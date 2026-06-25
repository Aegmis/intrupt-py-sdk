"""
Finance Agent — custom approval via outbound webhook + in-memory store.

This example shows how to use ``on_approval`` to bypass the intrupt approval
API entirely.  Approval requests are POSTed to a configurable webhook URL
(e.g. your internal ops dashboard, PagerDuty, OpsGenie).  The webhook handler
calls back to this agent's ``/decide`` endpoint to approve or reject.

No ApprovalMiddleware, no APPROVAL_BASE_URL, no API key needed.

Run:
    uvicorn intrupt_py_sdk.example.webhook_agent:app --port 8085

Simulate an approval (replace <approval_id> from the /call-tool response):
    curl -X POST http://localhost:8085/decide \\
         -H 'Content-Type: application/json' \\
         -d '{"approval_id": "<approval_id>", "approved": true}'
"""

import os
import uuid
import httpx
import threading
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.types import Command

from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required

load_dotenv()

AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8085")
WEBHOOK_URL = os.getenv("APPROVAL_WEBHOOK_URL", "")  # where to POST approval requests


# ── In-memory pending store ──────────────────────────────────────────────────
# Maps approval_id -> thread_id so /decide can resume the right graph thread.
# In production, replace with Redis or your database.

_pending: dict[str, str] = {}
_lock = threading.Lock()


def _store_pending(approval_id: str, thread_id: str) -> None:
    with _lock:
        _pending[approval_id] = thread_id


def _pop_pending(approval_id: str) -> str | None:
    with _lock:
        return _pending.pop(approval_id, None)


# ── Custom on_approval callable ──────────────────────────────────────────────

def webhook_approval(thread_id: str, v: dict) -> dict:
    """
    Called by ApprovalCallbackHandler the moment an interrupt fires.

    POSTs a JSON payload to WEBHOOK_URL so an external system can present the
    approval request to a human.  Returns ``{"approval_id": ...}`` to signal
    that the graph is pending; ApprovalGraph will include this in its response.

    The external system must later call ``POST /decide`` with the approval_id.
    """
    approval_id = str(uuid.uuid4())
    _store_pending(approval_id, thread_id)

    payload = {
        "approval_id": approval_id,
        "thread_id": thread_id,
        "action": v.get("action"),
        "message": v.get("message"),
        "tool": v.get("tool", {}),
        # tell the webhook where to call back
        "decide_url": f"{AGENT_PUBLIC_URL}/decide",
    }

    if WEBHOOK_URL:
        try:
            httpx.post(WEBHOOK_URL, json=payload, timeout=5)
        except Exception as exc:
            # don't let a failed notification block the checkpoint
            print(f"[webhook_approval] webhook delivery failed: {exc}")
    else:
        # dev mode: just print so you can curl /decide manually
        print(
            f"\n[APPROVAL REQUIRED]\n"
            f"  action      : {payload['action']}\n"
            f"  message     : {payload['message']}\n"
            f"  tool kwargs : {payload['tool'].get('kwargs', {})}\n"
            f"  approval_id : {approval_id}\n"
            f"\n  To approve : curl -X POST {AGENT_PUBLIC_URL}/decide "
            f"-H 'Content-Type: application/json' "
            f"-d '{{\"approval_id\": \"{approval_id}\", \"approved\": true}}'\n"
        )

    return {"approval_id": approval_id}


# ── Tools ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_invoice(invoice_id: str) -> dict:
    """Look up an invoice by ID."""
    return {
        "invoice_id": invoice_id,
        "vendor": "Acme Corp",
        "amount": 4500.00,
        "currency": "USD",
        "status": "unpaid",
    }


@tool
@approval_required(
    action="pay_invoice",
    message="Approve this payment — funds will be transferred immediately",
    args=["invoice_id", "vendor", "amount", "currency"],
)
def pay_invoice(invoice_id: str, vendor: str, amount: float, currency: str = "USD") -> dict:
    """Pay an invoice. Transfers funds to the vendor."""
    return {
        "status": "paid",
        "invoice_id": invoice_id,
        "vendor": vendor,
        "amount": amount,
        "currency": currency,
        "transaction_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
    }


tools = [get_invoice, pay_invoice]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


# ── Graph ────────────────────────────────────────────────────────────────────

def chat_node(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "END"


memory = MemorySaver()
graph = (
    StateGraph(AgentState)
    .add_node("chat_node", chat_node)
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "chat_node")
    .add_conditional_edges("chat_node", route)
    .add_edge("tools", "chat_node")
    .compile(checkpointer=memory)
)

# No ApprovalMiddleware, no client=, no callback_url —
# just hand the graph our own callable.
approval_graph = ApprovalGraph(
    graph=graph,
    on_approval=webhook_approval,
)


# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Webhook Approval Agent")


@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    if not payload.get("message"):
        raise HTTPException(status_code=400, detail="'message' required")
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    if payload.get("thread_id") and approval_graph.pending(thread_id):
        raise HTTPException(
            status_code=409,
            detail="thread has a pending approval — decide before sending new messages",
        )
    return approval_graph.invoke(
        {"messages": [{"role": "user", "content": payload["message"]}]},
        thread_id,
    )


@app.post("/decide")
async def decide(request: Request):
    """
    Webhook callback endpoint — called by the external approval system
    (or manually via curl in dev) to approve or reject a pending tool call.
    """
    payload = await request.json()
    approval_id = payload.get("approval_id")
    if not approval_id:
        raise HTTPException(status_code=400, detail="approval_id required")
    if "approved" not in payload:
        raise HTTPException(status_code=400, detail="approved required")

    thread_id = _pop_pending(approval_id)
    if thread_id is None:
        raise HTTPException(status_code=404, detail="unknown or already-decided approval_id")

    return approval_graph.resume(
        thread_id,
        approved=bool(payload["approved"]),
        approval_id=approval_id,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)
