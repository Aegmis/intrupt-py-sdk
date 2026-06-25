"""
Finance Agent — fully async approval via outbound webhook.

Demonstrates ``on_approval_async`` with ``httpx.AsyncClient`` so the event
loop is never blocked: the HTTP notification to the webhook and the graph
execution both run async end-to-end.

Run:
    uvicorn intrupt_py_sdk.example.async_webhook_agent:app --port 8086

Simulate an approval (use <approval_id> from the /call-tool response):
    curl -X POST http://localhost:8086/decide \\
         -H 'Content-Type: application/json' \\
         -d '{"approval_id": "<approval_id>", "approved": true}'
"""

import os
import uuid
import asyncio
import httpx
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

from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required

load_dotenv()

AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8086")
WEBHOOK_URL = os.getenv("APPROVAL_WEBHOOK_URL", "")


# ── In-memory pending store ──────────────────────────────────────────────────
# Maps approval_id -> asyncio.Event so /decide can unblock waiters,
# and approval_id -> thread_id so the graph knows which thread to resume.

_pending_threads: dict[str, str] = {}
_lock = asyncio.Lock()


async def _store_pending(approval_id: str, thread_id: str) -> None:
    async with _lock:
        _pending_threads[approval_id] = thread_id


async def _pop_pending(approval_id: str) -> str | None:
    async with _lock:
        return _pending_threads.pop(approval_id, None)


# ── Async on_approval callable ───────────────────────────────────────────────

async def async_webhook_approval(thread_id: str, v: dict) -> dict:
    """
    Async on_approval — called by AsyncApprovalCallbackHandler during ainvoke.

    Uses httpx.AsyncClient so the POST to the webhook does not block the event
    loop.  Returns {"approval_id": ...} which signals ApprovalGraph to put the
    response in "pending_approval" state.
    """
    approval_id = str(uuid.uuid4())
    await _store_pending(approval_id, thread_id)

    payload = {
        "approval_id": approval_id,
        "thread_id": thread_id,
        "action": v.get("action"),
        "message": v.get("message"),
        "tool": v.get("tool", {}),
        "decide_url": f"{AGENT_PUBLIC_URL}/decide",
    }

    if WEBHOOK_URL:
        async with httpx.AsyncClient() as http:
            try:
                await http.post(WEBHOOK_URL, json=payload, timeout=5)
            except Exception as exc:
                print(f"[async_webhook_approval] webhook delivery failed: {exc}")
    else:
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

# Pass on_approval_async — ApprovalGraph uses ainvoke/aresume paths only.
# No client=, no callback_url, no APPROVAL_API_KEY needed.
approval_graph = ApprovalGraph(
    graph=graph,
    on_approval_async=async_webhook_approval,
)


# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Async Webhook Approval Agent")


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

    # ainvoke: graph runs async, on_interrupt fires async_webhook_approval
    return await approval_graph.ainvoke(
        {"messages": [{"role": "user", "content": payload["message"]}]},
        thread_id,
    )


@app.post("/decide")
async def decide(request: Request):
    """Called by the external approval system (or curl in dev) to approve/reject."""
    payload = await request.json()
    approval_id = payload.get("approval_id")
    if not approval_id:
        raise HTTPException(status_code=400, detail="approval_id required")
    if "approved" not in payload:
        raise HTTPException(status_code=400, detail="approved required")

    thread_id = await _pop_pending(approval_id)
    if thread_id is None:
        raise HTTPException(status_code=404, detail="unknown or already-decided approval_id")

    # aresume: graph resumes async, same non-blocking path
    return await approval_graph.aresume(
        thread_id,
        approved=bool(payload["approved"]),
        approval_id=approval_id,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)
