"""
Policy Agent — auto-approve or auto-reject based on rules, escalate otherwise.

Rules are evaluated against the interrupt payload at interrupt time:
  • amount < 500           → auto-approve  (low-risk, no human needed)
  • amount > 50 000        → auto-reject   (over budget ceiling)
  • vendor in BLOCKLIST    → auto-reject
  • action starts "emergency_" → auto-approve (incident response fast path)
  • everything else        → escalate: store pending, print curl to terminal

For auto decisions the /call-tool endpoint immediately resumes the graph after
ainvoke returns, so the caller gets a final "complete" response in one shot.

Run:
    uvicorn intrupt_py_sdk.example.policy_agent:app --port 8088

Test (auto-approve, amount < 500):
    curl -X POST http://localhost:8088/call-tool \\
         -H 'Content-Type: application/json' \\
         -d '{"message": "Pay invoice INV-001 for $200 to Acme Corp"}'

Test (escalate to human, amount in range):
    curl -X POST http://localhost:8088/call-tool \\
         -H 'Content-Type: application/json' \\
         -d '{"message": "Pay invoice INV-002 for $2500 to Acme Corp"}'
"""

import asyncio
import os
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required

load_dotenv()

AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8088")

BLOCKED_VENDORS = {"BlockedCorp", "SanctionedLtd"}

# approval_id -> bool  (populated for auto-decisions; absent means needs human)
_auto_decisions: dict[str, bool] = {}
# approval_id -> thread_id  (populated for human-escalated approvals)
_pending: dict[str, str] = {}
_lock = asyncio.Lock()


def _evaluate_policy(v: dict) -> bool | None:
    """Return True/False for auto decisions, None to escalate to a human."""
    action = v.get("action", "")
    kwargs = v.get("tool", {}).get("kwargs", {})
    amount = float(kwargs.get("amount", 0))
    vendor = kwargs.get("vendor", "")

    if action.startswith("emergency_"):
        return True
    if vendor in BLOCKED_VENDORS:
        return False
    if amount < 500:
        return True
    if amount > 50_000:
        return False
    return None  # escalate


async def policy_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    decision = _evaluate_policy(v)

    async with _lock:
        if decision is not None:
            _auto_decisions[approval_id] = decision
            verdict = "AUTO-APPROVED" if decision else "AUTO-REJECTED"
            print(f"[policy] {verdict} — action={v.get('action')} amount={v.get('tool', {}).get('kwargs', {}).get('amount')}")
        else:
            _pending[approval_id] = thread_id
            print(
                f"\n[policy] ESCALATED — human decision required\n"
                f"  curl -X POST {AGENT_PUBLIC_URL}/decide "
                f"-H 'Content-Type: application/json' "
                f"-d '{{\"approval_id\": \"{approval_id}\", \"approved\": true}}'\n"
            )

    return {"approval_id": approval_id}


# ── Graph ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_invoice(invoice_id: str) -> dict:
    """Look up an invoice by ID."""
    return {"invoice_id": invoice_id, "vendor": "Acme Corp", "amount": 200.00, "status": "unpaid"}


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
        "transaction_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
    }


tools = [get_invoice, pay_invoice]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


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

approval_graph = ApprovalGraph(graph=graph, on_approval_async=policy_approval)

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Policy Approval Agent")


@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    if not payload.get("message"):
        raise HTTPException(status_code=400, detail="'message' required")

    thread_id = payload.get("thread_id") or str(uuid.uuid4())

    if payload.get("thread_id") and approval_graph.pending(thread_id):
        raise HTTPException(status_code=409, detail="thread has a pending human approval")

    result = await approval_graph.ainvoke(
        {"messages": [{"role": "user", "content": payload["message"]}]},
        thread_id,
    )

    # Auto-decisions are available immediately — resume in the same request.
    if result.get("status") == "pending_approval":
        aid = result["approval_id"]
        async with _lock:
            approved = _auto_decisions.pop(aid, None)
        if approved is not None:
            result = await approval_graph.aresume(thread_id, approved=approved, approval_id=aid)

    return result


@app.post("/decide")
async def decide(request: Request):
    """Human decision endpoint for escalated approvals."""
    payload = await request.json()
    aid = payload.get("approval_id")
    if not aid or "approved" not in payload:
        raise HTTPException(status_code=400, detail="approval_id and approved required")

    async with _lock:
        thread_id = _pending.pop(aid, None)
    if thread_id is None:
        raise HTTPException(status_code=404, detail="unknown or already-decided approval_id")

    return await approval_graph.aresume(thread_id, approved=bool(payload["approved"]), approval_id=aid)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)
