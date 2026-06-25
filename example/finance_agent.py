"""
Finance Agent — human approval for payments, refunds, and budget changes.

Use case: Accounts-payable AI that can look up invoices and balances freely
but must get human sign-off before moving any money.

Run:
    uvicorn intrupt_py_sdk.example.finance_agent:app --port 8084

Test:
    curl -X POST http://localhost:8084/call-tool \
         -H 'Content-Type: application/json' \
         -d '{"message": "Pay invoice INV-2024-0042 for $4,500 to Acme Corp"}'
"""

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
from langgraph.prebuilt import ToolNode

from langgraph.graph.message import add_messages

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required

load_dotenv()

ApprovalMiddleware(
    base_url=os.getenv("APPROVAL_BASE_URL", "http://localhost:8080"),
    api_key=os.getenv("APPROVAL_API_KEY"),
)
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8084")
_RESUME_SECRET = os.getenv("AGENT_RESUME_SECRET", "")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Read-only tools ──────────────────────────────────────────────────────────

@tool
def get_invoice(invoice_id: str) -> dict:
    """Look up an invoice by ID."""
    return {
        "invoice_id": invoice_id,
        "vendor": "Acme Corp",
        "amount": 4500.00,
        "currency": "USD",
        "due_date": "2026-07-01",
        "status": "unpaid",
        "line_items": [
            {"description": "Cloud infrastructure Q2", "amount": 3200.00},
            {"description": "Support retainer", "amount": 1300.00},
        ],
    }


@tool
def get_account_balance(account_id: str = "main") -> dict:
    """Get current balance for a company account."""
    return {"account_id": account_id, "balance": 128_450.00, "currency": "USD", "available": 120_000.00}


@tool
def list_pending_invoices() -> list[dict]:
    """List all unpaid invoices."""
    return [
        {"invoice_id": "INV-2024-0042", "vendor": "Acme Corp", "amount": 4500.00, "due_date": "2026-07-01"},
        {"invoice_id": "INV-2024-0043", "vendor": "DataDog", "amount": 890.00, "due_date": "2026-07-05"},
        {"invoice_id": "INV-2024-0044", "vendor": "AWS", "amount": 12_340.00, "due_date": "2026-07-10"},
    ]


@tool
def get_payment_history(vendor: str, limit: int = 5) -> list[dict]:
    """Get recent payment history for a vendor."""
    return [
        {"date": "2026-05-01", "vendor": vendor, "amount": 4500.00, "status": "paid", "ref": "PAY-001"},
        {"date": "2026-04-01", "vendor": vendor, "amount": 4500.00, "status": "paid", "ref": "PAY-002"},
    ]


# ── Mutating tools (approval required) ──────────────────────────────────────

@tool
@approval_required(
    action="pay_invoice",
    message="Approve this payment — funds will be transferred immediately upon approval",
    channel="slack",
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


@tool
@approval_required(
    action="issue_refund",
    message="Approve issuing this refund to the customer",
    channel="slack",
    args=["customer_id", "amount", "reason"],
)
def issue_refund(customer_id: str, amount: float, reason: str) -> dict:
    """Issue a refund to a customer."""
    return {
        "status": "refunded",
        "customer_id": customer_id,
        "amount": amount,
        "reason": reason,
        "refund_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
    }


@tool
@approval_required(
    action="update_budget",
    message="Approve this budget change — this affects spend limits for the team",
    channel="slack",
    args=["department", "category", "new_limit", "current_limit"],
)
def update_budget(department: str, category: str, new_limit: float, current_limit: float) -> dict:
    """Update a department's budget limit for a spend category."""
    return {
        "status": "updated",
        "department": department,
        "category": category,
        "previous_limit": current_limit,
        "new_limit": new_limit,
    }


tools = [get_invoice, get_account_balance, list_pending_invoices, get_payment_history,
         pay_invoice, issue_refund, update_budget]
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

approval_graph = ApprovalGraph(
    graph=graph,
    client=ApprovalMiddleware.get_client(),
    callback_url=f"{AGENT_PUBLIC_URL}/resume",
    callback_secret=_RESUME_SECRET,
)


# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Finance Agent")


@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    if not payload.get("message"):
        raise HTTPException(status_code=400, detail="'message' required")
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    return approval_graph.invoke({"messages": [{"role": "user", "content": payload["message"]}]}, thread_id)


@app.post("/resume")
async def resume(request: Request):
    if _RESUME_SECRET:
        if request.headers.get("X-Agent-Secret", "") != _RESUME_SECRET:
            raise HTTPException(status_code=401, detail="invalid X-Agent-Secret")
    payload = await request.json()
    thread_id = payload.get("thread_id")
    if not thread_id or "approved" not in payload:
        raise HTTPException(status_code=400, detail="thread_id and approved required")
    return approval_graph.resume(thread_id, approved=bool(payload["approved"]), approval_id=payload.get("approval_id"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
