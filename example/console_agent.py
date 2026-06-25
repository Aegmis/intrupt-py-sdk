"""
Console Agent — approval via interactive stdin prompt.

Ideal for local development and CLI tools. When a tool requires approval the
agent blocks and asks the developer directly in the terminal. The decision is
captured during ainvoke and the graph is resumed immediately — no separate
/decide call needed.

Run:
    uvicorn intrupt_py_sdk.example.console_agent:app --port 8087

Test:
    curl -X POST http://localhost:8087/call-tool \\
         -H 'Content-Type: application/json' \\
         -d '{"message": "Pay invoice INV-001 for $200 to Acme Corp"}'

The server will pause and print a y/n prompt to its terminal. Type y or n.
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

# approval_id -> bool: decisions made at the terminal during on_approval_async
_console_decisions: dict[str, bool] = {}


async def console_approval(thread_id: str, v: dict) -> dict:
    """Ask y/n on stdin without blocking the event loop."""
    approval_id = str(uuid.uuid4())

    kwargs = v.get("tool", {}).get("kwargs", {})
    prompt = (
        f"\n{'='*60}\n"
        f"[APPROVAL REQUIRED]\n"
        f"  thread  : {thread_id}\n"
        f"  action  : {v.get('action')}\n"
        f"  message : {v.get('message')}\n"
        f"  args    : {kwargs}\n"
        f"{'='*60}\n"
        f"Approve? [y/n]: "
    )

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, input, prompt)
    _console_decisions[approval_id] = answer.strip().lower() in ("y", "yes")
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

approval_graph = ApprovalGraph(graph=graph, on_approval_async=console_approval)

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Console Approval Agent")


@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    if not payload.get("message"):
        raise HTTPException(status_code=400, detail="'message' required")

    thread_id = payload.get("thread_id") or str(uuid.uuid4())

    result = await approval_graph.ainvoke(
        {"messages": [{"role": "user", "content": payload["message"]}]},
        thread_id,
    )

    # The terminal decision was captured during ainvoke — auto-resume now.
    if result.get("status") == "pending_approval":
        aid = result["approval_id"]
        approved = _console_decisions.pop(aid, False)
        result = await approval_graph.aresume(thread_id, approved=approved, approval_id=aid)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8087)
