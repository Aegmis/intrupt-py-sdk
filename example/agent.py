import os
import uuid
import time

from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langgraph.graph.message import add_messages
import requests

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import approval_required

load_dotenv()

# ── Initialise the approval client (singleton) ──────────────────────────────
ApprovalMiddleware(
    base_url=os.getenv("APPROVAL_BASE_URL", "http://localhost:8080"),
    api_key=os.getenv("APPROVAL_API_KEY"),
)
approval_client = ApprovalMiddleware.get_client()

AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8081")


# ── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    last_purchase: Optional[dict]


# ── Tools ────────────────────────────────────────────────────────────────────
@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')."""
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    r = requests.get(url, timeout=10)
    return r.json()

@tool
@approval_required(
    action="purchase_stock",
    message="Approve buying shares",
    channel="slack",
    args=["symbol", "quantity", "amount"],
)
def purchase_stock(symbol: str, quantity: int, amount: float) -> dict:
    """Purchase a given quantity of a stock symbol."""
    return {
        "status": "success",
        "message": f"Purchase order placed for {quantity} shares of {symbol}.",
        "symbol": symbol,
        "quantity": quantity,
        "amount": amount
    }


tools = [get_stock_price, purchase_stock]
llm   = ChatOpenAI().bind_tools(tools)


# ── Graph ────────────────────────────────────────────────────────────────────
def chat_node(state: AgentState):
    messages = state["messages"]
    return {"messages": [llm.invoke(messages)]}


def custom_tools_node(state: AgentState):
    """Custom tools node that tracks purchases for invoice generation."""
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)

    for msg in result.get("messages", []):
        if hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str):
                try:
                    import json
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(content, dict) and content.get("status") == "success":
                if "symbol" in content and "quantity" in content and "amount" in content:
                    return {"messages": result["messages"], "last_purchase": content}
                if "symbol" in content:
                    return {"messages": result["messages"]}

    return result

def invoice_generation_node(state: AgentState):
    """Generate invoice after successful purchase."""
    from langchain_core.messages import AIMessage

    purchase = state.get("last_purchase")
    if not purchase:
        return {"messages": [AIMessage(content="No purchase to generate invoice for.")]}

    invoice_id = str(uuid.uuid4())
    message = (
        f"Invoice generated for {purchase.get('quantity')} shares of "
        f"{purchase.get('symbol')}. Invoice ID: {invoice_id}"
    )
    return {"messages": [AIMessage(content=message)], "last_purchase": None}

def should_generate_invoice(state: AgentState) -> str:
    """Check if we should generate invoice after purchase."""
    return "invoice_generation_node" if state.get("last_purchase") else "chat_node"


def route_to_tools(state: AgentState) -> str:
    """Route to tools if the last message has tool calls."""
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "END"

memory = MemorySaver()
graph  = (
    StateGraph(AgentState)
    .add_node("chat_node", chat_node)
    .add_node("tools", custom_tools_node)
    .add_node("invoice_generation_node", invoice_generation_node)
    .add_edge(START, "chat_node")
    .add_conditional_edges("chat_node", route_to_tools)
    .add_conditional_edges("tools", should_generate_invoice)
    .add_edge("invoice_generation_node", "chat_node")
    .compile(checkpointer=memory)
)


# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI()
_RESUME_SECRET = os.getenv("AGENT_RESUME_SECRET", "")


def _build_response(thread_id: str, result: dict) -> dict:
    """Check for pending approval interrupts and create the approval request."""
    config = {"configurable": {"thread_id": thread_id}}
    state  = graph.get_state(config)
    for task in (state.tasks or ()):
        for itr in (task.interrupts or ()):
            v = getattr(itr, "value", itr)
            if isinstance(v, dict) and v.get("approval_required"):
                approval = approval_client.create_approval(
                    thread_id=thread_id,
                    action=v.get("action", "unknown"),
                    message=v.get("message", "Approval required"),
                    channel=v.get("channel", "slack"),
                    tool=v.get("tool", {}),
                    agent_callback_url=f"{AGENT_PUBLIC_URL}/resume",
                    agent_callback_secret=_RESUME_SECRET,
                )
                if "approval_id" in approval:
                    return {
                        "status": "pending_approval",
                        "thread_id": thread_id,
                        "approval_id": approval["approval_id"],
                    }
    return {
        "status": "complete",
        "thread_id": thread_id,
        "messages": [{"type": m.__class__.__name__, "content": m.content}
                     for m in result.get("messages", [])],
    }


@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    message = payload.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="'message' required")

    thread_id_from_client = payload.get("thread_id")
    thread_id = thread_id_from_client or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Reject new messages on a thread that already has a pending approval.
    # Sending a second message while the graph is interrupted corrupts the
    # checkpoint and leads to invalid message sequences on resume.
    if thread_id_from_client:
        existing = graph.get_state(config)
        for task in (existing.tasks or ()):
            for itr in (task.interrupts or ()):
                v = getattr(itr, "value", itr)
                if isinstance(v, dict) and v.get("approval_required"):
                    raise HTTPException(
                        status_code=409,
                        detail="thread has a pending approval — approve or reject before sending new messages",
                    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return _build_response(thread_id, result)


@app.post("/resume")
async def resume(request: Request):
    if _RESUME_SECRET:
        incoming = request.headers.get("X-Agent-Secret", "")
        if not incoming or incoming != _RESUME_SECRET:
            raise HTTPException(status_code=401, detail="missing or invalid X-Agent-Secret")

    payload = await request.json()
    thread_id = payload.get("thread_id")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    if "approved" not in payload:
        raise HTTPException(status_code=400, detail="approved required")

    config = {"configurable": {"thread_id": thread_id}}

    # Verify the thread is actually paused on an approval interrupt before
    # resuming. If the checkpoint is missing (server restarted while an
    # approval was in flight), invoking Command(resume=...) on a non-existent
    # checkpoint produces an invalid message sequence and triggers the
    # "messages[0].role == tool" error in chat_node.
    state = graph.get_state(config)
    has_pending = any(
        isinstance(getattr(itr, "value", itr), dict)
        and getattr(itr, "value", itr).get("approval_required")
        for task in (state.tasks or ())
        for itr in (task.interrupts or ())
    )
    if not has_pending:
        raise HTTPException(
            status_code=409,
            detail="thread is not paused on an approval (checkpoint missing or already decided)",
        )

    result = graph.invoke(
        Command(resume={"approved": bool(payload["approved"]),
                        "approval_id": payload.get("approval_id")}),
        config=config,
    )
    return _build_response(thread_id, result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
