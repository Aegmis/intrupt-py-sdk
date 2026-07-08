import hmac
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
from langgraph.graph.message import add_messages
import requests

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required

load_dotenv()

# ── Initialise the approval client (singleton) ──────────────────────────────
ApprovalMiddleware(
    base_url=os.getenv("AEGMIS_BASE_URL", "http://localhost:8080"),
    api_key=os.getenv("AEGMIS_API_KEY"),
)
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8081")
_RESUME_SECRET = os.getenv("AGENT_RESUME_SECRET", "")


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
    from langchain_core.messages import ToolMessage
    messages = list(state["messages"])
    # Strip leading ToolMessages — OpenAI rejects requests where messages[0].role == "tool",
    # which can happen after a server restart mid-approval when a checkpoint is replayed.
    while messages and isinstance(messages[0], ToolMessage):
        messages = messages[1:]
    if not messages:
        return {}
    return {"messages": [llm.invoke(messages)]}


async def custom_tools_node(state: AgentState):
    """Custom tools node that tracks purchases for invoice generation."""
    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke(state)

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

approval_graph = ApprovalGraph(
    graph=graph,
    callback_url=f"{AGENT_PUBLIC_URL}/resume",
    callback_secret=_RESUME_SECRET,
)


# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI()


def _raise_if_error(result: dict) -> dict:
    """Surface an SDK {"status": "error"} result as a real HTTP error.

    The approval API's status (e.g. 422 channel mismatch) rides along in
    "status_code"; without this the endpoint would return HTTP 200 with an
    error body, masking the failure from the caller.
    """
    if result.get("status") == "error":
        raise HTTPException(
            status_code=result.get("status_code", 502),
            detail=result.get("error", "approval failed"),
        )
    return result


@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    message = payload.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="'message' required")

    thread_id = payload.get("thread_id") or str(uuid.uuid4())

    # Reject new messages on a thread that already has a pending approval.
    if payload.get("thread_id") and approval_graph.pending(thread_id):
        raise HTTPException(
            status_code=409,
            detail="thread has a pending approval — approve or reject before sending new messages",
        )

    return _raise_if_error(
        await approval_graph.run({"messages": [{"role": "user", "content": message}]}, thread_id)
    )


@app.post("/resume")
async def resume(request: Request):
    # /resume can approve a paused run, so it must be authenticated. Fail CLOSED:
    # if AGENT_RESUME_SECRET is not configured, refuse to serve /resume outside
    # local dev rather than accepting unauthenticated approvals.
    if not _RESUME_SECRET:
        if os.getenv("APP_ENV", "local") != "local":
            raise HTTPException(status_code=500, detail="AGENT_RESUME_SECRET is not configured")
    # Constant-time compare so the secret can't be recovered via response timing.
    elif not hmac.compare_digest(request.headers.get("X-Agent-Secret", ""), _RESUME_SECRET):
        raise HTTPException(status_code=401, detail="missing or invalid X-Agent-Secret")

    payload = await request.json()
    thread_id = payload.get("thread_id")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    if "approved" not in payload:
        raise HTTPException(status_code=400, detail="approved required")

    if not approval_graph.pending(thread_id):
        raise HTTPException(
            status_code=409,
            detail="thread is not paused on an approval (no pending gate or already decided)",
        )

    result = await approval_graph.resume(
        thread_id,
        approved=bool(payload["approved"]),
        approval_id=payload.get("approval_id", ""),
    )
    if result.get("status") == "accepted":
        result = await approval_graph.wait_for_result(thread_id)
    return result


if __name__ == "__main__":
    import uvicorn
    from urllib.parse import urlparse
    # Listen on the same port advertised to the approval API as the callback host.
    _port = urlparse(AGENT_PUBLIC_URL).port or 8081
    uvicorn.run(app, host="0.0.0.0", port=_port)
