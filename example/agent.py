import os
import uuid
from typing import Annotated, Optional, TypedDict

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import approval_required

load_dotenv()


APPROVAL_API_URL = os.environ.get("APPROVAL_BASE_URL", "http://localhost:8080")
APPROVAL_API_KEY = os.environ.get("APPROVAL_API_KEY", "test-api-key")
AGENT_PUBLIC_URL = os.environ.get("AGENT_PUBLIC_URL", "http://localhost:8081")

ApprovalMiddleware(base_url=APPROVAL_API_URL, api_key=APPROVAL_API_KEY)
approval_client = ApprovalMiddleware.get_client()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI()


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
    args=["symbol", "quantity"],
)
def purchase_stock(symbol: str, quantity: int, config: RunnableConfig) -> dict:
    """Simulate purchasing a given quantity of a stock symbol.

    Pauses for human approval via @approval_required. The tool body only runs
    if the human approves; otherwise the decorator returns a cancelled record
    and this function is not invoked.
    """
    return {
        "status": "success",
        "message": f"Purchase order placed for {quantity} shares of {symbol}.",
        "symbol": symbol,
        "quantity": quantity,
    }


tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


memory = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

agent = graph.compile(checkpointer=memory)


app = FastAPI(title="Agent")


def _extract_pending_approval(state) -> Optional[dict]:
    """If the graph is paused on an approval interrupt, return its payload.

    Returns None if the run has no pending interrupt or the interrupt was not
    raised by `@approval_required`.
    """
    for task in (state.tasks or ()):
        for itr in (task.interrupts or ()):
            value = getattr(itr, "value", itr)
            if isinstance(value, dict) and value.get("approval_required"):
                return value
    return None


def _messages_to_jsonable(result: dict) -> list:
    out = []
    for m in result.get("messages", []):
        out.append({
            "type": m.__class__.__name__,
            "content": getattr(m, "content", None),
        })
    return out


@app.post("/call-tool")
async def call_tool(request: Request):
    """Start (or continue) a chat. If a tool requires approval the graph
    pauses, an approval is created on the API, and the response contains the
    `approval_id` + `thread_id` the caller can use to poll or correlate.
    """
    payload = await request.json()
    message = payload.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="'message' is required")

    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)

    state = agent.get_state(config)
    pending = _extract_pending_approval(state)
    if pending is not None:
        tool_info = pending.get("tool") or {}
        approval = approval_client.create_approval(
            thread_id=thread_id,
            action=pending.get("action", "unknown"),
            message=pending.get("message", "Approval required"),
            channel=pending.get("channel", "slack"),
            tool={
                "name": tool_info.get("name"),
                "kwargs": tool_info.get("kwargs") or {},
            },
            agent_callback_url=f"{AGENT_PUBLIC_URL}/resume",
        )
        return {
            "status": "pending_approval",
            "thread_id": thread_id,
            "approval_id": approval["approval_id"],
        }

    return {
        "status": "complete",
        "thread_id": thread_id,
        "messages": _messages_to_jsonable(result),
    }


@app.post("/resume")
async def resume(request: Request):
    """Resume an approval-paused run with the human's decision.

    Body: {"approval_id": str, "thread_id": str, "approved": bool}
    `thread_id` is required (it identifies the checkpoint to resume); the
    other fields are passed through to the interrupt as `Command(resume=...)`.
    """
    payload = await request.json()

    thread_id = payload.get("thread_id")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    if "approved" not in payload:
        raise HTTPException(status_code=400, detail="approved is required")

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        Command(resume={
            "approved": bool(payload["approved"]),
            "approval_id": payload.get("approval_id"),
        }),
        config=config,
    )
    return {
        "status": "complete",
        "thread_id": thread_id,
        "messages": _messages_to_jsonable(result),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
