import os
import time
import uuid
from typing import Annotated, Optional, TypedDict

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import approval_required

load_dotenv()


APPROVAL_API_URL = os.environ.get("APPROVAL_BASE_URL", "http://localhost:8080")
AGENT_PUBLIC_URL = os.environ.get("AGENT_PUBLIC_URL", "http://host.docker.internal:8081")
AGENT_API_KEY = os.environ.get("APPROVAL_API_KEY")  # API key format: sk_org_{org_id}_{hash}

if not AGENT_API_KEY:
    raise ValueError("APPROVAL_API_KEY environment variable is required (format: sk_org_{org_id}_{hash})")

ApprovalMiddleware(base_url=APPROVAL_API_URL, api_key=AGENT_API_KEY)
approval_client = ApprovalMiddleware.get_client()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    last_purchase: Optional[dict]


llm = ChatOpenAI()


@tool
@approval_required(
    action="get_stock_price",
    message="Approve getting stock price",
    channel="slack",
    args=["symbol"],
)
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
def purchase_stock(symbol: str, quantity: int, amount: float, config: RunnableConfig) -> dict:
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
        "amount": amount
    }

tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def custom_tools_node(state: ChatState):
    """Custom tools node that tracks purchases for invoice generation."""
    print("DEBUG: custom_tools_node called")
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    
    print(f"DEBUG: Tool result messages: {len(result.get('messages', []))}")
    
    # Check if purchase_stock was called and store details
    for msg in result.get("messages", []):
        print(f"DEBUG: Message type: {type(msg)}, content: {getattr(msg, 'content', None)}")
        if hasattr(msg, 'content'):
            content = msg.content
            # Parse content if it's a string (ToolMessage stores content as string)
            if isinstance(content, str):
                try:
                    import json
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Check if this is a successful purchase_stock result
            if isinstance(content, dict) and (content.get("status") == "success" and 
                "symbol" in content and 
                "quantity" in content and "amount" in content):
                print(f"DEBUG: Purchase detected: {content}")
                return {
                    "messages": result["messages"],
                    "last_purchase": content
                }            # Check if this is a successful purchase_stock result
            elif isinstance(content, dict) and (content.get("status") == "success" and 
                "Global Quote" in content):
                print(f"DEBUG: Stock price detected: {content}")
                return {
                    "messages": result["messages"]
                }
    
    print("DEBUG: No purchase detected, returning normal result")
    return result


def invoice_generation_node(state: ChatState):
    """Generate invoice after successful purchase."""
    from langchain_core.messages import AIMessage
    
    print("DEBUG: invoice_generation_node called")
    purchase = state.get("last_purchase")
    print(f"DEBUG: Purchase data: {purchase}")
    
    if not purchase:
        print("DEBUG: No purchase found")
        return {"messages": [AIMessage(content="No purchase to generate invoice for.")]}
    
    invoice = {
        "invoice_id": str(uuid.uuid4()),
        "symbol": purchase.get("symbol"),
        "quantity": purchase.get("quantity"),
        "amount": purchase.get("amount"),
        "timestamp": time.time(),
        "status": "generated"
    }
    
    message = f"Invoice generated for {purchase.get('quantity')} shares of {purchase.get('symbol')}. Invoice ID: {invoice['invoice_id']}"
    print(f"DEBUG: Invoice message: {message}")
    
    return {
        "messages": [AIMessage(content=message)],
        "last_purchase": None  # Clear after invoice generation
    }


def should_generate_invoice(state: ChatState) -> str:
    """Check if we should generate invoice after purchase."""
    print(f"DEBUG: should_generate_invoice called, last_purchase: {state.get('last_purchase')}")
    if state.get("last_purchase"):
        print("DEBUG: Routing to generate_invoice")
        return "generate_invoice"
    print("DEBUG: Routing to chat_node")
    return "chat_node"


def route_to_tools(state: ChatState) -> str:
    """Route to tools if the last message has tool calls."""
    last_message = state["messages"][-1] if state["messages"] else None
    print(f"DEBUG: route_to_tools called, last_message: {type(last_message)}")
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"DEBUG: Routing to tools, tool_calls: {last_message.tool_calls}")
        return "tools"
    print("DEBUG: Routing to END")
    return "END"


memory = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", custom_tools_node)
graph.add_node("generate_invoice", invoice_generation_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", route_to_tools)
graph.add_conditional_edges("tools", should_generate_invoice)
graph.add_edge("generate_invoice", "chat_node")

agent = graph.compile(checkpointer=memory)


app = FastAPI(title="Agent")

# Add CORS middleware to handle cross-origin requests from the web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _build_response(thread_id: str, result: dict) -> dict:
    """Inspect the post-invoke graph state and build the HTTP response.

    If the run paused on another approval interrupt (e.g. a second tool that
    also needs approval), create the next approval on the API and return
    `pending_approval`. Otherwise the run is finished — return the messages.

    Used by both `/call-tool` and `/resume` so a chain of approval-gated tools
    pauses once per tool instead of silently stalling after the first.
    """
    config = {"configurable": {"thread_id": thread_id}}
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
        if "approval_id" in approval:
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


@app.post("/call-tool")
async def call_tool(request: Request):
    """Start (or continue) a chat. If a tool requires approval the graph
    pauses, an approval is created on the API, and the response contains the
    `approval_id` + `thread_id` the caller can use to poll or correlate.

    Request payload:
    {
        "message": str,          # Required: user message
        "thread_id": str,        # Optional: conversation thread ID
    }

    Organization is determined from the API key (sk_org_{org_id}_{hash}).
    """
    payload = await request.json()
    message = payload.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="'message' is required")

    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)

    return _build_response(thread_id, result)


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

    # Debug: check state before resuming
    paused_state = agent.get_state(config)
    print(f"DEBUG: Paused state messages: {len(paused_state.values.get('messages', []))}")
    print(f"DEBUG: Paused state: {paused_state.values}")

    result = agent.invoke(
        Command(resume={
            "approved": bool(payload["approved"]),
            "approval_id": payload.get("approval_id"),
        }),
        config=config,
    )

    print(f"DEBUG: Result messages: {len(result.get('messages', []))}")

    # The resumed run may pause again on the next approval-gated tool. Reuse the
    # same pending-approval detection as /call-tool so the next approval is
    # created instead of the run silently stalling.
    return _build_response(thread_id, result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
