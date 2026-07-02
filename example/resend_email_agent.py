"""resend_email_agent.py — email-channel approval via the intrupt API + Resend.

Identical to agent.py except channel="email" on the gated tool.  The intrupt
API resolves the approver's email address from the matching policy, sends a
branded HTML email with HMAC-signed Approve / Reject links, and calls back to
this agent's /resume endpoint once the human clicks.

Required env vars (agent side):
    OPENAI_API_KEY
    APPROVAL_BASE_URL      e.g. https://api.intrupt.dev
    APPROVAL_API_KEY       sk_org_...
    AGENT_RESUME_SECRET    random secret — echoed in X-Agent-Secret on /resume

Required env vars (intrupt API side — not needed here):
    RESEND_API_KEY         re_...
    RESEND_FROM_EMAIL      Aegmis <noreply@yourdomain.com>
    EMAIL_DECISION_SECRET  random secret used to sign approve/reject URLs
    APPROVAL_API_BASE_URL  https://api.intrupt.dev

Run:
    python example/resend_email_agent.py

Try it:
    curl -X POST http://localhost:8095/call-tool \\
         -H 'Content-Type: application/json' \\
         -d '{"message": "buy 10 shares of AAPL"}'

Resume after human clicks Approve in email (the API does this automatically):
    curl -X POST http://localhost:8095/resume \\
         -H 'Content-Type: application/json' \\
         -H 'X-Agent-Secret: your-secret' \\
         -d '{"thread_id": "...", "approval_id": "...", "approved": true}'
"""

import os
import uuid
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

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required

load_dotenv()

# ── Approval client ───────────────────────────────────────────────────────────
ApprovalMiddleware(
    base_url=os.getenv("APPROVAL_BASE_URL", "http://localhost:8080"),
    api_key=os.getenv("APPROVAL_API_KEY"),
)
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8095")
_RESUME_SECRET = os.getenv("AGENT_RESUME_SECRET", "")


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    last_purchase: Optional[dict]


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
@approval_required(
    action="purchase_stock",
    message="Review and approve this stock purchase before funds move",
    channel="email",                        # ← intrupt API sends an email
    args=["symbol", "quantity", "amount"],
)
def purchase_stock(symbol: str, quantity: int, amount: float) -> dict:
    """Purchase a given quantity of a stock symbol."""
    return {
        "status": "success",
        "message": f"Purchase order placed for {quantity} shares of {symbol}.",
        "symbol": symbol,
        "quantity": quantity,
        "amount": amount,
    }


tools = [purchase_stock]
llm   = ChatOpenAI().bind_tools(tools)


# ── Graph ─────────────────────────────────────────────────────────────────────
def chat_node(state: AgentState):
    from langchain_core.messages import ToolMessage
    messages = list(state["messages"])
    while messages and isinstance(messages[0], ToolMessage):
        messages = messages[1:]
    if not messages:
        return {}
    return {"messages": [llm.invoke(messages)]}


def route_to_tools(state: AgentState) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "END"


memory = MemorySaver()
graph  = (
    StateGraph(AgentState)
    .add_node("chat_node", chat_node)
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "chat_node")
    .add_conditional_edges("chat_node", route_to_tools)
    .add_edge("tools", "chat_node")
    .compile(checkpointer=memory)
)

approval_graph = ApprovalGraph(
    graph=graph,
    callback_url=f"{AGENT_PUBLIC_URL}/resume",
    callback_secret=_RESUME_SECRET,
)


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Resend Email Approval Agent", version="1.0.0")


@app.post("/call-tool")
async def call_tool(request: Request):
    """Start or continue a conversation.

    Returns immediately with status="pending_approval" when a tool requires
    approval.  The intrupt API emails the approver; this endpoint is called
    again automatically via /resume once the human decides.
    """
    payload = await request.json()
    message = payload.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="'message' required")

    thread_id = payload.get("thread_id") or str(uuid.uuid4())

    if payload.get("thread_id") and approval_graph.pending(thread_id):
        raise HTTPException(
            status_code=409,
            detail="thread has a pending approval — wait for the email response",
        )

    return await approval_graph.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        thread_id,
    )


@app.post("/resume")
async def resume(request: Request):
    """Called by the intrupt API after the approver clicks Approve or Reject in email.

    The intrupt API signs this request with X-Agent-Secret (AGENT_RESUME_SECRET).
    """
    if _RESUME_SECRET and request.headers.get("X-Agent-Secret", "") != _RESUME_SECRET:
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
            detail="thread is not paused on an approval",
        )

    return await approval_graph.aresume(
        thread_id,
        approved=bool(payload["approved"]),
        approval_id=payload.get("approval_id", ""),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8095)
