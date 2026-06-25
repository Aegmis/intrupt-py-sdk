"""
Email Agent — human approval before sending any outbound email.

Use case: AI assistant that drafts and sends emails on behalf of a user.
Every send is gated by a human reviewer so nothing goes out unreviewed.

Run:
    uvicorn intrupt_py_sdk.example.email_agent:app --port 8082

Test:
    curl -X POST http://localhost:8082/call-tool \
         -H 'Content-Type: application/json' \
         -d '{"message": "Send a follow-up email to john@acme.com about our meeting yesterday"}'
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

ApprovalMiddleware(
    base_url=os.getenv("APPROVAL_BASE_URL"),
    api_key=os.getenv("APPROVAL_API_KEY"),
)
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8082")
_RESUME_SECRET = os.getenv("AGENT_RESUME_SECRET", "")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def draft_email(to: str, subject: str, body: str) -> dict:
    """Draft an email. Returns the draft for review before sending."""
    return {"draft": True, "to": to, "subject": subject, "body": body}


@tool
@approval_required(
    action="send_email",
    message="Review and approve this outbound email before it is sent",
    channel="slack",
    args=["to", "subject", "body"],
)
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to the recipient. Requires human approval."""
    # In production: integrate with SendGrid / Resend / SES here
    return {
        "status": "sent",
        "to": to,
        "subject": subject,
        "message_id": f"msg_{uuid.uuid4().hex[:8]}",
    }


@tool
def search_inbox(query: str) -> list[dict]:
    """Search recent emails by keyword (mock)."""
    return [
        {"from": "john@acme.com", "subject": "Re: Q3 proposal", "date": "2026-06-20"},
        {"from": "john@acme.com", "subject": "Meeting notes", "date": "2026-06-22"},
    ]


tools = [draft_email, send_email, search_inbox]
llm = ChatOpenAI().bind_tools(tools)


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

app = FastAPI(title="Email Agent")


@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    if not payload.get("message"):
        raise HTTPException(status_code=400, detail="'message' required")
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    x = approval_graph.invoke({"messages": [{"role": "user", "content": payload["message"]}]}, thread_id)
    print("=== Response ===")
    print(x)
    return x


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
    uvicorn.run(app, host="0.0.0.0", port=8082)
