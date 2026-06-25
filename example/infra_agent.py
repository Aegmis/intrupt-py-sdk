"""
Infrastructure Agent — human approval before any destructive cloud operation.

Use case: DevOps AI assistant that can query infra freely but must get
approval before creating, scaling, or deleting resources.

Run:
    uvicorn intrupt_py_sdk.example.infra_agent:app --port 8083

Test:
    curl -X POST http://localhost:8083/call-tool \
         -H 'Content-Type: application/json' \
         -d '{"message": "Scale the web-prod deployment to 10 replicas"}'
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
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8083")
_RESUME_SECRET = os.getenv("AGENT_RESUME_SECRET", "")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Read-only tools (no approval needed) ────────────────────────────────────

@tool
def list_deployments(environment: str = "production") -> list[dict]:
    """List all deployments in the given environment."""
    return [
        {"name": "web-prod", "replicas": 3, "image": "aegmis/web:v1.4.2", "status": "running"},
        {"name": "api-prod", "replicas": 2, "image": "aegmis/api:v1.4.2", "status": "running"},
        {"name": "worker-prod", "replicas": 1, "image": "aegmis/worker:v1.4.2", "status": "running"},
    ]


@tool
def get_deployment_logs(deployment: str, lines: int = 50) -> dict:
    """Fetch the last N log lines from a deployment."""
    return {
        "deployment": deployment,
        "lines": [f"[INFO] request processed in 42ms" for _ in range(lines)],
    }


@tool
def get_resource_usage(deployment: str) -> dict:
    """Get current CPU and memory usage for a deployment."""
    return {
        "deployment": deployment,
        "cpu_percent": 68.4,
        "memory_mb": 512,
        "requests_per_second": 120,
    }


# ── Mutating tools (approval required) ──────────────────────────────────────

@tool
@approval_required(
    action="scale_deployment",
    message="Approve scaling this deployment — this affects live traffic",
    channel="slack",
    args=["deployment", "replicas"],
)
def scale_deployment(deployment: str, replicas: int) -> dict:
    """Scale a deployment to the given number of replicas."""
    return {"status": "scaled", "deployment": deployment, "replicas": replicas}


@tool
@approval_required(
    action="delete_deployment",
    message="DANGER: Approve deleting this deployment — this is irreversible",
    channel="slack",
    args=["deployment", "environment"],
)
def delete_deployment(deployment: str, environment: str) -> dict:
    """Delete a deployment permanently."""
    return {"status": "deleted", "deployment": deployment, "environment": environment}


@tool
@approval_required(
    action="rollback_deployment",
    message="Approve rolling back this deployment to the previous image version",
    channel="slack",
    args=["deployment", "target_version"],
)
def rollback_deployment(deployment: str, target_version: str) -> dict:
    """Roll back a deployment to a previous image version."""
    return {"status": "rolled_back", "deployment": deployment, "version": target_version}


tools = [list_deployments, get_deployment_logs, get_resource_usage,
         scale_deployment, delete_deployment, rollback_deployment]
llm = ChatOpenAI(model="claude-sonnet-4-6").bind_tools(tools)


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

app = FastAPI(title="Infrastructure Agent")


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
    uvicorn.run(app, host="0.0.0.0", port=8083)
