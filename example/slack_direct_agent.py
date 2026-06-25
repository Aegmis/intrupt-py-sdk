"""
Slack Direct Agent — approval via a Block Kit message with Approve/Reject buttons.

Posts an interactive Slack message directly using slack_sdk (no intrupt API).
Slack sends an interactive callback to /slack/actions when the user clicks a
button. The handler verifies the signing secret, then resumes the graph.

Required env vars:
    SLACK_BOT_TOKEN      xoxb-... bot token
    SLACK_SIGNING_SECRET for verifying interactive callbacks
    SLACK_CHANNEL        channel ID or name, e.g. C012AB3CD or #approvals
    AGENT_PUBLIC_URL     publicly reachable base URL (Slack must reach /slack/actions)

Run:
    uvicorn intrupt_py_sdk.example.slack_direct_agent:app --port 8090

Test:
    curl -X POST http://localhost:8090/call-tool \\
         -H 'Content-Type: application/json' \\
         -d '{"message": "Pay invoice INV-003 for $900 to Acme Corp"}'

Then click Approve or Reject in Slack.
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Annotated, TypedDict
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from slack_sdk import WebClient

from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#approvals")
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8090")

_slack = WebClient(token=SLACK_BOT_TOKEN)
_pending: dict[str, str] = {}   # approval_id -> thread_id
_msg_ts: dict[str, str] = {}    # approval_id -> Slack message ts (for updating)
_lock = asyncio.Lock()


def _post_slack_message_sync(approval_id: str, v: dict) -> str | None:
    """Post Block Kit message; return message ts or None on failure."""
    kwargs = v.get("tool", {}).get("kwargs", {})
    fields = [
        {"type": "mrkdwn", "text": f"*{k}*\n{val}"}
        for k, val in kwargs.items()
    ]
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "⚠️ Approval Required", "emoji": True},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{v.get('message')}*"},
        },
        {
            "type": "section",
            "fields": fields or [{"type": "mrkdwn", "text": "_no args_"}],
        },
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Approve", "emoji": True},
                    "style": "primary",
                    "value": f"approve:{approval_id}",
                    "action_id": "approve",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Reject", "emoji": True},
                    "style": "danger",
                    "value": f"reject:{approval_id}",
                    "action_id": "reject",
                },
            ],
        },
    ]
    try:
        resp = _slack.chat_postMessage(channel=SLACK_CHANNEL, blocks=blocks, text=v.get("message"))
        return resp["ts"] if resp["ok"] else None
    except Exception as exc:
        print(f"[slack] failed to post message: {exc}")
        return None


async def slack_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())

    async with _lock:
        _pending[approval_id] = thread_id

    if SLACK_BOT_TOKEN:
        loop = asyncio.get_event_loop()
        ts = await loop.run_in_executor(None, _post_slack_message_sync, approval_id, v)
        if ts:
            async with _lock:
                _msg_ts[approval_id] = ts
            print(f"[slack] message posted (approval_id={approval_id})")
    else:
        print(
            f"\n[slack] SLACK_BOT_TOKEN not set — would post:\n"
            f"  action={v.get('action')}  kwargs={v.get('tool', {}).get('kwargs')}\n"
            f"  approval_id={approval_id}\n"
        )

    return {"approval_id": approval_id}


def _verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    if not SLACK_SIGNING_SECRET:
        return True  # skip verification in dev
    if abs(time.time() - int(timestamp)) > 300:
        return False
    base = f"v0:{timestamp}:{body.decode()}"
    expected = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), base.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ── Graph ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_invoice(invoice_id: str) -> dict:
    """Look up an invoice by ID."""
    return {"invoice_id": invoice_id, "vendor": "Acme Corp", "amount": 900.00, "status": "unpaid"}


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

approval_graph = ApprovalGraph(graph=graph, on_approval_async=slack_approval)

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Slack Direct Approval Agent")


@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    if not payload.get("message"):
        raise HTTPException(status_code=400, detail="'message' required")
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    if payload.get("thread_id") and approval_graph.pending(thread_id):
        raise HTTPException(status_code=409, detail="thread has a pending approval")
    return await approval_graph.ainvoke(
        {"messages": [{"role": "user", "content": payload["message"]}]},
        thread_id,
    )


@app.post("/slack/actions")
async def slack_actions(request: Request):
    """Slack interactive callback — fires when user clicks Approve or Reject."""
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "0")
    signature = request.headers.get("X-Slack-Signature", "")

    if not _verify_slack_signature(body, timestamp, signature):
        raise HTTPException(status_code=403, detail="invalid Slack signature")

    # Slack sends payload as URL-encoded form data
    form = parse_qs(body.decode())
    raw = form.get("payload", ["{}"])[0]
    payload = json.loads(raw)

    action = payload.get("actions", [{}])[0]
    value: str = action.get("value", "")

    if ":" not in value:
        return Response(status_code=200)

    decision_str, approval_id = value.split(":", 1)
    approved = decision_str == "approve"

    async with _lock:
        thread_id = _pending.pop(approval_id, None)
        ts = _msg_ts.pop(approval_id, None)

    if thread_id is None:
        return Response(status_code=200)  # already decided

    # Update the Slack message to show the decision
    if ts and SLACK_BOT_TOKEN:
        verdict_text = "✅ Approved" if approved else "❌ Rejected"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _slack.chat_update(
                channel=SLACK_CHANNEL,
                ts=ts,
                text=verdict_text,
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{verdict_text}* by <@{payload.get('user', {}).get('id', 'unknown')}>"},
                }],
            ),
        )

    await approval_graph.aresume(thread_id, approved=approved, approval_id=approval_id)
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
