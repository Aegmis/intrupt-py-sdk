"""
Telegram Agent — approval via Telegram bot inline keyboard buttons.

Sends an approval request as a Telegram message with ✅ Approve / ❌ Reject
inline buttons. Uses the Telegram Bot API directly via httpx.AsyncClient —
no extra library required. Telegram calls back to /telegram/webhook when the
user taps a button.

Required env vars:
    TELEGRAM_BOT_TOKEN   token from @BotFather, e.g. 123456:ABC-DEF...
    TELEGRAM_CHAT_ID     ID of the chat/group to send approvals to
    AGENT_PUBLIC_URL     publicly reachable base URL (Telegram must reach
                         /telegram/webhook — use ngrok or similar for local dev)

Setup webhook once (replace with your public URL):
    curl "https://api.telegram.org/bot<TOKEN>/setWebhook?url=<PUBLIC_URL>/telegram/webhook"

Run:
    uvicorn intrupt_py_sdk.example.telegram_agent:app --port 8091

Test:
    curl -X POST http://localhost:8091/call-tool \\
         -H 'Content-Type: application/json' \\
         -d '{"message": "Pay invoice INV-005 for $3200 to Acme Corp"}'

Then tap Approve or Reject in the Telegram chat.
"""

import asyncio
import os
import uuid
from typing import Annotated, TypedDict

import httpx
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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8091")

_TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

_pending: dict[str, str] = {}  # approval_id -> thread_id
_lock = asyncio.Lock()


async def _tg_post(method: str, **kwargs) -> dict:
    """Helper: POST to a Telegram Bot API method."""
    async with httpx.AsyncClient() as http:
        resp = await http.post(f"{_TG_API}/{method}", json=kwargs, timeout=10)
    resp.raise_for_status()
    return resp.json()


async def telegram_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    kwargs = v.get("tool", {}).get("kwargs", {})

    lines = "\n".join(f"  • <b>{k}</b>: {val}" for k, val in kwargs.items())
    text = (
        f"⚠️ <b>Approval Required</b>\n\n"
        f"<b>Action:</b> {v.get('action')}\n"
        f"<b>Message:</b> {v.get('message')}\n\n"
        f"<b>Parameters:</b>\n{lines or '  (none)'}\n\n"
        f"<code>thread: {thread_id[:8]}...</code>"
    )

    inline_keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": f"approve:{approval_id}"},
            {"text": "❌ Reject",  "callback_data": f"reject:{approval_id}"},
        ]]
    }

    async with _lock:
        _pending[approval_id] = thread_id

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            await _tg_post(
                "sendMessage",
                chat_id=TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="HTML",
                reply_markup=inline_keyboard,
            )
            print(f"[telegram] message sent (approval_id={approval_id})")
        except Exception as exc:
            print(f"[telegram] failed to send message: {exc}")
    else:
        print(
            f"\n[telegram] TELEGRAM_BOT_TOKEN/CHAT_ID not set — would send:\n"
            f"  action={v.get('action')}  approval_id={approval_id}\n"
        )

    return {"approval_id": approval_id}


# ── Graph ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_invoice(invoice_id: str) -> dict:
    """Look up an invoice by ID."""
    return {"invoice_id": invoice_id, "vendor": "Acme Corp", "amount": 3200.00, "status": "unpaid"}


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

approval_graph = ApprovalGraph(graph=graph, on_approval_async=telegram_approval)

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Telegram Approval Agent")


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


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Telegram calls this when the user taps an inline keyboard button."""
    update = await request.json()

    callback = update.get("callback_query")
    if not callback:
        return {"ok": True}  # ignore non-button updates

    callback_id = callback["id"]
    data: str = callback.get("data", "")

    if ":" not in data:
        return {"ok": True}

    decision_str, approval_id = data.split(":", 1)
    approved = decision_str == "approve"

    # Acknowledge the callback immediately so Telegram removes the loading spinner
    if TELEGRAM_BOT_TOKEN:
        try:
            await _tg_post("answerCallbackQuery", callback_query_id=callback_id)
        except Exception:
            pass

    async with _lock:
        thread_id = _pending.pop(approval_id, None)

    if thread_id is None:
        return {"ok": True}  # already decided

    # Edit the original message to reflect the decision
    if TELEGRAM_BOT_TOKEN:
        verdict = "✅ Approved" if approved else "❌ Rejected"
        user = callback.get("from", {}).get("username", "unknown")
        msg = callback.get("message", {})
        try:
            await _tg_post(
                "editMessageText",
                chat_id=msg.get("chat", {}).get("id"),
                message_id=msg.get("message_id"),
                text=f"{verdict} by @{user}",
                parse_mode="HTML",
            )
        except Exception:
            pass

    await approval_graph.aresume(thread_id, approved=approved, approval_id=approval_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)
