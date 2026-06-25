"""
SMTP Email Agent — approval via email with one-click Approve / Reject links.

The approver receives an HTML email. Clicking a link hits the agent's /decide
GET endpoint, which resumes the paused graph. Uses Python's stdlib smtplib
(wrapped in a thread executor so it doesn't block the event loop).

Required env vars:
    SMTP_HOST          default: smtp.gmail.com
    SMTP_PORT          default: 587
    SMTP_USER          your Gmail / SMTP username
    SMTP_PASSWORD      app password (not your account password)
    APPROVER_EMAIL     who receives the approval request
    AGENT_PUBLIC_URL   publicly reachable base URL of this agent (for links)

Run:
    uvicorn intrupt_py_sdk.example.smtp_email_agent:app --port 8089

Test:
    curl -X POST http://localhost:8089/call-tool \\
         -H 'Content-Type: application/json' \\
         -d '{"message": "Pay invoice INV-007 for $1500 to Acme Corp"}'

Then click the Approve or Reject link in the email that arrives.
"""

import asyncio
import os
import smtplib
import ssl
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
APPROVER_EMAIL = os.getenv("APPROVER_EMAIL", "")
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8089")

_pending: dict[str, str] = {}  # approval_id -> thread_id
_lock = asyncio.Lock()


def _send_email_sync(to: str, subject: str, html: str) -> None:
    """Blocking SMTP send — called inside run_in_executor."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to
    msg.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, to, msg.as_string())


async def smtp_email_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    kwargs = v.get("tool", {}).get("kwargs", {})

    approve_url = f"{AGENT_PUBLIC_URL}/decide?approval_id={approval_id}&approved=true"
    reject_url = f"{AGENT_PUBLIC_URL}/decide?approval_id={approval_id}&approved=false"

    rows = "".join(f"<tr><td><b>{k}</b></td><td>{val}</td></tr>" for k, val in kwargs.items())
    html = f"""
    <html><body style="font-family:sans-serif;max-width:600px;margin:auto">
      <h2 style="color:#c0392b">&#9888; Action requires your approval</h2>
      <p><b>Message:</b> {v.get("message")}</p>
      <p><b>Tool:</b> {v.get("tool", {}).get("name")}</p>
      <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f2f2f2"><th>Parameter</th><th>Value</th></tr>
        {rows}
      </table>
      <br>
      <a href="{approve_url}"
         style="background:#27ae60;color:white;padding:12px 24px;text-decoration:none;border-radius:4px;margin-right:12px">
        ✅ Approve
      </a>
      <a href="{reject_url}"
         style="background:#c0392b;color:white;padding:12px 24px;text-decoration:none;border-radius:4px">
        ❌ Reject
      </a>
      <p style="color:#888;font-size:12px;margin-top:24px">thread_id: {thread_id}</p>
    </body></html>
    """

    async with _lock:
        _pending[approval_id] = thread_id

    if SMTP_USER and APPROVER_EMAIL:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _send_email_sync, APPROVER_EMAIL, f"[Approval Required] {v.get('action')}", html
        )
        print(f"[smtp] approval email sent to {APPROVER_EMAIL} (approval_id={approval_id})")
    else:
        print(
            f"\n[smtp] SMTP not configured — approve manually:\n"
            f"  {approve_url}\n"
            f"  {reject_url}\n"
        )

    return {"approval_id": approval_id}


# ── Graph ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_invoice(invoice_id: str) -> dict:
    """Look up an invoice by ID."""
    return {"invoice_id": invoice_id, "vendor": "Acme Corp", "amount": 1500.00, "status": "unpaid"}


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

approval_graph = ApprovalGraph(graph=graph, on_approval_async=smtp_email_approval)

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="SMTP Email Approval Agent")


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


@app.get("/decide", response_class=HTMLResponse)
async def decide(approval_id: str, approved: str):
    """Browser-friendly endpoint linked from the email."""
    async with _lock:
        thread_id = _pending.pop(approval_id, None)
    if thread_id is None:
        return HTMLResponse("<h2>Already decided or unknown approval.</h2>", status_code=404)

    decision = approved.lower() in ("true", "1", "yes")
    await approval_graph.aresume(thread_id, approved=decision, approval_id=approval_id)

    verdict = "approved ✅" if decision else "rejected ❌"
    return HTMLResponse(
        f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
        f"<h2>Action {verdict}</h2>"
        f"<p>The agent has been notified and will continue.</p>"
        f"</body></html>"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089)
