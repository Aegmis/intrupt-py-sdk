# intrupt-py-sdk

Python SDK for adding human-in-the-loop approval gates to AI agents.

Supports **LangGraph**, **Google ADK**, **OpenAI Agents SDK**, and **CrewAI** — all sharing the same `gate.py` Future pattern and the same approval channel abstractions.

## Overview

```
Your agent tool
    └── @approval_required(...)      ← decorator from the adapter
            │  fires gate.request_approval(client, session_id, payload)
            │  suspends on a Future
            ▼
    on_approval_async / ApprovalClient.acreate_approval
            │  notifies the human (Slack, email, Telegram, console, …)
            │  returns {"approval_id": "..."}
            ▼
    Human clicks Approve / Reject
            │  your /resume (or /decide) endpoint is called
            ▼
    gate.resolve(approval_id, approved=True)
            │  unblocks the Future
            ▼
    Tool body executes (or returns cancelled)
```

The SDK is layered:

| Layer | What it does |
|---|---|
| `intrupt_py_sdk.core.gate` | Framework-agnostic Future registry — `request_approval` / `resolve` |
| `intrupt_py_sdk.core.client.ApprovalClient` | HTTP client for the intrupt approval API |
| `intrupt_py_sdk.adapters.approval_middleware.ApprovalMiddleware` | Process-wide singleton for the HTTP client |
| `intrupt_py_sdk.adapters.langgraph` | `ApprovalGraph` wrapper + `@approval_required` for LangGraph |
| `intrupt_py_sdk.adapters.google_adk` | `ApprovalRunner` + `@approval_required` for Google ADK |
| `intrupt_py_sdk.adapters.openai_agents` | `ApprovalAgentRunner` + `@approval_required` for OpenAI Agents SDK |
| `intrupt_py_sdk.adapters.crewai` | `ApprovalCrew` + `approval_required()` wrapper for CrewAI |

---

## Installation

```bash
pip install intrupt-py-sdk
# or with uv:
uv add intrupt-py-sdk
```

With test dependencies:

```bash
pip install "intrupt-py-sdk[test]"
```

---

## LangGraph Adapter

### Quick start

```python
import os
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from intrupt_py_sdk.adapters.langgraph import approval_required, ApprovalGraph

# 1. Decorate the tool that needs approval
@tool
@approval_required(
    action="purchase_stock",
    message="Review and approve this purchase before funds move",
    channel="slack",
    args=["symbol", "quantity", "amount"],   # kwargs forwarded to the approver
)
async def purchase_stock(symbol: str, quantity: int, amount: float) -> dict:
    """Buy shares of a stock."""
    return {"status": "success", "symbol": symbol, "quantity": quantity}

# 2. Build a standard LangGraph
memory = MemorySaver()
graph = (
    StateGraph(AgentState)
    .add_node("chat", chat_node)
    .add_node("tools", ToolNode([purchase_stock]))
    .add_edge(START, "chat")
    .add_conditional_edges("chat", route)
    .add_edge("tools", "chat")
    .compile(checkpointer=memory)
)

# 3. Wrap with ApprovalGraph — HTTP flow (intrupt API)
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

ApprovalMiddleware(
    base_url="https://api.intrupt.dev",
    api_key=os.getenv("APPROVAL_API_KEY"),
)

approval_graph = ApprovalGraph(
    graph=graph,
    callback_url="https://your-agent.example.com/resume",
    callback_secret=os.getenv("AGENT_RESUME_SECRET", ""),
    timeout=1.5,   # seconds to wait before returning pending_approval
)

# 4. Invoke
result = await approval_graph.ainvoke(
    {"messages": [{"role": "user", "content": "Buy 10 shares of AAPL"}]},
    thread_id="thread-abc",
)
# {"status": "pending_approval", "thread_id": "...", "approval_id": "..."}

# 5. Later — after the human approves
result = await approval_graph.aresume(
    thread_id="thread-abc",
    approved=True,
    approval_id="...",
)
# {"status": "complete", "thread_id": "...", "result": {...}, "messages": [...]}
```

### `ApprovalGraph` constructor

```python
ApprovalGraph(
    graph,                     # compiled StateGraph with MemorySaver checkpointer
    callback_url="",           # your agent's /resume endpoint URL
    callback_secret="",        # echoed in X-Agent-Secret on the resume callback
    on_approval_async=None,    # async (thread_id, payload) -> {"approval_id": "..."}
                               # use instead of the HTTP API for inline channels
    timeout=1.5,               # seconds to wait before returning pending_approval
    client=None,               # deprecated — call ApprovalMiddleware(...) at startup instead
)
```

Provide either `callback_url` (HTTP flow via intrupt API) **or** `on_approval_async` (inline callback) — not both.

| Scenario | Constructor args |
|---|---|
| intrupt approval API | `callback_url=..., callback_secret=...` + `ApprovalMiddleware(...)` at startup |
| Custom async channel (console, email, Slack, Telegram, policy) | `on_approval_async=my_async_fn` |

### Methods

```python
# Async (preferred inside FastAPI / async code)
result = await approval_graph.ainvoke(input_dict, thread_id, config=None)
result = await approval_graph.aresume(thread_id, approved, approval_id="")

# Sync aliases (same underlying logic)
result = await approval_graph.run(input_dict, thread_id, config=None)
result = await approval_graph.resume(thread_id, approved, approval_id="")

# State
state    = approval_graph.get_state(thread_id)
approval_graph.update_state(thread_id, values, as_node=None)
waiting  = approval_graph.pending(thread_id)   # -> bool
```

Pass an optional `config` dict to merge with the internal LangGraph config:

```python
result = await approval_graph.ainvoke(
    {"messages": [...]},
    thread_id="t1",
    config={
        "recursion_limit": 50,
        "tags": ["prod"],
        "configurable": {"model": "gpt-4o"},
    },
)
```

### Response shape

```python
# Tool intercepted — human notified
{"status": "pending_approval", "thread_id": "...", "approval_id": "..."}

# Graph finished (or resumed and finished)
{
    "status": "complete",
    "thread_id": "...",
    "result": {...},              # full graph state dict
    "messages": [{"type": "AIMessage", "content": "..."}],
}
```

### `@approval_required` decorator

```python
@tool
@approval_required(
    action="pay_invoice",              # label shown to the approver
    message="Approve this payment",    # human-readable reason
    channel="slack",                   # routing hint (informational)
    args=["invoice_id", "amount"],     # kwargs forwarded in the approval payload
)
def pay_invoice(invoice_id: str, amount: float) -> dict:
    ...
```

When the tool is called:
1. The approval payload is dispatched to your channel (via `on_approval_async` or the HTTP API).
2. The tool suspends on an asyncio Future.
3. On resume with `approved=True` — the original tool body runs.
4. On resume with `approved=False` — returns `{"status": "cancelled", "tool": "...", "message": "not approved"}`.

### Inline approval channels (`on_approval_async`)

Pass an async callback instead of the HTTP API. The callback receives `(thread_id, payload)` and must return `{"approval_id": "..."}`.

```python
async def my_channel(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    # ... notify the human somehow ...
    return {"approval_id": approval_id}

approval_graph = ApprovalGraph(graph=graph, on_approval_async=my_channel)
```

See the [Approval Channel Patterns](#approval-channel-patterns) section for ready-made examples.

---

## Google ADK Adapter

```python
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from intrupt_py_sdk.adapters.google_adk import approval_required, ApprovalRunner
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

# 1. Gate the tool function (must be async; ADK injects tool_context)
@approval_required(
    action="purchase_stock",
    message="Approve stock purchase?",
    channel="slack",
    args=["symbol", "quantity"],
)
async def purchase_stock(symbol: str, quantity: int, tool_context=None) -> str:
    """Buy shares of a stock."""
    return f"Purchased {quantity} shares of {symbol}"

# 2. Build the ADK agent
agent = LlmAgent(
    name="finance-bot",
    model="gemini-2.0-flash",
    tools=[purchase_stock],
)
session_service = InMemorySessionService()

# 3. Wire up the approval middleware
ApprovalMiddleware(
    base_url="https://api.intrupt.dev",
    api_key=os.getenv("APPROVAL_API_KEY"),
)

# 4. Wrap with ApprovalRunner
runner = ApprovalRunner(
    agent=agent,
    app_name="finance-bot",
    session_service=session_service,
    callback_url="https://your-agent.example.com/resume",
    callback_secret=os.getenv("AGENT_RESUME_SECRET", ""),
)

# 5. Run
result = await runner.run(session_id, "Buy 10 AAPL shares")
# {"status": "pending_approval", "session_id": "...", "approval_id": "..."}

result = await runner.resume(session_id, approved=True, approval_id="...")
# {"status": "complete", "session_id": "...", "result": "..."}
```

**Key difference from LangGraph:** ADK injects `tool_context` as a kwarg; the decorator reads `tool_context.invocation_context.session.id` to identify the session. `ApprovalRunner` uses `session_id` instead of `thread_id`. No `@tool` wrapper needed.

---

## OpenAI Agents SDK Adapter

```python
from agents import Agent, function_tool
from intrupt_py_sdk.adapters.openai_agents import approval_required, ApprovalAgentRunner
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

# 1. Gate the tool (apply approval_required inside @function_tool)
@function_tool
@approval_required(
    action="purchase_stock",
    message="Approve stock purchase?",
    channel="slack",
    args=["symbol", "quantity"],
)
async def purchase_stock(symbol: str, quantity: int) -> str:
    """Buy shares of a stock."""
    return f"Purchased {quantity} shares of {symbol}"

# 2. Build the OpenAI Agent
agent = Agent(
    name="Finance Bot",
    model="gpt-4o-mini",
    tools=[purchase_stock],
)

# 3. Wire up the approval middleware
ApprovalMiddleware(
    base_url="https://api.intrupt.dev",
    api_key=os.getenv("APPROVAL_API_KEY"),
)

# 4. Wrap with ApprovalAgentRunner
runner = ApprovalAgentRunner(
    agent=agent,
    callback_url="https://your-agent.example.com/resume",
    callback_secret=os.getenv("AGENT_RESUME_SECRET", ""),
)

# 5. Run
result = await runner.run(thread_id, "Buy 10 AAPL shares")
# {"status": "pending_approval", "thread_id": "...", "approval_id": "..."}

result = await runner.resume(thread_id, approved=True, approval_id="...")
# {"status": "complete", "thread_id": "...", "result": "..."}
```

**Key difference from LangGraph:** Apply `@approval_required` inside `@function_tool` (not `@tool`). Uses `Runner.run(agent, message)` under the hood. Thread ID flows via a `contextvars.ContextVar` set before the background task starts.

---

## CrewAI Adapter

```python
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from intrupt_py_sdk.adapters.crewai import approval_required, ApprovalCrew
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

# 1. Define a BaseTool
class PurchaseTool(BaseTool):
    name: str = "purchase_stock"
    description: str = "Buy shares of a stock."

    def _run(self, symbol: str, quantity: int) -> str:
        return f"Purchased {quantity} shares of {symbol}"

# 2. Wrap it with approval_required (a factory function, not a decorator)
gated_purchase = approval_required(
    PurchaseTool(),
    action="purchase_stock",
    message="Approve stock purchase?",
    channel="slack",
    args=["symbol", "quantity"],
)

# 3. Build the Crew
finance_agent = Agent(
    role="Finance Analyst",
    goal="Execute stock trades",
    tools=[gated_purchase],
)
task = Task(
    description="Buy 10 shares of AAPL",
    agent=finance_agent,
    expected_output="Trade confirmation",
)
crew = Crew(agents=[finance_agent], tasks=[task])

# 4. Wire up the approval middleware
ApprovalMiddleware(
    base_url="https://api.intrupt.dev",
    api_key=os.getenv("APPROVAL_API_KEY"),
)

# 5. Wrap with ApprovalCrew
approval_crew = ApprovalCrew(
    crew=crew,
    callback_url="https://your-agent.example.com/resume",
    callback_secret=os.getenv("AGENT_RESUME_SECRET", ""),
)

# 6. Kickoff
result = await approval_crew.kickoff(run_id, inputs={"request": "buy 10 AAPL"})
# {"status": "pending_approval", "run_id": "...", "approval_id": "..."}

result = await approval_crew.resume(run_id, approved=True, approval_id="...")
# {"status": "complete", "run_id": "...", "result": "..."}
```

**Key difference from LangGraph:** `approval_required` is a factory function (not a decorator) — it wraps a `BaseTool` instance and returns a new gated `BaseTool`. Uses `run_id` (matching CrewAI's kickoff semantics) and `crew.kickoff_async(inputs=...)` internally.

---

## Approval Channel Patterns

All channels use the `on_approval_async` parameter of `ApprovalGraph` (or the equivalent `ApprovalMiddleware` HTTP path). The callback signature is:

```python
async def my_channel(thread_id: str, payload: dict) -> dict:
    """
    payload keys: action, message, channel, tool
    tool = {"name": "...", "kwargs": {...}, "description": "..."}

    Return: {"approval_id": "..."}
    """
```

### Console (interactive stdin)

```python
import asyncio, uuid

_decisions: dict[str, bool] = {}

async def console_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, input, f"\nApprove '{v['action']}'? [y/n]: "
    )
    _decisions[approval_id] = answer.strip().lower() in ("y", "yes")
    return {"approval_id": approval_id}

approval_graph = ApprovalGraph(graph=graph, on_approval_async=console_approval)

# FastAPI handler — auto-resume immediately after stdin:
@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    thread_id = str(uuid.uuid4())
    result = await approval_graph.ainvoke(
        {"messages": [{"role": "user", "content": payload["message"]}]},
        thread_id,
    )
    if result["status"] == "pending_approval":
        aid = result["approval_id"]
        approved = _decisions.pop(aid, False)
        result = await approval_graph.aresume(thread_id, approved=approved, approval_id=aid)
    return result
```

See `example/console_agent.py` (port 8087).

### Policy engine (auto-approve / auto-reject by rule)

```python
import uuid

BLOCKED_VENDORS = {"BlockedCorp"}
_auto: dict[str, bool] = {}
_pending: dict[str, str] = {}   # approval_id -> thread_id (escalated to human)

async def policy_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    kwargs = v.get("tool", {}).get("kwargs", {})
    amount = float(kwargs.get("amount", 0))
    vendor = kwargs.get("vendor", "")

    if vendor in BLOCKED_VENDORS or amount > 50_000:
        _auto[approval_id] = False       # auto-reject
    elif amount < 500:
        _auto[approval_id] = True        # auto-approve
    else:
        _pending[approval_id] = thread_id   # escalate to human via /decide

    return {"approval_id": approval_id}

approval_graph = ApprovalGraph(graph=graph, on_approval_async=policy_approval)

# FastAPI handler:
@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    thread_id = str(uuid.uuid4())
    result = await approval_graph.ainvoke(
        {"messages": [{"role": "user", "content": payload["message"]}]},
        thread_id,
    )
    if result["status"] == "pending_approval":
        aid = result["approval_id"]
        if aid in _auto:
            result = await approval_graph.aresume(
                thread_id, approved=_auto.pop(aid), approval_id=aid
            )
    return result

@app.post("/decide")
async def decide(request: Request):
    body = await request.json()
    aid = body["approval_id"]
    thread_id = _pending.pop(aid, None)
    if thread_id is None:
        raise HTTPException(status_code=404, detail="unknown approval_id")
    return await approval_graph.aresume(thread_id, approved=body["approved"], approval_id=aid)
```

See `example/policy_agent.py` (port 8088).

### SMTP email (one-click approve/reject links)

```python
import asyncio, smtplib, ssl, uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST          = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT          = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER          = os.getenv("SMTP_USER", "")
SMTP_PASS          = os.getenv("SMTP_PASS", "")          # app password
APPROVAL_EMAIL_TO  = os.getenv("APPROVAL_EMAIL_TO", "")
AGENT_PUBLIC_URL   = os.getenv("AGENT_PUBLIC_URL", "http://localhost:8089")

_pending: dict[str, str] = {}   # approval_id -> thread_id

def _send_email_sync(to: str, subject: str, html: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to
    msg.attach(MIMEText(html, "html"))
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_USER, to, msg.as_string())

async def smtp_email_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    _pending[approval_id] = thread_id
    approve_url = f"{AGENT_PUBLIC_URL}/decide?approval_id={approval_id}&approved=true"
    reject_url  = f"{AGENT_PUBLIC_URL}/decide?approval_id={approval_id}&approved=false"
    html = f"""
    <p><b>{v['action']}</b> requires approval</p>
    <a href="{approve_url}">✅ Approve</a>  |  <a href="{reject_url}">❌ Reject</a>
    """
    if SMTP_USER and APPROVAL_EMAIL_TO:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _send_email_sync, APPROVAL_EMAIL_TO,
            f"[Approval Required] {v['action']}", html,
        )
    return {"approval_id": approval_id}

approval_graph = ApprovalGraph(graph=graph, on_approval_async=smtp_email_approval)

@app.get("/decide")
async def decide(approval_id: str, approved: str):
    thread_id = _pending.pop(approval_id, None)
    if thread_id is None:
        raise HTTPException(status_code=404)
    decision = approved.lower() in ("true", "1", "yes")
    await approval_graph.aresume(thread_id, approved=decision, approval_id=approval_id)
    return {"status": "decided", "approved": decision}
```

Uses `SMTP_SSL` on port 465. See `example/smtp_email_agent.py` (port 8089).

### Slack (direct Block Kit message)

```python
from slack_sdk import WebClient
import asyncio, uuid

_slack = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")
_pending: dict[str, str] = {}

def _post_blocks(approval_id: str, v: dict) -> None:
    _slack.chat_postMessage(
        channel=SLACK_CHANNEL_ID,
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{v['action']}* requires approval\n{v['message']}"},
            },
            {
                "type": "actions",
                "elements": [
                    {"type": "button", "text": {"type": "plain_text", "text": "Approve"},
                     "style": "primary", "value": f"approve:{approval_id}", "action_id": "approve"},
                    {"type": "button", "text": {"type": "plain_text", "text": "Reject"},
                     "style": "danger",  "value": f"reject:{approval_id}",  "action_id": "reject"},
                ],
            },
        ],
    )

async def slack_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    _pending[approval_id] = thread_id
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _post_blocks, approval_id, v)
    return {"approval_id": approval_id}

approval_graph = ApprovalGraph(graph=graph, on_approval_async=slack_approval)

# Slack Interactivity → Request URL must point to this endpoint:
@app.post("/slack/actions")
async def slack_actions(request: Request):
    import json
    form = await request.form()
    payload = json.loads(form["payload"])
    action = payload["actions"][0]
    kind, approval_id = action["value"].split(":", 1)
    thread_id = _pending.pop(approval_id, None)
    if thread_id:
        await approval_graph.aresume(thread_id, approved=(kind == "approve"), approval_id=approval_id)
    return {"ok": True}
```

See `example/slack_direct_agent.py` (port 8090).

### Telegram Bot

```python
import httpx, uuid

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
_TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
_pending: dict[str, str] = {}

async def telegram_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    _pending[approval_id] = thread_id
    async with httpx.AsyncClient() as http:
        await http.post(f"{_TG_API}/sendMessage", json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"*{v['action']}* requires approval\n{v['message']}",
            "parse_mode": "Markdown",
            "reply_markup": {"inline_keyboard": [[
                {"text": "✅ Approve", "callback_data": f"approve:{approval_id}"},
                {"text": "❌ Reject",  "callback_data": f"reject:{approval_id}"},
            ]]},
        })
    return {"approval_id": approval_id}

approval_graph = ApprovalGraph(graph=graph, on_approval_async=telegram_approval)

# Telegram webhook — register via:
# https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://your-agent/telegram/webhook
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    body = await request.json()
    cq = body.get("callback_query", {})
    data = cq.get("data", "")
    if ":" in data:
        kind, approval_id = data.split(":", 1)
        thread_id = _pending.pop(approval_id, None)
        if thread_id:
            await approval_graph.aresume(
                thread_id, approved=(kind == "approve"), approval_id=approval_id
            )
    return {"ok": True}
```

See `example/telegram_agent.py` (port 8091).

---

## HTTP Approval API (intrupt service)

When you want a managed approval service instead of an inline callback, use `ApprovalMiddleware` with the intrupt API. The API posts an interactive Slack (or other channel) message and calls your `/resume` endpoint when the human decides.

```python
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

# Call once at startup:
ApprovalMiddleware(
    base_url="https://api.intrupt.dev",
    api_key=os.getenv("APPROVAL_API_KEY"),
)

approval_graph = ApprovalGraph(
    graph=graph,
    callback_url="https://your-agent.example.com/resume",
    callback_secret=os.getenv("AGENT_RESUME_SECRET", ""),
)
```

Your `/resume` endpoint receives a POST from the intrupt service after the human decides:

```python
@app.post("/resume")
async def resume(request: Request):
    # Verify caller with X-Agent-Secret header if AGENT_RESUME_SECRET is set
    secret = os.getenv("AGENT_RESUME_SECRET", "")
    if secret and request.headers.get("X-Agent-Secret") != secret:
        raise HTTPException(status_code=401, detail="unauthorized")

    body = await request.json()
    if not body.get("thread_id") or body.get("approved") is None:
        raise HTTPException(status_code=400, detail="thread_id and approved required")
    if not approval_graph.pending(body["thread_id"]):
        raise HTTPException(status_code=409, detail="no pending approval for thread")

    return await approval_graph.aresume(
        body["thread_id"],
        approved=body["approved"],
        approval_id=body.get("approval_id", ""),
    )
```

See `example/agent.py` for the full reference implementation.

---

## Complete Example Agents

| File | Port | Approval channel | Framework |
|---|---|---|---|
| `example/agent.py` | 8081 | intrupt API → Slack | LangGraph |
| `example/console_agent.py` | 8087 | Interactive stdin | LangGraph |
| `example/policy_agent.py` | 8088 | Rule-based auto-approve/reject | LangGraph |
| `example/smtp_email_agent.py` | 8089 | SMTP email (SMTP_SSL + approve/reject links) | LangGraph |
| `example/slack_direct_agent.py` | 8090 | Direct Slack Block Kit messages | LangGraph |
| `example/telegram_agent.py` | 8091 | Telegram Bot inline keyboard | LangGraph |
| `example/google_adk_agent.py` | 8092 | intrupt API → Slack | Google ADK |
| `example/openai_agents_agent.py` | 8093 | intrupt API → Slack | OpenAI Agents SDK |
| `example/crewai_agent.py` | 8094 | intrupt API → Slack | CrewAI |

```bash
# Run the standard agent (requires .env with OPENAI_API_KEY + intrupt creds)
python example/agent.py

# Try it
curl -X POST http://localhost:8081/call-tool \
     -H 'Content-Type: application/json' \
     -d '{"message": "buy 10 shares of AAPL"}'

# Approve
curl -X POST http://localhost:8081/resume \
     -H 'Content-Type: application/json' \
     -H 'X-Agent-Secret: your-secret' \
     -d '{"thread_id": "...", "approval_id": "...", "approved": true}'
```

---

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `APPROVAL_BASE_URL` | `""` | Base URL of the intrupt approval API |
| `APPROVAL_API_KEY` | `""` | Bearer token (`sk_org_{org_id}_{hash}`) |
| `AGENT_RESUME_SECRET` | `""` | HMAC secret echoed on the `/resume` callback |

### `timeout` parameter

`timeout` (default `1.5` seconds) is how long `ainvoke` / `run` waits before concluding the graph hit an approval gate and returning `pending_approval`. Set it higher if your LLM or tool startup is slow:

```python
approval_graph = ApprovalGraph(graph=graph, on_approval_async=my_fn, timeout=3.0)
```

After the timeout fires, the SDK polls for up to 10 more seconds for the `approval_id` to appear (in case the HTTP call to the approval API is still in-flight). You will not see `approval_id: null` in the response under normal conditions.

### `ApprovalMiddleware` singleton

```python
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

# Initialize once at startup:
ApprovalMiddleware(base_url="...", api_key="...")

# Retrieve anywhere without re-passing credentials:
client = ApprovalMiddleware.get_client()
```

---

## Architecture notes

- **gate.py Future pattern** — `gate.request_approval(client, session_id, payload)` registers a pending `asyncio.Future` for `session_id`, calls `client.acreate_approval(thread_id=session_id, **payload)` asynchronously, and returns `(approval_id, Future)`. `gate.resolve(approval_id, approved)` sets the Future result, unblocking the decorator's `await future`. This pattern is identical across all four adapters.
- **Context vars** — Each adapter uses a `contextvars.ContextVar` (e.g. `_current_thread_id`) so concurrent requests do not share session identity. The var is set before `asyncio.create_task()` and inherited by the background task.
- **`on_approval_async` and `_OnApprovalClient`** — When `on_approval_async` is passed to `ApprovalGraph`, it is wrapped in `_OnApprovalClient`, a duck-typed adapter that exposes `acreate_approval(thread_id, **kwargs)`. This makes it compatible with `gate.py` without changing the gate or the decorator.
- **`_await_gate` polling** — After `timeout` fires, `ApprovalGraph` polls every 50 ms for up to 10 s for `gate.get_pending(thread_id)` to be populated. This avoids `approval_id: null` when the `acreate_approval` HTTP call takes longer than `timeout` to return.
- **`TestClient` context manager** — In tests, always use `with TestClient(app) as client:`. Without the context manager, each request creates a new anyio portal (new event loop), orphaning the background asyncio task spawned by `ApprovalGraph.run()` and causing `CancelledError` on resume.
- **`ApprovalMiddleware` singleton** — `_instance` is class-level. Test fixtures must reset it with `ApprovalMiddleware._instance = None` and patch `get_client().acreate_approval` on the instance (not the class) to avoid descriptor self-binding issues.

---

## Development

```bash
# Activate the in-tree venv
source .venv/bin/activate

# Run all tests
pytest -v

# Run a specific example
python example/console_agent.py

# Run the approval API
cd intrupt_api && python main.py

# Run the standard agent
python example/agent.py
```
