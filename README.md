# intrupt-py-sdk

Python SDK for adding human-in-the-loop approval gates to LangGraph agents.

## Overview

`intrupt-py-sdk` provides the `ApprovalGraph` wrapper and the `@approval_required` decorator. Together they intercept `interrupt()` events inside your LangGraph graph, dispatch them to a notification channel (Slack, email, webhook, or your own logic), and resume the agent once a human decides.

The SDK is layered:

| Layer | What it does |
|---|---|
| `intrupt_py_sdk.core.client.ApprovalClient` | HTTP client for the intrupt approval API |
| `intrupt_py_sdk.adapters.approval_middleware.ApprovalMiddleware` | Process-wide singleton for the client |
| `intrupt_py_sdk.adapters.langgraph.ApprovalGraph` | Wraps a compiled `StateGraph`; handles interrupt detection and resume |
| `intrupt_py_sdk.adapters.langgraph.approval_required` | Decorator that makes any LangGraph tool pause for human approval |

## Installation

```bash
pip install intrupt-py-sdk
# or with uv:
uv add intrupt-py-sdk
```

With test dependencies:

```bash
pip install intrupt-py-sdk[test]
```

---

## Quick Start

### 1. Decorate your tool

```python
from langchain_core.tools import tool
from intrupt_py_sdk.adapters.langgraph import approval_required

@tool
@approval_required(
    action="transfer_funds",
    message="Review and approve this transfer before funds move",
    channel="slack",
    args=["account", "amount"],      # kwargs forwarded to the approver
)
def transfer_funds(account: str, amount: float) -> dict:
    """Transfer funds to an external account."""
    return bank.transfer(account, amount)
```

### 2. Wrap your graph with `ApprovalGraph`

```python
from intrupt_py_sdk.adapters.langgraph import ApprovalGraph
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

# Initialise the approval client once at startup
ApprovalMiddleware(
    base_url="https://api.aegmis.com",
    api_key="sk_org_org_xxxx_yyyy",
)

approval_graph = ApprovalGraph(
    graph=compiled_graph,
    client=ApprovalMiddleware.get_client(),
    callback_url="https://your-agent.example.com/resume",
    callback_secret=os.getenv("AGENT_RESUME_SECRET", ""),
)
```

### 3. Invoke and handle the response

```python
result = approval_graph.invoke(
    {"messages": [{"role": "user", "content": "Transfer $500 to acct-123"}]},
    thread_id="thread-abc",
)

if result["status"] == "pending_approval":
    # A human has been notified; store result["approval_id"]
    print("Waiting for approval:", result["approval_id"])
elif result["status"] == "complete":
    # Full graph state is in result["result"]; messages are also surfaced top-level
    print(result["messages"])
    print(result["result"])   # all state fields, e.g. {"messages": [...], "last_purchase": {...}}

# Later — after the human clicks Approve/Reject in Slack:
result = approval_graph.resume(thread_id="thread-abc", approved=True)
```

---

## ApprovalGraph

`ApprovalGraph` is the main entry point. It wraps any compiled LangGraph `StateGraph` and transparently intercepts `interrupt()` events fired by `@approval_required`.

### Construction

```python
ApprovalGraph(
    graph,                     # compiled StateGraph with MemorySaver checkpointer
    client=None,               # ApprovalClient — required if on_approval is not set
    callback_url="",           # agent's /resume endpoint URL
    callback_secret="",        # AGENT_RESUME_SECRET echoed back on callback
    on_approval=None,          # custom sync callable (replaces client)
    on_approval_async=None,    # custom async callable (replaces client)
)
```

Provide either `client` **or** `on_approval`/`on_approval_async` — not both:

| Scenario | Constructor args |
|---|---|
| Use intrupt API (default) | `client=..., callback_url=..., callback_secret=...` |
| Custom sync channel | `on_approval=my_fn` |
| Custom async channel | `on_approval_async=my_async_fn` |

### Methods

```python
# Sync
result = approval_graph.invoke(input_dict, thread_id, config=None)
result = approval_graph.resume(thread_id, approved, approval_id=None, config=None)

# Async
result = await approval_graph.ainvoke(input_dict, thread_id, config=None)
result = await approval_graph.aresume(thread_id, approved, approval_id=None, config=None)

# Streaming (yields raw graph chunks)
for chunk in approval_graph.stream(input_dict, thread_id, config=None):
    ...
async for chunk in approval_graph.astream(input_dict, thread_id, config=None):
    ...

# State inspection / mutation
state    = approval_graph.get_state(thread_id)
approval_graph.update_state(thread_id, values, as_node=None)

# Check pending approval
is_waiting = approval_graph.pending(thread_id)  # -> bool
```

All invocation methods accept an optional `config` dict that is **merged** with the internal LangGraph config. You can use it to pass `recursion_limit`, `tags`, `metadata`, extra `configurable` keys (e.g. model choice), or your own callbacks (e.g. a LangSmith tracer):

```python
from langsmith import traceable

result = approval_graph.invoke(
    {"messages": [{"role": "user", "content": "Transfer $500 to acct-123"}]},
    thread_id="thread-abc",
    config={
        "recursion_limit": 50,
        "tags": ["prod", "finance"],
        "callbacks": [my_langsmith_handler],
        "configurable": {"model": "gpt-4o"},
    },
)
```

Your `configurable` keys are merged with the internal ones; `thread_id` always wins. Your `callbacks` are appended to the internal list so the approval handler is never dropped.

### Response shape

```python
# Tool interrupted — human notification sent
{"status": "pending_approval", "thread_id": "...", "approval_id": "..."}

# Graph ran to completion (or resumed and finished)
{
    "status": "complete",
    "thread_id": "...",
    "result": { ... },              # full graph state (all state fields)
    "messages": [{"type": "...", "content": "..."}],
}
```

`result` contains the raw graph state dict, so custom state fields (e.g. `last_purchase`, `invoice_id`) are accessible alongside `messages`.

### Streaming

`stream` / `astream` yield raw LangGraph chunks. When the graph pauses on an approval interrupt, a final sentinel chunk with the key `__approval__` is emitted:

```python
for chunk in approval_graph.stream(input_dict, thread_id):
    if "__approval__" in chunk:
        approval_info = chunk["__approval__"]
        # {"status": "pending_approval", "thread_id": "...", "approval_id": "..."}
    else:
        process(chunk)   # normal graph output
```

### State access

```python
# Read current graph state (e.g. to inspect messages after resume)
state = approval_graph.get_state(thread_id)

# Inject values directly into the checkpoint (useful for tests or manual corrections)
approval_graph.update_state(thread_id, {"last_purchase": None})
# As a specific node:
approval_graph.update_state(thread_id, {"messages": [...]}, as_node="chat_node")
```

---

## `@approval_required` decorator

Wraps a LangGraph `@tool` function so it pauses for human approval before executing.

```python
@tool
@approval_required(
    action="pay_invoice",              # label shown to the approver
    message="Approve this payment",    # human-readable description
    channel="slack",                   # hint for routing (informational)
    args=["invoice_id", "amount"],     # kwargs forwarded to the approval payload
)
def pay_invoice(invoice_id: str, amount: float) -> dict:
    ...
```

When the decorated tool is called:
1. `interrupt(payload)` fires — the graph checkpoints and pauses
2. `ApprovalGraph.on_interrupt` callback triggers — your `on_approval` callable is called
3. If `approved=True` on resume — the original tool body executes
4. If `approved=False` — returns `{"status": "cancelled", "tool": "...", "message": "not approved"}`

---

## Custom Approval Channels

Instead of the intrupt API, pass your own callable:

```python
def on_approval(thread_id: str, v: dict) -> dict:
    """
    Called synchronously the moment interrupt() fires.

    v keys:  approval_required, action, message, channel, tool
             tool = {"name": "...", "kwargs": {...}}

    Return:
      {"approval_id": "..."}   → response is {"status": "pending_approval", ...}
      {}                        → response is {"status": "complete", ...}
    """
    ...
    return {"approval_id": stored_id}
```

### Channel examples

#### Console (interactive stdin)

```python
import asyncio, uuid

_decisions: dict[str, bool] = {}

async def console_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, input, f"Approve {v['action']}? [y/n]: ")
    _decisions[approval_id] = answer.strip().lower() in ("y", "yes")
    return {"approval_id": approval_id}

approval_graph = ApprovalGraph(graph=graph, on_approval_async=console_approval)

# In /call-tool endpoint — auto-resume immediately after stdin:
result = await approval_graph.ainvoke(input_dict, thread_id)
if result["status"] == "pending_approval":
    aid = result["approval_id"]
    approved = _decisions.pop(aid, False)
    result = await approval_graph.aresume(thread_id, approved=approved, approval_id=aid)
```

#### Policy engine (auto-approve / auto-reject by rule)

```python
BLOCKED_VENDORS = {"BlockedCorp"}
_auto: dict[str, bool] = {}
_pending: dict[str, str] = {}

async def policy_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    kwargs = v.get("tool", {}).get("kwargs", {})
    amount = float(kwargs.get("amount", 0))
    vendor = kwargs.get("vendor", "")

    if vendor in BLOCKED_VENDORS or amount > 50_000:
        _auto[approval_id] = False
    elif amount < 500:
        _auto[approval_id] = True
    else:
        _pending[approval_id] = thread_id   # escalate to human

    return {"approval_id": approval_id}

# In /call-tool endpoint:
result = await approval_graph.ainvoke(input_dict, thread_id)
if result["status"] == "pending_approval":
    aid = result["approval_id"]
    if aid in _auto:
        result = await approval_graph.aresume(thread_id, approved=_auto.pop(aid), approval_id=aid)
```

#### SMTP email

```python
import smtplib, asyncio, uuid
from email.mime.text import MIMEText

_pending: dict[str, str] = {}

async def smtp_email_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    _pending[approval_id] = thread_id

    approve_url = f"{BASE_URL}/decide?approval_id={approval_id}&approved=true"
    reject_url  = f"{BASE_URL}/decide?approval_id={approval_id}&approved=false"
    body = f"Action: {v['action']}\n\nApprove: {approve_url}\nReject: {reject_url}"
    msg = MIMEText(body)
    msg["Subject"] = f"[Approval Required] {v['action']}"
    msg["From"] = SMTP_USER
    msg["To"]   = APPROVER_EMAIL

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send, msg)
    return {"approval_id": approval_id}

def _send(msg):
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASSWORD)
        s.sendmail(SMTP_USER, APPROVER_EMAIL, msg.as_string())
```

#### Slack (direct Block Kit message)

```python
from slack_sdk import WebClient
import uuid, asyncio

_slack = WebClient(token=SLACK_BOT_TOKEN)
_pending: dict[str, str] = {}

async def slack_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    _pending[approval_id] = thread_id

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _post_block_kit, approval_id, v)
    return {"approval_id": approval_id}

def _post_block_kit(approval_id: str, v: dict):
    _slack.chat_postMessage(
        channel=SLACK_CHANNEL_ID,
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{v['action']}* requires approval"}},
            {"type": "actions", "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "Approve"},
                 "style": "primary", "value": f"approve:{approval_id}", "action_id": "approve"},
                {"type": "button", "text": {"type": "plain_text", "text": "Reject"},
                 "style": "danger",  "value": f"reject:{approval_id}",  "action_id": "reject"},
            ]},
        ],
    )

# /slack/actions endpoint handles button callbacks:
# parse payload -> value = "approve:{id}" or "reject:{id}"
# -> approval_graph.aresume(thread_id, approved=True/False)
```

#### Telegram Bot

```python
import httpx, uuid

_TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
_pending: dict[str, str] = {}

async def telegram_approval(thread_id: str, v: dict) -> dict:
    approval_id = str(uuid.uuid4())
    _pending[approval_id] = thread_id

    async with httpx.AsyncClient() as http:
        await http.post(f"{_TG_API}/sendMessage", json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"*{v['action']}* requires approval",
            "parse_mode": "Markdown",
            "reply_markup": {"inline_keyboard": [[
                {"text": "Approve", "callback_data": f"approve:{approval_id}"},
                {"text": "Reject",  "callback_data": f"reject:{approval_id}"},
            ]]},
        })
    return {"approval_id": approval_id}

# /telegram/webhook endpoint:
# callback_query.data = "approve:{id}" or "reject:{id}"
# -> approval_graph.aresume(thread_id, approved=True/False)
```

---

## Async Usage

All `ApprovalGraph` methods have async equivalents. Use them with `async def` FastAPI handlers:

```python
approval_graph = ApprovalGraph(graph=graph, on_approval_async=my_async_fn)

@app.post("/call-tool")
async def call_tool(request: Request):
    payload = await request.json()
    thread_id = str(uuid.uuid4())
    result = await approval_graph.ainvoke(
        {"messages": [{"role": "user", "content": payload["message"]}]},
        thread_id,
        config={"tags": ["prod"]},   # optional — merged with internal config
    )
    return result

@app.post("/resume")
async def resume(request: Request):
    payload = await request.json()
    return await approval_graph.aresume(
        payload["thread_id"],
        approved=payload["approved"],
        approval_id=payload.get("approval_id"),
    )
```

### Streaming with FastAPI (`StreamingResponse`)

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/call-tool-stream")
async def call_tool_stream(request: Request):
    payload = await request.json()
    thread_id = payload.get("thread_id") or str(uuid.uuid4())

    async def generate():
        async for chunk in approval_graph.astream(
            {"messages": [{"role": "user", "content": payload["message"]}]},
            thread_id,
        ):
            yield json.dumps(chunk) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

---

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `APPROVAL_BASE_URL` | `""` | Base URL of the intrupt approval API |
| `APPROVAL_API_KEY` | `""` | Bearer token (format: `sk_org_{org_id}_{hash}`) |
| `AGENT_RESUME_SECRET` | `""` | HMAC secret echoed on the `/resume` callback |

### ApprovalMiddleware singleton

```python
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

# Call once at startup — anywhere in your app:
ApprovalMiddleware(
    base_url="https://api.aegmis.com",
    api_key=os.getenv("APPROVAL_API_KEY"),
)

# Retrieve the client elsewhere without re-passing credentials:
client = ApprovalMiddleware.get_client()
```

---

## Complete Example Agent

`example/agent.py` is a full FastAPI + LangGraph agent with:
- Two tools (`get_stock_price`, `purchase_stock`)
- `purchase_stock` guarded by `@approval_required`
- `/call-tool` endpoint that starts or continues a run
- `/resume` endpoint that accepts the human decision and resumes the graph
- 409 guard: rejects new messages on threads with a pending approval
- `AGENT_RESUME_SECRET` authentication on `/resume` (skipped when env var is empty)
- Leading-`ToolMessage` trim in `chat_node` to prevent OpenAI errors after server restart mid-approval

```bash
# Run the agent (requires .env with OPENAI_API_KEY, APPROVAL_BASE_URL, APPROVAL_API_KEY)
uv run python example/agent.py

# Start a run
curl -X POST http://localhost:8081/call-tool \
     -H 'Content-Type: application/json' \
     -d '{"message": "buy 10 shares of AAPL"}'

# Approve and resume
curl -X POST http://localhost:8081/resume \
     -H 'Content-Type: application/json' \
     -H 'X-Agent-Secret: your-secret' \
     -d '{"thread_id": "...", "approval_id": "...", "approved": true}'
```

Other example agents in `example/`:

| File | Port | Approval channel |
|---|---|---|
| `agent.py` | 8081 | intrupt API → Slack (default) |
| `console_agent.py` | 8087 | Interactive stdin |
| `policy_agent.py` | 8088 | Rule-based auto-approve/reject |
| `smtp_email_agent.py` | 8089 | SMTP email with approve/reject links |
| `slack_direct_agent.py` | 8090 | Direct Slack Block Kit messages |
| `telegram_agent.py` | 8091 | Telegram Bot inline keyboard |

---

## Development

```bash
# Install with dev deps
uv sync --extra test

# Install SDK in editable mode
uv pip install -e .

# Run tests
uv run pytest -v

# Run a specific example
uv run python example/console_agent.py
```
