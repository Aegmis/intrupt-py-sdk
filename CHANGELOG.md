# intrupt_py_sdk — Changelog

---

## 2026-06-30

### Fixed: `_ensure_session` in Google ADK adapter never created the session

**File:** `intrupt_py_sdk/adapters/google_adk.py`

`InMemorySessionService.get_session` returns `None` (not an exception) when a
session does not exist. The previous `_ensure_session` implementation discarded
the return value and unconditionally hit `return`, so `create_session` was never
called. `run_async` then raised `SessionNotFoundError` because no session had
been created.

**Fix:** the return value is now captured:

```python
getter = svc.get_session(app_name=app_name, user_id="user", session_id=session_id)
existing = (await getter) if asyncio.iscoroutine(getter) else getter
if existing is not None:
    return  # session already exists

creator = svc.create_session(app_name=app_name, user_id="user", session_id=session_id)
if asyncio.iscoroutine(creator):
    await creator
```

The same fix was applied to the pip-installed copy at
`demo/.venv/lib/.../intrupt_py_sdk/adapters/google_adk.py`.

---

### Fixed: `resume()` accepted duplicate/late calls silently

**File:** `intrupt_py_sdk/adapters/google_adk.py`

If `/resume` was called after the approval had already been resolved (e.g. a
double-submit from the email link, or a race between the email callback and
another caller), `gate.resolve` popped nothing and returned silently. The caller
received `{"status": "resuming"}` even though nothing actually happened, making
the issue invisible.

**Fix:** `resume()` now checks `gate.is_pending(approval_id)` before resolving:

```python
if not gate.is_pending(approval_id):
    return {"status": "already_resolved", "session_id": ..., "approval_id": ...}
```

This makes double-submit and late-arrival safe and diagnosable.

---

### Added: SSE subscriber system and `GET /events/{session_id}` endpoint

**File:** `intrupt_py_sdk/adapters/google_adk.py`

Polling `GET /result/{session_id}` works but requires the client to loop. A
Server-Sent Events interface lets clients receive each state transition the
moment it happens, with no polling overhead.

Three new methods on `ApprovalRunner`:

- **`subscribe(session_id) → asyncio.Queue`** — registers a new SSE subscriber
  and returns a queue. Every call to `_set_result` puts the new state dict on
  all registered queues for that session.
- **`unsubscribe(session_id, queue)`** — removes the queue and cleans up the
  empty dict entry.
- **`_set_result(session_id, result)`** — replaces all direct `self._results[...]`
  assignments. Stores the result and `put_nowait`s it to every subscriber,
  ensuring SSE clients always see every transition including `pending_approval`
  (written by the gate callback) and the final `complete` / `error` (written by
  `_run_agent`'s `finally`).

The demo agent exposes `GET /events/{session_id}` using `StreamingResponse`:
- Sends current state immediately on connect (no separate `GET /result` needed)
- Streams each state dict as a `data: {...}\n\n` SSE frame
- Sends `: keepalive` comments every 25 s so proxies don't close the connection
- Closes automatically when status reaches `"complete"` or `"error"`
- Multiple clients may connect to the same session concurrently

```bash
curl -N http://localhost:8082/events/<session_id>
# data: {"status": "in_progress", "session_id": "..."}
# data: {"status": "pending_approval", "session_id": "...", "approval_id": "..."}
# data: {"status": "complete", "session_id": "...", "result": "Purchase confirmed."}
```

---

### Fixed: `_run_agent` exceptions silently dropped; `GET /result` spun as `in_progress` forever

**File:** `intrupt_py_sdk/adapters/google_adk.py`

If `_run_agent` raised an exception after approval (e.g. `TypeError` from a tool
function with a required parameter the LLM did not supply, or any ADK error),
`self._results[session_id]` was never set. Because `session_id in runner._tasks`
stayed `True` (the completed-with-exception Task object remained in the dict),
`GET /result/{session_id}` returned `{"status": "in_progress"}` indefinitely and
the exception was silently lost.

**Fix:** `_run_agent` is now wrapped in try/except with a guaranteed `finally`
block that:
1. Sets `self._results[session_id]` to either the success result or
   `{"status": "error", "error": str(exc)}`.
2. Pops `session_id` from `self._tasks` so the session is fully cleaned up.

The `finally` block also unregisters gate pending-callbacks (previously in
`run()`'s `finally`) so they are cleaned up regardless of the code path taken.

---

### Changed: `run()` returns `in_progress` immediately; approval state polled via `GET /result`

**File:** `intrupt_py_sdk/adapters/google_adk.py`

Previously `run()` used `asyncio.wait(FIRST_COMPLETED)` to race the agent task
against an `asyncio.Event` that fires when the gate goes pending. While correct,
this meant `run()` could not return until after the Gemini LLM call **and** the
approval API HTTP call had both completed (2–5 s). Callers experienced multi-second
latency on every `POST /call-tool` even though the code path looked "immediate".

**Fix:** `run()` now returns `{"status": "in_progress", "session_id": ...}`
within microseconds, unconditionally. A closure registered via
`gate.register_pending_callback` writes `{"status": "pending_approval", ...}`
directly into `self._results[session_id]` the moment the gate fires — no polling
delay on the SDK side. Callers discover all state transitions by polling
`GET /result/{session_id}`:

| Status              | Meaning                                      |
|---------------------|----------------------------------------------|
| `"in_progress"`     | LLM thinking or tool executing               |
| `"pending_approval"`| Gate fired; `approval_id` included           |
| `"complete"`        | Finished; `result` has the final text        |
| `"error"`           | Failed; `error` has the exception message    |

`resume()` remains non-blocking (just calls `gate.resolve()` and returns
`{"status": "resuming"}`). After approval the task continues in the background;
the client polls until `"complete"` or `"error"`.

---

### `adapter` field added to all approval payloads

**Files:** `adapters/langgraph.py`, `adapters/google_adk.py`,
`adapters/openai_agents.py`, `adapters/crewai.py`

Each adapter now includes `"adapter": "<name>"` in the payload it sends to the
approval API. The value identifies which framework triggered the approval:

| Adapter file       | Value             |
|--------------------|-------------------|
| `langgraph.py`     | `"langgraph"`     |
| `google_adk.py`    | `"google_adk"`    |
| `openai_agents.py` | `"openai_agents"` |
| `crewai.py`        | `"crewai"`        |

`"adapter"` is not in `_RESERVED_FIELDS` in `core/client.py`, so it flows
through `**metadata` into the JSON body sent to the API without requiring any
change to `ApprovalClient`.

The API stores this value in the new `approvals.adapter` column and exposes it
in approval detail responses (see API changelog).

---
