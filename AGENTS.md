# AGENTS.md

Guidance for AI coding agents (and humans) working in the **`intrupt-py-sdk`** repository.
This is the Python client SDK + framework adapters for the intrupt / Aegmis human-in-the-loop (HITL) approval API. It is its own git repo (root: this directory), separate from the sibling `intrupt_api/` (server), `intrupt_js_sdk/`, and `intrupt_web/` projects.

> The nearest `CLAUDE.md` lives one directory up and describes the *whole* HITL monorepo, some of which is stale relative to this SDK (e.g. it predates the multi-framework adapters and the `gate.py` Future pattern). **When they disagree, this file and the code win.**

---

## Project overview

`intrupt-py-sdk` lets an AI agent pause its riskiest tool calls for human approval, route the request to a channel (Slack, email, Telegram, console, custom), and resume automatically once a human decides — with a full audit trail via the backend approval API.

- **Package name:** `intrupt-py-sdk` (import as `intrupt_py_sdk`)
- **Version:** `0.0.1a10` (alpha; see `CHANGELOG.md`)
- **Python:** `>=3.11`
- **Runtime deps:** `httpx>=0.27`, `langgraph>=0.2`
- **License:** MIT
- **Source:** https://github.com/Aegmis/intrupt-py-sdk

Supported agent frameworks (each has its own adapter, all sharing one core): **LangGraph**, **Google ADK**, **OpenAI Agents SDK**, **CrewAI**.

The public API surface (`intrupt_py_sdk/__init__.py`) is intentionally small: `ApprovalClient`, `ApprovalAPIError`, `ApprovalMiddleware`, `ApprovalGraph`, `approval_required`.

---

## Architecture

The end-to-end flow (framework-agnostic):

```
agent tool
  └── @approval_required(action, message, channel, args=[...])   ← from an adapter
        │  builds payload, calls gate.request_approval(client, session_id, payload)
        │  suspends on an asyncio.Future
        ▼
  client.acreate_approval(...)         ← ApprovalClient (HTTP) or _OnApprovalClient (inline callback)
        │  notifies the human (Slack / email / Telegram / console / policy)
        │  returns {"approval_id": "...", "status": "pending" | "approved" | ...}
        ▼
  human clicks Approve / Reject
        │  your /resume (or /decide) endpoint fires
        ▼
  gate.resolve(approval_id, approved)  ← sets the Future result (cross-loop safe)
        │
        ▼
  tool body runs (approved) OR returns {"status": "cancelled", ...} (rejected)
```

### Layers

| Module | Responsibility |
|---|---|
| `core/gate.py` | **The heart of the SDK.** Framework-agnostic registry of pending `asyncio.Future`s. `request_approval()` / `resolve()` / `get_pending()` / `is_pending()` / `register_pending_callback()`. Handles cross-event-loop resolution (see pitfalls). |
| `core/client.py` | `ApprovalClient` — sync (`create_approval`) + async (`acreate_approval`) HTTP wrapper. `ApprovalAPIError`, `approvals_enabled()`, `user_facing_error()`, `error_status_code()`. Extracts `org_id` from the API key. |
| `core/policy_engine.py` | **Empty (0 bytes)** — a placeholder. No policy logic lives here; rule-based approval is done in userland via `on_approval_async` (see `example/policy_agent.py`). Do not assume it does anything. |
| `adapters/approval_middleware.py` | `ApprovalMiddleware` — process-wide singleton holding one `ApprovalClient`. Construct once at startup; retrieve anywhere via `ApprovalMiddleware.get_client()`. |
| `adapters/langgraph.py` | `ApprovalGraph` wrapper + `@approval_required` decorator + `_OnApprovalClient`. Reference adapter. |
| `adapters/google_adk.py` | `ApprovalRunner` + `@approval_required`. Uses `session_id`; adds SSE subscribe/`_set_result`; non-blocking `run()` polled via `GET /result`. |
| `adapters/openai_agents.py` | `ApprovalAgentRunner` + `@approval_required`. Thread id flows via a `contextvars.ContextVar`. |
| `adapters/crewai.py` | `ApprovalCrew` + `approval_required()` **factory** (wraps a `BaseTool`, not a decorator). Runs tools in worker threads → exercises the cross-loop resolve path. |
| `utils/utils.py` | `_filter_kwargs(kwargs, allowed)` — strips framework plumbing (`config`, `RunnableConfig`) so only `args=[...]` keys reach the approver. |

### Cross-cutting patterns (must understand before editing adapters)

- **gate.py Future pattern** is identical across all four adapters. Don't reinvent it per adapter — extend the gate.
- **contextvars** carry session identity (`thread_id` / `session_id` / `run_id`) into background tasks so concurrent requests don't collide. Set the var *before* `asyncio.create_task()`.
- **`_OnApprovalClient`** duck-types `acreate_approval(thread_id, **kwargs)` so an inline `on_approval_async` callback is interchangeable with `ApprovalClient` at the gate boundary.
- **`adapter` field**: every adapter injects `"adapter": "<name>"` into the approval payload (`"langgraph"`, `"google_adk"`, `"openai_agents"`, `"crewai"`). It flows through `**metadata` because it is *not* in `_RESERVED_FIELDS`.

---

## Folder structure

```
intrupt_py_sdk/               # repo root
├── AGENTS.md                 # this file
├── README.md                 # full user-facing docs (adapters, channels, env)
├── CHANGELOG.md              # dated changes; read before assuming behavior
├── LICENSE                   # MIT
├── pyproject.toml            # metadata, deps, extras, pytest + coverage config
├── uv.lock                   # uv lockfile (this repo uses uv)
├── dist/                     # built wheels/sdists (gitignored)
├── intrupt_py_sdk/           # the package
│   ├── __init__.py           # public exports
│   ├── core/                 # gate.py, client.py, policy_engine.py (empty)
│   ├── adapters/             # approval_middleware + one file per framework
│   └── utils/                # _filter_kwargs
├── example/                  # runnable reference agents, one per channel/framework
└── tests/                    # pytest suite (see below)
```

### Example agents (`example/`)

Each runs as a standalone FastAPI agent on its own port:

| File | Port | Channel | Framework |
|---|---|---|---|
| `agent.py` | 8081 | intrupt API → Slack | LangGraph |
| `console_agent.py` | 8087 | stdin | LangGraph |
| `policy_agent.py` | 8088 | rule-based auto-approve/reject | LangGraph |
| `smtp_email_agent.py` | 8089 | SMTP email | LangGraph |
| `slack_direct_agent.py` | 8090 | direct Slack Block Kit | LangGraph |
| `telegram_agent.py` | 8091 | Telegram inline keyboard | LangGraph |
| `google_adk_agent.py` | 8092 | intrupt API → Slack | Google ADK |
| `openai_agents_agent.py` | 8093 | intrupt API → Slack | OpenAI Agents |
| `crewai_agent.py` | 8094 | intrupt API → Slack | CrewAI |
| `resend_email_agent.py` | 8095 | intrupt API → email (Resend) | LangGraph |

Also present: `async_webhook_agent.py`, `webhook_agent.py`, `email_agent.py`, `finance_agent.py`, `infra_agent.py`.

---

## Build commands

This repo uses **[uv](https://docs.astral.sh/uv/)** (see `uv.lock`). `pyproject.toml` declares no `[build-system]` table, so build with a PEP 517 frontend:

```bash
# Create/sync the environment from the lockfile (installs the package + deps)
uv sync

# Editable install into an existing venv (alternative to uv)
pip install -e .

# Install with a framework extra or the test group
pip install -e ".[test]"
pip install -e ".[google-adk]"     # or [openai-agents], [crewai]

# Build a wheel + sdist into dist/
uv build          # or: python -m build
```

There is an in-tree `.venv/` used during development — `source .venv/bin/activate` before running scripts if you rely on it.

Optional-dependency groups (`pyproject.toml`): `google-adk`, `openai-agents`, `crewai`, `test`.

---

## Test commands

Pytest is configured in `pyproject.toml` (`asyncio_mode = "auto"`, `testpaths = ["tests"]`, `addopts = "-ra --strict-markers"`, `pythonpath = ["..", "example"]`).

```bash
# All tests
pytest -v

# One file / one test
pytest tests/test_approval_graph.py -v
pytest tests/test_gate.py::test_resolve_cross_loop -v

# With coverage (config already in pyproject [tool.coverage.run])
pytest --cov=intrupt_py_sdk --cov-branch
```

Requires the `[test]` extra (`pytest`, `pytest-asyncio`, `langchain-core`, `langchain-openai`, `fastapi`, `uvicorn`, `python-dotenv`).

`tests/conftest.py` sets dummy env (`AEGMIS_BASE_URL`, `AEGMIS_API_KEY`, `AEGMIS_APPROVAL=true`, `OPENAI_API_KEY`) via `setdefault` so the suite never hits a real backend. The API key `sk_org_org_test1234_...` matches the `sk_org_{org_id}_{hash}` format the client parses.

Test files: `test_gate.py`, `test_gate_adapters.py`, `test_approval_graph.py`, `test_approval_decorator.py`, `test_agent_guards.py`, `test_agent_routes.py`, `test_sdk_client.py`, `test_examples.py`.

---

## Linting

**There is no configured linter, formatter, type checker, or CI in this repo** — no `ruff`/`black`/`flake8`/`mypy` config, no `.pre-commit-config.yaml`, no `.github/`. Match the existing style by reading neighboring code rather than running a tool. If you add tooling, add it as a `[project.optional-dependencies]` group and document it here — don't reformat the whole tree in an unrelated change.

---

## Coding conventions

Derived from the existing code — follow these:

- **Type hints** on public function signatures; `Optional[...]` from `typing`. Modern builtin generics (`dict[str, ...]`, `tuple[...]`) are used freely (Python 3.11+).
- **Docstrings** explain the *why*, especially non-obvious concurrency behavior (see `gate.py`). Module docstrings in adapters document install, env vars, and usage.
- **Leading underscore = private/internal**: `_pending`, `_filter_kwargs`, `_OnApprovalClient`, `_raise_for_status`, `_set_if_pending`. Don't export these; don't rely on them from userland.
- **Keyword-only public args**: `ApprovalClient.create_approval(*, thread_id, action, ...)`. Preserve `*` markers.
- **Logging, not printing**, in library code: `logger = logging.getLogger(__name__)`. `print()` is fine only in `example/`.
- **Errors surface, but readably**: raise `ApprovalAPIError` (carries `status_code`, `detail`, `request_id`); use `user_facing_error()` / `error_status_code()` at endpoint boundaries. `gate._surface_api_error` logs human-readable hints for 401/403/404/400.
- **Env-var contract** is centralized: read `AEGMIS_*` through `ApprovalClient` / `approvals_enabled()`, not scattered `os.environ` reads.
- **Payload field discipline**: new top-level approval fields must be added to `_RESERVED_FIELDS` in `core/client.py` *only if* they map to an explicit JSON key; free-form metadata should flow through `**metadata`.

---

## How to add new features

### Add a new framework adapter
1. Create `intrupt_py_sdk/adapters/<framework>.py`.
2. Reuse `core/gate.py` — call `gate.request_approval(...)` and await the returned Future; do **not** build a parallel Future registry.
3. Provide a `@approval_required` decorator (or factory, as CrewAI does) that builds the payload `{action, message, channel, tool: {name, description, kwargs}, adapter: "<framework>"}` and uses `_filter_kwargs(user_kwargs, args)` to strip plumbing.
4. Inject `"adapter": "<framework>"` into the payload.
5. Carry session identity through a `contextvars.ContextVar` set before spawning the background task.
6. Provide a `Runner`/`Graph`/`Crew` wrapper class with `run`/`ainvoke` + `resume`/`aresume` + `pending()`, returning the standard shapes (`{"status": "pending_approval" | "complete" | "cancelled" | "error", ...}`).
7. If the framework runs tools in a worker thread (like CrewAI), confirm the gate's cross-loop `call_soon_threadsafe` path is exercised.
8. Add an `example/<framework>_agent.py` on a new port and a `tests/` entry (mirror `test_gate_adapters.py`).
9. Add an optional-dependency extra in `pyproject.toml` if the framework isn't a core dep.
10. Export from `__init__.py` only if it belongs on the public surface.

### Add a new approval channel
Channels are **userland**, not SDK code — implement an `on_approval_async(session_id, payload) -> {"approval_id": "..."}` callback and a decide/resume endpoint, following the patterns in `README.md` and `example/` (console, policy, SMTP, Slack-direct, Telegram). The SDK only needs the callback to return an `approval_id`.

### Change the wire format
Edit both `create_approval` and `acreate_approval` in `core/client.py` (they must stay in sync), update `_RESERVED_FIELDS`, and coordinate with the `intrupt_api/` server — the field must exist on both sides. Note it in `CHANGELOG.md`.

---

## Deployment process

This is a **library**, deployed by publishing to PyPI — not a running service.

1. Bump `version` in `pyproject.toml` and add a dated entry to `CHANGELOG.md`.
2. `uv build` (artifacts land in `dist/`, which is gitignored).
3. Publish: `uv publish` (or `twine upload dist/*`) to PyPI as `intrupt-py-sdk`.
4. Users install with `pip install intrupt-py-sdk` / `uv add intrupt-py-sdk`.

The `example/` agents are for local demos, not production deploys. The production/deploy artifacts (`docker-compose.prod.yml`, deploy guides) in the parent monorepo belong to the **`intrupt_api/` server**, not this SDK.

Breaking changes (like the `APPROVAL_*` → `AEGMIS_*` env rename in `0.0.1a10`) must be called out prominently in `CHANGELOG.md` and `README.md` with a migration note.

---

## Common pitfalls

- **`policy_engine.py` is empty.** Policy/rule logic lives in userland `on_approval_async` callbacks (`example/policy_agent.py`), not here.
- **`ApprovalMiddleware` is a `__new__` singleton and idempotent.** Constructing it a second time is a **no-op** — it will *not* re-point the client. To change credentials in tests, reset `ApprovalMiddleware._instance = None`.
- **Cross-event-loop resolve.** A Future may be created on a different loop/thread than the one calling `resolve()` (CrewAI runs tools via `asyncio.run(...)` in a worker thread). The gate hops back via `call_soon_threadsafe`; a plain cross-loop `set_result` never wakes the waiter and the tool hangs forever. Preserve `_pending_loops` bookkeeping when editing `gate.py`.
- **Tests: always `with TestClient(app) as client:`.** Without the context manager each request spins a new anyio portal / event loop, orphaning the background task and causing `CancelledError` on resume.
- **`timeout` (default 1.5s)** is how long `ainvoke`/`run` waits before concluding a gate was hit and returning `pending_approval`; the SDK then polls up to 10s for `approval_id`. Raise it if your LLM/tool startup is slow.
- **`sync create_approval` blocks the event loop** — use `acreate_approval` on any async path.
- **`thread_id` is required** by `create_approval`/`acreate_approval` (needed to resume). Missing → `ValueError`.
- **Framework session-id naming differs**: LangGraph/OpenAI use `thread_id`, ADK uses `session_id`, CrewAI uses `run_id`. Keep them straight per adapter.
- **The parent `CLAUDE.md` is partly stale** for this SDK (predates the gate pattern and multi-adapter layout). Trust the code.

---

## Security considerations

- **Never commit secrets.** `.env` / `.env.*` are gitignored. `AEGMIS_API_KEY`, `AGENT_RESUME_SECRET`, `SLACK_BOT_TOKEN`, `OPENAI_API_KEY`, `SMTP_PASS`, etc. come from the environment only.
- **API key format `sk_org_{org_id}_{hash}`** — the client parses `org_id` out of it and scopes all calls to `/org/{org_id}/approval`. Treat the key as a bearer credential.
- **Verify the `/resume` callback.** The backend sends `X-Agent-Secret`; endpoints must compare it against `AGENT_RESUME_SECRET` and reject mismatches (`401`). See `example/agent.py`.
- **Signed one-click links.** The email channel's approve/reject links are HMAC-signed and single-use — that signature *is* the credential (no login). Don't weaken or log the `sig`; don't make links replayable.
- **Least-privilege payloads.** Only the kwargs named in `args=[...]` reach the approver (`_filter_kwargs`). Never widen this to forward whole `RunnableConfig`/`tool_context`/framework plumbing — it can leak internal state or secrets.
- **`AEGMIS_APPROVAL=false` auto-approves everything in-process** with no human and no backend call. It is a dev/test convenience — never ship it enabled to production.
- **Idempotent resume.** `resume()` checks `gate.is_pending(...)` and returns `already_resolved` for double-submits/races (e.g. email link clicked twice) rather than silently doing nothing.

---

## Do and Don't

**Do**
- Route all approval waits through `core/gate.py`; keep the Future pattern uniform across adapters.
- Keep `create_approval` and `acreate_approval` (`core/client.py`) in sync.
- Add a dated `CHANGELOG.md` entry for any behavior change; flag breaking changes loudly.
- Use `logging` in library code and `ApprovalClient`/`approvals_enabled()` for env access.
- Add a matching `example/*_agent.py` and tests when adding an adapter or channel.
- Use `with TestClient(app) as client:` and reset `ApprovalMiddleware._instance` in tests.
- Forward only `args=[...]` kwargs to approvers via `_filter_kwargs`.

**Don't**
- Don't re-point `ApprovalMiddleware` by constructing it twice (it's a no-op singleton).
- Don't cross-loop `set_result` a Future — use the gate's `resolve()` (`call_soon_threadsafe`).
- Don't call blocking `create_approval` from async code.
- Don't put policy logic in `core/policy_engine.py` expecting it to run — it's an empty placeholder; use `on_approval_async`.
- Don't forward `config` / `RunnableConfig` / `tool_context` or other plumbing into approval payloads.
- Don't add secrets, real API keys, or `.env` files to the repo.
- Don't ship with `AEGMIS_APPROVAL=false` to production.
- Don't trust the parent `CLAUDE.md` over the code for this SDK.
- Don't reformat the whole tree (no configured formatter) — match local style.
