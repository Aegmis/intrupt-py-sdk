"""Google ADK adapter for intrupt human-in-the-loop approvals.

Install
-------
This adapter requires the ``google-adk`` package which is **not** installed by
default. Install it with the bundled extras group::

    pip install "intrupt-py-sdk[google-adk]"

Or install the framework package directly::

    pip install google-adk

Required packages
-----------------
- ``google-adk>=0.1``   (provides ``google.adk``, ``google.adk.sessions``,
                          ``google.genai.types``)

Environment variables
---------------------
- ``AEGMIS_BASE_URL``   URL of the intrupt approval API  (e.g. ``http://localhost:8080``)
- ``AEGMIS_API_KEY``    API key for the approval API
- ``AGENT_PUBLIC_URL``    Public URL of this agent server (used as callback base)
- ``GOOGLE_API_KEY``      Gemini API key  (or ``GOOGLE_CLOUD_PROJECT`` for Vertex)

Usage
-----
::

    from intrupt_py_sdk.adapters.google_adk import approval_required, ApprovalRunner

    @approval_required(action="purchase_stock", message="Approve stock purchase?",
                       channel="slack", args=["symbol", "quantity"])
    async def purchase_stock(symbol: str, quantity: int, tool_context=None) -> str:
        ...

    runner = ApprovalRunner(
        agent=my_agent,
        app_name="finance_bot",
        session_service=session_svc,
        callback_url="https://my-agent.example.com/resume",
        callback_secret="...",
    )

    result = await runner.run(session_id, message)
    # if result["status"] == "pending_approval": wait for /resume call
    result = await runner.resume(session_id, approved=True, approval_id="...")
"""
import asyncio
import logging
import uuid
from functools import wraps
from typing import Callable, Optional

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.core.client import error_status_code, user_facing_error
from intrupt_py_sdk.utils.utils import _filter_kwargs

logger = logging.getLogger(__name__)

# Side-channel for tool-level API errors. Keyed by session_id. The tool wrapper
# stores an exception here and re-raises; after runner.run_async finishes,
# _run_agent checks this dict and surfaces {"status": "error"} instead of
# the LLM's natural-language summary of the error.
_tool_api_errors: dict[str, Exception] = {}

_CALLBACK_URL: str = ""
_CALLBACK_SECRET: str = ""


def configure(callback_url: str, callback_secret: str = "") -> None:
    global _CALLBACK_URL, _CALLBACK_SECRET
    _CALLBACK_URL = callback_url
    _CALLBACK_SECRET = callback_secret


def _session_id_from_tool_context(tool_context) -> Optional[str]:
    """Extract the ADK session id from a ToolContext across ADK versions.

    The attribute path has changed between releases: newer ADK exposes a
    ``session`` property directly on ToolContext (backed by the private
    ``_invocation_context``) and no longer has a public ``invocation_context``
    attribute. Getting this wrong is catastrophic: the gate would register the
    approval under a random uuid, so the /resume callback (which carries the
    real session id) can never resolve it and the tool hangs forever.
    """
    if tool_context is None:
        return None
    candidates = (
        lambda: tool_context.session.id,
        lambda: tool_context._invocation_context.session.id,
        lambda: tool_context.get_invocation_context().session.id,
        lambda: tool_context.invocation_context.session.id,
    )
    for get in candidates:
        try:
            sid = get()
        except Exception:
            continue
        if sid:
            return sid
    return None


def approval_required(
    action: str,
    message: str,
    channel: str = "slack",
    args: Optional[list] = None,
) -> Callable:
    """Decorator for ADK tool functions that require human approval.

    The decorated function must be async. ADK injects ``tool_context`` as a
    kwarg; the decorator reads the session id from it (via
    ``_session_id_from_tool_context``) to key the approval gate.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*fargs, **kwargs):
            tool_context = kwargs.get("tool_context")
            session_id = _session_id_from_tool_context(tool_context)
            if not session_id:
                # Falling back to a random id would make /resume unable to match
                # this approval — log loudly so the misconfiguration is visible.
                session_id = str(uuid.uuid4())
                logger.error(
                    "google_adk: could not read session id from tool_context for "
                    "tool %r; approvals for this call cannot be resumed. This is "
                    "an ADK-version/ToolContext API mismatch.", func.__name__,
                )

            filtered = _filter_kwargs(kwargs, args)
            payload = {
                "action": action,
                "message": message,
                "channel": channel,
                "tool": {
                    "name": func.__name__,
                    "description": func.__doc__ or "",
                    "kwargs": filtered,
                },
                "agent_callback_url": _CALLBACK_URL,
                "agent_callback_secret": _CALLBACK_SECRET,
                "adapter": "google_adk",
            }

            client = ApprovalMiddleware.get_client()
            try:
                approval_id, future = await gate.request_approval(client, session_id, payload)
            except Exception as exc:
                logger.error("approval API call failed for tool %r: %s", func.__name__, exc)
                _tool_api_errors[session_id] = exc
                raise

            approved = await future
            if not approved:
                return {"status": "cancelled", "tool": func.__name__}
            return await func(*fargs, **kwargs)

        return wrapper
    return decorator


class ApprovalRunner:
    """Wraps ``google.adk.Runner`` to expose pending-approval state.

    ``run()`` launches the ADK runner as a background asyncio task and returns
    immediately with ``{"status": "in_progress"}``. Callers discover state
    transitions either by polling ``GET /result/{session_id}`` or by opening
    ``GET /events/{session_id}`` (Server-Sent Events).

    State sequence::

        in_progress → pending_approval → in_progress → complete | error
    """

    def __init__(
        self,
        agent,
        app_name: str,
        session_service,
        callback_url: str,
        callback_secret: str = "",
        **runner_kwargs,
    ):
        from google.adk import Runner  # type: ignore[import]
        self._runner = Runner(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
            **runner_kwargs,
        )
        configure(callback_url, callback_secret)
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, dict] = {}
        # SSE: maps session_id → list of per-subscriber asyncio.Queue instances.
        self._event_queues: dict[str, list[asyncio.Queue]] = {}

    # ── SSE subscriber management ──────────────────────────────────────────────

    def subscribe(self, session_id: str) -> "asyncio.Queue[dict]":
        """Register a new SSE subscriber; returns a queue that receives state dicts."""
        q: asyncio.Queue = asyncio.Queue()
        self._event_queues.setdefault(session_id, []).append(q)
        return q

    def unsubscribe(self, session_id: str, queue: "asyncio.Queue[dict]") -> None:
        """Remove an SSE subscriber queue."""
        queues = self._event_queues.get(session_id, [])
        try:
            queues.remove(queue)
        except ValueError:
            pass
        if not queues:
            self._event_queues.pop(session_id, None)

    def _set_result(self, session_id: str, result: dict) -> None:
        """Store result and push it to every open SSE subscriber for this session."""
        self._results[session_id] = result
        for q in self._event_queues.get(session_id, []):
            q.put_nowait(result)

    # ── Core run / resume ──────────────────────────────────────────────────────

    async def run(self, session_id: str, message: str) -> dict:
        """Start the agent run as a background task and return immediately.

        Returns ``{"status": "in_progress", "session_id": ...}`` right away.
        State transitions are pushed to SSE subscribers (``subscribe()``) and
        are also available by polling ``GET /result/{session_id}``:

        - ``"in_progress"``      — LLM is thinking / tool is running
        - ``"pending_approval"`` — tool hit the gate; ``approval_id`` is included
        - ``"complete"``         — finished; ``result`` has the final text
        - ``"error"``            — failed; ``error`` has the exception message
        """
        task = asyncio.create_task(self._run_agent(session_id, message))
        self._tasks[session_id] = task

        def _on_approval_pending() -> None:
            approval_id = gate.get_pending(session_id)
            self._set_result(session_id, {
                "status": "pending_approval",
                "session_id": session_id,
                "approval_id": approval_id,
            })

        gate.register_pending_callback(session_id, _on_approval_pending)
        return {"status": "in_progress", "session_id": session_id}

    async def resume(self, session_id: str, approved: bool, approval_id: str = "") -> dict:
        """Unblock the pending approval gate and return immediately.

        The agent task continues in the background. The final result is pushed
        to SSE subscribers and stored in _results[session_id].
        """
        if session_id not in self._tasks:
            return {"status": "not_found", "session_id": session_id}
        if not approval_id:
            approval_id = gate.get_pending(session_id) or ""
        if not gate.is_pending(approval_id):
            return {"status": "already_resolved", "session_id": session_id, "approval_id": approval_id}
        gate.resolve(approval_id, approved)
        return {"status": "resuming", "session_id": session_id, "approval_id": approval_id}

    async def _ensure_session(self, session_id: str) -> None:
        """Create the ADK session if it doesn't exist yet.

        InMemorySessionService returns None (not an exception) when a session is
        not found, so we must capture the return value to decide whether to create.
        The service methods may be sync or async depending on ADK version.
        """
        svc = self._runner.session_service
        app_name = self._runner.app_name

        getter = svc.get_session(app_name=app_name, user_id="user", session_id=session_id)
        existing = (await getter) if asyncio.iscoroutine(getter) else getter
        if existing is not None:
            return  # session already exists

        try:
            creator = svc.create_session(app_name=app_name, user_id="user", session_id=session_id)
            if asyncio.iscoroutine(creator):
                await creator
        except Exception:
            pass  # concurrent request already created the session — safe to ignore

    async def _run_agent(self, session_id: str, message: str) -> dict:
        from google.genai.types import Content, Part  # type: ignore[import]

        result: dict = {"status": "error", "session_id": session_id, "error": "unknown", "status_code": 500}
        try:
            await self._ensure_session(session_id)

            content = Content(role="user", parts=[Part(text=message)])
            final_text = ""
            async for event in self._runner.run_async(
                user_id="user",
                session_id=session_id,
                new_message=content,
            ):
                if event.is_final_response() and event.content:
                    final_text = "".join(
                        p.text for p in event.content.parts if hasattr(p, "text")
                    )
            api_err = _tool_api_errors.pop(session_id, None)
            if api_err is not None:
                result = {"status": "error", "session_id": session_id, "error": user_facing_error(api_err), "status_code": error_status_code(api_err)}
            else:
                result = {"status": "complete", "session_id": session_id, "result": final_text}
        except Exception as exc:
            _tool_api_errors.pop(session_id, None)
            logger.exception("_run_agent failed for session %s: %s", session_id, exc)
            result = {"status": "error", "session_id": session_id, "error": user_facing_error(exc), "status_code": error_status_code(exc)}
        finally:
            # Unregister the pending callback registered in run() — it may not
            # have fired (e.g. auto-approved) and we must not leave a dangling ref.
            gate.unregister_pending_callbacks(session_id)
            # Push final result to SSE subscribers and store it before removing
            # the task — GET /result must never see a window where the session
            # is gone from both dicts.
            self._set_result(session_id, result)
            self._tasks.pop(session_id, None)
        return result
