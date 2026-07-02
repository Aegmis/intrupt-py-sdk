"""LangGraph adapter for intrupt human-in-the-loop approvals.

Install
-------
The core SDK already lists ``langgraph`` as a required dependency, so no
extras group is needed::

    pip install intrupt-py-sdk

To use the full agent example you also need the LangChain OpenAI integration::

    pip install intrupt-py-sdk langchain-openai

Required packages
-----------------
- ``langgraph>=0.2``        (provides ``langgraph``, ``langgraph.prebuilt``,
                              ``langgraph.checkpoint.memory``)
- ``langchain-core>=0.3``   (provides ``@tool``, ``BaseMessage``, ``add_messages``)
- ``langchain-openai>=0.2`` (provides ``ChatOpenAI`` — only needed in the agent)

Environment variables
---------------------
- ``APPROVAL_BASE_URL``    URL of the intrupt approval API  (e.g. ``http://localhost:8080``)
- ``APPROVAL_API_KEY``     API key for the approval API
- ``AGENT_PUBLIC_URL``     Public URL of this agent server (used as callback base)
- ``AGENT_RESUME_SECRET``  Random secret echoed as ``X-Agent-Secret`` on ``/resume`` callbacks
- ``OPENAI_API_KEY``       OpenAI API key (for ``ChatOpenAI``)

Uses the same gate.py Future pattern as the Google ADK, OpenAI Agents, and
CrewAI adapters — no LangGraph ``interrupt()`` involved.

Usage
-----
::

    from intrupt_py_sdk.adapters.langgraph import approval_required, ApprovalGraph

    @tool
    @approval_required(action="purchase_stock", message="Approve?", channel="slack",
                       args=["symbol", "quantity"])
    def purchase_stock(symbol: str, quantity: int) -> dict:
        ...  # only runs if approved

    approval_graph = ApprovalGraph(
        graph=graph,
        callback_url="http://localhost:8081/resume",
        callback_secret=os.getenv("AGENT_RESUME_SECRET", ""),
    )

    result = await approval_graph.run({"messages": [...]}, thread_id)
    # if result["status"] == "pending_approval": wait for /resume call
    result = await approval_graph.resume(thread_id, approved=True, approval_id="...")
"""
import asyncio
import contextvars
import logging
import uuid
from functools import wraps
from typing import Optional

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.core.client import error_status_code, user_facing_error
from intrupt_py_sdk.utils.utils import _filter_kwargs

logger = logging.getLogger(__name__)

# Side-channel for tool-level API errors. Keyed by thread_id. The tool wrapper
# stores an exception here and re-raises; after graph.ainvoke returns,
# _run_graph checks this dict and surfaces {"status": "error"} instead of
# the LLM's natural-language summary of the error ToolMessage.
_tool_api_errors: dict[str, Exception] = {}

_CALLBACK_URL: str = ""
_CALLBACK_SECRET: str = ""

# Each asyncio Task gets its own copy of these vars so concurrent runs don't
# share state — same pattern as the OpenAI Agents / CrewAI adapters.
_current_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "lg_thread_id", default=""
)
_current_on_approval_client: contextvars.ContextVar = contextvars.ContextVar(
    "lg_on_approval_client", default=None
)


class _OnApprovalClient:
    """Wraps an on_approval_async callback so gate.py can call acreate_approval."""

    def __init__(self, callback):
        self._callback = callback

    async def acreate_approval(self, *, thread_id: str, **kwargs) -> dict:
        result = await self._callback(thread_id, kwargs)
        return {
            "approval_id": result.get("approval_id", str(uuid.uuid4())),
            "status": "pending",
        }


def configure(callback_url: str, callback_secret: str = "") -> None:
    global _CALLBACK_URL, _CALLBACK_SECRET
    _CALLBACK_URL = callback_url
    _CALLBACK_SECRET = callback_secret


def approval_required(
    action: str = "",
    message: str = "",
    channel: str = "slack",
    args: Optional[list] = None,
) -> ...:
    """Decorate a tool so it pauses for human approval before executing.

    Apply *inside* ``@tool``::

        @tool
        @approval_required(action="...", message="...")
        def my_tool(...) -> dict: ...

    The thread_id is picked up from ``_current_thread_id``, which
    ``ApprovalGraph.run()`` sets before launching the graph task.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*fargs, **kwargs):
            # Strip LangChain's RunnableConfig plumbing before payload / func call.
            user_kwargs = {k: v for k, v in kwargs.items() if k != "config"}
            thread_id = _current_thread_id.get() or str(uuid.uuid4())
            payload = {
                "action": action or func.__name__,
                "message": message or f"Approval required for {func.__name__}",
                "channel": channel,
                "tool": {
                    "name": func.__name__,
                    "description": func.__doc__ or "",
                    "kwargs": _filter_kwargs(user_kwargs, args),
                },
                "agent_callback_url": _CALLBACK_URL,
                "agent_callback_secret": _CALLBACK_SECRET,
                "adapter": "langgraph",
            }
            inline = _current_on_approval_client.get()
            client = inline if inline is not None else ApprovalMiddleware.get_client()
            try:
                _, future = await gate.request_approval(client, thread_id, payload)
            except Exception as exc:
                logger.error("approval API call failed for tool %r: %s", func.__name__, exc)
                _tool_api_errors[thread_id] = exc
                raise
            approved = await future
            if not approved:
                return {
                    "status": "cancelled",
                    "tool": func.__name__,
                    "message": f"{func.__name__} was not approved",
                }
            if asyncio.iscoroutinefunction(func):
                return await func(*fargs, **user_kwargs)
            return func(*fargs, **user_kwargs)

        return wrapper
    return decorator


class ApprovalGraph:
    """Wraps a compiled LangGraph graph; handles approval gating and resume.

    Two-step flow:
      1. ``run()`` / ``ainvoke()`` launches ``graph.ainvoke`` as a background
         asyncio Task and waits up to ``timeout`` seconds. If an
         ``@approval_required`` tool fires before the timeout the call returns
         ``{"status": "pending_approval", "approval_id": "...", ...}``.
      2. ``resume()`` / ``aresume()`` calls ``gate.resolve()`` to unblock the
         Future and awaits the background task to completion.

    Args:
        graph:             Compiled LangGraph ``StateGraph``.
        callback_url:      URL the approval platform will POST to when the
                           human decides (e.g. ``http://myagent/resume``).
        callback_secret:   Optional secret echoed in ``X-Agent-Secret`` so
                           ``/resume`` can verify the caller.
        on_approval_async: Async callback ``(thread_id, payload) -> {"approval_id": ...}``
                           used instead of the HTTP approval API. Useful for
                           local/console approval, policy engines, etc.
        timeout:           Seconds to wait for an approval gate to fire before
                           returning ``pending_approval``. Default 1.5 s — set
                           higher if your LLM or tool startup is slow.
        client:            Deprecated. Pass a pre-built ``ApprovalMiddleware``
                           or ``ApprovalClient`` instance. Prefer calling
                           ``ApprovalMiddleware(base_url=...)`` before
                           constructing ``ApprovalGraph``.
    """

    def __init__(
        self,
        graph,
        callback_url: str = "",
        callback_secret: str = "",
        on_approval_async=None,
        timeout: float = 1.5,
        client=None,
        # kept for backwards compat with test helpers that used _timeout=
        _timeout: Optional[float] = None,
    ):
        self.graph = graph
        self._on_approval_async = on_approval_async
        if client is not None:
            # Legacy: accept a pre-built ApprovalMiddleware or ApprovalClient
            # and wire it into the singleton so approval_required can find it.
            actual = getattr(client, "client", client)
            ApprovalMiddleware._instance = object.__new__(ApprovalMiddleware)
            ApprovalMiddleware._instance.client = actual
        configure(callback_url, callback_secret)
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, dict] = {}
        self._timeout = _timeout if _timeout is not None else timeout

    async def run(self, input: dict, thread_id: str, config: Optional[dict] = None) -> dict:
        """Start (or restart) a graph run for *thread_id*.

        Returns immediately after ``timeout`` seconds if an approval gate fires.
        """
        _current_thread_id.set(thread_id)
        if self._on_approval_async:
            _current_on_approval_client.set(_OnApprovalClient(self._on_approval_async))
        task = asyncio.create_task(self._run_graph(thread_id, input, config))
        self._tasks[thread_id] = task
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=self._timeout)
        except asyncio.TimeoutError:
            # A tool API error is stored in _tool_api_errors before the SDK/graph
            # makes the extra LLM turn to handle it — check before polling the gate.
            api_err = _tool_api_errors.pop(thread_id, None)
            if api_err is not None:
                r = {"status": "error", "thread_id": thread_id, "error": user_facing_error(api_err),
                     "status_code": error_status_code(api_err)}
                self._results[thread_id] = r
                return r

            # acreate_approval (HTTP call) may still be in-flight when the
            # shield times out. Poll until the gate registers the approval_id
            # or the task finishes, whichever comes first.
            approval_id = await self._await_gate(thread_id, task)

            # If the task finished during _await_gate (e.g. the API auto-approved
            # the tool and the graph ran to completion without ever parking in
            # _pending), return the real result instead of a spurious pending_approval.
            if task.done() and not task.cancelled():
                try:
                    return task.result()
                except Exception as exc:
                    return {"status": "error", "thread_id": thread_id, "error": user_facing_error(exc),
                            "status_code": error_status_code(exc)}

            return {
                "status": "pending_approval",
                "thread_id": thread_id,
                "approval_id": approval_id,
            }
        except Exception as exc:
            # task raised before the timeout (graph.ainvoke raised because, e.g.,
            # a tool raised with no recovery LLM node in the graph).
            # _run_graph already cleaned _tool_api_errors before re-raising.
            r = {"status": "error", "thread_id": thread_id, "error": user_facing_error(exc),
                 "status_code": error_status_code(exc)}
            self._results[thread_id] = r
            return r

    async def resume(
        self,
        thread_id: str,
        approved: bool,
        approval_id: str = "",
    ) -> dict:
        """Unblock the gate Future and return immediately.

        The background task continues running (remaining LLM turns, tool calls).
        Callers that need the final result should poll _results[thread_id] or
        GET /result/{thread_id} — the long-poll in /call-tool already does this.
        Returning immediately here is critical: Slack webhooks require a 200 OK
        within 3 s or they retry, which would cause a double-resolve.
        """
        if not approval_id:
            approval_id = gate.get_pending(thread_id) or ""
        gate.resolve(approval_id, approved)
        # Check results first — covers the race where the task finishes immediately
        # after gate.resolve() and pops itself from _tasks before we check.
        result = self._results.get(thread_id)
        if result:
            return result
        if thread_id not in self._tasks:
            return {"status": "not_found", "thread_id": thread_id}
        return {"status": "accepted", "thread_id": thread_id, "approval_id": approval_id}

    async def ainvoke(self, input: dict, thread_id: str, config: Optional[dict] = None) -> dict:
        """Alias for run() — preferred name when using on_approval_async."""
        return await self.run(input, thread_id, config)

    async def aresume(self, thread_id: str, approved: bool, approval_id: str = "") -> dict:
        """Alias for resume()."""
        return await self.resume(thread_id, approved, approval_id)

    async def wait_for_result(self, thread_id: str, timeout: float = 10.0) -> dict:
        """Poll until the background task stores a terminal result for *thread_id*.

        Use this after ``resume()`` when the caller needs the final completed
        result rather than the immediate ``{"status": "accepted"}`` acknowledgement.
        Suitable for HTTP endpoints that can afford to wait (e.g. a /decide
        handler or a console auto-approval loop). Do NOT use from Slack webhook
        handlers — they require a 3-second response; return ``resume()`` directly.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        poll = 0.05
        while loop.time() < deadline:
            result = self._results.get(thread_id)
            if result and result.get("status") not in ("pending_approval", "accepted"):
                return result
            if thread_id not in self._tasks:
                # Task finished and removed itself — result must be in _results.
                return self._results.get(thread_id) or {
                    "status": "not_found",
                    "thread_id": thread_id,
                }
            await asyncio.sleep(poll)
        return {"status": "timeout", "thread_id": thread_id}

    def pending(self, thread_id: str) -> bool:
        """Return True if *thread_id* is paused on an approval gate."""
        return gate.get_pending(thread_id) is not None

    def get_state(self, thread_id: str):
        """Return the LangGraph checkpoint state for *thread_id*."""
        return self.graph.get_state({"configurable": {"thread_id": thread_id}})

    def update_state(self, thread_id: str, values: dict, as_node: Optional[str] = None):
        return self.graph.update_state(
            {"configurable": {"thread_id": thread_id}}, values, as_node=as_node
        )

    async def _await_gate(
        self, thread_id: str, task: asyncio.Task, poll: float = 0.05, extra: float = 10.0
    ) -> Optional[str]:
        """Poll until gate registers an approval_id for thread_id or task finishes.

        Called after the shield timeout — the acreate_approval HTTP call may
        still be in-flight, so we yield in short increments until:
          - the gate mapping appears  (real human-approval required), or
          - the task finishes         (auto-approved / no approval needed), or
          - the extra deadline passes (slow graph — rare, but handled by run()).
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + extra
        while loop.time() < deadline:
            approval_id = gate.get_pending(thread_id)
            if approval_id is not None:
                return approval_id
            if task.done():
                return None
            await asyncio.sleep(poll)
        # Deadline passed — return whatever the gate has (may be None).
        return gate.get_pending(thread_id)

    async def _run_graph(
        self, thread_id: str, input: dict, config: Optional[dict] = None
    ) -> dict:
        cfg: dict = {"configurable": {"thread_id": thread_id}}
        if config:
            cfg = {
                **config,
                "configurable": {**config.get("configurable", {}), "thread_id": thread_id},
            }
        try:
            result = await self.graph.ainvoke(input, config=cfg)
            api_err = _tool_api_errors.pop(thread_id, None)
            if api_err is not None:
                r: dict = {"status": "error", "thread_id": thread_id, "error": user_facing_error(api_err),
                           "status_code": error_status_code(api_err)}
            else:
                r = {
                    "status": "complete",
                    "thread_id": thread_id,
                    "result": result,
                    "messages": [
                        {"type": m.__class__.__name__, "content": m.content}
                        for m in result.get("messages", [])
                    ],
                }
        except Exception as exc:
            api_err = _tool_api_errors.pop(thread_id, None)
            err = api_err if api_err is not None else exc
            r = {
                "status": "error",
                "thread_id": thread_id,
                "error": user_facing_error(err),
                "status_code": error_status_code(err),
            }
            self._results[thread_id] = r
            raise
        finally:
            self._tasks.pop(thread_id, None)
        self._results[thread_id] = r
        return r
