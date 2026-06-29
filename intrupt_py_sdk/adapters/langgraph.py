"""LangGraph adapter for intrupt human-in-the-loop approvals.

Uses the same gate.py Future pattern as the Google ADK, OpenAI Agents, and
CrewAI adapters — no LangGraph ``interrupt()`` involved.

Usage::

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
import uuid
from functools import wraps
from typing import Optional

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.utils.utils import _filter_kwargs

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
            }
            inline = _current_on_approval_client.get()
            client = inline if inline is not None else ApprovalMiddleware.get_client()
            _, future = await gate.request_approval(client, thread_id, payload)
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
            # acreate_approval (HTTP call) may still be in-flight when the
            # shield times out. Poll until the gate registers the approval_id
            # or the task finishes, whichever comes first.
            approval_id = await self._await_gate(thread_id, task)
            return {
                "status": "pending_approval",
                "thread_id": thread_id,
                "approval_id": approval_id,
            }

    async def resume(
        self,
        thread_id: str,
        approved: bool,
        approval_id: str = "",
    ) -> dict:
        """Unblock the gate Future and await the background task to completion."""
        gate.resolve(approval_id, approved)
        task = self._tasks.get(thread_id)
        if task:
            await task
            return self._results.get(
                thread_id, {"status": "complete", "thread_id": thread_id}
            )
        return {"status": "not_found", "thread_id": thread_id}

    async def ainvoke(self, input: dict, thread_id: str, config: Optional[dict] = None) -> dict:
        """Alias for run() — preferred name when using on_approval_async."""
        return await self.run(input, thread_id, config)

    async def aresume(self, thread_id: str, approved: bool, approval_id: str = "") -> dict:
        """Alias for resume()."""
        return await self.resume(thread_id, approved, approval_id)

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
        still be in-flight, so we yield in short increments until the gate
        mapping appears (or the task dies unexpectedly).
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
        result = await self.graph.ainvoke(input, config=cfg)
        r: dict = {
            "status": "complete",
            "thread_id": thread_id,
            "result": result,
            "messages": [
                {"type": m.__class__.__name__, "content": m.content}
                for m in result.get("messages", [])
            ],
        }
        self._results[thread_id] = r
        return r
