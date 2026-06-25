import asyncio
import inspect
from functools import wraps
from typing import Callable, Optional
from langgraph.types import interrupt, Command
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent
from ..utils import _filter_kwargs


class ApprovalCallbackHandler(GraphCallbackHandler):
    """Fires an approval callable the moment LangGraph captures an interrupt.

    ``on_approval(thread_id, interrupt_value)`` is called with the raw interrupt
    payload dict from ``@approval_required``. It must return a dict; if that dict
    contains ``"approval_id"`` the graph is considered pending, otherwise complete.
    """

    def __init__(self, thread_id: str, on_approval: Callable[[str, dict], dict]):
        super().__init__()
        self.thread_id = thread_id
        self.on_approval = on_approval
        self.approval_result: Optional[dict] = None

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        if self.approval_result is not None:
            return
        for itr in event.interrupts:
            v = getattr(itr, "value", itr)
            if isinstance(v, dict) and v.get("approval_required"):
                self.approval_result = self.on_approval(self.thread_id, v)
                break


class AsyncApprovalCallbackHandler(GraphCallbackHandler):
    """Async variant — on_interrupt is a coroutine so ainvoke never blocks.

    ``on_approval`` may be either a regular function or a coroutine function.
    Sync callables are run in a thread-pool executor so they don't block the
    event loop, but prefer ``async def`` for true non-blocking behaviour.
    """

    def __init__(self, thread_id: str, on_approval: Callable):
        super().__init__()
        self.thread_id = thread_id
        self.on_approval = on_approval
        self.approval_result: Optional[dict] = None

    async def on_interrupt(self, event: GraphInterruptEvent) -> None:
        if self.approval_result is not None:
            return
        for itr in event.interrupts:
            v = getattr(itr, "value", itr)
            if isinstance(v, dict) and v.get("approval_required"):
                if inspect.iscoroutinefunction(self.on_approval):
                    self.approval_result = await self.on_approval(self.thread_id, v)
                else:
                    loop = asyncio.get_event_loop()
                    self.approval_result = await loop.run_in_executor(
                        None, self.on_approval, self.thread_id, v
                    )
                break


def _default_on_approval(client, callback_url: str, callback_secret: str):
    """Sync on_approval — delegates to ApprovalClient.create_approval."""
    def _call(thread_id: str, v: dict) -> dict:
        return client.create_approval(
            thread_id=thread_id,
            action=v.get("action", "unknown"),
            message=v.get("message", "Approval required"),
            channel=v.get("channel", "slack"),
            tool=v.get("tool", {}),
            agent_callback_url=callback_url,
            agent_callback_secret=callback_secret,
        )
    return _call


def _default_on_approval_async(client, callback_url: str, callback_secret: str):
    """Async on_approval — delegates to ApprovalClient.acreate_approval."""
    async def _call(thread_id: str, v: dict) -> dict:
        return await client.acreate_approval(
            thread_id=thread_id,
            action=v.get("action", "unknown"),
            message=v.get("message", "Approval required"),
            channel=v.get("channel", "slack"),
            tool=v.get("tool", {}),
            agent_callback_url=callback_url,
            agent_callback_secret=callback_secret,
        )
    return _call


class ApprovalGraph:
    """Wraps a compiled LangGraph graph and handles interrupt detection,
    approval creation, and resume — so agents need no boilerplate.

    Two construction styles:

    **Default** — delegates to the intrupt approval API::

        approval_graph = ApprovalGraph(
            graph=graph,
            client=approval_client,
            callback_url="http://localhost:8081/resume",
            callback_secret=os.getenv("AGENT_RESUME_SECRET", ""),
        )

    **Custom** — bring your own approval logic (useful for tests, other channels,
    or any scenario where you don't want to call the approval API)::

        def my_approval(thread_id: str, interrupt_value: dict) -> dict:
            # must return a dict; include "approval_id" to signal pending
            send_email(thread_id, interrupt_value)
            return {"approval_id": store_pending(thread_id)}

        approval_graph = ApprovalGraph(graph=graph, on_approval=my_approval)
    """

    def __init__(
        self,
        graph,
        client=None,
        callback_url: str = "",
        callback_secret: str = "",
        on_approval: Optional[Callable] = None,
        on_approval_async: Optional[Callable] = None,
    ):
        """
        ``on_approval``       — sync callable used by ``invoke`` / ``resume``.
        ``on_approval_async`` — async callable used by ``ainvoke`` / ``aresume``.

        If only one is provided it is used for both paths (sync is wrapped in an
        executor for the async path; async raises if called from the sync path).
        If neither is provided, ``client`` must be given and both defaults are
        built from it automatically.
        """
        if on_approval is None and on_approval_async is None and client is None:
            raise ValueError(
                "Provide 'client', 'on_approval', or 'on_approval_async'."
            )
        self.graph = graph
        self._on_approval = (
            on_approval
            or (None if on_approval_async else _default_on_approval(client, callback_url, callback_secret))
        )
        self._on_approval_async = (
            on_approval_async
            or (None if on_approval else _default_on_approval_async(client, callback_url, callback_secret))
        )

    def _make_handler(self, thread_id: str) -> ApprovalCallbackHandler:
        if self._on_approval is None:
            raise RuntimeError("No sync on_approval — use ainvoke instead.")
        return ApprovalCallbackHandler(thread_id=thread_id, on_approval=self._on_approval)

    def _make_async_handler(self, thread_id: str) -> AsyncApprovalCallbackHandler:
        fn = self._on_approval_async or self._on_approval
        if fn is None:
            raise RuntimeError("No on_approval callable configured.")
        return AsyncApprovalCallbackHandler(thread_id=thread_id, on_approval=fn)

    def invoke(self, input: dict, thread_id: str) -> dict:
        handler = self._make_handler(thread_id)
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}
        result = self.graph.invoke(input, config=config)
        return self._format_response(thread_id, result, handler)

    def resume(self, thread_id: str, approved: bool, approval_id: Optional[str] = None) -> dict:
        handler = self._make_handler(thread_id)
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}
        result = self.graph.invoke(
            Command(resume={"approved": approved, "approval_id": approval_id}),
            config=config,
        )
        return self._format_response(thread_id, result, handler)

    async def ainvoke(self, input: dict, thread_id: str) -> dict:
        handler = self._make_async_handler(thread_id)
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}
        result = await self.graph.ainvoke(input, config=config)
        return self._format_response(thread_id, result, handler)

    async def aresume(self, thread_id: str, approved: bool, approval_id: Optional[str] = None) -> dict:
        handler = self._make_async_handler(thread_id)
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}
        result = await self.graph.ainvoke(
            Command(resume={"approved": approved, "approval_id": approval_id}),
            config=config,
        )
        return self._format_response(thread_id, result, handler)

    def pending(self, thread_id: str) -> bool:
        """Return True if this thread is paused on an approval interrupt."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        return any(
            isinstance(getattr(itr, "value", itr), dict)
            and getattr(itr, "value", itr).get("approval_required")
            for task in (state.tasks or ())
            for itr in (task.interrupts or ())
        )

    def _format_response(self, thread_id: str, result: dict, handler: ApprovalCallbackHandler) -> dict:
        if handler.approval_result and "approval_id" in handler.approval_result:
            return {
                "status": "pending_approval",
                "thread_id": thread_id,
                "approval_id": handler.approval_result["approval_id"],
            }
        return {
            "status": "complete",
            "thread_id": thread_id,
            "messages": [
                {"type": m.__class__.__name__, "content": m.content}
                for m in result.get("messages", [])
            ],
        }


def approval_required(**configs):
    """Decorate a tool so it pauses for human approval before executing.

    The decorated function pauses via `langgraph.types.interrupt(...)`, which
    commits the checkpoint *before* any external side-effect (Slack DM, email)
    is fired. The agent's `/call-tool` handler observes the interrupt, creates
    an approval record on the API, and the human's decision flows back through
    `/resume`. The tool body only runs if the decision is `approved`.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            payload = {
                "approval_required": True,
                "action": configs.get("action", func.__name__),
                "message": configs.get(
                    "message", f"Approval required for {func.__name__}"
                ),
                "channel": configs.get("channel", "slack"),
                "tool": {
                    "name": func.__name__,
                    "description": func.__doc__,
                    "kwargs": _filter_kwargs(kwargs, configs.get("args")),
                },
            }

            decision = interrupt(payload)

            if not isinstance(decision, dict) or not decision.get("approved"):
                return {
                    "status": "cancelled",
                    "tool": func.__name__,
                    "message": f"{func.__name__} was not approved",
                }

            return func(*args, **kwargs)

        return wrapper

    return decorator


