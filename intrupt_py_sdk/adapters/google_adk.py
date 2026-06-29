"""Google ADK adapter for intrupt human-in-the-loop approvals.

Usage::

    from intrupt_py_sdk.adapters.google_adk import approval_required, ApprovalRunner

    @approval_required(action="purchase_stock", message="Approve stock purchase?", channel="slack", args=["symbol", "quantity"])
    async def purchase_stock(symbol: str, quantity: int, tool_context=None) -> str:
        ...

    runner = ApprovalRunner(
        agent=my_agent,
        app_name="finance-bot",
        session_service=session_svc,
        callback_url="https://my-agent.example.com/resume",
        callback_secret="...",
    )

    result = await runner.run(session_id, message)
    # if result["status"] == "pending_approval": wait for /resume call
    result = await runner.resume(session_id, approved=True, approval_id="...")
"""
import asyncio
import uuid
from functools import wraps
from typing import Callable, Optional

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.utils.utils import _filter_kwargs

_CALLBACK_URL: str = ""
_CALLBACK_SECRET: str = ""


def configure(callback_url: str, callback_secret: str = "") -> None:
    global _CALLBACK_URL, _CALLBACK_SECRET
    _CALLBACK_URL = callback_url
    _CALLBACK_SECRET = callback_secret


def approval_required(
    action: str,
    message: str,
    channel: str = "slack",
    args: Optional[list] = None,
) -> Callable:
    """Decorator for ADK tool functions that require human approval.

    The decorated function must be async. ADK injects ``tool_context`` as a
    kwarg; the decorator reads ``tool_context.invocation_context.session.id``
    to get the session identity for the gate.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*fargs, **kwargs):
            tool_context = kwargs.get("tool_context")
            try:
                session_id = tool_context.invocation_context.session.id
            except AttributeError:
                session_id = str(uuid.uuid4())

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
            }

            client = ApprovalMiddleware.get_client()
            approval_id, future = await gate.request_approval(client, session_id, payload)

            approved = await future
            if not approved:
                return {"status": "cancelled", "tool": func.__name__}
            return await func(*fargs, **kwargs)

        return wrapper
    return decorator


class ApprovalRunner:
    """Wraps ``google.adk.Runner`` to expose pending-approval state.

    ``run()`` launches the ADK runner as a background asyncio task and returns
    within 1.5 s. If the agent's tool fires an approval request before then,
    the background task suspends on the gate Future and ``run()`` returns
    ``{"status": "pending_approval", ...}``. Call ``resume()`` after the
    human decides to unblock the task and collect the final result.
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

    async def run(self, session_id: str, message: str) -> dict:
        task = asyncio.create_task(self._run_agent(session_id, message))
        self._tasks[session_id] = task
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
            self._results[session_id] = result
            return result
        except asyncio.TimeoutError:
            approval_id = gate.get_pending(session_id)
            return {
                "status": "pending_approval",
                "session_id": session_id,
                "approval_id": approval_id,
            }

    async def resume(self, session_id: str, approved: bool, approval_id: str) -> dict:
        gate.resolve(approval_id, approved)
        task = self._tasks.get(session_id)
        if task:
            await task
            return self._results.get(session_id, {"status": "complete", "session_id": session_id})
        return {"status": "not_found", "session_id": session_id}

    async def _run_agent(self, session_id: str, message: str) -> dict:
        from google.genai.types import Content, Part  # type: ignore[import]

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
        result = {"status": "complete", "session_id": session_id, "result": final_text}
        self._results[session_id] = result
        return result
