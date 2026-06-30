"""OpenAI Agents SDK adapter for intrupt human-in-the-loop approvals.

Install
-------
This adapter requires the ``openai-agents`` package which is **not** installed
by default. Install it with the bundled extras group::

    pip install "intrupt-py-sdk[openai-agents]"

Or install the framework package directly::

    pip install openai-agents

Required packages
-----------------
- ``openai-agents>=0.0.3``  (provides ``agents``, ``agents.tool``)

Environment variables
---------------------
- ``APPROVAL_BASE_URL``   URL of the intrupt approval API  (e.g. ``http://localhost:8080``)
- ``APPROVAL_API_KEY``    API key for the approval API
- ``AGENT_PUBLIC_URL``    Public URL of this agent server (used as callback base)
- ``OPENAI_API_KEY``      OpenAI API key

Usage
-----
::

    from intrupt_py_sdk.adapters.openai_agents import approval_required, ApprovalAgentRunner

    @function_tool
    @approval_required(action="purchase_stock", message="Approve stock purchase?",
                       channel="slack", args=["symbol", "quantity"])
    async def purchase_stock(symbol: str, quantity: int) -> str:
        ...

    runner = ApprovalAgentRunner(
        agent=my_agent,
        callback_url="https://my-agent.example.com/resume",
        callback_secret="...",
    )

    result = await runner.run(thread_id, message)
    # if result["status"] == "pending_approval": wait for /resume call
    result = await runner.resume(thread_id, approved=True, approval_id="...")
"""
import asyncio
import contextvars
import uuid
from functools import wraps
from typing import Callable, Optional

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.utils.utils import _filter_kwargs

_CALLBACK_URL: str = ""
_CALLBACK_SECRET: str = ""

# Each asyncio Task inherits its own copy of this context var, so concurrent
# runs don't share a thread_id.
_current_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "oai_thread_id", default=""
)


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
    """Decorator for OpenAI Agents SDK ``@function_tool`` functions.

    Apply *inside* ``@function_tool``::

        @function_tool
        @approval_required(action="...", message="...")
        async def my_tool(...) -> str: ...

    The thread_id is picked up from the ``_current_thread_id`` context var,
    which ``ApprovalAgentRunner.run()`` sets before launching the agent task.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*fargs, **kwargs):
            thread_id = _current_thread_id.get() or str(uuid.uuid4())
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
                "adapter": "openai_agents",
            }

            client = ApprovalMiddleware.get_client()
            approval_id, future = await gate.request_approval(client, thread_id, payload)

            approved = await future
            if not approved:
                return {"status": "cancelled", "tool": func.__name__}
            return await func(*fargs, **kwargs)

        return wrapper
    return decorator


class ApprovalAgentRunner:
    """Wraps an OpenAI Agents SDK Agent to expose pending-approval state.

    ``run()`` sets the context var, launches the agent in a background task,
    and returns within 1.5 s. If an ``@approval_required`` tool fires before
    then, the task suspends on the gate Future and ``run()`` returns
    ``{"status": "pending_approval", ...}``. Call ``resume()`` to unblock.
    """

    def __init__(self, agent, callback_url: str, callback_secret: str = ""):
        self._agent = agent
        configure(callback_url, callback_secret)
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, dict] = {}

    async def run(self, thread_id: str, message: str) -> dict:
        _current_thread_id.set(thread_id)
        task = asyncio.create_task(self._run_agent(thread_id, message))
        self._tasks[thread_id] = task
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
            return result
        except asyncio.TimeoutError:
            approval_id = gate.get_pending(thread_id)
            return {
                "status": "pending_approval",
                "thread_id": thread_id,
                "approval_id": approval_id,
            }

    async def resume(self, thread_id: str, approved: bool, approval_id: str) -> dict:
        gate.resolve(approval_id, approved)
        task = self._tasks.get(thread_id)
        if task:
            await task
            return self._results.get(thread_id, {"status": "complete", "thread_id": thread_id})
        return {"status": "not_found", "thread_id": thread_id}

    async def _run_agent(self, thread_id: str, message: str) -> dict:
        from agents import Runner  # type: ignore[import]

        result = await Runner.run(self._agent, message)
        r = {
            "status": "complete",
            "thread_id": thread_id,
            "result": result.final_output,
        }
        self._results[thread_id] = r
        return r
