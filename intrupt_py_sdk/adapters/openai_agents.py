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
import inspect
import logging
import uuid
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.core.client import error_status_code, user_facing_error
from intrupt_py_sdk.utils.utils import _filter_kwargs

_CALLBACK_URL: str = ""
_CALLBACK_SECRET: str = ""

# Each asyncio Task inherits its own copy of this context var, so concurrent
# runs don't share a thread_id.
_current_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "oai_thread_id", default=""
)

# Side-channel for tool-level API errors. Keyed by thread_id. The tool wrapper
# stores an exception here and re-raises it (so the SDK sees an error); after
# Runner.run returns, _run_agent checks this dict and surfaces a structured
# {"status": "error"} response instead of the LLM's natural-language summary.
_tool_api_errors: dict[str, Exception] = {}


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
        # Capture parameter names once at decoration time for positional-arg normalisation.
        _param_names = list(inspect.signature(func).parameters.keys())

        @wraps(func)
        async def wrapper(*fargs, **kwargs):
            # The openai-agents SDK may call the function with positional args.
            # Merge them into kwargs so _filter_kwargs and the approval payload
            # always see named arguments regardless of call convention.
            if fargs:
                kwargs = {**dict(zip(_param_names, fargs)), **kwargs}
                fargs = ()
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
            try:
                approval_id, future = await gate.request_approval(client, thread_id, payload)
            except Exception as exc:
                logger.error("approval API call failed for tool %r: %s", func.__name__, exc)
                _tool_api_errors[thread_id] = exc
                raise

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
            # A tool API error (e.g. channel mismatch) is stored in _tool_api_errors
            # before the SDK makes the extra LLM "handle the error" call that pushes
            # past the timeout. Surface it immediately instead of returning pending_approval.
            api_err = _tool_api_errors.pop(thread_id, None)
            if api_err is not None:
                r = {"status": "error", "thread_id": thread_id, "error": user_facing_error(api_err), "status_code": error_status_code(api_err)}
                self._results[thread_id] = r
                return r
            approval_id = gate.get_pending(thread_id)
            return {
                "status": "pending_approval",
                "thread_id": thread_id,
                "approval_id": approval_id,
            }

    async def resume(self, thread_id: str, approved: bool, approval_id: str = "") -> dict:
        """Unblock the gate Future and return immediately.

        The agent task continues in the background. Poll _results[thread_id] or
        GET /result/{thread_id} for the final result. Returning immediately here
        is critical: Slack webhooks time out after 3 s and retry if we block.
        """
        if not approval_id:
            approval_id = gate.get_pending(thread_id) or ""
        gate.resolve(approval_id, approved)
        result = self._results.get(thread_id)
        if result:
            return result
        if thread_id not in self._tasks:
            return {"status": "not_found", "thread_id": thread_id}
        return {"status": "accepted", "thread_id": thread_id, "approval_id": approval_id}

    async def _run_agent(self, thread_id: str, message: str) -> dict:
        from agents import Runner  # type: ignore[import]

        try:
            result = await Runner.run(self._agent, message)
            api_err = _tool_api_errors.pop(thread_id, None)
            if api_err is not None:
                r = {"status": "error", "thread_id": thread_id, "error": user_facing_error(api_err), "status_code": error_status_code(api_err)}
            else:
                r = {
                    "status": "complete",
                    "thread_id": thread_id,
                    "result": result.final_output,
                }
        except Exception as exc:
            _tool_api_errors.pop(thread_id, None)
            logger.exception("_run_agent failed for thread %s: %s", thread_id, exc)
            r = {"status": "error", "thread_id": thread_id, "error": user_facing_error(exc), "status_code": error_status_code(exc)}
        finally:
            self._tasks.pop(thread_id, None)
        self._results[thread_id] = r
        return r
