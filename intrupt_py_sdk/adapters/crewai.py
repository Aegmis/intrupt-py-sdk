"""CrewAI adapter for intrupt human-in-the-loop approvals.

Install
-------
This adapter requires the ``crewai`` package which is **not** installed by
default. Install it with the bundled extras group::

    pip install "intrupt-py-sdk[crewai]"

Or install the framework package directly::

    pip install crewai

Required packages
-----------------
- ``crewai>=0.1``  (provides ``crewai``, ``crewai.tools.BaseTool``)

Environment variables
---------------------
- ``APPROVAL_BASE_URL``   URL of the intrupt approval API  (e.g. ``http://localhost:8080``)
- ``APPROVAL_API_KEY``    API key for the approval API
- ``AGENT_PUBLIC_URL``    Public URL of this agent server (used as callback base)
- ``OPENAI_API_KEY``      OpenAI API key (CrewAI uses OpenAI by default)

Usage
-----
::

    from intrupt_py_sdk.adapters.crewai import approval_required, ApprovalCrew

    class PurchaseTool(BaseTool):
        name = "purchase_stock"
        description = "Buy shares of a stock."
        ...

    gated_purchase = approval_required(
        PurchaseTool(),
        action="purchase_stock",
        message="Approve stock purchase?",
        channel="slack",
        args=["symbol", "quantity"],
    )

    crew = ApprovalCrew(
        crew=Crew(agents=[...], tasks=[...], tools=[gated_purchase]),
        callback_url="https://my-agent.example.com/resume",
        callback_secret="...",
    )

    result = await crew.kickoff(run_id, inputs={"request": "buy AAPL"})
    # if result["status"] == "pending_approval": wait for /resume call
    result = await crew.resume(run_id, approved=True, approval_id="...")
"""
import asyncio
import contextvars
import uuid
from typing import Optional, Type

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.utils.utils import _filter_kwargs

_CALLBACK_URL: str = ""
_CALLBACK_SECRET: str = ""

# Each asyncio Task inherits its own copy.
_current_run_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "crew_run_id", default=""
)


def configure(callback_url: str, callback_secret: str = "") -> None:
    global _CALLBACK_URL, _CALLBACK_SECRET
    _CALLBACK_URL = callback_url
    _CALLBACK_SECRET = callback_secret


def approval_required(
    tool,
    action: str,
    message: str,
    channel: str = "slack",
    args: Optional[list] = None,
):
    """Wrap a CrewAI ``BaseTool`` instance with an approval gate.

    Returns a new ``BaseTool`` instance whose ``_arun``/``_run`` intercept
    the call, POST to the intrupt approval API, and suspend until the human
    decides. The original tool's logic only runs if approved.

    Args:
        tool:    An instantiated ``BaseTool`` subclass to gate.
        action:  Short approval action identifier.
        message: Human-readable reason shown in the approval request.
        channel: Dispatch channel (default ``"slack"``).
        args:    Tool kwarg names to include in the approval payload; ``None``
                 includes all kwargs (except framework-internal ones).
    """
    from crewai.tools import BaseTool  # type: ignore[import]

    _original_tool = tool
    # Use the raw (pre-validator) description from the class definition so the
    # gated tool doesn't nest CrewAI's auto-augmented description.
    _raw_desc = (
        type(_original_tool).model_fields.get("description").default
        if hasattr(type(_original_tool), "model_fields")
        else _original_tool.description
    )

    class _GatedTool(BaseTool):
        name: str = _original_tool.name
        description: str = _raw_desc

        async def _arun(self, **kwargs):
            run_id = _current_run_id.get() or str(uuid.uuid4())
            filtered = _filter_kwargs(kwargs, args)
            payload = {
                "action": action,
                "message": message,
                "channel": channel,
                "tool": {
                    "name": _original_tool.name,
                    "description": _original_tool.description,
                    "kwargs": filtered,
                },
                "agent_callback_url": _CALLBACK_URL,
                "agent_callback_secret": _CALLBACK_SECRET,
                "adapter": "crewai",
            }
            client = ApprovalMiddleware.get_client()
            approval_id, future = await gate.request_approval(client, run_id, payload)
            approved = await future
            if not approved:
                return f"Action cancelled by human reviewer: {_original_tool.name}"
            return await _original_tool._arun(**kwargs)

        def _run(self, **kwargs):
            return asyncio.run(self._arun(**kwargs))

    return _GatedTool()


class ApprovalCrew:
    """Wraps a CrewAI ``Crew`` to expose pending-approval state.

    ``kickoff()`` sets the run_id context var, launches ``crew.kickoff_async``
    as a background task, and returns within 1.5 s. If an ``approval_required``
    tool fires before then, the task suspends on the gate Future and
    ``kickoff()`` returns ``{"status": "pending_approval", ...}``. Call
    ``resume()`` to unblock.
    """

    def __init__(self, crew, callback_url: str, callback_secret: str = ""):
        self._crew = crew
        configure(callback_url, callback_secret)
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, dict] = {}

    async def kickoff(self, run_id: str, inputs: dict) -> dict:
        _current_run_id.set(run_id)
        task = asyncio.create_task(self._run_crew(run_id, inputs))
        self._tasks[run_id] = task
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
            return result
        except asyncio.TimeoutError:
            approval_id = gate.get_pending(run_id)
            return {
                "status": "pending_approval",
                "run_id": run_id,
                "approval_id": approval_id,
            }

    async def resume(self, run_id: str, approved: bool, approval_id: str) -> dict:
        gate.resolve(approval_id, approved)
        task = self._tasks.get(run_id)
        if task:
            await task
            return self._results.get(run_id, {"status": "complete", "run_id": run_id})
        return {"status": "not_found", "run_id": run_id}

    async def _run_crew(self, run_id: str, inputs: dict) -> dict:
        result = await self._crew.kickoff_async(inputs=inputs)
        r = {"status": "complete", "run_id": run_id, "result": str(result)}
        self._results[run_id] = r
        return r
