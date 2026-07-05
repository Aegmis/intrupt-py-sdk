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
- ``AEGMIS_BASE_URL``   URL of the intrupt approval API  (e.g. ``http://localhost:8080``)
- ``AEGMIS_API_KEY``    API key for the approval API
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
import logging
import uuid
from typing import Optional, Type

logger = logging.getLogger(__name__)

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.core.client import error_status_code, user_facing_error
from intrupt_py_sdk.utils.utils import _filter_kwargs

_CALLBACK_URL: str = ""
_CALLBACK_SECRET: str = ""

# Each asyncio Task inherits its own copy.
_current_run_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "crew_run_id", default=""
)
# Keyed by run_id. Set by _GatedTool._arun when the approval API call fails.
# Checked in _run_crew after kickoff_async returns so the error surfaces as
# status="error" instead of being swallowed by CrewAI into an LLM summary.
# Module-level dict (not a ContextVar) because CrewAI calls _arun in a
# sub-task whose context changes don't propagate back to the parent task.
_tool_api_errors: dict[str, Exception] = {}


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
    # Forward the original tool's args_schema so the LLM sees the correct
    # parameter names and types. Without this, _GatedTool gets an empty schema
    # and the agent never generates the right arguments.
    _original_schema = getattr(_original_tool, "args_schema", None)

    class _GatedTool(BaseTool):
        name: str = _original_tool.name
        description: str = _raw_desc
        args_schema: type = _original_schema  # type: ignore[assignment]

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
            try:
                approval_id, future = await gate.request_approval(client, run_id, payload)
            except Exception as exc:
                # Record in the module-level dict (keyed by run_id) so _run_crew
                # can surface it as status="error". A ContextVar won't work here
                # because CrewAI calls _arun in a sub-task whose context changes
                # don't propagate back to the parent task.
                _tool_api_errors[run_id] = exc
                logger.error("approval API call failed for tool %r: %s", _original_tool.name, exc)
                raise
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
        self._resolved: set[str] = set()  # approval_ids already resolved — guards against retries

    async def kickoff(self, run_id: str, inputs: dict) -> dict:
        # Event fires as soon as the gated tool registers an approval with the
        # gate — which happens inside the background task, potentially long after
        # the LLM finishes deciding to call the tool.
        approval_ready = asyncio.Event()
        gate.register_pending_callback(run_id, approval_ready.set)

        _current_run_id.set(run_id)
        task = asyncio.create_task(self._run_crew(run_id, inputs))
        self._tasks[run_id] = task

        # Race: crew finishes normally vs. approval gate fires vs. outer timeout.
        crew_done = asyncio.ensure_future(asyncio.shield(task))
        approval_fired = asyncio.ensure_future(approval_ready.wait())
        done, pending = await asyncio.wait(
            [crew_done, approval_fired],
            timeout=120,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Cancel the helper futures to avoid warnings; don't cancel the real task.
        for f in (crew_done, approval_fired):
            f.cancel()

        gate.unregister_pending_callbacks(run_id)

        if crew_done in done or task.done():
            return task.result()

        # Approval gate fired (or outer timeout hit — still return pending).
        approval_id = gate.get_pending(run_id)
        return {
            "status": "pending_approval",
            "run_id": run_id,
            "approval_id": approval_id,
        }

    def resume(self, run_id: str, approved: bool, approval_id: str = "") -> dict:
        """Resolve the gate and return immediately — do not await the crew task.

        Callers (the /resume HTTP handler) must return a response quickly so
        that Slack (or any other webhook source) does not time out and retry.
        The crew continues running in the background; poll /result/{run_id}
        or listen for the agent_callback_url to know when it finishes.

        Returns ``{"status": "already_resolved"}`` when called more than once
        for the same approval_id so the caller can detect and ignore duplicates.
        """
        if not approval_id:
            approval_id = gate.get_pending(run_id) or ""
        if approval_id in self._resolved:
            logger.warning("duplicate resume for approval_id %s (run_id %s) — ignored", approval_id, run_id)
            return {"status": "already_resolved", "run_id": run_id, "approval_id": approval_id}
        self._resolved.add(approval_id)
        resolved = gate.resolve(approval_id, approved)
        if not resolved:
            logger.warning("gate.resolve(%s) found no pending future — approval may have timed out", approval_id)
        return {"status": "accepted", "run_id": run_id, "approval_id": approval_id}

    async def _run_crew(self, run_id: str, inputs: dict) -> dict:
        try:
            result = await self._crew.kickoff_async(inputs=inputs)
            api_err = _tool_api_errors.pop(run_id, None)
            if api_err is not None:
                r = {"status": "error", "run_id": run_id, "error": user_facing_error(api_err), "status_code": error_status_code(api_err)}
            else:
                r = {"status": "complete", "run_id": run_id, "result": str(result)}
        except Exception as exc:
            _tool_api_errors.pop(run_id, None)
            logger.exception("_run_crew failed for run %s: %s", run_id, exc)
            r = {"status": "error", "run_id": run_id, "error": user_facing_error(exc), "status_code": error_status_code(exc)}
        self._results[run_id] = r
        return r
