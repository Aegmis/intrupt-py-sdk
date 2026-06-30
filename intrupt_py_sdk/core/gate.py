import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_pending: dict[str, asyncio.Future] = {}
_session_to_approval: dict[str, str] = {}
# Callbacks fired when a session transitions to pending-approval state.
_pending_callbacks: dict[str, list[Callable]] = {}


async def request_approval(client, session_id: str, payload: dict) -> tuple[str, asyncio.Future]:
    """POST to approval API, create a Future, return (approval_id, future).

    If the API auto-approves (policy engine or enforce_policies=false), the
    returned future is already resolved so the tool continues immediately.
    Any registered pending callback for this session is fired when the approval
    transitions to 'pending', so callers using wait_for_approval() wake up.
    """
    try:
        result = await client.acreate_approval(thread_id=session_id, **payload)
    except Exception as exc:
        _surface_api_error(exc)
        raise
    status = result.get("status", "")
    if status != "pending":
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result(status in ("approved", "audited"))
        return result.get("approval_id", ""), fut
    approval_id = result["approval_id"]
    fut = asyncio.get_event_loop().create_future()
    _pending[approval_id] = fut
    _session_to_approval[session_id] = approval_id
    # Notify any waiter (e.g. ApprovalRunner.run) that approval is now pending.
    for cb in _pending_callbacks.get(session_id, []):
        try:
            cb()
        except Exception:
            logger.exception("gate: pending callback raised for session %s", session_id)
    return approval_id, fut


def register_pending_callback(session_id: str, callback: Callable) -> None:
    """Register a zero-argument callable to be called when session_id goes pending."""
    _pending_callbacks.setdefault(session_id, []).append(callback)


def unregister_pending_callbacks(session_id: str) -> None:
    """Remove all pending callbacks for session_id."""
    _pending_callbacks.pop(session_id, None)


def resolve(approval_id: str, approved: bool) -> None:
    """Called by the adapter's /resume endpoint to unblock the waiting tool."""
    fut = _pending.pop(approval_id, None)
    stale = [k for k, v in _session_to_approval.items() if v == approval_id]
    for k in stale:
        _session_to_approval.pop(k, None)
    if fut and not fut.done():
        fut.set_result(approved)


def get_pending(session_id: str) -> Optional[str]:
    """Return the approval_id currently blocking session_id, or None."""
    return _session_to_approval.get(session_id)


def is_pending(approval_id: str) -> bool:
    return approval_id in _pending


def _surface_api_error(exc: Exception) -> None:
    """Log a human-readable explanation for common approval API errors."""
    try:
        import httpx
        if not isinstance(exc, httpx.HTTPStatusError):
            logger.error("approval API call failed: %s", exc)
            return
        status = exc.response.status_code
        try:
            detail = exc.response.json().get("detail", exc.response.text[:200])
        except Exception:
            detail = exc.response.text[:200]
        url = str(exc.request.url)
        if status == 404:
            logger.error(
                "approval API returned 404 for %s — APPROVAL_BASE_URL is pointing "
                "at the wrong server. The approval API runs on port 8080; the agent ",
                url,
            )
        elif status == 401:
            logger.error(
                "approval API returned 401 — API key is invalid or expired. "
                "Check APPROVAL_API_KEY in your .env and regenerate it at "
                "Account → API Keys if it has expired. Detail: %s", detail
            )
        elif status == 400:
            logger.error(
                "approval API returned 400 — bad request. Detail: %s. "
                "If the error is 'unsupported channel: email', restart the "
                "intrupt API server so the updated channel list takes effect.",
                detail,
            )
        elif status == 403:
            logger.error(
                "approval API returned 403 — API key does not have access to "
                "this org. Check APPROVAL_API_KEY and APPROVAL_BASE_URL. Detail: %s", detail
            )
        else:
            logger.error("approval API returned %d for %s: %s", status, url, detail)
    except Exception:
        logger.error("approval API call failed: %s", exc)
