import asyncio
from typing import Optional

_pending: dict[str, asyncio.Future] = {}
_session_to_approval: dict[str, str] = {}


async def request_approval(client, session_id: str, payload: dict) -> tuple[str, asyncio.Future]:
    """POST to approval API, create a Future, return (approval_id, future).

    If the API auto-approves (policy engine or enforce_policies=false), the
    returned future is already resolved so the tool continues immediately.
    """
    result = await client.acreate_approval(thread_id=session_id, **payload)
    status = result.get("status", "")
    if status != "pending":
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result(status in ("approved", "audited"))
        return result.get("approval_id", ""), fut
    approval_id = result["approval_id"]
    fut = asyncio.get_event_loop().create_future()
    _pending[approval_id] = fut
    _session_to_approval[session_id] = approval_id
    return approval_id, fut


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
