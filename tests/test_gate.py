"""Unit tests for core/gate.py — the asyncio Future gate."""
import asyncio
import pytest
from unittest.mock import AsyncMock

import sys, types

# Ensure the SDK package is importable from the tests directory
sys.path.insert(0, ".")

from intrupt_py_sdk.core.gate import (
    _pending,
    _session_to_approval,
    get_pending,
    is_pending,
    request_approval,
    resolve,
)


@pytest.fixture(autouse=True)
def clear_gate():
    """Reset module-level dicts before every test."""
    _pending.clear()
    _session_to_approval.clear()
    yield
    _pending.clear()
    _session_to_approval.clear()


class TestRequestApproval:
    async def test_pending_status_creates_future(self):
        client = AsyncMock()
        client.acreate_approval.return_value = {"status": "pending", "approval_id": "ap-1"}

        approval_id, fut = await request_approval(
            client, "sess-1",
            {"action": "buy", "message": "ok", "channel": "slack",
             "tool": {"name": "t", "description": "", "kwargs": {}},
             "agent_callback_url": "", "agent_callback_secret": ""},
        )
        assert approval_id == "ap-1"
        assert not fut.done()
        assert is_pending("ap-1")
        assert get_pending("sess-1") == "ap-1"

    async def test_auto_approved_returns_resolved_future(self):
        client = AsyncMock()
        client.acreate_approval.return_value = {"status": "approved", "approval_id": "ap-2"}

        approval_id, fut = await request_approval(
            client, "sess-2",
            {"action": "buy", "message": "ok", "channel": "slack",
             "tool": {"name": "t", "description": "", "kwargs": {}},
             "agent_callback_url": "", "agent_callback_secret": ""},
        )
        assert fut.done()
        assert fut.result() is True
        assert not is_pending("ap-2")

    async def test_audited_status_treated_as_approved(self):
        client = AsyncMock()
        client.acreate_approval.return_value = {"status": "audited", "approval_id": "ap-3"}

        _, fut = await request_approval(
            client, "sess-3",
            {"action": "buy", "message": "ok", "channel": "slack",
             "tool": {"name": "t", "description": "", "kwargs": {}},
             "agent_callback_url": "", "agent_callback_secret": ""},
        )
        assert fut.done() and fut.result() is True


class TestResolve:
    async def test_resolve_approved_unblocks_future(self):
        client = AsyncMock()
        client.acreate_approval.return_value = {"status": "pending", "approval_id": "ap-4"}

        _, fut = await request_approval(
            client, "sess-4",
            {"action": "buy", "message": "ok", "channel": "slack",
             "tool": {"name": "t", "description": "", "kwargs": {}},
             "agent_callback_url": "", "agent_callback_secret": ""},
        )

        resolve("ap-4", approved=True)
        assert fut.done()
        assert fut.result() is True
        assert not is_pending("ap-4")
        assert get_pending("sess-4") is None

    async def test_resolve_rejected_sets_false(self):
        client = AsyncMock()
        client.acreate_approval.return_value = {"status": "pending", "approval_id": "ap-5"}

        _, fut = await request_approval(
            client, "sess-5",
            {"action": "buy", "message": "ok", "channel": "slack",
             "tool": {"name": "t", "description": "", "kwargs": {}},
             "agent_callback_url": "", "agent_callback_secret": ""},
        )
        resolve("ap-5", approved=False)
        assert fut.result() is False

    def test_resolve_unknown_id_is_noop(self):
        resolve("does-not-exist", approved=True)  # must not raise

    async def test_double_resolve_is_safe(self):
        client = AsyncMock()
        client.acreate_approval.return_value = {"status": "pending", "approval_id": "ap-6"}

        _, fut = await request_approval(
            client, "sess-6",
            {"action": "buy", "message": "ok", "channel": "slack",
             "tool": {"name": "t", "description": "", "kwargs": {}},
             "agent_callback_url": "", "agent_callback_secret": ""},
        )
        resolve("ap-6", approved=True)
        resolve("ap-6", approved=False)  # second call — future already done, noop
        assert fut.result() is True


class TestGetPending:
    def test_unknown_session_returns_none(self):
        assert get_pending("no-such-session") is None
