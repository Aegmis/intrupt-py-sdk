"""Tests for the @approval_required decorator on the LangGraph adapter.

The decorator no longer uses langgraph.types.interrupt — it gates through
gate.py's asyncio Future, identical to the ADK / OpenAI Agents / CrewAI
adapters. Tests exercise the decorator in isolation (no full graph run needed).
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from intrupt_py_sdk.adapters.langgraph import approval_required, _current_thread_id
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.core.gate import _pending, _session_to_approval
from intrupt_py_sdk.utils import _filter_kwargs


@pytest.fixture(autouse=True)
def clear_state():
    ApprovalMiddleware._instance = None
    _pending.clear()
    _session_to_approval.clear()
    yield
    _pending.clear()
    _session_to_approval.clear()
    ApprovalMiddleware._instance = None


def _make_client(status="pending", approval_id="ap-1"):
    client = AsyncMock()
    client.acreate_approval.return_value = {"status": status, "approval_id": approval_id}
    return client


class TestApprovalRequired:
    async def test_approved_runs_tool_body(self):
        client = _make_client(status="pending", approval_id="ap-lg-1")

        @approval_required(action="buy", message="ok?", channel="slack", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            return {"status": "success", "symbol": symbol}

        with patch("intrupt_py_sdk.adapters.langgraph.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            _current_thread_id.set("t-lg-1")

            async def _approve():
                await asyncio.sleep(0.05)
                gate.resolve("ap-lg-1", approved=True)

            asyncio.create_task(_approve())
            result = await buy(symbol="AAPL")

        assert result == {"status": "success", "symbol": "AAPL"}

    async def test_rejected_returns_cancelled(self):
        client = _make_client(status="pending", approval_id="ap-lg-2")

        @approval_required(action="sell", message="ok?", channel="slack")
        def sell(symbol: str) -> dict:
            """Sell."""
            return {"status": "success", "symbol": symbol}

        with patch("intrupt_py_sdk.adapters.langgraph.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            _current_thread_id.set("t-lg-2")

            async def _reject():
                await asyncio.sleep(0.05)
                gate.resolve("ap-lg-2", approved=False)

            asyncio.create_task(_reject())
            result = await sell(symbol="TSLA")

        assert result["status"] == "cancelled"
        assert result["tool"] == "sell"

    async def test_auto_approved_skips_gate(self):
        client = _make_client(status="approved", approval_id="")

        @approval_required(action="ping", message="ok?", channel="slack")
        def ping() -> str:
            """Ping."""
            return "pong"

        with patch("intrupt_py_sdk.adapters.langgraph.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            _current_thread_id.set("t-lg-auto")
            result = await ping()

        assert result == "pong"

    async def test_async_tool_body_supported(self):
        client = _make_client(status="pending", approval_id="ap-lg-async")

        @approval_required(action="fetch", message="ok?")
        async def fetch(url: str) -> str:
            """Fetch."""
            return f"fetched:{url}"

        with patch("intrupt_py_sdk.adapters.langgraph.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            _current_thread_id.set("t-lg-async")

            async def _approve():
                await asyncio.sleep(0.05)
                gate.resolve("ap-lg-async", approved=True)

            asyncio.create_task(_approve())
            result = await fetch(url="http://example.com")

        assert result == "fetched:http://example.com"

    async def test_args_filter_limits_payload_kwargs(self):
        captured = {}
        client = AsyncMock()

        async def _capture(**kwargs):
            captured.update(kwargs)
            return {"status": "pending", "approval_id": "ap-filt"}

        client.acreate_approval.side_effect = _capture

        @approval_required(action="x", message="ok?", args=["symbol"])
        def x(symbol: str, secret: str = "shh") -> dict:
            """x."""
            return {}

        with patch("intrupt_py_sdk.adapters.langgraph.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            _current_thread_id.set("t-filt")

            async def _approve():
                await asyncio.sleep(0.05)
                gate.resolve("ap-filt", approved=True)

            asyncio.create_task(_approve())
            await x(symbol="AAPL", secret="hush")

        assert captured["tool"]["kwargs"] == {"symbol": "AAPL"}
        assert "secret" not in captured["tool"]["kwargs"]

    async def test_action_defaults_to_function_name(self):
        captured = {}
        client = AsyncMock()

        async def _capture(**kwargs):
            captured.update(kwargs)
            return {"status": "pending", "approval_id": "ap-name"}

        client.acreate_approval.side_effect = _capture

        @approval_required()
        def my_fn() -> dict:
            """No-op."""
            return {}

        with patch("intrupt_py_sdk.adapters.langgraph.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            _current_thread_id.set("t-name")

            async def _approve():
                await asyncio.sleep(0.05)
                gate.resolve("ap-name", approved=True)

            asyncio.create_task(_approve())
            await my_fn()

        assert captured["action"] == "my_fn"
        assert captured["message"] == "Approval required for my_fn"

    async def test_callback_url_forwarded_to_api(self):
        from intrupt_py_sdk.adapters.langgraph import configure
        configure("http://agent/resume", "s3cr3t")

        captured = {}
        client = AsyncMock()

        async def _capture(**kwargs):
            captured.update(kwargs)
            return {"status": "pending", "approval_id": "ap-cb"}

        client.acreate_approval.side_effect = _capture

        @approval_required(action="act", message="ok?")
        def act() -> dict:
            """Act."""
            return {}

        with patch("intrupt_py_sdk.adapters.langgraph.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            _current_thread_id.set("t-cb")

            async def _approve():
                await asyncio.sleep(0.05)
                gate.resolve("ap-cb", approved=True)

            asyncio.create_task(_approve())
            await act()

        assert captured["agent_callback_url"] == "http://agent/resume"
        assert captured["agent_callback_secret"] == "s3cr3t"


class TestFilterKwargs:
    def test_no_allowlist_drops_config(self):
        assert _filter_kwargs({"a": 1, "config": {"x": 1}}, None) == {"a": 1}

    def test_with_allowlist_only_keeps_allowed(self):
        assert _filter_kwargs({"a": 1, "b": 2, "c": 3}, ["a", "c"]) == {"a": 1, "c": 3}

    def test_empty_allowlist_shows_nothing(self):
        assert _filter_kwargs({"a": 1, "config": {}}, []) == {}


class TestMiddlewareSingleton:
    def test_second_construction_does_not_replace_client(self):
        ApprovalMiddleware._instance = None
        m1 = ApprovalMiddleware(base_url="http://first", api_key="sk_org_org_test1_abcdef0123456789")
        c1 = m1.client
        m2 = ApprovalMiddleware(base_url="http://second", api_key="sk_org_org_test2_abcdef0123456789")
        assert m1 is m2
        assert m2.client is c1
        assert c1.base_url == "http://first"

    def test_get_client_before_init_raises(self):
        ApprovalMiddleware._instance = None
        with pytest.raises(RuntimeError):
            ApprovalMiddleware.get_client()
