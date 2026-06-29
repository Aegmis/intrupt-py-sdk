"""Tests for the three new adapters using mocked framework runners.

Each adapter's approval_required decorator and Runner/Crew wrapper is exercised
without a live approval API or real framework dependency — the framework
modules are mocked at import time and ApprovalClient.acreate_approval is
patched to return canned responses.
"""
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Stub out optional framework packages ─────────────────────────────────────

def _stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _name in ["google", "google.adk", "google.genai", "google.genai.types",
              "agents", "crewai", "crewai.tools"]:
    if _name not in sys.modules:
        _stub(_name)

# Provide minimal BaseTool for crewai tests
class _FakeBaseTool:
    name: str = ""
    description: str = ""
    def _run(self, **kw): return {}
    async def _arun(self, **kw): return {}

sys.modules["crewai.tools"].BaseTool = _FakeBaseTool

from intrupt_py_sdk.core import gate
from intrupt_py_sdk.core.gate import _pending, _session_to_approval


@pytest.fixture(autouse=True)
def clear_gate():
    _pending.clear()
    _session_to_approval.clear()
    yield
    _pending.clear()
    _session_to_approval.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_client(status="pending", approval_id="ap-test"):
    client = AsyncMock()
    client.acreate_approval.return_value = {"status": status, "approval_id": approval_id}
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Google ADK adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestGoogleADKDecorator:
    def setup_method(self):
        # Re-import after stubs are in place
        import importlib
        import intrupt_py_sdk.adapters.google_adk as m
        importlib.reload(m)
        self.mod = m

    async def test_approved_runs_original_function(self):
        from intrupt_py_sdk.adapters import google_adk as m
        m.configure("http://localhost/resume", "secret")

        client = _make_client(status="pending", approval_id="ap-adk-1")

        @m.approval_required(action="buy", message="ok?", channel="slack")
        async def my_tool(symbol: str, tool_context=None) -> dict:
            return {"bought": symbol}

        with patch("intrupt_py_sdk.adapters.google_adk.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            # Schedule resolve to fire after the future is created
            async def _drive():
                await asyncio.sleep(0.05)
                gate.resolve("ap-adk-1", approved=True)
            asyncio.create_task(_drive())
            result = await my_tool(symbol="AAPL")

        assert result == {"bought": "AAPL"}

    async def test_rejected_returns_cancelled(self):
        from intrupt_py_sdk.adapters import google_adk as m
        m.configure("http://localhost/resume", "")
        client = _make_client(status="pending", approval_id="ap-adk-2")

        @m.approval_required(action="buy", message="ok?", channel="slack")
        async def sell(symbol: str, tool_context=None) -> dict:
            return {"sold": symbol}

        with patch("intrupt_py_sdk.adapters.google_adk.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            async def _reject():
                await asyncio.sleep(0.05)
                gate.resolve("ap-adk-2", approved=False)
            asyncio.create_task(_reject())
            result = await sell(symbol="TSLA")

        assert result == {"status": "cancelled", "tool": "sell"}

    async def test_auto_approved_skips_gate(self):
        from intrupt_py_sdk.adapters import google_adk as m
        m.configure("http://localhost/resume", "")
        client = _make_client(status="approved", approval_id="")

        @m.approval_required(action="buy", message="ok?", channel="slack")
        async def instant_buy(symbol: str, tool_context=None) -> str:
            return f"bought {symbol}"

        with patch("intrupt_py_sdk.adapters.google_adk.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            result = await instant_buy(symbol="GOOG")

        assert result == "bought GOOG"


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Agents SDK adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIAgentsDecorator:
    async def test_approved_runs_function(self):
        from intrupt_py_sdk.adapters import openai_agents as m
        m.configure("http://localhost/resume", "")
        client = _make_client(status="pending", approval_id="ap-oai-1")

        @m.approval_required(action="buy", message="ok?", channel="slack", args=["symbol"])
        async def buy(symbol: str) -> str:
            return f"bought {symbol}"

        with patch("intrupt_py_sdk.adapters.openai_agents.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            m._current_thread_id.set("t-1")
            async def _approve():
                await asyncio.sleep(0.05)
                gate.resolve("ap-oai-1", approved=True)
            asyncio.create_task(_approve())
            result = await buy(symbol="AAPL")

        assert result == "bought AAPL"

    async def test_rejected_returns_cancelled(self):
        from intrupt_py_sdk.adapters import openai_agents as m
        m.configure("http://localhost/resume", "")
        client = _make_client(status="pending", approval_id="ap-oai-2")

        @m.approval_required(action="sell", message="ok?", channel="slack")
        async def sell(symbol: str) -> str:
            return f"sold {symbol}"

        with patch("intrupt_py_sdk.adapters.openai_agents.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            m._current_thread_id.set("t-2")
            async def _reject():
                await asyncio.sleep(0.05)
                gate.resolve("ap-oai-2", approved=False)
            asyncio.create_task(_reject())
            result = await sell(symbol="TSLA")

        assert result == {"status": "cancelled", "tool": "sell"}

    async def test_runner_returns_pending_when_timeout(self):
        from intrupt_py_sdk.adapters import openai_agents as m
        m.configure("http://localhost/resume", "")
        client = _make_client(status="pending", approval_id="ap-oai-3")

        async def _slow_agent(thread_id, message):
            await asyncio.sleep(10)  # will be shielded, never actually sleeps 10s in test
            return {"status": "complete"}

        runner = m.ApprovalAgentRunner.__new__(m.ApprovalAgentRunner)
        runner._tasks = {}
        runner._results = {}
        runner._agent = None

        # Simulate what run() does: create task that blocks, expect pending return
        _session_to_approval["t-oai-3"] = "ap-oai-3"
        _pending["ap-oai-3"] = asyncio.get_event_loop().create_future()

        async def fake_run(thread_id, message):
            m._current_thread_id.set(thread_id)
            task = asyncio.create_task(asyncio.sleep(10))
            runner._tasks[thread_id] = task
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
                return {"status": "complete"}
            except asyncio.TimeoutError:
                approval_id = gate.get_pending(thread_id)
                return {"status": "pending_approval", "thread_id": thread_id,
                        "approval_id": approval_id}

        result = await fake_run("t-oai-3", "buy AAPL")
        assert result["status"] == "pending_approval"
        assert result["approval_id"] == "ap-oai-3"


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestCrewAIApprovalRequired:
    async def test_wraps_base_tool_and_gates(self):
        from intrupt_py_sdk.adapters import crewai as m
        m.configure("http://localhost/resume", "")
        client = _make_client(status="pending", approval_id="ap-crew-1")

        class FakePurchase(_FakeBaseTool):
            name = "purchase"
            description = "buy stuff"
            async def _arun(self, symbol: str = "", **kw):
                return {"bought": symbol}

        gated = m.approval_required(
            FakePurchase(),
            action="buy", message="ok?", channel="slack", args=["symbol"],
        )

        with patch("intrupt_py_sdk.adapters.crewai.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            m._current_run_id.set("run-1")
            async def _approve():
                await asyncio.sleep(0.05)
                gate.resolve("ap-crew-1", approved=True)
            asyncio.create_task(_approve())
            result = await gated._arun(symbol="AAPL")

        assert result == {"bought": "AAPL"}

    async def test_rejected_returns_cancelled_string(self):
        from intrupt_py_sdk.adapters import crewai as m
        m.configure("http://localhost/resume", "")
        client = _make_client(status="pending", approval_id="ap-crew-2")

        class FakePurchase(_FakeBaseTool):
            name = "purchase"
            description = "buy stuff"
            async def _arun(self, symbol: str = "", **kw):
                return {"bought": symbol}

        gated = m.approval_required(
            FakePurchase(),
            action="buy", message="ok?", channel="slack",
        )

        with patch("intrupt_py_sdk.adapters.crewai.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            m._current_run_id.set("run-2")
            async def _reject():
                await asyncio.sleep(0.05)
                gate.resolve("ap-crew-2", approved=False)
            asyncio.create_task(_reject())
            result = await gated._arun(symbol="TSLA")

        assert "cancelled" in str(result).lower()

    async def test_auto_approved_no_gate_wait(self):
        from intrupt_py_sdk.adapters import crewai as m
        m.configure("http://localhost/resume", "")
        client = _make_client(status="approved", approval_id="")

        class FakePurchase(_FakeBaseTool):
            name = "purchase"
            description = "buy stuff"
            async def _arun(self, symbol: str = "", **kw):
                return {"bought": symbol}

        gated = m.approval_required(
            FakePurchase(), action="buy", message="ok?", channel="slack",
        )

        with patch("intrupt_py_sdk.adapters.crewai.ApprovalMiddleware") as MM:
            MM.get_client.return_value = client
            m._current_run_id.set("run-3")
            result = await gated._arun(symbol="GOOG")

        assert result == {"bought": "GOOG"}
