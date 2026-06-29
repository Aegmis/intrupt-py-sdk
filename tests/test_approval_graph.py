"""Tests for ApprovalGraph — the LangGraph runner using the gate.py pattern.

All tests are async (asyncio_mode = "auto" in pyproject.toml).
ApprovalGraph is constructed with _timeout=0.05 so tests don't wait 1.5 s for
the gate shield to fire.
"""
import asyncio
from typing import Annotated, TypedDict
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required
from intrupt_py_sdk.core import gate
from intrupt_py_sdk.core.gate import _pending, _session_to_approval


# ── Helpers ───────────────────────────────────────────────────────────────────

class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _tool_call(name: str, args: dict, tc_id: str = "tc1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tc_id, "type": "tool_call"}],
    )


def _build_graph(tool_fn):
    g = StateGraph(_State)
    g.add_node("tools", ToolNode([tool_fn]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile(checkpointer=MemorySaver())


def _make_client(status="pending", approval_id="APR-001"):
    client = AsyncMock()
    client.acreate_approval.return_value = {"status": status, "approval_id": approval_id}
    return client


def _make_ag(graph, client=None, callback_url="http://agent/resume", callback_secret="secret"):
    """Build an ApprovalGraph with a short timeout and a patched middleware."""
    ag = ApprovalGraph(
        graph=graph,
        callback_url=callback_url,
        callback_secret=callback_secret,
        _timeout=0.05,
    )
    if client:
        ApprovalMiddleware._instance = None
        ApprovalMiddleware._instance = object.__new__(ApprovalMiddleware)
        ApprovalMiddleware._instance.client = client
    return ag


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_state():
    ApprovalMiddleware._instance = None
    _pending.clear()
    _session_to_approval.clear()
    yield
    _pending.clear()
    _session_to_approval.clear()
    ApprovalMiddleware._instance = None


# ── Tests: run() ──────────────────────────────────────────────────────────────

class TestRun:
    async def test_approval_gate_returns_pending(self):
        client = _make_client(approval_id="APR-XYZ")

        @tool
        @approval_required(action="buy", message="Approve?", channel="slack", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            return {"status": "success", "symbol": symbol}

        graph = _build_graph(buy)
        ag = _make_ag(graph, client=client)

        result = await ag.run(
            {"messages": [_tool_call("buy", {"symbol": "AAPL"})]},
            thread_id="T1",
        )

        assert result["status"] == "pending_approval"
        assert result["thread_id"] == "T1"
        assert result["approval_id"] == "APR-XYZ"

    async def test_no_approval_returns_complete(self):
        @tool
        def safe_tool(x: int) -> dict:
            """No approval needed."""
            return {"result": x * 2}

        graph = _build_graph(safe_tool)
        ag = _make_ag(graph)

        result = await ag.run(
            {"messages": [_tool_call("safe_tool", {"x": 5})]},
            thread_id="T-safe",
        )

        assert result["status"] == "complete"
        assert result["thread_id"] == "T-safe"
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0

    async def test_acreate_approval_called_with_correct_fields(self):
        client = _make_client(approval_id="APR-fields")

        @tool
        @approval_required(action="transfer", message="Approve transfer", channel="slack", args=["amount"])
        def transfer(amount: float) -> dict:
            """Transfer."""
            return {"transferred": amount}

        graph = _build_graph(transfer)
        ag = _make_ag(graph, client=client, callback_url="http://myagent/resume", callback_secret="s3cr3t")

        await ag.run(
            {"messages": [_tool_call("transfer", {"amount": 500.0})]},
            thread_id="T-fields",
        )

        client.acreate_approval.assert_called_once()
        kwargs = client.acreate_approval.call_args.kwargs
        assert kwargs["thread_id"] == "T-fields"
        assert kwargs["action"] == "transfer"
        assert kwargs["message"] == "Approve transfer"
        assert kwargs["channel"] == "slack"
        assert kwargs["agent_callback_url"] == "http://myagent/resume"
        assert kwargs["agent_callback_secret"] == "s3cr3t"
        assert kwargs["tool"]["kwargs"] == {"amount": 500.0}

    async def test_acreate_approval_not_called_when_no_gate(self):
        client = _make_client()

        @tool
        def noop() -> dict:
            """No gate."""
            return {"ok": True}

        graph = _build_graph(noop)
        ag = _make_ag(graph, client=client)

        await ag.run({"messages": [_tool_call("noop", {})]}, thread_id="T-noop")

        client.acreate_approval.assert_not_called()

    async def test_complete_response_contains_messages(self):
        @tool
        def echo(msg: str) -> dict:
            """Echo."""
            return {"echo": msg}

        graph = _build_graph(echo)
        ag = _make_ag(graph)

        result = await ag.run(
            {"messages": [_tool_call("echo", {"msg": "hello"})]},
            thread_id="T-echo",
        )

        assert result["status"] == "complete"
        assert all("type" in m and "content" in m for m in result["messages"])


# ── Tests: pending() ──────────────────────────────────────────────────────────

class TestPending:
    async def test_true_when_thread_is_paused(self):
        client = _make_client(approval_id="APR-pend")

        @tool
        @approval_required(action="act", args=[])
        def act() -> dict:
            """Act."""
            return {}

        graph = _build_graph(act)
        ag = _make_ag(graph, client=client)

        await ag.run({"messages": [_tool_call("act", {})]}, thread_id="T-pend")

        assert ag.pending("T-pend") is True

    async def test_false_when_no_gate(self):
        @tool
        def safe() -> dict:
            """Safe."""
            return {"ok": True}

        graph = _build_graph(safe)
        ag = _make_ag(graph)

        await ag.run({"messages": [_tool_call("safe", {})]}, thread_id="T-no-pend")

        assert ag.pending("T-no-pend") is False

    async def test_false_for_unknown_thread(self):
        @tool
        def any_tool() -> dict:
            """Tool."""
            return {}

        graph = _build_graph(any_tool)
        ag = _make_ag(graph)

        assert ag.pending("thread-that-never-existed") is False

    async def test_false_after_resume(self):
        client = _make_client(approval_id="APR-after")
        ran = []

        @tool
        @approval_required(action="go", args=[])
        def go() -> dict:
            """Go."""
            ran.append(True)
            return {"went": True}

        graph = _build_graph(go)
        ag = _make_ag(graph, client=client)

        await ag.run({"messages": [_tool_call("go", {})]}, thread_id="T-after")
        assert ag.pending("T-after") is True

        await ag.resume("T-after", approved=True, approval_id="APR-after")
        assert ag.pending("T-after") is False


# ── Tests: resume() ───────────────────────────────────────────────────────────

class TestResume:
    async def test_approved_true_runs_tool_body(self):
        client = _make_client(approval_id="APR-ap")
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"status": "success", "symbol": symbol}

        graph = _build_graph(buy)
        ag = _make_ag(graph, client=client)

        await ag.run({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, thread_id="T-ap")
        result = await ag.resume("T-ap", approved=True, approval_id="APR-ap")

        assert ran == ["AAPL"]
        assert result["status"] == "complete"
        assert result["thread_id"] == "T-ap"

    async def test_approved_false_skips_tool_body(self):
        client = _make_client(approval_id="APR-rej")
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"status": "success"}

        graph = _build_graph(buy)
        ag = _make_ag(graph, client=client)

        await ag.run({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, thread_id="T-rej")
        result = await ag.resume("T-rej", approved=False, approval_id="APR-rej")

        assert ran == []
        assert result["status"] == "complete"
        contents = " ".join(str(m.get("content", "")) for m in result["messages"])
        assert "cancelled" in contents

    async def test_resume_response_shape(self):
        client = _make_client(approval_id="APR-shape")

        @tool
        @approval_required(action="x", args=[])
        def x() -> dict:
            """x."""
            return {"done": True}

        graph = _build_graph(x)
        ag = _make_ag(graph, client=client)

        await ag.run({"messages": [_tool_call("x", {})]}, thread_id="T-shape")
        result = await ag.resume("T-shape", approved=True, approval_id="APR-shape")

        assert "status" in result
        assert "thread_id" in result
        assert "messages" in result
        assert result["thread_id"] == "T-shape"


# ── Tests: full round-trip ────────────────────────────────────────────────────

class TestRoundTrip:
    async def test_run_then_approve(self):
        client = _make_client(approval_id="APR-RT-1")
        ran = []

        @tool
        @approval_required(action="pay", message="Approve payment", channel="slack", args=["amount"])
        def pay(amount: float) -> dict:
            """Pay."""
            ran.append(amount)
            return {"paid": amount, "status": "success"}

        graph = _build_graph(pay)
        ag = _make_ag(graph, client=client)

        step1 = await ag.run(
            {"messages": [_tool_call("pay", {"amount": 99.9})]},
            thread_id="T-rt",
        )
        assert step1["status"] == "pending_approval"
        assert step1["approval_id"] == "APR-RT-1"
        assert ag.pending("T-rt") is True

        step2 = await ag.resume("T-rt", approved=True, approval_id="APR-RT-1")
        assert step2["status"] == "complete"
        assert ran == [99.9]
        assert ag.pending("T-rt") is False

    async def test_run_then_reject(self):
        client = _make_client(approval_id="APR-RT-2")
        ran = []

        @tool
        @approval_required(action="delete", message="Confirm delete", channel="slack", args=["id"])
        def delete(id: str) -> dict:
            """Delete."""
            ran.append(id)
            return {"deleted": id}

        graph = _build_graph(delete)
        ag = _make_ag(graph, client=client)

        step1 = await ag.run(
            {"messages": [_tool_call("delete", {"id": "rec-42"})]},
            thread_id="T-rt-rej",
        )
        assert step1["status"] == "pending_approval"
        assert ag.pending("T-rt-rej") is True

        step2 = await ag.resume("T-rt-rej", approved=False, approval_id="APR-RT-2")
        assert step2["status"] == "complete"
        assert ran == []
        assert ag.pending("T-rt-rej") is False

    async def test_multiple_threads_are_independent(self):
        # Single client — generates a distinct approval_id per thread_id so
        # gate.py can track each thread independently.
        client = AsyncMock()

        async def _per_thread(**kwargs):
            return {"status": "pending", "approval_id": f"APR-{kwargs['thread_id']}"}

        client.acreate_approval.side_effect = _per_thread
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"status": "success", "symbol": symbol}

        graph = _build_graph(buy)
        ag = _make_ag(graph, client=client)

        r_a = await ag.run({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, thread_id="T-A")
        r_b = await ag.run({"messages": [_tool_call("buy", {"symbol": "TSLA"})]}, thread_id="T-B")

        assert r_a["status"] == "pending_approval"
        assert r_b["status"] == "pending_approval"
        assert r_a["approval_id"] == "APR-T-A"
        assert r_b["approval_id"] == "APR-T-B"
        assert ag.pending("T-A") is True
        assert ag.pending("T-B") is True

        await ag.resume("T-A", approved=True, approval_id="APR-T-A")
        await ag.resume("T-B", approved=False, approval_id="APR-T-B")

        assert "AAPL" in ran
        assert "TSLA" not in ran
        assert ag.pending("T-A") is False
        assert ag.pending("T-B") is False
