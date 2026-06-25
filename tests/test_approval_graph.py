"""Tests for ApprovalGraph — the wrapper that replaces _build_response boilerplate.

Covers:
  - invoke(): interrupt detected → create_approval called, pending_approval returned
  - invoke(): no interrupt → complete returned with messages
  - resume(): approved=True → tool body runs, complete returned
  - resume(): approved=False → cancellation returned, tool body skipped
  - pending(): True when thread is paused, False otherwise
  - callback_url and callback_secret forwarded to create_approval
  - Full round-trip: invoke → resume approved
  - Full round-trip: invoke → resume rejected
"""

from typing import Annotated, TypedDict
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import ApprovalGraph, approval_required


# ── Helpers ──────────────────────────────────────────────────────────────────

class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _tool_call(name: str, args: dict, tc_id: str = "tc1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tc_id, "type": "tool_call"}],
    )


def _build_graph(tool_fn) -> object:
    g = StateGraph(_State)
    g.add_node("tools", ToolNode([tool_fn]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile(checkpointer=MemorySaver())


def _mock_client(approval_id: str = "APR-001") -> MagicMock:
    client = MagicMock()
    client.create_approval.return_value = {"approval_id": approval_id, "status": "pending"}
    return client


def _make_approval_graph(graph, client=None, callback_url="http://agent/resume", callback_secret="secret123"):
    return ApprovalGraph(
        graph=graph,
        client=client or _mock_client(),
        callback_url=callback_url,
        callback_secret=callback_secret,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Each test gets a fresh ApprovalMiddleware singleton."""
    ApprovalMiddleware._instance = None
    yield
    ApprovalMiddleware._instance = None


# ── Tests: invoke() ──────────────────────────────────────────────────────────

class TestInvoke:
    def test_interrupt_returns_pending_approval(self):
        @tool
        @approval_required(action="buy", message="Approve?", channel="slack", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            return {"status": "success", "symbol": symbol}

        graph = _build_graph(buy)
        client = _mock_client("APR-XYZ")
        ag = _make_approval_graph(graph, client=client)

        result = ag.invoke(
            {"messages": [_tool_call("buy", {"symbol": "AAPL"})]},
            thread_id="T1",
        )

        assert result["status"] == "pending_approval"
        assert result["thread_id"] == "T1"
        assert result["approval_id"] == "APR-XYZ"

    def test_no_interrupt_returns_complete(self):
        @tool
        def safe_tool(x: int) -> dict:
            """No approval needed."""
            return {"result": x * 2}

        graph = _build_graph(safe_tool)
        ag = _make_approval_graph(graph)

        result = ag.invoke(
            {"messages": [_tool_call("safe_tool", {"x": 5})]},
            thread_id="T-safe",
        )

        assert result["status"] == "complete"
        assert result["thread_id"] == "T-safe"
        assert isinstance(result["messages"], list)

    def test_create_approval_called_with_correct_fields(self):
        @tool
        @approval_required(action="transfer", message="Approve transfer", channel="slack", args=["amount"])
        def transfer(amount: float) -> dict:
            """Transfer."""
            return {"transferred": amount}

        graph = _build_graph(transfer)
        client = _mock_client()
        ag = _make_approval_graph(graph, client=client, callback_url="http://myagent/resume", callback_secret="s3cr3t")

        ag.invoke({"messages": [_tool_call("transfer", {"amount": 500.0})]}, thread_id="T-fields")

        client.create_approval.assert_called_once()
        kwargs = client.create_approval.call_args.kwargs
        assert kwargs["thread_id"] == "T-fields"
        assert kwargs["action"] == "transfer"
        assert kwargs["message"] == "Approve transfer"
        assert kwargs["channel"] == "slack"
        assert kwargs["agent_callback_url"] == "http://myagent/resume"
        assert kwargs["agent_callback_secret"] == "s3cr3t"
        assert kwargs["tool"]["kwargs"] == {"amount": 500.0}

    def test_create_approval_not_called_when_no_interrupt(self):
        @tool
        def noop() -> dict:
            """No interrupt."""
            return {"ok": True}

        graph = _build_graph(noop)
        client = _mock_client()
        ag = _make_approval_graph(graph, client=client)

        ag.invoke({"messages": [_tool_call("noop", {})]}, thread_id="T-noop")

        client.create_approval.assert_not_called()

    def test_complete_response_contains_messages(self):
        @tool
        def echo(msg: str) -> dict:
            """Echo."""
            return {"echo": msg}

        graph = _build_graph(echo)
        ag = _make_approval_graph(graph)

        result = ag.invoke({"messages": [_tool_call("echo", {"msg": "hello"})]}, thread_id="T-echo")

        assert result["status"] == "complete"
        assert len(result["messages"]) > 0
        assert all("type" in m and "content" in m for m in result["messages"])


# ── Tests: pending() ─────────────────────────────────────────────────────────

class TestPending:
    def test_true_when_thread_is_paused(self):
        @tool
        @approval_required(action="act", args=[])
        def act() -> dict:
            """Act."""
            return {}

        graph = _build_graph(act)
        ag = _make_approval_graph(graph)

        ag.invoke({"messages": [_tool_call("act", {})]}, thread_id="T-pend")

        assert ag.pending("T-pend") is True

    def test_false_when_thread_has_no_interrupt(self):
        @tool
        def safe() -> dict:
            """Safe."""
            return {"ok": True}

        graph = _build_graph(safe)
        ag = _make_approval_graph(graph)

        ag.invoke({"messages": [_tool_call("safe", {})]}, thread_id="T-no-pend")

        assert ag.pending("T-no-pend") is False

    def test_false_for_unknown_thread(self):
        @tool
        def any_tool() -> dict:
            """Tool."""
            return {}

        graph = _build_graph(any_tool)
        ag = _make_approval_graph(graph)

        assert ag.pending("thread-that-never-existed") is False

    def test_false_after_resume(self):
        ran = []

        @tool
        @approval_required(action="go", args=[])
        def go() -> dict:
            """Go."""
            ran.append(True)
            return {"went": True}

        graph = _build_graph(go)
        ag = _make_approval_graph(graph)

        ag.invoke({"messages": [_tool_call("go", {})]}, thread_id="T-after")
        assert ag.pending("T-after") is True

        ag.resume("T-after", approved=True)
        assert ag.pending("T-after") is False


# ── Tests: resume() ───────────────────────────────────────────────────────────

class TestResume:
    def test_approved_true_runs_tool_body(self):
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"status": "success", "symbol": symbol}

        graph = _build_graph(buy)
        ag = _make_approval_graph(graph)

        ag.invoke({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, thread_id="T-ap")
        result = ag.resume("T-ap", approved=True, approval_id="APR-1")

        assert ran == ["AAPL"]
        assert result["status"] == "complete"
        assert result["thread_id"] == "T-ap"

    def test_approved_false_skips_tool_body(self):
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"status": "success"}

        graph = _build_graph(buy)
        ag = _make_approval_graph(graph)

        ag.invoke({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, thread_id="T-rej")
        result = ag.resume("T-rej", approved=False)

        assert ran == []
        assert result["status"] == "complete"
        contents = " ".join(str(m.get("content", "")) for m in result["messages"])
        assert "cancelled" in contents

    def test_resume_complete_response_shape(self):
        @tool
        @approval_required(action="x", args=[])
        def x() -> dict:
            """x."""
            return {"done": True}

        graph = _build_graph(x)
        ag = _make_approval_graph(graph)

        ag.invoke({"messages": [_tool_call("x", {})]}, thread_id="T-shape")
        result = ag.resume("T-shape", approved=True)

        assert "status" in result
        assert "thread_id" in result
        assert "messages" in result
        assert result["thread_id"] == "T-shape"

    def test_approval_id_forwarded_to_graph(self):
        decisions = []

        @tool
        @approval_required(action="decide", args=[])
        def decide() -> dict:
            """Decide."""
            return {"ok": True}

        graph = _build_graph(decide)
        ag = _make_approval_graph(graph)

        ag.invoke({"messages": [_tool_call("decide", {})]}, thread_id="T-id")

        from langgraph.types import Command

        original_invoke = graph.invoke

        captured = []

        def capturing_invoke(input, **kwargs):
            if isinstance(input, Command):
                captured.append(input.resume)
            return original_invoke(input, **kwargs)

        graph.invoke = capturing_invoke
        ag.resume("T-id", approved=True, approval_id="APR-999")

        assert captured[0]["approval_id"] == "APR-999"
        assert captured[0]["approved"] is True


# ── Tests: full round-trip ────────────────────────────────────────────────────

class TestRoundTrip:
    def test_invoke_then_approve(self):
        ran = []

        @tool
        @approval_required(action="pay", message="Approve payment", channel="slack", args=["amount"])
        def pay(amount: float) -> dict:
            """Pay."""
            ran.append(amount)
            return {"paid": amount, "status": "success"}

        graph = _build_graph(pay)
        client = _mock_client("APR-RT-1")
        ag = _make_approval_graph(graph, client=client)

        # Step 1: invoke pauses
        step1 = ag.invoke({"messages": [_tool_call("pay", {"amount": 99.9})]}, thread_id="T-rt")
        assert step1["status"] == "pending_approval"
        assert step1["approval_id"] == "APR-RT-1"
        assert ag.pending("T-rt") is True

        # Step 2: resume approved
        step2 = ag.resume("T-rt", approved=True, approval_id="APR-RT-1")
        assert step2["status"] == "complete"
        assert ran == [99.9]
        assert ag.pending("T-rt") is False

    def test_invoke_then_reject(self):
        ran = []

        @tool
        @approval_required(action="delete", message="Confirm delete", channel="slack", args=["id"])
        def delete(id: str) -> dict:
            """Delete."""
            ran.append(id)
            return {"deleted": id}

        graph = _build_graph(delete)
        ag = _make_approval_graph(graph)

        step1 = ag.invoke({"messages": [_tool_call("delete", {"id": "rec-42"})]}, thread_id="T-rt-rej")
        assert step1["status"] == "pending_approval"
        assert ag.pending("T-rt-rej") is True

        step2 = ag.resume("T-rt-rej", approved=False)
        assert step2["status"] == "complete"
        assert ran == []
        assert ag.pending("T-rt-rej") is False

    def test_multiple_threads_are_independent(self):
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"status": "success", "symbol": symbol}

        graph = _build_graph(buy)
        ag = _make_approval_graph(graph)

        ag.invoke({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, thread_id="T-A")
        ag.invoke({"messages": [_tool_call("buy", {"symbol": "TSLA"})]}, thread_id="T-B")

        assert ag.pending("T-A") is True
        assert ag.pending("T-B") is True

        # Approve A, reject B
        ag.resume("T-A", approved=True)
        ag.resume("T-B", approved=False)

        assert "AAPL" in ran
        assert "TSLA" not in ran
        assert ag.pending("T-A") is False
        assert ag.pending("T-B") is False
