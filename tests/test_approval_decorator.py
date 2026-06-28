from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from intrupt_py_sdk.adapters.langgraph import approval_required
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.utils import _filter_kwargs


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_graph(tool_fn):
    g = StateGraph(_State)
    g.add_node("tools", ToolNode([tool_fn]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile(checkpointer=MemorySaver())


def _tool_call(name, args, tool_call_id="tc1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tool_call_id, "type": "tool_call"}],
    )


class TestInterruptPayload:
    def test_payload_carries_sentinel_and_kwargs(self):
        ran = []

        @tool
        @approval_required(action="do_thing", message="ok?", channel="slack",
                           args=["symbol", "quantity"])
        def do_thing(symbol: str, quantity: int) -> dict:
            """Do a thing."""
            ran.append((symbol, quantity))
            return {"ok": True, "symbol": symbol, "quantity": quantity}

        graph = _build_graph(do_thing)
        cfg = {"configurable": {"thread_id": "t1"}}

        graph.invoke({"messages": [_tool_call("do_thing", {"symbol": "AAPL", "quantity": 5})]}, config=cfg)

        # Tool body must NOT have run yet — we're paused on the interrupt.
        assert ran == []

        state = graph.get_state(cfg)
        interrupts = [i for task in state.tasks for i in (task.interrupts or ())]
        assert len(interrupts) == 1
        payload = interrupts[0].value
        assert payload["approval_required"] is True
        assert payload["action"] == "do_thing"
        assert payload["message"] == "ok?"
        assert payload["channel"] == "slack"
        assert payload["tool"]["name"] == "do_thing"
        assert payload["tool"]["kwargs"] == {"symbol": "AAPL", "quantity": 5}

    def test_args_filter_excludes_non_listed_kwargs(self):
        @tool
        @approval_required(action="x", args=["symbol"])
        def x(symbol: str, secret: str = "shh") -> dict:
            """Filter kwargs test."""
            return {"symbol": symbol}

        graph = _build_graph(x)
        cfg = {"configurable": {"thread_id": "t-filter"}}
        graph.invoke({"messages": [_tool_call("x", {"symbol": "AAPL", "secret": "hush"})]}, config=cfg)
        payload = next(
            i.value for task in graph.get_state(cfg).tasks for i in (task.interrupts or ())
        )
        assert payload["tool"]["kwargs"] == {"symbol": "AAPL"}
        assert "secret" not in payload["tool"]["kwargs"]

    def test_action_defaults_to_function_name(self):
        @tool
        @approval_required()
        def my_fn() -> dict:
            """No-op."""
            return {}

        graph = _build_graph(my_fn)
        cfg = {"configurable": {"thread_id": "t-def"}}
        graph.invoke({"messages": [_tool_call("my_fn", {})]}, config=cfg)
        payload = next(
            i.value for task in graph.get_state(cfg).tasks for i in (task.interrupts or ())
        )
        assert payload["action"] == "my_fn"
        assert payload["message"] == "Approval required for my_fn"


class TestResume:
    def test_approved_runs_tool_body(self):
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"status": "success", "symbol": symbol}

        graph = _build_graph(buy)
        cfg = {"configurable": {"thread_id": "t-ap"}}
        graph.invoke({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, config=cfg)

        result = graph.invoke(Command(resume={"approved": True}), config=cfg)
        assert ran == ["AAPL"]
        tool_msg = [m for m in result["messages"] if m.__class__.__name__ == "ToolMessage"][-1]
        assert "success" in tool_msg.content

    def test_rejected_skips_tool_body(self):
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)  # must not happen
            return {"status": "success"}

        graph = _build_graph(buy)
        cfg = {"configurable": {"thread_id": "t-rej"}}
        graph.invoke({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, config=cfg)

        result = graph.invoke(Command(resume={"approved": False}), config=cfg)
        assert ran == []
        tool_msg = [m for m in result["messages"] if m.__class__.__name__ == "ToolMessage"][-1]
        assert "cancelled" in tool_msg.content
        assert "not approved" in tool_msg.content

    def test_missing_approved_key_treated_as_rejection(self):
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"ok": True}

        graph = _build_graph(buy)
        cfg = {"configurable": {"thread_id": "t-missing"}}
        graph.invoke({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, config=cfg)
        graph.invoke(Command(resume={}), config=cfg)
        assert ran == []

    def test_non_dict_resume_value_treated_as_rejection(self):
        """If the resume payload is the wrong shape (e.g. a bare string),
        the decorator must not raise and must not run the tool."""
        ran = []

        @tool
        @approval_required(action="buy", args=["symbol"])
        def buy(symbol: str) -> dict:
            """Buy."""
            ran.append(symbol)
            return {"ok": True}

        graph = _build_graph(buy)
        cfg = {"configurable": {"thread_id": "t-bad"}}
        graph.invoke({"messages": [_tool_call("buy", {"symbol": "AAPL"})]}, config=cfg)
        graph.invoke(Command(resume="approved"), config=cfg)  # wrong shape
        assert ran == []


class TestFilterKwargs:
    def test_no_allowlist_drops_config(self):
        # When no `args` is configured, we still strip `config` because that's
        # framework plumbing (RunnableConfig), not human-readable data.
        assert _filter_kwargs({"a": 1, "config": {"x": 1}}, None) == {"a": 1}

    def test_with_allowlist_only_keeps_allowed(self):
        assert _filter_kwargs({"a": 1, "b": 2, "c": 3}, ["a", "c"]) == {"a": 1, "c": 3}

    def test_empty_allowlist_shows_nothing(self):
        assert _filter_kwargs({"a": 1, "config": {}}, []) == {}


class TestMiddlewareSingleton:
    def test_second_construction_does_not_replace_client(self):
        # Reset for isolation — other tests may have constructed the singleton.
        ApprovalMiddleware._instance = None

        m1 = ApprovalMiddleware(base_url="http://first", api_key="sk_org_org_test1_abcdef0123456789")
        c1 = m1.client
        m2 = ApprovalMiddleware(base_url="http://second", api_key="sk_org_org_test2_abcdef0123456789")

        assert m1 is m2
        assert m2.client is c1
        # Original config is preserved — second construction must not silently
        # repoint the shared client (the prior bug).
        assert c1.base_url == "http://first"
        assert c1.api_key == "sk_org_org_test1_abcdef0123456789"

    def test_get_client_before_init_raises(self):
        ApprovalMiddleware._instance = None
        with pytest.raises(RuntimeError):
            ApprovalMiddleware.get_client()
