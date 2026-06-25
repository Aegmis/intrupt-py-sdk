"""Tests for guards added to the example agent (example/agent.py) in a prior session.

Coverage:
- /call-tool 409 when a thread already has a pending approval
- /resume 409 when the thread is not paused (missing or dead checkpoint)
- AGENT_RESUME_SECRET authentication (missing/wrong → 401)
- chat_node leading-tool-message trim (prevents OpenAI "messages[0].role == tool" error)
"""

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage


# ─── Fake LLM ────────────────────────────────────────────────────────────────

class _FakeLLM:
    """Always emits one purchase_stock tool call."""
    def invoke(self, messages, **kwargs):
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "purchase_stock",
                "args": {"symbol": "AAPL", "quantity": 5, "amount": 100.0},
                "id": "tc-test",
                "type": "tool_call",
            }],
        )

    def bind_tools(self, tools):
        return self


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_agent_client(monkeypatch, *, resume_secret: str = ""):
    """Load agent.py with stubs; optionally set AGENT_RESUME_SECRET."""
    for mod in ("agent",):
        sys.modules.pop(mod, None)

    sdk_posts: list = []

    def fake_post(url, headers=None, json=None, timeout=None):
        sdk_posts.append({"url": url, "json": json, "headers": headers or {}})
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"approval_id": "A-stub", "status": "pending"})
        return resp

    import intrupt_py_sdk.core.client as sdk_client_mod
    monkeypatch.setattr(sdk_client_mod.httpx, "post", fake_post)

    import langchain_openai
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda *a, **kw: _FakeLLM())

    from intrupt_py_sdk.adapters import approval_middleware as adapter_mod
    adapter_mod.ApprovalMiddleware._instance = None

    if resume_secret:
        monkeypatch.setenv("AGENT_RESUME_SECRET", resume_secret)
    else:
        monkeypatch.delenv("AGENT_RESUME_SECRET", raising=False)

    import agent
    importlib.reload(agent)

    return TestClient(agent.app), sdk_posts


@pytest.fixture
def agent_client(monkeypatch):
    client, sdk_posts = _make_agent_client(monkeypatch)
    return client, sdk_posts


@pytest.fixture
def agent_client_with_secret(monkeypatch):
    client, sdk_posts = _make_agent_client(monkeypatch, resume_secret="test-resume-secret")
    return client, sdk_posts


# ─── /call-tool 409 guard ─────────────────────────────────────────────────────

class TestCallToolPendingGuard:
    def test_second_message_on_interrupted_thread_is_409(self, agent_client):
        client, _ = agent_client

        # First call: creates an approval interrupt.
        r1 = client.post("/call-tool", json={"message": "buy AAPL", "thread_id": "T-dup"})
        assert r1.status_code == 200
        assert r1.json()["status"] == "pending_approval"

        # Second call on the same thread: must be rejected while approval is pending.
        r2 = client.post("/call-tool", json={"message": "sell AAPL", "thread_id": "T-dup"})
        assert r2.status_code == 409

    def test_no_thread_id_always_allowed(self, agent_client):
        """Without an explicit thread_id, each call gets a fresh uuid thread —
        there can never be a duplicate-pending collision."""
        client, _ = agent_client

        r1 = client.post("/call-tool", json={"message": "buy AAPL"})
        assert r1.status_code == 200
        assert r1.json()["status"] == "pending_approval"

        # Second call with no thread_id → fresh thread, no conflict.
        r2 = client.post("/call-tool", json={"message": "buy AAPL"})
        assert r2.status_code == 200

    def test_thread_with_completed_approval_accepts_new_message(self, agent_client):
        """After resume, the interrupt is cleared — a new message must succeed."""
        client, _ = agent_client

        started = client.post("/call-tool", json={"message": "buy", "thread_id": "T-ok"}).json()
        assert started["status"] == "pending_approval"

        client.post("/resume", json={
            "thread_id": "T-ok",
            "approval_id": started["approval_id"],
            "approved": True,
        })

        r = client.post("/call-tool", json={"message": "buy again", "thread_id": "T-ok"})
        # After resume the thread is no longer interrupted — no 409.
        assert r.status_code != 409


# ─── /resume 409 guard ───────────────────────────────────────────────────────

class TestResumePendingGuard:
    def test_resume_on_unknown_thread_is_409(self, agent_client):
        """Calling /resume for a thread that was never started (or whose checkpoint
        was lost on server restart) must return 409, not crash."""
        client, _ = agent_client
        r = client.post("/resume", json={"thread_id": "never-existed", "approved": True})
        assert r.status_code == 409

    def test_resume_already_decided_thread_is_409(self, agent_client):
        """After a thread has been resumed once, its interrupt is gone. A second
        /resume attempt must return 409."""
        client, _ = agent_client

        started = client.post("/call-tool", json={"message": "buy"}).json()
        tid = started["thread_id"]

        client.post("/resume", json={
            "thread_id": tid,
            "approval_id": started["approval_id"],
            "approved": True,
        })

        r2 = client.post("/resume", json={"thread_id": tid, "approved": True})
        assert r2.status_code == 409


# ─── AGENT_RESUME_SECRET auth ────────────────────────────────────────────────

class TestAgentResumeSecretAuth:
    def test_missing_secret_header_returns_401(self, agent_client_with_secret):
        client, _ = agent_client_with_secret

        started = client.post("/call-tool", json={"message": "buy"}).json()
        tid = started["thread_id"]

        # No X-Agent-Secret header at all.
        r = client.post("/resume", json={"thread_id": tid, "approved": True})
        assert r.status_code == 401

    def test_wrong_secret_returns_401(self, agent_client_with_secret):
        client, _ = agent_client_with_secret

        started = client.post("/call-tool", json={"message": "buy"}).json()
        tid = started["thread_id"]

        r = client.post(
            "/resume",
            json={"thread_id": tid, "approved": True},
            headers={"X-Agent-Secret": "wrong-secret"},
        )
        assert r.status_code == 401

    def test_correct_secret_proceeds_past_auth(self, agent_client_with_secret):
        client, _ = agent_client_with_secret

        started = client.post("/call-tool", json={"message": "buy"}).json()
        tid = started["thread_id"]

        r = client.post(
            "/resume",
            json={
                "thread_id": tid,
                "approval_id": started["approval_id"],
                "approved": True,
            },
            headers={"X-Agent-Secret": "test-resume-secret"},
        )
        # Auth passed — status is not 401 (may be 200 or 409 from next guard).
        assert r.status_code != 401

    def test_no_secret_configured_any_request_allowed(self, agent_client):
        """When AGENT_RESUME_SECRET is empty, the auth check is skipped entirely."""
        client, _ = agent_client

        started = client.post("/call-tool", json={"message": "buy"}).json()
        tid = started["thread_id"]

        # No header needed.
        r = client.post("/resume", json={
            "thread_id": tid,
            "approval_id": started["approval_id"],
            "approved": True,
        })
        assert r.status_code != 401


# ─── agent_callback_secret forwarded via SDK ─────────────────────────────────

class TestAgentCallbackSecretInSdkPost:
    def test_resume_secret_sent_to_approval_api(self, agent_client_with_secret):
        """/call-tool must pass agent_callback_secret to create_approval so the
        approval platform can echo it back in the X-Agent-Secret header when
        calling /resume."""
        client, sdk_posts = agent_client_with_secret

        r = client.post("/call-tool", json={"message": "buy AAPL"})
        assert r.status_code == 200

        assert len(sdk_posts) >= 1
        body = sdk_posts[-1]["json"]
        # The secret must be present — the platform stores it opaquely and
        # echoes it when triggering the callback, which this agent validates.
        assert body.get("agent_callback_secret") == "test-resume-secret"


# ─── chat_node leading-tool-message trim ─────────────────────────────────────

class TestChatNodeTrim:
    """Verify that chat_node drops leading ToolMessages before invoking the LLM.

    The guard prevents openai.BadRequestError when a checkpoint is loaded that
    starts with a ToolMessage (e.g. after a server restart mid-approval).
    """

    def test_leading_tool_message_is_dropped(self, monkeypatch):
        for mod in ("agent",):
            sys.modules.pop(mod, None)

        invoked_with: list = []

        class _TrackingLLM:
            def invoke(self, messages, **kwargs):
                invoked_with.append(list(messages))
                return AIMessage(content="ok")

            def bind_tools(self, tools):
                return self

        import langchain_openai
        monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda *a, **kw: _TrackingLLM())

        from intrupt_py_sdk.adapters import approval_middleware as adapter_mod
        adapter_mod.ApprovalMiddleware._instance = None

        import intrupt_py_sdk.core.client as sdk_client_mod
        monkeypatch.setattr(
            sdk_client_mod.httpx, "post",
            lambda *a, **kw: MagicMock(
                status_code=200,
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"approval_id": "X", "status": "pending"}),
            )
        )

        import agent
        importlib.reload(agent)

        # Build a state where the first message is a ToolMessage — the pathological
        # scenario that triggers the OpenAI error after a server restart.
        tool_msg = ToolMessage(content="prior result", tool_call_id="tc-old")
        human_msg = AIMessage(content="buy AAPL")
        state = {"messages": [tool_msg, human_msg], "last_purchase": None}

        agent.chat_node(state)

        assert len(invoked_with) == 1
        msgs_seen = invoked_with[0]
        # The ToolMessage must have been stripped.
        from langchain_core.messages import ToolMessage as TM
        assert not any(isinstance(m, TM) for m in msgs_seen)

    def test_all_tool_messages_returns_empty(self, monkeypatch):
        """A state that is *only* ToolMessages must not crash — chat_node returns
        an empty update (no LLM call) when nothing remains after the trim."""
        for mod in ("agent",):
            sys.modules.pop(mod, None)

        invoked_with: list = []

        class _TrackingLLM:
            def invoke(self, messages, **kwargs):
                invoked_with.append(list(messages))
                return AIMessage(content="ok")

            def bind_tools(self, tools):
                return self

        import langchain_openai
        monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda *a, **kw: _TrackingLLM())

        from intrupt_py_sdk.adapters import approval_middleware as adapter_mod
        adapter_mod.ApprovalMiddleware._instance = None

        import intrupt_py_sdk.core.client as sdk_client_mod
        monkeypatch.setattr(
            sdk_client_mod.httpx, "post",
            lambda *a, **kw: MagicMock(
                status_code=200,
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"approval_id": "X", "status": "pending"}),
            )
        )

        import agent
        importlib.reload(agent)

        tool_msg1 = ToolMessage(content="r1", tool_call_id="tc1")
        tool_msg2 = ToolMessage(content="r2", tool_call_id="tc2")
        state = {"messages": [tool_msg1, tool_msg2], "last_purchase": None}

        result = agent.chat_node(state)

        # No LLM call made — trim consumed all messages.
        assert invoked_with == []
        # Returned update must not raise and must be an empty dict (or have no new messages).
        assert isinstance(result, dict)
        assert not result.get("messages")
