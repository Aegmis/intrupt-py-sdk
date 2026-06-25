"""End-to-end tests for the agent's /call-tool and /resume endpoints.

We replace the LLM with a deterministic stub that always emits a single
`purchase_stock` tool call. This lets us drive the graph without an OpenAI
key and pins the test to the integration we actually care about: the
approval interrupt + resume round-trip.
"""

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


class _FakeLLM:
    """Bound-to-tools LLM stub that always replies with a single tool call."""

    def invoke(self, messages, **kwargs):
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "purchase_stock",
                "args": {"symbol": "AAPL", "quantity": 5},
                "id": "tc1",
                "type": "tool_call",
            }],
        )

    def bind_tools(self, tools):
        return self


@pytest.fixture
def agent_client(monkeypatch):
    """Reimport agent.py with a stubbed LLM and a stubbed SDK HTTP layer."""
    # Drop any cached version so module-level code (graph compile, route
    # registration) runs against our stubs.
    for mod in ("agent",):
        sys.modules.pop(mod, None)

    # Stub the SDK's outbound HTTP so /call-tool doesn't hit a live API.
    sdk_posts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        sdk_posts.append({"url": url, "json": json})
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"approval_id": "A-stub", "status": "pending"})
        return resp

    import intrupt_py_sdk.core.client as sdk_client_mod
    monkeypatch.setattr(sdk_client_mod.httpx, "post", fake_post)

    # Stub ChatOpenAI before importing agent so module-level llm = ChatOpenAI() uses our fake.
    import langchain_openai
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda *a, **kw: _FakeLLM())

    # Reset the ApprovalMiddleware singleton so it picks up env from this test.
    from intrupt_py_sdk.adapters import approval_middleware as adapter_mod
    adapter_mod.ApprovalMiddleware._instance = None

    import agent
    importlib.reload(agent)

    return TestClient(agent.app), sdk_posts


class TestCallTool:
    def test_pauses_on_approval_and_creates_request(self, agent_client):
        client, sdk_posts = agent_client
        r = client.post("/call-tool", json={"message": "buy 5 AAPL"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "pending_approval"
        assert body["approval_id"] == "A-stub"
        assert isinstance(body["thread_id"], str) and body["thread_id"]

        # The SDK was actually called and the body carries the right
        # correlation: thread_id matches, agent_callback_url points back at
        # the agent.
        assert len(sdk_posts) == 1
        sent = sdk_posts[0]["json"]
        assert sent["thread_id"] == body["thread_id"]
        assert sent["action"] == "purchase_stock"
        assert sent["channel"] == "slack"
        assert sent["tool_kwargs"] == {"symbol": "AAPL", "quantity": 5}
        assert sent["agent_callback_url"].endswith("/resume")

    def test_explicit_thread_id_is_used(self, agent_client):
        client, sdk_posts = agent_client
        r = client.post("/call-tool", json={"message": "buy", "thread_id": "T-explicit"})
        assert r.json()["thread_id"] == "T-explicit"
        assert sdk_posts[-1]["json"]["thread_id"] == "T-explicit"

    def test_missing_message_400(self, agent_client):
        client, _ = agent_client
        r = client.post("/call-tool", json={})
        assert r.status_code == 400


class TestResume:
    def test_resume_with_approved_runs_tool_body(self, agent_client):
        client, _ = agent_client
        started = client.post("/call-tool", json={"message": "buy"}).json()
        tid = started["thread_id"]

        r = client.post(
            "/resume",
            json={"thread_id": tid, "approval_id": started["approval_id"], "approved": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "complete"
        assert body["thread_id"] == tid

        # The tool body produced a "success" payload — find the ToolMessage.
        contents = " ".join(str(m.get("content", "")) for m in body["messages"])
        assert "success" in contents
        assert "AAPL" in contents

    def test_resume_with_rejected_does_not_run_tool_body(self, agent_client):
        client, _ = agent_client
        started = client.post("/call-tool", json={"message": "buy"}).json()
        tid = started["thread_id"]

        r = client.post(
            "/resume",
            json={"thread_id": tid, "approval_id": started["approval_id"], "approved": False},
        )
        assert r.status_code == 200
        contents = " ".join(str(m.get("content", "")) for m in r.json()["messages"])
        assert "cancelled" in contents
        assert "not approved" in contents
        # The success payload from the tool body must NOT appear.
        assert "Purchase order placed" not in contents

    def test_missing_thread_id_400(self, agent_client):
        client, _ = agent_client
        r = client.post("/resume", json={"approved": True})
        assert r.status_code == 400

    def test_missing_approved_400(self, agent_client):
        client, _ = agent_client
        r = client.post("/resume", json={"thread_id": "T1"})
        assert r.status_code == 400
