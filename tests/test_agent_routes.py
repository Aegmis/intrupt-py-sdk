"""End-to-end tests for the agent's /call-tool and /resume endpoints.

We replace the LLM with a deterministic stub that always emits a single
``purchase_stock`` tool call. The approval API is mocked via
``ApprovalClient.acreate_approval`` (async), matching the gate.py path.
"""

import importlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


class _FakeLLM:
    """Bound-to-tools LLM stub.

    Emits a ``purchase_stock`` tool call on the first turn; returns a plain
    text message on subsequent turns so the graph doesn't loop after the tool
    runs.
    """

    def invoke(self, messages, **kwargs):
        from langchain_core.messages import ToolMessage
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="Done.")
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "purchase_stock",
                "args": {"symbol": "AAPL", "quantity": 5, "amount": 100.0},
                "id": "tc1",
                "type": "tool_call",
            }],
        )

    def bind_tools(self, tools):
        return self


@pytest.fixture
def agent_client(monkeypatch):
    """Reimport agent.py with a stubbed LLM and a mocked approval API.

    Uses TestClient as a context manager so all requests in a test share a
    single anyio portal (event loop). This is required for the background
    asyncio Task spawned by ApprovalGraph.run() to survive across the
    /call-tool → /resume request boundary.
    """
    for mod in ("agent",):
        sys.modules.pop(mod, None)

    # Prevent load_dotenv() in agent.py from loading AGENT_RESUME_SECRET from
    # the repo-root .env — tests don't send the secret header.
    monkeypatch.setenv("AGENT_RESUME_SECRET", "")

    # Stub ChatOpenAI before importing agent so module-level llm uses our fake.
    import langchain_openai
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda *a, **kw: _FakeLLM())

    # Reset the ApprovalMiddleware singleton so it picks up env from this test.
    from intrupt_py_sdk.adapters import approval_middleware as adapter_mod
    adapter_mod.ApprovalMiddleware._instance = None

    import agent
    importlib.reload(agent)

    # Patch the live ApprovalClient instance directly (avoids self-binding
    # issues that arise when patching async methods at the class level).
    approval_calls = []

    async def fake_acreate_approval(**kwargs):
        approval_calls.append(kwargs)
        return {"approval_id": "A-stub", "status": "pending"}

    from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
    ApprovalMiddleware.get_client().acreate_approval = fake_acreate_approval

    # Use a very short gate timeout so tests don't wait 1.5 s.
    agent.approval_graph._timeout = 0.05

    # Context-manager form keeps one event loop alive across all requests.
    with TestClient(agent.app) as client:
        yield client, approval_calls


class TestCallTool:
    def test_pauses_on_approval_and_creates_request(self, agent_client):
        client, approval_calls = agent_client
        r = client.post("/call-tool", json={"message": "buy 5 AAPL"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "pending_approval"
        assert body["approval_id"] == "A-stub"
        assert isinstance(body["thread_id"], str) and body["thread_id"]

        assert len(approval_calls) == 1
        sent = approval_calls[0]
        assert sent["thread_id"] == body["thread_id"]
        assert sent["action"] == "purchase_stock"
        assert sent["channel"] == "slack"
        assert sent["tool"]["kwargs"] == {"symbol": "AAPL", "quantity": 5, "amount": 100.0}
        assert str(sent.get("agent_callback_url", "")).endswith("/resume")

    def test_explicit_thread_id_is_used(self, agent_client):
        client, approval_calls = agent_client
        r = client.post("/call-tool", json={"message": "buy", "thread_id": "T-explicit"})
        assert r.json()["thread_id"] == "T-explicit"
        assert approval_calls[-1]["thread_id"] == "T-explicit"

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
        assert "Purchase order placed" not in contents

    def test_missing_thread_id_400(self, agent_client):
        client, _ = agent_client
        r = client.post("/resume", json={"approved": True})
        assert r.status_code == 400

    def test_missing_approved_400(self, agent_client):
        client, _ = agent_client
        r = client.post("/resume", json={"thread_id": "T1"})
        assert r.status_code == 400
