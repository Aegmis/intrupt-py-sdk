"""
Tests for the 5 on_approval_async example agents.

Each test class:
  1. Stubs the LLM so no OpenAI calls are made
  2. Stubs / skips external I/O (stdin, SMTP, Slack, Telegram)
  3. Tests the approval callable logic directly where it's pure
  4. Tests the full /call-tool → decision endpoint round-trip

Multi-request tests (call-tool → decide/webhook) use TestClient as a context
manager so all requests share one anyio portal (event loop), which allows the
background asyncio Task spawned by ApprovalGraph.run() to survive across the
request boundary.
"""

import asyncio
import hashlib
import hmac
import json
import smtplib
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


# ── Shared fake LLM ───────────────────────────────────────────────────────────

class _FakeLLM:
    """Emits a single pay_invoice tool call on the first turn; returns plain text
    after the tool runs so the graph terminates cleanly."""

    def __init__(self, *, amount: float = 250.0, vendor: str = "Acme Corp"):
        self._args = {
            "invoice_id": "INV-001",
            "vendor": vendor,
            "amount": amount,
            "currency": "USD",
        }

    def invoke(self, messages, **kwargs):
        from langchain_core.messages import ToolMessage
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="Done.")
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "pay_invoice",
                "args": self._args,
                "id": "tc-test",
                "type": "tool_call",
            }],
        )

    def bind_tools(self, tools):
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 1. Console Agent (console_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleAgent:
    """
    console_approval asks y/n via stdin (run_in_executor), stores the decision
    in _console_decisions, and /call-tool auto-resumes immediately.
    """

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        import console_agent as ca
        ca.llm = _FakeLLM(amount=500.0)
        ca._console_decisions.clear()
        ca.approval_graph._timeout = 0.05
        yield ca
        ca._console_decisions.clear()

    @pytest.fixture
    def approving_client(self, setup):
        ca = setup
        _AID = "APR-CONS-APPROVE"

        async def stub(thread_id, v):
            ca._console_decisions[_AID] = True
            return {"approval_id": _AID}

        ca.approval_graph._on_approval_async = stub
        # Console auto-resumes within one request — no context manager needed.
        return TestClient(ca.app)

    @pytest.fixture
    def rejecting_client(self, setup):
        ca = setup
        _AID = "APR-CONS-REJECT"

        async def stub(thread_id, v):
            ca._console_decisions[_AID] = False
            return {"approval_id": _AID}

        ca.approval_graph._on_approval_async = stub
        return TestClient(ca.app)

    def test_approved_call_returns_complete_with_tool_result(self, approving_client):
        r = approving_client.post("/call-tool", json={"message": "pay invoice INV-001"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "complete"
        contents = " ".join(str(m.get("content", "")) for m in body["messages"])
        assert "paid" in contents or "success" in contents

    def test_rejected_call_returns_complete_with_cancellation(self, rejecting_client):
        r = rejecting_client.post("/call-tool", json={"message": "pay invoice INV-001"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "complete"
        contents = " ".join(str(m.get("content", "")) for m in body["messages"])
        assert "cancelled" in contents

    def test_missing_message_is_400(self, approving_client):
        r = approving_client.post("/call-tool", json={})
        assert r.status_code == 400

    def test_console_approval_stores_decision(self, setup, monkeypatch):
        """Unit test: console_approval with mocked input stores the answer."""
        ca = setup
        monkeypatch.setattr("builtins.input", lambda *a: "y")

        result = asyncio.run(
            ca.console_approval("T-unit", {
                "action": "pay", "message": "approve?",
                "tool": {"name": "pay_invoice", "kwargs": {"amount": 100}},
            })
        )

        assert "approval_id" in result
        aid = result["approval_id"]
        assert ca._console_decisions[aid] is True

    def test_console_approval_stores_rejection(self, setup, monkeypatch):
        ca = setup
        monkeypatch.setattr("builtins.input", lambda *a: "n")

        result = asyncio.run(
            ca.console_approval("T-unit2", {
                "action": "pay", "message": "approve?",
                "tool": {"name": "pay_invoice", "kwargs": {}},
            })
        )

        aid = result["approval_id"]
        assert ca._console_decisions[aid] is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Policy Agent (policy_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyAgent:
    """
    policy_approval evaluates rules and either auto-decides (stored in
    _auto_decisions) or escalates (stored in _pending). /call-tool auto-resumes
    auto-decisions; /decide handles escalated cases.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        import policy_agent as pa
        pa._auto_decisions.clear()
        pa._pending.clear()
        pa.approval_graph._timeout = 0.05
        yield pa
        pa._auto_decisions.clear()
        pa._pending.clear()

    # ── Pure unit tests for _evaluate_policy ─────────────────────────────────

    def test_evaluate_policy_low_amount_auto_approves(self, setup):
        pa = setup
        v = {"action": "pay_invoice", "tool": {"kwargs": {"amount": 100.0, "vendor": "Acme"}}}
        assert pa._evaluate_policy(v) is True

    def test_evaluate_policy_high_amount_auto_rejects(self, setup):
        pa = setup
        v = {"action": "pay_invoice", "tool": {"kwargs": {"amount": 60000.0, "vendor": "Acme"}}}
        assert pa._evaluate_policy(v) is False

    def test_evaluate_policy_mid_range_escalates(self, setup):
        pa = setup
        v = {"action": "pay_invoice", "tool": {"kwargs": {"amount": 5000.0, "vendor": "Acme"}}}
        assert pa._evaluate_policy(v) is None

    def test_evaluate_policy_blocked_vendor_auto_rejects(self, setup):
        pa = setup
        blocked = next(iter(pa.BLOCKED_VENDORS))
        v = {"action": "pay_invoice", "tool": {"kwargs": {"amount": 100.0, "vendor": blocked}}}
        assert pa._evaluate_policy(v) is False

    def test_evaluate_policy_emergency_action_auto_approves(self, setup):
        pa = setup
        v = {"action": "emergency_override", "tool": {"kwargs": {"amount": 99999.0, "vendor": "Acme"}}}
        assert pa._evaluate_policy(v) is True

    # ── Integration tests via /call-tool ──────────────────────────────────────

    def test_call_tool_auto_approves_low_amount(self, setup):
        pa = setup
        pa.llm = _FakeLLM(amount=200.0)
        r = TestClient(pa.app).post("/call-tool", json={"message": "pay"})
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

    def test_call_tool_auto_rejects_high_amount(self, setup):
        pa = setup
        pa.llm = _FakeLLM(amount=55000.0)
        r = TestClient(pa.app).post("/call-tool", json={"message": "pay"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "complete"
        contents = " ".join(str(m.get("content", "")) for m in body["messages"])
        assert "cancelled" in contents

    def test_call_tool_escalates_mid_range_to_pending(self, setup):
        pa = setup
        pa.llm = _FakeLLM(amount=5000.0)
        r = TestClient(pa.app).post("/call-tool", json={"message": "pay"})
        assert r.status_code == 200
        assert r.json()["status"] == "pending_approval"

    def test_decide_approves_escalated_request(self, setup):
        pa = setup
        pa.llm = _FakeLLM(amount=5000.0)

        with TestClient(pa.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            assert step1["status"] == "pending_approval"
            aid = step1["approval_id"]
            tid = step1["thread_id"]

            pa._pending[aid] = tid  # policy_approval already stored this

            step2 = client.post("/decide", json={"approval_id": aid, "approved": True}).json()
            assert step2["status"] == "complete"

    def test_decide_rejects_escalated_request(self, setup):
        pa = setup
        pa.llm = _FakeLLM(amount=5000.0)

        with TestClient(pa.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            aid = step1["approval_id"]
            tid = step1["thread_id"]
            pa._pending[aid] = tid

            step2 = client.post("/decide", json={"approval_id": aid, "approved": False}).json()
            assert step2["status"] == "complete"
            contents = " ".join(str(m.get("content", "")) for m in step2["messages"])
            assert "cancelled" in contents

    def test_decide_unknown_approval_id_404(self, setup):
        pa = setup
        r = TestClient(pa.app).post("/decide", json={"approval_id": "UNKNOWN", "approved": True})
        assert r.status_code == 404

    def test_missing_message_400(self, setup):
        pa = setup
        r = TestClient(pa.app).post("/call-tool", json={})
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 3. SMTP Email Agent (smtp_email_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestSmtpEmailAgent:
    """
    smtp_email_approval sends HTML email via SMTP and stores in _pending.
    /decide?approval_id=...&approved=true/false resumes the graph.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        import smtp_email_agent as sea
        sea.llm = _FakeLLM(amount=300.0)
        sea._pending.clear()
        sea.approval_graph._timeout = 0.05
        yield sea
        sea._pending.clear()

    @pytest.fixture
    def mock_smtp(self, monkeypatch):
        """Stub smtplib.SMTP_SSL so no real SMTP connection is made."""
        import smtp_email_agent as sea
        smtp_inst = MagicMock()
        smtp_inst.__enter__ = lambda s: s
        smtp_inst.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *a, **kw: smtp_inst)
        monkeypatch.setattr(sea, "SMTP_HOST", "smtp.test")
        monkeypatch.setattr(sea, "SMTP_USER", "user@test.com")
        monkeypatch.setattr(sea, "SMTP_PASS", "secret")
        monkeypatch.setattr(sea, "APPROVAL_EMAIL_TO", "approver@test.com")
        return smtp_inst

    def test_call_tool_returns_pending_approval(self, setup):
        # SMTP_HOST is empty so smtp_email_approval skips sending — still stores pending
        sea = setup
        r = TestClient(sea.app).post("/call-tool", json={"message": "pay"})
        assert r.status_code == 200
        assert r.json()["status"] == "pending_approval"
        assert "approval_id" in r.json()

    def test_smtp_email_sent_when_credentials_set(self, setup, mock_smtp):
        sea = setup
        r = TestClient(sea.app).post("/call-tool", json={"message": "pay"})
        assert r.status_code == 200
        assert r.json()["status"] == "pending_approval"
        mock_smtp.login.assert_called_once()
        mock_smtp.sendmail.assert_called_once()

    def test_decide_approve_returns_complete(self, setup):
        sea = setup

        with TestClient(sea.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            aid = step1["approval_id"]
            tid = step1["thread_id"]
            sea._pending[aid] = tid  # smtp_email_approval already stored this

            r = client.get(f"/decide?approval_id={aid}&approved=true")
            assert r.status_code == 200
            assert "approved" in r.text.lower() or "notified" in r.text.lower()

    def test_decide_reject_returns_200_html(self, setup):
        sea = setup

        with TestClient(sea.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            aid = step1["approval_id"]
            tid = step1["thread_id"]
            sea._pending[aid] = tid

            r = client.get(f"/decide?approval_id={aid}&approved=false")
            assert r.status_code == 200
            assert "rejected" in r.text.lower()

    def test_decide_unknown_approval_id_404(self, setup):
        sea = setup
        r = TestClient(sea.app).get("/decide?approval_id=UNKNOWN&approved=true")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 4. Slack Direct Agent (slack_direct_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

def _slack_signature(body: bytes, secret: str) -> tuple[str, str]:
    """Compute a valid Slack request signature for testing."""
    ts = str(int(time.time()))
    base = f"v0:{ts}:{body.decode()}"
    sig = "v0=" + hmac.new(secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    return ts, sig


class TestSlackDirectAgent:
    """
    slack_approval posts a Block Kit message; when SLACK_BOT_TOKEN is empty it
    skips the API call but still populates _pending.
    /slack/actions receives the button callback and resumes the graph.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        import slack_direct_agent as sda
        sda.llm = _FakeLLM(amount=900.0)
        sda._pending.clear()
        sda._msg_ts.clear()
        sda.approval_graph._timeout = 0.05
        yield sda
        sda._pending.clear()
        sda._msg_ts.clear()

    def test_call_tool_returns_pending_approval(self, setup):
        # SLACK_BOT_TOKEN empty → dev-mode, skips API but stores pending
        sda = setup
        r = TestClient(sda.app).post("/call-tool", json={"message": "pay"})
        assert r.status_code == 200
        assert r.json()["status"] == "pending_approval"

    def test_slack_actions_approve_resumes_graph(self, setup):
        sda = setup

        with TestClient(sda.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            aid = step1["approval_id"]
            tid = step1["thread_id"]
            sda._pending[aid] = tid

            action_payload = json.dumps({
                "type": "block_actions",
                "actions": [{"action_id": "approve", "value": f"approve:{aid}"}],
                "message": {"ts": "111.222"},
                "channel": {"id": "C123"},
            })
            body = f"payload={action_payload}".encode()

            r = client.post(
                "/slack/actions",
                content=body,
                headers={"content-type": "application/x-www-form-urlencoded",
                         "X-Slack-Request-Timestamp": str(int(time.time())),
                         "X-Slack-Signature": "v0=skip"},
            )
            assert r.status_code == 200
            assert aid not in sda._pending  # consumed

    def test_slack_actions_reject_resumes_graph(self, setup):
        sda = setup

        with TestClient(sda.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            aid = step1["approval_id"]
            tid = step1["thread_id"]
            sda._pending[aid] = tid

            action_payload = json.dumps({
                "type": "block_actions",
                "actions": [{"action_id": "reject", "value": f"reject:{aid}"}],
                "message": {"ts": "111.222"},
                "channel": {"id": "C123"},
            })
            body = f"payload={action_payload}".encode()

            r = client.post(
                "/slack/actions",
                content=body,
                headers={"content-type": "application/x-www-form-urlencoded",
                         "X-Slack-Request-Timestamp": str(int(time.time())),
                         "X-Slack-Signature": "v0=skip"},
            )
            assert r.status_code == 200

    def test_slack_actions_invalid_signature_is_403(self, setup, monkeypatch):
        sda = setup
        # Set a signing secret so verification actually runs
        monkeypatch.setattr(sda, "SLACK_SIGNING_SECRET", "real-secret")

        body = b"payload=%7B%22actions%22%3A%5B%5D%7D"
        r = TestClient(sda.app).post(
            "/slack/actions",
            content=body,
            headers={"content-type": "application/x-www-form-urlencoded",
                     "X-Slack-Request-Timestamp": str(int(time.time())),
                     "X-Slack-Signature": "v0=badsig"},
        )
        assert r.status_code == 403

    def test_slack_actions_valid_hmac_is_accepted(self, setup, monkeypatch):
        sda = setup
        secret = "test-signing-secret"
        monkeypatch.setattr(sda, "SLACK_SIGNING_SECRET", secret)

        with TestClient(sda.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            aid = step1["approval_id"]
            sda._pending[aid] = step1["thread_id"]

            action_payload = json.dumps({
                "actions": [{"action_id": "approve", "value": f"approve:{aid}"}],
                "message": {"ts": "1.2"}, "channel": {"id": "C1"},
            })
            raw_body = f"payload={action_payload}".encode()
            ts, sig = _slack_signature(raw_body, secret)

            r = client.post(
                "/slack/actions",
                content=raw_body,
                headers={"content-type": "application/x-www-form-urlencoded",
                         "X-Slack-Request-Timestamp": ts,
                         "X-Slack-Signature": sig},
            )
            assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 5. Telegram Agent (telegram_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestTelegramAgent:
    """
    telegram_approval calls Telegram Bot API via httpx.AsyncClient.
    When TELEGRAM_BOT_TOKEN is empty it skips the API but stores in _pending.
    /telegram/webhook receives inline keyboard callbacks and resumes the graph.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        import telegram_agent as ta
        ta.llm = _FakeLLM(amount=3200.0)
        ta._pending.clear()
        ta.approval_graph._timeout = 0.05
        yield ta
        ta._pending.clear()

    def test_call_tool_returns_pending_approval(self, setup):
        # TELEGRAM_BOT_TOKEN empty → dev-mode, skips API but stores pending
        ta = setup
        r = TestClient(ta.app).post("/call-tool", json={"message": "pay"})
        assert r.status_code == 200
        assert r.json()["status"] == "pending_approval"

    def test_telegram_webhook_approve_resumes_graph(self, setup):
        ta = setup

        with TestClient(ta.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            aid = step1["approval_id"]
            tid = step1["thread_id"]
            ta._pending[aid] = tid

            update = {
                "update_id": 1,
                "callback_query": {
                    "id": "cq-001",
                    "data": f"approve:{aid}",
                    "message": {"message_id": 42, "chat": {"id": 99999}},
                },
            }
            r = client.post("/telegram/webhook", json=update)
            assert r.status_code == 200
            assert aid not in ta._pending  # consumed

    def test_telegram_webhook_reject_resumes_graph(self, setup):
        ta = setup

        with TestClient(ta.app) as client:
            step1 = client.post("/call-tool", json={"message": "pay"}).json()
            aid = step1["approval_id"]
            tid = step1["thread_id"]
            ta._pending[aid] = tid

            update = {
                "update_id": 2,
                "callback_query": {
                    "id": "cq-002",
                    "data": f"reject:{aid}",
                    "message": {"message_id": 43, "chat": {"id": 99999}},
                },
            }
            r = client.post("/telegram/webhook", json=update)
            assert r.status_code == 200

    def test_telegram_webhook_non_callback_update_ignored(self, setup):
        ta = setup
        # A regular message update (not a button press) must return 200 silently
        update = {"update_id": 3, "message": {"text": "hello", "chat": {"id": 1}}}
        r = TestClient(ta.app).post("/telegram/webhook", json=update)
        assert r.status_code == 200

    def test_telegram_webhook_unknown_approval_id_ignored(self, setup):
        ta = setup
        # Already-decided or unknown approval_id must not crash
        update = {
            "update_id": 4,
            "callback_query": {
                "id": "cq-003",
                "data": "approve:NONEXISTENT",
                "message": {"message_id": 1, "chat": {"id": 1}},
            },
        }
        r = TestClient(ta.app).post("/telegram/webhook", json=update)
        assert r.status_code == 200
