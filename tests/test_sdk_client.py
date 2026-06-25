import httpx
import pytest
from unittest.mock import MagicMock

from intrupt_py_sdk.core.client import ApprovalClient


_VALID_KEY = "sk_org_org_test1234_abcdef0123456789"
_ORG_ID = "org_test1234"


@pytest.fixture
def client():
    return ApprovalClient(base_url="http://api.test", api_key=_VALID_KEY)


def _ok_response(json_body, status_code=200):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body
    r.raise_for_status = MagicMock()
    return r


def _http_status_error(status_code: int = 502, message: str = "Bad Gateway"):
    req = httpx.Request("POST", "http://api.test")
    resp = httpx.Response(status_code, request=req)
    return httpx.HTTPStatusError(f"{status_code} {message}", request=req, response=resp)


class TestCreateApproval:
    def test_requires_thread_id(self, client, monkeypatch):
        called = []
        monkeypatch.setattr(httpx, "post", lambda *a, **kw: called.append(kw) or _ok_response({}))
        with pytest.raises(ValueError, match="thread_id"):
            client.create_approval(
                thread_id="",
                action="a", message="m", channel="slack",
                tool={"name": "t", "kwargs": {}},
            )
        assert called == []

    def test_posts_to_approval_endpoint(self, client, monkeypatch):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return _ok_response({"approval_id": "A1", "status": "pending"})

        monkeypatch.setattr(httpx, "post", fake_post)

        result = client.create_approval(
            thread_id="T1", action="purchase_stock",
            message="approve?", channel="slack",
            tool={"name": "purchase_stock", "kwargs": {"symbol": "AAPL", "quantity": 5}},
            agent_callback_url="http://agent/resume",
        )

        assert result == {"approval_id": "A1", "status": "pending"}
        assert captured["url"] == f"http://api.test/org/{_ORG_ID}/approval"
        assert captured["headers"]["Authorization"] == f"Bearer {_VALID_KEY}"
        assert captured["timeout"] is not None

        body = captured["json"]
        assert body["thread_id"] == "T1"
        assert body["action"] == "purchase_stock"
        assert body["channel"] == "slack"
        assert body["tool_name"] == "purchase_stock"
        assert body["tool_kwargs"] == {"symbol": "AAPL", "quantity": 5}
        assert body["agent_callback_url"] == "http://agent/resume"

    def test_does_not_send_workflow_id(self, client, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            httpx, "post",
            lambda url, **kw: captured.update(kw) or _ok_response({"approval_id": "A", "status": "pending"})
        )
        client.create_approval(
            thread_id="T", action="a", message="m", channel="slack",
            tool={"name": "t", "kwargs": {}},
        )
        assert "workflow_id" not in captured["json"]

    def test_no_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("APPROVAL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key is required"):
            ApprovalClient(base_url="http://x", api_key=None)

    def test_propagates_http_error(self, client, monkeypatch):
        bad = MagicMock()
        bad.raise_for_status.side_effect = _http_status_error(502, "Bad Gateway")
        monkeypatch.setattr(httpx, "post", lambda *a, **kw: bad)

        with pytest.raises(httpx.HTTPStatusError):
            client.create_approval(
                thread_id="T", action="a", message="m", channel="slack",
                tool={"name": "t", "kwargs": {}},
            )

    def test_base_url_trailing_slash_stripped(self, monkeypatch):
        c = ApprovalClient(base_url="http://api.test/", api_key=_VALID_KEY)
        captured = {}
        monkeypatch.setattr(
            httpx, "post",
            lambda url, **kw: captured.update({"url": url}) or _ok_response({"approval_id": "A", "status": "pending"})
        )
        c.create_approval(
            thread_id="T", action="a", message="m", channel="slack",
            tool={"name": "t", "kwargs": {}},
        )
        assert captured["url"] == f"http://api.test/org/{_ORG_ID}/approval"


class TestHooks:
    def test_emit_invokes_registered_callbacks(self):
        c = ApprovalClient(base_url="http://x", api_key=_VALID_KEY)
        seen = []
        c.add_hook("approval.created", lambda p: seen.append(("a", p)))
        c.add_hook("approval.created", lambda p: seen.append(("b", p)))
        c.emit("approval.created", {"id": "A"})
        assert seen == [("a", {"id": "A"}), ("b", {"id": "A"})]

    def test_emit_unknown_event_is_silent(self):
        ApprovalClient(base_url="http://x", api_key=_VALID_KEY).emit("nope", {})

    def test_invalid_key_format_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid API key format"):
            ApprovalClient(base_url="http://x", api_key="bad-key")

    def test_key_without_org_prefix_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid org_id"):
            ApprovalClient(base_url="http://x", api_key="sk_org_notorgid_abcdef0123456789")
