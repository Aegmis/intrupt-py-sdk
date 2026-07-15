"""OpenTelemetry emission for governed tool calls.

Design contract — this is the OPPOSITE of the approval gate:
  * fail-OPEN: any tracing error is swallowed; telemetry must never break or
    block a tool call. (The gate is fail-CLOSED.)
  * push-only: spans/metrics are POSTed OUTBOUND to the obs service over OTLP.
    The agent opens no inbound port.
  * no-op unless configured: if `init(endpoint)` is never called with a real
    endpoint, `tool_span()` yields a null recorder and costs nothing.

Usage (agent startup):
    from intrupt_py_sdk.core import observability
    observability.init(os.getenv("AEGMIS_OTLP_ENDPOINT"),   # e.g. http://localhost:8090
                       org_id=os.getenv("AEGMIS_ORG_ID"))   # tenant tag for dashboard scoping
    ...
    observability.shutdown()   # flush on exit (important for short-lived agents)
"""
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

_tracer = None
_approvals = None       # counter
_wait_seconds = None    # histogram
_provider = None        # TracerProvider, kept for force_flush on shutdown
_meter_provider = None  # MeterProvider, kept for force_flush on shutdown
_endpoint = None        # obs service base URL, used for the REST lifecycle channel
_service = "aegmis-agent"
_org_id = None          # tenant tag stamped on every pending/resolve + OTLP resource
_obs_unreachable = False # True after a push fails; re-armed on recovery (one-time warn)
_obs_warn_lock = threading.Lock()


def _post_json(path: str, body: dict) -> None:
    """Fire-and-forget POST to the obs service. Runs in a daemon thread so it
    never blocks the tool call; fully fail-open (a down obs service is ignored)."""
    if not _endpoint:
        return

    def _run():
        global _obs_unreachable
        try:
            import httpx
            httpx.post(f"{_endpoint}{path}", json=body, timeout=3.0)
            # Recovered: if we had warned about being down, note it resumed (once).
            if _obs_unreachable:
                with _obs_warn_lock:
                    if _obs_unreachable:
                        _obs_unreachable = False
                        logger.warning(
                            "observability: endpoint %s reachable again — telemetry resumed",
                            _endpoint,
                        )
        except Exception as exc:
            # First failure of an outage → one WARNING (re-armed on recovery), so a
            # dead obs service is visible without spamming a line per tool call.
            if not _obs_unreachable:
                with _obs_warn_lock:
                    if not _obs_unreachable:
                        _obs_unreachable = True
                        logger.warning(
                            "observability: endpoint %s unreachable (%s) — telemetry paused. "
                            "Governed tool calls are UNAFFECTED (observability is fail-open).",
                            _endpoint, exc.__class__.__name__,
                        )
            logger.debug("observability: POST %s failed", path, exc_info=True)

    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        logger.debug("observability: could not start POST thread", exc_info=True)


def _org_from_api_key() -> Optional[str]:
    """Identify the agent's org from AEGMIS_API_KEY (format sk_org_{org_id}_{hash}),
    e.g. sk_org_org_0819dfb9_<hash> → org_0819dfb9. Returns None if unparseable.

    This lets the SDK identify itself — callers don't have to pass/AEGMIS_ORG_ID."""
    key = os.getenv("AEGMIS_API_KEY", "")
    if not key.startswith("sk_org_"):
        return None
    after = key[7:]                     # drop "sk_org_"
    i = after.rfind("_")                # last "_" separates org_id from the hash
    if i <= 0:
        return None
    org = after[:i]
    return org if org.startswith("org_") else None


def _tracing_enabled(flag: Optional[bool]) -> bool:
    """Resolve the tracing switch: explicit arg wins, else the AEGMIS_ENABLE_TRACING
    env var. Default is TRUE (tracing on). Set AEGMIS_ENABLE_TRACING=false to opt out."""
    if flag is not None:
        return bool(flag)
    v = os.getenv("AEGMIS_ENABLE_TRACING")
    if v is None:
        return True  # default: enabled
    return v.strip().lower() not in ("0", "false", "no", "off")


def init(endpoint: Optional[str], service_name: str = "aegmis-agent",
         org_id: Optional[str] = None, enabled: Optional[bool] = None) -> None:
    """Enable OTLP + REST emission to `endpoint`.

    org_id:  tenant tag stamped on every record. If omitted, the SDK identifies
             itself — first AEGMIS_ORG_ID, else derived from AEGMIS_API_KEY. You
             normally do NOT need to pass this.
    enabled: tracing switch. Default True (tracing on). Pass False — or set
             AEGMIS_ENABLE_TRACING=false — to turn all emission off (no-op).
    """
    global _tracer, _approvals, _wait_seconds, _provider, _meter_provider, _endpoint, _service, _org_id
    if not _tracing_enabled(enabled):
        logger.info("observability: tracing disabled (AEGMIS_ENABLE_TRACING=false / enabled=False)")
        return
    if not endpoint:
        logger.debug("observability: no endpoint configured; tracing disabled")
        return
    # SDK identifies its own tenant: explicit org_id → AEGMIS_ORG_ID → API key.
    if org_id is None:
        org_id = os.getenv("AEGMIS_ORG_ID") or _org_from_api_key()
    if org_id:
        logger.debug("observability: tagging tenant org_id=%s", org_id)
    else:
        logger.warning("observability: no org_id (no AEGMIS_ORG_ID / unparseable AEGMIS_API_KEY) "
                       "— records will be untagged and hidden from org-scoped dashboards")
    # Set the REST endpoint FIRST so the pending/resolve lifecycle works even if
    # the OTel span/metric setup below fails for any reason.
    _endpoint = endpoint.rstrip("/")
    _service = service_name
    _org_id = org_id
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = _endpoint
        res_attrs = {"service.name": service_name}
        if org_id:
            res_attrs["aegmis.org_id"] = org_id   # survives the OTLP span path too
        res = Resource.create(res_attrs)

        _provider = TracerProvider(resource=res)
        # Batched + background thread → export never blocks the tool call.
        _provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        trace.set_tracer_provider(_provider)

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics")
        )
        _meter_provider = MeterProvider(resource=res, metric_readers=[reader])
        metrics.set_meter_provider(_meter_provider)

        _tracer = trace.get_tracer("aegmis")
        meter = metrics.get_meter("aegmis")
        _approvals = meter.create_counter(
            "aegmis.approvals", description="governed tool outcomes"
        )
        _wait_seconds = meter.create_histogram(
            "aegmis.approval.wait_seconds", unit="s",
            description="seconds a governed tool waited on the approval gate",
        )
        logger.info("observability: emitting OTLP to %s", endpoint)
    except Exception:
        # Never let a tracing setup problem take down the agent.
        logger.exception("observability: init failed; continuing untraced")
        _tracer = None


def shutdown(timeout_ms: int = 3000) -> None:
    """Flush buffered spans. Call on process/app shutdown so nothing is lost."""
    if _provider is not None:
        try:
            _provider.force_flush(timeout_millis=timeout_ms)
        except Exception:
            logger.exception("observability: span flush on shutdown failed")
    if _meter_provider is not None:
        try:
            _meter_provider.force_flush(timeout_millis=timeout_ms)
        except Exception:
            logger.exception("observability: metric flush on shutdown failed")


class _NullRec:
    def requested(self, *_a, **_k): ...
    def finish(self, *_a, **_k): ...


class _Rec:
    """Records the approval lifecycle: a 'pending' record posted the moment
    approval is requested (so the dashboard shows it live as WAITING), then a
    'resolve' when the human responds — plus the OTel span/metrics.

    The span may be None (OTel disabled); span ops are then skipped, but the
    REST lifecycle still runs so the dashboard works."""

    def __init__(self, span, adapter: str, action: str, started: float,
                 tool: str, thread_id: str):
        self._s = span
        self._adapter = adapter
        self._action = action
        self._t0 = started
        self._tool = tool
        self._thread_id = thread_id
        self._approval_id: Optional[str] = None
        self._auto_status: Optional[str] = None  # set when the policy decided (no human)
        self._done = False

    def requested(self, approval_id: str, payload: Optional[dict] = None) -> None:
        """Approval requested. Posts the WAITING row (with full payload) at once.

        Auto-decided calls (below-threshold allow / audit-only / auto-reject)
        have NO approval_id — only an audit_id. They are recorded too, keyed by
        the audit id, so the dashboard shows their args and that the policy
        (not a human) decided them."""
        self._auto_status = (payload or {}).get("_auto_status")
        if not approval_id and self._auto_status:
            # Key the record by the platform's audit id (or a local id as a
            # last resort) so pending + resolve land on one row.
            approval_id = (payload or {}).get("_audit_id") or f"auto-{uuid.uuid4().hex[:12]}"
        self._approval_id = approval_id
        # Span event (guarded — span may be None).
        try:
            if self._s is not None:
                if approval_id:
                    self._s.set_attribute("aegmis.approval.id", approval_id)
                self._s.add_event("approval.requested",
                                  {"aegmis.approval.id": approval_id or ""})
        except Exception:
            logger.debug("observability: requested() span op failed", exc_info=True)
        # Live 'waiting' row with the tool's payload. Payload/args go ONLY here
        # (our own service), never into the OTLP span that may reach a 3rd party.
        if approval_id:
            tool = payload.get("tool", {}) if payload else {}
            _post_json("/v1/pending", {
                "approval_id": approval_id,
                "org_id": _org_id,
                "tool": (tool.get("name") if isinstance(tool, dict) else None) or self._tool,
                "action": self._action,
                "adapter": self._adapter,
                "thread_id": self._thread_id,
                "service": _service,
                "message": (payload or {}).get("message"),
                "kwargs": tool.get("kwargs") if isinstance(tool, dict) else None,
                "channel": (payload or {}).get("channel"),
                # Which policy/conditions triggered the approval — stashed on the
                # payload by gate.request_approval from the API response.
                "policy_eval": (payload or {}).get("_policy_eval"),
            })

    def finish(self, status: str, policy_id: Optional[str] = None) -> None:
        # status: approved | rejected | audited | error
        if self._done:
            return
        self._done = True
        waited = time.monotonic() - self._t0
        # Policy-decided calls (no human): record as 'audited', not 'approved',
        # so the dashboard clearly shows the call was auto-allowed (e.g. amount
        # under the policy threshold) rather than human-approved.
        reason: Optional[str] = None
        if self._auto_status:
            if status == "approved":
                status = "audited"
                reason = "auto-approved by policy (below approval threshold / audit-only)"
            elif status == "rejected":
                reason = "auto-rejected by policy"
        # Span attrs + metrics (guarded).
        try:
            if self._s is not None:
                self._s.set_attribute("aegmis.approval.status", status)
                if policy_id:
                    self._s.set_attribute("aegmis.policy.id", policy_id)
                self._s.add_event("approval.resolved",
                                  {"aegmis.approval.status": status,
                                   "aegmis.wait_seconds": waited})
            attrs = {"status": status, "adapter": self._adapter, "action": self._action}
            if _approvals:
                _approvals.add(1, attrs)
            if _wait_seconds:
                _wait_seconds.record(waited, attrs)
        except Exception:
            logger.debug("observability: finish() span/metric op failed", exc_info=True)
        # Update the row → status flips from waiting to the outcome. Approver
        # identity is NOT sent from the agent — the obs service pulls it from
        # the approval platform's /internal/obs feed (system of record).
        if self._approval_id:
            _post_json("/v1/resolve", {
                "approval_id": self._approval_id,
                "org_id": _org_id,
                "status": status,
                "wait_s": waited,
                "reason": reason,
            })


@contextmanager
def tool_span(tool_name: str, adapter: str, action: str, thread_id: str = ""):
    """Open a governed-tool context. Yields a recorder (null if disabled).

    Drives BOTH the REST approval lifecycle (pending/resolve → live dashboard)
    and, when OTel is available, an open span whose duration is the end-to-end
    latency. Works with the span disabled — the lifecycle only needs _endpoint.
    """
    if not _endpoint:
        yield _NullRec()
        return

    started = time.monotonic()

    # No OTel tracer (import/setup failed) → run the REST lifecycle without a span.
    if _tracer is None:
        rec = _Rec(None, adapter, action, started, tool_name, thread_id)
        try:
            yield rec
        except Exception:
            rec.finish("error")
            raise
        return

    from opentelemetry.trace import Status, StatusCode
    with _tracer.start_as_current_span(f"tool.{tool_name}") as span:
        try:
            # OTel GenAI + OpenInference-friendly attrs → any viewer understands them.
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", tool_name)
            span.set_attribute("aegmis.adapter", adapter)
            span.set_attribute("aegmis.action", action)
            if thread_id:
                span.set_attribute("aegmis.thread_id", thread_id)
        except Exception:
            logger.debug("observability: attr set failed", exc_info=True)

        rec = _Rec(span, adapter, action, started, tool_name, thread_id)
        try:
            yield rec
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            try:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                rec.finish("error")
            except Exception:
                logger.debug("observability: error recording failed", exc_info=True)
            raise
