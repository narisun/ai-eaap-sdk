"""Tests for OTelEventAuditSink — records via IObservabilityProvider.record_event."""
from __future__ import annotations

import pytest

from ai_core.audit import AuditEvent, AuditRecord
from ai_core.audit.otel_event import OTelEventAuditSink

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_otel_event_sink_records_via_observability(fake_observability) -> None:  # type: ignore[no-untyped-def]
    """OTelEventAuditSink calls record_event with eaap.audit.<event> name."""
    sink = OTelEventAuditSink(fake_observability)
    await sink.record(AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        tool_name="search", tool_version=1,
        agent_id="a", tenant_id="t",
        decision_path="eaap/policy/allow",
        decision_allowed=True,
        decision_reason="ok",
    ))
    events = [(name, dict(attrs)) for name, attrs in fake_observability.events]
    assert any(name == "eaap.audit.policy.decision" for name, _ in events)
    matching = next(attrs for name, attrs in events if name == "eaap.audit.policy.decision")
    assert matching["audit.tool_name"] == "search"
    assert matching["audit.tool_version"] == 1
    assert matching["audit.decision_allowed"] is True


@pytest.mark.asyncio
async def test_otel_event_sink_swallows_backend_errors(fake_observability) -> None:  # type: ignore[no-untyped-def]
    """If the observability provider's record_event raises, the sink swallows."""
    class _BadObservability:
        async def record_event(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("backend down")
        # Stubs for unused interface methods (sink doesn't call them).
        def start_span(self, *args: object, **kwargs: object): ...
        async def record_llm_usage(self, *args: object, **kwargs: object) -> None: ...
        async def shutdown(self) -> None: ...

    sink = OTelEventAuditSink(_BadObservability())  # type: ignore[arg-type]
    # Must not raise.
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))


@pytest.mark.asyncio
async def test_otel_event_sink_emits_empty_string_for_undefined_decision(
    fake_observability,
) -> None:
    """When AuditRecord.decision_allowed is None (no policy decision),
    OTel attribute audit.decision_allowed should be empty string, not False."""
    sink = OTelEventAuditSink(fake_observability)
    await sink.record(AuditRecord.now(
        AuditEvent.TOOL_INVOCATION_COMPLETED,
        tool_name="x", tool_version=1,
        agent_id="a", tenant_id="t",
        # decision_allowed not set — defaults to None
    ))

    events = fake_observability.events
    matching = next(attrs for name, attrs in events
                     if name == "eaap.audit.tool.invocation.completed")
    assert matching["audit.decision_allowed"] == ""


@pytest.mark.asyncio
async def test_otel_event_sink_flush_is_noop() -> None:
    """OTelEventAuditSink.flush is a no-op (observability owns its flush)."""
    class _NoopObs:
        def start_span(self, *args, **kwargs): ...  # noqa: ANN
        async def record_llm_usage(self, *args, **kwargs) -> None: ...
        async def record_event(self, *args, **kwargs) -> None: ...
        async def shutdown(self) -> None: ...

    sink = OTelEventAuditSink(_NoopObs())  # type: ignore[arg-type]
    await sink.flush()
