"""Tests for AuditRecord, AuditEvent, NullAuditSink, and the redactor protocol."""
from __future__ import annotations

import pytest

from ai_core.audit import AuditEvent, AuditRecord, NullAuditSink

pytestmark = pytest.mark.unit


def test_audit_event_values() -> None:
    """AuditEvent string values are stable identifiers."""
    assert AuditEvent.POLICY_DECISION.value == "policy.decision"
    assert AuditEvent.TOOL_INVOCATION_STARTED.value == "tool.invocation.started"
    assert AuditEvent.TOOL_INVOCATION_COMPLETED.value == "tool.invocation.completed"
    assert AuditEvent.TOOL_INVOCATION_FAILED.value == "tool.invocation.failed"


def test_audit_record_now_populates_timestamp() -> None:
    """AuditRecord.now() stamps a UTC timestamp."""
    rec = AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        tool_name="search", tool_version=1,
        agent_id="a", tenant_id="t",
        decision_path="eaap/policy/allow",
        decision_allowed=True,
    )
    assert rec.event == AuditEvent.POLICY_DECISION
    assert rec.tool_name == "search"
    assert rec.tool_version == 1
    assert rec.decision_allowed is True
    assert rec.timestamp.tzinfo is not None  # has tzinfo (UTC)


def test_audit_record_default_redactor_is_identity() -> None:
    """Without a redactor, payload passes through unchanged."""
    rec = AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        payload={"input": {"q": "hi"}},
    )
    assert rec.payload == {"input": {"q": "hi"}}


def test_audit_record_with_custom_redactor() -> None:
    """Caller can supply a redactor that strips sensitive fields."""
    def _strip_password(payload):  # type: ignore[no-untyped-def]
        return {k: v for k, v in payload.items() if k != "password"}

    rec = AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        payload={"input": "hi", "password": "secret"},
        redactor=_strip_password,
    )
    assert "password" not in rec.payload
    assert rec.payload["input"] == "hi"


def test_audit_record_is_frozen() -> None:
    """AuditRecord is immutable."""
    rec = AuditRecord.now(AuditEvent.POLICY_DECISION)
    with pytest.raises(AttributeError):
        rec.tool_name = "x"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_null_audit_sink_record_is_noop() -> None:
    sink = NullAuditSink()
    # Should not raise; should not require any setup.
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))


@pytest.mark.asyncio
async def test_null_audit_sink_flush_is_idempotent() -> None:
    sink = NullAuditSink()
    await sink.flush()
    await sink.flush()  # second call must not raise
