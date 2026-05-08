"""Audit sink abstraction + record + event types.

An :class:`IAuditSink` records discrete events that need durable, queryable
retention for compliance: policy decisions, tool invocation outcomes, and
(optionally, redacted) payloads. Concrete sinks ship in this subpackage.

Sinks NEVER raise from :meth:`record` or :meth:`flush` — any backend error
must be swallowed internally and logged. Audit is best-effort by design;
its failure must not block the calling pipeline.
"""

from __future__ import annotations

import enum
from typing import Protocol, runtime_checkable
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class AuditEvent(enum.StrEnum):
    """Discrete event types recorded by the audit sink."""

    POLICY_DECISION = "policy.decision"
    TOOL_INVOCATION_STARTED = "tool.invocation.started"
    TOOL_INVOCATION_COMPLETED = "tool.invocation.completed"
    TOOL_INVOCATION_FAILED = "tool.invocation.failed"


# Optional pluggable redaction. Default identity. Implementations may strip PII
# before the record reaches the sink.
PayloadRedactor = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _identity_redactor(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(payload)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Immutable audit record. Sinks accept records via :meth:`IAuditSink.record`."""

    event: AuditEvent
    timestamp: datetime
    tool_name: str | None
    tool_version: int | None
    agent_id: str | None
    tenant_id: str | None
    decision_path: str | None
    decision_allowed: bool | None
    decision_reason: str | None
    error_code: str | None
    payload: Mapping[str, Any]
    latency_ms: float | None

    @classmethod
    def now(
        cls,
        event: AuditEvent,
        *,
        tool_name: str | None = None,
        tool_version: int | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
        decision_path: str | None = None,
        decision_allowed: bool | None = None,
        decision_reason: str | None = None,
        error_code: str | None = None,
        payload: Mapping[str, Any] | None = None,
        latency_ms: float | None = None,
        redactor: PayloadRedactor = _identity_redactor,
    ) -> AuditRecord:
        return cls(
            event=event,
            timestamp=datetime.now(UTC),
            tool_name=tool_name,
            tool_version=tool_version,
            agent_id=agent_id,
            tenant_id=tenant_id,
            decision_path=decision_path,
            decision_allowed=decision_allowed,
            decision_reason=decision_reason,
            error_code=error_code,
            payload=dict(redactor(payload or {})),
            latency_ms=latency_ms,
        )


@runtime_checkable
class IAuditSink(Protocol):
    """Durable record of policy and tool events.

    Implementations MUST:

    * be safe for concurrent use across coroutines;
    * never raise from :meth:`record` or :meth:`flush`;
    * make :meth:`flush` idempotent (called by Container.stop at shutdown).
    """

    async def record(self, record: AuditRecord) -> None:
        """Persist a single audit record. Best-effort; never raises."""
        ...

    async def flush(self) -> None:
        """Flush any buffered records. Idempotent."""
        ...


__all__ = [
    "AuditEvent",
    "AuditRecord",
    "IAuditSink",
    "PayloadRedactor",
]
