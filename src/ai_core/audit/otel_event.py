"""OTel-event audit sink — records audit events via IObservabilityProvider.

Trace-shaped retention applies (sampled, time-series). For compliance-grade
durability use :class:`JsonlFileAuditSink` or a custom backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.observability.logging import get_logger

if TYPE_CHECKING:
    from ai_core.di.interfaces import IObservabilityProvider

_logger = get_logger(__name__)


class OTelEventAuditSink(IAuditSink):
    """Records audit events as observability events.

    Each record produces one ``eaap.audit.<event>`` event with structured
    attributes carrying the audit fields (tool, agent, decision, latency).
    The ``payload`` field is intentionally NOT emitted to OTel (cardinality
    concern); use :class:`JsonlFileAuditSink` for payload retention.
    """

    def __init__(self, observability: IObservabilityProvider) -> None:
        self._obs = observability

    async def record(self, record: AuditRecord) -> None:
        try:
            await self._obs.record_event(
                f"eaap.audit.{record.event.value}",
                attributes=_record_to_attributes(record),
            )
        except Exception as exc:  # sinks NEVER raise
            _logger.warning(
                "audit.otel_sink.failed",
                audit_event=record.event.value, error=str(exc),
                error_type=type(exc).__name__,
            )

    async def flush(self) -> None:
        return None


def _record_to_attributes(record: AuditRecord) -> dict[str, Any]:
    """Render an AuditRecord into a flat OTel-attribute dict (scalars only)."""
    return {
        "audit.timestamp": record.timestamp.isoformat(),
        "audit.tool_name": record.tool_name or "",
        "audit.tool_version": record.tool_version or 0,
        "audit.agent_id": record.agent_id or "",
        "audit.tenant_id": record.tenant_id or "",
        "audit.decision_path": record.decision_path or "",
        "audit.decision_allowed": (
            record.decision_allowed if record.decision_allowed is not None else False
        ),
        "audit.decision_reason": record.decision_reason or "",
        "audit.error_code": record.error_code or "",
        "audit.latency_ms": record.latency_ms or 0.0,
    }


__all__ = ["OTelEventAuditSink"]
