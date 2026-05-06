"""SentryAuditSink — forward audit records to Sentry as first-class events.

Ships under the optional ``[sentry]`` extra. ``record`` maps each
:class:`AuditRecord` to ``sentry_sdk.capture_event`` with auto-level
inference (warning if denied or errored; info otherwise). Errors swallowed
per the :class:`IAuditSink` never-raise contract.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.exceptions import ConfigurationError

_logger = logging.getLogger(__name__)


class SentryAuditSink(IAuditSink):
    """Forward :class:`AuditRecord` instances to Sentry.

    Args:
        dsn: Sentry project DSN.
        environment: Optional environment tag on every event.
        release: Optional release identifier.
        sample_rate: 0.0-1.0 fraction of events to send.

    Raises:
        ConfigurationError: If ``sentry-sdk`` is not installed
            (``error_code='config.optional_dep_missing'``,
            ``details={'extra': 'sentry'}``).
    """

    def __init__(
        self,
        *,
        dsn: str,
        environment: str | None = None,
        release: str | None = None,
        sample_rate: float = 1.0,
    ) -> None:
        try:
            import sentry_sdk  # noqa: PLC0415
        except ImportError as exc:
            raise ConfigurationError(
                "Sentry sink requires the 'sentry' optional dependency. "
                "Install with: pip install ai-core-sdk[sentry]",
                error_code="config.optional_dep_missing",
                details={"extra": "sentry"},
                cause=exc,
            ) from exc

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            sample_rate=sample_rate,
            max_breadcrumbs=0,
        )
        self._sentry_sdk = sentry_sdk

    async def record(self, record: AuditRecord) -> None:
        try:
            level = (
                "warning"
                if (record.decision_allowed is False or record.error_code is not None)
                else "info"
            )
            event: dict[str, Any] = {
                "message": f"eaap.audit.{record.event.value}",
                "level": level,
                "timestamp": record.timestamp.isoformat(),
                "tags": {
                    "audit.event": record.event.value,
                    "audit.tool_name": record.tool_name or "",
                    "audit.tool_version": (
                        str(record.tool_version) if record.tool_version is not None else ""
                    ),
                    "audit.agent_id": record.agent_id or "",
                    "audit.tenant_id": record.tenant_id or "",
                    "audit.decision_path": record.decision_path or "",
                    "audit.decision_allowed": (
                        str(record.decision_allowed)
                        if record.decision_allowed is not None
                        else ""
                    ),
                    "audit.error_code": record.error_code or "",
                },
                "extra": {
                    "payload": dict(record.payload),
                    "decision_reason": record.decision_reason,
                    "latency_ms": record.latency_ms,
                },
            }
            self._sentry_sdk.capture_event(event)  # type: ignore[arg-type]
        except Exception as exc:  # audit-sink never-raise contract
            _logger.warning(
                "audit.sentry.record_failed",
                extra={
                    "event": record.event.value,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    async def flush(self) -> None:
        try:
            self._sentry_sdk.flush(timeout=5.0)
        except Exception as exc:  # audit-sink never-raise contract
            _logger.warning(
                "audit.sentry.flush_failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )


__all__ = ["SentryAuditSink"]
