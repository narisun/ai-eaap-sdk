"""DatadogAuditSink — forward audit records to Datadog as first-class events.

Ships under the optional ``[datadog]`` extra. ``record`` maps each
:class:`AuditRecord` to a ``datadog.api.Event.create`` call with auto
alert_type. ``flush`` is a no-op (Datadog events API has no buffer).
Errors swallowed per the :class:`IAuditSink` never-raise contract.
"""

from __future__ import annotations

import json
import logging

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.exceptions import ConfigurationError

_logger = logging.getLogger(__name__)


class DatadogAuditSink(IAuditSink):
    """Forward :class:`AuditRecord` instances to Datadog.

    Args:
        api_key: Required Datadog API key.
        app_key: Optional application key.
        site: Datadog site (default ``datadoghq.com``).
        source: Source name on emitted events.
        environment: Optional environment tag added as ``env:<value>``.

    Raises:
        ConfigurationError: If ``datadog`` is not installed
            (``error_code='config.optional_dep_missing'``,
            ``details={'extra': 'datadog'}``).
    """

    def __init__(
        self,
        *,
        api_key: str,
        app_key: str | None = None,
        site: str = "datadoghq.com",
        source: str = "ai-core-sdk",
        environment: str | None = None,
    ) -> None:
        try:
            import datadog  # noqa: PLC0415
        except ImportError as exc:
            raise ConfigurationError(
                "Datadog sink requires the 'datadog' optional dependency. "
                "Install with: pip install ai-core-sdk[datadog]",
                error_code="config.optional_dep_missing",
                details={"extra": "datadog"},
                cause=exc,
            ) from exc

        datadog.initialize(
            api_key=api_key,
            app_key=app_key,
            api_host=f"https://api.{site}",
        )
        self._datadog = datadog
        self._source = source
        self._environment = environment

    async def record(self, record: AuditRecord) -> None:
        try:
            alert_type = (
                "warning"
                if (record.decision_allowed is False or record.error_code is not None)
                else "info"
            )
            tags: list[str] = [
                f"event:{record.event.value}",
                f"tool_name:{record.tool_name or 'unknown'}",
                f"agent_id:{record.agent_id or 'unknown'}",
                f"tenant_id:{record.tenant_id or 'unknown'}",
            ]
            if record.error_code:
                tags.append(f"error_code:{record.error_code}")
            if self._environment:
                tags.append(f"env:{self._environment}")
            if record.decision_path:
                tags.append(f"decision_path:{record.decision_path}")
            if record.decision_allowed is not None:
                tags.append(f"decision_allowed:{record.decision_allowed}")

            text_payload = json.dumps({
                "payload": dict(record.payload),
                "decision_reason": record.decision_reason,
                "latency_ms": record.latency_ms,
            })

            self._datadog.api.Event.create(  # type: ignore[attr-defined,no-untyped-call]
                title=f"eaap.audit.{record.event.value}",
                text=text_payload,
                tags=tags,
                alert_type=alert_type,
                source_type_name=self._source,
            )
        except Exception as exc:
            _logger.warning(
                "audit.datadog.record_failed",
                extra={
                    "event": record.event.value,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    async def flush(self) -> None:
        # Datadog events API is synchronous-per-call; no buffer to flush.
        return


__all__ = ["DatadogAuditSink"]
