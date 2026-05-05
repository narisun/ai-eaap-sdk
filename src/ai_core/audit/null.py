"""No-op audit sink. Default DI binding for development."""

from __future__ import annotations

from ai_core.audit.interface import AuditRecord, IAuditSink


class NullAuditSink(IAuditSink):
    """Audit sink that drops every record. Default for local development."""

    async def record(self, record: AuditRecord) -> None:
        return None

    async def flush(self) -> None:
        return None


__all__ = ["NullAuditSink"]
