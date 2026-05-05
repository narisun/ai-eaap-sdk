"""Line-delimited JSON audit sink.

Suitable for dev/test/single-tenant deployments. Buffered writes via
``asyncio.to_thread`` keep the event loop responsive. ``flush()`` is
called by ``Container.stop`` at shutdown to drain any partial buffer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.observability.logging import get_logger

_logger = get_logger(__name__)


class JsonlFileAuditSink(IAuditSink):
    """Append audit records as line-delimited JSON to a local file.

    Args:
        path: Filesystem path where audit records are appended.
        buffer_size: Number of records to buffer before flushing to disk.
    """

    def __init__(self, path: Path | str, *, buffer_size: int = 64) -> None:
        self._path = Path(path)
        self._buffer: list[dict[str, Any]] = []
        self._buffer_size = buffer_size
        self._lock = asyncio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def record(self, record: AuditRecord) -> None:
        try:
            payload = _record_to_dict(record)
            async with self._lock:
                self._buffer.append(payload)
                if len(self._buffer) >= self._buffer_size:
                    await self._flush_locked()
        except Exception as exc:  # sinks NEVER raise
            _logger.warning(
                "audit.jsonl_sink.failed",
                audit_event=record.event.value, error=str(exc),
                error_type=type(exc).__name__,
            )

    async def flush(self) -> None:
        try:
            async with self._lock:
                await self._flush_locked()
        except Exception as exc:  # sinks NEVER raise
            _logger.warning(
                "audit.jsonl_sink.flush_failed",
                error=str(exc), error_type=type(exc).__name__,
            )

    async def _flush_locked(self) -> None:
        """Drain the buffer to disk. Caller MUST hold ``self._lock``."""
        if not self._buffer:
            return
        records = self._buffer
        self._buffer = []
        await asyncio.to_thread(self._append_lines, records)

    def _append_lines(self, records: list[dict[str, Any]]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def _record_to_dict(record: AuditRecord) -> dict[str, Any]:
    return {
        "event": record.event.value,
        "timestamp": record.timestamp.isoformat(),
        "tool_name": record.tool_name,
        "tool_version": record.tool_version,
        "agent_id": record.agent_id,
        "tenant_id": record.tenant_id,
        "decision_path": record.decision_path,
        "decision_allowed": record.decision_allowed,
        "decision_reason": record.decision_reason,
        "error_code": record.error_code,
        "latency_ms": record.latency_ms,
        "payload": dict(record.payload),
    }


__all__ = ["JsonlFileAuditSink"]
