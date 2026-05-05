"""Meta-test: every IAuditSink implementation MUST swallow internal errors."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai_core.audit import AuditEvent, AuditRecord
from ai_core.audit.jsonl import JsonlFileAuditSink
from ai_core.audit.null import NullAuditSink
from ai_core.audit.otel_event import OTelEventAuditSink

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_null_sink_record_never_raises() -> None:
    sink = NullAuditSink()
    # Even with a malformed-shape AuditRecord, NullAuditSink never raises.
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))
    await sink.flush()


@pytest.mark.asyncio
async def test_otel_event_sink_record_never_raises_when_backend_fails() -> None:
    """Backend exception inside record_event must be swallowed."""
    class _BadObs:
        def start_span(self, *args: Any, **kwargs: Any) -> Any: ...
        async def record_llm_usage(self, *args: Any, **kwargs: Any) -> None: ...
        async def record_event(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("backend down")
        async def shutdown(self) -> None: ...

    sink = OTelEventAuditSink(_BadObs())  # type: ignore[arg-type]
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))


@pytest.mark.asyncio
async def test_jsonl_sink_record_never_raises_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = JsonlFileAuditSink(tmp_path / "audit.jsonl", buffer_size=1)

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full (test-injected)")

    monkeypatch.setattr(sink, "_append_lines", _explode)
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))


@pytest.mark.asyncio
async def test_jsonl_sink_flush_never_raises_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = JsonlFileAuditSink(tmp_path / "audit.jsonl", buffer_size=100)
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full (test-injected)")

    monkeypatch.setattr(sink, "_append_lines", _explode)
    await sink.flush()  # must not raise
