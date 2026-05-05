"""Tests for JsonlFileAuditSink."""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ai_core.audit import AuditEvent, AuditRecord
from ai_core.audit.jsonl import JsonlFileAuditSink

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_jsonl_sink_writes_line_delimited(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path, buffer_size=2)
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION,
                                       tool_name="a", agent_id="x"))
    await sink.record(AuditRecord.now(AuditEvent.TOOL_INVOCATION_COMPLETED,
                                       tool_name="b", latency_ms=12.5))
    # Buffer fills at 2 records → flush triggers.
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    record_a = json.loads(lines[0])
    assert record_a["event"] == "policy.decision"
    assert record_a["tool_name"] == "a"
    record_b = json.loads(lines[1])
    assert record_b["event"] == "tool.invocation.completed"
    assert record_b["latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_jsonl_sink_flush_drains_buffer(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path, buffer_size=100)
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))
    # Buffer is below threshold; nothing on disk yet.
    assert not path.exists() or path.read_text() == ""
    await sink.flush()
    # After flush, the record is on disk.
    assert path.read_text().count("\n") == 1


@pytest.mark.asyncio
async def test_jsonl_sink_flush_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path)
    await sink.flush()  # buffer empty
    await sink.flush()  # second call must not raise
    assert not path.exists() or path.read_text() == ""


@pytest.mark.asyncio
async def test_jsonl_sink_handles_concurrent_record_calls(tmp_path: Path) -> None:
    """Concurrent record() calls should not interleave bytes within a line."""
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path, buffer_size=1)  # flush after every record
    await asyncio.gather(*(
        sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION,
                                     tool_name=f"tool-{i}"))
        for i in range(20)
    ))
    lines = path.read_text().splitlines()
    assert len(lines) == 20
    # Every line is a valid JSON object — no interleaving.
    for line in lines:
        json.loads(line)


@pytest.mark.asyncio
async def test_jsonl_sink_swallows_write_errors(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """If the file write fails, record() must NOT raise."""
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path, buffer_size=1)

    def _explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk full (test-injected)")

    monkeypatch.setattr(sink, "_append_lines", _explode)
    # Must not raise.
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))
