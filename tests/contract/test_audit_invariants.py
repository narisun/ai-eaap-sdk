"""IAuditSink concretes must never raise from record() or flush().

Phase 1 contract: backend errors are swallowed; audit is best-effort.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Force-import every audit sink module so __subclasses__() picks them up,
# including those behind optional-dep extras.
from ai_core.audit import AuditEvent, AuditRecord, IAuditSink

# Optional sinks — may not be installed in this environment.
for _modname in ("ai_core.audit.sentry", "ai_core.audit.datadog"):
    with contextlib.suppress(ImportError):
        importlib.import_module(_modname)


def _all_concrete_sinks() -> list[type[Any]]:
    seen: set[type[Any]] = set()
    stack: list[type[Any]] = list(IAuditSink.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    return sorted(
        (c for c in seen if not inspect.isabstract(c)),
        key=lambda c: c.__qualname__,
    )


def _make_test_record() -> AuditRecord:
    return AuditRecord.now(
        AuditEvent.TOOL_INVOCATION_COMPLETED,
        tool_name="test", tool_version=1,
        agent_id="a", tenant_id="t",
        payload={"input": {"q": "x"}},
    )


def _construct_sink_with_failing_backend(
    sink_cls: type[Any], monkeypatch: pytest.MonkeyPatch
) -> IAuditSink:
    """Return a constructed instance whose underlying transport raises on use.

    Uses monkeypatch.setitem so any sys.modules mutations auto-revert
    after the test, preventing pollution of subsequent unit tests that
    may import the real sentry_sdk / datadog modules.
    """
    name = sink_cls.__qualname__
    if name == "NullAuditSink":
        # NullAuditSink is a no-op; nothing to fault-inject. Return as-is.
        return sink_cls()  # type: ignore[no-any-return]
    if name == "JsonlFileAuditSink":
        # Use a read-only temp directory so the parent mkdir() succeeds
        # but the actual file write is refused by the OS.
        tmp_dir = tempfile.mkdtemp()
        os.chmod(tmp_dir, 0o555)  # read + execute only; writes denied
        bad_path = Path(tmp_dir) / "audit.jsonl"
        sink = sink_cls(bad_path)
        # Restore permissions so the directory can be cleaned up later.
        os.chmod(tmp_dir, 0o755)
        return sink  # type: ignore[no-any-return]
    if name == "OTelEventAuditSink":
        # Pass a fake observability provider whose record_event raises.
        fake_obs = MagicMock()

        async def _raise(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("backend-down")

        fake_obs.record_event = _raise
        return sink_cls(fake_obs)  # type: ignore[no-any-return]
    if name == "SentryAuditSink":
        fake = MagicMock()
        fake.capture_event.side_effect = RuntimeError("backend-down")
        fake.flush.side_effect = RuntimeError("backend-down")
        monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
        # Force a fresh import of the sink module so it sees the fake.
        monkeypatch.delitem(sys.modules, "ai_core.audit.sentry", raising=False)
        from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
        return SentryAuditSink(dsn="https://x@x/1")
    if name == "DatadogAuditSink":
        fake = MagicMock()
        fake.api.Event.create.side_effect = RuntimeError("backend-down")
        monkeypatch.setitem(sys.modules, "datadog", fake)
        monkeypatch.delitem(sys.modules, "ai_core.audit.datadog", raising=False)
        from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
        return DatadogAuditSink(api_key="dd-key")
    pytest.skip(f"No fault-injection harness defined for {name}")


@pytest.mark.parametrize(
    "sink_cls", _all_concrete_sinks(), ids=lambda c: c.__qualname__
)
def test_audit_sink_record_never_raises(
    sink_cls: type[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = _construct_sink_with_failing_backend(sink_cls, monkeypatch)
    record = _make_test_record()
    # Must NOT raise — Phase 1 contract.
    asyncio.run(sink.record(record))


@pytest.mark.parametrize(
    "sink_cls", _all_concrete_sinks(), ids=lambda c: c.__qualname__
)
def test_audit_sink_flush_never_raises(
    sink_cls: type[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = _construct_sink_with_failing_backend(sink_cls, monkeypatch)
    # Must NOT raise — Phase 1 contract.
    asyncio.run(sink.flush())


def test_at_least_three_concrete_sinks_exist() -> None:
    sinks = _all_concrete_sinks()
    assert len(sinks) >= 3, (
        f"Expected >=3 concrete IAuditSink subclasses, found "
        f"{[c.__qualname__ for c in sinks]}"
    )
