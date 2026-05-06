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
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Force-import every audit sink module so __subclasses__() picks them up,
# including those behind optional-dep extras.
from ai_core.audit import AuditEvent, AuditRecord, IAuditSink

# Optional sinks — may not be installed in this environment.
for _modname in ("ai_core.audit.sentry", "ai_core.audit.datadog"):
    with contextlib.suppress(ImportError):
        importlib.import_module(_modname)

pytestmark = pytest.mark.contract


def _all_concrete_sinks() -> list[type[IAuditSink]]:
    seen: set[type[IAuditSink]] = set()
    stack: list[type[IAuditSink]] = list(IAuditSink.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    # Filter out test-fixture sinks. Phase 9 moved the public fakes to
    # ai_core.testing, so we exclude that module path too.
    return sorted(
        (
            c for c in seen
            if not inspect.isabstract(c)
            and not c.__module__.startswith("tests.")
            and not c.__module__.startswith("ai_core.testing")
            and c.__module__ != "conftest"  # pytest-loaded conftest.py shows up as 'conftest'
        ),
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
    sink_cls: type[IAuditSink],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> IAuditSink:
    """Return a constructed instance whose underlying transport raises on use.

    Uses monkeypatch.setitem so any sys.modules mutations auto-revert
    after the test, preventing pollution of subsequent unit tests that
    may import the real sentry_sdk / datadog modules.
    """
    name = sink_cls.__qualname__
    if name == "NullAuditSink":
        # NullAuditSink is a no-op; nothing to fault-inject. Return as-is.
        return sink_cls()
    if name == "JsonlFileAuditSink":
        # Create the file in a normal-permissions dir so __init__'s mkdir succeeds,
        # then lock the dir so subsequent write/append fails.
        # Use buffer_size=1 so the very first record() call triggers an immediate
        # disk write (rather than buffering), ensuring the fault is actually exercised.
        bad_path = tmp_path / "audit.jsonl"
        sink = sink_cls(bad_path, buffer_size=1)  # type: ignore[call-arg]
        os.chmod(tmp_path, 0o555)
        # Restore write permission BEFORE pytest tears down tmp_path
        # (pytest can't traverse a 0o555 dir to delete it).
        request.addfinalizer(lambda: os.chmod(tmp_path, 0o755))
        return sink
    if name == "OTelEventAuditSink":
        # Pass a fake observability provider whose record_event raises.
        fake_obs = MagicMock()

        async def _raise(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("backend-down")

        fake_obs.record_event = _raise
        return sink_cls(fake_obs)  # type: ignore[call-arg]
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
    sink_cls: type[IAuditSink],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    sink = _construct_sink_with_failing_backend(sink_cls, monkeypatch, tmp_path, request)
    record = _make_test_record()
    # Must NOT raise — Phase 1 contract.
    asyncio.run(sink.record(record))


@pytest.mark.parametrize(
    "sink_cls", _all_concrete_sinks(), ids=lambda c: c.__qualname__
)
def test_audit_sink_flush_never_raises(
    sink_cls: type[IAuditSink],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    sink = _construct_sink_with_failing_backend(sink_cls, monkeypatch, tmp_path, request)
    # Must NOT raise — Phase 1 contract.
    asyncio.run(sink.flush())


def test_at_least_three_concrete_sinks_exist() -> None:
    sinks = _all_concrete_sinks()
    assert len(sinks) >= 3, (
        f"Expected >=3 concrete IAuditSink subclasses, found "
        f"{[c.__qualname__ for c in sinks]}"
    )
