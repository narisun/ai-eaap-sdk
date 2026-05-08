"""Tests for SentryAuditSink."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from ai_core.audit.interface import AuditEvent, AuditRecord
from ai_core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_sentry_sdk(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace sentry_sdk in sys.modules with a MagicMock recording all calls."""
    fake = MagicMock()
    fake.VERSION = "fake-1.0.0"
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    return fake


def _record(
    *,
    event: AuditEvent = AuditEvent.TOOL_INVOCATION_COMPLETED,
    decision_allowed: bool | None = None,
    error_code: str | None = None,
) -> AuditRecord:
    return AuditRecord.now(
        event,
        tool_name="search",
        tool_version=1,
        agent_id="agent-x",
        tenant_id="tenant-y",
        decision_allowed=decision_allowed,
        error_code=error_code,
        payload={"input": {"q": "hello"}},
    )


def test_init_calls_sentry_sdk_init_with_dsn(fake_sentry_sdk: MagicMock) -> None:
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    sink = SentryAuditSink(
        dsn="https://abc@sentry.example.com/42",
        environment="prod",
        release="v1.2.3",
        sample_rate=0.5,
    )
    fake_sentry_sdk.init.assert_called_once()
    kwargs = fake_sentry_sdk.init.call_args.kwargs
    assert kwargs["dsn"] == "https://abc@sentry.example.com/42"
    assert kwargs["environment"] == "prod"
    assert kwargs["release"] == "v1.2.3"
    assert kwargs["sample_rate"] == 0.5
    assert sink is not None


@pytest.mark.asyncio
async def test_record_emits_capture_event_with_info_level_for_allowed(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.record(_record(decision_allowed=True))
    fake_sentry_sdk.capture_event.assert_called_once()
    event = fake_sentry_sdk.capture_event.call_args.args[0]
    assert event["level"] == "info"


@pytest.mark.asyncio
async def test_record_emits_warning_level_for_decision_denied(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.record(_record(decision_allowed=False))
    event = fake_sentry_sdk.capture_event.call_args.args[0]
    assert event["level"] == "warning"


@pytest.mark.asyncio
async def test_record_emits_warning_level_for_error_code_set(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.record(_record(error_code="tool.invocation_failed"))
    event = fake_sentry_sdk.capture_event.call_args.args[0]
    assert event["level"] == "warning"


@pytest.mark.asyncio
async def test_record_includes_audit_tags(fake_sentry_sdk: MagicMock) -> None:
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.record(_record(decision_allowed=True))
    event = fake_sentry_sdk.capture_event.call_args.args[0]
    tags = event["tags"]
    assert tags["audit.tool_name"] == "search"
    assert tags["audit.agent_id"] == "agent-x"
    assert tags["audit.tenant_id"] == "tenant-y"
    assert tags["audit.event"] == AuditEvent.TOOL_INVOCATION_COMPLETED.value


@pytest.mark.asyncio
async def test_record_swallows_exception_from_capture_event(
    fake_sentry_sdk: MagicMock,
) -> None:
    fake_sentry_sdk.capture_event.side_effect = RuntimeError("backend down")
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    sink = SentryAuditSink(dsn="https://x@x/1")
    # Must not raise.
    await sink.record(_record())


@pytest.mark.asyncio
async def test_flush_calls_sentry_sdk_flush_with_timeout(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.flush()
    fake_sentry_sdk.flush.assert_called_once_with(timeout=5.0)


def test_missing_sentry_sdk_raises_configuration_error_with_extra_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If sentry_sdk is not installed, instantiating the sink raises ConfigurationError."""
    # Force the import to fail by removing sentry_sdk from sys.modules
    # AND blocking the import.
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    # Reload the sink module to retrigger the import inside __init__.
    if "ai_core.audit.sentry" in sys.modules:
        del sys.modules["ai_core.audit.sentry"]
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    with pytest.raises(ConfigurationError) as exc:
        SentryAuditSink(dsn="https://x@x/1")
    assert exc.value.error_code == "config.optional_dep_missing"
    assert exc.value.details["extra"] == "sentry"


def test_provide_audit_sink_sentry_without_dsn_raises_configuration_error(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.config.settings import AppSettings  # noqa: PLC0415
    from ai_core.di.interfaces import IObservabilityProvider  # noqa: PLC0415
    from ai_core.di.module import AgentModule  # noqa: PLC0415

    settings = AppSettings()
    settings.audit.sink_type = "sentry"  # type: ignore[assignment]
    # sentry_dsn is None by default

    obs = MagicMock(spec=IObservabilityProvider)
    module = AgentModule()
    with pytest.raises(ConfigurationError) as exc:
        module.provide_audit_sink(settings.audit, obs)
    assert exc.value.error_code == "config.invalid"
    assert "sentry_dsn" in exc.value.message.lower()
