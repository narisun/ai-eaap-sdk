"""Tests for DatadogAuditSink."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from ai_core.audit.interface import AuditEvent, AuditRecord
from ai_core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_datadog(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace `datadog` in sys.modules with a MagicMock."""
    fake = MagicMock()
    fake.__version__ = "fake-0.50.0"
    fake.api = MagicMock()
    fake.api.Event = MagicMock()
    monkeypatch.setitem(sys.modules, "datadog", fake)
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


def test_init_calls_datadog_initialize_with_api_key_and_site(
    fake_datadog: MagicMock,
) -> None:
    from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
    sink = DatadogAuditSink(
        api_key="dd-api-1",
        app_key="dd-app-1",
        site="datadoghq.eu",
        source="my-app",
        environment="prod",
    )
    fake_datadog.initialize.assert_called_once()
    kwargs = fake_datadog.initialize.call_args.kwargs
    assert kwargs["api_key"] == "dd-api-1"
    assert kwargs["app_key"] == "dd-app-1"
    assert kwargs["api_host"] == "https://api.datadoghq.eu"
    assert sink is not None


@pytest.mark.asyncio
async def test_record_emits_event_create_with_info_alert_type(
    fake_datadog: MagicMock,
) -> None:
    from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
    sink = DatadogAuditSink(api_key="k")
    await sink.record(_record(decision_allowed=True))
    fake_datadog.api.Event.create.assert_called_once()
    kwargs = fake_datadog.api.Event.create.call_args.kwargs
    assert kwargs["alert_type"] == "info"


@pytest.mark.asyncio
async def test_record_emits_warning_alert_type_for_denied(
    fake_datadog: MagicMock,
) -> None:
    from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
    sink = DatadogAuditSink(api_key="k")
    await sink.record(_record(decision_allowed=False))
    kwargs = fake_datadog.api.Event.create.call_args.kwargs
    assert kwargs["alert_type"] == "warning"


@pytest.mark.asyncio
async def test_record_includes_event_tags(fake_datadog: MagicMock) -> None:
    from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
    sink = DatadogAuditSink(api_key="k", environment="prod")
    await sink.record(_record(decision_allowed=True))
    kwargs = fake_datadog.api.Event.create.call_args.kwargs
    tags = kwargs["tags"]
    # All expected tags present.
    assert any(t.startswith("event:") for t in tags)
    assert any(t.startswith("tool_name:search") for t in tags)
    assert any(t.startswith("agent_id:agent-x") for t in tags)
    assert any(t.startswith("tenant_id:tenant-y") for t in tags)
    assert any(t.startswith("env:prod") for t in tags)


@pytest.mark.asyncio
async def test_record_swallows_exception_from_event_create(
    fake_datadog: MagicMock,
) -> None:
    fake_datadog.api.Event.create.side_effect = RuntimeError("backend down")
    from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
    sink = DatadogAuditSink(api_key="k")
    await sink.record(_record())  # must not raise


@pytest.mark.asyncio
async def test_flush_is_noop(fake_datadog: MagicMock) -> None:
    from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
    sink = DatadogAuditSink(api_key="k")
    await sink.flush()
    # Datadog has no flush API; the sink's flush is a no-op.
    fake_datadog.api.Event.create.assert_not_called()


def test_missing_datadog_raises_configuration_error_with_extra_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "datadog", None)
    if "ai_core.audit.datadog" in sys.modules:
        del sys.modules["ai_core.audit.datadog"]
    from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
    with pytest.raises(ConfigurationError) as exc:
        DatadogAuditSink(api_key="k")
    assert exc.value.error_code == "config.optional_dep_missing"
    assert exc.value.details["extra"] == "datadog"


def test_provide_audit_sink_datadog_without_api_key_raises_configuration_error(
    fake_datadog: MagicMock,
) -> None:
    from ai_core.config.settings import AppSettings  # noqa: PLC0415
    from ai_core.di.interfaces import IObservabilityProvider  # noqa: PLC0415
    from ai_core.di.module import AgentModule  # noqa: PLC0415

    settings = AppSettings()
    settings.audit.sink_type = "datadog"  # type: ignore[assignment]
    # datadog_api_key is None by default

    obs = MagicMock(spec=IObservabilityProvider)
    module = AgentModule()
    with pytest.raises(ConfigurationError) as exc:
        module.provide_audit_sink(settings.audit, obs)
    assert exc.value.error_code == "config.invalid"
    assert "datadog_api_key" in exc.value.message.lower()
