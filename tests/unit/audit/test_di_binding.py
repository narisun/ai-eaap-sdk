"""Tests for AgentModule's audit-sink binding."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_core.audit import IAuditSink, JsonlFileAuditSink, NullAuditSink, OTelEventAuditSink
from ai_core.config.settings import AppSettings, AuditSettings
from ai_core.di import AgentModule, Container
from ai_core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


def test_default_sink_is_null() -> None:
    container = Container.build([AgentModule()])
    sink = container.get(IAuditSink)
    assert isinstance(sink, NullAuditSink)


def test_otel_event_sink_when_configured() -> None:
    settings = AppSettings(audit=AuditSettings(sink_type="otel_event"))
    container = Container.build([AgentModule(settings=settings)])
    sink = container.get(IAuditSink)
    assert isinstance(sink, OTelEventAuditSink)


def test_jsonl_sink_when_configured(tmp_path: Path) -> None:
    settings = AppSettings(audit=AuditSettings(sink_type="jsonl",
                                                jsonl_path=tmp_path / "audit.jsonl"))
    container = Container.build([AgentModule(settings=settings)])
    sink = container.get(IAuditSink)
    assert isinstance(sink, JsonlFileAuditSink)


def test_jsonl_sink_without_path_raises() -> None:
    settings = AppSettings(audit=AuditSettings(sink_type="jsonl", jsonl_path=None))
    container = Container.build([AgentModule(settings=settings)])
    with pytest.raises(ConfigurationError):
        container.get(IAuditSink)
