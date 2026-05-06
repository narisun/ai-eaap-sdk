"""Tests for AgentModule.provide_payload_redactor."""
from __future__ import annotations

import pytest

from ai_core.audit.interface import _identity_redactor
from ai_core.audit.redaction import ChainRedactor
from ai_core.config.settings import AppSettings
from ai_core.di.module import AgentModule

pytestmark = pytest.mark.unit


def _make_settings(profile: str) -> AppSettings:
    settings = AppSettings()
    # Pydantic v2 frozen-ish models still allow attribute assignment on nested settings.
    settings.audit.redaction_profile = profile  # type: ignore[assignment]
    return settings


def test_provide_payload_redactor_off_returns_identity() -> None:
    settings = _make_settings("off")
    module = AgentModule()
    redactor = module.provide_payload_redactor(settings)
    assert redactor is _identity_redactor


def test_provide_payload_redactor_standard_returns_chain() -> None:
    settings = _make_settings("standard")
    module = AgentModule()
    redactor = module.provide_payload_redactor(settings)
    assert isinstance(redactor, ChainRedactor)
    # Behavior check: redacts both an email AND a password key.
    out = redactor({"contact": "alice@x.io", "password": "hunter2"})
    assert out["contact"] == "<redacted-email>"
    assert out["password"] == "[REDACTED]"


def test_provide_payload_redactor_strict_includes_long_number() -> None:
    settings = _make_settings("strict")
    module = AgentModule()
    redactor = module.provide_payload_redactor(settings)
    # 8-digit number gets redacted under strict profile.
    out = redactor({"id": "see id 12345678 attached"})
    assert "<redacted-long_number>" in out["id"]
