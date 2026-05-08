"""Smoke tests for :mod:`ai_core.config.settings`."""

from __future__ import annotations

import pytest

from ai_core.config.settings import (
    AppSettings,
    Environment,
    LogLevel,
)

pytestmark = pytest.mark.unit


def test_defaults_are_sensible() -> None:
    settings = AppSettings()

    assert settings.environment is Environment.LOCAL
    assert settings.service_name == "ai-eaap-sdk"
    assert settings.database.pool_size == 10
    assert settings.llm.max_retries == 3
    assert settings.observability.log_level is LogLevel.INFO
    assert settings.security.fail_closed is True
    assert settings.budget.enabled is True


def test_env_vars_with_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EAAP_ENVIRONMENT", "staging")
    monkeypatch.setenv("EAAP_DATABASE__POOL_SIZE", "42")
    monkeypatch.setenv("EAAP_LLM__DEFAULT_MODEL", "bedrock/anthropic.claude-opus-4-7")
    monkeypatch.setenv("EAAP_SECURITY__FAIL_CLOSED", "false")

    settings = AppSettings()

    assert settings.environment is Environment.STAGING
    assert settings.database.pool_size == 42
    assert settings.llm.default_model == "bedrock/anthropic.claude-opus-4-7"
    assert settings.security.fail_closed is False


def test_service_name_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AppSettings(service_name="   ")


def test_appsettings_is_freshly_constructed() -> None:
    """No process-wide cache: each AppSettings() returns a new instance.

    The DI container is the only intended sharing seam (one container =>
    one bound singleton). This test exists to pin the v1 contract that
    ``get_settings()`` is gone and direct construction is the path.
    """
    a = AppSettings()
    b = AppSettings()
    assert a is not b
    # Equal value (both built from the same env / defaults), but not identity-shared.
    assert a.model_dump() == b.model_dump()


def test_is_production_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EAAP_ENVIRONMENT", "prod")
    settings = AppSettings()
    assert settings.is_production() is True
