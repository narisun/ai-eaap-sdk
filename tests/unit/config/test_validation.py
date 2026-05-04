"""Unit tests for the runtime-config validation helpers."""
from __future__ import annotations

import pytest

from ai_core.config.settings import AgentSettings, AppSettings, LLMSettings
from ai_core.config.validation import ConfigIssue, ValidationContext
from ai_core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


def test_config_issue_is_frozen() -> None:
    issue = ConfigIssue(path="x.y", message="m", hint="h")
    with pytest.raises(AttributeError):
        issue.path = "z"  # type: ignore[misc]
    assert issue.path == "x.y"
    assert issue.message == "m"
    assert issue.hint == "h"


def test_validation_context_collects_issues() -> None:
    ctx = ValidationContext()
    assert not ctx.has_issues
    ctx.fail("a.b", "broken", hint="fix it")
    ctx.fail("c.d", "also broken")
    assert ctx.has_issues
    assert len(ctx.issues) == 2  # type: ignore[unreachable]
    assert ctx.issues[0] == ConfigIssue(path="a.b", message="broken", hint="fix it")
    assert ctx.issues[1] == ConfigIssue(path="c.d", message="also broken", hint=None)


def test_validation_context_starts_empty() -> None:
    assert ValidationContext().issues == []


# ---------------------------------------------------------------------------
# AppSettings.validate_for_runtime
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> AppSettings:
    """Build an AppSettings with targeted overrides; defaults are otherwise valid."""
    return AppSettings(**overrides)  # type: ignore[arg-type]


def test_validate_passes_on_default_settings(fake_secret_manager_factory: object) -> None:
    s = AppSettings()
    s.validate_for_runtime(secret_manager=fake_secret_manager_factory({}))  # type: ignore[operator]


def test_validate_rejects_empty_default_model() -> None:
    s = AppSettings(llm=LLMSettings(default_model=""))
    with pytest.raises(ConfigurationError) as exc:
        s.validate_for_runtime()
    issues = exc.value.details["issues"]
    assert len(issues) == 1
    assert issues[0]["path"] == "llm.default_model"
    assert "non-empty" in issues[0]["message"]
    assert issues[0]["hint"] is not None
    assert "EAAP_LLM__DEFAULT_MODEL" in issues[0]["hint"]


def test_validate_rejects_blank_fallback_model() -> None:
    s = AppSettings(llm=LLMSettings(default_model="m", fallback_models=["good", "  "]))
    with pytest.raises(ConfigurationError) as exc:
        s.validate_for_runtime()
    issues = exc.value.details["issues"]
    assert any(i["path"].startswith("llm.fallback_models[") for i in issues)


def test_validate_rejects_compaction_target_above_threshold() -> None:
    s = AppSettings(
        agent=AgentSettings(
            memory_compaction_token_threshold=1000,
            memory_compaction_target_tokens=2000,
        )
    )
    with pytest.raises(ConfigurationError) as exc:
        s.validate_for_runtime()
    issues = exc.value.details["issues"]
    assert any(
        i["path"] == "agent.memory_compaction_target_tokens"
        and "less than" in i["message"]
        for i in issues
    )


def test_validate_collects_all_issues() -> None:
    s = AppSettings(
        llm=LLMSettings(default_model="", fallback_models=[""]),
        agent=AgentSettings(
            memory_compaction_token_threshold=512,
            memory_compaction_target_tokens=10000,
        ),
    )
    with pytest.raises(ConfigurationError) as exc:
        s.validate_for_runtime()
    paths = sorted(i["path"] for i in exc.value.details["issues"])
    assert paths == [
        "agent.memory_compaction_target_tokens",
        "llm.default_model",
        "llm.fallback_models[0]",
    ]
    assert "3 issue(s)" in exc.value.message


def test_validate_rejects_non_isecretmanager() -> None:
    s = AppSettings()
    with pytest.raises(ConfigurationError) as exc:
        s.validate_for_runtime(secret_manager="not-a-manager")  # type: ignore[arg-type]
    issues = exc.value.details["issues"]
    assert any(i["path"] == "secret_manager" for i in issues)
