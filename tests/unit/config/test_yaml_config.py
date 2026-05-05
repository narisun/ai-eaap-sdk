"""Tests for the YAML config loader and AppSettings YAML integration."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from ai_core.config.settings import _EAAP_CONFIG_PATH_ENV, AppSettings, _resolve_config_path
from ai_core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _resolve_config_path
# ---------------------------------------------------------------------------
def test_resolve_config_path_returns_none_when_nothing_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No env var, no eaap.yaml in CWD → None."""
    monkeypatch.delenv(_EAAP_CONFIG_PATH_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert _resolve_config_path() is None


def test_resolve_config_path_returns_cwd_eaap_yaml_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """eaap.yaml exists in CWD → returns it."""
    monkeypatch.delenv(_EAAP_CONFIG_PATH_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    yaml_path = tmp_path / "eaap.yaml"
    yaml_path.write_text("# empty\n")
    result = _resolve_config_path()
    assert result == yaml_path


def test_resolve_config_path_honours_explicit_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EAAP_CONFIG_PATH=/some/file.yaml (exists) → returns that path."""
    explicit = tmp_path / "custom.yaml"
    explicit.write_text("# empty\n")
    monkeypatch.setenv(_EAAP_CONFIG_PATH_ENV, str(explicit))
    monkeypatch.chdir(tmp_path)
    assert _resolve_config_path() == explicit


def test_resolve_config_path_explicit_overrides_auto_discover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both EAAP_CONFIG_PATH AND eaap.yaml exist → explicit wins."""
    explicit = tmp_path / "custom.yaml"
    explicit.write_text("# explicit\n")
    auto = tmp_path / "eaap.yaml"
    auto.write_text("# auto\n")
    monkeypatch.setenv(_EAAP_CONFIG_PATH_ENV, str(explicit))
    monkeypatch.chdir(tmp_path)
    assert _resolve_config_path() == explicit


def test_resolve_config_path_raises_for_missing_explicit_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EAAP_CONFIG_PATH=/no/such/file → FileNotFoundError."""
    missing = tmp_path / "does_not_exist.yaml"
    monkeypatch.setenv(_EAAP_CONFIG_PATH_ENV, str(missing))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        _resolve_config_path()
    assert _EAAP_CONFIG_PATH_ENV in str(exc.value)
    assert str(missing) in str(exc.value)


def test_resolve_config_path_treats_empty_env_var_as_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EAAP_CONFIG_PATH='' → falls through to auto-discover."""
    monkeypatch.setenv(_EAAP_CONFIG_PATH_ENV, "")
    monkeypatch.chdir(tmp_path)
    # No eaap.yaml in tmp_path → None
    assert _resolve_config_path() is None
    # Now create one and re-check.
    (tmp_path / "eaap.yaml").write_text("# empty\n")
    assert _resolve_config_path() == tmp_path / "eaap.yaml"


# ---------------------------------------------------------------------------
# AppSettings YAML integration
# ---------------------------------------------------------------------------
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every EAAP_* env var so tests start from a clean slate."""
    for name in list(os.environ):
        if name.startswith("EAAP_"):
            monkeypatch.delenv(name, raising=False)


def test_app_settings_loads_yaml_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """YAML llm.prompt_cache_min_messages: 10 → settings.llm.prompt_cache_min_messages == 10."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eaap.yaml").write_text(
        "llm:\n"
        "  prompt_cache_min_messages: 10\n"
    )
    settings = AppSettings()
    assert settings.llm.prompt_cache_min_messages == 10


def test_app_settings_env_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """YAML says 10, env says 20 → 20 wins (env > yaml)."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eaap.yaml").write_text(
        "llm:\n"
        "  prompt_cache_min_messages: 10\n"
    )
    monkeypatch.setenv("EAAP_LLM__PROMPT_CACHE_MIN_MESSAGES", "20")
    settings = AppSettings()
    assert settings.llm.prompt_cache_min_messages == 20


def test_app_settings_yaml_overrides_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """YAML key set → YAML value used; YAML key omitted → field default."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eaap.yaml").write_text(
        "mcp:\n"
        "  pool_enabled: false\n"
    )
    settings = AppSettings()
    # Set in YAML.
    assert settings.mcp.pool_enabled is False
    # Default (300.0) — YAML didn't set it.
    assert settings.mcp.pool_idle_seconds == 300.0


def test_app_settings_handles_empty_yaml_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty eaap.yaml → settings load from defaults; no error."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eaap.yaml").write_text("# nothing but a comment\n")
    settings = AppSettings()
    # Defaults intact.
    assert settings.mcp.pool_enabled is True
    assert settings.llm.prompt_cache_enabled is True


def test_app_settings_raises_configuration_error_on_malformed_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """YAML with syntax error → ConfigurationError(error_code='config.yaml_parse_failed')."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eaap.yaml").write_text("llm:\n  prompt_cache_min_messages: [oops\n")
    with pytest.raises(ConfigurationError) as exc:
        AppSettings()
    assert exc.value.error_code == "config.yaml_parse_failed"
    assert "eaap.yaml" in exc.value.message or "eaap.yaml" in str(exc.value.details)


def test_app_settings_raises_configuration_error_on_top_level_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """YAML top-level list → ConfigurationError(error_code='config.yaml_parse_failed')."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eaap.yaml").write_text("- 1\n- 2\n- 3\n")
    with pytest.raises(ConfigurationError) as exc:
        AppSettings()
    assert exc.value.error_code == "config.yaml_parse_failed"


def test_app_settings_explicit_path_missing_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EAAP_CONFIG_PATH pointing to missing file → ConfigurationError."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_EAAP_CONFIG_PATH_ENV, str(tmp_path / "missing.yaml"))
    with pytest.raises(ConfigurationError) as exc:
        AppSettings()
    assert exc.value.error_code == "config.yaml_path_missing"
    assert exc.value.details["env_var"] == "EAAP_CONFIG_PATH"
