# ai-core-sdk Phase 5 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship YAML-based config (`./eaap.yaml` auto-discovery; env > yaml > defaults) and refresh the `eaap init` scaffold to expose Phase 4 settings + finalise the dangling `policies/agent.rego` + `policies/api.rego` starter templates.

**Architecture:** Override `AppSettings.settings_customise_sources` (a pydantic-settings hook) to insert a `YamlConfigSettingsSource` between env and defaults. A small pure helper `_resolve_config_path()` handles auto-discover-vs-explicit semantics. Errors wrap as `ConfigurationError` with new dotted-lowercase `error_code` constants. `eaap init` ships a starter `eaap.yaml.j2`, an extended `.env.example`, and the two .rego files (already authored in the working tree, just untracked).

**Tech Stack:** Python 3.11+, Pydantic v2, `pydantic-settings>=2.10` (`YamlConfigSettingsSource`), `pyyaml>=6.0` (new runtime dep — `types-PyYAML` already declared), `pytest` + `ruff` + `mypy --strict`. Spec: `docs/superpowers/specs/2026-05-05-ai-core-sdk-phase-5-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-5-yaml-config-scaffold-dx` (already checked out off `main` post-PR-merge; carries the Phase 5 spec at `6e73051`).

**Working-state hygiene** — do NOT touch:
- `README.md` (top-level)
- `src/ai_core/cli/main.py` (the runner is unchanged; only its templates change)
- `src/ai_core/cli/templates/init/README.md.j2`
- `src/ai_core/cli/templates/init/src/main.py.j2`
- All other `src/` files (Phase 5 is purely DX/config; no behavioural changes elsewhere)

**Mypy baseline:** 21 strict errors in 8 files (post-Phase-4). Total must remain ≤ 21 after every commit.

**Ruff baseline:** 211 errors at `d634b75` (Phase 4 merge commit). Total must remain ≤ 211.

**Per-step gate (every commit):**
- `ruff check <files-touched>` — no NEW violations vs the pre-task ruff state.
- `pytest tests/unit tests/component -q` — must pass (excluding pre-existing `respx`/`aiosqlite` collection errors).
- `mypy <files-touched-by-this-task>` — no new strict errors.
- `mypy src 2>&1 | tail -1` — total ≤ 21.

**Python interpreter:** `/Users/admin-h26/EAAP/.venv/bin/python` (no local venv inside the SDK).

**Per-task commit message convention:** Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `style:`, `build:`).

**Error code convention:** Existing exceptions use dotted-lowercase (`config.invalid`, `llm.timeout`, `di.resolution_failed`). Phase 5 follows: `config.yaml_path_missing` and `config.yaml_parse_failed`.

---

## Task 1 — YAML config loader

Adds runtime YAML support via `pydantic-settings`. Independent of Task 2.

**Files:**
- Modify: `pyproject.toml` (add `pyyaml>=6.0,<7.0` dep; pin `pydantic-settings>=2.10,<3.0`)
- Modify: `src/ai_core/config/settings.py` (add `_resolve_config_path` helper; override `AppSettings.settings_customise_sources`)
- Test: NEW `tests/unit/config/test_yaml_config.py` (~12 tests)

### 1a — Add the runtime dependency

- [ ] **Step 1.1: Add pyyaml + pin pydantic-settings**

In `pyproject.toml`, find the `dependencies = [...]` list and add the line:

```toml
    "pyyaml>=6.0,<7.0",
```

Find the existing `"pydantic-settings>=2.2.0",` line and update to:

```toml
    "pydantic-settings>=2.10,<3.0",
```

(pinning the lower bound to 2.10 ensures `YamlConfigSettingsSource` is available with the constructor signature this plan uses; pinning the upper bound under 3.0 protects against breaking changes in the source-customisation API.)

- [ ] **Step 1.2: Sync the venv**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pip install -e /Users/admin-h26/EAAP/ai-core-sdk 2>&1 | tail -5
```

Expected: a single line "Successfully installed ai-core-sdk-...". `pyyaml` is already in the venv (PyYAML 6.0.3); `pydantic-settings` is at 2.14.0. The pip resolver should be a no-op.

- [ ] **Step 1.3: Smoke import**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import yaml; from pydantic_settings import YamlConfigSettingsSource; print('ok', yaml.__version__)"
```

Expected: `ok 6.0.3`.

### 1b — `_resolve_config_path` helper

- [ ] **Step 1.4: Create the test directory if missing**

```bash
mkdir -p /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/config
[ -f /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/config/__init__.py ] || touch /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/config/__init__.py
```

Expected: directory exists with `__init__.py`. (May already exist from earlier work.)

- [ ] **Step 1.5: Write failing tests for `_resolve_config_path`**

Create `tests/unit/config/test_yaml_config.py`:

```python
"""Tests for the YAML config loader and AppSettings YAML integration."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_core.config.settings import _EAAP_CONFIG_PATH_ENV, _resolve_config_path

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
```

- [ ] **Step 1.6: Run tests to verify they fail with ImportError**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/config/test_yaml_config.py -v 2>&1 | tail -10
```

Expected: ImportError on `_resolve_config_path` / `_EAAP_CONFIG_PATH_ENV` (they don't exist yet in `ai_core.config.settings`).

- [ ] **Step 1.7: Implement `_resolve_config_path`**

In `src/ai_core/config/settings.py`, find the import block at the top (currently around lines 16-28). Add `os` to the standard-library imports. The full top block should now have:

```python
from __future__ import annotations

import enum
import os
from functools import lru_cache
from pathlib import Path  # noqa: TC003
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_core.config.secrets import ISecretManager
from ai_core.config.validation import ValidationContext
from ai_core.exceptions import ConfigurationError
```

Then add the helper between the imports and the existing `class LogLevel` declaration (around line 34, before `class LogLevel(str, enum.Enum):`):

```python
# ---------------------------------------------------------------------------
# YAML config support (Phase 5)
# ---------------------------------------------------------------------------
_EAAP_CONFIG_PATH_ENV = "EAAP_CONFIG_PATH"


def _resolve_config_path() -> Path | None:
    """Resolve the YAML config file location, or ``None`` if no YAML in this run.

    Resolution order:
        1. ``EAAP_CONFIG_PATH`` env var (explicit; missing-file is an error).
        2. ``./eaap.yaml`` in the current working directory (auto-discover; missing is silent).
        3. ``None`` (no YAML configured).

    Raises:
        FileNotFoundError: If ``EAAP_CONFIG_PATH`` is set but points to a
            non-existent file. Explicit configuration must succeed.
    """
    explicit = os.environ.get(_EAAP_CONFIG_PATH_ENV, "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(
                f"{_EAAP_CONFIG_PATH_ENV}={explicit!r} but the file does not exist"
            )
        return path
    auto = Path.cwd() / "eaap.yaml"
    return auto if auto.is_file() else None
```

- [ ] **Step 1.8: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/config/test_yaml_config.py -v 2>&1 | tail -15
```

Expected: 6 passed (all `_resolve_config_path` tests). Other tests (the YAML-loading ones added next) won't exist yet.

### 1c — `AppSettings.settings_customise_sources` override

- [ ] **Step 1.9: Append YAML-loading tests**

Append to `tests/unit/config/test_yaml_config.py`:

```python


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
    # Late import so the module-level lru_cache (if any) doesn't capture stale env state.
    from ai_core.config.settings import AppSettings
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
    from ai_core.config.settings import AppSettings
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
    from ai_core.config.settings import AppSettings
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
    from ai_core.config.settings import AppSettings
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
    from ai_core.config.settings import AppSettings
    from ai_core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError) as exc:
        AppSettings()
    assert exc.value.error_code == "config.yaml_parse_failed"
    assert "eaap.yaml" in exc.value.message or "eaap.yaml" in str(exc.value.details)


def test_app_settings_raises_configuration_error_on_top_level_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """YAML containing a top-level list → ConfigurationError(error_code='config.yaml_parse_failed')."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eaap.yaml").write_text("- 1\n- 2\n- 3\n")
    from ai_core.config.settings import AppSettings
    from ai_core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError) as exc:
        AppSettings()
    assert exc.value.error_code == "config.yaml_parse_failed"


def test_app_settings_explicit_path_missing_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EAAP_CONFIG_PATH=/no/such/file → ConfigurationError(error_code='config.yaml_path_missing')."""
    _isolate_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_EAAP_CONFIG_PATH_ENV, str(tmp_path / "missing.yaml"))
    from ai_core.config.settings import AppSettings
    from ai_core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError) as exc:
        AppSettings()
    assert exc.value.error_code == "config.yaml_path_missing"
```

- [ ] **Step 1.10: Run new tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/config/test_yaml_config.py -v 2>&1 | tail -25
```

Expected: 6 `_resolve_config_path` tests still pass; 7 new YAML-loading tests fail because `AppSettings` doesn't yet integrate YAML — env-only behaviour means YAML values are ignored and no `ConfigurationError` is raised.

- [ ] **Step 1.11: Override `settings_customise_sources` on `AppSettings`**

In `src/ai_core/config/settings.py`, find the existing `from pydantic_settings import BaseSettings, SettingsConfigDict` import line. Update to include `PydanticBaseSettingsSource` and `YamlConfigSettingsSource`:

```python
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
```

Find `class AppSettings(BaseSettings):`. Inside the class body, after the `validate_for_runtime` method (or at the end of the class — anywhere is fine), add the override:

```python
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert YAML between dotenv and file-secret; resolution is left-to-right.

        Order returned (highest precedence first):
            init args > env vars > .env file > eaap.yaml > file-secret > field defaults.

        Auto-discovery and explicit-path semantics live in :func:`_resolve_config_path`.
        Parse / shape errors are wrapped in :class:`ConfigurationError`.
        """
        try:
            yaml_path = _resolve_config_path()
        except FileNotFoundError as exc:
            raise ConfigurationError(
                str(exc),
                error_code="config.yaml_path_missing",
                details={"env_var": _EAAP_CONFIG_PATH_ENV},
                cause=exc,
            ) from exc

        if yaml_path is None:
            return (init_settings, env_settings, dotenv_settings, file_secret_settings)

        try:
            yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)
        except Exception as exc:  # noqa: BLE001 — wrap any parse/shape error uniformly
            raise ConfigurationError(
                f"Failed to load eaap.yaml at {yaml_path}: {exc}",
                error_code="config.yaml_parse_failed",
                details={"path": str(yaml_path)},
                cause=exc,
            ) from exc

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_settings,
            file_secret_settings,
        )
```

- [ ] **Step 1.12: Run all yaml_config tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/config/test_yaml_config.py -v 2>&1 | tail -25
```

Expected: 13 passed.

If `test_app_settings_raises_configuration_error_on_top_level_list` fails because `YamlConfigSettingsSource` doesn't raise at construction time for a top-level list, the `Exception`-wrapping happens later (during `__call__` inside pydantic-settings' source-merge). In that case the wrap surface is `AppSettings()` itself — `pydantic_settings.exceptions.SettingsError` propagates and is NOT a `ConfigurationError`. To handle this, additionally wrap the call site at the top of `AppSettings.__init__` is an option, BUT the simpler fix is to catch the error during `settings_customise_sources` by force-resolving the YAML once eagerly. If the test fails for this reason, update the override:

```python
        try:
            yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)
            # Force eager parse + shape check by invoking the source once.
            yaml_settings()
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(
                f"Failed to load eaap.yaml at {yaml_path}: {exc}",
                error_code="config.yaml_parse_failed",
                details={"path": str(yaml_path)},
                cause=exc,
            ) from exc
```

(Calling the source as a callable forces it to read+parse+validate-shape; subsequent merge-time calls will use the already-parsed result.)

Re-run the test. If still failing, report status `BLOCKED` with the actual exception stacktrace — the controller will provide context. Otherwise proceed.

### 1d — Lint, type-check, commit

- [ ] **Step 1.13: Lint + type-check the Task 1 change set**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/config/settings.py \
    tests/unit/config/test_yaml_config.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/config/settings.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations on these two files; mypy on `settings.py` clean; project total ≤ 21.

- [ ] **Step 1.14: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 360+ passing (348 baseline + 13 new); 9 pre-existing langgraph errors unchanged. Other tests must NOT regress — if anything in `tests/unit/config/` or `tests/component/` that previously passed now fails, the CWD-isolation in the new tests may be leaking. Investigate and fix.

- [ ] **Step 1.15: Commit Task 1**

```bash
git add pyproject.toml \
        src/ai_core/config/settings.py \
        tests/unit/config/__init__.py \
        tests/unit/config/test_yaml_config.py
git commit -m "feat(config): YAML config loader — eaap.yaml auto-discover with EAAP_CONFIG_PATH override"
```

(`tests/unit/config/__init__.py` may already be tracked. If `git status` shows it's missing or new, include it; otherwise the `git add` will silently skip.)

---

## Task 2 — `eaap init` scaffold refresh

Adds `eaap.yaml.j2` template, extends `.env.example` with Phase 4 keys, and finalises the two policy files. Independent of Task 1 (could be done in parallel; sequential here for predictable review).

**Files:**
- Create: `src/ai_core/cli/templates/init/eaap.yaml.j2`
- Modify: `src/ai_core/cli/templates/init/_DOT_env.example`
- Track (already exist as untracked): `src/ai_core/cli/templates/init/policies/agent.rego`
- Track (already exist as untracked): `src/ai_core/cli/templates/init/policies/api.rego`
- Modify: `tests/unit/cli/test_main.py` (extend with 3 new assertions)

### 2a — Add the `eaap.yaml.j2` template + tests

- [ ] **Step 2.1: Write a failing test for `eaap.yaml` rendering**

Append to `tests/unit/cli/test_main.py` (after the existing `test_init_*` tests, before the `# eaap generate` section):

```python
def test_init_renders_eaap_yaml(runner: CliRunner, tmp_path: Path) -> None:
    """`eaap init NAME` renders an eaap.yaml with commented Phase 4 keys."""
    result = runner.invoke(app, ["init", "my-app", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    yaml_path = tmp_path / "my-app" / "eaap.yaml"
    assert yaml_path.is_file()
    content = yaml_path.read_text()
    # Commented top-level groups demonstrating Phase 4 settings.
    assert "# llm:" in content
    assert "# mcp:" in content
    assert "# security:" in content
    # Inline comment explaining precedence.
    assert "env vars" in content.lower()


def test_init_env_example_includes_phase4_keys(runner: CliRunner, tmp_path: Path) -> None:
    """`.env.example` exposes Phase 4 env vars (commented)."""
    runner.invoke(app, ["init", "my-app", "--path", str(tmp_path)])
    env_example = (tmp_path / "my-app" / ".env.example").read_text()
    assert "EAAP_LLM__PROMPT_CACHE_ENABLED" in env_example
    assert "EAAP_MCP__POOL_ENABLED" in env_example
    assert "EAAP_SECURITY__OPA_HEALTH_PATH" in env_example
    assert "EAAP_CONFIG_PATH" in env_example


def test_init_renders_starter_policies_with_content(
    runner: CliRunner, tmp_path: Path
) -> None:
    """`policies/agent.rego` and `policies/api.rego` are rendered with non-trivial content."""
    runner.invoke(app, ["init", "my-app", "--path", str(tmp_path)])
    agent = (tmp_path / "my-app" / "policies" / "agent.rego").read_text()
    api = (tmp_path / "my-app" / "policies" / "api.rego").read_text()
    # Both files declare a Rego package and a default-deny rule.
    assert "package eaap.agent" in agent
    assert "default allow := false" in agent
    assert "package eaap.api" in api
    assert "default allow := false" in api
```

- [ ] **Step 2.2: Run the new tests — verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/cli/test_main.py -v -k "test_init_renders_eaap_yaml or test_init_env_example_includes_phase4_keys or test_init_renders_starter_policies_with_content" 2>&1 | tail -20
```

Expected:
- `test_init_renders_eaap_yaml` fails (no `eaap.yaml` in scaffold output).
- `test_init_env_example_includes_phase4_keys` fails (Phase 4 keys not in env example yet).
- `test_init_renders_starter_policies_with_content` may PASS already because the .rego files are in the working tree (untracked but read-by-iter_template_files). If it passes, that's fine — the assertion still verifies the content shape.

- [ ] **Step 2.3: Create `eaap.yaml.j2` template**

Create `src/ai_core/cli/templates/init/eaap.yaml.j2`:

```yaml
# Generated by `eaap init`. Customize per-deployment values here, or set
# corresponding environment variables (see .env.example for the EAAP_* names).
# Precedence: env vars > eaap.yaml > field defaults.
#
# All keys below are commented out — they show the SDK defaults so you can
# uncomment selectively. Top-level keys mirror the AppSettings groups.

# llm:
#   prompt_cache_enabled: true
#   prompt_cache_min_messages: 6
#   prompt_cache_min_tokens: 1024
#   request_timeout_seconds: 30.0

# mcp:
#   pool_enabled: true
#   pool_idle_seconds: 300.0

# security:
#   opa_url: "http://localhost:8181"
#   opa_health_path: "/health"
#   opa_request_timeout_seconds: 5.0
#   fail_closed: true

# health:
#   probe_timeout_seconds: 2.0

# observability:
#   service_name: "{{ project_name }}"
```

(`{{ project_name }}` is the Jinja2 variable provided by `eaap init` — already substituted by the existing renderer.)

- [ ] **Step 2.4: Run the eaap.yaml test — verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/cli/test_main.py::test_init_renders_eaap_yaml -v 2>&1 | tail -10
```

Expected: PASS.

### 2b — Extend `.env.example` with Phase 4 keys

- [ ] **Step 2.5: Append Phase 4 env-var lines to `_DOT_env.example`**

Open `src/ai_core/cli/templates/init/_DOT_env.example` and append (after the last existing line, with a blank line above for readability):

```
# --- Phase 4 settings (uncomment and customize per-deployment) ---

# Anthropic prompt caching (LLMSettings)
# EAAP_LLM__PROMPT_CACHE_ENABLED=true
# EAAP_LLM__PROMPT_CACHE_MIN_MESSAGES=6
# EAAP_LLM__PROMPT_CACHE_MIN_TOKENS=1024

# MCP connection pooling (MCPSettings)
# EAAP_MCP__POOL_ENABLED=true
# EAAP_MCP__POOL_IDLE_SECONDS=300.0

# OPA health probe path (SecuritySettings)
# EAAP_SECURITY__OPA_HEALTH_PATH=/health

# Optional: explicit YAML config path (auto-discovers ./eaap.yaml otherwise)
# EAAP_CONFIG_PATH=/etc/eaap/config.yaml
```

- [ ] **Step 2.6: Run the env-example test — verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/cli/test_main.py::test_init_env_example_includes_phase4_keys -v 2>&1 | tail -10
```

Expected: PASS.

### 2c — Finalise the .rego policies

The two files already exist in the working tree (untracked). They have well-authored content that matches the spec's intent. The only action needed is to commit them — no edits required, unless the existing assertions in the new test fail.

- [ ] **Step 2.7: Verify the existing .rego content matches the test assertions**

```bash
grep -E "^package |^default allow" /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/cli/templates/init/policies/agent.rego /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/cli/templates/init/policies/api.rego
```

Expected output must include lines that satisfy the test assertions:

```
src/ai_core/cli/templates/init/policies/agent.rego:package eaap.agent.tool_call    # contains "package eaap.agent"
src/ai_core/cli/templates/init/policies/agent.rego:default allow := false
src/ai_core/cli/templates/init/policies/api.rego:package eaap.api
src/ai_core/cli/templates/init/policies/api.rego:default allow := false
```

If the lines are present, no edits needed — the test will pass once the files are tracked. If lines are missing, edit the files to satisfy the test assertions:
- `agent.rego` must contain a `package eaap.agent...` line and `default allow := false`.
- `api.rego` must contain `package eaap.api` and `default allow := false`.

(Do NOT replace the file content if it already contains richer logic; the test only checks for the presence of these two lines.)

- [ ] **Step 2.8: Run the policies test — verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/cli/test_main.py::test_init_renders_starter_policies_with_content -v 2>&1 | tail -10
```

Expected: PASS.

### 2d — Lint, type-check, full suite, commit

- [ ] **Step 2.9: Lint + type-check the Task 2 change set**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    tests/unit/cli/test_main.py \
    src/ai_core/cli/templates/init/_DOT_env.example \
    src/ai_core/cli/templates/init/eaap.yaml.j2 2>&1 | tail -5
```

(Ruff doesn't lint `.example` / `.j2` / `.rego` — only `.py`. The above command is a sanity check that the test file is clean.)

Expected: no new violations.

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy tests/unit/cli/test_main.py 2>&1 | tail -3
```

Expected: clean (or no new errors vs baseline; existing test-file mypy state is the reference).

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: ≤ 21.

- [ ] **Step 2.10: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 363+ passing (Task 1's 13 new tests + Task 2's 3 new = 16 total new); 9 pre-existing langgraph errors unchanged.

- [ ] **Step 2.11: Commit Task 2**

```bash
git add src/ai_core/cli/templates/init/eaap.yaml.j2 \
        src/ai_core/cli/templates/init/_DOT_env.example \
        src/ai_core/cli/templates/init/policies/agent.rego \
        src/ai_core/cli/templates/init/policies/api.rego \
        tests/unit/cli/test_main.py
git commit -m "feat(cli): eaap init scaffold refresh — eaap.yaml.j2, .env.example Phase 4 keys, starter policies"
```

---

## Task 3 — End-of-phase smoke gate

Verification only. No code changes.

- [ ] **Step 3.1: Full test suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Report total passes / fails / errors. Identify any new failures (not pre-existing).

Expected: 363+ passing, 9 pre-existing errors unchanged.

- [ ] **Step 3.2: Lint the entire tree**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src tests 2>&1 | tail -3
```

Expected: 211 errors total (= post-Phase-4 baseline `d634b75`). No NEW categories.

- [ ] **Step 3.3: Mypy strict**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `Found 21 errors in 8 files (checked 62 source files)`.

- [ ] **Step 3.4: `eaap init` smoke**

```bash
SMOKE=$(mktemp -d)
/Users/admin-h26/EAAP/.venv/bin/python -m ai_core.cli.main init smoke-app --path "$SMOKE"
ls -la "$SMOKE/smoke-app"
echo "---eaap.yaml---"
cat "$SMOKE/smoke-app/eaap.yaml" | head -10
echo "---.env.example tail---"
tail -20 "$SMOKE/smoke-app/.env.example"
echo "---agent.rego head---"
head -8 "$SMOKE/smoke-app/policies/agent.rego"
echo "---api.rego head---"
head -8 "$SMOKE/smoke-app/policies/api.rego"
```

Expected: directory listing shows `eaap.yaml`, `policies/agent.rego`, `policies/api.rego`, `.env.example`. The eaap.yaml head shows the `# llm:` block. The .env.example tail shows Phase 4 keys. Both .rego files have `package eaap...` and `default allow := false`.

- [ ] **Step 3.5: YAML loading smoke from a generated project**

```bash
cd "$SMOKE/smoke-app"
/Users/admin-h26/EAAP/.venv/bin/python -c "
from ai_core.config.settings import AppSettings
s = AppSettings()
print('llm.prompt_cache_enabled =', s.llm.prompt_cache_enabled)
print('mcp.pool_enabled         =', s.mcp.pool_enabled)
print('security.opa_health_path =', s.security.opa_health_path)
print('OK — yaml present (commented), defaults loaded')
"
cd -
```

Expected: prints the three default values (`True True /health`) and the OK line. No exception.

Then test that uncommenting a YAML key actually takes effect:

```bash
cd "$SMOKE/smoke-app"
# Uncomment the mcp.pool_enabled line in eaap.yaml.
sed -i.bak 's/^# mcp:/mcp:/' eaap.yaml
sed -i.bak 's/^#   pool_enabled: true/  pool_enabled: false/' eaap.yaml
/Users/admin-h26/EAAP/.venv/bin/python -c "
from ai_core.config.settings import AppSettings
s = AppSettings()
assert s.mcp.pool_enabled is False, f'expected False, got {s.mcp.pool_enabled}'
print('OK — yaml override took effect')
"
cd -
```

Expected: prints `OK — yaml override took effect`. No exception.

- [ ] **Step 3.6: Verify Phase 5 surface symbols**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
from ai_core.config.settings import (
    AppSettings, _resolve_config_path, _EAAP_CONFIG_PATH_ENV
)
# Helper exists.
assert callable(_resolve_config_path)
# Constant exists.
assert _EAAP_CONFIG_PATH_ENV == 'EAAP_CONFIG_PATH'
# Override exists on the class.
assert hasattr(AppSettings, 'settings_customise_sources')
print('Phase 5 symbols OK')
"
```

Expected: `Phase 5 symbols OK`.

- [ ] **Step 3.7: Capture phase summary**

```bash
git log --oneline 6e73051..HEAD
```

Expected: 2 conventional-commit subjects (one per Task).

- [ ] **Step 3.8: Do NOT push automatically**

```bash
git status
echo ""
echo "Suggested next step:"
echo "git push origin feat/phase-5-yaml-config-scaffold-dx"
echo "gh pr create --title 'feat: Phase 5 — YAML config + eaap init scaffold refresh'"
```

---

## Out-of-scope reminders

For traceability, deferred to Phase 6+:

- Contract tests for the public 28-name surface
- Testcontainers replacement of component-test fakes (Postgres / OPA)
- README.md.j2 / src/main.py.j2 template refresh
- 1-hour cache TTL beta (`cache_control: {ttl: "1h"}`)
- Vertex AI Anthropic prefix recognition (`vertex_ai/claude-...`)
- Multi-connection MCP pool (`max_connections > 1`)
- Pool health-check probe
- `error_code` lookup registry / constants module (Phase 5 adds 2 new codes inline; full registry is Phase 6+)
- Concrete `PayloadRedactor` implementations (PII strippers)
- Sentry / Datadog audit-sink reference implementations

If a step starts pulling work from this list, stop and confirm scope with the user.
