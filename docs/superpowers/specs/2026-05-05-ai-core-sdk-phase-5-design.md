# ai-core-sdk Phase 5 — Design

**Date:** 2026-05-05
**Branch:** `feat/phase-5-yaml-config-scaffold-dx`
**Status:** Awaiting user review

## Goal

DX layer: ship YAML-based config and refresh `eaap init` scaffolding so SDK consumers can configure the runtime via `eaap.yaml` (or env vars, or both) and start a new project with a working scaffold that exposes Phase 4 features. After Phase 5, `eaap init` produces a project with `eaap.yaml`, an updated `.env.example`, and committed starter Rego policies; runtime config resolves env > yaml > defaults transparently.

## Scope (2 items)

### DX (2)

1. **YAML-based config layered onto Pydantic Settings.** `AppSettings` overrides `settings_customise_sources` to insert a `YamlConfigSettingsSource` between env and field defaults. File auto-discovers at `./eaap.yaml`; `EAAP_CONFIG_PATH` env var overrides. Precedence: env > yaml > defaults. Nested-mirror YAML structure (top-level keys map to settings groups). New helper `_resolve_config_path() -> Path | None` handles the discover/explicit/missing logic with explicit-path-missing failing loud and auto-discover-missing falling through silently. New runtime dependency `pyyaml>=6.0` added (currently only `types-PyYAML` was declared; the runtime stub is used transitively but not pinned). Errors wrapped in the existing `ConfigurationError` exception with new `error_code` constants `EAAP_YAML_PARSE_FAILED` and `EAAP_YAML_SHAPE_INVALID`.

2. **`eaap init` scaffold refresh.** New starter `eaap.yaml.j2` template demonstrating the Phase 4 settings (commented out, with explanatory inline comments). `_DOT_env.example` extended with Phase 4 env vars (`EAAP_MCP__POOL_ENABLED`, `EAAP_MCP__POOL_IDLE_SECONDS`, `EAAP_LLM__PROMPT_CACHE_ENABLED`, `EAAP_LLM__PROMPT_CACHE_MIN_MESSAGES`, `EAAP_LLM__PROMPT_CACHE_MIN_TOKENS`, `EAAP_SECURITY__OPA_HEALTH_PATH`). The two existing untracked `policies/agent.rego` + `policies/api.rego` files (held over from Phase 1's scaffolding) are finalised as committed starter policies with deny-by-default semantics and a placeholder allow rule keyed on `input.principal.groups`.

## Non-goals (deferred to Phase 6+)

- Contract tests for the public 28-name surface
- Testcontainers replacement of component-test fakes (Postgres / OPA)
- README.md.j2 / src/main.py.j2 template refresh
- 1-hour cache TTL beta
- Vertex AI Anthropic prefix recognition
- Multi-connection MCP pool
- Pool health-check probe
- `error_code` lookup registry / constants module (we add 2 new constants ad-hoc in Phase 5; full registry is Phase 6+)
- Concrete `PayloadRedactor` implementations
- Sentry / Datadog audit-sink reference implementations

## Constraints

- Pre-1.0; **no backwards-compatibility requirement.** Adding YAML support is purely additive — apps that don't ship an `eaap.yaml` keep the env-only behaviour.
- Per-step gate: `ruff check` no new violations + `mypy <touched files>` strict + `pytest tests/unit tests/component`.
- Project mypy total stays ≤ 21 (post-Phase-4 baseline).
- Project ruff total stays ≤ 211 (post-Phase-4 baseline at `d634b75`).
- End-of-phase smoke against `my-eaap-app` and the canonical 28-name top-level surface must continue to work.

## Module layout

```
src/ai_core/
├── config/settings.py              # MODIFIED — settings_customise_sources hook + _resolve_config_path()
├── exceptions.py                   # MODIFIED (1 line) — new error_code constants are referenced; the ConfigurationError class itself is unchanged
│
├── cli/templates/init/
│   ├── eaap.yaml.j2                # NEW — starter YAML template with commented Phase 4 keys
│   ├── _DOT_env.example            # MODIFIED — Phase 4 env vars
│   └── policies/
│       ├── agent.rego              # COMMITTED (was untracked) — starter agent policy
│       └── api.rego                # COMMITTED (was untracked) — starter API policy
│
└── (no other src changes)

pyproject.toml                      # MODIFIED — add `pyyaml>=6.0` to runtime dependencies

tests/
├── unit/config/test_yaml_config.py # NEW (~12 tests)
└── unit/cli/test_main.py           # MODIFIED — assert eaap.yaml is rendered + .env.example contains Phase 4 keys + policies render non-empty
```

### Files NOT touched

- `README.md` (top-level)
- `src/ai_core/cli/main.py` (the runner is unchanged; only its templates change)
- `src/ai_core/cli/templates/init/README.md.j2` (deferred)
- `src/ai_core/cli/templates/init/src/main.py.j2` (deferred)
- All other src files (Phase 5 is purely DX/config; no behavioural changes)

## Component 1 — YAML config

### 1a. `_resolve_config_path` helper

```python
import os
from pathlib import Path

_EAAP_CONFIG_PATH_ENV = "EAAP_CONFIG_PATH"


def _resolve_config_path() -> Path | None:
    """Resolve the YAML config file location, or None if no YAML in this run.

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

### 1b. `AppSettings.settings_customise_sources` override

```python
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

class AppSettings(BaseSettings):
    """Top-level settings for the ai-core-sdk runtime."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="EAAP_",
        extra="ignore",
    )

    # ... existing fields (llm: LLMSettings, mcp: MCPSettings, ...) ...

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert YAML between env and defaults; resolution is left-to-right.

        Order:
            init args > env vars > .env file > eaap.yaml > field defaults > file-secret.

        Auto-discovery and explicit-path semantics live in :func:`_resolve_config_path`.
        Parse / shape errors are wrapped in ``ConfigurationError``.
        """
        try:
            yaml_path = _resolve_config_path()
        except FileNotFoundError as exc:
            raise ConfigurationError(
                str(exc),
                error_code="EAAP_YAML_PATH_MISSING",
                cause=exc,
            ) from exc

        if yaml_path is None:
            return (init_settings, env_settings, dotenv_settings, file_secret_settings)

        try:
            yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)
        except Exception as exc:  # YAMLError or pydantic-settings shape error
            raise ConfigurationError(
                f"Failed to load eaap.yaml at {yaml_path}: {exc}",
                error_code="EAAP_YAML_PARSE_FAILED",
                details={"path": str(yaml_path)},
                cause=exc,
            ) from exc

        return (init_settings, env_settings, dotenv_settings, yaml_settings, file_secret_settings)
```

### 1c. New error_code constants

`ConfigurationError` already exists from Phase 1; only two new `error_code` string values are added (no class changes). Used as inline string literals in `config/settings.py`:

- `"EAAP_YAML_PATH_MISSING"` — explicit `EAAP_CONFIG_PATH` set but file missing.
- `"EAAP_YAML_PARSE_FAILED"` — anything wrong with the YAML body itself (parse error, top-level shape, encoding). The single error code keeps the handler simple — the underlying cause is preserved via `cause=exc` so the original `yaml.YAMLError` / `TypeError` message is still visible in the chained exception. Differentiating shape-vs-parse adds branching for marginal user value; consumers who care can inspect `exc.__cause__`.

(Phase 6 may collect all SDK error codes into a registry. Phase 5 stays minimal.)

## Component 2 — `eaap init` scaffold refresh

### 2a. `eaap.yaml.j2` template

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
#   max_tokens_per_request: 4096

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

(`{{ project_name }}` is the existing Jinja2 variable provided by `eaap init`; see `cli/main.py`.)

### 2b. `_DOT_env.example` extensions

Append after the existing block:

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

### 2c. `policies/agent.rego` (starter)

```rego
# Starter agent policy — invoked by ToolInvoker at tool call time.
# Customize the allow rules to enforce your tenant/role/scope semantics.
# By default this policy denies any tool invocation.

package eaap.agent

import rego.v1

# Default: deny.
default allow := false
default reason := "no allow rule matched"

# Example allow rule: members of the "admin" group can invoke any tool.
# Replace with your real authorization logic.
allow if {
    "admin" in input.principal.groups
}

reason := "admin group member" if {
    "admin" in input.principal.groups
}
```

### 2d. `policies/api.rego` (starter)

```rego
# Starter API policy — invoked by FastAPI security dependencies on inbound requests.
# Customize per-route to enforce your tenant/role/scope semantics.
# By default this policy denies any request.

package eaap.api

import rego.v1

# Default: deny.
default allow := false
default reason := "no allow rule matched"

# Example allow rule: requests with a non-empty principal.sub are allowed.
# Replace with your real authentication/authorization logic.
allow if {
    input.principal.sub != null
    input.principal.sub != ""
}

reason := "authenticated principal" if {
    input.principal.sub != null
    input.principal.sub != ""
}
```

### 2e. `eaap init` runner — no code changes

`src/ai_core/cli/main.py` already iterates `iter_template_files(INIT_TEMPLATE_PACKAGE)` and renders Jinja2-templated files. Adding `eaap.yaml.j2` and the two `.rego` files to the template package directory is sufficient — they are picked up automatically. The `.j2` extension is stripped by the existing renderer; the `.rego` files don't have `.j2` and are copied verbatim.

## Error handling — consolidated (Phase 5 deltas)

| Path | Behaviour |
|---|---|
| `_resolve_config_path()` — `EAAP_CONFIG_PATH` set, file missing | Raises `FileNotFoundError`. Caught in `settings_customise_sources` and re-raised as `ConfigurationError(error_code="EAAP_YAML_PATH_MISSING")`. |
| `_resolve_config_path()` — auto-discover, no `eaap.yaml` | Returns `None`; YAML source is omitted from the source list. Settings load env + defaults only. |
| `YamlConfigSettingsSource(...)` constructor — yaml parse error | Caught in `settings_customise_sources`, re-raised as `ConfigurationError(error_code="EAAP_YAML_PARSE_FAILED")` with `path` in details. |
| YAML top-level is not a mapping | pydantic-settings' YamlConfigSettingsSource raises `TypeError` (or similar). Caught by the same broad handler as parse errors and re-raised as `ConfigurationError(error_code="EAAP_YAML_PARSE_FAILED")`; original exception is chained via `cause=exc` so the underlying message is preserved. |
| YAML field-value type mismatch | Pydantic's standard `ValidationError` propagates from `AppSettings(...)` instantiation — already produces good messages with field path. Not re-wrapped (the field path is the readable signal). |
| YAML file empty / comments-only | `yaml.safe_load` returns `None`; pydantic-settings treats as empty mapping → no values from YAML; falls through to defaults. |

Phase 1-4 invariants preserved:
- Exception hierarchy + `error_code` field — preserved (we add 3 new constants, no class changes).
- `eaap.error.code` span attribute auto-emission — unaffected (no instrumentation changes).
- Audit-sink / probe never-raises contracts — unaffected.

## Testing strategy

Per-step gate identical to Phases 1-4. Project mypy total stays ≤ 21. Project ruff total stays ≤ 211.

### Per-step test additions

| Step | Tests |
|---|---|
| 1. YAML config | ~12 unit tests in `tests/unit/config/test_yaml_config.py` |
| 2. `eaap init` scaffold | ~3 test additions in `tests/unit/cli/test_main.py` |
| 3. Smoke gate | full pytest + ruff + mypy + my-eaap-app + `eaap init` invocation |

### Test detail — `tests/unit/config/test_yaml_config.py`

| Name | Asserts |
|---|---|
| `test_resolve_config_path_returns_none_when_nothing_exists` | No env var, no `eaap.yaml` in CWD → `None` |
| `test_resolve_config_path_returns_cwd_eaap_yaml_when_present` | `eaap.yaml` exists in tmp_cwd → returns it |
| `test_resolve_config_path_honours_explicit_env_var` | `EAAP_CONFIG_PATH=/some/file.yaml` (exists) → returns that path |
| `test_resolve_config_path_explicit_overrides_auto_discover` | Both `EAAP_CONFIG_PATH` AND `eaap.yaml` exist → explicit wins |
| `test_resolve_config_path_raises_for_missing_explicit_path` | `EAAP_CONFIG_PATH=/no/such/file` → `FileNotFoundError` |
| `test_resolve_config_path_treats_empty_env_var_as_unset` | `EAAP_CONFIG_PATH=""` → falls through to auto-discover |
| `test_app_settings_loads_yaml_values` | YAML `llm.prompt_cache_min_messages: 10` → `settings.llm.prompt_cache_min_messages == 10` |
| `test_app_settings_env_overrides_yaml` | YAML says 10, env says 20 → 20 wins |
| `test_app_settings_yaml_overrides_defaults` | YAML omits a key → field default; YAML sets a key → YAML value |
| `test_app_settings_handles_empty_yaml_file` | Empty `eaap.yaml` → settings load defaults; no error |
| `test_app_settings_raises_configuration_error_on_malformed_yaml` | YAML with syntax error → `ConfigurationError` with `error_code="EAAP_YAML_PARSE_FAILED"` and path in details |
| `test_app_settings_raises_configuration_error_on_top_level_list` | YAML containing `[1, 2, 3]` → `ConfigurationError` with `error_code="EAAP_YAML_PARSE_FAILED"` (catch-all for body-level errors; cause chained) |

### Test detail — `tests/unit/cli/test_main.py` extensions

| Name | Asserts |
|---|---|
| `test_init_renders_eaap_yaml` | After `eaap init NAME` in tmp_dir, `tmp_dir/NAME/eaap.yaml` exists and contains the commented `# llm:` block |
| `test_init_env_example_includes_phase4_keys` | `_DOT_env.example` rendering contains `EAAP_LLM__PROMPT_CACHE_ENABLED` and `EAAP_MCP__POOL_ENABLED` |
| `test_init_renders_starter_policies` | `tmp_dir/NAME/policies/agent.rego` and `policies/api.rego` exist and are non-empty (basic smoke) |

### Risk register

| Risk | Mitigation |
|---|---|
| `pydantic-settings` 3.x changes the `settings_customise_sources` signature | Pin `pydantic-settings>=2.10,<3.0`; add a focused unit test that asserts the source ordering is what we expect. |
| `YamlConfigSettingsSource` constructor changes between minor versions | Same pin; document the required version in the docstring; the unit test for "yaml beats default" will catch breakage. |
| Host app inadvertently picks up a stray `eaap.yaml` from CWD during tests | Document the auto-discover behaviour; tests for the SDK's own suite chdir to tmp_path before instantiating settings. |
| YAML parser dependency drift (PyYAML 7.x?) | Pin `pyyaml>=6.0,<7.0`. |
| `eaap init` rendering breaks because the new template files have `.yaml.j2` extension that the renderer doesn't recognize | Verified — `cli/main.py` strips trailing `.j2` from any template name. The new template uses the existing convention. |

### End-of-phase smoke gate

- Full `pytest tests/unit tests/component` green.
- `ruff check src tests` total ≤ 211 (no new vs `d634b75`).
- `mypy src --strict` total ≤ 21.
- All 28 canonical names import.
- `eaap init test-project --path /tmp/eaap-smoke-XXX` produces a directory containing `eaap.yaml`, `.env.example` with Phase 4 keys, and non-empty `policies/agent.rego` + `policies/api.rego`.
- `cd /tmp/eaap-smoke-XXX && /Users/admin-h26/EAAP/.venv/bin/python -c "from ai_core.config.settings import AppSettings; s = AppSettings(); print(s.llm.prompt_cache_enabled)"` works without error (the `eaap.yaml` is fully commented so this loads defaults; smoke verifies no crash).

### Coverage target

≥85% on new code (`config/settings.py` additions, `_resolve_config_path`). Existing coverage must not regress.

## Implementation order (bottom-up)

| Step | Deliverable | Tests | Dependencies |
|---|---|---|---|
| 1 | YAML config (`_resolve_config_path` + `settings_customise_sources` + pyyaml dep) | ~12 unit tests | none |
| 2 | `eaap init` scaffold refresh (eaap.yaml.j2, .env.example, policies/) | ~3 test additions | none (parallel-safe with step 1) |
| 3 | End-of-phase smoke gate | full pytest + ruff + mypy + `eaap init` smoke | all |

Step 1 and Step 2 are independent — could even run in parallel — but the plan treats them sequentially for predictable review cadence.

## Constraints — recap

- 3 implementation steps; per-step gate is ruff (no new) + mypy strict (touched files) + pytest unit/component.
- Project mypy total stays ≤ 21.
- Project ruff total stays ≤ 211.
- End-of-phase smoke gate must pass before merge.
- New runtime dependency: `pyyaml>=6.0,<7.0`.
- No backwards-compatibility shims (pre-1.0). YAML support is purely additive.
