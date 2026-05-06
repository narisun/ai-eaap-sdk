# ai-core-sdk Phase 8 — Design

**Date:** 2026-05-06
**Branch:** `feat/phase-8-stability-bundle`
**Status:** Awaiting user review

## Goal

Stability bundle: ship the `ErrorCode` enum (consolidating ~20 string `error_code` literals scattered across Phases 1-6 into a single `StrEnum`-backed registry), prep for Sentry SDK v3 by verifying our minimal usage works against it and bumping the version pin, and clear the Python 3.16 `asyncio.iscoroutinefunction` deprecation warning. After Phase 8, contributors get IDE-autocomplete + type safety on every typed exception's `error_code`, the SDK is ready for `sentry-sdk>=3.0`, and CI output is one deprecation warning quieter.

## Scope (3 items)

### Stability (3)

1. **`ErrorCode` StrEnum registry.** Add `class ErrorCode(str, enum.Enum)` to `src/ai_core/exceptions.py` enumerating every `error_code` string referenced in production code across Phases 1-6 (see § Component 1 for the full list — 19 members). Each concrete subclass of `EAAPBaseException` updates its `DEFAULT_CODE` to reference an `ErrorCode` member (`DEFAULT_CODE = ErrorCode.CONFIG_INVALID` instead of `DEFAULT_CODE = "config.invalid"`). The string values are unchanged — `StrEnum.value` returns the existing string, so the existing contract tests (`test_exception_default_code_is_dotted_lowercase`, `test_exception_mirrors_error_code_into_details`) keep passing without modification. The 4 codes raised inline at construction sites (`config/settings.py`'s YAML helpers — 2 sites; `audit/sentry.py` + `audit/datadog.py`'s missing-dep checks — 1 site each) update from raw strings to `ErrorCode.<member>`. `ErrorCode` is added to `ai_core.__all__` (29 → 30 names); Phase 7's `EXPECTED_PUBLIC_NAMES` updates from 29 → 30 entries — the deliberate two-place edit the contract test was designed for.

2. **Sentry SDK v3 verify-and-bump.** Install `sentry-sdk==3.x.y` (latest stable 3.x) in the dev venv, run `tests/unit/audit/test_sentry_sink.py` against it, address any breakage in `src/ai_core/audit/sentry.py` (likely 0-5 lines of code change given our minimal `init` / `capture_event` / `flush` surface), bump the optional-dep pin from `sentry-sdk>=1.40,<3.0` to `sentry-sdk>=1.40,<4.0` in `pyproject.toml`. Users on v1.x or v2.x continue to work; users can now opt into v3.x. If v3 introduces a hard incompatibility we can't paper over while keeping v2.x compat, fall back to ship just items 1 and 3 and document the blocker — verify-and-bump is not all-in.

3. **Clear the asyncio deprecation in `schema/registry.py:248`.** Single-line replacement: `asyncio.iscoroutinefunction(func)` → `inspect.iscoroutinefunction(func)`. The `inspect` import is already present in the file (used for `inspect.isasyncgenfunction` on the same line). Drop the `import asyncio` if it's no longer used elsewhere in the file. Existing 5 tests in `tests/unit/schema/test_registry.py` cover this surface and should pass unchanged — `inspect.iscoroutinefunction` is the same function the deprecated `asyncio.iscoroutinefunction` re-exports since Python 3.12.

## Non-goals (deferred to Phase 9+)

- Phase 4 cost/latency closure (Vertex AI Anthropic prefix, 1-hour cache TTL beta, multi-conn MCP pool, pool health-check probe)
- Sink polish (Sentry breadcrumbs / performance tracing, Datadog metrics / DogStatsD / log streaming)
- Redaction polish (custom regex at settings layer, per-sink redaction overrides, Microsoft Presidio / Faker / spaCy backends)
- `ErrorCode` metadata (description, severity, dashboard catalog) — Phase 8 ships the bare enum; metadata can land as a sibling module in a future phase if a real consumer asks for it
- Sentry SDK v3-specific features (e.g., new `Scope.fork()` API) — Phase 8 only verifies our minimal usage compiles against v3; we don't adopt new APIs
- CI workflow updates — host project owns CI configuration

## Constraints

- Pre-1.0; **no backwards-compatibility requirement.** `ErrorCode` is purely additive — existing callers passing raw strings continue to work because `ErrorCode` IS a `str` via StrEnum.
- Per-step gate: `ruff check` no new violations + `mypy <touched files>` strict + `pytest tests/unit tests/component tests/contract` + `pytest tests/integration` (Docker-conditional, auto-skip otherwise).
- Project mypy `src` total stays ≤ 21 (post-Phase-3 baseline).
- Project ruff total stays ≤ 211 (post-Phase-7 baseline at `3c26ca3`).
- End-of-phase smoke (`eaap init`, surface symbols, etc.) must continue to work.
- Sentry SDK pin bump only happens if verification passes; otherwise scope drops to items 1 and 3.

## Module layout

```
src/ai_core/
├── exceptions.py                   # MODIFIED — add ErrorCode StrEnum + update 16 DEFAULT_CODE refs
├── audit/sentry.py                 # POSSIBLY MODIFIED (only if v3 breaks our minimal usage)
├── audit/datadog.py                # MODIFIED (1 line) — config.optional_dep_missing → ErrorCode.CONFIG_OPTIONAL_DEP_MISSING
├── config/settings.py              # MODIFIED (2 lines) — config.yaml_path_missing / config.yaml_parse_failed → ErrorCode.<member>
└── schema/registry.py              # MODIFIED (1 line) — asyncio.iscoroutinefunction → inspect.iscoroutinefunction

src/ai_core/__init__.py             # MODIFIED — re-export ErrorCode (29 → 30 names)

pyproject.toml                      # MODIFIED — bump sentry pin from <3.0 to <4.0 (post-verify)

tests/contract/test_public_surface.py    # MODIFIED — EXPECTED_PUBLIC_NAMES 29 → 30 (add "ErrorCode")
tests/unit/audit/test_sentry_sink.py     # POSSIBLY MODIFIED (only if v3 changes test fixtures' API)
tests/unit/exceptions/test_error_code.py # NEW (~30 LOC, 2 tests asserting enum invariants)
```

### Files NOT touched

- `README.md`
- `src/ai_core/cli/main.py`, `cli/scaffold.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`
- `tests/unit/component/**`
- `tests/integration/**` (Phase 7 territory; Phase 8 doesn't add or modify integration tests)

## Component 1 — `ErrorCode` StrEnum registry

### 1a. Enum definition

In `src/ai_core/exceptions.py`, define `ErrorCode` BEFORE the existing exception classes (must be defined before `class ConfigurationError` references `ErrorCode.CONFIG_INVALID`):

```python
import enum


class ErrorCode(str, enum.Enum):
    """Canonical error codes for typed SDK exceptions.

    Members exhaustively cover every ``error_code`` string referenced by
    production code in Phases 1-6. Values are dotted-lowercase strings;
    ``ErrorCode`` inherits from ``str`` so members are directly comparable
    with raw strings::

        if exc.error_code == ErrorCode.CONFIG_INVALID:
            ...

    Adding a new code requires:
    1. Adding a member here.
    2. Wiring it into the appropriate exception class's ``DEFAULT_CODE``
       OR using ``ErrorCode.<member>`` directly at the construction site.

    The contract test ``test_every_concrete_exception_default_code_is_an_errorcode_member``
    will catch any new exception class that bypasses the enum.
    """

    # Configuration (Phases 1, 5)
    CONFIG_INVALID = "config.invalid"
    CONFIG_SECRET_NOT_RESOLVED = "config.secret_not_resolved"
    CONFIG_YAML_PATH_MISSING = "config.yaml_path_missing"           # Phase 5
    CONFIG_YAML_PARSE_FAILED = "config.yaml_parse_failed"           # Phase 5
    CONFIG_OPTIONAL_DEP_MISSING = "config.optional_dep_missing"     # Phase 6

    # Dependency injection (Phase 1)
    DI_RESOLUTION_FAILED = "di.resolution_failed"

    # Storage / persistence (Phase 1)
    STORAGE_ERROR = "storage.error"
    CHECKPOINT_ERROR = "checkpoint.error"

    # Policy / authorization (Phase 1)
    POLICY_DENIED = "policy.denied"

    # LLM (Phase 1)
    LLM_INVOCATION_ERROR = "llm.invocation_error"
    LLM_TIMEOUT = "llm.timeout"
    BUDGET_EXCEEDED = "budget.exceeded"

    # Schema / validation (Phase 1)
    SCHEMA_VALIDATION_FAILED = "schema.validation_failed"
    TOOL_VALIDATION_FAILED = "tool.validation_failed"

    # Tool execution (Phase 1)
    TOOL_EXECUTION_FAILED = "tool.execution_failed"

    # Agent runtime (Phase 1)
    AGENT_RUNTIME_ERROR = "agent.runtime_error"
    AGENT_RECURSION_LIMIT = "agent.recursion_limit"

    # Registry (Phase 1)
    REGISTRY_ERROR = "registry.error"

    # MCP transport (Phase 1)
    MCP_TRANSPORT_ERROR = "mcp.transport_error"
```

(Total: 19 members. Verified against `git grep "DEFAULT_CODE" src/ai_core/exceptions.py` + `grep "error_code=" src/ai_core/{config,audit}/` to ensure exhaustiveness; the implementation plan's first step refines this list by re-running the grep against the working tree.)

### 1b. Exception class updates

Each concrete `EAAPBaseException` subclass updates its `DEFAULT_CODE` to reference an `ErrorCode` member. Example:

```python
# Before
class ConfigurationError(EAAPBaseException):
    DEFAULT_CODE = "config.invalid"

# After
class ConfigurationError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.CONFIG_INVALID
```

The `DEFAULT_CODE` field annotation on `EAAPBaseException` stays `str` — `ErrorCode` is a `str` subclass via StrEnum, so no annotation change. Tests that compare `DEFAULT_CODE == "config.invalid"` keep passing.

16 concrete classes update (per Phase 7's discovery): ConfigurationError, SecretResolutionError, DependencyResolutionError, StorageError, CheckpointError, PolicyDenialError, LLMInvocationError, LLMTimeoutError, BudgetExceededError, SchemaValidationError, ToolValidationError, ToolExecutionError, AgentRuntimeError, AgentRecursionLimitError, RegistryError, MCPTransportError.

### 1c. Inline `error_code=` call-site updates

Find every `error_code="..."` string literal in production code (NOT in test files) and replace with `ErrorCode.<member>`:

```python
# src/ai_core/config/settings.py
# Before:
raise ConfigurationError(..., error_code="config.yaml_path_missing", ...)
# After:
raise ConfigurationError(..., error_code=ErrorCode.CONFIG_YAML_PATH_MISSING, ...)

# src/ai_core/audit/sentry.py + audit/datadog.py
# Before:
raise ConfigurationError(..., error_code="config.optional_dep_missing", details={"extra": "sentry"})
# After:
raise ConfigurationError(..., error_code=ErrorCode.CONFIG_OPTIONAL_DEP_MISSING, details={"extra": "sentry"})
```

Test files (`tests/unit/audit/test_sentry_sink.py`, `tests/unit/audit/test_datadog_sink.py`, `tests/unit/config/test_yaml_config.py`) keep their string-literal assertions (`assert exc.value.error_code == "config.yaml_path_missing"`) because StrEnum equality with plain strings still works. No test-side changes for these.

### 1d. Public surface update

`src/ai_core/__init__.py`:

```python
from ai_core.exceptions import (
    AgentRecursionLimitError,
    AgentRuntimeError,
    BudgetExceededError,
    ConfigurationError,
    DependencyResolutionError,
    EAAPBaseException,
    ErrorCode,                      # NEW (Phase 8)
    LLMInvocationError,
    ...,
)

__all__ = [
    ...,
    "ErrorCode",                    # NEW (Phase 8)
    ...,
]
```

`tests/contract/test_public_surface.py` — `EXPECTED_PUBLIC_NAMES` updates from 29 → 30 entries (add `"ErrorCode"`). The contract test's existing failure message ("Unexpected new exports") is the deliberate prompt for this two-place edit.

### 1e. New enum-self tests

`tests/unit/exceptions/test_error_code.py` (NEW):

```python
"""Sanity tests for the ErrorCode StrEnum registry."""
from __future__ import annotations

import pytest

from ai_core.exceptions import EAAPBaseException, ErrorCode

pytestmark = pytest.mark.unit


def test_error_code_values_are_unique_and_dotted_lowercase() -> None:
    values = [member.value for member in ErrorCode]
    assert len(set(values)) == len(values), "ErrorCode has duplicate values"
    for value in values:
        assert value, f"ErrorCode member has empty value"
        assert value == value.lower(), f"{value!r} not lowercase"
        assert "." in value, f"{value!r} not dotted"


def _all_concrete_exceptions() -> list[type[EAAPBaseException]]:
    seen: set[type[EAAPBaseException]] = set()
    stack = list(EAAPBaseException.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    return sorted(seen, key=lambda c: c.__qualname__)


@pytest.mark.parametrize(
    "exc_cls", _all_concrete_exceptions(), ids=lambda c: c.__qualname__
)
def test_every_concrete_exception_default_code_is_an_errorcode_member(
    exc_cls: type[EAAPBaseException],
) -> None:
    """A new exception class with an inline string DEFAULT_CODE bypasses the
    enum and won't appear in the catalog. This test catches that drift."""
    valid_values = {member.value for member in ErrorCode}
    assert exc_cls.DEFAULT_CODE in valid_values, (
        f"{exc_cls.__qualname__}.DEFAULT_CODE = {exc_cls.DEFAULT_CODE!r} "
        f"is not in ErrorCode. Add the new code to ErrorCode."
    )
```

(2 test functions; the second is parametrized over ~16 exception classes → ~17 collected items.)

## Component 2 — Sentry SDK v3 verify-and-bump

### 2a. Verification flow

```bash
# Install v3 over current 2.x
/Users/admin-h26/EAAP/.venv/bin/python -m pip install --upgrade "sentry-sdk>=3.0,<4.0"

# Run the existing 9 tests
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_sentry_sink.py -v
```

If green → bump pin. If red → identify breaking change.

### 2b. Likely breaking changes (preventive review)

Based on Sentry SDK v3 release notes (consulted at plan-write time):

- `sentry_sdk.init(...)` signature: most params unchanged; `max_breadcrumbs=0` still works.
- `sentry_sdk.capture_event(event)` signature: still accepts a dict; `level` field still accepts `"info"`/`"warning"`.
- `sentry_sdk.flush(timeout=...)` unchanged.
- `Hub` removed; replaced with `Scope.fork()` — we don't use `Hub`, so no impact.
- The `event_id` return type may have shifted; we don't use the return.

Likely zero code changes. The `# type: ignore[arg-type]` on `capture_event(event)` may need to become `# type: ignore[<different-code>]` if v3's typed stubs evolved.

### 2c. Pin bump

```toml
# pyproject.toml
[project.optional-dependencies]
sentry = ["sentry-sdk>=1.40,<4.0"]   # was <3.0
```

Users can install ai-core-sdk[sentry] with v1.x, v2.x, OR v3.x — broadest compat range.

### 2d. Fallback

If verification fails AND the breakage requires v3-only code paths (i.e., we'd lose v2.x compat to support v3), the task drops the pin bump entirely:

- Document the v3 incompatibility in `audit/sentry.py` as a comment block.
- Leave `pyproject.toml` pin at `<3.0`.
- Open a Phase 9+ follow-up to revisit when v2 is fully sunset.

The fallback is the explicit "we cleanly support v1+v2, but not v3 yet" position — better than half-supporting v3 with conditional code paths.

## Component 3 — `asyncio.iscoroutinefunction` → `inspect.iscoroutinefunction`

### 3a. Single-line replacement

`src/ai_core/schema/registry.py:248`:

```python
# Before
if asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):

# After
if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
```

### 3b. `import asyncio` cleanup

If `asyncio` is no longer referenced anywhere else in `schema/registry.py`, remove `import asyncio` from the imports block. Verify:

```bash
grep -n "asyncio\." /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/schema/registry.py
```

If only the now-replaced line referenced it, drop the import. If anything else uses `asyncio.<something>`, keep it.

### 3c. Verify no test changes needed

`tests/unit/schema/test_registry.py` has 5 tests covering coroutine vs sync function detection (`test_validate_tool_sync_parses_input_validates_output`, `test_validate_tool_async_parses_input_validates_output`, `test_validate_tool_rejects_bad_input`, `test_validate_tool_rejects_bad_output`, `test_validate_tool_passes_through_already_typed_inputs`). All 5 should pass unchanged because `inspect.iscoroutinefunction` and `asyncio.iscoroutinefunction` return identical results for `async def` functions in Python 3.11+ — the deprecated function in `asyncio` is a thin re-export of the `inspect` version since Python 3.12.

## Error handling — consolidated (Phase 8 deltas)

| Path | Behaviour |
|---|---|
| Caller passes raw string `error_code="config.invalid"` (legacy code, host integration) | Works identically — `ErrorCode.CONFIG_INVALID == "config.invalid"` evaluates True via StrEnum. Backwards-compatible. |
| Caller passes `ErrorCode.CONFIG_INVALID` (new code) | Stored on the exception as the StrEnum member; `exc.error_code == "config.invalid"` still True. |
| Caller passes an `ErrorCode` member that doesn't exist (typo) | AttributeError at the call site (`ErrorCode.NONEXISTENT` raises immediately). Caught at edit time by IDE / type checker. |
| New exception class added with inline `DEFAULT_CODE = "some.new_code"` (bypasses enum) | The new contract test `test_every_concrete_exception_default_code_is_an_errorcode_member` fails with a clear error message instructing the contributor to add the code to `ErrorCode`. |
| Sentry SDK v3 changes `capture_event` event-dict shape | Test `test_record_emits_capture_event_with_info_level_for_allowed` fails; implementer adjusts `audit/sentry.py:71-95` minimally. |
| Sentry SDK v3 deletes `sentry_sdk.flush(timeout=...)` | Highly unlikely; if it does, fall back to ship items 1 and 3 only. |
| `inspect.iscoroutinefunction` returns different result than `asyncio.iscoroutinefunction` for some edge case | Highly unlikely; same function since 3.12. Existing 5 tests catch any regression. |

Phase 1-7 invariants preserved.

## Testing strategy

Per-step gate identical to Phase 7. Project mypy `src` total stays ≤ 21. Project ruff total stays ≤ 211.

### Per-step test additions

| Step | Tests |
|---|---|
| 1. ErrorCode registry | 2 new tests in `tests/unit/exceptions/test_error_code.py` (~17 collected after parametrize); 1 update in `tests/contract/test_public_surface.py` |
| 2. Sentry SDK v3 verify-and-bump | Existing 9 sentry tests must pass unchanged against v3 |
| 3. asyncio fix | Existing 5 schema/registry tests must pass unchanged |
| 4. Smoke gate | Full pytest + ruff + mypy + integration (Docker-conditional) |

### Risk register

| Risk | Mitigation |
|---|---|
| ErrorCode enum has missing members | Discovery step in Task 1 runs `git grep "error_code=" src/` before adding members; new contract test catches drift. |
| Sentry SDK v3 has hard incompatibility | Fallback: ship items 1 and 3 only; document v3 blocker. |
| `inspect.iscoroutinefunction` behaves differently in Python 3.11 vs 3.14 | Both are aliases of the same C function since 3.12; no runtime difference. |
| `class ErrorCode(str, enum.Enum)` differs from `enum.StrEnum` (3.11+) | Use `enum.StrEnum` if available (3.11+); the codebase requires 3.11+ per pyproject.toml. Phase 6's `AuditEvent` already uses `StrEnum` — same pattern. |
| `EXPECTED_PUBLIC_NAMES` update adds drift opportunity for Phase 9+ | Single source of truth in the contract test; future contributors update both `__all__` and the set together. |
| Adding `ErrorCode` causes mypy errors on `DEFAULT_CODE` field | `class EAAPBaseException: DEFAULT_CODE: str = "eaap.unknown"` continues to type-check because StrEnum members ARE `str`. Verify with `mypy src` post-change. |
| `import enum` at top of `exceptions.py` shadows or interferes | `enum` is a stdlib module; harmless. Add the import at the top with the other stdlib imports. |

### End-of-phase smoke gate

- `pytest tests/unit tests/component tests/contract -q` green (≥ 466 passing: 464 baseline + ~2 enum tests; the parametrize on the second test adds ~17 collected items so the actual delta is +18-19 items, but most are folded into the existing exception count).
- `pytest tests/integration -q` Docker-conditional.
- `ruff check src tests` total ≤ 211.
- `mypy src --strict` total ≤ 21.
- All 30 canonical names import (29 + `ErrorCode`).
- No `asyncio.iscoroutinefunction` deprecation warning in pytest output.
- `pip install -e ".[sentry]"` succeeds with `sentry-sdk>=3.0,<4.0` available (post-bump).
- `eaap init` regression smoke (Phase 5 invariant) still produces a working scaffold.

### Coverage target

Phase 8 doesn't materially change line coverage. The 2 new enum tests add coverage on `exceptions.py`'s new enum.

## Implementation order (bottom-up)

| Step | Deliverable | Tests | Dependencies |
|---|---|---|---|
| 1 | `ErrorCode` StrEnum + 16 DEFAULT_CODE updates + 4 inline call-site updates + `__all__` update + contract test update + 2 new enum-self tests | ~2 new tests; 1 contract update | none |
| 2 | Sentry SDK v3 verify-and-bump | existing 9 sentry tests must pass against v3 | none (independent) |
| 3 | `asyncio.iscoroutinefunction` → `inspect.iscoroutinefunction` | existing 5 schema/registry tests must pass | none (independent) |
| 4 | End-of-phase smoke gate | full pytest + ruff + mypy + Docker-conditional | all |

Tasks 2 and 3 are fully independent of Task 1 (could run in parallel; sequential here for predictable review cadence).

## Constraints — recap

- 4 implementation steps; per-step gate is ruff (no new) + mypy strict (touched files) + pytest unit/component/contract + Docker-conditional pytest integration.
- Project mypy `src` total stays ≤ 21.
- Project ruff total stays ≤ 211.
- End-of-phase smoke gate must pass before merge.
- Public surface grows by exactly 1 name (`ErrorCode`); 29 → 30.
- Sentry SDK pin bump only happens if verification passes; otherwise scope drops to items 1 and 3.
- No new optional deps. No new top-level subpackages.
