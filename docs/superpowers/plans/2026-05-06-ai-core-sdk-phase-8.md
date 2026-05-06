# ai-core-sdk Phase 8 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `ErrorCode` StrEnum registry (consolidating every `error_code` string referenced across Phases 1-6 into a single enum), prep for Sentry SDK v3 by verifying our minimal usage works against it and bumping the version pin, and clear the Python 3.16 `asyncio.iscoroutinefunction` deprecation warning.

**Architecture:** Three independent test-discoverable changes. (1) Add `class ErrorCode(str, enum.Enum)` to `src/ai_core/exceptions.py`, update 16 existing `DEFAULT_CODE` references and 9 inline `error_code="..."` call sites, expose `ErrorCode` in `ai_core.__all__`. (2) Install `sentry-sdk>=3.0,<4.0`, run existing sentry tests, fix any breakage, bump pin. (3) Replace `asyncio.iscoroutinefunction` with `inspect.iscoroutinefunction` in `schema/registry.py:248`.

**Tech Stack:** Python 3.11+, Pydantic v2, `enum.StrEnum`-style class (`class ErrorCode(str, enum.Enum)`), pytest, ruff, mypy strict. Spec: `docs/superpowers/specs/2026-05-06-ai-core-sdk-phase-8-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-8-stability-bundle` (already checked out off `main` post-Phase-7-merge; carries the Phase 8 spec at `8eee368`).

**Working-state hygiene** — do NOT touch:
- `README.md` (top-level)
- `src/ai_core/cli/main.py`, `cli/scaffold.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`
- `tests/integration/**` (Phase 7 territory)

**Mypy baseline:** 21 strict errors in 8 files. `mypy src` total must remain ≤ 21.
**Ruff baseline:** 211 errors at `3c26ca3` (Phase 7 merge commit). Total must remain ≤ 211.
**Pytest baseline (post-Phase-7):** 464 passing across `tests/unit tests/component tests/contract`. The 9 langgraph errors that haunted Phases 1-7 are GONE — verified.

**Per-step gate (every commit):**
- `ruff check <files-touched>` — no NEW violations
- `pytest tests/unit tests/component tests/contract -q` — must pass
- `pytest tests/integration -q` — must pass when Docker is up; auto-skip otherwise
- `mypy <files-touched>` strict — no new errors
- `mypy src 2>&1 | tail -1` — total ≤ 21

**Python interpreter:** `/Users/admin-h26/EAAP/.venv/bin/python` (no local venv inside the SDK).

**CRITICAL — actual codebase error_code values (verified by `grep`):** The Phase 8 spec drafted some enum values from memory; the plan uses the EXACT strings present in production code today. Changing them would emit different `error_code` values to dashboards and break consumers reading them. Phase 8 is a zero-behavior-change refactor.

The 16 concrete `EAAPBaseException` subclasses and their actual `DEFAULT_CODE` strings (from `grep -n "DEFAULT_CODE" src/ai_core/exceptions.py`):

| Class | DEFAULT_CODE (actual value) |
|---|---|
| ConfigurationError | `"config.invalid"` |
| SecretResolutionError | `"config.secret_not_resolved"` |
| DependencyResolutionError | `"di.resolution_failed"` |
| StorageError | `"storage.error"` |
| CheckpointError | `"storage.checkpoint_failed"` |
| PolicyDenialError | `"policy.denied"` |
| LLMInvocationError | `"llm.invocation_failed"` |
| LLMTimeoutError | `"llm.timeout"` |
| BudgetExceededError | `"llm.budget_exceeded"` |
| SchemaValidationError | `"schema.invalid"` |
| ToolValidationError | `"tool.validation_failed"` |
| ToolExecutionError | `"tool.execution_failed"` |
| AgentRuntimeError | `"agent.runtime_error"` |
| AgentRecursionLimitError | `"agent.recursion_limit"` |
| RegistryError | `"registry.error"` |
| MCPTransportError | `"mcp.transport_failed"` |

Plus 4 inline-only codes (no class home; raised at construction sites):
- `"config.yaml_path_missing"` (Phase 5; `config/settings.py:538`)
- `"config.yaml_parse_failed"` (Phase 5; `config/settings.py:556`)
- `"config.optional_dep_missing"` (Phase 6; `audit/sentry.py:49`, `audit/datadog.py:51`)
- `"llm.empty_response"` (`llm/litellm_client.py:268`)

**Total: 20 distinct codes → 20 ErrorCode members.**

Inline `error_code="..."` call sites (9 total) that need to be updated to `ErrorCode.<member>`:

```
src/ai_core/config/settings.py:538     "config.yaml_path_missing"
src/ai_core/config/settings.py:556     "config.yaml_parse_failed"
src/ai_core/di/module.py:288           "config.invalid"
src/ai_core/di/module.py:296           "config.invalid"
src/ai_core/di/module.py:309           "config.invalid"
src/ai_core/di/module.py:324           "config.invalid"
src/ai_core/llm/litellm_client.py:268  "llm.empty_response"
src/ai_core/audit/sentry.py:49         "config.optional_dep_missing"
src/ai_core/audit/datadog.py:51        "config.optional_dep_missing"
```

**Per-task commit message convention:** Conventional Commits.

---

## Task 1 — `ErrorCode` StrEnum registry

Adds the enum, updates 16 `DEFAULT_CODE` references, updates 9 inline `error_code="..."` call sites, exposes `ErrorCode` on `ai_core.__all__`, updates Phase 7's `EXPECTED_PUBLIC_NAMES` set, adds 2 new enum-self tests.

**Files:**
- Modify: `src/ai_core/exceptions.py` — add `ErrorCode` enum + update 16 `DEFAULT_CODE` lines
- Modify: `src/ai_core/__init__.py` — export `ErrorCode`
- Modify: `src/ai_core/config/settings.py` — 2 inline updates
- Modify: `src/ai_core/di/module.py` — 4 inline updates
- Modify: `src/ai_core/audit/sentry.py` — 1 inline update
- Modify: `src/ai_core/audit/datadog.py` — 1 inline update
- Modify: `src/ai_core/llm/litellm_client.py` — 1 inline update
- Modify: `tests/contract/test_public_surface.py` — `EXPECTED_PUBLIC_NAMES` 29 → 30
- Create: `tests/unit/exceptions/__init__.py` (empty)
- Create: `tests/unit/exceptions/test_error_code.py` — 2 new tests

### 1a — Add the `ErrorCode` enum

- [ ] **Step 1.1: Inspect existing `exceptions.py` imports**

```bash
head -30 /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/exceptions.py
```

Confirm `import enum` is NOT yet present at the top. If it is, skip the import-add part of Step 1.2.

- [ ] **Step 1.2: Add `import enum` and `class ErrorCode` to `exceptions.py`**

In `src/ai_core/exceptions.py`, add `import enum` to the imports block at the top. Then immediately AFTER the existing module docstring + imports and BEFORE `class EAAPBaseException(Exception):`, insert the `ErrorCode` definition:

```python
class ErrorCode(str, enum.Enum):
    """Canonical error codes for typed SDK exceptions.

    Members exhaustively cover every ``error_code`` string referenced by
    production code in Phases 1-6. Values are dotted-lowercase strings;
    ``ErrorCode`` inherits from ``str`` so members are directly comparable
    with raw strings::

        if exc.error_code == ErrorCode.CONFIG_INVALID:
            ...

    Adding a new code requires:
      1. Adding a member here (dotted-lowercase value).
      2. Wiring it into the appropriate exception class's ``DEFAULT_CODE``
         OR using ``ErrorCode.<member>`` directly at the construction site.

    The contract test
    ``test_every_concrete_exception_default_code_is_an_errorcode_member``
    catches any new exception class that bypasses the enum.
    """

    # Configuration (Phases 1, 5, 6)
    CONFIG_INVALID = "config.invalid"
    CONFIG_SECRET_NOT_RESOLVED = "config.secret_not_resolved"
    CONFIG_YAML_PATH_MISSING = "config.yaml_path_missing"
    CONFIG_YAML_PARSE_FAILED = "config.yaml_parse_failed"
    CONFIG_OPTIONAL_DEP_MISSING = "config.optional_dep_missing"

    # Dependency injection (Phase 1)
    DI_RESOLUTION_FAILED = "di.resolution_failed"

    # Storage / persistence (Phase 1)
    STORAGE_ERROR = "storage.error"
    STORAGE_CHECKPOINT_FAILED = "storage.checkpoint_failed"

    # Policy / authorization (Phase 1)
    POLICY_DENIED = "policy.denied"

    # LLM (Phase 1)
    LLM_INVOCATION_FAILED = "llm.invocation_failed"
    LLM_TIMEOUT = "llm.timeout"
    LLM_BUDGET_EXCEEDED = "llm.budget_exceeded"
    LLM_EMPTY_RESPONSE = "llm.empty_response"

    # Schema / validation (Phase 1)
    SCHEMA_INVALID = "schema.invalid"
    TOOL_VALIDATION_FAILED = "tool.validation_failed"

    # Tool execution (Phase 1)
    TOOL_EXECUTION_FAILED = "tool.execution_failed"

    # Agent runtime (Phase 1)
    AGENT_RUNTIME_ERROR = "agent.runtime_error"
    AGENT_RECURSION_LIMIT = "agent.recursion_limit"

    # Registry (Phase 1)
    REGISTRY_ERROR = "registry.error"

    # MCP transport (Phase 1)
    MCP_TRANSPORT_FAILED = "mcp.transport_failed"
```

The class MUST be defined BEFORE any class that references its members. Place it immediately after `class EAAPBaseException(Exception):` is fine — `EAAPBaseException` doesn't reference `ErrorCode`, only its subclasses do.

- [ ] **Step 1.3: Verify the file still parses**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "from ai_core.exceptions import ErrorCode; print(len(list(ErrorCode)), 'members')"
```

Expected: `20 members`.

- [ ] **Step 1.4: Run `mypy src` to confirm no type errors from the new enum**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `Found 21 errors in 8 files (checked 68 source files)` — unchanged baseline.

If new errors appear, the most likely causes are: (a) `enum` not imported, (b) `ErrorCode` defined after a class that references it. Fix and re-run.

### 1b — Update 16 `DEFAULT_CODE` references in `exceptions.py`

- [ ] **Step 1.5: Replace each `DEFAULT_CODE = "..."` with `DEFAULT_CODE = ErrorCode.<MEMBER>`**

Apply these 16 edits in `src/ai_core/exceptions.py`:

```python
class ConfigurationError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.CONFIG_INVALID  # was "config.invalid"


class SecretResolutionError(ConfigurationError):
    DEFAULT_CODE = ErrorCode.CONFIG_SECRET_NOT_RESOLVED  # was "config.secret_not_resolved"


class DependencyResolutionError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.DI_RESOLUTION_FAILED  # was "di.resolution_failed"


class StorageError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.STORAGE_ERROR  # was "storage.error"


class CheckpointError(StorageError):
    DEFAULT_CODE = ErrorCode.STORAGE_CHECKPOINT_FAILED  # was "storage.checkpoint_failed"


class PolicyDenialError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.POLICY_DENIED  # was "policy.denied"


class LLMInvocationError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.LLM_INVOCATION_FAILED  # was "llm.invocation_failed"


class LLMTimeoutError(LLMInvocationError):
    DEFAULT_CODE = ErrorCode.LLM_TIMEOUT  # was "llm.timeout"


class BudgetExceededError(LLMInvocationError):
    DEFAULT_CODE = ErrorCode.LLM_BUDGET_EXCEEDED  # was "llm.budget_exceeded"


class SchemaValidationError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.SCHEMA_INVALID  # was "schema.invalid"


class ToolValidationError(SchemaValidationError):
    DEFAULT_CODE = ErrorCode.TOOL_VALIDATION_FAILED  # was "tool.validation_failed"


class ToolExecutionError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.TOOL_EXECUTION_FAILED  # was "tool.execution_failed"


class AgentRuntimeError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.AGENT_RUNTIME_ERROR  # was "agent.runtime_error"


class AgentRecursionLimitError(AgentRuntimeError):
    DEFAULT_CODE = ErrorCode.AGENT_RECURSION_LIMIT  # was "agent.recursion_limit"


class RegistryError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.REGISTRY_ERROR  # was "registry.error"


class MCPTransportError(EAAPBaseException):
    DEFAULT_CODE = ErrorCode.MCP_TRANSPORT_FAILED  # was "mcp.transport_failed"
```

The class docstrings, attributes, and methods are otherwise unchanged. Each edit is a single line replacement.

(Drop the `# was "..."` comments after editing — they're just guidance for the implementer to verify the right value; final code has clean lines.)

- [ ] **Step 1.6: Verify the contract tests still pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/contract/test_exception_invariants.py -v 2>&1 | tail -10
```

Expected: 33 passed (16 classes × 2 parametrized tests + 1 sanity = 33). The contract tests are agnostic to whether `DEFAULT_CODE` holds a string or a StrEnum member — both compare equal to the same dotted-lowercase value.

If a test fails because `exc.error_code` returned an `ErrorCode` member (rather than a `str`) and the test does `assert exc.error_code == "config.invalid"`: that's still `True` via StrEnum equality, so the test should still pass. Investigate any failure carefully — there may be a value mismatch between the enum member and the original string (e.g., a typo).

### 1c — Update 9 inline `error_code="..."` call sites

- [ ] **Step 1.7: Update `config/settings.py` (2 sites)**

In `src/ai_core/config/settings.py`, find lines 538 and 556. Replace the raw strings:

```python
# Line ~538 — inside the explicit-path-missing branch
raise ConfigurationError(
    str(exc),
    error_code=ErrorCode.CONFIG_YAML_PATH_MISSING,  # was "config.yaml_path_missing"
    details={"env_var": _EAAP_CONFIG_PATH_ENV},
    cause=exc,
) from exc

# Line ~556 — inside the YAML parse-failure branch
raise ConfigurationError(
    f"Failed to load eaap.yaml at {yaml_path}: {exc}",
    error_code=ErrorCode.CONFIG_YAML_PARSE_FAILED,  # was "config.yaml_parse_failed"
    details={"path": str(yaml_path)},
    cause=exc,
) from exc
```

Add the import at the top of the file (or extend the existing `from ai_core.exceptions import ConfigurationError` line):

```python
from ai_core.exceptions import ConfigurationError, ErrorCode
```

(Verify the existing import line; the file may already group multiple imports.)

- [ ] **Step 1.8: Update `di/module.py` (4 sites)**

In `src/ai_core/di/module.py`, find lines 288, 296, 309, 324 — all 4 raise `ConfigurationError(..., error_code="config.invalid", ...)`. Replace the string literal with `ErrorCode.CONFIG_INVALID`:

```python
# Each of the 4 sites becomes:
raise ConfigurationError(
    "...",
    error_code=ErrorCode.CONFIG_INVALID,  # was "config.invalid"
)
```

Add the import at the top of the file (extend the existing `from ai_core.exceptions import ConfigurationError` line):

```python
from ai_core.exceptions import ConfigurationError, ErrorCode
```

- [ ] **Step 1.9: Update `audit/sentry.py` (1 site)**

In `src/ai_core/audit/sentry.py`, find line 49. Replace:

```python
raise ConfigurationError(
    "Sentry sink requires the 'sentry' optional dependency. "
    "Install with: pip install ai-core-sdk[sentry]",
    error_code=ErrorCode.CONFIG_OPTIONAL_DEP_MISSING,  # was "config.optional_dep_missing"
    details={"extra": "sentry"},
    cause=exc,
) from exc
```

Update the import at the top:

```python
from ai_core.exceptions import ConfigurationError, ErrorCode
```

The docstring at line 31 references the literal string `'config.optional_dep_missing'`. Leave the docstring as-is (it's documentation; consumers reading docs see the actual error_code value as a string, which matches the StrEnum value).

- [ ] **Step 1.10: Update `audit/datadog.py` (1 site)**

In `src/ai_core/audit/datadog.py`, find line 51. Apply the same edit pattern as Step 1.9:

```python
raise ConfigurationError(
    "Datadog sink requires the 'datadog' optional dependency. "
    "Install with: pip install ai-core-sdk[datadog]",
    error_code=ErrorCode.CONFIG_OPTIONAL_DEP_MISSING,  # was "config.optional_dep_missing"
    details={"extra": "datadog"},
    cause=exc,
) from exc
```

Update the import:

```python
from ai_core.exceptions import ConfigurationError, ErrorCode
```

- [ ] **Step 1.11: Update `llm/litellm_client.py` (1 site)**

In `src/ai_core/llm/litellm_client.py`, find line 268. The code raises an LLMInvocationError variant with `error_code="llm.empty_response"`. Read the actual line first to see the exact exception class being constructed:

```bash
sed -n '260,275p' /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/llm/litellm_client.py
```

Then update:

```python
# Whatever exception class is being raised, replace:
error_code="llm.empty_response"
# with:
error_code=ErrorCode.LLM_EMPTY_RESPONSE
```

Update the import. The file likely already imports a specific exception class (e.g., `LLMInvocationError`); extend that import:

```python
from ai_core.exceptions import LLMInvocationError, ErrorCode  # adjust to actual classes
```

- [ ] **Step 1.12: Run linting on all modified files**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/exceptions.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py \
    src/ai_core/audit/sentry.py \
    src/ai_core/audit/datadog.py \
    src/ai_core/llm/litellm_client.py
```

Expected: no NEW violations vs pre-task state.

### 1d — Add `ErrorCode` to public surface

- [ ] **Step 1.13: Update `src/ai_core/__init__.py`**

In `src/ai_core/__init__.py`, locate the `from ai_core.exceptions import (...)` block at line 37. Add `ErrorCode` (alphabetical order):

```python
from ai_core.exceptions import (
    AgentRecursionLimitError,
    AgentRuntimeError,
    BudgetExceededError,
    ConfigurationError,
    DependencyResolutionError,
    EAAPBaseException,
    ErrorCode,                          # NEW
    LLMInvocationError,
    LLMTimeoutError,
    MCPTransportError,
    PolicyDenialError,
    RegistryError,
    SchemaValidationError,
    SecretResolutionError,
    StorageError,
    ToolExecutionError,
    ToolValidationError,
)
```

(The alphabetical position depends on the existing layout; place `ErrorCode` between `EAAPBaseException` and `LLMInvocationError`.)

In the `__all__` list (line ~69), add `"ErrorCode"`:

```python
__all__ = [
    ...,
    "EAAPBaseException",
    "ErrorCode",                        # NEW
    ...,
]
```

(Maintain alphabetical order of the `__all__` entries.)

- [ ] **Step 1.14: Verify the surface count**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import ai_core; print(len(ai_core.__all__), 'exports'); print('ErrorCode' in ai_core.__all__)"
```

Expected: `30 exports` and `True`.

### 1e — Update Phase 7's contract test

- [ ] **Step 1.15: Update `EXPECTED_PUBLIC_NAMES` in the contract test**

In `tests/contract/test_public_surface.py`, find the `EXPECTED_PUBLIC_NAMES: frozenset[str]` definition. Add `"ErrorCode"`:

```python
EXPECTED_PUBLIC_NAMES: frozenset[str] = frozenset({
    "AICoreApp",
    "AgentRecursionLimitError",
    "AgentRuntimeError",
    "AgentState",
    "AuditEvent",
    "AuditRecord",
    "BaseAgent",
    "BudgetExceededError",
    "ConfigurationError",
    "DependencyResolutionError",
    "EAAPBaseException",
    "ErrorCode",                        # NEW (Phase 8)
    "HealthSnapshot",
    "IAuditSink",
    "IHealthProbe",
    "LLMInvocationError",
    "LLMTimeoutError",
    "MCPTransportError",
    "PolicyDenialError",
    "ProbeResult",
    "RegistryError",
    "SchemaValidationError",
    "SecretResolutionError",
    "StorageError",
    "Tool",
    "ToolExecutionError",
    "ToolSpec",
    "ToolValidationError",
    "new_agent_state",
    "tool",
})
```

(30 entries total. Maintain alphabetical order.)

- [ ] **Step 1.16: Run the contract surface test**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/contract/test_public_surface.py -v 2>&1 | tail -10
```

Expected: 1 passed.

### 1f — Add new enum-self tests

- [ ] **Step 1.17: Create the test directory if missing**

```bash
mkdir -p /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/exceptions
[ -f /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/exceptions/__init__.py ] || touch /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/exceptions/__init__.py
```

- [ ] **Step 1.18: Write the enum-self tests**

Create `tests/unit/exceptions/test_error_code.py`:

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
        assert value, "ErrorCode member has empty value"
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

- [ ] **Step 1.19: Run the new enum tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/exceptions/test_error_code.py -v 2>&1 | tail -10
```

Expected: 17 passed (1 unique-and-dotted-lowercase + 16 parametrized × `_all_concrete_exceptions`).

### 1g — Lint, type-check, full suite, commit

- [ ] **Step 1.20: Lint + type-check the Task 1 change set**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/exceptions.py \
    src/ai_core/__init__.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py \
    src/ai_core/audit/sentry.py \
    src/ai_core/audit/datadog.py \
    src/ai_core/llm/litellm_client.py \
    tests/contract/test_public_surface.py \
    tests/unit/exceptions/test_error_code.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/exceptions.py \
    src/ai_core/__init__.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py \
    src/ai_core/audit/sentry.py \
    src/ai_core/audit/datadog.py \
    src/ai_core/llm/litellm_client.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on touched files clean; project total ≤ 21.

- [ ] **Step 1.21: Run full unit + component + contract suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -10
```

Expected: ≥481 passing (464 baseline + 17 new enum tests = 481).

- [ ] **Step 1.22: Commit Task 1**

```bash
git add src/ai_core/exceptions.py \
        src/ai_core/__init__.py \
        src/ai_core/config/settings.py \
        src/ai_core/di/module.py \
        src/ai_core/audit/sentry.py \
        src/ai_core/audit/datadog.py \
        src/ai_core/llm/litellm_client.py \
        tests/contract/test_public_surface.py \
        tests/unit/exceptions/__init__.py \
        tests/unit/exceptions/test_error_code.py
git commit -m "feat(exceptions): ErrorCode StrEnum registry + 16 DEFAULT_CODE updates + 9 inline call-site updates"
```

---

## Task 2 — Sentry SDK v3 verify-and-bump

**Files:**
- Modify: `pyproject.toml` (bump pin)
- Possibly modify: `src/ai_core/audit/sentry.py` (only if v3 breaks the existing tests)
- Possibly modify: `tests/unit/audit/test_sentry_sink.py` (only if v3 changes test fixture API)

### 2a — Install Sentry SDK v3

- [ ] **Step 2.1: Install latest stable v3.x in the dev venv**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pip install --upgrade "sentry-sdk>=3.0,<4.0" 2>&1 | tail -5
```

Expected: `Successfully installed sentry-sdk-3.x.y` (where 3.x.y is the latest stable). If v3 isn't yet on PyPI's stable channel, abort Task 2 and document in the report — Phase 8 falls back to ship just Tasks 1 and 3.

- [ ] **Step 2.2: Smoke import**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import sentry_sdk; print('ok', sentry_sdk.VERSION)"
```

Expected: `ok 3.x.y`.

### 2b — Run existing sentry tests against v3

- [ ] **Step 2.3: Run the existing sentry tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_sentry_sink.py -v 2>&1 | tail -15
```

**If 9 passed:** v3 is compatible with our minimal usage. Proceed to Step 2.6.

**If any failed:** investigate the failure type:

- **`AttributeError: module 'sentry_sdk' has no attribute 'init'`** → v3 removed the function. Hard incompatibility; abort and document.
- **`TypeError: capture_event() got an unexpected keyword argument`** → event-dict shape changed. Adjust `audit/sentry.py` (likely a `level` field expecting an enum/Literal instead of bare string). Make the fix backwards-compatible if possible (works under v2 AND v3).
- **`type: ignore[arg-type]` now unused** → mypy complains about a stale `# type: ignore`. Adjust the ignore code or remove it.
- **Test fixture mock no longer matches** → `MagicMock()` for `sentry_sdk` now needs different attributes. Adjust the test fixture's mock setup minimally.

For each failure type, fix `audit/sentry.py` and/or the test fixture, re-run.

If after 2 fix attempts the suite still fails AND the fix would lose v2.x compatibility, abort: report `BLOCKED` with the v3 incompatibility details. Phase 8's fallback skips the pin bump.

- [ ] **Step 2.4: Run mypy on the touched file**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/audit/sentry.py 2>&1 | tail -3
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `audit/sentry.py` clean; `mypy src` total ≤ 21.

If a `# type: ignore[arg-type]` was previously needed for v2's stubs and is no longer needed under v3 (because v3's stubs are looser), the line will produce an unused-noqa warning. Remove the ignore. If a different ignore is needed, adjust.

- [ ] **Step 2.5: Verify v2.x compat is preserved**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pip install "sentry-sdk>=2.0,<3.0" 2>&1 | tail -3
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_sentry_sink.py -v 2>&1 | tail -10
```

Expected: all 9 sentry tests pass under v2.x as well. If v2.x now fails, the fix in Step 2.3 broke v2.x compat — adjust to support both, or abort and skip the pin bump.

After verification, restore v3:

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pip install --upgrade "sentry-sdk>=3.0,<4.0" 2>&1 | tail -3
```

### 2c — Bump the pin

- [ ] **Step 2.6: Update `pyproject.toml`**

In `pyproject.toml`, find the `[project.optional-dependencies]` `sentry` line:

```toml
sentry = ["sentry-sdk>=1.40,<3.0"]
```

Update to:

```toml
sentry = ["sentry-sdk>=1.40,<4.0"]
```

(Range now spans v1.x, v2.x, AND v3.x — broadest compat.)

- [ ] **Step 2.7: Verify the pin update**

```bash
grep -n "sentry-sdk" /Users/admin-h26/EAAP/ai-core-sdk/pyproject.toml
```

Expected: shows the new `<4.0` upper bound.

### 2d — Lint, type-check, full suite, commit

- [ ] **Step 2.8: Run full unit + component + contract suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -10
```

Expected: ≥481 passing (Task 1's count); no regressions from Task 2 changes.

- [ ] **Step 2.9: Commit Task 2**

```bash
git add pyproject.toml src/ai_core/audit/sentry.py tests/unit/audit/test_sentry_sink.py
git commit -m "build(sentry): bump sentry-sdk pin to <4.0 after verifying v3 compat"
```

(If `audit/sentry.py` and the test file weren't modified, only `pyproject.toml` is staged; the commit is just the pin bump.)

If verification failed AND Phase 8 fell back to skipping the pin bump: skip Steps 2.6–2.9 entirely. Document the v3 blocker in the Task 4 smoke-gate report.

---

## Task 3 — `asyncio.iscoroutinefunction` → `inspect.iscoroutinefunction`

**Files:**
- Modify: `src/ai_core/schema/registry.py:248` (1 line)
- Possibly modify: `src/ai_core/schema/registry.py` imports (drop `import asyncio` if unused)

### 3a — Replace the deprecated call

- [ ] **Step 3.1: Update line 248**

In `src/ai_core/schema/registry.py:248`, replace:

```python
# Before
            if asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):

# After
            if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
```

(Single-character delta: `asyncio.` → `inspect.`.)

- [ ] **Step 3.2: Check whether `import asyncio` is still needed**

```bash
grep -n "asyncio\." /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/schema/registry.py
```

If `asyncio.<something>` appears anywhere else in the file (e.g., `asyncio.Lock()` at line 70 — yes per pre-flight grep), KEEP the `import asyncio`. If it only appeared in the now-replaced line 248, drop the import.

Per the pre-flight grep, `registry.py:70` uses `asyncio.Lock()`, so KEEP the import.

### 3b — Verify

- [ ] **Step 3.3: Run the schema/registry tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/schema/test_registry.py -v 2>&1 | tail -10
```

Expected: 5 passed (or whatever the existing count is — should be identical pre/post-fix).

- [ ] **Step 3.4: Verify the deprecation warning is gone**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/schema/test_registry.py -W error::DeprecationWarning -v 2>&1 | tail -10
```

Expected: 5 passed with no `asyncio.iscoroutinefunction` deprecation warning. (The `-W error::DeprecationWarning` flag promotes warnings to errors so any leftover triggers a test failure.)

If a different deprecation appears (e.g., from another library), that's NOT a Phase 8 issue — leave it for a future task.

### 3c — Lint, type-check, full suite, commit

- [ ] **Step 3.5: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/schema/registry.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/schema/registry.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on `registry.py` clean; project total ≤ 21.

- [ ] **Step 3.6: Run full unit + component + contract suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -10
```

Expected: ≥481 passing (Task 2's count; Task 3 adds zero tests).

- [ ] **Step 3.7: Commit Task 3**

```bash
git add src/ai_core/schema/registry.py
git commit -m "fix(schema): use inspect.iscoroutinefunction (asyncio.iscoroutinefunction deprecated in Python 3.16)"
```

---

## Task 4 — End-of-phase smoke gate

Verification only. No code changes.

- [ ] **Step 4.1: Full test suite (always-run)**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -5
```

Expected: ≥481 passing; 0 errors.

- [ ] **Step 4.2: Integration suite (Docker-conditional)**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/integration -v 2>&1 | tail -10
```

Expected: 7 passed (Docker up) OR 1 passed + 6 skipped (Docker down — the bad-DSN test runs without Docker per Phase 7).

- [ ] **Step 4.3: Lint the entire tree**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src tests 2>&1 | grep "Found"
```

Expected: 211 errors total (= post-Phase-7 baseline). No NEW violations.

- [ ] **Step 4.4: Mypy strict**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `Found 21 errors in 8 files (checked 68 source files)`.

- [ ] **Step 4.5: Verify Phase 8 surface symbols**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
import ai_core
assert len(ai_core.__all__) == 30, f'Expected 30 exports, got {len(ai_core.__all__)}'
assert 'ErrorCode' in ai_core.__all__
from ai_core import ErrorCode
assert len(list(ErrorCode)) == 20, f'Expected 20 ErrorCode members, got {len(list(ErrorCode))}'
# Spot-check a member.
assert ErrorCode.CONFIG_INVALID == 'config.invalid'
print('Phase 8 surface OK')
"
```

Expected: `Phase 8 surface OK`.

- [ ] **Step 4.6: Verify the asyncio deprecation is gone**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit -q 2>&1 | grep -i "asyncio.iscoroutinefunction" | head -5
```

Expected: empty output (no deprecation warning on `asyncio.iscoroutinefunction` anywhere).

- [ ] **Step 4.7: Verify the Sentry pin is bumped (if Task 2 succeeded)**

```bash
grep -n "sentry-sdk" /Users/admin-h26/EAAP/ai-core-sdk/pyproject.toml
```

Expected: `sentry = ["sentry-sdk>=1.40,<4.0"]`. If Task 2 was skipped (v3 incompatibility), the pin remains `<3.0` and that's documented in Task 2's report.

- [ ] **Step 4.8: `eaap init` regression smoke (Phase 5 invariant)**

```bash
SMOKE=$(mktemp -d)
/Users/admin-h26/EAAP/.venv/bin/python -m ai_core.cli.main init smoke-app --path "$SMOKE" 2>&1 | tail -2
test -f "$SMOKE/smoke-app/eaap.yaml" && echo "eaap.yaml present"
test -f "$SMOKE/smoke-app/policies/agent.rego" && echo "agent.rego present"
rm -rf "$SMOKE"
```

Expected: 2 "present" lines.

- [ ] **Step 4.9: Capture phase summary**

```bash
git log --oneline 8eee368..HEAD
```

Expected: 3 conventional-commit subjects (one per Task) plus any review-fix commits.

- [ ] **Step 4.10: Do NOT push automatically**

```bash
git status
echo ""
echo "Suggested next step:"
echo "git push origin feat/phase-8-stability-bundle"
echo "gh pr create --title 'feat: Phase 8 — stability bundle (ErrorCode + Sentry v3 prep + asyncio fix)'"
```

---

## Out-of-scope reminders

For traceability, deferred to Phase 9+:

- Phase 4 cost/latency closure (Vertex AI Anthropic prefix, 1-hour cache TTL beta, multi-conn MCP pool, pool health-check probe)
- Sink polish (Sentry breadcrumbs / performance tracing, Datadog metrics / DogStatsD / log streaming)
- Redaction polish (custom regex at settings layer, per-sink redaction overrides, Microsoft Presidio / Faker / spaCy backends)
- ErrorCode metadata (description, severity, dashboard catalog) — Phase 8 ships the bare enum
- Sentry SDK v3-specific features (e.g., new `Scope.fork()` API)
- CI workflow updates

If a step starts pulling work from this list, stop and confirm scope with the user.
