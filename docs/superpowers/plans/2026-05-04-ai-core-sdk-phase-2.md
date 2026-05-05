# ai-core-sdk Phase 2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Production-reliability hardening + Phase-1 polish in a single batch — empty-response detection, compaction timeout, `error_code` field on every exception, new `LLMTimeoutError`/`MCPTransportError`, observability `fail_open` toggle, `NoOpPolicyEvaluator` as DI default, `BaseAgent.compile()` auto-registers tools, `HealthSnapshot` polish.

**Architecture:** Bottom-up — `error_code` retrofit lands first so subsequent steps use it for new exception calls. Each task touches one or two related modules; per-task gate is ruff + mypy strict (touched files only) + pytest unit/component. Project mypy strict total stays at-or-below the 23-error baseline.

**Tech Stack:** Python 3.11+, Pydantic v2, `injector` for DI, LangGraph, OpenTelemetry, `litellm`, `pytest` + ruff + mypy strict. Spec: `docs/superpowers/specs/2026-05-04-ai-core-sdk-phase-2-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-1-facade-tool-validation` (Phase 1 work + Phase 2 spec live here; Phase 2 implementation will continue on this branch unless the user pushes Phase 1 first and asks to start a new branch).

**Working-state hygiene:** the branch carries unrelated WIP (CLI templates, README, policies). **Do not touch any of these files** during this plan:
- `README.md`
- `src/ai_core/cli/main.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`

If a step's planned change accidentally lands in any of those paths, stop and re-evaluate.

**Mypy baseline:** the existing tree has 23 pre-existing strict errors in 8 files. Total error count must remain ≤ 23 after every commit.

**Per-step gate (every commit):**
- `ruff check <files-touched>` — must be clean.
- `pytest tests/unit tests/component -q` — must pass (excluding pre-existing `respx` and `aiosqlite` collection errors that already failed before Phase 1).
- `mypy <files-touched-by-this-task>` — strict-clean.
- Total project mypy error count never exceeds 23 (capture with `mypy src 2>&1 | tail -1`).

**Python interpreter:** `/Users/admin-h26/EAAP/.venv/bin/python` (no local venv inside the SDK).

**Per-task commit message convention:** Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).

---

## Task 1 — `error_code` retrofit + `LLMTimeoutError` + `MCPTransportError`

The largest task in Phase 2. Touches every existing exception subclass to add `DEFAULT_CODE`, plus the base class to populate `details["error_code"]`.

**Files:**
- Modify: `src/ai_core/exceptions.py`
- Test: `tests/unit/test_exceptions.py` (existing — extend with new tests)

### 1a — Modify `EAAPBaseException` to accept and auto-populate `error_code`

- [ ] **Step 1.1: Write the failing tests for the base-class behavior**

Append to `tests/unit/test_exceptions.py`:

```python
# ---------------------------------------------------------------------------
# error_code field — Phase 2
# ---------------------------------------------------------------------------
from ai_core.exceptions import (
    AgentRecursionLimitError,
    AgentRuntimeError,
    BudgetExceededError,
    CheckpointError,
    ConfigurationError,
    DependencyResolutionError,
    LLMInvocationError,
    LLMTimeoutError,
    MCPTransportError,
    PolicyDenialError,
    RegistryError,
    SecretResolutionError,
    StorageError,
)


def test_base_default_code() -> None:
    err = EAAPBaseException("x")
    assert err.error_code == "eaap.unknown"
    assert err.details["error_code"] == "eaap.unknown"


def test_per_instance_override_wins() -> None:
    err = LLMInvocationError("x", error_code="llm.context_length_exceeded")
    assert err.error_code == "llm.context_length_exceeded"
    assert err.details["error_code"] == "llm.context_length_exceeded"


def test_subclass_default_codes() -> None:
    cases: list[tuple[type[EAAPBaseException], str]] = [
        (ConfigurationError, "config.invalid"),
        (SecretResolutionError, "config.secret_not_resolved"),
        (DependencyResolutionError, "di.resolution_failed"),
        (StorageError, "storage.error"),
        (CheckpointError, "storage.checkpoint_failed"),
        (PolicyDenialError, "policy.denied"),
        (LLMInvocationError, "llm.invocation_failed"),
        (LLMTimeoutError, "llm.timeout"),
        (BudgetExceededError, "llm.budget_exceeded"),
        (SchemaValidationError, "schema.invalid"),
        (ToolValidationError, "tool.validation_failed"),
        (ToolExecutionError, "tool.execution_failed"),
        (AgentRuntimeError, "agent.runtime_error"),
        (AgentRecursionLimitError, "agent.recursion_limit"),
        (RegistryError, "registry.error"),
        (MCPTransportError, "mcp.transport_failed"),
    ]
    for cls, expected_code in cases:
        err = cls("msg")
        assert err.error_code == expected_code, f"{cls.__name__} got {err.error_code}"
        assert err.details["error_code"] == expected_code


def test_existing_details_preserved_with_error_code_added() -> None:
    err = LLMInvocationError(
        "x", details={"model": "gpt-4", "attempts": 3}
    )
    assert err.details == {
        "model": "gpt-4",
        "attempts": 3,
        "error_code": "llm.invocation_failed",
    }


def test_explicit_details_error_code_is_preserved() -> None:
    """If caller passes details with 'error_code' key, that value wins."""
    err = LLMInvocationError(
        "x", details={"error_code": "llm.custom"}, error_code="llm.timeout"
    )
    # The error_code arg sets self.error_code; setdefault leaves the explicit
    # details["error_code"] intact.
    assert err.error_code == "llm.timeout"
    assert err.details["error_code"] == "llm.custom"


def test_llm_timeout_error_lineage() -> None:
    err = LLMTimeoutError("timed out")
    assert isinstance(err, LLMInvocationError)
    assert isinstance(err, EAAPBaseException)
    assert err.error_code == "llm.timeout"


def test_mcp_transport_error_lineage() -> None:
    err = MCPTransportError("transport down")
    assert isinstance(err, EAAPBaseException)
    assert not isinstance(err, LLMInvocationError)
    assert err.error_code == "mcp.transport_failed"
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/test_exceptions.py -q
```
Expected: ImportError on `LLMTimeoutError` / `MCPTransportError`, plus `AttributeError: 'EAAPBaseException' object has no attribute 'error_code'`.

- [ ] **Step 1.3: Replace `src/ai_core/exceptions.py` with the Phase-2 version**

Read the existing file first to understand the structure (it has 13 exception classes plus `__all__`). Replace its contents with:

```python
"""Custom exception hierarchy for the EAAP SDK.

Every error raised by the SDK derives from :class:`EAAPBaseException`,
allowing host applications to catch SDK errors generically while still
distinguishing between sub-domains (configuration, persistence, policy,
LLM, …) for targeted handling and metrics.

Errors carry an optional structured ``details`` mapping that flows into
OpenTelemetry span attributes and LangFuse trace metadata so that
downstream operators can correlate failures without parsing strings.

Each subclass declares a ``DEFAULT_CODE`` class attribute (Phase 2) used
to auto-populate ``error_code`` when callers don't pass one explicitly.
The code lands in ``details["error_code"]`` and as the OTel span
attribute ``error.code`` so dashboards can aggregate uniformly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class EAAPBaseException(Exception):
    """Base class for all SDK-raised exceptions.

    Attributes:
        message: Human-readable description of the error.
        details: Optional structured context attached for observability.
        cause: Optional underlying exception preserved for chained tracebacks.
        error_code: Dotted, lowercase code (e.g. ``"llm.timeout"``) used by
            dashboards. Defaults to the subclass's ``DEFAULT_CODE``.
    """

    DEFAULT_CODE: str = "eaap.unknown"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: dict[str, Any] = dict(details or {})
        self.error_code: str = error_code or type(self).DEFAULT_CODE
        # Auto-populate so observability surfaces that walk `details` pick it up.
        self.details.setdefault("error_code", self.error_code)
        self.cause: BaseException | None = cause
        if cause is not None:
            self.__cause__ = cause

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}(message={self.message!r}, "
            f"error_code={self.error_code!r}, details={self.details!r})"
        )


# ---------------------------------------------------------------------------
# Configuration / secrets
# ---------------------------------------------------------------------------
class ConfigurationError(EAAPBaseException):
    """Raised when required configuration is missing or invalid."""

    DEFAULT_CODE = "config.invalid"


class SecretResolutionError(ConfigurationError):
    """Raised when a secret cannot be resolved from a secret backend."""

    DEFAULT_CODE = "config.secret_not_resolved"


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------
class DependencyResolutionError(EAAPBaseException):
    """Raised when the DI container cannot satisfy a binding."""

    DEFAULT_CODE = "di.resolution_failed"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class StorageError(EAAPBaseException):
    """Base class for persistence failures (SQL, vector, blob)."""

    DEFAULT_CODE = "storage.error"


class CheckpointError(StorageError):
    """Raised when reading or writing a LangGraph checkpoint fails."""

    DEFAULT_CODE = "storage.checkpoint_failed"


# ---------------------------------------------------------------------------
# Security & policy
# ---------------------------------------------------------------------------
class PolicyDenialError(EAAPBaseException):
    """Raised when an OPA policy denies a request or tool invocation."""

    DEFAULT_CODE = "policy.denied"


# ---------------------------------------------------------------------------
# LLM / budgeting
# ---------------------------------------------------------------------------
class LLMInvocationError(EAAPBaseException):
    """Raised when an LLM call fails after retry exhaustion."""

    DEFAULT_CODE = "llm.invocation_failed"


class LLMTimeoutError(LLMInvocationError):
    """Raised when an LLM call exceeds its configured timeout (post-retry)."""

    DEFAULT_CODE = "llm.timeout"


class BudgetExceededError(LLMInvocationError):
    """Raised when an agent or tenant has consumed its allocated quota."""

    DEFAULT_CODE = "llm.budget_exceeded"


# ---------------------------------------------------------------------------
# Schema / contract
# ---------------------------------------------------------------------------
class SchemaValidationError(EAAPBaseException):
    """Raised when a payload does not match the expected versioned schema."""

    DEFAULT_CODE = "schema.invalid"


class ToolValidationError(SchemaValidationError):
    """Tool input or output failed Pydantic validation.

    The ``details`` payload carries:

    * ``tool`` — the tool name,
    * ``version`` — the registered version,
    * ``side`` — ``"input"`` or ``"output"``,
    * ``errors`` — Pydantic ``error.errors()`` list.
    """

    DEFAULT_CODE = "tool.validation_failed"


class ToolExecutionError(EAAPBaseException):
    """A tool handler raised. The original exception is preserved via ``__cause__``.

    The ``details`` payload carries ``tool``, ``version``, and (when known)
    ``agent_id`` / ``tenant_id`` so dashboards can correlate failures with
    the calling agent.
    """

    DEFAULT_CODE = "tool.execution_failed"


# ---------------------------------------------------------------------------
# Agent runtime
# ---------------------------------------------------------------------------
class AgentRuntimeError(EAAPBaseException):
    """Base class for agent-runtime failures (recursion limit, etc.)."""

    DEFAULT_CODE = "agent.runtime_error"


class AgentRecursionLimitError(AgentRuntimeError):
    """Raised when an agent exceeds its configured recursion limit.

    The ``details`` payload includes ``agent_id`` and ``limit`` so
    operators can correlate dashboards with the exact agent that
    looped.
    """

    DEFAULT_CODE = "agent.recursion_limit"


# ---------------------------------------------------------------------------
# MCP / registry
# ---------------------------------------------------------------------------
class RegistryError(EAAPBaseException):
    """Raised when a component registry operation fails."""

    DEFAULT_CODE = "registry.error"


class MCPTransportError(EAAPBaseException):
    """Raised when an MCP transport (stdio/http/sse) fails to open or operate.

    The ``details`` payload carries ``component_id`` and ``transport`` so
    operators can identify which server failed.
    """

    DEFAULT_CODE = "mcp.transport_failed"


__all__ = [
    "AgentRecursionLimitError",
    "AgentRuntimeError",
    "BudgetExceededError",
    "CheckpointError",
    "ConfigurationError",
    "DependencyResolutionError",
    "EAAPBaseException",
    "LLMInvocationError",
    "LLMTimeoutError",
    "MCPTransportError",
    "PolicyDenialError",
    "RegistryError",
    "SchemaValidationError",
    "SecretResolutionError",
    "StorageError",
    "ToolExecutionError",
    "ToolValidationError",
]
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/test_exceptions.py -q
```
Expected: 10 passed (3 existing + 7 new).

- [ ] **Step 1.5: Audit and update existing tests that assert on `details` shape**

Many existing tests assert `exc.details == {...}` with literal dicts. Now `details` always includes `"error_code"`. Run:

```bash
grep -rn "exc.value.details\b" tests/ | head -30
grep -rn "\.details ==" tests/ | head -30
```

Update any test that asserts `exc.value.details == {literal_dict_without_error_code}` to either:
- Add `"error_code": "<expected_code>"` to the expected dict, OR
- Use `details["specific_key"] == value` instead of full-dict equality.

Specifically check (these are likely callers):
- `tests/unit/config/test_validation.py` — `test_validate_collects_all_issues` and similar
- `tests/unit/tools/test_invoker.py` — assertions on `exc.details`
- `tests/component/test_agent_tool_loop.py` — any assertions on tool error details

Run the full unit + component suite to confirm no failures:

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -5
```

If any test fails because of the new `details["error_code"]` key, update the assertion in the simplest way (prefer `assert exc.details["..."] == ...` over full-dict equality). Re-run until green.

- [ ] **Step 1.6: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/exceptions.py tests/unit/test_exceptions.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/exceptions.py tests/unit/test_exceptions.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: ruff clean; mypy on touched files clean; total project errors ≤ 23.

- [ ] **Step 1.7: Commit**

```bash
git add src/ai_core/exceptions.py tests/unit/test_exceptions.py
# Plus any test files updated in Step 1.5:
git add tests/unit/config/test_validation.py tests/unit/tools/test_invoker.py tests/component/test_agent_tool_loop.py 2>/dev/null || true
git commit -m "feat(exceptions): error_code retrofit + LLMTimeoutError + MCPTransportError"
```

---

## Task 2 — `LLMResponse.finish_reason` + empty-response detection + LLMTimeoutError raise

**Files:**
- Modify: `src/ai_core/di/interfaces.py`
- Modify: `src/ai_core/llm/litellm_client.py`
- Test: `tests/unit/llm/test_litellm_client.py` (likely exists; if not, create)
- Modify (test fakes): `tests/unit/agents/test_memory.py` — `FakeLLM` needs `finish_reason` parameter
- Modify (test fakes): `tests/component/test_agent_tool_loop.py` — `_ScriptedLLM` needs `finish_reason` parameter
- Modify (test fakes): `tests/unit/app/test_runtime.py` — `_StubLLM` needs `finish_reason` parameter

### 2a — Add `finish_reason` to `LLMResponse`

- [ ] **Step 2.1: Modify the dataclass**

In `src/ai_core/di/interfaces.py`, find `class LLMResponse:` and add a `finish_reason` field:

```python
@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalised LLM completion result."""

    model: str
    content: str
    tool_calls: Sequence[Mapping[str, Any]]
    usage: LLMUsage
    raw: Mapping[str, Any]
    finish_reason: str | None = None   # NEW — None means upstream didn't report
```

- [ ] **Step 2.2: Verify existing test fakes still construct `LLMResponse` correctly**

The default `None` keeps existing test fakes working. Run:

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -5
```

Expected: still passing. If any failure, the test was using positional args; switch it to keyword args or add `finish_reason="stop"` explicitly.

- [ ] **Step 2.3: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/di/interfaces.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/di/interfaces.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 23.

### 2b — Empty-response detection in `_normalise_response`

- [ ] **Step 2.4: Write failing tests**

The exact path of the LiteLLM client tests may already exist; check `tests/unit/llm/`. If `test_litellm_client.py` exists, append. If not, create the file and an `__init__.py`:

```bash
mkdir -p tests/unit/llm
[ -f tests/unit/llm/__init__.py ] || touch tests/unit/llm/__init__.py
```

Create or extend `tests/unit/llm/test_litellm_client.py`:

```python
"""Unit tests for the _normalise_response helper and LLMTimeoutError mapping."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_core.exceptions import LLMInvocationError, LLMTimeoutError
from ai_core.llm.litellm_client import _normalise_response

pytestmark = pytest.mark.unit


def _raw(
    *,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    model: str = "gpt-x",
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{
            "message": {"content": content, "tool_calls": tool_calls or []},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_normal_content_response() -> None:
    response = _normalise_response("gpt-x", _raw(content="hi", finish_reason="stop"))
    assert response.content == "hi"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"


def test_tool_only_response_succeeds() -> None:
    """content=='' AND tool_calls!=[] is a valid function-call response — must NOT raise."""
    response = _normalise_response("gpt-x", _raw(
        content="",
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "x", "arguments": "{}"}}],
        finish_reason="tool_calls",
    ))
    assert response.content == ""
    assert response.tool_calls != []
    assert response.finish_reason == "tool_calls"


def test_empty_response_raises_llm_invocation_error() -> None:
    """content=='' AND tool_calls==[] is the silent-data-loss case — must raise."""
    with pytest.raises(LLMInvocationError) as exc:
        _normalise_response("gpt-x", _raw(content="", finish_reason="length"))
    assert exc.value.error_code == "llm.empty_response"
    assert exc.value.details["finish_reason"] == "length"
    assert exc.value.details["model"] == "gpt-x"
    assert "raw_keys" in exc.value.details


def test_empty_response_with_no_finish_reason() -> None:
    with pytest.raises(LLMInvocationError) as exc:
        _normalise_response("gpt-x", _raw(content=""))
    assert exc.value.details["finish_reason"] is None


def test_finish_reason_none_when_upstream_omits() -> None:
    """If choices[0] has no finish_reason field, LLMResponse.finish_reason is None."""
    raw = _raw(content="hello")
    raw["choices"][0].pop("finish_reason", None)
    response = _normalise_response("gpt-x", raw)
    assert response.finish_reason is None
```

- [ ] **Step 2.5: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_litellm_client.py -q
```

Expected: failures pointing at missing empty-response handling.

- [ ] **Step 2.6: Update `_normalise_response` in `src/ai_core/llm/litellm_client.py`**

Find the existing `_normalise_response` function (currently around line 228). Replace its body with:

```python
def _normalise_response(model: str, raw: Any) -> LLMResponse:
    """Convert a LiteLLM response object/dict into an :class:`LLMResponse`.

    Raises:
        LLMInvocationError: If the response has neither content nor tool_calls
            (silent-data-loss case — likely truncation or content filter).
    """
    payload: Mapping[str, Any] = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)

    choices = payload.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    content = message.get("content") or ""
    tool_calls = list(message.get("tool_calls") or [])
    finish_reason = first.get("finish_reason")

    # Silent-data-loss detection: no content AND no tool_calls means the response
    # was either truncated, content-filtered, or malformed upstream. Raise so the
    # caller doesn't silently propagate an empty assistant turn into the agent loop.
    if not content and not tool_calls:
        raise LLMInvocationError(
            f"LLM returned empty response (finish_reason={finish_reason!r})",
            details={
                "model": model,
                "finish_reason": finish_reason,
                "raw_keys": sorted(str(k) for k in payload.keys()),
            },
            error_code="llm.empty_response",
        )

    usage_blob = payload.get("usage") or {}
    usage = LLMUsage(
        prompt_tokens=int(usage_blob.get("prompt_tokens", 0)),
        completion_tokens=int(usage_blob.get("completion_tokens", 0)),
        total_tokens=int(
            usage_blob.get(
                "total_tokens",
                int(usage_blob.get("prompt_tokens", 0))
                + int(usage_blob.get("completion_tokens", 0)),
            )
        ),
        cost_usd=_extract_cost(raw),
    )
    return LLMResponse(
        model=str(payload.get("model") or model),
        content=str(content),
        tool_calls=tool_calls,
        usage=usage,
        raw=payload,
        finish_reason=finish_reason,
    )
```

- [ ] **Step 2.7: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_litellm_client.py -q
```

Expected: 5 passed.

### 2c — Map retry-exhausted Timeout to `LLMTimeoutError`

- [ ] **Step 2.8: Write failing test**

Append to `tests/unit/llm/test_litellm_client.py`:

```python
import asyncio as _asyncio

from collections.abc import Sequence
from typing import Any

from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import (
    BudgetCheck,
    IBudgetService,
    IObservabilityProvider,
    SpanContext,
)
from ai_core.llm.litellm_client import LiteLLMClient
from contextlib import asynccontextmanager


class _AlwaysAllowBudget(IBudgetService):
    async def check(self, *, tenant_id: str | None, agent_id: str | None,
                    estimated_tokens: int) -> BudgetCheck:
        return BudgetCheck(allowed=True, remaining_tokens=None, remaining_usd=None)

    async def record_usage(self, *, tenant_id: str | None, agent_id: str | None,
                           prompt_tokens: int, completion_tokens: int,
                           cost_usd: float) -> None:
        return None


class _NoOpObservability(IObservabilityProvider):
    def start_span(self, name: str, *, attributes=None):
        @asynccontextmanager
        async def _cm():
            yield SpanContext(name=name, trace_id="t", span_id="s",
                              backend_handles={})
        return _cm()

    async def record_llm_usage(self, **kwargs: Any) -> None:
        return None

    async def record_event(self, name: str, *, attributes=None) -> None:
        return None

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_retry_exhausted_timeout_raises_llm_timeout_error(monkeypatch) -> None:
    """litellm.Timeout after retry exhaustion -> LLMTimeoutError (not LLMInvocationError)."""
    from litellm.exceptions import Timeout as LiteLLMTimeout

    settings = AppSettings()
    settings.llm.max_retries = 0  # one attempt; fail fast
    settings.llm.retry_initial_backoff_seconds = 0.01
    settings.llm.retry_max_backoff_seconds = 0.01

    async def _always_timeout(**kwargs: Any) -> Any:
        raise LiteLLMTimeout(
            message="upstream timed out",
            model="x",
            llm_provider="test",
        )

    monkeypatch.setattr("litellm.acompletion", _always_timeout)

    client = LiteLLMClient(
        settings=settings,
        budget=_AlwaysAllowBudget(),
        observability=_NoOpObservability(),
    )
    with pytest.raises(LLMTimeoutError) as exc:
        await client.complete(model="gpt-x", messages=[{"role": "user", "content": "hi"}])
    assert exc.value.error_code == "llm.timeout"
    assert exc.value.details["model"] == "gpt-x"


@pytest.mark.asyncio
async def test_retry_exhausted_non_timeout_raises_llm_invocation_error(monkeypatch) -> None:
    """Non-timeout transient error after retries -> generic LLMInvocationError."""
    from litellm.exceptions import RateLimitError

    settings = AppSettings()
    settings.llm.max_retries = 0
    settings.llm.retry_initial_backoff_seconds = 0.01
    settings.llm.retry_max_backoff_seconds = 0.01

    async def _always_429(**kwargs: Any) -> Any:
        raise RateLimitError(
            message="rate limited",
            model="x",
            llm_provider="test",
        )

    monkeypatch.setattr("litellm.acompletion", _always_429)

    client = LiteLLMClient(
        settings=settings,
        budget=_AlwaysAllowBudget(),
        observability=_NoOpObservability(),
    )
    with pytest.raises(LLMInvocationError) as exc:
        await client.complete(model="gpt-x", messages=[{"role": "user", "content": "hi"}])
    # Generic invocation_failed, NOT llm.timeout.
    assert exc.value.error_code == "llm.invocation_failed"
    assert not isinstance(exc.value, LLMTimeoutError)
```

- [ ] **Step 2.9: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_litellm_client.py::test_retry_exhausted_timeout_raises_llm_timeout_error -v
```

Expected: assertion fails because the current code raises generic `LLMInvocationError` for all retry-exhausted errors.

- [ ] **Step 2.10: Update the retry-exhaustion handler**

In `src/ai_core/llm/litellm_client.py`, find the `except RetryError as exc:` block (currently around line 161). Update it to distinguish timeout-class errors:

```python
            except RetryError as exc:
                last = exc.last_attempt.exception() if exc.last_attempt else exc
                if isinstance(last, Timeout):
                    raise LLMTimeoutError(
                        f"LLM call timed out after {cfg.max_retries + 1} attempts",
                        details={"model": resolved_model, "attempts": cfg.max_retries + 1},
                        cause=last,
                    ) from last
                raise LLMInvocationError(
                    "LLM invocation failed after retries",
                    details={"model": resolved_model, "attempts": cfg.max_retries + 1},
                    cause=last,
                ) from last
```

Add `LLMTimeoutError` to the import at the top of the file:

```python
from ai_core.exceptions import BudgetExceededError, LLMInvocationError, LLMTimeoutError
```

(`Timeout` is already imported from `litellm.exceptions`.)

- [ ] **Step 2.11: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_litellm_client.py -q
```

Expected: all tests pass (5 from §2b + 2 new = 7).

- [ ] **Step 2.12: Run full unit + component suite to verify no regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -5
```

Expected: still passing.

- [ ] **Step 2.13: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/llm/litellm_client.py src/ai_core/di/interfaces.py tests/unit/llm/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/llm/litellm_client.py src/ai_core/di/interfaces.py tests/unit/llm/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 23.

- [ ] **Step 2.14: Commit**

```bash
git add src/ai_core/di/interfaces.py src/ai_core/llm/litellm_client.py tests/unit/llm/
git commit -m "feat(llm): finish_reason on LLMResponse, empty-response detection, LLMTimeoutError mapping"
```

---

## Task 3 — `MemoryManager.compact` timeout

**Files:**
- Modify: `src/ai_core/config/settings.py`
- Modify: `src/ai_core/agents/memory.py`
- Test: `tests/unit/agents/test_memory.py` (existing — extend)

- [ ] **Step 3.1: Add `compaction_timeout_seconds` to `AgentSettings`**

In `src/ai_core/config/settings.py`, find `class AgentSettings` and add the new field:

```python
class AgentSettings(BaseSettings):
    """Agent runtime defaults (memory compaction, recursion limits, …)."""

    model_config = SettingsConfigDict(extra="ignore")

    memory_compaction_token_threshold: int = Field(default=8_000, ge=512)
    memory_compaction_target_tokens: int = Field(default=2_000, ge=128)
    compaction_timeout_seconds: float = Field(default=30.0, gt=0)  # NEW
    max_recursion_depth: int = Field(default=25, ge=1, le=200)
    essential_entity_keys: list[str] = Field(
        default_factory=lambda: ["user_id", "tenant_id", "session_id", "task_id"],
        description="State keys that must be preserved across compactions.",
    )
```

- [ ] **Step 3.2: Write failing test for skip-on-timeout**

Append to `tests/unit/agents/test_memory.py`:

```python
import asyncio as _asyncio


class _SlowFakeLLM(ILLMClient):
    """Sleeps before returning, simulating a slow upstream."""

    def __init__(self, sleep_seconds: float) -> None:
        self._sleep = sleep_seconds
        self.calls = 0

    async def complete(
        self,
        *,
        model: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        await _asyncio.sleep(self._sleep)
        return LLMResponse(
            model=model or "fake",
            content="summary",
            tool_calls=[],
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.0),
            raw={},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_compact_skips_on_timeout() -> None:
    """When the LLM takes longer than compaction_timeout_seconds, compact() returns
    state unchanged and logs a WARNING — no crash."""
    settings = AppSettings()
    settings.agent.compaction_timeout_seconds = 0.05  # 50ms cap
    slow_llm = _SlowFakeLLM(sleep_seconds=0.2)  # 200ms hang
    counter = FakeTokenCounter([10_000, 0, 0])
    mgr = MemoryManager(settings=settings, llm=slow_llm, token_counter=counter)

    state = new_agent_state(
        initial_messages=[{"role": "user", "content": "hi"}],
        essential={"tenant_id": "t1"},
        metadata={"agent_id": "a1"},
    )

    result = await mgr.compact(state, agent_id="a1", tenant_id="t1")
    # State returned unchanged on timeout (skip-and-warn).
    assert result is state
    assert slow_llm.calls == 1


@pytest.mark.asyncio
async def test_compact_succeeds_within_timeout() -> None:
    """Compaction completes normally when LLM responds within the budget."""
    settings = AppSettings()
    settings.agent.compaction_timeout_seconds = 1.0
    fast_llm = FakeLLM(summary="hello world")
    counter = FakeTokenCounter([10_000, 0, 0])
    mgr = MemoryManager(settings=settings, llm=fast_llm, token_counter=counter)

    state = new_agent_state(
        initial_messages=[{"role": "user", "content": "hi"}],
        essential={"tenant_id": "t1"},
    )

    result = await mgr.compact(state, agent_id="a1", tenant_id="t1")
    # Compaction succeeded — state has summary attached.
    assert result is not state
    assert result.get("summary") == "hello world"
```

- [ ] **Step 3.3: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/agents/test_memory.py::test_compact_skips_on_timeout -v
```

Expected: test fails because `compact` currently has no timeout — it waits the full 200ms and returns a compacted state instead of the original.

- [ ] **Step 3.4: Refactor `MemoryManager.compact` to wrap `_do_compact` in `asyncio.wait_for`**

In `src/ai_core/agents/memory.py`:

1. Add `import asyncio` and `import logging` near the top of the file (after existing imports).
2. Add module-level logger after the imports: `_logger = logging.getLogger(__name__)`.
3. Rename the existing `compact` body to `_do_compact` (same signature; private).
4. Replace the public `compact` method with the timeout wrapper.

The new public `compact`:

```python
    async def compact(
        self,
        state: AgentState,
        *,
        model: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
    ) -> AgentState:
        """Summarise the message history while preserving Essential Entities.

        Wraps :meth:`_do_compact` in :func:`asyncio.wait_for` using the
        configured ``compaction_timeout_seconds`` budget. On timeout, logs
        a WARNING and returns the input state unchanged so the agent run
        continues. The state may exceed the threshold next turn; the
        next-turn ``should_compact`` check will retry.
        """
        timeout = self._settings.agent.compaction_timeout_seconds
        try:
            return await asyncio.wait_for(
                self._do_compact(
                    state,
                    model=model,
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            _logger.warning(
                "Compaction skipped: LLM call exceeded %.1fs timeout "
                "(agent_id=%s, tenant_id=%s)",
                timeout, agent_id, tenant_id,
            )
            return state
```

The renamed private `_do_compact` keeps the existing body of the old `compact` (lines that build essentials, call the LLM, construct the new state). The signature is identical.

- [ ] **Step 3.5: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/agents/test_memory.py -q
```

Expected: all existing tests still pass + 2 new tests pass.

- [ ] **Step 3.6: Run full unit + component suite for regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -5
```

Expected: still passing.

- [ ] **Step 3.7: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/agents/memory.py src/ai_core/config/settings.py tests/unit/agents/test_memory.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/agents/memory.py src/ai_core/config/settings.py tests/unit/agents/test_memory.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 23.

- [ ] **Step 3.8: Commit**

```bash
git add src/ai_core/agents/memory.py src/ai_core/config/settings.py tests/unit/agents/test_memory.py
git commit -m "feat(agents): MemoryManager.compact() wraps in asyncio.wait_for; skip-and-WARN on timeout"
```

---

## Task 4 — `ObservabilitySettings.fail_open` + OTel `error.code` emission + `FakeObservabilityProvider` extension

**Files:**
- Modify: `src/ai_core/config/settings.py`
- Modify: `src/ai_core/observability/real.py`
- Modify: `tests/conftest.py` — `FakeObservabilityProvider` records `error_code`
- Test: `tests/unit/observability/test_real.py` (likely exists; if not, create) — fail_open behavior + OTel `error.code` emission

### 4a — Settings field

- [ ] **Step 4.1: Add `fail_open` to `ObservabilitySettings`**

In `src/ai_core/config/settings.py`, find `class ObservabilitySettings` and add the field after `log_level`:

```python
    fail_open: bool = Field(
        default=True,
        description=(
            "When True (default, recommended for production), backend errors "
            "(OTel exporter, LangFuse client) are caught and logged. When False "
            "(recommended for local/dev), they re-raise so misconfigured "
            "exporters surface immediately."
        ),
    )
```

### 4b — `RealObservabilityProvider._swallow_or_raise` helper + apply

- [ ] **Step 4.2: Read the existing observability file to identify operational-vs-shutdown swallow points**

```bash
grep -n "except Exception\|except BaseException" /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/observability/real.py
```

There are ~9 swallow blocks. Identify which are **operational** (user-flow errors during a span/event) vs **shutdown/teardown** (errors during `shutdown()` or LangFuse flush). Heuristic:
- Inside `_span()`, `start_span`, `record_event`, `record_llm_usage` — **operational** (apply `fail_open`).
- Inside `shutdown()`, finalizer paths — leave as-is (always swallow during teardown).

If unclear, read the surrounding 5 lines for context.

- [ ] **Step 4.3: Add the helper and read the setting in `__init__`**

In `src/ai_core/observability/real.py`, find `class RealObservabilityProvider` and add the helper method (place it near other private helpers):

```python
    def _swallow_or_raise(self, exc: BaseException, context: str) -> None:
        """Either log-and-suppress (fail_open=True) or re-raise."""
        if self._fail_open:
            _logger.warning("Observability backend error in %s: %s", context, exc)
            return
        raise exc
```

In `RealObservabilityProvider.__init__`, after the existing field assignments, add:

```python
        self._fail_open: bool = settings.fail_open
```

- [ ] **Step 4.4: Replace each operational `except Exception:` block**

For each operational swallow point identified in 4.2, transform:

```python
except Exception as exc:  # noqa: BLE001 — observability must never raise
    _logger.warning(...)
```

into:

```python
except Exception as exc:  # noqa: BLE001 — observability boundary, controlled by fail_open
    self._swallow_or_raise(exc, "<context-name>")
```

Pick a short context-name describing what failed: e.g. `"start_span"`, `"record_llm_usage"`, `"langfuse_event"`, `"langfuse_trace"`. The tests below assert on the failure path; the exact message is not critical.

**Do NOT modify shutdown/teardown swallow blocks** — those must always swallow regardless of `fail_open`.

### 4c — OTel `error.code` emission

- [ ] **Step 4.5: Add the EAAPBaseException-aware exception handling to `_span`**

Find the async context manager that wraps OTel spans (likely in `_span` or directly in `start_span`). The current pattern is:

```python
try:
    yield span_context
except BaseException as exc:
    span.record_exception(exc)
    raise
```

Update to first check for `EAAPBaseException` and tag `error.code`:

```python
try:
    yield span_context
except BaseException as exc:
    if isinstance(exc, EAAPBaseException):
        span.set_attribute("error.code", exc.error_code)
        for k, v in (exc.details or {}).items():
            if isinstance(v, (str, int, float, bool)):
                span.set_attribute(f"error.details.{k}", v)
    span.record_exception(exc)
    raise
```

Add the import at the top of `src/ai_core/observability/real.py`:

```python
from ai_core.exceptions import EAAPBaseException
```

The exact span object (`span`) name depends on the existing structure. Use the local name already in scope.

### 4d — `FakeObservabilityProvider` records `error_code`

- [ ] **Step 4.6: Extend the fake**

In `tests/conftest.py`, find `class _RecordedSpan` and add the `error_code` field:

```python
@dataclass(slots=True)
class _RecordedSpan:
    name: str
    attributes: Mapping[str, Any]
    exception: BaseException | None = None
    error_code: str | None = None         # NEW
```

In `FakeObservabilityProvider.start_span`'s context manager, in the `except BaseException` branch, also record `error_code` when applicable:

```python
            except BaseException as exc:
                recorded.exception = exc
                # Mirror RealObservabilityProvider's error_code tagging behavior.
                from ai_core.exceptions import EAAPBaseException as _EAAP
                if isinstance(exc, _EAAP):
                    recorded.error_code = exc.error_code
                raise
```

Also add a new fake at the bottom of `tests/conftest.py`:

```python
class FakeBrokenObservabilityProvider(IObservabilityProvider):
    """Observability provider whose start_span raises immediately on enter — useful
    for testing fail_open behaviour at the consumer (e.g., LiteLLMClient).
    """

    def start_span(self, name: str, *, attributes: Mapping[str, Any] | None = None):
        @asynccontextmanager
        async def _cm() -> AsyncIterator[SpanContext]:
            raise RuntimeError(f"observability backend down (start_span '{name}')")
            yield  # type: ignore[unreachable]  # for static checker

        return _cm()

    async def record_llm_usage(self, **kwargs: Any) -> None:
        raise RuntimeError("observability backend down (record_llm_usage)")

    async def record_event(self, name: str, *, attributes: Mapping[str, Any] | None = None) -> None:
        raise RuntimeError("observability backend down (record_event)")

    async def shutdown(self) -> None:
        return None  # shutdown intentionally tolerant
```

Add a fixture at the bottom for direct use in observability tests:

```python
@pytest.fixture
def fake_broken_observability() -> FakeBrokenObservabilityProvider:
    return FakeBrokenObservabilityProvider()
```

### 4e — Tests

- [ ] **Step 4.7: Write failing tests**

Create or extend `tests/unit/observability/test_real.py`:

```python
"""Tests for fail_open + error.code attribute emission on RealObservabilityProvider."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_core.config.settings import AppSettings, ObservabilitySettings
from ai_core.exceptions import EAAPBaseException, LLMInvocationError
from ai_core.observability.real import RealObservabilityProvider

pytestmark = pytest.mark.unit


def _settings(*, fail_open: bool) -> AppSettings:
    s = AppSettings()
    s.observability = ObservabilitySettings(fail_open=fail_open)
    return s


@pytest.mark.asyncio
async def test_fail_open_true_swallows_backend_error() -> None:
    """When fail_open=True (default), backend errors are logged but not raised."""
    provider = RealObservabilityProvider(_settings(fail_open=True))

    # Force an exception inside the span's tracer code by patching the tracer.
    with patch.object(provider, "_tracer", new=MagicMock()):
        provider._tracer.start_as_current_span.side_effect = RuntimeError("backend down")
        # start_span must not raise — fail_open swallows.
        async with provider.start_span("x", attributes={}):
            pass


@pytest.mark.asyncio
async def test_fail_open_false_raises_backend_error() -> None:
    """When fail_open=False, backend errors propagate."""
    provider = RealObservabilityProvider(_settings(fail_open=False))

    with patch.object(provider, "_tracer", new=MagicMock()):
        provider._tracer.start_as_current_span.side_effect = RuntimeError("backend down")
        with pytest.raises(RuntimeError, match="backend down"):
            async with provider.start_span("x", attributes={}):
                pass


@pytest.mark.asyncio
async def test_eaap_exception_tags_error_code_on_span() -> None:
    """When an EAAPBaseException propagates inside a span, span.set_attribute('error.code', ...) fires."""
    provider = RealObservabilityProvider(_settings(fail_open=True))

    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value.__enter__.return_value = fake_span

    with patch.object(provider, "_tracer", new=fake_tracer):
        with pytest.raises(LLMInvocationError):
            async with provider.start_span("test.span", attributes={}):
                raise LLMInvocationError(
                    "some failure",
                    details={"model": "gpt-x", "attempts": 3},
                )

    # Verify error.code attribute was set.
    fake_span.set_attribute.assert_any_call("error.code", "llm.invocation_failed")
    # Verify scalar details landed as attributes.
    fake_span.set_attribute.assert_any_call("error.details.model", "gpt-x")
    fake_span.set_attribute.assert_any_call("error.details.attempts", 3)
```

Note: the exact patch targets (`provider._tracer`, `start_as_current_span`) depend on the actual `RealObservabilityProvider` implementation. If the patches don't line up with reality (because the provider uses a different attribute name or a different context manager style), adapt the patch to match the actual code. The semantics — "force a backend exception, observe behavior" and "verify error.code attribute" — are what matters.

- [ ] **Step 4.8: Run tests to verify they fail**

```bash
mkdir -p tests/unit/observability
[ -f tests/unit/observability/__init__.py ] || touch tests/unit/observability/__init__.py
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/observability/test_real.py -q
```

Expected: `fail_open` tests pass on the True case (default behavior already swallows) but fail on the False case (no toggle yet); `error_code` test fails (no tagging logic yet).

If the patch targets are wrong (because of structural mismatch), the tests may fail at AttributeError. Read the source in `observability/real.py` and adjust patch targets.

- [ ] **Step 4.9: Run tests to verify all pass after the implementation in 4.3-4.5 is complete**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/observability/test_real.py -q
```

Expected: 3 passed.

- [ ] **Step 4.10: Run full unit + component suite for regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -5
```

Expected: still passing.

- [ ] **Step 4.11: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/observability/real.py src/ai_core/config/settings.py tests/conftest.py tests/unit/observability/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/observability/real.py src/ai_core/config/settings.py tests/conftest.py tests/unit/observability/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 23.

- [ ] **Step 4.12: Commit**

```bash
git add src/ai_core/observability/real.py src/ai_core/config/settings.py tests/conftest.py tests/unit/observability/
git commit -m "feat(observability): fail_open toggle + auto error.code emission on EAAPBaseException"
```

---

## Task 5 — `NoOpPolicyEvaluator` + DI rebinding + `ProductionSecurityModule`

**Files:**
- Create: `src/ai_core/security/noop_policy.py`
- Modify: `src/ai_core/di/module.py`
- Test: `tests/unit/security/test_noop_policy.py` (new)
- Test: `tests/unit/di/test_module_security.py` (new)

- [ ] **Step 5.1: Write failing tests for `NoOpPolicyEvaluator`**

```bash
mkdir -p tests/unit/security
[ -f tests/unit/security/__init__.py ] || touch tests/unit/security/__init__.py
```

Create `tests/unit/security/test_noop_policy.py`:

```python
"""Tests for the NoOpPolicyEvaluator."""
from __future__ import annotations

import pytest

from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision
from ai_core.security.noop_policy import NoOpPolicyEvaluator

pytestmark = pytest.mark.unit


def test_noop_implements_ipolicyevaluator() -> None:
    assert isinstance(NoOpPolicyEvaluator(), IPolicyEvaluator)


@pytest.mark.asyncio
async def test_noop_always_allows() -> None:
    evaluator = NoOpPolicyEvaluator()
    decision = await evaluator.evaluate(
        decision_path="anything", input={"user": "anyone"}
    )
    assert isinstance(decision, PolicyDecision)
    assert decision.allowed is True
    assert decision.obligations == {}
    assert decision.reason == "no-op evaluator (development only)"
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/security/test_noop_policy.py -q
```

Expected: ImportError on `ai_core.security.noop_policy`.

- [ ] **Step 5.3: Create `src/ai_core/security/noop_policy.py`**

```python
"""Always-allow policy evaluator for development without OPA running.

Production deployments MUST override the default DI binding with a real
evaluator (e.g. :class:`OPAPolicyEvaluator`) via
:class:`ai_core.di.module.ProductionSecurityModule`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision


class NoOpPolicyEvaluator(IPolicyEvaluator):
    """Policy evaluator that returns ``PolicyDecision(allowed=True)`` for every call.

    Use this in local development environments where standing up OPA is overkill.
    The reason field is set to a recognisable string so audit trails show that no
    real policy evaluation occurred.
    """

    async def evaluate(
        self, *, decision_path: str, input: Mapping[str, Any]
    ) -> PolicyDecision:
        """Always allow."""
        return PolicyDecision(
            allowed=True,
            obligations={},
            reason="no-op evaluator (development only)",
        )


__all__ = ["NoOpPolicyEvaluator"]
```

- [ ] **Step 5.4: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/security/test_noop_policy.py -q
```

Expected: 2 passed.

- [ ] **Step 5.5: Write failing test for the DI rebinding**

```bash
mkdir -p tests/unit/di
```

Create `tests/unit/di/test_module_security.py`:

```python
"""Tests verifying the security DI binding behavior in Phase 2.

- AgentModule binds NoOpPolicyEvaluator by default.
- ProductionSecurityModule swaps in OPAPolicyEvaluator when added.
"""
from __future__ import annotations

import pytest

from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import IPolicyEvaluator
from ai_core.di.module import ProductionSecurityModule
from ai_core.security.noop_policy import NoOpPolicyEvaluator
from ai_core.security.opa import OPAPolicyEvaluator

pytestmark = pytest.mark.unit


def test_default_container_resolves_noop_policy() -> None:
    """`Container.build([AgentModule()])` returns the NoOp evaluator."""
    container = Container.build([AgentModule()])
    evaluator = container.get(IPolicyEvaluator)
    assert isinstance(evaluator, NoOpPolicyEvaluator)


def test_production_security_module_swaps_to_opa() -> None:
    """Adding ProductionSecurityModule rebinds the evaluator to OPAPolicyEvaluator."""
    container = Container.build([AgentModule(), ProductionSecurityModule()])
    evaluator = container.get(IPolicyEvaluator)
    assert isinstance(evaluator, OPAPolicyEvaluator)
```

- [ ] **Step 5.6: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/di/test_module_security.py -q
```

Expected: failures — `AgentModule` currently binds OPA by default; `ProductionSecurityModule` doesn't exist.

- [ ] **Step 5.7: Update `src/ai_core/di/module.py`**

Find the existing `provide_policy_evaluator` method on `AgentModule` and replace with:

```python
    @singleton
    @provider
    def provide_policy_evaluator(self) -> IPolicyEvaluator:
        """Return the default :class:`NoOpPolicyEvaluator`.

        Production deployments must override this binding with
        :class:`ProductionSecurityModule` (or a custom module that binds a
        real evaluator) to enable policy enforcement.
        """
        from ai_core.security.noop_policy import NoOpPolicyEvaluator
        return NoOpPolicyEvaluator()
```

Remove the existing `OPAPolicyEvaluator` import at the top of the file (line ~51) since `AgentModule` no longer uses it directly.

Add a new module class at the end of the file (before `__all__`):

```python
class ProductionSecurityModule(Module):
    """Opt-in DI module that binds :class:`OPAPolicyEvaluator` over the NoOp default.

    Compose with :class:`AgentModule` for production:

    .. code-block:: python

        from ai_core.app import AICoreApp
        from ai_core.di.module import ProductionSecurityModule

        async with AICoreApp(modules=[ProductionSecurityModule()]) as app:
            ...
    """

    @singleton
    @provider
    def provide_policy_evaluator(self, settings: AppSettings) -> IPolicyEvaluator:
        """Return the OPA-backed policy evaluator. Loaded from `ai_core.security.opa`."""
        from ai_core.security.opa import OPAPolicyEvaluator
        return OPAPolicyEvaluator(settings)
```

If the file has an `__all__` list at the bottom, add `"ProductionSecurityModule"` to it.

- [ ] **Step 5.8: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/di/test_module_security.py tests/unit/security/test_noop_policy.py -q
```

Expected: 4 passed.

- [ ] **Step 5.9: Run full unit + component suite to check for regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: still passing. **Watch carefully for tests that previously asserted on OPA being the default policy** — those need updating to either use NoOp OR explicitly construct an OPA evaluator.

If `tests/unit/security/test_opa.py` was already broken (missing `respx`), it stays broken — it's not regressed by this change.

- [ ] **Step 5.10: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/security/noop_policy.py src/ai_core/di/module.py tests/unit/security/test_noop_policy.py tests/unit/di/test_module_security.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/security/noop_policy.py src/ai_core/di/module.py tests/unit/security/test_noop_policy.py tests/unit/di/test_module_security.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 23.

- [ ] **Step 5.11: Commit**

```bash
git add src/ai_core/security/noop_policy.py src/ai_core/di/module.py tests/unit/security/test_noop_policy.py tests/unit/di/test_module_security.py
git commit -m "feat(security): NoOpPolicyEvaluator default + opt-in ProductionSecurityModule for OPA"
```

---

## Task 6 — `BaseAgent.compile()` auto-registers tools

**Files:**
- Modify: `src/ai_core/agents/base.py`
- Test: `tests/component/test_agent_tool_loop.py` (existing — extend)

- [ ] **Step 6.1: Write failing test**

Append to `tests/component/test_agent_tool_loop.py`:

```python
@pytest.mark.asyncio
async def test_compile_auto_registers_tools_with_schema_registry(
    fake_observability, fake_policy_evaluator_factory
) -> None:
    """compile() must populate SchemaRegistry with every ToolSpec returned by tools()."""
    from ai_core.schema.registry import SchemaRegistry

    llm = _ScriptedLLM([_llm_msg(content="ok")])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)

    # Before compile: registry has no record of count_tool.
    registry = container.get(SchemaRegistry)
    with pytest.raises(Exception):
        registry.get("count", version=1)

    # Compile and verify registration happened.
    agent.compile()
    record = registry.get("count", version=1)
    assert record.input_schema is _In
    assert record.output_schema is _Out

    # Re-compile must be idempotent — no double-registration error.
    agent.compile()
```

- [ ] **Step 6.2: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/component/test_agent_tool_loop.py::test_compile_auto_registers_tools_with_schema_registry -v
```

Expected: failure — `compile()` does not currently register with the SchemaRegistry.

- [ ] **Step 6.3: Modify `BaseAgent.compile()`**

In `src/ai_core/agents/base.py`, find the `compile()` method and add the auto-registration loop. The current method has this shape (after Phase 1):

```python
def compile(self, *, checkpointer: Any | None = None) -> Any:
    if self._graph is not None:
        return self._graph
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("compact", self._compaction_node)
    graph.add_node("agent", self._agent_node)

    sdk_tools = [t for t in self.tools() if isinstance(t, ToolSpec)]
    install_loop = self.auto_tool_loop and bool(sdk_tools)
    ...
```

Add the registration loop between the `sdk_tools` line and the `install_loop` check:

```python
def compile(self, *, checkpointer: Any | None = None) -> Any:
    if self._graph is not None:
        return self._graph
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("compact", self._compaction_node)
    graph.add_node("agent", self._agent_node)

    sdk_tools = [t for t in self.tools() if isinstance(t, ToolSpec)]

    # Phase 2: auto-register each ToolSpec with the SchemaRegistry so that
    # `app.register_tools(*specs)` is optional. Idempotent — re-compile is fine.
    for spec in sdk_tools:
        self._tool_invoker.register(spec)

    install_loop = self.auto_tool_loop and bool(sdk_tools)
    # ... rest of compile() unchanged
```

- [ ] **Step 6.4: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/component/test_agent_tool_loop.py -q
```

Expected: all passing (existing + 1 new).

- [ ] **Step 6.5: Run full suite for regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -5
```

Expected: still passing.

- [ ] **Step 6.6: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/agents/base.py tests/component/test_agent_tool_loop.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/agents/base.py tests/component/test_agent_tool_loop.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 23.

- [ ] **Step 6.7: Commit**

```bash
git add src/ai_core/agents/base.py tests/component/test_agent_tool_loop.py
git commit -m "feat(agents): BaseAgent.compile() auto-registers ToolSpecs with SchemaRegistry"
```

---

## Task 7 — `AICoreApp.health` polish

**Files:**
- Modify: `src/ai_core/app/runtime.py`
- Test: `tests/unit/app/test_runtime.py` (existing — extend)

- [ ] **Step 7.1: Write failing tests for the new health shape**

Append to `tests/unit/app/test_runtime.py`:

```python
@pytest.mark.asyncio
async def test_health_components_populated_after_entry(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        snap = app.health
        assert snap.status == "ok"
        assert snap.components == {
            "settings": "ok",
            "container": "ok",
            "tool_invoker": "unknown",
            "policy_evaluator": "unknown",
            "observability": "unknown",
        }


@pytest.mark.asyncio
async def test_health_service_name_field(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """HealthSnapshot has a `service_name` field (renamed from settings_version)."""
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        snap = app.health
        assert snap.service_name == app.settings.service_name
        assert not hasattr(snap, "settings_version")


@pytest.mark.asyncio
async def test_health_before_entry_has_empty_components_and_blank_service_name() -> None:
    """Before __aenter__, components is empty dict and service_name is empty string."""
    app = AICoreApp()
    snap = app.health
    assert snap.status == "down"
    assert snap.components == {}
    assert snap.service_name == ""
```

- [ ] **Step 7.2: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py -q
```

Expected: failures — `components` is currently `{}` and the field is named `settings_version`.

- [ ] **Step 7.3: Modify `src/ai_core/app/runtime.py`**

Find `class HealthSnapshot:`. Rename the field:

```python
@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Coarse application health snapshot returned by :py:attr:`AICoreApp.health`."""

    status: Literal["ok", "degraded", "down"]
    components: dict[str, Literal["ok", "unknown"]]
    service_name: str       # was settings_version (Phase 2 rename)
```

Find the `health` property and replace its body:

```python
    @property
    def health(self) -> HealthSnapshot:
        if not self._entered or self._settings is None:
            return HealthSnapshot(
                status="down",
                components={},
                service_name="",
            )
        return HealthSnapshot(
            status="ok" if not self._closed else "down",
            components={
                "settings": "ok",
                "container": "ok",
                "tool_invoker": "unknown",
                "policy_evaluator": "unknown",
                "observability": "unknown",
            },
            service_name=self._settings.service_name,
        )
```

- [ ] **Step 7.4: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py -q
```

Expected: all passing.

- [ ] **Step 7.5: Run full suite for regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -5
```

Expected: still passing.

- [ ] **Step 7.6: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/app/runtime.py tests/unit/app/test_runtime.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/app/runtime.py tests/unit/app/test_runtime.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 23.

- [ ] **Step 7.7: Commit**

```bash
git add src/ai_core/app/runtime.py tests/unit/app/test_runtime.py
git commit -m "fix(app): HealthSnapshot.service_name + populated components dict"
```

---

## Task 8 — `MCPTransportError` wrap in `transports.py`

**Files:**
- Modify: `src/ai_core/mcp/transports.py`
- Test: `tests/unit/mcp/test_transports.py` (likely exists; if not, create)

- [ ] **Step 8.1: Read the existing transport file**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import ai_core.mcp.transports as t; print(t.__file__)"
```

Read the file in full. Key points to identify:
- Where `from fastmcp import Client` (or similar) is imported (likely deferred to call time).
- Where transport-specific objects (`StdioTransport`, `SSETransport`, `StreamableHttpTransport`) are constructed.
- Where the actual connection is made (likely inside an `async with` block).

- [ ] **Step 8.2: Write failing tests**

```bash
mkdir -p tests/unit/mcp
[ -f tests/unit/mcp/__init__.py ] || touch tests/unit/mcp/__init__.py
```

Create `tests/unit/mcp/test_transports.py`:

```python
"""Tests for FastMCPConnectionFactory error wrapping (MCPTransportError)."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from ai_core.exceptions import MCPTransportError
from ai_core.mcp.transports import FastMCPConnectionFactory, MCPServerSpec

pytestmark = pytest.mark.unit


def _spec() -> MCPServerSpec:
    return MCPServerSpec(
        component_id="test-server",
        transport="stdio",
        target="/usr/bin/echo",
    )


def test_missing_fastmcp_raises_mcp_transport_error() -> None:
    """If fastmcp isn't importable, open() raises MCPTransportError with a helpful hint."""
    factory = FastMCPConnectionFactory()
    spec = _spec()

    # Force ImportError by removing fastmcp from sys.modules and shadowing the import.
    # The factory's open() calls __import__('fastmcp') deferred — patching __import__ traps it.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "fastmcp" or name.startswith("fastmcp."):
            raise ImportError(f"No module named '{name}' (test-injected)")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        with pytest.raises(MCPTransportError) as exc:
            # Touch the context manager — open() may be lazy until __aenter__.
            ctx = factory.open(spec)
            # If open() returned a context manager, entering it triggers the import.
            if hasattr(ctx, "__aenter__"):
                import asyncio
                async def _enter() -> None:
                    async with ctx:
                        pass
                asyncio.run(_enter())

    assert exc.value.error_code == "mcp.transport_failed"
    assert "fastmcp" in exc.value.message.lower() or "fastmcp" in str(exc.value.cause).lower()
    assert exc.value.details["component_id"] == "test-server"
    assert exc.value.details["transport"] == "stdio"
```

- [ ] **Step 8.3: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/mcp/test_transports.py -q
```

Expected: a raw `ImportError` (or whatever the underlying failure is) leaks instead of `MCPTransportError`.

- [ ] **Step 8.4: Wrap the import failure in `transports.py`**

Find where `from fastmcp import ...` (or the equivalent dynamic import) happens inside `FastMCPConnectionFactory.open` (or its inner async generator). Wrap with:

```python
try:
    from fastmcp import Client  # or whichever symbols are needed
    # ... transport-specific instantiation
except ImportError as exc:
    raise MCPTransportError(
        "FastMCP is not installed; install with `pip install ai-core-sdk[mcp]`",
        details={"component_id": spec.component_id, "transport": spec.transport},
        cause=exc,
    ) from exc
```

If the existing code structure deferred imports through a function (e.g., `_open_stdio`, `_open_http`), wrap in each function the same way.

For runtime connection failures (httpx/anyio errors during `async with client:`), the implementer at task time decides the cleanest place to catch — likely an `async with` wrapper that catches `(httpx.HTTPError, anyio.BrokenResourceError, OSError)` and re-raises as `MCPTransportError`. The contract (callers see only `MCPTransportError`) is what matters.

Add the import at the top of `src/ai_core/mcp/transports.py`:

```python
from ai_core.exceptions import MCPTransportError, RegistryError
```

(`RegistryError` is likely already imported.)

- [ ] **Step 8.5: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/mcp/test_transports.py -q
```

Expected: passed. If the test still fails because of patch-target mismatch, adjust the test to match the actual import location (the implementation is correct — the test needs to invoke the right pathway).

- [ ] **Step 8.6: Run full suite for regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -5
```

Expected: still passing.

- [ ] **Step 8.7: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/mcp/transports.py tests/unit/mcp/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/mcp/transports.py tests/unit/mcp/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean (the file currently has 2 pre-existing mypy errors at lines 107 and 111 — DON'T regress those, but also don't fix them in scope); total project ≤ 23.

- [ ] **Step 8.8: Commit**

```bash
git add src/ai_core/mcp/transports.py tests/unit/mcp/
git commit -m "feat(mcp): wrap fastmcp transport ImportError + connection failures as MCPTransportError"
```

---

## Task 9 — Top-level package exports for new exception types

**Files:**
- Modify: `src/ai_core/__init__.py`
- Test: `tests/unit/test_top_level_imports.py` (existing — extend)

- [ ] **Step 9.1: Write failing test**

In `tests/unit/test_top_level_imports.py`, find `test_canonical_imports_exist` and extend the import list to include `LLMTimeoutError` and `MCPTransportError`:

```python
def test_canonical_imports_exist() -> None:
    from ai_core import (
        AICoreApp,
        AgentRecursionLimitError,
        AgentRuntimeError,
        AgentState,
        BaseAgent,
        BudgetExceededError,
        ConfigurationError,
        DependencyResolutionError,
        EAAPBaseException,
        HealthSnapshot,
        LLMInvocationError,
        LLMTimeoutError,        # NEW (Phase 2)
        MCPTransportError,      # NEW (Phase 2)
        PolicyDenialError,
        RegistryError,
        SchemaValidationError,
        SecretResolutionError,
        StorageError,
        Tool,
        ToolExecutionError,
        ToolSpec,
        ToolValidationError,
        new_agent_state,
        tool,
    )
    locals_dict = locals()
    for name in [
        "AICoreApp", "AgentRecursionLimitError", "AgentRuntimeError",
        "AgentState", "BaseAgent", "BudgetExceededError", "ConfigurationError",
        "DependencyResolutionError", "EAAPBaseException", "HealthSnapshot",
        "LLMInvocationError", "LLMTimeoutError",
        "MCPTransportError",
        "PolicyDenialError", "RegistryError", "SchemaValidationError",
        "SecretResolutionError", "StorageError", "Tool", "ToolExecutionError",
        "ToolSpec", "ToolValidationError", "new_agent_state", "tool",
    ]:
        assert locals_dict[name] is not None
```

- [ ] **Step 9.2: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/test_top_level_imports.py -q
```

Expected: ImportError on `LLMTimeoutError` / `MCPTransportError`.

- [ ] **Step 9.3: Update `src/ai_core/__init__.py`**

Find the imports from `ai_core.exceptions` and add the two new names:

```python
from ai_core.exceptions import (
    AgentRecursionLimitError,
    AgentRuntimeError,
    BudgetExceededError,
    ConfigurationError,
    DependencyResolutionError,
    EAAPBaseException,
    LLMInvocationError,
    LLMTimeoutError,           # NEW (Phase 2)
    MCPTransportError,         # NEW (Phase 2)
    PolicyDenialError,
    RegistryError,
    SchemaValidationError,
    SecretResolutionError,
    StorageError,
    ToolExecutionError,
    ToolValidationError,
)
```

Add them to `__all__` in the appropriate section (Exceptions group):

```python
__all__ = [
    # Application
    "AICoreApp",
    "HealthSnapshot",
    # Agents
    "BaseAgent",
    "AgentState",
    "new_agent_state",
    # Tools
    "tool",
    "Tool",
    "ToolSpec",
    # Exceptions
    "EAAPBaseException",
    "ConfigurationError",
    "DependencyResolutionError",
    "SecretResolutionError",
    "StorageError",
    "PolicyDenialError",
    "BudgetExceededError",
    "LLMInvocationError",
    "LLMTimeoutError",       # NEW
    "SchemaValidationError",
    "ToolValidationError",
    "ToolExecutionError",
    "AgentRuntimeError",
    "AgentRecursionLimitError",
    "RegistryError",
    "MCPTransportError",     # NEW
]
```

- [ ] **Step 9.4: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/test_top_level_imports.py -q
```

Expected: 2 passed.

- [ ] **Step 9.5: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/__init__.py tests/unit/test_top_level_imports.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/__init__.py tests/unit/test_top_level_imports.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 23.

- [ ] **Step 9.6: Commit**

```bash
git add src/ai_core/__init__.py tests/unit/test_top_level_imports.py
git commit -m "feat: add LLMTimeoutError + MCPTransportError to top-level exports"
```

---

## Task 10 — End-of-phase smoke gate

**Files:** none (verification only).

This task is the gate that closes Phase 2. Same shape as Phase 1's smoke gate.

- [ ] **Step 10.1: Full test suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Report:
- Total passes / fails / errors.
- Which failures/errors are pre-existing (`respx`/`aiosqlite`) vs new.

- [ ] **Step 10.2: Lint the entire tree**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src tests
```

Expected: no NEW violations introduced by Phase 2.

- [ ] **Step 10.3: Mypy strict — total error count must not regress**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `Found N errors in M files (checked X source files)` with `N <= 23`.

- [ ] **Step 10.4: Smoke against `my-eaap-app`**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import importlib; importlib.import_module('ai_core'); print('ai_core imported ok')"
```

Then test the canonical "Hello, agent" path with the new exceptions:

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
from ai_core import (
    AICoreApp, BaseAgent, tool,
    AgentState, new_agent_state, Tool, ToolSpec, HealthSnapshot,
    EAAPBaseException, ConfigurationError, DependencyResolutionError,
    SecretResolutionError, StorageError, PolicyDenialError, BudgetExceededError,
    LLMInvocationError, LLMTimeoutError, SchemaValidationError, ToolValidationError,
    ToolExecutionError, AgentRuntimeError, AgentRecursionLimitError, RegistryError,
    MCPTransportError,
)
# Spot-check that error_code is populated.
assert LLMInvocationError('x').error_code == 'llm.invocation_failed'
assert LLMTimeoutError('x').error_code == 'llm.timeout'
assert MCPTransportError('x').error_code == 'mcp.transport_failed'
print('Phase 2 canonical surface ok; error_codes populated')
"
```

- [ ] **Step 10.5: Capture phase summary**

```bash
git log --oneline b1033cc..HEAD
```

This shows every Phase 2 commit (since the spec landed at `b1033cc`). Verify the commit graph reads as a clean phase: ~10 commits, conventional-commit subjects.

- [ ] **Step 10.6: Do NOT push the branch automatically**

Skip `git push`. The user will decide when to push. Just report `git status` and the suggested PR command:

```bash
git status
echo "Suggested next step:"
echo "git push -u origin feat/phase-1-facade-tool-validation"
echo "gh pr create --title 'feat: Phase 2 — resilience hardening + Phase 1 polish'"
```

---

## Out-of-scope reminders

For traceability, here is what is **deferred to Phase 3+** and must not creep into Phase 2:

- Real health probes (OPA ping, DB connect, model lookup).
- Structured logging (`structlog`).
- Audit sink for OPA decisions.
- Anthropic prompt caching.
- MCP connection pooling.
- `eaap init` scaffolding updates (the WIP on `main` covers some of this — kept untouched).
- `tests/unit/security/test_opa.py` env fix (`respx` install) — env work, not in this phase.
- Settings hash field in `HealthSnapshot`.

If a step starts pulling work from this list, stop and confirm scope with the user.
