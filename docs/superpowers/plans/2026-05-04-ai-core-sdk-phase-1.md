# ai-core-sdk Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `AICoreApp` lifecycle facade, the `@tool` decorator pipeline, and collect-all `validate_for_runtime()` on `AppSettings` so SDK consumers can write a working agent in 10 lines from top-level imports only.

**Architecture:** Bottom-up — build leaf collaborators first, compose upward. Three new subpackages (`app/`, `tools/`, plus `config/validation.py`); modify `agents/base.py`, `di/module.py`, `exceptions.py`, top-level `__init__.py`. No deletions; pre-1.0 so no backwards-compat shims.

**Tech Stack:** Python 3.11+, Pydantic v2, `injector` for DI, LangGraph for graph topology, OpenTelemetry for spans, `pytest` + ruff + mypy strict. Spec: `docs/superpowers/specs/2026-05-04-ai-core-sdk-phase-1-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-1-facade-tool-validation` (already created off `main`).

**Working-state hygiene:** the branch carries unrelated WIP (CLI templates, README, policies). **Do not touch any of these files** during this plan:
- `README.md`
- `src/ai_core/cli/main.py`, `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`

If a step's planned change accidentally lands in any of those paths, stop and re-evaluate.

**Mypy baseline:** the existing tree has 23 pre-existing strict errors in 8 files. New code MUST be strict-clean; total error count MUST NOT regress.

**Per-step gate (every commit):**
- `ruff check src tests` — must be clean.
- `pytest tests/unit tests/component -q` — must pass.
- `mypy <files-touched-by-this-task>` — strict-clean (i.e., the files this task creates/modifies show no errors).
- Total project mypy error count never exceeds 23 (capture with `mypy src 2>&1 | tail -1`).

**Per-task commit message convention:** Conventional Commits (`feat:`, `test:`, `refactor:`, `docs:`) — match the style of `8b8f3a5 Initial check-in - Pass 5` for short subjects.

---

## Task 1 — Add reusable test fakes to `conftest.py`

**Why first:** every later task imports these fakes. Locking the fake interfaces first prevents drift across files.

**Files:**
- Modify: `tests/conftest.py`
- Test: tests live alongside their callsites; this task is verified by import + a single sanity test.

- [ ] **Step 1.1: Write the failing import test**

Create `tests/unit/test_conftest_fakes.py`:

```python
"""Smoke test that the reusable fakes are importable and behave correctly."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_fake_policy_evaluator_allow(fake_policy_evaluator_factory):
    evaluator = fake_policy_evaluator_factory(default_allow=True)
    import asyncio
    decision = asyncio.run(evaluator.evaluate(decision_path="x", input={}))
    assert decision.allowed is True


def test_fake_policy_evaluator_deny(fake_policy_evaluator_factory):
    evaluator = fake_policy_evaluator_factory(default_allow=False, reason="nope")
    import asyncio
    decision = asyncio.run(evaluator.evaluate(decision_path="x", input={}))
    assert decision.allowed is False
    assert decision.reason == "nope"


def test_fake_observability_records_spans(fake_observability):
    import asyncio

    async def go() -> None:
        async with fake_observability.start_span("a", attributes={"k": "v"}):
            await fake_observability.record_event("e", attributes={"x": 1})

    asyncio.run(go())
    assert [s.name for s in fake_observability.spans] == ["a"]
    assert fake_observability.spans[0].attributes == {"k": "v"}
    assert fake_observability.events == [("e", {"x": 1})]


def test_fake_secret_manager_resolves(fake_secret_manager_factory):
    from ai_core.config.secrets import SecretRef
    import asyncio

    mgr = fake_secret_manager_factory({("env", "MY_KEY"): "secret-val"})
    val = asyncio.run(mgr.resolve(SecretRef(backend="env", name="MY_KEY")))
    assert val == "secret-val"
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd /Users/admin-h26/EAAP/ai-core-sdk
.venv/bin/python -m pytest tests/unit/test_conftest_fakes.py -q
```
Expected: `ERRORS` — fixtures not defined.

(If `.venv` isn't local, use `/Users/admin-h26/EAAP/.venv/bin/python` — that's the active venv per the existing setup.)

- [ ] **Step 1.3: Add the fakes and fixtures to `tests/conftest.py`**

Replace the contents of `tests/conftest.py` with:

```python
"""Shared pytest fixtures for the SDK test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from ai_core.config.secrets import ISecretManager, SecretRef
from ai_core.config.settings import AppSettings, get_settings
from ai_core.di.interfaces import (
    IObservabilityProvider,
    IPolicyEvaluator,
    PolicyDecision,
    SpanContext,
)
from ai_core.exceptions import SecretResolutionError


# ---------------------------------------------------------------------------
# Settings cache hygiene (existing fixtures — preserved)
# ---------------------------------------------------------------------------
@pytest.fixture
def clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fresh_settings() -> AppSettings:
    return AppSettings()


# ---------------------------------------------------------------------------
# FakePolicyEvaluator
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _PolicyCall:
    decision_path: str
    input: Mapping[str, Any]


class FakePolicyEvaluator(IPolicyEvaluator):
    """Deterministic IPolicyEvaluator for tests.

    Configure with a default verdict and optional per-path overrides.
    Records every call for inspection.
    """

    def __init__(
        self,
        *,
        default_allow: bool = True,
        reason: str | None = None,
        overrides: Mapping[str, PolicyDecision] | None = None,
    ) -> None:
        self._default_allow = default_allow
        self._reason = reason
        self._overrides = dict(overrides or {})
        self.calls: list[_PolicyCall] = []

    async def evaluate(
        self, *, decision_path: str, input: Mapping[str, Any]
    ) -> PolicyDecision:
        self.calls.append(_PolicyCall(decision_path=decision_path, input=dict(input)))
        if decision_path in self._overrides:
            return self._overrides[decision_path]
        return PolicyDecision(
            allowed=self._default_allow,
            obligations={},
            reason=self._reason,
        )


@pytest.fixture
def fake_policy_evaluator_factory():
    def _make(
        *,
        default_allow: bool = True,
        reason: str | None = None,
        overrides: Mapping[str, PolicyDecision] | None = None,
    ) -> FakePolicyEvaluator:
        return FakePolicyEvaluator(
            default_allow=default_allow, reason=reason, overrides=overrides
        )

    return _make


# ---------------------------------------------------------------------------
# FakeObservabilityProvider
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _RecordedSpan:
    name: str
    attributes: Mapping[str, Any]
    exception: BaseException | None = None


class FakeObservabilityProvider(IObservabilityProvider):
    """Records spans, events, and LLM-usage entries for assertion in tests."""

    def __init__(self) -> None:
        self.spans: list[_RecordedSpan] = []
        self.events: list[tuple[str, Mapping[str, Any]]] = []
        self.usage: list[Mapping[str, Any]] = []
        self.shutdowns: int = 0

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ):
        recorded = _RecordedSpan(name=name, attributes=dict(attributes or {}))
        self.spans.append(recorded)

        @asynccontextmanager
        async def _cm() -> AsyncIterator[SpanContext]:
            try:
                yield SpanContext(
                    name=name,
                    trace_id="trace-fake",
                    span_id=f"span-{len(self.spans)}",
                    backend_handles={},
                )
            except BaseException as exc:
                recorded.exception = exc
                raise

        return _cm()

    async def record_llm_usage(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_usd: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.usage.append(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "attributes": dict(attributes or {}),
            }
        )

    async def record_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append((name, dict(attributes or {})))

    async def shutdown(self) -> None:
        self.shutdowns += 1


@pytest.fixture
def fake_observability() -> FakeObservabilityProvider:
    return FakeObservabilityProvider()


# ---------------------------------------------------------------------------
# FakeSecretManager
# ---------------------------------------------------------------------------
class FakeSecretManager(ISecretManager):
    """In-memory ISecretManager keyed by (backend, name)."""

    def __init__(self, mapping: Mapping[tuple[str, str], str] | None = None) -> None:
        self._mapping = dict(mapping or {})

    async def resolve(self, ref: SecretRef) -> str:
        try:
            return self._mapping[(ref.backend, ref.name)]
        except KeyError as exc:
            raise SecretResolutionError(
                f"FakeSecretManager has no value for {ref.backend}/{ref.name}",
                details={"backend": ref.backend, "name": ref.name},
                cause=exc,
            ) from exc


@pytest.fixture
def fake_secret_manager_factory():
    def _make(mapping: Mapping[tuple[str, str], str] | None = None) -> FakeSecretManager:
        return FakeSecretManager(mapping or {})

    return _make
```

- [ ] **Step 1.4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/test_conftest_fakes.py -q
```
Expected: 4 passed.

- [ ] **Step 1.5: Run full unit suite to verify no regression**

```bash
.venv/bin/python -m pytest tests/unit -q
```
Expected: all existing tests still pass.

- [ ] **Step 1.6: Lint + type-check the new code**

```bash
.venv/bin/python -m ruff check tests/conftest.py tests/unit/test_conftest_fakes.py
.venv/bin/python -m mypy tests/conftest.py tests/unit/test_conftest_fakes.py
```
Expected: both clean.

- [ ] **Step 1.7: Commit**

```bash
git add tests/conftest.py tests/unit/test_conftest_fakes.py
git commit -m "test: add reusable FakePolicyEvaluator, FakeObservabilityProvider, FakeSecretManager"
```

---

## Task 2 — `config/validation.py` + `AppSettings.validate_for_runtime()`

**Files:**
- Create: `src/ai_core/config/validation.py`
- Modify: `src/ai_core/config/settings.py`
- Test: `tests/unit/config/test_validation.py`

### 2a — Create the helper module first (it's a leaf)

- [ ] **Step 2.1: Write the failing test for `ConfigIssue` + `ValidationContext`**

Create `tests/unit/config/test_validation.py`:

```python
"""Unit tests for the runtime-config validation helpers."""
from __future__ import annotations

import pytest

from ai_core.config.validation import ConfigIssue, ValidationContext

pytestmark = pytest.mark.unit


def test_config_issue_is_frozen():
    issue = ConfigIssue(path="x.y", message="m", hint="h")
    with pytest.raises(Exception):
        issue.path = "z"  # type: ignore[misc]
    assert issue.path == "x.y"
    assert issue.message == "m"
    assert issue.hint == "h"


def test_validation_context_collects_issues():
    ctx = ValidationContext()
    assert ctx.has_issues is False
    ctx.fail("a.b", "broken", hint="fix it")
    ctx.fail("c.d", "also broken")
    assert ctx.has_issues is True
    assert len(ctx.issues) == 2
    assert ctx.issues[0] == ConfigIssue(path="a.b", message="broken", hint="fix it")
    assert ctx.issues[1] == ConfigIssue(path="c.d", message="also broken", hint=None)


def test_validation_context_starts_empty():
    assert ValidationContext().issues == []
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/config/test_validation.py -q
```
Expected: collection error — module not found.

- [ ] **Step 2.3: Implement `config/validation.py`**

```python
"""Runtime configuration validation helpers.

These primitives back :meth:`ai_core.config.settings.AppSettings.validate_for_runtime`.
They accumulate every problem before raising so a developer can fix the entire
configuration in one pass instead of one error at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """A single runtime-configuration problem.

    Attributes:
        path: Dotted path inside :class:`AppSettings`, e.g. ``"llm.default_model"``.
        message: Short human-readable description of what is wrong.
        hint: Optional, actionable instruction for the developer (env var name,
            override pattern, doc link, …). ``None`` when there is no useful hint.
    """

    path: str
    message: str
    hint: str | None = None


@dataclass(slots=True)
class ValidationContext:
    """Accumulator passed through individual validators.

    Validators call :py:meth:`fail` to record problems. The context is consumed
    by :meth:`AppSettings.validate_for_runtime`, which raises
    :class:`ai_core.exceptions.ConfigurationError` when :py:attr:`has_issues`
    is ``True``.
    """

    issues: list[ConfigIssue] = field(default_factory=list)

    def fail(self, path: str, message: str, hint: str | None = None) -> None:
        """Record a single problem; never raises."""
        self.issues.append(ConfigIssue(path=path, message=message, hint=hint))

    @property
    def has_issues(self) -> bool:
        """True when at least one validator has called :py:meth:`fail`."""
        return bool(self.issues)


__all__ = ["ConfigIssue", "ValidationContext"]
```

- [ ] **Step 2.4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/config/test_validation.py -q
```
Expected: 3 passed.

- [ ] **Step 2.5: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/config/validation.py tests/unit/config/test_validation.py
.venv/bin/python -m mypy src/ai_core/config/validation.py tests/unit/config/test_validation.py
```
Expected: both clean.

- [ ] **Step 2.6: Commit**

```bash
mkdir -p tests/unit/config
git add src/ai_core/config/validation.py tests/unit/config/test_validation.py
git commit -m "feat(config): add ConfigIssue + ValidationContext for collect-all settings validation"
```

### 2b — Add `validate_for_runtime` on `AppSettings`

- [ ] **Step 2.7: Write failing tests for the validator method**

Append to `tests/unit/config/test_validation.py`:

```python
# ---------------------------------------------------------------------------
# AppSettings.validate_for_runtime
# ---------------------------------------------------------------------------
from ai_core.config.settings import AppSettings, AgentSettings, LLMSettings
from ai_core.exceptions import ConfigurationError


def _settings(**overrides):
    """Build an AppSettings with targeted overrides; defaults are otherwise valid."""
    return AppSettings(**overrides)


def test_validate_passes_on_default_settings(fake_secret_manager_factory):
    s = AppSettings()
    s.validate_for_runtime(secret_manager=fake_secret_manager_factory({}))


def test_validate_rejects_empty_default_model():
    s = AppSettings(llm=LLMSettings(default_model=""))
    with pytest.raises(ConfigurationError) as exc:
        s.validate_for_runtime()
    issues = exc.value.details["issues"]
    assert len(issues) == 1
    assert issues[0]["path"] == "llm.default_model"
    assert "non-empty" in issues[0]["message"]
    assert issues[0]["hint"] is not None
    assert "EAAP_LLM__DEFAULT_MODEL" in issues[0]["hint"]


def test_validate_rejects_blank_fallback_model():
    s = AppSettings(llm=LLMSettings(default_model="m", fallback_models=["good", "  "]))
    with pytest.raises(ConfigurationError) as exc:
        s.validate_for_runtime()
    issues = exc.value.details["issues"]
    assert any(i["path"].startswith("llm.fallback_models[") for i in issues)


def test_validate_rejects_compaction_target_above_threshold():
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


def test_validate_collects_all_issues():
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


def test_validate_rejects_non_isecretmanager():
    s = AppSettings()
    with pytest.raises(ConfigurationError) as exc:
        s.validate_for_runtime(secret_manager="not-a-manager")  # type: ignore[arg-type]
    issues = exc.value.details["issues"]
    assert any(i["path"] == "secret_manager" for i in issues)
```

- [ ] **Step 2.8: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/config/test_validation.py -q
```
Expected: failures pointing at missing `validate_for_runtime` method.

- [ ] **Step 2.9: Implement `validate_for_runtime` on `AppSettings`**

Open `src/ai_core/config/settings.py`. Add this import near the top of the file (after the existing `from pydantic_settings import BaseSettings, SettingsConfigDict` line):

```python
from ai_core.config.validation import ValidationContext
```

Then add the method to `AppSettings` (place it just below `is_production`):

```python
    def validate_for_runtime(
        self,
        *,
        secret_manager: "ISecretManager | None" = None,
    ) -> None:
        """Collect-all runtime validation. Raises :class:`ConfigurationError` once.

        Pydantic enforces type and ``Field`` constraints at construction time,
        but a few real-world invariants slip through:

        * Required strings can be set to the empty string by an environment
          override.
        * Cross-field invariants (e.g. compaction target < threshold) cannot
          be expressed as single-field constraints.
        * The ``secret_manager`` parameter is forward-looking: future
          settings groups may carry :class:`SecretRef` instances, and we
          want the wiring to be in place before that happens.

        Args:
            secret_manager: Optional :class:`ISecretManager` used to resolve
                any :class:`SecretRef` instances reachable from this settings
                tree. If ``None``, secret-resolution checks are skipped.

        Raises:
            ConfigurationError: If at least one issue is found. The exception
                ``details["issues"]`` field is a list of
                ``{"path", "message", "hint"}`` dicts — one per problem.
        """
        from ai_core.config.secrets import ISecretManager  # local to avoid cycles
        from ai_core.exceptions import ConfigurationError  # local to avoid cycles

        ctx = ValidationContext()

        # llm.default_model must be a non-empty, non-blank string.
        if not (self.llm.default_model and self.llm.default_model.strip()):
            ctx.fail(
                path="llm.default_model",
                message="must be a non-empty model identifier",
                hint=(
                    "set env var EAAP_LLM__DEFAULT_MODEL=<model-id> "
                    "or override AppSettings.llm.default_model in code"
                ),
            )

        # llm.fallback_models entries must be non-blank.
        for idx, model in enumerate(self.llm.fallback_models):
            if not (model and model.strip()):
                ctx.fail(
                    path=f"llm.fallback_models[{idx}]",
                    message="fallback model identifier must be non-empty",
                    hint="remove the entry or set a real model id",
                )

        # Cross-field: compaction target must be strictly less than threshold.
        if (
            self.agent.memory_compaction_target_tokens
            >= self.agent.memory_compaction_token_threshold
        ):
            ctx.fail(
                path="agent.memory_compaction_target_tokens",
                message=(
                    "must be strictly less than "
                    "agent.memory_compaction_token_threshold "
                    f"(target={self.agent.memory_compaction_target_tokens}, "
                    f"threshold={self.agent.memory_compaction_token_threshold})"
                ),
                hint=(
                    "set EAAP_AGENT__MEMORY_COMPACTION_TARGET_TOKENS to a value "
                    "strictly less than EAAP_AGENT__MEMORY_COMPACTION_TOKEN_THRESHOLD"
                ),
            )

        # secret_manager type-check (forward-looking).
        if secret_manager is not None and not isinstance(secret_manager, ISecretManager):
            ctx.fail(
                path="secret_manager",
                message=f"must be an ISecretManager, got {type(secret_manager).__name__}",
                hint="pass an instance of ai_core.config.secrets.ISecretManager",
            )

        if ctx.has_issues:
            raise ConfigurationError(
                f"Runtime configuration is invalid: {len(ctx.issues)} issue(s) found.",
                details={"issues": [
                    {"path": i.path, "message": i.message, "hint": i.hint}
                    for i in ctx.issues
                ]},
            )
```

Also, at the top of `settings.py`, add a `TYPE_CHECKING` import block so the `"ISecretManager | None"` annotation resolves under `mypy --strict` without creating a runtime cycle:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_core.config.secrets import ISecretManager
```

Place this after the existing `from typing import Annotated, Literal` import.

- [ ] **Step 2.10: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/config/test_validation.py -q
```
Expected: 9 passed.

- [ ] **Step 2.11: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/config/settings.py src/ai_core/config/validation.py tests/unit/config/test_validation.py
.venv/bin/python -m mypy src/ai_core/config/settings.py src/ai_core/config/validation.py tests/unit/config/test_validation.py
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: ruff clean; mypy on touched files clean; total project errors `<= 23`.

- [ ] **Step 2.12: Run full unit suite**

```bash
.venv/bin/python -m pytest tests/unit -q
```
Expected: all green; no regressions.

- [ ] **Step 2.13: Commit**

```bash
git add src/ai_core/config/settings.py tests/unit/config/test_validation.py
git commit -m "feat(config): add AppSettings.validate_for_runtime() with collect-all validation"
```

---

## Task 3 — Add `ToolValidationError` + `ToolExecutionError` to `exceptions.py`

**Files:**
- Modify: `src/ai_core/exceptions.py`
- Test: `tests/unit/test_exceptions.py` (new file; pattern matches `tests/unit/agents/test_*.py`)

- [ ] **Step 3.1: Write the failing test**

Create `tests/unit/test_exceptions.py`:

```python
"""Tests for the new tool-related exception types."""
from __future__ import annotations

import pytest

from ai_core.exceptions import (
    EAAPBaseException,
    SchemaValidationError,
    ToolExecutionError,
    ToolValidationError,
)

pytestmark = pytest.mark.unit


def test_tool_validation_error_is_schema_validation_error():
    err = ToolValidationError(
        "bad",
        details={"tool": "x", "version": 1, "side": "input", "errors": []},
    )
    assert isinstance(err, SchemaValidationError)
    assert isinstance(err, EAAPBaseException)
    assert err.details["side"] == "input"


def test_tool_execution_error_chains_cause():
    cause = RuntimeError("inner")
    err = ToolExecutionError(
        "bad",
        details={"tool": "x", "version": 1, "agent_id": "a", "tenant_id": "t"},
        cause=cause,
    )
    assert err.__cause__ is cause
    assert err.cause is cause
    assert err.details["agent_id"] == "a"


def test_tool_execution_error_is_eaap_base_exception():
    err = ToolExecutionError("x", details={"tool": "x", "version": 1})
    assert isinstance(err, EAAPBaseException)
    # Not a SchemaValidationError — that's reserved for validation failures.
    assert not isinstance(err, SchemaValidationError)
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/test_exceptions.py -q
```
Expected: ImportError.

- [ ] **Step 3.3: Add the exceptions**

In `src/ai_core/exceptions.py`, just below the `SchemaValidationError` declaration block, insert:

```python
class ToolValidationError(SchemaValidationError):
    """Tool input or output failed Pydantic validation.

    The ``details`` payload carries:

    * ``tool`` — the tool name,
    * ``version`` — the registered version,
    * ``side`` — ``"input"`` or ``"output"``,
    * ``errors`` — Pydantic ``error.errors()`` list.
    """


class ToolExecutionError(EAAPBaseException):
    """A tool handler raised. The original exception is preserved via ``__cause__``.

    The ``details`` payload carries ``tool``, ``version``, and (when known)
    ``agent_id`` / ``tenant_id`` so dashboards can correlate failures with
    the calling agent.
    """
```

Update the `__all__` list at the bottom of the file to include the two new names (insert them after `"SchemaValidationError"`):

```python
    "SchemaValidationError",
    "ToolValidationError",
    "ToolExecutionError",
```

- [ ] **Step 3.4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/test_exceptions.py -q
```
Expected: 3 passed.

- [ ] **Step 3.5: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/exceptions.py tests/unit/test_exceptions.py
.venv/bin/python -m mypy src/ai_core/exceptions.py tests/unit/test_exceptions.py
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: clean; total errors `<= 23`.

- [ ] **Step 3.6: Commit**

```bash
git add src/ai_core/exceptions.py tests/unit/test_exceptions.py
git commit -m "feat(exceptions): add ToolValidationError and ToolExecutionError"
```

---

## Task 4 — `tools/spec.py` — `ToolSpec` dataclass + `Tool` Protocol

**Files:**
- Create: `src/ai_core/tools/__init__.py` (placeholder; flesh out later)
- Create: `src/ai_core/tools/spec.py`
- Test: `tests/unit/tools/test_spec.py`

- [ ] **Step 4.1: Create the empty subpackage**

```bash
mkdir -p src/ai_core/tools tests/unit/tools
```

Create `src/ai_core/tools/__init__.py`:

```python
"""Tool authoring primitives — the @tool decorator and runtime invoker.

Phase 1 — see docs/superpowers/specs/2026-05-04-ai-core-sdk-phase-1-design.md.
"""

from __future__ import annotations

from ai_core.tools.spec import Tool, ToolSpec

__all__ = ["Tool", "ToolSpec"]
```

- [ ] **Step 4.2: Write the failing test**

Create `tests/unit/tools/test_spec.py`:

```python
"""Tests for ToolSpec dataclass and the Tool Protocol."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ai_core.tools.spec import Tool, ToolSpec

pytestmark = pytest.mark.unit


class _In(BaseModel):
    query: str
    limit: int = 10


class _Out(BaseModel):
    items: list[str]


async def _handler(payload: _In) -> _Out:  # noqa: ARG001
    return _Out(items=[])


def _spec() -> ToolSpec:
    return ToolSpec(
        name="search",
        version=1,
        description="search items",
        input_model=_In,
        output_model=_Out,
        handler=_handler,
        opa_path="eaap/agent/tool_call/allow",
    )


def test_toolspec_is_frozen():
    spec = _spec()
    with pytest.raises(Exception):
        spec.name = "other"  # type: ignore[misc]


def test_openai_schema_round_trips_through_json():
    schema = _spec().openai_schema()
    blob = json.dumps(schema)
    restored = json.loads(blob)
    assert restored["type"] == "function"
    assert restored["function"]["name"] == "search"
    assert restored["function"]["description"] == "search items"
    assert "query" in restored["function"]["parameters"]["properties"]


def test_toolspec_satisfies_tool_protocol():
    spec = _spec()
    # Structural subtype: ToolSpec instances must be usable as Tool.
    assert isinstance(spec, Tool)  # runtime_checkable Protocol
    assert spec.name == "search"
    assert spec.version == 1


def test_toolspec_eq_uses_name_and_version():
    a = _spec()
    b = _spec()
    assert a == b
    c = ToolSpec(
        name="search",
        version=2,
        description="search items",
        input_model=_In,
        output_model=_Out,
        handler=_handler,
        opa_path=None,
    )
    assert a != c
```

- [ ] **Step 4.3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/tools/test_spec.py -q
```
Expected: ImportError on `ai_core.tools.spec`.

- [ ] **Step 4.4: Implement `tools/spec.py`**

```python
"""Immutable tool descriptor + structural Protocol.

A :class:`ToolSpec` is what the ``@tool`` decorator produces. It carries
everything needed to (1) advertise the tool to an LLM via OpenAI's
function-calling schema, (2) validate a call's input/output, and (3) route
the call through OPA for authorisation.

ToolSpec deliberately does **not** depend on DI or observability. Those
concerns live in :class:`ai_core.tools.invoker.ToolInvoker`, which is
constructed once at app startup and passed each spec at invoke time.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


ToolHandler = Callable[[BaseModel], Awaitable[BaseModel]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable description of a tool available to an agent.

    Attributes:
        name: Logical tool identifier (must match across versions for upgrades).
        version: Positive integer — incremented on breaking schema changes.
        description: Human-readable description shown to the LLM and dashboards.
        input_model: Pydantic model used to validate raw arguments.
        output_model: Pydantic model used to validate the handler's return.
        handler: The async callable implementing the tool's behaviour.
        opa_path: Decision path for OPA. ``None`` skips policy enforcement
            (suitable for read-only or system tools).
    """

    name: str
    version: int
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    opa_path: str | None

    def openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


@runtime_checkable
class Tool(Protocol):
    """Structural type for anything an agent can advertise to its LLM."""

    name: str
    version: int

    def openai_schema(self) -> dict[str, Any]: ...


__all__ = ["Tool", "ToolHandler", "ToolSpec"]
```

- [ ] **Step 4.5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/tools/test_spec.py -q
```
Expected: 4 passed.

- [ ] **Step 4.6: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/tools tests/unit/tools
.venv/bin/python -m mypy src/ai_core/tools tests/unit/tools
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: clean; total `<= 23`.

- [ ] **Step 4.7: Commit**

```bash
git add src/ai_core/tools/__init__.py src/ai_core/tools/spec.py tests/unit/tools/test_spec.py
git commit -m "feat(tools): add ToolSpec dataclass and Tool Protocol"
```

---

## Task 5 — `tools/decorator.py` — the `@tool` decorator

**Files:**
- Create: `src/ai_core/tools/decorator.py`
- Modify: `src/ai_core/tools/__init__.py` (re-export `tool`)
- Test: `tests/unit/tools/test_decorator.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/unit/tools/test_decorator.py`:

```python
"""Tests for the @tool decorator."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from ai_core.tools import ToolSpec, tool

pytestmark = pytest.mark.unit


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    n: int


def test_decorator_returns_toolspec_with_inferred_models():
    @tool(name="x", version=1, description="docstring overridden")
    async def fn(payload: _In) -> _Out:
        return _Out(n=0)

    assert isinstance(fn, ToolSpec)
    assert fn.name == "x"
    assert fn.version == 1
    assert fn.description == "docstring overridden"
    assert fn.input_model is _In
    assert fn.output_model is _Out
    assert fn.opa_path == "eaap/agent/tool_call/allow"


def test_description_falls_back_to_docstring():
    @tool(name="x", version=1)
    async def fn(payload: _In) -> _Out:
        """The real description."""
        return _Out(n=0)

    assert fn.description == "The real description."


def test_description_falls_back_to_empty_when_no_docstring():
    @tool(name="x", version=1)
    async def fn(payload: _In) -> _Out:
        return _Out(n=0)

    assert fn.description == ""


def test_decorator_rejects_sync_function():
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)  # type: ignore[arg-type]
        def fn(payload: _In) -> _Out:
            return _Out(n=0)
    assert "async" in str(exc.value).lower()


def test_decorator_rejects_non_basemodel_input():
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)
        async def fn(payload: dict) -> _Out:  # type: ignore[type-arg]
            return _Out(n=0)
    assert "BaseModel" in str(exc.value)


def test_decorator_rejects_non_basemodel_return():
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)
        async def fn(payload: _In) -> dict:  # type: ignore[type-arg]
            return {}
    assert "BaseModel" in str(exc.value)


def test_decorator_rejects_zero_or_multi_param_function():
    with pytest.raises(TypeError):
        @tool(name="x", version=1)
        async def fn() -> _Out:
            return _Out(n=0)

    with pytest.raises(TypeError):
        @tool(name="x", version=1)
        async def fn2(a: _In, b: int) -> _Out:  # noqa: ARG001
            return _Out(n=0)


def test_opa_path_can_be_disabled():
    @tool(name="x", version=1, opa_path=None)
    async def fn(payload: _In) -> _Out:
        return _Out(n=0)

    assert fn.opa_path is None


@pytest.mark.asyncio
async def test_decorated_handler_round_trips():
    @tool(name="x", version=1)
    async def fn(payload: _In) -> _Out:
        return _Out(n=len(payload.q))

    out = await fn.handler(_In(q="hello"))
    assert out == _Out(n=5)
```

- [ ] **Step 5.2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/tools/test_decorator.py -q
```
Expected: ImportError on `tool`.

- [ ] **Step 5.3: Implement `tools/decorator.py`**

```python
"""``@tool`` decorator — converts an async Pydantic-typed function into a ToolSpec.

The decorator enforces a small contract at definition time so that runtime
invocation never has to ask "is this even shaped like a tool?":

* the decorated callable must be ``async``,
* it must accept exactly one positional parameter typed as a Pydantic
  ``BaseModel`` subclass,
* its return annotation must be a Pydantic ``BaseModel`` subclass.

Definition-time has zero dependency on DI, observability, or OPA — those
concerns are wired in by :class:`ai_core.tools.invoker.ToolInvoker`.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable

from pydantic import BaseModel

from ai_core.tools.spec import ToolHandler, ToolSpec


def tool(
    *,
    name: str,
    version: int = 1,
    description: str | None = None,
    opa_path: str | None = "eaap/agent/tool_call/allow",
) -> Callable[[ToolHandler], ToolSpec]:
    """Decorate an async function to expose it as a SDK tool.

    Args:
        name: Logical tool identifier.
        version: Positive integer schema version (must be ``>= 1``).
        description: Optional human-readable description. If omitted, the
            decorated function's docstring is used (or ``""`` if absent).
        opa_path: OPA decision path consulted before the handler runs. Pass
            ``None`` to skip policy enforcement.

    Returns:
        A decorator that consumes the function and returns a :class:`ToolSpec`.

    Raises:
        TypeError: If the function is not async, is not single-arg, or its
            input/return annotations are not Pydantic ``BaseModel`` subclasses.
        ValueError: If ``name`` is empty or ``version`` is less than 1.
    """
    if not name:
        raise ValueError("tool(): name must be non-empty")
    if version < 1:
        raise ValueError("tool(): version must be >= 1")

    def decorate(fn: ToolHandler) -> ToolSpec:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@tool requires an async function; '{fn.__name__}' is sync."
            )

        sig = inspect.signature(fn)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(params) != 1:
            raise TypeError(
                f"@tool requires exactly one positional parameter; "
                f"'{fn.__name__}' has {len(params)}."
            )

        # Resolve string annotations (e.g. when `from __future__ import annotations`
        # is in use in the user's module).
        try:
            hints = typing.get_type_hints(fn)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(
                f"@tool could not resolve type hints for '{fn.__name__}': {exc}"
            ) from exc

        param_name = params[0].name
        input_type = hints.get(param_name)
        if not (isinstance(input_type, type) and issubclass(input_type, BaseModel)):
            raise TypeError(
                f"@tool '{fn.__name__}' parameter '{param_name}' must be "
                f"annotated with a Pydantic BaseModel subclass; got {input_type!r}."
            )

        output_type = hints.get("return")
        if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
            raise TypeError(
                f"@tool '{fn.__name__}' return annotation must be a Pydantic "
                f"BaseModel subclass; got {output_type!r}."
            )

        resolved_description = description
        if resolved_description is None:
            doc = inspect.getdoc(fn) or ""
            resolved_description = doc.strip()

        return ToolSpec(
            name=name,
            version=version,
            description=resolved_description,
            input_model=input_type,
            output_model=output_type,
            handler=fn,
            opa_path=opa_path,
        )

    return decorate


__all__ = ["tool"]
```

- [ ] **Step 5.4: Re-export from package `__init__`**

Replace the contents of `src/ai_core/tools/__init__.py` with:

```python
"""Tool authoring primitives — the @tool decorator, ToolSpec, and Tool Protocol."""

from __future__ import annotations

from ai_core.tools.decorator import tool
from ai_core.tools.spec import Tool, ToolHandler, ToolSpec

__all__ = ["Tool", "ToolHandler", "ToolSpec", "tool"]
```

- [ ] **Step 5.5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/tools/test_decorator.py -q
```
Expected: 9 passed.

- [ ] **Step 5.6: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/tools tests/unit/tools
.venv/bin/python -m mypy src/ai_core/tools tests/unit/tools
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: clean; total `<= 23`.

- [ ] **Step 5.7: Commit**

```bash
git add src/ai_core/tools/__init__.py src/ai_core/tools/decorator.py tests/unit/tools/test_decorator.py
git commit -m "feat(tools): add @tool decorator producing ToolSpec from async Pydantic-typed functions"
```

---

## Task 6 — `tools/invoker.py` — `ToolInvoker` runtime pipeline

**Files:**
- Create: `src/ai_core/tools/invoker.py`
- Modify: `src/ai_core/tools/__init__.py` (re-export `ToolInvoker`)
- Test: `tests/unit/tools/test_invoker.py`

- [ ] **Step 6.1: Write the failing tests**

Create `tests/unit/tools/test_invoker.py`:

```python
"""Tests for the ToolInvoker pipeline."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from ai_core.di.interfaces import PolicyDecision
from ai_core.exceptions import (
    PolicyDenialError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.schema.registry import SchemaRegistry
from ai_core.tools import tool
from ai_core.tools.invoker import ToolInvoker

pytestmark = pytest.mark.unit


class _In(BaseModel):
    q: str
    limit: int = Field(default=10, ge=1)


class _Out(BaseModel):
    items: list[str]


@tool(name="search", version=1, description="d")
async def _search(payload: _In) -> _Out:
    return _Out(items=[payload.q] * payload.limit)


def _invoker(fake_observability, fake_policy_evaluator_factory, *, allow=True, reason=None):
    return ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(default_allow=allow, reason=reason),
        registry=SchemaRegistry(),
    )


@pytest.mark.asyncio
async def test_happy_path(fake_observability, fake_policy_evaluator_factory):
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    result = await inv.invoke(_search, {"q": "hi", "limit": 2}, agent_id="a", tenant_id="t")
    assert result == {"items": ["hi", "hi"]}
    assert [s.name for s in fake_observability.spans] == ["tool.invoke"]
    span = fake_observability.spans[0]
    assert span.attributes["tool.name"] == "search"
    assert span.attributes["tool.version"] == 1
    assert span.attributes["agent_id"] == "a"
    assert span.attributes["tenant_id"] == "t"
    assert ("tool.completed", {"tool.name": "search", "tool.version": 1}) in [
        (n, dict(a)) for n, a in fake_observability.events
    ]


@pytest.mark.asyncio
async def test_input_validation_failure(fake_observability, fake_policy_evaluator_factory):
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolValidationError) as exc:
        await inv.invoke(_search, {"q": "x", "limit": -1})
    assert exc.value.details["side"] == "input"
    assert exc.value.details["tool"] == "search"
    assert exc.value.details["version"] == 1


@pytest.mark.asyncio
async def test_input_validation_runs_before_opa(fake_observability, fake_policy_evaluator_factory):
    """Input failure must short-circuit before OPA is consulted."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    policy = inv._policy  # type: ignore[attr-defined]  # introspection for the test
    with pytest.raises(ToolValidationError):
        await inv.invoke(_search, {"q": "x", "limit": -1})
    assert policy.calls == []  # OPA never called


@pytest.mark.asyncio
async def test_opa_deny_raises_policy_denial(fake_observability, fake_policy_evaluator_factory):
    inv = _invoker(
        fake_observability, fake_policy_evaluator_factory, allow=False, reason="denied"
    )
    with pytest.raises(PolicyDenialError) as exc:
        await inv.invoke(_search, {"q": "x", "limit": 1})
    assert exc.value.details["tool"] == "search"
    assert "denied" in exc.value.message.lower() or exc.value.details.get("reason") == "denied"


@pytest.mark.asyncio
async def test_handler_raise_wraps_as_execution_error(
    fake_observability, fake_policy_evaluator_factory
):
    @tool(name="boom", version=1)
    async def boom(payload: _In) -> _Out:
        raise RuntimeError("kaboom")

    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolExecutionError) as exc:
        await inv.invoke(boom, {"q": "x", "limit": 1}, agent_id="a")
    assert exc.value.details["tool"] == "boom"
    assert exc.value.details["agent_id"] == "a"
    assert isinstance(exc.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_output_validation_failure(fake_observability, fake_policy_evaluator_factory):
    @tool(name="lying", version=1)
    async def lying(payload: _In) -> _Out:
        # Cast around mypy: hand back a non-conforming dict so the invoker
        # has to validate-and-fail.
        return {"wrong": True}  # type: ignore[return-value]

    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolValidationError) as exc:
        await inv.invoke(lying, {"q": "x", "limit": 1})
    assert exc.value.details["side"] == "output"
    assert exc.value.details["tool"] == "lying"


@pytest.mark.asyncio
async def test_opa_path_none_skips_policy(fake_observability, fake_policy_evaluator_factory):
    @tool(name="public", version=1, opa_path=None)
    async def public(payload: _In) -> _Out:
        return _Out(items=[])

    inv = _invoker(
        fake_observability, fake_policy_evaluator_factory, allow=False, reason="denied"
    )
    # Even though OPA is wired to deny, opa_path=None bypasses it.
    result = await inv.invoke(public, {"q": "x", "limit": 1})
    assert result == {"items": []}


@pytest.mark.asyncio
async def test_policy_none_skips_opa(fake_observability):
    inv = ToolInvoker(observability=fake_observability, policy=None, registry=None)
    result = await inv.invoke(_search, {"q": "hi", "limit": 1})
    assert result == {"items": ["hi"]}


@pytest.mark.asyncio
async def test_register_with_registry_is_idempotent(
    fake_observability, fake_policy_evaluator_factory
):
    registry = SchemaRegistry()
    inv = ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(),
        registry=registry,
    )
    inv.register(_search)
    inv.register(_search)  # idempotent — does NOT raise
    rec = registry.get("search", version=1)
    assert rec.input_schema is _In
    assert rec.output_schema is _Out


@pytest.mark.asyncio
async def test_span_records_exception_on_handler_raise(
    fake_observability, fake_policy_evaluator_factory
):
    @tool(name="explode", version=1)
    async def explode(payload: _In) -> _Out:
        raise RuntimeError("kaboom")

    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolExecutionError):
        await inv.invoke(explode, {"q": "x", "limit": 1})
    assert fake_observability.spans[0].exception is not None


@pytest.mark.asyncio
async def test_pipeline_order_input_then_opa_then_handler(
    fake_observability, fake_policy_evaluator_factory
):
    """When all stages succeed, exactly one OPA call is made for an allowed verdict."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    await inv.invoke(_search, {"q": "x", "limit": 1}, principal={"sub": "u1"})
    policy = inv._policy  # type: ignore[attr-defined]
    assert len(policy.calls) == 1
    call = policy.calls[0]
    assert call.decision_path == "eaap/agent/tool_call/allow"
    assert call.input["tool"] == "search"
    assert call.input["payload"] == {"q": "x", "limit": 1}
    assert call.input["user"] == {"sub": "u1"}
```

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/tools/test_invoker.py -q
```
Expected: ImportError on `ToolInvoker`.

- [ ] **Step 6.3: Implement `tools/invoker.py`**

```python
"""``ToolInvoker`` — runs the schema → OPA → span → handler pipeline.

The invoker is constructed once at app boot (via the DI container) and
holds references to:

* an :class:`IObservabilityProvider` for spans + events,
* an optional :class:`IPolicyEvaluator` for OPA enforcement,
* an optional :class:`SchemaRegistry` for cross-tool discovery.

It is stateless w.r.t. specs — each :py:meth:`invoke` call takes the spec
as a parameter so the same invoker handles every tool in the application.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from ai_core.di.interfaces import IObservabilityProvider, IPolicyEvaluator
from ai_core.exceptions import (
    PolicyDenialError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.schema.registry import SchemaRegistry
from ai_core.tools.spec import ToolSpec

_logger = logging.getLogger(__name__)


class ToolInvoker:
    """Runs ``ToolSpec`` instances through the SDK's standard tool-call pipeline.

    Pipeline:

    1. Validate ``raw_args`` -> ``spec.input_model`` (``ToolValidationError`` on fail).
    2. If ``spec.opa_path`` and a policy evaluator is wired, evaluate; deny -> ``PolicyDenialError``.
    3. Open OTel span ``"tool.invoke"`` carrying tool/version/agent/tenant attributes.
    4. ``await spec.handler(payload)``; raise -> ``ToolExecutionError`` chained.
    5. Validate output via ``spec.output_model.model_validate`` (``ToolValidationError`` on fail).
    6. Emit ``"tool.completed"`` event and return ``output.model_dump()``.
    """

    def __init__(
        self,
        *,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator | None = None,
        registry: SchemaRegistry | None = None,
    ) -> None:
        self._observability = observability
        self._policy = policy
        self._registry = registry

    def register(self, spec: ToolSpec) -> None:
        """Register a spec with the underlying :class:`SchemaRegistry`. Idempotent.

        No-op when this invoker was constructed without a registry.
        """
        if self._registry is None:
            return
        try:
            self._registry.register(
                spec.name,
                spec.version,
                input_schema=spec.input_model,
                output_schema=spec.output_model,
                description=spec.description,
            )
        except Exception:  # noqa: BLE001
            # Already registered with the same (name, version): treat as idempotent.
            existing = self._registry.get(spec.name, version=spec.version)
            if (
                existing.input_schema is not spec.input_model
                or existing.output_schema is not spec.output_model
            ):
                raise

    async def invoke(
        self,
        spec: ToolSpec,
        raw_args: Mapping[str, Any],
        *,
        principal: Mapping[str, Any] | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Mapping[str, Any]:
        # ----- 1. Input validation -------------------------------------------------
        try:
            payload = spec.input_model.model_validate(dict(raw_args))
        except ValidationError as exc:
            raise ToolValidationError(
                f"Tool '{spec.name}' v{spec.version} input failed validation.",
                details={
                    "tool": spec.name,
                    "version": spec.version,
                    "side": "input",
                    "errors": exc.errors(),
                },
                cause=exc,
            ) from exc

        # ----- 2. OPA enforcement --------------------------------------------------
        if spec.opa_path is not None and self._policy is not None:
            decision = await self._policy.evaluate(
                decision_path=spec.opa_path,
                input={
                    "tool": spec.name,
                    "version": spec.version,
                    "payload": payload.model_dump(),
                    "user": dict(principal or {}),
                    "agent_id": agent_id,
                    "tenant_id": tenant_id,
                },
            )
            if not decision.allowed:
                raise PolicyDenialError(
                    f"Tool '{spec.name}' v{spec.version} denied by policy: "
                    f"{decision.reason or 'no reason provided'}",
                    details={
                        "tool": spec.name,
                        "version": spec.version,
                        "reason": decision.reason,
                        "agent_id": agent_id,
                        "tenant_id": tenant_id,
                    },
                )

        # ----- 3. Span + 4. handler call -----------------------------------------
        attrs: dict[str, Any] = {
            "tool.name": spec.name,
            "tool.version": spec.version,
            "agent_id": agent_id or "",
            "tenant_id": tenant_id or "",
        }
        async with self._observability.start_span("tool.invoke", attributes=attrs):
            try:
                result: Any = await spec.handler(payload)
            except Exception as exc:  # noqa: BLE001
                raise ToolExecutionError(
                    f"Tool '{spec.name}' v{spec.version} failed: {exc}",
                    details={
                        "tool": spec.name,
                        "version": spec.version,
                        "agent_id": agent_id,
                        "tenant_id": tenant_id,
                    },
                    cause=exc,
                ) from exc

        # ----- 5. Output validation ------------------------------------------------
        try:
            validated: BaseModel = spec.output_model.model_validate(result)
        except ValidationError as exc:
            _logger.warning(
                "Tool '%s' v%s returned a non-conforming object; this is a handler bug.",
                spec.name,
                spec.version,
            )
            raise ToolValidationError(
                f"Tool '{spec.name}' v{spec.version} returned invalid data.",
                details={
                    "tool": spec.name,
                    "version": spec.version,
                    "side": "output",
                    "errors": exc.errors(),
                },
                cause=exc,
            ) from exc

        # ----- 6. Completion event -------------------------------------------------
        await self._observability.record_event(
            "tool.completed",
            attributes={"tool.name": spec.name, "tool.version": spec.version},
        )
        return validated.model_dump()


__all__ = ["ToolInvoker"]
```

- [ ] **Step 6.4: Re-export `ToolInvoker`**

Update `src/ai_core/tools/__init__.py`:

```python
"""Tool authoring primitives — the @tool decorator, ToolSpec, ToolInvoker."""

from __future__ import annotations

from ai_core.tools.decorator import tool
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.spec import Tool, ToolHandler, ToolSpec

__all__ = ["Tool", "ToolHandler", "ToolInvoker", "ToolSpec", "tool"]
```

- [ ] **Step 6.5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/tools/test_invoker.py -q
```
Expected: 11 passed.

- [ ] **Step 6.6: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/tools tests/unit/tools
.venv/bin/python -m mypy src/ai_core/tools tests/unit/tools
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: clean; total `<= 23`.

- [ ] **Step 6.7: Commit**

```bash
git add src/ai_core/tools/__init__.py src/ai_core/tools/invoker.py tests/unit/tools/test_invoker.py
git commit -m "feat(tools): add ToolInvoker — schema/OPA/span/handler/output pipeline"
```

---

## Task 7 — Bind `ToolInvoker` in the DI graph

**Files:**
- Modify: `src/ai_core/di/module.py`
- Test: `tests/unit/di/test_module_tool_invoker.py`

- [ ] **Step 7.1: Write the failing test**

Create `tests/unit/di/test_module_tool_invoker.py`:

```python
"""Tests that the DI graph resolves ToolInvoker as a singleton."""
from __future__ import annotations

import pytest

from ai_core.di import AgentModule, Container
from ai_core.tools.invoker import ToolInvoker

pytestmark = pytest.mark.unit


def test_container_resolves_tool_invoker():
    container = Container.build([AgentModule()])
    invoker = container.get(ToolInvoker)
    assert isinstance(invoker, ToolInvoker)


def test_tool_invoker_is_singleton():
    container = Container.build([AgentModule()])
    a = container.get(ToolInvoker)
    b = container.get(ToolInvoker)
    assert a is b
```

- [ ] **Step 7.2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/di/test_module_tool_invoker.py -q
```
Expected: `DependencyResolutionError`.

- [ ] **Step 7.3: Add the provider to `AgentModule`**

In `src/ai_core/di/module.py`, add this import alongside the existing tool/registry imports:

```python
from ai_core.tools.invoker import ToolInvoker
```

Add this provider method on `AgentModule` (place it near the existing `provide_observability` / `provide_policy_evaluator` block — anywhere within the class):

```python
    @singleton
    @provider
    def provide_tool_invoker(
        self,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator,
        registry: SchemaRegistry,
    ) -> ToolInvoker:
        """Return the singleton :class:`ToolInvoker` wired to the SDK's services."""
        return ToolInvoker(observability=observability, policy=policy, registry=registry)
```

- [ ] **Step 7.4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/di/test_module_tool_invoker.py tests/unit/di -q
```
Expected: 2 new passes; existing DI tests still pass.

- [ ] **Step 7.5: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/di/module.py tests/unit/di/test_module_tool_invoker.py
.venv/bin/python -m mypy src/ai_core/di/module.py tests/unit/di/test_module_tool_invoker.py
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: clean; total `<= 23`.

- [ ] **Step 7.6: Commit**

```bash
git add src/ai_core/di/module.py tests/unit/di/test_module_tool_invoker.py
git commit -m "feat(di): bind ToolInvoker as a singleton in AgentModule"
```

---

## Task 8 — `BaseAgent` integration: inject `ToolInvoker`, accept `Tool` objects, auto-install tool node

**Files:**
- Modify: `src/ai_core/agents/base.py`
- Test: `tests/component/test_agent_tool_loop.py` (new)

This is the largest change. It touches the agent graph topology and is exercised by component-level tests.

### 8a — Inject `ToolInvoker` and accept `Tool`-protocol tools

- [ ] **Step 8.1: Write a failing component test for the auto-install**

Create `tests/component/test_agent_tool_loop.py`:

```python
"""Component tests for BaseAgent + @tool integration."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel, Field

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import (
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
    LLMResponse,
    LLMUsage,
)
from ai_core.exceptions import AgentRecursionLimitError
from ai_core.tools import Tool, tool
from injector import Module, provider, singleton

pytestmark = pytest.mark.component


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    n: int = Field(..., ge=0)


@tool(name="count", version=1)
async def count_tool(payload: _In) -> _Out:
    """Return the length of the query string."""
    return _Out(n=len(payload.q))


class _ScriptedLLM(ILLMClient):
    """Returns a queue of pre-baked completions."""

    def __init__(self, scripts: Sequence[LLMResponse]) -> None:
        self._scripts = list(scripts)
        self.calls: list[Sequence[Mapping[str, Any]]] = []

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
        self.calls.append(list(messages))
        return self._scripts.pop(0)


def _llm_msg(content: str = "", tool_calls: Sequence[Mapping[str, Any]] = ()) -> LLMResponse:
    return LLMResponse(
        model="fake",
        content=content,
        tool_calls=list(tool_calls),
        usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.0),
        raw={},
    )


class _DemoAgent(BaseAgent):
    agent_id = "demo"
    _tools: tuple[Tool, ...] = (count_tool,)

    def system_prompt(self) -> str:
        return "You are a counting agent."

    def tools(self) -> Sequence[Tool]:
        return self._tools


def _build(llm: ILLMClient, fake_observability, fake_policy_evaluator_factory):
    class _Fakes(Module):
        @singleton
        @provider
        def llm(self) -> ILLMClient:
            return llm

        @singleton
        @provider
        def obs(self) -> IObservabilityProvider:
            return fake_observability

        @singleton
        @provider
        def policy(self) -> IPolicyEvaluator:
            return fake_policy_evaluator_factory(default_allow=True)

    return Container.build([AgentModule(), _Fakes()])


@pytest.mark.asyncio
async def test_agent_with_tool_runs_loop_and_returns_final_answer(
    fake_observability, fake_policy_evaluator_factory
):
    # First LLM turn requests a tool call; second returns the final answer.
    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": '{"q":"hello"}'},
        }]),
        _llm_msg(content="The count is 5."),
    ])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count hello"}])
    final = state["messages"][-1]
    final_content = getattr(final, "content", None) or final["content"]
    assert "5" in final_content


@pytest.mark.asyncio
async def test_agent_without_tool_calls_terminates(
    fake_observability, fake_policy_evaluator_factory
):
    llm = _ScriptedLLM([_llm_msg(content="Hi.")])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "hello"}])
    final = state["messages"][-1]
    final_content = getattr(final, "content", None) or final["content"]
    assert final_content == "Hi."


@pytest.mark.asyncio
async def test_tool_validation_error_surfaces_as_toolmessage(
    fake_observability, fake_policy_evaluator_factory
):
    # LLM passes invalid args (q is missing); LLM then explains.
    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": '{}'},
        }]),
        _llm_msg(content="I see — I need a query."),
    ])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count"}])
    # The second LLM call must have seen a tool message describing the error.
    tool_msg_seen = False
    for messages in llm.calls:
        for m in messages:
            if (m.get("role") == "tool") and "validation" in (m.get("content") or "").lower():
                tool_msg_seen = True
    assert tool_msg_seen, "Expected a ToolMessage describing the validation failure."


@pytest.mark.asyncio
async def test_policy_denial_surfaces_as_toolmessage(
    fake_observability, fake_policy_evaluator_factory
):
    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": '{"q":"x"}'},
        }]),
        _llm_msg(content="Ok, can't run that."),
    ])

    class _Fakes(Module):
        @singleton
        @provider
        def llm(self) -> ILLMClient:
            return llm

        @singleton
        @provider
        def obs(self) -> IObservabilityProvider:
            return fake_observability

        @singleton
        @provider
        def policy(self) -> IPolicyEvaluator:
            return fake_policy_evaluator_factory(default_allow=False, reason="not allowed")

    container = Container.build([AgentModule(), _Fakes()])
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count x"}])
    final = state["messages"][-1]
    content = getattr(final, "content", None) or final["content"]
    assert "can't run" in content.lower() or "ok" in content.lower()


@pytest.mark.asyncio
async def test_auto_tool_loop_can_be_disabled(
    fake_observability, fake_policy_evaluator_factory
):
    class _NoLoopAgent(_DemoAgent):
        auto_tool_loop = False

    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": '{"q":"x"}'},
        }]),
    ])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_NoLoopAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count x"}])
    # No second LLM call — graph terminated after the first turn.
    assert len(llm.calls) == 1
    # Final message has the unanswered tool_call attached.
    last = state["messages"][-1]
    tc = getattr(last, "tool_calls", None) or (last.get("tool_calls") if isinstance(last, dict) else None)
    assert tc, "Expected the unanswered tool_calls to remain on the final message."
```

- [ ] **Step 8.2: Run the test to verify it fails**

```bash
mkdir -p tests/component
.venv/bin/python -m pytest tests/component/test_agent_tool_loop.py -q
```
Expected: failures referencing missing `ToolInvoker` injection or missing tool dispatch node.

- [ ] **Step 8.3: Modify `src/ai_core/agents/base.py`**

Open the file. Update the imports (top of file, replace the existing import block):

```python
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from injector import inject
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from opentelemetry import baggage
from opentelemetry import context as otel_context

from ai_core.agents.memory import IMemoryManager, to_openai_messages
from ai_core.agents.state import AgentState, new_agent_state
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import ILLMClient, IObservabilityProvider
from ai_core.exceptions import (
    AgentRecursionLimitError,
    PolicyDenialError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.spec import Tool, ToolSpec

_logger = logging.getLogger(__name__)
```

Update `BaseAgent.__init__` signature:

```python
    @inject
    def __init__(
        self,
        settings: AppSettings,
        llm: ILLMClient,
        memory: IMemoryManager,
        observability: IObservabilityProvider,
        tool_invoker: ToolInvoker,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._memory = memory
        self._observability = observability
        self._tool_invoker = tool_invoker
        self._graph: Any | None = None
```

Add a class-level toggle just below `agent_id`:

```python
    #: When True, ``compile()`` auto-installs a tool dispatch loop whenever
    #: ``tools()`` returns at least one ``Tool``-protocol object. Subclasses
    #: that prefer to wire their own tool handling can set this to False.
    auto_tool_loop: bool = True
```

Update the `tools()` signature so subclasses can return either form (Phase 1 picks the union):

```python
    def tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Return tool definitions or ``Tool``-protocol objects (default: empty)."""
        return ()
```

Replace the `compile()` method body with one that branches on tool kinds:

```python
    def compile(self, *, checkpointer: Any | None = None) -> Any:
        if self._graph is not None:
            return self._graph
        graph: StateGraph[AgentState] = StateGraph(AgentState)
        graph.add_node("compact", self._compaction_node)
        graph.add_node("agent", self._agent_node)

        sdk_tools = [t for t in self.tools() if isinstance(t, ToolSpec)]
        install_loop = self.auto_tool_loop and bool(sdk_tools)

        if install_loop:
            graph.add_node("tool", self._tool_node)
            graph.add_conditional_edges(
                START,
                self._router_should_compact,
                {True: "compact", False: "agent"},
            )
            graph.add_edge("compact", "agent")
            graph.add_conditional_edges(
                "agent",
                self._router_after_agent,
                {True: "tool", False: END},
            )
            graph.add_edge("tool", "agent")
        else:
            graph.add_conditional_edges(
                START,
                self._router_should_compact,
                {True: "compact", False: "agent"},
            )
            graph.add_edge("compact", "agent")
            graph.add_edge("agent", END)

        self.extend_graph(graph)
        self._graph = graph.compile(checkpointer=checkpointer)
        return self._graph
```

Replace `_agent_node` so its tool list passes through the SDK tools' `openai_schema()`:

```python
    async def _agent_node(self, state: AgentState) -> AgentState:
        history = list(state.get("messages") or [])
        prompt: list[Mapping[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            *to_openai_messages(history),
        ]
        essentials = state.get("essential_entities") or {}

        tool_payload: list[Mapping[str, Any]] = []
        for t in self.tools():
            if isinstance(t, ToolSpec):
                tool_payload.append(t.openai_schema())
            elif isinstance(t, Mapping):
                tool_payload.append(t)
            # Anything else satisfies the Protocol structurally — render its schema.
            elif hasattr(t, "openai_schema"):
                tool_payload.append(t.openai_schema())

        response = await self._llm.complete(
            model=None,
            messages=prompt,
            tools=tool_payload or None,
            tenant_id=str(essentials.get("tenant_id") or "") or None,
            agent_id=self.agent_id,
        )
        appended: list[Any]
        if response.tool_calls:
            appended = [AIMessage(
                content=response.content,
                tool_calls=[
                    {
                        "id": tc.get("id") or f"call-{i}",
                        "name": tc.get("function", {}).get("name", ""),
                        "args": json.loads(tc.get("function", {}).get("arguments") or "{}"),
                    }
                    for i, tc in enumerate(response.tool_calls)
                ],
            )]
        else:
            appended = [AIMessage(content=response.content)]
        return AgentState(
            messages=appended,
            token_count=response.usage.prompt_tokens + response.usage.completion_tokens,
        )
```

Add the new tool-dispatch node and routers below `_agent_node`:

```python
    async def _tool_node(self, state: AgentState) -> AgentState:
        """Dispatch all tool calls on the most recent assistant message."""
        history = list(state.get("messages") or [])
        last = history[-1] if history else None
        tool_calls = getattr(last, "tool_calls", None) or []
        sdk_tools_by_name: dict[str, ToolSpec] = {
            t.name: t for t in self.tools() if isinstance(t, ToolSpec)
        }
        essentials = state.get("essential_entities") or {}
        tenant_id = str(essentials.get("tenant_id") or "") or None

        appended: list[Any] = []
        for tc in tool_calls:
            tc_id = tc.get("id") if isinstance(tc, Mapping) else getattr(tc, "id", "")
            name = tc.get("name") if isinstance(tc, Mapping) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, Mapping) else getattr(tc, "args", {}) or {}
            spec = sdk_tools_by_name.get(name)
            if spec is None:
                appended.append(ToolMessage(
                    content=f"Unknown tool '{name}'.",
                    tool_call_id=tc_id or "",
                    name=name or "",
                ))
                continue
            try:
                result = await self._tool_invoker.invoke(
                    spec,
                    args if isinstance(args, Mapping) else {},
                    agent_id=self.agent_id,
                    tenant_id=tenant_id,
                )
                appended.append(ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tc_id or "",
                    name=name,
                ))
            except ToolValidationError as exc:
                appended.append(ToolMessage(
                    content=(
                        f"Validation failed for '{name}': "
                        f"{exc.message} ({exc.details.get('errors', [])[:1]})"
                    ),
                    tool_call_id=tc_id or "",
                    name=name,
                ))
            except PolicyDenialError as exc:
                appended.append(ToolMessage(
                    content=f"Tool '{name}' denied by policy: {exc.details.get('reason') or exc.message}",
                    tool_call_id=tc_id or "",
                    name=name,
                ))
            except ToolExecutionError as exc:
                _logger.error(
                    "Tool '%s' execution error", name, exc_info=exc.__cause__,
                )
                appended.append(ToolMessage(
                    content=f"Tool '{name}' failed: {exc.message}",
                    tool_call_id=tc_id or "",
                    name=name,
                ))
        return AgentState(messages=appended)

    def _router_after_agent(self, state: AgentState) -> bool:
        """True -> there is at least one outstanding tool_call to dispatch."""
        history = list(state.get("messages") or [])
        last = history[-1] if history else None
        return bool(getattr(last, "tool_calls", None))
```

- [ ] **Step 8.4: Run component tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/component/test_agent_tool_loop.py -q
```
Expected: 5 passed. (If LangGraph's `add_messages` reducer normalises message types differently, adjust `tool_calls` extraction accordingly — the test makes this debuggable by asserting on raw content.)

- [ ] **Step 8.5: Run full unit + component suite**

```bash
.venv/bin/python -m pytest tests/unit tests/component -q
```
Expected: all green; no regressions.

- [ ] **Step 8.6: Type-check + lint**

Note: `agents/base.py` had 4 pre-existing mypy strict errors (`StateGraph` generic args, `list()` arg type, `no-any-return`). The edit must NOT add new errors. The rewritten `compile()` already provides a generic to `StateGraph[AgentState]`, which fixes 2 of those 4. Acceptable to leave the rest.

```bash
.venv/bin/python -m ruff check src/ai_core/agents/base.py tests/component/test_agent_tool_loop.py
.venv/bin/python -m mypy src/ai_core/agents/base.py tests/component/test_agent_tool_loop.py
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: ruff clean; mypy on touched files reports at most the same number of errors that were present before this task; project total `<= 23`.

- [ ] **Step 8.7: Commit**

```bash
git add src/ai_core/agents/base.py tests/component/test_agent_tool_loop.py
git commit -m "feat(agents): auto-install tool dispatch loop when @tool tools are returned"
```

---

## Task 9 — `app/runtime.py` — `AICoreApp` facade

**Files:**
- Create: `src/ai_core/app/__init__.py`
- Create: `src/ai_core/app/runtime.py`
- Test: `tests/unit/app/test_runtime.py`

- [ ] **Step 9.1: Write failing tests**

Create `tests/unit/app/test_runtime.py`:

```python
"""Tests for the AICoreApp facade."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from injector import Module, provider, singleton

from ai_core.app import AICoreApp, HealthSnapshot
from ai_core.config.settings import AgentSettings, AppSettings, LLMSettings
from ai_core.di.interfaces import (
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
    LLMResponse,
    LLMUsage,
)
from ai_core.exceptions import ConfigurationError
from ai_core.tools.invoker import ToolInvoker

pytestmark = pytest.mark.unit


def _bad_settings() -> AppSettings:
    return AppSettings(
        llm=LLMSettings(default_model=""),
        agent=AgentSettings(
            memory_compaction_token_threshold=512,
            memory_compaction_target_tokens=10000,
        ),
    )


class _StubLLM(ILLMClient):
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
        return LLMResponse(
            model="stub",
            content="ok",
            tool_calls=[],
            usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0),
            raw={},
        )


def _override_module(fake_observability, fake_policy_evaluator_factory) -> Module:
    class _M(Module):
        @singleton
        @provider
        def llm(self) -> ILLMClient:
            return _StubLLM()

        @singleton
        @provider
        def obs(self) -> IObservabilityProvider:
            return fake_observability

        @singleton
        @provider
        def policy(self) -> IPolicyEvaluator:
            return fake_policy_evaluator_factory(default_allow=True)

    return _M()


@pytest.mark.asyncio
async def test_aenter_runs_validation_and_builds_container(
    fake_observability, fake_policy_evaluator_factory
):
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        assert app.settings is not None
        assert app.observability is fake_observability
        assert app.policy_evaluator is not None
        assert isinstance(app.container.get(ToolInvoker), ToolInvoker)


@pytest.mark.asyncio
async def test_aenter_fails_fast_on_invalid_settings():
    app = AICoreApp(settings=_bad_settings())
    with pytest.raises(ConfigurationError) as exc:
        await app.__aenter__()
    assert "issue(s)" in exc.value.message


@pytest.mark.asyncio
async def test_aexit_is_idempotent(
    fake_observability, fake_policy_evaluator_factory
):
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    await app.__aenter__()
    await app.__aexit__(None, None, None)
    await app.__aexit__(None, None, None)  # double-close must not raise


@pytest.mark.asyncio
async def test_health_snapshot_is_ok_after_entry(
    fake_observability, fake_policy_evaluator_factory
):
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        snap = app.health
        assert isinstance(snap, HealthSnapshot)
        assert snap.status == "ok"


@pytest.mark.asyncio
async def test_register_tools_is_idempotent(
    fake_observability, fake_policy_evaluator_factory
):
    from pydantic import BaseModel
    from ai_core.tools import tool

    class _In(BaseModel):
        q: str

    class _Out(BaseModel):
        n: int

    @tool(name="x", version=1)
    async def x(p: _In) -> _Out:
        return _Out(n=0)

    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        app.register_tools(x)
        app.register_tools(x)  # idempotent — must not raise


@pytest.mark.asyncio
async def test_methods_raise_before_entry():
    app = AICoreApp()
    with pytest.raises(RuntimeError):
        _ = app.settings
    with pytest.raises(RuntimeError):
        _ = app.container
```

- [ ] **Step 9.2: Run tests to verify they fail**

```bash
mkdir -p tests/unit/app
.venv/bin/python -m pytest tests/unit/app/test_runtime.py -q
```
Expected: ImportError on `ai_core.app`.

- [ ] **Step 9.3: Implement `app/runtime.py`**

```python
"""Application-level lifecycle facade.

A consumer holds one :class:`AICoreApp`, enters it as an async context
manager, and resolves agents through :py:meth:`agent`. The app is the
canonical wiring layer between :class:`AppSettings`, the DI container, the
:class:`ComponentRegistry`, and the :class:`ToolInvoker`.

Power users may still build a :class:`Container` directly, but the facade
is the documented "pit of success" path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from injector import Module

from ai_core.config.secrets import EnvSecretManager, ISecretManager
from ai_core.config.settings import AppSettings, get_settings
from ai_core.di.container import Container
from ai_core.di.interfaces import IObservabilityProvider, IPolicyEvaluator
from ai_core.di.module import AgentModule
from ai_core.mcp.registry import ComponentRegistry, RegisteredComponent
from ai_core.mcp.transports import (
    FastMCPConnectionFactory,
    IMCPConnectionFactory,
    MCPServerSpec,
)
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.spec import ToolSpec

A = TypeVar("A")


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Coarse application health snapshot returned by :py:attr:`AICoreApp.health`."""

    status: Literal["ok", "degraded", "down"]
    components: dict[str, Literal["ok", "unknown"]]
    settings_version: str


class AICoreApp:
    """Lifecycle facade for an SDK consumer.

    Args:
        settings: Optional pre-built :class:`AppSettings`. When omitted, settings
            are loaded lazily via :func:`ai_core.config.settings.get_settings`.
        modules: Extra DI modules layered after :class:`AgentModule`. Useful for
            tests (fake providers) and for production overrides (custom
            ``ISecretManager``, alternative ``IPolicyEvaluator``).
        secret_manager: Optional :class:`ISecretManager`. Defaults to
            :class:`EnvSecretManager`.

    Use as an async context manager::

        async with AICoreApp() as app:
            agent = app.agent(MyAgent)
            state = await agent.ainvoke(messages=[...])
    """

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        modules: Sequence[Module] = (),
        secret_manager: ISecretManager | None = None,
    ) -> None:
        self._user_settings = settings
        self._user_modules = tuple(modules)
        self._user_secret_manager = secret_manager

        self._settings: AppSettings | None = None
        self._secret_manager: ISecretManager | None = None
        self._container: Container | None = None
        self._entered: bool = False
        self._closed: bool = False

    # ----- Lifecycle ----------------------------------------------------------
    async def __aenter__(self) -> AICoreApp:
        self._settings = self._user_settings or get_settings()
        self._secret_manager = self._user_secret_manager or EnvSecretManager()
        # Fail fast — collect-all validation surfaces every issue at once.
        self._settings.validate_for_runtime(secret_manager=self._secret_manager)
        self._container = Container.build([
            AgentModule(
                settings=self._settings,
                secret_manager=self._secret_manager,
            ),
            *self._user_modules,
        ])
        await self._container.start()
        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._closed or self._container is None:
            return
        await self._container.stop()
        self._closed = True

    # ----- Public API ---------------------------------------------------------
    def agent(self, cls: type[A]) -> A:
        """Resolve an agent class. ``ToolInvoker`` is auto-injected."""
        return self._require_container().get(cls)

    def register_tools(self, *specs: ToolSpec) -> None:
        """Register one or more :class:`ToolSpec` with the SDK's SchemaRegistry.

        Idempotent — registering the same spec twice is a no-op.
        """
        invoker = self._require_container().get(ToolInvoker)
        for spec in specs:
            invoker.register(spec)

    async def register_mcp(
        self,
        spec: MCPServerSpec,
        *,
        replace: bool = False,
    ) -> RegisteredComponent:
        """Register an MCP server spec with the :class:`ComponentRegistry`.

        Returns the :class:`RegisteredComponent` record so callers can
        introspect or unregister later.
        """
        container = self._require_container()
        registry = container.get(ComponentRegistry)
        factory = container.get(IMCPConnectionFactory)
        wrapper = _MCPComponent(spec=spec, factory=factory)
        return await registry.register(
            wrapper, component_type="mcp_server", replace=replace
        )

    # ----- Properties ---------------------------------------------------------
    @property
    def settings(self) -> AppSettings:
        if self._settings is None:
            raise RuntimeError("AICoreApp has not been entered yet.")
        return self._settings

    @property
    def container(self) -> Container:
        return self._require_container()

    @property
    def policy_evaluator(self) -> IPolicyEvaluator:
        return self._require_container().get(IPolicyEvaluator)

    @property
    def observability(self) -> IObservabilityProvider:
        return self._require_container().get(IObservabilityProvider)

    @property
    def health(self) -> HealthSnapshot:
        return HealthSnapshot(
            status="ok" if self._entered and not self._closed else "down",
            components={},
            settings_version=self.settings.service_name,
        )

    # ----- Internal -----------------------------------------------------------
    def _require_container(self) -> Container:
        if self._container is None:
            raise RuntimeError("AICoreApp has not been entered yet.")
        return self._container


class _MCPComponent:
    """Adapter that wraps an MCPServerSpec as an IComponent for the registry."""

    def __init__(self, *, spec: MCPServerSpec, factory: IMCPConnectionFactory) -> None:
        self.component_id = spec.component_id
        self.component_type = "mcp_server"
        self._spec = spec
        self._factory = factory

    async def health_check(self) -> bool:
        try:
            async with self._factory.open(self._spec):
                return True
        except Exception:  # noqa: BLE001
            return False


__all__ = ["AICoreApp", "HealthSnapshot"]
```

- [ ] **Step 9.4: Create `app/__init__.py`**

```python
"""Application-level facade — single-import entry point for AI engineers."""

from __future__ import annotations

from ai_core.app.runtime import AICoreApp, HealthSnapshot

__all__ = ["AICoreApp", "HealthSnapshot"]
```

- [ ] **Step 9.5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/app/test_runtime.py -q
```
Expected: 6 passed.

- [ ] **Step 9.6: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/app tests/unit/app
.venv/bin/python -m mypy src/ai_core/app tests/unit/app
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: clean; total `<= 23`.

- [ ] **Step 9.7: Commit**

```bash
git add src/ai_core/app/__init__.py src/ai_core/app/runtime.py tests/unit/app/test_runtime.py
git commit -m "feat(app): add AICoreApp lifecycle facade with collect-all settings validation"
```

---

## Task 10 — Curate top-level `src/ai_core/__init__.py` exports

**Files:**
- Modify: `src/ai_core/__init__.py`
- Test: `tests/unit/test_top_level_imports.py`

- [ ] **Step 10.1: Write the failing test**

Create `tests/unit/test_top_level_imports.py`:

```python
"""Asserts the curated top-level public surface is reachable."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_canonical_imports_exist():
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
    # Smoke: each name is non-None.
    locals_dict = locals()
    for name in [
        "AICoreApp", "AgentRecursionLimitError", "AgentRuntimeError",
        "AgentState", "BaseAgent", "BudgetExceededError", "ConfigurationError",
        "DependencyResolutionError", "EAAPBaseException", "HealthSnapshot",
        "LLMInvocationError", "PolicyDenialError", "RegistryError",
        "SchemaValidationError", "SecretResolutionError", "StorageError",
        "Tool", "ToolExecutionError", "ToolSpec", "ToolValidationError",
        "new_agent_state", "tool",
    ]:
        assert locals_dict[name] is not None


def test_version_string_is_set():
    import ai_core
    assert isinstance(ai_core.__version__, str)
    assert ai_core.__version__
```

- [ ] **Step 10.2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/test_top_level_imports.py -q
```
Expected: ImportError on at least `AICoreApp`, `tool`, `ToolSpec`.

- [ ] **Step 10.3: Update `src/ai_core/__init__.py`**

Replace the file contents with:

```python
"""ai_core — Enterprise Agentic AI Platform (EAAP) core SDK.

Curated public surface — these names are the documented entry points.
Power-user surface (Container, AgentModule, the I* interfaces) lives in
:mod:`ai_core.di.*`; per-subsystem detail (config, schema, security, mcp)
lives in its respective subpackage.

"Hello, agent" example::

    from ai_core import AICoreApp, BaseAgent, tool
    from pydantic import BaseModel

    class HiIn(BaseModel): name: str
    class HiOut(BaseModel): greeting: str

    @tool(name="say_hi", version=1)
    async def say_hi(p: HiIn) -> HiOut:
        return HiOut(greeting=f"Hi {p.name}!")

    class Greeter(BaseAgent):
        agent_id = "greeter"
        def system_prompt(self) -> str: return "You greet people."
        def tools(self): return [say_hi]

    async def main() -> None:
        async with AICoreApp() as app:
            agent = app.agent(Greeter)
            ...

"""

from __future__ import annotations

from ai_core.agents import AgentState, BaseAgent, new_agent_state
from ai_core.app import AICoreApp, HealthSnapshot
from ai_core.exceptions import (
    AgentRecursionLimitError,
    AgentRuntimeError,
    BudgetExceededError,
    ConfigurationError,
    DependencyResolutionError,
    EAAPBaseException,
    LLMInvocationError,
    PolicyDenialError,
    RegistryError,
    SchemaValidationError,
    SecretResolutionError,
    StorageError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.tools import Tool, ToolSpec, tool

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
    "SchemaValidationError",
    "ToolValidationError",
    "ToolExecutionError",
    "AgentRuntimeError",
    "AgentRecursionLimitError",
    "RegistryError",
]

__version__ = "0.1.0"
```

- [ ] **Step 10.4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/test_top_level_imports.py -q
```
Expected: 2 passed.

- [ ] **Step 10.5: Type-check + lint**

```bash
.venv/bin/python -m ruff check src/ai_core/__init__.py tests/unit/test_top_level_imports.py
.venv/bin/python -m mypy src/ai_core/__init__.py tests/unit/test_top_level_imports.py
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: clean; total `<= 23`.

- [ ] **Step 10.6: Commit**

```bash
git add src/ai_core/__init__.py tests/unit/test_top_level_imports.py
git commit -m "feat: curate top-level package exports for the Hello-Agent ergonomics"
```

---

## Task 11 — End-of-phase smoke gate

**Files:** none (verification + smoke test against `my-eaap-app`).

This task is the gate that closes Phase 1. It must pass before merging.

- [ ] **Step 11.1: Full test suite**

```bash
.venv/bin/python -m pytest tests/unit tests/component -q
```
Expected: all green.

- [ ] **Step 11.2: Lint the entire tree**

```bash
.venv/bin/python -m ruff check src tests
```
Expected: clean.

- [ ] **Step 11.3: Mypy strict — total error count must not regress**

```bash
.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: `Found N errors in M files (checked X source files)` with `N <= 23`. If `N > 23`, identify which new errors were introduced and fix them in the most recently modified file.

- [ ] **Step 11.4: Smoke against `my-eaap-app`**

```bash
cd /Users/admin-h26/EAAP/my-eaap-app
ls
.venv/bin/python -m pytest -q 2>&1 | tail -20 || true
.venv/bin/python -c "import importlib; importlib.import_module('ai_core'); print('imported ok')"
```

Expected: `my-eaap-app`'s tests pass (or its hello-world entry runs); the import smoke prints "imported ok".

If `my-eaap-app` has no tests, manually run its main entry point (look in its `pyproject.toml` for a `[project.scripts]` block, or run `.venv/bin/python -m my_eaap_app` if it exposes one).

If the smoke fails, the most likely cause is a top-level import that no longer resolves — re-run `pytest tests/unit/test_top_level_imports.py` to confirm the SDK side is healthy, then update `my-eaap-app` to use the new top-level imports.

- [ ] **Step 11.5: Capture phase summary in a commit message**

```bash
cd /Users/admin-h26/EAAP/ai-core-sdk
git log --oneline main..HEAD
```

Verify the commit graph reads as a clean phase: roughly 11 commits, one per task, conventional-commit subjects. If acceptable, push the branch:

```bash
git push -u origin feat/phase-1-facade-tool-validation
```

- [ ] **Step 11.6: Open the PR (optional — user-driven)**

Open a PR from `feat/phase-1-facade-tool-validation` into `main` with the design doc linked in the body. Phase 1 is complete when the PR is merged.

---

## Out-of-scope reminders

For traceability, here is what is **deferred to Phase 2+** and must not creep into Phase 1:

- I/O-based health probes (OPA ping, DB connect, model lookup).
- Prompt caching (Anthropic `cache_control` headers).
- MCP connection pooling.
- Audit sink for OPA decisions.
- `error_code` field on `EAAPBaseException`.
- `LLMTimeoutError`, `MCPTransportError`.
- `structlog` adoption.
- `eaap init` scaffold updates (the WIP on `main` covers some of this — kept untouched).
- Contract test suites for ABCs, Testcontainers integration tests.

If a step starts pulling work from this list, stop and confirm scope with the user.
