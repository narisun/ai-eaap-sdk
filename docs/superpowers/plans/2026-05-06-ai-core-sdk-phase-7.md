# ai-core-sdk Phase 7 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the public 29-name surface and Phase 1-6 invariants via contract tests; add Testcontainers-backed integration tests that exercise the SDK against real Postgres and OPA.

**Architecture:** Two purely-test-side deliverables, no production-code changes. (1) `tests/contract/` — five files pinning surface membership, audit/health never-raise contracts, exception `error_code` mirroring, and Container teardown order. (2) `tests/integration/` — Testcontainers fixtures (session-scoped, auto-skip when Docker unavailable) plus Postgres + OPA integration tests against the real backends.

**Tech Stack:** Python 3.11+, pytest, ruff, mypy strict. New dev dep: `testcontainers[postgres]>=4.0,<5.0`. Spec: `docs/superpowers/specs/2026-05-06-ai-core-sdk-phase-7-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-7-test-quality` (already checked out off `main` post-Phase-6-merge; carries the Phase 7 spec at `70976c9`).

**Working-state hygiene** — do NOT touch:
- `README.md` (top-level)
- `src/**` — entire src tree (Phase 7 is purely test-side)
- `tests/unit/**` — existing unit tests
- `tests/component/**` — existing component tests
- `src/ai_core/cli/templates/init/**` (the Postgres + OPA images and the policies live here, but Phase 7 only READS them, doesn't modify)

**Mypy baseline:** 21 strict errors in 8 files (post-Phase-6). Total must remain ≤ 21 after every commit.
**Ruff baseline:** 211 errors at `d5f03d5` (Phase 6 merge commit, verified). Total must remain ≤ 211.
**Pytest baseline:** 405 passing + 9 pre-existing langgraph errors (post-Phase-6).

**Per-step gate (every commit):**
- `ruff check <files-touched>` — no NEW violations.
- `pytest tests/unit tests/component tests/contract -q` — must pass (excluding the pre-existing langgraph errors).
- `pytest tests/integration -q` — must pass when Docker is up; auto-skip otherwise (verified by the controller during the smoke gate).
- `mypy src 2>&1 | tail -1` — total ≤ 21. Note: Phase 7 doesn't change `src/`, so `mypy src` should not change at all.

**Python interpreter:** `/Users/admin-h26/EAAP/.venv/bin/python` (no local venv inside the SDK).

**Pre-resolved unknowns (controller verified):**
- `OPAPolicyEvaluator` lives at `ai_core.security.opa.OPAPolicyEvaluator`.
- `IPolicyEvaluator` interface: `async def evaluate(self, *, decision_path: str, input: Mapping[str, Any]) -> PolicyDecision`. There is NO `allow(...)` method — the spec example referenced `allow(...)` and the plan corrects this to `evaluate(...)`.
- `PolicyDecision` dataclass: `allowed: bool`, `obligations: Mapping[str, Any]`, `reason: str | None`.
- 3 concrete `IHealthProbe` subclasses: `OPAReachabilityProbe` (component="opa"), `DatabaseProbe` (component="database"), `ModelLookupProbe` (component="model_lookup").
- 16 concrete `EAAPBaseException` subclasses: `ConfigurationError`, `SecretResolutionError`, `DependencyResolutionError`, `StorageError`, `CheckpointError`, `PolicyDenialError`, `LLMInvocationError`, `LLMTimeoutError`, `BudgetExceededError`, `SchemaValidationError`, `ToolValidationError`, `ToolExecutionError`, `AgentRuntimeError`, `AgentRecursionLimitError`, `RegistryError`, `MCPTransportError`.
- 3-5 concrete `IAuditSink` subclasses depending on optional deps installed: `NullAuditSink`, `JsonlFileAuditSink`, `OTelEventAuditSink`, plus `SentryAuditSink` and `DatadogAuditSink` if `[sentry]` / `[datadog]` extras are installed.
- 5 entries in `Container._teardown_sdk_resources` order: `("observability.shutdown", IObservabilityProvider, ("shutdown",))`, `("audit.flush", IAuditSink, ("flush",))`, `("mcp_pool.aclose", IMCPConnectionFactory, ("aclose",))`, `("policy_evaluator.aclose", IPolicyEvaluator, ("aclose",))`, `("engine.dispose", AsyncEngine, ("dispose",))`.

**Per-task commit message convention:** Conventional Commits.

---

## Task 1 — Contract tests

Five test files in `tests/contract/`. Independent of Tasks 2 and 3.

**Files:**
- Create: `tests/contract/__init__.py`
- Create: `tests/contract/conftest.py`
- Create: `tests/contract/test_public_surface.py`
- Create: `tests/contract/test_audit_invariants.py`
- Create: `tests/contract/test_health_invariants.py`
- Create: `tests/contract/test_exception_invariants.py`
- Create: `tests/contract/test_container_lifecycle.py`

### 1a — Skeleton

- [ ] **Step 1.1: Create the contract directory + bare files**

```bash
mkdir -p /Users/admin-h26/EAAP/ai-core-sdk/tests/contract
touch /Users/admin-h26/EAAP/ai-core-sdk/tests/contract/__init__.py
```

Create `tests/contract/conftest.py`:

```python
"""Pytest configuration for contract tests.

Contract tests pin SDK promises (public surface, never-raise contracts,
error_code mirroring, container lifecycle) at Phase 6's end. They run
in-process with no infrastructure — the Docker-conditional integration
tests live under tests/integration/.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Tag every test in this directory with pytest.mark.contract."""
    for item in items:
        item.add_marker(pytest.mark.contract)
```

(The marker `contract` is registered via the per-directory conftest. The project's existing `pyproject.toml` `[tool.pytest.ini_options]` may need a `markers` registration — checked in Step 1.2 below.)

- [ ] **Step 1.2: Register the contract marker if needed**

```bash
grep -A 5 "\[tool.pytest" /Users/admin-h26/EAAP/ai-core-sdk/pyproject.toml | head -15
```

If `markers` is configured in `[tool.pytest.ini_options]`, add `"contract: contract / surface tests"` and `"integration: testcontainers-backed integration tests"` to the list. Otherwise no change needed (pytest treats unknown markers as warnings unless `--strict-markers` is set).

### 1b — `test_public_surface.py`

- [ ] **Step 1.3: Write the public-surface test**

Create `tests/contract/test_public_surface.py`:

```python
"""Pin the SDK's top-level public surface.

If you add or remove a name from ai_core.__all__, you must also update
EXPECTED_PUBLIC_NAMES below. The two-place edit is deliberate — it forces
contributors to acknowledge that the public surface has changed.
"""
from __future__ import annotations

import ai_core


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


def test_public_surface_matches_expected() -> None:
    actual = set(ai_core.__all__)
    missing = EXPECTED_PUBLIC_NAMES - actual
    extra = actual - EXPECTED_PUBLIC_NAMES
    assert not missing, f"Missing exports: {sorted(missing)}"
    assert not extra, f"Unexpected new exports: {sorted(extra)}"
```

- [ ] **Step 1.4: Run the test**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/contract/test_public_surface.py -v 2>&1 | tail -10
```

Expected: 1 passed.

### 1c — `test_exception_invariants.py`

- [ ] **Step 1.5: Write the exception invariants test**

Create `tests/contract/test_exception_invariants.py`:

```python
"""Every typed exception must define a non-empty dotted-lowercase DEFAULT_CODE
and mirror it into details['error_code'] at construction time.
"""
from __future__ import annotations

import pytest

from ai_core.exceptions import EAAPBaseException


def _all_concrete_exceptions() -> list[type[EAAPBaseException]]:
    """Return all concrete subclasses of EAAPBaseException, recursively."""
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
def test_exception_default_code_is_dotted_lowercase(
    exc_cls: type[EAAPBaseException],
) -> None:
    code = exc_cls.DEFAULT_CODE
    assert code, f"{exc_cls.__qualname__}.DEFAULT_CODE is empty"
    assert code == code.lower(), (
        f"{exc_cls.__qualname__}.DEFAULT_CODE not lowercase: {code!r}"
    )
    assert "." in code, (
        f"{exc_cls.__qualname__}.DEFAULT_CODE not dotted: {code!r}"
    )


@pytest.mark.parametrize(
    "exc_cls", _all_concrete_exceptions(), ids=lambda c: c.__qualname__
)
def test_exception_mirrors_error_code_into_details(
    exc_cls: type[EAAPBaseException],
) -> None:
    exc = exc_cls("test message")
    assert exc.error_code == exc_cls.DEFAULT_CODE
    assert exc.details["error_code"] == exc.error_code


def test_at_least_fifteen_concrete_exceptions_exist() -> None:
    """Sanity check: regress if discovery breaks (pre-Phase-7 count is 16)."""
    classes = _all_concrete_exceptions()
    assert len(classes) >= 15, (
        f"Expected >=15 concrete exceptions, found {len(classes)}: "
        f"{[c.__qualname__ for c in classes]}"
    )
```

- [ ] **Step 1.6: Run the test**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/contract/test_exception_invariants.py -v 2>&1 | tail -20
```

Expected: ~33 passed (16 classes × 2 parametrized tests = 32, plus 1 sanity test = 33). If a concrete exception class fails one of the assertions, the test report will identify which class needs fixing — but at Phase 6 end all 16 should already comply (they were verified during prior phases).

If any test fails, STOP and report `BLOCKED`. The expected outcome is 33/33 passing; a failure would indicate a pre-existing defect in the SDK that needs investigation, not a Phase 7 issue.

### 1d — `test_audit_invariants.py`

- [ ] **Step 1.7: Write the audit invariants test**

Create `tests/contract/test_audit_invariants.py`:

```python
"""IAuditSink concretes must never raise from record() or flush().

Phase 1 contract: backend errors are swallowed; audit is best-effort.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Force-import every audit sink module so __subclasses__() picks them up,
# including those behind optional-dep extras.
import ai_core.audit  # noqa: F401
import ai_core.audit.jsonl  # noqa: F401
import ai_core.audit.null  # noqa: F401
import ai_core.audit.otel_event  # noqa: F401

# Optional sinks — may not be installed in this environment.
for _modname in ("ai_core.audit.sentry", "ai_core.audit.datadog"):
    try:
        importlib.import_module(_modname)
    except ImportError:
        pass

from ai_core.audit import AuditEvent, AuditRecord, IAuditSink


def _all_concrete_sinks() -> list[type[IAuditSink]]:
    seen: set[type[IAuditSink]] = set()
    stack = list(IAuditSink.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    return sorted(
        (c for c in seen if not inspect.isabstract(c)),
        key=lambda c: c.__qualname__,
    )


def _make_test_record() -> AuditRecord:
    return AuditRecord.now(
        AuditEvent.TOOL_INVOCATION_COMPLETED,
        tool_name="test", tool_version=1,
        agent_id="a", tenant_id="t",
        payload={"input": {"q": "x"}},
    )


def _construct_sink_with_failing_backend(
    sink_cls: type[IAuditSink], monkeypatch: pytest.MonkeyPatch
) -> IAuditSink:
    """Return a constructed instance whose underlying transport raises on use.

    Uses monkeypatch.setitem so any sys.modules mutations auto-revert
    after the test, preventing pollution of subsequent unit tests that
    may import the real sentry_sdk / datadog modules.
    """
    import sys
    name = sink_cls.__qualname__
    if name == "NullAuditSink":
        # NullAuditSink is a no-op; nothing to fault-inject. Return as-is.
        return sink_cls()
    if name == "JsonlFileAuditSink":
        # Point at a path the OS will refuse on write.
        bad_path = Path("/proc/this/path/cannot/be/written.jsonl")
        return sink_cls(bad_path)
    if name == "OTelEventAuditSink":
        # Pass a fake observability provider whose record_event raises.
        fake_obs = MagicMock()
        async def _raise(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("backend-down")
        fake_obs.record_event = _raise
        return sink_cls(fake_obs)
    if name == "SentryAuditSink":
        fake = MagicMock()
        fake.capture_event.side_effect = RuntimeError("backend-down")
        fake.flush.side_effect = RuntimeError("backend-down")
        monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
        # Force a fresh import of the sink module so it sees the fake.
        monkeypatch.delitem(sys.modules, "ai_core.audit.sentry", raising=False)
        from ai_core.audit.sentry import SentryAuditSink as _S
        return _S(dsn="https://x@x/1")
    if name == "DatadogAuditSink":
        fake = MagicMock()
        fake.api.Event.create.side_effect = RuntimeError("backend-down")
        monkeypatch.setitem(sys.modules, "datadog", fake)
        monkeypatch.delitem(sys.modules, "ai_core.audit.datadog", raising=False)
        from ai_core.audit.datadog import DatadogAuditSink as _D
        return _D(api_key="dd-key")
    pytest.skip(f"No fault-injection harness defined for {name}")


@pytest.mark.parametrize(
    "sink_cls", _all_concrete_sinks(), ids=lambda c: c.__qualname__
)
def test_audit_sink_record_never_raises(
    sink_cls: type[IAuditSink], monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = _construct_sink_with_failing_backend(sink_cls, monkeypatch)
    record = _make_test_record()
    # Must NOT raise — Phase 1 contract.
    asyncio.run(sink.record(record))


@pytest.mark.parametrize(
    "sink_cls", _all_concrete_sinks(), ids=lambda c: c.__qualname__
)
def test_audit_sink_flush_never_raises(
    sink_cls: type[IAuditSink], monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = _construct_sink_with_failing_backend(sink_cls, monkeypatch)
    # Must NOT raise — Phase 1 contract.
    asyncio.run(sink.flush())


def test_at_least_three_concrete_sinks_exist() -> None:
    sinks = _all_concrete_sinks()
    assert len(sinks) >= 3, (
        f"Expected >=3 concrete IAuditSink subclasses, found "
        f"{[c.__qualname__ for c in sinks]}"
    )
```

- [ ] **Step 1.8: Run the audit invariants test**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/contract/test_audit_invariants.py -v 2>&1 | tail -25
```

Expected: ≥7 passed (3 sinks × 2 parametrized tests = 6, plus 1 sanity test = 7; with `[sentry]` and `[datadog]` installed → 5 sinks × 2 = 10, plus 1 = 11). If any sink raises, the test reports which one and the exact exception — that's a real contract violation that needs fixing in the sink, not the test.

### 1e — `test_health_invariants.py`

- [ ] **Step 1.9: Write the health-probe invariants test**

Create `tests/contract/test_health_invariants.py`:

```python
"""IHealthProbe concretes must never raise from probe(); they return a
ProbeResult (potentially with status='down' or 'error') instead.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

# Force-import the probes module so __subclasses__() finds the concretes.
import ai_core.health.probes  # noqa: F401
from ai_core.config.settings import AppSettings, SecuritySettings
from ai_core.health import IHealthProbe, ProbeResult


def _all_concrete_probes() -> list[type[IHealthProbe]]:
    seen: set[type[IHealthProbe]] = set()
    stack = list(IHealthProbe.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    return sorted(
        (c for c in seen if not inspect.isabstract(c)),
        key=lambda c: c.__qualname__,
    )


def _construct_probe_with_failing_dependency(
    probe_cls: type[IHealthProbe],
) -> IHealthProbe:
    name = probe_cls.__qualname__
    if name == "OPAReachabilityProbe":
        # Point at a host:port nothing's listening on.
        settings = AppSettings()
        settings.security = SecuritySettings(opa_url="http://127.0.0.1:1")
        return probe_cls(settings)
    if name == "DatabaseProbe":
        # Engine that fails on connect.
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(
            "postgresql+asyncpg://x:y@127.0.0.1:1/x", connect_args={"timeout": 1}
        )
        return probe_cls(engine)
    if name == "ModelLookupProbe":
        # Settings with a model that won't resolve.
        settings = AppSettings()
        return probe_cls(settings)
    pytest.skip(f"No fault-injection harness defined for {name}")


@pytest.mark.parametrize(
    "probe_cls", _all_concrete_probes(), ids=lambda c: c.__qualname__
)
def test_health_probe_never_raises(probe_cls: type[IHealthProbe]) -> None:
    probe = _construct_probe_with_failing_dependency(probe_cls)
    # Must NOT raise — must return a ProbeResult (even with status="down"/"error").
    result = asyncio.run(probe.probe())
    assert isinstance(result, ProbeResult)


def test_at_least_three_concrete_probes_exist() -> None:
    probes = _all_concrete_probes()
    assert len(probes) >= 3, (
        f"Expected >=3 concrete IHealthProbe subclasses, found "
        f"{[c.__qualname__ for c in probes]}"
    )
```

- [ ] **Step 1.10: Run the health invariants test**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/contract/test_health_invariants.py -v 2>&1 | tail -15
```

Expected: 4 passed (3 probes × 1 test + 1 sanity = 4).

If `ModelLookupProbe.probe()` raises with the empty-settings construction, that's a Phase 6 defect — STOP and report. (The probe's responsibility is to return a `ProbeResult` even when the model registry is empty.)

### 1f — `test_container_lifecycle.py`

- [ ] **Step 1.11: Write the container-lifecycle test**

Create `tests/contract/test_container_lifecycle.py`:

```python
"""Container.stop() must invoke the 5 documented teardown steps in order.

Order (Phase 6-end): observability.shutdown -> audit.flush ->
mcp_pool.aclose -> policy_evaluator.aclose -> engine.dispose.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from injector import Module, provider, singleton

from ai_core.audit import IAuditSink
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import (
    IObservabilityProvider,
    IPolicyEvaluator,
)
from ai_core.mcp.transports import IMCPConnectionFactory


class _SpyModule(Module):
    """Override every Container teardown target with a spy that records call order."""

    def __init__(self, call_log: list[str]) -> None:
        self._call_log = call_log

    def configure(self, binder: Any) -> None:
        # Build spies whose teardown method appends a tag to call_log.
        log = self._call_log

        class _SpyObservability:
            async def shutdown(self) -> None:
                log.append("observability.shutdown")
            async def record_event(self, *_a: Any, **_k: Any) -> None: ...
            async def record_llm_usage(self, *_a: Any, **_k: Any) -> None: ...
            def start_span(self, *_a: Any, **_k: Any) -> Any: ...

        class _SpyAudit:
            async def record(self, _r: Any) -> None: ...
            async def flush(self) -> None:
                log.append("audit.flush")

        class _SpyMCP:
            def open(self, _spec: Any) -> Any: ...
            async def aclose(self) -> None:
                log.append("mcp_pool.aclose")

        class _SpyPolicy:
            async def evaluate(self, **_k: Any) -> Any: ...
            async def aclose(self) -> None:
                log.append("policy_evaluator.aclose")

        # AsyncEngine is third-party; we wrap it with an AsyncMock instance
        # whose dispose appends to the log.
        from sqlalchemy.ext.asyncio import AsyncEngine
        spy_engine = AsyncMock(spec=AsyncEngine)
        async def _dispose_spy() -> None:
            log.append("engine.dispose")
        spy_engine.dispose.side_effect = _dispose_spy

        binder.bind(IObservabilityProvider, to=_SpyObservability(), scope=singleton)
        binder.bind(IAuditSink, to=_SpyAudit(), scope=singleton)
        binder.bind(IMCPConnectionFactory, to=_SpyMCP(), scope=singleton)
        binder.bind(IPolicyEvaluator, to=_SpyPolicy(), scope=singleton)
        binder.bind(AsyncEngine, to=spy_engine, scope=singleton)


@pytest.mark.asyncio
async def test_container_teardown_calls_steps_in_documented_order() -> None:
    call_log: list[str] = []
    container = Container.build([AgentModule(), _SpyModule(call_log)])
    await container.start()
    await container.stop()

    assert call_log == [
        "observability.shutdown",
        "audit.flush",
        "mcp_pool.aclose",
        "policy_evaluator.aclose",
        "engine.dispose",
    ], f"Teardown order incorrect: {call_log}"
```

- [ ] **Step 1.12: Run the lifecycle test**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/contract/test_container_lifecycle.py -v 2>&1 | tail -10
```

Expected: 1 passed.

If the test fails on container.start() with a DI resolution error, the spy module bindings may need adjustment to satisfy the full DI graph. STOP and report `NEEDS_CONTEXT` with the exact stack trace — the controller will provide guidance on which dependencies to also override.

### 1g — Lint, type-check, full suite, commit

- [ ] **Step 1.13: Lint + type-check the Task 1 change set**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check tests/contract/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy tests/contract/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations on the new test files; mypy on `tests/contract/` clean; project total `mypy src` ≤ 21 (unchanged because no `src/` files were touched).

- [ ] **Step 1.14: Run unit + component + contract suites**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -10
```

Expected: 450+ passing (405 baseline + ~45 contract collected items); 9 pre-existing langgraph errors unchanged.

- [ ] **Step 1.15: Commit Task 1**

```bash
git add tests/contract/
git commit -m "test(contract): pin public surface + cross-cutting Phase 1-6 invariants"
```

---

## Task 2 — Testcontainers infra + Postgres integration

Adds the `testcontainers` dev dep, the integration directory, the session-scoped Docker fixtures, and the Postgres integration tests. Builds on Task 1 (no source dependency, just chronologically after).

**Files:**
- Modify: `pyproject.toml` (add `testcontainers[postgres]>=4.0,<5.0` to `[project.optional-dependencies] dev`)
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_database_integration.py`

### 2a — Add the dev dependency

- [ ] **Step 2.1: Add `testcontainers[postgres]` to the dev extra**

```bash
grep -n "\[project.optional-dependencies\]\|dev =" /Users/admin-h26/EAAP/ai-core-sdk/pyproject.toml | head -10
```

Find the `dev = [...]` list inside `[project.optional-dependencies]`. Add the line:

```toml
    "testcontainers[postgres]>=4.0,<5.0",
```

Preserve alphabetical order if the existing list is sorted; otherwise just append.

- [ ] **Step 2.2: Sync the venv**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pip install -e "/Users/admin-h26/EAAP/ai-core-sdk[dev]" 2>&1 | tail -5
```

Expected: `Successfully installed testcontainers-X.Y.Z` (and any sub-deps like `docker`, `psycopg2-binary`).

- [ ] **Step 2.3: Smoke import**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
import testcontainers
from testcontainers.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
print('testcontainers ok', testcontainers.__version__)
"
```

Expected: `testcontainers ok 4.x.y`. (If the import fails, `pip install` may need a retry — testcontainers has changed import paths between versions.)

### 2b — Create the integration directory + conftest

- [ ] **Step 2.4: Create the directory + bare files**

```bash
mkdir -p /Users/admin-h26/EAAP/ai-core-sdk/tests/integration
touch /Users/admin-h26/EAAP/ai-core-sdk/tests/integration/__init__.py
```

- [ ] **Step 2.5: Write the integration conftest**

Create `tests/integration/conftest.py`:

```python
"""Session-scoped Testcontainers fixtures with Docker-availability gating.

Every test in tests/integration/ skips automatically when Docker is
unavailable. When Docker is up, fixtures spin up Postgres + OPA once
per session and tests share them.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from testcontainers.core.container import DockerContainer
    from testcontainers.postgres import PostgresContainer

# Locate the eaap init starter policies — used by OPA fixture (Task 3).
POLICIES_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "ai_core" / "cli" / "templates" / "init" / "policies"
)
OPA_IMAGE = "openpolicyagent/opa:0.66.0"
POSTGRES_IMAGE = "postgres:16"


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Probe the Docker socket; return False if unreachable."""
    try:
        from testcontainers.core.docker_client import DockerClient  # noqa: PLC0415
        DockerClient().client.ping()
    except Exception:  # noqa: BLE001 — any failure means Docker is unusable
        return False
    return True


@pytest.fixture(scope="session")
def postgres_container(
    docker_available: bool,  # noqa: FBT001
) -> Iterator["PostgresContainer"]:
    if not docker_available:
        pytest.skip("Docker not available — integration tests skipped")
    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        yield pg


@pytest.fixture(scope="session")
def opa_container(
    docker_available: bool,  # noqa: FBT001
) -> Iterator["DockerContainer"]:
    if not docker_available:
        pytest.skip("Docker not available — integration tests skipped")
    from testcontainers.core.container import DockerContainer  # noqa: PLC0415
    from testcontainers.core.waiting_utils import wait_for_logs  # noqa: PLC0415
    container = (
        DockerContainer(OPA_IMAGE)
        .with_command("run --server --addr 0.0.0.0:8181 /policies")
        .with_volume_mapping(str(POLICIES_DIR), "/policies", "ro")
        .with_exposed_ports(8181)
    )
    with container as opa:
        wait_for_logs(opa, "Server started", timeout=30)
        yield opa


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Tag every test in this directory with pytest.mark.integration."""
    for item in items:
        item.add_marker(pytest.mark.integration)
```

(The OPA fixture is defined now even though only the Postgres tests use it in Task 2. Task 3 will add `test_opa_integration.py` that consumes it without needing to extend conftest.)

### 2c — Postgres integration tests

- [ ] **Step 2.6: Write the database integration tests**

Create `tests/integration/test_database_integration.py`:

```python
"""Postgres integration tests via Testcontainers.

Auto-skip when Docker is unavailable (see conftest.docker_available).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ai_core.health.probes import DatabaseProbe

if TYPE_CHECKING:
    from testcontainers.postgres import PostgresContainer


def _asyncpg_dsn(pg: "PostgresContainer") -> str:
    """Convert Testcontainers' default psycopg2 DSN to asyncpg form."""
    return pg.get_connection_url().replace(
        "postgresql+psycopg2://", "postgresql+asyncpg://"
    )


@pytest.mark.asyncio
async def test_async_engine_connects_to_real_postgres(
    postgres_container: "PostgresContainer",
) -> None:
    engine = create_async_engine(_asyncpg_dsn(postgres_container))
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_database_probe_returns_ok_against_real_postgres(
    postgres_container: "PostgresContainer",
) -> None:
    engine = create_async_engine(_asyncpg_dsn(postgres_container))
    try:
        probe = DatabaseProbe(engine)
        result = await probe.probe()
        assert result.status == "ok"
        assert result.component == "database"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_database_probe_returns_down_when_postgres_unreachable() -> None:
    """No fixture needed — exercises probe's never-raise contract on bad DSN."""
    engine = create_async_engine(
        "postgresql+asyncpg://x:y@127.0.0.1:1/x", connect_args={"timeout": 1}
    )
    try:
        probe = DatabaseProbe(engine)
        result = await probe.probe()
        # Probe MUST return a result (not raise) even when the backend is unreachable.
        assert result.status in {"down", "error", "degraded"}
        assert result.component == "database"
    finally:
        await engine.dispose()
```

- [ ] **Step 2.7: Run the integration tests (verify they work AND that they skip when Docker is down)**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/integration/test_database_integration.py -v 2>&1 | tail -15
```

Expected: If Docker is up, 3 passed. If Docker is unavailable, 2 skipped + 1 passed (the bad-DSN test doesn't need Docker — it uses an unreachable port and verifies the never-raise contract).

If you're on a machine without Docker and want to verify the Docker-conditional path works:

```bash
DOCKER_HOST=unix:///nonexistent.sock /Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/integration/test_database_integration.py -v 2>&1 | tail -10
```

Expected: 2 skipped (`postgres_container` fixture skip), 1 passed.

### 2d — Lint, type-check, full suite, commit

- [ ] **Step 2.8: Lint + type-check the Task 2 change set**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check tests/integration/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy tests/integration/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on `tests/integration/` clean; project total `mypy src` ≤ 21.

- [ ] **Step 2.9: Run unit + component + contract + integration suites**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract tests/integration -q 2>&1 | tail -10
```

Expected: 450+ passing from contract/unit/component (unchanged from Task 1's gate); plus integration tests either passing (Docker up) or skipping (Docker down). Net delta: +3 collected items.

- [ ] **Step 2.10: Commit Task 2**

```bash
git add pyproject.toml tests/integration/__init__.py tests/integration/conftest.py tests/integration/test_database_integration.py
git commit -m "test(integration): testcontainers infra + Postgres integration tests"
```

---

## Task 3 — OPA integration

Adds the OPA integration test file. Consumes the `opa_container` fixture defined in Task 2's conftest.

**Files:**
- Create: `tests/integration/test_opa_integration.py`

### 3a — OPA integration tests

- [ ] **Step 3.1: Verify the conftest's OPA fixture is in place**

```bash
grep -n "opa_container\|OPA_IMAGE\|POLICIES_DIR" /Users/admin-h26/EAAP/ai-core-sdk/tests/integration/conftest.py | head -5
```

Expected: confirm `opa_container` fixture exists from Task 2.

- [ ] **Step 3.2: Verify the bundled .rego policies exist with package declarations**

```bash
grep -E "^package " /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/cli/templates/init/policies/agent.rego /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/cli/templates/init/policies/api.rego
```

Expected:
```
.../agent.rego:package eaap.agent.tool_call
.../api.rego:package eaap.api
```

These package paths drive the `opa_decision_path` settings in the test below. If either is missing, STOP and report `BLOCKED` — Phase 7 depends on Phase 5's policy scaffold.

- [ ] **Step 3.3: Write the OPA integration tests**

Create `tests/integration/test_opa_integration.py`:

```python
"""OPA integration tests via Testcontainers.

Loads the eaap init starter policies (policies/agent.rego, policies/api.rego)
into a real OPA server and exercises OPAReachabilityProbe + OPAPolicyEvaluator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ai_core.config.settings import AppSettings, SecuritySettings
from ai_core.health.probes import OPAReachabilityProbe
from ai_core.security.opa import OPAPolicyEvaluator

if TYPE_CHECKING:
    from testcontainers.core.container import DockerContainer


def _opa_url(opa: "DockerContainer") -> str:
    host = opa.get_container_host_ip()
    port = opa.get_exposed_port(8181)
    return f"http://{host}:{port}"


def _make_settings(opa: "DockerContainer", **security_overrides: object) -> AppSettings:
    settings = AppSettings()
    settings.security = SecuritySettings(opa_url=_opa_url(opa), **security_overrides)
    return settings


@pytest.mark.asyncio
async def test_opa_reachability_probe_returns_ok_against_real_opa(
    opa_container: "DockerContainer",
) -> None:
    settings = _make_settings(opa_container)
    probe = OPAReachabilityProbe(settings)
    result = await probe.probe()
    assert result.status == "ok"
    assert result.component == "opa"


@pytest.mark.asyncio
async def test_opa_reachability_probe_honours_custom_health_path(
    opa_container: "DockerContainer",
) -> None:
    settings = _make_settings(opa_container, opa_health_path="/health")
    probe = OPAReachabilityProbe(settings)
    result = await probe.probe()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_opa_policy_evaluator_evaluates_starter_agent_policy(
    opa_container: "DockerContainer",
) -> None:
    """Exercise the eaap init starter agent policy.

    The agent.rego shipped with `eaap init` defines:
        package eaap.agent.tool_call
        default allow := false
        allow if { not denied_tool }
        denied_tool if { input.tool.name == "delete_everything" }

    So a tool call that is NOT in the deny list should return allowed=True.
    """
    settings = _make_settings(opa_container)
    evaluator = OPAPolicyEvaluator(settings)
    try:
        decision = await evaluator.evaluate(
            decision_path="eaap/agent/tool_call/allow",
            input={"tool": {"name": "search"}, "principal": {"sub": "user-1"}},
        )
        assert decision.allowed is True
    finally:
        await evaluator.aclose()


@pytest.mark.asyncio
async def test_opa_policy_evaluator_denies_blocked_tool(
    opa_container: "DockerContainer",
) -> None:
    """The starter policy denies tool.name=='delete_everything'."""
    settings = _make_settings(opa_container)
    evaluator = OPAPolicyEvaluator(settings)
    try:
        decision = await evaluator.evaluate(
            decision_path="eaap/agent/tool_call/allow",
            input={"tool": {"name": "delete_everything"}, "principal": {"sub": "user-1"}},
        )
        assert decision.allowed is False
    finally:
        await evaluator.aclose()
```

- [ ] **Step 3.4: Run the OPA integration tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/integration/test_opa_integration.py -v 2>&1 | tail -15
```

Expected: 4 passed when Docker is up; 4 skipped when Docker is down.

If a test fails because the OPA decision shape differs from what the test expects (e.g., OPA returns `{"result": false}` rather than `{"result": {"allow": false}}`), check the `OPAPolicyEvaluator.evaluate` implementation in `src/ai_core/security/opa.py` for the exact request/response format. The decision_path passed must match the policy's package path joined by `/`.

If the test fails with a connection-refused error after `opa_container` fixture says it's healthy, the `wait_for_logs` timeout in conftest may need increasing — try 60s instead of 30s.

### 3b — Lint, type-check, full suite, commit

- [ ] **Step 3.5: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check tests/integration/test_opa_integration.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy tests/integration/test_opa_integration.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on the new file clean; project total `mypy src` ≤ 21.

- [ ] **Step 3.6: Run unit + component + contract + integration suites**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract tests/integration -q 2>&1 | tail -10
```

Expected: same passing counts as Task 2 plus 4 new integration tests (passed if Docker up, skipped if not).

- [ ] **Step 3.7: Commit Task 3**

```bash
git add tests/integration/test_opa_integration.py
git commit -m "test(integration): OPA integration tests against real opa server with starter policies"
```

---

## Task 4 — End-of-phase smoke gate

Verification only. No code changes.

- [ ] **Step 4.1: Full test suite (always-run)**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -5
```

Expected: ≥450 passing (405 baseline + ~45 contract); 9 pre-existing langgraph errors unchanged.

- [ ] **Step 4.2: Integration suite (Docker-conditional)**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/integration -v 2>&1 | tail -15
```

Expected: 7 passed (Docker up) OR 6 skipped + 1 passed (Docker down — the bad-DSN database test runs without Docker).

If Docker is up but tests fail, investigate the specific test:
- Postgres tests failing with connection errors → check `postgres_container.get_connection_url()` output and asyncpg DSN conversion.
- OPA tests failing → check `wait_for_logs(opa, "Server started")` actually triggered, OR the policy package paths match what the test asserts.

- [ ] **Step 4.3: Lint the entire tree**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src tests 2>&1 | grep "Found"
```

Expected: 211 errors total (= post-Phase-6 baseline). No NEW violations.

- [ ] **Step 4.4: Mypy strict**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `Found 21 errors in 8 files (checked 68 source files)`. Phase 7 doesn't change `src/` so this should be byte-for-byte unchanged.

- [ ] **Step 4.5: Verify Phase 7 surface**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
import ai_core
assert len(ai_core.__all__) == 29, f'Expected 29 exports, got {len(ai_core.__all__)}'
from tests.contract.test_public_surface import EXPECTED_PUBLIC_NAMES
assert set(ai_core.__all__) == EXPECTED_PUBLIC_NAMES
print('Phase 7 surface OK')
"
```

Expected: `Phase 7 surface OK`.

(This is also asserted by the contract test — this manual check is a belt-and-suspenders sanity verification.)

- [ ] **Step 4.6: `eaap init` regression smoke (Phase 5 invariant)**

```bash
SMOKE=$(mktemp -d)
/Users/admin-h26/EAAP/.venv/bin/python -m ai_core.cli.main init smoke-app --path "$SMOKE" 2>&1 | tail -2
ls "$SMOKE/smoke-app/"
test -f "$SMOKE/smoke-app/eaap.yaml" && echo "eaap.yaml present"
test -f "$SMOKE/smoke-app/policies/agent.rego" && echo "agent.rego present"
test -f "$SMOKE/smoke-app/policies/api.rego" && echo "api.rego present"
rm -rf "$SMOKE"
```

Expected: tree listing + 3 "present" lines. Confirms Phase 5's scaffold invariant is intact.

- [ ] **Step 4.7: Capture phase summary**

```bash
git log --oneline 70976c9..HEAD
```

Expected: 3 conventional-commit subjects (Tasks 1, 2, 3) plus any review-fix commits.

- [ ] **Step 4.8: Do NOT push automatically**

```bash
git status
echo ""
echo "Suggested next step:"
echo "git push origin feat/phase-7-test-quality"
echo "gh pr create --title 'feat: Phase 7 — test quality (contract tests + Testcontainers integration)'"
```

---

## Out-of-scope reminders

For traceability, deferred to Phase 8+:

- Reviving the 9 pre-existing `tests/unit/persistence/test_langgraph_checkpoint.py` errors
- CI workflow updates (host project owns CI)
- Sentry/Datadog mock-server containers
- Phase 4 follow-ups (Vertex AI Anthropic prefix, 1-hour cache TTL, multi-conn MCP pool, pool health probe)
- Sink polish (Sentry breadcrumbs, Datadog metrics/DogStatsD/logs)
- Redaction polish (custom regex at settings, per-sink overrides, Presidio/Faker/spaCy)
- `error_code` registry / constants module
- Sentry SDK v3 migration

If a step starts pulling work from this list, stop and confirm scope with the user.
