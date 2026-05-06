# ai-core-sdk Phase 7 — Design

**Date:** 2026-05-06
**Branch:** `feat/phase-7-test-quality`
**Status:** Awaiting user review

## Goal

Test-quality layer: lock the public 29-name surface and the cross-cutting Phase 1-6 invariants (audit-sink-never-raises, probe-never-raises, exception `error_code`, container teardown order) via contract tests, and add Testcontainers-backed integration tests that exercise the SDK against real Postgres and OPA backends. After Phase 7, accidental surface drift fails CI immediately and integration tests give actual confidence that fakes weren't lying.

## Scope (2 items)

### Test quality (2)

1. **Contract tests** in a new `tests/contract/` directory. Five test files, all running in-process with no infrastructure:
   - `test_public_surface.py` — asserts `set(ai_core.__all__)` exactly matches a frozen `EXPECTED_PUBLIC_NAMES: frozenset[str]` of 29 entries. New exports require a deliberate two-place edit (the package `__all__` AND this set).
   - `test_audit_invariants.py` — discovers all `IAuditSink` subclasses via `__subclasses__()` and asserts every concrete one swallows exceptions in `record(...)` and `flush()` when its underlying transport raises (parametrized fault injection).
   - `test_health_invariants.py` — same shape for `IHealthProbe`: every concrete subclass returns a `ProbeResult` (not raises) when its dependency raises.
   - `test_exception_invariants.py` — every concrete subclass of `EAAPBaseException` defines a non-empty dotted-lowercase `DEFAULT_CODE` and `details["error_code"]` mirrors `self.error_code` after construction.
   - `test_container_lifecycle.py` — `Container._teardown_sdk_resources` calls the 5 documented steps (`observability.shutdown` → `audit.flush` → `mcp_pool.aclose` → `policy_evaluator.aclose` → `engine.dispose`) in exactly that order; verified by spying.

2. **Integration tests** in a new `tests/integration/` directory. Testcontainers-backed; auto-skip when Docker is unavailable.
   - `conftest.py` — session-scoped `postgres_container` and `opa_container` fixtures. `docker_available` fixture probes the Docker socket via `testcontainers.core.docker_client.DockerClient().client.ping()`; if it raises, every test in the directory skips with `"Docker not available — integration tests skipped"`.
   - `test_database_integration.py` — `AppSettings(database__dsn=...)` against the real Postgres container; asserts the async SQLAlchemy engine starts, executes a trivial `SELECT 1` query, and disposes cleanly. Also exercises `DatabaseProbe.probe()` against the real DB.
   - `test_opa_integration.py` — bring up an OPA container with the `eaap init` starter policies (`agent.rego` + `api.rego`) loaded via volume mount. Exercise `OPAReachabilityProbe` against the real OPA `/health` endpoint and `OPAPolicyEvaluator.allow(...)` against a real Rego decision.

## Non-goals (deferred to Phase 8+)

- Reviving the 9 pre-existing `tests/unit/persistence/test_langgraph_checkpoint.py` errors (separate scope; `respx`/`aiosqlite` collection issue).
- CI configuration changes (`.github/workflows/` updates) — host project owns CI.
- Sentry/Datadog mock-server containers — those stay mocked at the SDK level.
- Phase 4 follow-ups (Vertex AI Anthropic prefix, 1-hour cache TTL, multi-conn MCP pool, pool health probe).
- Sink polish (Sentry breadcrumbs, Datadog metrics/DogStatsD/logs).
- Redaction polish (custom regex at settings layer, per-sink redaction overrides, Presidio/Faker/spaCy backends).
- `error_code` registry / constants module.
- Sentry SDK v3 migration.
- Contract tests for Phase 7-internal types (we test what existed at Phase 6's end).

## Constraints

- Pre-1.0; **no backwards-compatibility requirement.** Tests are purely additive; production code is untouched.
- Per-step gate: `ruff check` no new violations + `mypy <touched files>` strict + `pytest tests/unit tests/component tests/contract` + `pytest tests/integration` (Docker-conditional, auto-skip otherwise).
- Project mypy total stays ≤ 21 (post-Phase-6 baseline). Test-side additions don't affect `mypy src`.
- Project ruff total stays ≤ 211 (post-Phase-6 baseline at `d5f03d5`).
- New dev dependency `testcontainers[postgres]>=4.0,<5.0` added to `[project.optional-dependencies] dev` only — does NOT affect default `pip install ai-core-sdk` footprint.
- End-of-phase smoke (`eaap init`, surface symbols, etc.) must continue to work.

## Module layout

```
tests/
├── contract/                       # NEW
│   ├── __init__.py                 # NEW (empty)
│   ├── conftest.py                 # NEW (~10 LOC, registers pytest.mark.contract)
│   ├── test_public_surface.py      # NEW (~25 LOC, 1 test + frozen 29-name set)
│   ├── test_audit_invariants.py    # NEW (~50 LOC, parametrized over IAuditSink concretes)
│   ├── test_health_invariants.py   # NEW (~50 LOC, parametrized over IHealthProbe concretes)
│   ├── test_exception_invariants.py # NEW (~40 LOC, parametrized over EAAPBaseException concretes)
│   └── test_container_lifecycle.py # NEW (~50 LOC, 1 test asserting teardown order)
│
└── integration/                    # NEW
    ├── __init__.py                 # NEW (empty)
    ├── conftest.py                 # NEW (~80 LOC, session fixtures + skip logic)
    ├── test_database_integration.py # NEW (~60 LOC, 3 tests vs real Postgres)
    └── test_opa_integration.py     # NEW (~80 LOC, 3 tests vs real OPA)

pyproject.toml                      # MODIFIED — testcontainers added to [dev]

src/                                # NOT TOUCHED
```

### Files NOT touched

- `README.md`
- `src/**` (entire src tree — Phase 7 is purely test-side)
- `tests/unit/**` (existing unit tests stay as-is)
- `tests/component/**` (existing component tests stay as-is)
- `src/ai_core/cli/templates/init/**`

## Component 1 — Contract tests

### 1a. `test_public_surface.py`

```python
"""Pin the SDK's top-level public surface.

If you add or remove a name from ai_core.__all__, you must also update
EXPECTED_PUBLIC_NAMES below. The two-place edit is deliberate — it forces
contributors to acknowledge that the public surface has changed.
"""
from __future__ import annotations

import pytest

import ai_core

pytestmark = pytest.mark.contract

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

### 1b. `test_audit_invariants.py`

```python
"""IAuditSink concretes must never raise from record() or flush().

Phase 1 contract: backend errors are swallowed; audit is best-effort.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

# Force-import every sink module so __subclasses__() picks them up,
# even ones that live behind optional-dep extras.
import ai_core.audit
import ai_core.audit.jsonl
import ai_core.audit.null
import ai_core.audit.otel_event

# Optional sinks (may not be installed in dev environment)
for _modname in ("ai_core.audit.sentry", "ai_core.audit.datadog"):
    try:
        importlib.import_module(_modname)
    except ImportError:
        pass

from ai_core.audit import IAuditSink

pytestmark = pytest.mark.contract


def _all_concrete_sinks() -> list[type[IAuditSink]]:
    """Walk subclass tree, return concrete (non-abstract) subclasses."""
    seen: set[type[IAuditSink]] = set()
    stack = list(IAuditSink.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return sorted(
        (c for c in seen if not inspect.isabstract(c)),
        key=lambda c: c.__qualname__,
    )


@pytest.mark.parametrize("sink_cls", _all_concrete_sinks(), ids=lambda c: c.__qualname__)
def test_audit_sink_record_never_raises(sink_cls: type[IAuditSink]) -> None:
    """Use a fault-injecting wrapper to force every concrete sink to face a
    failing transport; assert record() returns normally without raising."""
    # Strategy: subclass the concrete sink and override the underlying call
    # with one that raises. Each sink type has a different strategy below.
    # Simplest approach: monkey-patch a known internal method on the
    # constructed instance with a function that raises, then await record().
    # ... actual fault-injection by sink type ...
    pytest.skip("Fault-injection harness — see implementation plan for per-sink strategies")


def test_at_least_three_concrete_sinks_exist() -> None:
    """Sanity check: regress if discovery breaks."""
    sinks = _all_concrete_sinks()
    assert len(sinks) >= 3, f"Expected >=3 concrete sinks, found {[c.__qualname__ for c in sinks]}"
```

(The fault-injection harness is per-sink — `NullAuditSink` is a no-op so trivially never raises; `JsonlFileAuditSink` is exercised by writing to a non-writable directory; `OTelEventAuditSink` by giving it a fake observability provider that raises; `SentryAuditSink`/`DatadogAuditSink` by mocking their respective SDK to raise. The plan spells out the per-sink strategy.)

### 1c. `test_health_invariants.py`

Same shape as 1b but for `IHealthProbe`. Concrete subclasses include `OPAReachabilityProbe`, `DatabaseProbe`, `ModelLookupProbe`. Each is constructed with a dependency that raises (e.g., a `httpx.AsyncClient` that fails, an `AsyncEngine.connect()` that raises) and `probe()` is asserted to return a `ProbeResult` with `status="error"` rather than raise.

### 1d. `test_exception_invariants.py`

```python
"""Every typed exception must define a non-empty dotted-lowercase DEFAULT_CODE
and mirror it into details['error_code'] at construction time.
"""
from __future__ import annotations

import pytest

from ai_core.exceptions import EAAPBaseException

pytestmark = pytest.mark.contract


def _all_concrete_exceptions() -> list[type[EAAPBaseException]]:
    seen: set[type[EAAPBaseException]] = set()
    stack = list(EAAPBaseException.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    return sorted(seen, key=lambda c: c.__qualname__)


@pytest.mark.parametrize("exc_cls", _all_concrete_exceptions(), ids=lambda c: c.__qualname__)
def test_exception_default_code_is_dotted_lowercase(
    exc_cls: type[EAAPBaseException],
) -> None:
    code = exc_cls.DEFAULT_CODE
    assert code, f"{exc_cls.__qualname__}.DEFAULT_CODE is empty"
    assert code == code.lower(), f"{exc_cls.__qualname__}.DEFAULT_CODE not lowercase: {code!r}"
    assert "." in code, f"{exc_cls.__qualname__}.DEFAULT_CODE not dotted: {code!r}"


@pytest.mark.parametrize("exc_cls", _all_concrete_exceptions(), ids=lambda c: c.__qualname__)
def test_exception_mirrors_error_code_into_details(
    exc_cls: type[EAAPBaseException],
) -> None:
    exc = exc_cls("test message")
    assert exc.details["error_code"] == exc.error_code == exc_cls.DEFAULT_CODE
```

### 1e. `test_container_lifecycle.py`

Constructs a `Container` with a custom test module that registers spies on each of the 5 teardown targets (`IObservabilityProvider`, `IAuditSink`, `IMCPConnectionFactory`, `IPolicyEvaluator`, `AsyncEngine`). Calls `container.stop()` and asserts the call order matches the documented sequence.

## Component 2 — Integration tests

### 2a. `tests/integration/conftest.py`

```python
"""Session-scoped Testcontainers fixtures with Docker-availability gating."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from testcontainers.core.container import DockerContainer
    from testcontainers.postgres import PostgresContainer

POLICIES_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "ai_core" / "cli" / "templates" / "init" / "policies"
)
OPA_IMAGE = "openpolicyagent/opa:0.66.0"
POSTGRES_IMAGE = "postgres:16"


@pytest.fixture(scope="session")
def docker_available() -> bool:
    try:
        from testcontainers.core.docker_client import DockerClient  # noqa: PLC0415
        DockerClient().client.ping()
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def postgres_container(docker_available: bool) -> Iterator["PostgresContainer"]:
    if not docker_available:
        pytest.skip("Docker not available — integration tests skipped")
    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        yield pg


@pytest.fixture(scope="session")
def opa_container(docker_available: bool) -> Iterator["DockerContainer"]:
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
```

### 2b. `test_database_integration.py`

```python
"""Postgres integration tests via Testcontainers.

Auto-skip when Docker is unavailable.
"""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from ai_core.config.settings import AppSettings, DatabaseSettings
from ai_core.health.probes import DatabaseProbe

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_async_engine_connects_to_real_postgres(postgres_container) -> None:
    # Convert the JDBC-style URL Testcontainers gives us into asyncpg form.
    raw = postgres_container.get_connection_url()
    dsn = raw.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    engine = create_async_engine(dsn)
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_database_probe_returns_ok_against_real_postgres(
    postgres_container,
) -> None:
    raw = postgres_container.get_connection_url()
    dsn = raw.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    engine = create_async_engine(dsn)
    probe = DatabaseProbe(engine)
    result = await probe.probe()
    assert result.status == "ok"
    await engine.dispose()


@pytest.mark.asyncio
async def test_database_probe_returns_error_when_postgres_unreachable(
    postgres_container,
) -> None:
    # Swap the host port to one nothing's listening on.
    bad_dsn = "postgresql+asyncpg://x:y@127.0.0.1:1/x"
    engine = create_async_engine(bad_dsn, connect_args={"timeout": 1})
    probe = DatabaseProbe(engine)
    result = await probe.probe()
    assert result.status == "error"  # NEVER raises; returns degraded
    await engine.dispose()
```

### 2c. `test_opa_integration.py`

```python
"""OPA integration tests via Testcontainers.

Loads the eaap init starter policies (policies/agent.rego, policies/api.rego)
into a real OPA server and exercises OPAReachabilityProbe + OPAPolicyEvaluator.
"""
from __future__ import annotations

import pytest

from ai_core.config.settings import AppSettings, SecuritySettings
from ai_core.health.probes import OPAReachabilityProbe

pytestmark = pytest.mark.integration


def _opa_url(opa_container) -> str:
    host = opa_container.get_container_host_ip()
    port = opa_container.get_exposed_port(8181)
    return f"http://{host}:{port}"


@pytest.mark.asyncio
async def test_opa_reachability_probe_returns_ok_against_real_opa(opa_container) -> None:
    settings = AppSettings()
    settings.security = SecuritySettings(opa_url=_opa_url(opa_container))
    probe = OPAReachabilityProbe(settings)
    result = await probe.probe()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_opa_reachability_probe_honours_custom_health_path(opa_container) -> None:
    settings = AppSettings()
    settings.security = SecuritySettings(
        opa_url=_opa_url(opa_container),
        opa_health_path="/health",  # explicit
    )
    probe = OPAReachabilityProbe(settings)
    result = await probe.probe()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_opa_policy_evaluator_evaluates_starter_policy(opa_container) -> None:
    """Exercise the eaap init starter agent policy: deny by default."""
    from ai_core.security.opa_evaluator import OPAPolicyEvaluator
    settings = AppSettings()
    settings.security = SecuritySettings(
        opa_url=_opa_url(opa_container),
        opa_decision_path="eaap/agent/tool_call/allow",
    )
    evaluator = OPAPolicyEvaluator(settings)
    decision = await evaluator.allow(
        principal={"sub": "user-1"},
        action="tool.invoke",
        input={"tool": {"name": "search"}},
    )
    # Default-deny per the starter policy.
    assert decision.allowed is False
```

(The exact `OPAPolicyEvaluator` API may differ in this codebase; the implementation plan resolves the import path and method shape from `src/ai_core/security/`.)

## Error handling — consolidated (Phase 7 deltas)

Phase 7 introduces no new production-code error paths. The deltas are entirely test-side:

| Path | Behaviour |
|---|---|
| Contract test fails (surface drift, sink leaks an exception, etc.) | Standard pytest fail. Fix is to update the contract or the implementation. |
| Docker socket unreachable when running `pytest tests/integration` | `docker_available` returns False; every test in the directory skips with a single message. |
| Docker is up but `docker pull` fails | Testcontainers raises `DockerException`; pytest reports as error (NOT skipped) — loud failure is correct. |
| OPA container starts but policies fail to load | `wait_for_logs` returns; subsequent test fails — failure surfaces via the test, not the fixture. |
| Postgres container start exceeds 60s | Testcontainers default timeout fires; pytest error. No retry layer. |
| Test in `tests/contract/` accidentally requires Docker | Caught by review; not enforced via test. |

Phase 1-6 invariants preserved by virtue of being un-touched.

## Testing strategy

This phase IS the testing strategy.

### Per-step gate (Phase 7 update)

Phases 1-6 used: `pytest tests/unit tests/component -q`. Phase 7 extends to:

```bash
# Always-run (no Docker needed)
pytest tests/unit tests/component tests/contract -q

# Docker-conditional (run when Docker is up; auto-skip otherwise)
pytest tests/integration -q
```

Both must pass. The `tests/integration` invocation is a no-op (all-skipped) when Docker is unavailable.

### Test counts (target)

Counting **collected pytest items** (post-parametrize expansion), not test functions:

- Contract: ~45 collected items.
  - `test_public_surface.py`: 1 item.
  - `test_audit_invariants.py`: 1 sanity test + 1 parametrized fault-injection test × 3-5 concrete sinks = ~5-6 items.
  - `test_health_invariants.py`: 1 sanity test + 1 parametrized fault-injection test × 3 probes = ~4 items.
  - `test_exception_invariants.py`: 2 parametrized tests × ~16 concrete exception classes = ~32 items.
  - `test_container_lifecycle.py`: 1 item.
- Integration: 6 collected items (3 database + 3 opa); all-skip when Docker unavailable.
- Net pytest delta: ~+45 items when Docker is unavailable; ~+51 items when Docker is up.

### Risk register

| Risk | Mitigation |
|---|---|
| Frozen 29-name surface set drifts as future phases add exports | Failure is the desired signal — contributors must explicitly update `EXPECTED_PUBLIC_NAMES`. Documented in the test file's docstring. |
| `IAuditSink.__subclasses__()` doesn't pick up `SentryAuditSink`/`DatadogAuditSink` because their modules aren't imported | Test fixture force-imports both in a try/except (skipping if optional dep missing). |
| Testcontainers `docker_available` probe is slow on first call | Session-scoped; called once per test run. |
| `openpolicyagent/opa:0.66.0` image disappears | Pin via the `OPA_IMAGE` constant. |
| `postgres:16` image gets a breaking change | Documented; bump as routine maintenance. |
| Contract test parametrization misses an existing concrete sink/probe | The `_all_concrete_sinks()` helper walks the subclass tree recursively. A "≥3 concrete sinks" assertion catches discovery regressions. |
| Integration tests block CI when Docker socket isn't mounted | Auto-skip is the mitigation; CI runs integration tests when Docker IS present (separate workflow step), not blocking the unit gate. |
| Per-sink fault-injection harness drifts as new sinks are added | Pattern documented; the parametrize fixture is the single point to extend. |
| `OPAPolicyEvaluator` may not be the correct class name | Implementation plan does a `grep -n "class.*PolicyEvaluator" src/ai_core/security/` to resolve. |

### End-of-phase smoke gate

- `pytest tests/unit tests/component tests/contract -q` green (≥450 passing total: 405 baseline + ~45 contract collected items).
- `pytest tests/integration -q` green when Docker is up (≥6 passing); all-skip otherwise.
- `ruff check src tests` total ≤ 211 (no new vs `d5f03d5`).
- `mypy src --strict` total ≤ 21.
- All 29 canonical names import.
- `eaap init` regression smoke (Phase 5 invariant) still produces a working scaffold.
- Manual sanity: with Docker running, `pytest tests/integration -v` produces real output.

### Coverage target

Contract tests don't materially change line coverage. Integration tests bump coverage on `health/probes.py::OPAReachabilityProbe`, `health/probes.py::DatabaseProbe`, and `security/opa_evaluator.py` (or wherever the OPA evaluator class lives) above current numbers since they exercise live HTTP/SQL paths.

## Implementation order (bottom-up)

| Step | Deliverable | Tests | Dependencies |
|---|---|---|---|
| 1 | Contract tests (5 files) + dev-dep updates | ~14 contract tests | none |
| 2 | Testcontainers infra + Postgres integration | ~3 integration tests | step 1 (shared dev-dep landed) |
| 3 | OPA integration | ~3 integration tests | step 2 (shares conftest) |
| 4 | End-of-phase smoke gate | full pytest + ruff + mypy + Docker-conditional integration | all |

Tasks 2 and 3 share `tests/integration/conftest.py`. Sequential ordering keeps the diff focused per task.

## Constraints — recap

- 4 implementation steps; per-step gate is ruff (no new) + mypy strict (touched files; src not changed) + pytest unit/component/contract + Docker-conditional pytest integration.
- Project mypy total stays ≤ 21.
- Project ruff total stays ≤ 211.
- End-of-phase smoke gate must pass before merge.
- New dev dependency: `testcontainers[postgres]>=4.0,<5.0`.
- No production code changes. Phase 7 is purely test-side.
