# ai-core-sdk Phase 4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cost/latency hardening (Anthropic prompt caching + MCP connection pooling) plus 8 Phase 3 polish items flagged Important by final review.

**Architecture:** Bottom-up — polish batch (8 small fixes) first → MCP pooling (foundational connection lifecycle) → prompt caching (LLM hot path). Per-step gate: ruff (no new violations) + mypy strict (touched files) + pytest unit/component. Project mypy total stays ≤ 21.

**Tech Stack:** Python 3.11+, Pydantic v2, `injector`, `litellm`, FastMCP, structlog, OpenTelemetry, `pytest` + ruff + mypy strict. Spec: `docs/superpowers/specs/2026-05-05-ai-core-sdk-phase-4-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-1-facade-tool-validation` (carries Phase 1 + 2 + 3 + Phase 4 spec).

**Working-state hygiene** — do NOT touch:
- `README.md`
- `src/ai_core/cli/main.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`

**Mypy baseline:** 21 strict errors in 8 files (post-Phase-3). Total must remain ≤ 21 after every commit.

**Per-step gate (every commit):**
- `ruff check <files-touched>` — no NEW violations vs the pre-task ruff state.
- `pytest tests/unit tests/component -q` — must pass (excluding pre-existing `respx`/`aiosqlite` collection errors).
- `mypy <files-touched-by-this-task>` — no new strict errors.
- `mypy src 2>&1 | tail -1` — total ≤ 21.

**Python interpreter:** `/Users/admin-h26/EAAP/.venv/bin/python` (no local venv inside the SDK).

**Per-task commit message convention:** Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `style:`, `build:`).

---

## Task 1 — Phase 3 polish batch (items 3-10)

Eight self-contained fixes. Done first to clear the deck before the larger pooling/caching work.

**Files:**
- Modify: `src/ai_core/config/settings.py` (item 3 — `opa_health_path`)
- Modify: `src/ai_core/health/probes.py` (items 3 + 4 — `OPAReachabilityProbe`, delete `SettingsProbe`)
- Modify: `src/ai_core/health/__init__.py` (item 4 — drop SettingsProbe export)
- Modify: `src/ai_core/di/module.py` (item 4 — drop SettingsProbe from default probes)
- Modify: `src/ai_core/app/runtime.py` (item 5 — HealthSnapshot MappingProxyType)
- Modify: `src/ai_core/tools/invoker.py` (items 6 + 7 + 10)
- Modify: `src/ai_core/observability/logging.py` (item 8 — docstring only)
- Modify: `src/ai_core/audit/otel_event.py` (item 9)
- Test: extensions to `test_probes.py`, `test_runtime.py`, `test_invoker.py`, `test_otel_event_sink.py`
- Test: NEW `tests/unit/health/test_opa_health_path.py`

### 1a — `opa_health_path` setting (item 3)

- [ ] **Step 1.1: Write failing test**

Create `tests/unit/health/test_opa_health_path.py`:

```python
"""Tests for the opa_health_path config wiring."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_core.config.settings import AppSettings, SecuritySettings
from ai_core.health.probes import OPAReachabilityProbe

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_opa_probe_uses_configured_health_path() -> None:
    """OPAReachabilityProbe constructs URL from settings.security.opa_health_path."""
    settings = AppSettings()
    settings.security = SecuritySettings(opa_health_path="/opa/health")
    probe = OPAReachabilityProbe(settings)

    fake_response = MagicMock(status_code=200)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()

    # Verify the URL contains the configured path.
    fake_client.get.assert_called_once()
    called_url = fake_client.get.call_args.args[0]
    assert called_url.endswith("/opa/health")
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_opa_probe_default_path_is_health() -> None:
    """When opa_health_path is unset, default is /health."""
    settings = AppSettings()
    probe = OPAReachabilityProbe(settings)

    fake_response = MagicMock(status_code=200)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        await probe.probe()

    called_url = fake_client.get.call_args.args[0]
    assert called_url.endswith("/health")


@pytest.mark.asyncio
async def test_opa_probe_path_without_leading_slash_normalised() -> None:
    """opa_health_path='health' (no leading slash) is normalised to '/health'."""
    settings = AppSettings()
    settings.security = SecuritySettings(opa_health_path="health")
    probe = OPAReachabilityProbe(settings)

    fake_response = MagicMock(status_code=200)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        await probe.probe()

    called_url = fake_client.get.call_args.args[0]
    # Should NOT contain "//" between base and path.
    assert "//health" not in called_url
    assert called_url.endswith("/health")
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/health/test_opa_health_path.py -v
```
Expected: failures because `SecuritySettings.opa_health_path` doesn't exist yet.

- [ ] **Step 1.3: Add the setting**

In `src/ai_core/config/settings.py`, find `class SecuritySettings(BaseSettings):` and add the field after the existing `fail_closed` field:

```python
    opa_health_path: str = Field(
        default="/health",
        description=(
            "Path appended to opa_url for the reachability probe. Override for "
            "deployments where OPA is mounted at a non-standard prefix "
            "(e.g., '/opa/health' behind an API gateway)."
        ),
    )
```

- [ ] **Step 1.4: Update `OPAReachabilityProbe.__init__`**

In `src/ai_core/health/probes.py`, find `class OPAReachabilityProbe`. Replace its `__init__`:

```python
    def __init__(self, settings: AppSettings) -> None:
        base = str(settings.security.opa_url).rstrip("/")
        path = settings.security.opa_health_path
        if not path.startswith("/"):
            path = "/" + path
        self._url = base + path
        self._timeout = settings.security.opa_request_timeout_seconds
```

- [ ] **Step 1.5: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/health/test_opa_health_path.py tests/unit/health/test_probes.py -v 2>&1 | tail -15
```
Expected: 3 new + existing probe tests all passing.

### 1b — Remove `SettingsProbe` (item 4)

- [ ] **Step 1.6: Delete the class from `src/ai_core/health/probes.py`**

Find `class SettingsProbe(IHealthProbe):` and delete the class definition. Also remove `SettingsProbe` from the `__all__` list at the bottom of the file.

- [ ] **Step 1.7: Update `src/ai_core/health/__init__.py`**

Remove `SettingsProbe` from the import block:

```python
# Before
from ai_core.health.probes import (
    DatabaseProbe,
    ModelLookupProbe,
    OPAReachabilityProbe,
    SettingsProbe,
)

# After
from ai_core.health.probes import (
    DatabaseProbe,
    ModelLookupProbe,
    OPAReachabilityProbe,
)
```

Remove `"SettingsProbe"` from `__all__`.

- [ ] **Step 1.8: Update `AgentModule.provide_health_probes`**

In `src/ai_core/di/module.py`, find `provide_health_probes`. Remove `SettingsProbe(settings)` from the returned list. Also remove the `SettingsProbe` from the imports if it's named explicitly:

```python
# Update the local import inside the method
from ai_core.health.probes import (  # noqa: PLC0415
    DatabaseProbe, ModelLookupProbe, OPAReachabilityProbe,
)
return [
    OPAReachabilityProbe(settings),
    DatabaseProbe(engine),
    ModelLookupProbe(settings),
]
```

- [ ] **Step 1.9: Delete the SettingsProbe tests**

In `tests/unit/health/test_probes.py`, find and delete:
- `test_settings_probe_always_ok`

In `tests/unit/health/test_never_raises.py`, find and delete:
- `test_settings_probe_never_raises`

In `tests/unit/health/test_interface.py`, no SettingsProbe-specific test exists (the file tests the abstract types only) — verify and skip if true.

- [ ] **Step 1.10: Update `tests/unit/app/test_runtime.py`**

Find every test asserting on `health.components` keys. Look for `"settings"` in expected key sets. Update:

```bash
grep -n '"settings"\|set\(snap.components.keys' tests/unit/app/test_runtime.py
```

For each match in test code that includes `"settings"` among the expected `snap.components` keys, remove the `"settings"` literal from the expected set/list. The remaining real-probe keys come from each probe's `component` class attribute:

- `OPAReachabilityProbe.component` → `"opa"`
- `DatabaseProbe.component` → `"database"`
- `ModelLookupProbe.component` → `"model_lookup"`

Read those three classes once (`grep -n 'component =' src/ai_core/health/probes.py`) to confirm the literals before editing the assertions. The post-Phase-4 expected set should contain exactly those three keys (no `"settings"`).

- [ ] **Step 1.11: Run tests to verify no regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/health tests/unit/app -v 2>&1 | tail -20
```
Expected: all health + app tests pass; no `SettingsProbe` references remain.

### 1c — `HealthSnapshot` immutable interior (item 5)

- [ ] **Step 1.12: Write failing test**

Append to `tests/unit/app/test_runtime.py`:

```python
@pytest.mark.asyncio
async def test_health_snapshot_components_is_read_only(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """HealthSnapshot.components should be a read-only Mapping (MappingProxyType)
    so callers can't mutate the snapshot."""
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        snap = await app.health()
    # Mutation should raise TypeError (frozen-style behavior).
    with pytest.raises(TypeError):
        snap.components["x"] = "ok"  # type: ignore[index]
    with pytest.raises(TypeError):
        snap.component_details["x"] = "y"  # type: ignore[index]
```

- [ ] **Step 1.13: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py::test_health_snapshot_components_is_read_only -v
```
Expected: fails (mutation currently succeeds because `components` is a regular dict).

- [ ] **Step 1.14: Modify `HealthSnapshot`**

In `src/ai_core/app/runtime.py`, find `class HealthSnapshot:`. Add the `MappingProxyType` import at the top of the file:

```python
from types import MappingProxyType
```

Update the dataclass:

```python
@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Coarse application health snapshot returned by :py:meth:`AICoreApp.health`."""

    status: HealthStatus
    components: Mapping[str, HealthStatus]
    component_details: Mapping[str, str | None]
    service_name: str

    def __post_init__(self) -> None:
        # Wrap the dicts in read-only MappingProxyType so callers can't mutate
        # the snapshot. Mutation attempts raise TypeError, matching frozen=True semantics.
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "component_details", MappingProxyType(dict(self.component_details)))
```

(Field annotations change from `dict[str, ...]` to `Mapping[str, ...]`. `Mapping` should already be imported via `collections.abc`; verify.)

- [ ] **Step 1.15: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py -v 2>&1 | tail -15
```
Expected: all passing.

### 1d — `AuditRecord.now()` skip for `NullAuditSink` (item 6)

- [ ] **Step 1.16: Write failing test**

Append to `tests/unit/tools/test_invoker.py`:

```python
@pytest.mark.asyncio
async def test_invoker_skips_audit_record_allocation_for_null_sink(
    fake_observability, fake_policy_evaluator_factory, monkeypatch
) -> None:
    """When the audit sink is NullAuditSink (default), AuditRecord.now() should not be called."""
    from ai_core.audit import AuditRecord
    from ai_core.audit.null import NullAuditSink

    inv = ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(default_allow=True),
        registry=SchemaRegistry(),
        audit=NullAuditSink(),  # default
    )

    # Spy on AuditRecord.now to verify it is NOT called.
    call_count = 0
    original_now = AuditRecord.now

    @classmethod  # type: ignore[misc]
    def _spy_now(cls, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_now(*args, **kwargs)

    monkeypatch.setattr(AuditRecord, "now", _spy_now)

    await inv.invoke(_search, {"q": "x", "limit": 1}, agent_id="a")
    assert call_count == 0, f"AuditRecord.now called {call_count} times for NullAuditSink"
```

- [ ] **Step 1.17: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py::test_invoker_skips_audit_record_allocation_for_null_sink -v
```
Expected: fails — `AuditRecord.now` is currently called regardless.

- [ ] **Step 1.18: Add the `_records_audit` gate flag**

In `src/ai_core/tools/invoker.py`, find `__init__` and add the flag after the existing `self._audit` assignment:

```python
def __init__(
    self,
    *,
    observability: IObservabilityProvider,
    policy: IPolicyEvaluator | None = None,
    registry: SchemaRegistry | None = None,
    audit: IAuditSink | None = None,
) -> None:
    from ai_core.audit.null import NullAuditSink  # noqa: PLC0415
    self._observability = observability
    self._policy = policy
    self._registry = registry
    self._audit: IAuditSink = audit or NullAuditSink()
    # Phase 4: skip AuditRecord.now() allocation entirely when the sink is no-op.
    self._records_audit: bool = not isinstance(self._audit, NullAuditSink)
```

- [ ] **Step 1.19: Gate every audit call site**

Find every `await self._audit.record(AuditRecord.now(...))` in `invoke`. Wrap each with `if self._records_audit:`:

```python
# Before
await self._audit.record(AuditRecord.now(
    AuditEvent.POLICY_DECISION,
    ...
))

# After
if self._records_audit:
    await self._audit.record(AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        ...
    ))
```

There are three sites (POLICY_DECISION, TOOL_INVOCATION_COMPLETED, TOOL_INVOCATION_FAILED). Apply to all three.

- [ ] **Step 1.20: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py -v 2>&1 | tail -20
```
Expected: all passing.

### 1e — `principal` in audit payload (item 7)

- [ ] **Step 1.21: Write failing test**

Append to `tests/unit/tools/test_invoker.py`:

```python
@pytest.mark.asyncio
async def test_invoker_records_principal_in_audit_payload(
    fake_observability, fake_policy_evaluator_factory, fake_audit_sink
) -> None:
    """POLICY_DECISION audit record should include 'user' field with principal data."""
    from ai_core.audit import AuditEvent

    inv = ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(default_allow=True),
        registry=SchemaRegistry(),
        audit=fake_audit_sink,
    )
    await inv.invoke(
        _search, {"q": "x", "limit": 1},
        principal={"sub": "user-42", "groups": ["admin"]},
        agent_id="a", tenant_id="t",
    )

    policy_records = [r for r in fake_audit_sink.records if r.event == AuditEvent.POLICY_DECISION]
    assert len(policy_records) == 1
    payload = policy_records[0].payload
    assert "user" in payload
    assert payload["user"] == {"sub": "user-42", "groups": ["admin"]}
    # Also: the existing input field is preserved.
    assert "input" in payload
```

- [ ] **Step 1.22: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py::test_invoker_records_principal_in_audit_payload -v
```
Expected: fails because the `payload` currently doesn't include `"user"`.

- [ ] **Step 1.23: Update the POLICY_DECISION audit record**

In `src/ai_core/tools/invoker.py`, find the `POLICY_DECISION` audit call (around line 142). Update the `payload` arg:

```python
# Before
payload={"input": payload.model_dump()},

# After
payload={
    "input": payload.model_dump(),
    "user": dict(principal or {}),
},
```

- [ ] **Step 1.24: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py -v 2>&1 | tail -15
```
Expected: all passing.

### 1f — `_ContextVarMergingDict` precedence docstring (item 8)

- [ ] **Step 1.25: Update `bind_context` and module docstrings**

In `src/ai_core/observability/logging.py`, find `def bind_context(...)`. Update the docstring:

```python
def bind_context(**kwargs: Any) -> dict[str, contextvars.Token[Any]]:
    """Bind ``kwargs`` as ContextVar fields visible on every log line.

    Returns a token-map; pass it to :func:`unbind_context` to release.

    Precedence:
        Explicit logger keyword arguments take precedence over ContextVar-bound
        values when keys collide. After ``bind_context(agent_id="A")``, a call
        ``logger.warning("event", agent_id="B")`` emits ``agent_id="B"``. This
        matches structlog's ``_ContextVarMergingDict`` semantics — the per-call
        kwargs are the most specific binding.
    """
    ...
```

Then find the module-level docstring at the top of the file. Append to the asyncio-isolation paragraph:

```
The merge precedence is **explicit kwargs > ContextVar fields**: a per-call
``logger.warning("event", agent_id="B")`` overrides ``bind_context(agent_id="A")``.
```

(Documentation only — no test changes.)

### 1g — `OTelEventAuditSink` `decision_allowed=None` → `""` (item 9)

- [ ] **Step 1.26: Write failing test**

Append to `tests/unit/audit/test_otel_event_sink.py`:

```python
@pytest.mark.asyncio
async def test_otel_event_sink_emits_empty_string_for_undefined_decision(
    fake_observability,
) -> None:
    """When AuditRecord.decision_allowed is None (no policy decision),
    OTel attribute audit.decision_allowed should be empty string, not False."""
    sink = OTelEventAuditSink(fake_observability)
    await sink.record(AuditRecord.now(
        AuditEvent.TOOL_INVOCATION_COMPLETED,
        tool_name="x", tool_version=1,
        agent_id="a", tenant_id="t",
        # decision_allowed not set — defaults to None
    ))

    events = fake_observability.events
    matching = next(attrs for name, attrs in events
                     if name == "eaap.audit.tool.invocation.completed")
    assert matching["audit.decision_allowed"] == ""
```

- [ ] **Step 1.27: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_otel_event_sink.py::test_otel_event_sink_emits_empty_string_for_undefined_decision -v
```
Expected: fails — currently emits `False`.

- [ ] **Step 1.28: Update the `_record_to_attributes` helper**

In `src/ai_core/audit/otel_event.py`, find `_record_to_attributes`. Change the `audit.decision_allowed` line:

```python
# Before
"audit.decision_allowed": (
    record.decision_allowed if record.decision_allowed is not None else False
),

# After
"audit.decision_allowed": (
    record.decision_allowed if record.decision_allowed is not None else ""
),
```

- [ ] **Step 1.29: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_otel_event_sink.py -v
```
Expected: all passing.

### 1h — `tool.completed` post-span calls protected (item 10)

- [ ] **Step 1.30: Write failing test**

Append to `tests/unit/tools/test_invoker.py`:

```python
@pytest.mark.asyncio
async def test_invoker_swallows_tool_completed_event_failure(
    fake_policy_evaluator_factory, fake_audit_sink
) -> None:
    """A failure in the post-span record_event('tool.completed') must NOT propagate
    as a tool failure — the user's tool call already succeeded."""
    # FakeObservabilityProvider that fails on record_event.
    class _BadObs:
        spans: list = []
        def start_span(self, name, *, attributes=None):
            from contextlib import asynccontextmanager
            from ai_core.di.interfaces import SpanContext
            @asynccontextmanager
            async def _cm():
                yield SpanContext(name=name, trace_id="t", span_id="s",
                                  backend_handles={})
            return _cm()
        async def record_llm_usage(self, **kwargs): pass
        async def record_event(self, name, *, attributes=None):
            raise RuntimeError("backend down (test-injected)")
        async def shutdown(self): pass

    inv = ToolInvoker(
        observability=_BadObs(),  # type: ignore[arg-type]
        policy=fake_policy_evaluator_factory(default_allow=True),
        registry=SchemaRegistry(),
        audit=fake_audit_sink,
    )
    # Should NOT raise — the user's tool call returns successfully.
    result = await inv.invoke(_search, {"q": "x", "limit": 1}, agent_id="a")
    assert result == {"items": ["x"]}
```

- [ ] **Step 1.31: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py::test_invoker_swallows_tool_completed_event_failure -v
```
Expected: fails — `record_event` failure currently propagates as a tool failure.

- [ ] **Step 1.32: Wrap the post-span `record_event` in try/except**

In `src/ai_core/tools/invoker.py`, find the post-span `record_event("tool.completed", ...)` call. Wrap with try/except:

```python
# Before
latency_ms = (time.monotonic() - started) * 1000.0
await self._observability.record_event(
    "tool.completed",
    attributes={**attrs, "latency_ms": latency_ms},
)

# After
latency_ms = (time.monotonic() - started) * 1000.0
try:
    await self._observability.record_event(
        "tool.completed",
        attributes={**attrs, "latency_ms": latency_ms},
    )
except Exception as exc:  # noqa: BLE001 — observability boundary; never fail the tool result
    _logger.warning(
        "tool.completed_event_failed",
        tool_name=spec.name, agent_id=agent_id, tenant_id=tenant_id,
        error=str(exc), error_type=type(exc).__name__,
    )
```

(The `_logger` is already imported via `get_logger(__name__)` from Phase 3 Task 2.)

- [ ] **Step 1.33: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py -v 2>&1 | tail -15
```
Expected: all passing.

### 1i — Lint, type-check, commit

- [ ] **Step 1.34: Lint + type-check the full Task 1 change set**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/config/settings.py \
    src/ai_core/health/probes.py \
    src/ai_core/health/__init__.py \
    src/ai_core/di/module.py \
    src/ai_core/app/runtime.py \
    src/ai_core/tools/invoker.py \
    src/ai_core/observability/logging.py \
    src/ai_core/audit/otel_event.py \
    tests/unit/health/ \
    tests/unit/app/test_runtime.py \
    tests/unit/tools/test_invoker.py \
    tests/unit/audit/test_otel_event_sink.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/config/settings.py \
    src/ai_core/health/probes.py \
    src/ai_core/health/__init__.py \
    src/ai_core/di/module.py \
    src/ai_core/app/runtime.py \
    src/ai_core/tools/invoker.py \
    src/ai_core/observability/logging.py \
    src/ai_core/audit/otel_event.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: no new violations; mypy ≤ 21.

- [ ] **Step 1.35: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```
Expected: 305+ passing; 9 pre-existing errors unchanged.

- [ ] **Step 1.36: Commit**

```bash
git add src/ai_core/config/settings.py \
        src/ai_core/health/probes.py \
        src/ai_core/health/__init__.py \
        src/ai_core/di/module.py \
        src/ai_core/app/runtime.py \
        src/ai_core/tools/invoker.py \
        src/ai_core/observability/logging.py \
        src/ai_core/audit/otel_event.py \
        tests/unit/health/ \
        tests/unit/app/test_runtime.py \
        tests/unit/tools/test_invoker.py \
        tests/unit/audit/test_otel_event_sink.py
git commit -m "refactor: Phase 3 polish batch — opa_health_path, drop SettingsProbe, immutable HealthSnapshot, NullAuditSink skip, principal in audit, decision_allowed empty string, tool.completed swallow"
```

---

## Task 2 — MCP connection pooling

**Files:**
- Create: `src/ai_core/mcp/_pool.py`
- Modify: `src/ai_core/mcp/transports.py` (rename + alias + integrate pool)
- Modify: `src/ai_core/config/settings.py` (add `MCPSettings`)
- Modify: `src/ai_core/di/module.py` (wire settings into `provide_mcp_connection_factory`)
- Modify: `src/ai_core/di/container.py` (add `mcp_pool.aclose` teardown step)
- Test: `tests/unit/mcp/test_pool.py` (new)
- Test: extensions to `tests/unit/mcp/test_transports.py`

### 2a — Add `MCPSettings`

- [ ] **Step 2.1: Add the settings class**

In `src/ai_core/config/settings.py`, add the new class before `class AppSettings`:

```python
class MCPSettings(BaseSettings):
    """MCP transport / pool configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    pool_enabled: bool = Field(
        default=True,
        description=(
            "When True (default), MCP connections are pooled per component_id "
            "and reused across calls. Set False for debugging or to ensure "
            "every call uses a fresh transport."
        ),
    )
    pool_idle_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description=(
            "Connections idle longer than this are closed and reopened on next "
            "checkout. Default 5 minutes — matches typical server-side timeout."
        ),
    )
```

In `class AppSettings(BaseSettings):`, add the new field next to other groups:

```python
mcp: MCPSettings = Field(default_factory=MCPSettings)
```

### 2b — Implement the pool

- [ ] **Step 2.2: Write failing tests**

```bash
mkdir -p tests/unit/mcp
[ -f tests/unit/mcp/__init__.py ] || touch tests/unit/mcp/__init__.py
```

Create `tests/unit/mcp/test_pool.py`:

```python
"""Tests for the internal MCP connection pool."""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from ai_core.exceptions import MCPTransportError
from ai_core.mcp._pool import _MCPConnectionPool
from ai_core.mcp.transports import MCPServerSpec

pytestmark = pytest.mark.unit


class _FakeFastMCPClient:
    """Records open/close + use calls for assertion."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.uses = 0
        self.closed = False

    async def use(self) -> None:
        if self.closed:
            raise RuntimeError(f"{self.label} closed")
        self.uses += 1


def _make_opener() -> tuple[Any, list[_FakeFastMCPClient]]:
    """Return (opener_callable, opened_clients_list)."""
    opened: list[_FakeFastMCPClient] = []

    def _opener(spec: MCPServerSpec):
        @asynccontextmanager
        async def _cm() -> AsyncIterator[_FakeFastMCPClient]:
            client = _FakeFastMCPClient(label=spec.component_id)
            opened.append(client)
            try:
                yield client
            finally:
                client.closed = True
        return _cm()

    return _opener, opened


def _spec(component_id: str = "s1") -> MCPServerSpec:
    return MCPServerSpec(
        component_id=component_id, transport="stdio", target="/usr/bin/echo",
    )


@pytest.mark.asyncio
async def test_pool_reuses_connection_for_same_spec() -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)

    async with pool.acquire(_spec()) as c1:
        await c1.use()
    async with pool.acquire(_spec()) as c2:
        await c2.use()

    assert len(opened) == 1, "second call should reuse the first connection"
    assert opened[0].uses == 2

    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_opens_separate_connections_for_different_specs() -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)

    async with pool.acquire(_spec("s1")):
        pass
    async with pool.acquire(_spec("s2")):
        pass

    assert len(opened) == 2
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_evicts_stale_connection_after_idle_ttl(monkeypatch) -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=0.05)

    async with pool.acquire(_spec()):
        pass

    # Wait beyond idle TTL.
    await asyncio.sleep(0.1)

    async with pool.acquire(_spec()):
        pass

    assert len(opened) == 2, "stale connection should have been reopened"
    assert opened[0].closed is True
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_evicts_connection_on_in_flight_error() -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)

    with pytest.raises(RuntimeError, match="boom"):
        async with pool.acquire(_spec()) as c:
            assert c is not None
            raise RuntimeError("boom")

    # Next acquire should open a fresh connection (the broken one was evicted).
    async with pool.acquire(_spec()):
        pass

    assert len(opened) == 2
    assert opened[0].closed is True
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_aclose_closes_all_connections() -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)

    async with pool.acquire(_spec("s1")):
        pass
    async with pool.acquire(_spec("s2")):
        pass

    await pool.aclose()

    assert all(c.closed for c in opened)


@pytest.mark.asyncio
async def test_pool_acquire_after_aclose_raises_mcp_transport_error() -> None:
    opener, _ = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)
    await pool.aclose()

    with pytest.raises(MCPTransportError) as exc:
        async with pool.acquire(_spec()):
            pass
    assert "closed" in exc.value.message.lower()
```

- [ ] **Step 2.3: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/mcp/test_pool.py -q
```
Expected: ImportError on `ai_core.mcp._pool`.

- [ ] **Step 2.4: Create `src/ai_core/mcp/_pool.py`**

```python
"""Internal MCP connection pool — single-connection-per-spec with idle TTL.

Not part of the SDK's public API. Consumers reach this via
:class:`PoolingMCPConnectionFactory` in :mod:`ai_core.mcp.transports`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_core.exceptions import MCPTransportError
from ai_core.observability.logging import get_logger

if TYPE_CHECKING:
    from ai_core.mcp.transports import MCPServerSpec

_logger = get_logger(__name__)

_OpenerFn = Callable[["MCPServerSpec"], AbstractAsyncContextManager[Any]]


@dataclass(slots=True)
class _PooledConnection:
    """Live FastMCP client + its enclosing context manager + last-used bookkeeping."""

    client: Any
    cm: AbstractAsyncContextManager[Any]
    spec: "MCPServerSpec"
    last_used: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _MCPConnectionPool:
    """Per-component_id single-connection pool.

    Concurrency model:
        Each component_id has at most one connection. Concurrent calls to the
        same component_id serialize on the connection's ``lock``. The
        ``connections`` dict is guarded by ``self._lock`` to avoid races
        when two callers race for a missing connection.

    Lifecycle:
        ``aclose()`` closes every pooled connection. Called by Container teardown.

    Idle TTL:
        On checkout, if ``time.monotonic() - last_used > idle_seconds``, the
        connection is torn down and reopened.
    """

    def __init__(
        self,
        *,
        opener: _OpenerFn,
        idle_seconds: float,
    ) -> None:
        self._opener = opener
        self._idle_seconds = idle_seconds
        self._connections: dict[str, _PooledConnection] = {}
        self._lock = asyncio.Lock()
        self._closed: bool = False

    @asynccontextmanager
    async def acquire(self, spec: "MCPServerSpec") -> AsyncIterator[Any]:
        """Yield a live FastMCP client for ``spec``. Serialised per spec."""
        if self._closed:
            raise MCPTransportError(
                "MCP connection pool is closed",
                details={"component_id": spec.component_id, "transport": spec.transport},
            )

        async with self._lock:
            entry = self._connections.get(spec.component_id)
            if entry is None or self._is_stale(entry):
                if entry is not None:
                    await self._close_one(entry)
                entry = await self._open_entry(spec)
                self._connections[spec.component_id] = entry

        async with entry.lock:
            try:
                yield entry.client
                entry.last_used = time.monotonic()
            except Exception:
                async with self._lock:
                    if self._connections.get(spec.component_id) is entry:
                        del self._connections[spec.component_id]
                await self._close_one(entry)
                raise

    def _is_stale(self, entry: _PooledConnection) -> bool:
        return (time.monotonic() - entry.last_used) > self._idle_seconds

    async def _open_entry(self, spec: "MCPServerSpec") -> _PooledConnection:
        cm = self._opener(spec)
        try:
            client = await cm.__aenter__()
        except Exception as exc:
            raise MCPTransportError(
                f"MCP transport '{spec.transport}' connection failed: {exc}",
                details={"component_id": spec.component_id, "transport": spec.transport},
                cause=exc,
            ) from exc
        return _PooledConnection(
            client=client, cm=cm, spec=spec, last_used=time.monotonic(),
        )

    async def _close_one(self, entry: _PooledConnection) -> None:
        try:
            await entry.cm.__aexit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "mcp.pool.connection_close_failed",
                component_id=entry.spec.component_id,
                transport=entry.spec.transport,
                error=str(exc), error_type=type(exc).__name__,
            )

    async def aclose(self) -> None:
        """Close every pooled connection. Idempotent."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._connections.values())
            self._connections.clear()
        for entry in entries:
            await self._close_one(entry)


__all__ = ["_MCPConnectionPool"]
```

- [ ] **Step 2.5: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/mcp/test_pool.py -v
```
Expected: 6 passed.

### 2c — Modify `transports.py` to use the pool

- [ ] **Step 2.6: Write integration tests**

Append to `tests/unit/mcp/test_transports.py`:

```python
@pytest.mark.asyncio
async def test_factory_pool_enabled_reuses_connection(monkeypatch) -> None:
    """When pool_enabled=True, two open() calls for same spec reuse the connection."""
    import sys, types
    factory = PoolingMCPConnectionFactory(pool_enabled=True, pool_idle_seconds=300.0)
    spec = MCPServerSpec(component_id="reuse-test", transport="stdio", target="/usr/bin/echo")

    opened: list[Any] = []

    class _BoomClient:
        def __init__(self, *args, **kwargs): opened.append(self); self.closed = False
        async def __aenter__(self): return self
        async def __aexit__(self, *_): self.closed = True

    fake_fastmcp = types.SimpleNamespace(Client=_BoomClient)
    fake_transports = types.SimpleNamespace(
        StdioTransport=lambda **kw: object(),
        SSETransport=lambda **kw: object(),
        StreamableHttpTransport=lambda **kw: object(),
    )
    with patch.dict(sys.modules, {
        "fastmcp": fake_fastmcp,
        "fastmcp.client.transports": fake_transports,
    }):
        async with factory.open(spec):
            pass
        async with factory.open(spec):
            pass

    # With pooling, only ONE _BoomClient should be constructed.
    assert len(opened) == 1
    await factory.aclose()


@pytest.mark.asyncio
async def test_factory_pool_disabled_opens_fresh_each_call(monkeypatch) -> None:
    """When pool_enabled=False, each open() call constructs a new client."""
    import sys, types
    factory = PoolingMCPConnectionFactory(pool_enabled=False)
    spec = MCPServerSpec(component_id="fresh-test", transport="stdio", target="/usr/bin/echo")

    opened: list[Any] = []

    class _BoomClient:
        def __init__(self, *args, **kwargs): opened.append(self)
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None

    fake_fastmcp = types.SimpleNamespace(Client=_BoomClient)
    fake_transports = types.SimpleNamespace(
        StdioTransport=lambda **kw: object(),
        SSETransport=lambda **kw: object(),
        StreamableHttpTransport=lambda **kw: object(),
    )
    with patch.dict(sys.modules, {
        "fastmcp": fake_fastmcp,
        "fastmcp.client.transports": fake_transports,
    }):
        async with factory.open(spec):
            pass
        async with factory.open(spec):
            pass

    assert len(opened) == 2
    await factory.aclose()


@pytest.mark.asyncio
async def test_factory_aclose_drains_pool_when_enabled() -> None:
    """factory.aclose() closes all pooled connections (pool_enabled=True case)."""
    factory = PoolingMCPConnectionFactory(pool_enabled=True)
    # No opens — aclose should be a safe no-op.
    await factory.aclose()
    # Second call also OK.
    await factory.aclose()


@pytest.mark.asyncio
async def test_factory_aclose_noop_when_pool_disabled() -> None:
    """factory.aclose() is a safe no-op when pool_enabled=False."""
    factory = PoolingMCPConnectionFactory(pool_enabled=False)
    await factory.aclose()
```

- [ ] **Step 2.7: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/mcp/test_transports.py -v 2>&1 | tail -10
```
Expected: failures because `PoolingMCPConnectionFactory` doesn't exist yet.

- [ ] **Step 2.8: Refactor `transports.py`**

In `src/ai_core/mcp/transports.py`, perform a **rename + add wrapping**, not a full rewrite:

1. **Rename the class**: change `class FastMCPConnectionFactory(IMCPConnectionFactory):` to `class PoolingMCPConnectionFactory(IMCPConnectionFactory):`. Update the docstring to read:

   ```
   FastMCP-backed factory with optional per-spec connection pooling.

   When ``pool_enabled=True`` (default), connections are reused across calls
   until they exceed ``pool_idle_seconds`` of inactivity. When ``False``,
   each ``open()`` returns a fresh CM (pre-Phase-4 behaviour).

   Closed at app shutdown via :meth:`aclose` (called by
   ``Container._teardown_sdk_resources``).

   Raises:
       MCPTransportError: If FastMCP is not installed, transport-class import
           fails, or the connection itself fails.
       RegistryError: If ``spec.transport`` is not a supported value.
   ```

2. **Replace the existing `__init__`** with the pool-aware version:

   ```python
   def __init__(self, *, pool_enabled: bool = True,
                pool_idle_seconds: float = 300.0) -> None:
       from ai_core.mcp._pool import _MCPConnectionPool  # noqa: PLC0415
       self._pool_enabled = pool_enabled
       self._pool: _MCPConnectionPool | None = (
           _MCPConnectionPool(opener=self._open, idle_seconds=pool_idle_seconds)
           if pool_enabled else None
       )
   ```

3. **Replace the existing `open` method** with this dispatching version:

   ```python
   def open(self, spec: MCPServerSpec) -> AbstractAsyncContextManager[Any]:
       if self._pool is not None:
           return self._pool.acquire(spec)
       return self._open(spec)
   ```

4. **Preserve the existing `_open` method body verbatim.** It already has signature `@asynccontextmanager async def _open(self, spec: MCPServerSpec) -> AsyncIterator[Any]:` and contains all the FastMCP transport-construction code (Client + StdioTransport / SSETransport / StreamableHttpTransport branches + MCPTransportError wrapping + RegistryError on unknown transport). Do NOT delete or rewrite this method — only its callers change.

5. **Add `aclose` method** at the end of the class:

   ```python
   async def aclose(self) -> None:
       """Close all pooled connections. Idempotent. No-op when pool disabled."""
       if self._pool is not None:
           await self._pool.aclose()
   ```

6. **Add the alias** at module bottom (just before `__all__`):

   ```python
   # Pre-1.0 alias — kept for downstream importers.
   FastMCPConnectionFactory = PoolingMCPConnectionFactory
   ```

Update `__all__` at the bottom of the file:

```python
__all__ = [
    "MCPServerSpec",
    "MCPTransport",
    "IMCPConnectionFactory",
    "PoolingMCPConnectionFactory",
    "FastMCPConnectionFactory",  # alias
]
```

Add the imports needed at the top:

```python
from collections.abc import AsyncIterator
```

(The `AbstractAsyncContextManager` and `asynccontextmanager` should already be imported.)

- [ ] **Step 2.9: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/mcp/ -v 2>&1 | tail -20
```
Expected: all passing (existing + new).

### 2d — DI binding + Container teardown

- [ ] **Step 2.10: Update `provide_mcp_connection_factory`**

In `src/ai_core/di/module.py`, find `provide_mcp_connection_factory`. Update its signature and body to accept settings:

```python
@singleton
@provider
def provide_mcp_connection_factory(self, settings: AppSettings) -> IMCPConnectionFactory:
    """Return the pooling MCP connection factory."""
    return PoolingMCPConnectionFactory(
        pool_enabled=settings.mcp.pool_enabled,
        pool_idle_seconds=settings.mcp.pool_idle_seconds,
    )
```

Verify the import at the top of the file uses `PoolingMCPConnectionFactory` (or via the alias `FastMCPConnectionFactory` — both work).

- [ ] **Step 2.11: Update `Container._teardown_sdk_resources`**

In `src/ai_core/di/container.py`, find `_teardown_sdk_resources`. Add the new step BEFORE `engine.dispose`:

```python
from ai_core.mcp.transports import IMCPConnectionFactory  # add to local imports at top of method

steps: list[tuple[str, type[Any], tuple[str, ...]]] = [
    ("observability.shutdown", IObservabilityProvider, ("shutdown",)),
    ("audit.flush", IAuditSink, ("flush",)),
    ("mcp_pool.aclose", IMCPConnectionFactory, ("aclose",)),  # NEW
    ("policy_evaluator.aclose", IPolicyEvaluator, ("aclose",)),
    ("engine.dispose", AsyncEngine, ("dispose",)),
]
```

- [ ] **Step 2.12: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```
Expected: 305+ passing.

### 2e — Lint, type-check, commit

- [ ] **Step 2.13: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/mcp/_pool.py \
    src/ai_core/mcp/transports.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py \
    src/ai_core/di/container.py \
    tests/unit/mcp/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/mcp/_pool.py \
    src/ai_core/mcp/transports.py \
    src/ai_core/di/module.py \
    src/ai_core/di/container.py \
    tests/unit/mcp/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: no new violations; mypy ≤ 21.

- [ ] **Step 2.14: Commit**

```bash
git add src/ai_core/mcp/_pool.py \
        src/ai_core/mcp/transports.py \
        src/ai_core/config/settings.py \
        src/ai_core/di/module.py \
        src/ai_core/di/container.py \
        tests/unit/mcp/
git commit -m "feat(mcp): connection pooling — _MCPConnectionPool + PoolingMCPConnectionFactory + DI wiring"
```

---

## Task 3 — Anthropic prompt caching

**Files:**
- Create: `src/ai_core/llm/_prompt_cache.py`
- Modify: `src/ai_core/llm/litellm_client.py` (call apply_prompt_cache)
- Modify: `src/ai_core/config/settings.py` (LLMSettings additions)
- Test: `tests/unit/llm/test_prompt_cache.py` (new)
- Test: extensions to `tests/unit/llm/test_litellm_client.py`

### 3a — Pure-function helpers

- [ ] **Step 3.1: Write failing tests**

Create `tests/unit/llm/test_prompt_cache.py`:

```python
"""Tests for the prompt-cache pure-function helpers."""
from __future__ import annotations

from typing import Any

import pytest

from ai_core.llm._prompt_cache import apply_prompt_cache, supports_prompt_cache

pytestmark = pytest.mark.unit


# --- supports_prompt_cache ---

@pytest.mark.parametrize("model", [
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-haiku",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-7",
])
def test_supports_prompt_cache_anthropic_models(model: str) -> None:
    assert supports_prompt_cache(model) is True


@pytest.mark.parametrize("model", [
    "openai/gpt-4o",
    "gpt-4-turbo",
    "bedrock/amazon.titan-text-express-v1",
    "vertex_ai/claude-3-5-sonnet",  # Vertex AI prefix not supported in Phase 4
    "azure/gpt-4",
])
def test_supports_prompt_cache_rejects_non_anthropic(model: str) -> None:
    assert supports_prompt_cache(model) is False


# --- apply_prompt_cache: skip cases ---

def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def _conversation() -> list[dict[str, Any]]:
    """Build a typical 6-turn conversation for cache application."""
    return [
        _msg("system", "You are a helpful assistant."),
        _msg("user", "Question 1"),
        _msg("assistant", "Answer 1"),
        _msg("user", "Question 2"),
        _msg("assistant", "Answer 2"),
        _msg("user", "Question 3"),  # latest user turn
    ]


def test_apply_skipped_when_disabled() -> None:
    msgs = _conversation()
    result_msgs, result_tools = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=False, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    # Returns originals — no cache_control inserted.
    assert all(isinstance(m["content"], str) for m in result_msgs)


def test_apply_skipped_for_non_anthropic_model() -> None:
    msgs = _conversation()
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="openai/gpt-4o",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    assert all(isinstance(m["content"], str) for m in result_msgs)


def test_apply_skipped_below_message_threshold() -> None:
    msgs = [_msg("system", "sys"), _msg("user", "hi")]
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=True, min_messages=10, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    assert all(isinstance(m["content"], str) for m in result_msgs)


def test_apply_skipped_below_token_threshold() -> None:
    msgs = _conversation()
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=10_000,
        estimated_tokens=500,
    )
    assert all(isinstance(m["content"], str) for m in result_msgs)


# --- apply_prompt_cache: applied cases ---

def test_apply_inserts_breakpoint_on_system_prompt() -> None:
    msgs = _conversation()
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="anthropic/claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    # System message should have its content as a list with cache_control on the last block.
    sys_msg = result_msgs[0]
    assert sys_msg["role"] == "system"
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_apply_inserts_breakpoint_on_last_stable_assistant() -> None:
    msgs = _conversation()
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="anthropic/claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    # The assistant message right before the latest user turn (index 4) should have cache_control.
    last_stable_assistant = result_msgs[4]
    assert last_stable_assistant["role"] == "assistant"
    assert isinstance(last_stable_assistant["content"], list)
    assert last_stable_assistant["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_apply_no_assistant_yet_skips_history_breakpoint() -> None:
    """When conversation has only system + user (no assistant yet), only system gets cached."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "hi"),
        _msg("user", "follow-up"),  # no assistant turn yet
    ]
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    # System has cache_control.
    assert isinstance(result_msgs[0]["content"], list)
    # No assistant message → no second breakpoint.
    user_msgs = [m for m in result_msgs[1:] if m["role"] == "user"]
    for m in user_msgs:
        if isinstance(m["content"], list):
            for block in m["content"]:
                assert "cache_control" not in block, "user messages should NOT be cached"


def test_apply_handles_pre_structured_content() -> None:
    """When content is already a list of blocks, cache_control is added to the LAST block."""
    msgs = [
        {"role": "system", "content": [
            {"type": "text", "text": "block 1"},
            {"type": "text", "text": "block 2"},
        ]},
        _msg("user", "hi"),
        _msg("assistant", "hello"),
        _msg("user", "again"),
    ]
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    sys_content = result_msgs[0]["content"]
    assert sys_content[0].get("cache_control") is None
    assert sys_content[1].get("cache_control") == {"type": "ephemeral"}


def test_apply_caches_last_tool_when_tools_present() -> None:
    msgs = _conversation()
    tools = [
        {"type": "function", "function": {"name": "tool_a", "description": "...", "parameters": {}}},
        {"type": "function", "function": {"name": "tool_b", "description": "...", "parameters": {}}},
    ]
    _, result_tools = apply_prompt_cache(
        msgs, tools=tools, model="claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    assert result_tools is not None
    # First tool unchanged.
    assert "cache_control" not in result_tools[0]
    # Last tool gets cache_control.
    assert result_tools[-1].get("cache_control") == {"type": "ephemeral"}
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_prompt_cache.py -q
```
Expected: ImportError on `ai_core.llm._prompt_cache`.

- [ ] **Step 3.3: Implement `src/ai_core/llm/_prompt_cache.py`**

```python
"""Anthropic prompt-cache helpers — pure functions, no I/O.

Exposes:
- :func:`supports_prompt_cache` — provider detection by model prefix
- :func:`apply_prompt_cache` — non-mutating transform that adds
  ``cache_control`` blocks to messages and tools when caching is appropriate

The helpers do not depend on DI, observability, or LLM clients. They take
a list of messages and configuration scalars, and return a (possibly
modified) list ready to pass to ``litellm.acompletion``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_ANTHROPIC_PREFIXES = ("anthropic/", "bedrock/anthropic.", "claude-")


def supports_prompt_cache(model: str) -> bool:
    """Return True iff the model identifier targets Anthropic Claude.

    Recognises three common LiteLLM forms:
      - ``anthropic/claude-3-5-sonnet-20241022``
      - ``bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0``
      - ``claude-3-5-sonnet-20241022`` (bare Anthropic SDK style)

    Returns False for OpenAI, Azure, Vertex AI Anthropic (`vertex_ai/...`),
    and other providers — Phase 4 leaves Vertex Anthropic as a follow-up.
    """
    lowered = model.lower()
    return any(lowered.startswith(p) for p in _ANTHROPIC_PREFIXES)


def apply_prompt_cache(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None,
    model: str,
    enabled: bool,
    min_messages: int,
    min_estimated_tokens: int,
    estimated_tokens: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]] | None]:
    """Return (messages, tools) with cache_control breakpoints applied where appropriate.

    Returns the originals (as fresh lists) unchanged when:
      - ``enabled`` is False, OR
      - the model doesn't support caching (non-Anthropic provider), OR
      - ``len(messages) < min_messages``, OR
      - ``estimated_tokens < min_estimated_tokens``.

    Otherwise returns lists with cache_control content blocks inserted at:
      1. End of the system prompt (if present).
      2. End of the last assistant message before the trailing user turn
         (if the conversation has a stable history boundary).
      3. End of the tool list (if tools are present).

    Anthropic supports up to 4 cache breakpoints; this helper uses at most 3.
    """
    if not enabled or not supports_prompt_cache(model):
        return list(messages), list(tools) if tools is not None else None
    if len(messages) < min_messages or estimated_tokens < min_estimated_tokens:
        return list(messages), list(tools) if tools is not None else None

    # Convert all messages to a fresh list (cache-control blocks are added in place).
    cached_messages: list[Mapping[str, Any]] = [_with_cache_control(m, breakpoint=False) for m in messages]

    # Breakpoint 1: end of the first system message.
    for i, m in enumerate(cached_messages):
        if m.get("role") == "system":
            cached_messages[i] = _with_cache_control(m, breakpoint=True)
            break

    # Breakpoint 2: last assistant message before the trailing user turn.
    last_assistant_idx = _find_last_stable_assistant(cached_messages)
    if last_assistant_idx is not None:
        cached_messages[last_assistant_idx] = _with_cache_control(
            cached_messages[last_assistant_idx], breakpoint=True
        )

    # Breakpoint 3: last tool schema (if present).
    cached_tools: list[Mapping[str, Any]] | None = list(tools) if tools is not None else None
    if cached_tools:
        cached_tools[-1] = _with_tool_cache_control(cached_tools[-1])

    return cached_messages, cached_tools


def _with_cache_control(
    message: Mapping[str, Any], *, breakpoint: bool
) -> dict[str, Any]:
    """Convert message.content to structured form; tag last block as breakpoint if asked."""
    content = message.get("content")
    if isinstance(content, str):
        block: dict[str, Any] = {"type": "text", "text": content}
        if breakpoint:
            block["cache_control"] = {"type": "ephemeral"}
        return {**message, "content": [block]}
    if breakpoint and isinstance(content, list) and content:
        new_content = list(content)
        last_block = dict(new_content[-1])
        last_block["cache_control"] = {"type": "ephemeral"}
        new_content[-1] = last_block
        return {**message, "content": new_content}
    return dict(message)


def _with_tool_cache_control(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Tag the tool with cache_control to cache the tool list up to and including this entry."""
    return {**tool, "cache_control": {"type": "ephemeral"}}


def _find_last_stable_assistant(messages: Sequence[Mapping[str, Any]]) -> int | None:
    """Return index of the last assistant message before the trailing user turn.

    Returns None if the conversation doesn't have a stable history boundary
    (no assistant turns yet, or the last message isn't a user turn).
    """
    if not messages or messages[-1].get("role") != "user":
        return None
    for i in range(len(messages) - 2, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return None


__all__ = ["apply_prompt_cache", "supports_prompt_cache"]
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_prompt_cache.py -v 2>&1 | tail -20
```
Expected: 10+ tests passing.

### 3b — Settings additions

- [ ] **Step 3.5: Add the settings**

In `src/ai_core/config/settings.py`, find `class LLMSettings(BaseSettings):` and add the fields after the existing fields:

```python
    prompt_cache_enabled: bool = Field(
        default=True,
        description=(
            "When True (default), automatically apply Anthropic cache_control "
            "headers to system prompts and stable conversation history. Skipped "
            "for non-Anthropic providers and for prompts below the configured "
            "thresholds. Set False to disable for tests that require deterministic "
            "cache-miss responses."
        ),
    )
    prompt_cache_min_messages: int = Field(default=6, ge=2)
    prompt_cache_min_tokens: int = Field(default=1024, ge=512)
```

### 3c — Wire into `LiteLLMClient.complete`

- [ ] **Step 3.6: Write failing integration tests**

Append to `tests/unit/llm/test_litellm_client.py`:

```python
@pytest.mark.asyncio
async def test_complete_applies_cache_for_anthropic_above_threshold(
    monkeypatch, fake_observability, fake_budget,
) -> None:
    """For Anthropic models above the threshold, cache_control should be added."""
    settings = AppSettings()
    settings.llm.prompt_cache_enabled = True
    settings.llm.prompt_cache_min_messages = 2
    settings.llm.prompt_cache_min_tokens = 1  # force cache application

    captured: dict[str, Any] = {}

    async def _capture_call(**kwargs):
        captured.update(kwargs)
        return {
            "model": "claude-3-5-sonnet",
            "choices": [{"message": {"content": "ok", "tool_calls": []},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("litellm.acompletion", _capture_call)

    client = LiteLLMClient(
        settings=settings, budget=fake_budget, observability=fake_observability,
    )
    await client.complete(
        model="anthropic/claude-3-5-sonnet",
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "follow-up"},
        ],
    )

    sent_messages = captured["messages"]
    # System message should have cache_control.
    sys_content = sent_messages[0]["content"]
    assert isinstance(sys_content, list)
    assert sys_content[-1].get("cache_control") == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_complete_skips_cache_for_openai(
    monkeypatch, fake_observability, fake_budget,
) -> None:
    """OpenAI models should not get cache_control added."""
    settings = AppSettings()
    settings.llm.prompt_cache_enabled = True
    settings.llm.prompt_cache_min_messages = 2
    settings.llm.prompt_cache_min_tokens = 1

    captured: dict[str, Any] = {}

    async def _capture_call(**kwargs):
        captured.update(kwargs)
        return {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "ok", "tool_calls": []},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("litellm.acompletion", _capture_call)

    client = LiteLLMClient(
        settings=settings, budget=fake_budget, observability=fake_observability,
    )
    await client.complete(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "follow-up"},
        ],
    )

    sent_messages = captured["messages"]
    # OpenAI: messages should still have str content (no cache_control).
    for m in sent_messages:
        assert isinstance(m["content"], str)


@pytest.mark.asyncio
async def test_complete_skips_cache_when_setting_disabled(
    monkeypatch, fake_observability, fake_budget,
) -> None:
    """prompt_cache_enabled=False — even Anthropic + above threshold should not get cache_control."""
    settings = AppSettings()
    settings.llm.prompt_cache_enabled = False  # disabled

    captured: dict[str, Any] = {}

    async def _capture_call(**kwargs):
        captured.update(kwargs)
        return {
            "model": "claude-3-5-sonnet",
            "choices": [{"message": {"content": "ok", "tool_calls": []},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("litellm.acompletion", _capture_call)

    client = LiteLLMClient(
        settings=settings, budget=fake_budget, observability=fake_observability,
    )
    await client.complete(
        model="anthropic/claude-3-5-sonnet",
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "follow-up"},
        ],
    )

    sent_messages = captured["messages"]
    for m in sent_messages:
        assert isinstance(m["content"], str)
```

- [ ] **Step 3.7: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_litellm_client.py -v -k "test_complete_applies_cache or test_complete_skips_cache"
```
Expected: failures because `LiteLLMClient.complete` doesn't yet invoke `apply_prompt_cache`.

- [ ] **Step 3.8: Wire `apply_prompt_cache` into `LiteLLMClient.complete`**

In `src/ai_core/llm/litellm_client.py`, find where `request_kwargs` is constructed (around line 132). Insert the call to `apply_prompt_cache` BEFORE `request_kwargs`:

Add the import at the top of the file:

```python
from ai_core.llm._prompt_cache import apply_prompt_cache
```

Then update the request construction:

```python
        # Phase 4: apply prompt cache control if model supports it
        cache_cfg = self._settings.llm
        cached_messages, cached_tools = apply_prompt_cache(
            messages,
            tools=tools,
            model=resolved_model,
            enabled=cache_cfg.prompt_cache_enabled,
            min_messages=cache_cfg.prompt_cache_min_messages,
            min_estimated_tokens=cache_cfg.prompt_cache_min_tokens,
            estimated_tokens=estimated,
        )

        request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": cached_messages,
            "timeout": cfg.request_timeout_seconds,
        }
        if cached_tools is not None:
            request_kwargs["tools"] = cached_tools
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        # ... rest unchanged
```

(Remove the original `"messages": list(messages)` and the `request_kwargs["tools"] = list(tools)` block — both are now driven by `cached_messages` / `cached_tools`.)

- [ ] **Step 3.9: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/ -v 2>&1 | tail -25
```
Expected: all passing.

### 3d — Lint, type-check, commit

- [ ] **Step 3.10: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/llm/_prompt_cache.py \
    src/ai_core/llm/litellm_client.py \
    src/ai_core/config/settings.py \
    tests/unit/llm/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/llm/_prompt_cache.py \
    src/ai_core/llm/litellm_client.py \
    tests/unit/llm/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: no new violations; mypy ≤ 21.

- [ ] **Step 3.11: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```
Expected: 320+ passing.

- [ ] **Step 3.12: Commit**

```bash
git add src/ai_core/llm/_prompt_cache.py \
        src/ai_core/llm/litellm_client.py \
        src/ai_core/config/settings.py \
        tests/unit/llm/
git commit -m "feat(llm): Anthropic prompt caching — auto-apply cache_control to system + history + tools"
```

---

## Task 4 — End-of-phase smoke gate

**Files:** none (verification only).

- [ ] **Step 4.1: Full test suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Report total passes / fails / errors. Identify any new failures (not pre-existing).

- [ ] **Step 4.2: Lint the entire tree**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src tests 2>&1 | tail -5
```

Expected: no NEW categories vs `7fe92da` (pre-Phase-4).

- [ ] **Step 4.3: Mypy strict**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```
Expected: `Found N errors in M files (checked X source files)` with `N <= 21`.

- [ ] **Step 4.4: Smoke against `my-eaap-app`**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import importlib; importlib.import_module('ai_core'); print('ai_core imported ok')"
```

Verify Phase 4 surface:

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
from ai_core.llm._prompt_cache import apply_prompt_cache, supports_prompt_cache
from ai_core.mcp.transports import PoolingMCPConnectionFactory, FastMCPConnectionFactory
from ai_core.config.settings import MCPSettings, LLMSettings, SecuritySettings

# Caching helpers exist.
assert supports_prompt_cache('anthropic/claude-3-5-sonnet') is True
assert supports_prompt_cache('openai/gpt-4o') is False

# Pool factory + alias both work.
assert PoolingMCPConnectionFactory is FastMCPConnectionFactory or issubclass(FastMCPConnectionFactory, PoolingMCPConnectionFactory) or PoolingMCPConnectionFactory.__name__ == FastMCPConnectionFactory.__name__

# New settings fields exist.
mcp = MCPSettings()
assert mcp.pool_enabled is True
assert mcp.pool_idle_seconds == 300.0

llm = LLMSettings()
assert llm.prompt_cache_enabled is True
assert llm.prompt_cache_min_messages == 6
assert llm.prompt_cache_min_tokens == 1024

sec = SecuritySettings()
assert sec.opa_health_path == '/health'

print('Phase 4 settings + symbols OK')
"
```

Verify the Phase 1+2+3 canonical surface (28 names) still imports cleanly.

- [ ] **Step 4.5: Capture phase summary**

```bash
git log --oneline 7fe92da..HEAD
```

Expected: ~3 conventional-commit subjects (one per implementation task).

- [ ] **Step 4.6: Do NOT push automatically**

```bash
git status
echo ""
echo "Suggested next step:"
echo "git push origin feat/phase-1-facade-tool-validation"
echo "gh pr create --title 'feat: Phase 4 — cost/latency hardening + Phase 3 polish'"
```

---

## Out-of-scope reminders

For traceability, deferred to Phase 5+:

- 1-hour cache TTL beta (`cache_control: {ttl: "1h"}`)
- Vertex AI Anthropic prefix recognition (`vertex_ai/claude-...`)
- Multi-connection MCP pool (`max_connections > 1`)
- Pool health-check probe (e.g., `client.list_tools()` on checkout)
- `error_code` lookup registry / constants module
- Concrete `PayloadRedactor` implementations (PII strippers)
- Sentry / Datadog audit-sink reference implementations
- `eaap init` scaffold updates / YAML config / contract tests / Testcontainers

If a step starts pulling work from this list, stop and confirm scope with the user.
