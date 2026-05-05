# ai-core-sdk Phase 3 — Design

**Date:** 2026-05-05
**Branch:** continues `feat/phase-1-facade-tool-validation` (or new `feat/phase-3-operability`)
**Status:** Awaiting user review

## Goal

Operability hardening + Phase 2 coherence fixes. After Phase 3, the SDK can be deployed into a production environment where `/healthz` returns real signal, logs are structured for queryable ingestion, every dashboard query against `eaap.error.code` returns the expected counts (no more parent-span-only tagging), and an audit trail of policy decisions and tool invocations is durably recorded.

## Scope (9 items)

### Operability (3)

1. **Real health probes.** `app.health` becomes async; runs a configurable list of `IHealthProbe` instances in parallel via `asyncio.gather` with per-probe timeout. Default probes: `SettingsProbe`, `OPAReachabilityProbe`, `DatabaseProbe`, `ModelLookupProbe`. `HealthSnapshot.components` populated with real `"ok"|"degraded"|"down"` per subsystem; new `component_details` field for human-readable detail.
2. **`structlog` adoption.** Add `structlog` as a dependency. Configure JSON renderer for prod, key-value for dev. Module-level `_logger = get_logger(__name__)` from a new `observability/logging.py` seam. Auto-bind `agent_id`/`tenant_id`/`thread_id` via `ContextVar`. Migrate ~8-10 high-value operational callsites to event-name + structured kwargs; pre-existing stdlib calls keep working via `LoggerFactory`.
3. **`IAuditSink` ABC + 3 sinks.** Durable record of every policy decision and tool invocation outcome. `NullAuditSink` (default), `OTelEventAuditSink` (records via `IObservabilityProvider.record_event`), `JsonlFileAuditSink` (line-delimited JSON to a file). DI-bound; switchable via `settings.audit.sink_type`. Hooks into the widened `tool.invoke` span at OPA decision and tool completion / failure points.

### Phase 2 coherence Important fixes (2)

4. **Widen `tool.invoke` span to wrap all 6 pipeline steps.** Currently steps 1 (input validation), 2 (OPA), and 5 (output validation) raise *outside* the span; only step 4 (handler) is inside. Phase 3 wraps all 6 so all four error categories trigger the `eaap.error.code` auto-emission from Phase 2 Task 4.
5. **Move `_normalise_response` inside `llm.complete` span.** Currently the call sits *after* the span CM closes, so `LLMInvocationError(error_code="llm.empty_response")` doesn't tag the LLM-call span. One-line fix.

### Phase 2 minor follow-ups (4)

6. **`MemoryManager.compact` also skips `LLMTimeoutError`.** Currently catches only `asyncio.TimeoutError`; add a parallel branch for the LLM-client-level timeout. Distinct event name so dashboards can split "we hit our budget" vs "the LLM client itself timed out".
7. **`LLMResponse.finish_reason` Literal typing.** Type as `Literal["stop","length","tool_calls","content_filter","function_call"] | str | None` for IDE help on the common cases plus `str` fallback for unknown providers.
8. **`HealthSnapshot.components` test loosened.** The Phase 2 test asserted exact dict equality; loosen to set-of-keys + per-key value assertion only on known-good values, so Phase 3's real probes don't break the test.
9. **Promote `IBudgetService` fake to `tests/conftest.py`.** Currently `FakeBudget` is duplicated in three test files. Single shared `FakeBudgetService` + `fake_budget` fixture.

## Non-goals (explicitly deferred)

- Anthropic prompt caching — Phase 4 or later.
- MCP connection pooling — Phase 4 or later.
- Sentry / Datadog integrations — users implement custom `IAuditSink` if needed.
- Per-tenant audit retention policies — backend-side concern (Postgres triggers, S3 lifecycle rules).
- Concrete PII redactor implementations — the `PayloadRedactor` hook is shipped; concrete redactors are post-Phase-3.
- `error_code` lookup registry / constants module — Phase 4 if requested.
- Sentry-style breadcrumb buffer — out of scope.
- Per-database-connection pool probe — `DatabaseProbe` only does `SELECT 1`.

## Constraints

- Pre-1.0; **no backwards-compatibility requirement.** `app.health` becomes `async` (was sync). The Phase 2 polish test that asserted on the sync property is updated.
- Per-step gate: `ruff check` no new violations + `mypy <touched files>` strict + `pytest tests/unit tests/component`.
- Project mypy strict total stays at-or-below the 21-error baseline (post-Phase-2).
- End-of-phase smoke against `my-eaap-app` and the canonical top-level imports must continue to work.

## Module layout

```
src/ai_core/
├── audit/                        # NEW — audit sink subsystem
│   ├── __init__.py               # exports IAuditSink, AuditRecord, NullAuditSink,
│   │                             #   OTelEventAuditSink, JsonlFileAuditSink, AuditEvent,
│   │                             #   PayloadRedactor
│   ├── interface.py              # IAuditSink ABC + AuditRecord dataclass + AuditEvent enum
│   │                             #   + PayloadRedactor type alias + identity redactor
│   ├── null.py                   # NullAuditSink (default)
│   ├── otel_event.py             # OTelEventAuditSink (records via IObservabilityProvider)
│   └── jsonl.py                  # JsonlFileAuditSink (line-delimited JSON to file)
│
├── health/                       # NEW — health-probe subsystem
│   ├── __init__.py               # exports IHealthProbe, ProbeResult, HealthStatus
│   ├── interface.py              # IHealthProbe ABC + ProbeResult dataclass + HealthStatus type
│   └── probes.py                 # SettingsProbe, OPAReachabilityProbe, DatabaseProbe,
│                                 #   ModelLookupProbe
│
├── observability/
│   ├── real.py                   # MODIFIED — uses get_logger from logging.py for the ~5 migrated callsites
│   └── logging.py                # NEW — get_logger() + bind_context + unbind_context + configure
│
├── tools/invoker.py              # MODIFIED — widen `tool.invoke` span to wrap all 6 steps;
│                                 #   call audit sink at policy-decision + completion + failure points
│
├── llm/litellm_client.py         # MODIFIED — move _normalise_response inside `llm.complete` span
│
├── di/interfaces.py              # MODIFIED — LLMResponse.finish_reason becomes
│                                 #   Literal[...] | str | None (item 7)
│
├── di/module.py                  # MODIFIED — bind IAuditSink (default NullAuditSink, switchable
│                                 #   via settings.audit.sink_type); bind list[IHealthProbe]
│                                 #   (default 4 probes)
│
├── agents/memory.py              # MODIFIED — also catch LLMTimeoutError in skip-and-WARN (item 6);
│                                 #   migrate compaction logs to structlog event names
│
├── agents/base.py                # MODIFIED — bind_context for agent_id/tenant_id/thread_id
│                                 #   in ainvoke; migrate tool execution log to structlog
│
├── app/runtime.py                # MODIFIED — async `app.health()`; new _HealthCheckRunner;
│                                 #   HealthSnapshot.component_details added; configure structlog
│                                 #   in __aenter__
│
├── config/settings.py            # MODIFIED — adds:
│                                 #   • observability.log_format: Literal["text", "structured"] = "text"
│                                 #   • audit: AuditSettings (sink_type, jsonl_path)
│                                 #   • health: HealthSettings (probe_timeout_seconds)
│
├── di/container.py               # MODIFIED — _teardown_sdk_resources adds audit.flush step
│
└── __init__.py                   # MODIFIED — top-level exports for IAuditSink, AuditRecord,
                                  #   AuditEvent, IHealthProbe, ProbeResult
```

## Files NOT touched (carrying WIP from before Phase 1)

- `README.md`
- `src/ai_core/cli/main.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`

## Component 1 — Phase 2 coherence batch (items 4-9)

These ride on Phase 2 abstractions and ship first as small, self-contained fixes.

### 1a. Widen `tool.invoke` span (item 4)

`src/ai_core/tools/invoker.py` — wrap all 6 pipeline steps inside the `start_span("tool.invoke")` async context manager:

```python
async def invoke(
    self,
    spec: ToolSpec,
    raw_args: Mapping[str, Any],
    *,
    principal: Mapping[str, Any] | None = None,
    agent_id: str | None = None,
    tenant_id: str | None = None,
) -> Mapping[str, Any]:
    attrs = {
        "tool.name": spec.name, "tool.version": spec.version,
        "agent_id": agent_id or "", "tenant_id": tenant_id or "",
    }
    started = time.monotonic()
    async with self._observability.start_span("tool.invoke", attributes=attrs):
        try:
            payload = spec.input_model.model_validate(dict(raw_args))   # ToolValidationError(input)
        except ValidationError as exc:
            raise ToolValidationError(...) from exc

        if spec.opa_path is not None and self._policy is not None:
            decision = await self._policy.evaluate(decision_path=spec.opa_path, input={...})
            await self._audit.record(AuditRecord.now(  # see component 3
                event=AuditEvent.POLICY_DECISION,
                tool_name=spec.name, tool_version=spec.version,
                agent_id=agent_id, tenant_id=tenant_id,
                decision_path=spec.opa_path,
                decision_allowed=decision.allowed,
                decision_reason=decision.reason,
                payload={"input": payload.model_dump()},
            ))
            if not decision.allowed:
                raise PolicyDenialError(...)

        try:
            result = await spec.handler(payload)
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(..., cause=exc) from exc

        try:
            validated = spec.output_model.model_validate(result)         # ToolValidationError(output)
        except ValidationError as exc:
            raise ToolValidationError(..., side="output") from exc

    latency_ms = (time.monotonic() - started) * 1000.0
    await self._observability.record_event(
        "tool.completed",
        attributes={**attrs, "latency_ms": latency_ms},
    )
    await self._audit.record(AuditRecord.now(
        event=AuditEvent.TOOL_INVOCATION_COMPLETED,
        tool_name=spec.name, tool_version=spec.version,
        agent_id=agent_id, tenant_id=tenant_id,
        latency_ms=latency_ms,
    ))
    return validated.model_dump()
```

For failures, audit sink records `TOOL_INVOCATION_FAILED` in a unified handler placed *after* the span CM closes (so the span has already been tagged with `eaap.error.code`):

```python
try:
    async with self._observability.start_span(...):
        ...
        return validated.model_dump()  # success path returns inside try
except (ToolValidationError, PolicyDenialError, ToolExecutionError) as exc:
    latency_ms = (time.monotonic() - started) * 1000.0
    await self._audit.record(AuditRecord.now(
        event=AuditEvent.TOOL_INVOCATION_FAILED,
        tool_name=spec.name, tool_version=spec.version,
        agent_id=agent_id, tenant_id=tenant_id,
        error_code=exc.error_code,
        latency_ms=latency_ms,
    ))
    raise
```

### 1b. Move `_normalise_response` inside `llm.complete` span (item 5)

`src/ai_core/llm/litellm_client.py`:

```python
async with self._observability.start_span("llm.complete", attributes=attributes):
    raw = await self._call_with_retry(request_kwargs)
    response = _normalise_response(resolved_model, raw)  # was OUTSIDE the span; now INSIDE
# usage recording stays outside (pure metric emit, not measured)
await self._observability.record_llm_usage(...)
```

### 1c. `MemoryManager.compact` also skips `LLMTimeoutError` (item 6)

`src/ai_core/agents/memory.py`:

```python
async def compact(self, state, *, model=None, tenant_id=None, agent_id=None) -> AgentState:
    timeout = self._settings.agent.compaction_timeout_seconds
    try:
        return await asyncio.wait_for(
            self._do_compact(state, model=model, tenant_id=tenant_id, agent_id=agent_id),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        _logger.warning(
            "compaction.skipped.budget_exceeded",
            timeout_seconds=timeout, agent_id=agent_id, tenant_id=tenant_id,
        )
        return state
    except LLMTimeoutError as exc:
        _logger.warning(
            "compaction.skipped.llm_timeout",
            agent_id=agent_id, tenant_id=tenant_id, error_code=exc.error_code,
        )
        return state
```

### 1d. `LLMResponse.finish_reason` Literal typing (item 7)

`src/ai_core/di/interfaces.py`:

```python
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | str

@dataclass(frozen=True, slots=True)
class LLMResponse:
    ...
    finish_reason: FinishReason | None = None
```

### 1e. `HealthSnapshot.components` test loosened (item 8)

`tests/unit/app/test_runtime.py`. Replace exact-dict equality:

```python
assert set(snap.components.keys()) == {
    "settings", "container", "tool_invoker", "policy_evaluator", "observability",
}
assert snap.components["settings"] == "ok"
assert snap.components["container"] == "ok"
# Other component values may be "ok" or "unknown" depending on Phase 3 probes.
```

### 1f. `FakeBudgetService` promotion (item 9)

Move from per-test-file `FakeBudget` definitions to a single `tests/conftest.py` class + `fake_budget` fixture (always-allow, records calls).

## Component 2 — `structlog` adoption

### 2a. `src/ai_core/observability/logging.py` (NEW)

Single seam. All other modules import `get_logger` from here, never from `structlog` directly.

```python
from contextvars import ContextVar
from typing import Any, Literal

import structlog
from structlog.stdlib import LoggerFactory
from structlog.typing import FilteringBoundLogger


_request_context: ContextVar[Mapping[str, Any]] = ContextVar(
    "_eaap_request_context", default={}
)


def configure(*, log_format: Literal["text", "structured"] = "text",
              log_level: str = "INFO") -> None:
    """Configure structlog. Idempotent — safe to call from AICoreApp.__aenter__."""
    structlog.reset_defaults()
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _bind_request_context,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
    ]
    if log_format == "structured":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO),
        ),
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _bind_request_context(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for k, v in _request_context.get().items():
        event_dict.setdefault(k, v)
    return event_dict


def bind_context(**kwargs: Any) -> Any:
    """Push kwargs into the request-scoped ContextVar; return reset token."""
    current = dict(_request_context.get())
    current.update(kwargs)
    return _request_context.set(current)


def unbind_context(token: Any) -> None:
    """Reset the ContextVar to the prior token's snapshot."""
    _request_context.reset(token)


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    return structlog.get_logger(name)
```

### 2b. `AICoreApp` configures structlog at boot

```python
async def __aenter__(self) -> AICoreApp:
    self._settings = self._user_settings or get_settings()
    self._secret_manager = self._user_secret_manager or EnvSecretManager()
    self._settings.validate_for_runtime(secret_manager=self._secret_manager)
    # NEW: configure structlog before anything else logs.
    from ai_core.observability.logging import configure as _configure_logging
    _configure_logging(
        log_format=self._settings.observability.log_format,
        log_level=self._settings.observability.log_level.value,
    )
    self._container = Container.build([...])
    await self._container.start()
    self._entered = True
    return self
```

### 2c. `BaseAgent.ainvoke` binds request context

```python
async def ainvoke(self, *, messages, essential=None, tenant_id=None, thread_id=None) -> AgentState:
    from ai_core.observability.logging import bind_context, unbind_context
    log_token = bind_context(
        agent_id=self.agent_id, tenant_id=tenant_id, thread_id=thread_id,
    )
    try:
        # ... existing baggage attach + span code ...
    finally:
        unbind_context(log_token)
```

### 2d. Migration targets (~8-10 callsites)

| File | Old | New event name |
|---|---|---|
| `agents/memory.py` | "Compaction skipped: ..." | `compaction.skipped.budget_exceeded`, `compaction.skipped.llm_timeout` |
| `tools/invoker.py` | "Tool 'X' v1 returned non-conforming object" | `tool.output_validation_failed` |
| `agents/base.py` | "Tool 'X' execution error" | `tool.execution_error` |
| `observability/real.py` | "Observability backend error in X" (×2) | `observability.backend_error`, `langfuse.helper_failed` |

All other `_logger.*` calls keep their stdlib format. structlog wraps stdlib via `LoggerFactory`, so they all flow through the same renderer. Pre-existing message-shape stays valid. Phase 3 migrates *existing* operational callsites; it does **not** add new instrumentation (e.g., per-retry telemetry in `litellm_client.py`) — that's a future phase if needed.

### 2e. Setting

```python
class ObservabilitySettings(BaseSettings):
    ...
    log_format: Literal["text", "structured"] = Field(
        default="text",
        description=(
            "When 'text' (default), logs render as colorized key=value for local dev. "
            "When 'structured', logs render as JSON for production ingestion."
        ),
    )
```

## Component 3 — `IAuditSink` ABC + 3 sinks

### 3a. `src/ai_core/audit/interface.py`

```python
class AuditEvent(str, enum.Enum):
    POLICY_DECISION = "policy.decision"
    TOOL_INVOCATION_STARTED = "tool.invocation.started"
    TOOL_INVOCATION_COMPLETED = "tool.invocation.completed"
    TOOL_INVOCATION_FAILED = "tool.invocation.failed"


PayloadRedactor = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _identity_redactor(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(payload)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    event: AuditEvent
    timestamp: datetime
    tool_name: str | None
    tool_version: int | None
    agent_id: str | None
    tenant_id: str | None
    decision_path: str | None
    decision_allowed: bool | None
    decision_reason: str | None
    error_code: str | None
    payload: Mapping[str, Any]
    latency_ms: float | None

    @classmethod
    def now(cls, event: AuditEvent, *, ..., redactor: PayloadRedactor = _identity_redactor) -> AuditRecord: ...


class IAuditSink(ABC):
    @abstractmethod
    async def record(self, record: AuditRecord) -> None: ...
    @abstractmethod
    async def flush(self) -> None: ...
```

### 3b. Three concrete sinks

- **`NullAuditSink`** — no-op; default DI binding for development.
- **`OTelEventAuditSink`** — records via `IObservabilityProvider.record_event(f"eaap.audit.{event}", attrs)`. Trace-shaped retention; fine for non-compliance use cases.
- **`JsonlFileAuditSink`** — line-delimited JSON to a configurable file path. Buffered (default 64 records); `asyncio.to_thread` for the file write. `flush()` is idempotent and called by `Container.stop`.

All three swallow internal exceptions — sinks NEVER raise from `record` or `flush`.

### 3c. DI binding

```python
@singleton
@provider
def provide_audit_sink(self, settings: AppSettings,
                      observability: IObservabilityProvider) -> IAuditSink:
    sink_type = settings.audit.sink_type
    if sink_type == "null":
        return NullAuditSink()
    if sink_type == "otel_event":
        return OTelEventAuditSink(observability)
    if sink_type == "jsonl":
        if settings.audit.jsonl_path is None:
            raise ConfigurationError(
                "audit.sink_type='jsonl' requires audit.jsonl_path to be set"
            )
        return JsonlFileAuditSink(settings.audit.jsonl_path)
    raise ConfigurationError(f"Unknown audit.sink_type: {sink_type!r}")
```

### 3d. `ToolInvoker` integration

Constructor gains optional `audit: IAuditSink | None = None` (defaults to `NullAuditSink()` if not provided). DI wires the real sink in production. Records on:

- Step 2 (after OPA) — `POLICY_DECISION` with `decision_allowed`/`reason`/`payload={"input": ...}`.
- After step 5 (success) — `TOOL_INVOCATION_COMPLETED` with `latency_ms`.
- Catch-all `except` after the span — `TOOL_INVOCATION_FAILED` with `error_code` and `latency_ms`.

### 3e. Settings

```python
class AuditSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    sink_type: Literal["null", "otel_event", "jsonl"] = "null"
    jsonl_path: Path | None = None


class AppSettings(BaseSettings):
    ...
    audit: AuditSettings = Field(default_factory=AuditSettings)
```

### 3f. `Container.stop` calls `audit.flush()`

```python
steps = [
    ("observability.shutdown", IObservabilityProvider, ("shutdown",)),
    ("audit.flush", IAuditSink, ("flush",)),  # NEW
    ...
]
```

### 3g. Top-level exports

```python
from ai_core.audit import IAuditSink, AuditRecord, AuditEvent
__all__ = [..., "IAuditSink", "AuditRecord", "AuditEvent"]
```

## Component 4 — Real health probes

### 4a. `src/ai_core/health/interface.py`

```python
HealthStatus = Literal["ok", "degraded", "down"]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    component: str
    status: HealthStatus
    detail: str | None = None


class IHealthProbe(ABC):
    component: str

    @abstractmethod
    async def probe(self) -> ProbeResult:
        """Run the probe. Implementations MUST NOT raise — return
        ProbeResult(status='down') with detail explaining the failure."""
```

### 4b. Four concrete probes

- **`SettingsProbe`** — always returns `ok` (settings validated at app entry).
- **`OPAReachabilityProbe`** — `GET <opa_url>/health` via `httpx`. `ok` on 2xx/4xx, `degraded` on 5xx, `down` on connect error / timeout.
- **`DatabaseProbe`** — `SELECT 1` via the existing `AsyncEngine`. `down` on any failure.
- **`ModelLookupProbe`** — `litellm.utils.get_supported_openai_params(model)` via `asyncio.to_thread`. `degraded` if model not recognized; `down` on lookup error.

All four catch `Exception` internally and return `ProbeResult(status="down", detail="<short>")` on failure.

### 4c. `_HealthCheckRunner` (in `app/runtime.py`)

```python
class _HealthCheckRunner:
    def __init__(self, probes: Sequence[IHealthProbe], *, timeout_seconds: float) -> None: ...

    async def run(self) -> list[ProbeResult]:
        async def _run_one(probe: IHealthProbe) -> ProbeResult:
            try:
                return await asyncio.wait_for(probe.probe(), timeout=self._timeout)
            except asyncio.TimeoutError:
                return ProbeResult(component=probe.component, status="down",
                                   detail=f"probe_timeout_{self._timeout}s")
            except Exception as exc:  # noqa: BLE001
                return ProbeResult(component=probe.component, status="down",
                                   detail=f"probe_error: {type(exc).__name__}")
        return await asyncio.gather(*(_run_one(p) for p in self._probes))
```

Defense in depth — even a probe that misbehaves (raises despite the contract) is caught by the runner.

### 4d. `AICoreApp.health` becomes async

```python
@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    status: HealthStatus
    components: dict[str, HealthStatus]
    component_details: dict[str, str | None]   # NEW
    service_name: str


class AICoreApp:
    async def health(self) -> HealthSnapshot:
        if not self._entered or self._settings is None:
            return HealthSnapshot(status="down", components={},
                                  component_details={}, service_name="")
        runner = _HealthCheckRunner(
            self._container.get(list[IHealthProbe]),
            timeout_seconds=self._settings.health.probe_timeout_seconds,
        )
        results = await runner.run()
        components = {r.component: r.status for r in results}
        details = {r.component: r.detail for r in results}
        if any(s == "down" for s in components.values()):
            roll_up: HealthStatus = "down"
        elif any(s == "degraded" for s in components.values()):
            roll_up = "degraded"
        else:
            roll_up = "ok"
        if self._closed:
            roll_up = "down"
        return HealthSnapshot(status=roll_up, components=components,
                              component_details=details,
                              service_name=self._settings.service_name)
```

### 4e. DI binding

```python
@singleton
@provider
def provide_health_probes(self, settings: AppSettings,
                          engine: AsyncEngine) -> list[IHealthProbe]:
    return [
        SettingsProbe(settings),
        OPAReachabilityProbe(settings),
        DatabaseProbe(engine),
        ModelLookupProbe(settings),
    ]
```

### 4f. Settings

```python
class HealthSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    probe_timeout_seconds: float = Field(default=2.0, gt=0)


class AppSettings(BaseSettings):
    ...
    health: HealthSettings = Field(default_factory=HealthSettings)
```

### 4g. Top-level exports

```python
from ai_core.health import IHealthProbe, ProbeResult
__all__ = [..., "IHealthProbe", "ProbeResult"]
```

## Error handling — consolidated (Phase 3 deltas)

Phase 3 doesn't introduce new exception types. The novel things are:

### Updated propagation

| Error | Phase 2 | Phase 3 |
|---|---|---|
| `ToolValidationError` (input) | parent-span tag only | tagged on `tool.invoke` span (widened) |
| `PolicyDenialError` | parent-span tag only | tagged on `tool.invoke` span |
| `ToolValidationError` (output) | parent-span tag only | tagged on `tool.invoke` span |
| `LLMInvocationError(error_code="llm.empty_response")` | parent-span tag only | tagged on `llm.complete` span |
| `LLMTimeoutError` from compaction | crashed agent | skip-and-WARN with distinct event name |

### "Never raises" contracts (new)

- **`IAuditSink.record` / `flush`** — implementations MUST swallow internally and log via structlog. Audit failures must never block tool calls.
- **`IHealthProbe.probe`** — implementations MUST return `ProbeResult(status="down", detail="...")` on failure rather than raise. Defense in depth: `_HealthCheckRunner` catches anything that escapes.

### Logging policy

- Phase 1 + 2 used stdlib `logging` (preserved for ~30 callsites).
- Phase 3 migrates ~8-10 high-value operational callsites to structlog event-name + structured kwargs. Convention: dotted, lowercase, snake_case after the dot, parallel to `error_code` codes from Phase 2.

### What's not changing

- Phase 2 exception table (16 classes + DEFAULT_CODE) — preserved.
- `EAAPBaseException.error_code` semantics — preserved.
- OTel span attribute namespace (`eaap.error.code`, `eaap.error.details.{k}`) — preserved.
- `RealObservabilityProvider._should_swallow` semantics — preserved.

## Testing strategy

Per-step gate: `ruff check` no new violations + `mypy <touched files>` strict + `pytest tests/unit tests/component`. Project mypy stays ≤ 21.

### Reusable fakes — Phase 3 additions

`tests/conftest.py`:

- **`FakeBudgetService`** + `fake_budget` fixture (item 9 — promoted from per-file fakes).
- **`FakeAuditSink`** + `fake_audit_sink` fixture — records `AuditRecord` instances for assertion; `flush()` no-op.
- **`FakeHealthProbe`** + factory — configurable probe that returns a pre-baked `ProbeResult` (or simulates timeout/raise for runner-level tests).

### Per-step test additions

| Step | Test additions |
|---|---|
| 1. Phase 2 coherence | ~6 test updates: widened-span assertions, `_normalise_response`-inside-span, `LLMTimeoutError` in compaction, Literal typing smoke, components-test loosening, `FakeBudget` migration. |
| 2. structlog | ~6 unit tests in `tests/unit/observability/test_logging.py`: configure idempotency, get_logger return type, ContextVar binding/unbinding, JSON renderer output, console renderer output, `capture_logs` integration. |
| 3. Audit sink | ~10 unit tests in `tests/unit/audit/`: AuditRecord construction + redaction; NullSink no-op; OTelEventSink calls record_event; OTelEventSink swallows backend errors; JsonlSink writes line-delimited; JsonlSink buffers + flushes; JsonlSink flush idempotent; concurrency-safe flush; ToolInvoker integration; DI binding swap. |
| 4. Health probes | ~8 unit tests in `tests/unit/health/test_probes.py` (per probe + per failure mode); ~4 component tests in `test_runtime.py` (async health, roll-up, timeout, before-entry). |
| 5. Smoke gate | full pytest + ruff + mypy + my-eaap-app. |

### Additional meta-test (one-time enforcement)

`tests/unit/audit/test_never_raises.py` and `tests/unit/health/test_never_raises.py` — parametrized tests that monkeypatch each sink's / probe's internal call to raise `RuntimeError`, then assert `await sink.record(...)` / `await probe.probe()` does NOT raise. Locks the contract by structure.

### Existing tests that need updates (Step 1 audit)

- Find every `LLMResponse(...)` constructor call in tests; loosen any test asserting `finish_reason` is exactly a specific str (item 7).
- Find every assertion on `tool.invoke` span exception position; widen to "span has `eaap.error.code`" rather than "exception happened outside span" (item 4).
- Find every `_normalise_response` test asserting "raises before span"; flip to "raises inside span" (item 5).
- Find duplicated `FakeBudget` definitions; delete after promoting to conftest (item 9).
- Update `test_health_components_populated_after_entry` and friends to assert set-of-keys + known-good values (item 8).
- Update any test expecting sync `app.health` to use `await app.health()` (component 4).

### Risk register

| Risk | Mitigation |
|---|---|
| structlog migration breaks existing `caplog`-based tests | structlog wraps stdlib via `LoggerFactory`; existing `caplog` calls keep working. New tests use `structlog.testing.capture_logs`. |
| Audit sink integration in `ToolInvoker` adds I/O on every tool call | `NullAuditSink` default = no-op; `OTelEventAuditSink` async via observability; `JsonlFileAuditSink` buffers + thread-pool writes. None block. |
| Async `app.health` breaks consumers | Pre-1.0 — acceptable. Step 1 test audit updates the Phase 2 polish tests. |
| Real probes hit external services in CI | Probes use `respx`/in-memory SQLite/no-op model lookup in tests. Production wiring uses real probes; unit tests inject `FakeHealthProbe`. |
| `litellm.utils.get_supported_openai_params` slow on first call | Probe uses `asyncio.to_thread`; per-probe timeout caps it. |
| `JsonlFileAuditSink` blocks on disk I/O | `asyncio.to_thread` for file write. |
| Audit + observability infinite loop (sink uses observability that uses sink) | `OTelEventAuditSink` only depends on `IObservabilityProvider`; observability does not depend on audit. No cycle. |
| Phase 2 `FakeObservabilityProvider` doesn't record `tool.completed` event with `latency_ms` | Phase 3 adds `latency_ms` to the event attrs; existing `test_invoker.py` assertions should still match because they use `assert_any_call` semantics. Verify in Step 1. |

### Coverage target

≥85% on new code (`audit/`, `health/`, `observability/logging.py`). Existing coverage must not regress.

## Implementation order (Approach 1 — bottom-up)

| Step | Deliverable | Tests | Dependencies |
|---|---|---|---|
| 1 | Phase 2 coherence batch (items 4-9) | ~6 test updates + conftest additions | none |
| 2 | structlog adoption: `observability/logging.py`, configure in `__aenter__`, bind_context in `BaseAgent.ainvoke`, migrate ~8-10 callsites | ~6 unit tests in `tests/unit/observability/test_logging.py` | Step 1 |
| 3 | `IAuditSink` + 3 sinks + DI binding + `ToolInvoker` integration + meta-test | ~10 unit tests in `tests/unit/audit/` + 1 meta-test | Step 1 |
| 4 | `IHealthProbe` + 4 probes + `_HealthCheckRunner` + async `app.health` | ~8 unit + 4 component tests + 1 meta-test | Step 1 |
| 5 | End-of-phase smoke gate | full pytest + ruff + mypy + my-eaap-app smoke | all |

## Constraints — recap

- 4 implementation steps + 1 smoke gate = ~5 plan tasks (smaller cadence than Phase 1 + 2 because more is bundled per step; the per-step gate keeps each step independently shippable).
- Per-step gate: ruff + mypy strict + pytest. Project mypy total ≤ 21.
- End-of-phase smoke must pass before merge.
- No backwards-compatibility shims (pre-1.0).
- Two new top-level subpackages (`audit/`, `health/`); no existing subpackage deletions.
