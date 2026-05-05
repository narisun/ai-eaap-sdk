# ai-core-sdk Phase 4 — Design

**Date:** 2026-05-05
**Branch:** continues `feat/phase-1-facade-tool-validation` (or new `feat/phase-4-cost-latency-polish`)
**Status:** Awaiting user review

## Goal

Cost/latency hardening + Phase 3 polish. After Phase 4 the SDK can ship to production deployments running multi-turn Anthropic agents with prompt-cache savings (50-90% cost reduction on stable system prompts) and pooled MCP connections (avoiding per-call subprocess spawn / TLS handshake), while every Phase 3 follow-up flagged as Important by the final review is closed.

## Scope (10 items)

### Cost/latency (2)

1. **Anthropic prompt caching.** `LiteLLMClient.complete` auto-applies `cache_control: {type: "ephemeral"}` to (a) the system prompt and (b) the last assistant message before the trailing user turn. Threshold-gated (skip if `len(messages) < 6` or `estimated_tokens < 1024`). Provider-detected via model prefix (`anthropic/`, `bedrock/anthropic.`, `claude-`). Skipped for non-Anthropic. Tools list (when present) gets a third breakpoint. Configurable via `LLMSettings.prompt_cache_enabled` (default True), `prompt_cache_min_messages` (default 6), `prompt_cache_min_tokens` (default 1024).

2. **MCP connection pooling.** New private `_MCPConnectionPool` (single connection per `component_id`, idle-TTL eviction). `FastMCPConnectionFactory` becomes `PoolingMCPConnectionFactory`; old name retained as alias. Lifecycle-managed via `Container._teardown_sdk_resources`. Configurable via `MCPSettings.pool_enabled` (default True), `pool_idle_seconds` (default 300.0). Concurrent calls to the same spec serialize on a per-spec `asyncio.Lock`. Connection eviction-on-error provides self-healing.

### Phase 3 polish (8)

3. **`SecuritySettings.opa_health_path: str = "/health"`.** `OPAReachabilityProbe` reads from setting instead of hardcoding. Allows non-standard mounts (e.g., `/opa/health`).

4. **Remove `SettingsProbe`.** Always returns `ok` and conveys no real signal — its presence in `health.components` is misleading filler. Default probe list shrinks to 3 (`OPAReachabilityProbe`, `DatabaseProbe`, `ModelLookupProbe`).

5. **`HealthSnapshot` immutable interior.** `components` and `component_details` wrapped in `MappingProxyType` at construction. Field annotations change from `dict` to `Mapping`. Mutation attempts raise `TypeError`.

6. **`AuditRecord.now()` skip for `NullAuditSink`.** `ToolInvoker.__init__` sets `_records_audit: bool = not isinstance(audit, NullAuditSink)`. Each audit call is gated, avoiding `datetime.now(UTC)` + dict allocations on the hot path when audit is disabled.

7. **Include `principal` in audit `payload`.** `POLICY_DECISION` audit record's `payload` extends from `{"input": ...}` to `{"input": ..., "user": dict(principal or {})}`. Same `PayloadRedactor` mechanism still applies.

8. **`_ContextVarMergingDict` precedence docstring.** `bind_context` docstring + module docstring document that explicit logger kwargs override ContextVar-bound values when keys collide.

9. **`OTelEventAuditSink` `decision_allowed=None` → `""`** instead of `False`. Avoids SIEM misread "denied tool that ran anyway."

10. **`tool.completed` post-span calls protected.** `record_event("tool.completed", ...)` and the post-span audit record wrapped in `try/except` (logs `tool.completed_event_failed` at WARNING, swallows). Backend failures don't flip a successful tool call to failure from the caller's perspective.

## Non-goals (deferred to Phase 5+)

- 1-hour cache TTL beta (`cache_control: {ttl: "1h"}`)
- Vertex AI Anthropic prefix recognition (`vertex_ai/claude-...`)
- Multi-connection MCP pool (`max_connections > 1`)
- Pool health-check probe (e.g., `client.list_tools()` on checkout)
- `error_code` lookup registry / constants module
- Concrete `PayloadRedactor` implementations (PII strippers)
- Sentry / Datadog audit-sink reference implementations
- `eaap init` scaffold updates / YAML config / contract tests / Testcontainers

## Constraints

- Pre-1.0; **no backwards-compatibility requirement.** `FastMCPConnectionFactory` keeps its name as an alias for `PoolingMCPConnectionFactory`. `HealthSnapshot.components` becomes read-only at runtime (mutation now raises `TypeError`). `SettingsProbe` deletion changes `health.components` keys.
- Per-step gate: `ruff check` no new violations + `mypy <touched files>` strict + `pytest tests/unit tests/component`.
- Project mypy total stays ≤ 21 (post-Phase-3 baseline).
- End-of-phase smoke against `my-eaap-app` and the canonical 28-name top-level surface must continue to work.

## Module layout

```
src/ai_core/
├── llm/litellm_client.py        # MODIFIED — call apply_prompt_cache before request_kwargs
├── llm/_prompt_cache.py         # NEW — pure-function helpers (provider detection, breakpoint placement)
│
├── mcp/transports.py            # MODIFIED — PoolingMCPConnectionFactory; FastMCPConnectionFactory alias
├── mcp/_pool.py                 # NEW — internal _MCPConnectionPool
│
├── config/settings.py           # MODIFIED — adds:
│                                #   • llm.prompt_cache_enabled / _min_tokens / _min_messages
│                                #   • mcp.pool_enabled / pool_idle_seconds
│                                #   • security.opa_health_path
│
├── di/container.py              # MODIFIED — _teardown_sdk_resources adds mcp_pool.aclose
├── di/module.py                 # MODIFIED — provide_mcp_connection_factory wires settings
│
├── audit/interface.py           # unchanged (docstring tweak only)
├── audit/otel_event.py          # MODIFIED — decision_allowed=None → ""
│
├── tools/invoker.py             # MODIFIED — items 6 + 7 + 10:
│                                #   • _records_audit gate flag in __init__
│                                #   • principal added to POLICY_DECISION payload
│                                #   • post-span record_event/audit calls wrapped in try/except
│
├── observability/logging.py     # MODIFIED — bind_context docstring update (item 8)
│
├── app/runtime.py               # MODIFIED — HealthSnapshot.__post_init__ wraps components
│
├── health/probes.py             # MODIFIED — SettingsProbe deleted; OPAReachabilityProbe reads opa_health_path
│
└── health/__init__.py           # MODIFIED — drop SettingsProbe export
```

### Files NOT touched

- `README.md`
- `src/ai_core/cli/main.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`

### Test additions

```
tests/unit/llm/test_prompt_cache.py        # NEW (~10 tests for the pure-function helpers)
tests/unit/llm/test_litellm_client.py      # extended with caching-integration tests (~3 tests)
tests/unit/mcp/test_pool.py                # NEW (~6 tests for _MCPConnectionPool)
tests/unit/mcp/test_transports.py          # extended with pooling-integration tests (~3 tests)
tests/unit/audit/test_otel_event_sink.py   # extended with decision_allowed=None test (1 test)
tests/unit/tools/test_invoker.py           # extended with NullAuditSink-skip + principal-in-payload tests (~3 tests)
tests/unit/health/test_probes.py           # SettingsProbe tests deleted (4 removed)
tests/unit/health/test_opa_health_path.py  # NEW (1 test for opa_health_path config)
tests/unit/app/test_runtime.py             # extended with MappingProxyType immutability test (1 test)
```

## Component 1 — Phase 3 polish batch (items 3-10)

Eight self-contained fixes that ride on existing Phase 1/2/3 abstractions. Done first to clear the deck. Detail captured in §1a–1h below.

### 1a. `SecuritySettings.opa_health_path` (item 3)

```python
class SecuritySettings(BaseSettings):
    ...
    opa_health_path: str = Field(
        default="/health",
        description=(
            "Path appended to opa_url for the reachability probe. Override for "
            "deployments where OPA is mounted at a non-standard prefix "
            "(e.g., '/opa/health' behind an API gateway)."
        ),
    )
```

`OPAReachabilityProbe.__init__`:

```python
def __init__(self, settings: AppSettings) -> None:
    base = str(settings.security.opa_url).rstrip("/")
    path = settings.security.opa_health_path
    if not path.startswith("/"):
        path = "/" + path
    self._url = base + path
    self._timeout = settings.security.opa_request_timeout_seconds
```

### 1b. Remove `SettingsProbe` (item 4)

- Delete the class from `health/probes.py`.
- Remove from `health/__init__.py` exports.
- Update `AgentModule.provide_health_probes` to return only 3 probes.
- Delete the SettingsProbe tests from `test_probes.py` and `test_never_raises.py`.

### 1c. `HealthSnapshot` immutable interior (item 5)

```python
from types import MappingProxyType

@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    status: HealthStatus
    components: Mapping[str, HealthStatus]
    component_details: Mapping[str, str | None]
    service_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "components",
                           MappingProxyType(dict(self.components)))
        object.__setattr__(self, "component_details",
                           MappingProxyType(dict(self.component_details)))
```

### 1d. `AuditRecord.now()` skip for `NullAuditSink` (item 6)

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
    self._records_audit: bool = not isinstance(self._audit, NullAuditSink)
```

Each audit call site wraps with `if self._records_audit:`.

### 1e. `principal` in audit payload (item 7)

```python
if self._records_audit:
    await self._audit.record(AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        tool_name=spec.name, tool_version=spec.version,
        agent_id=agent_id, tenant_id=tenant_id,
        decision_path=spec.opa_path,
        decision_allowed=decision.allowed,
        decision_reason=decision.reason,
        payload={
            "input": payload.model_dump(),
            "user": dict(principal or {}),
        },
    ))
```

### 1f. `_ContextVarMergingDict` precedence docstring (item 8)

`bind_context` docstring + module docstring extended with:

> **Precedence:** Explicit logger keyword arguments take precedence over ContextVar-bound values when keys collide. After `bind_context(agent_id="A")`, a call `logger.warning("event", agent_id="B")` emits `agent_id="B"`. This matches structlog's `_ContextVarMergingDict` semantics — the per-call kwargs are the most specific binding.

### 1g. `OTelEventAuditSink` `decision_allowed=None` → `""` (item 9)

```python
return {
    ...
    "audit.decision_allowed": (
        record.decision_allowed if record.decision_allowed is not None else ""
    ),
    ...
}
```

### 1h. `tool.completed` post-span calls protected (item 10)

```python
latency_ms = (time.monotonic() - started) * 1000.0
try:
    await self._observability.record_event(
        "tool.completed",
        attributes={**attrs, "latency_ms": latency_ms},
    )
except Exception as exc:  # noqa: BLE001
    _logger.warning(
        "tool.completed_event_failed",
        tool_name=spec.name, agent_id=agent_id, tenant_id=tenant_id,
        error=str(exc), error_type=type(exc).__name__,
    )

if self._records_audit:
    await self._audit.record(AuditRecord.now(
        AuditEvent.TOOL_INVOCATION_COMPLETED,
        tool_name=spec.name, tool_version=spec.version,
        agent_id=agent_id, tenant_id=tenant_id,
        latency_ms=latency_ms,
    ))
return validated.model_dump(mode="json")
```

## Component 2 — MCP connection pooling

### 2a. `_MCPConnectionPool` (private)

`src/ai_core/mcp/_pool.py` — single-connection-per-spec pool with idle-TTL eviction. Key behaviours:

- `acquire(spec)` async context manager yields a live FastMCP client. Concurrent calls to the same `component_id` serialize on a per-conn `asyncio.Lock`.
- Stale-connection detection: `time.monotonic() - last_used > idle_seconds` triggers reopen on next checkout.
- In-flight error inside `async with pool.acquire(spec):` evicts the connection (self-healing).
- `aclose()` closes every pooled connection. Idempotent.
- `acquire()` after `aclose()` raises `MCPTransportError`.

### 2b. `PoolingMCPConnectionFactory`

`src/ai_core/mcp/transports.py` modified. The old `FastMCPConnectionFactory` class is renamed; old name retained as alias.

```python
class PoolingMCPConnectionFactory(IMCPConnectionFactory):
    def __init__(self, *, pool_enabled: bool = True,
                 pool_idle_seconds: float = 300.0) -> None:
        self._pool_enabled = pool_enabled
        self._pool: _MCPConnectionPool | None = (
            _MCPConnectionPool(opener=self._open_unpooled, idle_seconds=pool_idle_seconds)
            if pool_enabled else None
        )

    def open(self, spec: MCPServerSpec) -> AbstractAsyncContextManager[Any]:
        if self._pool is not None:
            return self._pool.acquire(spec)
        return self._open_unpooled(spec)

    @asynccontextmanager
    def _open_unpooled(self, spec: MCPServerSpec) -> AsyncIterator[Any]:
        # ... existing per-call FastMCP transport-construction code ...

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()


FastMCPConnectionFactory = PoolingMCPConnectionFactory  # alias
```

### 2c. Settings

```python
class MCPSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    pool_enabled: bool = Field(default=True, description="...")
    pool_idle_seconds: float = Field(default=300.0, gt=0.0, description="...")


class AppSettings(BaseSettings):
    ...
    mcp: MCPSettings = Field(default_factory=MCPSettings)
```

### 2d. DI binding

```python
@singleton
@provider
def provide_mcp_connection_factory(self, settings: AppSettings) -> IMCPConnectionFactory:
    return PoolingMCPConnectionFactory(
        pool_enabled=settings.mcp.pool_enabled,
        pool_idle_seconds=settings.mcp.pool_idle_seconds,
    )
```

### 2e. Container teardown

```python
steps: list[tuple[str, type[Any], tuple[str, ...]]] = [
    ("observability.shutdown", IObservabilityProvider, ("shutdown",)),
    ("audit.flush", IAuditSink, ("flush",)),
    ("mcp_pool.aclose", IMCPConnectionFactory, ("aclose",)),  # NEW
    ("policy_evaluator.aclose", IPolicyEvaluator, ("aclose",)),
    ("engine.dispose", AsyncEngine, ("dispose",)),
]
```

`PoolingMCPConnectionFactory.aclose` is a no-op when `pool_enabled=False`.

### 2f. Stale-connection detection — time-based only

No `client.ping()` probe. Reasoning: FastMCP's ping may not exist on all transports, and idle-TTL is conservative enough. Self-healing via in-flight error eviction handles "connection looked alive but server timed it out."

## Component 3 — Anthropic prompt caching

### 3a. `_prompt_cache.py` (pure functions)

```python
_ANTHROPIC_PREFIXES = ("anthropic/", "bedrock/anthropic.", "claude-")


def supports_prompt_cache(model: str) -> bool:
    lowered = model.lower()
    return any(lowered.startswith(p) for p in _ANTHROPIC_PREFIXES)


def apply_prompt_cache(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    model: str,
    enabled: bool,
    min_messages: int,
    min_estimated_tokens: int,
    estimated_tokens: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]] | None]:
    """Return (messages, tools) with cache_control breakpoints applied where appropriate.

    Returns the originals unchanged when:
      - enabled is False, OR
      - the model doesn't support caching, OR
      - len(messages) < min_messages, OR
      - estimated_tokens < min_estimated_tokens.

    Otherwise returns deep-copied lists with cache_control content blocks inserted.
    """
    if not enabled or not supports_prompt_cache(model):
        return list(messages), list(tools) if tools else None
    if len(messages) < min_messages or estimated_tokens < min_estimated_tokens:
        return list(messages), list(tools) if tools else None

    cached_messages = [_with_cache_control(m, breakpoint=False) for m in messages]
    # Breakpoint 1: end of system prompt.
    for i, m in enumerate(cached_messages):
        if m.get("role") == "system":
            cached_messages[i] = _with_cache_control(m, breakpoint=True)
            break
    # Breakpoint 2: last assistant before trailing user.
    last_assistant_idx = _find_last_stable_assistant(cached_messages)
    if last_assistant_idx is not None:
        cached_messages[last_assistant_idx] = _with_cache_control(
            cached_messages[last_assistant_idx], breakpoint=True
        )
    # Breakpoint 3: last tool schema.
    cached_tools = list(tools) if tools else None
    if cached_tools:
        cached_tools[-1] = _with_tool_cache_control(cached_tools[-1])
    return cached_messages, cached_tools
```

(Helper functions `_with_cache_control`, `_with_tool_cache_control`, `_find_last_stable_assistant` defined inline in the same file — pure, straight dict transformations.)

### 3b. Integration in `LiteLLMClient.complete`

Before `request_kwargs` is built:

```python
from ai_core.llm._prompt_cache import apply_prompt_cache

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
# ... rest unchanged
```

### 3c. Settings

```python
class LLMSettings(BaseSettings):
    ...
    prompt_cache_enabled: bool = Field(default=True, description="...")
    prompt_cache_min_messages: int = Field(default=6, ge=2)
    prompt_cache_min_tokens: int = Field(default=1024, ge=512)
```

### 3d. Provider-coverage notes

- **1-hour TTL** beta — Phase 4 uses default 5-min. Future config flag.
- **Bedrock** prompt caching uses the same `cache_control` syntax via LiteLLM; provider detection covers `bedrock/anthropic.` prefix.
- **OpenAI prompt caching (gpt-4o)** — automatic on OpenAI's side; helper correctly skips and lets OpenAI handle it.
- **Vertex AI Anthropic** — `vertex_ai/claude-...` prefix not in Phase 4. One-line follow-up.

## Error handling — consolidated (Phase 4 deltas)

Phase 4 doesn't introduce new exception types. Key changes:

| Path | Behaviour |
|---|---|
| `_MCPConnectionPool.acquire` after `aclose()` | `MCPTransportError("MCP connection pool is closed")` |
| `_MCPConnectionPool._open` raises | Wrapped as `MCPTransportError` (existing semantic) |
| Pool eviction-on-error | In-flight exception inside `async with acquire(spec)` propagates verbatim; pool drops the connection |
| `_close_one` failure | Logged at WARNING (`mcp.pool.connection_close_failed`); never propagates |
| `tool.completed` post-span `record_event` failure | Caught + logged (`tool.completed_event_failed`); user sees successful tool result |
| `apply_prompt_cache` | Pure function — never raises, never catches. Malformed input passes through unchanged. |

Phase 1-3 invariants preserved:
- Exception hierarchy + `error_code` field — preserved.
- `eaap.error.code` span attribute auto-emission — preserved.
- Audit-sink / probe never-raises contracts — preserved.
- structlog `bind_context` / `unbind_context` semantics — preserved.

## Testing strategy

Per-step gate identical to Phases 1-3. Project mypy total stays ≤ 21.

### Per-step test additions

| Step | Tests |
|---|---|
| 1. Polish batch | ~8 test updates (no new test files; some deletions for SettingsProbe) |
| 2. MCP pooling | ~6 unit tests in `test_pool.py` + ~3 in `test_transports.py` |
| 3. Prompt caching | ~10 unit tests in `test_prompt_cache.py` + ~3 in `test_litellm_client.py` |
| 4. Smoke gate | full pytest + ruff + mypy + my-eaap-app |

### Reusable fakes

`_FakeFastMCPClient` is local to `test_pool.py` (not promoted to conftest — only the pool tests need it).

### Risk register

| Risk | Mitigation |
|---|---|
| Removing `SettingsProbe` breaks tests asserting on its key | Step 1 deliberately audits + updates before later steps start |
| `MappingProxyType` wrap changes runtime type; consumers using `dict.update()` break | Pre-1.0; documented as breaking change; tests use `dict(snap.components)` for fresh dicts |
| Pool's per-spec lock serializes concurrent calls — could hide concurrency bugs | Documented as known limitation; multi-connection upgrade is Phase 5+ |
| `apply_prompt_cache` deep-copies lists — extra allocations | Profiled negligible (~1µs/call vs network latency) |
| LiteLLM cache_control API drift across versions | `litellm>=1.35.0` pin; helper small enough to fix in follow-up |
| Pool eviction-on-error closes a still-good connection (transient error) | Acceptable: next call gets fresh connection, self-healing. |

### End-of-phase smoke gate

- Full `pytest tests/unit tests/component` green.
- `ruff check src tests` no new violations vs `10ee979`.
- `mypy src --strict` total ≤ 21.
- All 28 canonical names import.
- `await app.health()` returns 3 probes (no `settings` key).
- `import ai_core` from `my-eaap-app` works.

### Coverage target

≥85% on new code (`mcp/_pool.py`, `llm/_prompt_cache.py`). Existing coverage must not regress.

## Implementation order (Approach 1 — bottom-up)

| Step | Deliverable | Tests | Dependencies |
|---|---|---|---|
| 1 | Polish batch (items 3-10) | ~8 test updates | none |
| 2 | MCP connection pooling | ~6 + ~3 = ~9 tests | none (after Step 1's fixes) |
| 3 | Anthropic prompt caching | ~10 + ~3 = ~13 tests | none (independent of Step 2) |
| 4 | End-of-phase smoke gate | full pytest + ruff + mypy + my-eaap-app | all |

## Constraints — recap

- 4 implementation steps; per-step gate is ruff (no new) + mypy strict (touched files) + pytest unit/component.
- Project mypy total stays ≤ 21.
- End-of-phase smoke gate must pass before merge.
- No backwards-compatibility shims (pre-1.0). `FastMCPConnectionFactory` keeps its name as alias for `PoolingMCPConnectionFactory`. `HealthSnapshot.components` becomes runtime-immutable.
- One new private file per cost/latency item (`mcp/_pool.py`, `llm/_prompt_cache.py`); no new top-level subpackages.
