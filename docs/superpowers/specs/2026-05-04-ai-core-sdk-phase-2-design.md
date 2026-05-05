# ai-core-sdk Phase 2 — Design

**Date:** 2026-05-04
**Branch:** continues `feat/phase-1-facade-tool-validation` (or new `feat/phase-2-resilience-polish`)
**Status:** Awaiting user review

## Goal

Production-reliability hardening + Phase-1 polish, in a single tight batch. After Phase 2, the SDK can ship an agent into a production environment without silent data loss, undocumented failure modes, missing OPA enforcement, or DX traps from Phase 1.

## Scope (9 items)

### Resilience (5)

1. **LLM empty-response fix.** Currently `_normalise_response` returns `content=""` on truncation/content-filter responses. Phase 2 raises `LLMInvocationError(error_code="llm.empty_response")` and surfaces `finish_reason` on `LLMResponse`.
2. **Compaction timeout.** `MemoryManager.compact()` wraps the LLM call in `asyncio.wait_for(timeout)`. On timeout, log WARNING and skip — don't crash the agent.
3. **`error_code` field on `EAAPBaseException`.** Required-with-fallback-default. Each subclass declares a `DEFAULT_CODE` class attribute. Auto-populated into `details["error_code"]` and emitted as OTel span attribute `error.code`.
4. **New exception types.** `LLMTimeoutError` (← `LLMInvocationError`) for upstream timeouts; `MCPTransportError` (← `EAAPBaseException`) for FastMCP transport failures.
5. **Observability `fail_open` toggle.** `settings.observability.fail_open: bool = True`. When `False`, the broad `except` blocks in `RealObservabilityProvider` re-raise instead of swallowing — backend misconfiguration surfaces in dev.

### Phase-1 polish (4)

6. **Auto-register tools on `BaseAgent.compile()`.** Calls `self._tool_invoker.register(spec)` for each `ToolSpec` in `self.tools()`. Closes the DX trap where users wonder if `app.register_tools(*specs)` is required (it isn't, after Phase 2).
7. **`NoOpPolicyEvaluator` as DI default.** New `src/ai_core/security/noop_policy.py`. `AgentModule.provide_policy_evaluator` returns NoOp; production overrides with new opt-in `ProductionSecurityModule` that wires `OPAPolicyEvaluator`.
8. **`health.components` placeholder shape.** After successful `__aenter__`, the components dict is `{"settings": "ok", "container": "ok", "tool_invoker": "unknown", "policy_evaluator": "unknown", "observability": "unknown"}`. Real probes are Phase 3.
9. **`HealthSnapshot.settings_version` → `service_name`.** Field rename — the value is `service_name` already.

## Non-goals (explicitly deferred)

- Real health probes (OPA ping, DB connect, model lookup) — Phase 3.
- Structured logging (`structlog`) — Phase 3.
- Audit sink for OPA decisions — Phase 3.
- Anthropic prompt caching — Phase 3 or 4.
- MCP connection pooling — Phase 3.
- `eaap init` scaffolding updates — Phase 3 or 4.
- `tests/unit/security/test_opa.py` env fix (`respx` install) — env work, not in this phase.

## Constraints

- Pre-1.0; **no backwards-compatibility requirement.** Default DI binding for `IPolicyEvaluator` changes from OPA to NoOp.
- Every step gates on `ruff check`, `mypy <touched files>` strict, `pytest tests/unit tests/component` green.
- Total project mypy strict errors must remain at-or-below the 23-error baseline.
- End-of-phase smoke against `my-eaap-app` and the canonical `from ai_core import ...` path must continue to work.

## Module layout

```
src/ai_core/
├── exceptions.py              # MODIFIED — error_code on base, DEFAULT_CODE on subclasses,
│                                            new LLMTimeoutError + MCPTransportError
│
├── di/interfaces.py           # MODIFIED — LLMResponse gains finish_reason: str | None = None
│
├── llm/litellm_client.py      # MODIFIED — empty-response detection raises LLMInvocationError;
│                                            timeout-class errors raise LLMTimeoutError;
│                                            _normalise_response surfaces finish_reason
│
├── agents/memory.py           # MODIFIED — MemoryManager.compact() wraps LLM in
│                                            asyncio.wait_for; skip-and-WARN on TimeoutError
│
├── agents/base.py             # MODIFIED — compile() auto-registers each ToolSpec via
│                                            self._tool_invoker.register(spec)
│
├── config/settings.py         # MODIFIED — ObservabilitySettings.fail_open: bool = True;
│                                            AgentSettings.compaction_timeout_seconds: float = 30.0
│
├── observability/real.py      # MODIFIED — _swallow_or_raise helper reads fail_open;
│                                            _span tags error.code on EAAPBaseException
│
├── security/
│   ├── opa.py                 # unchanged
│   └── noop_policy.py         # NEW — NoOpPolicyEvaluator (always allow)
│
├── di/module.py               # MODIFIED — AgentModule binds NoOp by default;
│                                            new ProductionSecurityModule binds OPAPolicyEvaluator
│
├── app/runtime.py             # MODIFIED — health.components shape; settings_version → service_name
│
└── mcp/transports.py          # MODIFIED — wrap fastmcp ImportErrors / connection failures
                                              as MCPTransportError
```

## Files NOT touched (carrying WIP from before Phase 1)
- `README.md`
- `src/ai_core/cli/main.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`

## Component 1 — `error_code` field + new exceptions

### `EAAPBaseException` constructor

```python
class EAAPBaseException(Exception):
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
        # Auto-populate so every observability surface that walks `details` picks it up.
        self.details.setdefault("error_code", self.error_code)
        self.cause: BaseException | None = cause
        if cause is not None:
            self.__cause__ = cause
```

### Subclass `DEFAULT_CODE` mapping

| Class | `DEFAULT_CODE` |
|---|---|
| `EAAPBaseException` | `eaap.unknown` |
| `ConfigurationError` | `config.invalid` |
| `SecretResolutionError` | `config.secret_not_resolved` |
| `DependencyResolutionError` | `di.resolution_failed` |
| `StorageError` | `storage.error` |
| `CheckpointError` | `storage.checkpoint_failed` |
| `PolicyDenialError` | `policy.denied` |
| `LLMInvocationError` | `llm.invocation_failed` |
| `LLMTimeoutError` (NEW) | `llm.timeout` |
| `BudgetExceededError` | `llm.budget_exceeded` |
| `SchemaValidationError` | `schema.invalid` |
| `ToolValidationError` | `tool.validation_failed` |
| `ToolExecutionError` | `tool.execution_failed` |
| `AgentRuntimeError` | `agent.runtime_error` |
| `AgentRecursionLimitError` | `agent.recursion_limit` |
| `RegistryError` | `registry.error` |
| `MCPTransportError` (NEW) | `mcp.transport_failed` |

### Naming convention

Dotted, lowercase, snake_case after the dot. First segment is the subsystem (`llm`, `tool`, `mcp`, `agent`, `policy`, `config`, `schema`, `di`, `storage`, `registry`, `eaap`). Second segment names the failure mode in past tense (`failed`, `denied`, `not_resolved`, `exceeded`, `timeout`).

### New exception types

```python
class LLMTimeoutError(LLMInvocationError):
    """LLM call exceeded its configured timeout."""
    DEFAULT_CODE = "llm.timeout"


class MCPTransportError(EAAPBaseException):
    """An MCP transport (stdio/http/sse) failed to open or operate."""
    DEFAULT_CODE = "mcp.transport_failed"
```

### Top-level export updates

`src/ai_core/__init__.py` adds `LLMTimeoutError` and `MCPTransportError` to the curated public surface.

## Component 2 — `LLMResponse.finish_reason` + empty-response fix

### `LLMResponse` extension

```python
@dataclass(frozen=True, slots=True)
class LLMResponse:
    model: str
    content: str
    tool_calls: Sequence[Mapping[str, Any]]
    usage: LLMUsage
    raw: Mapping[str, Any]
    finish_reason: str | None = None   # NEW — None means upstream did not report
```

`None` is semantically distinct from a string value: it indicates the upstream (LiteLLM/provider) did not report a finish_reason at all.

### `_normalise_response` extracts `finish_reason` and raises on empty

```python
def _normalise_response(model: str, raw: Any) -> LLMResponse:
    payload: Mapping[str, Any] = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
    choices = payload.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    content = message.get("content") or ""
    tool_calls = list(message.get("tool_calls") or [])
    finish_reason = first.get("finish_reason")

    if not content and not tool_calls:
        raise LLMInvocationError(
            f"LLM returned empty response (finish_reason={finish_reason!r})",
            details={
                "model": model,
                "finish_reason": finish_reason,
                "raw_keys": sorted(payload.keys()),
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

### `LiteLLMClient.complete` distinguishes timeouts

```python
except RetryError as exc:
    last = exc.last_attempt.exception() if exc.last_attempt else exc
    if isinstance(last, (litellm.Timeout, asyncio.TimeoutError, TimeoutError)):
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

If `litellm.Timeout` import path differs across `litellm` versions, fall back to `type(last).__name__ == "Timeout"`.

### Behavior summary

| LLM upstream condition | Raises |
|---|---|
| Network timeout / connection reset (transient, retried) | After retries: `LLMTimeoutError` |
| HTTP 4xx (auth, bad-request) — non-retried | `LLMInvocationError` (default code) |
| HTTP 5xx (transient, retried) | After retries: `LLMInvocationError` |
| Returns `content=""` AND `tool_calls=[]` | `LLMInvocationError(error_code="llm.empty_response")` |
| Returns `content=""` AND `tool_calls=[...]` | Normal — no error |
| Returns `content="..."` | Normal — no error |

## Component 3 — Compaction timeout

### Settings

```python
class AgentSettings(BaseSettings):
    ...
    compaction_timeout_seconds: float = Field(default=30.0, gt=0)
```

### `MemoryManager.compact` wraps the LLM call

```python
class MemoryManager(IMemoryManager):
    async def compact(
        self,
        state: AgentState,
        *,
        tenant_id: str | None = None,
        agent_id: str | None = None,
    ) -> AgentState:
        timeout = self._settings.agent.compaction_timeout_seconds
        try:
            return await asyncio.wait_for(
                self._do_compact(state, tenant_id=tenant_id, agent_id=agent_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            _logger.warning(
                "Compaction skipped: LLM call exceeded %.1fs timeout (agent_id=%s, tenant_id=%s)",
                timeout, agent_id, tenant_id,
            )
            return state
```

`_do_compact` is the existing body (renamed) — pulled into a separate method so the wrap-with-timeout pattern is clean.

### Validator extension

None needed. Pydantic `Field(gt=0)` on `compaction_timeout_seconds` already enforces `> 0` at construction time, so a duplicate check in `validate_for_runtime` would be unreachable. The Pydantic-level error message ("ensure this value is greater than 0") is sufficient.

### Why skip-on-timeout is safe

- Compaction reduces context size; without it, the *next* `should_compact()` check triggers again. Threshold logic is idempotent.
- Worst case: persistent compaction failures cause the context to grow until the LLM client returns "context length exceeded". The agent's existing recursion limit caps this. Eventually the agent run terminates with a real exception.
- The skip is logged at WARNING with `agent_id` + `tenant_id`, so SREs can build a "compaction timeout rate per tenant" dashboard alert.

## Component 4 — Observability `fail_open` toggle + OTel `error.code` emission

### Settings

```python
class ObservabilitySettings(BaseSettings):
    ...
    fail_open: bool = Field(
        default=True,
        description=(
            "When True (default, recommended for production), backend errors "
            "are caught and logged. When False (recommended for local/dev), "
            "they re-raise so misconfigured exporters surface immediately."
        ),
    )
```

### `RealObservabilityProvider._swallow_or_raise` helper

```python
class RealObservabilityProvider(IObservabilityProvider):
    def __init__(self, settings: ObservabilitySettings, ...):
        ...
        self._fail_open: bool = settings.fail_open

    def _swallow_or_raise(self, exc: BaseException, context: str) -> None:
        if self._fail_open:
            _logger.warning("Observability backend error in %s: %s", context, exc)
            return
        raise exc
```

Each existing `except Exception as exc:` block calls `self._swallow_or_raise(exc, "<context>")`.

### OTel `error.code` emission

Inside the `start_span` async context manager, on exception exit:

```python
except EAAPBaseException as exc:
    span.set_attribute("error.code", exc.error_code)
    for k, v in (exc.details or {}).items():
        if isinstance(v, (str, int, float, bool)):
            span.set_attribute(f"error.details.{k}", v)
    span.record_exception(exc)
    raise
except BaseException as exc:
    span.record_exception(exc)
    raise
```

This is interleaved with the existing LangFuse handling.

### `FakeObservabilityProvider` mirrors

```python
@dataclass(slots=True)
class _RecordedSpan:
    name: str
    attributes: Mapping[str, Any]
    exception: BaseException | None = None
    error_code: str | None = None         # NEW

# In start_span context manager:
except BaseException as exc:
    recorded.exception = exc
    if isinstance(exc, EAAPBaseException):
        recorded.error_code = exc.error_code
    raise
```

A new `FakeBrokenObservabilityProvider` whose `start_span` raises immediately is added to `tests/conftest.py` for `fail_open` tests.

## Component 5 — `NoOpPolicyEvaluator` + DI rebinding

### `src/ai_core/security/noop_policy.py` (NEW)

```python
from collections.abc import Mapping
from typing import Any

from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision


class NoOpPolicyEvaluator(IPolicyEvaluator):
    """Policy evaluator that always allows. Suitable for development without OPA running.

    Production deployments MUST override the default DI binding with a real
    evaluator (e.g. :class:`OPAPolicyEvaluator`) via :class:`ProductionSecurityModule`.
    """

    async def evaluate(
        self, *, decision_path: str, input: Mapping[str, Any]
    ) -> PolicyDecision:
        return PolicyDecision(
            allowed=True,
            obligations={},
            reason="no-op evaluator (development only)",
        )
```

### DI rebinding in `di/module.py`

```python
class AgentModule(Module):
    """Default top-level module — uses NoOpPolicyEvaluator by default."""
    ...
    @singleton
    @provider
    def provide_policy_evaluator(self) -> IPolicyEvaluator:
        from ai_core.security.noop_policy import NoOpPolicyEvaluator
        return NoOpPolicyEvaluator()


class ProductionSecurityModule(Module):
    """Opt-in module that swaps NoOp for the real OPA-backed evaluator."""

    @singleton
    @provider
    def provide_policy_evaluator(self, settings: AppSettings) -> IPolicyEvaluator:
        from ai_core.security.opa import OPAPolicyEvaluator
        return OPAPolicyEvaluator(settings.security)
```

Production usage:

```python
from ai_core.di.module import ProductionSecurityModule

async with AICoreApp(modules=[ProductionSecurityModule()]) as app:
    ...
```

`AgentModule`'s previous OPA binding is removed; OPA loads only when `ProductionSecurityModule` is added. Local dev needs zero infrastructure.

## Component 6 — `BaseAgent.compile()` auto-registers tools

```python
def compile(self, *, checkpointer: Any | None = None) -> Any:
    if self._graph is not None:
        return self._graph
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("compact", self._compaction_node)
    graph.add_node("agent", self._agent_node)

    sdk_tools = [t for t in self.tools() if isinstance(t, ToolSpec)]

    # NEW: register each ToolSpec with the SchemaRegistry. Idempotent.
    for spec in sdk_tools:
        self._tool_invoker.register(spec)

    install_loop = self.auto_tool_loop and bool(sdk_tools)
    # ... rest of compile() unchanged
```

After Phase 2, `app.register_tools(*specs)` is the **power-user surface** for tools not attached to any agent (e.g., MCP-server-side tools, schema-export-only tools). Users who only define agent tools never call `register_tools`.

## Component 7 — `AICoreApp.health` polish

### `HealthSnapshot` dataclass — field rename

```python
@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    status: Literal["ok", "degraded", "down"]
    components: dict[str, Literal["ok", "unknown"]]
    service_name: str       # was settings_version
```

### `health` property — populated components

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

## Component 8 — `MCPTransportError` wrap

`mcp/transports.py` becomes the single place where transport-level errors map to SDK errors.

```python
class FastMCPConnectionFactory(IMCPConnectionFactory):
    def open(self, spec: MCPServerSpec) -> AbstractAsyncContextManager[Any]:
        try:
            from fastmcp import Client
            # ... existing transport selection logic
        except ImportError as exc:
            raise MCPTransportError(
                "FastMCP is not installed; install with `pip install ai-core-sdk[mcp]`",
                details={"component_id": spec.component_id, "transport": spec.transport},
                cause=exc,
            ) from exc
        # Connection failures inside the async context manager are wrapped via
        # an inner async generator that catches httpx/anyio errors and re-raises
        # as MCPTransportError.
```

The exact connection-failure wrapping depends on FastMCP's exception surface; the implementer handles this at task time. The contract is: callers see only `MCPTransportError` from this module.

## Error handling — consolidated table

| Error | Raised in | Default `error_code` | Phase 2 change |
|---|---|---|---|
| `ConfigurationError` | `validate_for_runtime` | `config.invalid` | new code |
| `SecretResolutionError` | `secrets.py` | `config.secret_not_resolved` | new code |
| `DependencyResolutionError` | `di/container.py` | `di.resolution_failed` | new code |
| `StorageError` / `CheckpointError` | `persistence/*` | `storage.error` / `storage.checkpoint_failed` | new codes |
| `PolicyDenialError` | `ToolInvoker` step 2 | `policy.denied` | new code |
| `LLMInvocationError` | `LiteLLMClient` | `llm.invocation_failed` | new code; specific `llm.empty_response` overrides at empty-response site |
| `LLMTimeoutError` (NEW) | `LiteLLMClient` after retries | `llm.timeout` | new exception + code |
| `BudgetExceededError` | `LiteLLMClient` step 1 | `llm.budget_exceeded` | new code |
| `SchemaValidationError` | `SchemaRegistry` | `schema.invalid` | new code |
| `ToolValidationError` | `ToolInvoker` steps 1+5 | `tool.validation_failed` | new code |
| `ToolExecutionError` | `ToolInvoker` step 4 | `tool.execution_failed` | new code |
| `AgentRuntimeError` | `BaseAgent` | `agent.runtime_error` | new code |
| `AgentRecursionLimitError` | `BaseAgent.ainvoke` | `agent.recursion_limit` | new code |
| `RegistryError` | `ComponentRegistry` | `registry.error` | new code |
| `MCPTransportError` (NEW) | `FastMCPConnectionFactory` | `mcp.transport_failed` | new exception + code |

## Implementation order (Approach 1 — bottom-up)

| Step | Deliverable | Tests | Dependencies |
|---|---|---|---|
| 1 | `error_code` retrofit + `LLMTimeoutError` + `MCPTransportError` | ~6 unit tests in `test_exceptions.py` | none |
| 2 | `LLMResponse.finish_reason` + empty-response fix + timeout-class fix | ~4 unit tests in `test_litellm_client.py` | Step 1 |
| 3 | `MemoryManager.compact` timeout + `compaction_timeout_seconds` setting | ~3 unit tests in `test_memory.py` | Step 1 |
| 4 | `ObservabilitySettings.fail_open` + `_swallow_or_raise` + OTel `error.code` emission | ~3 unit tests in `test_real_observability.py`; `FakeObservabilityProvider` extension | Step 1 |
| 5 | `NoOpPolicyEvaluator` + DI rebinding + `ProductionSecurityModule` | ~3 unit tests in security + DI dirs | none |
| 6 | `BaseAgent.compile()` auto-registers ToolSpecs | ~1 component test extension | none (Phase-1 polish only) |
| 7 | `AICoreApp.health` polish (`service_name` rename + components shape) | ~2 unit test extensions in `test_runtime.py` | none |
| 8 | `MCPTransportError` wrap in `transports.py` | ~2 unit tests in `tests/unit/mcp/` | Step 1 |
| 9 | End-of-phase smoke gate | full pytest + ruff + mypy + my-eaap-app smoke | all |

Total: ~22 new tests + updates to existing fakes (`FakeLLM`, `_ScriptedLLM` get `finish_reason` defaults; `FakeObservabilityProvider` records `error_code`).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `error_code` retrofit breaks tests asserting `details` dict shape | Step 1 explicitly grep-and-update before adding new tests; the auto-populated `details["error_code"]` may surprise existing assertions |
| `LLMResponse.finish_reason` breaks every test fake constructing `LLMResponse` | `None` default keeps non-passing constructs working; Step 2 audits `tests/` for `LLMResponse(` and updates as needed |
| Default DI binding swap (NoOp vs OPA) breaks tests relying on OPA-by-default | `tests/unit/security/test_opa.py` is currently broken (`respx` missing) — out of scope; rebinding work happens in `test_module_*.py`, not OPA tests |
| OTel `error.code` attribute emission interferes with existing span assertions | `FakeObservabilityProvider` extension records `error_code` separately; existing tests assert on `attributes` dict and won't see the new attribute unless they look |
| `litellm.Timeout` import path varies across litellm versions | Fallback to `type(last).__name__ == "Timeout"` check |
| MCP transport-failure wrapping depends on FastMCP exception surface | Implementer handles at task time; only the contract ("callers see `MCPTransportError`") is locked here |

## Constraints — recap

- 9 implementation steps + 1 smoke gate = ~10 plan tasks (matches Phase 1 cadence: 11 tasks).
- Per-step gate: ruff + mypy strict + pytest. Project mypy total ≤ 23.
- End-of-phase smoke must pass before the user merges.
- No backwards-compatibility shims (pre-1.0).
