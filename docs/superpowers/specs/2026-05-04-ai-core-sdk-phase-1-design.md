# ai-core-sdk Phase 1 — Design

**Date:** 2026-05-04
**Branch:** `feat/phase-1-facade-tool-validation`
**Status:** Awaiting user review

## Goal

Lift the SDK from "a DI primitive an AI engineer can use if they read the source" to "an SDK an AI engineer can use after reading one README page." Three concrete deliverables:

1. **`AICoreApp`** — an opinionated lifecycle facade so a developer never assembles a DI container by hand.
2. **`@tool` decorator** — one decorator that derives the OpenAI tool schema from a Pydantic input model, registers with `SchemaRegistry`, validates input/output at runtime, opens an OTel span, and routes through OPA.
3. **`AppSettings.validate_for_runtime()`** — collect-all schema validation that fails fast at app entry instead of at first LLM call.

## Non-goals (deferred to later phases)

- I/O-based health probes, prompt caching, MCP connection pooling, audit sink, structured logging, `error_code` field on exceptions, `LLMTimeoutError` / `MCPTransportError`, `eaap init` scaffold updates, contract test suites.

## Constraints

- Pre-1.0; **no backwards-compatibility requirement**. Existing public surface (`Container`, `AgentModule`) is preserved because it's still useful, not because it's frozen.
- Every step gates on `ruff check src tests`, `mypy src --strict`, and `pytest tests/unit tests/component`.
- End-of-phase smoke gate: the consumer at `/Users/admin-h26/EAAP/my-eaap-app` must still run.

## Module layout

Phase 1 adds three new modules and modifies three existing files. No deletions.

```
src/ai_core/
├── __init__.py                # CURATED — top-level exports of facade + agents + tools
│
├── app/                       # NEW — application facade
│   ├── __init__.py            # exports AICoreApp, HealthSnapshot
│   └── runtime.py             # AICoreApp class
│
├── tools/                     # NEW — @tool decorator + types
│   ├── __init__.py            # exports tool, Tool, ToolSpec
│   ├── spec.py                # ToolSpec dataclass + Tool protocol
│   ├── decorator.py           # @tool implementation
│   └── invoker.py             # ToolInvoker — schema + OPA + span + body pipeline
│
├── config/
│   ├── settings.py            # MODIFIED — adds AppSettings.validate_for_runtime()
│   └── validation.py          # NEW — collect-all validator helpers + ConfigIssue
│
├── exceptions.py              # MODIFIED — adds ToolValidationError, ToolExecutionError
│
├── di/module.py               # MODIFIED — adds ToolInvoker provider binding
│
└── agents/base.py             # MODIFIED — accepts Tool objects, dispatches via ToolInvoker,
                                  auto-installs tool_node when @tool tools are returned
```

### Boundaries

- `tools/` knows about `SchemaRegistry`, `IPolicyEvaluator`, `IObservabilityProvider`. Does **not** know about `BaseAgent` or `Container`.
- `app/` is the only module that touches both `Container` and `tools/`. It is the wiring layer.
- `BaseAgent` learns about `Tool` objects but stays a thin orchestrator.
- `config/validation.py` is a sibling helper so `settings.py` does not bloat.

## Component 1 — `AppSettings.validate_for_runtime()`

Collect-all schema validation, no I/O. Surfaces every config problem at once with actionable hints, before any DI binding happens.

### API

```python
# config/validation.py
@dataclass(frozen=True)
class ConfigIssue:
    path: str          # dotted, e.g. "llm.default_model"
    message: str       # what's wrong
    hint: str | None   # how to fix

class ValidationContext:
    issues: list[ConfigIssue]
    def fail(self, path: str, message: str, hint: str | None = None) -> None: ...
    @property
    def has_issues(self) -> bool: ...

# config/settings.py — additive method on AppSettings
def validate_for_runtime(self, *, secret_manager: ISecretManager | None = None) -> None:
    """Collect-all validation. Raises ConfigurationError once with all issues."""
```

### Checks (Phase 1, no I/O)

| Path | Check |
|---|---|
| `llm.default_model` | non-empty |
| `llm.proxy_base_url` | valid URL if set |
| `database.url` | valid SQLAlchemy URL string if set |
| `security.opa_url` | required + valid URL when `security.policy_enforcement_enabled` |
| `security.jwt_*` | required when `security.jwt_enabled` |
| every `SecretRef` | resolves via the supplied `secret_manager` |
| `agent.max_recursion_depth` | `>= 1` |
| `agent.compaction_target_tokens` < `agent.compaction_threshold_tokens` | cross-field invariant |

### Failure mode

```python
raise ConfigurationError(
    "Runtime configuration is invalid: 3 issue(s) found.",
    details={"issues": [{"path": "...", "message": "...", "hint": "..."}, ...]},
)
```

Count appears in the message so log scrapes do not require parsing `details`.

### Out of scope

No `litellm.utils` lookup, DB connect, OPA ping, or Langfuse handshake — those belong with the Phase-2 health-check work.

## Component 2 — `@tool` decorator + `ToolSpec` + `ToolInvoker`

Three collaborators with separated responsibilities.

### 2a. The decorator (definition-time, zero DI)

```python
# tools/decorator.py
def tool(
    *,
    name: str,
    version: int = 1,
    description: str | None = None,                     # falls back to docstring
    opa_path: str | None = "eaap/agent/tool_call/allow",
) -> Callable[[ToolHandler], ToolSpec]:
    ...
```

Decorator inspects the function signature and asserts:
- async function,
- exactly one positional parameter typed as a `BaseModel` subclass,
- return annotation is a `BaseModel` subclass.

Returns a `ToolSpec`, not the original function. Definition-time has zero DI dependency; observability/OPA are passed at invoke time.

### 2b. `ToolSpec` (immutable descriptor)

```python
# tools/spec.py
@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: int
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[BaseModel]]
    opa_path: str | None

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }

class Tool(Protocol):
    name: str
    version: int
    def openai_schema(self) -> dict[str, Any]: ...
```

`ToolSpec` satisfies `Tool`. The Protocol is forward-compat (e.g., remote MCP tools could implement `Tool` later).

### 2c. `ToolInvoker` (runtime — owns the pipeline)

```python
# tools/invoker.py
class ToolInvoker:
    def __init__(
        self,
        *,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator | None = None,
        registry: SchemaRegistry | None = None,
    ) -> None: ...

    async def invoke(
        self,
        spec: ToolSpec,
        raw_args: Mapping[str, Any],
        *,
        principal: Mapping[str, Any] | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Mapping[str, Any]:
        """
        Pipeline:
          1. Validate input  (raw_args -> spec.input_model)  -> ToolValidationError(side="input")
          2. If spec.opa_path and policy: evaluate           -> PolicyDenialError on deny
          3. Open OTel span "tool.invoke" with attrs
          4. await spec.handler(payload)                     -> ToolExecutionError on raise (chained)
          5. spec.output_model.model_validate(result)        -> ToolValidationError(side="output")
                # accepts an output_model instance or a shape-correct dict
          6. record_event("tool.completed", ...)
          7. Return validated.model_dump()
        """
```

Pipeline-order rationale:
- Input validation runs **before** OPA so policies reference typed fields, not raw JSON.
- Output validation runs **after** the handler so a buggy handler is caught with a clear error rather than corrupting the LLM's view.

### 2d. New exceptions

```python
# exceptions.py — additive
class ToolValidationError(SchemaValidationError):
    """Tool input or output failed Pydantic validation.
    details: {tool, version, side: 'input' | 'output', errors: [...]}
    """

class ToolExecutionError(EAAPBaseException):
    """Tool handler raised. Original cause preserved via __cause__.
    details: {tool, version, agent_id, tenant_id}
    """
```

### 2e. `BaseAgent` integration

Today's graph is one-shot (`agent → END`). For `@tool` tools to actually run, `BaseAgent.compile()` auto-installs a `tool_node` and a loop **when** `tools()` returns at least one `Tool`-protocol object:

```
START → (compact | agent)
agent → has_tool_calls?  ─yes→ tool_node → agent   (loop, bounded by max_recursion_depth)
                          └no─→ END
```

`tool_node` reads `state.messages[-1].tool_calls`, looks up each by name in `self.tools()`, dispatches via the injected `ToolInvoker`, and appends `ToolMessage` results.

Subclasses opt out by setting `auto_tool_loop = False`. Legacy raw-`Mapping` tools get no loop (caller wires it). This is the SDK handling the cross-cutting concern by default but staying overridable.

### 2f. `SchemaRegistry` interaction

`AICoreApp.register_tools(*specs)` registers each `ToolSpec` with the existing `SchemaRegistry` keyed by `(name, version)` so the CLI's schema-export and cross-tool discovery still work. The legacy `validate_tool` decorator is kept for non-`@tool` tool functions.

## Component 3 — `AICoreApp` facade

One object owns the lifecycle: settings → DI → MCP registry → shutdown. `validate_for_runtime` runs before the container is built. `ToolInvoker` is bound in the DI graph (no instance mutation).

### API

```python
# app/runtime.py
class AICoreApp:
    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        modules: Sequence[Module] = (),
        secret_manager: ISecretManager | None = None,
    ) -> None: ...

    async def __aenter__(self) -> "AICoreApp":
        self._settings = self._settings or get_settings()
        self._secret_manager = self._secret_manager or EnvSecretManager()
        self._settings.validate_for_runtime(secret_manager=self._secret_manager)
        self._container = Container.build([
            AgentModule(settings=self._settings, secret_manager=self._secret_manager),
            *self._modules,
        ])
        await self._container.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._container.stop()    # idempotent

    def agent(self, cls: type[A]) -> A:
        return self._container.get(cls)  # ToolInvoker auto-injected

    def register_tools(self, *specs: ToolSpec) -> None: ...
    def register_mcp(self, spec: MCPServerSpec) -> ComponentHandle: ...

    @property
    def policy_evaluator(self) -> IPolicyEvaluator: ...
    @property
    def observability(self) -> IObservabilityProvider: ...
    @property
    def settings(self) -> AppSettings: ...
    @property
    def container(self) -> Container: ...     # power-user escape hatch
    @property
    def health(self) -> HealthSnapshot: ...   # Phase-1 stub: always 'ok' if entered
```

### DI binding for `ToolInvoker`

```python
# di/module.py — additive
@singleton
@provider
def provide_tool_invoker(
    self,
    observability: IObservabilityProvider,
    policy: IPolicyEvaluator,
    registry: SchemaRegistry,
) -> ToolInvoker:
    return ToolInvoker(observability=observability, policy=policy, registry=registry)
```

### `BaseAgent` injects `ToolInvoker`

```python
class BaseAgent(ABC):
    @inject
    def __init__(
        self,
        settings: AppSettings,
        llm: ILLMClient,
        memory: IMemoryManager,
        observability: IObservabilityProvider,
        tool_invoker: ToolInvoker,         # NEW
    ) -> None: ...
```

### Health stub

```python
@dataclass(frozen=True)
class HealthSnapshot:
    status: Literal["ok", "degraded", "down"]
    components: dict[str, Literal["ok", "unknown"]]
    settings_version: str
```

Phase 1 returns `status="ok"` if `__aenter__` succeeded. Real per-component probes are Phase 2.

## Top-level package exports

```python
# src/ai_core/__init__.py
from ai_core.app import AICoreApp, HealthSnapshot
from ai_core.agents import BaseAgent, AgentState, new_agent_state
from ai_core.tools import tool, Tool, ToolSpec
from ai_core.exceptions import (
    EAAPBaseException, ConfigurationError, DependencyResolutionError,
    SecretResolutionError, StorageError, PolicyDenialError, BudgetExceededError,
    LLMInvocationError, SchemaValidationError,
    ToolValidationError, ToolExecutionError,                # NEW
    AgentRuntimeError, AgentRecursionLimitError, RegistryError,
)
__version__ = "0.1.0"
```

A new developer reads two lines and has the canonical "Hello, agent" example:

```python
from ai_core import AICoreApp, BaseAgent, tool
from pydantic import BaseModel
```

`Container`, `AgentModule`, the `I*` interfaces, `AppSettings`, `ISecretManager`, `SchemaRegistry`, `IPolicyEvaluator`, `JWTVerifier`, `MCPServerSpec` are intentionally **not** at the top level — available at their subpackage paths for power users.

## Error handling — consolidated

### Where each error originates

| Error | Raised in | Propagation | LLM sees |
|---|---|---|---|
| `ConfigurationError` | `validate_for_runtime` | re-raised verbatim from `__aenter__` | n/a |
| `SecretResolutionError` | `validate_for_runtime` | becomes a `ConfigurationError` issue | n/a |
| `ToolValidationError` (input) | `ToolInvoker` step 1 | `tool_node` -> `ToolMessage("Input validation failed: ...")` | recoverable |
| `PolicyDenialError` | `ToolInvoker` step 2 | `tool_node` -> `ToolMessage("Tool denied by policy: <reason>")` | recoverable |
| `ToolExecutionError` | `ToolInvoker` step 4 | `tool_node` -> `ToolMessage("Tool 'X' failed: <short>")` | recoverable |
| `ToolValidationError` (output) | `ToolInvoker` step 5 | `tool_node` -> `ToolMessage` + WARN log (handler bug) | recoverable |
| `AgentRecursionLimitError` | LangGraph wrapper | terminates run | n/a |
| `BudgetExceededError`, `LLMInvocationError` | `LiteLLMClient` | propagate up unchanged | n/a |

Two clean rules:
- **Tool-call errors -> LLM gets a message.** Agent never crashes on a tool failure; LLM decides whether to retry, switch, or apologize.
- **Boot/config/budget/recursion errors -> caller gets the exception.** These are program-level failures, not conversational ones.

### Error-message quality bar (the "rustc rule")

Every `EAAPBaseException` in Phase 1 follows: **what failed, why, and what to do next**.

```python
ConfigurationError(
    "Runtime configuration is invalid: 2 issue(s) found.",
    details={"issues": [
        {"path": "llm.default_model", "message": "must be non-empty",
         "hint": "set env var EAAP_LLM__DEFAULT_MODEL=<model-id> or override AppSettings.llm.default_model in code"},
    ]},
)

ToolValidationError(
    "Tool 'search_orders' v1 input failed validation: 'limit' must be >= 1, got -3.",
    details={"tool": "search_orders", "version": 1, "side": "input",
             "errors": [{"loc": ("limit",), "msg": "..."}]},
)

ToolExecutionError(
    "Tool 'search_orders' v1 failed: ConnectionError contacting orders-api.",
    details={"tool": "search_orders", "version": 1, "agent_id": "concierge", "tenant_id": "acme"},
    cause=original_exc,
)
```

`ToolMessage` content shown to the LLM drops infra detail (no stack frames, no internal paths) but keeps the actionable signal. Full structured detail still flows to OTel via `_observability`.

### Logging policy (additive, stdlib `logging`)

| Level | What goes here |
|---|---|
| `DEBUG` | Tool dispatch decisions, OPA inputs, schema-registry hits |
| `INFO` | Tool invocation start/end with name + version + duration |
| `WARNING` | Output-validation failures (handler bugs); fallbacks engaged |
| `ERROR` | `ToolExecutionError` chains; `ConfigurationError` issue list |

Logger names follow `ai_core.<subpackage>` so callers can mute selectively.

`structlog` adoption is Phase 2.

## Testing strategy

Per-step gate: `ruff check src tests` clean, `mypy src --strict` clean, `pytest tests/unit tests/component` green.

### Reusable fakes (in `tests/conftest.py`)

| Fake | Replaces |
|---|---|
| `FakePolicyEvaluator` | `IPolicyEvaluator` |
| `FakeObservabilityProvider` | `IObservabilityProvider` (records spans/events for assertion) |
| `FakeSecretManager` | `ISecretManager` |
| `FakeLLMClient` | `ILLMClient` (already exists) |
| `FakeMemoryManager` | `IMemoryManager` (already exists) |

### Step 1 — `validate_for_runtime` (~12 unit tests)

One test per check; one collect-all test (3 simultaneous holes -> 3 issues); happy-path test.

### Step 2 — `@tool` / `ToolSpec` / `ToolInvoker` (~20 unit tests)

- Decorator (5): rejects sync funcs, non-Pydantic args, non-Pydantic returns; falls back to docstring; returns populated `ToolSpec`.
- ToolSpec (3): `openai_schema()` round-trips through `json.dumps`; equality on (name, version).
- ToolInvoker (12): happy path; input-validation failure; OPA deny; handler raise (chained cause); output-validation failure; `opa_path=None` skips OPA; `policy=None` skips OPA; span attributes correct; exception recorded on span; pipeline order asserted; `tool.completed` event fires; idempotent `SchemaRegistry` registration.

### Step 3 — `BaseAgent` integration (~6 component tests)

- `@tool` tools auto-install the loop.
- Multi-step `agent -> tool -> agent -> END`.
- `ToolValidationError`/`ToolExecutionError` surface as `ToolMessage` content; LLM-recovery path exercised.
- `PolicyDenialError` surfaces as `ToolMessage` content.
- `auto_tool_loop = False` opts out.
- Recursion limit caps the loop.

### Step 4 — `AICoreApp` facade (~8 unit + 2 component)

Unit: validation runs in `__aenter__`; failure does not build container; `__aexit__` is idempotent; `agent(cls)` returns instance with DI-injected `ToolInvoker`; `register_tools` idempotent; `register_mcp` returns handle; properties resolve; `container` escape hatch works.

Component: end-to-end "Hello, agent with `@tool`" using only top-level imports + `FakeLLMClient`.

### Step 5 — End-of-phase smoke gate

- Full `pytest` green.
- `ruff check src tests` clean.
- `mypy src --strict` clean.
- `my-eaap-app` runs against the new SDK.

### Coverage target

≥85% on new code (`app/`, `tools/`, `config/validation.py`). Existing coverage must not regress.

## Implementation order (Approach 1 — bottom-up)

1. Settings validator (smallest, isolated). Tests + ruff + mypy.
2. `@tool` + `ToolSpec` + `ToolInvoker`. Tests + ruff + mypy.
3. `BaseAgent` integration (auto loop, `ToolInvoker` injection). Component tests + ruff + mypy.
4. `AICoreApp` facade + top-level exports. Unit + component tests + ruff + mypy.
5. End-of-phase smoke gate (full suite + `my-eaap-app`).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `BaseAgent` graph change breaks existing agents | `auto_tool_loop = False` opt-out; legacy `Sequence[Mapping]` tools get no loop. |
| `mypy --strict` not currently green on existing code | Run baseline first; fix only new-code issues; defer pre-existing failures to a tracked follow-up. |
| `injector` cycle when `ToolInvoker` requires `IPolicyEvaluator` and same module also wires policy | Bind `ToolInvoker` in same `AgentModule` as a `@singleton @provider`; `injector` handles ordering. |
| `register_tools` called before `__aenter__` | Raise `RuntimeError("AICoreApp not entered")` from any method requiring a built container. |
