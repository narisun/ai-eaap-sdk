# Changelog

All notable changes to `ai-eaap-sdk` are documented here. The format
roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased] — Phase 14: agent compositional primitives

First slice of the Phase 14 work: ship higher-level agent patterns
(supervisor, planner, verifier, harness) as first-class primitives so
AI engineers can build cutting-edge multi-agent applications without
hand-rolling LangGraph subgraphs. Each pattern auto-inherits v1's
cross-cutting concerns (DI, observability, policy, budget, audit, error
handling).

### Added

- **`ai_core.agents.SupervisorAgent`** — coordinates child
  `BaseAgent` instances through LLM-driven tool-call routing.
  Subclasses declare `children() -> Mapping[str, type[BaseAgent]]`;
  each entry becomes a tool the supervisor's LLM can invoke.
  Children-as-tools by design: 100 % reuse of `ToolInvoker`'s
  validation / OPA / audit / observability / `IToolErrorRenderer`
  pipelines. Each child runs through its own `BaseAgent.ainvoke`,
  with its own observability span nested under the supervisor's,
  its own budget binding (same tenant pool, separate agent
  budget), and OTel baggage propagation tagging delegation flow.

  Default child contract: `TaskInput(task: str, context: str | None)`
  → `TaskOutput(result: str)`. Hosts override
  `child_input_schema(name)` / `child_output_schema(name)` /
  `render_child_input` / `render_child_output` for typed contracts.

- **`ai_core.agents.AgentResolver`** (DI-aware sub-agent resolver) —
  thin wrapper over `Container.get` so compositional patterns don't
  smuggle the container into agent code. Provided by
  `AgentModule.provide_agent_resolver`; the running `Container`
  self-binds in its constructor.

- **New top-level exports**: `SupervisorAgent`, `TaskInput`,
  `TaskOutput`. Public surface contract test updated.

- **`AgentRuntime` gains** `agent_resolver: AgentResolver`. Existing
  `BaseAgent` subclasses that inherited the implicit constructor
  pass-through unchanged.

- **`Container.__init__`** self-binds via
  `injector.binder.bind(Container, to=self)` so internal providers
  (`AgentResolver`, FastAPI integrations) can receive the running
  container without closure-state hacks.

- **Runnable example** at `examples/supervisor_demo/run.py` —
  scripted-LLM demonstration with two children (`TriageAgent`,
  `ResearchAgent`).

### Fixed

- `tests/integration/test_opa_integration.py` was carrying a
  pre-existing v1 PR #1 regression: `OPAReachabilityProbe` and
  `OPAPolicyEvaluator` switched to `SecuritySettings` parameters in
  v1, but the integration test still passed `AppSettings`. CI's
  `-m "not integration"` filter hid the breakage. Fixed in passing.

### Tests

- 628 passed, 6 Docker-skipped (+11 supervisor unit tests, +2
  end-to-end component tests covering full DI + LangGraph flow).
- mypy strict: clean.
- ruff: clean.

## [1.0.1] — 2026-05-08

Type-checking and lint cleanup pass on top of v1.0.0. **No behavioural
changes**; the wheel built from this commit is byte-functional with
the v1.0.0 wheel for SDK consumers.

### Fixed

- `ai_core.observability.real`: `lf_span.end(status_message=str(exc))`
  closure-on-`exc` issue. The lambda captured `exc` from an
  `except` block; Python clears `except as exc` bindings on
  block-exit, so the deferred lambda would raise `NameError` if
  `_safe_lf_call` ever moved to async-deferred execution. Captures
  `err_msg = str(exc)` ahead of the lambda. Caught by ruff
  `F821` and verified.
- `ai_core.persistence.langgraph_checkpoint`: `aput` / `aget_tuple` /
  `alist` / `aput_writes` parameter and return-type annotations
  switched from `dict[str, Any]` to `RunnableConfig` so they no
  longer violate the Liskov substitution principle against
  LangGraph's `BaseCheckpointSaver`.
- `ai_core.agents.base._tool_node`: coerce LLM-supplied
  `tool_call.id` / `tool_call.name` to `str` before passing to the
  `IToolErrorRenderer`. Prevents a `TypeError` if a misbehaving
  provider returns `None` for either field.
- `ai_core.di.interfaces.ILLMClient.astream`: declared `async def`
  to match the implementations in `LiteLLMClient` and
  `ScriptedLLM`. Existing call shape (`await llm.astream(...)`
  followed by `async for chunk in stream:`) is unchanged.
- `ai_core.cli.main._jinja_env`: explicit `# noqa: S701` with
  rationale — the scaffold subcommand renders Python source files,
  not HTML, so HTML autoescape would mangle generated code.

### Build

- `pyproject.toml`: removed the redundant
  `[tool.hatch.build.targets.wheel.force-include]` block. The
  `packages = ["src/ai_core"]` directive already includes
  `cli/templates/`; listing them in force-include duplicated paths
  inside the wheel zip and emitted "Duplicate name" warnings on
  every build.
- `ai_core.__version__` now derives from
  `importlib.metadata.version("ai-eaap-sdk")` (already in v1.0.0
  re-cut, recapped here for the changelog record).

### Tooling

- `pyproject.toml` ruff ignore list expanded for our deliberate
  patterns: `PLC0415` (lazy imports for optional extras), `B008`
  (Pydantic `Field(default_factory=...)` and FastAPI `Depends`),
  `ANN401` (`Any` at untyped third-party boundaries). Reduces
  noise without weakening real checks.
- `pyproject.toml` mypy overrides extended to include
  `sentry_sdk.*` and `datadog.*`; both ship without stubs.
- 132 ruff auto-fixes applied (unused `# noqa` comments, trivial
  re-ordering, etc.). Tests untouched.
- 36 mypy strict errors fixed. `mypy src` is now **clean** —
  "Success: no issues found in 84 source files".

### Tests

- 611 passed, 6 Docker-skipped — same as v1.0.0.

[1.0.1]: https://github.com/narisun/ai-eaap-sdk/releases/tag/v1.0.1

## [1.0.0] — 2026-05-08

First production-ready release. Bundles Phase 13 cost/latency hardening
with a senior-architect-level v1 cleanup pass: 17 of 18 reviewed items
landed across six stacked PRs (#1–#6, plus the integration PR #7).

### Breaking

- **`BaseAgent` constructor**: replaced the six-argument `@inject`
  constructor with a single `runtime: AgentRuntime` parameter.
  Subclasses that explicitly listed all six args must switch to
  `(runtime: AgentRuntime)` and call `super().__init__(runtime)`.
  Subclasses that inherited the implicit ctor and only override
  `system_prompt` / `tools` / `mcp_servers` / `extend_graph` need no
  code changes.
- **`get_settings()` removed**: the `lru_cached` process-wide
  `AppSettings` accessor in `ai_core.config.settings` is gone.
  Construct `AppSettings()` directly (Pydantic Settings reads env /
  YAML / defaults at construction) or pass an instance to `AICoreApp`
  / `AgentModule`. The DI container is the only intended sharing
  seam.
- **Heavyweight provider adapters demoted to optional extras**:
  `litellm`, `langfuse`, and `fastmcp` are no longer installed by
  `pip install ai-eaap-sdk`. Add `[litellm]`, `[langfuse]`, `[mcp]`,
  or `[all]`:

      pip install ai-eaap-sdk[litellm,langfuse,mcp]

  The default `ILLMClient` binding is now `RaiseOnUseLLMClient`,
  which raises `ConfigurationError` with an install hint on first
  call. Hosts compose `LiteLLMModule()` alongside `AgentModule` to
  enable the LiteLLM-backed adapter, or bind their own
  `ILLMClient`.
- **`AICoreApp` facade**: dropped `app.policy_evaluator` and
  `app.observability` properties (they covered only a slice of bound
  interfaces). Use `app.get(IPolicyEvaluator)` /
  `app.get(IObservabilityProvider)` — the new `app.get(interface)`
  accessor is the curated seam.
- **Distribution rename**: package name is `ai-eaap-sdk`. Imports
  remain `import ai_core`.

### Added

- `AgentRuntime` (frozen dataclass) — bundle of SDK collaborators
  (settings, llm, memory, observability, tool invoker, mcp factory,
  tool error renderer, tool resolver, tool registrar) injected as a
  single argument to `BaseAgent`.
- `ICompactionLLM` — distinct Protocol so memory compaction can
  route to a cheaper model than agent reasoning. Default-aliased
  to the request `ILLMClient`.
- `Container.register_agent(cls)` and
  `AICoreApp.register_agent(cls)` for fail-fast agent binding
  (auto-bind retained for back-compat; scheduled for v2 removal).
- `AICoreApp.add_health_probe(probe)` convenience for one-off
  registration; documented multibind extension via
  `@multiprovider`.
- `ai_core.audit` registry pattern: `register_audit_sink(name, factory)`
  + `ai_eaap_sdk.audit_sinks` setuptools entry-point group for
  third-party sink discovery.
- `IToolErrorRenderer` — pluggable rendering of tool-dispatch
  failures into `ToolMessage` instances. `DefaultToolErrorRenderer`
  preserves pre-v1 English text.
- `ILLMClient.astream(...)` — streaming completion on the Protocol;
  real implementation in `LiteLLMClient` with the same budget /
  retry / observability / SLO semantics as `complete()`.
  `ScriptedLLM` supports it for tests.
- `make_tool(...)` — DI-aware tool factory accepting bound methods,
  closures, partials, lambdas. Complements `@tool` (which by design
  rejects methods).
- `IToolResolver` + `ToolRegistrar` — extracted from `BaseAgent` so
  `compile()` is pure graph construction. Hosts override resolution
  for cross-agent caching / tenant filtering / MCP stubbing.
- `ToolMiddleware` — around-advice chain wrapping the
  `ToolInvoker` pipeline. Hosts add cross-cutting concerns
  (rate-limit, sandbox, PII scrub, structured-output repair) via
  DI multibind.
- Typed exception details: nine stable-schema exception classes
  expose a frozen `@dataclass` payload via
  `exc.as_typed_details()`. Heterogeneous classes
  (`PolicyDenialError`, `ConfigurationError`, `RegistryError`,
  `LLMInvocationError`) keep raw-dict details for now.
- Lazy-import contract test (`tests/contract/test_lazy_imports.py`)
  runs in a subprocess and asserts `import ai_core` pulls zero
  modules from `{litellm, langfuse, fastmcp}` into `sys.modules`.
- Per-tenant / per-agent budget overrides
  (`AppSettings.budget.overrides`) with most-specific-wins
  resolution and per-axis (token / USD) override composition.
- Latency SLO observability: `LLMSettings.latency_slo_ms` triggers
  an `llm.slo_violated` event when an LLM call exceeds the
  threshold (alert-only; distinct from `request_timeout_seconds`).

### Changed

- Per-subsystem settings injection: every concrete service now
  injects only its slice (`LLMSettings`, `BudgetSettings`,
  `SecuritySettings`, `AgentSettings`, `DatabaseSettings`, …)
  instead of the whole `AppSettings`.
- `I*` interfaces converted from ABC to `@runtime_checkable
  Protocol` for structural typing. ABCs retained where shared
  default-method logic exists (`ISecretManager`, `JWTVerifier`,
  `BaseAgent`).

### Deferred

- Splitting `BaseAgent` into a pluggable-orchestrator shape so
  `langgraph` and `langchain-core` can also become optional
  extras. Tracked as future work; current `BaseAgent` imports
  `StateGraph` / `AIMessage` / `ToolMessage` at module load.

### Tests

611 passed, 6 Docker-skipped at the v1.0 commit. Net new since
0.x: +34 unit tests across `tests/unit/tools/test_factory.py`,
`tests/unit/tools/test_middleware.py`,
`tests/unit/exceptions/test_typed_details.py`,
`tests/unit/testing/test_scripted_llm.py`, and
`tests/contract/test_lazy_imports.py`.

[1.0.0]: https://github.com/narisun/ai-eaap-sdk/releases/tag/v1.0.0
