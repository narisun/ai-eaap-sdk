# Changelog

All notable changes to `ai-eaap-sdk` are documented here. The format
roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased] — Phase 14: agent compositional primitives

Higher-level agent patterns (supervisor, planner, verifier, harness)
as first-class primitives so AI engineers can build cutting-edge
multi-agent applications without hand-rolling LangGraph subgraphs.
Each pattern auto-inherits v1's cross-cutting concerns (DI,
observability, policy, budget, audit, error handling).

### Added — slice 4: HarnessAgent (capture-only tracing)

- **`ai_core.agents.HarnessAgent`** — replay-grade tracing primitive.
  Wraps a child :class:`BaseAgent` and records every LLM call and tool
  dispatch the wrapped agent makes into a structured :class:`Trace`,
  exposed via ``HarnessAgent.last_trace``. Hosts persist via
  ``trace.model_dump_json()`` and feed the result to dashboards,
  evals, or future replay machinery.

  Capture mechanism: the harness builds a customised
  :class:`AgentRuntime` via ``dataclasses.replace`` whose ``llm`` is a
  ``_CapturingLLMClient`` (satisfies :class:`ILLMClient`
  structurally) and whose ``tool_invoker`` is a
  ``_CapturingToolInvoker`` (satisfies :class:`IToolInvoker`). Every
  other collaborator (memory, observability, audit, MCP, agent
  resolver) is shared with the harness's own runtime, so capture is
  non-intrusive.

  Scope is **capture only**; replay is deferred to a future slice.
  The :class:`Trace` data model is the public surface that future
  replay logic builds on — JSON-serialisable so traces persist and
  replay across SDK versions without coupling to in-memory types.

  Caveat: :class:`MemoryManager` compaction uses the separate
  :class:`ICompactionLLM` DI binding and is therefore **not**
  captured in v1. Hosts that need compaction-LLM capture override
  the binding in their test container.

- **`ai_core.agents.{Trace, TraceEvent, LLMCallRecord,
  ToolDispatchRecord}`** — Pydantic data model exposed on the public
  surface so hosts can introspect traces programmatically.
  ``ToolDispatchRecord`` captures both ``outcome="ok"`` (with the
  validated result) and ``outcome="error"`` (with exception type and
  message); failures are recorded then re-raised so the wrapped
  agent's error-recovery paths fire normally.

- **`ai_core.tools.invoker.IToolInvoker`** — new ``runtime_checkable``
  Protocol capturing the dispatch surface
  (:meth:`invoke` / :meth:`register`) the agent runtime consumes.
  :class:`AgentRuntime.tool_invoker` is now typed as the Protocol so
  hosts (and SDK primitives like :class:`HarnessAgent`) can interpose
  wrapping implementations without subclassing the concrete
  :class:`ToolInvoker`. The concrete class is unchanged and satisfies
  the Protocol structurally.

- **Runnable example** at ``examples/harness_demo/run.py`` — wraps a
  real support agent with a tool, runs it under capture, and renders
  the captured event sequence + a JSON snippet of the trace.

### Added — slice 3: VerifierAgent

- **`ai_core.agents.VerifierAgent`** — output-verification primitive.
  Wraps a single child :class:`BaseAgent`. After the child produces
  a final answer, the verifier issues a separate LLM call against
  a host-supplied rubric and gets back a structured
  :class:`Verdict` (passed / feedback / issues / score). On
  ``Verdict.passed=False``, the wrapped agent re-runs with the
  verdict's feedback injected as a new user message; up to
  ``max_retries`` retries.

  Direct LLM call (not synthetic tool) for the verification step —
  verification is a control-flow decision, not a tool dispatch, and
  routing through :class:`ToolInvoker` would add no value (no OPA
  on the verification act itself). The verifier still emits its own
  ``agent.verify`` span for observability and goes through its own
  ``IBudgetService`` binding.

  Composition: a verifier can wrap a :class:`SupervisorAgent` (gate
  multi-agent flows on a final verification), a
  :class:`PlanningAgent` (verify the plan-and-execute final answer),
  or another :class:`VerifierAgent` (layered critique). A verifier
  can also be a child of a supervisor.

  Strict-mode default (``strict=True``): raise
  :class:`AgentRuntimeError` when verification fails after
  ``max_retries`` so unverified answers can't silently leak.
  ``strict=False`` returns the last attempt with the final verdict
  in ``state.metadata["last_verdict"]`` for hosts that want to
  inspect and decide.

  Verdict history accumulates in
  ``state.scratchpad["verifications"]`` so eval / replay surfaces
  see every retry.

- **`ai_core.agents.Verdict`** — Pydantic data model exposed on the
  public surface. Hosts can reach into ``state.metadata["last_verdict"]``
  programmatically (dashboards, eval harnesses, alerting).

- **Runnable example** at ``examples/verifier_demo/run.py`` —
  citation-checking verifier wrapping a factual answerer; shows the
  fail → feedback → revise → pass cycle.

### Added — slice 2: PlanningAgent

- **`ai_core.agents.PlanningAgent`** — plan-and-execute primitive.
  The LLM declares a structured plan via a synthetic ``_make_plan``
  tool call, then executes each step via the user's work tools. Plan
  history is preserved in ``state.scratchpad["plans"]`` so
  re-planning is informed by what didn't work. ``max_replans`` caps
  revisions; once hit, the system prompt nudges the LLM to finalize.

  Implementation: dynamic system prompt that switches between four
  modes (initial / executing / done / replan-cap-reached) based on
  live plan state read from ``self._current_state``. The
  ``_make_plan`` tool's handler stashes plans on a side-channel that
  the overridden ``_tool_node`` merges into ``state.scratchpad``.
  Step status is tracked implicitly — the LLM revises step status
  by calling ``_make_plan`` again with the updated step list.

- **`ai_core.agents.{Plan, PlanStep, PlanAck, StepStatus}`** —
  Pydantic data model exposed on the public surface so hosts can
  introspect the plan from ``state.scratchpad`` programmatically
  (eval, replay, dashboards).

- **`AgentState.scratchpad`** — new optional ``dict[str, Any]``
  field for per-pattern scratch space. Phase 14 patterns store
  typed payloads at well-known keys (``scratchpad["plans"]`` for
  ``PlanningAgent``; future: ``scratchpad["verifications"]`` for
  ``VerifierAgent``). Default reducer is overwrite; patterns own
  any merge semantics they need internally.

- **Runnable example** at ``examples/planner_demo/run.py`` —
  end-to-end plan-declare-execute-finalize loop with a scripted
  LLM and one work tool.

### Added — slice 1: SupervisorAgent

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
