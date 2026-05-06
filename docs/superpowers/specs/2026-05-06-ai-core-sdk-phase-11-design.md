# Phase 11 — Agent-Side MCP Integration: Design

**Status:** Draft
**Date:** 2026-05-06
**Branch:** `feat/phase-11-mcp-agent-integration`
**Predecessor:** Phase 10 (PR #9, merged at `8ab9ffb`)

---

## 1. Architecture

Phase 11 closes the agent-MCP loop. Today the SDK ships transport-layer infrastructure (`MCPServerSpec`, `FastMCPConnectionFactory`) and an example client, but agents have no way to use a remote MCP server's tools as if they were local `@tool`s. Phase 11 makes that work end-to-end.

Design choices, settled during brainstorming:

- **Declarative on the agent class.** New `mcp_servers() -> Sequence[MCPServerSpec]` on `BaseAgent`, mirroring the existing `tools()` method. Per-agent wiring; explicit; no new dynamic-discovery infrastructure to design.
- **Permissive `MCPToolSpec` subclass.** MCP tools become first-class `ToolSpec` instances with passthrough Pydantic input/output models. `openai_schema()` is overridden to return FastMCP's `inputSchema` directly so the LLM sees real types. Input validation is delegated to the MCP server (the source of truth); `ToolInvoker`'s OPA / audit / observability pipeline runs uniformly across local and MCP tools.
- **Lazy resolution on first turn.** The first time an agent runs, it connects to its declared MCP servers, calls `list_tools()`, synthesizes `MCPToolSpec`s, and caches per-instance. Subsequent turns reuse the cache. No new lifecycle API; no `Container.start()` changes; misconfigured servers surface when the agent actually runs.
- **Flat tool names + per-server OPA path.** Conflicts (local vs MCP, or MCP vs MCP) raise `RegistryError` at resolution time. OPA decision path is configured per-server on `MCPServerSpec`; falls through to `ToolInvoker`'s standard policy machinery.

### Three deliverables

1. **`MCPToolSpec`** — frozen `ToolSpec` subclass that exposes a remote MCP tool via the `Tool` protocol. Permissive Pydantic models for `ToolInvoker` API contract; `openai_schema()` returns FastMCP's raw `inputSchema`. Handler is a closure that opens a pooled connection, calls `call_tool`, and maps `CallToolResult` to the output model.
2. **First-turn resolver path on `BaseAgent`** — new `mcp_servers()` method (default empty), plus async `_all_tools()` helper that runs resolution once per instance, merges MCP specs into the tools the LLM sees and the dispatcher dispatches to, and runs the conflict check.
3. **`MCPServerSpec.opa_decision_path: str | None = None`** — new optional field. When set, every tool call from that server checks that OPA path through `ToolInvoker`'s existing policy code. When unset, MCP tool calls skip OPA (matching local tools without `opa_path`).

### Module layout

```
src/ai_core/mcp/
├── __init__.py              # add MCPToolSpec to public exports
├── transports.py            # MCPServerSpec gains opa_decision_path field
├── tools.py                 # NEW: MCPToolSpec class + result mapping helpers
└── resolver.py              # NEW: async resolve_mcp_tools(servers, factory)

src/ai_core/agents/
└── base.py                  # extended: mcp_servers(), _all_tools() async helper, agent_node + _tool_node use the merged list

src/ai_core/di/
└── module.py                # AgentModule: bind IMCPConnectionFactory if not already bound

examples/mcp_server_demo/
├── README.md                # updated: drop "not yet shown" caveat; document agent_demo.py
└── agent_demo.py            # NEW: agent using the demo server's tools end-to-end

tests/unit/mcp/
├── test_tools.py            # NEW: MCPToolSpec construction, openai_schema passthrough, handler error mapping
└── test_resolver.py         # NEW: resolver with mocked factory, conflict detection

tests/integration/mcp/
└── test_agent_uses_mcp.py   # NEW: spawns demo FastMCP server, agent invokes echo through full pipeline

scripts/run_examples.sh      # extended: agent_demo.py added to the smoke gate
```

### Out of scope (deferred)

- `ComponentRegistry`-based dynamic MCP discovery (was option B in the brainstorming).
- JSON-Schema → strict Pydantic conversion (would replace permissive validation).
- Per-tool synthesized OPA paths (only per-server for v1).
- Tool-name prefixing (only flat names + compile-error for v1).
- Streaming results, tool progress, MCP resources/prompts (only `tools` in v1).
- Hot-reload: tools refresh only on container restart; if the server adds tools while the agent is running, the agent doesn't see them.

---

## 2. Components & data flow

### Component 1 — `MCPServerSpec` extension (`src/ai_core/mcp/transports.py`)

Add one field, default-`None`, fully backward-compatible:

```python
@dataclass(slots=True, frozen=True)
class MCPServerSpec:
    component_id: str
    transport: MCPTransport
    target: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    opa_decision_path: str | None = None   # NEW
```

### Component 2 — `MCPToolSpec` (`src/ai_core/mcp/tools.py`, new)

Frozen `ToolSpec` subclass. Two new fields: `mcp_server_spec` (the source server) and `mcp_input_schema` (FastMCP's raw `inputSchema` dict). One method override:

```python
@dataclass(frozen=True, slots=True)
class MCPToolSpec(ToolSpec):
    mcp_server_spec: MCPServerSpec
    mcp_input_schema: Mapping[str, Any]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.mcp_input_schema),
            },
        }
```

`input_model` and `output_model` are permissive Pydantic models defined in the same file:

```python
class _MCPPassthroughInput(BaseModel):
    model_config = ConfigDict(extra="allow")

class _MCPPassthroughOutput(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    value: Any = None
```

The `handler` is constructed in the resolver. It dumps the input model back to a dict, opens a pooled connection from the factory, calls `call_tool(name, args_dict)`, and maps the result:

- `result.is_error == True` → raise `ToolExecutionError("MCP tool returned error", details={"tool": name, "server": component_id, "content": ...})`.
- `result.data is not None` → wrap as `_MCPPassthroughOutput(value=result.data)`.
- otherwise → concatenate `content[*].text` and wrap as `_MCPPassthroughOutput(value=concatenated_text)`.

`opa_path` on the spec is set from `server.opa_decision_path` — `ToolInvoker`'s existing OPA pipeline does the rest with no Phase-11-specific code path.

### Component 3 — Resolver (`src/ai_core/mcp/resolver.py`, new)

Single async helper:

```python
async def resolve_mcp_tools(
    servers: Sequence[MCPServerSpec],
    factory: IMCPConnectionFactory,
) -> list[MCPToolSpec]:
    """For each server, list_tools and build MCPToolSpecs. Per-server connections are pooled.

    Raises:
        MCPTransportError: if any server is unreachable.
        RegistryError: if two MCP tools share a name (across servers or duplicated).
    """
    seen_names: set[str] = set()
    out: list[MCPToolSpec] = []
    for spec in servers:
        async with factory.open(spec) as client:
            tools = await client.list_tools()
        for fastmcp_tool in tools:
            name = fastmcp_tool.name
            if name in seen_names:
                raise RegistryError(
                    f"MCP tool name {name!r} appears in multiple servers",
                    details={"name": name, "server": spec.component_id},
                )
            seen_names.add(name)
            out.append(_build_mcp_tool_spec(spec, fastmcp_tool, factory))
    return out
```

`_build_mcp_tool_spec` constructs the closure handler and the `MCPToolSpec`. The pool is reused on every call (factory is `PoolingMCPConnectionFactory` by default).

### Component 4 — `BaseAgent` integration (`src/ai_core/agents/base.py`)

Three small changes:

1. **New injected dep.** `mcp_factory: IMCPConnectionFactory` added to `BaseAgent.__init__` and bound in `AgentModule.provide_mcp_connection_factory`.
2. **New overridable method:**

   ```python
   def mcp_servers(self) -> Sequence[MCPServerSpec]:
       """Override to declare MCP servers whose tools the agent uses."""
       return ()
   ```

3. **New async helper for first-turn resolution:**

   ```python
   async def _all_tools(self) -> list[Tool]:
       async with self._mcp_resolution_lock:
           if self._mcp_resolved is None:
               servers = list(self.mcp_servers())
               resolved = await resolve_mcp_tools(servers, self._mcp_factory) if servers else []
               local_names = {t.name for t in self.tools() if isinstance(t, ToolSpec)}
               for mcp_spec in resolved:
                   if mcp_spec.name in local_names:
                       raise RegistryError(
                           f"MCP tool name {mcp_spec.name!r} conflicts with a local tool",
                           details={"tool": mcp_spec.name},
                       )
                   self._tool_invoker.register(mcp_spec)
               self._mcp_resolved = resolved
       return list(self.tools()) + self._mcp_resolved
   ```

The two existing call sites — `agent_node`'s tool-payload build and `_tool_node`'s name-lookup dict — switch from `self.tools()` to `await self._all_tools()`. Both callers are already async.

In `BaseAgent.__init__`: `self._mcp_resolved: list[MCPToolSpec] | None = None` (None means "not yet resolved") and `self._mcp_resolution_lock = asyncio.Lock()` are initialized to support the lazy path and serialize concurrent first-turn races.

### Component 5 — Example (`examples/mcp_server_demo/agent_demo.py`, new)

A new third file in the demo directory, parallel to `run_client.py`. Defines an `MCPAgent` subclass overriding `mcp_servers()` to declare the demo server (spawning `server.py` as a stdio subprocess via `MCPServerSpec(transport="stdio", target=sys.executable, args=(str(server_path),))`). Drives the agent with a `ScriptedLLM` that emits a tool-call for `echo`, prints the result.

The README is updated to drop the "what's not yet shown" caveat (the agent-side adapter no longer being roadmap), document `agent_demo.py`, and add it as a "Run" alternative.

### Component 6 — Tests

- **`tests/unit/mcp/test_tools.py`** — pure unit tests of `MCPToolSpec`. Construct one with a fake schema; assert `openai_schema()` returns FastMCP's `inputSchema` directly (not Pydantic-derived). Test the handler with a mock connection factory: success path (data + content), error path (`is_error → ToolExecutionError`), content-concatenation path (no data, multiple text content items).
- **`tests/unit/mcp/test_resolver.py`** — mock `IMCPConnectionFactory.open` to yield a fake client whose `list_tools()` returns canned data. Assert: each MCP tool produces an `MCPToolSpec` with `opa_path` set from `server.opa_decision_path`; conflict detection raises `RegistryError` (test both same-server-twice and cross-server cases).
- **`tests/integration/mcp/test_agent_uses_mcp.py`** — uses the real `examples/mcp_server_demo/server.py` spawned as stdio subprocess via the production `PoolingMCPConnectionFactory`. Agent with `mcp_servers()` declared, `ScriptedLLM` emits a tool-call for `echo`, runs `ainvoke`. Asserts: tool resolved at first turn (cache populated); `FakeAuditSink` recorded the invocation with the right `tool_name`; result message has `value="..."`. No Docker needed — stdio transport is purely subprocess.

Backward-compat: existing agent tests must continue to pass without modification. `BaseAgent.mcp_servers()` default returning `()` ensures opt-out is the default. The new `IMCPConnectionFactory` injection is required, but `AgentModule` provides the binding by default.

The smoke gate (`scripts/run_examples.sh`) gets a new entry: `examples/mcp_server_demo/agent_demo.py`. Skips with the existing message if FastMCP is unavailable.

### Data flow

```
1. User: defines agent with `mcp_servers() -> [MCPServerSpec(...)]`
2. User: `await agent.ainvoke(...)`
3. agent_node (first call): `_all_tools()` triggers `resolve_mcp_tools()`
       → for each server: factory.open() → list_tools()
       → for each tool: build MCPToolSpec with closure handler
       → conflict check vs local tools
       → register each MCPToolSpec with ToolInvoker (idempotent)
       → cache on agent instance
4. agent_node: sends merged tool list (local + MCP) as openai_schema to LLM
5. LLM: emits tool_call by name
6. _tool_node: name lookup finds MCPToolSpec → ToolInvoker.invoke(spec, args)
       → input validation (no-op)
       → OPA policy (if opa_decision_path set; standard machinery)
       → audit BEFORE
       → handler: factory.open() (pooled) → call_tool(name, args) → result mapping
       → audit AFTER
       → output validation (no-op)
7. Result string flows back to LLM as tool message.
8. Subsequent turns: `_all_tools()` returns cached list, no re-resolution.
```

---

## 3. Error handling, testing, constraints

### Error handling

- **MCP server unreachable at first turn.** `factory.open(spec)` raises `MCPTransportError` (defined in `ai_core.exceptions`). The resolver doesn't catch it — propagation is up to `agent.ainvoke()`'s caller. The first turn fails loud; subsequent retries hit the same code path. Rationale: misconfigured MCP servers are real bugs and should not silently degrade an agent's capabilities. Users who want fail-soft behavior can wrap `mcp_servers()` to filter unreachable servers themselves.
- **`list_tools()` returns nothing.** The resolver returns `[]` for that server. No-op; agent runs with whatever tools resolved from other servers (or just local tools).
- **MCP tool returns `is_error=True`.** Handler raises `ToolExecutionError("MCP tool returned error", details={"tool": name, "server": component_id, "content": ...})`. `ToolInvoker`'s standard error path handles audit / observability / error-message-back-to-LLM uniformly with local tool failures.
- **Tool name conflict.** Resolver raises `RegistryError` at first-turn resolution. The error names the conflicting tool and the offending server. Resolution is "rename one of them, or drop the conflicting MCP server from `mcp_servers()`."
- **Concurrent first-turn races.** An `asyncio.Lock` per agent instance serializes the resolution block. Hot only on the very first turn — no contention afterward.
- **OPA decision path set but evaluator unavailable.** Standard `ToolInvoker` machinery — same behavior as a local tool with `opa_path` set when the policy evaluator can't reach OPA. No new code path.
- **Permissive Pydantic models.** Input is a passthrough — `model_validate(any_dict)` accepts everything. The MCP server validates server-side; if it rejects with a malformed-args error, that surfaces as `is_error=True` and we re-raise as `ToolExecutionError`. Output similarly accepts whatever shape FastMCP returns.

### Testing strategy

- **Unit tests** (`tests/unit/mcp/test_tools.py`, `test_resolver.py`) — fully mocked, no I/O. Cover `MCPToolSpec` construction + `openai_schema()` + handler dispatch + result mapping; resolver's conflict detection + per-server `opa_path` propagation.
- **Integration test** (`tests/integration/mcp/test_agent_uses_mcp.py`) — exercises the real `examples/mcp_server_demo/server.py` as a stdio subprocess, full agent pipeline. No Docker. End-to-end assertions: tool resolved, dispatch hits `ToolInvoker.invoke`, audit recorded, result shape correct.
- **Backward-compat verification.** Existing agent tests must pass without modification. `BaseAgent.mcp_servers()` default returning `()` ensures agents that don't opt in see no behavior change.
- **Smoke gate.** `examples/mcp_server_demo/agent_demo.py` becomes a new entry in `scripts/run_examples.sh`. Skips when FastMCP isn't installed (same condition as `run_client.py`).

### Constraints

- **Python 3.11+ required.** Subclassing `@dataclass(frozen=True, slots=True)` parent with another `@dataclass(frozen=True, slots=True)` child requires 3.11+. The SDK already requires 3.11+ per `pyproject.toml`, so no new constraint.
- **`IMCPConnectionFactory` must be DI-bound.** If it's not currently bound in `AgentModule`, Phase 11 adds the binding (`provide_mcp_connection_factory`).
- **No new runtime dependencies.** `fastmcp` is already a core dep (`pyproject.toml:46`). Resolver and tool spec use only stdlib + Pydantic + the SDK's own modules.
- **No streaming, no progress, no MCP resources/prompts.** v1 ships tool dispatch only.
- **No hot-reload of the MCP tool list.** Once resolved, an agent's MCP tool set is fixed for the lifetime of the agent instance.
- **Closure-captured factory.** Each `MCPToolSpec.handler` captures `IMCPConnectionFactory` at resolution time. Already-resolved agents keep using the captured factory even if the DI binding is later replaced. Matches the existing DI pattern — most overrides happen before agent instantiation.
- **Subclass `__init__` forwarding.** If a custom agent overrides `BaseAgent.__init__`, it must forward `mcp_factory` via `super().__init__(...)`. Existing agents in the SDK and examples don't override `__init__` (they rely on `@inject`), so this is theoretical — but worth knowing for users who do.
