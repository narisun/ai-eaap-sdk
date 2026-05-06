# Phase 12 — MCP Resources + Prompts: Design

**Status:** Draft
**Date:** 2026-05-06
**Branch:** `feat/phase-12-mcp-resources-prompts`
**Predecessor:** Phase 11 (PR #10, merged at `ac78ef6`)

---

## 1. Architecture

Phase 12 extends the agent ↔ MCP integration along three small axes, all anchored on the patterns Phase 11 established:

- **MCP resources as tools.** Each resource the server exposes via `list_resources()` becomes a read-only tool the LLM can call. Same lazy-first-turn resolution + conflict detection + `ToolInvoker` dispatch as Phase 11 tools. The only thing that changes is the handler closure (calls `read_resource(uri)` instead of `call_tool(name, args)`).
- **MCP prompts as application helpers.** `await agent.list_prompts()` / `await agent.get_prompt(name, args)` — async methods that fetch fresh each call (no lazy cache). Application-invoked, not LLM-invoked. The application splices the templated messages into `ainvoke(messages=...)` itself.
- **`unwrap_mcp_tool_message()` helper.** A free function in `ai_core.mcp.tools` that takes a `ToolMessage.content` JSON string and returns the unwrapped `_MCPPassthroughOutput.value` when present. Closes the pedagogical wart Phase 11's reviewer flagged.

### Three deliverables

1. **`MCPResourceSpec`** — frozen subclass of `MCPToolSpec` adding `mcp_resource_uri: str`. Inherits all Phase 11 plumbing (passthrough Pydantic, `ToolInvoker` integration). The new field carries the URI; the handler closure dispatches via `client.read_resource(uri)`. `openai_schema()` returns `{"type": "object", "properties": {}}` since resources take no parameters.

2. **Prompt API on `BaseAgent`.**
   - `await agent.list_prompts() -> list[MCPPrompt]` — opens one pooled connection per declared MCP server, calls `list_prompts()` on each, merges results with origin server tagged on each `MCPPrompt`. Fetched fresh each call.
   - `await agent.get_prompt(name, arguments, *, server=None) -> list[MCPPromptMessage]` — finds the prompt by name across declared servers (or just on `server` if provided), calls FastMCP `get_prompt(name, arguments)`, maps `GetPromptResult` to a list of `MCPPromptMessage(role, content)`.
   - New types in `ai_core/mcp/prompts.py`: `MCPPrompt(name, description, arguments, mcp_server_spec)`, `MCPPromptArgument(name, description, required)`, `MCPPromptMessage(role, content)` — frozen dataclasses.

3. **`unwrap_mcp_tool_message`** in `ai_core/mcp/tools.py`. Single-line semantics: parse JSON, if dict has exactly one key `"value"` return that value, else return the parsed JSON or raw string unchanged. Documented for use against a `ToolMessage.content` string.

### Module layout

```
src/ai_core/mcp/
├── tools.py           # add MCPResourceSpec subclass + unwrap_mcp_tool_message helper
├── resolver.py        # extend with resolve_mcp_resources(); update internals
├── prompts.py         # NEW: MCPPrompt, MCPPromptArgument, MCPPromptMessage types
├── __init__.py        # export new public types

src/ai_core/agents/
└── base.py            # _all_tools() merges tool + resource specs; new list_prompts() + get_prompt() async methods

examples/mcp_server_demo/
├── server.py          # extend: add a `documentation` resource and a `summarize_text` prompt
├── agent_demo.py      # extend: show resource being read by the LLM and unwrap helper in action
├── prompt_demo.py     # NEW: shows application-invoked prompt → message splice → ainvoke
└── README.md          # document the resource + prompt demos

tests/unit/mcp/
├── test_tools.py      # extend: MCPResourceSpec + unwrap helper
├── test_resolver.py   # extend: resolve_mcp_resources + conflict cases
└── test_prompts.py    # NEW: list_prompts, get_prompt with mocked factory

tests/unit/agents/
└── test_base_mcp.py   # extend: list_prompts/get_prompt unit tests; conflict cases that include resources

tests/integration/mcp/
├── test_agent_uses_mcp.py  # extend: agent reads a resource end-to-end
└── test_prompts.py    # NEW: real prompt fetch end-to-end

scripts/run_examples.sh  # add prompt_demo
```

### Out of scope (explicit deferrals)

- **Resource templates** (`list_resource_templates()` → parameterized URIs). Phase 13 candidate.
- **Hot-reload via `notifications/list_changed`.** Phase 13 candidate (the architecturally-heaviest piece).
- **Binary resources.** v1 only handles `TextResourceContents`; `BlobResourceContents` get a placeholder string `<binary content suppressed: N block(s)>` and a logged warning.
- **LLM-callable prompts.** Application-invoked only.
- **Strict client-side validation of prompt arguments.** v1 takes `arguments: dict[str, Any]` — author is responsible for matching the prompt's declared arguments. Pydantic validation of prompt args is a Phase 13+ enhancement.
- **Audit / OPA on prompt fetches.** `list_prompts` and `get_prompt` are application-invoked; trust is the application's responsibility. ToolInvoker's audit and OPA pipelines apply only to resources (which go through the tool path).

### Naming & namespaces

- Resources are exposed using the resource's `name` field directly (no `read_` prefix). Same flat-namespace + compile-time conflict-error as Phase 11 tools. Conflicts: local `@tool` named `docs` AND resource named `docs` → `RegistryError`. Two MCP servers' resources sharing a name → same.
- Resource `opa_path` propagates from `MCPServerSpec.opa_decision_path` (the field added in Phase 11) — same trust-per-server model.
- Prompt names share the same flat namespace across servers; cross-server name conflict raises `RegistryError` at `list_prompts()` time.

---

## 2. Components & data flow

### Component 1 — `MCPResourceSpec` (extends `src/ai_core/mcp/tools.py`)

Frozen subclass of `MCPToolSpec`. Adds one field; overrides `openai_schema()` to return zero-parameter shape.

```python
@dataclass(frozen=True, slots=True)
class MCPResourceSpec(MCPToolSpec):
    """An MCP resource exposed as a parameter-less read-only tool.

    The handler closure (built by the resolver) hardcodes the resource URI
    and dispatches via `client.read_resource(uri)`. Result mapping mirrors
    Phase 11's tool-call mapping: TextResourceContents are concatenated;
    BlobResourceContents are replaced with a placeholder string and logged.
    """
    mcp_resource_uri: str

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }
```

The handler closure (built in resolver):

```python
async def _handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
    async with factory.open(server) as client:
        contents = await client.read_resource(uri)
    text_parts: list[str] = []
    binary_count = 0
    for c in contents:
        if getattr(c, "text", None) is not None:
            text_parts.append(c.text)
        else:
            binary_count += 1
    if binary_count:
        _logger.warning(
            "mcp.resource.binary_suppressed",
            uri=uri, server=component_id, count=binary_count,
        )
        text_parts.append(f"<binary content suppressed: {binary_count} block(s)>")
    return _MCPPassthroughOutput(value="\n".join(text_parts))
```

### Component 2 — `unwrap_mcp_tool_message` helper (in `src/ai_core/mcp/tools.py`)

Free function. Closes the `{"value": ...}` pedagogical wart:

```python
def unwrap_mcp_tool_message(content: str) -> Any:
    """Unwrap MCPToolSpec's {"value": ...} envelope from a ToolMessage.content string.

    Returns the inner value when content is a JSON object with exactly one key
    "value" (the standard MCPToolSpec/MCPResourceSpec envelope). Otherwise
    returns the parsed JSON, or the raw string if it's not JSON at all.

    Use this in the agent's tool-message reading code to avoid re-implementing
    the unwrap pattern that examples/mcp_server_demo/agent_demo.py inlines.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, dict) and set(parsed.keys()) == {"value"}:
        return parsed["value"]
    return parsed
```

### Component 3 — Resolver extensions (`src/ai_core/mcp/resolver.py`)

Add `resolve_mcp_resources` parallel to `resolve_mcp_tools`. BaseAgent calls both; they share the connection pool (the second call hits the warm connection):

```python
async def resolve_mcp_resources(
    servers: Sequence[MCPServerSpec],
    factory: IMCPConnectionFactory,
) -> list[MCPResourceSpec]:
    """Discover resources on each server and return them as MCPResourceSpec instances.

    Each resource becomes a parameter-less read-only tool. Same conflict-detection
    semantics as resolve_mcp_tools — duplicate names across servers raise RegistryError.

    Servers that don't expose resources (list_resources returns "method not found"
    error) are silently skipped.
    """
    seen_names: set[str] = set()
    out: list[MCPResourceSpec] = []
    for spec in servers:
        async with factory.open(spec) as client:
            try:
                resources = await client.list_resources()
            except McpError as exc:
                if _is_method_not_found(exc):
                    continue  # server doesn't expose resources
                raise
        for fastmcp_resource in resources:
            name = fastmcp_resource.name
            if name in seen_names:
                raise RegistryError(
                    f"MCP resource name {name!r} appears in multiple servers",
                    details={"name": name, "server": spec.component_id},
                )
            seen_names.add(name)
            out.append(_build_mcp_resource_spec(spec, fastmcp_resource, factory))
    return out
```

`_build_mcp_resource_spec` mirrors `_build_mcp_tool_spec` from Phase 11: builds the closure handler, returns `MCPResourceSpec(...)` with `name=fastmcp_resource.name`, `description=fastmcp_resource.description or ""`, `mcp_resource_uri=str(fastmcp_resource.uri)`, `opa_path=server.opa_decision_path`.

`_is_method_not_found(exc)` is a small predicate the implementer defines after probing FastMCP's actual exception shape:
- If FastMCP/MCP raises a typed `MethodNotFoundError` subclass, use `isinstance(exc, MethodNotFoundError)`.
- Else check the JSON-RPC code if exposed (`exc.error.code == -32601` or similar).
- Last resort: case-insensitive string match against `"method not found"` in `str(exc)`.

The predicate lives in `resolver.py` (used by both `resolve_mcp_resources` and the prompt API in `base.py`). Centralizing it here means a single line changes if FastMCP's exception shape evolves.

### Component 4 — Prompt types (new file `src/ai_core/mcp/prompts.py`)

```python
@dataclass(frozen=True, slots=True)
class MCPPromptArgument:
    """One argument declaration on an MCP prompt template."""
    name: str
    description: str | None
    required: bool

@dataclass(frozen=True, slots=True)
class MCPPrompt:
    """A prompt template advertised by an MCP server."""
    name: str
    description: str | None
    arguments: tuple[MCPPromptArgument, ...]
    mcp_server_spec: MCPServerSpec  # which server this prompt came from

@dataclass(frozen=True, slots=True)
class MCPPromptMessage:
    """One message from a fetched prompt template (after argument substitution)."""
    role: str  # "user" or "assistant"
    content: str  # text only in v1; binary content blocks suppressed

__all__ = ["MCPPrompt", "MCPPromptArgument", "MCPPromptMessage"]
```

### Component 5 — `BaseAgent` prompt API (`src/ai_core/agents/base.py`)

Two new async methods. Neither uses the lazy lock — prompt invocations are application-driven.

```python
async def list_prompts(self) -> list[MCPPrompt]:
    """List all prompts across declared MCP servers.

    Fetched fresh each call (no cache).

    Raises:
        RegistryError: When two servers expose prompts with the same name.
        MCPTransportError: Propagated from the connection factory.
    """
    out: list[MCPPrompt] = []
    seen_names: set[str] = set()
    for server in self.mcp_servers():
        async with self._mcp_factory.open(server) as client:
            try:
                prompts = await client.list_prompts()
            except McpError as exc:
                if _is_method_not_found(exc):
                    continue  # server doesn't expose prompts
                raise
        for p in prompts:
            if p.name in seen_names:
                raise RegistryError(
                    f"MCP prompt name {p.name!r} appears in multiple servers",
                    details={"name": p.name, "server": server.component_id},
                )
            seen_names.add(p.name)
            out.append(_to_mcp_prompt(p, server))
    return out

async def get_prompt(
    self,
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    server: str | None = None,
) -> list[MCPPromptMessage]:
    """Fetch a templated prompt's messages by name.

    Args:
        name: The prompt's name (must match what `list_prompts()` returned).
        arguments: Argument dict to substitute into the template.
        server: Optional component_id of the server known to host the prompt.
            When omitted, the agent searches across all declared servers
            (one list_prompts call each until found) — N round-trips worst case.

    Returns:
        List of MCPPromptMessage instances, ready to splice into ainvoke(messages=...).

    Raises:
        RegistryError: When the prompt is not found on any declared server.
    """
    for srv in self.mcp_servers():
        if server is not None and srv.component_id != server:
            continue
        async with self._mcp_factory.open(srv) as client:
            try:
                prompts = await client.list_prompts()
            except McpError:
                continue
            if not any(p.name == name for p in prompts):
                continue
            result = await client.get_prompt(name, dict(arguments or {}))
        return [_to_mcp_prompt_message(m) for m in result.messages]
    raise RegistryError(
        f"MCP prompt {name!r} not found in any declared server",
        details={"name": name, "hint_server": server},
    )
```

### Component 6 — `_all_tools()` extension (`src/ai_core/agents/base.py`)

The existing `_all_tools()` body is updated to call both resolvers. Conflict check operates on the merged name set (local tools + MCP tools + MCP resources):

```python
async def _all_tools(self) -> list[Tool | Mapping[str, Any]]:
    if self._mcp_resolved is None:
        async with self._mcp_resolution_lock:
            if self._mcp_resolved is None:
                servers = list(self.mcp_servers())
                if servers:
                    tools_resolved = await resolve_mcp_tools(servers, self._mcp_factory)
                    resources_resolved = await resolve_mcp_resources(servers, self._mcp_factory)
                    resolved: list[MCPToolSpec] = list(tools_resolved) + list(resources_resolved)
                else:
                    resolved = []
                local_names = {
                    t.name for t in self.tools() if isinstance(t, ToolSpec)
                }
                mcp_names_seen: set[str] = set()
                for mcp_spec in resolved:
                    if mcp_spec.name in local_names:
                        kind = "resource" if isinstance(mcp_spec, MCPResourceSpec) else "tool"
                        raise RegistryError(
                            f"MCP {kind} name {mcp_spec.name!r} conflicts with a local tool",
                            details={"tool": mcp_spec.name},
                        )
                    if mcp_spec.name in mcp_names_seen:
                        raise RegistryError(
                            f"MCP name {mcp_spec.name!r} appears in both tools and resources",
                            details={"name": mcp_spec.name},
                        )
                    mcp_names_seen.add(mcp_spec.name)
                    self._tool_invoker.register(mcp_spec)
                self._mcp_resolved = resolved
    return list(self.tools()) + list(self._mcp_resolved)
```

### Component 7 — Examples

**`examples/mcp_server_demo/server.py`** gets two additions:

```python
@mcp.resource("mcp-demo://documentation")
def documentation() -> str:
    """Project documentation — exposed as an MCP resource."""
    return (
        "This is the demo MCP server's documentation.\n\n"
        "It exposes:\n"
        "  - echo(text): repeat a string\n"
        "  - current_time(): UTC ISO-8601 timestamp\n"
        "  - documentation: this resource\n"
        "  - summarize_text(text): a prompt template"
    )

@mcp.prompt()
def summarize_text(text: str) -> str:
    """Generate a summarization prompt for a given text."""
    return f"Please summarize the following text in one sentence:\n\n{text}"
```

**`examples/mcp_server_demo/agent_demo.py`** updated: ScriptedLLM emits a tool_call for the `documentation` resource (no args); the demo prints the unwrapped result via the new `unwrap_mcp_tool_message` helper.

**`examples/mcp_server_demo/prompt_demo.py`** (new): application-side script. Lists prompts, fetches `summarize_text` with sample text, splices messages into `agent.ainvoke(messages=...)`, prints the agent's response. Demonstrates the application-invoked pattern.

### Data flow

**Resource read (LLM-driven):**
```
1. Agent.ainvoke() — first turn
2. _all_tools() resolves: list_tools() + list_resources() per server
3. Each resource → MCPResourceSpec with hardcoded URI in handler closure
4. LLM sees `documentation` as a tool with no parameters
5. LLM emits tool_call(name="documentation", args={})
6. _tool_node looks up MCPResourceSpec, calls ToolInvoker.invoke
7. ToolInvoker → handler → factory.open() → client.read_resource(uri) → text
8. _MCPPassthroughOutput(value=text) → ToolMessage with {"value": "..."}
9. Author code may call unwrap_mcp_tool_message(msg.content) to display cleanly
```

**Prompt fetch (application-driven):**
```
1. Application: prompts = await agent.list_prompts() → list[MCPPrompt]
2. App lets user pick "summarize_text" with text="..."
3. App: messages = await agent.get_prompt("summarize_text", {"text": "..."})
4. App: final = await agent.ainvoke(messages=messages, tenant_id=...)
5. Agent runs normally with the templated messages prepended
```

---

## 3. Error handling, testing, constraints

### Error handling

- **Server doesn't expose resources or prompts.** Many MCP servers are tool-only. `list_resources()` / `list_prompts()` on such servers raises an `McpError`-shaped "method not found" error. Both `resolve_mcp_resources` and `list_prompts` swallow this specific case (method-not-found → empty result for that server) and log a debug-level message. Other failures (transport errors, protocol parse errors) propagate. The implementer must verify FastMCP's exception shape during coding — string-match against the specific JSON-RPC code or a typed predicate, NOT a broad `except Exception` swallow.

- **Binary resources.** `read_resource()` returns `list[TextResourceContents | BlobResourceContents]`. v1 only handles text. Each binary block is replaced with a placeholder string `<binary content suppressed: N block(s)>` and a warning is logged with `uri`, `server`, `count`. Authors who need binary handling can write a custom tool until a future phase adds first-class support.

- **Resource read fails mid-call.** If `client.read_resource(uri)` itself fails (e.g., file not found on the server side), the error propagates out of the handler and `ToolInvoker` wraps it in `ToolExecutionError` (same path as a tool error in Phase 11). The error message identifies the resource by URI and server.

- **Name conflicts.** The conflict check runs at first-turn resolution and at `list_prompts` time:
  - Local `@tool` vs MCP tool → existing Phase 11 `RegistryError`.
  - Local `@tool` vs MCP resource → new check, same `RegistryError` shape.
  - MCP tool vs MCP resource (cross-server or same-server) → `RegistryError` at `_all_tools()` time.
  - MCP prompt name conflict across servers → `RegistryError` at `list_prompts()` time.
  - Conflict messages name the offending server's `component_id`.

- **`get_prompt(name)` not found.** Searched across all declared servers; if no server has a prompt by that name, `RegistryError` with `details={"name": name, "hint_server": server}`. The error message suggests calling `list_prompts()` to enumerate.

- **`get_prompt` with malformed arguments.** Propagated from FastMCP. Server-side validation surfaces as the server's error message; the handler doesn't pre-validate.

- **Concurrent first-turn races.** Existing `asyncio.Lock` from Phase 11 still serializes the resolution. The lock now covers BOTH `resolve_mcp_tools` and `resolve_mcp_resources` calls (sequential within the lock body, not parallel — adding parallelism would require a different cache structure and is YAGNI for v1).

- **`list_prompts` / `get_prompt` are NOT audited.** Different from tool/resource invocations which go through `ToolInvoker`'s audit pipeline. Prompts are application-invoked discovery/fetch — applications wrap the call themselves if audit is needed.

- **`list_prompts` / `get_prompt` do NOT check OPA.** Same reasoning. The application controls when prompts are fetched; trust is the application's responsibility.

### Testing strategy

- **Unit tests** (`tests/unit/mcp/test_tools.py`, `test_resolver.py`, `test_prompts.py`, `tests/unit/agents/test_base_mcp.py`):
  - `MCPResourceSpec` construction; `openai_schema()` returns parameter-less shape; subclass-of-`MCPToolSpec` invariant.
  - `unwrap_mcp_tool_message`: dict with `{"value": ...}` → unwraps; dict with multiple keys → returns parsed dict; non-JSON string → returns raw; non-string types raise `TypeError`.
  - `resolve_mcp_resources` with mocked factory: discovers resources, builds `MCPResourceSpec`s with correct URIs; conflict detection across servers; "method not found" → silent skip.
  - Handler closure: text content concatenation; binary content placeholder + warning; multi-content mix.
  - `list_prompts` with mocked factory: discovers prompts, populates `MCPPrompt` from FastMCP types; cross-server name conflict raises.
  - `get_prompt(name, args)` with mocked factory: finds prompt across servers; explicit `server=` skips search; not-found raises with hint.
  - `_all_tools()` extended: merges tools and resources; conflict detection covers the merged name set; resources registered with `ToolInvoker`.

- **Integration tests** (`tests/integration/mcp/test_agent_uses_mcp.py` extension; new `test_prompts.py`):
  - Agent reads `documentation` resource end-to-end via real FastMCP subprocess; `ToolMessage.content` contains the documentation text wrapped in `{"value": ...}`; `unwrap_mcp_tool_message` returns the unwrapped string.
  - `prompt_demo.py` integration test: app calls `list_prompts()`, picks `summarize_text`, calls `get_prompt(...)`, splices messages into `ainvoke()`, asserts agent runs successfully.

- **Backward-compat verification.** All Phase 11 tests must pass without modification. `BaseAgent.mcp_servers()` returning empty still produces zero MCP-served entries in `_all_tools()`. `resolve_mcp_tools` continues to behave exactly as in Phase 11; the new `resolve_mcp_resources` is parallel.

- **Smoke gate.** `examples/mcp_server_demo/prompt_demo.py` becomes a new entry in `scripts/run_examples.sh`. Skips on the same `fastmcp not installed` condition as `mcp_server_demo` and `mcp_agent_demo`.

### Constraints

- **No new runtime dependencies.** FastMCP already provides `list_resources`, `read_resource`, `list_resource_templates`, `list_prompts`, `get_prompt` on `Client`. Pydantic v2 already imported. No new deps needed.

- **Same connection model.** Per-call pooled connections via `PoolingMCPConnectionFactory`. No persistent subscriptions. Hot-reload deferred to Phase 13.

- **Phase 11 surface unchanged.** `MCPToolSpec`, `resolve_mcp_tools`, `_all_tools()` keep their existing semantics. `MCPResourceSpec` is a subclass — additive, not breaking. `_all_tools()` body is extended to call both resolvers, but the public signature is unchanged.

- **Backward compat for `BaseAgent.__init__`.** No new injected dependencies. `IMCPConnectionFactory` is already injected; the new prompt methods just use the same factory.

- **No new exceptions.** `RegistryError`, `MCPTransportError`, `ToolExecutionError` cover all error paths.

- **`MCPResourceSpec.mcp_resource_uri: str` (not `AnyUrl`).** FastMCP returns `pydantic.AnyUrl` for resource URIs; we store as plain `str` to keep `MCPResourceSpec` Pydantic-free at the field-type level (matching `MCPToolSpec`'s style). The handler does `str(uri)` on the way in.

- **Prompt naming follows the same flat-namespace + cross-server-conflict rule as tools and resources.** Future phases that need finer granularity (e.g., prompt categories, prefixed names) can layer on top.

- **`McpError` import path is FastMCP-version-dependent.** The implementer must verify (likely `from fastmcp.exceptions import McpError` or `from mcp.shared.exceptions import McpError`) and use the import that matches the version pinned in `pyproject.toml`. If the predicate for "method not found" needs adjustment based on actual error shape, that's a coding-time finding.
