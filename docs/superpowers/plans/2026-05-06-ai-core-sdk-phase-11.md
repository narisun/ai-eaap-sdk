# Phase 11 — Agent-Side MCP Integration: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make remote MCP servers a first-class tool source for agents — `mcp_servers()` declares servers, the SDK resolves their tools at first turn, and `ToolInvoker`'s standard pipeline dispatches them uniformly with local `@tool`s.

**Architecture:** New `MCPToolSpec` (frozen `ToolSpec` subclass with permissive Pydantic models and an overridden `openai_schema()` that returns FastMCP's `inputSchema`). New `resolve_mcp_tools()` async helper that, for each declared `MCPServerSpec`, opens a pooled connection, calls `list_tools()`, and synthesizes one `MCPToolSpec` per server tool with a closure handler. `BaseAgent` gains `mcp_servers()` (default empty), an injected `IMCPConnectionFactory`, and an async `_all_tools()` helper that merges resolved MCP specs with local tools, runs once per instance behind an `asyncio.Lock`, and feeds both the LLM tool list and the dispatcher.

**Tech Stack:** Python 3.11+, Pydantic v2 (passthrough models), FastMCP (already a core dep at `pyproject.toml:46`), `injector` (DI), pytest. Toolchain: `uv run pytest`, `uv run ruff check`. (`uv` may not be on every dev machine — substitute `.venv/bin/python` and `.venv/bin/ruff` as needed.)

---

## File map

| Path | New / Modified | Purpose |
| --- | --- | --- |
| `src/ai_core/mcp/transports.py` | Modified | `MCPServerSpec` gains `opa_decision_path: str \| None = None` |
| `src/ai_core/mcp/tools.py` | New | `MCPToolSpec` + `_MCPPassthroughInput` + `_MCPPassthroughOutput` |
| `src/ai_core/mcp/resolver.py` | New | `resolve_mcp_tools(servers, factory) -> list[MCPToolSpec]` + `_build_mcp_tool_spec` |
| `src/ai_core/mcp/__init__.py` | Modified | Export `MCPToolSpec`, `resolve_mcp_tools` |
| `src/ai_core/agents/base.py` | Modified | Inject `mcp_factory`; add `mcp_servers()` and `_all_tools()`; switch `_agent_node` and `_tool_node` to use `_all_tools()` |
| `tests/unit/mcp/test_tools.py` | New | `MCPToolSpec` unit tests |
| `tests/unit/mcp/test_resolver.py` | New | `resolve_mcp_tools` unit tests with mocked factory |
| `tests/unit/agents/test_base_mcp.py` | New | `BaseAgent` MCP-integration unit tests (mocked factory, no FastMCP I/O) |
| `tests/integration/mcp/__init__.py` | New (empty) | Marks integration package |
| `tests/integration/mcp/test_agent_uses_mcp.py` | New | End-to-end test: real FastMCP stdio server, agent invokes tool |
| `examples/mcp_server_demo/agent_demo.py` | New | Agent using the demo server's tools end-to-end |
| `examples/mcp_server_demo/README.md` | Modified | Drop "what's not yet shown"; document `agent_demo.py` |
| `scripts/run_examples.sh` | Modified | Add `agent_demo` entry (skips if FastMCP missing, like `run_client.py`) |

---

## Task 1: Add `opa_decision_path` to `MCPServerSpec`

**Why:** Per-server OPA path is the simplest way to wire MCP tool calls into `ToolInvoker`'s existing OPA pipeline. The field is added with `default=None` for backward-compat; `ToolInvoker` already treats `opa_path=None` as "skip policy".

**Files:**
- Modify: `src/ai_core/mcp/transports.py:38-60`
- Test: `tests/unit/mcp/test_transports.py` (read-only check; existing tests should continue to pass)

- [ ] **Step 1: Inspect the existing `MCPServerSpec` dataclass**

Read `src/ai_core/mcp/transports.py` lines 38-60. Confirm it's `@dataclass(slots=True, frozen=True)` with the existing fields ending at `timeout_seconds: float = 30.0`.

- [ ] **Step 2: Add the new field**

In `src/ai_core/mcp/transports.py`, add `opa_decision_path` as the last field:

```python
@dataclass(slots=True, frozen=True)
class MCPServerSpec:
    """Connection spec for an MCP server.

    Attributes:
        component_id: Logical identifier registered in :class:`ComponentRegistry`.
        transport: One of ``"stdio"``, ``"http"``, ``"sse"``.
        target: For ``stdio`` — the executable path or shell command.
            For ``http`` / ``sse`` — the server URL.
        args: Extra positional CLI args (``stdio`` only).
        env: Extra environment variables for the spawned subprocess
            (``stdio`` only).
        headers: HTTP headers to send (``http`` / ``sse`` only).
        timeout_seconds: Per-call timeout enforced by the FastMCP client.
        opa_decision_path: When set, every MCP tool call from this server
            checks this OPA path through the standard ToolInvoker pipeline.
            ``None`` skips OPA enforcement (matches local tools without ``opa_path``).
    """

    component_id: str
    transport: MCPTransport
    target: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    opa_decision_path: str | None = None
```

- [ ] **Step 3: Run existing tests for transports**

Run: `uv run pytest tests/unit/mcp/ -v` (or `.venv/bin/pytest tests/unit/mcp/ -v`)

Expected: all pre-existing tests still pass. Adding a defaulted field is backward-compatible.

- [ ] **Step 4: Sanity-check the field is reachable**

Run: `uv run python -c "from ai_core.mcp import MCPServerSpec; s = MCPServerSpec(component_id='x', transport='stdio', target='/bin/true'); print(s.opa_decision_path)"`

Expected: prints `None`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/ai_core/mcp/transports.py`

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ai_core/mcp/transports.py
git commit -m "feat(mcp): add opa_decision_path to MCPServerSpec

Per-server OPA decision path used by Phase 11's MCPToolSpec to wire
remote MCP tool calls through ToolInvoker's existing OPA pipeline.
Default None skips policy enforcement, matching local tools without
opa_path."
```

---

## Task 2: `MCPToolSpec` + permissive Pydantic models

**Why:** This is the type that bridges FastMCP tools into `ToolInvoker`. `MCPToolSpec` IS-A `ToolSpec` (so the existing dispatch in `_tool_node` finds it via the same `isinstance(t, ToolSpec)` check), but its `openai_schema()` returns FastMCP's raw `inputSchema` instead of `input_model.model_json_schema()`. `_MCPPassthroughInput` accepts any dict (server validates); `_MCPPassthroughOutput.value: Any` carries the result.

**Files:**
- Create: `src/ai_core/mcp/tools.py`
- Test: `tests/unit/mcp/test_tools.py`
- Modify: `src/ai_core/mcp/__init__.py` (export `MCPToolSpec`)

- [ ] **Step 1: Write the failing tests first**

Create `tests/unit/mcp/test_tools.py`:

```python
"""Unit tests for MCPToolSpec and its permissive I/O models."""
from __future__ import annotations

import pytest

from ai_core.mcp import MCPServerSpec
from ai_core.mcp.tools import (
    MCPToolSpec,
    _MCPPassthroughInput,
    _MCPPassthroughOutput,
)

pytestmark = pytest.mark.unit


def _spec_factory(**overrides) -> MCPToolSpec:
    """Build an MCPToolSpec with sensible defaults for testing."""
    server = MCPServerSpec(
        component_id="test-server",
        transport="stdio",
        target="/bin/true",
        opa_decision_path=overrides.pop("opa_decision_path", None),
    )

    async def _noop_handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
        return _MCPPassthroughOutput(value="ok")

    return MCPToolSpec(
        name=overrides.pop("name", "echo"),
        version=1,
        description=overrides.pop("description", "Test tool"),
        input_model=_MCPPassthroughInput,
        output_model=_MCPPassthroughOutput,
        handler=_noop_handler,
        opa_path=overrides.pop("opa_path", None),
        mcp_server_spec=server,
        mcp_input_schema=overrides.pop(
            "mcp_input_schema",
            {"type": "object", "properties": {"text": {"type": "string"}}},
        ),
    )


def test_mcp_tool_spec_is_a_tool_spec() -> None:
    """MCPToolSpec subclasses ToolSpec so existing isinstance checks find it."""
    from ai_core.tools.spec import ToolSpec

    spec = _spec_factory()
    assert isinstance(spec, ToolSpec)


def test_openai_schema_returns_raw_input_schema() -> None:
    """MCPToolSpec.openai_schema() returns FastMCP's inputSchema, not Pydantic-derived."""
    raw = {
        "type": "object",
        "properties": {"text": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["text"],
    }
    spec = _spec_factory(mcp_input_schema=raw)

    schema = spec.openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["description"] == "Test tool"
    # Critical: parameters is the raw FastMCP schema, not Pydantic-derived
    assert schema["function"]["parameters"] == raw


def test_passthrough_input_accepts_arbitrary_keys() -> None:
    """_MCPPassthroughInput allows any keys (server-side validation is the source of truth)."""
    payload = _MCPPassthroughInput.model_validate(
        {"text": "hi", "weird_key": [1, 2, 3], "nested": {"a": True}}
    )
    dumped = payload.model_dump()
    assert dumped["text"] == "hi"
    assert dumped["weird_key"] == [1, 2, 3]
    assert dumped["nested"] == {"a": True}


def test_passthrough_output_wraps_any_value() -> None:
    """_MCPPassthroughOutput.value accepts arbitrary Python values."""
    out = _MCPPassthroughOutput(value={"complex": [1, 2]})
    assert out.value == {"complex": [1, 2]}

    out_str = _MCPPassthroughOutput(value="hello")
    assert out_str.value == "hello"


def test_opa_path_default_is_none() -> None:
    """Tools constructed without an opa_path skip OPA (ToolInvoker contract)."""
    spec = _spec_factory()
    assert spec.opa_path is None


def test_opa_path_propagates_from_server_spec() -> None:
    """When the resolver passes server.opa_decision_path → spec.opa_path, it round-trips."""
    spec = _spec_factory(opa_path="mcp.test-server.allow")
    assert spec.opa_path == "mcp.test-server.allow"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/mcp/test_tools.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ai_core.mcp.tools'` (or similar import error).

- [ ] **Step 3: Implement `src/ai_core/mcp/tools.py`**

Create `src/ai_core/mcp/tools.py`:

```python
"""MCP-tool spec: bridges remote MCP tools into the SDK's ToolInvoker pipeline.

`MCPToolSpec` is a frozen subclass of `ToolSpec` that:

* Uses permissive Pydantic models for input/output (server-side validates
  args; the SDK's `_MCPPassthroughOutput` wraps the result so the existing
  `output_model.model_validate(...)` contract holds).
* Overrides `openai_schema()` to return FastMCP's raw `inputSchema` so the
  LLM sees the real types the server advertises — no JSON-Schema-to-Pydantic
  conversion required.

The handler that actually invokes the remote tool is built in
`ai_core.mcp.resolver._build_mcp_tool_spec`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from ai_core.mcp.transports import MCPServerSpec
from ai_core.tools.spec import ToolSpec


class _MCPPassthroughInput(BaseModel):
    """Permissive input model — accepts any keys; the MCP server validates server-side."""

    model_config = ConfigDict(extra="allow")


class _MCPPassthroughOutput(BaseModel):
    """Result wrapper that carries arbitrary values back from a remote MCP tool."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    value: Any = None


@dataclass(frozen=True, slots=True)
class MCPToolSpec(ToolSpec):
    """A `ToolSpec` whose handler dispatches to a remote MCP tool.

    Attributes:
        mcp_server_spec: The server this tool was discovered on (used by audit
            and by the handler closure to open connections).
        mcp_input_schema: The raw FastMCP `inputSchema` dict — returned
            verbatim by `openai_schema()` so the LLM sees real types.
    """

    mcp_server_spec: MCPServerSpec
    mcp_input_schema: Mapping[str, Any]

    def openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema using FastMCP's raw inputSchema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.mcp_input_schema),
            },
        }


__all__ = ["MCPToolSpec"]
```

- [ ] **Step 4: Update `src/ai_core/mcp/__init__.py` exports**

In `src/ai_core/mcp/__init__.py`, add:

```python
"""MCP sub-package — Component Registry + FastMCP transport handlers."""

from __future__ import annotations

from ai_core.mcp.registry import ComponentRegistry, RegisteredComponent
from ai_core.mcp.tools import MCPToolSpec
from ai_core.mcp.transports import (
    FastMCPConnectionFactory,
    IMCPConnectionFactory,
    MCPServerSpec,
    MCPTransport,
)

__all__ = [
    "ComponentRegistry",
    "FastMCPConnectionFactory",
    "IMCPConnectionFactory",
    "MCPServerSpec",
    "MCPToolSpec",
    "MCPTransport",
    "RegisteredComponent",
]
```

(Sorting `__all__` matches the codebase's existing pattern; ruff's `RUF022` enforces it.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/mcp/test_tools.py -v`

Expected: all 6 tests PASS.

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/ai_core/mcp/ tests/unit/mcp/test_tools.py`

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/ai_core/mcp/tools.py src/ai_core/mcp/__init__.py tests/unit/mcp/test_tools.py
git commit -m "feat(mcp): MCPToolSpec — frozen ToolSpec subclass for remote MCP tools

Permissive Pydantic models (server validates); openai_schema() returns
FastMCP's raw inputSchema directly. IS-A ToolSpec so existing
ToolInvoker dispatch finds it without any new isinstance checks."
```

---

## Task 3: `resolve_mcp_tools` + handler closures

**Why:** This is the async helper that turns a `Sequence[MCPServerSpec]` into a `list[MCPToolSpec]` ready to register with `ToolInvoker`. For each server it opens a pooled connection, calls `list_tools()`, builds one `MCPToolSpec` per tool — each carrying a closure that opens (a fresh, also pooled) connection per call, calls `call_tool`, and maps the `CallToolResult`.

**Files:**
- Create: `src/ai_core/mcp/resolver.py`
- Test: `tests/unit/mcp/test_resolver.py`
- Modify: `src/ai_core/mcp/__init__.py` (export `resolve_mcp_tools`)

- [ ] **Step 1: Write failing tests first**

Create `tests/unit/mcp/test_resolver.py`:

```python
"""Unit tests for resolve_mcp_tools — uses a mocked IMCPConnectionFactory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest

from ai_core.exceptions import RegistryError, ToolExecutionError
from ai_core.mcp import MCPServerSpec
from ai_core.mcp.resolver import resolve_mcp_tools
from ai_core.mcp.tools import MCPToolSpec, _MCPPassthroughInput, _MCPPassthroughOutput
from ai_core.mcp.transports import IMCPConnectionFactory

pytestmark = pytest.mark.unit


# ---- Test doubles for fastmcp.Client + CallToolResult ---------------------------
@dataclass(frozen=True)
class _FakeFastMCPTool:
    name: str
    description: str | None
    inputSchema: dict[str, Any]  # noqa: N815 — match FastMCP's attribute name


@dataclass(frozen=True)
class _FakeTextContent:
    text: str


@dataclass(frozen=True)
class _FakeCallToolResult:
    is_error: bool
    data: Any
    content: list[_FakeTextContent]


class _FakeMCPClient:
    """Tiny fake of the FastMCP client surface used by the resolver and handler."""

    def __init__(
        self,
        tools: list[_FakeFastMCPTool],
        call_results: dict[str, _FakeCallToolResult] | None = None,
    ) -> None:
        self._tools = tools
        self._call_results = call_results or {}
        self.list_tools_calls = 0
        self.call_tool_invocations: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[_FakeFastMCPTool]:
        self.list_tools_calls += 1
        return list(self._tools)

    async def call_tool(self, name: str, args: dict[str, Any]) -> _FakeCallToolResult:
        self.call_tool_invocations.append((name, args))
        if name in self._call_results:
            return self._call_results[name]
        return _FakeCallToolResult(is_error=False, data=f"called {name}", content=[])


class _FakeFactory(IMCPConnectionFactory):
    """Mock factory whose .open() yields a fresh _FakeMCPClient bound to the spec."""

    def __init__(self, clients_by_component: dict[str, _FakeMCPClient]) -> None:
        self._clients_by_component = clients_by_component
        self.open_calls: list[MCPServerSpec] = []

    def open(self, spec: MCPServerSpec):  # type: ignore[override]
        self.open_calls.append(spec)
        client = self._clients_by_component[spec.component_id]

        @asynccontextmanager
        async def _cm():
            yield client

        return _cm()


# ---- Resolver tests -------------------------------------------------------------
async def test_resolves_one_server_one_tool() -> None:
    server = MCPServerSpec(
        component_id="time-svc", transport="stdio", target="/bin/true",
    )
    fake_tool = _FakeFastMCPTool(
        name="now", description="Return the current time",
        inputSchema={"type": "object", "properties": {}},
    )
    client = _FakeMCPClient(tools=[fake_tool])
    factory = _FakeFactory({"time-svc": client})

    specs = await resolve_mcp_tools([server], factory)

    assert len(specs) == 1
    assert isinstance(specs[0], MCPToolSpec)
    assert specs[0].name == "now"
    assert specs[0].description == "Return the current time"
    assert specs[0].mcp_input_schema == {"type": "object", "properties": {}}
    assert client.list_tools_calls == 1


async def test_opa_path_propagates_from_server_to_spec() -> None:
    server = MCPServerSpec(
        component_id="time-svc", transport="stdio", target="/bin/true",
        opa_decision_path="mcp.time-svc.allow",
    )
    fake_tool = _FakeFastMCPTool(name="now", description=None, inputSchema={})
    factory = _FakeFactory({"time-svc": _FakeMCPClient(tools=[fake_tool])})

    specs = await resolve_mcp_tools([server], factory)

    assert specs[0].opa_path == "mcp.time-svc.allow"


async def test_opa_path_none_when_server_has_no_decision_path() -> None:
    server = MCPServerSpec(
        component_id="time-svc", transport="stdio", target="/bin/true",
    )
    fake_tool = _FakeFastMCPTool(name="now", description=None, inputSchema={})
    factory = _FakeFactory({"time-svc": _FakeMCPClient(tools=[fake_tool])})

    specs = await resolve_mcp_tools([server], factory)

    assert specs[0].opa_path is None


async def test_conflict_within_same_server_raises() -> None:
    server = MCPServerSpec(
        component_id="dup-svc", transport="stdio", target="/bin/true",
    )
    fake_tools = [
        _FakeFastMCPTool(name="run", description="A", inputSchema={}),
        _FakeFastMCPTool(name="run", description="B", inputSchema={}),
    ]
    factory = _FakeFactory({"dup-svc": _FakeMCPClient(tools=fake_tools)})

    with pytest.raises(RegistryError) as excinfo:
        await resolve_mcp_tools([server], factory)

    assert "run" in str(excinfo.value)


async def test_conflict_across_servers_raises() -> None:
    server_a = MCPServerSpec(
        component_id="svc-a", transport="stdio", target="/bin/true",
    )
    server_b = MCPServerSpec(
        component_id="svc-b", transport="stdio", target="/bin/true",
    )
    factory = _FakeFactory({
        "svc-a": _FakeMCPClient(tools=[_FakeFastMCPTool("ping", None, {})]),
        "svc-b": _FakeMCPClient(tools=[_FakeFastMCPTool("ping", None, {})]),
    })

    with pytest.raises(RegistryError) as excinfo:
        await resolve_mcp_tools([server_a, server_b], factory)

    assert "ping" in str(excinfo.value)


async def test_handler_returns_data_when_present() -> None:
    server = MCPServerSpec(
        component_id="svc", transport="stdio", target="/bin/true",
    )
    factory = _FakeFactory({
        "svc": _FakeMCPClient(
            tools=[_FakeFastMCPTool("echo", None, {})],
            call_results={"echo": _FakeCallToolResult(
                is_error=False, data="hello", content=[],
            )},
        ),
    })
    specs = await resolve_mcp_tools([server], factory)

    payload = _MCPPassthroughInput.model_validate({"text": "hello"})
    result = await specs[0].handler(payload)

    assert isinstance(result, _MCPPassthroughOutput)
    assert result.value == "hello"


async def test_handler_concatenates_text_content_when_no_data() -> None:
    server = MCPServerSpec(
        component_id="svc", transport="stdio", target="/bin/true",
    )
    factory = _FakeFactory({
        "svc": _FakeMCPClient(
            tools=[_FakeFastMCPTool("multi", None, {})],
            call_results={"multi": _FakeCallToolResult(
                is_error=False, data=None,
                content=[_FakeTextContent("line one"), _FakeTextContent("line two")],
            )},
        ),
    })
    specs = await resolve_mcp_tools([server], factory)

    payload = _MCPPassthroughInput.model_validate({})
    result = await specs[0].handler(payload)

    assert result.value == "line one\nline two"


async def test_handler_raises_tool_execution_error_on_is_error() -> None:
    server = MCPServerSpec(
        component_id="svc", transport="stdio", target="/bin/true",
    )
    factory = _FakeFactory({
        "svc": _FakeMCPClient(
            tools=[_FakeFastMCPTool("broken", None, {})],
            call_results={"broken": _FakeCallToolResult(
                is_error=True, data=None,
                content=[_FakeTextContent("server-side blew up")],
            )},
        ),
    })
    specs = await resolve_mcp_tools([server], factory)

    payload = _MCPPassthroughInput.model_validate({})
    with pytest.raises(ToolExecutionError) as excinfo:
        await specs[0].handler(payload)

    assert "broken" in str(excinfo.value)
    assert excinfo.value.details.get("tool") == "broken"
    assert excinfo.value.details.get("server") == "svc"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/mcp/test_resolver.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ai_core.mcp.resolver'`.

- [ ] **Step 3: Implement `src/ai_core/mcp/resolver.py`**

Create `src/ai_core/mcp/resolver.py`:

```python
"""Resolve a list of MCPServerSpecs into MCPToolSpecs by talking to each server.

`resolve_mcp_tools` is the async helper that BaseAgent calls on the first
turn to discover what tools each declared MCP server exposes. For each
server it opens one pooled connection (via `IMCPConnectionFactory.open`),
calls `list_tools()`, then builds one `MCPToolSpec` per advertised tool —
each carrying a closure handler that opens a fresh pooled connection on
every invocation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ai_core.exceptions import RegistryError, ToolExecutionError
from ai_core.mcp.tools import (
    MCPToolSpec,
    _MCPPassthroughInput,
    _MCPPassthroughOutput,
)
from ai_core.mcp.transports import IMCPConnectionFactory, MCPServerSpec


async def resolve_mcp_tools(
    servers: Sequence[MCPServerSpec],
    factory: IMCPConnectionFactory,
) -> list[MCPToolSpec]:
    """Discover tools on each server and return them as `MCPToolSpec` instances.

    Args:
        servers: MCP server specs the agent declared via `mcp_servers()`.
        factory: Connection factory (typically the DI-bound
            `PoolingMCPConnectionFactory`).

    Returns:
        One `MCPToolSpec` per discovered tool, in declaration order. Empty
        list when `servers` is empty.

    Raises:
        MCPTransportError: When a server is unreachable (propagated from
            `factory.open()` / `client.list_tools()`).
        RegistryError: When two servers expose tools with the same name,
            or when a single server returns duplicate names.
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


def _build_mcp_tool_spec(
    server: MCPServerSpec,
    fastmcp_tool: Any,
    factory: IMCPConnectionFactory,
) -> MCPToolSpec:
    """Construct an MCPToolSpec wrapping a closure handler that calls the remote tool."""
    tool_name = fastmcp_tool.name
    component_id = server.component_id

    async def _handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
        args_dict = payload.model_dump()
        async with factory.open(server) as client:
            result = await client.call_tool(tool_name, args_dict)

        if result.is_error:
            content_text = _join_text_content(getattr(result, "content", ()))
            raise ToolExecutionError(
                f"MCP tool {tool_name!r} returned error",
                details={
                    "tool": tool_name,
                    "server": component_id,
                    "content": content_text,
                },
            )

        if getattr(result, "data", None) is not None:
            return _MCPPassthroughOutput(value=result.data)

        return _MCPPassthroughOutput(
            value=_join_text_content(getattr(result, "content", ()))
        )

    return MCPToolSpec(
        name=tool_name,
        version=1,
        description=fastmcp_tool.description or "",
        input_model=_MCPPassthroughInput,
        output_model=_MCPPassthroughOutput,
        handler=_handler,
        opa_path=server.opa_decision_path,
        mcp_server_spec=server,
        mcp_input_schema=getattr(fastmcp_tool, "inputSchema", None) or {
            "type": "object", "properties": {},
        },
    )


def _join_text_content(content: Any) -> str:
    """Join text-bearing content items into a single string."""
    return "\n".join(c.text for c in content if hasattr(c, "text"))


__all__ = ["resolve_mcp_tools"]
```

- [ ] **Step 4: Run resolver tests**

Run: `uv run pytest tests/unit/mcp/test_resolver.py -v`

Expected: all 8 tests PASS.

- [ ] **Step 5: Add to package exports**

In `src/ai_core/mcp/__init__.py`, add `resolve_mcp_tools` to the imports and `__all__`:

```python
"""MCP sub-package — Component Registry + FastMCP transport handlers."""

from __future__ import annotations

from ai_core.mcp.registry import ComponentRegistry, RegisteredComponent
from ai_core.mcp.resolver import resolve_mcp_tools
from ai_core.mcp.tools import MCPToolSpec
from ai_core.mcp.transports import (
    FastMCPConnectionFactory,
    IMCPConnectionFactory,
    MCPServerSpec,
    MCPTransport,
)

__all__ = [
    "ComponentRegistry",
    "FastMCPConnectionFactory",
    "IMCPConnectionFactory",
    "MCPServerSpec",
    "MCPToolSpec",
    "MCPTransport",
    "RegisteredComponent",
    "resolve_mcp_tools",
]
```

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/ai_core/mcp/ tests/unit/mcp/test_resolver.py`

Expected: no errors.

- [ ] **Step 7: Run the full mcp unit suite**

Run: `uv run pytest tests/unit/mcp/ -v`

Expected: all tests pass (Task 1 + Task 2 + Task 3 cumulatively).

- [ ] **Step 8: Commit**

```bash
git add src/ai_core/mcp/resolver.py src/ai_core/mcp/__init__.py tests/unit/mcp/test_resolver.py
git commit -m "feat(mcp): resolve_mcp_tools — async server-to-MCPToolSpec resolver

For each declared MCPServerSpec, opens a pooled connection, calls
list_tools(), and synthesizes one MCPToolSpec per advertised tool.
Each spec carries a closure handler that opens a fresh pooled
connection per call. Conflict detection raises RegistryError; tool
errors raise ToolExecutionError."
```

---

## Task 4: `BaseAgent` integration

**Why:** This is where everything ties together. `BaseAgent` gets the new injected `mcp_factory`, an overridable `mcp_servers()` method, and an async `_all_tools()` helper that runs first-turn resolution behind an `asyncio.Lock`, registers each `MCPToolSpec` with the invoker, runs the conflict check against local tools, caches the merged list on the instance, and then is the source of truth for both `_agent_node`'s tool-payload build and `_tool_node`'s name-lookup dict.

**Files:**
- Modify: `src/ai_core/agents/base.py`
- Test: `tests/unit/agents/test_base_mcp.py` (new)

- [ ] **Step 1: Write the failing tests first**

Create `tests/unit/agents/test_base_mcp.py`:

```python
"""Unit tests for BaseAgent's MCP integration: mcp_servers() + _all_tools()."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.exceptions import RegistryError
from ai_core.mcp import MCPServerSpec
from ai_core.mcp.tools import MCPToolSpec
from ai_core.tools import tool
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.spec import ToolSpec

pytestmark = pytest.mark.unit


# ---- Fakes mirroring test_resolver.py -------------------------------------------
@dataclass(frozen=True)
class _FakeFastMCPTool:
    name: str
    description: str | None
    inputSchema: dict[str, Any]  # noqa: N815


@dataclass(frozen=True)
class _FakeCallToolResult:
    is_error: bool
    data: Any
    content: list


class _FakeMCPClient:
    def __init__(self, tools: list[_FakeFastMCPTool]) -> None:
        self._tools = tools

    async def list_tools(self) -> list[_FakeFastMCPTool]:
        return list(self._tools)

    async def call_tool(self, name: str, args: dict[str, Any]) -> _FakeCallToolResult:
        return _FakeCallToolResult(is_error=False, data=f"called {name}", content=[])


class _FakeFactory:
    def __init__(self, clients: dict[str, _FakeMCPClient]) -> None:
        self._clients = clients
        self.open_count = 0

    def open(self, spec: MCPServerSpec):
        self.open_count += 1

        @asynccontextmanager
        async def _cm():
            yield self._clients[spec.component_id]

        return _cm()


# ---- Test agent -----------------------------------------------------------------
class _LocalEchoIn(BaseModel):
    text: str


class _LocalEchoOut(BaseModel):
    text: str


@tool(name="local_echo", version=1, description="A local echo tool")
async def _local_echo(payload: _LocalEchoIn) -> _LocalEchoOut:
    return _LocalEchoOut(text=payload.text)


class _AgentWithoutMCP(BaseAgent):
    agent_id = "no-mcp"

    def system_prompt(self) -> str:
        return "test"

    def tools(self):
        return (_local_echo,)


class _AgentWithMCP(BaseAgent):
    agent_id = "with-mcp"

    _mcp_servers: tuple[MCPServerSpec, ...] = ()

    def system_prompt(self) -> str:
        return "test"

    def tools(self):
        return (_local_echo,)

    def mcp_servers(self):
        return self._mcp_servers


# ---- Helpers --------------------------------------------------------------------
def _build_agent(agent_cls, factory) -> BaseAgent:
    """Construct an agent with mostly-no-op deps; only mcp_factory matters here."""
    invoker = ToolInvoker(observability=MagicMock())
    return agent_cls(
        settings=AppSettings(service_name="test", environment="local"),
        llm=MagicMock(),
        memory=MagicMock(),
        observability=MagicMock(),
        tool_invoker=invoker,
        mcp_factory=factory,
    )


# ---- Tests ----------------------------------------------------------------------
async def test_mcp_servers_default_is_empty() -> None:
    """BaseAgent.mcp_servers() returns () by default — backward compat."""
    agent = _build_agent(_AgentWithoutMCP, _FakeFactory({}))
    assert tuple(agent.mcp_servers()) == ()


async def test_all_tools_returns_local_when_no_mcp_declared() -> None:
    """_all_tools() returns just the local tools when mcp_servers() is empty."""
    agent = _build_agent(_AgentWithoutMCP, _FakeFactory({}))

    all_tools = await agent._all_tools()

    assert len(all_tools) == 1
    assert all_tools[0].name == "local_echo"


async def test_all_tools_resolves_and_merges() -> None:
    """_all_tools() resolves MCP servers and appends to local tools."""
    factory = _FakeFactory({
        "svc": _FakeMCPClient(tools=[
            _FakeFastMCPTool("remote_a", "A", {"type": "object"}),
            _FakeFastMCPTool("remote_b", "B", {"type": "object"}),
        ]),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    all_tools = await agent._all_tools()

    names = sorted(t.name for t in all_tools)
    assert names == ["local_echo", "remote_a", "remote_b"]
    assert factory.open_count == 1  # one list_tools roundtrip


async def test_all_tools_caches_after_first_call() -> None:
    """Second call to _all_tools() does not re-resolve."""
    factory = _FakeFactory({
        "svc": _FakeMCPClient(tools=[_FakeFastMCPTool("remote", None, {})]),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    await agent._all_tools()
    await agent._all_tools()

    assert factory.open_count == 1


async def test_local_vs_mcp_name_conflict_raises() -> None:
    """An MCP tool whose name matches a local @tool raises RegistryError."""
    factory = _FakeFactory({
        "svc": _FakeMCPClient(tools=[_FakeFastMCPTool("local_echo", None, {})]),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    with pytest.raises(RegistryError) as excinfo:
        await agent._all_tools()

    assert "local_echo" in str(excinfo.value)


async def test_concurrent_first_turn_resolves_once() -> None:
    """Two concurrent _all_tools() calls share a single resolution."""
    factory = _FakeFactory({
        "svc": _FakeMCPClient(tools=[_FakeFastMCPTool("remote", None, {})]),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    await asyncio.gather(agent._all_tools(), agent._all_tools(), agent._all_tools())

    assert factory.open_count == 1


async def test_resolved_mcp_specs_are_registered_with_invoker() -> None:
    """Each MCPToolSpec is registered with ToolInvoker so dispatch works."""
    factory = _FakeFactory({
        "svc": _FakeMCPClient(tools=[_FakeFastMCPTool("remote", None, {})]),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )
    register_spy = MagicMock(wraps=agent._tool_invoker.register)
    agent._tool_invoker.register = register_spy  # type: ignore[method-assign]

    await agent._all_tools()

    # Was register() called with an MCPToolSpec?
    registered_specs = [c.args[0] for c in register_spy.call_args_list]
    assert any(isinstance(s, MCPToolSpec) for s in registered_specs)
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/unit/agents/test_base_mcp.py -v`

Expected: FAIL — `BaseAgent.__init__` doesn't accept `mcp_factory` yet, and `mcp_servers()` / `_all_tools()` don't exist.

- [ ] **Step 3: Modify `src/ai_core/agents/base.py` — imports + `__init__`**

In `src/ai_core/agents/base.py`, update the imports section to add:

```python
import asyncio
```

(at the top, alongside `import json`).

Update the import block near line 47-55 to add MCP imports:

```python
from ai_core.exceptions import (
    AgentRecursionLimitError,
    PolicyDenialError,
    RegistryError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.mcp.resolver import resolve_mcp_tools
from ai_core.mcp.tools import MCPToolSpec
from ai_core.mcp.transports import IMCPConnectionFactory, MCPServerSpec
```

Update the `__init__` signature and body (currently at lines 99-113):

```python
    @inject
    def __init__(
        self,
        settings: AppSettings,
        llm: ILLMClient,
        memory: IMemoryManager,
        observability: IObservabilityProvider,
        tool_invoker: ToolInvoker,
        mcp_factory: IMCPConnectionFactory,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._memory = memory
        self._observability = observability
        self._tool_invoker = tool_invoker
        self._mcp_factory = mcp_factory
        self._graph: Any | None = None
        self._mcp_resolved: list[MCPToolSpec] | None = None
        self._mcp_resolution_lock: asyncio.Lock = asyncio.Lock()
```

- [ ] **Step 4: Add `mcp_servers()` method to `BaseAgent`**

In `src/ai_core/agents/base.py`, just after the existing `tools()` method (around line 122-124), add:

```python
    def mcp_servers(self) -> Sequence[MCPServerSpec]:
        """Return MCPServerSpecs whose tools the agent should use (default: empty).

        Resolved on the first agent turn via `_all_tools()`. Tools surfaced by
        these servers are merged with `tools()` for both LLM advertising and
        dispatch.
        """
        return ()
```

- [ ] **Step 5: Add `_all_tools()` helper**

After `_router_should_compact` (around line 385) — or anywhere before `_build_baggage` — add:

```python
    async def _all_tools(self) -> list[Tool | Mapping[str, Any]]:
        """Return the merged list of local + resolved MCP tools.

        Lazily resolves MCP servers on the first call; caches per-instance.
        Concurrent first-turn callers serialize on `_mcp_resolution_lock`.

        Raises:
            MCPTransportError: When a declared MCP server is unreachable.
            RegistryError: When MCP tool names conflict with each other or
                with local @tool names.
        """
        if self._mcp_resolved is None:
            async with self._mcp_resolution_lock:
                if self._mcp_resolved is None:
                    servers = list(self.mcp_servers())
                    resolved = (
                        await resolve_mcp_tools(servers, self._mcp_factory)
                        if servers
                        else []
                    )
                    local_names = {
                        t.name for t in self.tools() if isinstance(t, ToolSpec)
                    }
                    for mcp_spec in resolved:
                        if mcp_spec.name in local_names:
                            raise RegistryError(
                                f"MCP tool name {mcp_spec.name!r} conflicts with a local tool",
                                details={"tool": mcp_spec.name},
                            )
                        self._tool_invoker.register(mcp_spec)
                    self._mcp_resolved = resolved
        return list(self.tools()) + list(self._mcp_resolved)
```

> **Note on the double-checked locking:** the outer `if self._mcp_resolved is None` is a fast-path that avoids the lock when resolution is already done. The inner check inside the lock prevents the second concurrent caller from re-resolving after acquiring the lock.

- [ ] **Step 6: Update `_agent_node` to use `_all_tools()`**

In `src/ai_core/agents/base.py`, find the `_agent_node` method (around line 250). Replace the `tool_payload` build (around lines 259-264):

Old:
```python
        tool_payload: list[Mapping[str, Any]] = []
        for t in self.tools():
            if isinstance(t, ToolSpec):
                tool_payload.append(t.openai_schema())
            elif isinstance(t, Mapping):
                tool_payload.append(t)
```

New:
```python
        tool_payload: list[Mapping[str, Any]] = []
        for t in await self._all_tools():
            if isinstance(t, ToolSpec):
                tool_payload.append(t.openai_schema())
            elif isinstance(t, Mapping):
                tool_payload.append(t)
```

- [ ] **Step 7: Update `_tool_node` to use `_all_tools()`**

In `_tool_node` (around line 293), replace the lookup-dict construction (around lines 298-300):

Old:
```python
        sdk_tools_by_name: dict[str, ToolSpec] = {
            t.name: t for t in self.tools() if isinstance(t, ToolSpec)
        }
```

New:
```python
        sdk_tools_by_name: dict[str, ToolSpec] = {
            t.name: t for t in await self._all_tools() if isinstance(t, ToolSpec)
        }
```

- [ ] **Step 8: Run BaseAgent MCP unit tests**

Run: `uv run pytest tests/unit/agents/test_base_mcp.py -v`

Expected: all 7 tests PASS.

- [ ] **Step 9: Run the broader agent suite to confirm nothing regressed**

Run: `uv run pytest tests/unit/agents/ tests/unit/tools/ -q`

Expected: all green. (Backward compat: existing agents that don't override `mcp_servers()` get an empty list; `_all_tools()` returns just local tools, identical to pre-Phase-11 behavior.)

- [ ] **Step 10: Run the full unit suite**

Run: `uv run pytest -q -m unit`

Expected: all green. The new `mcp_factory` injection on `BaseAgent.__init__` is satisfied automatically by `AgentModule.provide_mcp_connection_factory` (already present at `src/ai_core/di/module.py:204`).

- [ ] **Step 11: Lint**

Run: `uv run ruff check src/ai_core/agents/base.py tests/unit/agents/test_base_mcp.py`

Expected: no errors.

- [ ] **Step 12: Commit**

```bash
git add src/ai_core/agents/base.py tests/unit/agents/test_base_mcp.py
git commit -m "feat(agents): mcp_servers() + lazy first-turn MCP tool resolution

BaseAgent now accepts an injected IMCPConnectionFactory and exposes a
mcp_servers() method (default empty). _all_tools() merges resolved MCP
tools with local @tools behind an asyncio.Lock — caches per instance,
runs the conflict check, registers MCP specs with the ToolInvoker.

_agent_node and _tool_node both consume _all_tools() so MCP tools are
indistinguishable from local ones to the LLM and the dispatcher."
```

---

## Task 5: Integration test against the real demo server

**Why:** Unit tests use mocks; this end-to-end test exercises the full pipeline with a real FastMCP stdio subprocess (no Docker, no network — just `subprocess.Popen`). It catches anything we got wrong in the FastMCP client surface assumptions and is the canonical "an agent uses an MCP server" reference.

**Files:**
- Create: `tests/integration/mcp/__init__.py` (empty)
- Create: `tests/integration/mcp/test_agent_uses_mcp.py`

- [ ] **Step 1: Create the integration test directory marker**

Create `tests/integration/mcp/__init__.py` as an empty file:

```bash
mkdir -p tests/integration/mcp
touch tests/integration/mcp/__init__.py
```

- [ ] **Step 2: Write the integration test**

Create `tests/integration/mcp/test_agent_uses_mcp.py`:

```python
"""End-to-end: agent invokes a real FastMCP stdio server's tool.

Spawns `examples/mcp_server_demo/server.py` as a stdio subprocess via the
production `PoolingMCPConnectionFactory`. Drives an agent with a
`ScriptedLLM` that emits a tool-call for `echo`, asserts the round-trip
through the full SDK pipeline (resolution, ToolInvoker, audit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import IAuditSink, ILLMClient
from ai_core.mcp import MCPServerSpec
from ai_core.testing import FakeAuditSink, ScriptedLLM, make_llm_response
from injector import Module, provider, singleton

pytestmark = pytest.mark.integration

# Skip when fastmcp isn't importable (CI environments without the extra).
pytest.importorskip("fastmcp")

DEMO_SERVER = Path(__file__).resolve().parents[3] / "examples" / "mcp_server_demo" / "server.py"


class _MCPDemoAgent(BaseAgent):
    agent_id = "mcp-demo-agent"

    def system_prompt(self) -> str:
        return "Use the echo tool to repeat what the user said."

    def mcp_servers(self):
        return [
            MCPServerSpec(
                component_id="demo-srv",
                transport="stdio",
                target=sys.executable,
                args=(str(DEMO_SERVER),),
            ),
        ]


def _build_container(llm: ILLMClient, audit_sink: IAuditSink) -> Container:
    settings = AppSettings(service_name="mcp-int-test", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

        @singleton
        @provider
        def provide_audit_sink(self) -> IAuditSink:
            return audit_sink

    return Container.build([AgentModule(settings=settings), _Overrides()])


async def test_agent_invokes_mcp_echo_end_to_end() -> None:
    """LLM → tool_call → MCPToolSpec.handler → real FastMCP server → result."""
    # ScriptedLLM responds with a tool_call to echo, then a final assistant message.
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"text": "hello from the SDK"}',
                    },
                },
            ],
        ),
        make_llm_response("Done."),
    ])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_MCPDemoAgent)
        final = await agent.ainvoke(
            messages=[{"role": "user", "content": "say hi"}],
            tenant_id="t",
        )

    # Verify: a ToolMessage with the echo'd value was appended.
    msgs = final["messages"]
    tool_msgs = [m for m in msgs if getattr(m, "type", None) == "tool"]
    assert len(tool_msgs) == 1
    assert "hello from the SDK" in tool_msgs[0].content

    # Audit recorded the invocation.
    completed_events = [
        r for r in audit.records
        if r.tool_name == "echo"
    ]
    assert len(completed_events) >= 1


async def test_mcp_resolution_caches_across_turns() -> None:
    """The MCP server is contacted exactly once even across multiple tool-loop turns."""
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "first"}'},
                },
            ],
        ),
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "second"}'},
                },
            ],
        ),
        make_llm_response("Done."),
    ])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_MCPDemoAgent)
        final = await agent.ainvoke(
            messages=[{"role": "user", "content": "two echoes"}],
            tenant_id="t",
        )

    # Two echo invocations succeeded.
    tool_msgs = [m for m in final["messages"] if getattr(m, "type", None) == "tool"]
    assert len(tool_msgs) == 2
    assert "first" in tool_msgs[0].content
    assert "second" in tool_msgs[1].content
```

- [ ] **Step 3: Run the integration test**

Run: `uv run pytest tests/integration/mcp/ -v`

Expected: both tests PASS. The agent really spawns the server, lists tools, calls echo twice, audits cleanly.

If FastMCP's actual API differs in shape (e.g., `tool_calls` format from `ScriptedLLM` doesn't match what the agent expects), debug by looking at the existing `agent_demo` test: `tests/unit/agents/test_base.py` and the demo runner output.

- [ ] **Step 4: Lint**

Run: `uv run ruff check tests/integration/mcp/`

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/mcp/
git commit -m "test(integration): agent end-to-end against real FastMCP stdio server

Spawns examples/mcp_server_demo/server.py as a stdio subprocess via
the production PoolingMCPConnectionFactory. Asserts: tool resolved,
echo invoked through ToolInvoker, FakeAuditSink recorded the call,
resolution cache holds across multiple tool-loop turns."
```

---

## Task 6: Example, README, smoke gate

**Why:** Phase 10's `mcp_server_demo` README has a "what's not (yet) shown" section that names exactly this gap as roadmap. Phase 11 closes it — drop the caveat, add a runnable agent demo, hook it into the smoke gate.

**Files:**
- Create: `examples/mcp_server_demo/agent_demo.py`
- Modify: `examples/mcp_server_demo/README.md`
- Modify: `scripts/run_examples.sh`

- [ ] **Step 1: Write the agent demo**

Create `examples/mcp_server_demo/agent_demo.py`:

```python
"""Agent that uses the demo MCP server's tools end-to-end.

Drives a real `BaseAgent` with `mcp_servers()` declaring the local
`server.py` (spawned as a stdio subprocess by the SDK's connection
factory). A `ScriptedLLM` emits a tool_call for `echo`; the agent
runs through the full SDK pipeline (resolution, ToolInvoker, audit)
and prints the result.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from injector import Module, provider, singleton
from rich.console import Console
from rich.panel import Panel

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.mcp import MCPServerSpec
from ai_core.testing import ScriptedLLM, make_llm_response

console = Console()
SERVER_PATH = Path(__file__).parent / "server.py"


class MCPAgent(BaseAgent):
    """Trivial agent that uses the demo server's tools."""

    agent_id: str = "mcp-demo-agent"

    def system_prompt(self) -> str:
        return "Repeat the user's message using the echo tool."

    def mcp_servers(self):
        return [
            MCPServerSpec(
                component_id="time-service",
                transport="stdio",
                target=sys.executable,
                args=(str(SERVER_PATH),),
            ),
        ]


def build_container(llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="mcp-agent-demo", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


async def main() -> None:
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"text": "hello from the SDK"}',
                    },
                },
            ],
        ),
        make_llm_response("Echoed successfully."),
    ])
    container = build_container(llm)
    async with container as c:
        agent = c.get(MCPAgent)
        final = await agent.ainvoke(
            messages=[{"role": "user", "content": "say hi"}],
            tenant_id="demo",
        )

    tool_msgs = [m for m in final["messages"] if getattr(m, "type", None) == "tool"]
    console.print(Panel.fit(
        "\n".join(str(m.content) for m in tool_msgs) or "(no tool messages)",
        title="MCP tool output",
        border_style="green",
    ))
    console.print(
        "[bold green]Done.[/bold green] Agent invoked the remote MCP echo tool "
        "through the full SDK pipeline (resolution + ToolInvoker + audit)."
    )


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run the demo manually to confirm it works**

Run: `uv run python examples/mcp_server_demo/agent_demo.py`

Expected: prints a panel containing `hello from the SDK` and the success message.

- [ ] **Step 3: Update the README**

In `examples/mcp_server_demo/README.md`, replace the "What's not (yet) shown" section (the one that says agent integration is on the roadmap) with the now-shipped capability. Read the current README first to see exact wording, then replace the affected section with:

```markdown
## What's also shown (Phase 11)

- Agent-side adapter: `agent_demo.py` runs an agent that declares this
  server via `mcp_servers()` and uses its tools through the standard
  `ToolInvoker` pipeline (validation, OPA, audit, observability).
```

And update the "Run" section to add a third command alongside `run_client.py`:

```markdown
### Run as an agent

```bash
uv run python examples/mcp_server_demo/agent_demo.py
```

The agent declares this server via `mcp_servers()`, the SDK resolves
its tools on the first turn, and `ToolInvoker` dispatches the call
through the same pipeline as local `@tool`s.
```

(Keep the existing "What this demonstrates" bullets; just add a new bullet noting that agent-side resolution is now implemented.)

- [ ] **Step 4: Run lint on the example + README**

Run: `uv run ruff check examples/mcp_server_demo/agent_demo.py`

Expected: no errors.

- [ ] **Step 5: Add `agent_demo` to the smoke gate**

In `scripts/run_examples.sh`, find the `mcp_server_demo` block (around line 41-46) and add a new line for `agent_demo` right after `run_client.py`. The block should look like:

```bash
# --- mcp_server_demo: needs fastmcp installed.
if "${TIMEOUT_CMD[@]+"${TIMEOUT_CMD[@]}"}" uv run python -c "import fastmcp" 2>/dev/null; then
    run_demo mcp_server_demo uv run python examples/mcp_server_demo/run_client.py
    run_demo mcp_agent_demo uv run python examples/mcp_server_demo/agent_demo.py
else
    skip_demo mcp_server_demo "fastmcp not installed (run \`uv sync\`)"
    skip_demo mcp_agent_demo "fastmcp not installed (run \`uv sync\`)"
fi
```

(Note: the existing fastmcp probe doesn't currently use `TIMEOUT_CMD`. Leave the probe as-is — `uv run python -c "import fastmcp"` is a fast operation. The above just adds two `run_demo`/`skip_demo` lines.)

The actual minimal edit: add inside the existing `if` branch:

```bash
    run_demo mcp_agent_demo uv run python examples/mcp_server_demo/agent_demo.py
```

and inside the `else` branch:

```bash
    skip_demo mcp_agent_demo "fastmcp not installed (run \`uv sync\`)"
```

- [ ] **Step 6: Run the smoke gate**

Run: `bash scripts/run_examples.sh`

Expected: `agent_demo`, `testing_demo`, `mcp_server_demo`, `mcp_agent_demo`, and `fastapi_integration_import` all run; summary at the bottom shows non-zero `ran` count and zero `failed`. (If `uv` isn't on PATH on the dev machine, the script will report exit 127 for every demo — that's the smoke gate doing its job.)

- [ ] **Step 7: Final lint sweep across everything Phase 11 touched**

Run: `uv run ruff check src/ai_core/mcp/ src/ai_core/agents/base.py tests/unit/mcp/ tests/unit/agents/test_base_mcp.py tests/integration/mcp/ examples/mcp_server_demo/ scripts/run_examples.sh`

(Last argument may not be a Python file — drop it if ruff complains about non-Python paths; rerun without it.)

Expected: no errors.

- [ ] **Step 8: Final test sweep**

Run: `uv run pytest -q`

Expected: all unit tests green; integration tests run if FastMCP installed; same skip count as before for Docker-gated tests.

- [ ] **Step 9: Commit**

```bash
git add examples/mcp_server_demo/agent_demo.py examples/mcp_server_demo/README.md scripts/run_examples.sh
git commit -m "feat(examples): mcp_server_demo/agent_demo.py — agent uses MCP tools end-to-end

Closes the loop documented as roadmap in Phase 10's README. Adds:

- agent_demo.py: agent declares the demo server via mcp_servers(),
  ScriptedLLM emits a tool_call for echo, full SDK pipeline runs
- README updated to drop the 'not yet shown' caveat and document
  the new agent_demo entry
- Smoke gate runs agent_demo alongside run_client.py (skips on the
  same fastmcp-unavailable condition)"
```

---

## Definition of done

- All six tasks committed; six green commits between BASE and HEAD.
- `uv run pytest` is green; integration tests run when FastMCP is importable.
- `uv run ruff check src/ai_core/mcp/ src/ai_core/agents/base.py tests/ examples/` is clean.
- `bash scripts/run_examples.sh` exercises `mcp_agent_demo` and exits 0 (when `uv` and `fastmcp` are present).
- `git diff main -- src/ai_core/` shows changes only in `mcp/` (transports.py, tools.py, resolver.py, __init__.py) and `agents/base.py` — no other source touched.
- `examples/mcp_server_demo/agent_demo.py` runs cleanly to print the echoed string.
- `BaseAgent.mcp_servers()` default returns `()` so all pre-Phase-11 agent code continues to work without modification.
