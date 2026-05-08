"""Unit tests for BaseAgent's MCP integration: mcp_servers() + _all_tools()."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from pydantic import BaseModel

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.exceptions import RegistryError
from ai_core.mcp import MCPServerSpec
from ai_core.mcp.prompts import MCPPrompt, MCPPromptArgument, MCPPromptMessage
from ai_core.mcp.tools import MCPResourceSpec, MCPToolSpec
from ai_core.tools import tool
from ai_core.tools.invoker import ToolInvoker

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

    async def list_resources(self) -> list:
        return []

    async def call_tool(self, name: str, args: dict[str, Any], **_: Any) -> _FakeCallToolResult:
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
    from ai_core.agents.runtime import AgentRuntime
    from ai_core.agents.tool_errors import DefaultToolErrorRenderer
    from ai_core.tools.registrar import ToolRegistrar
    from ai_core.tools.resolver import DefaultToolResolver
    invoker = ToolInvoker(observability=MagicMock())
    runtime = AgentRuntime(
        agent_settings=AppSettings(service_name="test", environment="local").agent,
        llm=MagicMock(),
        memory=MagicMock(),
        observability=MagicMock(),
        tool_invoker=invoker,
        mcp_factory=factory,
        tool_error_renderer=DefaultToolErrorRenderer(),
        tool_resolver=DefaultToolResolver(factory, invoker),
        tool_registrar=ToolRegistrar(invoker),
    )
    return agent_cls(runtime)


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
    assert factory.open_count == 2  # one list_tools + one list_resources roundtrip


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

    assert factory.open_count == 2  # tools + resources resolved once; second call hits cache


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

    assert factory.open_count == 2  # tools + resources resolved once despite concurrent callers


class _AgentMCPOnly(BaseAgent):
    """Agent with NO local @tool — only MCP servers."""

    agent_id = "mcp-only"

    _mcp_servers: tuple[MCPServerSpec, ...] = ()

    def system_prompt(self) -> str:
        return "test"

    def mcp_servers(self):
        return self._mcp_servers


async def test_compile_installs_tool_loop_for_mcp_only_agent() -> None:
    """An agent with no tools() but a non-empty mcp_servers() must still get the tool loop."""
    factory = _FakeFactory({
        "svc": _FakeMCPClient(tools=[_FakeFastMCPTool("remote", None, {})]),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    compiled = agent.compile()

    # The compiled graph must include the 'tool' node for the loop to function.
    # LangGraph's compiled graph exposes nodes via .get_graph().nodes (a Mapping).
    nodes = set(compiled.get_graph().nodes.keys())
    assert "tool" in nodes, (
        f"Expected 'tool' node in compiled graph but got {nodes}. "
        f"This means compile()'s install_loop decision missed the MCP-only path."
    )


async def test_compile_skips_tool_loop_for_agent_with_no_tools_at_all() -> None:
    """Sanity check: when both tools() and mcp_servers() are empty, no tool loop installed."""
    factory = _FakeFactory({})
    agent = _build_agent(_AgentMCPOnly, factory)
    # _mcp_servers stays default ()

    compiled = agent.compile()

    nodes = set(compiled.get_graph().nodes.keys())
    assert "tool" not in nodes, (
        f"Expected NO 'tool' node when tools() and mcp_servers() are both empty, "
        f"but got {nodes}."
    )


async def test_resolved_mcp_specs_are_registered_with_invoker() -> None:
    """Each MCPToolSpec is registered with ToolInvoker so dispatch works."""
    factory = _FakeFactory({
        "svc": _FakeMCPClient(tools=[_FakeFastMCPTool("remote", None, {})]),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )
    register_spy = MagicMock(wraps=agent.runtime.tool_invoker.register)
    agent.runtime.tool_invoker.register = register_spy  # type: ignore[method-assign]

    await agent._all_tools()

    # Was register() called with an MCPToolSpec?
    registered_specs = [c.args[0] for c in register_spy.call_args_list]
    assert any(isinstance(s, MCPToolSpec) for s in registered_specs)


# ---------------------------------------------------------------------------
# _all_tools() resource merging (Phase 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeFastMCPResource:
    name: str
    description: str | None
    uri: str


class _FakeMCPClientWithResources:
    """Fake client supporting list_tools, list_resources, read_resource, call_tool."""

    def __init__(
        self,
        tools: list[_FakeFastMCPTool] | None = None,
        resources: list[_FakeFastMCPResource] | None = None,
    ) -> None:
        self._tools = tools or []
        self._resources = resources or []

    async def list_tools(self) -> list[_FakeFastMCPTool]:
        return list(self._tools)

    async def list_resources(self) -> list[_FakeFastMCPResource]:
        return list(self._resources)

    async def read_resource(self, uri: str) -> list:
        @dataclass
        class _Text:
            text: str
            blob: bytes | None = None

        return [_Text(text=f"contents of {uri}")]

    async def call_tool(self, name: str, args: dict[str, Any], **_: Any):
        return _FakeCallToolResult(is_error=False, data=f"called {name}", content=[])


class _FakeFactoryWithResources:
    def __init__(self, clients: dict[str, _FakeMCPClientWithResources]) -> None:
        self._clients = clients
        self.open_count = 0

    def open(self, spec: MCPServerSpec):
        self.open_count += 1

        @asynccontextmanager
        async def _cm():
            yield self._clients[spec.component_id]

        return _cm()


async def test_all_tools_merges_resources_with_tools() -> None:
    """_all_tools() returns local + MCP tools + MCP resources."""
    factory = _FakeFactoryWithResources({
        "svc": _FakeMCPClientWithResources(
            tools=[_FakeFastMCPTool("remote_tool", "tool desc", {})],
            resources=[_FakeFastMCPResource("docs", "Project docs", "mcp://docs")],
        ),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    all_tools = await agent._all_tools()

    names = sorted(t.name for t in all_tools)
    assert names == ["docs", "local_echo", "remote_tool"]


async def test_all_tools_resource_is_mcp_resource_spec() -> None:
    """The resolved resource is an MCPResourceSpec, registered with the invoker."""
    factory = _FakeFactoryWithResources({
        "svc": _FakeMCPClientWithResources(
            resources=[_FakeFastMCPResource("docs", "d", "mcp://docs")],
        ),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    all_tools = await agent._all_tools()

    resource_specs = [t for t in all_tools if isinstance(t, MCPResourceSpec)]
    assert len(resource_specs) == 1
    assert resource_specs[0].name == "docs"
    assert resource_specs[0].mcp_resource_uri == "mcp://docs"


async def test_resource_vs_local_tool_conflict_raises_with_kind() -> None:
    """An MCP resource named the same as a local tool → RegistryError saying 'resource'."""
    factory = _FakeFactoryWithResources({
        "svc": _FakeMCPClientWithResources(
            resources=[_FakeFastMCPResource("local_echo", "d", "mcp://x")],
        ),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    with pytest.raises(RegistryError) as excinfo:
        await agent._all_tools()

    assert "local_echo" in str(excinfo.value)
    assert "resource" in str(excinfo.value)


async def test_tool_and_resource_with_same_name_conflict_raises() -> None:
    """An MCP tool and an MCP resource sharing a name → RegistryError."""
    factory = _FakeFactoryWithResources({
        "svc": _FakeMCPClientWithResources(
            tools=[_FakeFastMCPTool("ambiguous", "t", {})],
            resources=[_FakeFastMCPResource("ambiguous", "r", "mcp://x")],
        ),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    with pytest.raises(RegistryError) as excinfo:
        await agent._all_tools()

    assert "ambiguous" in str(excinfo.value)


# ---------------------------------------------------------------------------
# list_prompts() / get_prompt() tests (Phase 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeFastMCPPromptArg:
    name: str
    description: str | None
    required: bool


@dataclass(frozen=True)
class _FakeFastMCPPrompt:
    name: str
    description: str | None
    arguments: list[_FakeFastMCPPromptArg]


@dataclass(frozen=True)
class _FakeTextContent:
    """Stand-in for mcp.types.TextContent."""
    text: str
    type: str = "text"


@dataclass(frozen=True)
class _FakeFastMCPPromptMessage:
    role: str
    content: object  # may be _FakeTextContent or other


@dataclass(frozen=True)
class _FakeGetPromptResult:
    messages: list[_FakeFastMCPPromptMessage]


class _FakeMCPClientWithPrompts:
    """Fake client supporting list_prompts + get_prompt + the tool/resource methods."""

    def __init__(
        self,
        prompts: list[_FakeFastMCPPrompt] | None = None,
        get_prompt_results: dict[str, _FakeGetPromptResult] | None = None,
        list_prompts_raises: BaseException | None = None,
    ) -> None:
        self._prompts = prompts or []
        self._results = get_prompt_results or {}
        self._list_raises = list_prompts_raises
        self.get_prompt_invocations: list[tuple[str, dict]] = []

    async def list_prompts(self) -> list[_FakeFastMCPPrompt]:
        if self._list_raises is not None:
            raise self._list_raises
        return list(self._prompts)

    async def get_prompt(self, name: str, arguments: dict) -> _FakeGetPromptResult:
        self.get_prompt_invocations.append((name, dict(arguments)))
        if name in self._results:
            return self._results[name]
        return _FakeGetPromptResult(messages=[
            _FakeFastMCPPromptMessage(
                role="user",
                content=_FakeTextContent(text=f"prompt {name}"),
            ),
        ])

    # Methods for tool/resource compatibility (not exercised by these tests):
    async def list_tools(self) -> list:
        return []

    async def list_resources(self) -> list:
        return []


class _FakePromptFactory:
    def __init__(self, clients: dict[str, _FakeMCPClientWithPrompts]) -> None:
        self._clients = clients

    def open(self, spec: MCPServerSpec):
        @asynccontextmanager
        async def _cm():
            yield self._clients[spec.component_id]

        return _cm()


# ---- list_prompts ----

async def test_list_prompts_returns_typed_mcp_prompts() -> None:
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    factory = _FakePromptFactory({
        "svc": _FakeMCPClientWithPrompts(prompts=[
            _FakeFastMCPPrompt(
                name="summarize",
                description="Summarize text",
                arguments=[
                    _FakeFastMCPPromptArg(name="text", description="Input", required=True),
                ],
            ),
        ]),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    prompts = await agent.list_prompts()

    assert len(prompts) == 1
    assert isinstance(prompts[0], MCPPrompt)
    assert prompts[0].name == "summarize"
    assert prompts[0].description == "Summarize text"
    assert prompts[0].arguments == (
        MCPPromptArgument(name="text", description="Input", required=True),
    )
    assert prompts[0].mcp_server_spec.component_id == "svc"


async def test_list_prompts_silently_skips_method_not_found() -> None:
    """Servers that don't expose prompts → empty result for that server."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    factory = _FakePromptFactory({
        "svc": _FakeMCPClientWithPrompts(
            prompts=[],
            list_prompts_raises=McpError(ErrorData(code=-32601, message="Method not found")),
        ),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    prompts = await agent.list_prompts()

    assert prompts == []


async def test_list_prompts_cross_server_conflict_raises() -> None:
    """Two servers exposing same prompt name → RegistryError."""
    server_a = MCPServerSpec(component_id="a", transport="stdio", target="/bin/true")
    server_b = MCPServerSpec(component_id="b", transport="stdio", target="/bin/true")
    factory = _FakePromptFactory({
        "a": _FakeMCPClientWithPrompts(prompts=[
            _FakeFastMCPPrompt(name="summarize", description=None, arguments=[]),
        ]),
        "b": _FakeMCPClientWithPrompts(prompts=[
            _FakeFastMCPPrompt(name="summarize", description=None, arguments=[]),
        ]),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server_a, server_b)

    with pytest.raises(RegistryError) as excinfo:
        await agent.list_prompts()

    assert "summarize" in str(excinfo.value)


# ---- get_prompt ----

async def test_get_prompt_returns_typed_messages() -> None:
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    client = _FakeMCPClientWithPrompts(
        prompts=[_FakeFastMCPPrompt(name="summarize", description=None, arguments=[])],
        get_prompt_results={
            "summarize": _FakeGetPromptResult(messages=[
                _FakeFastMCPPromptMessage(
                    role="user",
                    content=_FakeTextContent(text="Please summarize:"),
                ),
                _FakeFastMCPPromptMessage(
                    role="assistant",
                    content=_FakeTextContent(text="Sure, what's the input?"),
                ),
            ]),
        },
    )
    factory = _FakePromptFactory({"svc": client})
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    messages = await agent.get_prompt("summarize", {"text": "hello"})

    assert messages == [
        MCPPromptMessage(role="user", content="Please summarize:"),
        MCPPromptMessage(role="assistant", content="Sure, what's the input?"),
    ]
    assert client.get_prompt_invocations == [("summarize", {"text": "hello"})]


async def test_get_prompt_with_explicit_server_skips_search() -> None:
    """When server= is provided, only that server is queried."""
    server_a = MCPServerSpec(component_id="a", transport="stdio", target="/bin/true")
    server_b = MCPServerSpec(component_id="b", transport="stdio", target="/bin/true")

    client_a = _FakeMCPClientWithPrompts(prompts=[
        _FakeFastMCPPrompt(name="x", description=None, arguments=[]),
    ])
    client_b = _FakeMCPClientWithPrompts(prompts=[
        _FakeFastMCPPrompt(name="x", description=None, arguments=[]),
    ])
    factory = _FakePromptFactory({"a": client_a, "b": client_b})
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server_a, server_b)

    await agent.get_prompt("x", server="b")

    assert client_a.get_prompt_invocations == []
    assert len(client_b.get_prompt_invocations) == 1


async def test_get_prompt_not_found_raises_registry_error() -> None:
    """If no server has the prompt, RegistryError with hint."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    factory = _FakePromptFactory({
        "svc": _FakeMCPClientWithPrompts(prompts=[
            _FakeFastMCPPrompt(name="other", description=None, arguments=[]),
        ]),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    with pytest.raises(RegistryError) as excinfo:
        await agent.get_prompt("missing")

    assert "missing" in str(excinfo.value)


async def test_get_prompt_drops_non_text_content() -> None:
    """Non-TextContent message bodies become empty content (binary suppressed)."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")

    @dataclass(frozen=True)
    class _FakeImage:
        type: str = "image"

    client = _FakeMCPClientWithPrompts(
        prompts=[_FakeFastMCPPrompt(name="x", description=None, arguments=[])],
        get_prompt_results={
            "x": _FakeGetPromptResult(messages=[
                _FakeFastMCPPromptMessage(role="user", content=_FakeImage()),
                _FakeFastMCPPromptMessage(
                    role="assistant", content=_FakeTextContent(text="text reply")
                ),
            ]),
        },
    )
    factory = _FakePromptFactory({"svc": client})
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    messages = await agent.get_prompt("x")

    assert messages[0].content == ""  # image suppressed
    assert messages[1].content == "text reply"
