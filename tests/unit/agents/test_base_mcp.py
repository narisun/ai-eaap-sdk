"""Unit tests for BaseAgent's MCP integration: mcp_servers() + _all_tools()."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.exceptions import RegistryError
from ai_core.mcp import MCPServerSpec
from ai_core.mcp.tools import MCPToolSpec
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
