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
