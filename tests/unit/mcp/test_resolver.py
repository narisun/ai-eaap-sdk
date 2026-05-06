"""Unit tests for resolve_mcp_tools — uses a mocked IMCPConnectionFactory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from ai_core.exceptions import RegistryError, ToolExecutionError
from ai_core.mcp import MCPServerSpec
from ai_core.mcp.resolver import is_method_not_found, resolve_mcp_resources, resolve_mcp_tools
from ai_core.mcp.tools import (
    MCPResourceSpec,
    MCPToolSpec,
    _MCPPassthroughInput,
    _MCPPassthroughOutput,
)
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
        self.call_tool_invocations: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    async def list_tools(self) -> list[_FakeFastMCPTool]:
        self.list_tools_calls += 1
        return list(self._tools)

    async def call_tool(
        self, name: str, args: dict[str, Any], **kwargs: Any,
    ) -> _FakeCallToolResult:
        self.call_tool_invocations.append((name, args, kwargs))
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


async def test_handler_passes_raise_on_error_false() -> None:
    """The handler must pass raise_on_error=False so is_error maps to ToolExecutionError."""
    server = MCPServerSpec(
        component_id="svc", transport="stdio", target="/bin/true",
    )
    client = _FakeMCPClient(tools=[_FakeFastMCPTool("echo", None, {})])
    factory = _FakeFactory({"svc": client})
    specs = await resolve_mcp_tools([server], factory)

    payload = _MCPPassthroughInput.model_validate({"text": "hi"})
    await specs[0].handler(payload)

    # Verify the call kwargs included raise_on_error=False.
    assert len(client.call_tool_invocations) == 1
    name, _args, kwargs = client.call_tool_invocations[0]
    assert name == "echo"
    assert kwargs.get("raise_on_error") is False


async def test_handler_preserves_empty_input_schema() -> None:
    """An empty {} inputSchema is preserved verbatim (not replaced with type:object stub)."""
    server = MCPServerSpec(
        component_id="svc", transport="stdio", target="/bin/true",
    )
    client = _FakeMCPClient(tools=[_FakeFastMCPTool("anything_goes", None, {})])
    factory = _FakeFactory({"svc": client})
    specs = await resolve_mcp_tools([server], factory)

    assert specs[0].mcp_input_schema == {}


async def test_handler_substitutes_default_when_input_schema_is_none() -> None:
    """A None inputSchema (rare but defensive) gets the default object schema."""

    class _ToolWithoutSchema:
        name = "no-schema-tool"
        description = None
        inputSchema = None  # noqa: N815 — match FastMCP's attribute name

    server = MCPServerSpec(
        component_id="svc", transport="stdio", target="/bin/true",
    )
    factory = _FakeFactory({"svc": _FakeMCPClient(tools=[_ToolWithoutSchema()])})

    specs = await resolve_mcp_tools([server], factory)

    assert specs[0].mcp_input_schema == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# resolve_mcp_resources tests (Phase 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeFastMCPResource:
    name: str
    description: str | None
    uri: str


@dataclass(frozen=True)
class _FakeReadContents:
    """Stand-in for TextResourceContents."""
    text: str | None = None
    blob: bytes | None = None


class _FakeMCPResourceClient:
    """Fake client supporting list_resources + read_resource."""

    def __init__(
        self,
        resources: list[_FakeFastMCPResource],
        read_results: dict[str, list[_FakeReadContents]] | None = None,
        list_resources_raises: BaseException | None = None,
    ) -> None:
        self._resources = resources
        self._read_results = read_results or {}
        self._list_raises = list_resources_raises
        self.read_resource_invocations: list[str] = []

    async def list_resources(self) -> list[_FakeFastMCPResource]:
        if self._list_raises is not None:
            raise self._list_raises
        return list(self._resources)

    async def read_resource(self, uri: str) -> list[_FakeReadContents]:
        self.read_resource_invocations.append(uri)
        return self._read_results.get(uri, [_FakeReadContents(text=f"contents of {uri}")])


def _make_method_not_found_error() -> Exception:
    """Build a real McpError with code -32601."""
    return McpError(ErrorData(code=-32601, message="Method not found"))


# ---- _is_method_not_found ------------------------------------------------------

def test_is_method_not_found_true_on_minus_32601() -> None:
    """The predicate accepts a real McpError with code -32601."""
    exc = _make_method_not_found_error()
    assert is_method_not_found(exc) is True


def test_is_method_not_found_false_on_other_codes() -> None:
    """The predicate rejects McpErrors with different codes."""
    exc = McpError(ErrorData(code=-32602, message="Invalid params"))
    assert is_method_not_found(exc) is False


def test_is_method_not_found_false_on_unrelated_exception() -> None:
    """Non-McpError exceptions are not method-not-found."""
    assert is_method_not_found(ValueError("nope")) is False
    assert is_method_not_found(RuntimeError("transport gone")) is False


# ---- resolve_mcp_resources -----------------------------------------------------

async def test_resolves_one_server_one_resource() -> None:
    server = MCPServerSpec(
        component_id="docs-svc", transport="stdio", target="/bin/true",
    )
    fake_resource = _FakeFastMCPResource(
        name="documentation", description="Project docs",
        uri="mcp-demo://docs",
    )
    client = _FakeMCPResourceClient(resources=[fake_resource])
    factory = _FakeFactory({"docs-svc": client})

    specs = await resolve_mcp_resources([server], factory)

    assert len(specs) == 1
    assert isinstance(specs[0], MCPResourceSpec)
    assert specs[0].name == "documentation"
    assert specs[0].description == "Project docs"
    assert specs[0].mcp_resource_uri == "mcp-demo://docs"
    assert specs[0].mcp_input_schema == {"type": "object", "properties": {}}


async def test_resolve_resources_silently_skips_method_not_found() -> None:
    """Servers that don't expose resources raise McpError(-32601); we skip them."""
    server = MCPServerSpec(
        component_id="tool-only", transport="stdio", target="/bin/true",
    )
    client = _FakeMCPResourceClient(
        resources=[],
        list_resources_raises=_make_method_not_found_error(),
    )
    factory = _FakeFactory({"tool-only": client})

    specs = await resolve_mcp_resources([server], factory)

    assert specs == []


async def test_resolve_resources_propagates_other_mcp_errors() -> None:
    """Non-method-not-found McpErrors propagate (e.g., -32602 invalid params)."""
    server = MCPServerSpec(
        component_id="broken", transport="stdio", target="/bin/true",
    )
    client = _FakeMCPResourceClient(
        resources=[],
        list_resources_raises=McpError(ErrorData(code=-32602, message="invalid params")),
    )
    factory = _FakeFactory({"broken": client})

    with pytest.raises(McpError):
        await resolve_mcp_resources([server], factory)


async def test_resolve_resources_opa_path_propagates() -> None:
    """server.opa_decision_path → spec.opa_path."""
    server = MCPServerSpec(
        component_id="docs", transport="stdio", target="/bin/true",
        opa_decision_path="mcp.docs.allow",
    )
    fake_resource = _FakeFastMCPResource(name="readme", description=None, uri="mcp://readme")
    factory = _FakeFactory({"docs": _FakeMCPResourceClient(resources=[fake_resource])})

    specs = await resolve_mcp_resources([server], factory)

    assert specs[0].opa_path == "mcp.docs.allow"


async def test_resolve_resources_conflict_across_servers_raises() -> None:
    """Two servers exposing the same resource name → RegistryError."""
    server_a = MCPServerSpec(component_id="a", transport="stdio", target="/bin/true")
    server_b = MCPServerSpec(component_id="b", transport="stdio", target="/bin/true")
    factory = _FakeFactory({
        "a": _FakeMCPResourceClient(resources=[
            _FakeFastMCPResource(name="docs", description=None, uri="mcp://a/docs"),
        ]),
        "b": _FakeMCPResourceClient(resources=[
            _FakeFastMCPResource(name="docs", description=None, uri="mcp://b/docs"),
        ]),
    })

    with pytest.raises(RegistryError) as excinfo:
        await resolve_mcp_resources([server_a, server_b], factory)

    assert "docs" in str(excinfo.value)


async def test_resource_handler_returns_text_content() -> None:
    """The handler concatenates TextResourceContents and wraps in _MCPPassthroughOutput."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    client = _FakeMCPResourceClient(
        resources=[_FakeFastMCPResource(name="readme", description=None, uri="mcp://readme")],
        read_results={"mcp://readme": [
            _FakeReadContents(text="line one"),
            _FakeReadContents(text="line two"),
        ]},
    )
    factory = _FakeFactory({"svc": client})

    specs = await resolve_mcp_resources([server], factory)
    payload = _MCPPassthroughInput.model_validate({})
    result = await specs[0].handler(payload)

    assert isinstance(result, _MCPPassthroughOutput)
    assert result.value == "line one\nline two"
    assert client.read_resource_invocations == ["mcp://readme"]


async def test_resource_handler_substitutes_binary_with_placeholder() -> None:
    """BlobResourceContents become a placeholder with a logged warning."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    client = _FakeMCPResourceClient(
        resources=[_FakeFastMCPResource(name="image", description=None, uri="mcp://image")],
        read_results={"mcp://image": [
            _FakeReadContents(text=None, blob=b"\x89PNG..."),
            _FakeReadContents(text="caption"),
            _FakeReadContents(text=None, blob=b"more bytes"),
        ]},
    )
    factory = _FakeFactory({"svc": client})

    specs = await resolve_mcp_resources([server], factory)
    payload = _MCPPassthroughInput.model_validate({})
    result = await specs[0].handler(payload)

    # Two binary blocks → one placeholder; one text → "caption".
    # Order: text content emitted as encountered; placeholder appended after the loop.
    assert "caption" in result.value
    assert "<binary content suppressed: 2 block(s)>" in result.value
