"""Resolve a list of MCPServerSpecs into MCPToolSpecs by talking to each server.

`resolve_mcp_tools` is the async helper that BaseAgent calls on the first
turn to discover what tools each declared MCP server exposes. For each
server it opens one pooled connection (via `IMCPConnectionFactory.open`),
calls `list_tools()`, then builds one `MCPToolSpec` per advertised tool —
each carrying a closure handler that opens a fresh pooled connection on
every invocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_core.exceptions import RegistryError, ToolExecutionError
from ai_core.mcp.tools import (
    MCPResourceSpec,
    MCPToolSpec,
    _MCPPassthroughInput,
    _MCPPassthroughOutput,
)
from ai_core.mcp.transports import MCPServerSpec  # noqa: TC001 — runtime import for closure
from ai_core.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_core.mcp.transports import IMCPConnectionFactory

_logger = get_logger(__name__)


def _is_method_not_found(exc: BaseException) -> bool:
    """Return True if `exc` is an McpError signaling JSON-RPC method-not-found.

    Used by Phase 12's resolve_mcp_resources and by BaseAgent's prompt API to
    silently skip servers that don't advertise the resources/prompts methods.

    Centralized here so a single line changes if FastMCP's exception shape evolves.
    """
    from mcp.shared.exceptions import McpError  # noqa: PLC0415 — defer FastMCP import
    return isinstance(exc, McpError) and exc.error.code == -32601  # noqa: PLR2004


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
    fastmcp_tool: Any,  # noqa: ANN401 — intentionally untyped; FastMCP tool shape
    factory: IMCPConnectionFactory,
) -> MCPToolSpec:
    """Construct an MCPToolSpec wrapping a closure handler that calls the remote tool.

    The closure captures `tool_name`, `component_id`, `server`, and `factory`
    as locals of THIS function — not as references to outer-loop variables —
    so multiple specs built from the same loop iteration each call the
    correct server with the correct tool name. This is the standard
    factory-function pattern that prevents Python's late-binding closure
    footgun.

    Args:
        server: The MCP server this tool was discovered on. Captured by
            the handler closure to pass to `factory.open()` on each call.
        fastmcp_tool: The FastMCP `Tool` object returned by
            `client.list_tools()` (duck-typed: only `.name`, `.description`,
            `.inputSchema` are accessed).
        factory: The connection factory; captured by the closure to open
            a fresh pooled connection per invocation.

    Returns:
        A frozen `MCPToolSpec` ready to register with `ToolInvoker`.
    """
    tool_name = fastmcp_tool.name
    component_id = server.component_id

    async def _handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
        args_dict = payload.model_dump()
        async with factory.open(server) as client:
            result = await client.call_tool(tool_name, args_dict, raise_on_error=False)

        if result.is_error:
            content_text = _join_text_content(getattr(result, "content", ()))
            raise ToolExecutionError(
                f"MCP tool {tool_name!r} returned error",
                details={
                    "tool": tool_name,
                    "server": component_id,
                    "content": content_text,
                    "meta": getattr(result, "meta", None),
                },
            )

        if getattr(result, "data", None) is not None:
            return _MCPPassthroughOutput(value=result.data)

        return _MCPPassthroughOutput(
            value=_join_text_content(getattr(result, "content", ()))
        )

    _input_schema = getattr(fastmcp_tool, "inputSchema", None)
    return MCPToolSpec(
        name=tool_name,
        version=1,
        description=fastmcp_tool.description or "",
        input_model=_MCPPassthroughInput,
        output_model=_MCPPassthroughOutput,
        handler=_handler,
        opa_path=server.opa_decision_path,
        mcp_server_spec=server,
        mcp_input_schema=(
            _input_schema if _input_schema is not None
            else {"type": "object", "properties": {}}
        ),
    )


def _join_text_content(content: Any) -> str:  # noqa: ANN401 — heterogeneous content list
    """Join text-bearing content items into a single string."""
    return "\n".join(c.text for c in content if hasattr(c, "text"))


async def resolve_mcp_resources(
    servers: Sequence[MCPServerSpec],
    factory: IMCPConnectionFactory,
) -> list[MCPResourceSpec]:
    """Discover resources on each server and return them as MCPResourceSpec instances.

    Each resource becomes a parameter-less read-only tool. Handler closures capture
    the URI and dispatch via `client.read_resource(uri)`. Same conflict-detection
    semantics as `resolve_mcp_tools` — duplicate names across servers raise
    RegistryError.

    Servers that don't expose resources (list_resources raises McpError with
    code -32601) are silently skipped. Other errors propagate.

    Args:
        servers: MCP server specs the agent declared via `mcp_servers()`.
        factory: Connection factory (typically the DI-bound `PoolingMCPConnectionFactory`).

    Returns:
        One `MCPResourceSpec` per discovered resource. Empty when no servers
        expose any.

    Raises:
        MCPTransportError: When a server is unreachable.
        RegistryError: When two servers expose resources with the same name.
        McpError: For any non-method-not-found protocol error.
    """
    seen_names: set[str] = set()
    out: list[MCPResourceSpec] = []
    for spec in servers:
        async with factory.open(spec) as client:
            try:
                resources = await client.list_resources()
            except Exception as exc:  # we re-raise unless it's method-not-found
                if _is_method_not_found(exc):
                    continue
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


def _build_mcp_resource_spec(
    server: MCPServerSpec,
    fastmcp_resource: Any,  # noqa: ANN401 — FastMCP resource shape is duck-typed
    factory: IMCPConnectionFactory,
) -> MCPResourceSpec:
    """Construct an MCPResourceSpec wrapping a closure handler that reads the resource.

    The closure captures `uri`, `component_id`, `server`, and `factory` as locals
    of THIS function — the standard factory-function pattern that prevents
    Python's late-binding closure footgun.

    Args:
        server: The MCP server this resource was discovered on. Captured by the
            handler closure to pass to `factory.open()` on each call.
        fastmcp_resource: The FastMCP `Resource` object returned by
            `client.list_resources()` (duck-typed: only `.name`, `.description`,
            `.uri` are accessed).
        factory: The connection factory; captured by the closure to open a fresh
            pooled connection per invocation.

    Returns:
        A frozen `MCPResourceSpec` ready to register with `ToolInvoker`.
    """
    name = fastmcp_resource.name
    description = fastmcp_resource.description or ""
    uri = str(fastmcp_resource.uri)
    component_id = server.component_id

    async def _handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
        # payload is empty (resources take no args); ignored.
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

    return MCPResourceSpec(
        name=name,
        version=1,
        description=description,
        input_model=_MCPPassthroughInput,
        output_model=_MCPPassthroughOutput,
        handler=_handler,
        opa_path=server.opa_decision_path,
        mcp_server_spec=server,
        mcp_input_schema={"type": "object", "properties": {}},
        mcp_resource_uri=uri,
    )


__all__ = ["resolve_mcp_resources", "resolve_mcp_tools"]
