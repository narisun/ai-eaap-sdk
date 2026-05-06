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
    MCPToolSpec,
    _MCPPassthroughInput,
    _MCPPassthroughOutput,
)
from ai_core.mcp.transports import MCPServerSpec  # noqa: TC001 — runtime import for closure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_core.mcp.transports import IMCPConnectionFactory


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


def _join_text_content(content: Any) -> str:  # noqa: ANN401 — heterogeneous content list
    """Join text-bearing content items into a single string."""
    return "\n".join(c.text for c in content if hasattr(c, "text"))


__all__ = ["resolve_mcp_tools"]
