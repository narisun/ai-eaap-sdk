"""MCP sub-package — Component Registry + FastMCP transport handlers."""

from __future__ import annotations

from ai_core.mcp.registry import ComponentRegistry, RegisteredComponent
from ai_core.mcp.resolver import resolve_mcp_resources, resolve_mcp_tools
from ai_core.mcp.tools import MCPResourceSpec, MCPToolSpec, unwrap_mcp_tool_message
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
    "MCPResourceSpec",
    "MCPServerSpec",
    "MCPToolSpec",
    "MCPTransport",
    "RegisteredComponent",
    "resolve_mcp_resources",
    "resolve_mcp_tools",
    "unwrap_mcp_tool_message",
]
