"""MCP sub-package — Component Registry + FastMCP transport handlers."""

from __future__ import annotations

from ai_core.mcp.registry import ComponentRegistry, RegisteredComponent
from ai_core.mcp.transports import (
    FastMCPConnectionFactory,
    IMCPConnectionFactory,
    MCPServerSpec,
    MCPTransport,
    PoolingMCPConnectionFactory,
)

__all__ = [
    "ComponentRegistry",
    "FastMCPConnectionFactory",
    "IMCPConnectionFactory",
    "MCPServerSpec",
    "MCPTransport",
    "PoolingMCPConnectionFactory",
    "RegisteredComponent",
]
