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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ai_core.tools.spec import ToolSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ai_core.mcp.transports import MCPServerSpec


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
