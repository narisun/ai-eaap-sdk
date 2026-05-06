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

import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ai_core.mcp.transports import (
    MCPServerSpec,  # noqa: TC001  # runtime import — guards get_type_hints()
)
from ai_core.tools.spec import ToolSpec

if TYPE_CHECKING:
    from collections.abc import Mapping


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
                "parameters": copy.deepcopy(self.mcp_input_schema),
            },
        }


@dataclass(frozen=True, slots=True)
class MCPResourceSpec(MCPToolSpec):
    """An MCP resource exposed as a parameter-less read-only tool.

    Phase 12 maps each resource the server advertises via `list_resources()`
    to one `MCPResourceSpec`. The handler closure (built by the resolver)
    hardcodes `mcp_resource_uri` and dispatches via `client.read_resource(uri)`.

    Attributes:
        mcp_resource_uri: The resource's URI (stored as plain str; FastMCP
            returns `pydantic.AnyUrl` which the resolver casts via `str()`
            on the way in).
    """

    mcp_resource_uri: str

    def openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema with no parameters."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }


def unwrap_mcp_tool_message(content: str) -> Any:  # noqa: ANN401
    """Unwrap MCPToolSpec's {"value": ...} envelope from a ToolMessage.content string.

    Returns the inner value when content is a JSON object with exactly one key
    "value" (the standard MCPToolSpec/MCPResourceSpec envelope). Otherwise
    returns the parsed JSON, or the raw string if it's not JSON at all.

    Use this to display MCP tool results cleanly in user-facing UIs without
    re-implementing the unwrap pattern inline.

    Args:
        content: The `ToolMessage.content` string from an MCP tool dispatch.

    Returns:
        The unwrapped value, parsed JSON, or raw string.

    Raises:
        TypeError: If `content` is not a string.
    """
    if not isinstance(content, str):
        raise TypeError(
            f"unwrap_mcp_tool_message expected str, got {type(content).__name__}"
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, dict) and set(parsed.keys()) == {"value"}:
        return parsed["value"]
    return parsed


__all__ = ["MCPResourceSpec", "MCPToolSpec", "unwrap_mcp_tool_message"]
