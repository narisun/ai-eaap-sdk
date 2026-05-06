"""MCP prompt types — used by BaseAgent.list_prompts() / get_prompt().

Phase 12 exposes MCP prompts as application-invoked helpers (not LLM-callable
tools). The application fetches a prompt by name, splices the resulting
messages into ainvoke(messages=...), and runs the agent.

All types are frozen dataclasses:
- MCPPromptArgument: one argument declaration on a prompt template.
- MCPPrompt: a prompt template with its origin server tagged.
- MCPPromptMessage: one message after argument substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_core.mcp.transports import MCPServerSpec


@dataclass(frozen=True, slots=True)
class MCPPromptArgument:
    """One argument declaration on an MCP prompt template.

    Mirrors mcp.types.PromptArgument but kept SDK-internal so the public
    API doesn't depend on the FastMCP/mcp Python types directly.

    Attributes:
        name: The argument's name (used as a key when calling get_prompt).
        description: Human-readable description (may be None).
        required: Whether the argument must be provided.
    """

    name: str
    description: str | None
    required: bool


@dataclass(frozen=True, slots=True)
class MCPPrompt:
    """A prompt template advertised by an MCP server.

    Attributes:
        name: The prompt's name; unique across declared servers (a conflict
            during list_prompts raises RegistryError).
        description: Human-readable description (may be None).
        arguments: Tuple of MCPPromptArgument declarations (may be empty).
        mcp_server_spec: Which server this prompt came from. Useful for
            passing back to get_prompt(name, args, server=spec.component_id)
            to skip the cross-server search.
    """

    name: str
    description: str | None
    arguments: tuple[MCPPromptArgument, ...]
    mcp_server_spec: MCPServerSpec


@dataclass(frozen=True, slots=True)
class MCPPromptMessage:
    """One message from a fetched prompt template (after argument substitution).

    Attributes:
        role: Typically "user" or "assistant" — passed through as `str` from
            the MCP protocol without client-side validation, so future protocol
            extensions are forward-compatible.
        content: Text content. Binary content blocks (images, etc.) from the
            FastMCP message are dropped in v1; only TextContent.text is preserved.
    """

    role: str
    content: str


__all__ = ["MCPPrompt", "MCPPromptArgument", "MCPPromptMessage"]
