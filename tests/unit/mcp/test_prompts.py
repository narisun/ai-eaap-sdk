"""Unit tests for MCP prompt types (Phase 12)."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ai_core.mcp import MCPServerSpec
from ai_core.mcp.prompts import (
    MCPPrompt,
    MCPPromptArgument,
    MCPPromptMessage,
)

pytestmark = pytest.mark.unit


def _server() -> MCPServerSpec:
    return MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")


def test_prompt_argument_is_frozen() -> None:
    arg = MCPPromptArgument(name="text", description="The text", required=True)
    with pytest.raises(FrozenInstanceError):
        arg.required = False  # type: ignore[misc]


def test_prompt_is_frozen_with_arguments_tuple() -> None:
    """MCPPrompt holds arguments as a tuple (frozen, hashable)."""
    args = (
        MCPPromptArgument(name="text", description="Input", required=True),
        MCPPromptArgument(name="locale", description=None, required=False),
    )
    prompt = MCPPrompt(
        name="summarize",
        description="Summarize the text",
        arguments=args,
        mcp_server_spec=_server(),
    )

    assert prompt.name == "summarize"
    assert prompt.arguments[0].name == "text"
    assert prompt.arguments[1].required is False
    with pytest.raises(FrozenInstanceError):
        prompt.name = "different"  # type: ignore[misc]


def test_prompt_message_role_and_content() -> None:
    msg = MCPPromptMessage(role="user", content="Hello, world.")
    assert msg.role == "user"
    assert msg.content == "Hello, world."
    with pytest.raises(FrozenInstanceError):
        msg.content = "different"  # type: ignore[misc]


def test_prompt_with_no_arguments() -> None:
    """A prompt with no arguments uses an empty tuple."""
    prompt = MCPPrompt(
        name="ping",
        description=None,
        arguments=(),
        mcp_server_spec=_server(),
    )
    assert prompt.arguments == ()
