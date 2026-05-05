"""Tests for the prompt-cache pure-function helpers."""
from __future__ import annotations

from typing import Any

import pytest

from ai_core.llm._prompt_cache import apply_prompt_cache, supports_prompt_cache

pytestmark = pytest.mark.unit


# --- supports_prompt_cache ---

@pytest.mark.parametrize("model", [
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-haiku",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-7",
])
def test_supports_prompt_cache_anthropic_models(model: str) -> None:
    assert supports_prompt_cache(model) is True


@pytest.mark.parametrize("model", [
    "openai/gpt-4o",
    "gpt-4-turbo",
    "bedrock/amazon.titan-text-express-v1",
    "vertex_ai/claude-3-5-sonnet",  # Vertex AI prefix not supported in Phase 4
    "azure/gpt-4",
])
def test_supports_prompt_cache_rejects_non_anthropic(model: str) -> None:
    assert supports_prompt_cache(model) is False


# --- apply_prompt_cache: skip cases ---

def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def _conversation() -> list[dict[str, Any]]:
    """Build a typical 6-turn conversation for cache application."""
    return [
        _msg("system", "You are a helpful assistant."),
        _msg("user", "Question 1"),
        _msg("assistant", "Answer 1"),
        _msg("user", "Question 2"),
        _msg("assistant", "Answer 2"),
        _msg("user", "Question 3"),  # latest user turn
    ]


def test_apply_skipped_when_disabled() -> None:
    msgs = _conversation()
    result_msgs, result_tools = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=False, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    # Returns equivalent messages, but as a fresh list (defensive copy).
    assert result_msgs == list(msgs)
    assert result_msgs is not msgs
    assert result_tools is None


def test_apply_skipped_for_non_anthropic_model() -> None:
    msgs = _conversation()
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="openai/gpt-4o",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    assert all(isinstance(m["content"], str) for m in result_msgs)


def test_apply_skipped_below_message_threshold() -> None:
    msgs = [_msg("system", "sys"), _msg("user", "hi")]
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=True, min_messages=10, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    assert all(isinstance(m["content"], str) for m in result_msgs)


def test_apply_skipped_below_token_threshold() -> None:
    msgs = _conversation()
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=10_000,
        estimated_tokens=500,
    )
    assert all(isinstance(m["content"], str) for m in result_msgs)


def test_apply_skipped_returns_fresh_tools_list() -> None:
    """When skipped, the returned tools list is a fresh copy, not the caller's list."""
    msgs = [_msg("system", "sys"), _msg("user", "hi")]
    tools = [{"type": "function", "function": {"name": "x"}}]
    _, result_tools = apply_prompt_cache(
        msgs, tools=tools, model="openai/gpt-4o",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    assert result_tools == tools
    assert result_tools is not tools  # fresh list, not aliased


# --- apply_prompt_cache: applied cases ---

def test_apply_inserts_breakpoint_on_system_prompt() -> None:
    msgs = _conversation()
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="anthropic/claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    # System message should have its content as a list with cache_control on the last block.
    sys_msg = result_msgs[0]
    assert sys_msg["role"] == "system"
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_apply_inserts_breakpoint_on_last_stable_assistant() -> None:
    msgs = _conversation()
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="anthropic/claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    # The assistant message right before the latest user turn (index 4) should have cache_control.
    last_stable_assistant = result_msgs[4]
    assert last_stable_assistant["role"] == "assistant"
    assert isinstance(last_stable_assistant["content"], list)
    assert last_stable_assistant["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_apply_no_assistant_yet_skips_history_breakpoint() -> None:
    """When conversation has only system + user (no assistant yet), only system gets cached."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "hi"),
        _msg("user", "follow-up"),  # no assistant turn yet
    ]
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    # System has cache_control.
    assert isinstance(result_msgs[0]["content"], list)
    # No assistant message → no second breakpoint.
    user_msgs = [m for m in result_msgs[1:] if m["role"] == "user"]
    for m in user_msgs:
        if isinstance(m["content"], list):
            for block in m["content"]:
                assert "cache_control" not in block, "user messages should NOT be cached"


def test_apply_handles_pre_structured_content() -> None:
    """When content is already a list of blocks, cache_control is added to the LAST block."""
    msgs = [
        {"role": "system", "content": [
            {"type": "text", "text": "block 1"},
            {"type": "text", "text": "block 2"},
        ]},
        _msg("user", "hi"),
        _msg("assistant", "hello"),
        _msg("user", "again"),
    ]
    result_msgs, _ = apply_prompt_cache(
        msgs, tools=None, model="claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    sys_content = result_msgs[0]["content"]
    assert sys_content[0].get("cache_control") is None
    assert sys_content[1].get("cache_control") == {"type": "ephemeral"}


def test_apply_caches_last_tool_when_tools_present() -> None:
    msgs = _conversation()
    tools = [
        {"type": "function", "function": {
            "name": "tool_a", "description": "...", "parameters": {},
        }},
        {"type": "function", "function": {
            "name": "tool_b", "description": "...", "parameters": {},
        }},
    ]
    _, result_tools = apply_prompt_cache(
        msgs, tools=tools, model="claude-3-5-sonnet",
        enabled=True, min_messages=2, min_estimated_tokens=100,
        estimated_tokens=2048,
    )
    assert result_tools is not None
    # First tool unchanged.
    assert "cache_control" not in result_tools[0]
    # Last tool gets cache_control.
    assert result_tools[-1].get("cache_control") == {"type": "ephemeral"}
