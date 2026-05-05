"""Anthropic prompt-cache helpers — pure functions, no I/O.

Exposes:
- :func:`supports_prompt_cache` — provider detection by model prefix
- :func:`apply_prompt_cache` — non-mutating transform that adds
  ``cache_control`` blocks to messages and tools when caching is appropriate

The helpers do not depend on DI, observability, or LLM clients. They take
a list of messages and configuration scalars, and return a (possibly
modified) list ready to pass to ``litellm.acompletion``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_ANTHROPIC_PREFIXES = ("anthropic/", "bedrock/anthropic.", "claude-")


def supports_prompt_cache(model: str) -> bool:
    """Return True iff the model identifier targets Anthropic Claude.

    Recognises three common LiteLLM forms:
      - ``anthropic/claude-3-5-sonnet-20241022``
      - ``bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0``
      - ``claude-3-5-sonnet-20241022`` (bare Anthropic SDK style)

    Returns False for OpenAI, Azure, Vertex AI Anthropic (`vertex_ai/...`),
    and other providers — Phase 4 leaves Vertex Anthropic as a follow-up.
    """
    lowered = model.lower()
    return any(lowered.startswith(p) for p in _ANTHROPIC_PREFIXES)


def apply_prompt_cache(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None,
    model: str,
    enabled: bool,
    min_messages: int,
    min_estimated_tokens: int,
    estimated_tokens: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]] | None]:
    """Return (messages, tools) with cache_control breakpoints applied where appropriate.

    Returns the originals (as fresh lists) unchanged when:
      - ``enabled`` is False, OR
      - the model doesn't support caching (non-Anthropic provider), OR
      - ``len(messages) < min_messages``, OR
      - ``estimated_tokens < min_estimated_tokens``.

    Otherwise returns lists with cache_control content blocks inserted at:
      1. End of the system prompt (if present).
      2. End of the last assistant message before the trailing user turn
         (if the conversation has a stable history boundary).
      3. End of the tool list (if tools are present).

    Anthropic supports up to 4 cache breakpoints; this helper uses at most 3.
    """
    if not enabled or not supports_prompt_cache(model):
        return list(messages), list(tools) if tools is not None else None
    if len(messages) < min_messages or estimated_tokens < min_estimated_tokens:
        return list(messages), list(tools) if tools is not None else None

    # Convert all messages to a fresh list (cache-control blocks are added in place).
    cached_messages: list[Mapping[str, Any]] = [
        _with_cache_control(m, breakpoint=False) for m in messages
    ]

    # Breakpoint 1: end of the first system message.
    for i, m in enumerate(cached_messages):
        if m.get("role") == "system":
            cached_messages[i] = _with_cache_control(m, breakpoint=True)
            break

    # Breakpoint 2: last assistant message before the trailing user turn.
    last_assistant_idx = _find_last_stable_assistant(cached_messages)
    if last_assistant_idx is not None:
        cached_messages[last_assistant_idx] = _with_cache_control(
            cached_messages[last_assistant_idx], breakpoint=True
        )

    # Breakpoint 3: last tool schema (if present).
    cached_tools: list[Mapping[str, Any]] | None = list(tools) if tools is not None else None
    if cached_tools:
        cached_tools[-1] = _with_tool_cache_control(cached_tools[-1])

    return cached_messages, cached_tools


def _with_cache_control(
    message: Mapping[str, Any], *, breakpoint: bool
) -> dict[str, Any]:
    """Convert message.content to structured form; tag last block as breakpoint if asked."""
    content = message.get("content")
    if isinstance(content, str):
        block: dict[str, Any] = {"type": "text", "text": content}
        if breakpoint:
            block["cache_control"] = {"type": "ephemeral"}
        return {**message, "content": [block]}
    if breakpoint and isinstance(content, list) and content:
        new_content = list(content)
        last_block = dict(new_content[-1])
        last_block["cache_control"] = {"type": "ephemeral"}
        new_content[-1] = last_block
        return {**message, "content": new_content}
    return dict(message)


def _with_tool_cache_control(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Tag the tool with cache_control to cache the tool list up to and including this entry."""
    return {**tool, "cache_control": {"type": "ephemeral"}}


def _find_last_stable_assistant(messages: Sequence[Mapping[str, Any]]) -> int | None:
    """Return index of the last assistant message before the trailing user turn.

    Returns None if the conversation doesn't have a stable history boundary
    (no assistant turns yet, or the last message isn't a user turn).
    """
    if not messages or messages[-1].get("role") != "user":
        return None
    for i in range(len(messages) - 2, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return None


__all__ = ["apply_prompt_cache", "supports_prompt_cache"]
