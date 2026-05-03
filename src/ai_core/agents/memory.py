"""Memory + compaction logic for SDK agents.

Design:

* :class:`MemoryManager` is *stateless* — it operates on the
  :class:`AgentState` mapping passed by the LangGraph compaction node.
* It depends on a :class:`TokenCounter` (defaults to LiteLLM's tokenizer)
  so unit tests can inject deterministic counts.
* Compaction calls a *summarization chain* — modelled here as one
  :meth:`ILLMClient.complete` call with a system + user prompt. The
  prompt asks the LLM to summarise the conversation while explicitly
  preserving Essential Entities.
* "Essential Entities" are collected from
  :attr:`AgentSettings.essential_entity_keys` plus anything already
  present in :attr:`AgentState.essential_entities`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import litellm
from injector import inject

from ai_core.agents.state import AgentState
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import ILLMClient


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
@runtime_checkable
class TokenCounter(Protocol):
    """Counts prompt tokens for a list of chat messages."""

    def count(self, messages: Sequence[Mapping[str, Any]], *, model: str) -> int:
        """Return the prompt-token count for ``messages``."""
        ...


class LiteLLMTokenCounter:
    """Default :class:`TokenCounter` backed by :func:`litellm.token_counter`."""

    def count(self, messages: Sequence[Mapping[str, Any]], *, model: str) -> int:
        try:
            return int(litellm.token_counter(model=model, messages=list(messages)))
        except Exception:  # noqa: BLE001 — fall back to character heuristic
            approx = sum(len(str(m.get("content", ""))) for m in messages)
            return max(0, approx // 4)


# ---------------------------------------------------------------------------
# Compaction prompts
# ---------------------------------------------------------------------------
_COMPACTION_SYSTEM_PROMPT = (
    "You are a conversation summariser for an enterprise AI agent. "
    "Produce a faithful, concise narrative of the dialogue so far that "
    "captures every decision, fact, and action taken. The summary will "
    "REPLACE the older messages in the agent's working memory.\n\n"
    "Hard requirements:\n"
    "1. Preserve every Essential Entity verbatim.\n"
    "2. Keep all task IDs, ticket numbers, file paths, and code identifiers.\n"
    "3. Preserve outstanding todos, errors, and unanswered questions.\n"
    "4. Aim for under {target_tokens} tokens. Drop social pleasantries first.\n"
    "5. Output ONLY the summary — no preamble, no headings."
)


def _format_essential_entities(entities: Mapping[str, Any]) -> str:
    if not entities:
        return "(none recorded)"
    return "\n".join(f"- {k}: {v!r}" for k, v in entities.items())


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------
class MemoryManager:
    """Decide when to compact agent memory and execute the compaction.

    Args:
        settings: Aggregated application settings (``agent`` group consumed).
        llm: LLM client used by the summarization chain.
        token_counter: Token counter used by :meth:`should_compact`.
    """

    @inject
    def __init__(
        self,
        settings: AppSettings,
        llm: ILLMClient,
        token_counter: TokenCounter,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._counter = token_counter

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def should_compact(self, state: AgentState, *, model: str | None = None) -> bool:
        """Return ``True`` when current message tokens exceed the configured threshold.

        Args:
            state: Current agent state.
            model: Optional model id used for tokenization. Defaults to
                :attr:`LLMSettings.default_model`.

        Returns:
            Whether the compaction node should run before the next LLM call.
        """
        threshold = self._settings.agent.memory_compaction_token_threshold
        messages = state.get("messages") or []
        if not messages:
            return False
        used = self._counter.count(messages, model=model or self._settings.llm.default_model)
        return used > threshold

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------
    async def compact(
        self,
        state: AgentState,
        *,
        model: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
    ) -> AgentState:
        """Summarise the message history while preserving Essential Entities.

        Args:
            state: Current agent state.
            model: Optional model id to use for the summarization chain.
            tenant_id: Forwarded to the LLM client for budget enforcement.
            agent_id: Forwarded to the LLM client for budget enforcement.

        Returns:
            A *new* :class:`AgentState` whose ``messages`` list has been
            replaced with a single summary message plus the most recent
            user/assistant exchange. ``essential_entities`` is preserved
            verbatim and a brief ``summary`` is recorded.
        """
        messages = list(state.get("messages") or [])
        if not messages:
            return state

        cfg = self._settings.agent
        essentials = self._collect_essentials(state)

        target = cfg.memory_compaction_target_tokens
        system_prompt = _COMPACTION_SYSTEM_PROMPT.format(target_tokens=target)
        user_prompt = (
            "ESSENTIAL ENTITIES (must appear in summary verbatim):\n"
            f"{_format_essential_entities(essentials)}\n\n"
            "CONVERSATION:\n"
            f"{_render_history(messages)}"
        )

        summary_response = await self._llm.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tenant_id=tenant_id,
            agent_id=agent_id,
            temperature=0.0,
        )
        summary_text = summary_response.content.strip()

        # Keep the tail of the conversation (last user msg + last assistant msg)
        # so that the post-compaction agent has immediate context to act on.
        tail = _trailing_user_assistant_pair(messages)
        new_messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"[Conversation summary]\n{summary_text}"},
            *tail,
        ]

        new_state: AgentState = AgentState(
            messages=new_messages,
            essential_entities=dict(essentials),
            token_count=self._counter.count(
                new_messages,
                model=model or self._settings.llm.default_model,
            ),
            compaction_count=int(state.get("compaction_count", 0)) + 1,
            summary=summary_text,
            metadata=dict(state.get("metadata") or {}),
        )
        return new_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_essentials(self, state: AgentState) -> dict[str, Any]:
        """Merge configured essential keys with any already in state."""
        essentials: dict[str, Any] = {}
        existing = state.get("essential_entities") or {}
        for key in self._settings.agent.essential_entity_keys:
            if key in existing:
                essentials[key] = existing[key]
        # Also preserve any host-defined essentials not in the configured list.
        for key, value in existing.items():
            essentials.setdefault(key, value)
        return essentials


# ---------------------------------------------------------------------------
# Module-private message helpers
# ---------------------------------------------------------------------------
def _render_history(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render messages as a readable transcript for the summariser."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "?"))
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(part.get("text", part)) for part in content)
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _trailing_user_assistant_pair(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the most recent user and assistant messages, in order."""
    last_user: dict[str, Any] | None = None
    last_assistant: dict[str, Any] | None = None
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "assistant" and last_assistant is None:
            last_assistant = dict(msg)
        elif role == "user" and last_user is None:
            last_user = dict(msg)
        if last_user and last_assistant:
            break
    tail: list[dict[str, Any]] = []
    if last_user:
        tail.append(last_user)
    if last_assistant:
        tail.append(last_assistant)
    return tail


__all__ = ["MemoryManager", "TokenCounter", "LiteLLMTokenCounter"]
