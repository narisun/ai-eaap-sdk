"""Memory + compaction logic for SDK agents.

Design:

* :class:`IMemoryManager` is the abstract contract that
  :class:`BaseAgent` depends on. Hosts that need
  per-agent / per-tenant compaction strategies subclass it and
  override the binding in their DI module.
* :class:`MemoryManager` is the production implementation. It is
  *stateless* — it operates on the :class:`AgentState` mapping passed
  by the LangGraph compaction node and depends on a
  :class:`TokenCounter` (defaults to LiteLLM's tokenizer) so unit
  tests can inject deterministic counts.
* Compaction calls a *summarization chain* — modelled here as one
  :meth:`ILLMClient.complete` call with a system + user prompt. The
  prompt asks the LLM to summarise the conversation while explicitly
  preserving Essential Entities.
* "Essential Entities" are collected from
  :attr:`AgentSettings.essential_entity_keys` plus anything already
  present in :attr:`AgentState.essential_entities`.

Replacement vs. append semantics
--------------------------------
The compaction node returns a state whose ``messages`` list begins
with :class:`langchain_core.messages.RemoveMessage` carrying the
LangGraph-recognised id ``"__remove_all__"``. The ``add_messages``
reducer interprets this marker as "wipe the existing history" before
appending the rest of the returned messages. Without this, the
``add_messages`` reducer would *append* the summary to the existing
history and compaction would grow tokens instead of compressing them.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import litellm
from injector import inject
from langchain_core.messages import BaseMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from ai_core.agents.state import AgentState
from ai_core.config.settings import AgentSettings, LLMSettings
from ai_core.di.interfaces import ILLMClient
from ai_core.exceptions import LLMTimeoutError
from ai_core.observability.logging import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
@runtime_checkable
class TokenCounter(Protocol):
    """Counts prompt tokens for a list of chat messages."""

    def count(self, messages: Sequence[Any], *, model: str) -> int:
        """Return the prompt-token count for ``messages``."""
        ...


class LiteLLMTokenCounter:
    """Default :class:`TokenCounter` backed by :func:`litellm.token_counter`."""

    def count(self, messages: Sequence[Any], *, model: str) -> int:
        normalised = [_msg_to_dict(m) for m in messages]
        try:
            return int(litellm.token_counter(model=model, messages=normalised))
        except Exception as exc:  # noqa: BLE001 — fall back to character heuristic
            _logger.debug(
                "token_counter.fallback",
                model=model,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            approx = sum(len(_msg_content(m)) for m in messages)
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
# IMemoryManager
# ---------------------------------------------------------------------------
class IMemoryManager(ABC):
    """Abstract contract for agent memory management.

    Implementations decide *when* to compact (`should_compact`) and
    *how* (`compact`). The default :class:`MemoryManager` runs a
    summarisation chain; alternative strategies (recency-only,
    semantic clustering, tiered storage) plug in by overriding the
    binding in the host's DI module.
    """

    @abstractmethod
    def should_compact(self, state: AgentState, *, model: str | None = None) -> bool:
        """Return ``True`` when the state should be compacted before the next turn."""

    @abstractmethod
    async def compact(
        self,
        state: AgentState,
        *,
        model: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
    ) -> AgentState:
        """Return a new state with compressed history and Essential Entities preserved."""


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------
class MemoryManager(IMemoryManager):
    """Default :class:`IMemoryManager` — summarisation-chain based.

    Args:
        agent_settings: Agent runtime configuration (compaction thresholds,
            target tokens, timeout, essential entity keys).
        llm_settings: LLM configuration — only ``default_model`` is consumed
            so token counting and compaction can default consistently.
        llm: LLM client used by the summarization chain.
        token_counter: Token counter used by :meth:`should_compact`.
    """

    @inject
    def __init__(
        self,
        agent_settings: AgentSettings,
        llm_settings: LLMSettings,
        llm: ILLMClient,
        token_counter: TokenCounter,
    ) -> None:
        self._agent_cfg = agent_settings
        self._llm_cfg = llm_settings
        self._llm = llm
        self._counter = token_counter

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def should_compact(self, state: AgentState, *, model: str | None = None) -> bool:
        """Return ``True`` when current message tokens exceed the configured threshold."""
        threshold = self._agent_cfg.memory_compaction_token_threshold
        messages = state.get("messages") or []
        if not messages:
            return False
        used = self._counter.count(messages, model=model or self._llm_cfg.default_model)
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

        Wraps :meth:`_do_compact` in :func:`asyncio.wait_for` using the
        configured ``compaction_timeout_seconds`` budget. On timeout, logs
        a WARNING and returns the input state unchanged so the agent run
        continues. The state may exceed the threshold next turn; the
        next-turn ``should_compact`` check will retry.
        """
        timeout = self._agent_cfg.compaction_timeout_seconds
        try:
            return await asyncio.wait_for(
                self._do_compact(
                    state,
                    model=model,
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            _logger.warning(
                "compaction.skipped.budget_exceeded",
                timeout_seconds=timeout, agent_id=agent_id, tenant_id=tenant_id,
            )
            return state
        except LLMTimeoutError as exc:
            _logger.warning(
                "compaction.skipped.llm_timeout",
                agent_id=agent_id, tenant_id=tenant_id, error_code=exc.error_code,
            )
            return state

    async def _do_compact(
        self,
        state: AgentState,
        *,
        model: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
    ) -> AgentState:
        """Internal compaction implementation — called by :meth:`compact`.

        Returns a new :class:`AgentState` whose ``messages`` field starts
        with a :class:`RemoveMessage` marker so the ``add_messages``
        reducer wipes existing history before appending the summary
        and the trailing user/assistant pair. ``essential_entities`` is
        preserved verbatim and a brief ``summary`` is recorded.
        """
        messages = list(state.get("messages") or [])
        if not messages:
            return state

        cfg = self._agent_cfg
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

        tail = _trailing_user_assistant_pair(messages)
        replacement: list[Any] = [
            # Tells add_messages to drop every existing message before applying
            # the rest of this list — without it the summary would be appended.
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            {"role": "system", "content": f"[Conversation summary]\n{summary_text}"},
            *tail,
        ]

        return AgentState(
            messages=replacement,
            essential_entities=dict(essentials),
            token_count=self._counter.count(
                replacement[1:],  # token count for the actual content, excludes the marker
                model=model or self._llm_cfg.default_model,
            ),
            compaction_count=int(state.get("compaction_count", 0)) + 1,
            summary=summary_text,
            metadata=dict(state.get("metadata") or {}),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_essentials(self, state: AgentState) -> dict[str, Any]:
        """Merge configured essential keys with any already in state."""
        essentials: dict[str, Any] = {}
        existing = state.get("essential_entities") or {}
        for key in self._agent_cfg.essential_entity_keys:
            if key in existing:
                essentials[key] = existing[key]
        for key, value in existing.items():
            essentials.setdefault(key, value)
        return essentials


# ---------------------------------------------------------------------------
# Module-private message helpers
# ---------------------------------------------------------------------------
_LC_TYPE_TO_ROLE: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
    "remove": "remove",
}


def _msg_role(msg: Any) -> str:
    """Extract a normalised role from either a dict or a LangChain Message."""
    if isinstance(msg, BaseMessage):
        return _LC_TYPE_TO_ROLE.get(msg.type, msg.type)
    if isinstance(msg, Mapping):
        role = msg.get("role")
        return str(role) if role is not None else "?"
    return "?"


def _msg_content(msg: Any) -> str:
    """Extract textual content from either a dict or a LangChain Message."""
    content: Any
    if isinstance(msg, BaseMessage):
        content = msg.content
    elif isinstance(msg, Mapping):
        content = msg.get("content", "")
    else:
        content = ""
    if isinstance(content, list):
        # Some providers return content as a list of segments — join their text.
        return " ".join(
            str(part.get("text", part)) if isinstance(part, Mapping) else str(part)
            for part in content
        )
    return str(content)


def _msg_to_dict(msg: Any) -> Mapping[str, Any]:
    """Coerce a message to the OpenAI-style dict shape used by tokenisers."""
    return to_openai_message(msg)


def to_openai_message(msg: Any) -> dict[str, Any]:
    """Normalise a message into an OpenAI-style chat dict (``role`` + ``content``).

    LangGraph's :func:`add_messages` reducer converts dict messages into
    typed :class:`langchain_core.messages.BaseMessage` instances after
    they pass through the graph. LiteLLM and most OpenAI-compatible
    providers expect the ``{role, content}`` dict shape, so callers that
    forward state messages onto a downstream LLM SHOULD normalise via
    this helper.

    Args:
        msg: Either an OpenAI-style dict, a LangChain ``BaseMessage``, or
            any value with a ``content`` attribute.

    Returns:
        A new dict with at minimum ``role`` and ``content`` keys.
        Assistant tool calls are preserved as-is when present.
    """
    if isinstance(msg, Mapping):
        return dict(msg)
    if isinstance(msg, BaseMessage):
        out: dict[str, Any] = {"role": _msg_role(msg), "content": _msg_content(msg)}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # AIMessage exposes tool_calls as a list of TypedDicts; OpenAI's
            # provider accepts either shape, so we pass-through verbatim.
            out["tool_calls"] = list(tool_calls)
        return out
    return {"role": "user", "content": str(msg)}


def to_openai_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Apply :func:`to_openai_message` to every element of ``messages``."""
    return [to_openai_message(m) for m in messages]


def _render_history(messages: Sequence[Any]) -> str:
    """Render messages as a readable transcript for the summariser."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, RemoveMessage):
            continue
        role = _msg_role(msg)
        lines.append(f"{role.upper()}: {_msg_content(msg)}")
    return "\n".join(lines)


def _trailing_user_assistant_pair(messages: Sequence[Any]) -> list[Any]:
    """Return the most recent user and assistant messages (in chronological order).

    Returned messages are passed through verbatim — dicts stay dicts,
    LangChain Message objects stay typed. The ``add_messages`` reducer
    handles either shape downstream.
    """
    last_user: Any = None
    last_assistant: Any = None
    for msg in reversed(messages):
        role = _msg_role(msg)
        if role == "assistant" and last_assistant is None:
            last_assistant = msg
        elif role == "user" and last_user is None:
            last_user = msg
        if last_user is not None and last_assistant is not None:
            break
    tail: list[Any] = []
    if last_user is not None:
        tail.append(last_user)
    if last_assistant is not None:
        tail.append(last_assistant)
    return tail


__all__ = [
    "IMemoryManager",
    "MemoryManager",
    "TokenCounter",
    "LiteLLMTokenCounter",
    "to_openai_message",
    "to_openai_messages",
]
