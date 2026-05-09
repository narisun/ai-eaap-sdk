"""LangGraph-compatible agent state definitions.

The state is a :class:`TypedDict` so that LangGraph's ``StateGraph`` can
infer reducers per-key. Key choices:

* ``messages`` uses LangGraph's :func:`add_messages` reducer so that
  agent and tool nodes can append messages without overwriting history.
* ``essential_entities`` is a flat mapping that survives memory
  compaction (see :class:`MemoryManager`). Reducer-level merging keeps
  newer values without dropping any keys already present.
* ``token_count`` is updated by the agent node after every LLM call so
  that the compaction router can decide deterministically whether to
  branch into the compaction node before the next turn.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def _merge_entities(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reducer for :attr:`AgentState.essential_entities`.

    Merges ``right`` on top of ``left`` so that updates win, but no
    previously-recorded entity is silently dropped.
    """
    merged: dict[str, Any] = dict(left or {})
    if right:
        merged.update(right)
    return merged


class EssentialEntities(TypedDict, total=False):
    """Convenience typed-dict for well-known entity keys.

    These keys are populated by host applications when the agent is
    invoked and *must* survive memory compaction. Hosts may add their
    own keys — :attr:`AgentState.essential_entities` is loosely typed
    to accommodate them.
    """

    user_id: str
    tenant_id: str
    session_id: str
    task_id: str


class AgentState(TypedDict, total=False):
    """Default state schema for SDK-built agents.

    Attributes:
        messages: Chat-message history (with ``add_messages`` reducer).
        essential_entities: Persistent context preserved across compaction.
        token_count: Last-known prompt-token count, used for compaction routing.
        compaction_count: Number of compactions performed in this thread.
        summary: Latest narrative summary produced by the compaction node.
        metadata: Free-form mapping for run-level metadata.
        scratchpad: Free-form per-pattern scratch space — Phase 14 patterns
            store typed payloads at well-known keys (``scratchpad["plan"]``
            for :class:`PlanningAgent`, ``scratchpad["verifications"]`` for
            :class:`VerifierAgent`, etc.). Default reducer is overwrite;
            patterns that need merge semantics own that contract internally.
    """

    messages: Annotated[list[dict[str, Any]], add_messages]
    essential_entities: Annotated[dict[str, Any], _merge_entities]
    token_count: int
    compaction_count: int
    summary: str
    metadata: dict[str, Any]
    scratchpad: dict[str, Any]


def new_agent_state(
    *,
    initial_messages: list[dict[str, Any]] | None = None,
    essential: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AgentState:
    """Construct an :class:`AgentState` with sensible defaults.

    Args:
        initial_messages: Initial chat history.
        essential: Initial essential-entity values.
        metadata: Initial run metadata.

    Returns:
        A fully-populated :class:`AgentState`.
    """
    return AgentState(
        messages=list(initial_messages or []),
        essential_entities=dict(essential or {}),
        token_count=0,
        compaction_count=0,
        summary="",
        metadata=dict(metadata or {}),
        scratchpad={},
    )


__all__ = ["AgentState", "EssentialEntities", "new_agent_state"]
