"""Abstract :class:`BaseAgent` integrating LangGraph + SDK services.

A subclass implements three things:

1. :meth:`tools` — return the OpenAI-style tool definitions the agent
   can invoke (or ``[]``).
2. :meth:`system_prompt` — return the system prompt for the agent.
3. *(optional)* :meth:`extend_graph` — add custom nodes/edges to the
   compiled :class:`StateGraph` before it is finalised.

The base class wires:

* an ``agent`` node that calls :class:`ILLMClient` with the system prompt
  prepended to the conversation;
* a ``compaction`` node that delegates to :class:`MemoryManager.compact`;
* a router that runs compaction *before* each agent turn whenever
  :meth:`MemoryManager.should_compact` returns ``True``;
* an optional checkpointer (LangGraph-native) supplied by callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from injector import inject
from langgraph.graph import END, START, StateGraph

from ai_core.agents.memory import MemoryManager
from ai_core.agents.state import AgentState, new_agent_state
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import ILLMClient, IObservabilityProvider


class BaseAgent(ABC):
    """Abstract base class for SDK-built LangGraph agents.

    Args:
        settings: Aggregated application settings.
        llm: LLM client used by the agent node.
        memory: Memory manager used by the compaction node.
        observability: Provider used to wrap top-level invocations in spans.
    """

    #: Logical identifier — override in subclasses (used for budgeting + tracing).
    agent_id: str = "base-agent"

    @inject
    def __init__(
        self,
        settings: AppSettings,
        llm: ILLMClient,
        memory: MemoryManager,
        observability: IObservabilityProvider,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._memory = memory
        self._observability = observability
        self._graph: Any | None = None

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""

    def tools(self) -> Sequence[Mapping[str, Any]]:
        """Return tool definitions in OpenAI-style format. Defaults to none."""
        return ()

    def extend_graph(self, graph: StateGraph) -> None:
        """Hook for subclasses to add custom nodes/edges before compile."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compile(self, *, checkpointer: Any | None = None) -> Any:
        """Build + compile the LangGraph. Idempotent.

        Args:
            checkpointer: Optional LangGraph-native checkpointer instance.

        Returns:
            The compiled LangGraph runnable.
        """
        if self._graph is not None:
            return self._graph
        graph: StateGraph = StateGraph(AgentState)

        graph.add_node("compact", self._compaction_node)
        graph.add_node("agent", self._agent_node)

        graph.add_conditional_edges(
            START,
            self._router_should_compact,
            {True: "compact", False: "agent"},
        )
        graph.add_edge("compact", "agent")
        graph.add_edge("agent", END)

        self.extend_graph(graph)
        self._graph = graph.compile(checkpointer=checkpointer)
        return self._graph

    async def ainvoke(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        essential: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        """Run the agent end-to-end against an initial message list.

        Args:
            messages: Initial chat history (``role`` + ``content`` dicts).
            essential: Initial Essential Entities to seed the state with.
            tenant_id: Tenant identifier used for budgeting/tracing.
            thread_id: Optional LangGraph thread identifier (enables checkpointer).

        Returns:
            Final :class:`AgentState` after graph execution.
        """
        compiled = self.compile()
        initial = new_agent_state(
            initial_messages=list(messages),
            essential={**(essential or {}), "tenant_id": tenant_id or ""},
            metadata={"agent_id": self.agent_id},
        )
        config: dict[str, Any] = {}
        if thread_id is not None:
            config["configurable"] = {"thread_id": thread_id}

        attributes = {"agent.id": self.agent_id, "agent.tenant_id": tenant_id or ""}
        async with self._observability.start_span("agent.ainvoke", attributes=attributes):
            result = await compiled.ainvoke(initial, config=config or None)
        return result

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    async def _agent_node(self, state: AgentState) -> AgentState:
        """LangGraph node that performs one LLM turn."""
        history = list(state.get("messages") or [])
        prompt: list[Mapping[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            *history,
        ]
        essentials = state.get("essential_entities") or {}
        response = await self._llm.complete(
            model=None,
            messages=prompt,
            tools=list(self.tools()) or None,
            tenant_id=str(essentials.get("tenant_id") or "") or None,
            agent_id=self.agent_id,
        )
        appended: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": response.content,
                **({"tool_calls": list(response.tool_calls)} if response.tool_calls else {}),
            }
        ]
        return AgentState(
            messages=appended,
            token_count=response.usage.prompt_tokens + response.usage.completion_tokens,
        )

    async def _compaction_node(self, state: AgentState) -> AgentState:
        """LangGraph node that delegates to :class:`MemoryManager`."""
        essentials = state.get("essential_entities") or {}
        return await self._memory.compact(
            state,
            tenant_id=str(essentials.get("tenant_id") or "") or None,
            agent_id=self.agent_id,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _router_should_compact(self, state: AgentState) -> bool:
        """Conditional edge: True → compact, False → agent."""
        return self._memory.should_compact(state)


__all__ = ["BaseAgent"]
