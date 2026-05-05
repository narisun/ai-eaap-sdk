"""Abstract :class:`BaseAgent` integrating LangGraph + SDK services.

A subclass implements three things:

1. :meth:`tools` — return either SDK ``Tool``-protocol objects (e.g. via
   :func:`@tool`) or raw OpenAI-style dicts (or ``()``).
2. :meth:`system_prompt` — return the system prompt for the agent.
3. *(optional)* :meth:`extend_graph` — add custom nodes/edges to the
   compiled :class:`StateGraph` before it is finalised.

The base class wires:

* an ``agent`` node that calls :class:`ILLMClient` with the system prompt
  prepended to the conversation;
* a ``compaction`` node that delegates to
  :class:`IMemoryManager.compact`;
* a router that runs compaction *before* each agent turn whenever
  :meth:`IMemoryManager.should_compact` returns ``True``;
* OpenTelemetry **Baggage** propagation for ``agent_id`` / ``tenant_id``
  / ``user_id`` / ``session_id`` / ``task_id`` / ``thread_id`` so that
  every downstream span (and LangFuse trace) is automatically tagged
  for cross-tenant aggregation;
* an explicit ``recursion_limit`` from
  :attr:`AgentSettings.max_recursion_depth` so a confused LLM cannot
  spin in an infinite loop;
* an optional checkpointer (LangGraph-native) supplied by callers.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from injector import inject
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from opentelemetry import baggage
from opentelemetry import context as otel_context

from ai_core.agents.memory import IMemoryManager, to_openai_messages
from ai_core.agents.state import AgentState, new_agent_state
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import ILLMClient, IObservabilityProvider
from ai_core.exceptions import (
    AgentRecursionLimitError,
    PolicyDenialError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.observability.logging import bind_context, get_logger, unbind_context
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.spec import Tool, ToolSpec

_logger = get_logger(__name__)


def _parse_tool_call_args(arguments: str | None) -> dict[str, Any]:
    """Parse JSON tool-call arguments; return a sentinel dict on malformed input.

    Args:
        arguments: Raw JSON string from the LLM's ``function.arguments`` field.

    Returns:
        A dict of parsed arguments, or ``{"__parse_error__": <raw>}`` when the
        string is not valid JSON or does not decode to a mapping.
    """
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"__parse_error__": arguments}
    return parsed if isinstance(parsed, dict) else {"__parse_error__": str(parsed)}


class BaseAgent(ABC):
    """Abstract base class for SDK-built LangGraph agents.

    Args:
        settings: Aggregated application settings.
        llm: LLM client used by the agent node.
        memory: Memory manager (interface) used by the compaction node.
        observability: Provider used to wrap top-level invocations in spans.
        tool_invoker: Invoker used to dispatch SDK tools through the
            validation → policy → handler pipeline.
    """

    #: Logical identifier — override in subclasses (used for budgeting + tracing).
    agent_id: str = "base-agent"

    #: When True, ``compile()`` auto-installs a tool dispatch loop whenever
    #: ``tools()`` returns at least one ``Tool``-protocol object. Subclasses
    #: that prefer to wire their own tool handling can set this to False.
    auto_tool_loop: bool = True

    @inject
    def __init__(
        self,
        settings: AppSettings,
        llm: ILLMClient,
        memory: IMemoryManager,
        observability: IObservabilityProvider,
        tool_invoker: ToolInvoker,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._memory = memory
        self._observability = observability
        self._tool_invoker = tool_invoker
        self._graph: Any | None = None

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""

    def tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Return tool definitions or ``Tool``-protocol objects (default: empty)."""
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
        graph: StateGraph[AgentState] = StateGraph(AgentState)
        graph.add_node("compact", self._compaction_node)
        graph.add_node("agent", self._agent_node)

        sdk_tools = [t for t in self.tools() if isinstance(t, ToolSpec)]

        # Phase 2: auto-register each ToolSpec with the SchemaRegistry so that
        # `app.register_tools(*specs)` is optional. Idempotent — re-compile is fine.
        for spec in sdk_tools:
            self._tool_invoker.register(spec)

        install_loop = self.auto_tool_loop and bool(sdk_tools)

        # START routing is invariant across branches.
        graph.add_conditional_edges(
            START,
            self._router_should_compact,
            {True: "compact", False: "agent"},
        )
        graph.add_edge("compact", "agent")

        if install_loop:
            graph.add_node("tool", self._tool_node)
            graph.add_conditional_edges(
                "agent",
                self._router_after_agent,
                {True: "tool", False: END},
            )
            graph.add_edge("tool", "agent")
        else:
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

        Raises:
            AgentRecursionLimitError: If the graph exceeds
                :attr:`AgentSettings.max_recursion_depth`.
        """
        log_token = bind_context(
            agent_id=self.agent_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
        )
        try:
            compiled = self.compile()
            initial = new_agent_state(
                initial_messages=list(messages),
                essential={**(essential or {}), "tenant_id": tenant_id or ""},
                metadata={"agent_id": self.agent_id},
            )

            recursion_limit = self._settings.agent.max_recursion_depth
            config: dict[str, Any] = {"recursion_limit": recursion_limit}
            if thread_id is not None:
                config["configurable"] = {"thread_id": thread_id}

            attributes = {"agent.id": self.agent_id, "agent.tenant_id": tenant_id or ""}

            # Stash agent context in OTel Baggage so every downstream span (LLM
            # call, tool call, MCP request) is automatically tagged for
            # cross-tenant aggregation in OTel + LangFuse.
            token = otel_context.attach(self._build_baggage(tenant_id, thread_id, essential))
            try:
                async with self._observability.start_span("agent.ainvoke", attributes=attributes):
                    try:
                        result = await compiled.ainvoke(initial, config=config)
                    except GraphRecursionError as exc:
                        raise AgentRecursionLimitError(
                            "Agent exceeded recursion limit",
                            details={
                                "agent_id": self.agent_id,
                                "tenant_id": tenant_id,
                                "thread_id": thread_id,
                                "limit": recursion_limit,
                            },
                            cause=exc,
                        ) from exc
            finally:
                otel_context.detach(token)
            return result
        finally:
            unbind_context(log_token)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    async def _agent_node(self, state: AgentState) -> AgentState:
        """LangGraph node that performs one LLM turn."""
        history = list(state.get("messages") or [])
        prompt: list[Mapping[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            *to_openai_messages(history),
        ]
        essentials = state.get("essential_entities") or {}

        tool_payload: list[Mapping[str, Any]] = []
        for t in self.tools():
            if isinstance(t, ToolSpec):
                tool_payload.append(t.openai_schema())
            elif isinstance(t, Mapping):
                tool_payload.append(t)

        response = await self._llm.complete(
            model=None,
            messages=prompt,
            tools=tool_payload or None,
            tenant_id=str(essentials.get("tenant_id") or "") or None,
            agent_id=self.agent_id,
        )
        appended: list[Any]
        if response.tool_calls:
            appended = [AIMessage(
                content=response.content,
                tool_calls=[
                    {
                        "id": tc.get("id") or f"call-{i}",
                        "name": tc.get("function", {}).get("name", ""),
                        "args": _parse_tool_call_args(tc.get("function", {}).get("arguments")),
                    }
                    for i, tc in enumerate(response.tool_calls)
                ],
            )]
        else:
            appended = [AIMessage(content=response.content)]
        return AgentState(
            messages=appended,
            token_count=response.usage.prompt_tokens + response.usage.completion_tokens,
        )

    async def _tool_node(self, state: AgentState) -> AgentState:
        """Dispatch all tool calls on the most recent assistant message."""
        history = list(state.get("messages") or [])
        last = history[-1] if history else None
        tool_calls = getattr(last, "tool_calls", None) or []
        sdk_tools_by_name: dict[str, ToolSpec] = {
            t.name: t for t in self.tools() if isinstance(t, ToolSpec)
        }
        essentials = state.get("essential_entities") or {}
        tenant_id = str(essentials.get("tenant_id") or "") or None

        appended: list[Any] = []
        for tc in tool_calls:
            tc_id = tc.get("id") if isinstance(tc, Mapping) else getattr(tc, "id", "")
            name = tc.get("name") if isinstance(tc, Mapping) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, Mapping) else getattr(tc, "args", {}) or {}
            if isinstance(args, Mapping) and "__parse_error__" in args:
                raw = args["__parse_error__"]
                appended.append(ToolMessage(
                    content=f"Tool '{name}' arguments were not valid JSON: {raw!r}",
                    tool_call_id=tc_id or "",
                    name=name,
                ))
                continue
            spec = sdk_tools_by_name.get(name)
            if spec is None:
                appended.append(ToolMessage(
                    content=f"Unknown tool '{name}'.",
                    tool_call_id=tc_id or "",
                    name=name or "",
                ))
                continue
            try:
                result = await self._tool_invoker.invoke(
                    spec,
                    args if isinstance(args, Mapping) else {},
                    agent_id=self.agent_id,
                    tenant_id=tenant_id,
                )
                appended.append(ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tc_id or "",
                    name=name,
                ))
            except ToolValidationError as exc:
                first_err = exc.details.get("errors", [{}])[0] if exc.details.get("errors") else {}
                msg = first_err.get("msg") if isinstance(first_err, Mapping) else None
                appended.append(ToolMessage(
                    content=f"Validation failed for '{name}': {msg or exc.message}",
                    tool_call_id=tc_id or "",
                    name=name,
                ))
            except PolicyDenialError as exc:
                reason = exc.details.get("reason") or exc.message
                appended.append(ToolMessage(
                    content=f"Tool '{name}' denied by policy: {reason}",
                    tool_call_id=tc_id or "",
                    name=name,
                ))
            except ToolExecutionError as exc:
                _logger.error(
                    "tool.execution_error",
                    tool_name=name, agent_id=self.agent_id,
                    exc_info=exc,
                )
                appended.append(ToolMessage(
                    content=f"Tool '{name}' failed: {exc.message}",
                    tool_call_id=tc_id or "",
                    name=name,
                ))
        return AgentState(messages=appended)

    def _router_after_agent(self, state: AgentState) -> bool:
        """True -> there is at least one outstanding tool_call to dispatch."""
        history = list(state.get("messages") or [])
        last = history[-1] if history else None
        return bool(getattr(last, "tool_calls", None))

    async def _compaction_node(self, state: AgentState) -> AgentState:
        """LangGraph node that delegates to :class:`IMemoryManager`."""
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

    # ------------------------------------------------------------------
    # Baggage
    # ------------------------------------------------------------------
    def _build_baggage(
        self,
        tenant_id: str | None,
        thread_id: str | None,
        essential: Mapping[str, Any] | None,
    ) -> Any:
        """Construct an OTel Context with ``eaap.*`` Baggage entries set.

        The returned context can be attached with :func:`otel_context.attach`
        so every downstream span observes the same agent attribution
        without per-call injection.
        """
        ctx = otel_context.get_current()
        ctx = baggage.set_baggage("eaap.agent_id", self.agent_id, context=ctx)
        if tenant_id:
            ctx = baggage.set_baggage("eaap.tenant_id", tenant_id, context=ctx)
        if thread_id:
            ctx = baggage.set_baggage("eaap.thread_id", thread_id, context=ctx)
        for key in ("user_id", "session_id", "task_id"):
            value = (essential or {}).get(key)
            if value:
                ctx = baggage.set_baggage(f"eaap.{key}", str(value), context=ctx)
        return ctx


__all__ = ["BaseAgent"]
