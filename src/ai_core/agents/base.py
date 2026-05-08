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

import asyncio
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

from ai_core.agents.memory import to_openai_messages
from ai_core.agents.runtime import AgentRuntime
from ai_core.agents.state import AgentState, new_agent_state
from ai_core.exceptions import (
    AgentRecursionLimitError,
    PolicyDenialError,
    RegistryError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.mcp.prompts import MCPPrompt, MCPPromptArgument, MCPPromptMessage
from ai_core.mcp.resolver import (
    is_method_not_found,
    resolve_mcp_resources,
    resolve_mcp_tools,
)
from ai_core.mcp.tools import MCPResourceSpec, MCPToolSpec
from ai_core.mcp.transports import MCPServerSpec  # noqa: TC001
from ai_core.observability.logging import bind_context, get_logger, unbind_context
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

    Subclasses receive a single :class:`AgentRuntime` argument carrying every
    SDK collaborator (LLM client, memory manager, tool invoker, observability,
    MCP factory, agent settings). Subclasses that need additional dependencies
    inject them alongside the runtime::

        class MyAgent(BaseAgent):
            @inject
            def __init__(
                self,
                runtime: AgentRuntime,
                repo: MyRepository,
            ) -> None:
                super().__init__(runtime)
                self._repo = repo

    Args:
        runtime: Bundle of SDK services. Constructed by the DI container via
            :func:`AgentModule.provide_agent_runtime`.
    """

    #: Logical identifier — override in subclasses (used for budgeting + tracing).
    agent_id: str = "base-agent"

    #: When True, ``compile()`` auto-installs a tool dispatch loop whenever
    #: ``tools()`` returns at least one ``Tool``-protocol object. Subclasses
    #: that prefer to wire their own tool handling can set this to False.
    auto_tool_loop: bool = True

    @inject
    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime
        self._graph: Any | None = None
        self._mcp_resolved: list[MCPToolSpec] | None = None
        self._mcp_resolution_lock: asyncio.Lock = asyncio.Lock()

    @property
    def runtime(self) -> AgentRuntime:
        """Return the bundle of SDK services injected at construction time."""
        return self._runtime

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""

    def tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Return tool definitions or ``Tool``-protocol objects (default: empty)."""
        return ()

    def mcp_servers(self) -> Sequence[MCPServerSpec]:
        """Return MCPServerSpecs whose tools the agent should use (default: empty).

        Resolved on the first agent turn via `_all_tools()`. Tools surfaced by
        these servers are merged with `tools()` for both LLM advertising and
        dispatch.
        """
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
            self._runtime.tool_invoker.register(spec)

        install_loop = self.auto_tool_loop and (bool(sdk_tools) or bool(list(self.mcp_servers())))

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

            recursion_limit = self._runtime.agent_settings.max_recursion_depth
            config: dict[str, Any] = {"recursion_limit": recursion_limit}
            if thread_id is not None:
                config["configurable"] = {"thread_id": thread_id}

            attributes = {"agent.id": self.agent_id, "agent.tenant_id": tenant_id or ""}

            # Stash agent context in OTel Baggage so every downstream span (LLM
            # call, tool call, MCP request) is automatically tagged for
            # cross-tenant aggregation in OTel + LangFuse.
            token = otel_context.attach(self._build_baggage(tenant_id, thread_id, essential))
            try:
                async with self._runtime.observability.start_span("agent.ainvoke", attributes=attributes):
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
        for t in await self._all_tools():
            if isinstance(t, ToolSpec):
                tool_payload.append(t.openai_schema())
            elif isinstance(t, Mapping):
                tool_payload.append(t)

        response = await self._runtime.llm.complete(
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
        """Dispatch all tool calls on the most recent assistant message.

        Each tool-call failure is rendered into a ``ToolMessage`` via the
        injected :class:`IToolErrorRenderer` so the LLM gets feedback on
        the next turn rather than a short-circuited graph.
        """
        history = list(state.get("messages") or [])
        last = history[-1] if history else None
        tool_calls = getattr(last, "tool_calls", None) or []
        sdk_tools_by_name: dict[str, ToolSpec] = {
            t.name: t for t in await self._all_tools() if isinstance(t, ToolSpec)
        }
        essentials = state.get("essential_entities") or {}
        tenant_id = str(essentials.get("tenant_id") or "") or None
        renderer = self._runtime.tool_error_renderer

        appended: list[Any] = []
        for tc in tool_calls:
            tc_id = tc.get("id") if isinstance(tc, Mapping) else getattr(tc, "id", "")
            name = tc.get("name") if isinstance(tc, Mapping) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, Mapping) else getattr(tc, "args", {}) or {}
            tool_call_id = tc_id or ""
            if isinstance(args, Mapping) and "__parse_error__" in args:
                appended.append(renderer.render_parse_error(
                    tool_name=name,
                    tool_call_id=tool_call_id,
                    raw=str(args["__parse_error__"]),
                ))
                continue
            spec = sdk_tools_by_name.get(name)
            if spec is None:
                appended.append(renderer.render_unknown_tool(
                    tool_name=name, tool_call_id=tool_call_id,
                ))
                continue
            try:
                result = await self._runtime.tool_invoker.invoke(
                    spec,
                    args if isinstance(args, Mapping) else {},
                    agent_id=self.agent_id,
                    tenant_id=tenant_id,
                )
                appended.append(ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call_id,
                    name=name,
                ))
            except ToolValidationError as exc:
                appended.append(renderer.render_validation_error(
                    tool_name=name, tool_call_id=tool_call_id, error=exc,
                ))
            except PolicyDenialError as exc:
                appended.append(renderer.render_policy_denial(
                    tool_name=name, tool_call_id=tool_call_id, error=exc,
                ))
            except ToolExecutionError as exc:
                _logger.error(
                    "tool.execution_error",
                    tool_name=name, agent_id=self.agent_id,
                    exc_info=exc,
                )
                appended.append(renderer.render_execution_error(
                    tool_name=name, tool_call_id=tool_call_id, error=exc,
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
        return await self._runtime.memory.compact(
            state,
            tenant_id=str(essentials.get("tenant_id") or "") or None,
            agent_id=self.agent_id,
        )

    async def _all_tools(self) -> list[Tool | Mapping[str, Any]]:
        """Return the merged list of local + resolved MCP tools + resources.

        Lazily resolves MCP servers on the first call; caches per-instance.
        Concurrent first-turn callers serialize on `_mcp_resolution_lock`.

        Raises:
            MCPTransportError: When a declared MCP server is unreachable.
            RegistryError: When MCP names conflict with each other or with
                local @tool names. Conflicts span tools-vs-tools, tools-vs-resources,
                and any of those vs local @tools.
        """
        if self._mcp_resolved is None:
            async with self._mcp_resolution_lock:
                if self._mcp_resolved is None:
                    servers = list(self.mcp_servers())
                    if servers:
                        tools_resolved = await resolve_mcp_tools(servers, self._runtime.mcp_factory)
                        resources_resolved = await resolve_mcp_resources(servers, self._runtime.mcp_factory)
                        resolved: list[MCPToolSpec] = (
                            list(tools_resolved) + list(resources_resolved)
                        )
                    else:
                        resolved = []
                    local_names = {
                        t.name for t in self.tools() if isinstance(t, ToolSpec)
                    }
                    mcp_names_seen: set[str] = set()
                    for mcp_spec in resolved:
                        if mcp_spec.name in local_names:
                            kind = "resource" if isinstance(mcp_spec, MCPResourceSpec) else "tool"
                            raise RegistryError(
                                f"MCP {kind} name {mcp_spec.name!r} conflicts with a local tool",
                                details={"tool": mcp_spec.name},
                            )
                        if mcp_spec.name in mcp_names_seen:
                            raise RegistryError(
                                f"MCP name {mcp_spec.name!r} appears in both tools and resources "
                                f"on declared servers",
                                details={"name": mcp_spec.name},
                            )
                        mcp_names_seen.add(mcp_spec.name)
                        self._runtime.tool_invoker.register(mcp_spec)
                    self._mcp_resolved = resolved
        return list(self.tools()) + list(self._mcp_resolved)

    async def list_prompts(self) -> list[MCPPrompt]:
        """List all prompts across declared MCP servers.

        Fetched fresh each call (no cache — application-invoked patterns vary,
        consistency matters more than throughput).

        Returns:
            One MCPPrompt per discovered prompt, with its origin server tagged.

        Raises:
            RegistryError: When two servers expose prompts with the same name.
            MCPTransportError: Propagated from the connection factory.
        """
        out: list[MCPPrompt] = []
        seen_names: set[str] = set()
        for server in self.mcp_servers():
            async with self._runtime.mcp_factory.open(server) as client:
                try:
                    prompts = await client.list_prompts()
                except Exception as exc:  # noqa: BLE001, RUF100 — narrow via predicate
                    if is_method_not_found(exc):
                        continue
                    raise
            for p in prompts:
                if p.name in seen_names:
                    raise RegistryError(
                        f"MCP prompt name {p.name!r} appears in multiple servers",
                        details={"name": p.name, "server": server.component_id},
                    )
                seen_names.add(p.name)
                out.append(_to_mcp_prompt(p, server))
        return out

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        server: str | None = None,
    ) -> list[MCPPromptMessage]:
        """Fetch a templated prompt's messages by name.

        Args:
            name: The prompt's name (must match what `list_prompts()` returned).
            arguments: Argument dict to substitute into the template.
            server: Optional component_id of the server known to host the prompt.
                When omitted, the agent searches across declared servers (one
                list_prompts call each until found).

        Round trips: when `server` is provided, exactly two RPCs (one
        list_prompts to verify, one get_prompt to fetch). When omitted with
        N declared servers, up to 2N round trips in the worst case (prompt
        on the last server checked). Use `server=...` for prompt fetches in
        hot paths.

        Returns:
            List of MCPPromptMessage instances, ready to splice into
            ainvoke(messages=...).

        Raises:
            RegistryError: When the prompt is not found on any declared server.
            MCPTransportError: Propagated from the connection factory.
        """
        for srv in self.mcp_servers():
            if server is not None and srv.component_id != server:
                continue
            async with self._runtime.mcp_factory.open(srv) as client:
                try:
                    prompts = await client.list_prompts()
                except Exception as exc:  # noqa: BLE001, RUF100 — narrow via predicate
                    if is_method_not_found(exc):
                        continue
                    raise
                if not any(p.name == name for p in prompts):
                    continue
                result = await client.get_prompt(name, dict(arguments or {}))
            return [_to_mcp_prompt_message(m) for m in result.messages]
        raise RegistryError(
            f"MCP prompt {name!r} not found in any declared server",
            details={"name": name, "hint_server": server},
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _router_should_compact(self, state: AgentState) -> bool:
        """Conditional edge: True → compact, False → agent."""
        return self._runtime.memory.should_compact(state)

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


def _to_mcp_prompt(fastmcp_prompt: Any, server: MCPServerSpec) -> MCPPrompt:  # noqa: ANN401
    """Map a FastMCP `Prompt` to our typed MCPPrompt."""
    args = tuple(
        MCPPromptArgument(
            name=a.name,
            description=getattr(a, "description", None),
            required=getattr(a, "required", False),
        )
        for a in (getattr(fastmcp_prompt, "arguments", None) or [])
    )
    return MCPPrompt(
        name=fastmcp_prompt.name,
        description=getattr(fastmcp_prompt, "description", None),
        arguments=args,
        mcp_server_spec=server,
    )


def _to_mcp_prompt_message(fastmcp_msg: Any) -> MCPPromptMessage:  # noqa: ANN401
    """Map a FastMCP `PromptMessage` to our typed MCPPromptMessage.

    PromptMessage.content is a union of TextContent | ImageContent | … .
    v1 only handles TextContent; other types yield empty content with a
    debug-level log (parallel to resource binary content suppression in
    resolver.py).
    """
    content_obj = getattr(fastmcp_msg, "content", None)
    text = ""
    role = str(fastmcp_msg.role)
    if content_obj is not None and getattr(content_obj, "type", None) == "text":
        text = getattr(content_obj, "text", "") or ""
    elif content_obj is not None:
        _logger.debug(
            "mcp.prompt.non_text_content_dropped",
            role=role,
            content_type=getattr(content_obj, "type", "unknown"),
        )
    return MCPPromptMessage(role=role, content=text)


__all__ = ["BaseAgent"]
