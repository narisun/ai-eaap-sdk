"""Frozen service bag injected into :class:`BaseAgent` subclasses.

:class:`AgentRuntime` collects every SDK-provided collaborator a
LangGraph-driven agent needs (LLM client, memory manager, tool invoker,
observability, MCP factory, agent-runtime configuration). Subclasses of
:class:`ai_core.agents.BaseAgent` receive a single ``runtime`` argument
through DI, freeing them to add their own dependencies via ``@inject``
without copying or mirroring the SDK's growing service surface.

Why composition rather than a fat ``__init__``
----------------------------------------------
Pre-v1, :class:`BaseAgent` exposed a six-argument ``@inject`` constructor.
A subclass that wanted its own dependency had three bad choices:

* repeat all six arguments verbatim (and break every time the SDK added
  a seventh collaborator),
* skip ``@inject`` and lose DI for the new dependency, or
* abandon :class:`BaseAgent` entirely.

By boxing the SDK collaborators into an :class:`AgentRuntime`, adding a
new SDK collaborator becomes a non-breaking change: subclasses keep
``def __init__(self, runtime: AgentRuntime, my_db: DBSession) -> None``
unchanged.

Stability contract
------------------
:class:`AgentRuntime` is frozen and slotted; the field set is part of
the public SDK surface and changes only on major-version bumps.
Field additions are backwards-compatible (subclasses pass-through via
``super().__init__(runtime)``); removals are not. New SDK collaborators
land here first; in-place mutation is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # IMemoryManager / IToolErrorRenderer / AgentResolver live in sibling
    # modules under ai_core.agents.* — declared lazily here to avoid the
    # import cycles that would arise if runtime.py loaded them at module
    # init time.
    from ai_core.agents._resolver import AgentResolver
    from ai_core.agents.memory import IMemoryManager
    from ai_core.agents.tool_errors import IToolErrorRenderer
    from ai_core.config.settings import AgentSettings
    from ai_core.di.interfaces import ILLMClient, IObservabilityProvider
    from ai_core.mcp.transports import IMCPConnectionFactory
    from ai_core.tools.invoker import ToolInvoker
    from ai_core.tools.registrar import ToolRegistrar
    from ai_core.tools.resolver import IToolResolver


@dataclass(frozen=True, slots=True)
class AgentRuntime:
    """Bundle of SDK services injected into every :class:`BaseAgent`.

    Attributes:
        agent_settings: Agent-runtime configuration (compaction thresholds,
            recursion depth, essential entity keys).
        llm: LLM client used by the agent node for chat completions.
        memory: Memory manager invoked by the compaction node.
        observability: Span + event sink for tracing and metrics.
        tool_invoker: Validated, policy-aware tool dispatcher.
        mcp_factory: Factory that opens MCP server connections lazily.
        tool_error_renderer: Strategy that turns tool-dispatch failures
            into ``ToolMessage`` instances for the next LLM turn. Override
            via DI for strict-failure semantics or localized text.
        tool_resolver: Strategy for resolving an agent's local tools and
            declared MCP servers into a single dispatchable list. Default
            implementation reproduces pre-v1 behaviour.
        tool_registrar: Strategy for registering local :class:`ToolSpec`
            instances with the invoker at compile time. Extracted from
            :meth:`BaseAgent.compile` so graph construction stays pure.
        agent_resolver: DI-aware resolver for sub-agents. Used by
            compositional patterns (:class:`SupervisorAgent` and
            friends) to resolve child :class:`BaseAgent` instances at
            runtime without leaking the container into agent code.
    """

    agent_settings: AgentSettings
    llm: ILLMClient
    memory: IMemoryManager
    observability: IObservabilityProvider
    tool_invoker: ToolInvoker
    mcp_factory: IMCPConnectionFactory
    tool_error_renderer: IToolErrorRenderer
    tool_resolver: IToolResolver
    tool_registrar: ToolRegistrar
    agent_resolver: AgentResolver


__all__ = ["AgentRuntime"]
