"""Pluggable resolution of an agent's local + MCP tools.

Pre-v1, :class:`BaseAgent._all_tools` did three things at once:

* called :func:`resolve_mcp_tools` / :func:`resolve_mcp_resources` for
  every declared MCP server,
* enforced naming invariants (local vs MCP, MCP-tool vs MCP-resource),
* registered each resolved :class:`MCPToolSpec` with the
  :class:`ToolInvoker` so the runtime can dispatch them later.

That mixed orchestration with I/O and registry side-effects, leaving
hosts no clean seam to override resolution behaviour (e.g. caching MCP
results across agents, redacting tools by tenant, or stubbing the MCP
backend in tests). This module extracts the resolution policy behind
:class:`IToolResolver` and ships :class:`DefaultToolResolver` as the
standard implementation.

The conflict-detection rules and registration side-effects are
preserved exactly so existing :class:`BaseAgent` subclasses see no
behavioural change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ai_core.exceptions import RegistryError
from ai_core.mcp.resolver import resolve_mcp_resources, resolve_mcp_tools
from ai_core.mcp.tools import MCPResourceSpec, MCPToolSpec
from ai_core.tools.spec import Tool, ToolSpec

if TYPE_CHECKING:
    from ai_core.mcp.transports import IMCPConnectionFactory, MCPServerSpec
    from ai_core.tools.invoker import ToolInvoker


@runtime_checkable
class IToolResolver(Protocol):
    """Resolves an agent's declared tools to a concrete dispatchable list.

    A resolver is responsible for:

    * fanning out to declared MCP servers and merging their tools and
      resources into the local tool set,
    * raising :class:`ai_core.exceptions.RegistryError` on naming
      conflicts (local vs MCP, MCP-tool vs MCP-resource), and
    * registering every resolved :class:`MCPToolSpec` with the
      :class:`ToolInvoker` so dispatch can find it later.

    Hosts that want to cache MCP resolutions across agents, redact tools
    based on tenant, or fake the MCP backend in tests bind a custom
    :class:`IToolResolver` via DI.
    """

    async def resolve(
        self,
        *,
        local_tools: Sequence[Tool | Mapping[str, Any]],
        mcp_servers: Sequence[MCPServerSpec],
    ) -> Sequence[MCPToolSpec]:
        """Resolve MCP-derived specs and register them with the invoker.

        Returns:
            The list of :class:`MCPToolSpec` instances produced by the
            declared MCP servers (already registered with
            :class:`ToolInvoker`). The agent merges these with
            ``local_tools`` itself.
        """
        ...


class DefaultToolResolver:
    """Pre-v1 :class:`BaseAgent._all_tools` behaviour, packaged as a service.

    Args:
        mcp_factory: Connection factory used to open MCP transports.
        tool_invoker: Invoker used to register resolved MCP specs so the
            runtime can dispatch them on subsequent turns.
    """

    def __init__(
        self,
        mcp_factory: IMCPConnectionFactory,
        tool_invoker: ToolInvoker,
    ) -> None:
        self._mcp_factory = mcp_factory
        self._tool_invoker = tool_invoker

    async def resolve(
        self,
        *,
        local_tools: Sequence[Tool | Mapping[str, Any]],
        mcp_servers: Sequence[MCPServerSpec],
    ) -> Sequence[MCPToolSpec]:
        servers = list(mcp_servers)
        if not servers:
            return ()

        tools_resolved = await resolve_mcp_tools(servers, self._mcp_factory)
        resources_resolved = await resolve_mcp_resources(servers, self._mcp_factory)
        resolved: list[MCPToolSpec] = list(tools_resolved) + list(resources_resolved)

        local_names = {t.name for t in local_tools if isinstance(t, ToolSpec)}
        mcp_names_seen: set[str] = set()
        for mcp_spec in resolved:
            if mcp_spec.name in local_names:
                kind = (
                    "resource" if isinstance(mcp_spec, MCPResourceSpec) else "tool"
                )
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
            self._tool_invoker.register(mcp_spec)
        return resolved


__all__ = ["DefaultToolResolver", "IToolResolver"]
