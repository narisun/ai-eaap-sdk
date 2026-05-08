"""Bulk registration of :class:`ToolSpec` with the :class:`ToolInvoker`.

Before v1, :class:`BaseAgent.compile` mutated the schema registry as a
side effect of building the LangGraph. That coupled graph construction
to registry state and made it harder to reason about which tools an
invoker actually knew about. :class:`ToolRegistrar` extracts the
registration step into a single explicit call so :meth:`compile`
becomes pure graph assembly.

The behaviour is unchanged: each :class:`ToolSpec` is registered
idempotently with the underlying :class:`ToolInvoker` (which in turn
delegates to the :class:`SchemaRegistry`). Hosts that want to gate
registration (e.g. by tenant or feature flag) bind a custom
implementation conforming to the :data:`__call__` shape used by
:class:`BaseAgent`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_core.tools.invoker import ToolInvoker
    from ai_core.tools.spec import ToolSpec


class ToolRegistrar:
    """Centralises the ``ToolInvoker.register`` calls scattered across BaseAgent.

    Args:
        tool_invoker: The runtime invoker whose schema registry is the
            target of registration.
    """

    def __init__(self, tool_invoker: ToolInvoker) -> None:
        self._tool_invoker = tool_invoker

    def register_all(self, specs: Iterable[ToolSpec]) -> None:
        """Register every spec in ``specs``. Idempotent.

        Args:
            specs: Iterable of :class:`ToolSpec` to register. Mixing
                already-registered specs with new ones is safe — the
                underlying registry treats matching ``(name, version)``
                as a no-op.
        """
        for spec in specs:
            self._tool_invoker.register(spec)


__all__ = ["ToolRegistrar"]
