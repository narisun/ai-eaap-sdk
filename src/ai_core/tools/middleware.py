"""Around-advice middleware for :class:`ToolInvoker`.

The :class:`ToolInvoker` ships a fixed pipeline (input validation → OPA
enforcement + audit → handler → output validation → audit) that covers
every requirement the platform team has agreed on so far. Hosts that
need additional cross-cutting concerns — per-tool rate limiting, request
sandboxing, PII scrubbing on input, structured-output repair, dev-only
breadcrumb logging — pre-v1 had to either subclass :class:`ToolInvoker`
(and copy the entire pipeline) or wrap every tool's handler manually
before registering.

This module introduces :class:`ToolMiddleware`, a thin
chain-of-responsibility hook that **wraps** the built-in pipeline. The
SDK's own validation / policy / audit logic does not change shape; it
runs inside the innermost layer. Hosts register middlewares via DI
multibind:

    class MyToolMiddlewares(Module):
        @multiprovider
        def provide_extras(self) -> list[ToolMiddleware]:
            return [PerTenantRateLimiter(), PIIScrubber()]

    async with AICoreApp(modules=[MyToolMiddlewares()]) as app:
        ...

The default :class:`AgentModule` contributes an empty list, so
applications that don't register middlewares run with the exact same
:class:`ToolInvoker` behaviour as pre-v1.

Ordering rules
--------------
* Multibind preserves provider order: middlewares from the first module
  that contributes are at the start of the chain (outermost layer);
  later contributions are wrapped inside.
* Each middleware sees the same :class:`ToolCallContext` instance and
  decides when to call ``await call_next()`` — before, after, around,
  or not at all (early-return / short-circuit).
* If a middleware raises, the SDK pipeline below it is skipped; the
  agent's :class:`IToolErrorRenderer` will turn the exception into the
  next-turn ``ToolMessage``.
* Successful return values must be JSON-serialisable mappings — the
  same contract as :class:`ToolInvoker.invoke`'s return type — because
  downstream code feeds the result back to the LLM verbatim.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ai_core.tools.spec import ToolSpec


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    """Per-invocation snapshot passed to every middleware.

    Attributes:
        spec: The :class:`ToolSpec` being dispatched.
        raw_args: The raw argument mapping the LLM produced (pre-validation).
        principal: Optional caller-identity mapping (matches the
            :meth:`ToolInvoker.invoke` ``principal`` argument).
        agent_id: Logical agent identifier from the calling agent.
        tenant_id: Tenant identifier propagated from the agent's state.
    """

    spec: ToolSpec
    raw_args: Mapping[str, Any]
    principal: Mapping[str, Any] | None
    agent_id: str | None
    tenant_id: str | None


#: Continuation passed to a middleware. Calling it runs the next
#: middleware (or the built-in pipeline if this is the innermost wrap).
MiddlewareNext = Callable[[], Awaitable[Mapping[str, Any]]]


@runtime_checkable
class ToolMiddleware(Protocol):
    """Around-advice hook wrapping the built-in :class:`ToolInvoker` pipeline.

    Implementations decide whether to call ``call_next``, transform its
    result, short-circuit with their own response, or instrument the
    surrounding span.
    """

    async def __call__(
        self,
        ctx: ToolCallContext,
        call_next: MiddlewareNext,
    ) -> Mapping[str, Any]:
        """Wrap one tool dispatch.

        Args:
            ctx: Snapshot of the incoming call.
            call_next: Continuation; ``await``-ing it runs the next
                middleware in the chain (or the built-in pipeline at
                the innermost layer).

        Returns:
            The tool's JSON-serialisable result mapping.
        """
        ...


__all__ = ["MiddlewareNext", "ToolCallContext", "ToolMiddleware"]
