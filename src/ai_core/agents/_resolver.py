"""DI-aware resolver for sub-agents.

:class:`SupervisorAgent` and other compositional patterns need to resolve
child :class:`BaseAgent` instances at runtime without binding the
container into agent code directly. :class:`AgentResolver` wraps the
container behind a narrow interface so:

* sub-agent resolution stays a single explicit seam (no container leaks
  scattered through agent code), and
* tests can swap the resolver for a fake (e.g. ``FakeAgentResolver`` that
  returns a pre-baked dict of agents) without standing up a real DI
  container.

Each ``resolve`` call returns whatever scope the container's binding
produces. With v1.0's auto-bind default that's a fresh instance per
call; hosts that want shared instances across a session bind the agent
class as ``@singleton`` in their own module, and the resolver returns
the cached instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ai_core.agents.base import BaseAgent
    from ai_core.di.container import Container

A = TypeVar("A", bound="BaseAgent")


class AgentResolver:
    """Resolve :class:`BaseAgent` subclasses through the DI container.

    Args:
        container: The DI container to delegate resolution to. Provided
            via ``AgentModule.provide_agent_resolver`` so the resolver
            and the runtime share the same container instance.
    """

    __slots__ = ("_container",)

    def __init__(self, container: Container) -> None:
        self._container = container

    def resolve(self, cls: type[A]) -> A:
        """Return an instance of the agent class.

        Args:
            cls: A concrete :class:`BaseAgent` subclass.

        Returns:
            A fully-DI-resolved instance with its own
            :class:`AgentRuntime` (and therefore its own observability
            span context, budget binding, and policy evaluator).

        Raises:
            DependencyResolutionError: If ``cls`` cannot be resolved
                (e.g. its constructor needs a dependency that is not
                bound).
        """
        return self._container.get(cls)


__all__ = ["AgentResolver"]
