"""Dependency-injection sub-package.

The DI layer is the *only* place where concrete classes are bound to
abstract interfaces. Application code MUST depend on the abstractions
exposed by :mod:`ai_core.di.interfaces` and resolve them through the
container exposed by :mod:`ai_core.di.container`.

Typical bootstrap (in a host service)::

    from ai_core.di import Container, AgentModule

    container = Container.build(modules=[AgentModule()])
    agent = container.get(MyAgent)            # MyAgent's deps auto-wired

Implementation note:
    :class:`AgentModule` imports concrete classes from every domain
    sub-package (``llm``, ``agents``, ``persistence``, …). Several of
    those modules in turn import the interfaces declared here. To keep
    that import graph acyclic we expose :class:`AgentModule` via
    :pep:`562` ``__getattr__`` instead of an eager re-export. Callers
    still write ``from ai_core.di import AgentModule``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_core.config.secrets import ISecretManager
from ai_core.di.container import Container
from ai_core.di.interfaces import (
    IBudgetService,
    ICheckpointSaver,
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
    IStorageProvider,
)

if TYPE_CHECKING:
    from ai_core.di.module import AgentModule


__all__ = [
    "Container",
    "AgentModule",
    "IStorageProvider",
    "ISecretManager",
    "IObservabilityProvider",
    "ILLMClient",
    "IBudgetService",
    "IPolicyEvaluator",
    "ICheckpointSaver",
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for :class:`AgentModule`.

    Importing :class:`AgentModule` eagerly here would pull in concrete
    classes from every domain module, several of which transitively
    import :mod:`ai_core.di.interfaces` — creating a circular import
    during package initialization. Resolving the symbol on first
    attribute access defers the heavy import until the cycle is closed.
    """
    if name == "AgentModule":
        from ai_core.di.module import AgentModule as _AgentModule

        return _AgentModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
