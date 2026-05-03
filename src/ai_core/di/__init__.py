"""Dependency-injection sub-package.

The DI layer is the *only* place where concrete classes are bound to
abstract interfaces. Application code MUST depend on the abstractions
exposed by :mod:`ai_core.di.interfaces` and resolve them through the
container exposed by :mod:`ai_core.di.container`.

Typical bootstrap (in a host service)::

    from ai_core.di import Container, AgentModule

    container = Container.build(modules=[AgentModule()])
    agent = container.get(MyAgent)            # MyAgent's deps auto-wired
"""

from __future__ import annotations

from ai_core.di.container import Container
from ai_core.di.interfaces import (
    IBudgetService,
    ICheckpointSaver,
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
    IStorageProvider,
)
from ai_core.di.module import AgentModule

# ISecretManager is intentionally re-exported from config to avoid a cycle
# while still letting downstream code import it from `ai_core.di`.
from ai_core.config.secrets import ISecretManager

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
