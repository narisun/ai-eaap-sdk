"""ai_core — Enterprise Agentic AI Platform (EAAP) core SDK.

This package exposes the building blocks required to assemble an
agentic application: dependency-injection container, configuration,
observability, persistence, security, LLM proxying, and MCP lifecycle.

All cross-cutting interfaces are defined as :pep:`544` ABCs so that
concrete implementations can be swapped via the DI container without
modifying calling code.
"""

from __future__ import annotations

from ai_core.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    DependencyResolutionError,
    EAAPBaseException,
    LLMInvocationError,
    PolicyDenialError,
    SchemaValidationError,
    SecretResolutionError,
    StorageError,
)

__all__ = [
    "EAAPBaseException",
    "ConfigurationError",
    "DependencyResolutionError",
    "SecretResolutionError",
    "StorageError",
    "PolicyDenialError",
    "BudgetExceededError",
    "LLMInvocationError",
    "SchemaValidationError",
]

__version__ = "0.1.0"
