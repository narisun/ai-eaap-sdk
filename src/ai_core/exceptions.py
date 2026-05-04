"""Custom exception hierarchy for the EAAP SDK.

Every error raised by the SDK derives from :class:`EAAPBaseException`,
allowing host applications to catch SDK errors generically while still
distinguishing between sub-domains (configuration, persistence, policy,
LLM, …) for targeted handling and metrics.

Errors carry an optional structured ``details`` mapping that flows into
OpenTelemetry span attributes and LangFuse trace metadata so that
downstream operators can correlate failures without parsing strings.
"""

from __future__ import annotations

from typing import Any


class EAAPBaseException(Exception):
    """Base class for all SDK-raised exceptions.

    Attributes:
        message: Human-readable description of the error.
        details: Optional structured context attached for observability.
        cause: Optional underlying exception preserved for chained tracebacks.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: dict[str, Any] = dict(details or {})
        self.cause: BaseException | None = cause
        if cause is not None:
            self.__cause__ = cause

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(message={self.message!r}, details={self.details!r})"


# ---------------------------------------------------------------------------
# Configuration / secrets
# ---------------------------------------------------------------------------
class ConfigurationError(EAAPBaseException):
    """Raised when required configuration is missing or invalid."""


class SecretResolutionError(ConfigurationError):
    """Raised when a secret cannot be resolved from a secret backend."""


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------
class DependencyResolutionError(EAAPBaseException):
    """Raised when the DI container cannot satisfy a binding."""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class StorageError(EAAPBaseException):
    """Base class for persistence failures (SQL, vector, blob)."""


class CheckpointError(StorageError):
    """Raised when reading or writing a LangGraph checkpoint fails."""


# ---------------------------------------------------------------------------
# Security & policy
# ---------------------------------------------------------------------------
class PolicyDenialError(EAAPBaseException):
    """Raised when an OPA policy denies a request or tool invocation."""


# ---------------------------------------------------------------------------
# LLM / budgeting
# ---------------------------------------------------------------------------
class LLMInvocationError(EAAPBaseException):
    """Raised when an LLM call fails after retry exhaustion."""


class BudgetExceededError(LLMInvocationError):
    """Raised when an agent or tenant has consumed its allocated quota."""


# ---------------------------------------------------------------------------
# Schema / contract
# ---------------------------------------------------------------------------
class SchemaValidationError(EAAPBaseException):
    """Raised when a payload does not match the expected versioned schema."""


class ToolValidationError(SchemaValidationError):
    """Tool input or output failed Pydantic validation.

    The ``details`` payload carries:

    * ``tool`` — the tool name,
    * ``version`` — the registered version,
    * ``side`` — ``"input"`` or ``"output"``,
    * ``errors`` — Pydantic ``error.errors()`` list.
    """


class ToolExecutionError(EAAPBaseException):
    """A tool handler raised. The original exception is preserved via ``__cause__``.

    The ``details`` payload carries ``tool``, ``version``, and (when known)
    ``agent_id`` / ``tenant_id`` so dashboards can correlate failures with
    the calling agent.
    """


# ---------------------------------------------------------------------------
# Agent runtime
# ---------------------------------------------------------------------------
class AgentRuntimeError(EAAPBaseException):
    """Base class for agent-runtime failures (recursion limit, etc.)."""


class AgentRecursionLimitError(AgentRuntimeError):
    """Raised when an agent exceeds its configured recursion limit.

    The ``details`` payload includes ``agent_id`` and ``limit`` so
    operators can correlate dashboards with the exact agent that
    looped.
    """


# ---------------------------------------------------------------------------
# MCP / registry
# ---------------------------------------------------------------------------
class RegistryError(EAAPBaseException):
    """Raised when a component registry operation fails."""


__all__ = [
    "AgentRecursionLimitError",
    "AgentRuntimeError",
    "BudgetExceededError",
    "CheckpointError",
    "ConfigurationError",
    "DependencyResolutionError",
    "EAAPBaseException",
    "LLMInvocationError",
    "PolicyDenialError",
    "RegistryError",
    "SchemaValidationError",
    "SecretResolutionError",
    "StorageError",
    "ToolExecutionError",
    "ToolValidationError",
]
