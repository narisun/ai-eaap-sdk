"""Custom exception hierarchy for the EAAP SDK.

Every error raised by the SDK derives from :class:`EAAPBaseException`,
allowing host applications to catch SDK errors generically while still
distinguishing between sub-domains (configuration, persistence, policy,
LLM, …) for targeted handling and metrics.

Errors carry an optional structured ``details`` mapping that flows into
OpenTelemetry span attributes and LangFuse trace metadata so that
downstream operators can correlate failures without parsing strings.

Each subclass declares a ``DEFAULT_CODE`` class attribute (Phase 2) used
to auto-populate ``error_code`` when callers don't pass one explicitly.
The code lands in ``details["error_code"]`` and as the OTel span
attribute ``error.code`` so dashboards can aggregate uniformly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class EAAPBaseException(Exception):
    """Base class for all SDK-raised exceptions.

    Attributes:
        message: Human-readable description of the error.
        details: Optional structured context attached for observability.
        cause: Optional underlying exception preserved for chained tracebacks.
        error_code: Dotted, lowercase code (e.g. ``"llm.timeout"``) used by
            dashboards. Defaults to the subclass's ``DEFAULT_CODE``.
    """

    DEFAULT_CODE: str = "eaap.unknown"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: dict[str, Any] = dict(details or {})
        self.error_code: str = error_code or type(self).DEFAULT_CODE
        # Auto-populate so observability surfaces that walk `details` pick it up.
        self.details.setdefault("error_code", self.error_code)
        self.cause: BaseException | None = cause
        if cause is not None:
            self.__cause__ = cause

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}(message={self.message!r}, "
            f"error_code={self.error_code!r}, details={self.details!r})"
        )


# ---------------------------------------------------------------------------
# Configuration / secrets
# ---------------------------------------------------------------------------
class ConfigurationError(EAAPBaseException):
    """Raised when required configuration is missing or invalid."""

    DEFAULT_CODE = "config.invalid"


class SecretResolutionError(ConfigurationError):
    """Raised when a secret cannot be resolved from a secret backend."""

    DEFAULT_CODE = "config.secret_not_resolved"


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------
class DependencyResolutionError(EAAPBaseException):
    """Raised when the DI container cannot satisfy a binding."""

    DEFAULT_CODE = "di.resolution_failed"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class StorageError(EAAPBaseException):
    """Base class for persistence failures (SQL, vector, blob)."""

    DEFAULT_CODE = "storage.error"


class CheckpointError(StorageError):
    """Raised when reading or writing a LangGraph checkpoint fails."""

    DEFAULT_CODE = "storage.checkpoint_failed"


# ---------------------------------------------------------------------------
# Security & policy
# ---------------------------------------------------------------------------
class PolicyDenialError(EAAPBaseException):
    """Raised when an OPA policy denies a request or tool invocation."""

    DEFAULT_CODE = "policy.denied"


# ---------------------------------------------------------------------------
# LLM / budgeting
# ---------------------------------------------------------------------------
class LLMInvocationError(EAAPBaseException):
    """Raised when an LLM call fails after retry exhaustion."""

    DEFAULT_CODE = "llm.invocation_failed"


class LLMTimeoutError(LLMInvocationError):
    """Raised when an LLM call exceeds its configured timeout (post-retry)."""

    DEFAULT_CODE = "llm.timeout"


class BudgetExceededError(LLMInvocationError):
    """Raised when an agent or tenant has consumed its allocated quota."""

    DEFAULT_CODE = "llm.budget_exceeded"


# ---------------------------------------------------------------------------
# Schema / contract
# ---------------------------------------------------------------------------
class SchemaValidationError(EAAPBaseException):
    """Raised when a payload does not match the expected versioned schema."""

    DEFAULT_CODE = "schema.invalid"


class ToolValidationError(SchemaValidationError):
    """Tool input or output failed Pydantic validation.

    The ``details`` payload carries:

    * ``tool`` — the tool name,
    * ``version`` — the registered version,
    * ``side`` — ``"input"`` or ``"output"``,
    * ``errors`` — Pydantic ``error.errors()`` list.
    """

    DEFAULT_CODE = "tool.validation_failed"


class ToolExecutionError(EAAPBaseException):
    """A tool handler raised. The original exception is preserved via ``__cause__``.

    The ``details`` payload carries ``tool``, ``version``, and (when known)
    ``agent_id`` / ``tenant_id`` so dashboards can correlate failures with
    the calling agent.
    """

    DEFAULT_CODE = "tool.execution_failed"


# ---------------------------------------------------------------------------
# Agent runtime
# ---------------------------------------------------------------------------
class AgentRuntimeError(EAAPBaseException):
    """Base class for agent-runtime failures (recursion limit, etc.)."""

    DEFAULT_CODE = "agent.runtime_error"


class AgentRecursionLimitError(AgentRuntimeError):
    """Raised when an agent exceeds its configured recursion limit.

    The ``details`` payload includes ``agent_id`` and ``limit`` so
    operators can correlate dashboards with the exact agent that
    looped.
    """

    DEFAULT_CODE = "agent.recursion_limit"


# ---------------------------------------------------------------------------
# MCP / registry
# ---------------------------------------------------------------------------
class RegistryError(EAAPBaseException):
    """Raised when a component registry operation fails."""

    DEFAULT_CODE = "registry.error"


class MCPTransportError(EAAPBaseException):
    """Raised when an MCP transport (stdio/http/sse) fails to open or operate.

    The ``details`` payload carries ``component_id`` and ``transport`` so
    operators can identify which server failed.
    """

    DEFAULT_CODE = "mcp.transport_failed"


__all__ = [
    "AgentRecursionLimitError",
    "AgentRuntimeError",
    "BudgetExceededError",
    "CheckpointError",
    "ConfigurationError",
    "DependencyResolutionError",
    "EAAPBaseException",
    "LLMInvocationError",
    "LLMTimeoutError",
    "MCPTransportError",
    "PolicyDenialError",
    "RegistryError",
    "SchemaValidationError",
    "SecretResolutionError",
    "StorageError",
    "ToolExecutionError",
    "ToolValidationError",
]
