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

Typed details (v1)
==================
Subclasses with a stable ``details`` schema expose a typed
``@dataclass`` payload reachable via ``exc.as_typed_details()``. The
raw dict still works (back-compat); the typed accessor is the
recommended path for SREs / consumers who route or alert on errors::

    try:
        await invoker.invoke(spec, args)
    except ToolValidationError as exc:
        info = exc.as_typed_details()
        logger.warning("tool %s failed validation on %s side", info.tool, info.side)
        emit_metric("tool.validation_failed", tags={"tool": info.tool})

Heterogeneous classes (``PolicyDenialError``, ``ConfigurationError``,
``RegistryError``, ``LLMInvocationError``) keep raw-dict ``details``
because their keys vary too much to type usefully. Future PRs may
introduce per-call-site sub-classes if any of those settle on a stable
shape.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


# ---------------------------------------------------------------------------
# Typed details payloads
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ToolValidationDetails:
    """Typed payload for :class:`ToolValidationError.details`.

    Attributes:
        tool: Logical tool name.
        version: Registered :class:`ToolSpec` version.
        side: Either ``"input"`` (raw args failed validation) or
            ``"output"`` (handler return value failed validation).
        errors: Pydantic ``ValidationError.errors()`` list.
    """

    tool: str
    version: int
    side: str
    errors: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ToolExecutionDetails:
    """Typed payload for :class:`ToolExecutionError.details`.

    The original handler exception is preserved on ``__cause__``; this
    payload only carries identity.
    """

    tool: str
    version: int
    agent_id: str | None
    tenant_id: str | None


@dataclass(frozen=True, slots=True)
class BudgetExceededDetails:
    """Typed payload for :class:`BudgetExceededError.details`.

    All fields are optional individually because the budget service may
    deny on either the token or the USD axis without recording the
    other; ``reason`` carries the human-readable trigger.
    """

    tenant_id: str | None
    agent_id: str | None
    model: str | None
    estimated_tokens: int | None
    remaining_tokens: int | None
    remaining_usd: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class LLMTimeoutDetails:
    """Typed payload for :class:`LLMTimeoutError.details`."""

    model: str
    attempts: int


@dataclass(frozen=True, slots=True)
class MCPTransportDetails:
    """Typed payload for :class:`MCPTransportError.details`."""

    component_id: str
    transport: str


@dataclass(frozen=True, slots=True)
class AgentRecursionDetails:
    """Typed payload for :class:`AgentRecursionLimitError.details`."""

    agent_id: str | None
    tenant_id: str | None
    thread_id: str | None
    limit: int


@dataclass(frozen=True, slots=True)
class SecretResolutionDetails:
    """Typed payload for :class:`SecretResolutionError.details`."""

    backend: str
    name: str


@dataclass(frozen=True, slots=True)
class DependencyResolutionDetails:
    """Typed payload for :class:`DependencyResolutionError.details`."""

    interface: str


@dataclass(frozen=True, slots=True)
class CheckpointDetails:
    """Typed payload for :class:`CheckpointError.details`."""

    thread_id: str
    checkpoint_id: str | None


class ErrorCode(enum.StrEnum):
    """Canonical error codes for typed SDK exceptions.

    Members exhaustively cover every ``error_code`` string referenced by
    production code in Phases 1-6. Values are dotted-lowercase strings;
    ``ErrorCode`` inherits from ``str`` so members are directly comparable
    with raw strings::

        if exc.error_code == ErrorCode.CONFIG_INVALID:
            ...

    Adding a new code requires:
      1. Adding a member here (dotted-lowercase value).
      2. Wiring it into the appropriate exception class's ``DEFAULT_CODE``
         OR using ``ErrorCode.<member>`` directly at the construction site.

    The contract test
    ``test_every_concrete_exception_default_code_is_an_errorcode_member``
    catches any new exception class that bypasses the enum.
    """

    # Configuration (Phases 1, 5, 6)
    CONFIG_INVALID = "config.invalid"
    CONFIG_SECRET_NOT_RESOLVED = "config.secret_not_resolved"  # noqa: S105
    CONFIG_YAML_PATH_MISSING = "config.yaml_path_missing"
    CONFIG_YAML_PARSE_FAILED = "config.yaml_parse_failed"
    CONFIG_OPTIONAL_DEP_MISSING = "config.optional_dep_missing"

    # Dependency injection (Phase 1)
    DI_RESOLUTION_FAILED = "di.resolution_failed"

    # Storage / persistence (Phase 1)
    STORAGE_ERROR = "storage.error"
    STORAGE_CHECKPOINT_FAILED = "storage.checkpoint_failed"

    # Policy / authorization (Phase 1)
    POLICY_DENIED = "policy.denied"

    # LLM (Phase 1)
    LLM_INVOCATION_FAILED = "llm.invocation_failed"
    LLM_TIMEOUT = "llm.timeout"
    LLM_BUDGET_EXCEEDED = "llm.budget_exceeded"
    LLM_EMPTY_RESPONSE = "llm.empty_response"

    # Schema / validation (Phase 1)
    SCHEMA_INVALID = "schema.invalid"
    TOOL_VALIDATION_FAILED = "tool.validation_failed"

    # Tool execution (Phase 1)
    TOOL_EXECUTION_FAILED = "tool.execution_failed"

    # Agent runtime (Phase 1)
    AGENT_RUNTIME_ERROR = "agent.runtime_error"
    AGENT_RECURSION_LIMIT = "agent.recursion_limit"

    # Registry (Phase 1)
    REGISTRY_ERROR = "registry.error"

    # MCP transport (Phase 1)
    MCP_TRANSPORT_FAILED = "mcp.transport_failed"


class EAAPBaseException(Exception):  # noqa: N818
    """Base class for all SDK-raised exceptions.

    The "Exception" suffix instead of "Error" is a deliberate, stable name
    choice — it's part of the v1.0 public API. Renaming is a breaking
    change and is not on the roadmap.


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
        # Mirror error_code into details so observability surfaces that walk
        # the dict see the same value as `self.error_code`. Last-write-wins —
        # any pre-existing details["error_code"] from the caller is overwritten.
        self.details["error_code"] = self.error_code
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

    DEFAULT_CODE = ErrorCode.CONFIG_INVALID


class SecretResolutionError(ConfigurationError):
    """Raised when a secret cannot be resolved from a secret backend."""

    DEFAULT_CODE = ErrorCode.CONFIG_SECRET_NOT_RESOLVED

    def as_typed_details(self) -> SecretResolutionDetails:
        """Return ``self.details`` as a :class:`SecretResolutionDetails`."""
        return SecretResolutionDetails(
            backend=str(self.details.get("backend", "")),
            name=str(self.details.get("name", "")),
        )


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------
class DependencyResolutionError(EAAPBaseException):
    """Raised when the DI container cannot satisfy a binding."""

    DEFAULT_CODE = ErrorCode.DI_RESOLUTION_FAILED

    def as_typed_details(self) -> DependencyResolutionDetails:
        """Return ``self.details`` as a :class:`DependencyResolutionDetails`."""
        return DependencyResolutionDetails(
            interface=str(self.details.get("interface", "")),
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class StorageError(EAAPBaseException):
    """Base class for persistence failures (SQL, vector, blob)."""

    DEFAULT_CODE = ErrorCode.STORAGE_ERROR


class CheckpointError(StorageError):
    """Raised when reading or writing a LangGraph checkpoint fails."""

    DEFAULT_CODE = ErrorCode.STORAGE_CHECKPOINT_FAILED

    def as_typed_details(self) -> CheckpointDetails:
        """Return ``self.details`` as a :class:`CheckpointDetails`."""
        ck = self.details.get("checkpoint_id")
        return CheckpointDetails(
            thread_id=str(self.details.get("thread_id", "")),
            checkpoint_id=str(ck) if ck is not None else None,
        )


# ---------------------------------------------------------------------------
# Security & policy
# ---------------------------------------------------------------------------
class PolicyDenialError(EAAPBaseException):
    """Raised when an OPA policy denies a request or tool invocation."""

    DEFAULT_CODE = ErrorCode.POLICY_DENIED


# ---------------------------------------------------------------------------
# LLM / budgeting
# ---------------------------------------------------------------------------
class LLMInvocationError(EAAPBaseException):
    """Raised when an LLM call fails after retry exhaustion."""

    DEFAULT_CODE = ErrorCode.LLM_INVOCATION_FAILED


class LLMTimeoutError(LLMInvocationError):
    """Raised when an LLM call exceeds its configured timeout (post-retry)."""

    DEFAULT_CODE = ErrorCode.LLM_TIMEOUT

    def as_typed_details(self) -> LLMTimeoutDetails:
        """Return ``self.details`` as a :class:`LLMTimeoutDetails`."""
        return LLMTimeoutDetails(
            model=str(self.details.get("model", "")),
            attempts=int(self.details.get("attempts", 0)),
        )


class BudgetExceededError(LLMInvocationError):
    """Raised when an agent or tenant has consumed its allocated quota."""

    DEFAULT_CODE = ErrorCode.LLM_BUDGET_EXCEEDED

    def as_typed_details(self) -> BudgetExceededDetails:
        """Return ``self.details`` as a :class:`BudgetExceededDetails`."""
        d = self.details
        rt = d.get("remaining_tokens")
        ru = d.get("remaining_usd")
        et = d.get("estimated_tokens")
        return BudgetExceededDetails(
            tenant_id=d.get("tenant_id"),
            agent_id=d.get("agent_id"),
            model=d.get("model"),
            estimated_tokens=int(et) if et is not None else None,
            remaining_tokens=int(rt) if rt is not None else None,
            remaining_usd=float(ru) if ru is not None else None,
            reason=d.get("reason"),
        )


# ---------------------------------------------------------------------------
# Schema / contract
# ---------------------------------------------------------------------------
class SchemaValidationError(EAAPBaseException):
    """Raised when a payload does not match the expected versioned schema."""

    DEFAULT_CODE = ErrorCode.SCHEMA_INVALID


class ToolValidationError(SchemaValidationError):
    """Tool input or output failed Pydantic validation.

    The ``details`` payload carries:

    * ``tool`` — the tool name,
    * ``version`` — the registered version,
    * ``side`` — ``"input"`` or ``"output"``,
    * ``errors`` — Pydantic ``error.errors()`` list.
    """

    DEFAULT_CODE = ErrorCode.TOOL_VALIDATION_FAILED

    def as_typed_details(self) -> ToolValidationDetails:
        """Return ``self.details`` as a :class:`ToolValidationDetails`."""
        errs = self.details.get("errors") or ()
        return ToolValidationDetails(
            tool=str(self.details.get("tool", "")),
            version=int(self.details.get("version", 0)),
            side=str(self.details.get("side", "")),
            errors=tuple(errs),
        )


class ToolExecutionError(EAAPBaseException):
    """A tool handler raised. The original exception is preserved via ``__cause__``.

    The ``details`` payload carries ``tool``, ``version``, and (when known)
    ``agent_id`` / ``tenant_id`` so dashboards can correlate failures with
    the calling agent.
    """

    DEFAULT_CODE = ErrorCode.TOOL_EXECUTION_FAILED

    def as_typed_details(self) -> ToolExecutionDetails:
        """Return ``self.details`` as a :class:`ToolExecutionDetails`."""
        return ToolExecutionDetails(
            tool=str(self.details.get("tool", "")),
            version=int(self.details.get("version", 0)),
            agent_id=self.details.get("agent_id"),
            tenant_id=self.details.get("tenant_id"),
        )


# ---------------------------------------------------------------------------
# Agent runtime
# ---------------------------------------------------------------------------
class AgentRuntimeError(EAAPBaseException):
    """Base class for agent-runtime failures (recursion limit, etc.)."""

    DEFAULT_CODE = ErrorCode.AGENT_RUNTIME_ERROR


class AgentRecursionLimitError(AgentRuntimeError):
    """Raised when an agent exceeds its configured recursion limit.

    The ``details`` payload includes ``agent_id`` and ``limit`` so
    operators can correlate dashboards with the exact agent that
    looped.
    """

    DEFAULT_CODE = ErrorCode.AGENT_RECURSION_LIMIT

    def as_typed_details(self) -> AgentRecursionDetails:
        """Return ``self.details`` as a :class:`AgentRecursionDetails`."""
        return AgentRecursionDetails(
            agent_id=self.details.get("agent_id"),
            tenant_id=self.details.get("tenant_id"),
            thread_id=self.details.get("thread_id"),
            limit=int(self.details.get("limit", 0)),
        )


# ---------------------------------------------------------------------------
# MCP / registry
# ---------------------------------------------------------------------------
class RegistryError(EAAPBaseException):
    """Raised when a component registry operation fails."""

    DEFAULT_CODE = ErrorCode.REGISTRY_ERROR


class MCPTransportError(EAAPBaseException):
    """Raised when an MCP transport (stdio/http/sse) fails to open or operate.

    The ``details`` payload carries ``component_id`` and ``transport`` so
    operators can identify which server failed.
    """

    DEFAULT_CODE = ErrorCode.MCP_TRANSPORT_FAILED

    def as_typed_details(self) -> MCPTransportDetails:
        """Return ``self.details`` as a :class:`MCPTransportDetails`."""
        return MCPTransportDetails(
            component_id=str(self.details.get("component_id", "")),
            transport=str(self.details.get("transport", "")),
        )


__all__ = [
    "AgentRecursionDetails",
    "AgentRecursionLimitError",
    "AgentRuntimeError",
    "BudgetExceededDetails",
    "BudgetExceededError",
    "CheckpointDetails",
    "CheckpointError",
    "ConfigurationError",
    "DependencyResolutionDetails",
    "DependencyResolutionError",
    "EAAPBaseException",
    "ErrorCode",
    "LLMInvocationError",
    "LLMTimeoutDetails",
    "LLMTimeoutError",
    "MCPTransportDetails",
    "MCPTransportError",
    "PolicyDenialError",
    "RegistryError",
    "SchemaValidationError",
    "SecretResolutionDetails",
    "SecretResolutionError",
    "StorageError",
    "ToolExecutionDetails",
    "ToolExecutionError",
    "ToolValidationDetails",
    "ToolValidationError",
]
