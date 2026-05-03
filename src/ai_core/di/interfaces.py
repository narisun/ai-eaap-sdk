"""Abstract Base Classes for every external dependency used by the SDK.

These interfaces form the *seam* between the SDK and the host
environment. Concrete implementations (S3, AWS Secrets Manager,
LiteLLM, OPA, OTel, …) are bound to these ABCs by an :class:`injector`
:class:`~injector.Module` (see :mod:`ai_core.di.module`).

The interfaces are deliberately minimal — broad enough to cover real
production needs but narrow enough that fakes for unit testing can be
written in a few lines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class IStorageProvider(ABC):
    """Object-storage abstraction (S3, GCS, Azure Blob, local FS).

    Implementations are expected to be safe for concurrent use across
    coroutines and to perform their own connection pooling.
    """

    @abstractmethod
    async def put_object(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        """Upload ``body`` under ``key``.

        Args:
            key: Object key (path inside the bucket).
            body: Raw bytes payload.
            content_type: Optional MIME type stored with the object.
            metadata: Optional user-defined metadata.

        Returns:
            A backend-specific object identifier (e.g. ``s3://bucket/key``).
        """

    @abstractmethod
    async def get_object(self, key: str) -> bytes:
        """Return the raw bytes stored under ``key``.

        Raises:
            ai_core.exceptions.StorageError: If the object cannot be read.
        """

    @abstractmethod
    async def delete_object(self, key: str) -> None:
        """Remove the object stored under ``key``. Idempotent."""

    @abstractmethod
    async def list_objects(self, prefix: str) -> AsyncIterator[str]:
        """Yield object keys whose name starts with ``prefix``."""


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SpanContext:
    """Lightweight span handle returned by :class:`IObservabilityProvider`.

    Concrete providers may attach richer state (OTel ``Span``,
    LangFuse trace handles) via the ``backend_handles`` mapping; callers
    should treat the contents as opaque.
    """

    name: str
    trace_id: str
    span_id: str
    backend_handles: Mapping[str, Any]


class IObservabilityProvider(ABC):
    """Span + trace + LLM-usage logging abstraction.

    A single provider fans out to OpenTelemetry **and** LangFuse so that
    instrumentation code stays vendor-agnostic.
    """

    @abstractmethod
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[SpanContext]:
        """Open a new span as an async context manager.

        Args:
            name: Span name (typically ``module.function``).
            attributes: Optional structured attributes added to the span.

        Returns:
            An async context manager yielding a :class:`SpanContext`.
        """

    @abstractmethod
    async def record_llm_usage(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_usd: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit a usage metric covering one LLM invocation."""

    @abstractmethod
    async def record_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit an arbitrary structured event (audit, business signal)."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Flush exporters and release resources."""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Token + cost telemetry returned alongside every LLM completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalised LLM completion result."""

    model: str
    content: str
    tool_calls: Sequence[Mapping[str, Any]]
    usage: LLMUsage
    raw: Mapping[str, Any]


class ILLMClient(ABC):
    """LiteLLM-backed completion client with budgeting + retries.

    Note:
        Concrete implementations live in :mod:`ai_core.llm` and wrap
        :func:`litellm.acompletion` with tenacity-based retries and a
        :class:`IBudgetService` precheck.
    """

    @abstractmethod
    async def complete(
        self,
        *,
        model: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a chat completion.

        Args:
            model: Optional override; defaults to ``LLMSettings.default_model``.
            messages: OpenAI-style chat message dicts.
            tools: Optional tool definitions (function-calling).
            temperature: Sampling temperature.
            max_tokens: Output token cap.
            tenant_id: Tenant identifier — used for budget enforcement.
            agent_id: Logical agent identifier — used for budget + tracing.
            extra: Extra provider-specific kwargs forwarded verbatim.

        Returns:
            A normalised :class:`LLMResponse`.

        Raises:
            ai_core.exceptions.BudgetExceededError: If the budget precheck fails.
            ai_core.exceptions.LLMInvocationError: On retry exhaustion.
        """


# ---------------------------------------------------------------------------
# Budgeting
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BudgetCheck:
    """Result of a budget precheck."""

    allowed: bool
    remaining_tokens: int | None
    remaining_usd: float | None
    reason: str | None = None


class IBudgetService(ABC):
    """Per-tenant / per-agent quota enforcement."""

    @abstractmethod
    async def check(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        estimated_tokens: int,
    ) -> BudgetCheck:
        """Return whether the projected request fits within remaining quota."""

    @abstractmethod
    async def record_usage(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """Persist actual usage after a successful LLM call."""


# ---------------------------------------------------------------------------
# Policy / OPA
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """OPA decision record returned to callers."""

    allowed: bool
    obligations: Mapping[str, Any]
    reason: str | None = None


class IPolicyEvaluator(ABC):
    """OPA-backed authorisation evaluator."""

    @abstractmethod
    async def evaluate(
        self,
        *,
        decision_path: str,
        input: Mapping[str, Any],
    ) -> PolicyDecision:
        """Submit ``input`` to OPA and return the decision document."""


# ---------------------------------------------------------------------------
# LangGraph checkpointing
# ---------------------------------------------------------------------------
class ICheckpointSaver(ABC):
    """Persist + restore LangGraph state across runs."""

    @abstractmethod
    async def save(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Persist a serialised LangGraph checkpoint."""

    @abstractmethod
    async def load(
        self,
        *,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        """Return the latest (or named) checkpoint, or ``None`` if absent."""

    @abstractmethod
    async def list(self, *, thread_id: str, limit: int = 10) -> Sequence[str]:
        """Return up to ``limit`` checkpoint ids for ``thread_id`` (newest first)."""


# ---------------------------------------------------------------------------
# MCP / agent registry — Protocol so callers can supply duck-typed objects
# ---------------------------------------------------------------------------
class IComponent(Protocol):
    """Minimal contract every registered component (Agent, MCP server) must satisfy."""

    component_id: str
    component_type: str  # e.g. "agent" | "mcp_server"

    async def health_check(self) -> bool:  # pragma: no cover - protocol stub
        ...


__all__ = [
    "IStorageProvider",
    "IObservabilityProvider",
    "SpanContext",
    "ILLMClient",
    "LLMResponse",
    "LLMUsage",
    "IBudgetService",
    "BudgetCheck",
    "IPolicyEvaluator",
    "PolicyDecision",
    "ICheckpointSaver",
    "IComponent",
]
