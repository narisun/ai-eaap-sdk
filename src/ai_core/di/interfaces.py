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

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
@runtime_checkable
class IStorageProvider(Protocol):
    """Object-storage abstraction (S3, GCS, Azure Blob, local FS).

    Implementations are expected to be safe for concurrent use across
    coroutines and to perform their own connection pooling.
    """

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
        ...

    async def get_object(self, key: str) -> bytes:
        """Return the raw bytes stored under ``key``.

        Raises:
            ai_core.exceptions.StorageError: If the object cannot be read.
        """
        ...

    async def delete_object(self, key: str) -> None:
        """Remove the object stored under ``key``. Idempotent."""
        ...

    async def list_objects(self, prefix: str) -> AsyncIterator[str]:
        """Yield object keys whose name starts with ``prefix``."""
        ...


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


@runtime_checkable
class IObservabilityProvider(Protocol):
    """Span + trace + LLM-usage logging abstraction.

    A single provider fans out to OpenTelemetry **and** LangFuse so that
    instrumentation code stays vendor-agnostic.
    """

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
        ...

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
        ...

    async def record_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit an arbitrary structured event (audit, business signal)."""
        ...

    async def shutdown(self) -> None:
        """Flush exporters and release resources."""
        ...


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
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        | str
        | None
    ) = None  # None means upstream didn't report


@dataclass(frozen=True, slots=True)
class LLMStreamChunk:
    """Incremental delta produced by :meth:`ILLMClient.astream`.

    A streaming response is a sequence of :class:`LLMStreamChunk` objects
    followed by a terminal chunk whose ``finish_reason`` is non-``None``.
    The terminal chunk also carries final ``usage`` if the upstream
    provider reports it; otherwise ``usage`` is ``None`` on every chunk
    and host code derives it from running totals.

    Attributes:
        model: Resolved model identifier (matches :attr:`LLMResponse.model`).
        delta_content: Text fragment to append to the running assistant
            message. Empty string is valid (e.g. tool-call-only chunk).
        delta_tool_calls: Partial tool-call records — providers stream
            these as fragmentary JSON; merging is the caller's
            responsibility (see ``ai_core.llm._stream_merge`` for a
            reference implementation).
        finish_reason: ``None`` until the terminal chunk; matches the
            same shape as :attr:`LLMResponse.finish_reason`.
        usage: Final usage tally on the terminal chunk only; ``None``
            on every other chunk.
        raw: Provider-native chunk payload, retained for callers that
            need fields the SDK does not normalise yet.
    """

    model: str
    delta_content: str
    delta_tool_calls: Sequence[Mapping[str, Any]]
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        | str
        | None
    ) = None
    usage: LLMUsage | None = None
    raw: Mapping[str, Any] | None = None


@runtime_checkable
class ICompactionLLM(Protocol):
    """Distinct LLM client used by :class:`MemoryManager` for summarisation.

    Structurally identical to :class:`ILLMClient` but bound separately so
    deployments can route compaction to a cheaper / faster model than the
    one that drives agent reasoning. The default :class:`AgentModule`
    binding aliases :class:`ICompactionLLM` to the request :class:`ILLMClient`,
    so behaviour is unchanged unless the host overrides the binding.
    """

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
        """Generate a completion. See :meth:`ILLMClient.complete`."""
        ...


@runtime_checkable
class ILLMClient(Protocol):
    """LiteLLM-backed completion client with budgeting + retries.

    Note:
        Concrete implementations live in :mod:`ai_core.llm` and wrap
        :func:`litellm.acompletion` with tenacity-based retries and a
        :class:`IBudgetService` precheck.
    """

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
        ...

    async def astream(
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
    ) -> AsyncIterator[LLMStreamChunk]:
        """Open a streaming chat completion and return an async iterator of deltas.

        Two-phase shape: ``await llm.astream(...)`` performs the budget
        pre-check, opens the stream (with retries on the open), and
        returns the iterator. Then::

            stream = await llm.astream(...)
            async for chunk in stream:
                ...

        The terminal chunk carries the final ``finish_reason`` and, when
        available, the final ``usage``. Non-terminal chunks always have
        ``finish_reason=None`` and ``usage=None``.

        Default implementations should still:

        * run the same budget pre-check as :meth:`complete` (and raise
          :class:`ai_core.exceptions.BudgetExceededError` before opening
          the stream),
        * record usage + cost when the terminal chunk arrives, and
        * surface the same retry / timeout / API-error semantics as
          :meth:`complete`.

        Hosts that don't need streaming can leave this method
        un-implemented; the SDK's default agent loop uses
        :meth:`complete`.

        Returns:
            An async iterator of :class:`LLMStreamChunk` objects.
        """
        ...


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


@runtime_checkable
class IBudgetService(Protocol):
    """Per-tenant / per-agent quota enforcement."""

    async def check(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        estimated_tokens: int,
    ) -> BudgetCheck:
        """Return whether the projected request fits within remaining quota."""
        ...

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
        ...


# ---------------------------------------------------------------------------
# Policy / OPA
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """OPA decision record returned to callers."""

    allowed: bool
    obligations: Mapping[str, Any]
    reason: str | None = None


@runtime_checkable
class IPolicyEvaluator(Protocol):
    """OPA-backed authorisation evaluator."""

    async def evaluate(
        self,
        *,
        decision_path: str,
        input: Mapping[str, Any],
    ) -> PolicyDecision:
        """Submit ``input`` to OPA and return the decision document."""
        ...


# ---------------------------------------------------------------------------
# LangGraph checkpointing
# ---------------------------------------------------------------------------
@runtime_checkable
class ICheckpointSaver(Protocol):
    """Persist + restore LangGraph state across runs."""

    async def save(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Persist a serialised LangGraph checkpoint."""
        ...

    async def load(
        self,
        *,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        """Return the latest (or named) checkpoint, or ``None`` if absent."""
        ...

    async def list(self, *, thread_id: str, limit: int = 10) -> Sequence[str]:
        """Return up to ``limit`` checkpoint ids for ``thread_id`` (newest first)."""
        ...


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
    "BudgetCheck",
    "IBudgetService",
    "ICheckpointSaver",
    "ICompactionLLM",
    "IComponent",
    "ILLMClient",
    "IObservabilityProvider",
    "IPolicyEvaluator",
    "IStorageProvider",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMUsage",
    "PolicyDecision",
    "SpanContext",
]
