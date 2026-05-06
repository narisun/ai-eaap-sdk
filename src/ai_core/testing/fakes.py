"""Fake implementations of the SDK's public protocols, for use in tests.

These fakes are deliberately simple, in-memory, and synchronous-where-possible
so consumer tests can assert on observable state without setting up real
backends. They are NOT production-quality.

Imported by :mod:`ai_core.testing` (no pytest dependency) and re-exported
from :mod:`ai_core.testing.pytest_plugin` for fixture-based access.
"""

from __future__ import annotations

# Stdlib imports needed by the 5 classes (assembled from tests/conftest.py imports).
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# SDK protocol imports (the fakes implement these).
from ai_core.audit import AuditRecord, IAuditSink
from ai_core.config.secrets import ISecretManager, SecretRef
from ai_core.di.interfaces import (
    BudgetCheck,
    IBudgetService,
    IObservabilityProvider,
    IPolicyEvaluator,
    PolicyDecision,
    SpanContext,
)
from ai_core.exceptions import EAAPBaseException, SecretResolutionError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from contextlib import AbstractAsyncContextManager


# ---------------------------------------------------------------------------
# FakePolicyEvaluator
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _PolicyCall:
    decision_path: str
    input: Mapping[str, Any]


class FakePolicyEvaluator(IPolicyEvaluator):
    """Deterministic IPolicyEvaluator for tests.

    Configure with a default verdict and optional per-path overrides.
    Records every call for inspection.
    """

    def __init__(
        self,
        *,
        default_allow: bool = True,
        reason: str | None = None,
        overrides: Mapping[str, PolicyDecision] | None = None,
    ) -> None:
        self._default_allow = default_allow
        self._reason = reason
        self._overrides = dict(overrides or {})
        self.calls: list[_PolicyCall] = []

    async def evaluate(
        self, *, decision_path: str, input: Mapping[str, Any]
    ) -> PolicyDecision:
        self.calls.append(_PolicyCall(decision_path=decision_path, input=dict(input)))
        if decision_path in self._overrides:
            return self._overrides[decision_path]
        return PolicyDecision(
            allowed=self._default_allow,
            obligations={},
            reason=self._reason,
        )


# ---------------------------------------------------------------------------
# FakeObservabilityProvider
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _RecordedSpan:
    name: str
    attributes: Mapping[str, Any]
    exception: BaseException | None = None
    error_code: str | None = None  # NEW — mirrors RealObservabilityProvider's error.code tagging


class FakeObservabilityProvider(IObservabilityProvider):
    """Records spans, events, and LLM-usage entries for assertion in tests."""

    def __init__(self) -> None:
        self.spans: list[_RecordedSpan] = []
        self.events: list[tuple[str, Mapping[str, Any]]] = []
        self.usage: list[Mapping[str, Any]] = []
        self.shutdowns: int = 0

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[SpanContext]:
        recorded = _RecordedSpan(name=name, attributes=dict(attributes or {}))
        self.spans.append(recorded)

        @asynccontextmanager
        async def _cm() -> AsyncIterator[SpanContext]:
            try:
                yield SpanContext(
                    name=name,
                    trace_id="trace-fake",
                    span_id=f"span-{len(self.spans)}",
                    backend_handles={},
                )
            except BaseException as exc:
                recorded.exception = exc
                # Mirror RealObservabilityProvider's error_code tagging behavior.
                if isinstance(exc, EAAPBaseException):
                    recorded.error_code = exc.error_code
                raise

        return _cm()

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
        self.usage.append(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "attributes": dict(attributes or {}),
            }
        )

    async def record_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append((name, dict(attributes or {})))

    async def shutdown(self) -> None:
        self.shutdowns += 1


# ---------------------------------------------------------------------------
# FakeSecretManager
# ---------------------------------------------------------------------------
class FakeSecretManager(ISecretManager):
    """In-memory ISecretManager keyed by (backend, name)."""

    def __init__(self, mapping: Mapping[tuple[str, str], str] | None = None) -> None:
        self._mapping = dict(mapping or {})

    async def resolve(self, ref: SecretRef) -> str:
        try:
            return self._mapping[(ref.backend, ref.name)]
        except KeyError as exc:
            raise SecretResolutionError(
                f"FakeSecretManager has no value for {ref.backend}/{ref.name}",
                details={"backend": ref.backend, "name": ref.name},
                cause=exc,
            ) from exc


# ---------------------------------------------------------------------------
# FakeBudgetService — promoted from per-test-file definitions (Phase 3 item 9)
# ---------------------------------------------------------------------------
class FakeBudgetService(IBudgetService):
    """Always-allow IBudgetService for tests. Records call kwargs for assertion."""

    def __init__(self) -> None:
        self.checks: list[Mapping[str, Any]] = []
        self.usages: list[Mapping[str, Any]] = []

    async def check(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        estimated_tokens: int,
    ) -> BudgetCheck:
        self.checks.append(
            {
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "estimated_tokens": estimated_tokens,
            },
        )
        return BudgetCheck(allowed=True, remaining_tokens=None, remaining_usd=None)

    async def record_usage(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        self.usages.append(
            {
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
            },
        )


# ---------------------------------------------------------------------------
# FakeAuditSink — Phase 3 Task 3
# ---------------------------------------------------------------------------
class FakeAuditSink(IAuditSink):
    """Records audit records for assertion in tests."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []
        self.flushes: int = 0

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)

    async def flush(self) -> None:
        self.flushes += 1


__all__ = [
    "FakeAuditSink",
    "FakeBudgetService",
    "FakeObservabilityProvider",
    "FakePolicyEvaluator",
    "FakeSecretManager",
]
