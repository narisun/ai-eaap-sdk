"""No-op observability provider.

Useful as a default DI binding so that calling code never has to
``if provider is not None`` around span/usage hooks. The provider
records nothing, allocates nothing, and is safe to use in tests and
local development where no OTel collector or LangFuse instance is
running.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from ai_core.di.interfaces import IObservabilityProvider, SpanContext


class NoOpObservabilityProvider(IObservabilityProvider):
    """Drop-all implementation of :class:`IObservabilityProvider`.

    Methods accept the same signatures as a real provider but perform
    no I/O. The yielded :class:`SpanContext` carries random IDs so that
    callers can still propagate trace identifiers in logs.
    """

    @asynccontextmanager
    async def _span(  # type: ignore[override]
        self,
        name: str,
        attributes: Mapping[str, Any] | None,
    ) -> AsyncIterator[SpanContext]:
        ctx = SpanContext(
            name=name,
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            backend_handles={},
        )
        try:
            yield ctx
        finally:  # nothing to flush, but keep the structure for parity
            pass

    def start_span(  # type: ignore[override]
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return an async context manager yielding a :class:`SpanContext`."""
        return self._span(name, attributes)

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
        """No-op."""
        return None

    async def record_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """No-op."""
        return None

    async def shutdown(self) -> None:
        """No-op."""
        return None


__all__ = ["NoOpObservabilityProvider"]
