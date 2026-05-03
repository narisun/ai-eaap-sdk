"""Production observability provider: OpenTelemetry + LangFuse.

Behaviour
---------
* **OTel** — initialises a process-wide :class:`TracerProvider` if one
  is not already set, attaching an OTLP/gRPC exporter when
  :attr:`ObservabilitySettings.otel_endpoint` is configured. Spans
  capture agent "thoughts" (start_span attributes) and tool calls
  (record_event).
* **LangFuse** — constructs a :class:`langfuse.Langfuse` client when
  both keys are configured. ``record_llm_usage`` writes a
  ``generation`` (prompt → completion with token + cost metadata) and
  ``record_event`` writes a top-level ``event``.
* **Graceful degradation** — when neither backend is configured the
  provider behaves like the :class:`NoOpObservabilityProvider`, with
  one improvement: it still allocates and propagates trace identifiers
  so log correlation works during local development.

Concurrency / context propagation
---------------------------------
LangFuse is not span-aware in the OTel sense, so the provider keeps an
async-task-scoped :class:`contextvars.ContextVar` pointing at the
*active LangFuse trace*. ``start_span`` opens a child LangFuse span on
the active trace (creating one if absent); ``record_llm_usage`` reads
the same ContextVar so generations land on the right trace.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from injector import inject

from ai_core.config.settings import AppSettings, ObservabilitySettings
from ai_core.di.interfaces import IObservabilityProvider, SpanContext

_logger = logging.getLogger(__name__)

# OTel imports kept module-level — they're required deps in pyproject.
from opentelemetry import trace as otel_trace  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: E402

# Track the active LangFuse trace per async task.
_active_lf_trace: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "ai_core_active_langfuse_trace", default=None
)


class RealObservabilityProvider(IObservabilityProvider):
    """OpenTelemetry + LangFuse implementation of :class:`IObservabilityProvider`.

    Args:
        settings: Aggregated application settings.
    """

    @inject
    def __init__(self, settings: AppSettings) -> None:
        self._cfg: ObservabilitySettings = settings.observability
        self._service_name = settings.service_name
        self._tracer = self._init_otel()
        self._langfuse = self._init_langfuse()

    # ------------------------------------------------------------------
    # IObservabilityProvider
    # ------------------------------------------------------------------
    def start_span(  # type: ignore[override]
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> Any:
        """See :meth:`IObservabilityProvider.start_span`."""
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
        """See :meth:`IObservabilityProvider.record_llm_usage`."""
        attrs = dict(attributes or {})
        attrs.update(
            {
                "llm.model": model,
                "llm.prompt_tokens": prompt_tokens,
                "llm.completion_tokens": completion_tokens,
                "llm.latency_ms": latency_ms,
                "llm.cost_usd": cost_usd if cost_usd is not None else 0.0,
            }
        )
        # OTel: emit as an event on the current span (if any) — cheap, structured.
        current = otel_trace.get_current_span()
        if current.is_recording():
            current.add_event("llm.usage", attributes=attrs)

        # LangFuse: a "generation" lives on the active trace.
        trace_handle = self._ensure_lf_trace(name="llm.complete")
        if trace_handle is not None:
            try:
                trace_handle.generation(
                    name="llm.complete",
                    model=model,
                    usage={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                        "unit": "TOKENS",
                        **({"total_cost": cost_usd} if cost_usd is not None else {}),
                    },
                    metadata={k: v for k, v in attrs.items() if not k.startswith("llm.")},
                )
            except Exception as exc:  # noqa: BLE001 — observability must never raise
                _logger.warning("LangFuse generation() failed: %s", exc)

    async def record_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """See :meth:`IObservabilityProvider.record_event`."""
        current = otel_trace.get_current_span()
        if current.is_recording():
            current.add_event(name, attributes=dict(attributes or {}))

        trace_handle = self._ensure_lf_trace(name=name)
        if trace_handle is not None:
            try:
                trace_handle.event(name=name, metadata=dict(attributes or {}))
            except Exception as exc:  # noqa: BLE001
                _logger.warning("LangFuse event() failed: %s", exc)

    async def shutdown(self) -> None:
        """Flush both backends. Idempotent and exception-safe."""
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("LangFuse flush() failed: %s", exc)
        provider = otel_trace.get_tracer_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("OTel TracerProvider shutdown() failed: %s", exc)

    # ------------------------------------------------------------------
    # Span context manager
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def _span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None,
    ) -> AsyncIterator[SpanContext]:
        attrs = dict(attributes or {})
        outer_trace_token: contextvars.Token[Any | None] | None = None

        # Open OTel span
        with self._tracer.start_as_current_span(name, attributes=attrs) as otel_span:
            otel_ctx = otel_span.get_span_context()
            trace_id_hex = format(otel_ctx.trace_id, "032x")
            span_id_hex = format(otel_ctx.span_id, "016x")

            lf_span: Any = None
            current_trace = _active_lf_trace.get()
            if self._langfuse is not None:
                if current_trace is None:
                    new_trace = self._safe_lf_call(
                        lambda: self._langfuse.trace(  # type: ignore[union-attr]
                            name=name,
                            metadata=attrs,
                        )
                    )
                    if new_trace is not None:
                        outer_trace_token = _active_lf_trace.set(new_trace)
                        current_trace = new_trace
                if current_trace is not None:
                    lf_span = self._safe_lf_call(
                        lambda: current_trace.span(name=name, metadata=attrs)
                    )

            ctx = SpanContext(
                name=name,
                trace_id=trace_id_hex,
                span_id=span_id_hex,
                backend_handles={"otel": otel_span, "langfuse": lf_span},
            )
            try:
                yield ctx
            except BaseException as exc:  # noqa: BLE001 — record then re-raise
                otel_span.record_exception(exc)
                otel_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(exc)))
                if lf_span is not None:
                    self._safe_lf_call(
                        lambda: lf_span.end(level="ERROR", status_message=str(exc))
                    )
                raise
            else:
                if lf_span is not None:
                    self._safe_lf_call(lambda: lf_span.end())
            finally:
                if outer_trace_token is not None:
                    _active_lf_trace.reset(outer_trace_token)

    # ------------------------------------------------------------------
    # Backend init
    # ------------------------------------------------------------------
    def _init_otel(self) -> Any:
        provider = otel_trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            # A TracerProvider has already been installed (e.g. by the host
            # application). Reuse it — don't double-install processors.
            return otel_trace.get_tracer(self._service_name)

        new_provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": self._service_name,
                    "deployment.environment": "unknown",
                }
            )
        )
        if self._cfg.otel_endpoint is not None:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(
                    endpoint=str(self._cfg.otel_endpoint),
                    insecure=self._cfg.otel_insecure,
                )
                new_provider.add_span_processor(BatchSpanProcessor(exporter))
            except Exception as exc:  # noqa: BLE001
                _logger.warning("OTel OTLP exporter setup failed: %s", exc)
        otel_trace.set_tracer_provider(new_provider)
        return otel_trace.get_tracer(self._service_name)

    def _init_langfuse(self) -> Any | None:
        if self._cfg.langfuse_public_key is None or self._cfg.langfuse_secret_key is None:
            return None
        try:
            from langfuse import Langfuse

            return Langfuse(
                public_key=self._cfg.langfuse_public_key.get_secret_value(),
                secret_key=self._cfg.langfuse_secret_key.get_secret_value(),
                host=str(self._cfg.langfuse_host) if self._cfg.langfuse_host else None,
            )
        except Exception as exc:  # noqa: BLE001 — degrade rather than crash
            _logger.warning("LangFuse init failed; continuing without it: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_lf_trace(self, *, name: str) -> Any | None:
        """Return the active LangFuse trace, creating one if necessary."""
        if self._langfuse is None:
            return None
        trace_handle = _active_lf_trace.get()
        if trace_handle is not None:
            return trace_handle
        new_trace = self._safe_lf_call(lambda: self._langfuse.trace(name=name))  # type: ignore[union-attr]
        if new_trace is not None:
            _active_lf_trace.set(new_trace)
        return new_trace

    @staticmethod
    def _safe_lf_call(call: Any) -> Any:
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("LangFuse call failed: %s", exc)
            return None


__all__ = ["RealObservabilityProvider"]
