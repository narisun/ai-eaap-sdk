"""Production observability provider: OpenTelemetry + LangFuse.

Behaviour
---------
* **OTel** — initialises a process-wide :class:`TracerProvider` if one
  is not already set, attaching an OTLP/gRPC exporter when
  :attr:`ObservabilitySettings.otel_endpoint` is configured. Spans
  capture agent "thoughts" (start_span attributes) and tool calls
  (record_event).
* **Dev mode console exporter** — when no collector endpoint is
  configured *and* the environment is local/dev, install an
  :class:`ConsoleSpanExporter` so developers see traces on stderr
  immediately without standing up infrastructure.
* **LangFuse** — constructs a :class:`langfuse.Langfuse` client when
  both keys are configured. ``record_llm_usage`` writes a
  ``generation`` (prompt → completion with token + cost metadata) and
  ``record_event`` writes a top-level ``event``.
* **OTel Baggage propagation** — every span opened via
  :py:meth:`start_span` copies any ``eaap.*`` Baggage entries onto its
  attributes (and onto the LangFuse span metadata). This means tenant
  /user/agent attribution flows automatically as long as the calling
  code sets Baggage at the entry point (FastAPI middleware,
  :class:`BaseAgent.ainvoke`, custom orchestrators).
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
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from injector import inject
from opentelemetry import baggage
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from ai_core.config.settings import AppSettings, Environment, ObservabilitySettings
from ai_core.di.interfaces import IObservabilityProvider, SpanContext
from ai_core.exceptions import EAAPBaseException
from ai_core.observability.logging import get_logger

_logger = get_logger(__name__)

# Track the active LangFuse trace per async task.
_active_lf_trace: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "ai_core_active_langfuse_trace", default=None
)

_DEV_ENVIRONMENTS: frozenset[Environment] = frozenset(
    {Environment.LOCAL, Environment.DEV}
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
        self._environment: Environment = settings.environment
        self._fail_open: bool = settings.observability.fail_open
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
        attrs.update(_baggage_attributes())

        current = otel_trace.get_current_span()
        if current.is_recording():
            current.add_event("llm.usage", attributes=attrs)

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
            except Exception as exc:  # noqa: BLE001 — observability boundary, controlled by fail_open
                if not self._should_swallow(exc, "record_llm_usage"):
                    raise

    async def record_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """See :meth:`IObservabilityProvider.record_event`."""
        attrs = dict(attributes or {})
        attrs.update(_baggage_attributes())

        current = otel_trace.get_current_span()
        if current.is_recording():
            current.add_event(name, attributes=attrs)

        trace_handle = self._ensure_lf_trace(name=name)
        if trace_handle is not None:
            try:
                trace_handle.event(name=name, metadata=attrs)
            except Exception as exc:  # noqa: BLE001 — observability boundary, controlled by fail_open
                if not self._should_swallow(exc, "record_event"):
                    raise

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
    async def _span(  # noqa: PLR0912 — graceful-degradation paths add necessary branches
        self,
        name: str,
        attributes: Mapping[str, Any] | None,
    ) -> AsyncIterator[SpanContext]:
        attrs = dict(attributes or {})
        # Promote Baggage to span attributes so attribution propagates without
        # the caller having to thread tenant/agent ids through every method.
        attrs.update(_baggage_attributes())

        outer_trace_token: contextvars.Token[Any | None] | None = None

        # Attempt to start the OTel span; respect fail_open on backend failure.
        try:
            _otel_cm = self._tracer.start_as_current_span(name, attributes=attrs)
        except Exception as exc:  # noqa: BLE001 — observability boundary, controlled by fail_open
            if not self._should_swallow(exc, "start_span"):
                raise
            # fail_open=True path: yield a no-op fallback so the caller's body still runs.
            yield SpanContext(name=name, trace_id="0" * 32, span_id="0" * 16, backend_handles={})
            return

        with _otel_cm as otel_span:
            otel_span_ctx = otel_span.get_span_context()
            trace_id_hex = format(otel_span_ctx.trace_id, "032x")
            span_id_hex = format(otel_span_ctx.span_id, "016x")

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
                if isinstance(exc, EAAPBaseException):
                    otel_span.set_attribute("eaap.error.code", exc.error_code)
                    for k, v in (exc.details or {}).items():
                        if isinstance(v, (str, int, float, bool)):
                            otel_span.set_attribute(f"eaap.error.details.{k}", v)
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
        """Build a per-instance :class:`TracerProvider`.

        The provider is also installed as the *global* OTel provider on a
        best-effort basis so that auto-instrumentation in third-party
        libraries lands on the same exporters. The global install is
        "first writer wins" in OTel — subsequent calls are no-ops — so
        we always use the instance's own provider for span emission to
        guarantee that this provider's exporter configuration takes
        effect even if the global slot was claimed earlier.
        """
        new_provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": self._service_name,
                    "deployment.environment": self._environment.value,
                }
            )
        )
        installed_exporter = False
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
                installed_exporter = True
            except Exception as exc:  # noqa: BLE001
                _logger.warning("OTel OTLP exporter setup failed: %s", exc)

        # Dev-mode console exporter: only when nothing else is exporting and
        # the environment is local/dev. SimpleSpanProcessor is intentional
        # here so developers see spans synchronously on stderr.
        if (
            not installed_exporter
            and self._cfg.console_export_in_dev
            and self._environment in _DEV_ENVIRONMENTS
        ):
            new_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        # Global install: skip if a real provider is already installed (the
        # OTel runtime only allows one and would otherwise emit an
        # "Overriding of current TracerProvider is not allowed" warning).
        # Spans always go through ``new_provider.get_tracer(...)`` below, so
        # this provider's exporters fire regardless of the global slot.
        existing = otel_trace.get_tracer_provider()
        if not isinstance(existing, TracerProvider):
            try:
                otel_trace.set_tracer_provider(new_provider)
            except Exception as exc:  # noqa: BLE001
                _logger.debug("OTel global TracerProvider install failed: %s", exc)

        self._otel_provider = new_provider
        return new_provider.get_tracer(self._service_name)

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
    def _should_swallow(self, exc: BaseException, context: str) -> bool:
        """Return True if the caller should swallow the error (fail_open=True).
        Return False if the caller should re-raise via bare `raise`.

        Always logs at WARNING level regardless of return value.
        """
        if self._fail_open:
            _logger.warning(
                "observability.backend_error",
                context=context, error=str(exc), error_type=type(exc).__name__,
                fail_open=True,
            )
            return True
        _logger.warning(
            "observability.backend_error",
            context=context, error=str(exc), error_type=type(exc).__name__,
            fail_open=False,
        )
        return False

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

    def _safe_lf_call(self, call: Callable[[], Any]) -> Any:
        """Invoke a LangFuse call best-effort. ALWAYS swallows backend errors —
        this is teardown shaped (used to end spans, etc.) and must not mask the
        user's original exception when called from inside an except block.
        fail_open does NOT govern this helper.
        """
        try:
            return call()
        except Exception as exc:  # teardown helper; always tolerant
            _logger.warning(
                "langfuse.helper_failed",
                error=str(exc), error_type=type(exc).__name__,
            )
            return None


def _baggage_attributes() -> dict[str, str]:
    """Return ``eaap.*`` Baggage entries as a flat span-attribute mapping."""
    out: dict[str, str] = {}
    for key, value in baggage.get_all().items():
        if key.startswith("eaap."):
            out[key] = str(value)
    return out


__all__ = ["RealObservabilityProvider"]
