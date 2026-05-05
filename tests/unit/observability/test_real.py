"""Unit tests for :class:`ai_core.observability.real.RealObservabilityProvider`.

The tests intentionally avoid spinning up a real OTel collector or
LangFuse server. OTel is exercised in *in-process* mode (no exporter
configured); LangFuse is mocked at the constructor level so the
provider's interactions can be observed deterministically.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_core.config.settings import AppSettings, ObservabilitySettings
from ai_core.exceptions import LLMInvocationError
from ai_core.observability.real import RealObservabilityProvider


pytestmark = pytest.mark.unit


def _settings_no_external() -> AppSettings:
    """No OTel endpoint, no LangFuse keys — exercises degraded-mode."""
    return AppSettings(service_name="obs-test")


def _settings_with_langfuse() -> AppSettings:
    return AppSettings(
        service_name="obs-test",
        observability={  # type: ignore[arg-type]
            "service_name": "obs-test",
            "langfuse_public_key": "pk-test",
            "langfuse_secret_key": "sk-test",
            "langfuse_host": "http://localhost:3000",
        },
    )


# ---------------------------------------------------------------------------
# Degraded mode
# ---------------------------------------------------------------------------
async def test_start_span_works_without_any_backend() -> None:
    provider = RealObservabilityProvider(_settings_no_external())
    async with provider.start_span("op.x", attributes={"k": 1}) as ctx:
        assert ctx.name == "op.x"
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16
        assert "otel" in ctx.backend_handles
        assert ctx.backend_handles["langfuse"] is None


async def test_record_llm_usage_no_op_without_langfuse() -> None:
    provider = RealObservabilityProvider(_settings_no_external())
    # Should not raise.
    await provider.record_llm_usage(
        model="m", prompt_tokens=10, completion_tokens=5, latency_ms=12.0, cost_usd=0.01
    )


async def test_record_event_no_op_without_langfuse() -> None:
    provider = RealObservabilityProvider(_settings_no_external())
    await provider.record_event("agent.thought", attributes={"k": "v"})


async def test_shutdown_idempotent() -> None:
    provider = RealObservabilityProvider(_settings_no_external())
    await provider.shutdown()
    await provider.shutdown()


# ---------------------------------------------------------------------------
# LangFuse delegation (mocked)
# ---------------------------------------------------------------------------
def _build_with_mock_langfuse(monkeypatch: pytest.MonkeyPatch) -> tuple[RealObservabilityProvider, MagicMock]:
    fake_trace = MagicMock(name="lf_trace")
    fake_trace.span = MagicMock(return_value=MagicMock(name="lf_span"))
    fake_trace.generation = MagicMock(name="lf_generation")
    fake_trace.event = MagicMock(name="lf_event")

    fake_client = MagicMock(name="Langfuse")
    fake_client.trace = MagicMock(return_value=fake_trace)
    fake_client.flush = MagicMock()

    def _fake_init(self: RealObservabilityProvider) -> Any:
        return fake_client

    monkeypatch.setattr(RealObservabilityProvider, "_init_langfuse", _fake_init)

    provider = RealObservabilityProvider(_settings_with_langfuse())
    return provider, fake_trace


async def test_start_span_creates_langfuse_trace_and_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, fake_trace = _build_with_mock_langfuse(monkeypatch)
    async with provider.start_span("agent.thought", attributes={"step": 1}) as ctx:
        assert ctx.backend_handles["langfuse"] is fake_trace.span.return_value
    fake_trace.span.assert_called_once()
    fake_trace.span.return_value.end.assert_called_once()


async def test_record_llm_usage_writes_langfuse_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, fake_trace = _build_with_mock_langfuse(monkeypatch)
    async with provider.start_span("llm.call"):
        await provider.record_llm_usage(
            model="claude-sonnet-4-6",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=120.5,
            cost_usd=0.0125,
            attributes={"agent.id": "a-1"},
        )
    fake_trace.generation.assert_called_once()
    args = fake_trace.generation.call_args.kwargs
    assert args["model"] == "claude-sonnet-4-6"
    assert args["usage"]["input"] == 100
    assert args["usage"]["output"] == 50
    assert args["usage"]["total_cost"] == pytest.approx(0.0125)


async def test_record_event_writes_langfuse_event(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, fake_trace = _build_with_mock_langfuse(monkeypatch)
    async with provider.start_span("agent.run"):
        await provider.record_event("tool.invoked", attributes={"tool": "search"})
    fake_trace.event.assert_called_once()


async def test_exception_inside_span_is_recorded_then_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, fake_trace = _build_with_mock_langfuse(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        async with provider.start_span("op.x"):
            raise RuntimeError("boom")
    fake_trace.span.return_value.end.assert_called_once()
    end_kwargs = fake_trace.span.return_value.end.call_args.kwargs
    assert end_kwargs.get("level") == "ERROR"


async def test_shutdown_flushes_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _build_with_mock_langfuse(monkeypatch)
    await provider.shutdown()
    provider._langfuse.flush.assert_called_once()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Dev-mode console exporter
# ---------------------------------------------------------------------------
def _has_console_processor(provider: RealObservabilityProvider) -> bool:
    """Return True if the per-instance TracerProvider has a console exporter."""
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    processors = provider._otel_provider._active_span_processor._span_processors  # type: ignore[attr-defined]
    return any(
        isinstance(p, SimpleSpanProcessor)
        and isinstance(p.span_exporter, ConsoleSpanExporter)
        for p in processors
    )


def test_console_exporter_installed_in_local_when_no_endpoint() -> None:
    settings = AppSettings(
        environment="local",
        observability={  # type: ignore[arg-type]
            "service_name": "dev-console-test",
            "console_export_in_dev": True,
        },
    )
    provider = RealObservabilityProvider(settings)
    assert _has_console_processor(provider)


def test_console_exporter_NOT_installed_in_prod() -> None:
    settings = AppSettings(
        environment="prod",
        observability={"service_name": "prod-test"},  # type: ignore[arg-type]
    )
    provider = RealObservabilityProvider(settings)
    assert not _has_console_processor(provider)


def test_console_exporter_disabled_when_flag_off() -> None:
    settings = AppSettings(
        environment="local",
        observability={  # type: ignore[arg-type]
            "service_name": "no-console",
            "console_export_in_dev": False,
        },
    )
    provider = RealObservabilityProvider(settings)
    assert not _has_console_processor(provider)


# ---------------------------------------------------------------------------
# Baggage propagation into span attributes
# ---------------------------------------------------------------------------
async def test_baggage_eaap_keys_copied_onto_span_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When eaap.* baggage is set in context, _span() must copy it onto the OTel span."""
    from opentelemetry import baggage
    from opentelemetry import context as otel_context

    provider = RealObservabilityProvider(_settings_no_external())

    ctx = otel_context.get_current()
    ctx = baggage.set_baggage("eaap.tenant_id", "acme", context=ctx)
    ctx = baggage.set_baggage("eaap.agent_id", "agent-1", context=ctx)
    # Non-eaap baggage must NOT be copied.
    ctx = baggage.set_baggage("other", "leaked", context=ctx)
    token = otel_context.attach(ctx)
    try:
        async with provider.start_span("agent.thought") as span_ctx:
            otel_span = span_ctx.backend_handles["otel"]
            attrs = dict(otel_span.attributes or {})
            assert attrs.get("eaap.tenant_id") == "acme"
            assert attrs.get("eaap.agent_id") == "agent-1"
            assert "other" not in attrs
    finally:
        otel_context.detach(token)


async def test_baggage_promoted_into_langfuse_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry import baggage
    from opentelemetry import context as otel_context

    provider, fake_trace = _build_with_mock_langfuse(monkeypatch)
    ctx = otel_context.get_current()
    ctx = baggage.set_baggage("eaap.tenant_id", "acme", context=ctx)
    token = otel_context.attach(ctx)
    try:
        async with provider.start_span("op"):
            pass
    finally:
        otel_context.detach(token)

    # The metadata kwarg passed to lf.span() should include eaap.tenant_id.
    metadata_seen = fake_trace.span.call_args.kwargs["metadata"]
    assert metadata_seen.get("eaap.tenant_id") == "acme"


# ---------------------------------------------------------------------------
# fail_open toggle + error.code emission (Phase 2 Task 4)
# ---------------------------------------------------------------------------
def _settings(*, fail_open: bool) -> AppSettings:
    s = AppSettings()
    s.observability = ObservabilitySettings(fail_open=fail_open)
    return s


@pytest.mark.asyncio
async def test_fail_open_true_swallows_backend_error() -> None:
    """When fail_open=True (default), backend errors are logged but not raised."""
    provider = RealObservabilityProvider(_settings(fail_open=True))

    # Force an exception inside the span's tracer code by patching the tracer.
    with patch.object(provider, "_tracer", new=MagicMock()):
        provider._tracer.start_as_current_span.side_effect = RuntimeError("backend down")
        # start_span must not raise — fail_open swallows.
        async with provider.start_span("x", attributes={}):
            pass


@pytest.mark.asyncio
async def test_fail_open_false_raises_backend_error() -> None:
    """When fail_open=False, backend errors propagate."""
    provider = RealObservabilityProvider(_settings(fail_open=False))

    with patch.object(provider, "_tracer", new=MagicMock()):
        provider._tracer.start_as_current_span.side_effect = RuntimeError("backend down")
        with pytest.raises(RuntimeError, match="backend down"):
            async with provider.start_span("x", attributes={}):
                pass


@pytest.mark.asyncio
async def test_eaap_exception_tags_error_code_on_span() -> None:
    """When an EAAPBaseException propagates inside a span, set_attribute fires with error.code."""
    provider = RealObservabilityProvider(_settings(fail_open=True))

    # Build a mock span context object with integer trace/span ids so that
    # format(..., "032x") / format(..., "016x") in _span() succeeds.
    fake_span_context = MagicMock()
    fake_span_context.trace_id = 0
    fake_span_context.span_id = 0

    fake_span = MagicMock()
    fake_span.get_span_context.return_value = fake_span_context

    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_span)
    fake_cm.__exit__ = MagicMock(return_value=False)

    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value = fake_cm

    with patch.object(provider, "_tracer", new=fake_tracer), pytest.raises(LLMInvocationError):
        async with provider.start_span("test.span", attributes={}):
            raise LLMInvocationError(
                "some failure",
                details={"model": "gpt-x", "attempts": 3},
            )

    # Verify error.code attribute was set.
    fake_span.set_attribute.assert_any_call("error.code", "llm.invocation_failed")
    # Verify scalar details landed as attributes.
    fake_span.set_attribute.assert_any_call("error.details.model", "gpt-x")
    fake_span.set_attribute.assert_any_call("error.details.attempts", 3)
