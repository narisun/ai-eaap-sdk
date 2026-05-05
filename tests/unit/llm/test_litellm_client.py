"""Unit tests for :class:`ai_core.llm.litellm_client.LiteLLMClient`.

LiteLLM's network calls are mocked at the module-attribute level so
that no real model is contacted. The transient-error path is exercised
by raising ``litellm.exceptions.RateLimitError`` from the mock for the
first N calls and verifying the client retries until success.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
from litellm.exceptions import APIConnectionError, RateLimitError, Timeout as LiteLLMTimeout

from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import (
    BudgetCheck,
    IBudgetService,
    IObservabilityProvider,
    SpanContext,
)
from ai_core.exceptions import BudgetExceededError, LLMInvocationError, LLMTimeoutError
from ai_core.llm.litellm_client import LiteLLMClient, _normalise_response
from ai_core.observability.noop import NoOpObservabilityProvider
from tests.conftest import FakeBudgetService


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeBudget(IBudgetService):
    def __init__(
        self,
        *,
        allowed: bool = True,
        reason: str | None = None,
    ) -> None:
        self.allowed = allowed
        self.reason = reason
        self.checks: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []

    async def check(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        estimated_tokens: int,
    ) -> BudgetCheck:
        self.checks.append(
            {"tenant_id": tenant_id, "agent_id": agent_id, "estimated_tokens": estimated_tokens}
        )
        return BudgetCheck(
            allowed=self.allowed,
            remaining_tokens=10_000,
            remaining_usd=10.0,
            reason=self.reason,
        )

    async def record_usage(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        self.records.append(
            {
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
            }
        )


class RecordingObservability(NoOpObservabilityProvider):
    def __init__(self) -> None:
        self.usage_calls: list[dict[str, Any]] = []
        self.events: list[tuple[str, Mapping[str, Any] | None]] = []

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
        self.usage_calls.append(
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
        self.events.append((name, attributes))


def _fake_response(content: str = "hello", *, prompt: int = 12, completion: int = 7) -> dict[str, Any]:
    return {
        "model": "fake/model",
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": []}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        "response_cost": 0.00123,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_client(
    *,
    budget: IBudgetService | None = None,
    observability: IObservabilityProvider | None = None,
    max_retries: int = 2,
) -> tuple[LiteLLMClient, FakeBudget, RecordingObservability]:
    settings = AppSettings(
        llm={  # type: ignore[arg-type]
            "default_model": "fake/model",
            "max_retries": max_retries,
            "retry_initial_backoff_seconds": 0.001,
            "retry_max_backoff_seconds": 0.002,
        },
    )
    fake_budget = budget if isinstance(budget, FakeBudget) else FakeBudget()
    obs = observability if isinstance(observability, RecordingObservability) else RecordingObservability()
    client = LiteLLMClient(settings, fake_budget, obs)
    return client, fake_budget, obs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
async def test_happy_path_returns_normalised_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client, budget, obs = _build_client()
    mock_acompletion = AsyncMock(return_value=_fake_response("ok"))
    monkeypatch.setattr("ai_core.llm.litellm_client.litellm.acompletion", mock_acompletion)

    response = await client.complete(
        model=None,
        messages=[{"role": "user", "content": "hi"}],
        tenant_id="tenant-a",
        agent_id="agent-1",
    )

    assert response.content == "ok"
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 19
    assert response.usage.cost_usd == pytest.approx(0.00123)
    assert mock_acompletion.await_count == 1
    assert budget.checks[0]["tenant_id"] == "tenant-a"
    assert budget.records[0]["prompt_tokens"] == 12
    assert obs.usage_calls[0]["model"] == "fake/model"


async def test_budget_denied_raises_without_calling_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = _build_client(
        budget=FakeBudget(allowed=False, reason="quota"),
    )
    mock_acompletion = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr("ai_core.llm.litellm_client.litellm.acompletion", mock_acompletion)

    with pytest.raises(BudgetExceededError) as ei:
        await client.complete(model=None, messages=[{"role": "user", "content": "hi"}])

    assert ei.value.details["reason"] == "quota"
    assert mock_acompletion.await_count == 0


async def test_retries_on_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _build_client(max_retries=3)
    mock_acompletion = AsyncMock(
        side_effect=[
            RateLimitError("slow down", "fake-llm", "fake/model"),
            APIConnectionError("flaky", "fake-llm", "fake/model"),
            _fake_response("recovered"),
        ]
    )
    monkeypatch.setattr("ai_core.llm.litellm_client.litellm.acompletion", mock_acompletion)

    response = await client.complete(
        model=None,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.content == "recovered"
    assert mock_acompletion.await_count == 3


async def test_exhausts_retries_and_raises_invocation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = _build_client(max_retries=2)
    mock_acompletion = AsyncMock(
        side_effect=RateLimitError("permanent", "fake-llm", "fake/model"),
    )
    monkeypatch.setattr("ai_core.llm.litellm_client.litellm.acompletion", mock_acompletion)

    with pytest.raises(LLMInvocationError) as ei:
        await client.complete(model=None, messages=[{"role": "user", "content": "hi"}])

    assert ei.value.details["model"] == "fake/model"
    # max_retries=2 => total attempts = 3
    assert mock_acompletion.await_count == 3


async def test_observability_called_with_token_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, obs = _build_client()
    monkeypatch.setattr(
        "ai_core.llm.litellm_client.litellm.acompletion",
        AsyncMock(return_value=_fake_response("y")),
    )

    await client.complete(
        model=None,
        messages=[{"role": "user", "content": "hi"}],
        tenant_id="t",
        agent_id="a",
    )

    assert len(obs.usage_calls) == 1
    call = obs.usage_calls[0]
    assert call["prompt_tokens"] == 12
    assert call["completion_tokens"] == 7
    assert call["attributes"]["llm.tenant_id"] == "t"
    assert call["attributes"]["llm.agent_id"] == "a"


async def test_extra_kwargs_forwarded_to_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _build_client()
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr("ai_core.llm.litellm_client.litellm.acompletion", fake_acompletion)

    await client.complete(
        model="override/model",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.2,
        max_tokens=128,
        extra={"top_p": 0.9},
    )

    assert captured["model"] == "override/model"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 128
    assert captured["top_p"] == 0.9


# ---------------------------------------------------------------------------
# §2b — _normalise_response: finish_reason + empty-response detection
# ---------------------------------------------------------------------------

def _raw(
    *,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    model: str = "gpt-x",
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{
            "message": {"content": content, "tool_calls": tool_calls or []},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_normal_content_response() -> None:
    response = _normalise_response("gpt-x", _raw(content="hi", finish_reason="stop"))
    assert response.content == "hi"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"


def test_tool_only_response_succeeds() -> None:
    """content=='' AND tool_calls!=[] is a valid function-call response — must NOT raise."""
    response = _normalise_response("gpt-x", _raw(
        content="",
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "x", "arguments": "{}"}}],
        finish_reason="tool_calls",
    ))
    assert response.content == ""
    assert response.tool_calls != []
    assert response.finish_reason == "tool_calls"


def test_empty_response_raises_llm_invocation_error() -> None:
    """content=='' AND tool_calls==[] is the silent-data-loss case — must raise."""
    with pytest.raises(LLMInvocationError) as exc:
        _normalise_response("gpt-x", _raw(content="", finish_reason="length"))
    assert exc.value.error_code == "llm.empty_response"
    assert exc.value.details["finish_reason"] == "length"
    assert exc.value.details["model"] == "gpt-x"
    assert "raw_keys" in exc.value.details


def test_empty_response_with_no_finish_reason() -> None:
    with pytest.raises(LLMInvocationError) as exc:
        _normalise_response("gpt-x", _raw(content=""))
    assert exc.value.details["finish_reason"] is None


def test_finish_reason_none_when_upstream_omits() -> None:
    """If choices[0] has no finish_reason field, LLMResponse.finish_reason is None."""
    raw = _raw(content="hello")
    raw["choices"][0].pop("finish_reason", None)
    response = _normalise_response("gpt-x", raw)
    assert response.finish_reason is None


# ---------------------------------------------------------------------------
# §2c — LLMTimeoutError mapping after retry exhaustion
# ---------------------------------------------------------------------------

class _NoOpObservability(IObservabilityProvider):
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> Any:
        @asynccontextmanager
        async def _cm() -> Any:
            yield SpanContext(name=name, trace_id="t", span_id="s", backend_handles={})
        return _cm()

    async def record_llm_usage(self, **kwargs: Any) -> None:
        return None

    async def record_event(self, name: str, *, attributes: Mapping[str, Any] | None = None) -> None:
        return None

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_retry_exhausted_timeout_raises_llm_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: E501
    """litellm.Timeout after retry exhaustion -> LLMTimeoutError (not LLMInvocationError)."""
    settings = AppSettings()
    settings.llm.max_retries = 0  # one attempt; fail fast
    settings.llm.retry_initial_backoff_seconds = 0.01
    settings.llm.retry_max_backoff_seconds = 0.01

    async def _always_timeout(**kwargs: Any) -> Any:
        raise LiteLLMTimeout(
            message="upstream timed out",
            model="x",
            llm_provider="test",
        )

    monkeypatch.setattr("litellm.acompletion", _always_timeout)

    client = LiteLLMClient(
        settings=settings,
        budget=FakeBudgetService(),
        observability=_NoOpObservability(),
    )
    with pytest.raises(LLMTimeoutError) as exc:
        await client.complete(model="gpt-x", messages=[{"role": "user", "content": "hi"}])
    assert exc.value.error_code == "llm.timeout"
    assert exc.value.details["model"] == "gpt-x"
    # Assert we went through the RetryError path (not a direct exception type).
    assert exc.value.details["attempts"] == 1  # max_retries=0 -> 1 attempt total
    assert "after" in exc.value.message.lower() or "timed out" in exc.value.message.lower()


@pytest.mark.asyncio
async def test_retry_exhausted_non_timeout_raises_llm_invocation_error(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: E501
    """Non-timeout transient error after retries -> generic LLMInvocationError."""
    settings = AppSettings()
    settings.llm.max_retries = 0
    settings.llm.retry_initial_backoff_seconds = 0.01
    settings.llm.retry_max_backoff_seconds = 0.01

    async def _always_429(**kwargs: Any) -> Any:
        raise RateLimitError(
            message="rate limited",
            model="x",
            llm_provider="test",
        )

    monkeypatch.setattr("litellm.acompletion", _always_429)

    client = LiteLLMClient(
        settings=settings,
        budget=FakeBudgetService(),
        observability=_NoOpObservability(),
    )
    with pytest.raises(LLMInvocationError) as exc:
        await client.complete(model="gpt-x", messages=[{"role": "user", "content": "hi"}])
    # Generic invocation_failed, NOT llm.timeout.
    assert exc.value.error_code == "llm.invocation_failed"
    assert not isinstance(exc.value, LLMTimeoutError)
    # Assert we went through the RetryError path (not _TRANSIENT_LLM_ERRORS).
    assert exc.value.details["attempts"] == 1
    assert "after retries" in exc.value.message


@pytest.mark.asyncio
async def test_empty_response_tags_llm_complete_span(
    monkeypatch: pytest.MonkeyPatch,
    fake_observability: Any,
    fake_budget: Any,
) -> None:
    """An empty LLM response must propagate inside the llm.complete span so
    eaap.error.code='llm.empty_response' is auto-emitted."""
    settings = AppSettings()
    settings.llm.max_retries = 0

    async def _empty_response(**kwargs: Any) -> Any:
        return {
            "model": "gpt-x",
            "choices": [{"message": {"content": "", "tool_calls": []},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        }

    monkeypatch.setattr("litellm.acompletion", _empty_response)

    client = LiteLLMClient(
        settings=settings,
        budget=fake_budget,
        observability=fake_observability,
    )
    with pytest.raises(LLMInvocationError) as exc:
        await client.complete(model="gpt-x",
                              messages=[{"role": "user", "content": "hi"}])
    assert exc.value.error_code == "llm.empty_response"
    spans = [s for s in fake_observability.spans if s.name == "llm.complete"]
    assert len(spans) == 1
    assert spans[0].error_code == "llm.empty_response"
