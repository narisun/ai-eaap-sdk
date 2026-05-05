"""Concrete :class:`ILLMClient` powered by `litellm <https://docs.litellm.ai>`_.

Responsibilities:

1. **Budget enforcement** — call :meth:`IBudgetService.check` with an
   estimated token count *before* spending money.
2. **Retry on transient failures** — wrap :func:`litellm.acompletion` in
   a Tenacity ``AsyncRetrying`` block with exponential backoff, retrying
   only on rate-limit / 5xx / connection / timeout errors.
3. **Usage capture** — on success, push token + cost telemetry to both
   :class:`IObservabilityProvider` and :class:`IBudgetService` so that
   subsequent budget checks reflect actual spend.

The client never logs the raw prompt/response content — operators get
shape (token counts, model, latency) by default and can opt into full
content capture through their observability provider.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import litellm
from injector import inject
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import (
    IBudgetService,
    ILLMClient,
    IObservabilityProvider,
    LLMResponse,
    LLMUsage,
)
from ai_core.exceptions import BudgetExceededError, LLMInvocationError, LLMTimeoutError

# Tenacity retries on these error types only — auth / bad-request must surface immediately.
_TRANSIENT_LLM_ERRORS: tuple[type[BaseException], ...] = (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    ServiceUnavailableError,
    Timeout,
)


def _estimate_tokens(messages: Sequence[Mapping[str, Any]], model: str) -> int:
    """Best-effort prompt-token estimate using LiteLLM's tokenizer."""
    try:
        return int(litellm.token_counter(model=model, messages=list(messages)))
    except Exception:  # noqa: BLE001 — fall back when tokeniser fails
        # Heuristic: ~4 chars/token. Good enough for budget pre-check.
        approx = sum(len(str(m.get("content", ""))) for m in messages)
        return max(1, approx // 4)


class LiteLLMClient(ILLMClient):
    """Production :class:`ILLMClient` wired with budgeting + retries + tracing.

    Args:
        settings: Aggregated application settings.
        budget: :class:`IBudgetService` for pre-call quota check + post-call
            usage recording.
        observability: :class:`IObservabilityProvider` for spans + usage.
    """

    @inject
    def __init__(
        self,
        settings: AppSettings,
        budget: IBudgetService,
        observability: IObservabilityProvider,
    ) -> None:
        self._settings = settings
        self._budget = budget
        self._observability = observability

    async def complete(  # noqa: PLR0913 — DI-friendly wide signature, mirrors ILLMClient
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
        """See :meth:`ILLMClient.complete`."""
        cfg = self._settings.llm
        resolved_model = model or cfg.default_model

        # --- 1. Budget pre-check --------------------------------------------------
        estimated = _estimate_tokens(messages, resolved_model)
        check = await self._budget.check(
            tenant_id=tenant_id,
            agent_id=agent_id,
            estimated_tokens=estimated,
        )
        if not check.allowed:
            raise BudgetExceededError(
                "Budget pre-check denied LLM invocation",
                details={
                    "tenant_id": tenant_id,
                    "agent_id": agent_id,
                    "model": resolved_model,
                    "estimated_tokens": estimated,
                    "remaining_tokens": check.remaining_tokens,
                    "remaining_usd": check.remaining_usd,
                    "reason": check.reason,
                },
            )

        # --- 2. Retry-wrapped LLM call -------------------------------------------
        request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": list(messages),
            "timeout": cfg.request_timeout_seconds,
        }
        if tools is not None:
            request_kwargs["tools"] = list(tools)
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if cfg.proxy_base_url is not None:
            request_kwargs["api_base"] = str(cfg.proxy_base_url)
        if cfg.proxy_api_key is not None:
            request_kwargs["api_key"] = cfg.proxy_api_key.get_secret_value()
        if extra:
            request_kwargs.update(dict(extra))

        attributes = {
            "llm.model": resolved_model,
            "llm.tenant_id": tenant_id or "",
            "llm.agent_id": agent_id or "",
            "llm.estimated_tokens": estimated,
        }

        async with self._observability.start_span("llm.complete", attributes=attributes):
            started = time.monotonic()
            try:
                raw = await self._call_with_retry(request_kwargs)
            except RetryError as exc:
                last = exc.last_attempt.exception() if exc.last_attempt else exc
                if isinstance(last, Timeout):
                    raise LLMTimeoutError(
                        f"LLM call timed out after {cfg.max_retries + 1} attempts",
                        details={"model": resolved_model, "attempts": cfg.max_retries + 1},
                        cause=last,
                    ) from last
                raise LLMInvocationError(
                    "LLM invocation failed after retries",
                    details={"model": resolved_model, "attempts": cfg.max_retries + 1},
                    cause=last,
                ) from last
            except Timeout as exc:
                raise LLMTimeoutError(
                    f"LLM call timed out after {cfg.max_retries + 1} attempts",
                    details={"model": resolved_model, "attempts": cfg.max_retries + 1},
                    cause=exc,
                ) from exc
            except _TRANSIENT_LLM_ERRORS as exc:
                raise LLMInvocationError(
                    "LLM invocation failed",
                    details={"model": resolved_model},
                    cause=exc,
                ) from exc
            except APIError as exc:
                # Non-transient LiteLLM errors (auth, bad request, etc.) — bubble verbatim semantics.
                raise LLMInvocationError(
                    "LLM API rejected the request",
                    details={"model": resolved_model, "status": getattr(exc, "status_code", None)},
                    cause=exc,
                ) from exc
            latency_ms = (time.monotonic() - started) * 1000.0

        # --- 3. Normalise response + record usage -------------------------------
        response = _normalise_response(resolved_model, raw)
        cost_usd = _extract_cost(raw)

        await self._observability.record_llm_usage(
            model=resolved_model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            attributes=attributes,
        )
        await self._budget.record_usage(
            tenant_id=tenant_id,
            agent_id=agent_id,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=cost_usd or 0.0,
        )
        return response

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------
    async def _call_with_retry(self, request_kwargs: Mapping[str, Any]) -> Any:
        cfg = self._settings.llm
        retrying = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(cfg.max_retries + 1),
            wait=wait_exponential(
                multiplier=cfg.retry_initial_backoff_seconds,
                max=cfg.retry_max_backoff_seconds,
            ),
            retry=retry_if_exception_type(_TRANSIENT_LLM_ERRORS),
        )
        async for attempt in retrying:
            with attempt:
                return await litellm.acompletion(**request_kwargs)
        # Unreachable: AsyncRetrying with reraise=True always returns or raises.
        raise LLMInvocationError("Unreachable retry exit")  # pragma: no cover


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _normalise_response(model: str, raw: Any) -> LLMResponse:
    """Convert a LiteLLM response object/dict into an :class:`LLMResponse`.

    Raises:
        LLMInvocationError: If the response has neither content nor tool_calls
            (silent-data-loss case — likely truncation or content filter).
    """
    payload: Mapping[str, Any] = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)

    choices = payload.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    content = message.get("content") or ""
    tool_calls = list(message.get("tool_calls") or [])
    finish_reason = first.get("finish_reason")

    # Silent-data-loss detection: no content AND no tool_calls means the response
    # was either truncated, content-filtered, or malformed upstream. Raise so the
    # caller doesn't silently propagate an empty assistant turn into the agent loop.
    if not content and not tool_calls:
        raise LLMInvocationError(
            f"LLM returned empty response (finish_reason={finish_reason!r})",
            details={
                "model": model,
                "finish_reason": finish_reason,
                "raw_keys": sorted(str(k) for k in payload.keys()),
            },
            error_code="llm.empty_response",
        )

    usage_blob = payload.get("usage") or {}
    usage = LLMUsage(
        prompt_tokens=int(usage_blob.get("prompt_tokens", 0)),
        completion_tokens=int(usage_blob.get("completion_tokens", 0)),
        total_tokens=int(
            usage_blob.get(
                "total_tokens",
                int(usage_blob.get("prompt_tokens", 0))
                + int(usage_blob.get("completion_tokens", 0)),
            )
        ),
        cost_usd=_extract_cost(raw),
    )
    return LLMResponse(
        model=str(payload.get("model") or model),
        content=str(content),
        tool_calls=tool_calls,
        usage=usage,
        raw=payload,
        finish_reason=finish_reason,
    )


def _extract_cost(raw: Any) -> float | None:
    """Pull a USD cost off a LiteLLM response if it provides one.

    LiteLLM injects ``response_cost`` either as an attribute or under
    ``_hidden_params``. Be permissive — return ``None`` when absent.
    """
    if raw is None:
        return None
    direct = getattr(raw, "response_cost", None)
    if isinstance(direct, (int, float)):
        return float(direct)
    hidden = getattr(raw, "_hidden_params", None) or {}
    cost = hidden.get("response_cost") if isinstance(hidden, Mapping) else None
    if isinstance(cost, (int, float)):
        return float(cost)
    if isinstance(raw, Mapping):
        cost = raw.get("response_cost")
        if isinstance(cost, (int, float)):
            return float(cost)
    return None


__all__ = ["LiteLLMClient"]
