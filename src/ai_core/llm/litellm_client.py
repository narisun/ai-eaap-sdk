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
from collections.abc import AsyncIterator, Mapping, Sequence
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

from ai_core.config.settings import LLMSettings
from ai_core.di.interfaces import (
    IBudgetService,
    ILLMClient,
    IObservabilityProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
)
from ai_core.exceptions import BudgetExceededError, ErrorCode, LLMInvocationError, LLMTimeoutError
from ai_core.llm._prompt_cache import apply_prompt_cache

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
        settings: The LLM configuration slice. Pass ``app_settings.llm`` when
            constructing manually; the DI container injects the slice
            automatically.
        budget: :class:`IBudgetService` for pre-call quota check + post-call
            usage recording.
        observability: :class:`IObservabilityProvider` for spans + usage.
    """

    @inject
    def __init__(
        self,
        settings: LLMSettings,
        budget: IBudgetService,
        observability: IObservabilityProvider,
    ) -> None:
        self._cfg = settings
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
        cfg = self._cfg
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
        # Phase 4: apply prompt cache control if model supports it
        cache_cfg = self._cfg
        cached_messages, cached_tools = apply_prompt_cache(
            messages,
            tools=tools,
            model=resolved_model,
            enabled=cache_cfg.prompt_cache_enabled,
            min_messages=cache_cfg.prompt_cache_min_messages,
            min_estimated_tokens=cache_cfg.prompt_cache_min_tokens,
            estimated_tokens=estimated,
        )

        request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": cached_messages,
            "timeout": cfg.request_timeout_seconds,
        }
        if cached_tools is not None:
            request_kwargs["tools"] = cached_tools
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
            except APIError as exc:
                # Non-transient LiteLLM errors (auth, bad request, etc.) — bubble verbatim semantics.
                raise LLMInvocationError(
                    "LLM API rejected the request",
                    details={"model": resolved_model, "status": getattr(exc, "status_code", None)},
                    cause=exc,
                ) from exc
            latency_ms = (time.monotonic() - started) * 1000.0
            response = _normalise_response(resolved_model, raw)  # inside span: errors get tagged

        # --- 3. Record usage (outside span — pure metric emit) ------------------
        cost_usd = _extract_cost(raw)

        await self._observability.record_llm_usage(
            model=resolved_model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            attributes=attributes,
        )
        # Phase 13: latency SLO alert-only check.
        slo_ms = self._cfg.latency_slo_ms
        if slo_ms is not None and latency_ms > slo_ms:
            await self._observability.record_event(
                "llm.slo_violated",
                attributes={
                    "llm.model": resolved_model,
                    "llm.tenant_id": tenant_id or "",
                    "llm.agent_id": agent_id or "",
                    "llm.latency_ms": latency_ms,
                    "llm.threshold_ms": slo_ms,
                },
            )
        await self._budget.record_usage(
            tenant_id=tenant_id,
            agent_id=agent_id,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=cost_usd or 0.0,
        )
        return response

    async def astream(  # noqa: PLR0913 — DI-friendly wide signature, mirrors ILLMClient
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
        """Stream a chat completion via :func:`litellm.acompletion(stream=True)`.

        The same budget pre-check, retry semantics, and observability span
        wrap the open-stream call as :meth:`complete`. Retries only apply
        to opening the stream — once the first chunk arrives, mid-stream
        provider errors are propagated to the caller verbatim. Usage and
        cost are recorded once on the terminal chunk.
        """
        cfg = self._cfg
        resolved_model = model or cfg.default_model
        estimated = _estimate_tokens(messages, resolved_model)
        check = await self._budget.check(
            tenant_id=tenant_id,
            agent_id=agent_id,
            estimated_tokens=estimated,
        )
        if not check.allowed:
            raise BudgetExceededError(
                "Budget pre-check denied LLM invocation (stream)",
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

        cached_messages, cached_tools = apply_prompt_cache(
            messages,
            tools=tools,
            model=resolved_model,
            enabled=cfg.prompt_cache_enabled,
            min_messages=cfg.prompt_cache_min_messages,
            min_estimated_tokens=cfg.prompt_cache_min_tokens,
            estimated_tokens=estimated,
        )

        request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": cached_messages,
            "timeout": cfg.request_timeout_seconds,
            "stream": True,
        }
        if cached_tools is not None:
            request_kwargs["tools"] = cached_tools
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
            "llm.stream": True,
        }
        return self._astream_iter(
            request_kwargs=request_kwargs,
            attributes=attributes,
            resolved_model=resolved_model,
            tenant_id=tenant_id,
            agent_id=agent_id,
        )

    async def _astream_iter(
        self,
        *,
        request_kwargs: Mapping[str, Any],
        attributes: Mapping[str, Any],
        resolved_model: str,
        tenant_id: str | None,
        agent_id: str | None,
    ) -> AsyncIterator[LLMStreamChunk]:
        cfg = self._cfg
        async with self._observability.start_span("llm.astream", attributes=attributes):
            started = time.monotonic()
            try:
                stream = await self._open_stream_with_retry(request_kwargs)
            except RetryError as exc:
                last = exc.last_attempt.exception() if exc.last_attempt else exc
                if isinstance(last, Timeout):
                    raise LLMTimeoutError(
                        f"LLM stream open timed out after {cfg.max_retries + 1} attempts",
                        details={"model": resolved_model, "attempts": cfg.max_retries + 1},
                        cause=last,
                    ) from last
                raise LLMInvocationError(
                    "LLM stream open failed after retries",
                    details={"model": resolved_model, "attempts": cfg.max_retries + 1},
                    cause=last,
                ) from last
            except APIError as exc:
                raise LLMInvocationError(
                    "LLM API rejected the stream request",
                    details={"model": resolved_model, "status": getattr(exc, "status_code", None)},
                    cause=exc,
                ) from exc

            running_prompt = 0
            running_completion = 0
            running_cost: float | None = None
            terminal_seen = False
            async for raw_chunk in stream:
                normalised = _normalise_stream_chunk(resolved_model, raw_chunk)
                if normalised.usage is not None:
                    running_prompt = normalised.usage.prompt_tokens or running_prompt
                    running_completion = (
                        normalised.usage.completion_tokens or running_completion
                    )
                    if normalised.usage.cost_usd is not None:
                        running_cost = normalised.usage.cost_usd
                if normalised.finish_reason is not None:
                    terminal_seen = True
                yield normalised

            latency_ms = (time.monotonic() - started) * 1000.0
            # Record usage/cost once at the end of the stream so behaviour
            # mirrors complete(); fall back to running totals when the
            # provider didn't send a final usage block.
            if not terminal_seen:
                # Stream terminated without a finish_reason — surface as an
                # invocation error so callers don't silently truncate.
                raise LLMInvocationError(
                    "LLM stream ended without a terminal finish_reason chunk",
                    details={"model": resolved_model},
                    error_code=ErrorCode.LLM_EMPTY_RESPONSE,
                )
            await self._observability.record_llm_usage(
                model=resolved_model,
                prompt_tokens=running_prompt,
                completion_tokens=running_completion,
                latency_ms=latency_ms,
                cost_usd=running_cost,
                attributes=attributes,
            )
            slo_ms = self._cfg.latency_slo_ms
            if slo_ms is not None and latency_ms > slo_ms:
                await self._observability.record_event(
                    "llm.slo_violated",
                    attributes={
                        "llm.model": resolved_model,
                        "llm.tenant_id": tenant_id or "",
                        "llm.agent_id": agent_id or "",
                        "llm.latency_ms": latency_ms,
                        "llm.threshold_ms": slo_ms,
                        "llm.stream": True,
                    },
                )
            await self._budget.record_usage(
                tenant_id=tenant_id,
                agent_id=agent_id,
                prompt_tokens=running_prompt,
                completion_tokens=running_completion,
                cost_usd=running_cost or 0.0,
            )

    async def _open_stream_with_retry(self, request_kwargs: Mapping[str, Any]) -> Any:
        cfg = self._cfg
        retrying = AsyncRetrying(
            reraise=False,
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
        raise LLMInvocationError("Unreachable retry exit")  # pragma: no cover

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------
    async def _call_with_retry(self, request_kwargs: Mapping[str, Any]) -> Any:
        cfg = self._cfg
        retrying = AsyncRetrying(
            reraise=False,  # Wrap exhausted retries in RetryError so we see attempts info.
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
        # Unreachable: AsyncRetrying with reraise=False raises RetryError on exhaustion.
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
                "raw_keys": sorted(str(k) for k in payload),
            },
            error_code=ErrorCode.LLM_EMPTY_RESPONSE,
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


def _normalise_stream_chunk(model: str, raw: Any) -> LLMStreamChunk:
    """Convert a LiteLLM streaming chunk into an :class:`LLMStreamChunk`.

    LiteLLM's chunk shape mirrors OpenAI's: ``choices[0].delta`` carries
    incremental ``content`` / ``tool_calls`` / ``role``; ``finish_reason``
    is set only on the terminal chunk; final ``usage`` arrives either on
    the terminal chunk or as a trailing ``[DONE]``-style record.
    """
    payload: Mapping[str, Any] = (
        raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
    )
    choices = payload.get("choices") or []
    first = choices[0] if choices else {}
    delta = first.get("delta") or {}
    content = delta.get("content") or ""
    tool_calls = list(delta.get("tool_calls") or [])
    finish_reason = first.get("finish_reason")
    usage_blob = payload.get("usage") or {}
    usage: LLMUsage | None = None
    if usage_blob:
        prompt = int(usage_blob.get("prompt_tokens") or 0)
        completion = int(usage_blob.get("completion_tokens") or 0)
        usage = LLMUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=int(
                usage_blob.get("total_tokens") or (prompt + completion)
            ),
            cost_usd=_extract_cost(raw),
        )
    return LLMStreamChunk(
        model=str(payload.get("model") or model),
        delta_content=str(content),
        delta_tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        raw=payload,
    )


__all__ = ["LiteLLMClient"]
