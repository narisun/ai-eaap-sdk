"""ScriptedLLM + make_llm_response builder for agent tests.

These helpers consolidate the per-test ad-hoc LLM fakes (FakeLLM,
_StubLLM, _ScriptedLLM, etc.) into a single canonical API. The
ScriptedLLM matches the full ``ILLMClient.complete`` signature so it
satisfies the abstract base class under mypy strict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMStreamChunk, LLMUsage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence


_DEFAULT_PROMPT_TOKENS = 10
_DEFAULT_COMPLETION_TOKENS = 20
_DEFAULT_MODEL = "test-model"


def make_llm_response(
    text: str = "",
    *,
    tool_calls: Sequence[Mapping[str, Any]] = (),
    finish_reason: str = "stop",
    prompt_tokens: int = _DEFAULT_PROMPT_TOKENS,
    completion_tokens: int = _DEFAULT_COMPLETION_TOKENS,
    model: str = _DEFAULT_MODEL,
) -> LLMResponse:
    """Build an :class:`LLMResponse` with sensible defaults.

    Convenience for tests that don't care about token accounting or
    model identity, only the response shape.

    Args:
        text: The string the assistant returned. Default empty.
        tool_calls: Optional sequence of OpenAI-style tool-call dicts.
        finish_reason: One of ``"stop"``, ``"length"``, ``"tool_calls"``,
            etc. Default ``"stop"``.
        prompt_tokens / completion_tokens: Usage counters; ``total_tokens``
            is computed as the sum.
        model: Model identifier echoed back on the response. Default ``"test-model"``.
    """
    return LLMResponse(
        model=model,
        content=text,
        tool_calls=list(tool_calls),
        usage=LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        raw={},
        finish_reason=finish_reason,
    )


class ScriptedLLM(ILLMClient):
    """Returns pre-constructed responses in sequence on each ``complete()`` call.

    Args:
        responses: Ordered sequence of ``LLMResponse`` to return.
        repeat_last: If ``True``, after exhausting ``responses``, return the
            last entry forever. If ``False`` (default), raise ``IndexError``
            on exhaustion so tests fail loudly when they need more responses
            than scripted.

    Raises:
        ValueError: At construction if ``responses`` is empty.
        IndexError: At call time if exhausted and ``repeat_last`` is False.
    """

    def __init__(
        self,
        responses: Sequence[LLMResponse],
        *,
        repeat_last: bool = False,
    ) -> None:
        if not responses:
            raise ValueError("ScriptedLLM requires at least one response")
        self._responses: tuple[LLMResponse, ...] = tuple(responses)
        self._repeat_last: bool = repeat_last
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({
            "model": model,
            "messages": [dict(m) for m in messages],
            "tools": [dict(t) for t in tools] if tools is not None else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "extra": dict(extra) if extra is not None else None,
        })
        idx = len(self.calls) - 1
        if idx < len(self._responses):
            return self._responses[idx]
        if self._repeat_last:
            return self._responses[-1]
        raise IndexError(
            f"ScriptedLLM exhausted: {len(self._responses)} responses "
            f"scripted but call #{len(self.calls)} requested. "
            f"Set repeat_last=True to keep returning the final response."
        )

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
        """Yield the scripted response as a single terminal chunk.

        Tests that need fragmented streams (multiple deltas before the
        terminal chunk) can construct chunks directly; this default
        keeps the common-case test path one-liner short.
        """
        response = await self.complete(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            tenant_id=tenant_id,
            agent_id=agent_id,
            extra=extra,
        )

        async def _gen() -> AsyncIterator[LLMStreamChunk]:
            yield LLMStreamChunk(
                model=response.model,
                delta_content=response.content,
                delta_tool_calls=list(response.tool_calls),
                finish_reason=response.finish_reason,
                usage=response.usage,
                raw=dict(response.raw),
            )

        return _gen()


__all__ = ["ScriptedLLM", "make_llm_response"]
