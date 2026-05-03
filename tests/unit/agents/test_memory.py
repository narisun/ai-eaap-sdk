"""Unit tests for :class:`ai_core.agents.memory.MemoryManager`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from ai_core.agents.memory import MemoryManager, TokenCounter
from ai_core.agents.state import new_agent_state
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMUsage


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeTokenCounter(TokenCounter):
    """Returns a fixed count, optionally varying per call for deterministic tests."""

    def __init__(self, counts: Sequence[int]) -> None:
        self._counts = list(counts)
        self.calls = 0

    def count(self, messages: Sequence[Mapping[str, Any]], *, model: str) -> int:
        idx = min(self.calls, len(self._counts) - 1)
        self.calls += 1
        return self._counts[idx]


class FakeLLM(ILLMClient):
    """Captures the prompt sent for compaction and returns a canned summary."""

    def __init__(self, summary: str = "summary text") -> None:
        self._summary = summary
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
        self.calls.append(
            {
                "model": model,
                "messages": [dict(m) for m in messages],
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "temperature": temperature,
            }
        )
        return LLMResponse(
            model=model or "fake",
            content=self._summary,
            tool_calls=[],
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.0),
            raw={},
        )


_THRESHOLD = 1_000
_TARGET = 200


def _make_settings(threshold: int = _THRESHOLD, target: int = _TARGET) -> AppSettings:
    return AppSettings(
        agent={  # type: ignore[arg-type]
            "memory_compaction_token_threshold": threshold,
            "memory_compaction_target_tokens": target,
            "essential_entity_keys": ["user_id", "tenant_id", "session_id", "task_id"],
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_should_compact_below_threshold() -> None:
    mgr = MemoryManager(_make_settings(), FakeLLM(), FakeTokenCounter([_THRESHOLD - 1]))
    state = new_agent_state(initial_messages=[{"role": "user", "content": "hi"}])
    assert mgr.should_compact(state) is False


def test_should_compact_at_threshold_is_false() -> None:
    mgr = MemoryManager(_make_settings(), FakeLLM(), FakeTokenCounter([_THRESHOLD]))
    state = new_agent_state(initial_messages=[{"role": "user", "content": "hi"}])
    assert mgr.should_compact(state) is False


def test_should_compact_above_threshold() -> None:
    mgr = MemoryManager(_make_settings(), FakeLLM(), FakeTokenCounter([_THRESHOLD + 1]))
    state = new_agent_state(initial_messages=[{"role": "user", "content": "hi"}])
    assert mgr.should_compact(state) is True


def test_should_compact_empty_messages_is_false() -> None:
    mgr = MemoryManager(_make_settings(), FakeLLM(), FakeTokenCounter([1_000_000]))
    state = new_agent_state()
    assert mgr.should_compact(state) is False


async def test_compact_returns_state_with_summary_and_essentials_preserved() -> None:
    fake_llm = FakeLLM(summary="condensed conversation")
    mgr = MemoryManager(_make_settings(), fake_llm, FakeTokenCounter([2 * _THRESHOLD, _TARGET]))

    state = new_agent_state(
        initial_messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "what is my task?"},
            {"role": "assistant", "content": "you are working on TASK-42"},
            {"role": "user", "content": "and the deadline?"},
        ],
        essential={
            "user_id": "u-1",
            "tenant_id": "t-9",
            "task_id": "TASK-42",
            "custom_host_key": "keep-me",
        },
    )

    new_state = await mgr.compact(state, tenant_id="t-9", agent_id="agent-test")

    # Summary text recorded.
    assert new_state["summary"] == "condensed conversation"

    # First message in the new history is the summary system message.
    assert new_state["messages"][0]["role"] == "system"
    assert "condensed conversation" in new_state["messages"][0]["content"]

    # Tail (most recent user message) is preserved post-compaction.
    contents = [m["content"] for m in new_state["messages"]]
    assert any("deadline" in c for c in contents)

    # Essential entities — both configured and host-defined — survive.
    essentials = new_state["essential_entities"]
    assert essentials["user_id"] == "u-1"
    assert essentials["tenant_id"] == "t-9"
    assert essentials["task_id"] == "TASK-42"
    assert essentials["custom_host_key"] == "keep-me"

    # Compaction count incremented.
    assert new_state["compaction_count"] == 1


async def test_compact_passes_essentials_to_summarisation_prompt() -> None:
    fake_llm = FakeLLM()
    mgr = MemoryManager(_make_settings(), fake_llm, FakeTokenCounter([2 * _THRESHOLD, _TARGET]))
    state = new_agent_state(
        initial_messages=[{"role": "user", "content": "hello"}],
        essential={"user_id": "u-1", "task_id": "TASK-42"},
    )

    await mgr.compact(state, tenant_id="t", agent_id="a")

    assert len(fake_llm.calls) == 1
    user_msg = fake_llm.calls[0]["messages"][1]["content"]
    assert "user_id" in user_msg
    assert "TASK-42" in user_msg
    assert fake_llm.calls[0]["temperature"] == 0.0
    assert fake_llm.calls[0]["tenant_id"] == "t"
    assert fake_llm.calls[0]["agent_id"] == "a"


async def test_compact_no_messages_returns_state_unchanged() -> None:
    mgr = MemoryManager(_make_settings(), FakeLLM(), FakeTokenCounter([0]))
    state = new_agent_state(essential={"user_id": "u-1"})
    out = await mgr.compact(state)
    assert out is state  # no-op


async def test_compact_increments_count_across_invocations() -> None:
    mgr = MemoryManager(
        _make_settings(),
        FakeLLM(),
        FakeTokenCounter([2 * _THRESHOLD, _TARGET, 2 * _THRESHOLD, _TARGET]),
    )
    state = new_agent_state(
        initial_messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first reply"},
        ],
        essential={"user_id": "u"},
    )
    s1 = await mgr.compact(state)
    s2 = await mgr.compact(s1)
    assert s1["compaction_count"] == 1
    assert s2["compaction_count"] == 2
