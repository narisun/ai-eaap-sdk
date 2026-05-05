"""Unit tests for :class:`ai_core.agents.memory.MemoryManager`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from ai_core.agents.memory import MemoryManager, TokenCounter
from ai_core.agents.state import new_agent_state
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMUsage


def _content_messages(state_messages: list[Any]) -> list[Any]:
    """Strip RemoveMessage markers from a compacted-state messages list."""
    return [m for m in state_messages if not isinstance(m, RemoveMessage)]


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

    # The first element MUST be a RemoveMessage(REMOVE_ALL_MESSAGES) marker so
    # the add_messages reducer wipes existing history before applying the rest.
    assert isinstance(new_state["messages"][0], RemoveMessage)
    assert new_state["messages"][0].id == REMOVE_ALL_MESSAGES

    # The summary system message follows the marker.
    summary_msg = new_state["messages"][1]
    assert summary_msg["role"] == "system"
    assert "condensed conversation" in summary_msg["content"]

    # Tail (most recent user message) is preserved post-compaction.
    contents = [m["content"] for m in _content_messages(new_state["messages"])]
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


async def test_compact_replaces_history_via_add_messages_reducer() -> None:
    """The RemoveMessage marker must cause add_messages to wipe non-trailing history.

    Compaction *intentionally* preserves the most recent user+assistant pair as
    immediate context for the next turn; it evicts everything before that and
    inserts a summary in its place.
    """
    fake_llm = FakeLLM(summary="compressed")
    mgr = MemoryManager(_make_settings(), fake_llm, FakeTokenCounter([2 * _THRESHOLD, _TARGET]))
    existing = [
        {"role": "user", "content": "old-1", "id": "m1"},
        {"role": "assistant", "content": "old-2", "id": "m2"},
        {"role": "user", "content": "old-3", "id": "m3"},
        {"role": "assistant", "content": "old-4", "id": "m4"},
    ]
    state = new_agent_state(initial_messages=existing, essential={"user_id": "u-1"})

    update = await mgr.compact(state)

    # Run the actual reducer the way LangGraph would on commit.
    after = add_messages(existing, update["messages"])

    after_ids = {getattr(m, "id", None) for m in after}
    # Evicted: every message *before* the trailing pair.
    assert "m1" not in after_ids
    assert "m2" not in after_ids
    # The summary system message is present.
    assert any(getattr(m, "type", None) == "system" for m in after)
    # The trailing user+assistant pair survived for immediate context.
    assert "m3" in after_ids
    assert "m4" in after_ids
    # And nothing more — the only survivors are summary + trail pair.
    non_summary_ids = {i for i in after_ids if i in {"m1", "m2", "m3", "m4"}}
    assert non_summary_ids == {"m3", "m4"}


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


import asyncio as _asyncio  # noqa: E402


class _SlowFakeLLM(ILLMClient):
    """Sleeps before returning, simulating a slow upstream."""

    def __init__(self, sleep_seconds: float) -> None:
        self._sleep = sleep_seconds
        self.calls = 0

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
        self.calls += 1
        await _asyncio.sleep(self._sleep)
        return LLMResponse(
            model=model or "fake",
            content="summary",
            tool_calls=[],
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.0),
            raw={},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_compact_skips_on_timeout() -> None:
    """When the LLM takes longer than compaction_timeout_seconds, compact() returns
    state unchanged and logs a WARNING — no crash."""
    settings = AppSettings()
    settings.agent.compaction_timeout_seconds = 0.05  # 50ms cap
    slow_llm = _SlowFakeLLM(sleep_seconds=0.2)  # 200ms hang
    counter = FakeTokenCounter([10_000, 0, 0])
    mgr = MemoryManager(settings=settings, llm=slow_llm, token_counter=counter)

    state = new_agent_state(
        initial_messages=[{"role": "user", "content": "hi"}],
        essential={"tenant_id": "t1"},
        metadata={"agent_id": "a1"},
    )

    result = await mgr.compact(state, agent_id="a1", tenant_id="t1")
    # State returned unchanged on timeout (skip-and-warn).
    assert result is state
    assert slow_llm.calls == 1


@pytest.mark.asyncio
async def test_compact_succeeds_within_timeout() -> None:
    """Compaction completes normally when LLM responds within the budget."""
    settings = AppSettings()
    settings.agent.compaction_timeout_seconds = 1.0
    fast_llm = FakeLLM(summary="hello world")
    counter = FakeTokenCounter([10_000, 0, 0])
    mgr = MemoryManager(settings=settings, llm=fast_llm, token_counter=counter)

    state = new_agent_state(
        initial_messages=[{"role": "user", "content": "hi"}],
        essential={"tenant_id": "t1"},
    )

    result = await mgr.compact(state, agent_id="a1", tenant_id="t1")
    # Compaction succeeded — state has summary attached.
    assert result is not state
    assert result.get("summary") == "hello world"
