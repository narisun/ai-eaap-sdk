"""Tests for ai_core.testing.make_llm_response."""
from __future__ import annotations

import pytest

from ai_core.di.interfaces import LLMResponse, LLMUsage
from ai_core.testing import make_llm_response

pytestmark = pytest.mark.unit


def test_make_llm_response_defaults() -> None:
    r = make_llm_response()
    assert isinstance(r, LLMResponse)
    assert r.content == ""
    assert r.finish_reason == "stop"
    assert r.model == "test-model"
    assert isinstance(r.usage, LLMUsage)
    assert r.usage.prompt_tokens == 10
    assert r.usage.completion_tokens == 20
    assert r.usage.total_tokens == 30
    assert r.tool_calls == []


def test_make_llm_response_with_text_only() -> None:
    r = make_llm_response("hi")
    assert r.content == "hi"
    assert r.finish_reason == "stop"
    assert r.usage.total_tokens == 30


def test_make_llm_response_with_all_fields() -> None:
    r = make_llm_response(
        "hello",
        tool_calls=[{"id": "c1", "function": {"name": "f"}}],
        finish_reason="tool_calls",
        prompt_tokens=5,
        completion_tokens=15,
        model="gpt-test",
    )
    assert r.content == "hello"
    assert r.tool_calls == [{"id": "c1", "function": {"name": "f"}}]
    assert r.finish_reason == "tool_calls"
    assert r.usage.prompt_tokens == 5
    assert r.usage.completion_tokens == 15
    assert r.usage.total_tokens == 20  # prompt + completion
    assert r.model == "gpt-test"
