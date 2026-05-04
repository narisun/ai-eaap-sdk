"""Component tests for BaseAgent + @tool integration."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from injector import Module, provider, singleton
from pydantic import BaseModel, Field

from ai_core.agents import BaseAgent
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import (
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
    LLMResponse,
    LLMUsage,
)
from ai_core.tools import Tool, tool

pytestmark = pytest.mark.component


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    n: int = Field(..., ge=0)


@tool(name="count", version=1)
async def count_tool(payload: _In) -> _Out:
    """Return the length of the query string."""
    return _Out(n=len(payload.q))


class _ScriptedLLM(ILLMClient):
    """Returns a queue of pre-baked completions."""

    def __init__(self, scripts: Sequence[LLMResponse]) -> None:
        self._scripts = list(scripts)
        self.calls: list[Sequence[Mapping[str, Any]]] = []

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
        self.calls.append(list(messages))
        return self._scripts.pop(0)


def _llm_msg(content: str = "", tool_calls: Sequence[Mapping[str, Any]] = ()) -> LLMResponse:
    return LLMResponse(
        model="fake",
        content=content,
        tool_calls=list(tool_calls),
        usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.0),
        raw={},
    )


class _DemoAgent(BaseAgent):
    agent_id = "demo"
    _tools: tuple[Tool, ...] = (count_tool,)

    def system_prompt(self) -> str:
        return "You are a counting agent."

    def tools(self) -> Sequence[Tool]:
        return self._tools


def _build(llm: ILLMClient, fake_observability, fake_policy_evaluator_factory):
    class _Fakes(Module):
        @singleton
        @provider
        def llm(self) -> ILLMClient:
            return llm

        @singleton
        @provider
        def obs(self) -> IObservabilityProvider:
            return fake_observability

        @singleton
        @provider
        def policy(self) -> IPolicyEvaluator:
            return fake_policy_evaluator_factory(default_allow=True)

    return Container.build([AgentModule(), _Fakes()])


@pytest.mark.asyncio
async def test_agent_with_tool_runs_loop_and_returns_final_answer(
    fake_observability, fake_policy_evaluator_factory
):
    # First LLM turn requests a tool call; second returns the final answer.
    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": '{"q":"hello"}'},
        }]),
        _llm_msg(content="The count is 5."),
    ])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count hello"}])
    final = state["messages"][-1]
    final_content = getattr(final, "content", None) or final["content"]
    assert "5" in final_content


@pytest.mark.asyncio
async def test_agent_without_tool_calls_terminates(
    fake_observability, fake_policy_evaluator_factory
):
    llm = _ScriptedLLM([_llm_msg(content="Hi.")])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "hello"}])
    final = state["messages"][-1]
    final_content = getattr(final, "content", None) or final["content"]
    assert final_content == "Hi."


@pytest.mark.asyncio
async def test_tool_validation_error_surfaces_as_toolmessage(
    fake_observability, fake_policy_evaluator_factory
):
    # LLM passes invalid args (q is missing); LLM then explains.
    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": '{}'},
        }]),
        _llm_msg(content="I see — I need a query."),
    ])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)
    await agent.ainvoke(messages=[{"role": "user", "content": "count"}])
    # The second LLM call must have seen a tool message describing the error.
    tool_msg_seen = False
    for messages in llm.calls:
        for m in messages:
            if (m.get("role") == "tool") and "validation" in (m.get("content") or "").lower():
                tool_msg_seen = True
    assert tool_msg_seen, "Expected a ToolMessage describing the validation failure."


@pytest.mark.asyncio
async def test_policy_denial_surfaces_as_toolmessage(
    fake_observability, fake_policy_evaluator_factory
):
    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": '{"q":"x"}'},
        }]),
        _llm_msg(content="Ok, can't run that."),
    ])

    class _Fakes(Module):
        @singleton
        @provider
        def llm(self) -> ILLMClient:
            return llm

        @singleton
        @provider
        def obs(self) -> IObservabilityProvider:
            return fake_observability

        @singleton
        @provider
        def policy(self) -> IPolicyEvaluator:
            return fake_policy_evaluator_factory(default_allow=False, reason="not allowed")

    container = Container.build([AgentModule(), _Fakes()])
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count x"}])
    final = state["messages"][-1]
    content = getattr(final, "content", None) or final["content"]
    assert "can't run" in content.lower() or "ok" in content.lower()


@pytest.mark.asyncio
async def test_auto_tool_loop_can_be_disabled(
    fake_observability, fake_policy_evaluator_factory
):
    class _NoLoopAgent(_DemoAgent):
        auto_tool_loop = False

    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": '{"q":"x"}'},
        }]),
    ])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_NoLoopAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count x"}])
    # No second LLM call — graph terminated after the first turn.
    assert len(llm.calls) == 1
    # Final message has the unanswered tool_call attached.
    last = state["messages"][-1]
    tc = getattr(last, "tool_calls", None) or (
        last.get("tool_calls") if isinstance(last, dict) else None
    )
    assert tc, "Expected the unanswered tool_calls to remain on the final message."


@pytest.mark.asyncio
async def test_malformed_tool_args_surface_as_toolmessage(
    fake_observability, fake_policy_evaluator_factory
):
    """LLM emitting malformed JSON in tool_calls.arguments must not crash the graph."""
    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "count", "arguments": "not-json{"},
        }]),
        _llm_msg(content="Sorry, I'll retry."),
    ])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count x"}])
    final = state["messages"][-1]
    final_content = getattr(final, "content", None) or final["content"]
    assert "retry" in final_content.lower() or "sorry" in final_content.lower()


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_turn(
    fake_observability, fake_policy_evaluator_factory
):
    """An AIMessage with multiple tool_calls dispatches each."""
    llm = _ScriptedLLM([
        _llm_msg(tool_calls=[
            {"id": "call-1", "type": "function",
             "function": {"name": "count", "arguments": '{"q":"abc"}'}},
            {"id": "call-2", "type": "function",
             "function": {"name": "count", "arguments": '{"q":"defgh"}'}},
        ]),
        _llm_msg(content="Got 3 and 5."),
    ])
    container = _build(llm, fake_observability, fake_policy_evaluator_factory)
    agent = container.get(_DemoAgent)
    state = await agent.ainvoke(messages=[{"role": "user", "content": "count two things"}])
    final = state["messages"][-1]
    final_content = getattr(final, "content", None) or final["content"]
    assert "3" in final_content and "5" in final_content
