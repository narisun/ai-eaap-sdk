"""Tests for the recursion-limit guard wired into :class:`BaseAgent.ainvoke`."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langgraph.errors import GraphRecursionError

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMUsage
from ai_core.exceptions import AgentRecursionLimitError

pytestmark = pytest.mark.unit


class _Echo(BaseAgent):
    agent_id = "echo"

    def system_prompt(self) -> str:
        return "test agent"


def _make_container(*, max_depth: int = 25) -> Container:
    settings = AppSettings(agent={"max_recursion_depth": max_depth})  # type: ignore[arg-type]

    fake_llm = AsyncMock(spec=ILLMClient)
    fake_llm.complete = AsyncMock(
        return_value=LLMResponse(
            model="m",
            content="ok",
            tool_calls=[],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            raw={},
        )
    )

    from injector import Module, provider, singleton

    class _Override(Module):
        @singleton
        @provider
        def llm(self) -> ILLMClient:
            return fake_llm

    return Container.build([AgentModule(settings=settings), _Override()])


async def test_recursion_limit_passed_to_compiled_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured max_recursion_depth must reach compiled.ainvoke as recursion_limit."""
    container = _make_container(max_depth=7)
    agent = container.get(_Echo)
    compiled = agent.compile()

    captured: dict[str, object] = {}

    async def _spy(state: object, config: object | None = None) -> object:
        captured["config"] = config
        return state

    monkeypatch.setattr(compiled, "ainvoke", _spy)
    await agent.ainvoke(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(captured["config"], dict)
    assert captured["config"]["recursion_limit"] == 7  # type: ignore[index]


async def test_graph_recursion_error_wrapped_in_structured_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _make_container(max_depth=3)
    agent = container.get(_Echo)
    compiled = agent.compile()

    async def _looping(state: object, config: object | None = None) -> object:
        raise GraphRecursionError("loop")

    monkeypatch.setattr(compiled, "ainvoke", _looping)

    with pytest.raises(AgentRecursionLimitError) as ei:
        await agent.ainvoke(messages=[{"role": "user", "content": "hi"}])

    assert ei.value.details["agent_id"] == "echo"
    assert ei.value.details["limit"] == 3
    assert isinstance(ei.value.__cause__, GraphRecursionError)
