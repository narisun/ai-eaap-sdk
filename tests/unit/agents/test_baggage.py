"""Tests for OTel Baggage propagation in :class:`BaseAgent.ainvoke`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock

import pytest
from opentelemetry import baggage

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMUsage


pytestmark = pytest.mark.unit


class _Echo(BaseAgent):
    agent_id = "echo-baggage"

    def system_prompt(self) -> str:
        return "test"


def _make_container() -> Container:
    settings = AppSettings(service_name="echo-baggage-test")
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


async def test_baggage_set_during_ainvoke_includes_agent_and_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside compiled.ainvoke we should see eaap.* baggage entries."""
    container = _make_container()
    agent = container.get(_Echo)
    compiled = agent.compile()

    seen: dict[str, str] = {}

    async def _capture(state: Mapping[str, Any], config: object | None = None) -> object:
        seen.update({k: str(v) for k, v in baggage.get_all().items() if k.startswith("eaap.")})
        return state

    monkeypatch.setattr(compiled, "ainvoke", _capture)

    await agent.ainvoke(
        messages=[{"role": "user", "content": "hi"}],
        essential={"user_id": "u-1", "session_id": "s-2", "task_id": "T-9"},
        tenant_id="acme",
        thread_id="conv-7",
    )

    assert seen == {
        "eaap.agent_id": "echo-baggage",
        "eaap.tenant_id": "acme",
        "eaap.thread_id": "conv-7",
        "eaap.user_id": "u-1",
        "eaap.session_id": "s-2",
        "eaap.task_id": "T-9",
    }


async def test_baggage_cleared_after_ainvoke_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Baggage context must be detached when ainvoke returns."""
    container = _make_container()
    agent = container.get(_Echo)
    compiled = agent.compile()

    async def _noop(state: object, config: object | None = None) -> object:
        return state

    monkeypatch.setattr(compiled, "ainvoke", _noop)

    await agent.ainvoke(
        messages=[{"role": "user", "content": "hi"}],
        tenant_id="acme",
    )

    # No leftover eaap.* entries in the surrounding context.
    leftover = {k: v for k, v in baggage.get_all().items() if k.startswith("eaap.")}
    assert leftover == {}
