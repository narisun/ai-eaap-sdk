"""Tests for the AICoreApp facade."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from injector import Module, provider, singleton
from pydantic import BaseModel

from ai_core.app import AICoreApp, HealthSnapshot
from ai_core.config.settings import AgentSettings, AppSettings, LLMSettings
from ai_core.di.interfaces import (
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
    LLMResponse,
    LLMUsage,
)
from ai_core.exceptions import ConfigurationError
from ai_core.tools import tool
from ai_core.tools.invoker import ToolInvoker

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from tests.conftest import FakeObservabilityProvider, FakePolicyEvaluator

pytestmark = pytest.mark.unit


def _bad_settings() -> AppSettings:
    return AppSettings(
        llm=LLMSettings(default_model=""),
        agent=AgentSettings(
            memory_compaction_token_threshold=512,
            memory_compaction_target_tokens=10000,
        ),
    )


class _StubLLM(ILLMClient):
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
        return LLMResponse(
            model="stub",
            content="ok",
            tool_calls=[],
            usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0),
            raw={},
        )


def _override_module(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> Module:
    class _M(Module):
        @singleton
        @provider
        def llm(self) -> ILLMClient:
            return _StubLLM()

        @singleton
        @provider
        def obs(self) -> IObservabilityProvider:
            return fake_observability

        @singleton
        @provider
        def policy(self) -> IPolicyEvaluator:
            return fake_policy_evaluator_factory(default_allow=True)

    return _M()


@pytest.mark.asyncio
async def test_aenter_runs_validation_and_builds_container(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        assert app.settings is not None
        assert app.observability is fake_observability
        assert app.policy_evaluator is not None
        assert isinstance(app.container.get(ToolInvoker), ToolInvoker)


@pytest.mark.asyncio
async def test_aenter_fails_fast_on_invalid_settings() -> None:
    app = AICoreApp(settings=_bad_settings())
    with pytest.raises(ConfigurationError) as exc:
        await app.__aenter__()
    assert "issue(s)" in exc.value.message


@pytest.mark.asyncio
async def test_aexit_is_idempotent(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    await app.__aenter__()
    await app.__aexit__(None, None, None)
    await app.__aexit__(None, None, None)  # double-close must not raise


@pytest.mark.asyncio
async def test_health_snapshot_is_ok_after_entry(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        snap = app.health
        assert isinstance(snap, HealthSnapshot)
        assert snap.status == "ok"


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    n: int


@tool(name="x", version=1)
async def _x(p: _In) -> _Out:
    return _Out(n=0)


@pytest.mark.asyncio
async def test_register_tools_is_idempotent(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        app.register_tools(_x)
        app.register_tools(_x)  # idempotent — must not raise


@pytest.mark.asyncio
async def test_methods_raise_before_entry() -> None:
    app = AICoreApp()
    with pytest.raises(RuntimeError):
        _ = app.settings
    with pytest.raises(RuntimeError):
        _ = app.container
    # health is the exception — must return status="down" without raising
    snap = app.health
    assert snap.status == "down"
    assert snap.settings_version == ""
