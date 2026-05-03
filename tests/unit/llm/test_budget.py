"""Unit tests for :class:`ai_core.llm.budget.InMemoryBudgetService`."""

from __future__ import annotations

import pytest

from ai_core.config.settings import AppSettings
from ai_core.llm.budget import InMemoryBudgetService


pytestmark = pytest.mark.unit


def _settings(*, enabled: bool = True, tokens: int = 100, usd: float = 1.0) -> AppSettings:
    return AppSettings(
        budget={  # type: ignore[arg-type]
            "enabled": enabled,
            "default_daily_token_limit": tokens,
            "default_daily_usd_limit": usd,
        },
    )


async def test_disabled_budget_always_allows() -> None:
    svc = InMemoryBudgetService(_settings(enabled=False))
    result = await svc.check(tenant_id=None, agent_id=None, estimated_tokens=10_000_000)
    assert result.allowed is True
    assert result.reason == "budget enforcement disabled"


async def test_within_limits_allows_and_records() -> None:
    svc = InMemoryBudgetService(_settings(tokens=1000))
    first = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=100)
    assert first.allowed is True
    await svc.record_usage(
        tenant_id="t", agent_id="a", prompt_tokens=80, completion_tokens=20, cost_usd=0.05
    )
    second = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=100)
    assert second.allowed is True
    assert second.remaining_tokens == 1000 - 100 - 100


async def test_token_limit_denies() -> None:
    svc = InMemoryBudgetService(_settings(tokens=50))
    await svc.record_usage(
        tenant_id="t", agent_id="a", prompt_tokens=40, completion_tokens=0, cost_usd=0.0
    )
    result = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=20)
    assert result.allowed is False
    assert "token" in (result.reason or "")


async def test_usd_limit_denies() -> None:
    svc = InMemoryBudgetService(_settings(tokens=10_000, usd=0.10))
    await svc.record_usage(
        tenant_id="t", agent_id="a", prompt_tokens=10, completion_tokens=10, cost_usd=0.20
    )
    result = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=10)
    assert result.allowed is False
    assert "USD" in (result.reason or "")


async def test_per_key_isolation() -> None:
    svc = InMemoryBudgetService(_settings(tokens=100))
    await svc.record_usage(
        tenant_id="t1", agent_id="a", prompt_tokens=90, completion_tokens=0, cost_usd=0.0
    )
    other = await svc.check(tenant_id="t2", agent_id="a", estimated_tokens=50)
    assert other.allowed is True
