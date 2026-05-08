"""Unit tests for :class:`ai_core.llm.budget.InMemoryBudgetService`."""

from __future__ import annotations

import pytest

from ai_core.config.settings import AppSettings, BudgetOverride
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
    svc = InMemoryBudgetService(_settings(enabled=False).budget)
    result = await svc.check(tenant_id=None, agent_id=None, estimated_tokens=10_000_000)
    assert result.allowed is True
    assert result.reason == "budget enforcement disabled"


async def test_within_limits_allows_and_records() -> None:
    svc = InMemoryBudgetService(_settings(tokens=1000).budget)
    first = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=100)
    assert first.allowed is True
    await svc.record_usage(
        tenant_id="t", agent_id="a", prompt_tokens=80, completion_tokens=20, cost_usd=0.05
    )
    second = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=100)
    assert second.allowed is True
    assert second.remaining_tokens == 1000 - 100 - 100


async def test_token_limit_denies() -> None:
    svc = InMemoryBudgetService(_settings(tokens=50).budget)
    await svc.record_usage(
        tenant_id="t", agent_id="a", prompt_tokens=40, completion_tokens=0, cost_usd=0.0
    )
    result = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=20)
    assert result.allowed is False
    assert "token" in (result.reason or "")


async def test_usd_limit_denies() -> None:
    svc = InMemoryBudgetService(_settings(tokens=10_000, usd=0.10).budget)
    await svc.record_usage(
        tenant_id="t", agent_id="a", prompt_tokens=10, completion_tokens=10, cost_usd=0.20
    )
    result = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=10)
    assert result.allowed is False
    assert "USD" in (result.reason or "")


async def test_per_key_isolation() -> None:
    svc = InMemoryBudgetService(_settings(tokens=100).budget)
    await svc.record_usage(
        tenant_id="t1", agent_id="a", prompt_tokens=90, completion_tokens=0, cost_usd=0.0
    )
    other = await svc.check(tenant_id="t2", agent_id="a", estimated_tokens=50)
    assert other.allowed is True


# ---------------------------------------------------------------------------
# Per-agent budget overrides (Phase 13)
# ---------------------------------------------------------------------------


def _settings_with_overrides(
    *,
    tokens: int = 1000,
    usd: float = 1.0,
    overrides: list[BudgetOverride] | None = None,
) -> AppSettings:
    """Build AppSettings with global defaults + an optional override list."""
    return AppSettings(
        budget={  # type: ignore[arg-type]
            "enabled": True,
            "default_daily_token_limit": tokens,
            "default_daily_usd_limit": usd,
            "overrides": overrides or [],
        },
    )


async def test_override_exact_tenant_agent_match() -> None:
    """An override with both tenant_id and agent_id set applies to that exact pair."""
    overrides = [
        BudgetOverride(
            tenant_id="acme",
            agent_id="customer-support",
            daily_token_limit=5000,
            daily_usd_limit=10.0,
        ),
    ]
    svc = InMemoryBudgetService(_settings_with_overrides(tokens=100, overrides=overrides).budget)

    # Matching key — uses override (5000 >= 1000 estimated, allowed)
    matched = await svc.check(
        tenant_id="acme", agent_id="customer-support", estimated_tokens=1000,
    )
    assert matched.allowed is True
    assert matched.remaining_tokens == 5000 - 1000

    # Non-matching key — falls back to global default (100 < 1000, denied)
    other = await svc.check(tenant_id="other", agent_id="other", estimated_tokens=1000)
    assert other.allowed is False


async def test_override_tenant_only_match() -> None:
    """An override with tenant_id set but agent_id=None matches all agents under that tenant."""
    overrides = [
        BudgetOverride(tenant_id="acme", agent_id=None, daily_token_limit=2000),
    ]
    svc = InMemoryBudgetService(_settings_with_overrides(tokens=100, overrides=overrides).budget)

    # Any agent under tenant "acme" gets 2000-token limit
    a = await svc.check(tenant_id="acme", agent_id="agent-x", estimated_tokens=500)
    b = await svc.check(tenant_id="acme", agent_id="agent-y", estimated_tokens=500)
    assert a.allowed is True
    assert b.allowed is True
    assert a.remaining_tokens == 2000 - 500
    assert b.remaining_tokens == 2000 - 500


async def test_override_agent_only_match() -> None:
    """An override with agent_id set but tenant_id=None matches that agent across all tenants."""
    overrides = [
        BudgetOverride(tenant_id=None, agent_id="reporting", daily_token_limit=3000),
    ]
    svc = InMemoryBudgetService(_settings_with_overrides(tokens=100, overrides=overrides).budget)

    a = await svc.check(tenant_id="t1", agent_id="reporting", estimated_tokens=500)
    b = await svc.check(tenant_id="t2", agent_id="reporting", estimated_tokens=500)
    assert a.allowed is True
    assert b.allowed is True
    assert a.remaining_tokens == 3000 - 500


async def test_override_most_specific_wins_when_both_match() -> None:
    """When multiple candidates match, most-specific (tenant, agent) wins over (tenant, None)."""
    overrides = [
        # Tenant-only override (less specific) — first in list
        BudgetOverride(tenant_id="acme", agent_id=None, daily_token_limit=1000),
        # Tenant+agent override (more specific) — second in list
        BudgetOverride(
            tenant_id="acme", agent_id="customer-support", daily_token_limit=5000,
        ),
    ]
    svc = InMemoryBudgetService(_settings_with_overrides(tokens=100, overrides=overrides).budget)

    # The (acme, customer-support) candidate is checked first → wins with 5000.
    result = await svc.check(
        tenant_id="acme", agent_id="customer-support", estimated_tokens=2000,
    )
    assert result.allowed is True
    assert result.remaining_tokens == 5000 - 2000


async def test_partial_override_composes_with_defaults() -> None:
    """An override that sets only daily_token_limit lets daily_usd_limit fall through to default."""
    overrides = [
        BudgetOverride(tenant_id="acme", agent_id=None, daily_token_limit=10000),
        # daily_usd_limit is None → not overridden
    ]
    svc = InMemoryBudgetService(
        _settings_with_overrides(tokens=100, usd=5.0, overrides=overrides).budget,
    )

    # Token side uses override (10_000), USD side uses default (5.0).
    # Record some usage to verify USD path is exercised.
    await svc.record_usage(
        tenant_id="acme", agent_id="x", prompt_tokens=10, completion_tokens=10, cost_usd=4.0,
    )
    # Should still be allowed: 4.0 < 5.0 (default), 20 + estimated < 10_000.
    result = await svc.check(tenant_id="acme", agent_id="x", estimated_tokens=100)
    assert result.allowed is True
    assert result.remaining_usd == pytest.approx(5.0 - 4.0)
    assert result.remaining_tokens == 10000 - 20 - 100


async def test_no_override_uses_global_defaults() -> None:
    """Empty overrides list = current behavior; falls through to default_daily_*."""
    svc = InMemoryBudgetService(_settings_with_overrides(tokens=500, usd=1.0, overrides=[]).budget)

    result = await svc.check(tenant_id="t", agent_id="a", estimated_tokens=100)
    assert result.allowed is True
    assert result.remaining_tokens == 500 - 100


async def test_first_matching_override_wins_for_duplicates() -> None:
    """When multiple overrides match the same key, the first one in the list wins per-field."""
    overrides = [
        BudgetOverride(
            tenant_id="acme", agent_id="x", daily_token_limit=1000,
        ),
        # This second override would match the same key but is shadowed by the first
        BudgetOverride(
            tenant_id="acme", agent_id="x", daily_token_limit=99999,
        ),
    ]
    svc = InMemoryBudgetService(_settings_with_overrides(tokens=100, overrides=overrides).budget)

    result = await svc.check(tenant_id="acme", agent_id="x", estimated_tokens=500)
    # First override wins: limit is 1000, projected is 500 → allowed.
    assert result.allowed is True
    assert result.remaining_tokens == 1000 - 500
