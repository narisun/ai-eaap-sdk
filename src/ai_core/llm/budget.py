"""Default in-memory :class:`IBudgetService` implementation.

The service keeps per-(tenant, agent) counters that reset every
calendar UTC day. It is intentionally simple — production deployments
should swap this with a Redis- or DB-backed implementation by binding
their own :class:`IBudgetService` in a custom DI module.

Attributes:
    Counters are stored in a ``dict`` guarded by an :class:`asyncio.Lock`.
    The lock is held for the duration of a check or record call, which
    is fine for the in-memory implementation but is exactly why hosts
    should swap in something distributed for multi-replica deployments.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from injector import inject

from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import BudgetCheck, IBudgetService


@dataclass(slots=True)
class _Counter:
    """Per-key counter — tracks tokens + spend for the current UTC day."""

    day: date
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class _Key:
    """Composite key combining tenant and agent identifiers."""

    tenant_id: str
    agent_id: str

    @classmethod
    def of(cls, tenant_id: str | None, agent_id: str | None) -> _Key:
        return cls(tenant_id=tenant_id or "_default_", agent_id=agent_id or "_default_")


class InMemoryBudgetService(IBudgetService):
    """Process-local quota enforcement.

    The service derives its limits from :class:`AppSettings.budget`.
    Limits are checked against the *projected* spend (current usage +
    estimated tokens for the pending request).

    Args:
        settings: Aggregated application settings.
    """

    @inject
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._counters: dict[tuple[str, str], _Counter] = {}
        self._lock = asyncio.Lock()

    async def check(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        estimated_tokens: int,
    ) -> BudgetCheck:
        """See :meth:`IBudgetService.check`."""
        cfg = self._settings.budget
        if not cfg.enabled:
            return BudgetCheck(
                allowed=True,
                remaining_tokens=None,
                remaining_usd=None,
                reason="budget enforcement disabled",
            )

        key = _Key.of(tenant_id, agent_id)
        async with self._lock:
            counter = self._get_or_reset_locked(key)
            projected = counter.total_tokens() + max(0, estimated_tokens)
            remaining_tokens = cfg.default_daily_token_limit - projected
            remaining_usd = cfg.default_daily_usd_limit - counter.cost_usd

            if cfg.default_daily_token_limit and projected > cfg.default_daily_token_limit:
                return BudgetCheck(
                    allowed=False,
                    remaining_tokens=max(0, cfg.default_daily_token_limit - counter.total_tokens()),
                    remaining_usd=remaining_usd,
                    reason="daily token limit exceeded",
                )
            if cfg.default_daily_usd_limit and counter.cost_usd >= cfg.default_daily_usd_limit:
                return BudgetCheck(
                    allowed=False,
                    remaining_tokens=remaining_tokens,
                    remaining_usd=0.0,
                    reason="daily USD limit exceeded",
                )
            return BudgetCheck(
                allowed=True,
                remaining_tokens=remaining_tokens,
                remaining_usd=remaining_usd,
            )

    async def record_usage(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """See :meth:`IBudgetService.record_usage`."""
        key = _Key.of(tenant_id, agent_id)
        async with self._lock:
            counter = self._get_or_reset_locked(key)
            counter.prompt_tokens += max(0, prompt_tokens)
            counter.completion_tokens += max(0, completion_tokens)
            counter.cost_usd += max(0.0, cost_usd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_or_reset_locked(self, key: _Key) -> _Counter:
        today = datetime.now(UTC).date()
        composite = (key.tenant_id, key.agent_id)
        counter = self._counters.get(composite)
        if counter is None or counter.day != today:
            counter = _Counter(day=today)
            self._counters[composite] = counter
        return counter


__all__ = ["InMemoryBudgetService"]
