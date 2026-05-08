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
from dataclasses import dataclass
from datetime import UTC, date, datetime

from injector import inject

from ai_core.config.settings import BudgetSettings
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

    The service derives its limits from :class:`BudgetSettings`.
    Limits are checked against the *projected* spend (current usage +
    estimated tokens for the pending request).

    Args:
        settings: The budget configuration slice. Pass
            ``app_settings.budget`` when constructing manually.
    """

    @inject
    def __init__(self, settings: BudgetSettings) -> None:
        self._cfg = settings
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
        cfg = self._cfg
        if not cfg.enabled:
            return BudgetCheck(
                allowed=True,
                remaining_tokens=None,
                remaining_usd=None,
                reason="budget enforcement disabled",
            )

        token_limit, usd_limit = self._resolve_limits(tenant_id, agent_id)

        key = _Key.of(tenant_id, agent_id)
        async with self._lock:
            counter = self._get_or_reset_locked(key)
            projected = counter.total_tokens() + max(0, estimated_tokens)
            remaining_tokens = token_limit - projected
            remaining_usd = usd_limit - counter.cost_usd

            if token_limit and projected > token_limit:
                return BudgetCheck(
                    allowed=False,
                    remaining_tokens=max(0, token_limit - counter.total_tokens()),
                    remaining_usd=remaining_usd,
                    reason="daily token limit exceeded",
                )
            if usd_limit and counter.cost_usd >= usd_limit:
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
    def _resolve_limits(
        self,
        tenant_id: str | None,
        agent_id: str | None,
    ) -> tuple[int, float]:
        """Resolve effective (daily_token_limit, daily_usd_limit) for a key.

        Walks override candidates from most-specific to least-specific:
        (tenant, agent), (tenant, None), (None, agent). For each candidate,
        finds matching override entries and fills in any field still unset.
        Falls back to settings defaults for any field not covered by overrides.

        Resolution iterates ``self._cfg.overrides`` in list order;
        first match wins per-field. Operators wanting deterministic precedence
        put more specific entries first.
        """
        cfg = self._cfg
        candidates: list[tuple[str | None, str | None]] = []
        if tenant_id is not None and agent_id is not None:
            candidates.append((tenant_id, agent_id))
        if tenant_id is not None:
            candidates.append((tenant_id, None))
        if agent_id is not None:
            candidates.append((None, agent_id))

        token: int | None = None
        usd: float | None = None
        for cand_tenant, cand_agent in candidates:
            for ov in cfg.overrides:
                if ov.tenant_id == cand_tenant and ov.agent_id == cand_agent:
                    if token is None and ov.daily_token_limit is not None:
                        token = ov.daily_token_limit
                    if usd is None and ov.daily_usd_limit is not None:
                        usd = ov.daily_usd_limit
            if token is not None and usd is not None:
                break

        return (
            token if token is not None else cfg.default_daily_token_limit,
            usd if usd is not None else cfg.default_daily_usd_limit,
        )

    def _get_or_reset_locked(self, key: _Key) -> _Counter:
        today = datetime.now(UTC).date()
        composite = (key.tenant_id, key.agent_id)
        counter = self._counters.get(composite)
        if counter is None or counter.day != today:
            counter = _Counter(day=today)
            self._counters[composite] = counter
        return counter


__all__ = ["InMemoryBudgetService"]
