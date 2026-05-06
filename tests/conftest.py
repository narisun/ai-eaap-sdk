"""Shared pytest fixtures for the SDK test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ai_core.config.settings import AppSettings, get_settings
from ai_core.testing import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from ai_core.di.interfaces import PolicyDecision


# ---------------------------------------------------------------------------
# Settings cache hygiene (existing fixtures — preserved)
# ---------------------------------------------------------------------------
@pytest.fixture
def clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fresh_settings() -> AppSettings:
    return AppSettings()


# ---------------------------------------------------------------------------
# FakePolicyEvaluator
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_policy_evaluator_factory() -> Callable[..., FakePolicyEvaluator]:
    def _make(
        *,
        default_allow: bool = True,
        reason: str | None = None,
        overrides: Mapping[str, PolicyDecision] | None = None,
    ) -> FakePolicyEvaluator:
        return FakePolicyEvaluator(
            default_allow=default_allow, reason=reason, overrides=overrides
        )

    return _make


# ---------------------------------------------------------------------------
# FakeObservabilityProvider
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_observability() -> FakeObservabilityProvider:
    return FakeObservabilityProvider()


# ---------------------------------------------------------------------------
# FakeSecretManager
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_secret_manager_factory() -> Callable[..., FakeSecretManager]:
    def _make(
        mapping: Mapping[tuple[str, str], str] | None = None,
    ) -> FakeSecretManager:
        return FakeSecretManager(mapping or {})

    return _make


# ---------------------------------------------------------------------------
# FakeBudgetService — promoted from per-test-file definitions (Phase 3 item 9)
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_budget() -> FakeBudgetService:
    return FakeBudgetService()


# ---------------------------------------------------------------------------
# FakeAuditSink — Phase 3 Task 3
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_audit_sink() -> FakeAuditSink:
    return FakeAuditSink()
