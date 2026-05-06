"""Pytest plugin exposing ai_core.testing fakes as fixtures.

Activate via your conftest.py::

    pytest_plugins = ["ai_core.testing.pytest_plugin"]

Then write tests that consume the fixtures::

    async def test_my_agent(scripted_llm_factory, fake_audit_sink):
        from ai_core.testing import make_llm_response
        llm = scripted_llm_factory([make_llm_response("ok")])
        # ...
        assert len(fake_audit_sink.records) == 1
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ai_core.testing.fakes import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)
from ai_core.testing.llm import ScriptedLLM

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ai_core.di.interfaces import LLMResponse


@pytest.fixture
def fake_audit_sink() -> FakeAuditSink:
    """Fresh per-test :class:`FakeAuditSink` instance."""
    return FakeAuditSink()


@pytest.fixture
def fake_observability() -> FakeObservabilityProvider:
    """Fresh per-test :class:`FakeObservabilityProvider` instance."""
    return FakeObservabilityProvider()


@pytest.fixture
def fake_budget() -> FakeBudgetService:
    """Fresh per-test :class:`FakeBudgetService` instance."""
    return FakeBudgetService()


@pytest.fixture
def fake_policy_evaluator_factory() -> Callable[..., FakePolicyEvaluator]:
    """Factory: ``factory(default_allow=True)`` returns a configured fake."""

    def _factory(
        *,
        default_allow: bool = True,
        reason: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> FakePolicyEvaluator:
        return FakePolicyEvaluator(
            default_allow=default_allow, reason=reason, overrides=overrides
        )

    return _factory


@pytest.fixture
def fake_secret_manager_factory() -> Callable[..., FakeSecretManager]:
    """Factory: ``factory({(backend, name): value})`` returns a configured fake."""

    def _factory(
        mapping: Mapping[tuple[str, str], str] | None = None,
    ) -> FakeSecretManager:
        return FakeSecretManager(mapping or {})

    return _factory


@pytest.fixture
def scripted_llm_factory() -> Callable[..., ScriptedLLM]:
    """Factory: ``factory([resp1, resp2], repeat_last=False)`` returns a ScriptedLLM."""

    def _factory(
        responses: Sequence[LLMResponse],
        *,
        repeat_last: bool = False,
    ) -> ScriptedLLM:
        return ScriptedLLM(list(responses), repeat_last=repeat_last)

    return _factory
