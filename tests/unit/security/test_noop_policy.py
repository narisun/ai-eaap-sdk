"""Tests for the NoOpPolicyEvaluator."""
from __future__ import annotations

import pytest

from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision
from ai_core.security.noop_policy import NoOpPolicyEvaluator

pytestmark = pytest.mark.unit


def test_noop_implements_ipolicyevaluator() -> None:
    assert isinstance(NoOpPolicyEvaluator(), IPolicyEvaluator)


@pytest.mark.asyncio
async def test_noop_always_allows() -> None:
    evaluator = NoOpPolicyEvaluator()
    decision = await evaluator.evaluate(
        decision_path="anything", input={"user": "anyone"}
    )
    assert isinstance(decision, PolicyDecision)
    assert decision.allowed is True
    assert decision.obligations == {}
    assert decision.reason == "no-op evaluator (development only)"
