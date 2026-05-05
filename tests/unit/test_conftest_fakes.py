"""Smoke test that the reusable fakes are importable and behave correctly."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from ai_core.config.secrets import SecretRef

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import FakeObservabilityProvider, FakePolicyEvaluator, FakeSecretManager

pytestmark = pytest.mark.unit


def test_fake_policy_evaluator_allow(
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    evaluator = fake_policy_evaluator_factory(default_allow=True)
    decision = asyncio.run(evaluator.evaluate(decision_path="x", input={}))
    assert decision.allowed is True


def test_fake_policy_evaluator_deny(
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    evaluator = fake_policy_evaluator_factory(default_allow=False, reason="nope")
    decision = asyncio.run(evaluator.evaluate(decision_path="x", input={}))
    assert decision.allowed is False
    assert decision.reason == "nope"


def test_fake_observability_records_spans(
    fake_observability: FakeObservabilityProvider,
) -> None:
    async def go() -> None:
        async with fake_observability.start_span("a", attributes={"k": "v"}):
            await fake_observability.record_event("e", attributes={"x": 1})

    asyncio.run(go())
    assert [s.name for s in fake_observability.spans] == ["a"]
    assert fake_observability.spans[0].attributes == {"k": "v"}
    assert fake_observability.events == [("e", {"x": 1})]


def test_fake_secret_manager_resolves(
    fake_secret_manager_factory: Callable[..., FakeSecretManager],
) -> None:
    mgr = fake_secret_manager_factory({("env", "MY_KEY"): "secret-val"})
    val = asyncio.run(mgr.resolve(SecretRef(backend="env", name="MY_KEY")))
    assert val == "secret-val"
