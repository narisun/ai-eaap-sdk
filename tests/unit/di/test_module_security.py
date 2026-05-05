"""Tests verifying the security DI binding behavior in Phase 2.

- AgentModule binds NoOpPolicyEvaluator by default.
- ProductionSecurityModule swaps in OPAPolicyEvaluator when added.
"""
from __future__ import annotations

import pytest

from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import IPolicyEvaluator
from ai_core.di.module import ProductionSecurityModule
from ai_core.security.noop_policy import NoOpPolicyEvaluator
from ai_core.security.opa import OPAPolicyEvaluator

pytestmark = pytest.mark.unit


def test_default_container_resolves_noop_policy() -> None:
    """`Container.build([AgentModule()])` returns the NoOp evaluator."""
    container = Container.build([AgentModule()])
    evaluator = container.get(IPolicyEvaluator)
    assert isinstance(evaluator, NoOpPolicyEvaluator)


def test_production_security_module_swaps_to_opa() -> None:
    """Adding ProductionSecurityModule rebinds the evaluator to OPAPolicyEvaluator."""
    container = Container.build([AgentModule(), ProductionSecurityModule()])
    evaluator = container.get(IPolicyEvaluator)
    assert isinstance(evaluator, OPAPolicyEvaluator)
