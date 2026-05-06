"""Public testing surface for SDK consumers.

Exports in-memory ``Fake*`` implementations of the SDK's public protocols
so consumer tests can assert against observable state without setting up
real backends::

    from ai_core.testing import FakeAuditSink, FakePolicyEvaluator
    from ai_core.testing import ScriptedLLM, make_llm_response
"""

from __future__ import annotations

from ai_core.testing.fakes import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)
from ai_core.testing.llm import ScriptedLLM, make_llm_response

__all__ = [
    "FakeAuditSink",
    "FakeBudgetService",
    "FakeObservabilityProvider",
    "FakePolicyEvaluator",
    "FakeSecretManager",
    "ScriptedLLM",
    "make_llm_response",
]
