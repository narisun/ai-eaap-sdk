"""Public testing surface for SDK consumers.

Activate the pytest plugin in your conftest.py::

    pytest_plugins = ["ai_core.testing.pytest_plugin"]

Then use the exported fakes (also importable directly here without pytest)::

    from ai_core.testing import FakeAuditSink, ScriptedLLM, make_llm_response
"""

from __future__ import annotations

from ai_core.testing.fakes import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)

__all__ = [
    "FakeAuditSink",
    "FakeBudgetService",
    "FakeObservabilityProvider",
    "FakePolicyEvaluator",
    "FakeSecretManager",
]
