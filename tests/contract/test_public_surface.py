"""Pin the SDK's top-level public surface.

If you add or remove a name from ai_core.__all__, you must also update
EXPECTED_PUBLIC_NAMES below. The two-place edit is deliberate — it forces
contributors to acknowledge that the public surface has changed.
"""
from __future__ import annotations

import pytest

import ai_core

pytestmark = pytest.mark.contract

EXPECTED_PUBLIC_NAMES: frozenset[str] = frozenset({
    "AICoreApp",
    "AgentRecursionLimitError",
    "AgentRuntime",
    "AgentRuntimeError",
    "AgentState",
    "AuditEvent",
    "AuditRecord",
    "BaseAgent",
    "BudgetExceededError",
    "ConfigurationError",
    "DependencyResolutionError",
    "EAAPBaseException",
    "ErrorCode",                        # NEW (Phase 8)
    "HealthSnapshot",
    "IAuditSink",
    "IHealthProbe",
    "LLMInvocationError",
    "LLMTimeoutError",
    "MCPTransportError",
    "PolicyDenialError",
    "ProbeResult",
    "RegistryError",
    "SchemaValidationError",
    "SecretResolutionError",
    "StorageError",
    "Plan",
    "PlanAck",
    "PlanStep",
    "PlanningAgent",
    "SupervisorAgent",
    "TaskInput",
    "TaskOutput",
    "Tool",
    "ToolExecutionError",
    "ToolSpec",
    "ToolValidationError",
    "Verdict",
    "VerifierAgent",
    "make_tool",
    "new_agent_state",
    "tool",
})


def test_public_surface_matches_expected() -> None:
    actual = set(ai_core.__all__)
    missing = EXPECTED_PUBLIC_NAMES - actual
    extra = actual - EXPECTED_PUBLIC_NAMES
    assert not missing, f"Missing exports: {sorted(missing)}"
    assert not extra, f"Unexpected new exports: {sorted(extra)}"
