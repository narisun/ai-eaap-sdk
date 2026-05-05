"""Asserts the curated top-level public surface is reachable."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_canonical_imports_exist() -> None:
    from ai_core import (  # noqa: PLC0415, F401, I001
        AICoreApp,
        AgentRecursionLimitError,
        AgentRuntimeError,
        AgentState,
        BaseAgent,
        BudgetExceededError,
        ConfigurationError,
        DependencyResolutionError,
        EAAPBaseException,
        HealthSnapshot,
        LLMInvocationError,
        PolicyDenialError,
        RegistryError,
        SchemaValidationError,
        SecretResolutionError,
        StorageError,
        Tool,
        ToolExecutionError,
        ToolSpec,
        ToolValidationError,
        new_agent_state,
        tool,
    )
    # Smoke: each name is non-None.
    locals_dict = locals()
    for name in [
        "AICoreApp", "AgentRecursionLimitError", "AgentRuntimeError",
        "AgentState", "BaseAgent", "BudgetExceededError", "ConfigurationError",
        "DependencyResolutionError", "EAAPBaseException", "HealthSnapshot",
        "LLMInvocationError", "PolicyDenialError", "RegistryError",
        "SchemaValidationError", "SecretResolutionError", "StorageError",
        "Tool", "ToolExecutionError", "ToolSpec", "ToolValidationError",
        "new_agent_state", "tool",
    ]:
        assert locals_dict[name] is not None


def test_version_string_is_set() -> None:
    import ai_core  # noqa: PLC0415
    assert isinstance(ai_core.__version__, str)
    assert ai_core.__version__
