"""Smoke tests for :mod:`ai_core.exceptions`."""

from __future__ import annotations

import pytest

from ai_core.exceptions import (
    AgentRecursionLimitError,
    AgentRuntimeError,
    BudgetExceededError,
    CheckpointError,
    ConfigurationError,
    DependencyResolutionError,
    EAAPBaseException,
    LLMInvocationError,
    LLMTimeoutError,
    MCPTransportError,
    PolicyDenialError,
    RegistryError,
    SchemaValidationError,
    SecretResolutionError,
    StorageError,
    ToolExecutionError,
    ToolValidationError,
)

pytestmark = pytest.mark.unit


def test_hierarchy_anchored_on_base() -> None:
    assert issubclass(ConfigurationError, EAAPBaseException)
    assert issubclass(SecretResolutionError, ConfigurationError)
    assert issubclass(LLMInvocationError, EAAPBaseException)
    assert issubclass(BudgetExceededError, LLMInvocationError)


def test_details_attached_and_immutable_per_instance() -> None:
    err = ConfigurationError("missing", details={"key": "DSN"})
    assert err.details["key"] == "DSN"  # error_code also added; check by key
    other = ConfigurationError("other")
    assert "key" not in other.details  # not shared with first instance


def test_cause_chain_is_preserved() -> None:
    inner = RuntimeError("boom")
    err = ConfigurationError("wrapped", cause=inner)
    assert err.__cause__ is inner
    assert err.cause is inner


def test_repr_includes_message_and_details() -> None:
    err = ConfigurationError("bad", details={"k": 1})
    rendered = repr(err)
    assert "ConfigurationError" in rendered
    assert "bad" in rendered
    assert "'k': 1" in rendered


def test_tool_validation_error_is_schema_validation_error() -> None:
    err = ToolValidationError(
        "bad",
        details={"tool": "x", "version": 1, "side": "input", "errors": []},
    )
    assert isinstance(err, SchemaValidationError)
    assert isinstance(err, EAAPBaseException)
    assert err.details["side"] == "input"


def test_tool_execution_error_chains_cause() -> None:
    cause = RuntimeError("inner")
    err = ToolExecutionError(
        "bad",
        details={"tool": "x", "version": 1, "agent_id": "a", "tenant_id": "t"},
        cause=cause,
    )
    assert err.__cause__ is cause
    assert err.cause is cause
    assert err.details["agent_id"] == "a"


def test_tool_execution_error_is_eaap_base_exception() -> None:
    err = ToolExecutionError("x", details={"tool": "x", "version": 1})
    assert isinstance(err, EAAPBaseException)
    # Not a SchemaValidationError — that's reserved for validation failures.
    assert not isinstance(err, SchemaValidationError)


# ---------------------------------------------------------------------------
# error_code field — Phase 2
# ---------------------------------------------------------------------------


def test_base_default_code() -> None:
    err = EAAPBaseException("x")
    assert err.error_code == "eaap.unknown"
    assert err.details["error_code"] == "eaap.unknown"


def test_per_instance_override_wins() -> None:
    err = LLMInvocationError("x", error_code="llm.context_length_exceeded")
    assert err.error_code == "llm.context_length_exceeded"
    assert err.details["error_code"] == "llm.context_length_exceeded"


@pytest.mark.parametrize(
    ("cls", "expected_code"),
    [
        (ConfigurationError, "config.invalid"),
        (SecretResolutionError, "config.secret_not_resolved"),
        (DependencyResolutionError, "di.resolution_failed"),
        (StorageError, "storage.error"),
        (CheckpointError, "storage.checkpoint_failed"),
        (PolicyDenialError, "policy.denied"),
        (LLMInvocationError, "llm.invocation_failed"),
        (LLMTimeoutError, "llm.timeout"),
        (BudgetExceededError, "llm.budget_exceeded"),
        (SchemaValidationError, "schema.invalid"),
        (ToolValidationError, "tool.validation_failed"),
        (ToolExecutionError, "tool.execution_failed"),
        (AgentRuntimeError, "agent.runtime_error"),
        (AgentRecursionLimitError, "agent.recursion_limit"),
        (RegistryError, "registry.error"),
        (MCPTransportError, "mcp.transport_failed"),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else str(v),
)
def test_subclass_default_codes(
    cls: type[EAAPBaseException], expected_code: str
) -> None:
    err = cls("msg")
    assert err.error_code == expected_code
    assert err.details["error_code"] == expected_code


def test_existing_details_preserved_with_error_code_added() -> None:
    err = LLMInvocationError(
        "x", details={"model": "gpt-4", "attempts": 3}
    )
    assert err.details == {
        "model": "gpt-4",
        "attempts": 3,
        "error_code": "llm.invocation_failed",
    }


def test_error_code_arg_overrides_details_error_code() -> None:
    """The error_code arg always wins; details['error_code'] mirrors self.error_code."""
    err = LLMInvocationError(
        "x", details={"error_code": "llm.custom"}, error_code="llm.timeout"
    )
    # error_code arg sets self.error_code AND overwrites any details["error_code"]
    # so dashboards never see divergence between the attribute and the dict.
    assert err.error_code == "llm.timeout"
    assert err.details["error_code"] == "llm.timeout"


def test_details_error_code_used_when_no_arg_given() -> None:
    """If no error_code arg is given, the subclass DEFAULT_CODE wins (still overwriting)."""
    err = LLMInvocationError("x", details={"error_code": "llm.something_else"})
    # No explicit error_code arg → DEFAULT_CODE is used, AND it overwrites
    # whatever the caller put in details. This is intentional last-write-wins.
    assert err.error_code == "llm.invocation_failed"
    assert err.details["error_code"] == "llm.invocation_failed"


def test_llm_timeout_error_lineage() -> None:
    err = LLMTimeoutError("timed out")
    assert isinstance(err, LLMInvocationError)
    assert isinstance(err, EAAPBaseException)
    assert err.error_code == "llm.timeout"


def test_mcp_transport_error_lineage() -> None:
    err = MCPTransportError("transport down")
    assert isinstance(err, EAAPBaseException)
    assert not isinstance(err, LLMInvocationError)
    assert err.error_code == "mcp.transport_failed"
