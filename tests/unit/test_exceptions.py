"""Smoke tests for :mod:`ai_core.exceptions`."""

from __future__ import annotations

import pytest

from ai_core.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    EAAPBaseException,
    LLMInvocationError,
    SchemaValidationError,
    SecretResolutionError,
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
    assert err.details == {"key": "DSN"}
    other = ConfigurationError("other")
    assert other.details == {}  # not shared with first instance


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
