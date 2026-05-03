"""Smoke tests for :mod:`ai_core.exceptions`."""

from __future__ import annotations

import pytest

from ai_core.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    EAAPBaseException,
    LLMInvocationError,
    SecretResolutionError,
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
