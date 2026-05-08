"""Tests for the v1 ``as_typed_details()`` accessor on SDK exceptions.

Each exception subclass with a stable ``details`` schema exposes a
typed dataclass payload reachable via ``exc.as_typed_details()``. These
tests pin the field mapping and the back-compat behaviour (raw dict
``details`` still works alongside the typed accessor).
"""

from __future__ import annotations

import pytest

from ai_core.exceptions import (
    AgentRecursionDetails,
    AgentRecursionLimitError,
    BudgetExceededDetails,
    BudgetExceededError,
    CheckpointDetails,
    CheckpointError,
    DependencyResolutionDetails,
    DependencyResolutionError,
    LLMTimeoutDetails,
    LLMTimeoutError,
    MCPTransportDetails,
    MCPTransportError,
    SecretResolutionDetails,
    SecretResolutionError,
    ToolExecutionDetails,
    ToolExecutionError,
    ToolValidationDetails,
    ToolValidationError,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ToolValidationError / ToolExecutionError
# ---------------------------------------------------------------------------
def test_tool_validation_typed_details() -> None:
    exc = ToolValidationError(
        "input failed",
        details={
            "tool": "lookup",
            "version": 2,
            "side": "input",
            "errors": [{"loc": ("name",), "msg": "Field required"}],
        },
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, ToolValidationDetails)
    assert typed.tool == "lookup"
    assert typed.version == 2
    assert typed.side == "input"
    assert len(typed.errors) == 1
    assert typed.errors[0]["msg"] == "Field required"


def test_tool_execution_typed_details_with_optional_ids() -> None:
    exc = ToolExecutionError(
        "handler raised",
        details={
            "tool": "lookup",
            "version": 1,
            "agent_id": "support-bot",
            "tenant_id": "acme",
        },
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, ToolExecutionDetails)
    assert typed.tool == "lookup"
    assert typed.version == 1
    assert typed.agent_id == "support-bot"
    assert typed.tenant_id == "acme"


def test_tool_execution_typed_details_drops_missing_optionals_to_none() -> None:
    exc = ToolExecutionError("handler raised", details={"tool": "x", "version": 1})
    typed = exc.as_typed_details()
    assert typed.agent_id is None
    assert typed.tenant_id is None


# ---------------------------------------------------------------------------
# BudgetExceededError / LLMTimeoutError
# ---------------------------------------------------------------------------
def test_budget_exceeded_typed_details_full_payload() -> None:
    exc = BudgetExceededError(
        "denied",
        details={
            "tenant_id": "acme",
            "agent_id": "summariser",
            "model": "claude-haiku",
            "estimated_tokens": 1500,
            "remaining_tokens": 500,
            "remaining_usd": 1.23,
            "reason": "daily token limit exceeded",
        },
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, BudgetExceededDetails)
    assert typed.tenant_id == "acme"
    assert typed.estimated_tokens == 1500
    assert typed.remaining_usd == pytest.approx(1.23)
    assert typed.reason == "daily token limit exceeded"


def test_budget_exceeded_typed_details_partial_payload_keeps_unset_as_none() -> None:
    exc = BudgetExceededError("denied", details={"reason": "no quota"})
    typed = exc.as_typed_details()
    assert typed.tenant_id is None
    assert typed.estimated_tokens is None
    assert typed.remaining_tokens is None
    assert typed.remaining_usd is None
    assert typed.reason == "no quota"


def test_llm_timeout_typed_details() -> None:
    exc = LLMTimeoutError(
        "timed out", details={"model": "gpt-4o-mini", "attempts": 3},
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, LLMTimeoutDetails)
    assert typed.model == "gpt-4o-mini"
    assert typed.attempts == 3


# ---------------------------------------------------------------------------
# Other stable schemas
# ---------------------------------------------------------------------------
def test_mcp_transport_typed_details() -> None:
    exc = MCPTransportError(
        "open failed", details={"component_id": "search-svc", "transport": "stdio"},
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, MCPTransportDetails)
    assert typed.component_id == "search-svc"
    assert typed.transport == "stdio"


def test_agent_recursion_typed_details() -> None:
    exc = AgentRecursionLimitError(
        "looped",
        details={
            "agent_id": "tutor",
            "tenant_id": "acme",
            "thread_id": "thread-7",
            "limit": 25,
        },
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, AgentRecursionDetails)
    assert typed.limit == 25
    assert typed.thread_id == "thread-7"


def test_secret_resolution_typed_details() -> None:
    exc = SecretResolutionError(
        "missing", details={"backend": "env", "name": "OPENAI_API_KEY"},
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, SecretResolutionDetails)
    assert typed.backend == "env"
    assert typed.name == "OPENAI_API_KEY"


def test_dependency_resolution_typed_details() -> None:
    exc = DependencyResolutionError(
        "unbound", details={"interface": "ILLMClient"},
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, DependencyResolutionDetails)
    assert typed.interface == "ILLMClient"


def test_checkpoint_typed_details_with_optional_id() -> None:
    exc = CheckpointError(
        "save failed",
        details={"thread_id": "t-1", "checkpoint_id": "c-42"},
    )
    typed = exc.as_typed_details()
    assert isinstance(typed, CheckpointDetails)
    assert typed.thread_id == "t-1"
    assert typed.checkpoint_id == "c-42"


def test_checkpoint_typed_details_omitted_id_is_none() -> None:
    exc = CheckpointError("save failed", details={"thread_id": "t-1"})
    typed = exc.as_typed_details()
    assert typed.checkpoint_id is None


# ---------------------------------------------------------------------------
# Back-compat: raw dict still works alongside typed accessor
# ---------------------------------------------------------------------------
def test_raw_dict_details_still_accessible_alongside_typed_view() -> None:
    """Typed accessor doesn't replace ``self.details``; both are usable."""
    exc = ToolValidationError(
        "x",
        details={"tool": "t", "version": 1, "side": "input", "errors": []},
    )
    assert exc.details["tool"] == "t"
    assert exc.details["error_code"] == "tool.validation_failed"  # populated by base
    typed = exc.as_typed_details()
    assert typed.tool == "t"


def test_typed_details_is_frozen() -> None:
    """Typed payloads are frozen dataclasses — immutable for safe sharing."""
    exc = MCPTransportError("x", details={"component_id": "c", "transport": "stdio"})
    typed = exc.as_typed_details()
    # A frozen dataclass with slots raises FrozenInstanceError on attribute
    # set; with slots alone it would be AttributeError. Accept either by
    # narrowing to the precise type that fires under dataclasses(frozen=True).
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        typed.component_id = "other"  # type: ignore[misc]
