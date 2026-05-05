"""Tests for the ToolInvoker pipeline."""
from __future__ import annotations

import json as _json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from ai_core.exceptions import (
    PolicyDenialError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.schema.registry import SchemaRegistry
from ai_core.tools import tool
from ai_core.tools.invoker import ToolInvoker

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import FakeObservabilityProvider, FakePolicyEvaluator

pytestmark = pytest.mark.unit


class _In(BaseModel):
    q: str
    limit: int = Field(default=10, ge=1)


# --- Module-level fixtures for datetime/UUID JSON-safety test -----------------


class _DateOut(BaseModel):
    when: datetime
    ref: UUID


@tool(name="ts", version=1)
async def _ts_tool(payload: _In) -> _DateOut:
    return _DateOut(
        when=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        ref=UUID("12345678-1234-5678-1234-567812345678"),
    )


class _Out(BaseModel):
    items: list[str]


@tool(name="search", version=1, description="d")
async def _search(payload: _In) -> _Out:
    return _Out(items=[payload.q] * payload.limit)


def _invoker(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
    *,
    allow: bool = True,
    reason: str | None = None,
) -> ToolInvoker:
    return ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(default_allow=allow, reason=reason),
        registry=SchemaRegistry(),
    )


@pytest.mark.asyncio
async def test_happy_path(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    result = await inv.invoke(_search, {"q": "hi", "limit": 2}, agent_id="a", tenant_id="t")
    assert result == {"items": ["hi", "hi"]}
    assert [s.name for s in fake_observability.spans] == ["tool.invoke"]
    span = fake_observability.spans[0]
    assert span.attributes["tool.name"] == "search"
    assert span.attributes["tool.version"] == 1
    assert span.attributes["agent_id"] == "a"
    assert span.attributes["tenant_id"] == "t"
    assert ("tool.completed", {
        "tool.name": "search", "tool.version": 1, "agent_id": "a", "tenant_id": "t",
    }) in [(n, dict(a)) for n, a in fake_observability.events]


@pytest.mark.asyncio
async def test_input_validation_failure(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolValidationError) as exc:
        await inv.invoke(_search, {"q": "x", "limit": -1})
    assert exc.value.details["side"] == "input"
    assert exc.value.details["tool"] == "search"
    assert exc.value.details["version"] == 1


@pytest.mark.asyncio
async def test_input_validation_runs_before_opa(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """Input failure must short-circuit before OPA is consulted."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    policy = cast("FakePolicyEvaluator", inv._policy)  # introspection for the test
    with pytest.raises(ToolValidationError):
        await inv.invoke(_search, {"q": "x", "limit": -1})
    assert policy.calls == []  # OPA never called


@pytest.mark.asyncio
async def test_opa_deny_raises_policy_denial(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    inv = _invoker(
        fake_observability, fake_policy_evaluator_factory, allow=False, reason="denied"
    )
    with pytest.raises(PolicyDenialError) as exc:
        await inv.invoke(_search, {"q": "x", "limit": 1})
    assert exc.value.details["tool"] == "search"
    assert "denied" in exc.value.message.lower() or exc.value.details.get("reason") == "denied"
    # OPA deny now propagates inside the span (gets eaap.error.code tagging).
    assert any(
        s.name == "tool.invoke" and s.error_code == "policy.denied"
        for s in fake_observability.spans
    )


@pytest.mark.asyncio
async def test_handler_raise_wraps_as_execution_error(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    @tool(name="boom", version=1)
    async def boom(payload: _In) -> _Out:
        raise RuntimeError("kaboom")

    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolExecutionError) as exc:
        await inv.invoke(boom, {"q": "x", "limit": 1}, agent_id="a")
    assert exc.value.details["tool"] == "boom"
    assert exc.value.details["agent_id"] == "a"
    assert isinstance(exc.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_output_validation_failure(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    @tool(name="lying", version=1)
    async def lying(payload: _In) -> _Out:
        # Cast around mypy: hand back a non-conforming dict so the invoker
        # has to validate-and-fail.
        return {"wrong": True}  # type: ignore[return-value]

    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolValidationError) as exc:
        await inv.invoke(lying, {"q": "x", "limit": 1})
    assert exc.value.details["side"] == "output"
    assert exc.value.details["tool"] == "lying"
    # Completion event must not fire when output validation fails.
    assert all(name != "tool.completed" for name, _ in fake_observability.events)


@pytest.mark.asyncio
async def test_opa_path_none_skips_policy(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    @tool(name="public", version=1, opa_path=None)
    async def public(payload: _In) -> _Out:
        return _Out(items=[])

    inv = _invoker(
        fake_observability, fake_policy_evaluator_factory, allow=False, reason="denied"
    )
    # Even though OPA is wired to deny, opa_path=None bypasses it.
    result = await inv.invoke(public, {"q": "x", "limit": 1})
    assert result == {"items": []}


@pytest.mark.asyncio
async def test_policy_none_skips_opa(fake_observability: FakeObservabilityProvider) -> None:
    inv = ToolInvoker(observability=fake_observability, policy=None, registry=None)
    result = await inv.invoke(_search, {"q": "hi", "limit": 1})
    assert result == {"items": ["hi"]}


@pytest.mark.asyncio
async def test_register_with_registry_is_idempotent(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    registry = SchemaRegistry()
    inv = ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(),
        registry=registry,
    )
    inv.register(_search)
    inv.register(_search)  # idempotent — does NOT raise
    rec = registry.get("search", version=1)
    assert rec.input_schema is _In
    assert rec.output_schema is _Out


@pytest.mark.asyncio
async def test_span_records_exception_on_handler_raise(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    @tool(name="explode", version=1)
    async def explode(payload: _In) -> _Out:
        raise RuntimeError("kaboom")

    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolExecutionError):
        await inv.invoke(explode, {"q": "x", "limit": 1})
    assert fake_observability.spans[0].exception is not None


@pytest.mark.asyncio
async def test_pipeline_order_input_then_opa_then_handler(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """When all stages succeed, exactly one OPA call is made for an allowed verdict."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    await inv.invoke(_search, {"q": "x", "limit": 1}, principal={"sub": "u1"})
    policy = cast("FakePolicyEvaluator", inv._policy)
    assert len(policy.calls) == 1
    call = policy.calls[0]
    assert call.decision_path == "eaap/agent/tool_call/allow"
    assert call.input["tool"] == "search"
    assert call.input["payload"] == {"q": "x", "limit": 1}
    assert call.input["user"] == {"sub": "u1"}


@pytest.mark.asyncio
async def test_invoke_returns_json_safe_dict_for_datetime_output(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """Output models containing datetime/UUID must round-trip through json.dumps."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    result = await inv.invoke(_ts_tool, {"q": "x", "limit": 1})
    # Must be JSON-serializable without error.
    blob = _json.dumps(result)
    parsed = _json.loads(blob)
    assert parsed["when"].startswith("2026-01-01T12:00:00")
    assert parsed["ref"] == "12345678-1234-5678-1234-567812345678"


@pytest.mark.asyncio
async def test_input_validation_error_tags_tool_invoke_span(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """ToolValidationError(side='input') must propagate inside the tool.invoke span."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolValidationError):
        await inv.invoke(_search, {"q": "x", "limit": -1})  # limit<1 fails Pydantic
    spans = [s for s in fake_observability.spans if s.name == "tool.invoke"]
    assert len(spans) == 1
    assert spans[0].error_code == "tool.validation_failed"


@pytest.mark.asyncio
async def test_policy_denial_tags_tool_invoke_span(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """PolicyDenialError must propagate inside the tool.invoke span."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory, allow=False, reason="denied")
    with pytest.raises(PolicyDenialError):
        await inv.invoke(_search, {"q": "x", "limit": 1})
    spans = [s for s in fake_observability.spans if s.name == "tool.invoke"]
    assert len(spans) == 1
    assert spans[0].error_code == "policy.denied"


@pytest.mark.asyncio
async def test_output_validation_error_tags_tool_invoke_span(
    fake_observability: FakeObservabilityProvider,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """ToolValidationError(side='output') must propagate inside the tool.invoke span."""

    @tool(name="lying", version=1)
    async def lying(payload: _In) -> _Out:
        return {"wrong": True}  # type: ignore[return-value]

    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolValidationError):
        await inv.invoke(lying, {"q": "x", "limit": 1})
    spans = [s for s in fake_observability.spans if s.name == "tool.invoke"]
    assert len(spans) == 1
    assert spans[0].error_code == "tool.validation_failed"
