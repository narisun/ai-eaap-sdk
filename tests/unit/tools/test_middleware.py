"""Tests for :class:`ai_core.tools.middleware.ToolMiddleware` integration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from ai_core.audit.null import NullAuditSink
from ai_core.testing import FakeObservabilityProvider
from ai_core.tools import ToolSpec, tool
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.middleware import ToolCallContext, ToolMiddleware

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    doubled: int


@tool(name="double", version=1, opa_path=None)
async def _double(p: _In) -> _Out:
    return _Out(doubled=p.value * 2)


def _build_invoker(
    middlewares: list[ToolMiddleware],
) -> tuple[ToolInvoker, list[str]]:
    """Construct a ToolInvoker with the given middlewares + capture log."""
    log: list[str] = []
    invoker = ToolInvoker(
        observability=FakeObservabilityProvider(),
        audit=NullAuditSink(),
        middlewares=middlewares,
    )
    return invoker, log


def _record_middleware(name: str, log: list[str]) -> ToolMiddleware:
    """Build a middleware that logs entry / exit so we can assert order."""

    async def _mw(
        ctx: ToolCallContext,
        call_next: Callable[[], Awaitable[Mapping[str, Any]]],
    ) -> Mapping[str, Any]:
        log.append(f"{name}:before")
        result = await call_next()
        log.append(f"{name}:after")
        return result

    return _mw


@pytest.mark.asyncio
async def test_middleware_chain_runs_in_outermost_first_order() -> None:
    """First registered middleware is the outermost layer."""
    log: list[str] = []
    invoker, _ = _build_invoker([
        _record_middleware("outer", log),
        _record_middleware("inner", log),
    ])

    result = await invoker.invoke(_double, {"value": 3})

    assert result == {"doubled": 6}
    assert log == [
        "outer:before",  # outer entered first
        "inner:before",  # inner entered next
        "inner:after",   # inner exited first (LIFO)
        "outer:after",   # outer exited last
    ]


@pytest.mark.asyncio
async def test_middleware_can_short_circuit_without_calling_next() -> None:
    """A middleware that returns without awaiting next bypasses the pipeline."""

    async def _short(
        ctx: ToolCallContext,
        call_next: Callable[[], Awaitable[Mapping[str, Any]]],
    ) -> Mapping[str, Any]:
        # Never call call_next; return a synthetic result instead.
        return {"doubled": -1}

    invoker, _ = _build_invoker([_short])
    result = await invoker.invoke(_double, {"value": 3})

    assert result == {"doubled": -1}


@pytest.mark.asyncio
async def test_no_middlewares_runs_pipeline_unchanged() -> None:
    """When middlewares=() the invoker behaves exactly like pre-v1."""
    invoker, _ = _build_invoker([])
    result = await invoker.invoke(_double, {"value": 21})
    assert result == {"doubled": 42}


@pytest.mark.asyncio
async def test_middleware_sees_spec_and_args_in_context() -> None:
    """Each middleware receives a ToolCallContext snapshot of the invocation."""
    seen: dict[str, Any] = {}

    async def _capture(
        ctx: ToolCallContext,
        call_next: Callable[[], Awaitable[Mapping[str, Any]]],
    ) -> Mapping[str, Any]:
        seen["spec"] = ctx.spec
        seen["raw_args"] = dict(ctx.raw_args)
        seen["agent_id"] = ctx.agent_id
        seen["tenant_id"] = ctx.tenant_id
        return await call_next()

    invoker, _ = _build_invoker([_capture])
    await invoker.invoke(
        _double, {"value": 7}, agent_id="my-agent", tenant_id="acme",
    )

    assert isinstance(seen["spec"], ToolSpec)
    assert seen["spec"].name == "double"
    assert seen["raw_args"] == {"value": 7}
    assert seen["agent_id"] == "my-agent"
    assert seen["tenant_id"] == "acme"
