"""Tests for :func:`ai_core.tools.make_tool`."""

from __future__ import annotations

import functools

import pytest
from pydantic import BaseModel

from ai_core.tools import make_tool

pytestmark = pytest.mark.unit


class _In(BaseModel):
    name: str


class _Out(BaseModel):
    greeting: str


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------
def test_make_tool_accepts_a_module_level_async_function() -> None:
    async def handler(p: _In) -> _Out:
        return _Out(greeting=f"hi {p.name}")

    spec = make_tool(name="hi", version=1, handler=handler)

    assert spec.name == "hi"
    assert spec.version == 1
    assert spec.input_model is _In
    assert spec.output_model is _Out
    assert spec.handler is handler


def test_make_tool_accepts_a_bound_method_with_di_resolved_state() -> None:
    """The whole point of make_tool: handler whose `self` carries a DI-resolved dep."""

    class _Service:
        def __init__(self, prefix: str) -> None:
            self._prefix = prefix

        async def hello(self, p: _In) -> _Out:
            return _Out(greeting=f"{self._prefix} {p.name}")

    svc = _Service("yo")
    spec = make_tool(name="hello", version=1, handler=svc.hello)

    assert spec.input_model is _In
    assert spec.output_model is _Out
    # The bound method retains its closure over `self`.
    import asyncio

    out = asyncio.run(spec.handler(_In(name="bob")))
    assert isinstance(out, _Out)
    assert out.greeting == "yo bob"


def test_make_tool_accepts_a_closure() -> None:
    def factory(suffix: str):
        async def handler(p: _In) -> _Out:
            return _Out(greeting=f"hi {p.name}{suffix}")

        return handler

    spec = make_tool(name="hi", version=2, handler=factory("!"))
    assert spec.version == 2


def test_make_tool_uses_handler_docstring_when_description_omitted() -> None:
    async def handler(p: _In) -> _Out:
        """Greet a user by name."""
        return _Out(greeting=p.name)

    spec = make_tool(name="hi", version=1, handler=handler)
    assert spec.description == "Greet a user by name."


def test_make_tool_explicit_description_wins_over_docstring() -> None:
    async def handler(p: _In) -> _Out:
        """Will not be used."""
        return _Out(greeting=p.name)

    spec = make_tool(
        name="hi", version=1, handler=handler, description="explicit",
    )
    assert spec.description == "explicit"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_make_tool_rejects_empty_name() -> None:
    async def handler(p: _In) -> _Out:
        return _Out(greeting=p.name)

    with pytest.raises(ValueError, match="non-empty"):
        make_tool(name="", version=1, handler=handler)


def test_make_tool_rejects_zero_version() -> None:
    async def handler(p: _In) -> _Out:
        return _Out(greeting=p.name)

    with pytest.raises(ValueError, match=">= 1"):
        make_tool(name="x", version=0, handler=handler)


def test_make_tool_rejects_sync_handler() -> None:
    def sync_handler(p: _In) -> _Out:
        return _Out(greeting=p.name)

    with pytest.raises(TypeError, match="async"):
        make_tool(name="x", version=1, handler=sync_handler)


def test_make_tool_rejects_handler_with_two_positional_args() -> None:
    async def handler(p: _In, extra: str) -> _Out:
        return _Out(greeting=p.name + extra)

    with pytest.raises(TypeError, match="exactly one positional"):
        make_tool(name="x", version=1, handler=handler)


def test_make_tool_rejects_handler_with_kwargs() -> None:
    async def handler(p: _In, **kw: str) -> _Out:
        return _Out(greeting=p.name)

    with pytest.raises(TypeError, match=r"\*\*kwargs"):
        make_tool(name="x", version=1, handler=handler)


def test_make_tool_rejects_handler_without_pydantic_input() -> None:
    async def handler(p: dict) -> _Out:  # type: ignore[type-arg]
        return _Out(greeting=str(p))

    with pytest.raises(TypeError, match="Pydantic BaseModel"):
        make_tool(name="x", version=1, handler=handler)


def test_make_tool_rejects_handler_without_pydantic_output() -> None:
    async def handler(p: _In) -> dict:  # type: ignore[type-arg]
        return {"greeting": p.name}

    with pytest.raises(TypeError, match="return annotation"):
        make_tool(name="x", version=1, handler=handler)


def test_make_tool_unwraps_partial_for_async_check() -> None:
    """functools.partial of an async function is still an async handler."""

    async def base(prefix: str, p: _In) -> _Out:
        return _Out(greeting=f"{prefix} {p.name}")

    bound = functools.partial(base, "yo")
    # functools.partial is callable but inspect.iscoroutinefunction on the
    # partial itself returns False on older Python; make_tool unwraps it.
    spec = make_tool(name="x", version=1, handler=bound)
    assert spec.input_model is _In
