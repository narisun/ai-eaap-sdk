"""Tests for the @tool decorator."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from ai_core.tools import ToolSpec, tool

pytestmark = pytest.mark.unit


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    n: int


def test_decorator_returns_toolspec_with_inferred_models() -> None:
    @tool(name="x", version=1, description="docstring overridden")
    async def fn(payload: _In) -> _Out:
        return _Out(n=0)

    assert isinstance(fn, ToolSpec)
    assert fn.name == "x"
    assert fn.version == 1
    assert fn.description == "docstring overridden"
    assert fn.input_model is _In
    assert fn.output_model is _Out
    assert fn.opa_path == "eaap/agent/tool_call/allow"


def test_description_falls_back_to_docstring() -> None:
    @tool(name="x", version=1)
    async def fn(payload: _In) -> _Out:
        """The real description."""
        return _Out(n=0)

    assert fn.description == "The real description."


def test_description_falls_back_to_empty_when_no_docstring() -> None:
    @tool(name="x", version=1)
    async def fn(payload: _In) -> _Out:
        return _Out(n=0)

    assert fn.description == ""


def test_decorator_rejects_sync_function() -> None:
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)  # type: ignore[arg-type]
        def fn(payload: _In) -> _Out:
            return _Out(n=0)
    assert "async" in str(exc.value).lower()


def test_decorator_rejects_non_basemodel_input() -> None:
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)  # type: ignore[type-var]
        async def fn(payload: dict) -> _Out:  # type: ignore[type-arg]
            return _Out(n=0)
    assert "BaseModel" in str(exc.value)


def test_decorator_rejects_non_basemodel_return() -> None:
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)  # type: ignore[type-var]
        async def fn(payload: _In) -> dict:  # type: ignore[type-arg]
            return {}
    assert "BaseModel" in str(exc.value)


def test_decorator_rejects_zero_or_multi_param_function() -> None:
    with pytest.raises(TypeError):
        @tool(name="x", version=1)  # type: ignore[arg-type]
        async def fn() -> _Out:
            return _Out(n=0)

    with pytest.raises(TypeError):
        @tool(name="x", version=1)  # type: ignore[arg-type]
        async def fn2(a: _In, b: _In) -> _Out:
            return _Out(n=0)


def test_opa_path_can_be_disabled() -> None:
    @tool(name="x", version=1, opa_path=None)
    async def fn(payload: _In) -> _Out:
        return _Out(n=0)

    assert fn.opa_path is None


@pytest.mark.asyncio
async def test_decorated_handler_round_trips() -> None:
    @tool(name="x", version=1)
    async def fn(payload: _In) -> _Out:
        return _Out(n=len(payload.q))

    out = await fn.handler(_In(q="hello"))
    assert out == _Out(n=5)


# ---------------------------------------------------------------------------
# Issue 2 — new validation-gap tests
# ---------------------------------------------------------------------------

def test_decorator_rejects_keyword_only_param() -> None:
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)  # type: ignore[arg-type]
        async def fn(*, payload: _In) -> _Out:
            return _Out(n=0)
    assert "keyword-only" in str(exc.value).lower()


def test_decorator_rejects_var_positional() -> None:
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)
        async def fn(payload: _In, *args: object) -> _Out:
            return _Out(n=0)
    assert "*args" in str(exc.value) or "var" in str(exc.value).lower()


def test_decorator_rejects_var_keyword() -> None:
    with pytest.raises(TypeError) as exc:
        @tool(name="x", version=1)
        async def fn(payload: _In, **kwargs: object) -> _Out:
            return _Out(n=0)
    assert "**kwargs" in str(exc.value) or "var" in str(exc.value).lower()


def test_decorator_rejects_method_with_self() -> None:
    def _build() -> None:
        @tool(name="x", version=1)  # type: ignore[arg-type]
        async def fn(self: object, payload: _In) -> _Out:
            return _Out(n=0)

    with pytest.raises(TypeError) as exc:
        _build()
    assert "module-level" in str(exc.value) or "method" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Minor M1 — TYPE_CHECKING / forward-reference failure test
# ---------------------------------------------------------------------------

def test_decorator_reports_unresolvable_typecheck_only_annotation() -> None:
    """If a user annotates with a TYPE_CHECKING-only symbol, error should be helpful."""
    src = (
        "from __future__ import annotations\n"
        "async def fn(payload: NotARealModel) -> NotARealModel:\n"
        "    return payload\n"
    )
    namespace: dict[str, object] = {}
    exec(src, namespace)
    fn = namespace["fn"]
    with pytest.raises(TypeError) as exc:
        tool(name="x", version=1)(fn)  # type: ignore[arg-type]
    assert "TYPE_CHECKING" in str(exc.value)


# ---------------------------------------------------------------------------
# Minor M5 — name/version validation tests
# ---------------------------------------------------------------------------

def test_decorator_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tool(name="", version=1)


def test_decorator_rejects_zero_version() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        tool(name="x", version=0)
