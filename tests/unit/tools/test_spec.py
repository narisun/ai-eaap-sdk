"""Tests for ToolSpec dataclass and the Tool Protocol."""
from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import BaseModel

from ai_core.tools.spec import Tool, ToolHandler, ToolSpec

pytestmark = pytest.mark.unit


class _In(BaseModel):
    query: str
    limit: int = 10


class _Out(BaseModel):
    items: list[str]


async def _handler(payload: _In) -> _Out:
    return _Out(items=[])


def _spec() -> ToolSpec:
    return ToolSpec(
        name="search",
        version=1,
        description="search items",
        input_model=_In,
        output_model=_Out,
        handler=cast("ToolHandler", _handler),
        opa_path="eaap/agent/tool_call/allow",
    )


def test_toolspec_is_frozen() -> None:
    spec = _spec()
    with pytest.raises(AttributeError):
        spec.name = "other"  # type: ignore[misc]


def test_openai_schema_round_trips_through_json() -> None:
    schema = _spec().openai_schema()
    blob = json.dumps(schema)
    restored = json.loads(blob)
    assert restored["type"] == "function"
    assert restored["function"]["name"] == "search"
    assert restored["function"]["description"] == "search items"
    assert "query" in restored["function"]["parameters"]["properties"]


def test_toolspec_satisfies_tool_protocol() -> None:
    spec = _spec()
    # Structural subtype: ToolSpec instances must be usable as Tool.
    assert isinstance(spec, Tool)  # runtime_checkable Protocol
    assert spec.name == "search"
    assert spec.version == 1


def test_toolspec_eq_uses_name_and_version() -> None:
    a = _spec()
    b = _spec()
    assert a == b
    c = ToolSpec(
        name="search",
        version=2,
        description="search items",
        input_model=_In,
        output_model=_Out,
        handler=cast("ToolHandler", _handler),
        opa_path=None,
    )
    assert a != c
