"""Unit tests for MCPToolSpec and its permissive I/O models."""
from __future__ import annotations

import pytest

from ai_core.mcp import MCPServerSpec
from ai_core.mcp.tools import (
    MCPToolSpec,
    _MCPPassthroughInput,
    _MCPPassthroughOutput,
)

pytestmark = pytest.mark.unit


def _spec_factory(**overrides) -> MCPToolSpec:
    """Build an MCPToolSpec with sensible defaults for testing."""
    server = MCPServerSpec(
        component_id="test-server",
        transport="stdio",
        target="/bin/true",
        opa_decision_path=overrides.pop("opa_decision_path", None),
    )

    async def _noop_handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
        return _MCPPassthroughOutput(value="ok")

    return MCPToolSpec(
        name=overrides.pop("name", "echo"),
        version=1,
        description=overrides.pop("description", "Test tool"),
        input_model=_MCPPassthroughInput,
        output_model=_MCPPassthroughOutput,
        handler=_noop_handler,
        opa_path=overrides.pop("opa_path", None),
        mcp_server_spec=server,
        mcp_input_schema=overrides.pop(
            "mcp_input_schema",
            {"type": "object", "properties": {"text": {"type": "string"}}},
        ),
    )


def test_mcp_tool_spec_is_a_tool_spec() -> None:
    """MCPToolSpec subclasses ToolSpec so existing isinstance checks find it."""
    from ai_core.tools.spec import ToolSpec  # noqa: PLC0415

    spec = _spec_factory()
    assert isinstance(spec, ToolSpec)


def test_openai_schema_returns_raw_input_schema() -> None:
    """MCPToolSpec.openai_schema() returns FastMCP's inputSchema, not Pydantic-derived."""
    raw = {
        "type": "object",
        "properties": {"text": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["text"],
    }
    spec = _spec_factory(mcp_input_schema=raw)

    schema = spec.openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["description"] == "Test tool"
    # Critical: parameters is the raw FastMCP schema, not Pydantic-derived
    assert schema["function"]["parameters"] == raw


def test_passthrough_input_accepts_arbitrary_keys() -> None:
    """_MCPPassthroughInput allows any keys (server-side validation is the source of truth)."""
    payload = _MCPPassthroughInput.model_validate(
        {"text": "hi", "weird_key": [1, 2, 3], "nested": {"a": True}}
    )
    dumped = payload.model_dump()
    assert dumped["text"] == "hi"
    assert dumped["weird_key"] == [1, 2, 3]
    assert dumped["nested"] == {"a": True}


def test_passthrough_output_wraps_any_value() -> None:
    """_MCPPassthroughOutput.value accepts arbitrary Python values."""
    out = _MCPPassthroughOutput(value={"complex": [1, 2]})
    assert out.value == {"complex": [1, 2]}

    out_str = _MCPPassthroughOutput(value="hello")
    assert out_str.value == "hello"


def test_opa_path_default_is_none() -> None:
    """Tools constructed without an opa_path skip OPA (ToolInvoker contract)."""
    spec = _spec_factory()
    assert spec.opa_path is None


def test_opa_path_propagates_from_server_spec() -> None:
    """When the resolver passes server.opa_decision_path → spec.opa_path, it round-trips."""
    spec = _spec_factory(opa_path="mcp.test-server.allow")
    assert spec.opa_path == "mcp.test-server.allow"
