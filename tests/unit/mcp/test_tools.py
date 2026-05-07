"""Unit tests for MCPToolSpec and its permissive I/O models."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ai_core.mcp import MCPServerSpec
from ai_core.mcp.tools import (
    MCPResourceSpec,
    MCPToolSpec,
    _MCPPassthroughInput,
    _MCPPassthroughOutput,
    unwrap_mcp_tool_message,
)
from ai_core.tools.spec import ToolSpec

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


def test_openai_schema_returned_dict_is_independent_of_source() -> None:
    """Mutating the returned parameters dict (or its nested values) doesn't affect the spec."""
    raw = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    }
    spec = _spec_factory(mcp_input_schema=raw)

    schema = spec.openai_schema()
    schema["function"]["parameters"]["properties"]["text"]["INJECTED"] = "MUTATED"
    schema["function"]["parameters"]["new_key"] = "MUTATED"

    # The original raw dict and the spec's stored schema must NOT see the mutation.
    assert "INJECTED" not in raw["properties"]["text"]
    assert "new_key" not in raw
    assert "INJECTED" not in spec.mcp_input_schema["properties"]["text"]
    assert "new_key" not in spec.mcp_input_schema


# ---------------------------------------------------------------------------
# MCPResourceSpec tests (Phase 12)
# ---------------------------------------------------------------------------


def _resource_spec_factory(**overrides) -> MCPResourceSpec:
    """Build an MCPResourceSpec with sensible defaults for testing."""
    server = MCPServerSpec(
        component_id="test-server",
        transport="stdio",
        target="/bin/true",
        opa_decision_path=overrides.pop("opa_decision_path", None),
    )

    async def _noop_handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
        return _MCPPassthroughOutput(value="ok")

    return MCPResourceSpec(
        name=overrides.pop("name", "documentation"),
        version=1,
        description=overrides.pop("description", "Project docs"),
        input_model=_MCPPassthroughInput,
        output_model=_MCPPassthroughOutput,
        handler=_noop_handler,
        opa_path=overrides.pop("opa_path", None),
        mcp_server_spec=server,
        mcp_input_schema={"type": "object", "properties": {}},
        mcp_resource_uri=overrides.pop("mcp_resource_uri", "mcp-demo://docs"),
    )


def test_mcp_resource_spec_is_a_mcp_tool_spec() -> None:
    """MCPResourceSpec subclasses MCPToolSpec → existing isinstance checks find it."""
    spec = _resource_spec_factory()
    assert isinstance(spec, MCPResourceSpec)
    assert isinstance(spec, MCPToolSpec)  # parent
    assert isinstance(spec, ToolSpec)  # grandparent


def test_resource_openai_schema_has_no_parameters() -> None:
    """Resources take no args; openai_schema returns a parameter-less shape."""
    spec = _resource_spec_factory(name="docs", description="Project docs")

    schema = spec.openai_schema()

    assert schema == {
        "type": "function",
        "function": {
            "name": "docs",
            "description": "Project docs",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_resource_uri_is_stored_as_string() -> None:
    """mcp_resource_uri is plain str, not pydantic AnyUrl (matches MCPToolSpec field-type style)."""
    spec = _resource_spec_factory(mcp_resource_uri="mcp-demo://docs")
    assert isinstance(spec.mcp_resource_uri, str)
    assert spec.mcp_resource_uri == "mcp-demo://docs"


def test_resource_opa_path_propagates() -> None:
    """When the resolver passes server.opa_decision_path → spec.opa_path, it round-trips."""
    spec = _resource_spec_factory(opa_path="mcp.test-server.allow")
    assert spec.opa_path == "mcp.test-server.allow"


def test_resource_spec_is_frozen() -> None:
    """MCPResourceSpec inherits frozen=True from ToolSpec."""
    spec = _resource_spec_factory()
    with pytest.raises(FrozenInstanceError):
        spec.mcp_resource_uri = "different://uri"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# unwrap_mcp_tool_message tests (Phase 12)
# ---------------------------------------------------------------------------


def test_unwrap_returns_value_for_single_key_envelope() -> None:
    """{'value': X} → X."""
    assert unwrap_mcp_tool_message('{"value": "hello"}') == "hello"
    assert unwrap_mcp_tool_message('{"value": 42}') == 42
    assert unwrap_mcp_tool_message('{"value": [1, 2]}') == [1, 2]
    assert unwrap_mcp_tool_message('{"value": null}') is None


def test_unwrap_returns_parsed_dict_when_multiple_keys() -> None:
    """A dict with more than just 'value' is returned as-is (parsed)."""
    raw = '{"value": "x", "meta": "y"}'
    assert unwrap_mcp_tool_message(raw) == {"value": "x", "meta": "y"}


def test_unwrap_returns_parsed_dict_when_no_value_key() -> None:
    """A dict without a 'value' key is returned parsed."""
    raw = '{"foo": "bar"}'
    assert unwrap_mcp_tool_message(raw) == {"foo": "bar"}


def test_unwrap_returns_parsed_non_dict() -> None:
    """JSON list / scalar at the top level is returned parsed."""
    assert unwrap_mcp_tool_message('[1, 2, 3]') == [1, 2, 3]
    assert unwrap_mcp_tool_message('"plain"') == "plain"
    assert unwrap_mcp_tool_message('42') == 42


def test_unwrap_returns_raw_string_when_not_json() -> None:
    """Non-JSON string is returned verbatim."""
    assert unwrap_mcp_tool_message("not json") == "not json"
    assert unwrap_mcp_tool_message("") == ""


def test_unwrap_raises_typeerror_on_non_string() -> None:
    """Non-string input is rejected."""
    with pytest.raises(TypeError):
        unwrap_mcp_tool_message(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        unwrap_mcp_tool_message(None)  # type: ignore[arg-type]
