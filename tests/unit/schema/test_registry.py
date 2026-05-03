"""Unit tests for :class:`ai_core.schema.registry.SchemaRegistry`."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from ai_core.exceptions import SchemaValidationError
from ai_core.schema import SchemaRegistry


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CreateTicketInV1(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = "low"


class CreateTicketOutV1(BaseModel):
    ticket_id: str


class CreateTicketInV2(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = "low"
    tenant_id: str


class CreateTicketOutV2(BaseModel):
    ticket_id: str
    href: str


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_register_and_get_round_trip() -> None:
    reg = SchemaRegistry()
    rec = reg.register(
        "create_ticket",
        1,
        input_schema=CreateTicketInV1,
        output_schema=CreateTicketOutV1,
        description="v1",
    )
    assert rec.name == "create_ticket"
    assert rec.version == 1
    assert reg.get("create_ticket", 1) is rec
    assert reg.get("create_ticket") is rec  # latest defaults
    assert ("create_ticket", 1) in reg
    assert reg.versions("create_ticket") == [1]


def test_latest_version_picks_highest() -> None:
    reg = SchemaRegistry()
    reg.register("x", 1, input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1)
    reg.register("x", 3, input_schema=CreateTicketInV2, output_schema=CreateTicketOutV2)
    reg.register("x", 2, input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1)
    assert reg.latest_version("x") == 3
    assert reg.get("x").version == 3


def test_duplicate_registration_raises_unless_replace() -> None:
    reg = SchemaRegistry()
    reg.register("x", 1, input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1)
    with pytest.raises(SchemaValidationError):
        reg.register("x", 1, input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1)
    reg.register(
        "x", 1, input_schema=CreateTicketInV2, output_schema=CreateTicketOutV2, replace=True
    )
    assert reg.get("x", 1).input_schema is CreateTicketInV2


def test_invalid_arguments_rejected() -> None:
    reg = SchemaRegistry()
    with pytest.raises(SchemaValidationError):
        reg.register("", 1, input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1)
    with pytest.raises(SchemaValidationError):
        reg.register("x", 0, input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1)
    with pytest.raises(SchemaValidationError):
        reg.register(
            "x", 1, input_schema=int,  # type: ignore[arg-type]
            output_schema=CreateTicketOutV1,
        )


def test_get_missing_raises() -> None:
    reg = SchemaRegistry()
    with pytest.raises(SchemaValidationError):
        reg.get("nope")
    with pytest.raises(SchemaValidationError):
        reg.get("nope", 1)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------
def test_validate_tool_sync_parses_input_validates_output() -> None:
    reg = SchemaRegistry()
    reg.register(
        "create_ticket", 1,
        input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1,
    )

    @reg.validate_tool("create_ticket")
    def create_ticket(payload: CreateTicketInV1) -> CreateTicketOutV1:
        assert isinstance(payload, CreateTicketInV1)
        return CreateTicketOutV1(ticket_id=f"T-{payload.title}")

    result = create_ticket({"title": "outage", "priority": "high"})
    assert isinstance(result, CreateTicketOutV1)
    assert result.ticket_id == "T-outage"


async def test_validate_tool_async_parses_input_validates_output() -> None:
    reg = SchemaRegistry()
    reg.register(
        "create_ticket", 1,
        input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1,
    )

    @reg.validate_tool("create_ticket")
    async def create_ticket(payload: CreateTicketInV1) -> CreateTicketOutV1:
        return CreateTicketOutV1(ticket_id=f"T-{payload.title}")

    result = await create_ticket({"title": "outage"})
    assert result.ticket_id == "T-outage"


async def test_validate_tool_rejects_bad_input() -> None:
    reg = SchemaRegistry()
    reg.register(
        "create_ticket", 1,
        input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1,
    )

    @reg.validate_tool("create_ticket")
    async def create_ticket(payload: CreateTicketInV1) -> CreateTicketOutV1:
        return CreateTicketOutV1(ticket_id="T")

    with pytest.raises(SchemaValidationError) as ei:
        await create_ticket({"title": ""})  # min_length violated
    assert ei.value.details["name"] == "create_ticket"
    assert ei.value.details["version"] == 1


async def test_validate_tool_rejects_bad_output() -> None:
    reg = SchemaRegistry()
    reg.register(
        "create_ticket", 1,
        input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1,
    )

    @reg.validate_tool("create_ticket")
    async def create_ticket(payload: CreateTicketInV1) -> CreateTicketOutV1:
        return {"wrong_field": True}  # type: ignore[return-value]

    with pytest.raises(SchemaValidationError) as ei:
        await create_ticket({"title": "x"})
    assert ei.value.details["name"] == "create_ticket"


def test_validate_tool_unknown_schema_raises_at_decoration_time() -> None:
    reg = SchemaRegistry()
    with pytest.raises(SchemaValidationError):
        @reg.validate_tool("nonexistent")
        def _tool(payload: CreateTicketInV1) -> CreateTicketOutV1:  # pragma: no cover
            return CreateTicketOutV1(ticket_id="T")


def test_validate_tool_passes_through_already_typed_inputs() -> None:
    reg = SchemaRegistry()
    reg.register(
        "create_ticket", 1,
        input_schema=CreateTicketInV1, output_schema=CreateTicketOutV1,
    )

    @reg.validate_tool("create_ticket")
    def create_ticket(payload: CreateTicketInV1) -> CreateTicketOutV1:
        return CreateTicketOutV1(ticket_id="T-already-typed")

    result = create_ticket(CreateTicketInV1(title="hi"))
    assert result.ticket_id == "T-already-typed"
