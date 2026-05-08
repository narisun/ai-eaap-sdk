"""Unit tests for :func:`ai_core.schema.export.export_schemas`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from ai_core.schema import SchemaRegistry, export_schemas

pytestmark = pytest.mark.unit


class TicketIn(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = "low"


class TicketOut(BaseModel):
    ticket_id: str


class SearchIn(BaseModel):
    query: str


class SearchOut(BaseModel):
    hits: list[str]


@pytest.fixture
def populated() -> SchemaRegistry:
    reg = SchemaRegistry()
    reg.register("create_ticket", 1, input_schema=TicketIn, output_schema=TicketOut, description="v1")
    reg.register("create_ticket", 2, input_schema=TicketIn, output_schema=TicketOut)
    reg.register("search", 1, input_schema=SearchIn, output_schema=SearchOut)
    return reg


def test_writes_two_files_per_record(populated: SchemaRegistry, tmp_path: Path) -> None:
    written = export_schemas(populated, tmp_path)

    expected = {
        tmp_path / "create_ticket.v1.input.json",
        tmp_path / "create_ticket.v1.output.json",
        tmp_path / "create_ticket.v2.input.json",
        tmp_path / "create_ticket.v2.output.json",
        tmp_path / "search.v1.input.json",
        tmp_path / "search.v1.output.json",
    }
    assert set(written) == expected
    for path in expected:
        assert path.is_file()


def test_returns_paths_in_deterministic_order(
    populated: SchemaRegistry, tmp_path: Path
) -> None:
    written = export_schemas(populated, tmp_path)
    assert written == sorted(written, key=lambda p: p.name)


def test_emits_valid_json_with_provenance(
    populated: SchemaRegistry, tmp_path: Path
) -> None:
    export_schemas(populated, tmp_path)
    schema = json.loads((tmp_path / "create_ticket.v1.input.json").read_text())
    assert schema["x-eaap-name"] == "create_ticket"
    assert schema["x-eaap-version"] == 1
    assert schema["x-eaap-kind"] == "input"
    assert schema["x-eaap-description"] == "v1"
    # The Pydantic-emitted schema is preserved.
    assert "title" in schema["properties"]
    assert schema["properties"]["title"]["type"] == "string"
    assert schema["$id"] == "eaap://schema/create_ticket/v1/input"


def test_no_overwrite_skips_existing_files(
    populated: SchemaRegistry, tmp_path: Path
) -> None:
    target = tmp_path / "create_ticket.v1.input.json"
    target.write_text("OLD")
    written = export_schemas(populated, tmp_path, overwrite=False)
    assert target not in written
    assert target.read_text() == "OLD"


def test_creates_output_dir_if_missing(populated: SchemaRegistry, tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "out"
    written = export_schemas(populated, nested)
    assert nested.is_dir()
    assert len(written) == 6


def test_empty_registry_writes_nothing(tmp_path: Path) -> None:
    written = export_schemas(SchemaRegistry(), tmp_path)
    assert written == []
