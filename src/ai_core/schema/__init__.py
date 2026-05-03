"""Schema sub-package — versioned Pydantic schemas for tool I/O contracts."""

from __future__ import annotations

from ai_core.schema.registry import (
    SchemaRecord,
    SchemaRegistry,
    SchemaVersion,
)

__all__ = ["SchemaRegistry", "SchemaRecord", "SchemaVersion"]
