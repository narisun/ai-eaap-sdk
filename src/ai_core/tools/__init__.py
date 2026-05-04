"""Tool authoring primitives — the @tool decorator, ToolSpec, and Tool Protocol."""

from __future__ import annotations

from ai_core.tools.decorator import tool
from ai_core.tools.spec import Tool, ToolHandler, ToolSpec

__all__ = ["Tool", "ToolHandler", "ToolSpec", "tool"]
