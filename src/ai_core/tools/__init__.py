"""Tool authoring primitives — the @tool decorator, ToolSpec, ToolInvoker."""

from __future__ import annotations

from ai_core.tools.decorator import tool
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.spec import Tool, ToolHandler, ToolSpec

__all__ = ["Tool", "ToolHandler", "ToolInvoker", "ToolSpec", "tool"]
