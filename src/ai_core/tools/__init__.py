"""Tool authoring primitives — the @tool decorator and runtime invoker.

Phase 1 — see docs/superpowers/specs/2026-05-04-ai-core-sdk-phase-1-design.md.
"""

from __future__ import annotations

from ai_core.tools.spec import Tool, ToolHandler, ToolSpec

__all__ = ["Tool", "ToolHandler", "ToolSpec"]
