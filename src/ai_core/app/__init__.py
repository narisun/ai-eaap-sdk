"""Application-level facade — single-import entry point for AI engineers."""

from __future__ import annotations

from ai_core.app.runtime import AICoreApp, HealthSnapshot

__all__ = ["AICoreApp", "HealthSnapshot"]
