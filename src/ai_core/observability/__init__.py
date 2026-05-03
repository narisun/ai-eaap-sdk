"""Observability sub-package.

Concrete OTel + LangFuse providers will land in a later step. For now
the SDK ships a :class:`NoOpObservabilityProvider` so that downstream
components (LLM client, agents) can call observability hooks
unconditionally without requiring a fully configured collector.
"""

from __future__ import annotations

from ai_core.observability.noop import NoOpObservabilityProvider

__all__ = ["NoOpObservabilityProvider"]
