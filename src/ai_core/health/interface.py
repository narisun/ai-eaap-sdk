"""Health-probe abstraction.

A :class:`IHealthProbe` runs a cheap reachability check against one
subsystem and returns a structured :class:`ProbeResult`. The
``HealthCheckRunner`` (in :mod:`ai_core.app.runtime`) fans out probes
in parallel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

HealthStatus = Literal["ok", "degraded", "down"]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of a single probe.

    Attributes:
        component: Subsystem name, e.g. ``"database"``, ``"opa"``, ``"model_lookup"``.
        status: ``"ok"`` (reachable + responsive), ``"degraded"`` (responding
            but with warnings), ``"down"`` (unreachable / errored).
        detail: Optional human-readable detail (latency_ms, error_code,
            response message).
    """

    component: str
    status: HealthStatus
    detail: str | None = None


class IHealthProbe(ABC):
    """One probe runs one component reachability check."""

    component: str  # class-level — name used in HealthSnapshot.components

    @abstractmethod
    async def probe(self) -> ProbeResult:
        """Run the probe. Implementations MUST NOT raise — return a
        :class:`ProbeResult` with ``status="down"`` and ``detail`` explaining
        the failure instead.
        """


__all__ = ["HealthStatus", "IHealthProbe", "ProbeResult"]
