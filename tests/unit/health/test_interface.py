"""Tests for IHealthProbe / ProbeResult / HealthStatus."""
from __future__ import annotations

import dataclasses

import pytest

from ai_core.health import IHealthProbe, ProbeResult

pytestmark = pytest.mark.unit


def test_probe_result_is_frozen() -> None:
    result = ProbeResult(component="db", status="ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "down"  # type: ignore[misc]


def test_probe_result_default_detail() -> None:
    result = ProbeResult(component="db", status="ok")
    assert result.detail is None


def test_probe_result_with_detail() -> None:
    result = ProbeResult(component="db", status="degraded",
                         detail="slow response (1.8s)")
    assert result.detail == "slow response (1.8s)"


def test_ihealthprobe_is_abstract() -> None:
    with pytest.raises(TypeError):
        IHealthProbe()  # type: ignore[abstract]
