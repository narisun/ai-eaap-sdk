"""Meta-test: every IHealthProbe implementation MUST return ProbeResult on failure."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from ai_core.config.settings import AppSettings
from ai_core.health.probes import (
    DatabaseProbe,
    ModelLookupProbe,
    OPAReachabilityProbe,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_opa_probe_never_raises_on_unexpected_error() -> None:
    probe = OPAReachabilityProbe(AppSettings())
    with patch("httpx.AsyncClient", side_effect=RuntimeError("totally unexpected")):
        result = await probe.probe()
    assert result.status == "down"  # not raised


@pytest.mark.asyncio
async def test_database_probe_never_raises_on_unexpected_error() -> None:
    """A bad engine that raises on connect() must produce ProbeResult(status='down')."""

    class _BadEngine:
        def connect(self, *_: object) -> Any:
            raise RuntimeError("totally unexpected")

    probe = DatabaseProbe(_BadEngine())  # type: ignore[arg-type]
    result = await probe.probe()
    assert result.status == "down"


@pytest.mark.asyncio
async def test_model_lookup_probe_never_raises_on_unexpected_error() -> None:
    probe = ModelLookupProbe(AppSettings())
    with patch(
        "litellm.utils.get_supported_openai_params",
        side_effect=RuntimeError("library bug"),
    ):
        result = await probe.probe()
    assert result.status == "down"
