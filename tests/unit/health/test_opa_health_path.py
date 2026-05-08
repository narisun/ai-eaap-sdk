"""Tests for the opa_health_path config wiring."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_core.config.settings import AppSettings, SecuritySettings
from ai_core.health.probes import OPAReachabilityProbe

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_opa_probe_uses_configured_health_path() -> None:
    """OPAReachabilityProbe constructs URL from settings.security.opa_health_path."""
    settings = AppSettings()
    settings.security = SecuritySettings(opa_health_path="/opa/health")
    probe = OPAReachabilityProbe(settings.security)

    fake_response = MagicMock(status_code=200)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()

    # Verify the URL contains the configured path.
    fake_client.get.assert_called_once()
    called_url = fake_client.get.call_args.args[0]
    assert called_url.endswith("/opa/health")
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_opa_probe_default_path_is_health() -> None:
    """When opa_health_path is unset, default is /health."""
    settings = AppSettings()
    probe = OPAReachabilityProbe(settings.security)

    fake_response = MagicMock(status_code=200)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        await probe.probe()

    called_url = fake_client.get.call_args.args[0]
    assert called_url.endswith("/health")


@pytest.mark.asyncio
async def test_opa_probe_path_without_leading_slash_normalised() -> None:
    """opa_health_path='health' (no leading slash) is normalised to '/health'."""
    settings = AppSettings()
    settings.security = SecuritySettings(opa_health_path="health")
    probe = OPAReachabilityProbe(settings.security)

    fake_response = MagicMock(status_code=200)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        await probe.probe()

    called_url = fake_client.get.call_args.args[0]
    # Should NOT contain "//" between base and path.
    assert "//health" not in called_url
    assert called_url.endswith("/health")
