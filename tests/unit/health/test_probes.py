"""Tests for the four shipped health probes."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_core.config.settings import AppSettings
from ai_core.health.probes import (
    DatabaseProbe,
    ModelLookupProbe,
    OPAReachabilityProbe,
    SettingsProbe,
)

pytestmark = pytest.mark.unit


# --- SettingsProbe ---

@pytest.mark.asyncio
async def test_settings_probe_always_ok() -> None:
    probe = SettingsProbe(AppSettings())
    result = await probe.probe()
    assert result.component == "settings"
    assert result.status == "ok"


# --- OPAReachabilityProbe ---

@pytest.mark.asyncio
async def test_opa_probe_ok_on_2xx() -> None:
    probe = OPAReachabilityProbe(AppSettings())
    fake_response = MagicMock(status_code=200)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()
    assert result.status == "ok"
    assert "200" in (result.detail or "")


@pytest.mark.asyncio
async def test_opa_probe_degraded_on_5xx() -> None:
    probe = OPAReachabilityProbe(AppSettings())
    fake_response = MagicMock(status_code=503)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()
    assert result.status == "degraded"


@pytest.mark.asyncio
async def test_opa_probe_down_on_connect_error() -> None:
    probe = OPAReachabilityProbe(AppSettings())
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.side_effect = httpx.ConnectError("refused")
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()
    assert result.status == "down"


@pytest.mark.asyncio
async def test_opa_probe_never_raises_on_unexpected_error() -> None:
    """Even on a non-httpx exception, the probe returns down rather than raising."""
    probe = OPAReachabilityProbe(AppSettings())
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.side_effect = RuntimeError("unexpected")
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()
    assert result.status == "down"


# --- DatabaseProbe ---

@pytest.mark.asyncio
async def test_database_probe_ok() -> None:
    fake_conn = AsyncMock()
    fake_conn.execute = AsyncMock(return_value=None)
    fake_ctx = AsyncMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=fake_conn)
    fake_ctx.__aexit__ = AsyncMock(return_value=None)
    fake_engine = MagicMock()
    fake_engine.connect.return_value = fake_ctx

    probe = DatabaseProbe(fake_engine)
    result = await probe.probe()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_database_probe_down_on_connect_failure() -> None:
    fake_engine = AsyncMock()
    fake_engine.connect.side_effect = ConnectionError("db unreachable")

    probe = DatabaseProbe(fake_engine)
    result = await probe.probe()
    assert result.status == "down"
    assert "connect_failed" in (result.detail or "")


# --- ModelLookupProbe ---

@pytest.mark.asyncio
async def test_model_lookup_probe_ok_for_known_model() -> None:
    """litellm.utils.get_supported_openai_params returns a list for known models."""
    settings = AppSettings()
    settings.llm.default_model = "gpt-4o-mini"
    probe = ModelLookupProbe(settings)
    with patch(
        "litellm.utils.get_supported_openai_params",
        return_value=["max_tokens", "temperature"],
    ):
        result = await probe.probe()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_model_lookup_probe_degraded_when_unknown() -> None:
    settings = AppSettings()
    probe = ModelLookupProbe(settings)
    with patch("litellm.utils.get_supported_openai_params", return_value=None):
        result = await probe.probe()
    assert result.status == "degraded"


@pytest.mark.asyncio
async def test_model_lookup_probe_down_on_lookup_error() -> None:
    settings = AppSettings()
    probe = ModelLookupProbe(settings)
    with patch(
        "litellm.utils.get_supported_openai_params",
        side_effect=RuntimeError("bad model"),
    ):
        result = await probe.probe()
    assert result.status == "down"
