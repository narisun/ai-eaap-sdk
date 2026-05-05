"""Concrete health probes shipped with the SDK."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import litellm.utils
from sqlalchemy import text

from ai_core.health.interface import IHealthProbe, ProbeResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ai_core.config.settings import AppSettings

_HTTP_ERROR_THRESHOLD = 500


class OPAReachabilityProbe(IHealthProbe):
    """Sends ``GET <opa_url>/health`` to verify OPA is reachable."""

    component = "opa"

    def __init__(self, settings: AppSettings) -> None:
        base = str(settings.security.opa_url).rstrip("/")
        path = settings.security.opa_health_path
        if not path.startswith("/"):
            path = "/" + path
        self._url = base + path
        self._timeout = settings.security.opa_request_timeout_seconds

    async def probe(self) -> ProbeResult:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._url)
            if response.status_code < _HTTP_ERROR_THRESHOLD:
                return ProbeResult(
                    component=self.component, status="ok",
                    detail=f"http_status={response.status_code}",
                )
            return ProbeResult(
                component=self.component, status="degraded",
                detail=f"http_status={response.status_code}",
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            return ProbeResult(
                component=self.component, status="down",
                detail=f"unreachable: {type(exc).__name__}",
            )
        except Exception as exc:  # probes never raise
            return ProbeResult(
                component=self.component, status="down",
                detail=f"error: {type(exc).__name__}",
            )


class DatabaseProbe(IHealthProbe):
    """Runs ``SELECT 1`` against the configured AsyncEngine."""

    component = "database"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def probe(self) -> ProbeResult:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return ProbeResult(component=self.component, status="ok")
        except Exception as exc:  # probes never raise
            return ProbeResult(
                component=self.component, status="down",
                detail=f"connect_failed: {type(exc).__name__}",
            )


class ModelLookupProbe(IHealthProbe):
    """Verifies ``litellm.utils.get_supported_openai_params`` resolves
    the configured default model."""

    component = "model_lookup"

    def __init__(self, settings: AppSettings) -> None:
        self._model = settings.llm.default_model

    async def probe(self) -> ProbeResult:
        try:
            params = await asyncio.to_thread(
                litellm.utils.get_supported_openai_params,  # type: ignore[attr-defined]
                self._model,
            )
            if params is None:
                return ProbeResult(
                    component=self.component, status="degraded",
                    detail=f"model {self._model!r} not recognized by litellm",
                )
            return ProbeResult(
                component=self.component, status="ok",
                detail=f"model={self._model}",
            )
        except Exception as exc:  # probes never raise
            return ProbeResult(
                component=self.component, status="down",
                detail=f"lookup_error: {type(exc).__name__}",
            )


__all__ = ["DatabaseProbe", "ModelLookupProbe", "OPAReachabilityProbe"]
