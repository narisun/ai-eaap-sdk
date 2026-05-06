"""Postgres integration tests via Testcontainers.

Auto-skip when Docker is unavailable (see conftest.docker_available).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ai_core.health.probes import DatabaseProbe

if TYPE_CHECKING:
    from testcontainers.postgres import PostgresContainer


def _asyncpg_dsn(pg: PostgresContainer) -> str:
    """Convert Testcontainers' default psycopg2 DSN to asyncpg form."""
    url: str = pg.get_connection_url()
    return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")


@pytest.mark.asyncio
async def test_async_engine_connects_to_real_postgres(
    postgres_container: PostgresContainer,
) -> None:
    engine = create_async_engine(_asyncpg_dsn(postgres_container))
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_database_probe_returns_ok_against_real_postgres(
    postgres_container: PostgresContainer,
) -> None:
    engine = create_async_engine(_asyncpg_dsn(postgres_container))
    try:
        probe = DatabaseProbe(engine)
        result = await probe.probe()
        assert result.status == "ok"
        assert result.component == "database"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_database_probe_returns_down_when_postgres_unreachable() -> None:
    """No fixture needed — exercises probe's never-raise contract on bad DSN."""
    engine = create_async_engine(
        "postgresql+asyncpg://x:y@127.0.0.1:1/x", connect_args={"timeout": 1}
    )
    try:
        probe = DatabaseProbe(engine)
        result = await probe.probe()
        # Probe MUST return a result (not raise) even when the backend is unreachable.
        assert result.status in {"down", "error", "degraded"}
        assert result.component == "database"
    finally:
        await engine.dispose()
