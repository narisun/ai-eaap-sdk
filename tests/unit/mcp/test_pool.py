"""Tests for the internal MCP connection pool."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest

from ai_core.exceptions import MCPTransportError
from ai_core.mcp._pool import _MCPConnectionPool
from ai_core.mcp.transports import MCPServerSpec

pytestmark = pytest.mark.unit


class _FakeFastMCPClient:
    """Records open/close + use calls for assertion."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.uses = 0
        self.closed = False

    async def use(self) -> None:
        if self.closed:
            raise RuntimeError(f"{self.label} closed")
        self.uses += 1


def _make_opener() -> tuple[Any, list[_FakeFastMCPClient]]:
    """Return (opener_callable, opened_clients_list)."""
    opened: list[_FakeFastMCPClient] = []

    def _opener(spec: MCPServerSpec) -> Any:
        @asynccontextmanager
        async def _cm() -> AsyncIterator[_FakeFastMCPClient]:
            client = _FakeFastMCPClient(label=spec.component_id)
            opened.append(client)
            try:
                yield client
            finally:
                client.closed = True
        return _cm()

    return _opener, opened


def _spec(component_id: str = "s1") -> MCPServerSpec:
    return MCPServerSpec(
        component_id=component_id, transport="stdio", target="/usr/bin/echo",
    )


@pytest.mark.asyncio
async def test_pool_reuses_connection_for_same_spec() -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)

    async with pool.acquire(_spec()) as c1:
        await c1.use()
    async with pool.acquire(_spec()) as c2:
        await c2.use()

    assert len(opened) == 1, "second call should reuse the first connection"
    assert opened[0].uses == 2

    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_opens_separate_connections_for_different_specs() -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)

    async with pool.acquire(_spec("s1")):
        pass
    async with pool.acquire(_spec("s2")):
        pass

    assert len(opened) == 2
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_evicts_stale_connection_after_idle_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=0.05)

    async with pool.acquire(_spec()):
        pass

    # Wait beyond idle TTL.
    await asyncio.sleep(0.1)

    async with pool.acquire(_spec()):
        pass

    assert len(opened) == 2, "stale connection should have been reopened"
    assert opened[0].closed is True
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_evicts_connection_on_in_flight_error() -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)

    with pytest.raises(RuntimeError, match="boom"):
        async with pool.acquire(_spec()) as c:
            assert c is not None
            raise RuntimeError("boom")

    # Next acquire should open a fresh connection (the broken one was evicted).
    async with pool.acquire(_spec()):
        pass

    assert len(opened) == 2
    assert opened[0].closed is True
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_aclose_closes_all_connections() -> None:
    opener, opened = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)

    async with pool.acquire(_spec("s1")):
        pass
    async with pool.acquire(_spec("s2")):
        pass

    await pool.aclose()

    assert all(c.closed for c in opened)


@pytest.mark.asyncio
async def test_pool_acquire_after_aclose_raises_mcp_transport_error() -> None:
    opener, _ = _make_opener()
    pool = _MCPConnectionPool(opener=opener, idle_seconds=300.0)
    await pool.aclose()

    with pytest.raises(MCPTransportError) as exc:
        async with pool.acquire(_spec()):
            pass
    assert "closed" in exc.value.message.lower()
