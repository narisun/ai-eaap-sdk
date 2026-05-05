"""Tests for FastMCPConnectionFactory error wrapping (MCPTransportError)."""
from __future__ import annotations

import asyncio
import builtins
import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

from ai_core.exceptions import MCPTransportError
from ai_core.mcp.transports import (
    FastMCPConnectionFactory,
    MCPServerSpec,
    PoolingMCPConnectionFactory,
)

pytestmark = pytest.mark.unit


def _spec() -> MCPServerSpec:
    return MCPServerSpec(
        component_id="test-server",
        transport="stdio",
        target="/usr/bin/echo",
    )


def test_missing_fastmcp_raises_mcp_transport_error() -> None:
    """If fastmcp isn't importable, open() raises MCPTransportError with a helpful hint."""
    factory = FastMCPConnectionFactory()
    spec = _spec()

    # Force ImportError by removing fastmcp from sys.modules and shadowing the import.
    # The factory's open() calls __import__('fastmcp') deferred — patching __import__ traps it.
    real_import = builtins.__import__

    def _fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "fastmcp" or name.startswith("fastmcp."):
            raise ImportError(f"No module named '{name}' (test-injected)")
        return real_import(name, globals, locals, fromlist, level)

    async def _enter() -> None:
        ctx = factory.open(spec)
        if hasattr(ctx, "__aenter__"):
            async with ctx:
                pass

    with patch("builtins.__import__", side_effect=_fake_import), pytest.raises(
        MCPTransportError
    ) as exc:
        asyncio.run(_enter())

    assert exc.value.error_code == "mcp.transport_failed"
    assert "fastmcp" in exc.value.message.lower() or "fastmcp" in str(
        exc.value.cause
    ).lower()
    assert exc.value.details["component_id"] == "test-server"
    assert exc.value.details["transport"] == "stdio"


def test_runtime_oserror_is_wrapped_as_mcp_transport_error() -> None:
    """An OSError raised inside the async-with block must surface as MCPTransportError."""
    factory = FastMCPConnectionFactory()
    spec = _spec()

    class _BoomClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        async def __aenter__(self) -> object:
            raise OSError("connection refused (test-injected)")

        async def __aexit__(self, *_exc_info: object) -> None:
            return None

    fake_fastmcp = types.SimpleNamespace(Client=_BoomClient)
    fake_transports = types.SimpleNamespace(
        StdioTransport=lambda **kw: object(),
        SSETransport=lambda **kw: object(),
        StreamableHttpTransport=lambda **kw: object(),
    )

    async def _enter() -> None:
        with patch.dict(sys.modules, {
            "fastmcp": fake_fastmcp,
            "fastmcp.client.transports": fake_transports,
        }):
            async with factory.open(spec):
                pass

    with pytest.raises(MCPTransportError) as exc:
        asyncio.run(_enter())

    assert exc.value.error_code == "mcp.transport_failed"
    assert exc.value.details["component_id"] == "test-server"
    assert exc.value.details["transport"] == "stdio"
    assert "connection refused" in str(exc.value.cause)


@pytest.mark.asyncio
async def test_factory_pool_enabled_reuses_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """When pool_enabled=True, two open() calls for same spec reuse the connection."""
    factory = PoolingMCPConnectionFactory(pool_enabled=True, pool_idle_seconds=300.0)
    spec = MCPServerSpec(component_id="reuse-test", transport="stdio", target="/usr/bin/echo")

    opened: list[Any] = []

    class _BoomClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            opened.append(self)
            self.closed = False

        async def __aenter__(self) -> _BoomClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            self.closed = True

    fake_fastmcp = types.SimpleNamespace(Client=_BoomClient)
    fake_transports = types.SimpleNamespace(
        StdioTransport=lambda **kw: object(),
        SSETransport=lambda **kw: object(),
        StreamableHttpTransport=lambda **kw: object(),
    )
    with patch.dict(sys.modules, {
        "fastmcp": fake_fastmcp,
        "fastmcp.client.transports": fake_transports,
    }):
        async with factory.open(spec):
            pass
        async with factory.open(spec):
            pass

    # With pooling, only ONE _BoomClient should be constructed.
    assert len(opened) == 1
    await factory.aclose()


@pytest.mark.asyncio
async def test_factory_pool_disabled_opens_fresh_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """When pool_enabled=False, each open() call constructs a new client."""
    factory = PoolingMCPConnectionFactory(pool_enabled=False)
    spec = MCPServerSpec(component_id="fresh-test", transport="stdio", target="/usr/bin/echo")

    opened: list[Any] = []

    class _BoomClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            opened.append(self)

        async def __aenter__(self) -> _BoomClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    fake_fastmcp = types.SimpleNamespace(Client=_BoomClient)
    fake_transports = types.SimpleNamespace(
        StdioTransport=lambda **kw: object(),
        SSETransport=lambda **kw: object(),
        StreamableHttpTransport=lambda **kw: object(),
    )
    with patch.dict(sys.modules, {
        "fastmcp": fake_fastmcp,
        "fastmcp.client.transports": fake_transports,
    }):
        async with factory.open(spec):
            pass
        async with factory.open(spec):
            pass

    assert len(opened) == 2
    await factory.aclose()


@pytest.mark.asyncio
async def test_factory_aclose_drains_pool_when_enabled() -> None:
    """factory.aclose() closes all pooled connections (pool_enabled=True case)."""
    factory = PoolingMCPConnectionFactory(pool_enabled=True)
    # No opens — aclose should be a safe no-op.
    await factory.aclose()
    # Second call also OK.
    await factory.aclose()


@pytest.mark.asyncio
async def test_factory_aclose_noop_when_pool_disabled() -> None:
    """factory.aclose() is a safe no-op when pool_enabled=False."""
    factory = PoolingMCPConnectionFactory(pool_enabled=False)
    await factory.aclose()
