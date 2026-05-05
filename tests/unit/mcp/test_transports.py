"""Tests for FastMCPConnectionFactory error wrapping (MCPTransportError)."""
from __future__ import annotations

import asyncio
import builtins
from typing import Any
from unittest.mock import patch

import pytest

from ai_core.exceptions import MCPTransportError
from ai_core.mcp.transports import FastMCPConnectionFactory, MCPServerSpec

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
