"""FastMCP-backed connection handlers.

FastMCP supports several transports — this module wraps the two most
common ones for enterprise deployments:

* **stdio** — for locally-spawned MCP servers (e.g. tools shipped as
  CLI binaries).
* **http** / **sse** — for remote MCP servers reached over HTTP. SSE is
  the FastMCP default; ``streamable-http`` is preferred when the server
  supports it.

The factory returns the FastMCP :class:`fastmcp.Client` as an async
context manager. Callers are expected to use ``async with`` so that
connections are properly closed.

Note:
    FastMCP imports are deferred to call time so that the SDK can be
    imported in environments where FastMCP is unavailable (e.g. unit
    tests that only exercise the Registry).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from ai_core.exceptions import MCPTransportError, RegistryError

MCPTransport = Literal["stdio", "http", "sse"]


@dataclass(slots=True, frozen=True)
class MCPServerSpec:
    """Connection spec for an MCP server.

    Attributes:
        component_id: Logical identifier registered in :class:`ComponentRegistry`.
        transport: One of ``"stdio"``, ``"http"``, ``"sse"``.
        target: For ``stdio`` — the executable path or shell command.
            For ``http`` / ``sse`` — the server URL.
        args: Extra positional CLI args (``stdio`` only).
        env: Extra environment variables for the spawned subprocess
            (``stdio`` only).
        headers: HTTP headers to send (``http`` / ``sse`` only).
        timeout_seconds: Per-call timeout enforced by the FastMCP client.
        opa_decision_path: When set, every MCP tool call from this server
            checks this OPA path through the standard ToolInvoker pipeline.
            ``None`` skips OPA enforcement (matches local tools without ``opa_path``).
    """

    component_id: str
    transport: MCPTransport
    target: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    opa_decision_path: str | None = None


class IMCPConnectionFactory(ABC):
    """Open FastMCP connections from an :class:`MCPServerSpec`."""

    @abstractmethod
    def open(self, spec: MCPServerSpec) -> AbstractAsyncContextManager[Any]:
        """Return an async context manager yielding a connected MCP client."""


class PoolingMCPConnectionFactory(IMCPConnectionFactory):
    """FastMCP-backed factory with optional per-spec connection pooling.

    When ``pool_enabled=True`` (default), connections are reused across calls
    until they exceed ``pool_idle_seconds`` of inactivity. When ``False``,
    each ``open()`` returns a fresh CM (pre-Phase-4 behaviour).

    Closed at app shutdown via :meth:`aclose` (called by
    ``Container._teardown_sdk_resources``).

    Raises:
        MCPTransportError: If FastMCP is not installed, transport-class import
            fails, or the connection itself fails.
        RegistryError: If ``spec.transport`` is not a supported value.
    """

    def __init__(self, *, pool_enabled: bool = True,
                 pool_idle_seconds: float = 300.0) -> None:
        from ai_core.mcp._pool import _MCPConnectionPool  # noqa: PLC0415
        self._pool_enabled = pool_enabled
        self._pool: _MCPConnectionPool | None = (
            _MCPConnectionPool(opener=self._open, idle_seconds=pool_idle_seconds)
            if pool_enabled else None
        )

    def open(self, spec: MCPServerSpec) -> AbstractAsyncContextManager[Any]:
        if self._pool is not None:
            return self._pool.acquire(spec)
        return self._open(spec)

    @asynccontextmanager
    async def _open(self, spec: MCPServerSpec) -> AsyncIterator[Any]:
        _transport_details = {
            "component_id": spec.component_id,
            "transport": spec.transport,
        }

        try:
            from fastmcp import Client  # local import keeps fastmcp optional at import time
        except ImportError as exc:
            raise MCPTransportError(
                "FastMCP is not installed; install with `pip install ai-core-sdk[mcp]`",
                details=_transport_details,
                cause=exc,
            ) from exc

        try:
            if spec.transport == "stdio":
                from fastmcp.client.transports import StdioTransport

                transport = StdioTransport(
                    command=spec.target,
                    args=list(spec.args),
                    env=dict(spec.env),
                )
            elif spec.transport == "sse":
                from fastmcp.client.transports import SSETransport

                transport = SSETransport(url=spec.target, headers=dict(spec.headers))
            elif spec.transport == "http":
                from fastmcp.client.transports import StreamableHttpTransport

                transport = StreamableHttpTransport(url=spec.target, headers=dict(spec.headers))
            else:  # pragma: no cover - exhaustive Literal check
                raise RegistryError(
                    f"Unsupported MCP transport {spec.transport!r}",
                    details=_transport_details,
                )
        except ImportError as exc:
            raise MCPTransportError(
                f"FastMCP transport class for {spec.transport!r} not found "
                "(version mismatch or partial install); "
                "upgrade with `pip install -U ai-core-sdk[mcp]`",
                details=_transport_details,
                cause=exc,
            ) from exc

        client = Client(transport, timeout=spec.timeout_seconds)
        try:
            async with client:
                yield client
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            # Preserve cancellation/exit semantics for the async-context-manager unwind.
            raise
        except Exception as exc:
            raise MCPTransportError(
                f"MCP transport '{spec.transport}' connection failed: {exc}",
                details=_transport_details,
                cause=exc,
            ) from exc

    async def aclose(self) -> None:
        """Close all pooled connections. Idempotent. No-op when pool disabled."""
        if self._pool is not None:
            await self._pool.aclose()


# Pre-1.0 alias — kept for downstream importers.
FastMCPConnectionFactory = PoolingMCPConnectionFactory

__all__ = [
    "MCPServerSpec",
    "MCPTransport",
    "IMCPConnectionFactory",
    "PoolingMCPConnectionFactory",
    "FastMCPConnectionFactory",  # alias
]
