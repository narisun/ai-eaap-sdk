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
from typing import Any, Literal

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
    """

    component_id: str
    transport: MCPTransport
    target: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0


class IMCPConnectionFactory(ABC):
    """Open FastMCP connections from an :class:`MCPServerSpec`."""

    @abstractmethod
    def open(self, spec: MCPServerSpec) -> AbstractAsyncContextManager[Any]:
        """Return an async context manager yielding a connected MCP client."""


class FastMCPConnectionFactory(IMCPConnectionFactory):
    """Default :class:`IMCPConnectionFactory` powered by FastMCP transports."""

    def open(self, spec: MCPServerSpec) -> AbstractAsyncContextManager[Any]:
        """Open a FastMCP client for ``spec``.

        Args:
            spec: Server connection specification.

        Returns:
            Async context manager yielding a connected FastMCP ``Client``.

        Raises:
            RegistryError: If the transport is unsupported or FastMCP is missing.
        """
        return self._open(spec)

    @asynccontextmanager
    async def _open(self, spec: MCPServerSpec) -> Any:
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
                "FastMCP is not installed; install with `pip install ai-core-sdk[mcp]`",
                details=_transport_details,
                cause=exc,
            ) from exc

        client = Client(transport, timeout=spec.timeout_seconds)
        try:
            async with client:
                yield client
        except OSError as exc:
            raise MCPTransportError(
                f"MCP transport '{spec.transport}' connection failed: {exc}",
                details=_transport_details,
                cause=exc,
            ) from exc
        except Exception as exc:
            # Catch httpx / anyio runtime errors without importing those libraries.
            exc_type = type(exc).__name__
            if any(
                exc_type.endswith(suffix)
                for suffix in ("HTTPError", "BrokenResourceError", "ClosedResourceError")
            ):
                raise MCPTransportError(
                    f"MCP transport '{spec.transport}' connection failed: {exc}",
                    details=_transport_details,
                    cause=exc,
                ) from exc
            raise


__all__ = [
    "MCPServerSpec",
    "MCPTransport",
    "IMCPConnectionFactory",
    "FastMCPConnectionFactory",
]
