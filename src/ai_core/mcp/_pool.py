"""Internal MCP connection pool — single-connection-per-spec with idle TTL.

Not part of the SDK's public API. Consumers reach this via
:class:`PoolingMCPConnectionFactory` in :mod:`ai_core.mcp.transports`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_core.exceptions import MCPTransportError
from ai_core.observability.logging import get_logger

if TYPE_CHECKING:
    from ai_core.mcp.transports import MCPServerSpec

_logger = get_logger(__name__)

_OpenerFn = Callable[["MCPServerSpec"], AbstractAsyncContextManager[Any]]


@dataclass(slots=True)
class _PooledConnection:
    """Live FastMCP client + its enclosing context manager + last-used bookkeeping."""

    client: Any
    cm: AbstractAsyncContextManager[Any]
    spec: MCPServerSpec
    last_used: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _MCPConnectionPool:
    """Per-component_id single-connection pool.

    Concurrency model:
        Each component_id has at most one connection. Concurrent calls to the
        same component_id serialize on the connection's ``lock``. The
        ``connections`` dict is guarded by ``self._lock`` to avoid races
        when two callers race for a missing connection.

        Known v1 limitation: ``self._lock`` is held while a missing connection
        is opened (and while a stale connection is closed before reopen).
        During those awaited operations, acquires for *other* component_ids
        will block on ``self._lock`` even though they target unrelated
        connections. Acceptable for current single-tenant SDK consumption
        (low spec cardinality, infrequent opens once warm). Multi-component_id
        parallel opens are a Phase 5+ follow-up.

    Lifecycle:
        ``aclose()`` closes every pooled connection. Called by Container teardown.

    Idle TTL:
        On checkout, if ``time.monotonic() - last_used > idle_seconds``, the
        connection is torn down and reopened.
    """

    def __init__(
        self,
        *,
        opener: _OpenerFn,
        idle_seconds: float,
    ) -> None:
        self._opener = opener
        self._idle_seconds = idle_seconds
        self._connections: dict[str, _PooledConnection] = {}
        self._lock = asyncio.Lock()
        self._closed: bool = False

    @asynccontextmanager
    async def acquire(self, spec: MCPServerSpec) -> AsyncIterator[Any]:
        """Yield a live FastMCP client for ``spec``. Serialised per spec."""
        if self._closed:
            raise MCPTransportError(
                "MCP connection pool is closed",
                details={"component_id": spec.component_id, "transport": spec.transport},
            )

        async with self._lock:
            entry = self._connections.get(spec.component_id)
            if entry is None or self._is_stale(entry):
                if entry is not None:
                    await self._close_one(entry)
                entry = await self._open_entry(spec)
                self._connections[spec.component_id] = entry

        async with entry.lock:
            try:
                yield entry.client
                entry.last_used = time.monotonic()
            except Exception:
                async with self._lock:
                    if self._connections.get(spec.component_id) is entry:
                        del self._connections[spec.component_id]
                await self._close_one(entry)
                raise

    def _is_stale(self, entry: _PooledConnection) -> bool:
        return (time.monotonic() - entry.last_used) > self._idle_seconds

    async def _open_entry(self, spec: MCPServerSpec) -> _PooledConnection:
        cm = self._opener(spec)
        try:
            client = await cm.__aenter__()
        except Exception as exc:
            raise MCPTransportError(
                f"MCP transport '{spec.transport}' connection failed: {exc}",
                details={"component_id": spec.component_id, "transport": spec.transport},
                cause=exc,
            ) from exc
        return _PooledConnection(
            client=client, cm=cm, spec=spec, last_used=time.monotonic(),
        )

    async def _close_one(self, entry: _PooledConnection) -> None:
        try:
            await entry.cm.__aexit__(None, None, None)
        except Exception as exc:
            _logger.warning(
                "mcp.pool.connection_close_failed",
                component_id=entry.spec.component_id,
                transport=entry.spec.transport,
                error=str(exc), error_type=type(exc).__name__,
            )

    async def aclose(self) -> None:
        """Close every pooled connection. Idempotent."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._connections.values())
            self._connections.clear()
        for entry in entries:
            await self._close_one(entry)


__all__ = ["_MCPConnectionPool"]
