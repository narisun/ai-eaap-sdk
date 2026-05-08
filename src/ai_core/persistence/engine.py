"""Async SQLAlchemy engine + session factory.

The engine is created lazily through DI. A factory wrapper is exposed
so that hosts can construct multiple engines (e.g. read-replica vs.
primary) by registering additional :class:`injector.Module` providers
without losing the default convenience binding.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ai_core.config.settings import DatabaseSettings


class EngineFactory:
    """Construct + cache a single :class:`AsyncEngine` from settings.

    Args:
        settings: The database configuration slice. Pass
            ``app_settings.database`` when constructing manually; the DI
            container injects the slice directly.
    """

    __slots__ = ("_engine", "_sessionmaker", "_settings")

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings: DatabaseSettings = settings
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    def engine(self) -> AsyncEngine:
        """Return the lazily-constructed :class:`AsyncEngine` singleton."""
        if self._engine is None:
            self._engine = create_async_engine(
                self._settings.dsn.get_secret_value(),
                pool_size=self._settings.pool_size,
                max_overflow=self._settings.max_overflow,
                pool_timeout=self._settings.pool_timeout_seconds,
                echo=self._settings.echo_sql,
                pool_pre_ping=True,
                future=True,
            )
        return self._engine

    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        """Return a cached ``async_sessionmaker`` bound to :py:meth:`engine`."""
        if self._sessionmaker is None:
            self._sessionmaker = async_sessionmaker(
                self.engine(),
                expire_on_commit=False,
                class_=AsyncSession,
            )
        return self._sessionmaker

    async def dispose(self) -> None:
        """Close the engine. Idempotent."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None


def async_engine_provider(settings: DatabaseSettings) -> AsyncEngine:
    """Convenience function used by the DI container to provide ``AsyncEngine``.

    Args:
        settings: The database settings slice, supplied by the container.

    Returns:
        A lazily-initialised :class:`AsyncEngine`.
    """
    return EngineFactory(settings).engine()


__all__ = ["EngineFactory", "async_engine_provider"]
