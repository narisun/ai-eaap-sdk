"""Postgres-backed :class:`ICheckpointSaver` implementation.

This saver persists *SDK-level* run checkpoints — opaque payloads keyed
by ``(thread_id, checkpoint_id)``. It is independent of LangGraph's
internal :class:`BaseCheckpointSaver`; an adapter that bridges the two
is intentionally deferred to a later step so that the SDK contract
remains stable across LangGraph versions.

Concurrency:
    All operations open a fresh ``AsyncSession`` and use ``session.begin()``
    for transactional safety. The underlying engine pool handles
    concurrency, so the saver itself is stateless and safe to share.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from injector import inject
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ai_core.di.interfaces import ICheckpointSaver
from ai_core.exceptions import CheckpointError
from ai_core.persistence.models import CheckpointRecord


class PostgresCheckpointSaver(ICheckpointSaver):
    """Persist checkpoints in Postgres via SQLAlchemy 2.0 async.

    Args:
        engine: Async engine bound by the DI container.
    """

    @inject
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    async def save(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Upsert a checkpoint identified by ``(thread_id, checkpoint_id)``.

        Args:
            thread_id: Conversation/run identifier.
            checkpoint_id: Monotonic per-thread identifier.
            payload: Arbitrary JSON-serialisable mapping.

        Raises:
            CheckpointError: If the database operation fails.
        """
        try:
            async with self._sessionmaker() as session, session.begin():
                existing = await session.get(
                    CheckpointRecord, {"thread_id": thread_id, "checkpoint_id": checkpoint_id}
                )
                if existing is None:
                    session.add(
                        CheckpointRecord(
                            thread_id=thread_id,
                            checkpoint_id=checkpoint_id,
                            payload=dict(payload),
                        )
                    )
                else:
                    existing.payload = dict(payload)
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to persist checkpoint",
                details={"thread_id": thread_id, "checkpoint_id": checkpoint_id},
                cause=exc,
            ) from exc

    async def load(
        self,
        *,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        """Return the requested checkpoint payload, or the latest for the thread.

        Args:
            thread_id: Conversation/run identifier.
            checkpoint_id: Optional specific checkpoint id; if ``None``, the
                most recently created checkpoint for the thread is returned.

        Returns:
            The payload mapping, or ``None`` if no matching checkpoint exists.

        Raises:
            CheckpointError: If the database query fails.
        """
        try:
            async with self._sessionmaker() as session:
                stmt = select(CheckpointRecord).where(CheckpointRecord.thread_id == thread_id)
                if checkpoint_id is not None:
                    stmt = stmt.where(CheckpointRecord.checkpoint_id == checkpoint_id)
                else:
                    stmt = stmt.order_by(CheckpointRecord.created_at.desc()).limit(1)
                row: CheckpointRecord | None = (await session.execute(stmt)).scalars().first()
                return None if row is None else dict(row.payload)
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to load checkpoint",
                details={"thread_id": thread_id, "checkpoint_id": checkpoint_id},
                cause=exc,
            ) from exc

    async def list(self, *, thread_id: str, limit: int = 10) -> Sequence[str]:
        """Return up to ``limit`` checkpoint ids for ``thread_id``, newest first.

        Args:
            thread_id: Conversation/run identifier.
            limit: Maximum number of ids to return.

        Returns:
            A list of checkpoint ids ordered by ``created_at`` descending.

        Raises:
            CheckpointError: If the database query fails.
        """
        if limit <= 0:
            return []
        try:
            async with self._sessionmaker() as session:
                stmt = (
                    select(CheckpointRecord.checkpoint_id)
                    .where(CheckpointRecord.thread_id == thread_id)
                    .order_by(CheckpointRecord.created_at.desc())
                    .limit(limit)
                )
                return list((await session.execute(stmt)).scalars().all())
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to list checkpoints",
                details={"thread_id": thread_id, "limit": limit},
                cause=exc,
            ) from exc

    async def delete_thread(self, *, thread_id: str) -> int:
        """Delete every checkpoint for ``thread_id``.

        Args:
            thread_id: Conversation/run identifier.

        Returns:
            The number of rows deleted.

        Raises:
            CheckpointError: If the database operation fails.
        """
        try:
            async with self._sessionmaker() as session, session.begin():
                result = await session.execute(
                    delete(CheckpointRecord).where(CheckpointRecord.thread_id == thread_id)
                )
                return int(result.rowcount or 0)
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to delete checkpoints",
                details={"thread_id": thread_id},
                cause=exc,
            ) from exc


__all__ = ["PostgresCheckpointSaver"]
