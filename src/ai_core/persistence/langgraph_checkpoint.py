"""LangGraph-native checkpoint saver.

This adapter satisfies LangGraph's :class:`BaseCheckpointSaver` contract
on top of the SDK's :class:`AsyncEngine`, so a compiled graph can use
the same Postgres database the SDK already manages::

    saver = container.get(LangGraphCheckpointSaver)
    graph = my_agent.compile(checkpointer=saver)

Schema
------
Two tables — see :mod:`ai_core.persistence.models`:

* ``eaap_langgraph_checkpoints`` — one row per
  ``(thread_id, checkpoint_ns, checkpoint_id)``; stores the serialised
  Checkpoint blob plus its metadata.
* ``eaap_langgraph_writes`` — one row per pending channel write
  attached to a checkpoint, indexed by ``(task_id, idx)`` for
  deterministic replay.

Concurrency
-----------
Each operation opens a fresh ``AsyncSession`` and uses
``session.begin()`` for transactional safety. The class is stateless
beyond the engine / session-maker handle and is safe to share across
coroutines.

Sync interface
--------------
The class deliberately leaves the inherited sync methods (``put``,
``get_tuple``, ``list``, ``put_writes``) raising ``NotImplementedError``
— ``BaseAgent`` and the rest of the SDK are async-first, and bridging
to a sync API would require either a blocking driver or an event-loop
hack. Callers that need the sync API should use LangGraph's
in-memory ``MemorySaver`` for tests or run the async API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from injector import inject
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ai_core.exceptions import CheckpointError
from ai_core.persistence.models import (
    LangGraphCheckpointRecord,
    LangGraphWriteRecord,
)


class LangGraphCheckpointSaver(BaseCheckpointSaver[str]):
    """Async-only LangGraph checkpoint saver backed by SQLAlchemy.

    Args:
        engine: Async SQLAlchemy engine bound by the DI container.

    Note:
        Tables must already exist. In production they're created by
        Alembic migrations; in tests, call
        ``Base.metadata.create_all`` against the engine before use.
    """

    @inject
    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(serde=JsonPlusSerializer())
        self._engine = engine
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    # ------------------------------------------------------------------
    # aput
    # ------------------------------------------------------------------
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a single checkpoint.

        Args:
            config: ``RunnableConfig``; must carry ``configurable.thread_id``.
            checkpoint: The :class:`Checkpoint` payload from LangGraph.
            metadata: The :class:`CheckpointMetadata` dict.
            new_versions: Channel-version map (unused — versions are
                embedded in the checkpoint blob).

        Returns:
            A new ``RunnableConfig`` carrying the freshly-written
            ``checkpoint_id`` so subsequent calls can chain.

        Raises:
            CheckpointError: On database failure.
        """
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        new_checkpoint_id = str(checkpoint["id"])
        parent_id = configurable.get("checkpoint_id")
        parent_str = str(parent_id) if parent_id else None

        type_, blob = self.serde.dumps_typed(checkpoint)

        try:
            async with self._sessionmaker() as session, session.begin():
                pk = {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": new_checkpoint_id,
                }
                existing = await session.get(LangGraphCheckpointRecord, pk)
                if existing is None:
                    session.add(
                        LangGraphCheckpointRecord(
                            **pk,
                            parent_checkpoint_id=parent_str,
                            checkpoint_type=type_,
                            checkpoint_blob=blob,
                            metadata_blob=dict(metadata or {}),
                        )
                    )
                else:
                    existing.parent_checkpoint_id = parent_str
                    existing.checkpoint_type = type_
                    existing.checkpoint_blob = blob
                    existing.metadata_blob = dict(metadata or {})
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to persist LangGraph checkpoint",
                details={
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": new_checkpoint_id,
                },
                cause=exc,
            ) from exc

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": new_checkpoint_id,
            }
        }

    # ------------------------------------------------------------------
    # aget_tuple
    # ------------------------------------------------------------------
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return the requested (or latest) checkpoint as a ``CheckpointTuple``.

        Args:
            config: ``RunnableConfig`` with ``configurable.thread_id`` and
                optionally ``checkpoint_ns`` / ``checkpoint_id``.

        Returns:
            A :class:`CheckpointTuple` (with pending writes bundled) or
            ``None`` if no matching checkpoint exists.

        Raises:
            CheckpointError: On database failure.
        """
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        requested_id = configurable.get("checkpoint_id")
        requested_id_str = str(requested_id) if requested_id else None

        try:
            async with self._sessionmaker() as session:
                stmt = select(LangGraphCheckpointRecord).where(
                    LangGraphCheckpointRecord.thread_id == thread_id,
                    LangGraphCheckpointRecord.checkpoint_ns == checkpoint_ns,
                )
                if requested_id_str is not None:
                    stmt = stmt.where(
                        LangGraphCheckpointRecord.checkpoint_id == requested_id_str
                    )
                else:
                    stmt = stmt.order_by(
                        LangGraphCheckpointRecord.created_at.desc()
                    ).limit(1)
                row = (await session.execute(stmt)).scalars().first()
                if row is None:
                    return None

                writes_stmt = (
                    select(LangGraphWriteRecord)
                    .where(
                        LangGraphWriteRecord.thread_id == thread_id,
                        LangGraphWriteRecord.checkpoint_ns == checkpoint_ns,
                        LangGraphWriteRecord.checkpoint_id == row.checkpoint_id,
                    )
                    .order_by(
                        LangGraphWriteRecord.task_id,
                        LangGraphWriteRecord.idx,
                    )
                )
                write_rows = list((await session.execute(writes_stmt)).scalars().all())
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to load LangGraph checkpoint",
                details={
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": requested_id_str,
                },
                cause=exc,
            ) from exc

        return self._row_to_tuple(row, write_rows)

    # ------------------------------------------------------------------
    # alist
    # ------------------------------------------------------------------
    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: Mapping[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Yield checkpoints for the supplied thread, newest first.

        Args:
            config: Required — must carry ``configurable.thread_id``.
            filter: Optional metadata key/value pairs (matched on the
                ``metadata_blob`` JSON column).
            before: Optional ``RunnableConfig`` whose ``checkpoint_id``
                acts as an upper bound — only checkpoints created
                strictly before that id are yielded.
            limit: Maximum number of tuples to yield.

        Yields:
            :class:`CheckpointTuple` values ordered by ``created_at`` desc.
        """
        if config is None or "configurable" not in config:
            return
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))

        before_id: str | None = None
        if before is not None:
            before_id = before.get("configurable", {}).get("checkpoint_id")
            before_id = str(before_id) if before_id else None

        try:
            async with self._sessionmaker() as session:
                stmt = select(LangGraphCheckpointRecord).where(
                    LangGraphCheckpointRecord.thread_id == thread_id,
                    LangGraphCheckpointRecord.checkpoint_ns == checkpoint_ns,
                )
                if before_id is not None:
                    cutoff = await session.execute(
                        select(LangGraphCheckpointRecord.created_at).where(
                            LangGraphCheckpointRecord.thread_id == thread_id,
                            LangGraphCheckpointRecord.checkpoint_ns == checkpoint_ns,
                            LangGraphCheckpointRecord.checkpoint_id == before_id,
                        )
                    )
                    cutoff_ts = cutoff.scalar_one_or_none()
                    if cutoff_ts is not None:
                        stmt = stmt.where(LangGraphCheckpointRecord.created_at < cutoff_ts)

                stmt = stmt.order_by(LangGraphCheckpointRecord.created_at.desc())
                if limit is not None:
                    stmt = stmt.limit(int(limit))

                rows = list((await session.execute(stmt)).scalars().all())

                for row in rows:
                    if filter and not _metadata_matches(row.metadata_blob, filter):
                        continue
                    writes_stmt = (
                        select(LangGraphWriteRecord)
                        .where(
                            LangGraphWriteRecord.thread_id == thread_id,
                            LangGraphWriteRecord.checkpoint_ns == checkpoint_ns,
                            LangGraphWriteRecord.checkpoint_id == row.checkpoint_id,
                        )
                        .order_by(
                            LangGraphWriteRecord.task_id,
                            LangGraphWriteRecord.idx,
                        )
                    )
                    write_rows = list(
                        (await session.execute(writes_stmt)).scalars().all()
                    )
                    yield self._row_to_tuple(row, write_rows)
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to list LangGraph checkpoints",
                details={"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # aput_writes
    # ------------------------------------------------------------------
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending channel writes that accompany a checkpoint.

        Args:
            config: Must carry ``configurable.thread_id`` and
                ``configurable.checkpoint_id``.
            writes: ``[(channel, value), ...]`` to persist in order.
            task_id: LangGraph task identifier producing the writes.
            task_path: Optional structural path of the task.

        Raises:
            CheckpointError: On database failure.
        """
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(configurable["checkpoint_id"])

        try:
            async with self._sessionmaker() as session, session.begin():
                for idx, (channel, value) in enumerate(writes):
                    type_, blob = self.serde.dumps_typed(value)
                    pk = {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "idx": idx,
                    }
                    existing = await session.get(LangGraphWriteRecord, pk)
                    if existing is None:
                        session.add(
                            LangGraphWriteRecord(
                                **pk,
                                channel=str(channel),
                                type_=type_,
                                blob=blob,
                                task_path=task_path,
                            )
                        )
                    else:
                        existing.channel = str(channel)
                        existing.type_ = type_
                        existing.blob = blob
                        existing.task_path = task_path
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to persist LangGraph pending writes",
                details={
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                },
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # adelete_thread (newer LangGraph API)
    # ------------------------------------------------------------------
    async def adelete_thread(self, thread_id: str) -> None:
        """Remove every checkpoint and pending write for ``thread_id``."""
        try:
            async with self._sessionmaker() as session, session.begin():
                await session.execute(
                    delete(LangGraphWriteRecord).where(
                        LangGraphWriteRecord.thread_id == thread_id
                    )
                )
                await session.execute(
                    delete(LangGraphCheckpointRecord).where(
                        LangGraphCheckpointRecord.thread_id == thread_id
                    )
                )
        except SQLAlchemyError as exc:
            raise CheckpointError(
                "Failed to delete LangGraph thread",
                details={"thread_id": thread_id},
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _row_to_tuple(
        self,
        row: LangGraphCheckpointRecord,
        write_rows: Sequence[LangGraphWriteRecord],
    ) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed((row.checkpoint_type, row.checkpoint_blob))
        metadata: CheckpointMetadata = dict(row.metadata_blob or {})  # type: ignore[assignment]
        config: RunnableConfig = {
            "configurable": {
                "thread_id": row.thread_id,
                "checkpoint_ns": row.checkpoint_ns,
                "checkpoint_id": row.checkpoint_id,
            }
        }
        parent_config: RunnableConfig | None = None
        if row.parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": row.thread_id,
                    "checkpoint_ns": row.checkpoint_ns,
                    "checkpoint_id": row.parent_checkpoint_id,
                }
            }
        pending_writes: list[tuple[str, str, Any]] = [
            (
                w.task_id,
                w.channel,
                self.serde.loads_typed((w.type_, w.blob)),
            )
            for w in write_rows
        ]
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )


def _metadata_matches(
    metadata: Mapping[str, Any] | None,
    filter_: Mapping[str, Any],
) -> bool:
    """Return ``True`` if every (k, v) in ``filter_`` matches ``metadata``."""
    if not metadata:
        return False
    for k, v in filter_.items():
        if metadata.get(k) != v:
            return False
    return True


__all__ = ["LangGraphCheckpointSaver"]
