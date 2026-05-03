"""SQLAlchemy ORM models persisted by the SDK.

The schema is intentionally minimal — host applications typically
extend :class:`Base` with their own domain tables. The SDK owns:

* :class:`CheckpointRecord` — opaque, SDK-level run checkpoints used
  by :class:`PostgresCheckpointSaver` for audit / replay / arbitrary
  state-keeping.
* :class:`LangGraphCheckpointRecord` — full LangGraph checkpoint
  payloads (serialised via :class:`JsonPlusSerializer`).
* :class:`LangGraphWriteRecord` — pending channel writes that
  accompany a :class:`LangGraphCheckpointRecord`.

JSON columns use :func:`sqlalchemy.JSON` with a PostgreSQL
``JSONB``-backed variant so production keeps the indexable JSONB type
while tests can run against SQLite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    Integer,
    LargeBinary,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on Postgres, plain JSON elsewhere — keeps SQLAlchemy schema portable.
_PORTABLE_JSON = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Declarative base for all SDK-owned tables."""


class CheckpointRecord(Base):
    """Persisted SDK-level checkpoint.

    Attributes:
        thread_id: Logical conversation/run identifier.
        checkpoint_id: Per-thread checkpoint identifier (monotonic / ulid).
        payload: Opaque JSON-serialisable payload supplied by the caller.
        created_at: UTC timestamp of insert.
    """

    __tablename__ = "eaap_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(_PORTABLE_JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("ix_eaap_checkpoints_thread_created", "thread_id", "created_at"),
    )


class LangGraphCheckpointRecord(Base):
    """Persisted LangGraph checkpoint (one row per (thread, ns, id)).

    Attributes:
        thread_id: LangGraph thread identifier (``configurable.thread_id``).
        checkpoint_ns: LangGraph namespace (``""`` for the root graph).
        checkpoint_id: Per-thread checkpoint identifier emitted by LangGraph.
        parent_checkpoint_id: Previous checkpoint id (``None`` for the first).
        checkpoint_type: Serialiser type tag returned by ``serde.dumps_typed``.
        checkpoint_blob: Serialised :class:`Checkpoint` blob.
        metadata_blob: JSON-serialised :class:`CheckpointMetadata`.
        created_at: UTC timestamp of insert (drives ``alist`` ordering).
    """

    __tablename__ = "eaap_langgraph_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(
        String(128), primary_key=True, default=""
    )
    checkpoint_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    checkpoint_type: Mapped[str] = mapped_column(String(64), nullable=False)
    checkpoint_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    metadata_blob: Mapped[dict[str, Any]] = mapped_column(_PORTABLE_JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index(
            "ix_eaap_langgraph_checkpoints_thread_ns_created",
            "thread_id",
            "checkpoint_ns",
            "created_at",
        ),
    )


class LangGraphWriteRecord(Base):
    """Persisted pending channel write attached to a LangGraph checkpoint.

    Attributes:
        thread_id: LangGraph thread identifier.
        checkpoint_ns: LangGraph namespace.
        checkpoint_id: Checkpoint these writes belong to.
        task_id: LangGraph task identifier emitting the write.
        idx: Per-task ordinal of this write (preserves original order).
        channel: Channel name receiving the write.
        type_: Serialiser type tag returned by ``serde.dumps_typed``.
        blob: Serialised value.
        task_path: Optional structural path of the task that emitted the write.
    """

    __tablename__ = "eaap_langgraph_writes"

    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(
        String(128), primary_key=True, default=""
    )
    checkpoint_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(String(256), nullable=False)
    # ``type`` is a SQL keyword on some dialects — rename the attribute, keep the column.
    type_: Mapped[str] = mapped_column("type", String(64), nullable=False)
    blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    task_path: Mapped[str] = mapped_column(String(256), nullable=False, default="")


__all__ = [
    "Base",
    "CheckpointRecord",
    "LangGraphCheckpointRecord",
    "LangGraphWriteRecord",
]
