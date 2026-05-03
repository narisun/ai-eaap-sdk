"""SQLAlchemy ORM models persisted by the SDK.

The schema is intentionally minimal — host applications typically
extend :class:`Base` with their own domain tables. The SDK only owns
checkpoint storage; everything else is opt-in.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all SDK-owned tables.

    Host applications may import this base and add their own tables to
    keep migrations centralised, but they are not required to.
    """


class CheckpointRecord(Base):
    """Persisted LangGraph checkpoint.

    Attributes:
        thread_id: Logical conversation/run identifier.
        checkpoint_id: Per-thread checkpoint identifier (monotonic / ulid).
        payload: Opaque LangGraph checkpoint blob, serialised as JSONB.
        created_at: UTC timestamp of insert.
    """

    __tablename__ = "eaap_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("ix_eaap_checkpoints_thread_created", "thread_id", "created_at"),
    )


__all__ = ["Base", "CheckpointRecord"]
