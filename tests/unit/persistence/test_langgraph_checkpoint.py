"""Unit tests for :class:`ai_core.persistence.LangGraphCheckpointSaver`.

Runs against an in-memory aiosqlite engine so the suite stays
hermetic. The schema in :mod:`ai_core.persistence.models` uses
``JSON().with_variant(JSONB(), "postgresql")`` so production keeps
JSONB while these tests get plain JSON-on-SQLite.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ai_core.persistence import LangGraphCheckpointSaver
from ai_core.persistence.models import Base

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def saver(engine: AsyncEngine) -> LangGraphCheckpointSaver:
    return LangGraphCheckpointSaver(engine)


def _make_checkpoint(checkpoint_id: str = "cp-1") -> Checkpoint:
    """Build a minimal but realistic :class:`Checkpoint` payload."""
    return {
        "v": 1,
        "ts": "2026-05-03T10:00:00.000000+00:00",
        "id": checkpoint_id,
        "channel_values": {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        },
        "channel_versions": {"messages": "v1"},
        "versions_seen": {},
    }


def _config(thread_id: str, *, checkpoint_id: str | None = None, ns: str = "") -> dict:
    inner: dict = {"thread_id": thread_id, "checkpoint_ns": ns}
    if checkpoint_id is not None:
        inner["checkpoint_id"] = checkpoint_id
    return {"configurable": inner}


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
async def test_aput_then_aget_round_trip(saver: LangGraphCheckpointSaver) -> None:
    cp = _make_checkpoint("cp-1")
    md: CheckpointMetadata = {"source": "input", "step": 0, "writes": {}, "parents": {}}  # type: ignore[typeddict-item]

    new_cfg = await saver.aput(_config("t-1"), cp, md, {})

    assert new_cfg["configurable"]["thread_id"] == "t-1"
    assert new_cfg["configurable"]["checkpoint_id"] == "cp-1"

    loaded = await saver.aget_tuple(_config("t-1", checkpoint_id="cp-1"))
    assert loaded is not None
    assert loaded.checkpoint["id"] == "cp-1"
    assert loaded.checkpoint["channel_values"]["messages"][0]["content"] == "hello"
    assert loaded.metadata == md
    assert loaded.parent_config is None
    assert loaded.pending_writes == []


async def test_aget_tuple_returns_latest_when_id_omitted(
    saver: LangGraphCheckpointSaver,
) -> None:
    await saver.aput(_config("t-1"), _make_checkpoint("cp-1"), {}, {})  # type: ignore[arg-type]
    await saver.aput(
        _config("t-1", checkpoint_id="cp-1"),
        _make_checkpoint("cp-2"),
        {},  # type: ignore[arg-type]
        {},
    )
    latest = await saver.aget_tuple(_config("t-1"))
    assert latest is not None
    assert latest.checkpoint["id"] == "cp-2"
    assert latest.parent_config is not None
    assert latest.parent_config["configurable"]["checkpoint_id"] == "cp-1"


async def test_aget_tuple_returns_none_for_unknown(
    saver: LangGraphCheckpointSaver,
) -> None:
    assert await saver.aget_tuple(_config("missing-thread")) is None
    await saver.aput(_config("t-1"), _make_checkpoint("cp-1"), {}, {})  # type: ignore[arg-type]
    assert (
        await saver.aget_tuple(_config("t-1", checkpoint_id="cp-zzz")) is None
    )


# ---------------------------------------------------------------------------
# alist
# ---------------------------------------------------------------------------
async def test_alist_yields_newest_first(saver: LangGraphCheckpointSaver) -> None:
    for i in (1, 2, 3):
        await saver.aput(
            _config("t-1", checkpoint_id=f"cp-{i - 1}" if i > 1 else None),
            _make_checkpoint(f"cp-{i}"),
            {},  # type: ignore[arg-type]
            {},
        )

    ids = [t.checkpoint["id"] async for t in saver.alist(_config("t-1"))]
    assert ids == ["cp-3", "cp-2", "cp-1"]


async def test_alist_respects_limit(saver: LangGraphCheckpointSaver) -> None:
    for i in (1, 2, 3):
        await saver.aput(_config("t-1"), _make_checkpoint(f"cp-{i}"), {}, {})  # type: ignore[arg-type]
    ids = [t.checkpoint["id"] async for t in saver.alist(_config("t-1"), limit=2)]
    assert len(ids) == 2


async def test_alist_filter_matches_metadata(
    saver: LangGraphCheckpointSaver,
) -> None:
    await saver.aput(
        _config("t-1"),
        _make_checkpoint("cp-1"),
        {"source": "input", "step": 0, "writes": {}, "parents": {}},  # type: ignore[arg-type]
        {},
    )
    await saver.aput(
        _config("t-1", checkpoint_id="cp-1"),
        _make_checkpoint("cp-2"),
        {"source": "loop", "step": 1, "writes": {}, "parents": {}},  # type: ignore[arg-type]
        {},
    )
    ids = [
        t.checkpoint["id"]
        async for t in saver.alist(_config("t-1"), filter={"source": "loop"})
    ]
    assert ids == ["cp-2"]


# ---------------------------------------------------------------------------
# aput_writes / pending writes round-trip
# ---------------------------------------------------------------------------
async def test_pending_writes_round_trip(saver: LangGraphCheckpointSaver) -> None:
    await saver.aput(_config("t-1"), _make_checkpoint("cp-1"), {}, {})  # type: ignore[arg-type]

    cfg = _config("t-1", checkpoint_id="cp-1")
    await saver.aput_writes(
        cfg,
        [("messages", {"role": "tool", "content": "result"}), ("counter", 7)],
        task_id="task-A",
    )
    await saver.aput_writes(
        cfg,
        [("messages", {"role": "tool", "content": "other"})],
        task_id="task-B",
    )

    loaded = await saver.aget_tuple(cfg)
    assert loaded is not None
    # 3 pending writes total, ordered by (task_id, idx).
    assert len(loaded.pending_writes) == 3
    task_ids = [w[0] for w in loaded.pending_writes]
    channels = [w[1] for w in loaded.pending_writes]
    assert task_ids == ["task-A", "task-A", "task-B"]
    assert channels == ["messages", "counter", "messages"]
    assert loaded.pending_writes[0][2]["content"] == "result"
    assert loaded.pending_writes[1][2] == 7


async def test_aput_writes_idempotent_on_same_task_idx(
    saver: LangGraphCheckpointSaver,
) -> None:
    await saver.aput(_config("t-1"), _make_checkpoint("cp-1"), {}, {})  # type: ignore[arg-type]
    cfg = _config("t-1", checkpoint_id="cp-1")
    await saver.aput_writes(cfg, [("c", "first")], task_id="task-A")
    await saver.aput_writes(cfg, [("c", "second")], task_id="task-A")  # overwrites idx=0
    loaded = await saver.aget_tuple(cfg)
    assert loaded is not None
    assert loaded.pending_writes == [("task-A", "c", "second")]


# ---------------------------------------------------------------------------
# adelete_thread
# ---------------------------------------------------------------------------
async def test_adelete_thread_removes_checkpoints_and_writes(
    saver: LangGraphCheckpointSaver,
) -> None:
    await saver.aput(_config("t-1"), _make_checkpoint("cp-1"), {}, {})  # type: ignore[arg-type]
    await saver.aput_writes(
        _config("t-1", checkpoint_id="cp-1"),
        [("c", 1)],
        task_id="task-A",
    )
    await saver.aput(_config("t-2"), _make_checkpoint("cp-1"), {}, {})  # type: ignore[arg-type]

    await saver.adelete_thread("t-1")

    assert await saver.aget_tuple(_config("t-1")) is None
    # other thread untouched
    other = await saver.aget_tuple(_config("t-2"))
    assert other is not None
