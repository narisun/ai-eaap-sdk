"""Tests for the structlog logging seam."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from structlog.testing import capture_logs

from ai_core.observability.logging import (
    bind_context,
    configure,
    get_logger,
    unbind_context,
)

pytestmark = pytest.mark.unit


def test_configure_idempotent() -> None:
    """configure() can be called multiple times without raising."""
    configure(log_format="text", log_level="INFO")
    configure(log_format="text", log_level="INFO")
    configure(log_format="structured", log_level="DEBUG")


def test_get_logger_returns_bound_logger() -> None:
    configure(log_format="text", log_level="INFO")
    logger = get_logger("ai_core.test")
    assert logger is not None
    assert callable(getattr(logger, "warning", None))


def test_bind_context_propagates_through_log_call() -> None:
    """ContextVar values land as fields on every log line."""
    configure(log_format="structured", log_level="DEBUG")
    token = bind_context(agent_id="agent-x", tenant_id="tenant-y")
    try:
        with capture_logs() as captured:
            logger = get_logger("ai_core.test")
            logger.warning("test.event")
        assert len(captured) == 1
        record = captured[0]
        assert record["event"] == "test.event"
        assert record["agent_id"] == "agent-x"
        assert record["tenant_id"] == "tenant-y"
    finally:
        unbind_context(token)


def test_unbind_context_clears_fields() -> None:
    """After unbind_context, log lines no longer carry the previously-bound fields."""
    configure(log_format="structured", log_level="DEBUG")
    token = bind_context(agent_id="agent-x")
    unbind_context(token)
    with capture_logs() as captured:
        logger = get_logger("ai_core.test")
        logger.warning("test.event")
    assert len(captured) == 1
    assert "agent_id" not in captured[0]


def test_structured_renderer_is_active_when_configured() -> None:
    """log_format='structured' produces JSON-renderable output."""
    configure(log_format="structured", log_level="DEBUG")
    logger = get_logger("ai_core.test")
    with capture_logs() as captured:
        logger.info("config.ok", value=42)
    assert captured[0]["event"] == "config.ok"
    assert captured[0]["value"] == 42


def test_text_renderer_is_active_when_configured() -> None:
    """log_format='text' (default) configures the console renderer."""
    configure(log_format="text", log_level="DEBUG")
    logger = get_logger("ai_core.test")
    with capture_logs() as captured:
        logger.info("config.ok", value=42)
    assert captured[0]["event"] == "config.ok"


def test_bind_context_accumulates_across_calls() -> None:
    """Successive bind_context calls accumulate; unbind only the most recent."""
    configure(log_format="structured", log_level="DEBUG")

    token_a = bind_context(agent_id="a")
    token_b = bind_context(tenant_id="t")
    try:
        with capture_logs() as captured:
            logger = get_logger("ai_core.test")
            logger.warning("test.event")
        record = captured[0]
        assert record["agent_id"] == "a"
        assert record["tenant_id"] == "t"

        # Unbinding token_b removes only tenant_id.
        unbind_context(token_b)
        with capture_logs() as captured2:
            logger.warning("test.event2")
        record2 = captured2[0]
        assert record2.get("agent_id") == "a"
        assert "tenant_id" not in record2
    finally:
        unbind_context(token_a)


@pytest.mark.asyncio
async def test_asyncio_task_isolation() -> None:
    """ContextVar-bound fields must NOT cross-contaminate between concurrent tasks."""
    configure(log_format="structured", log_level="DEBUG")

    captured_per_task: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}

    async def _task(label: str) -> None:
        token = bind_context(agent_id=f"agent-{label}", task_label=label)
        try:
            logger = get_logger(f"ai_core.test.{label}")
            # Log during concurrent execution (without capture_logs to avoid
            # structlog.testing.capture_logs context manager contention).
            logger.warning("task.event")
            await asyncio.sleep(0)  # yield to interleave with other tasks
            logger.info("task.event.followup")

            # After concurrent interleaving, capture a verification log to confirm
            # this task's context is still correct (not cross-contaminated).
            with capture_logs() as captured:
                logger.warning("task.verify")
            captured_per_task[label] = captured  # type: ignore[assignment]
        finally:
            unbind_context(token)

    await asyncio.gather(_task("A"), _task("B"), _task("C"))

    # Each task must have captured ONLY its own agent_id / task_label
    # in the verify log, proving isolation was maintained through concurrent execution.
    for label, events in captured_per_task.items():
        assert len(events) == 1, f"task {label}: expected 1 verify event, got {len(events)}"
        event = events[0]
        assert event.get("agent_id") == f"agent-{label}", (
            f"task {label} saw wrong agent_id: {event.get('agent_id')!r}"
        )
        assert event.get("task_label") == label, (
            f"task {label} saw wrong task_label: {event.get('task_label')!r}"
        )
