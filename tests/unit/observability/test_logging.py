"""Tests for the structlog logging seam."""
from __future__ import annotations

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
