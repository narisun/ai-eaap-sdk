"""Structured logging seam.

Other SDK modules import :func:`get_logger` from here, never from
``structlog`` directly. The seam makes it possible to replace structlog
later without touching every callsite.

The :func:`configure` function is called once by :class:`AICoreApp`
during ``__aenter__``. Tests may call it directly with their preferred
shape.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import structlog
import structlog.contextvars
from structlog.stdlib import LoggerFactory

if TYPE_CHECKING:
    import contextvars

    from structlog.typing import FilteringBoundLogger


class _ContextVarMergingDict(dict):  # type: ignore[type-arg]
    """dict subclass that folds structlog ContextVar context into every copy().

    ``BoundLoggerBase._process_event`` starts the event_dict from
    ``self._context.copy()``.  By merging the structlog contextvars here we
    get context propagation that works even when structlog's own
    ``merge_contextvars`` processor has been removed by
    ``structlog.testing.capture_logs``.
    """

    def copy(self) -> _ContextVarMergingDict:
        merged: dict[str, Any] = dict(structlog.contextvars.get_contextvars())
        merged.update(self)
        return _ContextVarMergingDict(merged)


def configure(
    *,
    log_format: Literal["text", "structured"] = "text",
    log_level: str = "INFO",
) -> None:
    """Configure structlog. Idempotent — safe to call multiple times."""
    structlog.reset_defaults()
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
    ]
    if log_format == "structured":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO),
        ),
        logger_factory=LoggerFactory(),
        context_class=_ContextVarMergingDict,
        cache_logger_on_first_use=False,
    )


def bind_context(**kwargs: Any) -> dict[str, contextvars.Token[Any]]:  # noqa: ANN401
    """Push ``kwargs`` into the request-scoped structlog ContextVars; return reset tokens."""
    return dict(structlog.contextvars.bind_contextvars(**kwargs))


def unbind_context(token: dict[str, contextvars.Token[Any]]) -> None:
    """Reset the structlog ContextVars to the prior snapshot using the tokens."""
    structlog.contextvars.reset_contextvars(**token)


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog logger bound to module ``name``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


__all__ = ["bind_context", "configure", "get_logger", "unbind_context"]
