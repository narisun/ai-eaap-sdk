"""Pluggable registry for :class:`IAuditSink` factories.

The default :class:`AgentModule` selects an audit sink by name from
:attr:`AuditSettings.sink_type`. Built-in sinks (``null``, ``jsonl``,
``otel_event``, ``sentry``, ``datadog``) register themselves at import
time. Third-party packages can extend the set without forking the SDK
either by:

* calling :func:`register_audit_sink` from any module loaded before the
  container starts, or
* declaring a ``ai_eaap_sdk.audit_sinks`` entry point in their package
  metadata. The entry point's value must resolve to a callable matching
  :data:`AuditSinkFactory`.

Example — third-party Loki sink::

    # in my_eaap_loki/sink.py
    from ai_core.audit import register_audit_sink

    def _loki_factory(audit_cfg, observability):
        return LokiAuditSink(url=audit_cfg.loki_url)  # extra field on a host-defined AuditSettings subclass

    register_audit_sink("loki", _loki_factory)

The registry is process-global and intentionally side-effecting: once a
factory is registered it stays for the lifetime of the interpreter, so
hosts call :func:`register_audit_sink` exactly once (at import or app
boot) per name.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ai_core.exceptions import ConfigurationError, ErrorCode

if TYPE_CHECKING:
    from ai_core.audit.interface import IAuditSink
    from ai_core.config.settings import AuditSettings
    from ai_core.di.interfaces import IObservabilityProvider

_logger = logging.getLogger(__name__)

#: Factory signature: ``(audit_cfg, observability) -> IAuditSink``.
AuditSinkFactory = Callable[
    ["AuditSettings", "IObservabilityProvider"], "IAuditSink"
]

_REGISTRY: dict[str, AuditSinkFactory] = {}
_ENTRY_POINTS_LOADED: bool = False
_ENTRY_POINT_GROUP = "ai_eaap_sdk.audit_sinks"


def register_audit_sink(name: str, factory: AuditSinkFactory) -> None:
    """Register a factory under ``name``.

    Re-registering the same name silently replaces the prior factory.
    Names should be lower-case identifiers (the value of
    :attr:`AuditSettings.sink_type`).

    Args:
        name: Identifier matched against :attr:`AuditSettings.sink_type`.
        factory: Callable that returns a configured :class:`IAuditSink`.
    """
    if not name:
        raise ValueError("register_audit_sink: name must be non-empty")
    _REGISTRY[name] = factory


def get_audit_sink_factory(name: str) -> AuditSinkFactory:
    """Return the factory for ``name`` or raise :class:`ConfigurationError`.

    Triggers entry-point discovery on first call.

    Raises:
        ConfigurationError: When no factory is registered for ``name``.
    """
    _load_entry_points_once()
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ConfigurationError(
            f"Unknown audit.sink_type: {name!r}",
            error_code=ErrorCode.CONFIG_INVALID,
            details={"sink_type": name, "available": sorted(_REGISTRY)},
        )
    return factory


def known_audit_sink_names() -> list[str]:
    """Return all registered names. Useful for diagnostics + ``--help`` output."""
    _load_entry_points_once()
    return sorted(_REGISTRY)


def _load_entry_points_once() -> None:
    """Discover and register third-party sinks declared as entry points.

    Called lazily so importing :mod:`ai_core.audit.registry` does not
    pull in arbitrary user code at SDK import time.
    """
    global _ENTRY_POINTS_LOADED  # noqa: PLW0603
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 — metadata API is best-effort
        _logger.debug("audit_sink.entry_points.scan_failed: %s", exc)
        return
    for ep in eps:
        try:
            factory = ep.load()
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "audit_sink.entry_point.load_failed name=%r value=%r error=%s",
                ep.name, ep.value, exc,
            )
            continue
        if not callable(factory):
            _logger.warning(
                "audit_sink.entry_point.not_callable name=%r value=%r",
                ep.name, ep.value,
            )
            continue
        register_audit_sink(ep.name, factory)


# ---------------------------------------------------------------------------
# Built-in factory registrations
# ---------------------------------------------------------------------------
def _null_factory(_cfg: AuditSettings, _obs: IObservabilityProvider) -> IAuditSink:
    from ai_core.audit.null import NullAuditSink  # noqa: PLC0415
    return NullAuditSink()


def _otel_event_factory(
    _cfg: AuditSettings, observability: IObservabilityProvider
) -> IAuditSink:
    from ai_core.audit.otel_event import OTelEventAuditSink  # noqa: PLC0415
    return OTelEventAuditSink(observability)


def _jsonl_factory(cfg: AuditSettings, _obs: IObservabilityProvider) -> IAuditSink:
    from ai_core.audit.jsonl import JsonlFileAuditSink  # noqa: PLC0415
    if cfg.jsonl_path is None:
        raise ConfigurationError(
            "audit.sink_type='jsonl' requires audit.jsonl_path to be set",
            error_code=ErrorCode.CONFIG_INVALID,
        )
    return JsonlFileAuditSink(cfg.jsonl_path)


def _sentry_factory(cfg: AuditSettings, _obs: IObservabilityProvider) -> IAuditSink:
    from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
    if cfg.sentry_dsn is None:
        raise ConfigurationError(
            "audit.sink_type='sentry' requires audit.sentry_dsn to be set",
            error_code=ErrorCode.CONFIG_INVALID,
        )
    return SentryAuditSink(
        dsn=cfg.sentry_dsn.get_secret_value(),
        environment=cfg.sentry_environment,
        release=cfg.sentry_release,
        sample_rate=cfg.sentry_sample_rate,
    )


def _datadog_factory(cfg: AuditSettings, _obs: IObservabilityProvider) -> IAuditSink:
    from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
    if cfg.datadog_api_key is None:
        raise ConfigurationError(
            "audit.sink_type='datadog' requires audit.datadog_api_key to be set",
            error_code=ErrorCode.CONFIG_INVALID,
        )
    return DatadogAuditSink(
        api_key=cfg.datadog_api_key.get_secret_value(),
        app_key=(
            cfg.datadog_app_key.get_secret_value()
            if cfg.datadog_app_key
            else None
        ),
        site=cfg.datadog_site,
        source=cfg.datadog_source,
        environment=cfg.datadog_environment,
    )


# Register built-ins at import time.
register_audit_sink("null", _null_factory)
register_audit_sink("otel_event", _otel_event_factory)
register_audit_sink("jsonl", _jsonl_factory)
register_audit_sink("sentry", _sentry_factory)
register_audit_sink("datadog", _datadog_factory)


__all__ = [
    "AuditSinkFactory",
    "get_audit_sink_factory",
    "known_audit_sink_names",
    "register_audit_sink",
]
