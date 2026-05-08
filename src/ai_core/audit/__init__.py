"""Audit subsystem — policy and tool event records for compliance."""

from __future__ import annotations

from ai_core.audit.interface import (
    AuditEvent,
    AuditRecord,
    IAuditSink,
    PayloadRedactor,
)
from ai_core.audit.jsonl import JsonlFileAuditSink
from ai_core.audit.null import NullAuditSink
from ai_core.audit.otel_event import OTelEventAuditSink
from ai_core.audit.redaction import (
    ChainRedactor,
    KeyNameRedactor,
    RegexRedactor,
)
from ai_core.audit.registry import (
    AuditSinkFactory,
    get_audit_sink_factory,
    known_audit_sink_names,
    register_audit_sink,
)

__all__ = [
    "AuditEvent",
    "AuditRecord",
    "AuditSinkFactory",
    "ChainRedactor",
    "IAuditSink",
    "JsonlFileAuditSink",
    "KeyNameRedactor",
    "NullAuditSink",
    "OTelEventAuditSink",
    "PayloadRedactor",
    "RegexRedactor",
    "get_audit_sink_factory",
    "known_audit_sink_names",
    "register_audit_sink",
]
