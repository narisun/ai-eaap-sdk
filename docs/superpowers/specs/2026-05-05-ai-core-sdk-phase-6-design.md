# ai-core-sdk Phase 6 — Design

**Date:** 2026-05-05
**Branch:** `feat/phase-6-prod-integrations`
**Status:** Awaiting user review

## Goal

Production integrations layer: ship concrete `PayloadRedactor` PII strippers and reference `IAuditSink` implementations for Sentry and Datadog. After Phase 6, an SDK consumer can configure `audit.sink_type=sentry` (or `datadog`) plus `audit.redaction_profile=standard` and get audit records redacted at the call site, dispatched as first-class events to their existing observability backend.

## Scope (3 items)

### Production integrations (3)

1. **PayloadRedactor concretes.** Three new pure-function classes in a new private subpackage `ai_core.audit.redaction`:
   - `RegexRedactor` — built-in patterns for `email`, `phone` (US E.164), `ssn`, `credit_card` (Luhn-checked), `ipv4`. Selectable via `enabled_patterns: set[Literal[...]]`. Replaces matches with `<redacted-{kind}>`. Recursive over nested dicts/lists.
   - `KeyNameRedactor` — case-insensitive key matching against a default set (`{password, passwd, secret, api_key, apikey, authorization, auth, token, access_token, refresh_token, bearer, cookie, session, private_key, ssh_key}`). Replaces values with `[REDACTED]`. Customisable via `redact_keys: set[str]`.
   - `ChainRedactor(*redactors)` — composes N redactors; output of one feeds the next.
   - DI binding: new `provide_payload_redactor(self, settings) -> PayloadRedactor` reads `AuditSettings.redaction_profile: Literal["off", "standard", "strict"] = "off"` and returns the appropriate chain. `ToolInvoker.__init__` accepts a new `redactor: PayloadRedactor | None = None` and threads it into every `AuditRecord.now(..., redactor=self._redactor)` call. Default `_identity_redactor` preserves Phase 1-5 behaviour for apps that don't opt in.

2. **`SentryAuditSink`** in `src/ai_core/audit/sentry.py`. Constructor accepts `dsn: str`, `environment: str | None`, `release: str | None`, `sample_rate: float = 1.0`. Calls `sentry_sdk.init(...)` once at construction (idempotent). `record(...)` maps `AuditRecord` → `sentry_sdk.capture_event(...)` with auto-level inference (warning if `decision_allowed=False` or `error_code is not None`; info otherwise) plus tags (`event`, `tool_name`, `tool_version`, `agent_id`, `tenant_id`, `decision_path`) and `extra` carrying the (already-redacted) payload. `flush()` calls `sentry_sdk.flush(timeout=5.0)`. Errors swallowed per `IAuditSink` contract. Optional dep group `[project.optional-dependencies] sentry = ["sentry-sdk>=1.40,<3.0"]`. Deferred import; missing → `ConfigurationError(error_code="config.optional_dep_missing", details={"extra": "sentry"})`.

3. **`DatadogAuditSink`** in `src/ai_core/audit/datadog.py`. Constructor accepts `api_key: str`, `app_key: str | None`, `site: str = "datadoghq.com"`, `source: str = "ai-core-sdk"`, `environment: str | None`. Calls `datadog.initialize(api_key=..., app_key=..., api_host=...)` at construction. `record(...)` maps `AuditRecord` → `datadog.api.Event.create(title=..., text=..., tags=[...], alert_type=...)` with `alert_type="warning"` when `decision_allowed=False` or `error_code is not None`, else `"info"`. `text` field carries the (already-redacted) payload as JSON. `flush()` is a no-op (Datadog events API has no buffer). Errors swallowed. Optional dep group `[project.optional-dependencies] datadog = ["datadog>=0.50,<1.0"]`. Deferred import; missing → `ConfigurationError(error_code="config.optional_dep_missing", details={"extra": "datadog"})`.

## Non-goals (deferred to Phase 7+)

- Sentry breadcrumbs / performance tracing integration
- Datadog metrics / DogStatsD / log streaming
- Custom regex patterns at the settings layer (free-form list)
- Per-sink redaction overrides
- Microsoft Presidio / Faker / spaCy as alternative redactor backends
- Redactor performance tuning beyond compiled-pattern reuse
- Contract tests for the public 28-name surface
- Testcontainers replacement of component-test fakes
- Phase 4 follow-ups (Vertex AI Anthropic prefix, 1-hour cache TTL, multi-conn MCP pool, pool health probe)
- `error_code` lookup registry / constants module
- Sentry SDK v3 migration (when released)

## Constraints

- Pre-1.0; **no backwards-compatibility requirement.** Adding redactors and sinks is purely additive — apps that don't opt in see no behaviour change. `AuditRecord.now(redactor=...)` already exists from Phase 1.
- Per-step gate: `ruff check` no new violations + `mypy <touched files>` strict + `pytest tests/unit tests/component`.
- Project mypy total stays ≤ 21 (post-Phase-3 baseline).
- Project ruff total stays ≤ 211 (post-Phase-4 baseline at `4c530d2`).
- New optional deps NOT added to the default `dependencies` list — only under `[project.optional-dependencies] sentry` / `datadog`. Default `pip install ai-core-sdk` footprint does not grow.
- End-of-phase smoke against `my-eaap-app` and the canonical 28-name top-level surface must continue to work.

## Module layout

```
src/ai_core/audit/
├── interface.py                  # unchanged (PayloadRedactor type alias + AuditRecord already support redaction)
├── redaction/                    # NEW subpackage
│   ├── __init__.py               # NEW — re-exports RegexRedactor, KeyNameRedactor, ChainRedactor
│   ├── regex.py                  # NEW — RegexRedactor + 5 built-in pattern definitions (Luhn helper)
│   ├── key_name.py               # NEW — KeyNameRedactor + DEFAULT_REDACT_KEYS set
│   └── chain.py                  # NEW — ChainRedactor
├── sentry.py                     # NEW — SentryAuditSink
├── datadog.py                    # NEW — DatadogAuditSink
├── jsonl.py                      # unchanged
├── otel_event.py                 # unchanged
└── null.py                       # unchanged

src/ai_core/audit/__init__.py     # MODIFIED — re-export the new redactor classes
src/ai_core/config/settings.py    # MODIFIED — extend AuditSettings (sink_type literal + redaction_profile + sentry_* + datadog_*)
src/ai_core/di/module.py          # MODIFIED — provide_payload_redactor + extend provide_audit_sink for sentry/datadog
src/ai_core/tools/invoker.py      # MODIFIED — accept redactor param; thread into AuditRecord.now()
src/ai_core/exceptions.py         # MODIFIED — add config.optional_dep_missing error_code constant (or use inline string; same convention as Phase 5)

pyproject.toml                    # MODIFIED — [project.optional-dependencies] sentry / datadog groups
```

### Files NOT touched

- `README.md` (top-level)
- `src/ai_core/cli/main.py`
- `src/ai_core/cli/templates/init/**` (Phase 5 already shipped these; Phase 6 doesn't touch them)
- `tests/unit/cli/test_main.py`

### Test additions

```
tests/unit/audit/redaction/
├── __init__.py                   # NEW
├── test_regex_redactor.py        # NEW (~12 tests for 5 patterns + edge cases)
├── test_key_name_redactor.py     # NEW (~6 tests)
├── test_chain_redactor.py        # NEW (~3 tests)
└── test_redaction_profile.py     # NEW (~3 tests for off/standard/strict)

tests/unit/audit/test_sentry_sink.py       # NEW (~6 tests, mocking sentry_sdk)
tests/unit/audit/test_datadog_sink.py      # NEW (~6 tests, mocking datadog SDK)
tests/unit/tools/test_invoker.py           # MODIFIED — +1 test verifying redactor threads into AuditRecord
tests/unit/di/test_module.py               # MODIFIED — +3 tests for new provider behaviour
```

Roughly 40 new tests; net pytest delta around +40 (no deletions).

## Component 1 — Redaction layer

### 1a. `RegexRedactor`

```python
# src/ai_core/audit/redaction/regex.py

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

PatternKind = Literal["email", "phone", "ssn", "credit_card", "ipv4", "long_number"]

# Patterns are compiled once at module import; safe to share.
_PATTERNS: dict[PatternKind, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "ssn":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "ipv4":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "long_number": re.compile(r"\b\d{6,}\b"),  # strict-profile only
}


def _luhn_check(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class RegexRedactor:
    """Strip PII patterns from string values inside a payload mapping.

    Replacement format: ``<redacted-{kind}>``.
    """

    def __init__(self, *, enabled_patterns: set[PatternKind]) -> None:
        # Validate: unknown patterns raise immediately.
        unknown = enabled_patterns - set(_PATTERNS.keys())
        if unknown:
            raise ValueError(f"Unknown patterns: {sorted(unknown)}")
        self._patterns: dict[PatternKind, re.Pattern[str]] = {
            k: _PATTERNS[k] for k in enabled_patterns
        }

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {k: self._redact_value(v) for k, v in payload.items()}

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_string(value)
        if isinstance(value, Mapping):
            return {k: self._redact_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        return value

    def _redact_string(self, s: str) -> str:
        result = s
        for kind, pattern in self._patterns.items():
            if kind == "credit_card":
                # Luhn-check each match; only redact valid card numbers.
                result = pattern.sub(
                    lambda m: f"<redacted-{kind}>" if _luhn_check(m.group(0)) else m.group(0),
                    result,
                )
            else:
                result = pattern.sub(f"<redacted-{kind}>", result)
        return result
```

### 1b. `KeyNameRedactor`

```python
# src/ai_core/audit/redaction/key_name.py

from collections.abc import Mapping
from typing import Any

DEFAULT_REDACT_KEYS: frozenset[str] = frozenset({
    "password", "passwd", "secret", "api_key", "apikey",
    "authorization", "auth", "token", "access_token", "refresh_token",
    "bearer", "cookie", "session", "private_key", "ssh_key",
})


class KeyNameRedactor:
    """Replace values for any key whose lowercase name matches the redact set.

    Replacement: literal string ``[REDACTED]``.
    """

    def __init__(self, redact_keys: set[str] | None = None) -> None:
        keys = redact_keys if redact_keys is not None else set(DEFAULT_REDACT_KEYS)
        self._redact_keys: frozenset[str] = frozenset(k.lower() for k in keys)

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {k: self._redact_value(k, v) for k, v in payload.items()}

    def _redact_value(self, key: str, value: Any) -> Any:
        if key.lower() in self._redact_keys:
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {k: self._redact_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(key, v) for v in value]  # propagate parent key context for list items
        return value
```

(Note: list-items inherit the parent key context, so `{"tokens": ["aaa", "bbb"]}` redacts to `{"tokens": ["[REDACTED]", "[REDACTED]"]}`. That's the natural reading; documented in the docstring.)

### 1c. `ChainRedactor`

```python
# src/ai_core/audit/redaction/chain.py

from collections.abc import Mapping
from typing import Any
from ai_core.audit import PayloadRedactor


class ChainRedactor:
    """Apply N redactors in order; output of one feeds the next."""

    def __init__(self, *redactors: PayloadRedactor) -> None:
        self._redactors: tuple[PayloadRedactor, ...] = redactors

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        result: Mapping[str, Any] = payload
        for r in self._redactors:
            result = r(result)
        return dict(result)
```

### 1d. DI binding

```python
# in di/module.py

@singleton
@provider
def provide_payload_redactor(self, settings: AppSettings) -> PayloadRedactor:
    """Return the configured PayloadRedactor — identity for profile='off'."""
    profile = settings.audit.redaction_profile
    if profile == "off":
        from ai_core.audit.interface import _identity_redactor  # noqa: PLC0415
        return _identity_redactor
    from ai_core.audit.redaction import (  # noqa: PLC0415
        ChainRedactor, KeyNameRedactor, RegexRedactor,
    )
    base_patterns: set[PatternKind] = {"email", "phone", "ssn", "credit_card", "ipv4"}
    if profile == "strict":
        base_patterns.add("long_number")
    return ChainRedactor(
        RegexRedactor(enabled_patterns=base_patterns),
        KeyNameRedactor(),  # default key set
    )
```

### 1e. `ToolInvoker` integration

```python
# in tools/invoker.py

def __init__(
    self,
    *,
    observability: IObservabilityProvider,
    policy: IPolicyEvaluator | None = None,
    registry: SchemaRegistry | None = None,
    audit: IAuditSink | None = None,
    redactor: PayloadRedactor | None = None,  # NEW
) -> None:
    from ai_core.audit.interface import _identity_redactor  # noqa: PLC0415
    from ai_core.audit.null import NullAuditSink  # noqa: PLC0415
    self._observability = observability
    self._policy = policy
    self._registry = registry
    self._audit: IAuditSink = audit or NullAuditSink()
    self._records_audit: bool = not isinstance(self._audit, NullAuditSink)
    self._redactor: PayloadRedactor = redactor or _identity_redactor  # NEW
```

Then every `AuditRecord.now(...)` call site adds `redactor=self._redactor`. Three call sites (POLICY_DECISION, TOOL_INVOCATION_COMPLETED, TOOL_INVOCATION_FAILED).

DI wires the new param via `AgentModule.provide_tool_invoker(...)` (or wherever `ToolInvoker` is currently constructed in DI).

## Component 2 — `SentryAuditSink`

```python
# src/ai_core/audit/sentry.py

from __future__ import annotations

import json
import logging
from typing import Any

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.exceptions import ConfigurationError

_logger = logging.getLogger(__name__)


class SentryAuditSink(IAuditSink):
    """Forward audit records to Sentry as first-class events.

    Audit-sink contract: never raise; flush idempotent; safe for concurrent use.
    """

    def __init__(
        self,
        *,
        dsn: str,
        environment: str | None = None,
        release: str | None = None,
        sample_rate: float = 1.0,
    ) -> None:
        try:
            import sentry_sdk  # noqa: PLC0415
        except ImportError as exc:
            raise ConfigurationError(
                "Sentry sink requires the 'sentry' optional dependency. "
                "Install with: pip install ai-core-sdk[sentry]",
                error_code="config.optional_dep_missing",
                details={"extra": "sentry"},
                cause=exc,
            ) from exc

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            sample_rate=sample_rate,
            # Audit events are server-side breadcrumbs; we don't send local crumbs.
            max_breadcrumbs=0,
        )
        self._sentry_sdk = sentry_sdk

    async def record(self, record: AuditRecord) -> None:
        try:
            level = (
                "warning"
                if (record.decision_allowed is False or record.error_code is not None)
                else "info"
            )
            self._sentry_sdk.capture_event({
                "message": f"eaap.audit.{record.event.value}",
                "level": level,
                "timestamp": record.timestamp.isoformat(),
                "tags": {
                    "audit.event": record.event.value,
                    "audit.tool_name": record.tool_name or "",
                    "audit.tool_version": (
                        str(record.tool_version) if record.tool_version is not None else ""
                    ),
                    "audit.agent_id": record.agent_id or "",
                    "audit.tenant_id": record.tenant_id or "",
                    "audit.decision_path": record.decision_path or "",
                    "audit.decision_allowed": (
                        str(record.decision_allowed)
                        if record.decision_allowed is not None
                        else ""
                    ),
                    "audit.error_code": record.error_code or "",
                },
                "extra": {
                    "payload": dict(record.payload),
                    "decision_reason": record.decision_reason,
                    "latency_ms": record.latency_ms,
                },
            })
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "audit.sentry.record_failed",
                extra={
                    "event": record.event.value,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    async def flush(self) -> None:
        try:
            self._sentry_sdk.flush(timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "audit.sentry.flush_failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
```

## Component 3 — `DatadogAuditSink`

```python
# src/ai_core/audit/datadog.py

from __future__ import annotations

import json
import logging

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.exceptions import ConfigurationError

_logger = logging.getLogger(__name__)


class DatadogAuditSink(IAuditSink):
    """Forward audit records to Datadog as first-class events via the Events API.

    Audit-sink contract: never raise; flush idempotent (no-op); safe for concurrent use.
    """

    def __init__(
        self,
        *,
        api_key: str,
        app_key: str | None = None,
        site: str = "datadoghq.com",
        source: str = "ai-core-sdk",
        environment: str | None = None,
    ) -> None:
        try:
            import datadog  # noqa: PLC0415
        except ImportError as exc:
            raise ConfigurationError(
                "Datadog sink requires the 'datadog' optional dependency. "
                "Install with: pip install ai-core-sdk[datadog]",
                error_code="config.optional_dep_missing",
                details={"extra": "datadog"},
                cause=exc,
            ) from exc

        datadog.initialize(
            api_key=api_key,
            app_key=app_key,
            api_host=f"https://api.{site}",
        )
        self._datadog = datadog
        self._source = source
        self._environment = environment

    async def record(self, record: AuditRecord) -> None:
        try:
            alert_type = (
                "warning"
                if (record.decision_allowed is False or record.error_code is not None)
                else "info"
            )
            tags = [
                f"event:{record.event.value}",
                f"tool_name:{record.tool_name or 'unknown'}",
                f"agent_id:{record.agent_id or 'unknown'}",
                f"tenant_id:{record.tenant_id or 'unknown'}",
            ]
            if record.error_code:
                tags.append(f"error_code:{record.error_code}")
            if self._environment:
                tags.append(f"env:{self._environment}")
            if record.decision_path:
                tags.append(f"decision_path:{record.decision_path}")
            if record.decision_allowed is not None:
                tags.append(f"decision_allowed:{record.decision_allowed}")

            text_payload = json.dumps({
                "payload": dict(record.payload),
                "decision_reason": record.decision_reason,
                "latency_ms": record.latency_ms,
            })

            self._datadog.api.Event.create(
                title=f"eaap.audit.{record.event.value}",
                text=text_payload,
                tags=tags,
                alert_type=alert_type,
                source_type_name=self._source,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "audit.datadog.record_failed",
                extra={
                    "event": record.event.value,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    async def flush(self) -> None:
        # Datadog events API is synchronous-per-call; no buffer to flush.
        return
```

## Component 4 — Settings + DI wiring

### Settings additions (`AuditSettings`)

```python
class AuditSettings(BaseSettings):
    """Audit-sink configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    sink_type: Literal["null", "otel_event", "jsonl", "sentry", "datadog"] = "null"
    jsonl_path: Path | None = None  # required when sink_type == "jsonl"

    # Phase 6 additions
    redaction_profile: Literal["off", "standard", "strict"] = "off"

    # Sentry
    sentry_dsn: SecretStr | None = None
    sentry_environment: str | None = None
    sentry_release: str | None = None
    sentry_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    # Datadog
    datadog_api_key: SecretStr | None = None
    datadog_app_key: SecretStr | None = None
    datadog_site: str = "datadoghq.com"
    datadog_source: str = "ai-core-sdk"
    datadog_environment: str | None = None
```

### Extended `provide_audit_sink`

```python
@singleton
@provider
def provide_audit_sink(
    self,
    settings: AppSettings,
    observability: IObservabilityProvider,
) -> IAuditSink:
    """Default audit sink — switchable via settings.audit.sink_type."""
    sink_type = settings.audit.sink_type
    if sink_type == "null":
        from ai_core.audit.null import NullAuditSink  # noqa: PLC0415
        return NullAuditSink()
    if sink_type == "otel_event":
        from ai_core.audit.otel_event import OTelEventAuditSink  # noqa: PLC0415
        return OTelEventAuditSink(observability)
    if sink_type == "jsonl":
        from ai_core.audit.jsonl import JsonlFileAuditSink  # noqa: PLC0415
        if settings.audit.jsonl_path is None:
            raise ConfigurationError(
                "audit.sink_type='jsonl' requires audit.jsonl_path to be set",
                error_code="config.invalid",
            )
        return JsonlFileAuditSink(settings.audit.jsonl_path)
    if sink_type == "sentry":
        from ai_core.audit.sentry import SentryAuditSink  # noqa: PLC0415
        if settings.audit.sentry_dsn is None:
            raise ConfigurationError(
                "audit.sink_type='sentry' requires audit.sentry_dsn to be set",
                error_code="config.invalid",
            )
        return SentryAuditSink(
            dsn=settings.audit.sentry_dsn.get_secret_value(),
            environment=settings.audit.sentry_environment,
            release=settings.audit.sentry_release,
            sample_rate=settings.audit.sentry_sample_rate,
        )
    if sink_type == "datadog":
        from ai_core.audit.datadog import DatadogAuditSink  # noqa: PLC0415
        if settings.audit.datadog_api_key is None:
            raise ConfigurationError(
                "audit.sink_type='datadog' requires audit.datadog_api_key to be set",
                error_code="config.invalid",
            )
        return DatadogAuditSink(
            api_key=settings.audit.datadog_api_key.get_secret_value(),
            app_key=(
                settings.audit.datadog_app_key.get_secret_value()
                if settings.audit.datadog_app_key
                else None
            ),
            site=settings.audit.datadog_site,
            source=settings.audit.datadog_source,
            environment=settings.audit.datadog_environment,
        )
    raise ConfigurationError(
        f"Unknown audit.sink_type: {sink_type!r}",
        error_code="config.invalid",
    )
```

### `pyproject.toml` extras

```toml
[project.optional-dependencies]
sentry = ["sentry-sdk>=1.40,<3.0"]
datadog = ["datadog>=0.50,<1.0"]
```

## Error handling — consolidated (Phase 6 deltas)

| Path | Behaviour |
|---|---|
| Redactor with unknown pattern (`RegexRedactor(enabled_patterns={"foo"})`) | `ValueError` at `__init__`. Not user-reachable through DI (profiles are Literal-typed). |
| Redactor on non-string / non-mapping / non-list value | Pass through unchanged. Best-effort. |
| Sentry import missing | `ConfigurationError(error_code="config.optional_dep_missing", details={"extra": "sentry"})`. |
| Datadog import missing | `ConfigurationError(error_code="config.optional_dep_missing", details={"extra": "datadog"})`. |
| Sentry/Datadog DSN/api_key missing | `ConfigurationError(error_code="config.invalid")` with field name in message. |
| Sentry `capture_event` raises | Caught + logged at WARNING (`audit.sentry.record_failed`); never propagates. |
| Sentry `flush` raises | Same — caught, logged, swallowed. |
| Datadog `Event.create` raises | Caught + logged at WARNING (`audit.datadog.record_failed`); never propagates. |
| Datadog `flush` | No-op (no buffer). |
| `ToolInvoker` with `redactor=None` | Default `_identity_redactor` — Phase 1-5 behaviour preserved. |

New `error_code` values:
- `"config.optional_dep_missing"` — ONE new code (catches both sentry and datadog cases via `details["extra"]`).

## Testing strategy

Per-step gate identical to Phases 1-5. Project mypy total stays ≤ 21. Project ruff total stays ≤ 211.

### Per-step test additions

| Step | Tests |
|---|---|
| 1. Redaction layer | ~24 tests (12 regex + 6 key_name + 3 chain + 3 profile) |
| 2. Sentry sink | ~6 tests + 1 invoker test |
| 3. Datadog sink | ~6 tests |
| 4. Smoke gate | full pytest + ruff + mypy + extras-install smoke |

### Test detail — redaction layer

`tests/unit/audit/redaction/test_regex_redactor.py` (~12 tests):
- `test_redacts_email_address`
- `test_redacts_phone_us_format`
- `test_redacts_ssn_with_dashes`
- `test_redacts_credit_card_passing_luhn`
- `test_does_not_redact_credit_card_failing_luhn`
- `test_redacts_ipv4_address`
- `test_redacts_recursively_through_nested_dict`
- `test_redacts_recursively_through_list_of_strings`
- `test_passes_through_non_string_non_dict_non_list_values` (int, None, bool)
- `test_selective_pattern_enable` (`enabled_patterns={"email"}` doesn't redact phone)
- `test_unknown_pattern_raises_value_error`

`tests/unit/audit/redaction/test_key_name_redactor.py` (~6 tests):
- `test_redacts_default_password_key`
- `test_redacts_default_api_key`
- `test_case_insensitive_match` (`Password`, `API_KEY`)
- `test_custom_key_set` (`KeyNameRedactor({"my_secret"})` redacts only that)
- `test_redacts_nested_keys`
- `test_redacts_inside_list_with_parent_key_context`

`tests/unit/audit/redaction/test_chain_redactor.py` (~3 tests):
- `test_two_redactor_chain_applies_both`
- `test_chain_order_matters` (regex first → key-name; reversed gives different result for some inputs)
- `test_empty_chain_is_identity`

`tests/unit/audit/redaction/test_redaction_profile.py` (~3 tests):
- `test_provide_payload_redactor_off_returns_identity`
- `test_provide_payload_redactor_standard_returns_chain` (verify it redacts both PII and key-names)
- `test_provide_payload_redactor_strict_includes_long_number` (verify the additional pattern)

### Test detail — Sentry sink

`tests/unit/audit/test_sentry_sink.py` (~6 tests, mocking `sentry_sdk`):
- `test_init_calls_sentry_sdk_init_with_dsn`
- `test_record_emits_capture_event_with_info_level_for_allowed`
- `test_record_emits_warning_level_for_decision_denied`
- `test_record_emits_warning_level_for_error_code_set`
- `test_record_includes_audit_tags`
- `test_record_swallows_exception_from_capture_event`
- `test_flush_calls_sentry_sdk_flush_with_timeout`
- `test_missing_sentry_sdk_raises_configuration_error_with_extra_detail`

(8 tests total; rounded "~6" in the table.)

### Test detail — Datadog sink

`tests/unit/audit/test_datadog_sink.py` (~6 tests, mocking `datadog`):
- `test_init_calls_datadog_initialize_with_api_key_and_site`
- `test_record_emits_event_create_with_info_alert_type`
- `test_record_emits_warning_alert_type_for_denied`
- `test_record_includes_event_tags`
- `test_record_swallows_exception_from_event_create`
- `test_flush_is_noop`
- `test_missing_datadog_raises_configuration_error_with_extra_detail`

### Test detail — `ToolInvoker` redactor wiring

`tests/unit/tools/test_invoker.py` (extend, +1 test):
- `test_invoker_threads_redactor_into_audit_record`: Pass a spy redactor to `ToolInvoker(redactor=spy)`; invoke a tool; assert spy was called with the policy-decision payload before the audit sink received it.

### Test detail — DI wiring

`tests/unit/di/test_module.py` (extend, +3 tests):
- `test_provide_payload_redactor_off_returns_identity`
- `test_provide_audit_sink_sentry_without_dsn_raises_configuration_error`
- `test_provide_audit_sink_datadog_with_api_key_returns_datadog_sink` (mocks `datadog` import)

### Reusable fakes

- A small `_FakeSentrySdk` with `init(*args, **kwargs)`, `capture_event(event)`, `flush(timeout)` recording call args. Lives in `test_sentry_sink.py` (not promoted to conftest).
- A `_FakeDatadog` with `initialize(...)`, `api.Event.create(...)`. Local to `test_datadog_sink.py`.

### Risk register

| Risk | Mitigation |
|---|---|
| `sentry-sdk` 3.x changes the `init` / `capture_event` API | Pin `<3.0`; tests use `monkeypatch` against function names — breakage surfaces in CI. |
| `datadog` SDK split / replaced by `ddtrace` | Pin `<1.0`; switch to direct HTTP API in a follow-up if needed. |
| Regex false-positives (e.g. timestamp redacted as IPv4) | Documented in test comments; users opt in via `redaction_profile`. False positives in audit logs are far better than PII leaks. |
| Luhn check rejects valid cards in obscure formats | Standard 13-19 digit formats covered (Visa/MC/Amex/Discover). Edge formats out of scope. |
| Sentry network outage causes audit pipeline to back up | Sink swallows + logs warning; never blocks call site. Verified by exception-swallow test. |
| `sentry_sdk.capture_event` is sync; called from async sink | `capture_event` is non-blocking — queued by SDK transport. No `asyncio.to_thread` needed. |
| Datadog `Event.create` is also sync HTTP | Same: SDK queues internally; we accept the per-call overhead since audit volume is low. |
| `datadog.initialize` global state pollutes tests | Each test resets `monkeypatch.setattr("datadog.api.Event.create", spy)` in setup; no real calls. |

### End-of-phase smoke gate

- Full `pytest tests/unit tests/component` green (≥ 410 passing — 364 baseline + ~46 new).
- `ruff check src tests` total ≤ 211 (no new vs `4c530d2`).
- `mypy src --strict` total ≤ 21.
- All 28 canonical names import.
- `pip install -e ".[sentry]"` works in a clean venv (manual; doesn't run in unit tests).
- `pip install -e ".[datadog]"` works.
- `eaap init smoke-app` still produces a working scaffold (no regressions).

### Coverage target

≥85% on new code (`audit/redaction/`, `audit/sentry.py`, `audit/datadog.py`). Existing coverage must not regress.

## Implementation order (bottom-up)

| Step | Deliverable | Tests | Dependencies |
|---|---|---|---|
| 1 | Redaction layer (3 classes + DI binding + ToolInvoker integration) | ~24 tests | none |
| 2 | SentryAuditSink + settings + DI wiring + `[sentry]` extra | ~7 tests | none (parallel-safe with step 1) |
| 3 | DatadogAuditSink + settings + DI wiring + `[datadog]` extra | ~7 tests | none |
| 4 | End-of-phase smoke gate | full pytest + ruff + mypy + manual extras-install | all |

Tasks 2 and 3 are independent of Task 1 (sinks consume already-redacted records but don't depend on the redactor classes existing — `IAuditSink.record` accepts whatever payload reaches it). Sequential ordering is for predictable review cadence.

## Constraints — recap

- 4 implementation steps; per-step gate is ruff (no new) + mypy strict (touched files) + pytest unit/component.
- Project mypy total stays ≤ 21.
- Project ruff total stays ≤ 211.
- End-of-phase smoke gate must pass before merge.
- New optional deps live ONLY under `[project.optional-dependencies]`. Default `pip install` footprint unchanged.
- One new private subpackage `audit/redaction/`; two new top-level audit modules (`sentry.py`, `datadog.py`); no other new top-level subpackages.
- Phase 6 introduces ONE new `error_code`: `"config.optional_dep_missing"`.
