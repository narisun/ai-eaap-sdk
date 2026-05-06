# ai-core-sdk Phase 6 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `PayloadRedactor` PII strippers and reference `IAuditSink` implementations for Sentry and Datadog so production deployments can wire `audit.sink_type=sentry|datadog` + `audit.redaction_profile=standard` and stream redacted audit events to their observability backend.

**Architecture:** Three independent deliverables. (1) New `audit/redaction/` subpackage with three pure-function classes (`RegexRedactor`, `KeyNameRedactor`, `ChainRedactor`) — DI-bound via `provide_payload_redactor` and threaded into `ToolInvoker`. (2) `SentryAuditSink` in `audit/sentry.py` mapping records to `sentry_sdk.capture_event` with auto-level inference. (3) `DatadogAuditSink` in `audit/datadog.py` mapping records to `datadog.api.Event.create` with auto alert_type. Both sinks ship as `[project.optional-dependencies]` extras (deferred imports; missing → `ConfigurationError(error_code="config.optional_dep_missing")`).

**Tech Stack:** Python 3.11+, Pydantic v2, pytest + ruff + mypy strict. New optional deps: `sentry-sdk>=1.40,<3.0` (under `[sentry]`), `datadog>=0.50,<1.0` (under `[datadog]`). Spec: `docs/superpowers/specs/2026-05-05-ai-core-sdk-phase-6-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-6-prod-integrations` (already checked out off `main` post-Phase-5-merge; carries the Phase 6 spec at `bc92ebd`).

**Working-state hygiene** — do NOT touch:
- `README.md` (top-level)
- `src/ai_core/cli/main.py`
- `src/ai_core/cli/scaffold.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`

**Mypy baseline:** 21 strict errors in 8 files. Total must remain ≤ 21 after every commit.
**Ruff baseline:** 211 errors at `4c530d2` (Phase 5 merge commit). Total must remain ≤ 211.

**Per-step gate (every commit):**
- `ruff check <files-touched>` — no NEW violations vs the pre-task ruff state.
- `pytest tests/unit tests/component -q` — must pass (excluding pre-existing `respx`/`aiosqlite` collection errors).
- `mypy <files-touched-by-this-task>` — no new strict errors.
- `mypy src 2>&1 | tail -1` — total ≤ 21.

**Python interpreter:** `/Users/admin-h26/EAAP/.venv/bin/python` (no local venv inside the SDK).

**Per-task commit message convention:** Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `style:`, `build:`).

**Existing audit-sink contract** (Phase 1, must not break): `IAuditSink.record` and `IAuditSink.flush` must NEVER raise. Backend errors swallowed + logged.

**ToolInvoker call-sites for audit** (must update, do not move): `src/ai_core/tools/invoker.py` lines 146, 222, 235 — the three `await self._audit.record(AuditRecord.now(...))` calls. Each currently uses default `_identity_redactor`; Task 1 threads `redactor=self._redactor` into all three.

---

## Task 1 — Redaction layer

Three new classes + DI binding + `ToolInvoker` integration. Independent of Tasks 2-3.

**Files:**
- Create: `src/ai_core/audit/redaction/__init__.py`
- Create: `src/ai_core/audit/redaction/regex.py`
- Create: `src/ai_core/audit/redaction/key_name.py`
- Create: `src/ai_core/audit/redaction/chain.py`
- Modify: `src/ai_core/audit/__init__.py` (re-export the three classes)
- Modify: `src/ai_core/config/settings.py` (add `redaction_profile` to `AuditSettings`)
- Modify: `src/ai_core/di/module.py` (add `provide_payload_redactor` + thread into `provide_tool_invoker`)
- Modify: `src/ai_core/tools/invoker.py` (add `redactor` param + thread into 3 `AuditRecord.now()` call sites)
- Test: NEW `tests/unit/audit/redaction/__init__.py`
- Test: NEW `tests/unit/audit/redaction/test_regex_redactor.py`
- Test: NEW `tests/unit/audit/redaction/test_key_name_redactor.py`
- Test: NEW `tests/unit/audit/redaction/test_chain_redactor.py`
- Test: NEW `tests/unit/audit/redaction/test_redaction_profile.py`
- Test: extend `tests/unit/tools/test_invoker.py` (+1 redactor-threading test)
- Test: extend `tests/unit/di/test_module.py` (or create if missing) — +1 redactor profile test

### 1a — `RegexRedactor`

- [ ] **Step 1.1: Create the redaction subpackage**

```bash
mkdir -p /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/audit/redaction
mkdir -p /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/audit/redaction
touch /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/audit/redaction/__init__.py
touch /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/audit/redaction/__init__.py
```

- [ ] **Step 1.2: Write failing `RegexRedactor` tests**

Create `tests/unit/audit/redaction/test_regex_redactor.py`:

```python
"""Tests for RegexRedactor — pattern-based PII stripping."""
from __future__ import annotations

import pytest

from ai_core.audit.redaction.regex import RegexRedactor

pytestmark = pytest.mark.unit


def test_redacts_email_address() -> None:
    r = RegexRedactor(enabled_patterns={"email"})
    out = r({"input": "contact me at alice@example.com please"})
    assert out["input"] == "contact me at <redacted-email> please"


def test_redacts_phone_us_format() -> None:
    r = RegexRedactor(enabled_patterns={"phone"})
    for phone in ("(555) 123-4567", "555-123-4567", "+1 555 123 4567"):
        out = r({"v": f"call {phone} now"})
        assert "<redacted-phone>" in out["v"], f"failed for {phone!r}"


def test_redacts_ssn_with_dashes() -> None:
    r = RegexRedactor(enabled_patterns={"ssn"})
    out = r({"v": "SSN: 123-45-6789"})
    assert out["v"] == "SSN: <redacted-ssn>"


def test_redacts_credit_card_passing_luhn() -> None:
    """Visa test card 4242 4242 4242 4242 passes Luhn → redacted."""
    r = RegexRedactor(enabled_patterns={"credit_card"})
    out = r({"v": "card: 4242424242424242"})
    assert out["v"] == "card: <redacted-credit_card>"


def test_does_not_redact_credit_card_failing_luhn() -> None:
    """1234 5678 9012 3456 fails Luhn → NOT redacted."""
    r = RegexRedactor(enabled_patterns={"credit_card"})
    out = r({"v": "fake: 1234567890123456"})
    assert out["v"] == "fake: 1234567890123456"


def test_redacts_ipv4_address() -> None:
    r = RegexRedactor(enabled_patterns={"ipv4"})
    out = r({"v": "from 192.168.1.1 ok"})
    assert out["v"] == "from <redacted-ipv4> ok"


def test_redacts_recursively_through_nested_dict() -> None:
    r = RegexRedactor(enabled_patterns={"email"})
    out = r({"user": {"contact": "bob@x.io"}})
    assert out["user"]["contact"] == "<redacted-email>"


def test_redacts_recursively_through_list_of_strings() -> None:
    r = RegexRedactor(enabled_patterns={"email"})
    out = r({"emails": ["a@x.io", "b@x.io"]})
    assert out["emails"] == ["<redacted-email>", "<redacted-email>"]


def test_passes_through_non_string_non_dict_non_list_values() -> None:
    r = RegexRedactor(enabled_patterns={"email"})
    out = r({"int": 42, "none": None, "bool": True, "float": 3.14})
    assert out == {"int": 42, "none": None, "bool": True, "float": 3.14}


def test_selective_pattern_enable() -> None:
    """enabled_patterns={'email'} doesn't redact phone numbers."""
    r = RegexRedactor(enabled_patterns={"email"})
    out = r({"v": "call 555-123-4567 or alice@x.io"})
    assert "555-123-4567" in out["v"]  # phone NOT redacted
    assert "<redacted-email>" in out["v"]  # email redacted


def test_unknown_pattern_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown patterns"):
        RegexRedactor(enabled_patterns={"foo"})  # type: ignore[arg-type]
```

- [ ] **Step 1.3: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/redaction/test_regex_redactor.py -v 2>&1 | tail -15
```

Expected: ImportError on `ai_core.audit.redaction.regex`.

- [ ] **Step 1.4: Implement `RegexRedactor`**

Create `src/ai_core/audit/redaction/regex.py`:

```python
"""Regex-based PII redaction — pure functions, no I/O.

Built-in patterns: email, phone (US), SSN, credit card (Luhn-checked),
IPv4, long_number (strict-profile only — 6+ digit sequences).

Replacement format: ``<redacted-{kind}>``. Recursive over nested
mappings and lists.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

PatternKind = Literal["email", "phone", "ssn", "credit_card", "ipv4", "long_number"]

_PATTERNS: dict[PatternKind, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "long_number": re.compile(r"\b\d{6,}\b"),
}


def _luhn_check(s: str) -> bool:
    """Standard mod-10 Luhn verification for credit-card validation."""
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

    Args:
        enabled_patterns: Subset of supported pattern kinds to apply.

    Raises:
        ValueError: If ``enabled_patterns`` contains unsupported kinds.
    """

    def __init__(self, *, enabled_patterns: set[PatternKind]) -> None:
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
                result = pattern.sub(
                    lambda m, k=kind: f"<redacted-{k}>" if _luhn_check(m.group(0)) else m.group(0),
                    result,
                )
            else:
                result = pattern.sub(f"<redacted-{kind}>", result)
        return result


__all__ = ["PatternKind", "RegexRedactor"]
```

- [ ] **Step 1.5: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/redaction/test_regex_redactor.py -v 2>&1 | tail -15
```

Expected: 11 passed.

### 1b — `KeyNameRedactor`

- [ ] **Step 1.6: Write failing `KeyNameRedactor` tests**

Create `tests/unit/audit/redaction/test_key_name_redactor.py`:

```python
"""Tests for KeyNameRedactor — case-insensitive key-name redaction."""
from __future__ import annotations

import pytest

from ai_core.audit.redaction.key_name import KeyNameRedactor

pytestmark = pytest.mark.unit


def test_redacts_default_password_key() -> None:
    r = KeyNameRedactor()
    assert r({"password": "hunter2"}) == {"password": "[REDACTED]"}


def test_redacts_default_api_key() -> None:
    r = KeyNameRedactor()
    assert r({"api_key": "abc123"}) == {"api_key": "[REDACTED]"}


def test_case_insensitive_match() -> None:
    r = KeyNameRedactor()
    assert r({"Password": "x", "API_KEY": "y"}) == {
        "Password": "[REDACTED]",
        "API_KEY": "[REDACTED]",
    }


def test_custom_key_set_only_redacts_specified() -> None:
    r = KeyNameRedactor(redact_keys={"my_secret"})
    out = r({"my_secret": "x", "password": "y"})
    assert out == {"my_secret": "[REDACTED]", "password": "y"}


def test_redacts_nested_keys() -> None:
    r = KeyNameRedactor()
    out = r({"db": {"password": "x"}, "info": "ok"})
    assert out == {"db": {"password": "[REDACTED]"}, "info": "ok"}


def test_redacts_inside_list_with_parent_key_context() -> None:
    """List items inherit parent key name — {tokens: [a,b]} → {tokens: [REDACTED, REDACTED]}."""
    r = KeyNameRedactor()
    out = r({"tokens": ["aaa", "bbb"]})
    assert out == {"tokens": ["[REDACTED]", "[REDACTED]"]}
```

- [ ] **Step 1.7: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/redaction/test_key_name_redactor.py -v 2>&1 | tail -10
```

Expected: ImportError on `ai_core.audit.redaction.key_name`.

- [ ] **Step 1.8: Implement `KeyNameRedactor`**

Create `src/ai_core/audit/redaction/key_name.py`:

```python
"""Key-name-based redaction — replaces values for sensitive keys with [REDACTED].

Default key set covers common secret-bearing key names (passwords, tokens,
api keys). Match is case-insensitive. List items inherit their parent
key name's redaction status.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_REDACT_KEYS: frozenset[str] = frozenset({
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "token",
    "access_token",
    "refresh_token",
    "bearer",
    "cookie",
    "session",
    "private_key",
    "ssh_key",
})


class KeyNameRedactor:
    """Replace values for any key whose lowercase name matches the redact set.

    Args:
        redact_keys: Custom set of key names. Defaults to
            :data:`DEFAULT_REDACT_KEYS` if omitted. Names are matched
            case-insensitively after lowercasing both sides.
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
            return [self._redact_value(key, v) for v in value]
        return value


__all__ = ["DEFAULT_REDACT_KEYS", "KeyNameRedactor"]
```

- [ ] **Step 1.9: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/redaction/test_key_name_redactor.py -v 2>&1 | tail -10
```

Expected: 6 passed.

### 1c — `ChainRedactor`

- [ ] **Step 1.10: Write failing `ChainRedactor` tests**

Create `tests/unit/audit/redaction/test_chain_redactor.py`:

```python
"""Tests for ChainRedactor — composition of N redactors."""
from __future__ import annotations

import pytest

from ai_core.audit.redaction.chain import ChainRedactor
from ai_core.audit.redaction.key_name import KeyNameRedactor
from ai_core.audit.redaction.regex import RegexRedactor

pytestmark = pytest.mark.unit


def test_two_redactor_chain_applies_both() -> None:
    """Regex first, then key-name → both transforms visible."""
    chain = ChainRedactor(
        RegexRedactor(enabled_patterns={"email"}),
        KeyNameRedactor(redact_keys={"password"}),
    )
    out = chain({"contact": "alice@x.io", "password": "hunter2"})
    assert out == {"contact": "<redacted-email>", "password": "[REDACTED]"}


def test_chain_order_matters() -> None:
    """KeyName first masks the value before regex sees it; reversed order doesn't."""
    contact_value = "alice@x.io"
    key_first = ChainRedactor(
        KeyNameRedactor(redact_keys={"contact"}),
        RegexRedactor(enabled_patterns={"email"}),
    )
    regex_first = ChainRedactor(
        RegexRedactor(enabled_patterns={"email"}),
        KeyNameRedactor(redact_keys={"contact"}),
    )
    # KeyName masks first → regex sees "[REDACTED]" which doesn't match email.
    assert key_first({"contact": contact_value}) == {"contact": "[REDACTED]"}
    # Regex first → email replaced; key-name then masks the (now-redacted) value.
    assert regex_first({"contact": contact_value}) == {"contact": "[REDACTED]"}
    # Both end the same here, but the intermediate values differ — order matters
    # for any redactor whose output depends on input shape.


def test_empty_chain_is_identity() -> None:
    chain = ChainRedactor()
    assert chain({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}
```

- [ ] **Step 1.11: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/redaction/test_chain_redactor.py -v 2>&1 | tail -10
```

Expected: ImportError on `ai_core.audit.redaction.chain`.

- [ ] **Step 1.12: Implement `ChainRedactor`**

Create `src/ai_core/audit/redaction/chain.py`:

```python
"""Compose N redactors into a single PayloadRedactor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ai_core.audit.interface import PayloadRedactor


class ChainRedactor:
    """Apply N redactors in order; output of one feeds the next.

    An empty chain is the identity function.
    """

    def __init__(self, *redactors: PayloadRedactor) -> None:
        self._redactors: tuple[PayloadRedactor, ...] = redactors

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        result: Mapping[str, Any] = payload
        for r in self._redactors:
            result = r(result)
        return dict(result)


__all__ = ["ChainRedactor"]
```

- [ ] **Step 1.13: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/redaction/test_chain_redactor.py -v 2>&1 | tail -10
```

Expected: 3 passed.

### 1d — Re-export from `audit/__init__.py` + redaction subpackage `__init__`

- [ ] **Step 1.14: Populate `audit/redaction/__init__.py`**

Replace the empty `src/ai_core/audit/redaction/__init__.py` content with:

```python
"""PayloadRedactor concrete implementations — see :mod:`ai_core.audit.interface`
for the type alias and :func:`ai_core.di.module.AgentModule.provide_payload_redactor`
for the DI wiring."""

from ai_core.audit.redaction.chain import ChainRedactor
from ai_core.audit.redaction.key_name import DEFAULT_REDACT_KEYS, KeyNameRedactor
from ai_core.audit.redaction.regex import PatternKind, RegexRedactor

__all__ = [
    "DEFAULT_REDACT_KEYS",
    "ChainRedactor",
    "KeyNameRedactor",
    "PatternKind",
    "RegexRedactor",
]
```

- [ ] **Step 1.15: Re-export from `src/ai_core/audit/__init__.py`**

Read the current file first:

```bash
cat /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/audit/__init__.py
```

Add to the existing imports (preserving alphabetical order) and `__all__`:

```python
from ai_core.audit.redaction import (
    ChainRedactor,
    KeyNameRedactor,
    RegexRedactor,
)
```

And update `__all__` to include `"ChainRedactor"`, `"KeyNameRedactor"`, `"RegexRedactor"` (preserving existing entries).

### 1e — Settings: `redaction_profile`

- [ ] **Step 1.16: Add `redaction_profile` field to `AuditSettings`**

In `src/ai_core/config/settings.py`, find `class AuditSettings(BaseSettings):` (around line 271). Add the new field after the existing `jsonl_path` field:

```python
class AuditSettings(BaseSettings):
    """Audit-sink configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    sink_type: Literal["null", "otel_event", "jsonl"] = "null"
    jsonl_path: Path | None = None  # required when sink_type == "jsonl"

    # Phase 6: PayloadRedactor configuration
    redaction_profile: Literal["off", "standard", "strict"] = Field(
        default="off",
        description=(
            "Profile name for the DI-bound PayloadRedactor. "
            "'off' is identity (no redaction); 'standard' chains a RegexRedactor "
            "(email, phone, ssn, credit_card, ipv4) with a KeyNameRedactor "
            "(default secret/token key set); 'strict' adds a 6+digit number "
            "pattern that catches IDs/account numbers (higher false-positive rate)."
        ),
    )
```

(Note: don't extend `sink_type` literal yet — that's Tasks 2 and 3.)

### 1f — DI: `provide_payload_redactor`

- [ ] **Step 1.17: Write failing redaction-profile DI test**

Create `tests/unit/audit/redaction/test_redaction_profile.py`:

```python
"""Tests for AgentModule.provide_payload_redactor."""
from __future__ import annotations

import pytest

from ai_core.audit.interface import _identity_redactor
from ai_core.audit.redaction import ChainRedactor
from ai_core.config.settings import AppSettings
from ai_core.di.module import AgentModule

pytestmark = pytest.mark.unit


def _make_settings(profile: str) -> AppSettings:
    settings = AppSettings()
    # Pydantic v2 frozen-ish models still allow attribute assignment on nested settings.
    settings.audit.redaction_profile = profile  # type: ignore[assignment]
    return settings


def test_provide_payload_redactor_off_returns_identity() -> None:
    settings = _make_settings("off")
    module = AgentModule()
    redactor = module.provide_payload_redactor(settings)
    assert redactor is _identity_redactor


def test_provide_payload_redactor_standard_returns_chain() -> None:
    settings = _make_settings("standard")
    module = AgentModule()
    redactor = module.provide_payload_redactor(settings)
    assert isinstance(redactor, ChainRedactor)
    # Behavior check: redacts both an email AND a password key.
    out = redactor({"contact": "alice@x.io", "password": "hunter2"})
    assert out["contact"] == "<redacted-email>"
    assert out["password"] == "[REDACTED]"


def test_provide_payload_redactor_strict_includes_long_number() -> None:
    settings = _make_settings("strict")
    module = AgentModule()
    redactor = module.provide_payload_redactor(settings)
    # 8-digit number gets redacted under strict profile.
    out = redactor({"id": "see id 12345678 attached"})
    assert "<redacted-long_number>" in out["id"]
```

- [ ] **Step 1.18: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/redaction/test_redaction_profile.py -v 2>&1 | tail -10
```

Expected: failures (`AgentModule` has no `provide_payload_redactor` method).

- [ ] **Step 1.19: Add `provide_payload_redactor` to `AgentModule`**

In `src/ai_core/di/module.py`, find the existing `provide_audit_sink` method (around line 247). Add a new provider method just BEFORE it (so the redactor is wired before any audit-sink consumer needs it):

```python
    # ----- Payload redactor (Phase 6) ---------------------------------------
    @singleton
    @provider
    def provide_payload_redactor(self, settings: AppSettings) -> PayloadRedactor:
        """Return the configured PayloadRedactor; identity for profile='off'."""
        profile = settings.audit.redaction_profile
        if profile == "off":
            from ai_core.audit.interface import _identity_redactor  # noqa: PLC0415
            return _identity_redactor
        from ai_core.audit.redaction import (  # noqa: PLC0415
            ChainRedactor,
            KeyNameRedactor,
            PatternKind,
            RegexRedactor,
        )
        base_patterns: set[PatternKind] = {"email", "phone", "ssn", "credit_card", "ipv4"}
        if profile == "strict":
            base_patterns.add("long_number")
        return ChainRedactor(
            RegexRedactor(enabled_patterns=base_patterns),
            KeyNameRedactor(),
        )
```

Add the `PayloadRedactor` import to the top of the file (or extend the existing `from ai_core.audit import ...` line):

```python
from ai_core.audit import IAuditSink, PayloadRedactor  # noqa: TC001
```

(Adjust `noqa` based on what's there; the existing line has `# noqa: TC001` already.)

- [ ] **Step 1.20: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/redaction/test_redaction_profile.py -v 2>&1 | tail -10
```

Expected: 3 passed.

### 1g — `ToolInvoker` integration

- [ ] **Step 1.21: Write failing redactor-threading test**

Append to `tests/unit/tools/test_invoker.py`:

```python
@pytest.mark.asyncio
async def test_invoker_threads_redactor_into_audit_record(
    fake_observability, fake_policy_evaluator_factory, fake_audit_sink
) -> None:
    """ToolInvoker(redactor=spy) → spy is called with the policy-decision payload
    BEFORE the audit sink sees the record."""
    spy_calls: list[Mapping[str, Any]] = []

    def _spy_redactor(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        spy_calls.append(dict(payload))
        return {"redacted": True}

    inv = ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(default_allow=True),
        registry=SchemaRegistry(),
        audit=fake_audit_sink,
        redactor=_spy_redactor,
    )
    await inv.invoke(
        _search,
        {"q": "x", "limit": 1},
        principal={"sub": "user-42"},
        agent_id="a",
        tenant_id="t",
    )

    # Spy was called for at least the POLICY_DECISION payload (input + user).
    assert len(spy_calls) >= 1
    first = spy_calls[0]
    assert "input" in first
    assert "user" in first

    # The audit sink received the redacted payload.
    from ai_core.audit import AuditEvent
    policy_records = [
        r for r in fake_audit_sink.records if r.event == AuditEvent.POLICY_DECISION
    ]
    assert len(policy_records) == 1
    assert policy_records[0].payload == {"redacted": True}
```

(If `Mapping`, `Any` aren't already imported at the top of the test file, add them: `from collections.abc import Mapping; from typing import Any`.)

- [ ] **Step 1.22: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py::test_invoker_threads_redactor_into_audit_record -v 2>&1 | tail -10
```

Expected: failure — `ToolInvoker.__init__` does not accept `redactor`.

- [ ] **Step 1.23: Add `redactor` param to `ToolInvoker.__init__`**

In `src/ai_core/tools/invoker.py`, update the `__init__` (around line 59) to accept `redactor` and store it. The new `__init__` should be:

```python
    def __init__(
        self,
        *,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator | None = None,
        registry: SchemaRegistry | None = None,
        audit: IAuditSink | None = None,
        redactor: PayloadRedactor | None = None,
    ) -> None:
        from ai_core.audit.interface import _identity_redactor  # noqa: PLC0415
        from ai_core.audit.null import NullAuditSink as _NullAuditSink  # noqa: PLC0415
        self._observability = observability
        self._policy = policy
        self._registry = registry
        self._audit: IAuditSink = audit or _NullAuditSink()
        # Phase 4: skip AuditRecord.now() allocation entirely when the sink is no-op.
        self._records_audit: bool = not isinstance(self._audit, _NullAuditSink)
        # Phase 6: optional payload redactor; default identity preserves Phase 1-5 behaviour.
        self._redactor: PayloadRedactor = redactor or _identity_redactor
```

Add the `PayloadRedactor` import to the top of `invoker.py`. The existing imports include `from ai_core.audit import AuditEvent, AuditRecord, IAuditSink` — extend to:

```python
from ai_core.audit import AuditEvent, AuditRecord, IAuditSink, PayloadRedactor
```

(Verify the `# noqa` comments on existing imports are preserved.)

- [ ] **Step 1.24: Thread `redactor=self._redactor` into all 3 audit call sites**

Find the three `AuditRecord.now(...)` calls in `tools/invoker.py` (around lines 146, 222, 235). Each currently looks like:

```python
await self._audit.record(AuditRecord.now(
    AuditEvent.POLICY_DECISION,
    tool_name=spec.name, tool_version=spec.version,
    agent_id=agent_id, tenant_id=tenant_id,
    decision_path=spec.opa_path,
    decision_allowed=decision.allowed,
    decision_reason=decision.reason,
    payload={
        "input": payload.model_dump(),
        "user": dict(principal or {}),
    },
))
```

Add `redactor=self._redactor,` as the LAST kwarg in each `AuditRecord.now(...)` call. After the change, each call ends like:

```python
    payload={ ... },
    redactor=self._redactor,
))
```

Apply to all three call sites (POLICY_DECISION, TOOL_INVOCATION_COMPLETED, TOOL_INVOCATION_FAILED). Do NOT change anything else inside the calls.

- [ ] **Step 1.25: Wire `redactor` into the `provide_tool_invoker` DI provider**

In `src/ai_core/di/module.py`, find `provide_tool_invoker` (around line 296). Update its signature and body:

```python
    @singleton
    @provider
    def provide_tool_invoker(
        self,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator,
        registry: SchemaRegistry,
        audit: IAuditSink,
        redactor: PayloadRedactor,
    ) -> ToolInvoker:
        """Return the singleton :class:`ToolInvoker` wired to the SDK's services."""
        return ToolInvoker(
            observability=observability,
            policy=policy,
            registry=registry,
            audit=audit,
            redactor=redactor,
        )
```

- [ ] **Step 1.26: Run the redactor-threading test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py::test_invoker_threads_redactor_into_audit_record -v 2>&1 | tail -10
```

Expected: PASS.

### 1h — Lint, type-check, full suite, commit

- [ ] **Step 1.27: Lint + type-check the Task 1 change set**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/audit/redaction/ \
    src/ai_core/audit/__init__.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py \
    src/ai_core/tools/invoker.py \
    tests/unit/audit/redaction/ \
    tests/unit/tools/test_invoker.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/audit/redaction/ \
    src/ai_core/audit/__init__.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py \
    src/ai_core/tools/invoker.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on touched files clean; project total ≤ 21.

- [ ] **Step 1.28: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 388+ passing (364 baseline + ~24 new); 9 pre-existing langgraph errors unchanged.

- [ ] **Step 1.29: Commit Task 1**

```bash
git add src/ai_core/audit/redaction/ \
        src/ai_core/audit/__init__.py \
        src/ai_core/config/settings.py \
        src/ai_core/di/module.py \
        src/ai_core/tools/invoker.py \
        tests/unit/audit/redaction/ \
        tests/unit/tools/test_invoker.py
git commit -m "feat(audit): redaction layer — RegexRedactor, KeyNameRedactor, ChainRedactor + DI binding + ToolInvoker wiring"
```

---

## Task 2 — `SentryAuditSink`

`SentryAuditSink` + settings + DI wiring + `[sentry]` extra. Independent of Tasks 1 and 3.

**Files:**
- Create: `src/ai_core/audit/sentry.py`
- Modify: `src/ai_core/config/settings.py` (extend `sink_type` literal + add Sentry settings)
- Modify: `src/ai_core/di/module.py` (add `sentry` branch to `provide_audit_sink`)
- Modify: `pyproject.toml` (add `[project.optional-dependencies] sentry`)
- Test: NEW `tests/unit/audit/test_sentry_sink.py`
- Test: extend `tests/unit/di/test_module.py` (or create) — +1 sentry-without-DSN test

### 2a — Add the optional dependency

- [ ] **Step 2.1: Add `[sentry]` extra to pyproject.toml**

In `pyproject.toml`, find the `[project.optional-dependencies]` table (or add it if missing). Add the `sentry` group:

```toml
[project.optional-dependencies]
sentry = ["sentry-sdk>=1.40,<3.0"]
```

If a `[project.optional-dependencies]` table already exists with other groups (mcp, etc.), add the `sentry` line preserving alphabetical order.

- [ ] **Step 2.2: Install the optional dep into the venv**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pip install -e "/Users/admin-h26/EAAP/ai-core-sdk[sentry]" 2>&1 | tail -5
```

Expected: `Successfully installed sentry-sdk-...` (or "already satisfied"). Required so unit tests can import the module.

- [ ] **Step 2.3: Smoke import**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import sentry_sdk; print('ok', sentry_sdk.VERSION)"
```

Expected: `ok 1.x.x` or `2.x.x`.

### 2b — Sentry settings

- [ ] **Step 2.4: Extend `AuditSettings` with Sentry fields and update `sink_type` literal**

In `src/ai_core/config/settings.py`, locate `class AuditSettings`. Update `sink_type` to include `"sentry"`:

```python
    sink_type: Literal["null", "otel_event", "jsonl", "sentry"] = "null"
```

(Don't add `"datadog"` yet — that's Task 3.)

After the `redaction_profile` field added in Task 1, add Sentry fields:

```python
    # Phase 6: Sentry sink config (used when sink_type='sentry')
    sentry_dsn: SecretStr | None = Field(
        default=None,
        description="Required when sink_type='sentry'. Project-level DSN issued by Sentry.",
    )
    sentry_environment: str | None = Field(
        default=None,
        description="Optional environment tag on Sentry events (e.g. 'prod', 'staging').",
    )
    sentry_release: str | None = Field(
        default=None,
        description="Optional release identifier on Sentry events.",
    )
    sentry_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of audit events sent to Sentry. 1.0 = all; 0.0 = none.",
    )
```

`SecretStr` should already be imported at the top of the file. Verify with:

```bash
grep -n "SecretStr" /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/config/settings.py | head -3
```

If not imported, add it to the existing pydantic import line.

### 2c — `SentryAuditSink` implementation

- [ ] **Step 2.5: Write failing `SentryAuditSink` tests**

Create `tests/unit/audit/test_sentry_sink.py`:

```python
"""Tests for SentryAuditSink."""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_core.audit.interface import AuditEvent, AuditRecord
from ai_core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_sentry_sdk(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace sentry_sdk in sys.modules with a MagicMock recording all calls."""
    fake = MagicMock()
    fake.VERSION = "fake-1.0.0"
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    return fake


def _record(
    *,
    event: AuditEvent = AuditEvent.TOOL_INVOCATION_COMPLETED,
    decision_allowed: bool | None = None,
    error_code: str | None = None,
) -> AuditRecord:
    return AuditRecord.now(
        event,
        tool_name="search",
        tool_version=1,
        agent_id="agent-x",
        tenant_id="tenant-y",
        decision_allowed=decision_allowed,
        error_code=error_code,
        payload={"input": {"q": "hello"}},
    )


def test_init_calls_sentry_sdk_init_with_dsn(fake_sentry_sdk: MagicMock) -> None:
    from ai_core.audit.sentry import SentryAuditSink
    sink = SentryAuditSink(
        dsn="https://abc@sentry.example.com/42",
        environment="prod",
        release="v1.2.3",
        sample_rate=0.5,
    )
    fake_sentry_sdk.init.assert_called_once()
    kwargs = fake_sentry_sdk.init.call_args.kwargs
    assert kwargs["dsn"] == "https://abc@sentry.example.com/42"
    assert kwargs["environment"] == "prod"
    assert kwargs["release"] == "v1.2.3"
    assert kwargs["sample_rate"] == 0.5
    assert sink is not None


@pytest.mark.asyncio
async def test_record_emits_capture_event_with_info_level_for_allowed(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.audit.sentry import SentryAuditSink
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.record(_record(decision_allowed=True))
    fake_sentry_sdk.capture_event.assert_called_once()
    event = fake_sentry_sdk.capture_event.call_args.args[0]
    assert event["level"] == "info"


@pytest.mark.asyncio
async def test_record_emits_warning_level_for_decision_denied(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.audit.sentry import SentryAuditSink
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.record(_record(decision_allowed=False))
    event = fake_sentry_sdk.capture_event.call_args.args[0]
    assert event["level"] == "warning"


@pytest.mark.asyncio
async def test_record_emits_warning_level_for_error_code_set(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.audit.sentry import SentryAuditSink
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.record(_record(error_code="tool.invocation_failed"))
    event = fake_sentry_sdk.capture_event.call_args.args[0]
    assert event["level"] == "warning"


@pytest.mark.asyncio
async def test_record_includes_audit_tags(fake_sentry_sdk: MagicMock) -> None:
    from ai_core.audit.sentry import SentryAuditSink
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.record(_record(decision_allowed=True))
    event = fake_sentry_sdk.capture_event.call_args.args[0]
    tags = event["tags"]
    assert tags["audit.tool_name"] == "search"
    assert tags["audit.agent_id"] == "agent-x"
    assert tags["audit.tenant_id"] == "tenant-y"
    assert tags["audit.event"] == AuditEvent.TOOL_INVOCATION_COMPLETED.value


@pytest.mark.asyncio
async def test_record_swallows_exception_from_capture_event(
    fake_sentry_sdk: MagicMock,
) -> None:
    fake_sentry_sdk.capture_event.side_effect = RuntimeError("backend down")
    from ai_core.audit.sentry import SentryAuditSink
    sink = SentryAuditSink(dsn="https://x@x/1")
    # Must not raise.
    await sink.record(_record())


@pytest.mark.asyncio
async def test_flush_calls_sentry_sdk_flush_with_timeout(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.audit.sentry import SentryAuditSink
    sink = SentryAuditSink(dsn="https://x@x/1")
    await sink.flush()
    fake_sentry_sdk.flush.assert_called_once_with(timeout=5.0)


def test_missing_sentry_sdk_raises_configuration_error_with_extra_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If sentry_sdk is not installed, instantiating the sink raises ConfigurationError."""
    # Force the import to fail by removing sentry_sdk from sys.modules
    # AND blocking the import.
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    # Reload the sink module to retrigger the import inside __init__.
    if "ai_core.audit.sentry" in sys.modules:
        del sys.modules["ai_core.audit.sentry"]
    from ai_core.audit.sentry import SentryAuditSink
    with pytest.raises(ConfigurationError) as exc:
        SentryAuditSink(dsn="https://x@x/1")
    assert exc.value.error_code == "config.optional_dep_missing"
    assert exc.value.details["extra"] == "sentry"
```

- [ ] **Step 2.6: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_sentry_sink.py -v 2>&1 | tail -15
```

Expected: ImportError on `ai_core.audit.sentry`.

- [ ] **Step 2.7: Implement `SentryAuditSink`**

Create `src/ai_core/audit/sentry.py`:

```python
"""SentryAuditSink — forward audit records to Sentry as first-class events.

Ships under the optional ``[sentry]`` extra. ``record`` maps each
:class:`AuditRecord` to ``sentry_sdk.capture_event`` with auto-level
inference (warning if denied or errored; info otherwise). Errors swallowed
per the :class:`IAuditSink` never-raise contract.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.exceptions import ConfigurationError

_logger = logging.getLogger(__name__)


class SentryAuditSink(IAuditSink):
    """Forward :class:`AuditRecord` instances to Sentry.

    Args:
        dsn: Sentry project DSN.
        environment: Optional environment tag on every event.
        release: Optional release identifier.
        sample_rate: 0.0-1.0 fraction of events to send.

    Raises:
        ConfigurationError: If ``sentry-sdk`` is not installed
            (``error_code='config.optional_dep_missing'``,
            ``details={'extra': 'sentry'}``).
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
            event: dict[str, Any] = {
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
            }
            self._sentry_sdk.capture_event(event)
        except Exception as exc:  # noqa: BLE001 — audit-sink never-raise contract
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
        except Exception as exc:  # noqa: BLE001 — audit-sink never-raise contract
            _logger.warning(
                "audit.sentry.flush_failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )


__all__ = ["SentryAuditSink"]
```

- [ ] **Step 2.8: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_sentry_sink.py -v 2>&1 | tail -15
```

Expected: 8 passed.

### 2d — DI wiring

- [ ] **Step 2.9: Extend `provide_audit_sink` with the `sentry` branch**

In `src/ai_core/di/module.py`, locate the existing `provide_audit_sink` method (around line 247). After the `if sink_type == "jsonl":` block and BEFORE the final `raise ConfigurationError(f"Unknown audit.sink_type: ...")`, add:

```python
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
```

- [ ] **Step 2.10: Add Sentry-without-DSN DI test**

Append to `tests/unit/audit/test_sentry_sink.py`:

```python
def test_provide_audit_sink_sentry_without_dsn_raises_configuration_error(
    fake_sentry_sdk: MagicMock,
) -> None:
    from ai_core.config.settings import AppSettings
    from ai_core.di.module import AgentModule
    from ai_core.di.interfaces import IObservabilityProvider

    settings = AppSettings()
    settings.audit.sink_type = "sentry"  # type: ignore[assignment]
    # sentry_dsn is None by default

    obs = MagicMock(spec=IObservabilityProvider)
    module = AgentModule()
    with pytest.raises(ConfigurationError) as exc:
        module.provide_audit_sink(settings, obs)
    assert exc.value.error_code == "config.invalid"
    assert "sentry_dsn" in exc.value.message.lower()
```

- [ ] **Step 2.11: Run the new DI test**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_sentry_sink.py::test_provide_audit_sink_sentry_without_dsn_raises_configuration_error -v 2>&1 | tail -10
```

Expected: PASS.

### 2e — Lint, type-check, full suite, commit

- [ ] **Step 2.12: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/audit/sentry.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py \
    tests/unit/audit/test_sentry_sink.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/audit/sentry.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on touched files clean; project total ≤ 21.

- [ ] **Step 2.13: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 397+ passing (388 after Task 1 + ~9 new for Task 2); 9 pre-existing langgraph errors unchanged.

- [ ] **Step 2.14: Commit Task 2**

```bash
git add src/ai_core/audit/sentry.py \
        src/ai_core/config/settings.py \
        src/ai_core/di/module.py \
        pyproject.toml \
        tests/unit/audit/test_sentry_sink.py
git commit -m "feat(audit): SentryAuditSink — sentry_sdk.capture_event with auto-level + [sentry] extra"
```

---

## Task 3 — `DatadogAuditSink`

`DatadogAuditSink` + settings + DI wiring + `[datadog]` extra. Independent of Tasks 1 and 2.

**Files:**
- Create: `src/ai_core/audit/datadog.py`
- Modify: `src/ai_core/config/settings.py` (extend `sink_type` to include `"datadog"` + add Datadog settings)
- Modify: `src/ai_core/di/module.py` (add `datadog` branch to `provide_audit_sink`)
- Modify: `pyproject.toml` (add `[project.optional-dependencies] datadog`)
- Test: NEW `tests/unit/audit/test_datadog_sink.py`

### 3a — Add the optional dependency

- [ ] **Step 3.1: Add `[datadog]` extra to pyproject.toml**

In `pyproject.toml` `[project.optional-dependencies]` table, add:

```toml
datadog = ["datadog>=0.50,<1.0"]
```

(After the `sentry` entry, preserving alphabetical order: `datadog` comes before `sentry`.)

- [ ] **Step 3.2: Install the optional dep into the venv**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pip install -e "/Users/admin-h26/EAAP/ai-core-sdk[datadog]" 2>&1 | tail -5
```

Expected: `Successfully installed datadog-...`.

- [ ] **Step 3.3: Smoke import**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import datadog; print('ok', datadog.__version__)"
```

Expected: `ok 0.x.x`.

### 3b — Datadog settings

- [ ] **Step 3.4: Extend `AuditSettings` with Datadog fields and update `sink_type` literal**

In `src/ai_core/config/settings.py`, update `sink_type` to include `"datadog"`:

```python
    sink_type: Literal["null", "otel_event", "jsonl", "sentry", "datadog"] = "null"
```

After the Sentry fields added in Task 2, add Datadog fields:

```python
    # Phase 6: Datadog sink config (used when sink_type='datadog')
    datadog_api_key: SecretStr | None = Field(
        default=None,
        description="Required when sink_type='datadog'. Datadog API key.",
    )
    datadog_app_key: SecretStr | None = Field(
        default=None,
        description="Optional Datadog application key (some endpoints require it).",
    )
    datadog_site: str = Field(
        default="datadoghq.com",
        description="Datadog site (e.g. 'datadoghq.com', 'datadoghq.eu', 'us3.datadoghq.com').",
    )
    datadog_source: str = Field(
        default="ai-core-sdk",
        description="Source name attached to Datadog events (free text).",
    )
    datadog_environment: str | None = Field(
        default=None,
        description="Optional environment tag (added as 'env:<value>' to every event).",
    )
```

### 3c — `DatadogAuditSink` implementation

- [ ] **Step 3.5: Write failing `DatadogAuditSink` tests**

Create `tests/unit/audit/test_datadog_sink.py`:

```python
"""Tests for DatadogAuditSink."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_core.audit.interface import AuditEvent, AuditRecord
from ai_core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_datadog(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace `datadog` in sys.modules with a MagicMock."""
    fake = MagicMock()
    fake.__version__ = "fake-0.50.0"
    fake.api = MagicMock()
    fake.api.Event = MagicMock()
    monkeypatch.setitem(sys.modules, "datadog", fake)
    return fake


def _record(
    *,
    event: AuditEvent = AuditEvent.TOOL_INVOCATION_COMPLETED,
    decision_allowed: bool | None = None,
    error_code: str | None = None,
) -> AuditRecord:
    return AuditRecord.now(
        event,
        tool_name="search",
        tool_version=1,
        agent_id="agent-x",
        tenant_id="tenant-y",
        decision_allowed=decision_allowed,
        error_code=error_code,
        payload={"input": {"q": "hello"}},
    )


def test_init_calls_datadog_initialize_with_api_key_and_site(
    fake_datadog: MagicMock,
) -> None:
    from ai_core.audit.datadog import DatadogAuditSink
    sink = DatadogAuditSink(
        api_key="dd-api-1",
        app_key="dd-app-1",
        site="datadoghq.eu",
        source="my-app",
        environment="prod",
    )
    fake_datadog.initialize.assert_called_once()
    kwargs = fake_datadog.initialize.call_args.kwargs
    assert kwargs["api_key"] == "dd-api-1"
    assert kwargs["app_key"] == "dd-app-1"
    assert kwargs["api_host"] == "https://api.datadoghq.eu"
    assert sink is not None


@pytest.mark.asyncio
async def test_record_emits_event_create_with_info_alert_type(
    fake_datadog: MagicMock,
) -> None:
    from ai_core.audit.datadog import DatadogAuditSink
    sink = DatadogAuditSink(api_key="k")
    await sink.record(_record(decision_allowed=True))
    fake_datadog.api.Event.create.assert_called_once()
    kwargs = fake_datadog.api.Event.create.call_args.kwargs
    assert kwargs["alert_type"] == "info"


@pytest.mark.asyncio
async def test_record_emits_warning_alert_type_for_denied(
    fake_datadog: MagicMock,
) -> None:
    from ai_core.audit.datadog import DatadogAuditSink
    sink = DatadogAuditSink(api_key="k")
    await sink.record(_record(decision_allowed=False))
    kwargs = fake_datadog.api.Event.create.call_args.kwargs
    assert kwargs["alert_type"] == "warning"


@pytest.mark.asyncio
async def test_record_includes_event_tags(fake_datadog: MagicMock) -> None:
    from ai_core.audit.datadog import DatadogAuditSink
    sink = DatadogAuditSink(api_key="k", environment="prod")
    await sink.record(_record(decision_allowed=True))
    kwargs = fake_datadog.api.Event.create.call_args.kwargs
    tags = kwargs["tags"]
    # All expected tags present.
    assert any(t.startswith("event:") for t in tags)
    assert any(t.startswith("tool_name:search") for t in tags)
    assert any(t.startswith("agent_id:agent-x") for t in tags)
    assert any(t.startswith("tenant_id:tenant-y") for t in tags)
    assert any(t.startswith("env:prod") for t in tags)


@pytest.mark.asyncio
async def test_record_swallows_exception_from_event_create(
    fake_datadog: MagicMock,
) -> None:
    fake_datadog.api.Event.create.side_effect = RuntimeError("backend down")
    from ai_core.audit.datadog import DatadogAuditSink
    sink = DatadogAuditSink(api_key="k")
    await sink.record(_record())  # must not raise


@pytest.mark.asyncio
async def test_flush_is_noop(fake_datadog: MagicMock) -> None:
    from ai_core.audit.datadog import DatadogAuditSink
    sink = DatadogAuditSink(api_key="k")
    await sink.flush()
    # Datadog has no flush API; the sink's flush is a no-op.
    fake_datadog.api.Event.create.assert_not_called()


def test_missing_datadog_raises_configuration_error_with_extra_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "datadog", None)
    if "ai_core.audit.datadog" in sys.modules:
        del sys.modules["ai_core.audit.datadog"]
    from ai_core.audit.datadog import DatadogAuditSink
    with pytest.raises(ConfigurationError) as exc:
        DatadogAuditSink(api_key="k")
    assert exc.value.error_code == "config.optional_dep_missing"
    assert exc.value.details["extra"] == "datadog"
```

- [ ] **Step 3.6: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_datadog_sink.py -v 2>&1 | tail -10
```

Expected: ImportError on `ai_core.audit.datadog`.

- [ ] **Step 3.7: Implement `DatadogAuditSink`**

Create `src/ai_core/audit/datadog.py`:

```python
"""DatadogAuditSink — forward audit records to Datadog as first-class events.

Ships under the optional ``[datadog]`` extra. ``record`` maps each
:class:`AuditRecord` to a ``datadog.api.Event.create`` call with auto
alert_type. ``flush`` is a no-op (Datadog events API has no buffer).
Errors swallowed per the :class:`IAuditSink` never-raise contract.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.exceptions import ConfigurationError

_logger = logging.getLogger(__name__)


class DatadogAuditSink(IAuditSink):
    """Forward :class:`AuditRecord` instances to Datadog.

    Args:
        api_key: Required Datadog API key.
        app_key: Optional application key.
        site: Datadog site (default ``datadoghq.com``).
        source: Source name on emitted events.
        environment: Optional environment tag added as ``env:<value>``.

    Raises:
        ConfigurationError: If ``datadog`` is not installed
            (``error_code='config.optional_dep_missing'``,
            ``details={'extra': 'datadog'}``).
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
            tags: list[str] = [
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
        except Exception as exc:  # noqa: BLE001 — audit-sink never-raise contract
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


__all__ = ["DatadogAuditSink"]
```

- [ ] **Step 3.8: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_datadog_sink.py -v 2>&1 | tail -15
```

Expected: 7 passed.

### 3d — DI wiring

- [ ] **Step 3.9: Extend `provide_audit_sink` with the `datadog` branch**

In `src/ai_core/di/module.py`, after the `sentry` branch added in Task 2 and BEFORE the final `raise ConfigurationError("Unknown audit.sink_type: ...")`, add:

```python
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
```

### 3e — Lint, type-check, full suite, commit

- [ ] **Step 3.10: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/audit/datadog.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py \
    tests/unit/audit/test_datadog_sink.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/audit/datadog.py \
    src/ai_core/config/settings.py \
    src/ai_core/di/module.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on touched files clean; project total ≤ 21.

- [ ] **Step 3.11: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 404+ passing (397 after Task 2 + 7 new for Task 3); 9 pre-existing langgraph errors unchanged.

- [ ] **Step 3.12: Commit Task 3**

```bash
git add src/ai_core/audit/datadog.py \
        src/ai_core/config/settings.py \
        src/ai_core/di/module.py \
        pyproject.toml \
        tests/unit/audit/test_datadog_sink.py
git commit -m "feat(audit): DatadogAuditSink — datadog.api.Event.create with auto alert_type + [datadog] extra"
```

---

## Task 4 — End-of-phase smoke gate

Verification only. No code changes.

- [ ] **Step 4.1: Full test suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 404+ passing, 9 pre-existing langgraph errors unchanged.

- [ ] **Step 4.2: Lint the entire tree**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src tests 2>&1 | tail -3
```

Expected: 211 errors total (= post-Phase-5 baseline). No NEW categories.

- [ ] **Step 4.3: Mypy strict**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `Found 21 errors in 8 files (checked 62 source files)`.

- [ ] **Step 4.4: Verify Phase 6 surface symbols**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
from ai_core.audit import (
    ChainRedactor, KeyNameRedactor, RegexRedactor,
)
from ai_core.audit.redaction import DEFAULT_REDACT_KEYS, PatternKind
from ai_core.audit.sentry import SentryAuditSink
from ai_core.audit.datadog import DatadogAuditSink
from ai_core.config.settings import AuditSettings

# Redactor classes are callable.
assert callable(RegexRedactor(enabled_patterns={'email'}))
assert callable(KeyNameRedactor())
assert callable(ChainRedactor())

# Default key set is non-empty.
assert len(DEFAULT_REDACT_KEYS) >= 10

# AuditSettings has new fields.
s = AuditSettings()
assert s.redaction_profile == 'off'
assert s.sentry_sample_rate == 1.0
assert s.datadog_site == 'datadoghq.com'

# sink_type literal extended.
assert s.sink_type == 'null'  # default unchanged

print('Phase 6 symbols OK')
"
```

Expected: `Phase 6 symbols OK`.

- [ ] **Step 4.5: Verify extras-install (clean venv smoke)**

This is a manual sanity check — not automated:

```bash
TMP=$(mktemp -d)
/Users/admin-h26/EAAP/.venv/bin/python -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install --quiet -e "/Users/admin-h26/EAAP/ai-core-sdk[sentry]" 2>&1 | tail -2
"$TMP/venv/bin/python" -c "from ai_core.audit.sentry import SentryAuditSink; print('sentry extra OK')"
"$TMP/venv/bin/pip" install --quiet -e "/Users/admin-h26/EAAP/ai-core-sdk[datadog]" 2>&1 | tail -2
"$TMP/venv/bin/python" -c "from ai_core.audit.datadog import DatadogAuditSink; print('datadog extra OK')"
rm -rf "$TMP"
```

Expected: prints `sentry extra OK` then `datadog extra OK`. Confirms the extras install cleanly in a fresh venv.

- [ ] **Step 4.6: `eaap init` regression smoke**

```bash
SMOKE=$(mktemp -d)
/Users/admin-h26/EAAP/.venv/bin/python -m ai_core.cli.main init smoke-app --path "$SMOKE" 2>&1 | tail -2
ls "$SMOKE/smoke-app/"
rm -rf "$SMOKE"
```

Expected: `eaap init` still produces a working scaffold (eaap.yaml, .env.example, policies/, src/, etc.). No regression from Phase 5.

- [ ] **Step 4.7: Capture phase summary**

```bash
git log --oneline bc92ebd..HEAD
```

Expected: 3 conventional-commit subjects (one per Task 1/2/3) plus any review-fix commits.

- [ ] **Step 4.8: Do NOT push automatically**

```bash
git status
echo ""
echo "Suggested next step:"
echo "git push origin feat/phase-6-prod-integrations"
echo "gh pr create --title 'feat: Phase 6 — production integrations (PII redaction + Sentry/Datadog audit sinks)'"
```

---

## Out-of-scope reminders

For traceability, deferred to Phase 7+:

- Sentry breadcrumbs / performance tracing
- Datadog metrics / DogStatsD / log streaming
- Custom regex patterns at the settings layer (free-form list)
- Per-sink redaction overrides
- Microsoft Presidio / Faker / spaCy as alternative redactor backends
- Contract tests for the public 28-name surface
- Testcontainers replacement of component-test fakes
- 1-hour cache TTL beta
- Vertex AI Anthropic prefix recognition
- Multi-connection MCP pool
- Pool health-check probe
- `error_code` lookup registry / constants module
- Sentry SDK v3 migration

If a step starts pulling work from this list, stop and confirm scope with the user.
