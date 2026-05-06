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
                _kind = kind  # capture in local variable for the inner function

                def _sub_cc(m: re.Match[str], k: str = _kind) -> str:
                    return f"<redacted-{k}>" if _luhn_check(m.group(0)) else m.group(0)

                result = pattern.sub(_sub_cc, result)
            else:
                result = pattern.sub(f"<redacted-{kind}>", result)
        return result


__all__ = ["PatternKind", "RegexRedactor"]
