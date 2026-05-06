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
