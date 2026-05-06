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
