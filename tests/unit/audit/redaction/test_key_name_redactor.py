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
