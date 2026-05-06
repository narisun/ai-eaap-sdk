"""Sanity tests for the ErrorCode StrEnum registry."""
from __future__ import annotations

import pytest

from ai_core.exceptions import EAAPBaseException, ErrorCode

pytestmark = pytest.mark.unit


def test_error_code_values_are_unique_and_dotted_lowercase() -> None:
    values = [member.value for member in ErrorCode]
    assert len(set(values)) == len(values), "ErrorCode has duplicate values"
    for value in values:
        assert value, "ErrorCode member has empty value"
        assert value == value.lower(), f"{value!r} not lowercase"
        assert "." in value, f"{value!r} not dotted"


def _all_concrete_exceptions() -> list[type[EAAPBaseException]]:
    seen: set[type[EAAPBaseException]] = set()
    stack = list(EAAPBaseException.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    return sorted(seen, key=lambda c: c.__qualname__)


@pytest.mark.parametrize(
    "exc_cls", _all_concrete_exceptions(), ids=lambda c: c.__qualname__
)
def test_every_concrete_exception_default_code_is_an_errorcode_member(
    exc_cls: type[EAAPBaseException],
) -> None:
    """A new exception class with an inline string DEFAULT_CODE bypasses the
    enum and won't appear in the catalog. This test catches that drift."""
    valid_values = {member.value for member in ErrorCode}
    assert exc_cls.DEFAULT_CODE in valid_values, (
        f"{exc_cls.__qualname__}.DEFAULT_CODE = {exc_cls.DEFAULT_CODE!r} "
        f"is not in ErrorCode. Add the new code to ErrorCode."
    )
