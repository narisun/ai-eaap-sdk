"""Every typed exception must define a non-empty dotted-lowercase DEFAULT_CODE
and mirror it into details['error_code'] at construction time.
"""
from __future__ import annotations

import pytest

from ai_core.exceptions import EAAPBaseException


def _all_concrete_exceptions() -> list[type[EAAPBaseException]]:
    """Return all concrete subclasses of EAAPBaseException, recursively."""
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
def test_exception_default_code_is_dotted_lowercase(
    exc_cls: type[EAAPBaseException],
) -> None:
    code = exc_cls.DEFAULT_CODE
    assert code, f"{exc_cls.__qualname__}.DEFAULT_CODE is empty"
    assert code == code.lower(), (
        f"{exc_cls.__qualname__}.DEFAULT_CODE not lowercase: {code!r}"
    )
    assert "." in code, (
        f"{exc_cls.__qualname__}.DEFAULT_CODE not dotted: {code!r}"
    )


@pytest.mark.parametrize(
    "exc_cls", _all_concrete_exceptions(), ids=lambda c: c.__qualname__
)
def test_exception_mirrors_error_code_into_details(
    exc_cls: type[EAAPBaseException],
) -> None:
    exc = exc_cls("test message")
    assert exc.error_code == exc_cls.DEFAULT_CODE
    assert exc.details["error_code"] == exc.error_code


def test_at_least_fifteen_concrete_exceptions_exist() -> None:
    """Sanity check: regress if discovery breaks (pre-Phase-7 count is 16)."""
    classes = _all_concrete_exceptions()
    assert len(classes) >= 15, (
        f"Expected >=15 concrete exceptions, found {len(classes)}: "
        f"{[c.__qualname__ for c in classes]}"
    )
