"""Compose N redactors into a single PayloadRedactor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

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
