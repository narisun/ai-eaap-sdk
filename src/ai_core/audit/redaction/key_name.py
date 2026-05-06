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
    "tokens",
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
            if isinstance(value, list):
                return [self._redact_value(key, v) for v in value]
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {k: self._redact_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(key, v) for v in value]
        return value


__all__ = ["DEFAULT_REDACT_KEYS", "KeyNameRedactor"]
