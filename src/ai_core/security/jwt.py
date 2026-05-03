"""JWT verification used by the FastAPI authorization dependency.

The SDK ships two implementations:

* :class:`HS256JWTVerifier` — HMAC-SHA256 with a shared secret. Suitable
  for service-to-service tokens where the issuer is internal.
* :class:`UnverifiedJWTDecoder` — decodes the token *without verifying
  the signature*. Intended for **dev/test only**, or for deployments
  where signature verification has already been enforced upstream
  (e.g. an API gateway). It still validates ``iss`` and ``aud`` claims
  if those are configured in :class:`SecuritySettings`.

Production hosts can plug a JWKS-backed verifier (RS256 / ES256) by
implementing :class:`JWTVerifier` and overriding the binding in their
own DI module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import jwt
from jwt import PyJWTError

from ai_core.config.settings import AppSettings, SecuritySettings
from ai_core.exceptions import PolicyDenialError

_logger = logging.getLogger(__name__)


class JWTVerifier(ABC):
    """Verify a bearer token and return its claims."""

    @abstractmethod
    def verify(self, token: str) -> dict[str, Any]:
        """Return the validated JWT claims.

        Args:
            token: The raw bearer token (without the ``Bearer `` prefix).

        Returns:
            The decoded JWT claims as a mapping.

        Raises:
            PolicyDenialError: If the token is malformed or fails verification.
        """


class HS256JWTVerifier(JWTVerifier):
    """HMAC-SHA256 verifier using a shared secret.

    Args:
        secret: HMAC shared secret.
        settings: Aggregated application settings (for audience/issuer claims).
    """

    def __init__(self, secret: str, settings: AppSettings) -> None:
        if not secret:
            raise ValueError("HS256JWTVerifier requires a non-empty secret")
        self._secret = secret
        self._sec: SecuritySettings = settings.security

    def verify(self, token: str) -> dict[str, Any]:
        """See :meth:`JWTVerifier.verify`."""
        try:
            return jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                audience=self._sec.jwt_audience,
                issuer=self._sec.jwt_issuer,
                options={"require": ["exp"]},
            )
        except PyJWTError as exc:
            raise PolicyDenialError(
                "JWT verification failed",
                details={"error_type": type(exc).__name__},
                cause=exc,
            ) from exc


class UnverifiedJWTDecoder(JWTVerifier):
    """Decode JWTs without signature verification — DEV / GATEWAY mode only.

    Use cases:

    * Local development where signing keys aren't available.
    * Production deployments where an upstream API gateway has already
      verified the signature and the FastAPI service only needs to read
      claims.

    The decoder still validates ``iss`` and ``aud`` if they are
    configured, so a malformed/spoofed token without the expected
    issuer or audience is rejected.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._sec: SecuritySettings = settings.security
        if not (settings.is_production() is False):
            _logger.warning(
                "UnverifiedJWTDecoder bound in production environment %r — "
                "signature verification is OFF",
                settings.environment.value,
            )

    def verify(self, token: str) -> dict[str, Any]:
        """See :meth:`JWTVerifier.verify`."""
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": True, "require": ["exp"]},
                algorithms=None,
            )
        except PyJWTError as exc:
            raise PolicyDenialError(
                "JWT decode failed",
                details={"error_type": type(exc).__name__},
                cause=exc,
            ) from exc

        if self._sec.jwt_issuer and claims.get("iss") != self._sec.jwt_issuer:
            raise PolicyDenialError(
                "JWT issuer mismatch",
                details={"expected": self._sec.jwt_issuer, "actual": claims.get("iss")},
            )
        if self._sec.jwt_audience:
            aud = claims.get("aud")
            audiences = aud if isinstance(aud, list) else [aud] if aud is not None else []
            if self._sec.jwt_audience not in audiences:
                raise PolicyDenialError(
                    "JWT audience mismatch",
                    details={"expected": self._sec.jwt_audience, "actual": aud},
                )
        return claims


__all__ = ["JWTVerifier", "HS256JWTVerifier", "UnverifiedJWTDecoder"]
