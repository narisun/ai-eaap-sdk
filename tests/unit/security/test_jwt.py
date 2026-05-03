"""Unit tests for :mod:`ai_core.security.jwt`."""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest

from ai_core.config.settings import AppSettings
from ai_core.exceptions import PolicyDenialError
from ai_core.security.jwt import HS256JWTVerifier, UnverifiedJWTDecoder


pytestmark = pytest.mark.unit


def _settings(*, audience: str | None = None, issuer: str | None = None) -> AppSettings:
    payload: dict[str, str | None] = {}
    if audience is not None:
        payload["jwt_audience"] = audience
    if issuer is not None:
        payload["jwt_issuer"] = issuer
    return AppSettings(security=payload)  # type: ignore[arg-type]


def _make_token(
    *,
    secret: str | None = None,
    algorithm: str = "HS256",
    aud: str | None = None,
    iss: str | None = None,
    exp_offset: int = 60,
    sub: str = "user-1",
    extra: dict[str, str] | None = None,
) -> str:
    claims: dict[str, object] = {"sub": sub, "exp": int(time.time()) + exp_offset}
    if aud is not None:
        claims["aud"] = aud
    if iss is not None:
        claims["iss"] = iss
    if extra:
        claims.update(extra)
    if secret is None:
        return pyjwt.encode(claims, "irrelevant", algorithm="none")
    return pyjwt.encode(claims, secret, algorithm=algorithm)


# ---------------------------------------------------------------------------
# HS256
# ---------------------------------------------------------------------------
def test_hs256_round_trip_with_aud_and_iss() -> None:
    token = _make_token(secret="topsecret", aud="eaap-app", iss="eaap-idp")
    verifier = HS256JWTVerifier("topsecret", _settings(audience="eaap-app", issuer="eaap-idp"))
    claims = verifier.verify(token)
    assert claims["sub"] == "user-1"


def test_hs256_rejects_wrong_audience() -> None:
    token = _make_token(secret="topsecret", aud="other", iss="eaap-idp")
    verifier = HS256JWTVerifier("topsecret", _settings(audience="eaap-app", issuer="eaap-idp"))
    with pytest.raises(PolicyDenialError):
        verifier.verify(token)


def test_hs256_rejects_wrong_secret() -> None:
    token = _make_token(secret="other-secret")
    verifier = HS256JWTVerifier("topsecret", _settings())
    with pytest.raises(PolicyDenialError):
        verifier.verify(token)


def test_hs256_rejects_expired_token() -> None:
    token = _make_token(secret="topsecret", exp_offset=-1)
    verifier = HS256JWTVerifier("topsecret", _settings())
    with pytest.raises(PolicyDenialError):
        verifier.verify(token)


def test_hs256_requires_non_empty_secret() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        HS256JWTVerifier("", _settings())


# ---------------------------------------------------------------------------
# UnverifiedJWTDecoder
# ---------------------------------------------------------------------------
def test_unverified_decoder_accepts_token_without_signature() -> None:
    token = _make_token(secret="anything")
    verifier = UnverifiedJWTDecoder(_settings())
    claims = verifier.verify(token)
    assert claims["sub"] == "user-1"


def test_unverified_decoder_enforces_audience_when_configured() -> None:
    token = _make_token(secret="x", aud="other")
    verifier = UnverifiedJWTDecoder(_settings(audience="eaap-app"))
    with pytest.raises(PolicyDenialError):
        verifier.verify(token)


def test_unverified_decoder_accepts_matching_audience() -> None:
    token = _make_token(secret="x", aud="eaap-app")
    verifier = UnverifiedJWTDecoder(_settings(audience="eaap-app"))
    assert verifier.verify(token)["aud"] == "eaap-app"


def test_unverified_decoder_enforces_issuer() -> None:
    token = _make_token(secret="x", iss="other-idp")
    verifier = UnverifiedJWTDecoder(_settings(issuer="eaap-idp"))
    with pytest.raises(PolicyDenialError):
        verifier.verify(token)


def test_unverified_decoder_rejects_expired_token() -> None:
    token = _make_token(secret="x", exp_offset=-5)
    verifier = UnverifiedJWTDecoder(_settings())
    with pytest.raises(PolicyDenialError):
        verifier.verify(token)
