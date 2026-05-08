"""Security sub-package — OPA evaluator, JWT verification, FastAPI dep, Guardrail."""

from __future__ import annotations

from ai_core.security.fastapi_dep import (
    AuthorizedPrincipal,
    OPAAuthorization,
    require_authorization,
)
from ai_core.security.guardrail import GuardrailDecision, GuardrailNode
from ai_core.security.jwt import (
    HS256JWTVerifier,
    JWTVerifier,
    UnverifiedJWTDecoder,
)
from ai_core.security.opa import OPAPolicyEvaluator

__all__ = [
    "AuthorizedPrincipal",
    "GuardrailDecision",
    "GuardrailNode",
    "HS256JWTVerifier",
    "JWTVerifier",
    "OPAAuthorization",
    "OPAPolicyEvaluator",
    "UnverifiedJWTDecoder",
    "require_authorization",
]
