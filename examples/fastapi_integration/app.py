"""FastAPI app demonstrating OPA-backed authorization via the SDK.

The endpoint enforces `data.eaap.api.allow` from the SDK's starter
api.rego policy: a JWT subject (`sub`) must equal the `user_id` path
parameter, otherwise OPA returns deny → FastAPI returns 403.

Run:

    uv run python examples/fastapi_integration/app.py

See README.md for OPA setup and curl examples.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import Depends, FastAPI
from injector import Module, provider, singleton

from ai_core.config.settings import AppSettings, Environment
from ai_core.di import AgentModule, Container
from ai_core.di.module import ProductionSecurityModule
from ai_core.security import AuthorizedPrincipal, OPAAuthorization
from ai_core.security.jwt import HS256JWTVerifier, JWTVerifier


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(
            f"environment variable {name} is required — set it before running this demo. "
            f"See examples/fastapi_integration/README.md."
        )
    return value


def build_app() -> FastAPI:
    jwt_secret = _require_env("DEMO_JWT_SECRET")
    # Demo defaults — AppSettings reads from os.environ. Callers (e.g. the
    # smoke gate or tests) can override these before calling build_app().
    os.environ.setdefault("EAAP_SECURITY__OPA_URL", "http://localhost:8181")
    os.environ.setdefault("EAAP_SECURITY__JWT_AUDIENCE", "ai-core-sdk-demo")
    os.environ.setdefault("EAAP_SECURITY__JWT_ISSUER", "ai-core-sdk-demo")

    settings = AppSettings(service_name="fastapi-demo", environment=Environment.LOCAL)

    class _DemoSecurityOverrides(Module):
        @singleton
        @provider
        def provide_jwt_verifier(self, settings_: AppSettings) -> JWTVerifier:
            return HS256JWTVerifier(jwt_secret, settings_)

    container = Container.build(
        [
            AgentModule(settings=settings),
            ProductionSecurityModule(),
            _DemoSecurityOverrides(),
        ]
    )
    authz = OPAAuthorization(container, decision_path="eaap/api/allow")

    app = FastAPI(title="ai-core-sdk FastAPI integration demo")

    @app.get("/users/{user_id}/profile")
    async def read_profile(
        user_id: str,
        principal: AuthorizedPrincipal = Depends(  # noqa: B008
            authz.requires(action="profile.read", resource="profile")  # noqa: B008
        ),
    ) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "subject": principal.subject,
            "email": principal.claims.get("email", "(unknown)"),
        }

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="127.0.0.1", port=8000)
