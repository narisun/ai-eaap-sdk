"""Unit tests for :mod:`ai_core.security.fastapi_dep`."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from injector import Module, provider, singleton

from ai_core.config.settings import AppSettings
from ai_core.di import Container, AgentModule
from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision
from ai_core.security import (
    AuthorizedPrincipal,
    HS256JWTVerifier,
    JWTVerifier,
    OPAAuthorization,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakePolicy(IPolicyEvaluator):
    def __init__(self, *, allow: bool, reason: str | None = None) -> None:
        self.allow = allow
        self.reason = reason
        self.calls: list[dict[str, Any]] = []

    async def evaluate(
        self,
        *,
        decision_path: str,
        input: Mapping[str, Any],
    ) -> PolicyDecision:
        self.calls.append({"decision_path": decision_path, "input": dict(input)})
        return PolicyDecision(allowed=self.allow, obligations={}, reason=self.reason)


def _build_container(
    *, allow: bool, reason: str | None = None, secret: str = "topsecret"
) -> tuple[Container, _FakePolicy]:
    settings = AppSettings(security={"opa_decision_path": "eaap/test/allow"})  # type: ignore[arg-type]
    policy = _FakePolicy(allow=allow, reason=reason)

    class _Override(Module):
        @singleton
        @provider
        def p(self) -> IPolicyEvaluator:
            return policy

        @singleton
        @provider
        def jwt(self) -> JWTVerifier:
            return HS256JWTVerifier(secret, settings)

    return Container.build([AgentModule(settings=settings), _Override()]), policy


def _make_app(container: Container) -> FastAPI:
    authz = OPAAuthorization(container)
    app = FastAPI()

    @app.get("/projects/{project_id}")
    async def read_project(
        project_id: str,
        principal: AuthorizedPrincipal = Depends(
            authz.requires(action="project.read", resource="project")
        ),
    ) -> dict[str, str]:
        return {"viewer": principal.subject, "project_id": project_id}

    return app


def _bearer(secret: str = "topsecret") -> str:
    token = pyjwt.encode(
        {"sub": "user-1", "exp": int(time.time()) + 60}, secret, algorithm="HS256"
    )
    return f"Bearer {token}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_allowed_request_returns_principal() -> None:
    container, policy = _build_container(allow=True)
    client = TestClient(_make_app(container))

    resp = client.get("/projects/p-7", headers={"Authorization": _bearer()})

    assert resp.status_code == 200
    assert resp.json() == {"viewer": "user-1", "project_id": "p-7"}
    assert policy.calls[0]["decision_path"] == "eaap/test/allow"
    assert policy.calls[0]["input"]["action"] == "project.read"
    assert policy.calls[0]["input"]["resource"] == "project"
    assert policy.calls[0]["input"]["request"]["path_params"]["project_id"] == "p-7"


def test_missing_token_returns_401() -> None:
    container, _ = _build_container(allow=True)
    client = TestClient(_make_app(container))
    resp = client.get("/projects/p-7")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_invalid_token_returns_401() -> None:
    container, _ = _build_container(allow=True)
    client = TestClient(_make_app(container))
    resp = client.get(
        "/projects/p-7", headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert resp.status_code == 401


def test_policy_denial_returns_403() -> None:
    container, _ = _build_container(allow=False, reason="not your project")
    client = TestClient(_make_app(container))
    resp = client.get("/projects/p-7", headers={"Authorization": _bearer()})
    assert resp.status_code == 403
    assert "not your project" in resp.text


def test_wrong_signature_returns_401() -> None:
    container, _ = _build_container(allow=True, secret="server-secret")
    client = TestClient(_make_app(container))
    resp = client.get(
        "/projects/p-7", headers={"Authorization": _bearer(secret="other-secret")}
    )
    assert resp.status_code == 401
