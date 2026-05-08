"""FastAPI dependency that authorises a request via JWT + OPA.

Usage::

    from fastapi import FastAPI, Depends
    from ai_core.di import Container
    from ai_core.di.module import AgentModule, ProductionSecurityModule
    from ai_core.security import OPAAuthorization, AuthorizedPrincipal

    container = Container.build([AgentModule(), ProductionSecurityModule()])
    authz = OPAAuthorization(container, decision_path="eaap/api/allow")

    app = FastAPI()

    @app.get("/projects/{project_id}")
    async def read_project(
        project_id: str,
        principal: AuthorizedPrincipal = Depends(
            authz.requires(action="project.read", resource="project")
        ),
    ) -> ...:
        ...

The :class:`ProductionSecurityModule` is REQUIRED for actual policy enforcement —
:class:`AgentModule` alone binds the always-allow :class:`NoOpPolicyEvaluator`,
which is appropriate for local development but not for any production deployment.

The dependency:

1. Extracts the bearer token from ``Authorization`` (or returns 401).
2. Decodes it via the bound :class:`JWTVerifier`.
3. Submits ``{user, action, resource, claims, request}`` to OPA.
4. Raises :class:`fastapi.HTTPException` 401/403 on rejection.
5. Returns :class:`AuthorizedPrincipal` so downstream handlers can read
   the verified claims and any obligations attached by OPA.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ai_core.config.settings import AppSettings
from ai_core.di.container import Container
from ai_core.di.interfaces import IPolicyEvaluator
from ai_core.exceptions import PolicyDenialError
from ai_core.security.jwt import JWTVerifier


@dataclass(slots=True)
class AuthorizedPrincipal:
    """Outcome of a successful authorization.

    Attributes:
        subject: The ``sub`` claim of the token (or ``""`` if absent).
        claims: All decoded JWT claims.
        obligations: Any obligations the OPA policy attached to the decision.
        reason: Optional ``reason`` from the OPA decision document.
    """

    subject: str
    claims: Mapping[str, Any]
    obligations: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None


class OPAAuthorization:
    """Factory that produces FastAPI dependency callables.

    Args:
        container: DI container holding bindings for :class:`JWTVerifier`,
            :class:`IPolicyEvaluator`, and :class:`AppSettings`.
        decision_path: Optional override for the OPA decision path. If
            omitted the value from :attr:`SecuritySettings.opa_decision_path`
            is used at request time.
    """

    def __init__(self, container: Container, *, decision_path: str | None = None) -> None:
        self._container = container
        self._decision_path = decision_path
        self._security_scheme = HTTPBearer(auto_error=False)

    def requires(
        self,
        *,
        action: str,
        resource: str | None = None,
    ) -> Callable[..., Awaitable[AuthorizedPrincipal]]:
        """Build a dependency callable that authorises ``action`` on ``resource``.

        Args:
            action: Action verb sent to OPA (e.g. ``"project.read"``).
            resource: Optional resource type. The actual resource id is
                expected to come from path/query parameters and is added
                to the OPA input under ``input.request``.

        Returns:
            A FastAPI-compatible coroutine usable with :func:`Depends`.
        """
        return require_authorization(
            container=self._container,
            decision_path=self._decision_path,
            action=action,
            resource=resource,
            security_scheme=self._security_scheme,
        )


def require_authorization(
    *,
    container: Container,
    action: str,
    resource: str | None = None,
    decision_path: str | None = None,
    security_scheme: HTTPBearer | None = None,
) -> Callable[..., Awaitable[AuthorizedPrincipal]]:
    """Build a one-shot FastAPI dependency for the given action/resource.

    See :class:`OPAAuthorization.requires` for the high-level API. This
    function exists for hosts that prefer not to construct an
    :class:`OPAAuthorization` instance.

    Args:
        container: DI container.
        action: Action verb sent to OPA.
        resource: Optional resource type.
        decision_path: Optional override for ``security.opa_decision_path``.
        security_scheme: Optional shared :class:`HTTPBearer` instance.

    Returns:
        A FastAPI-compatible coroutine usable with :func:`Depends`.
    """
    bearer = security_scheme or HTTPBearer(auto_error=False)

    async def _dep(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = None,
    ) -> AuthorizedPrincipal:
        # Resolve dependencies fresh per call so that overrides
        # (e.g. a test container with a fake evaluator) take effect.
        # mypy strict flags ``type[Protocol]`` as abstract, but the DI
        # container's binding makes the lookup concrete at runtime.
        verifier = container.get(JWTVerifier)  # type: ignore[type-abstract]
        evaluator = container.get(IPolicyEvaluator)  # type: ignore[type-abstract]
        settings = container.get(AppSettings)

        if credentials is None:
            credentials = await bearer(request)
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            claims = verifier.verify(credentials.credentials)
        except PolicyDenialError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=exc.message,
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        opa_input: dict[str, Any] = {
            "user": claims.get("sub", ""),
            "action": action,
            "resource": resource,
            "claims": claims,
            "request": {
                "method": request.method,
                "path": request.url.path,
                "path_params": dict(request.path_params),
                "query_params": dict(request.query_params),
            },
        }

        try:
            decision = await evaluator.evaluate(
                decision_path=decision_path or settings.security.opa_decision_path,
                input=opa_input,
            )
        except PolicyDenialError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=exc.message,
            ) from exc

        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=decision.reason or "policy denied",
            )

        return AuthorizedPrincipal(
            subject=str(claims.get("sub", "")),
            claims=claims,
            obligations=decision.obligations,
            reason=decision.reason,
        )

    return _dep


__all__ = ["AuthorizedPrincipal", "OPAAuthorization", "require_authorization"]
