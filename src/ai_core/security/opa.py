"""Open Policy Agent (OPA) policy evaluator.

Calls the OPA REST API at ``POST /v1/data/<decision_path>`` with the
caller-supplied input document. The response is normalised into a
:class:`PolicyDecision`.

Failure semantics
-----------------
Network or HTTP errors honor :attr:`SecuritySettings.fail_closed`:
* ``fail_closed=True`` (default) → :class:`PolicyDenialError` is raised so
  the calling middleware/guardrail rejects the request;
* ``fail_closed=False`` → returns ``PolicyDecision(allowed=True, …)`` with
  a ``reason`` describing the failure mode (use only for non-critical
  decisions).

Decision-document shape
-----------------------
The evaluator accepts either the OPA "minimal" shape (``result`` is a
boolean) or the structured shape::

    {"result": {"allow": true, "obligations": {...}, "reason": "..."}}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
from injector import inject

from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision
from ai_core.exceptions import PolicyDenialError


class OPAPolicyEvaluator(IPolicyEvaluator):
    """Concrete :class:`IPolicyEvaluator` backed by the OPA REST API.

    Args:
        settings: Aggregated application settings.
        client: Optional pre-configured :class:`httpx.AsyncClient`. If
            omitted, a client is constructed lazily on first call using
            ``security.opa_request_timeout_seconds``.
    """

    @inject
    def __init__(
        self,
        settings: AppSettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    async def evaluate(
        self,
        *,
        decision_path: str,
        input: Mapping[str, Any],
    ) -> PolicyDecision:
        """See :meth:`IPolicyEvaluator.evaluate`."""
        cfg = self._settings.security
        url = f"{str(cfg.opa_url).rstrip('/')}/v1/data/{decision_path.lstrip('/')}"

        client = await self._get_client()
        try:
            response = await client.post(
                url,
                json={"input": dict(input)},
                timeout=cfg.opa_request_timeout_seconds,
            )
            response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            return self._handle_failure(exc, decision_path=decision_path)

        try:
            payload = response.json()
        except ValueError as exc:
            return self._handle_failure(exc, decision_path=decision_path)

        return _decode_decision(payload)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        """Close the underlying HTTP client if this instance owns it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.security.opa_request_timeout_seconds,
            )
            self._owns_client = True
        return self._client

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    def _handle_failure(
        self,
        exc: BaseException,
        *,
        decision_path: str,
    ) -> PolicyDecision:
        cfg = self._settings.security
        if cfg.fail_closed:
            raise PolicyDenialError(
                "OPA evaluation failed (fail_closed=True)",
                details={
                    "decision_path": decision_path,
                    "opa_url": str(cfg.opa_url),
                    "error_type": type(exc).__name__,
                },
                cause=exc,
            ) from exc
        return PolicyDecision(
            allowed=True,
            obligations={},
            reason=f"opa-unavailable: {type(exc).__name__}",
        )


def _decode_decision(payload: Mapping[str, Any]) -> PolicyDecision:
    """Normalise an OPA response payload into a :class:`PolicyDecision`."""
    result = payload.get("result")
    if isinstance(result, bool):
        return PolicyDecision(allowed=result, obligations={})
    if isinstance(result, Mapping):
        allow = bool(result.get("allow", False))
        obligations = result.get("obligations") or {}
        if not isinstance(obligations, Mapping):
            obligations = {}
        reason = result.get("reason")
        return PolicyDecision(
            allowed=allow,
            obligations=dict(obligations),
            reason=str(reason) if reason is not None else None,
        )
    # Treat missing / unexpected shapes as deny — safer default than allow.
    return PolicyDecision(allowed=False, obligations={}, reason="malformed-opa-response")


__all__ = ["OPAPolicyEvaluator"]
