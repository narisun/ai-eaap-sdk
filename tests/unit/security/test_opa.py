"""Unit tests for :class:`ai_core.security.opa.OPAPolicyEvaluator`."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import PolicyDecision
from ai_core.exceptions import PolicyDenialError
from ai_core.security.opa import OPAPolicyEvaluator


pytestmark = pytest.mark.unit


def _settings(*, fail_closed: bool = True, opa_url: str = "http://opa:8181") -> AppSettings:
    return AppSettings(
        security={  # type: ignore[arg-type]
            "opa_url": opa_url,
            "opa_decision_path": "eaap/test/allow",
            "fail_closed": fail_closed,
            "opa_request_timeout_seconds": 1.0,
        },
    )


@respx.mock
async def test_decision_minimal_boolean_shape_allowed() -> None:
    respx.post("http://opa:8181/v1/data/eaap/test/allow").respond(
        200, json={"result": True}
    )
    evaluator = OPAPolicyEvaluator(_settings().security)
    decision = await evaluator.evaluate(decision_path="eaap/test/allow", input={"u": "a"})
    assert decision == PolicyDecision(allowed=True, obligations={}, reason=None)


@respx.mock
async def test_decision_structured_shape_denied_with_reason() -> None:
    respx.post("http://opa:8181/v1/data/eaap/test/allow").respond(
        200,
        json={
            "result": {
                "allow": False,
                "obligations": {"audit": "high"},
                "reason": "tenant suspended",
            }
        },
    )
    evaluator = OPAPolicyEvaluator(_settings().security)
    decision = await evaluator.evaluate(decision_path="eaap/test/allow", input={})
    assert decision.allowed is False
    assert decision.obligations == {"audit": "high"}
    assert decision.reason == "tenant suspended"


@respx.mock
async def test_malformed_response_treated_as_deny() -> None:
    respx.post("http://opa:8181/v1/data/eaap/test/allow").respond(
        200, json={"unexpected": "shape"}
    )
    evaluator = OPAPolicyEvaluator(_settings().security)
    decision = await evaluator.evaluate(decision_path="eaap/test/allow", input={})
    assert decision.allowed is False
    assert decision.reason == "malformed-opa-response"


@respx.mock
async def test_input_payload_wrapped_with_input_key() -> None:
    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read()
        return httpx.Response(200, json={"result": True})

    respx.post("http://opa:8181/v1/data/eaap/test/allow").mock(side_effect=_handler)

    evaluator = OPAPolicyEvaluator(_settings().security)
    await evaluator.evaluate(
        decision_path="eaap/test/allow",
        input={"action": "read", "user": "u-1"},
    )

    import json

    parsed = json.loads(captured["body"])
    assert parsed == {"input": {"action": "read", "user": "u-1"}}


@respx.mock
async def test_fail_closed_raises_on_network_error() -> None:
    respx.post("http://opa:8181/v1/data/eaap/test/allow").mock(
        side_effect=httpx.ConnectError("boom")
    )
    evaluator = OPAPolicyEvaluator(_settings(fail_closed=True).security)
    with pytest.raises(PolicyDenialError) as ei:
        await evaluator.evaluate(decision_path="eaap/test/allow", input={})
    assert ei.value.details["error_type"] == "ConnectError"


@respx.mock
async def test_fail_open_returns_allow_with_reason_on_network_error() -> None:
    respx.post("http://opa:8181/v1/data/eaap/test/allow").mock(
        side_effect=httpx.ConnectError("boom")
    )
    evaluator = OPAPolicyEvaluator(_settings(fail_closed=False).security)
    decision = await evaluator.evaluate(decision_path="eaap/test/allow", input={})
    assert decision.allowed is True
    assert decision.reason is not None
    assert "opa-unavailable" in decision.reason


@respx.mock
async def test_fail_closed_raises_on_5xx() -> None:
    respx.post("http://opa:8181/v1/data/eaap/test/allow").respond(500)
    evaluator = OPAPolicyEvaluator(_settings(fail_closed=True).security)
    with pytest.raises(PolicyDenialError):
        await evaluator.evaluate(decision_path="eaap/test/allow", input={})


async def test_aclose_idempotent() -> None:
    evaluator = OPAPolicyEvaluator(_settings().security)
    await evaluator.aclose()
    await evaluator.aclose()
