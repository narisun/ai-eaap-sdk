"""OPA integration tests via Testcontainers.

Loads the eaap init starter policies (policies/agent.rego, policies/api.rego)
into a real OPA server and exercises OPAReachabilityProbe + OPAPolicyEvaluator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ai_core.config.settings import AppSettings, SecuritySettings
from ai_core.health.probes import OPAReachabilityProbe
from ai_core.security.opa import OPAPolicyEvaluator

if TYPE_CHECKING:
    from testcontainers.core.container import DockerContainer

pytestmark = pytest.mark.integration


def _opa_url(opa: DockerContainer) -> str:
    host = opa.get_container_host_ip()
    port = opa.get_exposed_port(8181)
    return f"http://{host}:{port}"


def _make_settings(
    opa: DockerContainer,
    *,
    opa_health_path: str | None = None,
) -> AppSettings:
    settings = AppSettings()
    opa_url = _opa_url(opa)
    if opa_health_path is not None:
        settings.security = SecuritySettings(
            opa_url=opa_url,
            opa_health_path=opa_health_path,
        )
    else:
        settings.security = SecuritySettings(opa_url=opa_url)
    return settings


@pytest.mark.asyncio
async def test_opa_reachability_probe_returns_ok_against_real_opa(
    opa_container: DockerContainer,
) -> None:
    settings = _make_settings(opa_container)
    probe = OPAReachabilityProbe(settings.security)
    result = await probe.probe()
    assert result.status == "ok"
    assert result.component == "opa"


@pytest.mark.asyncio
async def test_opa_reachability_probe_honours_custom_health_path(
    opa_container: DockerContainer,
) -> None:
    settings = _make_settings(opa_container, opa_health_path="/health")
    probe = OPAReachabilityProbe(settings.security)
    result = await probe.probe()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_opa_policy_evaluator_evaluates_starter_agent_policy(
    opa_container: DockerContainer,
) -> None:
    """Exercise the eaap init starter agent policy.

    The agent.rego shipped with `eaap init` defines:
        package eaap.agent.tool_call
        default allow := false
        allow if { not denied_tool }
        denied_tool if { input.tool.name == "delete_everything" }

    So a tool call that is NOT in the deny list should return allowed=True.
    """
    settings = _make_settings(opa_container)
    evaluator = OPAPolicyEvaluator(settings.security)
    try:
        decision = await evaluator.evaluate(
            decision_path="eaap/agent/tool_call/allow",
            input={"tool": {"name": "search"}, "principal": {"sub": "user-1"}},
        )
        assert decision.allowed is True
    finally:
        await evaluator.aclose()


@pytest.mark.asyncio
async def test_opa_policy_evaluator_denies_blocked_tool(
    opa_container: DockerContainer,
) -> None:
    """The starter policy denies tool.name=='delete_everything'."""
    settings = _make_settings(opa_container)
    evaluator = OPAPolicyEvaluator(settings.security)
    try:
        decision = await evaluator.evaluate(
            decision_path="eaap/agent/tool_call/allow",
            input={"tool": {"name": "delete_everything"}, "principal": {"sub": "user-1"}},
        )
        assert decision.allowed is False
    finally:
        await evaluator.aclose()
