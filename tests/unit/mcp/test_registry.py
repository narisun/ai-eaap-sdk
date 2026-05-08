"""Unit tests for :class:`ai_core.mcp.registry.ComponentRegistry`."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_core.exceptions import RegistryError
from ai_core.mcp.registry import ComponentRegistry

pytestmark = pytest.mark.unit


@dataclass
class _StubComponent:
    component_id: str
    component_type: str
    healthy: bool = True
    raise_on_health: bool = False

    async def health_check(self) -> bool:
        if self.raise_on_health:
            raise RuntimeError("boom")
        return self.healthy


async def test_register_and_get_round_trip() -> None:
    reg = ComponentRegistry()
    comp = _StubComponent(component_id="agent-1", component_type="agent")
    await reg.register(comp, component_type="agent", metadata={"v": 1})

    record = reg.get("agent-1")
    assert record.component is comp
    assert record.component_type == "agent"
    assert record.metadata == {"v": 1}
    assert "agent-1" in reg
    assert len(reg) == 1


async def test_duplicate_registration_raises_unless_replace() -> None:
    reg = ComponentRegistry()
    comp1 = _StubComponent(component_id="x", component_type="agent")
    comp2 = _StubComponent(component_id="x", component_type="agent")
    await reg.register(comp1, component_type="agent")

    with pytest.raises(RegistryError):
        await reg.register(comp2, component_type="agent")

    await reg.register(comp2, component_type="agent", replace=True)
    assert reg.get("x").component is comp2


async def test_unregister_returns_bool() -> None:
    reg = ComponentRegistry()
    comp = _StubComponent(component_id="x", component_type="mcp_server")
    await reg.register(comp, component_type="mcp_server")
    assert await reg.unregister("x") is True
    assert await reg.unregister("x") is False


async def test_get_missing_raises_registry_error() -> None:
    reg = ComponentRegistry()
    with pytest.raises(RegistryError):
        reg.get("missing")


async def test_list_filters_by_type() -> None:
    reg = ComponentRegistry()
    await reg.register(
        _StubComponent("a-1", "agent"), component_type="agent"
    )
    await reg.register(
        _StubComponent("m-1", "mcp_server"), component_type="mcp_server"
    )
    assert {r.component.component_id for r in reg.list(component_type="agent")} == {"a-1"}
    assert {r.component.component_id for r in reg.list(component_type="mcp_server")} == {"m-1"}
    assert len(reg.list()) == 2


async def test_health_check_aggregates_results() -> None:
    reg = ComponentRegistry()
    await reg.register(_StubComponent("ok", "agent"), component_type="agent")
    await reg.register(
        _StubComponent("sick", "agent", healthy=False), component_type="agent"
    )
    await reg.register(
        _StubComponent("flaky", "agent", raise_on_health=True), component_type="agent"
    )

    results = await reg.health_check_all()
    assert results == {"ok": True, "sick": False, "flaky": False}
