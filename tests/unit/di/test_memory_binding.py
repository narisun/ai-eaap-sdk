"""Tests for the IMemoryManager DI binding."""

from __future__ import annotations

import pytest
from injector import Module, provider, singleton

from ai_core.agents import IMemoryManager, MemoryManager
from ai_core.agents.state import AgentState
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container

pytestmark = pytest.mark.unit


def test_default_iface_resolves_to_memory_manager() -> None:
    container = Container.build([AgentModule(settings=AppSettings())])
    iface = container.get(IMemoryManager)
    concrete = container.get(MemoryManager)
    assert isinstance(iface, MemoryManager)
    assert iface is concrete  # same singleton, two type aliases


def test_iface_can_be_overridden_independently() -> None:
    """Hosts that need a custom strategy override IMemoryManager only."""

    class _CountOnlyMemory(IMemoryManager):
        def should_compact(self, state, *, model=None) -> bool:  # type: ignore[no-untyped-def]
            return False

        async def compact(  # type: ignore[no-untyped-def]
            self, state, *, model=None, tenant_id=None, agent_id=None
        ) -> AgentState:
            return state

    fake = _CountOnlyMemory()

    class _Override(Module):
        @singleton
        @provider
        def m(self) -> IMemoryManager:
            return fake

    container = Container.build([AgentModule(settings=AppSettings()), _Override()])
    assert container.get(IMemoryManager) is fake
    # Concrete MemoryManager is still resolvable via the default binding.
    assert isinstance(container.get(MemoryManager), MemoryManager)
