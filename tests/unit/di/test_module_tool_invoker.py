"""Tests that the DI graph resolves ToolInvoker as a singleton."""
from __future__ import annotations

import pytest

from ai_core.di import AgentModule, Container
from ai_core.tools.invoker import ToolInvoker

pytestmark = pytest.mark.unit


def test_container_resolves_tool_invoker() -> None:
    container = Container.build([AgentModule()])
    invoker = container.get(ToolInvoker)
    assert isinstance(invoker, ToolInvoker)


def test_tool_invoker_is_singleton() -> None:
    container = Container.build([AgentModule()])
    a = container.get(ToolInvoker)
    b = container.get(ToolInvoker)
    assert a is b
