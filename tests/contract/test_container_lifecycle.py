"""Container.stop() must invoke the 5 documented teardown steps in order.

Order (Phase 6-end): observability.shutdown -> audit.flush ->
mcp_pool.aclose -> policy_evaluator.aclose -> engine.dispose.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from injector import Module, singleton
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_core.audit import IAuditSink
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import (
    IObservabilityProvider,
    IPolicyEvaluator,
)
from ai_core.mcp.transports import IMCPConnectionFactory


class _SpyModule(Module):
    """Override every Container teardown target with a spy that records call order."""

    def __init__(self, call_log: list[str]) -> None:
        self._call_log = call_log

    def configure(self, binder: Any) -> None:
        # Build spies whose teardown method appends a tag to call_log.
        log = self._call_log

        class _SpyObservability:
            async def shutdown(self) -> None:
                log.append("observability.shutdown")
            async def record_event(self, *_a: Any, **_k: Any) -> None: ...
            async def record_llm_usage(self, *_a: Any, **_k: Any) -> None: ...
            def start_span(self, *_a: Any, **_k: Any) -> Any: ...

        class _SpyAudit:
            async def record(self, _r: Any) -> None: ...
            async def flush(self) -> None:
                log.append("audit.flush")

        class _SpyMCP:
            def open(self, _spec: Any) -> Any: ...
            async def aclose(self) -> None:
                log.append("mcp_pool.aclose")

        class _SpyPolicy:
            async def evaluate(self, **_k: Any) -> Any: ...
            async def aclose(self) -> None:
                log.append("policy_evaluator.aclose")

        # AsyncEngine is third-party; we wrap it with an AsyncMock instance
        # whose dispose appends to the log.
        spy_engine = AsyncMock(spec=AsyncEngine)
        async def _dispose_spy() -> None:
            log.append("engine.dispose")
        spy_engine.dispose.side_effect = _dispose_spy

        binder.bind(IObservabilityProvider, to=_SpyObservability(), scope=singleton)
        binder.bind(IAuditSink, to=_SpyAudit(), scope=singleton)
        binder.bind(IMCPConnectionFactory, to=_SpyMCP(), scope=singleton)
        binder.bind(IPolicyEvaluator, to=_SpyPolicy(), scope=singleton)
        binder.bind(AsyncEngine, to=spy_engine, scope=singleton)


@pytest.mark.asyncio
async def test_container_teardown_calls_steps_in_documented_order() -> None:
    call_log: list[str] = []
    container = Container.build([AgentModule(), _SpyModule(call_log)])
    await container.start()
    await container.stop()

    assert call_log == [
        "observability.shutdown",
        "audit.flush",
        "mcp_pool.aclose",
        "policy_evaluator.aclose",
        "engine.dispose",
    ], f"Teardown order incorrect: {call_log}"
