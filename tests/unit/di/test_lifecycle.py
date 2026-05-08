"""Tests for :class:`Container.start` / :class:`Container.stop`."""

from __future__ import annotations

import pytest
from injector import Module, provider, singleton

from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import IObservabilityProvider, IPolicyEvaluator

pytestmark = pytest.mark.unit


class _AsyncHook:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def start_async(self) -> None:
        self.events.append("start")

    async def stop_async(self) -> None:
        self.events.append("stop")


class _SyncHook:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("sync-start")

    def stop(self) -> None:
        self.events.append("sync-stop")


# ---------------------------------------------------------------------------
# Hook order + idempotency
# ---------------------------------------------------------------------------
async def test_start_runs_async_hooks_then_marks_started() -> None:
    container = Container.build([AgentModule(settings=AppSettings())])
    h = _AsyncHook()
    container.add_lifecycle_hook(h)

    await container.start()
    assert h.events == ["start"]

    # Idempotent: a second start() does NOT re-run hooks.
    await container.start()
    assert h.events == ["start"]


async def test_stop_calls_hooks_in_reverse_registration_order() -> None:
    container = Container.build([AgentModule(settings=AppSettings())])
    order: list[str] = []

    class _Hook:
        def __init__(self, label: str) -> None:
            self.label = label

        async def start_async(self) -> None:
            order.append(f"start:{self.label}")

        async def stop_async(self) -> None:
            order.append(f"stop:{self.label}")

    container.add_lifecycle_hook(_Hook("A"))
    container.add_lifecycle_hook(_Hook("B"))

    await container.start()
    await container.stop()

    assert order == ["start:A", "start:B", "stop:B", "stop:A"]


async def test_stop_swallows_hook_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    container = Container.build([AgentModule(settings=AppSettings())])

    class _BadHook:
        async def stop_async(self) -> None:
            raise RuntimeError("boom-on-shutdown")

    container.add_lifecycle_hook(_BadHook())
    # Should not raise — failure logged, lifecycle continues.
    await container.stop()


async def test_supports_sync_start_stop_methods() -> None:
    container = Container.build([AgentModule(settings=AppSettings())])
    h = _SyncHook()
    container.add_lifecycle_hook(h)
    await container.start()
    await container.stop()
    assert h.events == ["sync-start", "sync-stop"]


async def test_async_with_runs_full_lifecycle() -> None:
    h = _AsyncHook()
    async with Container.build([AgentModule(settings=AppSettings())]) as container:
        container.add_lifecycle_hook(h)
        # The hook was registered AFTER start(), so only stop() will fire it.
        # Verify async-with semantics work for both registration paths.
        h2 = _AsyncHook()
        container.add_lifecycle_hook(h2)
    # On exit, both hooks' stop methods ran.
    assert h.events == ["stop"]
    assert h2.events == ["stop"]


# ---------------------------------------------------------------------------
# SDK-known teardown
# ---------------------------------------------------------------------------
async def test_stop_invokes_observability_shutdown_and_policy_aclose() -> None:
    settings = AppSettings()
    shutdown_called: list[bool] = []
    aclose_called: list[bool] = []

    class _Obs(IObservabilityProvider):
        def start_span(self, name, *, attributes=None):  # type: ignore[no-untyped-def]
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def _cm():  # type: ignore[no-untyped-def]
                yield None

            return _cm()

        async def record_llm_usage(self, **kwargs):  # type: ignore[no-untyped-def]
            return None

        async def record_event(self, name, *, attributes=None):  # type: ignore[no-untyped-def]
            return None

        async def shutdown(self) -> None:
            shutdown_called.append(True)

    class _Policy(IPolicyEvaluator):
        async def evaluate(self, *, decision_path, input):  # type: ignore[no-untyped-def]
            from ai_core.di.interfaces import PolicyDecision

            return PolicyDecision(allowed=True, obligations={})

        async def aclose(self) -> None:
            aclose_called.append(True)

    class _Override(Module):
        @singleton
        @provider
        def obs(self) -> IObservabilityProvider:
            return _Obs()

        @singleton
        @provider
        def policy(self) -> IPolicyEvaluator:
            return _Policy()

    container = Container.build([AgentModule(settings=settings), _Override()])
    await container.stop()
    assert shutdown_called == [True]
    assert aclose_called == [True]


async def test_stop_tolerates_missing_bindings() -> None:
    """Containers without the SDK bindings (e.g. minimal test containers) shouldn't crash."""

    class _BareModule(Module):
        pass

    container = Container([_BareModule()])
    # No observability, policy, or engine bindings — stop() should be a no-op.
    await container.stop()
