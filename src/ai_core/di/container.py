"""DI container façade around :class:`injector.Injector`.

The :class:`Container` is a thin wrapper that:

* enforces SDK conventions (modules are an ordered list; the *last*
  binding for a given type wins, which makes test overrides idempotent);
* surfaces SDK-flavored errors (:class:`DependencyResolutionError`)
  rather than the raw :class:`injector.Error`; and
* offers a convenience :py:meth:`override` method that returns a
  *new* container for tests, leaving the original untouched.
* exposes :py:meth:`start` / :py:meth:`stop` lifecycle hooks (and an
  async-context-manager protocol) so applications can warm up
  long-lived resources at boot and tear them down cleanly on shutdown.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

from injector import CallError, Injector, Module, UnsatisfiedRequirement

from ai_core.exceptions import DependencyResolutionError

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class Container:
    """Resolved DI container scoped to one application/test.

    The container is intentionally not a singleton — host code MAY
    construct multiple containers (e.g. one per worker, one per test).
    Within a single container, providers marked as ``@singleton`` are
    cached, but those caches do not leak across containers.

    Lifecycle:
        Call :py:meth:`start` once at application boot to run any
        registered start hooks; call :py:meth:`stop` at shutdown to
        run stop hooks (LIFO) and tear down SDK-known long-lived
        resources (observability, OPA, async DB engine). The container
        also implements the async context-manager protocol for
        ``async with Container.build(...) as c: ...`` style use.

    Attributes:
        injector: The underlying :class:`injector.Injector`. Use
            :py:meth:`get` for type-safe resolution rather than touching
            this attribute directly.
    """

    __slots__ = ("_lifecycle_hooks", "_modules", "_started", "injector")

    def __init__(self, modules: Sequence[Module]) -> None:
        self._modules: tuple[Module, ...] = tuple(modules)
        # auto_bind=True lets concrete user classes (e.g. host-defined Agent
        # subclasses) resolve without an explicit binding. Abstract interfaces
        # still require a binding, so this does not weaken the contract.
        self.injector: Injector = Injector(modules=list(modules), auto_bind=True)
        self._lifecycle_hooks: list[Any] = []
        self._started: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, modules: Iterable[Module] | None = None) -> Container:
        """Construct a container with the given modules.

        Args:
            modules: Ordered iterable of :class:`injector.Module` instances.
                If empty/None, an :class:`ai_core.di.AgentModule` with default
                settings is used.

        Returns:
            A ready-to-use :class:`Container`.
        """
        from ai_core.di.module import AgentModule

        mods = list(modules) if modules is not None else [AgentModule()]
        if not mods:
            mods = [AgentModule()]
        return cls(mods)

    def override(self, *additional: Module) -> Container:
        """Return a *new* container with extra modules appended.

        Later modules win, so this is the preferred way to install
        test fakes::

            container = Container.build()
            test_container = container.override(FakeProvidersModule())

        Args:
            *additional: Modules to layer on top of the existing ones.

        Returns:
            A new :class:`Container`. The original is unchanged.
        """
        return Container([*self._modules, *additional])

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def get(self, interface: type[T]) -> T:
        """Resolve an instance of ``interface`` from the container.

        Args:
            interface: The abstract type (ABC / protocol / concrete) to resolve.

        Returns:
            An instance bound to ``interface``.

        Raises:
            DependencyResolutionError: If no binding exists or instantiation fails.
        """
        try:
            return self.injector.get(interface)
        except (UnsatisfiedRequirement, CallError) as exc:
            raise DependencyResolutionError(
                f"Cannot resolve binding for {interface!r}",
                details={"interface": getattr(interface, "__qualname__", str(interface))},
                cause=exc,
            ) from exc

    def __contains__(self, interface: type[object]) -> bool:
        """Return ``True`` if a binding for ``interface`` is registered."""
        try:
            self.injector.get(interface)
        except (UnsatisfiedRequirement, CallError):
            return False
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def add_lifecycle_hook(self, hook: Any) -> None:
        """Register a host-defined component for lifecycle management.

        ``hook`` may expose any of (in priority order):

        * ``async def start_async(self)`` / ``async def stop_async(self)``
        * ``async def start(self)`` / ``async def stop(self)``
        * synchronous ``start()`` / ``stop()``

        Hooks fire in registration order on :py:meth:`start` and in
        reverse-registration order on :py:meth:`stop`.

        Args:
            hook: Any object with at least one of the above methods.
        """
        self._lifecycle_hooks.append(hook)

    async def start(self) -> None:
        """Run start hooks. Idempotent — safe to call once at app boot.

        The SDK's own concretes do not require explicit startup — they
        self-initialise on first use (lazy DB engine, lazy OPA client)
        or in their constructors (observability tracer/exporter). This
        method is therefore a passthrough for host-registered hooks.
        Override or extend in subclasses to add eager warmup behaviour.
        """
        if self._started:
            return
        for hook in self._lifecycle_hooks:
            await self._call_optional_method(hook, ("start_async", "start"), swallow=False)
        self._started = True

    async def stop(self) -> None:
        """Run stop hooks (LIFO) + tear down SDK-known long-lived resources.

        Best-effort: a failure in one teardown step is logged but does
        not prevent subsequent steps from running.
        """
        # Custom hooks first, in reverse order so resources released before
        # the SDK ones they may depend on (e.g. a custom cache that uses the
        # DB engine).
        for hook in reversed(self._lifecycle_hooks):
            await self._call_optional_method(hook, ("stop_async", "stop"), swallow=True)

        # SDK-owned long-lived resources, in dependency-friendly order.
        await self._teardown_sdk_resources()
        self._started = False

    async def __aenter__(self) -> Container:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _teardown_sdk_resources(self) -> None:
        """Best-effort teardown of long-lived SDK singletons."""
        # Local imports keep the lifecycle code from forcing import-time
        # cycles when the container is built without these bindings.
        from sqlalchemy.ext.asyncio import AsyncEngine

        from ai_core.audit import IAuditSink  # noqa: PLC0415
        from ai_core.di.interfaces import IObservabilityProvider, IPolicyEvaluator
        from ai_core.mcp.transports import IMCPConnectionFactory  # noqa: PLC0415

        steps: list[tuple[str, type[Any], tuple[str, ...]]] = [
            ("observability.shutdown", IObservabilityProvider, ("shutdown",)),
            ("audit.flush", IAuditSink, ("flush",)),
            ("mcp_pool.aclose", IMCPConnectionFactory, ("aclose",)),
            ("policy_evaluator.aclose", IPolicyEvaluator, ("aclose",)),
            ("engine.dispose", AsyncEngine, ("dispose",)),
        ]
        for label, interface, method_names in steps:
            try:
                target = self.injector.get(interface)
            except (UnsatisfiedRequirement, CallError):
                continue
            await self._call_optional_method(target, method_names, swallow=True, label=label)

    async def _call_optional_method(
        self,
        target: Any,
        method_names: tuple[str, ...],
        *,
        swallow: bool,
        label: str | None = None,
    ) -> None:
        """Call the first existing method on ``target`` from ``method_names``.

        Awaits the result if it's a coroutine. When ``swallow`` is True,
        exceptions are logged at WARNING and not re-raised — this is
        the right policy for shutdown paths.
        """
        method = next(
            (getattr(target, n) for n in method_names if callable(getattr(target, n, None))),
            None,
        )
        if method is None:
            return
        descriptor = label or f"{type(target).__name__}.{method.__name__}"
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001
            if swallow:
                _logger.warning("Lifecycle %s failed: %s", descriptor, exc)
                return
            raise


__all__ = ["Container"]
