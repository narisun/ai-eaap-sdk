"""DI container façade around :class:`injector.Injector`.

The :class:`Container` is a thin wrapper that:

* enforces SDK conventions (modules are an ordered list; the *last*
  binding for a given type wins, which makes test overrides idempotent);
* surfaces SDK-flavored errors (:class:`DependencyResolutionError`)
  rather than the raw :class:`injector.Error`; and
* offers a convenience :py:meth:`override` method that returns a
  *new* container for tests, leaving the original untouched.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

from injector import CallError, Injector, Module, UnsatisfiedRequirement

from ai_core.exceptions import DependencyResolutionError

T = TypeVar("T")


class Container:
    """Resolved DI container scoped to one application/test.

    The container is intentionally not a singleton — host code MAY
    construct multiple containers (e.g. one per worker, one per test).
    Within a single container, providers marked as ``@singleton`` are
    cached, but those caches do not leak across containers.

    Attributes:
        injector: The underlying :class:`injector.Injector`. Use
            :py:meth:`get` for type-safe resolution rather than touching
            this attribute directly.
    """

    __slots__ = ("_modules", "injector")

    def __init__(self, modules: Sequence[Module]) -> None:
        self._modules: tuple[Module, ...] = tuple(modules)
        self.injector: Injector = Injector(modules=list(modules), auto_bind=False)

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
        # Late import to avoid a cycle at module-load time.
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


__all__ = ["Container"]
