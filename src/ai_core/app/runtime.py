"""Application-level lifecycle facade.

A consumer holds one :class:`AICoreApp`, enters it as an async context
manager, and resolves agents through :py:meth:`agent`. The app is the
canonical wiring layer between :class:`AppSettings`, the DI container, the
:class:`ComponentRegistry`, and the :class:`ToolInvoker`.

Power users may still build a :class:`Container` directly, but the facade
is the documented "pit of success" path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

from ai_core.config.secrets import EnvSecretManager, ISecretManager
from ai_core.config.settings import AppSettings, get_settings
from ai_core.di.container import Container
from ai_core.di.interfaces import IObservabilityProvider, IPolicyEvaluator
from ai_core.di.module import AgentModule
from ai_core.health.interface import HealthStatus, IHealthProbe, ProbeResult
from ai_core.mcp.registry import ComponentRegistry, RegisteredComponent
from ai_core.mcp.transports import (
    IMCPConnectionFactory,
    MCPServerSpec,
)
from ai_core.observability.logging import configure as _configure_logging
from ai_core.tools.invoker import ToolInvoker

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from injector import Module

    from ai_core.tools.spec import ToolSpec

A = TypeVar("A")


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Coarse application health snapshot returned by :py:meth:`AICoreApp.health`."""

    status: HealthStatus
    components: Mapping[str, HealthStatus]
    component_details: Mapping[str, str | None]
    service_name: str

    def __post_init__(self) -> None:
        # Wrap the dicts in read-only MappingProxyType so callers can't mutate
        # the snapshot. Mutation attempts raise TypeError, matching frozen=True semantics.
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(
            self, "component_details", MappingProxyType(dict(self.component_details))
        )


class _HealthCheckRunner:
    """Fans out IHealthProbe instances in parallel with a per-probe timeout."""

    def __init__(self, probes: Sequence[IHealthProbe], *,
                 timeout_seconds: float) -> None:
        self._probes = list(probes)
        self._timeout = timeout_seconds

    async def run(self) -> list[ProbeResult]:
        async def _run_one(probe: IHealthProbe) -> ProbeResult:
            try:
                return await asyncio.wait_for(probe.probe(), timeout=self._timeout)
            except TimeoutError:
                return ProbeResult(
                    component=probe.component, status="down",
                    detail=f"probe_timeout_{self._timeout}s",
                )
            except Exception as exc:  # runner defence-in-depth — never raises
                return ProbeResult(
                    component=probe.component, status="down",
                    detail=f"probe_error: {type(exc).__name__}",
                )
        return list(await asyncio.gather(*(_run_one(p) for p in self._probes)))


class AICoreApp:
    """Lifecycle facade for an SDK consumer.

    Args:
        settings: Optional pre-built :class:`AppSettings`. When omitted, settings
            are loaded lazily via :func:`ai_core.config.settings.get_settings`.
        modules: Extra DI modules layered after :class:`AgentModule`. Useful for
            tests (fake providers) and for production overrides (custom
            ``ISecretManager``, alternative ``IPolicyEvaluator``).
        secret_manager: Optional :class:`ISecretManager`. Defaults to
            :class:`EnvSecretManager`.

    Use as an async context manager::

        async with AICoreApp() as app:
            agent = app.agent(MyAgent)
            state = await agent.ainvoke(messages=[...])
    """

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        modules: Sequence[Module] = (),
        secret_manager: ISecretManager | None = None,
    ) -> None:
        self._user_settings = settings
        self._user_modules = tuple(modules)
        self._user_secret_manager = secret_manager

        self._settings: AppSettings | None = None
        self._secret_manager: ISecretManager | None = None
        self._container: Container | None = None
        self._entered: bool = False
        self._closed: bool = False

    # ----- Lifecycle ----------------------------------------------------------
    async def __aenter__(self) -> AICoreApp:
        self._settings = self._user_settings or get_settings()
        self._secret_manager = self._user_secret_manager or EnvSecretManager()
        # Fail fast — collect-all validation surfaces every issue at once.
        self._settings.validate_for_runtime(secret_manager=self._secret_manager)
        # Phase 3: configure structlog before any code logs.
        _configure_logging(
            log_format=self._settings.observability.log_format,
            log_level=self._settings.observability.log_level.value,
        )
        self._container = Container.build([
            AgentModule(
                settings=self._settings,
                secret_manager=self._secret_manager,
            ),
            *self._user_modules,
        ])
        await self._container.start()
        self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        if self._closed or self._container is None:
            return
        await self._container.stop()
        self._closed = True

    # ----- Public API ---------------------------------------------------------
    def agent(self, cls: type[A]) -> A:
        """Resolve an agent class. ``ToolInvoker`` is auto-injected."""
        return self._require_container().get(cls)

    def register_agent(self, cls: type[A]) -> None:
        """Explicitly bind an agent class to the container.

        Call this once per agent class at app boot to surface missing
        bindings (typos in constructor type hints, etc.) immediately
        rather than relying on :py:attr:`Container.auto_bind`.
        """
        self._require_container().register_agent(cls)

    def add_health_probe(self, probe: IHealthProbe) -> None:
        """Append an extra health probe to the live container.

        Convenience for hosts that want to register a probe without
        authoring a small :class:`Module`. The probe is included in
        every subsequent :py:meth:`health` snapshot.

        Args:
            probe: An :class:`IHealthProbe` instance with a ``component``
                identifier.
        """
        existing = self._require_container().get(list[IHealthProbe])
        existing.append(probe)

    def register_tools(self, *specs: ToolSpec) -> None:
        """Register one or more :class:`ToolSpec` with the SDK's SchemaRegistry.

        Idempotent — registering the same spec twice is a no-op.
        """
        invoker = self._require_container().get(ToolInvoker)
        for spec in specs:
            invoker.register(spec)

    async def register_mcp(
        self,
        spec: MCPServerSpec,
        *,
        replace: bool = False,
    ) -> RegisteredComponent:
        """Register an MCP server spec with the :class:`ComponentRegistry`.

        Returns the :class:`RegisteredComponent` record so callers can
        introspect or unregister later.
        """
        container = self._require_container()
        registry = container.get(ComponentRegistry)
        factory = container.get(IMCPConnectionFactory)  # type: ignore[type-abstract]
        wrapper = _MCPComponent(spec=spec, factory=factory)
        return await registry.register(
            wrapper, component_type="mcp_server", replace=replace
        )

    async def health(self) -> HealthSnapshot:
        """Run all configured health probes in parallel and return the snapshot."""
        if not self._entered or self._settings is None:
            return HealthSnapshot(
                status="down",
                components={},
                component_details={},
                service_name="",
            )
        runner = _HealthCheckRunner(
            self._require_container().get(list[IHealthProbe]),
            timeout_seconds=self._settings.health.probe_timeout_seconds,
        )
        results = await runner.run()
        components: dict[str, HealthStatus] = {r.component: r.status for r in results}
        details: dict[str, str | None] = {r.component: r.detail for r in results}
        if any(s == "down" for s in components.values()):
            roll_up: HealthStatus = "down"
        elif any(s == "degraded" for s in components.values()):
            roll_up = "degraded"
        else:
            roll_up = "ok"
        if self._closed:
            roll_up = "down"
        return HealthSnapshot(
            status=roll_up,
            components=components,
            component_details=details,
            service_name=self._settings.service_name,
        )

    # ----- Properties ---------------------------------------------------------
    @property
    def settings(self) -> AppSettings:
        if self._settings is None:
            raise RuntimeError("AICoreApp has not been entered yet.")
        return self._settings

    @property
    def container(self) -> Container:
        return self._require_container()

    @property
    def policy_evaluator(self) -> IPolicyEvaluator:
        return self._require_container().get(IPolicyEvaluator)  # type: ignore[type-abstract]

    @property
    def observability(self) -> IObservabilityProvider:
        return self._require_container().get(IObservabilityProvider)  # type: ignore[type-abstract]

    # ----- Internal -----------------------------------------------------------
    def _require_container(self) -> Container:
        if self._container is None:
            raise RuntimeError("AICoreApp has not been entered yet.")
        return self._container


class _MCPComponent:
    """Adapter that wraps an MCPServerSpec as an IComponent for the registry."""

    def __init__(self, *, spec: MCPServerSpec, factory: IMCPConnectionFactory) -> None:
        self.component_id = spec.component_id
        self.component_type = "mcp_server"
        self._spec = spec
        self._factory = factory

    async def health_check(self) -> bool:
        try:
            async with self._factory.open(self._spec):
                return True
        except Exception:  # broad catch intentional: always return False on any failure
            return False


__all__ = ["AICoreApp", "HealthSnapshot"]
