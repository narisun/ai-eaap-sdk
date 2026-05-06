"""DI bindings for the SDK.

The :class:`AgentModule` is the canonical place where abstract
interfaces (see :mod:`ai_core.di.interfaces`) are bound to concrete
implementations. Bindings are organised so that:

* every binding is a ``@singleton`` provider (one instance per container);
* none of the providers performs I/O at instantiation — heavy resources
  (database engine, FastMCP transports) are constructed lazily on first
  use *after* injection;
* hosts override anything by passing additional :class:`Module`
  instances to :class:`Container.build` — the last binding wins.

Note:
    The package-level ``ai_core.di.__init__`` deliberately does *not*
    re-export this class eagerly; it uses :pep:`562` ``__getattr__`` to
    avoid the circular import that would otherwise arise when domain
    modules (which import from :mod:`ai_core.di.interfaces`) are loaded.
"""

from __future__ import annotations

from injector import Module, multiprovider, provider, singleton
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_core.agents.memory import (
    IMemoryManager,
    LiteLLMTokenCounter,
    MemoryManager,
    TokenCounter,
)
from ai_core.audit import IAuditSink  # noqa: TC001
from ai_core.config.secrets import EnvSecretManager, ISecretManager
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import (
    IBudgetService,
    ICheckpointSaver,
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
)
from ai_core.exceptions import ConfigurationError
from ai_core.health import IHealthProbe  # noqa: TC001
from ai_core.llm.budget import InMemoryBudgetService
from ai_core.llm.litellm_client import LiteLLMClient
from ai_core.mcp.registry import ComponentRegistry
from ai_core.mcp.transports import IMCPConnectionFactory, PoolingMCPConnectionFactory
from ai_core.observability.real import RealObservabilityProvider
from ai_core.persistence.checkpoint import PostgresCheckpointSaver
from ai_core.persistence.engine import EngineFactory
from ai_core.persistence.langgraph_checkpoint import LangGraphCheckpointSaver
from ai_core.schema.registry import SchemaRegistry
from ai_core.security.jwt import JWTVerifier, UnverifiedJWTDecoder
from ai_core.tools.invoker import ToolInvoker


class AgentModule(Module):
    """Default top-level DI module for agentic applications.

    Subclass or compose alongside this module to override bindings for
    specific environments (e.g. swap :class:`NoOpObservabilityProvider`
    for a real OTel/LangFuse provider in production).

    Args:
        settings: Optional pre-built :class:`AppSettings`. If omitted,
            settings are loaded from the environment.
        secret_manager: Optional :class:`ISecretManager`. Defaults to
            :class:`EnvSecretManager`.
    """

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        secret_manager: ISecretManager | None = None,
    ) -> None:
        self._settings = settings
        self._secret_manager = secret_manager

    # ----- Settings ---------------------------------------------------------
    @singleton
    @provider
    def provide_settings(self) -> AppSettings:
        """Return the bound :class:`AppSettings` singleton."""
        if self._settings is not None:
            return self._settings
        from ai_core.config.settings import get_settings

        return get_settings()

    # ----- Secret manager ---------------------------------------------------
    @singleton
    @provider
    def provide_secret_manager(self) -> ISecretManager:
        """Return the bound :class:`ISecretManager` singleton."""
        return self._secret_manager or EnvSecretManager()

    # ----- Observability ----------------------------------------------------
    @singleton
    @provider
    def provide_observability(self, settings: AppSettings) -> IObservabilityProvider:
        """Return the production OTel + LangFuse observability provider.

        The provider degrades gracefully: when no OTel endpoint and no
        LangFuse credentials are configured (the default), it still
        produces traces in-process so log correlation works during local
        development. To explicitly opt out, override this binding with
        :class:`ai_core.observability.NoOpObservabilityProvider`.
        """
        return RealObservabilityProvider(settings)

    # ----- Budget -----------------------------------------------------------
    @singleton
    @provider
    def provide_budget(self, settings: AppSettings) -> IBudgetService:
        """Return the in-memory budget service singleton."""
        return InMemoryBudgetService(settings)

    # ----- LLM client -------------------------------------------------------
    @singleton
    @provider
    def provide_llm_client(
        self,
        settings: AppSettings,
        budget: IBudgetService,
        observability: IObservabilityProvider,
    ) -> ILLMClient:
        """Return the LiteLLM-backed client singleton."""
        return LiteLLMClient(settings, budget, observability)

    # ----- Token counter + memory manager -----------------------------------
    @singleton
    @provider
    def provide_token_counter(self) -> TokenCounter:
        """Return the default LiteLLM-backed token counter."""
        return LiteLLMTokenCounter()

    @singleton
    @provider
    def provide_memory_manager(
        self,
        settings: AppSettings,
        llm: ILLMClient,
        counter: TokenCounter,
    ) -> MemoryManager:
        """Return the concrete :class:`MemoryManager` singleton."""
        return MemoryManager(settings, llm, counter)

    @singleton
    @provider
    def provide_memory_manager_interface(self, manager: MemoryManager) -> IMemoryManager:
        """Alias the :class:`IMemoryManager` interface to the concrete singleton.

        :class:`BaseAgent` depends on :class:`IMemoryManager` so hosts
        that need bespoke compaction strategies (recency-only, semantic
        clustering, tiered storage) can override this binding alone
        without touching the concrete :class:`MemoryManager` provider.
        """
        return manager

    # ----- Persistence ------------------------------------------------------
    @singleton
    @provider
    def provide_engine_factory(self, settings: AppSettings) -> EngineFactory:
        """Return the engine factory singleton (lazy connection)."""
        return EngineFactory(settings)

    @singleton
    @provider
    def provide_async_engine(self, factory: EngineFactory) -> AsyncEngine:
        """Return the lazily-constructed :class:`AsyncEngine`."""
        return factory.engine()

    @singleton
    @provider
    def provide_checkpoint_saver(self, engine: AsyncEngine) -> ICheckpointSaver:
        """Return the Postgres-backed checkpoint saver singleton."""
        return PostgresCheckpointSaver(engine)

    @singleton
    @provider
    def provide_langgraph_checkpoint_saver(
        self,
        engine: AsyncEngine,
    ) -> LangGraphCheckpointSaver:
        """Return the LangGraph-native checkpoint saver singleton.

        Bound by concrete type rather than LangGraph's
        :class:`BaseCheckpointSaver` to keep that interface optional —
        host applications that don't compile LangGraph graphs never
        pull the LangGraph types into their DI graph.
        """
        return LangGraphCheckpointSaver(engine)

    # ----- MCP --------------------------------------------------------------
    @singleton
    @provider
    def provide_component_registry(self) -> ComponentRegistry:
        """Return the in-memory component registry singleton."""
        return ComponentRegistry()

    @singleton
    @provider
    def provide_mcp_connection_factory(self, settings: AppSettings) -> IMCPConnectionFactory:
        """Return the pooling MCP connection factory."""
        return PoolingMCPConnectionFactory(
            pool_enabled=settings.mcp.pool_enabled,
            pool_idle_seconds=settings.mcp.pool_idle_seconds,
        )

    # ----- Security ---------------------------------------------------------
    @singleton
    @provider
    def provide_policy_evaluator(self) -> IPolicyEvaluator:
        """Return the default :class:`NoOpPolicyEvaluator`.

        Production deployments must override this binding with
        :class:`ProductionSecurityModule` (or a custom module that binds a
        real evaluator) to enable policy enforcement.
        """
        from ai_core.security.noop_policy import NoOpPolicyEvaluator  # noqa: PLC0415
        return NoOpPolicyEvaluator()

    @singleton
    @provider
    def provide_jwt_verifier(self, settings: AppSettings) -> JWTVerifier:
        """Return the default JWT verifier.

        Defaults to :class:`UnverifiedJWTDecoder` (decode-only, suitable
        for deployments where an upstream gateway has already validated
        the signature). Production hosts that terminate JWT verification
        inside the service should override this binding with
        :class:`HS256JWTVerifier` (or a JWKS verifier of their own).
        """
        return UnverifiedJWTDecoder(settings)

    # ----- Schema registry --------------------------------------------------
    @singleton
    @provider
    def provide_schema_registry(self) -> SchemaRegistry:
        """Return the in-process versioned-schema registry singleton."""
        return SchemaRegistry()

    # ----- Audit sink -------------------------------------------------------
    @singleton
    @provider
    def provide_audit_sink(
        self,
        settings: AppSettings,
        observability: IObservabilityProvider,
    ) -> IAuditSink:
        """Default audit sink — switchable via settings.audit.sink_type."""
        sink_type = settings.audit.sink_type
        if sink_type == "null":
            from ai_core.audit.null import NullAuditSink  # noqa: PLC0415
            return NullAuditSink()
        if sink_type == "otel_event":
            from ai_core.audit.otel_event import OTelEventAuditSink  # noqa: PLC0415
            return OTelEventAuditSink(observability)
        if sink_type == "jsonl":
            from ai_core.audit.jsonl import JsonlFileAuditSink  # noqa: PLC0415
            if settings.audit.jsonl_path is None:
                raise ConfigurationError(
                    "audit.sink_type='jsonl' requires audit.jsonl_path to be set",
                    error_code="config.invalid",
                )
            return JsonlFileAuditSink(settings.audit.jsonl_path)
        raise ConfigurationError(
            f"Unknown audit.sink_type: {sink_type!r}",
            error_code="config.invalid",
        )

    # ----- Health probes ----------------------------------------------------
    @singleton
    @multiprovider
    def provide_health_probes(
        self,
        settings: AppSettings,
        engine: AsyncEngine,
    ) -> list[IHealthProbe]:
        """Default health-probe set. Override in a custom module to add probes."""
        from ai_core.health.probes import (  # noqa: PLC0415
            DatabaseProbe,
            ModelLookupProbe,
            OPAReachabilityProbe,
        )
        return [
            OPAReachabilityProbe(settings),
            DatabaseProbe(engine),
            ModelLookupProbe(settings),
        ]

    # ----- Tool invoker -----------------------------------------------------
    @singleton
    @provider
    def provide_tool_invoker(
        self,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator,
        registry: SchemaRegistry,
        audit: IAuditSink,
    ) -> ToolInvoker:
        """Return the singleton :class:`ToolInvoker` wired to the SDK's services."""
        return ToolInvoker(
            observability=observability,
            policy=policy,
            registry=registry,
            audit=audit,
        )


class ProductionSecurityModule(Module):
    """Opt-in DI module that binds :class:`OPAPolicyEvaluator` over the NoOp default.

    Compose with :class:`AgentModule` for production:

    .. code-block:: python

        from ai_core.app import AICoreApp
        from ai_core.di.module import ProductionSecurityModule

        async with AICoreApp(modules=[ProductionSecurityModule()]) as app:
            ...
    """

    @singleton
    @provider
    def provide_policy_evaluator(self, settings: AppSettings) -> IPolicyEvaluator:
        """Return the OPA-backed policy evaluator. Loaded from `ai_core.security.opa`."""
        from ai_core.security.opa import OPAPolicyEvaluator  # noqa: PLC0415
        return OPAPolicyEvaluator(settings)


__all__ = ["AgentModule", "ProductionSecurityModule"]
