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
from ai_core.agents.runtime import AgentRuntime
from ai_core.agents.tool_errors import DefaultToolErrorRenderer, IToolErrorRenderer
from ai_core.audit import IAuditSink, PayloadRedactor  # noqa: TC001
from ai_core.config.secrets import EnvSecretManager, ISecretManager
from ai_core.config.settings import (
    AgentSettings,
    AppSettings,
    AuditSettings,
    BudgetSettings,
    DatabaseSettings,
    HealthSettings,
    LLMSettings,
    MCPSettings,
    ObservabilitySettings,
    SecuritySettings,
)
from ai_core.di.interfaces import (
    IBudgetService,
    ICheckpointSaver,
    ICompactionLLM,
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
)
from ai_core.health import IHealthProbe  # noqa: TC001
from ai_core.llm._raise import RaiseOnUseLLMClient
from ai_core.llm.budget import InMemoryBudgetService
from ai_core.mcp.registry import ComponentRegistry
from ai_core.mcp.transports import IMCPConnectionFactory, PoolingMCPConnectionFactory
from ai_core.observability.real import RealObservabilityProvider
from ai_core.persistence.checkpoint import PostgresCheckpointSaver
from ai_core.persistence.engine import EngineFactory
from ai_core.persistence.langgraph_checkpoint import LangGraphCheckpointSaver
from ai_core.schema.registry import SchemaRegistry
from ai_core.security.jwt import JWTVerifier, UnverifiedJWTDecoder
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.middleware import ToolMiddleware  # noqa: TC001
from ai_core.tools.registrar import ToolRegistrar
from ai_core.tools.resolver import DefaultToolResolver, IToolResolver


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
        """Return the bound :class:`AppSettings` singleton.

        When the host did not pass an explicit ``settings`` to the module,
        a fresh :class:`AppSettings` is constructed here. Pydantic Settings
        sources (env, ``.env``, YAML) are evaluated at construction time;
        the container caches the result as a singleton, so subsequent
        injections see the same instance without a process-global cache.
        """
        if self._settings is not None:
            return self._settings
        return AppSettings()

    # Per-subsystem settings slices — bind each nested settings group as its
    # own singleton so concrete services can inject only the slice they need.
    # This keeps unit tests honest: a fake LLM client does not need to fabricate
    # a full :class:`AppSettings` just to satisfy DI.
    @singleton
    @provider
    def provide_llm_settings(self, settings: AppSettings) -> LLMSettings:
        return settings.llm

    @singleton
    @provider
    def provide_agent_settings(self, settings: AppSettings) -> AgentSettings:
        return settings.agent

    @singleton
    @provider
    def provide_database_settings(self, settings: AppSettings) -> DatabaseSettings:
        return settings.database

    @singleton
    @provider
    def provide_observability_settings(self, settings: AppSettings) -> ObservabilitySettings:
        return settings.observability

    @singleton
    @provider
    def provide_security_settings(self, settings: AppSettings) -> SecuritySettings:
        return settings.security

    @singleton
    @provider
    def provide_audit_settings(self, settings: AppSettings) -> AuditSettings:
        return settings.audit

    @singleton
    @provider
    def provide_budget_settings(self, settings: AppSettings) -> BudgetSettings:
        return settings.budget

    @singleton
    @provider
    def provide_mcp_settings(self, settings: AppSettings) -> MCPSettings:
        return settings.mcp

    @singleton
    @provider
    def provide_health_settings(self, settings: AppSettings) -> HealthSettings:
        return settings.health

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
    def provide_budget(self, budget_settings: BudgetSettings) -> IBudgetService:
        """Return the in-memory budget service singleton."""
        return InMemoryBudgetService(budget_settings)

    # ----- LLM client -------------------------------------------------------
    @singleton
    @provider
    def provide_llm_client(self) -> ILLMClient:
        """Bind the raise-on-use default :class:`ILLMClient`.

        Hosts that want the LiteLLM-backed adapter compose
        :class:`ai_core.llm.LiteLLMModule` alongside this module — the
        last binding wins, so :class:`LiteLLMClient` overrides this
        stub. Hosts with their own LLM stack bind their own
        :class:`ILLMClient` implementation in a custom module.

        Tests that exercise non-LLM behaviour use
        :class:`ai_core.testing.ScriptedLLM`, which also overrides this
        binding via its own :class:`Module` and avoids requiring the
        ``[litellm]`` extra.
        """
        return RaiseOnUseLLMClient()

    # ----- Token counter + memory manager -----------------------------------
    @singleton
    @provider
    def provide_token_counter(self) -> TokenCounter:
        """Return the default LiteLLM-backed token counter."""
        return LiteLLMTokenCounter()

    @singleton
    @provider
    def provide_compaction_llm(self, llm: ILLMClient) -> ICompactionLLM:
        """Default :class:`ICompactionLLM` aliases the request LLM client.

        Override this binding (or supply a layered :class:`Module`) when
        you want compaction to use a cheaper/faster model than agent
        reasoning. Both interfaces are structurally identical, so any
        :class:`ILLMClient` implementation satisfies :class:`ICompactionLLM`.
        """
        # ILLMClient structurally satisfies ICompactionLLM (same Protocol shape).
        return llm

    @singleton
    @provider
    def provide_memory_manager(
        self,
        agent_settings: AgentSettings,
        llm_settings: LLMSettings,
        compaction_llm: ICompactionLLM,
        counter: TokenCounter,
    ) -> MemoryManager:
        """Return the concrete :class:`MemoryManager` singleton."""
        return MemoryManager(agent_settings, llm_settings, compaction_llm, counter)

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
    def provide_engine_factory(self, db_settings: DatabaseSettings) -> EngineFactory:
        """Return the engine factory singleton (lazy connection)."""
        return EngineFactory(db_settings)

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
    def provide_mcp_connection_factory(self, mcp_settings: MCPSettings) -> IMCPConnectionFactory:
        """Return the pooling MCP connection factory."""
        return PoolingMCPConnectionFactory(
            pool_enabled=mcp_settings.pool_enabled,
            pool_idle_seconds=mcp_settings.pool_idle_seconds,
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
        from ai_core.security.noop_policy import NoOpPolicyEvaluator
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

    # ----- Payload redactor (Phase 6) ---------------------------------------
    @singleton
    @provider
    def provide_payload_redactor(self, settings: AppSettings) -> PayloadRedactor:
        """Return the configured PayloadRedactor; identity for profile='off'."""
        profile = settings.audit.redaction_profile
        if profile == "off":
            from ai_core.audit.interface import _identity_redactor
            return _identity_redactor
        from ai_core.audit.redaction import (
            ChainRedactor,
            KeyNameRedactor,
            PatternKind,
            RegexRedactor,
        )
        base_patterns: set[PatternKind] = {"email", "phone", "ssn", "credit_card", "ipv4"}
        if profile == "strict":
            base_patterns.add("long_number")
        return ChainRedactor(
            RegexRedactor(enabled_patterns=base_patterns),
            KeyNameRedactor(),
        )

    # ----- Audit sink -------------------------------------------------------
    @singleton
    @provider
    def provide_audit_sink(
        self,
        audit_settings: AuditSettings,
        observability: IObservabilityProvider,
    ) -> IAuditSink:
        """Resolve an :class:`IAuditSink` via the pluggable registry.

        Built-in sinks (``null``, ``otel_event``, ``jsonl``, ``sentry``,
        ``datadog``) register themselves in
        :mod:`ai_core.audit.registry`. Third-party sinks register either
        by calling :func:`ai_core.audit.register_audit_sink` or by
        declaring an ``ai_eaap_sdk.audit_sinks`` entry point.
        """
        from ai_core.audit.registry import get_audit_sink_factory
        factory = get_audit_sink_factory(audit_settings.sink_type)
        return factory(audit_settings, observability)

    # ----- Health probes ----------------------------------------------------
    @singleton
    @multiprovider
    def provide_health_probes(
        self,
        security_settings: SecuritySettings,
        llm_settings: LLMSettings,
        engine: AsyncEngine,
    ) -> list[IHealthProbe]:
        """Default health-probe set.

        :class:`injector.Module` ``@multiprovider`` semantics allow hosts to
        **add** probes by including an extra :class:`Module` that also
        contributes a ``list[IHealthProbe]``. The lists are concatenated;
        no override of this provider is required.

        Example — register an extra probe alongside the defaults::

            class ExtraProbes(Module):
                @multiprovider
                def provide_extra(self) -> list[IHealthProbe]:
                    return [VectorDBProbe(), KafkaProbe()]

            async with AICoreApp(modules=[ExtraProbes()]) as app:
                ...

        For a one-off addition without authoring a Module, use
        :py:meth:`AICoreApp.add_health_probe`.
        """
        from ai_core.health.probes import (
            DatabaseProbe,
            ModelLookupProbe,
            OPAReachabilityProbe,
        )
        return [
            OPAReachabilityProbe(security_settings),
            DatabaseProbe(engine),
            ModelLookupProbe(llm_settings),
        ]

    # ----- Tool invoker -----------------------------------------------------
    @singleton
    @multiprovider
    def provide_tool_middlewares(self) -> list[ToolMiddleware]:
        """Default contribution to the :class:`ToolMiddleware` multibind.

        Returns an empty list so the default :class:`ToolInvoker`
        pipeline runs unwrapped — byte-identical to pre-v1 behaviour.
        Hosts add their own middlewares by including a :class:`Module`
        whose ``@multiprovider`` returns a non-empty list; injector
        concatenates the lists in module-registration order.
        """
        return []

    @singleton
    @provider
    def provide_tool_invoker(
        self,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator,
        registry: SchemaRegistry,
        audit: IAuditSink,
        redactor: PayloadRedactor,
        middlewares: list[ToolMiddleware],
    ) -> ToolInvoker:
        """Return the singleton :class:`ToolInvoker` wired to the SDK's services."""
        return ToolInvoker(
            observability=observability,
            policy=policy,
            registry=registry,
            audit=audit,
            redactor=redactor,
            middlewares=middlewares,
        )

    # ----- Tool error renderer ---------------------------------------------
    @singleton
    @provider
    def provide_tool_error_renderer(self) -> IToolErrorRenderer:
        """Default :class:`IToolErrorRenderer` — preserves pre-v1 English text.

        Override this binding to enforce strict failure (raise instead
        of returning a recovery message), localize text, or redact error
        details before they flow back to the LLM.
        """
        return DefaultToolErrorRenderer()

    # ----- Tool resolver + registrar ---------------------------------------
    @singleton
    @provider
    def provide_tool_resolver(
        self,
        mcp_factory: IMCPConnectionFactory,
        tool_invoker: ToolInvoker,
    ) -> IToolResolver:
        """Default :class:`IToolResolver` — pre-v1 MCP-merge behaviour.

        Hosts override this binding to cache MCP resolutions across
        agents, redact tools by tenant, or stub the MCP backend in
        tests.
        """
        return DefaultToolResolver(mcp_factory, tool_invoker)

    @singleton
    @provider
    def provide_tool_registrar(self, tool_invoker: ToolInvoker) -> ToolRegistrar:
        """Default :class:`ToolRegistrar` — bulk register on the invoker.

        Override the binding to gate registration (e.g. by tenant or
        feature flag) without touching :meth:`BaseAgent.compile`.
        """
        return ToolRegistrar(tool_invoker)

    # ----- Agent runtime ----------------------------------------------------
    @singleton
    @provider
    def provide_agent_runtime(
        self,
        agent_settings: AgentSettings,
        llm: ILLMClient,
        memory: IMemoryManager,
        observability: IObservabilityProvider,
        tool_invoker: ToolInvoker,
        mcp_factory: IMCPConnectionFactory,
        tool_error_renderer: IToolErrorRenderer,
        tool_resolver: IToolResolver,
        tool_registrar: ToolRegistrar,
    ) -> AgentRuntime:
        """Return the bundle of SDK services injected into :class:`BaseAgent`.

        Centralising these collaborators in :class:`AgentRuntime` lets
        :class:`BaseAgent` subclasses receive a single argument and add
        their own DI-resolved dependencies without mirroring the SDK's
        internal service surface.
        """
        return AgentRuntime(
            agent_settings=agent_settings,
            llm=llm,
            memory=memory,
            observability=observability,
            tool_invoker=tool_invoker,
            mcp_factory=mcp_factory,
            tool_error_renderer=tool_error_renderer,
            tool_resolver=tool_resolver,
            tool_registrar=tool_registrar,
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
    def provide_policy_evaluator(
        self, security_settings: SecuritySettings
    ) -> IPolicyEvaluator:
        """Return the OPA-backed policy evaluator. Loaded from `ai_core.security.opa`."""
        from ai_core.security.opa import OPAPolicyEvaluator
        return OPAPolicyEvaluator(security_settings)


__all__ = ["AgentModule", "ProductionSecurityModule"]
