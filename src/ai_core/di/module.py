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

from injector import Module, provider, singleton
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_core.agents.memory import (
    IMemoryManager,
    LiteLLMTokenCounter,
    MemoryManager,
    TokenCounter,
)
from ai_core.config.secrets import EnvSecretManager, ISecretManager
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import (
    IBudgetService,
    ICheckpointSaver,
    ILLMClient,
    IObservabilityProvider,
    IPolicyEvaluator,
)
from ai_core.llm.budget import InMemoryBudgetService
from ai_core.llm.litellm_client import LiteLLMClient
from ai_core.mcp.registry import ComponentRegistry
from ai_core.mcp.transports import FastMCPConnectionFactory, IMCPConnectionFactory
from ai_core.observability.real import RealObservabilityProvider
from ai_core.persistence.checkpoint import PostgresCheckpointSaver
from ai_core.persistence.engine import EngineFactory
from ai_core.persistence.langgraph_checkpoint import LangGraphCheckpointSaver
from ai_core.schema.registry import SchemaRegistry
from ai_core.security.jwt import JWTVerifier, UnverifiedJWTDecoder
from ai_core.security.opa import OPAPolicyEvaluator


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
    def provide_mcp_connection_factory(self) -> IMCPConnectionFactory:
        """Return the default FastMCP connection factory."""
        return FastMCPConnectionFactory()

    # ----- Security ---------------------------------------------------------
    @singleton
    @provider
    def provide_policy_evaluator(self, settings: AppSettings) -> IPolicyEvaluator:
        """Return the OPA-backed policy evaluator singleton."""
        return OPAPolicyEvaluator(settings)

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


__all__ = ["AgentModule"]
