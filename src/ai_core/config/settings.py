"""Strongly-typed application settings backed by Pydantic v2.

All settings are loaded from environment variables (and optional ``.env``
files) using :mod:`pydantic_settings`. Nested groups are addressed via a
double-underscore delimiter, so for example::

    EAAP_DATABASE__DSN=postgresql+asyncpg://user:pass@host/db
    EAAP_LLM__DEFAULT_MODEL=bedrock/anthropic.claude-sonnet-4-6
    EAAP_OBSERVABILITY__OTEL_ENDPOINT=http://otel-collector:4317

The constructed :class:`AppSettings` instance is meant to be bound as a
DI singleton (see :mod:`ai_core.di`); modules MUST receive their
configuration through DI rather than instantiating settings themselves.
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path  # noqa: TC003
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_core.config.secrets import ISecretManager
from ai_core.config.validation import ValidationContext
from ai_core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Enums / aliases
# ---------------------------------------------------------------------------
PortInt = Annotated[int, Field(ge=1, le=65_535)]


class LogLevel(str, enum.Enum):
    """Standard syslog-style log levels accepted by the SDK."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, enum.Enum):
    """Deployment environment classifier — drives default behaviors."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


# ---------------------------------------------------------------------------
# Nested settings
# ---------------------------------------------------------------------------
class DatabaseSettings(BaseSettings):
    """Async SQLAlchemy / Postgres configuration.

    Attributes:
        dsn: Async SQLAlchemy DSN, e.g. ``postgresql+asyncpg://...``.
        pool_size: Number of permanent connections in the pool.
        max_overflow: Burst connections beyond ``pool_size``.
        pool_timeout_seconds: Wait time before a checkout fails.
        echo_sql: Whether SQLAlchemy should echo SQL (DEBUG only).
    """

    model_config = SettingsConfigDict(extra="ignore")

    dsn: SecretStr = Field(
        default=SecretStr("postgresql+asyncpg://eaap:eaap@localhost:5432/eaap"),
        description="Async SQLAlchemy DSN.",
    )
    pool_size: int = Field(default=10, ge=1, le=200)
    max_overflow: int = Field(default=20, ge=0, le=500)
    pool_timeout_seconds: float = Field(default=30.0, gt=0)
    echo_sql: bool = False


class VectorDBSettings(BaseSettings):
    """Vector database configuration (pgvector / external)."""

    model_config = SettingsConfigDict(extra="ignore")

    backend: Literal["pgvector", "pinecone", "weaviate", "noop"] = "pgvector"
    collection: str = "eaap_default"
    embedding_dimensions: int = Field(default=1536, ge=1)
    endpoint: AnyHttpUrl | None = None
    api_key: SecretStr | None = None


class StorageSettings(BaseSettings):
    """Blob / object storage configuration (S3 by default)."""

    model_config = SettingsConfigDict(extra="ignore")

    backend: Literal["s3", "gcs", "azure_blob", "local"] = "s3"
    bucket: str = "eaap-artifacts"
    region: str = "us-east-1"
    endpoint_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional override for S3-compatible endpoints (MinIO, LocalStack).",
    )


class LLMSettings(BaseSettings):
    """LiteLLM-fronted LLM proxy configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    default_model: str = Field(
        default="bedrock/anthropic.claude-sonnet-4-6",
        description="LiteLLM model identifier used when an agent does not override.",
    )
    fallback_models: list[str] = Field(default_factory=list)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_initial_backoff_seconds: float = Field(default=0.5, gt=0)
    retry_max_backoff_seconds: float = Field(default=10.0, gt=0)
    proxy_base_url: AnyHttpUrl | None = None
    proxy_api_key: SecretStr | None = None


class BudgetSettings(BaseSettings):
    """Per-tenant / per-agent budget enforcement."""

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    default_daily_token_limit: int = Field(default=1_000_000, ge=0)
    default_daily_usd_limit: float = Field(default=50.0, ge=0.0)
    hard_fail_on_exceeded: bool = True


class ObservabilitySettings(BaseSettings):
    """OpenTelemetry + LangFuse observability configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    service_name: str = "ai-core-sdk"
    otel_endpoint: AnyHttpUrl | None = Field(
        default=None,
        description="OTLP/gRPC collector endpoint. Disables export when None.",
    )
    otel_insecure: bool = True
    console_export_in_dev: bool = Field(
        default=True,
        description=(
            "Install an OTel ConsoleSpanExporter when no collector endpoint "
            "is configured AND the environment is local/dev. Gives developers "
            "immediate trace visibility without standing up infrastructure."
        ),
    )
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    langfuse_host: AnyHttpUrl | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    log_level: LogLevel = LogLevel.INFO
    log_format: Literal["text", "structured"] = Field(
        default="text",
        description=(
            "When 'text' (default), logs render as colorized key=value for local dev. "
            "When 'structured', logs render as JSON for production ingestion."
        ),
    )
    fail_open: bool = Field(
        default=True,
        description=(
            "When True (default, recommended for production), backend errors "
            "(OTel exporter, LangFuse client) are caught and logged. When False "
            "(recommended for local/dev), they re-raise so misconfigured "
            "exporters surface immediately."
        ),
    )


class SecuritySettings(BaseSettings):
    """OPA / AuthZ configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    opa_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://localhost:8181"),
        description="Base URL of the OPA sidecar.",
    )
    opa_decision_path: str = Field(
        default="eaap/authz/allow",
        description="Decision document path; combined with /v1/data/<path>.",
    )
    opa_request_timeout_seconds: float = Field(default=2.0, gt=0)
    fail_closed: bool = Field(
        default=True,
        description="If True, deny on OPA error; if False, allow (use with caution).",
    )
    opa_health_path: str = Field(
        default="/health",
        description=(
            "Path appended to opa_url for the reachability probe. Override for "
            "deployments where OPA is mounted at a non-standard prefix "
            "(e.g., '/opa/health' behind an API gateway)."
        ),
    )
    jwt_audience: str | None = None
    jwt_issuer: str | None = None


class AgentSettings(BaseSettings):
    """Agent runtime defaults (memory compaction, recursion limits, …)."""

    model_config = SettingsConfigDict(extra="ignore")

    memory_compaction_token_threshold: int = Field(default=8_000, ge=512)
    memory_compaction_target_tokens: int = Field(default=2_000, ge=128)
    compaction_timeout_seconds: float = Field(default=30.0, gt=0)  # NEW
    max_recursion_depth: int = Field(default=25, ge=1, le=200)
    essential_entity_keys: list[str] = Field(
        default_factory=lambda: ["user_id", "tenant_id", "session_id", "task_id"],
        description="State keys that must be preserved across compactions.",
    )


class AuditSettings(BaseSettings):
    """Audit-sink configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    sink_type: Literal["null", "otel_event", "jsonl"] = "null"
    jsonl_path: Path | None = None  # required when sink_type == "jsonl"


class HealthSettings(BaseSettings):
    """Health-probe configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    probe_timeout_seconds: float = Field(default=2.0, gt=0)


class MCPSettings(BaseSettings):
    """MCP transport / pool configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    pool_enabled: bool = Field(
        default=True,
        description=(
            "When True (default), MCP connections are pooled per component_id "
            "and reused across calls. Set False for debugging or to ensure "
            "every call uses a fresh transport."
        ),
    )
    pool_idle_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description=(
            "Connections idle longer than this are closed and reopened on next "
            "checkout. Default 5 minutes — matches typical server-side timeout."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------
class AppSettings(BaseSettings):
    """Aggregated SDK settings — bound as a DI singleton.

    All env vars are prefixed with ``EAAP_`` and use ``__`` as a nested
    delimiter, e.g.::

        EAAP_ENVIRONMENT=staging
        EAAP_DATABASE__POOL_SIZE=25
        EAAP_LLM__DEFAULT_MODEL=bedrock/anthropic.claude-opus-4-7

    Attributes:
        environment: Deployment classifier influencing defaults.
        service_name: Human-readable service identity used in logs/traces.
        database: Async SQLAlchemy configuration.
        vector_db: Vector store configuration.
        storage: Object storage configuration.
        llm: LiteLLM proxy + retry configuration.
        budget: Quota enforcement configuration.
        observability: OpenTelemetry + LangFuse configuration.
        security: OPA / authz configuration.
        agent: Agent runtime defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="EAAP_",
        env_nested_delimiter="__",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Environment = Environment.LOCAL
    service_name: str = "ai-core-sdk"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    health: HealthSettings = Field(default_factory=HealthSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    @field_validator("service_name")
    @classmethod
    def _service_name_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("service_name must be a non-empty string")
        return value.strip()

    def is_production(self) -> bool:
        """Return ``True`` when running in a production-like environment."""
        return self.environment is Environment.PROD

    def validate_for_runtime(
        self,
        *,
        secret_manager: ISecretManager | None = None,
    ) -> None:
        """Collect-all runtime validation. Raises :class:`ConfigurationError` once.

        Pydantic enforces type and ``Field`` constraints at construction time,
        but a few real-world invariants slip through:

        * Required strings can be set to the empty string by an environment
          override.
        * Cross-field invariants (e.g. compaction target < threshold) cannot
          be expressed as single-field constraints.
        * The ``secret_manager`` parameter is forward-looking: future
          settings groups may carry :class:`SecretRef` instances, and we
          want the wiring to be in place before that happens.

        Args:
            secret_manager: Optional :class:`ISecretManager` used to resolve
                any :class:`SecretRef` instances reachable from this settings
                tree. If ``None``, secret-resolution checks are skipped.

        Raises:
            ConfigurationError: If at least one issue is found. The exception
                ``details["issues"]`` field is a list of
                ``{"path", "message", "hint"}`` dicts — one per problem.
        """
        ctx = ValidationContext()

        # llm.default_model must be a non-empty, non-blank string.
        if not (self.llm.default_model and self.llm.default_model.strip()):
            ctx.fail(
                path="llm.default_model",
                message="must be a non-empty model identifier",
                hint=(
                    "set env var EAAP_LLM__DEFAULT_MODEL=<model-id> "
                    "or override AppSettings.llm.default_model in code"
                ),
            )

        # llm.fallback_models entries must be non-blank.
        for idx, model in enumerate(self.llm.fallback_models):
            if not (model and model.strip()):
                ctx.fail(
                    path=f"llm.fallback_models[{idx}]",
                    message="fallback model identifier must be non-empty",
                    hint="remove the entry or set a real model id",
                )

        # Cross-field: compaction target must be strictly less than threshold.
        if (
            self.agent.memory_compaction_target_tokens
            >= self.agent.memory_compaction_token_threshold
        ):
            ctx.fail(
                path="agent.memory_compaction_target_tokens",
                message=(
                    "must be strictly less than "
                    "agent.memory_compaction_token_threshold "
                    f"(target={self.agent.memory_compaction_target_tokens}, "
                    f"threshold={self.agent.memory_compaction_token_threshold})"
                ),
                hint=(
                    "set EAAP_AGENT__MEMORY_COMPACTION_TARGET_TOKENS to a value "
                    "strictly less than EAAP_AGENT__MEMORY_COMPACTION_TOKEN_THRESHOLD"
                ),
            )

        # secret_manager type-check (forward-looking).
        # The isinstance guard is intentional: callers may pass a wrong type at
        # runtime even though the static type is ISecretManager | None.
        _sm: object = secret_manager  # widen to object so mypy does not mark the branch unreachable
        if _sm is not None and not isinstance(_sm, ISecretManager):
            ctx.fail(
                path="secret_manager",
                message=f"must be an ISecretManager, got {type(_sm).__name__}",
                hint="pass an instance of ai_core.config.secrets.ISecretManager",
            )

        if ctx.has_issues:
            raise ConfigurationError(
                f"Runtime configuration is invalid: {len(ctx.issues)} issue(s) found.",
                details={"issues": [
                    {"path": i.path, "message": i.message, "hint": i.hint}
                    for i in ctx.issues
                ]},
            )


# ---------------------------------------------------------------------------
# Accessor — used by the DI container only; never read directly elsewhere.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a process-wide cached :class:`AppSettings` instance.

    The cache is populated on first access and is intentionally process-scoped;
    tests should call :func:`get_settings.cache_clear` between cases or — better —
    inject an ``AppSettings`` override through the DI container instead of
    relying on this accessor.

    Returns:
        A validated :class:`AppSettings` populated from the environment.
    """
    return AppSettings()
