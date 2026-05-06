# Settings reference

Auto-generated from `src/ai_core/config/settings.py`. Do not edit by hand — run `uv run python scripts/generate_settings_doc.py` to regenerate.

Environment variable names use the prefix `EAAP_` and `__` as the nested-group delimiter (e.g. `EAAP_DATABASE__DSN`).

## AppSettings

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `environment` | `Environment` | `<Environment.LOCAL: 'local'>` | `EAAP_ENVIRONMENT` |  |
| `service_name` | `str` | `'ai-core-sdk'` | `EAAP_SERVICE_NAME` |  |
| `database` | `DatabaseSettings` | *(factory)* | `EAAP_DATABASE` |  |
| `vector_db` | `VectorDBSettings` | *(factory)* | `EAAP_VECTOR_DB` |  |
| `storage` | `StorageSettings` | *(factory)* | `EAAP_STORAGE` |  |
| `llm` | `LLMSettings` | *(factory)* | `EAAP_LLM` |  |
| `budget` | `BudgetSettings` | *(factory)* | `EAAP_BUDGET` |  |
| `observability` | `ObservabilitySettings` | *(factory)* | `EAAP_OBSERVABILITY` |  |
| `security` | `SecuritySettings` | *(factory)* | `EAAP_SECURITY` |  |
| `agent` | `AgentSettings` | *(factory)* | `EAAP_AGENT` |  |
| `audit` | `AuditSettings` | *(factory)* | `EAAP_AUDIT` |  |
| `health` | `HealthSettings` | *(factory)* | `EAAP_HEALTH` |  |
| `mcp` | `MCPSettings` | *(factory)* | `EAAP_MCP` |  |

## DatabaseSettings (`AppSettings.database`)

Async SQLAlchemy / Postgres configuration.

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `dsn` | `SecretStr` | `SecretStr('**********')` | `EAAP_DATABASE__DSN` | Async SQLAlchemy DSN. |
| `pool_size` | `int` | `10` | `EAAP_DATABASE__POOL_SIZE` |  |
| `max_overflow` | `int` | `20` | `EAAP_DATABASE__MAX_OVERFLOW` |  |
| `pool_timeout_seconds` | `float` | `30.0` | `EAAP_DATABASE__POOL_TIMEOUT_SECONDS` |  |
| `echo_sql` | `bool` | `False` | `EAAP_DATABASE__ECHO_SQL` |  |

## VectorDBSettings (`AppSettings.vector_db`)

Vector database configuration (pgvector / external).

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `backend` | `Literal[pgvector, pinecone, weaviate, noop]` | `'pgvector'` | `EAAP_VECTOR_DB__BACKEND` |  |
| `collection` | `str` | `'eaap_default'` | `EAAP_VECTOR_DB__COLLECTION` |  |
| `embedding_dimensions` | `int` | `1536` | `EAAP_VECTOR_DB__EMBEDDING_DIMENSIONS` |  |
| `endpoint` | `UnionType[AnyHttpUrl, None]` | `None` | `EAAP_VECTOR_DB__ENDPOINT` |  |
| `api_key` | `UnionType[SecretStr, None]` | `None` | `EAAP_VECTOR_DB__API_KEY` |  |

## StorageSettings (`AppSettings.storage`)

Blob / object storage configuration (S3 by default).

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `backend` | `Literal[s3, gcs, azure_blob, local]` | `'s3'` | `EAAP_STORAGE__BACKEND` |  |
| `bucket` | `str` | `'eaap-artifacts'` | `EAAP_STORAGE__BUCKET` |  |
| `region` | `str` | `'us-east-1'` | `EAAP_STORAGE__REGION` |  |
| `endpoint_url` | `UnionType[AnyHttpUrl, None]` | `None` | `EAAP_STORAGE__ENDPOINT_URL` | Optional override for S3-compatible endpoints (MinIO, LocalStack). |

## LLMSettings (`AppSettings.llm`)

LiteLLM-fronted LLM proxy configuration.

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `default_model` | `str` | `'bedrock/anthropic.claude-sonnet-4-6'` | `EAAP_LLM__DEFAULT_MODEL` | LiteLLM model identifier used when an agent does not override. |
| `fallback_models` | `list[str]` | *(factory)* | `EAAP_LLM__FALLBACK_MODELS` |  |
| `request_timeout_seconds` | `float` | `60.0` | `EAAP_LLM__REQUEST_TIMEOUT_SECONDS` |  |
| `max_retries` | `int` | `3` | `EAAP_LLM__MAX_RETRIES` |  |
| `retry_initial_backoff_seconds` | `float` | `0.5` | `EAAP_LLM__RETRY_INITIAL_BACKOFF_SECONDS` |  |
| `retry_max_backoff_seconds` | `float` | `10.0` | `EAAP_LLM__RETRY_MAX_BACKOFF_SECONDS` |  |
| `proxy_base_url` | `UnionType[AnyHttpUrl, None]` | `None` | `EAAP_LLM__PROXY_BASE_URL` |  |
| `proxy_api_key` | `UnionType[SecretStr, None]` | `None` | `EAAP_LLM__PROXY_API_KEY` |  |
| `prompt_cache_enabled` | `bool` | `True` | `EAAP_LLM__PROMPT_CACHE_ENABLED` | When True (default), automatically apply Anthropic cache_control headers to system prompts and stable conversation history. Skipped for non-Anthropic providers and for prompts below the configured thresholds. Set False to disable for tests that require deterministic cache-miss responses. |
| `prompt_cache_min_messages` | `int` | `6` | `EAAP_LLM__PROMPT_CACHE_MIN_MESSAGES` |  |
| `prompt_cache_min_tokens` | `int` | `1024` | `EAAP_LLM__PROMPT_CACHE_MIN_TOKENS` |  |

## BudgetSettings (`AppSettings.budget`)

Per-tenant / per-agent budget enforcement.

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `enabled` | `bool` | `True` | `EAAP_BUDGET__ENABLED` |  |
| `default_daily_token_limit` | `int` | `1000000` | `EAAP_BUDGET__DEFAULT_DAILY_TOKEN_LIMIT` |  |
| `default_daily_usd_limit` | `float` | `50.0` | `EAAP_BUDGET__DEFAULT_DAILY_USD_LIMIT` |  |
| `hard_fail_on_exceeded` | `bool` | `True` | `EAAP_BUDGET__HARD_FAIL_ON_EXCEEDED` |  |

## ObservabilitySettings (`AppSettings.observability`)

OpenTelemetry + LangFuse observability configuration.

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `service_name` | `str` | `'ai-core-sdk'` | `EAAP_OBSERVABILITY__SERVICE_NAME` |  |
| `otel_endpoint` | `UnionType[AnyHttpUrl, None]` | `None` | `EAAP_OBSERVABILITY__OTEL_ENDPOINT` | OTLP/gRPC collector endpoint. Disables export when None. |
| `otel_insecure` | `bool` | `True` | `EAAP_OBSERVABILITY__OTEL_INSECURE` |  |
| `console_export_in_dev` | `bool` | `True` | `EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV` | Install an OTel ConsoleSpanExporter when no collector endpoint is configured AND the environment is local/dev. Gives developers immediate trace visibility without standing up infrastructure. |
| `sample_ratio` | `float` | `1.0` | `EAAP_OBSERVABILITY__SAMPLE_RATIO` |  |
| `langfuse_host` | `UnionType[AnyHttpUrl, None]` | `None` | `EAAP_OBSERVABILITY__LANGFUSE_HOST` |  |
| `langfuse_public_key` | `UnionType[SecretStr, None]` | `None` | `EAAP_OBSERVABILITY__LANGFUSE_PUBLIC_KEY` |  |
| `langfuse_secret_key` | `UnionType[SecretStr, None]` | `None` | `EAAP_OBSERVABILITY__LANGFUSE_SECRET_KEY` |  |
| `log_level` | `LogLevel` | `<LogLevel.INFO: 'INFO'>` | `EAAP_OBSERVABILITY__LOG_LEVEL` |  |
| `log_format` | `Literal[text, structured]` | `'text'` | `EAAP_OBSERVABILITY__LOG_FORMAT` | When 'text' (default), logs render as colorized key=value for local dev. When 'structured', logs render as JSON for production ingestion. |
| `fail_open` | `bool` | `True` | `EAAP_OBSERVABILITY__FAIL_OPEN` | When True (default, recommended for production), backend errors (OTel exporter, LangFuse client) are caught and logged. When False (recommended for local/dev), they re-raise so misconfigured exporters surface immediately. |

## SecuritySettings (`AppSettings.security`)

OPA / AuthZ configuration.

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `opa_url` | `AnyHttpUrl` | `AnyHttpUrl('http://localhost:8181/')` | `EAAP_SECURITY__OPA_URL` | Base URL of the OPA sidecar. |
| `opa_decision_path` | `str` | `'eaap/authz/allow'` | `EAAP_SECURITY__OPA_DECISION_PATH` | Decision document path; combined with /v1/data/<path>. |
| `opa_request_timeout_seconds` | `float` | `2.0` | `EAAP_SECURITY__OPA_REQUEST_TIMEOUT_SECONDS` |  |
| `fail_closed` | `bool` | `True` | `EAAP_SECURITY__FAIL_CLOSED` | If True, deny on OPA error; if False, allow (use with caution). |
| `opa_health_path` | `str` | `'/health'` | `EAAP_SECURITY__OPA_HEALTH_PATH` | Path appended to opa_url for the reachability probe. Override for deployments where OPA is mounted at a non-standard prefix (e.g., '/opa/health' behind an API gateway). |
| `jwt_audience` | `UnionType[str, None]` | `None` | `EAAP_SECURITY__JWT_AUDIENCE` |  |
| `jwt_issuer` | `UnionType[str, None]` | `None` | `EAAP_SECURITY__JWT_ISSUER` |  |

## AgentSettings (`AppSettings.agent`)

Agent runtime defaults (memory compaction, recursion limits, …).

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `memory_compaction_token_threshold` | `int` | `8000` | `EAAP_AGENT__MEMORY_COMPACTION_TOKEN_THRESHOLD` |  |
| `memory_compaction_target_tokens` | `int` | `2000` | `EAAP_AGENT__MEMORY_COMPACTION_TARGET_TOKENS` |  |
| `compaction_timeout_seconds` | `float` | `30.0` | `EAAP_AGENT__COMPACTION_TIMEOUT_SECONDS` |  |
| `max_recursion_depth` | `int` | `25` | `EAAP_AGENT__MAX_RECURSION_DEPTH` |  |
| `essential_entity_keys` | `list[str]` | *(factory)* | `EAAP_AGENT__ESSENTIAL_ENTITY_KEYS` | State keys that must be preserved across compactions. |

## AuditSettings (`AppSettings.audit`)

Audit-sink configuration.

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `sink_type` | `Literal[null, otel_event, jsonl, sentry, datadog]` | `'null'` | `EAAP_AUDIT__SINK_TYPE` |  |
| `jsonl_path` | `UnionType[Path, None]` | `None` | `EAAP_AUDIT__JSONL_PATH` |  |
| `redaction_profile` | `Literal[off, standard, strict]` | `'off'` | `EAAP_AUDIT__REDACTION_PROFILE` | Profile name for the DI-bound PayloadRedactor. 'off' is identity (no redaction); 'standard' chains a RegexRedactor (email, phone, ssn, credit_card, ipv4) with a KeyNameRedactor (default secret/token key set); 'strict' adds a 6+digit number pattern that catches IDs/account numbers (higher false-positive rate). |
| `sentry_dsn` | `UnionType[SecretStr, None]` | `None` | `EAAP_AUDIT__SENTRY_DSN` | Required when sink_type='sentry'. Project-level DSN issued by Sentry. |
| `sentry_environment` | `UnionType[str, None]` | `None` | `EAAP_AUDIT__SENTRY_ENVIRONMENT` | Optional environment tag on Sentry events (e.g. 'prod', 'staging'). |
| `sentry_release` | `UnionType[str, None]` | `None` | `EAAP_AUDIT__SENTRY_RELEASE` | Optional release identifier on Sentry events. |
| `sentry_sample_rate` | `float` | `1.0` | `EAAP_AUDIT__SENTRY_SAMPLE_RATE` | Fraction of audit events sent to Sentry. 1.0 = all; 0.0 = none. |
| `datadog_api_key` | `UnionType[SecretStr, None]` | `None` | `EAAP_AUDIT__DATADOG_API_KEY` | Required when sink_type='datadog'. Datadog API key. |
| `datadog_app_key` | `UnionType[SecretStr, None]` | `None` | `EAAP_AUDIT__DATADOG_APP_KEY` | Optional Datadog application key (some endpoints require it). |
| `datadog_site` | `str` | `'datadoghq.com'` | `EAAP_AUDIT__DATADOG_SITE` | Datadog site (e.g. 'datadoghq.com', 'datadoghq.eu', 'us3.datadoghq.com'). |
| `datadog_source` | `str` | `'ai-core-sdk'` | `EAAP_AUDIT__DATADOG_SOURCE` | Source name attached to Datadog events (free text). |
| `datadog_environment` | `UnionType[str, None]` | `None` | `EAAP_AUDIT__DATADOG_ENVIRONMENT` | Optional environment tag (added as 'env:<value>' to every event). |

## HealthSettings (`AppSettings.health`)

Health-probe configuration.

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `probe_timeout_seconds` | `float` | `2.0` | `EAAP_HEALTH__PROBE_TIMEOUT_SECONDS` |  |

## MCPSettings (`AppSettings.mcp`)

MCP transport / pool configuration.

| Field | Type | Default | Env var | Description |
| --- | --- | --- | --- | --- |
| `pool_enabled` | `bool` | `True` | `EAAP_MCP__POOL_ENABLED` | When True (default), MCP connections are pooled per component_id and reused across calls. Set False for debugging or to ensure every call uses a fresh transport. |
| `pool_idle_seconds` | `float` | `300.0` | `EAAP_MCP__POOL_IDLE_SECONDS` | Connections idle longer than this are closed and reopened on next checkout. Default 5 minutes — matches typical server-side timeout. |

