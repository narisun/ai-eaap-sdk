"""Configuration sub-package.

Re-exports the primary settings and secret-manager symbols so that
host applications and downstream SDK modules can do::

    from ai_core.config import AppSettings, ISecretManager

without having to know the file layout.

Note: in v1 the process-wide ``get_settings()`` accessor was removed.
Construct :class:`AppSettings` directly (it reads env / YAML / defaults
via Pydantic Settings) or hand a pre-built instance to
:class:`ai_core.app.AICoreApp` / :class:`ai_core.di.AgentModule`. The DI
container caches the binding as a singleton — that is the only intended
sharing seam.
"""

from __future__ import annotations

from ai_core.config.secrets import (
    EnvSecretManager,
    ISecretManager,
    SecretRef,
)
from ai_core.config.settings import (
    AppSettings,
    BudgetSettings,
    DatabaseSettings,
    LLMSettings,
    LogLevel,
    ObservabilitySettings,
    SecuritySettings,
    StorageSettings,
    VectorDBSettings,
)

__all__ = [
    "AppSettings",
    "BudgetSettings",
    "DatabaseSettings",
    "EnvSecretManager",
    "ISecretManager",
    "LLMSettings",
    "LogLevel",
    "ObservabilitySettings",
    "SecretRef",
    "SecuritySettings",
    "StorageSettings",
    "VectorDBSettings",
]
