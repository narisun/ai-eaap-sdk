"""Configuration sub-package.

Re-exports the primary settings and secret-manager symbols so that
host applications and downstream SDK modules can do::

    from ai_core.config import AppSettings, ISecretManager, get_settings

without having to know the file layout.
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
    get_settings,
)

__all__ = [
    "AppSettings",
    "DatabaseSettings",
    "LLMSettings",
    "ObservabilitySettings",
    "SecuritySettings",
    "StorageSettings",
    "VectorDBSettings",
    "BudgetSettings",
    "LogLevel",
    "get_settings",
    "ISecretManager",
    "EnvSecretManager",
    "SecretRef",
]
