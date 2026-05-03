"""DI bindings for the SDK.

The :class:`AgentModule` is the canonical place where abstract
interfaces (see :mod:`ai_core.di.interfaces`) are bound to concrete
implementations. It is structured so that:

* **Step 1 (this commit)** binds settings + the default
  :class:`EnvSecretManager`. All other interfaces are *declared* but
  intentionally not yet bound to concrete classes — the corresponding
  modules (``llm``, ``observability``, ``security``, ``persistence``)
  arrive in subsequent steps and will register their concretes here.
* **Hosts override anything** by passing additional :class:`Module`
  instances to :class:`Container.build` — the last binding wins, which
  makes test fakes trivial to wire in.

Note:
    There are NO global singletons. Settings and clients are bound as
    DI singletons *within a container* — a fresh container yields a
    fresh dependency graph. Tests therefore enjoy full isolation.
"""

from __future__ import annotations

from injector import Module, provider, singleton

from ai_core.config.secrets import EnvSecretManager, ISecretManager
from ai_core.config.settings import AppSettings


class AgentModule(Module):
    """Default top-level DI module for agentic applications.

    Subclass this module (or compose alongside it) to override bindings
    for specific environments. For example, a production deployment may
    want::

        class ProdModule(AgentModule):
            def configure(self, binder: injector.Binder) -> None:
                super().configure(binder)
                binder.bind(IStorageProvider, to=S3StorageProvider, scope=singleton)
                binder.bind(ILLMClient, to=LiteLLMClient, scope=singleton)
                ...

    Args:
        settings: Optional pre-built :class:`AppSettings`. If omitted,
            settings are loaded from the environment via
            :func:`ai_core.config.settings.get_settings`.
        secret_manager: Optional :class:`ISecretManager` instance. Defaults
            to :class:`EnvSecretManager` (env-var backed).
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
        # Local import keeps the test seam clean: callers can override
        # via the constructor without monkey-patching get_settings().
        from ai_core.config.settings import get_settings

        return get_settings()

    # ----- Secret manager ---------------------------------------------------
    @singleton
    @provider
    def provide_secret_manager(self) -> ISecretManager:
        """Return the bound :class:`ISecretManager` singleton."""
        return self._secret_manager or EnvSecretManager()


__all__ = ["AgentModule"]
