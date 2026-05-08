"""Opt-in DI module that binds the LiteLLM-backed :class:`ILLMClient`.

Compose alongside :class:`ai_core.di.AgentModule` to enable the
production LLM adapter::

    from ai_core import AICoreApp
    from ai_core.llm import LiteLLMModule

    async with AICoreApp(modules=[LiteLLMModule()]) as app:
        agent = app.agent(MyAgent)
        ...

Without this module, the default :class:`AgentModule` binds
:class:`RaiseOnUseLLMClient`, which raises a :class:`ConfigurationError`
on first call directing the host to install ``ai-eaap-sdk[litellm]``.

Importing this module requires :mod:`litellm`; the ``ImportError`` from
a missing extra surfaces here at module-load time, not at SDK import,
keeping the base ``import ai_core`` lightweight.
"""

from __future__ import annotations

from injector import Module, provider, singleton

from ai_core.config.settings import LLMSettings
from ai_core.di.interfaces import IBudgetService, ILLMClient, IObservabilityProvider
from ai_core.llm.litellm_client import LiteLLMClient


class LiteLLMModule(Module):
    """Opt-in module binding :class:`LiteLLMClient` as the production LLM."""

    @singleton
    @provider
    def provide_llm_client(
        self,
        llm_settings: LLMSettings,
        budget: IBudgetService,
        observability: IObservabilityProvider,
    ) -> ILLMClient:
        """Return the LiteLLM-backed client singleton."""
        return LiteLLMClient(llm_settings, budget, observability)


__all__ = ["LiteLLMModule"]
