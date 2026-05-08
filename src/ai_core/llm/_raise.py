"""Default :class:`ILLMClient` binding when no real adapter is installed.

The pre-v1 default :class:`AgentModule` bound :class:`LiteLLMClient` as
the production :class:`ILLMClient`, which forced :mod:`litellm` (and
its ~30MB of transitive dependencies) onto every consumer of the SDK.
v1 demotes :mod:`litellm` to an optional extra; the default binding is
this stub, which raises a clear :class:`ConfigurationError` on first
use directing the host to install the right extra or supply their own
:class:`ILLMClient`.

Hosts that want the LiteLLM-backed adapter compose
:class:`ai_core.llm.LiteLLMModule` alongside :class:`AgentModule`::

    from ai_core import AICoreApp
    from ai_core.llm import LiteLLMModule

    async with AICoreApp(modules=[LiteLLMModule()]) as app: ...

Tests that exercise non-LLM behaviour use :class:`ScriptedLLM` from
:mod:`ai_core.testing` and do not need the extra installed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMStreamChunk
from ai_core.exceptions import ConfigurationError, ErrorCode

_INSTALL_HINT = (
    "No ILLMClient implementation is bound. Install the LiteLLM-backed "
    "default with `pip install ai-eaap-sdk[litellm]` and compose "
    "`LiteLLMModule()` alongside `AgentModule`, or bind your own "
    "ILLMClient implementation via a custom `Module`."
)


class RaiseOnUseLLMClient(ILLMClient):
    """Default :class:`ILLMClient` that fails loudly on first call.

    Importing the SDK does not require :mod:`litellm`; a host that
    invokes an agent without supplying an :class:`ILLMClient` gets a
    :class:`ConfigurationError` carrying the install hint instead of an
    obscure ``ImportError`` at module load.
    """

    async def complete(
        self,
        *,
        model: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        raise ConfigurationError(
            _INSTALL_HINT,
            error_code=ErrorCode.CONFIG_INVALID,
            details={
                "missing_binding": "ILLMClient",
                "extras": "litellm",
            },
        )

    def astream(
        self,
        *,
        model: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        raise ConfigurationError(
            _INSTALL_HINT,
            error_code=ErrorCode.CONFIG_INVALID,
            details={
                "missing_binding": "ILLMClient",
                "extras": "litellm",
            },
        )


__all__ = ["RaiseOnUseLLMClient"]
