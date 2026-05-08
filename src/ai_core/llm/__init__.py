"""LLM sub-package — LiteLLM-fronted clients with budgeting and retries.

Heavyweight adapters (``LiteLLMClient``, ``LiteLLMModule``) are loaded
lazily via :pep:`562` so ``import ai_core.llm`` does not pull
:mod:`litellm` into environments where the optional ``[litellm]`` extra
is absent. The default :class:`RaiseOnUseLLMClient` and the budget
service have no heavy deps and are eagerly available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_core.llm._raise import RaiseOnUseLLMClient
from ai_core.llm.budget import InMemoryBudgetService

if TYPE_CHECKING:  # pragma: no cover — surfaces names to type checkers without import
    from ai_core.llm.litellm_client import LiteLLMClient
    from ai_core.llm.module import LiteLLMModule

__all__ = [
    "InMemoryBudgetService",
    "LiteLLMClient",
    "LiteLLMModule",
    "RaiseOnUseLLMClient",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve LiteLLM-backed names so litellm stays optional."""
    if name == "LiteLLMClient":
        from ai_core.llm.litellm_client import LiteLLMClient
        return LiteLLMClient
    if name == "LiteLLMModule":
        from ai_core.llm.module import LiteLLMModule
        return LiteLLMModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
