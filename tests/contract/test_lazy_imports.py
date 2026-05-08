"""Pin the v1 contract that ``import ai_core`` does not pull in any extras.

Heavyweight provider adapters — :mod:`litellm`, :mod:`langfuse`,
:mod:`fastmcp` — moved to optional ``[litellm]`` / ``[langfuse]`` /
``[mcp]`` extras in v1. Hosts that supply their own LLM / observability /
MCP backends get a lightweight base install.

This test fails loudly if a future contributor adds a module-level
``import litellm`` / ``import langfuse`` / ``import fastmcp`` somewhere
on the ``ai_core`` import path. Either:

* defer the import to function scope (``import litellm`` inside the
  method that uses it), or
* gate it behind :pep:`562` ``__getattr__`` lazy resolution like
  :mod:`ai_core.llm.__init__` does for :class:`LiteLLMClient` and
  :class:`LiteLLMModule`.

Note: :mod:`langgraph` and :mod:`langchain_core` are intentionally
*not* checked here — :class:`BaseAgent` depends on them at module
load (StateGraph, AIMessage, ToolMessage). Making those optional
requires splitting :class:`BaseAgent` into a pluggable-orchestrator
shape and is tracked as future work.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.contract


_OPTIONAL_EXTRAS = ("litellm", "langfuse", "fastmcp")


def test_import_ai_core_does_not_pull_in_optional_extras() -> None:
    """Run the import audit in a fresh subprocess.

    Mutating ``sys.modules`` in-process to force a fresh ``import ai_core``
    invalidates module references already held by sibling tests in the
    same process and causes spurious failures. A subprocess is the only
    way to observe the import side-effects in true isolation.
    """
    probe = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import ai_core  # noqa: F401\n"
        f"extras = {_OPTIONAL_EXTRAS!r}\n"
        "after = set(sys.modules) - before\n"
        "pulled = sorted(m for m in after if m.split('.', 1)[0] in extras)\n"
        "import json; print(json.dumps(pulled))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    pulled = json.loads(result.stdout.strip().splitlines()[-1])
    assert not pulled, (
        f"`import ai_core` pulled in optional-extra modules: {pulled}. "
        "Defer the imports to function scope or gate them behind PEP 562 "
        "__getattr__ in the relevant package __init__."
    )
