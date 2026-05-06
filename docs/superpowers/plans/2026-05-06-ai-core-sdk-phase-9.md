# ai-core-sdk Phase 9 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ai_core.testing` so consumers can `pip install ai-core-sdk[testing]`, add `pytest_plugins = ["ai_core.testing.pytest_plugin"]` to their conftest, and write tests against the SDK without forking our internal `tests/conftest.py` fakes.

**Architecture:** Three independent test-side deliverables. (1) Migrate 5 existing `Fake*` classes from `tests/conftest.py` to a new `src/ai_core/testing/` subpackage, with the conftest re-importing them so existing 482 tests stay green. (2) Add `ScriptedLLM` + `make_llm_response` to consolidate 5+ ad-hoc LLM fakes into one canonical implementation. (3) Ship `pytest_plugin.py` exposing 6 fixtures + a `[testing]` optional-dep extra + a recipe doc at `docs/testing.md`.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest>=7.0 (new optional dep). No production code changes outside the new `testing/` subpackage. Spec: `docs/superpowers/specs/2026-05-06-ai-core-sdk-phase-9-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-9-public-testing-surface` (already checked out off `main` post-rego-fix-merge `d19084f`; carries the Phase 9 spec at `f31168b`).

**Working-state hygiene** — do NOT touch:
- `README.md` (top-level)
- `src/ai_core/cli/main.py`, `cli/scaffold.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`
- `tests/integration/**` (Phase 7 territory)
- `tests/contract/**` (Phase 7 territory; surface contract test stays at 30 names — Phase 9 does NOT add to top-level `__all__`)
- All `src/ai_core/` files outside the new `testing/` subpackage

**Mypy baseline:** 21 strict errors in 8 files. `mypy src` total must remain ≤ 21. New `testing/` files must type-check clean.

**Ruff baseline:** 211 errors at `d19084f` (rego-fix merge). Total must remain ≤ 211.

**Pytest baseline (post-Phase-8 + rego-fix merge):** 482 passing across `tests/unit tests/component tests/contract`.

**Per-step gate (every commit):**
- `ruff check <files-touched>` — no NEW violations
- `pytest tests/unit tests/component tests/contract -q` — must pass
- `pytest tests/integration -q` — must pass when Docker is up; auto-skip otherwise
- `mypy <files-touched>` strict — no new errors
- `mypy src 2>&1 | tail -1` — total ≤ 21

**Python interpreter:** `/Users/admin-h26/EAAP/.venv/bin/python` (no local venv inside the SDK).

**Pre-resolved facts (controller verified):**
- `LLMResponse` fields (in order): `model: str`, `content: str`, `tool_calls: Sequence[Mapping[str, Any]]`, `usage: LLMUsage`, `raw: Mapping[str, Any]`, `finish_reason: ... | None = None`. Constructor REQUIRES the first 5; `finish_reason` is optional.
- `LLMUsage` fields: `prompt_tokens: int`, `completion_tokens: int`, `total_tokens: int`, `cost_usd: float | None = None`.
- `ILLMClient.complete` is an abstract method with the exact signature:
  ```python
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
  ```
  All 8 keyword-only args; `ScriptedLLM.complete` MUST match this exactly to satisfy the ABC under mypy strict.
- `tests/conftest.py:53-282` contains the 5 `Fake*` classes that move to `src/ai_core/testing/fakes.py`. Their fixture wrappers (`fake_audit_sink`, etc.) live below the class definitions in the same file and stay where they are.
- `tests/unit/test_conftest_fakes.py` (54 lines) has ~10 tests for the fakes; this file MOVES to `tests/unit/testing/test_fakes.py` (new location).
- 4 ad-hoc LLM fakes in `tests/`:
  - `tests/unit/agents/test_memory.py` — `FakeLLM(summary="...")` + `_SlowFakeLLM` + inline `_RaisingLLM`
  - `tests/unit/app/test_runtime.py` — `_StubLLM` (fixed "ok" response)
  - `tests/component/test_agent_run.py` — `ScriptedLLM(responses: Sequence[str])` (string-only sequence)
  - `tests/component/test_agent_tool_loop.py` — `_ScriptedLLM` (richer; can carry tool_calls)
- The new public `ScriptedLLM` has a different constructor (`responses: Sequence[LLMResponse]`) than `tests/component/test_agent_run.py:ScriptedLLM` (`responses: Sequence[str]`). Migration touches each call site to wrap strings in `make_llm_response(...)`.

**Per-task commit message convention:** Conventional Commits.

---

## Task 1 — Migrate 5 Fake classes to `ai_core.testing`

**Files:**
- Create: `src/ai_core/testing/__init__.py`
- Create: `src/ai_core/testing/fakes.py`
- Create: `tests/unit/testing/__init__.py`
- Move + rename: `tests/unit/test_conftest_fakes.py` → `tests/unit/testing/test_fakes.py`
- Modify: `tests/conftest.py` — remove 5 class definitions; re-import from `ai_core.testing`
- Modify: `pyproject.toml` — add `[testing]` optional-dep extra

### 1a — Create the testing subpackage skeleton

- [ ] **Step 1.1: Create the new directories**

```bash
mkdir -p /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/testing
mkdir -p /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/testing
touch /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/testing/__init__.py
```

- [ ] **Step 1.2: Add the `[testing]` extra to `pyproject.toml`**

In `pyproject.toml`, find the `[project.optional-dependencies]` table. Add an entry:

```toml
testing = ["pytest>=7.0"]
```

Place it in alphabetical order with the existing extras (between `sentry` and `vector-pgvector`, or wherever `t` falls).

Verify:

```bash
grep -A 1 "^testing = " /Users/admin-h26/EAAP/ai-core-sdk/pyproject.toml
```

Expected: shows `testing = ["pytest>=7.0"]`.

### 1b — Move the 5 `Fake*` classes

- [ ] **Step 1.3: Read the existing class definitions**

```bash
sed -n '1,30p' /Users/admin-h26/EAAP/ai-core-sdk/tests/conftest.py
sed -n '50,290p' /Users/admin-h26/EAAP/ai-core-sdk/tests/conftest.py
```

You'll see the imports (lines 1-30) and the 5 class definitions (lines ~50-290). Note the exact imports each class needs: `IPolicyEvaluator`, `PolicyDecision`, `IObservabilityProvider`, `SpanContext`, `ISecretManager`, `IBudgetService`, `IAuditSink`, `AuditRecord`, etc. The classes also use stdlib types (`asyncio`, `contextlib.asynccontextmanager`, `dataclasses`, `datetime`, `typing.Any`, etc.).

- [ ] **Step 1.4: Create `src/ai_core/testing/fakes.py` with the 5 classes**

Copy the 5 class bodies VERBATIM from `tests/conftest.py` into a new file `src/ai_core/testing/fakes.py`. Add the necessary imports at the top. The file should look like:

```python
"""Fake implementations of the SDK's public protocols, for use in tests.

These fakes are deliberately simple, in-memory, and synchronous-where-possible
so consumer tests can assert on observable state without setting up real
backends. They are NOT production-quality.

Imported by :mod:`ai_core.testing` (no pytest dependency) and re-exported
from :mod:`ai_core.testing.pytest_plugin` for fixture-based access.
"""

from __future__ import annotations

# Stdlib imports needed by the 5 classes (assemble from tests/conftest.py imports).
import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# SDK protocol imports (the fakes implement these).
from ai_core.audit import AuditRecord, IAuditSink
from ai_core.config.secrets import ISecretManager
from ai_core.di.interfaces import (
    IBudgetService,
    IObservabilityProvider,
    IPolicyEvaluator,
    PolicyDecision,
    SpanContext,
)


# --- 5 class bodies copied verbatim from tests/conftest.py ---

class FakePolicyEvaluator(IPolicyEvaluator):
    """..."""
    # (paste the existing body unchanged)


class FakeObservabilityProvider(IObservabilityProvider):
    """..."""
    # (paste the existing body unchanged)


class FakeSecretManager(ISecretManager):
    """..."""
    # (paste the existing body unchanged)


class FakeBudgetService(IBudgetService):
    """..."""
    # (paste the existing body unchanged)


class FakeAuditSink(IAuditSink):
    """..."""
    # (paste the existing body unchanged)


__all__ = [
    "FakeAuditSink",
    "FakeBudgetService",
    "FakeObservabilityProvider",
    "FakePolicyEvaluator",
    "FakeSecretManager",
]
```

(The exact import set may differ — read the existing `tests/conftest.py` imports and copy only what the 5 classes use. Don't import test-only helpers like pytest fixtures.)

- [ ] **Step 1.5: Create `src/ai_core/testing/__init__.py`**

```python
"""Public testing surface for SDK consumers.

Activate the pytest plugin in your conftest.py::

    pytest_plugins = ["ai_core.testing.pytest_plugin"]

Then use the exported fakes (also importable directly here without pytest)::

    from ai_core.testing import FakeAuditSink, ScriptedLLM, make_llm_response
"""

from __future__ import annotations

from ai_core.testing.fakes import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)

__all__ = [
    "FakeAuditSink",
    "FakeBudgetService",
    "FakeObservabilityProvider",
    "FakePolicyEvaluator",
    "FakeSecretManager",
]
```

(Tasks 2 and 3 will extend `__all__` to add `ScriptedLLM` and `make_llm_response`.)

- [ ] **Step 1.6: Smoke-test the new module**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
from ai_core.testing import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)
import ai_core.testing as t
assert set(t.__all__) == {'FakeAuditSink', 'FakeBudgetService', 'FakeObservabilityProvider', 'FakePolicyEvaluator', 'FakeSecretManager'}
print('Phase 9 fakes imported OK')
"
```

Expected: `Phase 9 fakes imported OK`.

### 1c — Update `tests/conftest.py` to re-import

- [ ] **Step 1.7: Remove the 5 class bodies from `tests/conftest.py`**

In `tests/conftest.py`, DELETE the 5 class definitions (~lines 50-290 — actual range is whatever `Fake*` blocks span). The fixture functions (`fake_audit_sink`, `fake_observability`, etc., which sit lower in the file) MUST stay; they reference the classes by name.

After the existing imports block at the top, add a new import line:

```python
from ai_core.testing import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)
```

This brings the names into scope for the fixture functions below.

- [ ] **Step 1.8: Verify the 482-test baseline is still green**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -5
```

Expected: 482 passed. The class migration is structural; no test changes required because the fixture functions still hand out `Fake*` instances by the same names.

If any test fails, the migration introduced a subtle behavioural drift — investigate by comparing the moved class body against the deleted version (`git diff HEAD -- tests/conftest.py src/ai_core/testing/fakes.py`).

### 1d — Move `tests/unit/test_conftest_fakes.py` → `tests/unit/testing/test_fakes.py`

- [ ] **Step 1.9: Move the file**

```bash
git mv /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/test_conftest_fakes.py /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/testing/test_fakes.py
```

- [ ] **Step 1.10: Update its imports**

Read the file:

```bash
cat /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/testing/test_fakes.py
```

The file likely imports the `Fake*` classes from `conftest` directly (or relies on the fixture). If it imports from `tests.conftest` or similar, replace those imports with the new public path:

```python
from ai_core.testing import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)
```

If it consumes via fixture (function args like `fake_audit_sink`), the fixture is still defined in `tests/conftest.py` and works at the new path because pytest discovers conftests up the directory tree. No fixture-side change needed.

- [ ] **Step 1.11: Run the moved tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/testing/test_fakes.py -v 2>&1 | tail -15
```

Expected: same number of tests passing as before (~10 tests).

### 1e — Lint, type-check, full suite, commit

- [ ] **Step 1.12: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/testing/__init__.py \
    src/ai_core/testing/fakes.py \
    tests/conftest.py \
    tests/unit/testing/test_fakes.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/testing/__init__.py \
    src/ai_core/testing/fakes.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on the new src files clean; project total ≤ 21.

- [ ] **Step 1.13: Full unit + component + contract suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -5
```

Expected: 482 passed (unchanged — the move is structural).

- [ ] **Step 1.14: Commit Task 1**

```bash
git add src/ai_core/testing/__init__.py \
        src/ai_core/testing/fakes.py \
        tests/conftest.py \
        tests/unit/testing/__init__.py \
        tests/unit/testing/test_fakes.py \
        pyproject.toml
git commit -m "feat(testing): migrate 5 Fake classes to ai_core.testing + add [testing] extra"
```

(`git mv` from Step 1.9 stages the rename; `git add` here picks up everything else.)

---

## Task 2 — Add `ScriptedLLM` + `make_llm_response`

**Files:**
- Create: `src/ai_core/testing/llm.py`
- Modify: `src/ai_core/testing/__init__.py` (extend `__all__`)
- Create: `tests/unit/testing/test_scripted_llm.py`
- Create: `tests/unit/testing/test_make_llm_response.py`
- Modify: `tests/unit/agents/test_memory.py` (replace `FakeLLM` with `ScriptedLLM`)
- Modify: `tests/unit/app/test_runtime.py` (replace `_StubLLM` with `ScriptedLLM`)
- Modify: `tests/component/test_agent_run.py` (replace local `ScriptedLLM` with `ai_core.testing.ScriptedLLM`)
- Modify: `tests/component/test_agent_tool_loop.py` (replace `_ScriptedLLM` with `ai_core.testing.ScriptedLLM`)

### 2a — Implement `ScriptedLLM` + `make_llm_response`

- [ ] **Step 2.1: Write failing tests for `make_llm_response`**

Create `tests/unit/testing/test_make_llm_response.py`:

```python
"""Tests for ai_core.testing.make_llm_response."""
from __future__ import annotations

import pytest

from ai_core.di.interfaces import LLMResponse, LLMUsage
from ai_core.testing import make_llm_response

pytestmark = pytest.mark.unit


def test_make_llm_response_defaults() -> None:
    r = make_llm_response()
    assert isinstance(r, LLMResponse)
    assert r.content == ""
    assert r.finish_reason == "stop"
    assert r.model == "test-model"
    assert isinstance(r.usage, LLMUsage)
    assert r.usage.prompt_tokens == 10
    assert r.usage.completion_tokens == 20
    assert r.usage.total_tokens == 30
    assert r.tool_calls == []


def test_make_llm_response_with_text_only() -> None:
    r = make_llm_response("hi")
    assert r.content == "hi"
    assert r.finish_reason == "stop"
    assert r.usage.total_tokens == 30


def test_make_llm_response_with_all_fields() -> None:
    r = make_llm_response(
        "hello",
        tool_calls=[{"id": "c1", "function": {"name": "f"}}],
        finish_reason="tool_calls",
        prompt_tokens=5,
        completion_tokens=15,
        model="gpt-test",
    )
    assert r.content == "hello"
    assert r.tool_calls == [{"id": "c1", "function": {"name": "f"}}]
    assert r.finish_reason == "tool_calls"
    assert r.usage.prompt_tokens == 5
    assert r.usage.completion_tokens == 15
    assert r.usage.total_tokens == 20  # prompt + completion
    assert r.model == "gpt-test"
```

- [ ] **Step 2.2: Run the test — verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/testing/test_make_llm_response.py -v 2>&1 | tail -10
```

Expected: ImportError on `ai_core.testing.make_llm_response`.

- [ ] **Step 2.3: Write failing tests for `ScriptedLLM`**

Create `tests/unit/testing/test_scripted_llm.py`:

```python
"""Tests for ai_core.testing.ScriptedLLM."""
from __future__ import annotations

import pytest

from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_scripted_llm_returns_responses_in_order() -> None:
    r1 = make_llm_response("first")
    r2 = make_llm_response("second")
    llm = ScriptedLLM([r1, r2])
    out1 = await llm.complete(model=None, messages=[])
    out2 = await llm.complete(model=None, messages=[])
    assert out1.content == "first"
    assert out2.content == "second"


@pytest.mark.asyncio
async def test_scripted_llm_records_calls() -> None:
    llm = ScriptedLLM([make_llm_response("a"), make_llm_response("b")])
    await llm.complete(model="x", messages=[{"role": "user", "content": "1"}])
    await llm.complete(
        model="y",
        messages=[{"role": "user", "content": "2"}],
        tenant_id="t1",
    )
    assert len(llm.calls) == 2
    assert llm.calls[0]["model"] == "x"
    assert llm.calls[1]["tenant_id"] == "t1"


@pytest.mark.asyncio
async def test_scripted_llm_raises_index_error_on_exhaustion() -> None:
    llm = ScriptedLLM([make_llm_response("only")])
    await llm.complete(model=None, messages=[])
    with pytest.raises(IndexError, match="ScriptedLLM exhausted"):
        await llm.complete(model=None, messages=[])


@pytest.mark.asyncio
async def test_scripted_llm_repeat_last_keeps_returning_final_response() -> None:
    r1 = make_llm_response("first")
    r2 = make_llm_response("last")
    llm = ScriptedLLM([r1, r2], repeat_last=True)
    out1 = await llm.complete(model=None, messages=[])
    out2 = await llm.complete(model=None, messages=[])
    out3 = await llm.complete(model=None, messages=[])
    out4 = await llm.complete(model=None, messages=[])
    assert out1.content == "first"
    assert out2.content == "last"
    assert out3.content == "last"
    assert out4.content == "last"


def test_scripted_llm_init_rejects_empty_responses() -> None:
    with pytest.raises(ValueError, match="at least one response"):
        ScriptedLLM([])


def test_scripted_llm_satisfies_illmclient_protocol() -> None:
    llm = ScriptedLLM([make_llm_response()])
    assert isinstance(llm, ILLMClient)
```

- [ ] **Step 2.4: Run the test — verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/testing/test_scripted_llm.py -v 2>&1 | tail -15
```

Expected: ImportError on `ai_core.testing.ScriptedLLM`.

- [ ] **Step 2.5: Implement `src/ai_core/testing/llm.py`**

```python
"""ScriptedLLM + make_llm_response builder for agent tests.

These helpers consolidate the per-test ad-hoc LLM fakes (FakeLLM,
_StubLLM, _ScriptedLLM, etc.) into a single canonical API. The
ScriptedLLM matches the full ``ILLMClient.complete`` signature so it
satisfies the abstract base class under mypy strict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMUsage


_DEFAULT_PROMPT_TOKENS = 10
_DEFAULT_COMPLETION_TOKENS = 20
_DEFAULT_MODEL = "test-model"


def make_llm_response(
    text: str = "",
    *,
    tool_calls: Sequence[Mapping[str, Any]] = (),
    finish_reason: str = "stop",
    prompt_tokens: int = _DEFAULT_PROMPT_TOKENS,
    completion_tokens: int = _DEFAULT_COMPLETION_TOKENS,
    model: str = _DEFAULT_MODEL,
) -> LLMResponse:
    """Build an :class:`LLMResponse` with sensible defaults.

    Convenience for tests that don't care about token accounting or
    model identity, only the response shape.

    Args:
        text: The string the assistant returned. Default empty.
        tool_calls: Optional sequence of OpenAI-style tool-call dicts.
        finish_reason: One of ``"stop"``, ``"length"``, ``"tool_calls"``,
            etc. Default ``"stop"``.
        prompt_tokens / completion_tokens: Usage counters; ``total_tokens``
            is computed as the sum.
        model: Model identifier echoed back on the response. Default ``"test-model"``.
    """
    return LLMResponse(
        model=model,
        content=text,
        tool_calls=list(tool_calls),
        usage=LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        raw={},
        finish_reason=finish_reason,
    )


class ScriptedLLM(ILLMClient):
    """Returns pre-constructed responses in sequence on each ``complete()`` call.

    Args:
        responses: Ordered sequence of ``LLMResponse`` to return.
        repeat_last: If ``True``, after exhausting ``responses``, return the
            last entry forever. If ``False`` (default), raise ``IndexError``
            on exhaustion so tests fail loudly when they need more responses
            than scripted.

    Raises:
        ValueError: At construction if ``responses`` is empty.
        IndexError: At call time if exhausted and ``repeat_last`` is False.
    """

    def __init__(
        self,
        responses: Sequence[LLMResponse],
        *,
        repeat_last: bool = False,
    ) -> None:
        if not responses:
            raise ValueError("ScriptedLLM requires at least one response")
        self._responses: tuple[LLMResponse, ...] = tuple(responses)
        self._repeat_last: bool = repeat_last
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({
            "model": model,
            "messages": [dict(m) for m in messages],
            "tools": [dict(t) for t in tools] if tools is not None else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "extra": dict(extra) if extra is not None else None,
        })
        idx = len(self.calls) - 1
        if idx < len(self._responses):
            return self._responses[idx]
        if self._repeat_last:
            return self._responses[-1]
        raise IndexError(
            f"ScriptedLLM exhausted: {len(self._responses)} responses "
            f"scripted but call #{len(self.calls)} requested. "
            f"Set repeat_last=True to keep returning the final response."
        )


__all__ = ["ScriptedLLM", "make_llm_response"]
```

- [ ] **Step 2.6: Extend `src/ai_core/testing/__init__.py`**

Update the `__init__.py` to re-export the new names:

```python
"""Public testing surface for SDK consumers.
... (existing docstring)
"""

from __future__ import annotations

from ai_core.testing.fakes import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)
from ai_core.testing.llm import ScriptedLLM, make_llm_response

__all__ = [
    "FakeAuditSink",
    "FakeBudgetService",
    "FakeObservabilityProvider",
    "FakePolicyEvaluator",
    "FakeSecretManager",
    "ScriptedLLM",
    "make_llm_response",
]
```

- [ ] **Step 2.7: Run the new tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/testing/test_make_llm_response.py tests/unit/testing/test_scripted_llm.py -v 2>&1 | tail -15
```

Expected: 9 passed (3 builder + 6 ScriptedLLM tests).

### 2b — Replace ad-hoc LLM fakes in `tests/`

- [ ] **Step 2.8: Replace `_StubLLM` in `tests/unit/app/test_runtime.py`**

Read the file at the `_StubLLM` definition (~line 44):

```bash
sed -n '40,80p' /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/app/test_runtime.py
```

Find the `class _StubLLM(ILLMClient):` block (returns "ok"; ~30 LOC). Replace the class definition AND all usage sites:

```python
# Before
class _StubLLM(ILLMClient):
    async def complete(self, *, model: str | None, messages: ..., ...) -> LLMResponse:
        return LLMResponse(model="stub", content="ok", ...)


# Usage:
def _override_module(...):
    @provider
    def provide_llm(...) -> ILLMClient:
        return _StubLLM()

# After
from ai_core.testing import ScriptedLLM, make_llm_response


# Usage:
def _override_module(...):
    @provider
    def provide_llm(...) -> ILLMClient:
        return ScriptedLLM([make_llm_response("ok")], repeat_last=True)
```

`repeat_last=True` matches `_StubLLM`'s "always return ok" behaviour.

Add the `from ai_core.testing import ...` line at the top of the file. Drop the `from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMUsage` import lines if they're now unused (the file may still use `ILLMClient` as a type annotation — keep it if so).

- [ ] **Step 2.9: Run the affected tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py -v 2>&1 | tail -15
```

Expected: same number of tests passing as before the migration.

- [ ] **Step 2.10: Replace `FakeLLM` in `tests/unit/agents/test_memory.py`**

Read the file's `FakeLLM`, `_SlowFakeLLM`, and inline `_RaisingLLM`:

```bash
sed -n '40,80p' /Users/admin-h26/EAAP/ai-core-sdk/tests/unit/agents/test_memory.py
```

`FakeLLM(summary="...")` returns a single `LLMResponse(content=summary, ...)`. Migration: replace each `FakeLLM(summary="X")` with `ScriptedLLM([make_llm_response("X")], repeat_last=True)`.

`_SlowFakeLLM` adds an `asyncio.sleep(...)`. The new `ScriptedLLM` doesn't have a delay hook; if the slow behaviour is essential to the test, keep `_SlowFakeLLM` as a local class wrapping `ScriptedLLM` (or a lambda). Read the test to see what `_SlowFakeLLM` is asserting; if it's a timeout test, the `asyncio.sleep` is essential — keep the class but make it inherit from `ILLMClient` only (not `ScriptedLLM`).

`_RaisingLLM` (inline class in test_recursion.py or test_memory.py) raises a configurable exception. Same treatment — keep it as a local helper if its only consumer is one test. Note: this fake is part of the test file's local scope, NOT shared across tests, so it's fine to leave.

The migration goal is "consolidate the COMMON pattern (canned response sequence) into ScriptedLLM" — purpose-built local fakes (`_SlowFakeLLM`, `_RaisingLLM`) can stay in their test files.

After migration, the `class FakeLLM(ILLMClient):` definition is DELETED; `_SlowFakeLLM` and `_RaisingLLM` STAY.

Replace each `FakeLLM(summary=X)` call with the new form, and add `from ai_core.testing import ScriptedLLM, make_llm_response` at the top of the file.

- [ ] **Step 2.11: Run the affected tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/agents/test_memory.py -v 2>&1 | tail -15
```

Expected: same number of tests passing.

- [ ] **Step 2.12: Replace `ScriptedLLM` in `tests/component/test_agent_run.py`**

Read the file's `ScriptedLLM` class (~line 49):

```bash
sed -n '45,80p' /Users/admin-h26/EAAP/ai-core-sdk/tests/component/test_agent_run.py
```

This local `ScriptedLLM` takes `responses: Sequence[str]` (string-only). The new public `ScriptedLLM` takes `Sequence[LLMResponse]`. Migration:

```python
# Before
class ScriptedLLM(ILLMClient):
    def __init__(self, responses: Sequence[str]) -> None: ...

# In tests:
llm = ScriptedLLM(["first", "second"])

# After
from ai_core.testing import ScriptedLLM, make_llm_response

llm = ScriptedLLM([make_llm_response("first"), make_llm_response("second")])
```

DELETE the local `ScriptedLLM` class definition. Update each usage site to pass `[make_llm_response(s) for s in responses]` or directly inline as shown.

- [ ] **Step 2.13: Run the affected tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/component/test_agent_run.py -v 2>&1 | tail -15
```

Expected: same number of tests passing.

- [ ] **Step 2.14: Replace `_ScriptedLLM` in `tests/component/test_agent_tool_loop.py`**

Read:

```bash
sed -n '40,80p' /Users/admin-h26/EAAP/ai-core-sdk/tests/component/test_agent_tool_loop.py
```

This `_ScriptedLLM` takes a richer sequence — likely each entry is a `(content, tool_calls)` tuple or pre-built `LLMResponse`. Migration:

- If entries are already `LLMResponse` objects, just rename the local class' usage to import the public one.
- If entries are tuples or strings, wrap in `make_llm_response(content, tool_calls=...)` and pass the resulting list to the public `ScriptedLLM`.

DELETE the local `_ScriptedLLM` class. Add `from ai_core.testing import ScriptedLLM, make_llm_response`.

- [ ] **Step 2.15: Run the affected tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/component/test_agent_tool_loop.py -v 2>&1 | tail -15
```

Expected: same number of tests passing.

### 2c — Lint, type-check, full suite, commit

- [ ] **Step 2.16: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/testing/__init__.py \
    src/ai_core/testing/llm.py \
    tests/unit/testing/test_scripted_llm.py \
    tests/unit/testing/test_make_llm_response.py \
    tests/unit/agents/test_memory.py \
    tests/unit/app/test_runtime.py \
    tests/component/test_agent_run.py \
    tests/component/test_agent_tool_loop.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/testing/__init__.py \
    src/ai_core/testing/llm.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on the new src files clean; project total ≤ 21.

- [ ] **Step 2.17: Full unit + component + contract suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -5
```

Expected: ≥491 passing (482 baseline + 9 new tests = 491).

- [ ] **Step 2.18: Commit Task 2**

```bash
git add src/ai_core/testing/__init__.py \
        src/ai_core/testing/llm.py \
        tests/unit/testing/test_scripted_llm.py \
        tests/unit/testing/test_make_llm_response.py \
        tests/unit/agents/test_memory.py \
        tests/unit/app/test_runtime.py \
        tests/component/test_agent_run.py \
        tests/component/test_agent_tool_loop.py
git commit -m "feat(testing): ScriptedLLM + make_llm_response — consolidate ad-hoc LLM fakes"
```

---

## Task 3 — pytest plugin + recipe doc

**Files:**
- Create: `src/ai_core/testing/pytest_plugin.py`
- Create: `tests/unit/testing/test_pytest_plugin.py`
- Create: `docs/testing.md`

### 3a — Implement the plugin

- [ ] **Step 3.1: Write failing tests for the plugin**

Create `tests/unit/testing/test_pytest_plugin.py`:

```python
"""Tests for ai_core.testing.pytest_plugin via pytester."""
from __future__ import annotations

import pytest

# Activate the plugin under test for these specific tests.
pytestmark = pytest.mark.unit


def test_plugin_provides_fake_audit_sink_fixture(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(
        'pytest_plugins = ["ai_core.testing.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        from ai_core.testing import FakeAuditSink

        def test_uses_fake_audit_sink(fake_audit_sink):
            assert isinstance(fake_audit_sink, FakeAuditSink)
            assert fake_audit_sink.records == []
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_plugin_provides_fake_observability_fixture(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(
        'pytest_plugins = ["ai_core.testing.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        from ai_core.testing import FakeObservabilityProvider

        def test_uses_fake_observability(fake_observability):
            assert isinstance(fake_observability, FakeObservabilityProvider)
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_plugin_provides_scripted_llm_factory_fixture(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(
        'pytest_plugins = ["ai_core.testing.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        import pytest

        from ai_core.testing import ScriptedLLM, make_llm_response


        @pytest.mark.asyncio
        async def test_uses_factory(scripted_llm_factory):
            llm = scripted_llm_factory([make_llm_response("hello")])
            assert isinstance(llm, ScriptedLLM)
            out = await llm.complete(model=None, messages=[])
            assert out.content == "hello"
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_plugin_factory_fixtures_accept_kwargs(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(
        'pytest_plugins = ["ai_core.testing.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        from ai_core.testing import FakePolicyEvaluator, FakeSecretManager


        def test_factories(
            fake_policy_evaluator_factory, fake_secret_manager_factory
        ):
            policy = fake_policy_evaluator_factory(default_allow=False)
            assert isinstance(policy, FakePolicyEvaluator)

            secrets = fake_secret_manager_factory({"k": "v"})
            assert isinstance(secrets, FakeSecretManager)
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
```

Note: `pytester` is a built-in pytest fixture; no extra plugin needed. The `pytester.makeconftest(...)` and `pytester.makepyfile(...)` write a temporary conftest + test file, then `pytester.runpytest()` spawns a child pytest session with isolated cwd.

If `pytester` requires `pytester` plugin to be activated, add `pytest_plugins = ["pytester"]` at module top. (Modern pytest auto-loads it; if your version doesn't, the explicit activation is required.)

- [ ] **Step 3.2: Run the test — verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/testing/test_pytest_plugin.py -v 2>&1 | tail -10
```

Expected: ImportError on `ai_core.testing.pytest_plugin` from inside the spawned child session, OR failures because the fixtures don't exist.

- [ ] **Step 3.3: Implement `src/ai_core/testing/pytest_plugin.py`**

```python
"""Pytest plugin exposing ai_core.testing fakes as fixtures.

Activate via your conftest.py::

    pytest_plugins = ["ai_core.testing.pytest_plugin"]

Then write tests that consume the fixtures::

    async def test_my_agent(scripted_llm_factory, fake_audit_sink):
        from ai_core.testing import make_llm_response
        llm = scripted_llm_factory([make_llm_response("ok")])
        # ...
        assert len(fake_audit_sink.records) == 1
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from ai_core.di.interfaces import LLMResponse
from ai_core.testing.fakes import (
    FakeAuditSink,
    FakeBudgetService,
    FakeObservabilityProvider,
    FakePolicyEvaluator,
    FakeSecretManager,
)
from ai_core.testing.llm import ScriptedLLM


@pytest.fixture
def fake_audit_sink() -> FakeAuditSink:
    """Fresh per-test :class:`FakeAuditSink` instance."""
    return FakeAuditSink()


@pytest.fixture
def fake_observability() -> FakeObservabilityProvider:
    """Fresh per-test :class:`FakeObservabilityProvider` instance."""
    return FakeObservabilityProvider()


@pytest.fixture
def fake_budget() -> FakeBudgetService:
    """Fresh per-test :class:`FakeBudgetService` instance."""
    return FakeBudgetService()


@pytest.fixture
def fake_policy_evaluator_factory() -> Callable[..., FakePolicyEvaluator]:
    """Factory: ``factory(default_allow=True)`` returns a configured fake."""

    def _factory(
        *, default_allow: bool = True, **kwargs: Any
    ) -> FakePolicyEvaluator:
        return FakePolicyEvaluator(default_allow=default_allow, **kwargs)

    return _factory


@pytest.fixture
def fake_secret_manager_factory() -> Callable[..., FakeSecretManager]:
    """Factory: ``factory({"key": "value"})`` returns a configured fake."""

    def _factory(secrets: Mapping[str, str] | None = None) -> FakeSecretManager:
        return FakeSecretManager(secrets or {})

    return _factory


@pytest.fixture
def scripted_llm_factory() -> Callable[..., ScriptedLLM]:
    """Factory: ``factory([resp1, resp2], repeat_last=False)`` returns a ScriptedLLM."""

    def _factory(
        responses: Sequence[LLMResponse],
        *,
        repeat_last: bool = False,
    ) -> ScriptedLLM:
        return ScriptedLLM(list(responses), repeat_last=repeat_last)

    return _factory
```

The plugin imports `pytest` at module top. Consumers without pytest installed who try to import `ai_core.testing.pytest_plugin` get a clear `ImportError`.

The 6 fixtures match the existing internal `tests/conftest.py` shapes (`fake_audit_sink` / `fake_observability` / `fake_budget` are singletons; the three factories take config kwargs and return constructed instances).

Verify the `FakePolicyEvaluator` + `FakeSecretManager` class signatures by checking the fakes module:

```bash
grep -A 5 "class FakePolicyEvaluator\|class FakeSecretManager" /Users/admin-h26/EAAP/ai-core-sdk/src/ai_core/testing/fakes.py | head -20
```

Adjust the factory `**kwargs` forwarding to match the actual constructor signatures. If `FakePolicyEvaluator` doesn't accept `default_allow=...` directly, the existing internal `tests/conftest.py:fake_policy_evaluator_factory` shows the right invocation — copy it.

- [ ] **Step 3.4: Run the plugin tests — verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/testing/test_pytest_plugin.py -v 2>&1 | tail -15
```

Expected: 4 passed.

If any test fails because `pytester` isn't auto-activated, add `pytest_plugins = ["pytester"]` at the top of the test file (immediately after imports, before `pytestmark`).

### 3b — Recipe doc

- [ ] **Step 3.5: Write `docs/testing.md`**

Create `docs/testing.md`:

```markdown
# Testing your agent with `ai_core.testing`

The `ai_core.testing` subpackage ships test helpers — fakes for the SDK's
core protocols, a `ScriptedLLM` for canned LLM responses, and a pytest
plugin that makes them all available as fixtures.

## Install

```bash
pip install ai-core-sdk[testing]
```

This pulls in `pytest>=7.0` alongside the SDK. The fakes themselves work
without pytest — only the plugin module imports it.

## Activate the pytest plugin

Add this to your project's top-level `conftest.py`:

```python
pytest_plugins = ["ai_core.testing.pytest_plugin"]
```

That's it. Six fixtures are now available everywhere in your test suite:

- `fake_audit_sink` — fresh `FakeAuditSink` per test
- `fake_observability` — fresh `FakeObservabilityProvider` per test
- `fake_budget` — fresh `FakeBudgetService` per test
- `fake_policy_evaluator_factory` — call with `(default_allow=True/False)`
- `fake_secret_manager_factory` — call with a `{key: value}` dict
- `scripted_llm_factory` — call with a list of `LLMResponse` objects

## Recipe — testing an agent that calls one tool

```python
import pytest

from ai_core.testing import make_llm_response


@pytest.mark.asyncio
async def test_agent_calls_search_tool(
    scripted_llm_factory,
    fake_audit_sink,
    fake_policy_evaluator_factory,
):
    # Arrange: wire a fake LLM that asks for a tool call, then summarises.
    llm = scripted_llm_factory(
        [
            make_llm_response(
                "",
                tool_calls=[{"id": "c1", "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}}],
                finish_reason="tool_calls",
            ),
            make_llm_response("Found x"),
        ]
    )
    policy = fake_policy_evaluator_factory(default_allow=True)

    # Act: run your agent / tool invoker with these fakes wired in.
    # ... (your code here)

    # Assert: the audit sink recorded the policy decision + tool invocation.
    assert len(fake_audit_sink.records) == 2
    assert fake_audit_sink.records[0].event.value == "policy.decision"
```

## Recipe — testing a host service wired via DI

```python
from injector import Module, provider, singleton

from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import IPolicyEvaluator
from ai_core.testing import FakePolicyEvaluator


class _DenyAllOverride(Module):
    @singleton
    @provider
    def provide_policy(self) -> IPolicyEvaluator:
        return FakePolicyEvaluator(default_allow=False)


def test_my_service_denies_when_policy_denies():
    container = Container.build([AgentModule(), _DenyAllOverride()])
    # ... use container.get(...) to resolve your service ...
```

## What's NOT in `ai_core.testing`

- `Container.test_mode()` — explicit per-fixture wiring is the v1 pattern;
  drop in a Phase 10 follow-up if real consumer projects ask for the
  builder.
- `RaisingLLM` / `SlowLLM` — handle these in your test files via inline
  classes if needed; the canned-response common case is `ScriptedLLM`.
- Snapshot / replay — agent run recording is not yet shipped.

## More patterns

The SDK's own `tests/contract/` directory shows how the public surface
itself is pinned via parametrized tests over `__subclasses__()`. The
`tests/integration/` directory shows how to use Testcontainers with the
real Postgres + OPA backends. Both are worth reading if you need
patterns beyond the basic fixtures.
```

(Adjust the example code to match real names: verify `FakePolicyEvaluator(default_allow=...)` signature exists in your migrated `fakes.py`. If the constructor differs, update the recipe.)

### 3c — Lint, type-check, full suite, commit

- [ ] **Step 3.6: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/testing/pytest_plugin.py \
    tests/unit/testing/test_pytest_plugin.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/testing/pytest_plugin.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no NEW ruff violations; mypy on the plugin clean; project total ≤ 21.

- [ ] **Step 3.7: Full unit + component + contract suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -5
```

Expected: ≥495 passing (491 after Task 2 + 4 plugin tests).

- [ ] **Step 3.8: Commit Task 3**

```bash
git add src/ai_core/testing/pytest_plugin.py \
        tests/unit/testing/test_pytest_plugin.py \
        docs/testing.md
git commit -m "feat(testing): pytest plugin (6 fixtures) + docs/testing.md recipe"
```

---

## Task 4 — End-of-phase smoke gate

Verification only. No code changes.

- [ ] **Step 4.1: Full test suite (always-run)**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component tests/contract -q 2>&1 | tail -5
```

Expected: ≥495 passing; 0 errors.

- [ ] **Step 4.2: Integration suite (Docker-conditional)**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/integration -q 2>&1 | tail -5
```

Expected: 7 passed (Docker up) OR 1 passed + 6 skipped (Docker down — the bad-DSN test is Docker-independent).

- [ ] **Step 4.3: Lint the entire tree**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src tests 2>&1 | grep "Found"
```

Expected: 211 errors total (= post-Phase-7-rego-fix baseline).

- [ ] **Step 4.4: Mypy strict**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `Found 21 errors in 8 files (checked 72 source files)` — note the source-file count went up by 4 (new testing module).

- [ ] **Step 4.5: Public surface unchanged at 30 names**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
import ai_core
assert len(ai_core.__all__) == 30, f'Expected 30, got {len(ai_core.__all__)}'
assert 'ErrorCode' in ai_core.__all__
# Crucially: ai_core.testing.* should NOT be in top-level __all__.
assert 'ScriptedLLM' not in ai_core.__all__
assert 'FakeAuditSink' not in ai_core.__all__
print('Top-level surface OK: 30 names, no testing-module leakage')
"
```

Expected: `Top-level surface OK: 30 names, no testing-module leakage`.

- [ ] **Step 4.6: `ai_core.testing` surface check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
import ai_core.testing as t
expected = {
    'FakeAuditSink', 'FakeBudgetService', 'FakeObservabilityProvider',
    'FakePolicyEvaluator', 'FakeSecretManager',
    'ScriptedLLM', 'make_llm_response',
}
assert set(t.__all__) == expected, f'Mismatch: {set(t.__all__) ^ expected}'
print('ai_core.testing surface OK: 7 names')
"
```

Expected: `ai_core.testing surface OK: 7 names`.

- [ ] **Step 4.7: Clean-venv install of `[testing]` extra**

```bash
TMP=$(mktemp -d)
/Users/admin-h26/EAAP/.venv/bin/python -m venv "$TMP/v9"
"$TMP/v9/bin/pip" install --quiet -e "/Users/admin-h26/EAAP/ai-core-sdk[testing]" 2>&1 | tail -2
"$TMP/v9/bin/python" -c "
from ai_core.testing import FakeAuditSink, ScriptedLLM, make_llm_response
from ai_core.testing.pytest_plugin import fake_audit_sink, scripted_llm_factory
print('Clean-venv [testing] install OK')
"
rm -rf "$TMP"
```

Expected: `Clean-venv [testing] install OK`. Confirms pytest is pulled in transitively via the `[testing]` extra.

- [ ] **Step 4.8: `eaap init` regression smoke (Phase 5 invariant)**

```bash
SMOKE=$(mktemp -d)
/Users/admin-h26/EAAP/.venv/bin/python -m ai_core.cli.main init smoke-app --path "$SMOKE" 2>&1 | tail -2
test -f "$SMOKE/smoke-app/eaap.yaml" && echo "eaap.yaml present"
test -f "$SMOKE/smoke-app/policies/agent.rego" && echo "agent.rego present"
rm -rf "$SMOKE"
```

Expected: 2 "present" lines.

- [ ] **Step 4.9: Capture phase summary**

```bash
git log --oneline f31168b..HEAD
```

Expected: 3 conventional-commit subjects (one per implementation Task).

- [ ] **Step 4.10: Do NOT push automatically**

```bash
git status
echo ""
echo "Suggested next step:"
echo "git push origin feat/phase-9-public-testing-surface"
echo "gh pr create --title 'feat: Phase 9 — public testing surface (ai_core.testing + pytest plugin)'"
```

---

## Out-of-scope reminders

For traceability, deferred to Phase 10+:

- `Container.test_mode()` factory or `TestModule` DI builder
- `RaisingLLM` / `SlowLLM` purpose-built fakes (handled via inline lambdas)
- Snapshot / replay testing helpers
- Standalone `pytest-ai-core` distribution package
- Adding `ai_core.testing.*` symbols to top-level `ai_core.__all__`
- Top-level `README.md` / API reference site / Quickstart
- Robustness primitives (LLM retry, audit sink buffering, agent degraded-mode)
- Phase 4 cost/latency closure (Vertex AI Anthropic prefix, multi-conn MCP pool, etc.)

If a step starts pulling work from this list, stop and confirm scope with the user.
