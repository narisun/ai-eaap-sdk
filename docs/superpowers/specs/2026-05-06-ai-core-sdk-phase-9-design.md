# ai-core-sdk Phase 9 — Design

**Date:** 2026-05-06
**Branch:** `feat/phase-9-public-testing-surface`
**Status:** Awaiting user review

## Goal

Public testing surface: ship `ai_core.testing` so consumers can write tests against the SDK's protocols without forking our internal `tests/conftest.py` fakes. After Phase 9, an SDK consumer can `pip install ai-core-sdk[testing]`, add `pytest_plugins = ["ai_core.testing.pytest_plugin"]` to their `conftest.py`, and write `async def test_my_agent(scripted_llm_factory, fake_audit_sink): ...` without copying any code from our test suite.

## Scope (3 items)

### Public testing surface (3)

1. **Migrate 5 Fake classes to `ai_core.testing`.** Move `FakePolicyEvaluator`, `FakeObservabilityProvider`, `FakeSecretManager`, `FakeBudgetService`, `FakeAuditSink` from `tests/conftest.py:53-282` (where they're internal-only) to a new `src/ai_core/testing/fakes.py`. Class bodies are preserved verbatim — these are well-tested implementations already used by 9+ test files. The new `src/ai_core/testing/__init__.py` re-exports them. `tests/conftest.py` updates to re-import from the new public location; existing fixtures (`fake_audit_sink`, `fake_observability`, `fake_policy_evaluator_factory`, `fake_secret_manager_factory`, `fake_budget`) keep their function bodies and signatures so the existing 482-test surface stays green.

2. **Add `ScriptedLLM` + `make_llm_response` to `src/ai_core/testing/llm.py`.** Consolidates the 5+ ad-hoc LLM fakes scattered across `tests/unit/agents/test_memory.py`, `tests/unit/app/test_runtime.py`, `tests/component/test_agent_run.py`, `tests/component/test_agent_tool_loop.py` (`FakeLLM`, `_StubLLM`, `_SlowFakeLLM`, `_RaisingLLM`, `ScriptedLLM`, `_ScriptedLLM`). The canonical `ScriptedLLM(responses, *, repeat_last=False)` returns scripted responses in order; raises `IndexError` on exhaustion (or repeats final entry if `repeat_last=True`). Records `self.calls` for assertions. Companion `make_llm_response(text="", *, tool_calls=(), finish_reason="stop", prompt_tokens=10, completion_tokens=20, model="test-model") -> LLMResponse` builds a valid `LLMResponse` with sensible defaults — eliminates the boilerplate `LLMResponse(content=..., tool_calls=[], finish_reason=..., usage=LLMUsage(...), model=...)` that bloats every test. Internal migration replaces the ad-hoc fakes with `ScriptedLLM` (~6-10 file touches under `tests/`).

3. **Add `src/ai_core/testing/pytest_plugin.py` + `docs/testing.md` recipe.** Plugin module imports `pytest` at module top and exports 6 fixtures: `fake_audit_sink`, `fake_observability`, `fake_budget` (singletons), `fake_policy_evaluator_factory`, `fake_secret_manager_factory`, `scripted_llm_factory` (factories taking config kwargs). Consumer activation via `pytest_plugins = ["ai_core.testing.pytest_plugin"]` in their `conftest.py` (manual opt-in pattern; matches `httpx[testing]` / `fastapi.testing` precedent). New `[testing]` optional dep extra in `pyproject.toml` declares `pytest>=7.0`. Recipe document `docs/testing.md` walks through "testing an agent that calls a tool" and "testing a host service that wires the SDK's DI" using the new fixtures.

## Non-goals (deferred to Phase 10+)

- `Container.test_mode()` factory or `TestModule` DI builder (option C from scoping question 1; consumers get the explicit per-fixture pattern instead — adopt builder if real consumer feedback asks for it)
- `RaisingLLM` / `SlowLLM` purpose-built fakes (handled via inline `lambda` hooks if a consumer needs them)
- Snapshot / replay testing helpers (record an agent run, replay it as a regression test)
- Standalone `pytest-ai-core` distribution package
- Adding `ai_core.testing.*` symbols to top-level `ai_core.__all__` — testing helpers stay namespaced (matches FastAPI/Starlette convention)
- Top-level `README.md` / API reference site / Quickstart — Phase 10+ documentation theme
- Robustness primitives (LLM retry, audit sink buffering, agent degraded-mode) — Phase 10+

## Constraints

- Pre-1.0; **no backwards-compatibility requirement.** `ai_core.testing` is purely additive.
- Per-step gate: `ruff check` no new violations + `mypy <touched files>` strict + `pytest tests/unit tests/component tests/contract` + `pytest tests/integration` (Docker-conditional, auto-skip otherwise).
- Project mypy `src` total stays ≤ 21 (post-Phase-3 baseline). Phase 9 adds 4 new src files (`testing/__init__.py`, `testing/fakes.py`, `testing/llm.py`, `testing/pytest_plugin.py`) — they must type-check clean against strict mypy.
- Project ruff total stays ≤ 211 (post-Phase-7 baseline at `d19084f`, the rego fix merge).
- Public surface stays at exactly 30 names (`ai_core.__all__` unchanged).
- `ai_core.testing.__all__` has exactly 7 names: 5 Fake classes + `ScriptedLLM` + `make_llm_response`.
- New optional dep `pytest>=7.0` lives ONLY under `[testing]` extra. Default `pip install ai-core-sdk` footprint unchanged.
- Internal `tests/conftest.py` re-imports preserve the 482-test surface; no test file outside the migrated paths needs changes.

## Module layout

```
src/ai_core/testing/
├── __init__.py                 # NEW — re-exports the 7 public names
├── fakes.py                    # NEW — 5 Fake* classes (moved from tests/conftest.py)
├── llm.py                      # NEW — ScriptedLLM + make_llm_response()
└── pytest_plugin.py            # NEW — 6 fixtures (imports pytest at module top)

tests/conftest.py               # MODIFIED — re-import Fake* from ai_core.testing
tests/unit/agents/test_memory.py        # MODIFIED — replace ad-hoc FakeLLM with ScriptedLLM
tests/unit/app/test_runtime.py          # MODIFIED — replace _StubLLM with ScriptedLLM
tests/component/test_agent_run.py       # MODIFIED — replace ScriptedLLM (local) with ai_core.testing.ScriptedLLM
tests/component/test_agent_tool_loop.py # MODIFIED — replace _ScriptedLLM with ai_core.testing.ScriptedLLM

tests/unit/testing/                     # NEW — testing-module unit tests
├── __init__.py                         # NEW (empty)
├── test_fakes.py                       # NEW — moved from existing tests/unit/test_conftest_fakes.py (~15 tests)
├── test_scripted_llm.py                # NEW (~6 tests)
├── test_make_llm_response.py           # NEW (~3 tests)
└── test_pytest_plugin.py               # NEW (~4 tests using `pytester`)

docs/testing.md                         # NEW — recipe doc (~150 LOC markdown, 2-3 worked examples)

pyproject.toml                          # MODIFIED — [testing] extra group
```

### Files NOT touched

- `README.md`
- `src/ai_core/cli/main.py`, `cli/scaffold.py`, `cli/templates/init/**`
- `tests/unit/cli/test_main.py`
- `tests/integration/**`
- `tests/contract/**` (Phase 7 territory; surface contract test already at 30 names — Phase 9 doesn't add to top-level `__all__`)
- All `src/ai_core/` files outside the new `testing/` subpackage

## Component 1 — `ai_core.testing.fakes`

```python
# src/ai_core/testing/fakes.py
"""Fake implementations of the SDK's public protocols, for use in tests.

These fakes are deliberately simple, in-memory, and synchronous-where-possible
so consumer tests can assert on observable state without setting up real
backends. They are NOT production-quality.

Imported by ``ai_core.testing`` (no pytest dependency) and re-exported from
``ai_core.testing.pytest_plugin`` for fixture-based access.
"""

# (5 class bodies moved verbatim from tests/conftest.py:53-282)

class FakePolicyEvaluator(IPolicyEvaluator): ...
class FakeObservabilityProvider(IObservabilityProvider): ...
class FakeSecretManager(ISecretManager): ...
class FakeBudgetService(IBudgetService): ...
class FakeAuditSink(IAuditSink): ...
```

The migration is structural — class bodies, constructor signatures, observable attributes (`records`, `spans`, `events`, etc.), and methods are byte-for-byte identical to the existing definitions.

## Component 2 — `ai_core.testing.llm`

### 2a. `make_llm_response`

```python
# src/ai_core/testing/llm.py
def make_llm_response(
    text: str = "",
    *,
    tool_calls: Sequence[Mapping[str, Any]] = (),
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    model: str = "test-model",
) -> LLMResponse:
    """Build an LLMResponse with sensible defaults.

    Convenience for tests that don't care about token accounting or
    model identity, only the response shape.
    """
    return LLMResponse(
        content=text,
        tool_calls=list(tool_calls),
        finish_reason=finish_reason,
        usage=LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model=model,
    )
```

The exact `LLMResponse` field set comes from `ai_core.di.interfaces` — implementer reads the live class to confirm.

### 2b. `ScriptedLLM`

```python
class ScriptedLLM(ILLMClient):
    """Returns pre-constructed responses in sequence on each ``complete()`` call.

    Args:
        responses: Ordered list of LLMResponse to return.
        repeat_last: If True, after exhausting `responses`, return the last
            entry forever. If False (default), raise IndexError on exhaustion
            so tests fail loudly when they need more responses than scripted.

    Raises:
        ValueError: If ``responses`` is empty at construction time.
        IndexError: If ``complete()`` is called more times than the number of
            scripted responses (and ``repeat_last`` is False).
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

    async def complete(self, **kwargs: Any) -> LLMResponse:
        # Record the call for test assertions.
        self.calls.append(dict(kwargs))
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
```

The `**kwargs: Any` complete signature matches `ILLMClient.complete` — the implementer expands it to the real signature when porting (mypy strict will require explicit kwargs to satisfy the ABC).

## Component 3 — `ai_core.testing.pytest_plugin`

```python
# src/ai_core/testing/pytest_plugin.py
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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

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
    return FakeAuditSink()


@pytest.fixture
def fake_observability() -> FakeObservabilityProvider:
    return FakeObservabilityProvider()


@pytest.fixture
def fake_budget() -> FakeBudgetService:
    return FakeBudgetService()


@pytest.fixture
def fake_policy_evaluator_factory() -> Callable[..., FakePolicyEvaluator]:
    """Factory: pass ``default_allow=True/False`` and rule overrides."""
    def _factory(
        *, default_allow: bool = True, **kwargs: Any
    ) -> FakePolicyEvaluator:
        return FakePolicyEvaluator(default_allow=default_allow, **kwargs)
    return _factory


@pytest.fixture
def fake_secret_manager_factory() -> Callable[..., FakeSecretManager]:
    def _factory(secrets: Mapping[str, str] | None = None) -> FakeSecretManager:
        return FakeSecretManager(secrets or {})
    return _factory


@pytest.fixture
def scripted_llm_factory() -> Callable[..., ScriptedLLM]:
    """Factory: pass a list of LLMResponse objects (use ``make_llm_response()``)."""
    def _factory(
        responses: Sequence[Any], *, repeat_last: bool = False
    ) -> ScriptedLLM:
        return ScriptedLLM(list(responses), repeat_last=repeat_last)
    return _factory
```

The 6 fixtures mirror the existing internal fixture names and shapes; internal `tests/conftest.py` keeps its own definitions (which now construct from the public Fake classes) — internal test files don't need to switch fixture sources.

## Component 4 — `pyproject.toml` `[testing]` extra

```toml
[project.optional-dependencies]
testing = ["pytest>=7.0"]
```

Alphabetical position respects the existing extras order (`datadog`, `dev`, `mcp`, `secrets-aws`, `sentry`, `storage-aws`, `testing`, `vector-pgvector`).

## Component 5 — `docs/testing.md` recipe

Sections:
1. **Install**: `pip install ai-core-sdk[testing]`
2. **Activate**: `pytest_plugins = ["ai_core.testing.pytest_plugin"]` in conftest.py
3. **Recipe — testing an agent that calls one tool**: walk through `scripted_llm_factory([make_llm_response("ok", tool_calls=[...])])` + `fake_audit_sink` + assertion on `fake_audit_sink.records`
4. **Recipe — testing a host service wired via DI**: `Container.build([AgentModule(), MyOverrideModule()])` where `MyOverrideModule` binds `IPolicyEvaluator → FakePolicyEvaluator(default_allow=False)`
5. **Pointer to contract tests** as examples of more advanced patterns

~150 LOC of Markdown, 2-3 worked code blocks.

## Error handling — consolidated (Phase 9 deltas)

| Path | Behaviour |
|---|---|
| Consumer imports `ai_core.testing.pytest_plugin` without pytest installed | Top-of-module `import pytest` raises `ImportError`. Consumers who installed `ai-core-sdk[testing]` won't hit this. |
| Consumer imports `ai_core.testing` (just the fakes) without pytest installed | Works fine — no pytest dependency. |
| `ScriptedLLM` exhausted with `repeat_last=False` (default) | Raises `IndexError` with a message naming the call number, total scripted, and `repeat_last=True` suggestion. Loud failure prevents masked test bugs. |
| `ScriptedLLM([])` (empty responses) | `__init__` raises `ValueError("ScriptedLLM requires at least one response")`. |
| `make_llm_response()` called with no args | Returns a valid `LLMResponse` with empty content + 10/20/30 token usage. |
| Consumer fixture name collides with a fixture in their own conftest | Pytest's "closer fixture wins" rule means consumer-side overrides work cleanly. Documented in the recipe. |
| `FakeAuditSink.flush()` called twice | Idempotent — Phase 1 `IAuditSink` contract preserved. |
| `FakePolicyEvaluator(default_allow=True).evaluate(...)` | Returns `PolicyDecision(allowed=True, obligations={}, reason=None)` — same as the existing internal fixture-factory. |

Phase 1-8 invariants preserved by virtue of being un-touched.

## Testing strategy

Per-step gate identical to Phase 8. Project mypy `src` total stays ≤ 21. Project ruff total stays ≤ 211.

### Per-step test additions

| Step | Tests |
|---|---|
| 1. Migrate fakes | move existing 15 fake tests from `tests/unit/test_conftest_fakes.py` to `tests/unit/testing/test_fakes.py` (rename, no behaviour change) |
| 2. ScriptedLLM + make_llm_response | ~6 ScriptedLLM tests + ~3 make_llm_response tests in `tests/unit/testing/` |
| 3. Pytest plugin + recipe | ~4 plugin tests using pytest's `pytester` fixture (asserts each fixture is available + yields the right type) |
| 4. Smoke gate | full pytest + ruff + mypy + clean-venv install of `[testing]` extra + integration suite |

### Test detail — `tests/unit/testing/test_scripted_llm.py`

| Name | Asserts |
|---|---|
| `test_scripted_llm_returns_responses_in_order` | `ScriptedLLM([r1, r2]).complete(...)` returns r1 then r2 |
| `test_scripted_llm_records_calls` | After 2 calls, `llm.calls` has 2 entries with the kwargs from each call |
| `test_scripted_llm_raises_on_exhaustion` | 3rd call to `ScriptedLLM([r1, r2])` raises `IndexError` with informative message |
| `test_scripted_llm_repeat_last_returns_final_forever` | `ScriptedLLM([r1, r2], repeat_last=True)` returns r2 on calls 3, 4, 5, ... |
| `test_scripted_llm_init_rejects_empty_responses` | `ScriptedLLM([])` raises `ValueError` |
| `test_scripted_llm_satisfies_illmclient_protocol` | Instantiates without TypeError (mypy strict + abstract method check) |

### Test detail — `tests/unit/testing/test_make_llm_response.py`

| Name | Asserts |
|---|---|
| `test_make_llm_response_defaults` | `make_llm_response()` returns content="", finish_reason="stop", 10/20/30 tokens, model="test-model" |
| `test_make_llm_response_with_text` | `make_llm_response("hi")` returns content="hi" with default usage |
| `test_make_llm_response_with_all_fields` | All keyword args propagate; `total_tokens = prompt_tokens + completion_tokens` |

### Test detail — `tests/unit/testing/test_pytest_plugin.py`

Uses pytest's built-in `pytester` fixture to spawn child sessions and assert plugin behaviour:

| Name | Asserts |
|---|---|
| `test_plugin_provides_fake_audit_sink_fixture` | `pytester.runpytest()` succeeds when the test file consumes `fake_audit_sink` |
| `test_plugin_provides_scripted_llm_factory_fixture` | Same for `scripted_llm_factory` (test passes a 1-element response list) |
| `test_plugin_factory_fixtures_are_callable` | `fake_policy_evaluator_factory()` and `fake_secret_manager_factory()` return correct types |
| `test_plugin_imports_fail_loudly_without_pytest_activation` | Without `pytest_plugins = [...]`, fixtures are NOT auto-injected — pytester run reports fixture-not-found |

### Internal migration tests

Existing 482 tests stay green by construction:
- `tests/conftest.py` re-imports `Fake*` from `ai_core.testing`. The 9+ test files using `fake_audit_sink`/`fake_observability`/etc. fixtures see no behavioural change.
- Ad-hoc LLM fakes in 4 test files get replaced with `ScriptedLLM([make_llm_response(...)])`. Tests assert observable behaviour, not the fake's class name, so the migration is transparent.

### Risk register

| Risk | Mitigation |
|---|---|
| Migrated fakes have subtle behavioural drift after move | Verbatim move + re-import. Existing 482 tests are the regression guard. |
| `ScriptedLLM` signature drifts from `ILLMClient.complete()` | New `test_scripted_llm_satisfies_illmclient_protocol` instantiates it; mypy strict catches signature drift at type-check time. |
| `pytest_plugin.py` imports `pytest` at top → consumers without pytest hit `ImportError` even just trying `import ai_core.testing` | The `ai_core.testing.__init__` does NOT import the plugin module. Only `pytest_plugins = [...]` activation triggers the plugin import. |
| Plugin fixture names collide with consumer fixtures | Documented in the recipe. Pytest's "closer fixture wins" rule means consumer-side overrides work cleanly. |
| `pip install ai-core-sdk[testing]` doesn't pull pytest into the consumer's venv | The `[testing]` extra explicitly declares `pytest>=7.0`. Verified by Step 4's clean-venv install smoke. |
| Domain-specific behavior of replaced ad-hoc fakes is lost | Each migration-touched test (test_memory.py, test_runtime.py, test_agent_run.py, test_agent_tool_loop.py) is read carefully; domain-specific behaviour is encoded in the `responses` argument or accepted as scope reduction (most ad-hoc fakes were trivial single-response stubs). |
| `pytester` plugin tests are flaky on macOS / different platforms | Use the `pytester` fixture's standard pattern (spawns child pytest with isolated cwd); avoid filesystem assertions; tests run in CI-equivalent isolation. |

### End-of-phase smoke gate

- `pytest tests/unit tests/component tests/contract -q` green (≥510 passing — 482 baseline + ~28 new testing-module tests).
- `pytest tests/integration -q` green (7 passed when Docker up; 6 skipped + 1 passed otherwise).
- `ruff check src tests` total ≤ 211 (no new vs `d19084f`).
- `mypy src --strict` total ≤ 21.
- All 30 canonical names import (`ai_core.testing.*` is NOT in `ai_core.__all__`).
- `ai_core.testing.__all__` has exactly 7 names.
- Clean-venv smoke: `python -m venv /tmp/v9 && /tmp/v9/bin/pip install -e ".[testing]" && /tmp/v9/bin/python -c "from ai_core.testing.pytest_plugin import fake_audit_sink"` succeeds.
- `eaap init` regression smoke (Phase 5 invariant) still produces a working scaffold.

### Coverage target

Phase 9 doesn't materially change line coverage on `src/ai_core/`. The new `testing/` subpackage adds ~250 LOC of new src code; ~28 tests target it directly with ≥85% coverage.

## Implementation order (bottom-up)

| Step | Deliverable | Tests | Dependencies |
|---|---|---|---|
| 1 | Migrate 5 Fake classes to `src/ai_core/testing/{fakes.py,__init__.py}` + `[testing]` extra + update `tests/conftest.py` re-imports + move existing fake tests to `tests/unit/testing/` | 15 moved tests | none |
| 2 | Add `ScriptedLLM` + `make_llm_response` to `src/ai_core/testing/llm.py` + replace 5+ ad-hoc LLM fakes in `tests/` | ~9 new tests + ~6-10 file touches in tests/ | step 1 (re-export from `__init__`) |
| 3 | Add `src/ai_core/testing/pytest_plugin.py` with 6 fixtures + `docs/testing.md` recipe | ~4 plugin tests using `pytester` | step 1, step 2 |
| 4 | End-of-phase smoke gate | full pytest + ruff + mypy + clean-venv extras-install verification | all |

Tasks 2 and 3 depend on Task 1's migration of `Fake*` to `ai_core.testing`. Task 3 depends on Task 2's `ScriptedLLM` for the `scripted_llm_factory` fixture.

## Constraints — recap

- 4 implementation steps; per-step gate is ruff (no new) + mypy strict (touched files) + pytest unit/component/contract + Docker-conditional pytest integration.
- Project mypy `src` total stays ≤ 21.
- Project ruff total stays ≤ 211.
- End-of-phase smoke gate must pass before merge.
- Public surface stays at 30 names — `ai_core.testing.*` is namespaced, not added to `ai_core.__all__`.
- New optional dep `pytest>=7.0` lives ONLY under `[testing]` extra. No new top-level subpackages beyond `ai_core/testing/`.
