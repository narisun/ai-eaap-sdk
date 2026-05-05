# ai-core-sdk Phase 3 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operability hardening (real health probes, structlog adoption, audit sink) + Phase 2 coherence fixes (widened `tool.invoke` span, empty-response inside `llm.complete` span, `LLMTimeoutError` in compaction skip, Literal `finish_reason`, loosened test, promoted `FakeBudget`).

**Architecture:** Bottom-up — Phase 2 coherence first (small wins, ride on existing abstractions), then structlog (foundational for audit logs), then audit sink (uses structlog + the widened span), then real health probes (user-visible capstone making `app.health` async). Per-step gate is ruff + mypy strict (touched files only) + pytest unit/component. Project mypy strict total stays at-or-below the 21-error post-Phase-2 baseline.

**Tech Stack:** Python 3.11+, Pydantic v2, `injector` for DI, LangGraph, OpenTelemetry, `structlog>=24.1.0` (NEW), `httpx` (already a transitive dep via fastmcp), `pytest` + ruff + mypy strict. Spec: `docs/superpowers/specs/2026-05-05-ai-core-sdk-phase-3-design.md`.

---

## Pre-flight context

**Branch:** `feat/phase-1-facade-tool-validation` (carries Phase 1 + Phase 2 + Phase 3 spec). Phase 3 implementation continues on this branch unless the user starts a new branch.

**Working-state hygiene** — do NOT touch:
- `README.md`
- `src/ai_core/cli/main.py`
- `src/ai_core/cli/templates/init/**`
- `tests/unit/cli/test_main.py`

**Mypy baseline:** 21 strict errors in 8 files (post-Phase-2). Total must remain ≤ 21 after every commit.

**Per-step gate (every commit):**
- `ruff check <files-touched>` — no NEW violations vs the pre-task ruff state.
- `pytest tests/unit tests/component -q` — must pass (excluding pre-existing `respx`/`aiosqlite` collection errors).
- `mypy <files-touched-by-this-task>` — no new strict errors.
- `mypy src 2>&1 | tail -1` — total ≤ 21.

**Python interpreter:** `/Users/admin-h26/EAAP/.venv/bin/python` (no local venv inside the SDK).

**Per-task commit message convention:** Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `style:`, `build:`).

**`structlog` dependency:** must be added to `pyproject.toml`'s `dependencies` list before Step 2's tests run. Pin: `structlog>=24.1.0,<26`.

---

## Task 1 — Phase 2 coherence batch (items 4-9)

This task bundles 6 small fixes that ride on Phase 2 abstractions. Done first because every later step depends on the new test fixtures and assertion shapes.

**Files:**
- Modify: `src/ai_core/tools/invoker.py` — widen `tool.invoke` span (item 4)
- Modify: `src/ai_core/llm/litellm_client.py` — move `_normalise_response` inside span (item 5)
- Modify: `src/ai_core/agents/memory.py` — also catch `LLMTimeoutError` in compaction skip (item 6)
- Modify: `src/ai_core/di/interfaces.py` — `LLMResponse.finish_reason` Literal typing (item 7)
- Modify: `tests/unit/app/test_runtime.py` — loosen `health.components` assertion (item 8)
- Modify: `tests/conftest.py` — add `FakeBudgetService` (item 9)
- Modify: `tests/unit/agents/test_memory.py`, `tests/unit/llm/test_litellm_client.py` — replace local `FakeBudget` with shared
- Modify: `tests/unit/tools/test_invoker.py` — update span-position assertions (item 4 fallout)

### 1a — Widen `tool.invoke` span (item 4)

- [ ] **Step 1.1: Read the current `ToolInvoker.invoke` shape**

```bash
sed -n '87,190p' src/ai_core/tools/invoker.py
```

Confirm the steps are organised: input validation (lines 96-109) → OPA (111-135) → span+handler (137-157) → output validation (159-183) → completion event (185-187).

- [ ] **Step 1.2: Write a test that proves all 4 tool error categories tag the `tool.invoke` span with `eaap.error.code`**

The existing `tests/unit/tools/test_invoker.py` has a `fake_observability` fixture that records spans with `error_code` attribute (Phase 2 Task 4). Append a new test:

```python
@pytest.mark.asyncio
async def test_input_validation_error_tags_tool_invoke_span(
    fake_observability, fake_policy_evaluator_factory
) -> None:
    """ToolValidationError(side='input') must propagate inside the tool.invoke span."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolValidationError):
        await inv.invoke(_search, {"q": "x", "limit": -1})  # limit<1 fails Pydantic
    spans = [s for s in fake_observability.spans if s.name == "tool.invoke"]
    assert len(spans) == 1
    assert spans[0].error_code == "tool.validation_failed"


@pytest.mark.asyncio
async def test_policy_denial_tags_tool_invoke_span(
    fake_observability, fake_policy_evaluator_factory
) -> None:
    """PolicyDenialError must propagate inside the tool.invoke span."""
    inv = _invoker(fake_observability, fake_policy_evaluator_factory,
                   allow=False, reason="denied")
    with pytest.raises(PolicyDenialError):
        await inv.invoke(_search, {"q": "x", "limit": 1})
    spans = [s for s in fake_observability.spans if s.name == "tool.invoke"]
    assert len(spans) == 1
    assert spans[0].error_code == "policy.denied"


@pytest.mark.asyncio
async def test_output_validation_error_tags_tool_invoke_span(
    fake_observability, fake_policy_evaluator_factory
) -> None:
    """ToolValidationError(side='output') must propagate inside the tool.invoke span."""

    @tool(name="lying", version=1)
    async def lying(payload: _In) -> _Out:
        return {"wrong": True}  # type: ignore[return-value]

    inv = _invoker(fake_observability, fake_policy_evaluator_factory)
    with pytest.raises(ToolValidationError):
        await inv.invoke(lying, {"q": "x", "limit": 1})
    spans = [s for s in fake_observability.spans if s.name == "tool.invoke"]
    assert len(spans) == 1
    assert spans[0].error_code == "tool.validation_failed"
```

(`_invoker`, `_search`, `_In`, `_Out` are existing fixtures defined in the same test file — reuse them.)

- [ ] **Step 1.3: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py::test_input_validation_error_tags_tool_invoke_span tests/unit/tools/test_invoker.py::test_policy_denial_tags_tool_invoke_span tests/unit/tools/test_invoker.py::test_output_validation_error_tags_tool_invoke_span -v
```

Expected: all three fail because the span doesn't currently see those exceptions (input/OPA validation runs before span open; output validation runs after span close).

- [ ] **Step 1.4: Refactor `ToolInvoker.invoke` to wrap all 6 steps in the span**

Replace the `invoke` method body with:

```python
    async def invoke(
        self,
        spec: ToolSpec,
        raw_args: Mapping[str, Any],
        *,
        principal: Mapping[str, Any] | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Mapping[str, Any]:
        attrs: dict[str, Any] = {
            "tool.name": spec.name,
            "tool.version": spec.version,
            "agent_id": agent_id or "",
            "tenant_id": tenant_id or "",
        }
        async with self._observability.start_span("tool.invoke", attributes=attrs):
            # ----- 1. Input validation ------------------------------------
            try:
                payload = spec.input_model.model_validate(dict(raw_args))
            except ValidationError as exc:
                raise ToolValidationError(
                    f"Tool '{spec.name}' v{spec.version} input failed validation.",
                    details={
                        "tool": spec.name,
                        "version": spec.version,
                        "side": "input",
                        "errors": exc.errors(),
                    },
                    cause=exc,
                ) from exc

            # ----- 2. OPA enforcement -------------------------------------
            if spec.opa_path is not None and self._policy is not None:
                decision = await self._policy.evaluate(
                    decision_path=spec.opa_path,
                    input={
                        "tool": spec.name,
                        "version": spec.version,
                        "payload": payload.model_dump(),
                        "user": dict(principal or {}),
                        "agent_id": agent_id,
                        "tenant_id": tenant_id,
                    },
                )
                if not decision.allowed:
                    raise PolicyDenialError(
                        f"Tool '{spec.name}' v{spec.version} denied by policy: "
                        f"{decision.reason or 'no reason provided'}",
                        details={
                            "tool": spec.name,
                            "version": spec.version,
                            "reason": decision.reason,
                            "agent_id": agent_id,
                            "tenant_id": tenant_id,
                        },
                    )

            # ----- 3+4. Handler call --------------------------------------
            try:
                result: Any = await spec.handler(payload)
            except Exception as exc:  # noqa: BLE001 — wrap as ToolExecutionError
                raise ToolExecutionError(
                    f"Tool '{spec.name}' v{spec.version} failed: {exc}",
                    details={
                        "tool": spec.name,
                        "version": spec.version,
                        "agent_id": agent_id,
                        "tenant_id": tenant_id,
                    },
                    cause=exc,
                ) from exc

            # ----- 5. Output validation -----------------------------------
            try:
                validated: BaseModel = spec.output_model.model_validate(result)
            except ValidationError as exc:
                _logger.warning(
                    "Tool '%s' v%s returned a non-conforming object "
                    "(agent_id=%s, tenant_id=%s); this is a handler bug.",
                    spec.name,
                    spec.version,
                    agent_id,
                    tenant_id,
                )
                raise ToolValidationError(
                    f"Tool '{spec.name}' v{spec.version} returned invalid data.",
                    details={
                        "tool": spec.name,
                        "version": spec.version,
                        "side": "output",
                        "errors": exc.errors(),
                    },
                    cause=exc,
                ) from exc

        # ----- 6. Completion event (outside span — span already closed cleanly) -----
        await self._observability.record_event("tool.completed", attributes=attrs)
        return validated.model_dump(mode="json")
```

- [ ] **Step 1.5: Run the new tests to verify they pass + existing tests still pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py -v 2>&1 | tail -30
```

Expected: all tests pass (including the 3 new ones + ~15 existing ones).

The Phase 2 test `test_opa_deny_raises_policy_denial` previously asserted `fake_observability.spans == []` (no span on OPA-deny path). After widening, the span DOES open even on OPA deny — update that assertion:

```python
# Before:
# OPA deny short-circuits before the span opens.
assert fake_observability.spans == []

# After (Phase 3 widens span to wrap all 6 steps):
# OPA deny now propagates inside the span (gets eaap.error.code tagging).
assert any(s.name == "tool.invoke" and s.error_code == "policy.denied"
           for s in fake_observability.spans)
```

If `test_input_validation_runs_before_opa` exists, it should still pass because input validation still happens before OPA — the order didn't change, just the surrounding span scope.

If a test asserts on the EXACT order or count of `record_event` calls, verify it still holds (we didn't change when the completion event fires).

- [ ] **Step 1.6: Type-check + lint**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/tools/invoker.py tests/unit/tools/test_invoker.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/tools/invoker.py tests/unit/tools/test_invoker.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no new violations; mypy ≤ 21.

### 1b — Move `_normalise_response` inside `llm.complete` span (item 5)

- [ ] **Step 1.7: Write a test that proves empty-response error tags `llm.complete` span**

Append to `tests/unit/llm/test_litellm_client.py`:

```python
@pytest.mark.asyncio
async def test_empty_response_tags_llm_complete_span(monkeypatch, fake_observability,
                                                       fake_budget) -> None:
    """An empty LLM response must propagate inside the llm.complete span so
    eaap.error.code='llm.empty_response' is auto-emitted."""
    settings = AppSettings()
    settings.llm.max_retries = 0

    async def _empty_response(**kwargs: Any) -> Any:
        # Shape that triggers _normalise_response's empty-detection branch.
        return {
            "model": "gpt-x",
            "choices": [{"message": {"content": "", "tool_calls": []},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        }

    monkeypatch.setattr("litellm.acompletion", _empty_response)

    client = LiteLLMClient(
        settings=settings,
        budget=fake_budget,
        observability=fake_observability,
    )
    with pytest.raises(LLMInvocationError) as exc:
        await client.complete(model="gpt-x",
                              messages=[{"role": "user", "content": "hi"}])
    assert exc.value.error_code == "llm.empty_response"
    spans = [s for s in fake_observability.spans if s.name == "llm.complete"]
    assert len(spans) == 1
    assert spans[0].error_code == "llm.empty_response"
```

(`fake_budget` is the new conftest fixture — see step 1.13.)

- [ ] **Step 1.8: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_litellm_client.py::test_empty_response_tags_llm_complete_span -v
```

Expected: fails because `_normalise_response` is currently called *after* the `llm.complete` span context manager closes (line 184 in the current file).

- [ ] **Step 1.9: Move `_normalise_response` inside the span**

In `src/ai_core/llm/litellm_client.py`, find the block around lines 157-184. Currently:

```python
        async with self._observability.start_span("llm.complete", attributes=attributes):
            started = time.monotonic()
            try:
                raw = await self._call_with_retry(request_kwargs)
            except RetryError as exc:
                # ... existing handling
            latency_ms = (time.monotonic() - started) * 1000.0

        # --- 3. Normalise response + record usage -------------------------------
        response = _normalise_response(resolved_model, raw)
        cost_usd = _extract_cost(raw)
```

Change to keep `_normalise_response` inside the span:

```python
        async with self._observability.start_span("llm.complete", attributes=attributes):
            started = time.monotonic()
            try:
                raw = await self._call_with_retry(request_kwargs)
            except RetryError as exc:
                # ... existing handling
            latency_ms = (time.monotonic() - started) * 1000.0
            response = _normalise_response(resolved_model, raw)  # NEW: inside span

        # --- 3. Record usage (outside span — pure metric emit) ----------------
        cost_usd = _extract_cost(raw)
```

- [ ] **Step 1.10: Run test to verify it passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/llm/test_litellm_client.py -v 2>&1 | tail -20
```

Expected: all passing including the new test.

### 1c — `MemoryManager.compact` also skips `LLMTimeoutError` (item 6)

- [ ] **Step 1.11: Write failing test**

Append to `tests/unit/agents/test_memory.py`:

```python
@pytest.mark.asyncio
async def test_compact_skips_on_llm_timeout_error() -> None:
    """LLMTimeoutError raised by the LLM client must trigger skip-and-WARN
    (no agent crash). Distinct from asyncio.TimeoutError (which is the
    compaction-budget timeout we already handle)."""
    from ai_core.exceptions import LLMTimeoutError

    class _RaisingLLM(ILLMClient):
        async def complete(
            self, *, model, messages, tools=None, temperature=None, max_tokens=None,
            tenant_id=None, agent_id=None, extra=None,
        ) -> LLMResponse:
            raise LLMTimeoutError(
                "client-level timeout",
                details={"model": "fake", "attempts": 1},
            )

    settings = AppSettings()
    counter = FakeTokenCounter([10_000, 0, 0])
    mgr = MemoryManager(settings=settings, llm=_RaisingLLM(), token_counter=counter)

    state = new_agent_state(
        initial_messages=[{"role": "user", "content": "hi"}],
        essential={"tenant_id": "t1"},
    )

    result = await mgr.compact(state, agent_id="a1", tenant_id="t1")
    # Skip-and-WARN: state returned unchanged, no exception.
    assert result is state
```

- [ ] **Step 1.12: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/agents/test_memory.py::test_compact_skips_on_llm_timeout_error -v
```

Expected: fails — `LLMTimeoutError` propagates out of `compact()` because the existing `except` only catches `asyncio.TimeoutError`.

- [ ] **Step 1.13: Add the second except branch**

In `src/ai_core/agents/memory.py`, find the `compact` method's existing `except asyncio.TimeoutError:` block. Add a second branch right after:

```python
    async def compact(
        self,
        state: AgentState,
        *,
        model: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
    ) -> AgentState:
        """..."""
        timeout = self._settings.agent.compaction_timeout_seconds
        try:
            return await asyncio.wait_for(
                self._do_compact(state, model=model, tenant_id=tenant_id, agent_id=agent_id),
                timeout=timeout,
            )
        except TimeoutError:  # asyncio.TimeoutError == TimeoutError in 3.11+
            _logger.warning(
                "Compaction skipped: LLM call exceeded %.1fs timeout "
                "(agent_id=%s, tenant_id=%s)",
                timeout, agent_id, tenant_id,
            )
            return state
        except LLMTimeoutError as exc:
            # The LLM client raised its own timeout (network/upstream). Distinct from
            # the asyncio-level wait_for timeout: the budget wasn't exceeded — the
            # underlying call gave up. Skip-and-WARN with a different message so
            # dashboards can split the two cases.
            _logger.warning(
                "Compaction skipped: LLM client timeout (agent_id=%s, tenant_id=%s, "
                "error_code=%s)",
                agent_id, tenant_id, exc.error_code,
            )
            return state
```

Add the import at the top of `src/ai_core/agents/memory.py` (find the existing `from ai_core.exceptions import ...` if any, or add a fresh line):

```python
from ai_core.exceptions import LLMTimeoutError
```

- [ ] **Step 1.14: Run tests to verify the new test passes + no regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/agents/test_memory.py -v 2>&1 | tail -15
```

Expected: all 11+ tests pass.

### 1d — `LLMResponse.finish_reason` Literal typing (item 7)

- [ ] **Step 1.15: Update the dataclass**

In `src/ai_core/di/interfaces.py`, find `class LLMResponse:` and change the `finish_reason` annotation:

```python
# Before
finish_reason: str | None = None

# After
finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | str | None = None
```

Add the `Literal` import at the top of the file if not already present (it likely already is for other annotations):

```python
from typing import Any, Literal, Protocol
```

- [ ] **Step 1.16: Type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/di/interfaces.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 21.

(No new test — the runtime behavior is unchanged. The annotation is purely for IDE help.)

### 1e — Loosen `health.components` test (item 8)

- [ ] **Step 1.17: Update the existing test in `tests/unit/app/test_runtime.py`**

Find `test_health_components_populated_after_entry`. Currently asserts exact dict equality. Replace the assertion:

```python
# Before
assert snap.components == {
    "settings": "ok",
    "container": "ok",
    "tool_invoker": "unknown",
    "policy_evaluator": "unknown",
    "observability": "unknown",
}

# After (loosened — Phase 3 will populate real values, but the keys are stable)
assert set(snap.components.keys()) == {
    "settings", "container", "tool_invoker", "policy_evaluator", "observability",
}
assert snap.components["settings"] == "ok"
assert snap.components["container"] == "ok"
# Other component values may be "ok" or "unknown" depending on Phase 3 probes.
```

- [ ] **Step 1.18: Run test to verify it still passes**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py -v 2>&1 | tail -10
```

Expected: all passing.

### 1f — Promote `FakeBudgetService` to `tests/conftest.py` (item 9)

- [ ] **Step 1.19: Add `FakeBudgetService` + fixture to `tests/conftest.py`**

Append to `tests/conftest.py`:

```python
# ---------------------------------------------------------------------------
# FakeBudgetService — promoted from per-test-file definitions (Phase 3 item 9)
# ---------------------------------------------------------------------------
from ai_core.di.interfaces import BudgetCheck, IBudgetService


class FakeBudgetService(IBudgetService):
    """Always-allow IBudgetService for tests. Records call kwargs for assertion."""

    def __init__(self) -> None:
        self.checks: list[Mapping[str, Any]] = []
        self.usages: list[Mapping[str, Any]] = []

    async def check(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        estimated_tokens: int,
    ) -> BudgetCheck:
        self.checks.append(
            {"tenant_id": tenant_id, "agent_id": agent_id,
             "estimated_tokens": estimated_tokens},
        )
        return BudgetCheck(allowed=True, remaining_tokens=None, remaining_usd=None)

    async def record_usage(
        self,
        *,
        tenant_id: str | None,
        agent_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        self.usages.append({
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        })


@pytest.fixture
def fake_budget() -> FakeBudgetService:
    return FakeBudgetService()
```

(The `Mapping`, `Any` types are already imported in conftest from Phase 1+2.)

- [ ] **Step 1.20: Find and remove duplicate `FakeBudget`/`_FakeBudget`/`_AlwaysAllowBudget` definitions**

```bash
grep -rn "class.*Budget.*IBudgetService\|class _AlwaysAllowBudget\|class FakeBudget\b" tests/ | head -10
```

Expected duplicates:
- `tests/unit/llm/test_litellm_client.py` has `_AlwaysAllowBudget` (Phase 2 Task 2)
- Possibly other test files

For each duplicate, replace the class definition with a function-scoped use of the new `fake_budget` fixture. Update the test signatures to take `fake_budget` as a parameter.

Example for `test_litellm_client.py`:

```python
# Before
class _AlwaysAllowBudget(IBudgetService):
    async def check(self, *, tenant_id, agent_id, estimated_tokens) -> BudgetCheck:
        return BudgetCheck(allowed=True, remaining_tokens=None, remaining_usd=None)
    async def record_usage(self, *, tenant_id, agent_id, prompt_tokens,
                           completion_tokens, cost_usd) -> None: return None

# ... later ...
client = LiteLLMClient(settings=settings, budget=_AlwaysAllowBudget(), observability=...)

# After
# (delete the class entirely)

# Test signature gains fake_budget:
@pytest.mark.asyncio
async def test_retry_exhausted_timeout_raises_llm_timeout_error(
    monkeypatch: pytest.MonkeyPatch, fake_budget
) -> None:  # noqa: E501
    ...
    client = LiteLLMClient(settings=settings, budget=fake_budget,
                           observability=_NoOpObservability())
```

- [ ] **Step 1.21: Run full unit + component suite to verify no regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 250+ passing; 9 pre-existing errors unchanged.

### 1g — Lint, type-check, commit

- [ ] **Step 1.22: Lint + type-check the full Task 1 change set**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/tools/invoker.py \
    src/ai_core/llm/litellm_client.py \
    src/ai_core/agents/memory.py \
    src/ai_core/di/interfaces.py \
    tests/conftest.py \
    tests/unit/tools/test_invoker.py \
    tests/unit/llm/test_litellm_client.py \
    tests/unit/agents/test_memory.py \
    tests/unit/app/test_runtime.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy \
    src/ai_core/tools/invoker.py \
    src/ai_core/llm/litellm_client.py \
    src/ai_core/agents/memory.py \
    src/ai_core/di/interfaces.py \
    tests/conftest.py \
    tests/unit/tools/test_invoker.py \
    tests/unit/llm/test_litellm_client.py \
    tests/unit/agents/test_memory.py \
    tests/unit/app/test_runtime.py
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no new ruff violations; mypy on touched files clean; total ≤ 21.

- [ ] **Step 1.23: Commit**

```bash
git add src/ai_core/tools/invoker.py \
        src/ai_core/llm/litellm_client.py \
        src/ai_core/agents/memory.py \
        src/ai_core/di/interfaces.py \
        tests/conftest.py \
        tests/unit/tools/test_invoker.py \
        tests/unit/llm/test_litellm_client.py \
        tests/unit/agents/test_memory.py \
        tests/unit/app/test_runtime.py
git commit -m "refactor: Phase 2 coherence batch — widen tool.invoke span, empty-response in llm.complete span, LLMTimeoutError in compaction skip, Literal finish_reason, FakeBudgetService promotion"
```

---

## Task 2 — `structlog` adoption

**Files:**
- Modify: `pyproject.toml` — add `structlog>=24.1.0,<26` to dependencies
- Create: `src/ai_core/observability/logging.py` — get_logger + bind_context + configure
- Modify: `src/ai_core/config/settings.py` — `observability.log_format` setting
- Modify: `src/ai_core/app/runtime.py` — call `configure()` in `__aenter__`
- Modify: `src/ai_core/agents/base.py` — `bind_context` in `ainvoke`
- Modify: 4 callsite files — migrate ~6 high-value warnings to structlog event-name format
- Test: `tests/unit/observability/test_logging.py` (new)

### 2a — Add the dependency

- [ ] **Step 2.1: Add `structlog` to `pyproject.toml`**

Read `pyproject.toml` and find the `dependencies = [...]` list. Add `"structlog>=24.1.0,<26",` to it. Then install:

```bash
/Users/admin-h26/EAAP/.venv/bin/pip install -e /Users/admin-h26/EAAP/ai-core-sdk
```

Expected: `structlog` installs successfully.

- [ ] **Step 2.2: Verify the import works**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import structlog; print(structlog.__version__)"
```

Expected: prints a version ≥ 24.1.0.

### 2b — Create `observability/logging.py`

- [ ] **Step 2.3: Write failing tests**

Create `tests/unit/observability/test_logging.py`:

```python
"""Tests for the structlog logging seam."""
from __future__ import annotations

import logging
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

from ai_core.observability.logging import (
    bind_context,
    configure,
    get_logger,
    unbind_context,
)

pytestmark = pytest.mark.unit


def test_configure_idempotent() -> None:
    """configure() can be called multiple times without raising."""
    configure(log_format="text", log_level="INFO")
    configure(log_format="text", log_level="INFO")  # second call must not raise
    configure(log_format="structured", log_level="DEBUG")  # different config OK


def test_get_logger_returns_bound_logger() -> None:
    configure(log_format="text", log_level="INFO")
    logger = get_logger("ai_core.test")
    assert logger is not None
    # Method exists and is callable.
    assert callable(getattr(logger, "warning", None))


def test_bind_context_propagates_through_log_call() -> None:
    """ContextVar values land as fields on every log line."""
    configure(log_format="structured", log_level="DEBUG")
    token = bind_context(agent_id="agent-x", tenant_id="tenant-y")
    try:
        with capture_logs() as captured:
            logger = get_logger("ai_core.test")
            logger.warning("test.event")
        assert len(captured) == 1
        record = captured[0]
        assert record["event"] == "test.event"
        assert record["agent_id"] == "agent-x"
        assert record["tenant_id"] == "tenant-y"
    finally:
        unbind_context(token)


def test_unbind_context_clears_fields() -> None:
    """After unbind_context, log lines no longer carry the previously-bound fields."""
    configure(log_format="structured", log_level="DEBUG")
    token = bind_context(agent_id="agent-x")
    unbind_context(token)
    with capture_logs() as captured:
        logger = get_logger("ai_core.test")
        logger.warning("test.event")
    assert len(captured) == 1
    assert "agent_id" not in captured[0]


def test_structured_renderer_is_active_when_configured() -> None:
    """log_format='structured' produces JSON-renderable output."""
    configure(log_format="structured", log_level="DEBUG")
    # Sanity: configuration should not have raised; capture_logs operates
    # regardless of renderer, so we test by ensuring configure() succeeds.
    logger = get_logger("ai_core.test")
    with capture_logs() as captured:
        logger.info("config.ok", value=42)
    assert captured[0]["event"] == "config.ok"
    assert captured[0]["value"] == 42


def test_text_renderer_is_active_when_configured() -> None:
    """log_format='text' (default) configures the console renderer."""
    configure(log_format="text", log_level="DEBUG")
    logger = get_logger("ai_core.test")
    with capture_logs() as captured:
        logger.info("config.ok", value=42)
    assert captured[0]["event"] == "config.ok"
```

- [ ] **Step 2.4: Run test to verify it fails**

```bash
mkdir -p tests/unit/observability
[ -f tests/unit/observability/__init__.py ] || touch tests/unit/observability/__init__.py
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/observability/test_logging.py -q
```

Expected: ImportError on `ai_core.observability.logging`.

- [ ] **Step 2.5: Implement `src/ai_core/observability/logging.py`**

```python
"""Structured logging seam.

Other SDK modules import :func:`get_logger` from here, never from
``structlog`` directly. The seam makes it possible to replace structlog
later without touching every callsite.

The :func:`configure` function is called once by :class:`AICoreApp`
during ``__aenter__``. Tests may call it directly with their preferred
shape.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, Literal

import structlog
from structlog.stdlib import LoggerFactory

if TYPE_CHECKING:
    from structlog.typing import FilteringBoundLogger


# Per-task contextual binding. Updated by AICoreApp + BaseAgent on entry.
_request_context: ContextVar[Mapping[str, Any]] = ContextVar(
    "_eaap_request_context", default={}
)


def configure(
    *,
    log_format: Literal["text", "structured"] = "text",
    log_level: str = "INFO",
) -> None:
    """Configure structlog. Idempotent — safe to call multiple times."""
    structlog.reset_defaults()
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _bind_request_context,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
    ]
    if log_format == "structured":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO),
        ),
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _bind_request_context(
    _logger: Any, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject the ContextVar payload (agent_id, tenant_id, …) into every log line."""
    for k, v in _request_context.get().items():
        event_dict.setdefault(k, v)
    return event_dict


def bind_context(**kwargs: Any) -> Token[Mapping[str, Any]]:
    """Push ``kwargs`` into the request-scoped ContextVar; return reset token."""
    current = dict(_request_context.get())
    current.update(kwargs)
    return _request_context.set(current)


def unbind_context(token: Token[Mapping[str, Any]]) -> None:
    """Reset the ContextVar to the prior token's snapshot."""
    _request_context.reset(token)


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog logger bound to module ``name``."""
    return structlog.get_logger(name)


__all__ = ["bind_context", "configure", "get_logger", "unbind_context"]
```

- [ ] **Step 2.6: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/observability/test_logging.py -v
```

Expected: 6 passed.

### 2c — Add `log_format` setting

- [ ] **Step 2.7: Add the field to `ObservabilitySettings`**

In `src/ai_core/config/settings.py`, find `class ObservabilitySettings(BaseSettings):` and add the field after the existing `fail_open` field (or after `log_level` if it's adjacent):

```python
    log_format: Literal["text", "structured"] = Field(
        default="text",
        description=(
            "When 'text' (default), logs render as colorized key=value for local dev. "
            "When 'structured', logs render as JSON for production ingestion."
        ),
    )
```

(`Literal` is already imported at the top of `settings.py`.)

### 2d — Configure structlog in `AICoreApp.__aenter__`

- [ ] **Step 2.8: Update `AICoreApp.__aenter__`**

In `src/ai_core/app/runtime.py`, find `async def __aenter__`. After `validate_for_runtime` and before `Container.build`, add:

```python
        # Phase 3: configure structlog before any code logs.
        from ai_core.observability.logging import configure as _configure_logging
        _configure_logging(
            log_format=self._settings.observability.log_format,
            log_level=self._settings.observability.log_level.value,
        )
```

(Local import to avoid module-load-time structlog config.)

- [ ] **Step 2.9: Verify existing app tests still pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py -v 2>&1 | tail -10
```

Expected: all 9+ tests pass.

### 2e — Bind request context in `BaseAgent.ainvoke`

- [ ] **Step 2.10: Update `BaseAgent.ainvoke`**

In `src/ai_core/agents/base.py`, find `async def ainvoke`. Wrap the existing body in a try/finally that binds + unbinds context:

```python
    async def ainvoke(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        essential: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        """..."""
        from ai_core.observability.logging import bind_context, unbind_context

        log_token = bind_context(
            agent_id=self.agent_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
        )
        try:
            # ... existing body (compile, build_baggage, attach, span, ainvoke, detach, raise) ...
        finally:
            unbind_context(log_token)
```

- [ ] **Step 2.11: Verify existing agent tests still pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/agents tests/component -q 2>&1 | tail -10
```

Expected: all passing.

### 2f — Migrate ~6 high-value callsites to structlog event-name format

- [ ] **Step 2.12: Update `agents/memory.py` compaction warnings**

In `src/ai_core/agents/memory.py`:

```python
# Replace top-of-file:
#   import logging
#   _logger = logging.getLogger(__name__)
# with:
from ai_core.observability.logging import get_logger
_logger = get_logger(__name__)

# Update the two compaction-skip warnings:

# In the `except TimeoutError:` branch (Phase 2 + 1c):
_logger.warning(
    "compaction.skipped.budget_exceeded",
    timeout_seconds=timeout, agent_id=agent_id, tenant_id=tenant_id,
)

# In the `except LLMTimeoutError:` branch (Phase 3 1c):
_logger.warning(
    "compaction.skipped.llm_timeout",
    agent_id=agent_id, tenant_id=tenant_id, error_code=exc.error_code,
)
```

(Pre-existing stdlib logger calls in this file — if any — can stay as-is. structlog's `LoggerFactory` wraps stdlib so existing format strings still work.)

- [ ] **Step 2.13: Update `tools/invoker.py` output validation warning**

In `src/ai_core/tools/invoker.py`:

```python
# Replace top-of-file:
#   import logging
#   _logger = logging.getLogger(__name__)
# with:
from ai_core.observability.logging import get_logger
_logger = get_logger(__name__)

# Update the output-validation warning:
_logger.warning(
    "tool.output_validation_failed",
    tool_name=spec.name, tool_version=spec.version,
    agent_id=agent_id, tenant_id=tenant_id,
)
```

- [ ] **Step 2.14: Update `agents/base.py` tool execution error log**

In `src/ai_core/agents/base.py`, find the `except ToolExecutionError as exc:` block in `_tool_node`:

```python
# Replace top-of-file:
#   import logging
#   _logger = logging.getLogger(__name__)
# with:
from ai_core.observability.logging import get_logger
_logger = get_logger(__name__)

# Update the error log inside _tool_node:
_logger.error(
    "tool.execution_error",
    tool_name=name, agent_id=self.agent_id,
    exc_info=exc,
)
```

- [ ] **Step 2.15: Update `observability/real.py` backend-error logs**

In `src/ai_core/observability/real.py`, find the `_should_swallow` helper (Phase 2 Task 4 quality fix). Replace the two `_logger.warning(...)` calls inside it with structlog event-name format:

```python
# Top of file — change:
#   import logging
#   _logger = logging.getLogger(__name__)
# to:
from ai_core.observability.logging import get_logger
_logger = get_logger(__name__)

# Inside _should_swallow:
def _should_swallow(self, exc: BaseException, context: str) -> bool:
    if self._fail_open:
        _logger.warning(
            "observability.backend_error",
            context=context, error=str(exc), error_type=type(exc).__name__,
            fail_open=True,
        )
        return True
    _logger.warning(
        "observability.backend_error",
        context=context, error=str(exc), error_type=type(exc).__name__,
        fail_open=False,
    )
    return False
```

In the same file, find `_safe_lf_call`'s warning — update to:

```python
async def _safe_lf_call(self, ...):
    try:
        ...
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "langfuse.helper_failed",
            error=str(exc), error_type=type(exc).__name__,
        )
```

(Other `except Exception:` blocks in init/shutdown paths keep their existing stdlib `_logger.warning(...)` format — they're not in the migration scope.)

- [ ] **Step 2.16: Run full unit + component suite to verify no regressions**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 250+ passing.

- [ ] **Step 2.17: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check \
    src/ai_core/observability/ \
    src/ai_core/config/settings.py \
    src/ai_core/app/runtime.py \
    src/ai_core/agents/ \
    src/ai_core/tools/invoker.py \
    tests/unit/observability/test_logging.py \
    pyproject.toml
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/observability/ tests/unit/observability/
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: no new violations; mypy ≤ 21.

- [ ] **Step 2.18: Commit**

```bash
git add pyproject.toml \
        src/ai_core/observability/logging.py \
        src/ai_core/observability/real.py \
        src/ai_core/config/settings.py \
        src/ai_core/app/runtime.py \
        src/ai_core/agents/base.py \
        src/ai_core/agents/memory.py \
        src/ai_core/tools/invoker.py \
        tests/unit/observability/test_logging.py
git commit -m "feat(observability): adopt structlog with ContextVar binding; migrate 6 high-value operational logs to event-name format"
```

---

## Task 3 — `IAuditSink` ABC + 3 implementations + DI binding + `ToolInvoker` integration

**Files:**
- Create: `src/ai_core/audit/__init__.py` — public exports
- Create: `src/ai_core/audit/interface.py` — IAuditSink, AuditRecord, AuditEvent, PayloadRedactor
- Create: `src/ai_core/audit/null.py` — NullAuditSink
- Create: `src/ai_core/audit/otel_event.py` — OTelEventAuditSink
- Create: `src/ai_core/audit/jsonl.py` — JsonlFileAuditSink
- Modify: `src/ai_core/config/settings.py` — `AuditSettings`
- Modify: `src/ai_core/di/module.py` — bind `IAuditSink`
- Modify: `src/ai_core/di/container.py` — `audit.flush` step in teardown
- Modify: `src/ai_core/tools/invoker.py` — record audit events
- Modify: `src/ai_core/__init__.py` — top-level exports
- Test: `tests/unit/audit/` (new dir, ~10 tests + 1 meta-test)

### 3a — Create `audit/interface.py` and `audit/null.py`

- [ ] **Step 3.1: Write failing tests for `AuditRecord` and `NullAuditSink`**

```bash
mkdir -p src/ai_core/audit tests/unit/audit
[ -f tests/unit/audit/__init__.py ] || touch tests/unit/audit/__init__.py
```

Create `tests/unit/audit/test_interface.py`:

```python
"""Tests for AuditRecord, AuditEvent, NullAuditSink, and the redactor protocol."""
from __future__ import annotations

import pytest

from ai_core.audit import AuditEvent, AuditRecord, NullAuditSink

pytestmark = pytest.mark.unit


def test_audit_event_values() -> None:
    """AuditEvent string values are stable identifiers."""
    assert AuditEvent.POLICY_DECISION.value == "policy.decision"
    assert AuditEvent.TOOL_INVOCATION_STARTED.value == "tool.invocation.started"
    assert AuditEvent.TOOL_INVOCATION_COMPLETED.value == "tool.invocation.completed"
    assert AuditEvent.TOOL_INVOCATION_FAILED.value == "tool.invocation.failed"


def test_audit_record_now_populates_timestamp() -> None:
    """AuditRecord.now() stamps a UTC timestamp."""
    rec = AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        tool_name="search", tool_version=1,
        agent_id="a", tenant_id="t",
        decision_path="eaap/policy/allow",
        decision_allowed=True,
    )
    assert rec.event == AuditEvent.POLICY_DECISION
    assert rec.tool_name == "search"
    assert rec.tool_version == 1
    assert rec.decision_allowed is True
    assert rec.timestamp.tzinfo is not None  # has tzinfo (UTC)


def test_audit_record_default_redactor_is_identity() -> None:
    """Without a redactor, payload passes through unchanged."""
    rec = AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        payload={"input": {"q": "hi"}},
    )
    assert rec.payload == {"input": {"q": "hi"}}


def test_audit_record_with_custom_redactor() -> None:
    """Caller can supply a redactor that strips sensitive fields."""
    def _strip_password(payload):
        return {k: v for k, v in payload.items() if k != "password"}

    rec = AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        payload={"input": "hi", "password": "secret"},
        redactor=_strip_password,
    )
    assert "password" not in rec.payload
    assert rec.payload["input"] == "hi"


def test_audit_record_is_frozen() -> None:
    """AuditRecord is immutable."""
    rec = AuditRecord.now(AuditEvent.POLICY_DECISION)
    with pytest.raises(Exception):
        rec.tool_name = "x"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_null_audit_sink_record_is_noop() -> None:
    sink = NullAuditSink()
    # Should not raise; should not require any setup.
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))


@pytest.mark.asyncio
async def test_null_audit_sink_flush_is_idempotent() -> None:
    sink = NullAuditSink()
    await sink.flush()
    await sink.flush()  # second call must not raise
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_interface.py -q
```

Expected: ImportError on `ai_core.audit`.

- [ ] **Step 3.3: Create `src/ai_core/audit/interface.py`**

```python
"""Audit sink abstraction + record + event types.

An :class:`IAuditSink` records discrete events that need durable, queryable
retention for compliance: policy decisions, tool invocation outcomes, and
(optionally, redacted) payloads. Concrete sinks ship in this subpackage.

Sinks NEVER raise from :meth:`record` or :meth:`flush` — any backend error
must be swallowed internally and logged. Audit is best-effort by design;
its failure must not block the calling pipeline.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class AuditEvent(str, enum.Enum):
    """Discrete event types recorded by the audit sink."""

    POLICY_DECISION = "policy.decision"
    TOOL_INVOCATION_STARTED = "tool.invocation.started"
    TOOL_INVOCATION_COMPLETED = "tool.invocation.completed"
    TOOL_INVOCATION_FAILED = "tool.invocation.failed"


# Optional pluggable redaction. Default identity. Implementations may strip PII
# before the record reaches the sink.
PayloadRedactor = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _identity_redactor(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(payload)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Immutable audit record. Sinks accept records via :meth:`IAuditSink.record`."""

    event: AuditEvent
    timestamp: datetime
    tool_name: str | None
    tool_version: int | None
    agent_id: str | None
    tenant_id: str | None
    decision_path: str | None
    decision_allowed: bool | None
    decision_reason: str | None
    error_code: str | None
    payload: Mapping[str, Any]
    latency_ms: float | None

    @classmethod
    def now(
        cls,
        event: AuditEvent,
        *,
        tool_name: str | None = None,
        tool_version: int | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
        decision_path: str | None = None,
        decision_allowed: bool | None = None,
        decision_reason: str | None = None,
        error_code: str | None = None,
        payload: Mapping[str, Any] | None = None,
        latency_ms: float | None = None,
        redactor: PayloadRedactor = _identity_redactor,
    ) -> AuditRecord:
        return cls(
            event=event,
            timestamp=datetime.now(UTC),
            tool_name=tool_name,
            tool_version=tool_version,
            agent_id=agent_id,
            tenant_id=tenant_id,
            decision_path=decision_path,
            decision_allowed=decision_allowed,
            decision_reason=decision_reason,
            error_code=error_code,
            payload=dict(redactor(payload or {})),
            latency_ms=latency_ms,
        )


class IAuditSink(ABC):
    """Durable record of policy and tool events.

    Implementations MUST:

    * be safe for concurrent use across coroutines;
    * never raise from :meth:`record` or :meth:`flush`;
    * make :meth:`flush` idempotent (called by Container.stop at shutdown).
    """

    @abstractmethod
    async def record(self, record: AuditRecord) -> None:
        """Persist a single audit record. Best-effort; never raises."""

    @abstractmethod
    async def flush(self) -> None:
        """Flush any buffered records. Idempotent."""


__all__ = [
    "AuditEvent",
    "AuditRecord",
    "IAuditSink",
    "PayloadRedactor",
]
```

- [ ] **Step 3.4: Create `src/ai_core/audit/null.py`**

```python
"""No-op audit sink. Default DI binding for development."""

from __future__ import annotations

from ai_core.audit.interface import AuditRecord, IAuditSink


class NullAuditSink(IAuditSink):
    """Audit sink that drops every record. Default for local development."""

    async def record(self, record: AuditRecord) -> None:
        return None

    async def flush(self) -> None:
        return None


__all__ = ["NullAuditSink"]
```

- [ ] **Step 3.5: Create `src/ai_core/audit/__init__.py`**

```python
"""Audit subsystem — policy and tool event records for compliance."""

from __future__ import annotations

from ai_core.audit.interface import (
    AuditEvent,
    AuditRecord,
    IAuditSink,
    PayloadRedactor,
)
from ai_core.audit.null import NullAuditSink

__all__ = [
    "AuditEvent",
    "AuditRecord",
    "IAuditSink",
    "NullAuditSink",
    "PayloadRedactor",
]
```

- [ ] **Step 3.6: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_interface.py -v
```

Expected: 7 passed.

### 3b — Create `audit/otel_event.py`

- [ ] **Step 3.7: Write failing tests**

Create `tests/unit/audit/test_otel_event_sink.py`:

```python
"""Tests for OTelEventAuditSink — records via IObservabilityProvider.record_event."""
from __future__ import annotations

import pytest

from ai_core.audit import AuditEvent, AuditRecord
from ai_core.audit.otel_event import OTelEventAuditSink

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_otel_event_sink_records_via_observability(fake_observability) -> None:
    """OTelEventAuditSink calls record_event with eaap.audit.<event> name."""
    sink = OTelEventAuditSink(fake_observability)
    await sink.record(AuditRecord.now(
        AuditEvent.POLICY_DECISION,
        tool_name="search", tool_version=1,
        agent_id="a", tenant_id="t",
        decision_path="eaap/policy/allow",
        decision_allowed=True,
        decision_reason="ok",
    ))
    events = [(name, dict(attrs)) for name, attrs in fake_observability.events]
    assert any(name == "eaap.audit.policy.decision" for name, _ in events)
    matching = next(attrs for name, attrs in events if name == "eaap.audit.policy.decision")
    assert matching["audit.tool_name"] == "search"
    assert matching["audit.tool_version"] == 1
    assert matching["audit.decision_allowed"] is True


@pytest.mark.asyncio
async def test_otel_event_sink_swallows_backend_errors(fake_observability) -> None:
    """If the observability provider's record_event raises, the sink swallows."""
    class _BadObservability:
        async def record_event(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("backend down")
        # Stubs for unused interface methods (sink doesn't call them).
        def start_span(self, *args: object, **kwargs: object): ...  # noqa: ANN201
        async def record_llm_usage(self, *args: object, **kwargs: object) -> None: ...
        async def shutdown(self) -> None: ...

    sink = OTelEventAuditSink(_BadObservability())  # type: ignore[arg-type]
    # Must not raise.
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))


@pytest.mark.asyncio
async def test_otel_event_sink_flush_is_noop() -> None:
    """OTelEventAuditSink.flush is a no-op (observability owns its flush)."""
    class _NoopObs:
        def start_span(self, *args, **kwargs): ...  # noqa: ANN
        async def record_llm_usage(self, *args, **kwargs) -> None: ...
        async def record_event(self, *args, **kwargs) -> None: ...
        async def shutdown(self) -> None: ...

    sink = OTelEventAuditSink(_NoopObs())  # type: ignore[arg-type]
    await sink.flush()
```

- [ ] **Step 3.8: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_otel_event_sink.py -q
```

Expected: ImportError on `ai_core.audit.otel_event`.

- [ ] **Step 3.9: Create `src/ai_core/audit/otel_event.py`**

```python
"""OTel-event audit sink — records audit events via IObservabilityProvider.

Trace-shaped retention applies (sampled, time-series). For compliance-grade
durability use :class:`JsonlFileAuditSink` or a custom backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.observability.logging import get_logger

if TYPE_CHECKING:
    from ai_core.di.interfaces import IObservabilityProvider

_logger = get_logger(__name__)


class OTelEventAuditSink(IAuditSink):
    """Records audit events as observability events.

    Each record produces one ``eaap.audit.<event>`` event with structured
    attributes carrying the audit fields (tool, agent, decision, latency).
    The ``payload`` field is intentionally NOT emitted to OTel (cardinality
    concern); use :class:`JsonlFileAuditSink` for payload retention.
    """

    def __init__(self, observability: IObservabilityProvider) -> None:
        self._obs = observability

    async def record(self, record: AuditRecord) -> None:
        try:
            await self._obs.record_event(
                f"eaap.audit.{record.event.value}",
                attributes=_record_to_attributes(record),
            )
        except Exception as exc:  # noqa: BLE001 — sinks NEVER raise
            _logger.warning(
                "audit.otel_sink.failed",
                event=record.event.value, error=str(exc),
                error_type=type(exc).__name__,
            )

    async def flush(self) -> None:
        return None


def _record_to_attributes(record: AuditRecord) -> dict[str, Any]:
    """Render an AuditRecord into a flat OTel-attribute dict (scalars only)."""
    return {
        "audit.timestamp": record.timestamp.isoformat(),
        "audit.tool_name": record.tool_name or "",
        "audit.tool_version": record.tool_version or 0,
        "audit.agent_id": record.agent_id or "",
        "audit.tenant_id": record.tenant_id or "",
        "audit.decision_path": record.decision_path or "",
        "audit.decision_allowed": (
            record.decision_allowed if record.decision_allowed is not None else False
        ),
        "audit.decision_reason": record.decision_reason or "",
        "audit.error_code": record.error_code or "",
        "audit.latency_ms": record.latency_ms or 0.0,
    }


__all__ = ["OTelEventAuditSink"]
```

Update `src/ai_core/audit/__init__.py` to add the new export:

```python
from ai_core.audit.otel_event import OTelEventAuditSink
__all__ = [..., "OTelEventAuditSink"]
```

- [ ] **Step 3.10: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_otel_event_sink.py -v
```

Expected: 3 passed.

### 3c — Create `audit/jsonl.py`

- [ ] **Step 3.11: Write failing tests**

Create `tests/unit/audit/test_jsonl_sink.py`:

```python
"""Tests for JsonlFileAuditSink."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ai_core.audit import AuditEvent, AuditRecord
from ai_core.audit.jsonl import JsonlFileAuditSink

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_jsonl_sink_writes_line_delimited(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path, buffer_size=2)
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION,
                                       tool_name="a", agent_id="x"))
    await sink.record(AuditRecord.now(AuditEvent.TOOL_INVOCATION_COMPLETED,
                                       tool_name="b", latency_ms=12.5))
    # Buffer fills at 2 records → flush triggers.
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    record_a = json.loads(lines[0])
    assert record_a["event"] == "policy.decision"
    assert record_a["tool_name"] == "a"
    record_b = json.loads(lines[1])
    assert record_b["event"] == "tool.invocation.completed"
    assert record_b["latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_jsonl_sink_flush_drains_buffer(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path, buffer_size=100)
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))
    # Buffer is below threshold; nothing on disk yet.
    assert not path.exists() or path.read_text() == ""
    await sink.flush()
    # After flush, the record is on disk.
    assert path.read_text().count("\n") == 1


@pytest.mark.asyncio
async def test_jsonl_sink_flush_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path)
    await sink.flush()  # buffer empty
    await sink.flush()  # second call must not raise
    assert not path.exists() or path.read_text() == ""


@pytest.mark.asyncio
async def test_jsonl_sink_handles_concurrent_record_calls(tmp_path: Path) -> None:
    """Concurrent record() calls should not interleave bytes within a line."""
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path, buffer_size=1)  # flush after every record
    await asyncio.gather(*(
        sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION,
                                     tool_name=f"tool-{i}"))
        for i in range(20)
    ))
    lines = path.read_text().splitlines()
    assert len(lines) == 20
    # Every line is a valid JSON object — no interleaving.
    for line in lines:
        json.loads(line)


@pytest.mark.asyncio
async def test_jsonl_sink_swallows_write_errors(tmp_path: Path, monkeypatch) -> None:
    """If the file write fails, record() must NOT raise."""
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path, buffer_size=1)

    def _explode(*args, **kwargs):
        raise OSError("disk full (test-injected)")

    monkeypatch.setattr(sink, "_append_lines", _explode)
    # Must not raise.
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))
```

- [ ] **Step 3.12: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_jsonl_sink.py -q
```

Expected: ImportError.

- [ ] **Step 3.13: Create `src/ai_core/audit/jsonl.py`**

```python
"""Line-delimited JSON audit sink.

Suitable for dev/test/single-tenant deployments. Buffered writes via
``asyncio.to_thread`` keep the event loop responsive. ``flush()`` is
called by ``Container.stop`` at shutdown to drain any partial buffer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ai_core.audit.interface import AuditRecord, IAuditSink
from ai_core.observability.logging import get_logger

_logger = get_logger(__name__)


class JsonlFileAuditSink(IAuditSink):
    """Append audit records as line-delimited JSON to a local file.

    Args:
        path: Filesystem path where audit records are appended.
        buffer_size: Number of records to buffer before flushing to disk.
    """

    def __init__(self, path: Path | str, *, buffer_size: int = 64) -> None:
        self._path = Path(path)
        self._buffer: list[dict[str, Any]] = []
        self._buffer_size = buffer_size
        self._lock = asyncio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def record(self, record: AuditRecord) -> None:
        try:
            payload = _record_to_dict(record)
            async with self._lock:
                self._buffer.append(payload)
                if len(self._buffer) >= self._buffer_size:
                    await self._flush_locked()
        except Exception as exc:  # noqa: BLE001 — sinks NEVER raise
            _logger.warning(
                "audit.jsonl_sink.failed",
                event=record.event.value, error=str(exc),
                error_type=type(exc).__name__,
            )

    async def flush(self) -> None:
        try:
            async with self._lock:
                await self._flush_locked()
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "audit.jsonl_sink.flush_failed",
                error=str(exc), error_type=type(exc).__name__,
            )

    async def _flush_locked(self) -> None:
        """Drain the buffer to disk. Caller MUST hold ``self._lock``."""
        if not self._buffer:
            return
        records = self._buffer
        self._buffer = []
        await asyncio.to_thread(self._append_lines, records)

    def _append_lines(self, records: list[dict[str, Any]]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def _record_to_dict(record: AuditRecord) -> dict[str, Any]:
    return {
        "event": record.event.value,
        "timestamp": record.timestamp.isoformat(),
        "tool_name": record.tool_name,
        "tool_version": record.tool_version,
        "agent_id": record.agent_id,
        "tenant_id": record.tenant_id,
        "decision_path": record.decision_path,
        "decision_allowed": record.decision_allowed,
        "decision_reason": record.decision_reason,
        "error_code": record.error_code,
        "latency_ms": record.latency_ms,
        "payload": dict(record.payload),
    }


__all__ = ["JsonlFileAuditSink"]
```

Update `audit/__init__.py`:

```python
from ai_core.audit.jsonl import JsonlFileAuditSink
__all__ = [..., "JsonlFileAuditSink"]
```

- [ ] **Step 3.14: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit -v 2>&1 | tail -20
```

Expected: 15+ passed (7 + 3 + 5 = 15).

### 3d — DI binding + settings

- [ ] **Step 3.15: Add `AuditSettings` to `src/ai_core/config/settings.py`**

Add a new class definition before `AppSettings`:

```python
class AuditSettings(BaseSettings):
    """Audit-sink configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    sink_type: Literal["null", "otel_event", "jsonl"] = "null"
    jsonl_path: Path | None = None  # required when sink_type == "jsonl"
```

Add `from pathlib import Path` to the imports if not already present.

In `class AppSettings(BaseSettings):`, add the new field next to existing groups:

```python
audit: AuditSettings = Field(default_factory=AuditSettings)
```

- [ ] **Step 3.16: Add `provide_audit_sink` to `AgentModule`**

In `src/ai_core/di/module.py`, add the provider:

```python
@singleton
@provider
def provide_audit_sink(
    self,
    settings: AppSettings,
    observability: IObservabilityProvider,
) -> IAuditSink:
    """Default audit sink — switchable via settings.audit.sink_type."""
    sink_type = settings.audit.sink_type
    if sink_type == "null":
        from ai_core.audit.null import NullAuditSink  # noqa: PLC0415
        return NullAuditSink()
    if sink_type == "otel_event":
        from ai_core.audit.otel_event import OTelEventAuditSink  # noqa: PLC0415
        return OTelEventAuditSink(observability)
    if sink_type == "jsonl":
        from ai_core.audit.jsonl import JsonlFileAuditSink  # noqa: PLC0415
        if settings.audit.jsonl_path is None:
            raise ConfigurationError(
                "audit.sink_type='jsonl' requires audit.jsonl_path to be set",
                error_code="config.invalid",
            )
        return JsonlFileAuditSink(settings.audit.jsonl_path)
    raise ConfigurationError(
        f"Unknown audit.sink_type: {sink_type!r}",
        error_code="config.invalid",
    )
```

Add the import at the top of `module.py`:

```python
from ai_core.audit import IAuditSink
from ai_core.exceptions import ConfigurationError
```

- [ ] **Step 3.17: Add `audit.flush` to `Container._teardown_sdk_resources`**

In `src/ai_core/di/container.py`, find `_teardown_sdk_resources`. Add the new step:

```python
from ai_core.audit import IAuditSink  # local import inside the method body

steps: list[tuple[str, type[Any], tuple[str, ...]]] = [
    ("observability.shutdown", IObservabilityProvider, ("shutdown",)),
    ("audit.flush", IAuditSink, ("flush",)),  # NEW
    ("policy_evaluator.aclose", IPolicyEvaluator, ("aclose",)),
    ("engine.dispose", AsyncEngine, ("dispose",)),
]
```

(Keep the local import pattern matching the existing code.)

- [ ] **Step 3.18: Write a DI-binding test**

Create `tests/unit/audit/test_di_binding.py`:

```python
"""Tests for AgentModule's audit-sink binding."""
from __future__ import annotations

import pytest
from pathlib import Path

from ai_core.audit import IAuditSink, NullAuditSink, OTelEventAuditSink, JsonlFileAuditSink
from ai_core.config.settings import AppSettings, AuditSettings
from ai_core.di import AgentModule, Container
from ai_core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


def test_default_sink_is_null() -> None:
    container = Container.build([AgentModule()])
    sink = container.get(IAuditSink)
    assert isinstance(sink, NullAuditSink)


def test_otel_event_sink_when_configured() -> None:
    settings = AppSettings(audit=AuditSettings(sink_type="otel_event"))
    container = Container.build([AgentModule(settings=settings)])
    sink = container.get(IAuditSink)
    assert isinstance(sink, OTelEventAuditSink)


def test_jsonl_sink_when_configured(tmp_path: Path) -> None:
    settings = AppSettings(audit=AuditSettings(sink_type="jsonl",
                                                jsonl_path=tmp_path / "audit.jsonl"))
    container = Container.build([AgentModule(settings=settings)])
    sink = container.get(IAuditSink)
    assert isinstance(sink, JsonlFileAuditSink)


def test_jsonl_sink_without_path_raises() -> None:
    settings = AppSettings(audit=AuditSettings(sink_type="jsonl", jsonl_path=None))
    container = Container.build([AgentModule(settings=settings)])
    with pytest.raises(ConfigurationError):
        container.get(IAuditSink)
```

- [ ] **Step 3.19: Run DI tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit/test_di_binding.py -v
```

Expected: 4 passed.

### 3e — `ToolInvoker` integration

- [ ] **Step 3.20: Write failing tests for audit-sink integration**

Append to `tests/unit/tools/test_invoker.py`:

```python
@pytest.mark.asyncio
async def test_invoker_records_policy_decision(
    fake_observability, fake_policy_evaluator_factory, fake_audit_sink
) -> None:
    """ToolInvoker records POLICY_DECISION audit event after OPA evaluation."""
    from ai_core.audit import AuditEvent

    inv = ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(default_allow=True, reason="ok"),
        registry=SchemaRegistry(),
        audit=fake_audit_sink,
    )
    await inv.invoke(_search, {"q": "x", "limit": 1}, agent_id="a", tenant_id="t")
    events = [r.event for r in fake_audit_sink.records]
    assert AuditEvent.POLICY_DECISION in events
    assert AuditEvent.TOOL_INVOCATION_COMPLETED in events


@pytest.mark.asyncio
async def test_invoker_records_failure_on_handler_raise(
    fake_observability, fake_policy_evaluator_factory, fake_audit_sink
) -> None:
    """ToolInvoker records TOOL_INVOCATION_FAILED with error_code on handler raise."""
    from ai_core.audit import AuditEvent

    @tool(name="boom", version=1)
    async def boom(payload: _In) -> _Out:
        raise RuntimeError("kaboom")

    inv = ToolInvoker(
        observability=fake_observability,
        policy=fake_policy_evaluator_factory(),
        registry=SchemaRegistry(),
        audit=fake_audit_sink,
    )
    with pytest.raises(ToolExecutionError):
        await inv.invoke(boom, {"q": "x", "limit": 1}, agent_id="a")

    failed = [r for r in fake_audit_sink.records if r.event == AuditEvent.TOOL_INVOCATION_FAILED]
    assert len(failed) == 1
    assert failed[0].error_code == "tool.execution_failed"
    assert failed[0].agent_id == "a"
```

- [ ] **Step 3.21: Add `FakeAuditSink` to `tests/conftest.py`**

Append to `tests/conftest.py`:

```python
# ---------------------------------------------------------------------------
# FakeAuditSink — Phase 3
# ---------------------------------------------------------------------------
from ai_core.audit import AuditRecord, IAuditSink


class FakeAuditSink(IAuditSink):
    """Records audit records for assertion in tests."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []
        self.flushes: int = 0

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)

    async def flush(self) -> None:
        self.flushes += 1


@pytest.fixture
def fake_audit_sink() -> FakeAuditSink:
    return FakeAuditSink()
```

- [ ] **Step 3.22: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py::test_invoker_records_policy_decision tests/unit/tools/test_invoker.py::test_invoker_records_failure_on_handler_raise -v
```

Expected: failures because `ToolInvoker.__init__` doesn't accept `audit` kwarg yet.

- [ ] **Step 3.23: Modify `ToolInvoker` to accept and use `audit`**

In `src/ai_core/tools/invoker.py`:

Update imports near the top (replace existing imports as needed):

```python
import time
from collections.abc import Mapping  # if not already there
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ai_core.audit import AuditEvent, AuditRecord, NullAuditSink
from ai_core.exceptions import (
    PolicyDenialError,
    SchemaValidationError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.observability.logging import get_logger

if TYPE_CHECKING:
    from ai_core.audit import IAuditSink
    from ai_core.di.interfaces import IObservabilityProvider, IPolicyEvaluator
    from ai_core.schema.registry import SchemaRegistry
    from ai_core.tools.spec import ToolSpec

_logger = get_logger(__name__)
```

Update `__init__` to accept `audit`:

```python
    def __init__(
        self,
        *,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator | None = None,
        registry: SchemaRegistry | None = None,
        audit: IAuditSink | None = None,
    ) -> None:
        self._observability = observability
        self._policy = policy
        self._registry = registry
        self._audit: IAuditSink = audit or NullAuditSink()
```

Update `invoke` method body — wrap the existing widened-span logic with audit calls (after OPA + before completion + on failure):

```python
    async def invoke(
        self,
        spec: ToolSpec,
        raw_args: Mapping[str, Any],
        *,
        principal: Mapping[str, Any] | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Mapping[str, Any]:
        attrs: dict[str, Any] = {
            "tool.name": spec.name,
            "tool.version": spec.version,
            "agent_id": agent_id or "",
            "tenant_id": tenant_id or "",
        }
        started = time.monotonic()
        try:
            async with self._observability.start_span("tool.invoke", attributes=attrs):
                # Step 1: input validation
                try:
                    payload = spec.input_model.model_validate(dict(raw_args))
                except ValidationError as exc:
                    raise ToolValidationError(
                        f"Tool '{spec.name}' v{spec.version} input failed validation.",
                        details={"tool": spec.name, "version": spec.version,
                                 "side": "input", "errors": exc.errors()},
                        cause=exc,
                    ) from exc

                # Step 2: OPA enforcement + audit decision record
                if spec.opa_path is not None and self._policy is not None:
                    decision = await self._policy.evaluate(
                        decision_path=spec.opa_path,
                        input={
                            "tool": spec.name,
                            "version": spec.version,
                            "payload": payload.model_dump(),
                            "user": dict(principal or {}),
                            "agent_id": agent_id,
                            "tenant_id": tenant_id,
                        },
                    )
                    await self._audit.record(AuditRecord.now(
                        AuditEvent.POLICY_DECISION,
                        tool_name=spec.name, tool_version=spec.version,
                        agent_id=agent_id, tenant_id=tenant_id,
                        decision_path=spec.opa_path,
                        decision_allowed=decision.allowed,
                        decision_reason=decision.reason,
                        payload={"input": payload.model_dump()},
                    ))
                    if not decision.allowed:
                        raise PolicyDenialError(
                            f"Tool '{spec.name}' v{spec.version} denied by policy: "
                            f"{decision.reason or 'no reason provided'}",
                            details={"tool": spec.name, "version": spec.version,
                                     "reason": decision.reason,
                                     "agent_id": agent_id, "tenant_id": tenant_id},
                        )

                # Steps 3+4: handler call
                try:
                    result: Any = await spec.handler(payload)
                except Exception as exc:  # noqa: BLE001
                    raise ToolExecutionError(
                        f"Tool '{spec.name}' v{spec.version} failed: {exc}",
                        details={"tool": spec.name, "version": spec.version,
                                 "agent_id": agent_id, "tenant_id": tenant_id},
                        cause=exc,
                    ) from exc

                # Step 5: output validation
                try:
                    validated: BaseModel = spec.output_model.model_validate(result)
                except ValidationError as exc:
                    _logger.warning(
                        "tool.output_validation_failed",
                        tool_name=spec.name, tool_version=spec.version,
                        agent_id=agent_id, tenant_id=tenant_id,
                    )
                    raise ToolValidationError(
                        f"Tool '{spec.name}' v{spec.version} returned invalid data.",
                        details={"tool": spec.name, "version": spec.version,
                                 "side": "output", "errors": exc.errors()},
                        cause=exc,
                    ) from exc

            # Step 6: completion event + audit record
            latency_ms = (time.monotonic() - started) * 1000.0
            await self._observability.record_event(
                "tool.completed",
                attributes={**attrs, "latency_ms": latency_ms},
            )
            await self._audit.record(AuditRecord.now(
                AuditEvent.TOOL_INVOCATION_COMPLETED,
                tool_name=spec.name, tool_version=spec.version,
                agent_id=agent_id, tenant_id=tenant_id,
                latency_ms=latency_ms,
            ))
            return validated.model_dump(mode="json")
        except (ToolValidationError, PolicyDenialError, ToolExecutionError) as exc:
            latency_ms = (time.monotonic() - started) * 1000.0
            await self._audit.record(AuditRecord.now(
                AuditEvent.TOOL_INVOCATION_FAILED,
                tool_name=spec.name, tool_version=spec.version,
                agent_id=agent_id, tenant_id=tenant_id,
                error_code=exc.error_code,
                latency_ms=latency_ms,
            ))
            raise
```

- [ ] **Step 3.24: Update `AgentModule.provide_tool_invoker` to wire the audit sink**

In `src/ai_core/di/module.py`, find `provide_tool_invoker`. Update its signature and body:

```python
@singleton
@provider
def provide_tool_invoker(
    self,
    observability: IObservabilityProvider,
    policy: IPolicyEvaluator,
    registry: SchemaRegistry,
    audit: IAuditSink,
) -> ToolInvoker:
    """Return the singleton :class:`ToolInvoker` wired to the SDK's services."""
    return ToolInvoker(
        observability=observability,
        policy=policy,
        registry=registry,
        audit=audit,
    )
```

- [ ] **Step 3.25: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/tools/test_invoker.py tests/unit/audit -v 2>&1 | tail -25
```

Expected: all passing (existing 17+ in test_invoker.py + 2 new + 16+ in audit/).

### 3f — Top-level exports + meta-test

- [ ] **Step 3.26: Update `src/ai_core/__init__.py`**

Add the new audit imports and to `__all__`:

```python
from ai_core.audit import AuditEvent, AuditRecord, IAuditSink

__all__ = [
    ...,  # existing
    # Audit
    "IAuditSink",
    "AuditRecord",
    "AuditEvent",
]
```

- [ ] **Step 3.27: Update top-level imports test**

In `tests/unit/test_top_level_imports.py`, add `IAuditSink`, `AuditRecord`, `AuditEvent` to the import list and assertion list.

- [ ] **Step 3.28: Write the meta-test that all sinks "never raise"**

Create `tests/unit/audit/test_never_raises.py`:

```python
"""Meta-test: every IAuditSink implementation MUST swallow internal errors."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai_core.audit import AuditEvent, AuditRecord
from ai_core.audit.jsonl import JsonlFileAuditSink
from ai_core.audit.null import NullAuditSink
from ai_core.audit.otel_event import OTelEventAuditSink

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_null_sink_record_never_raises() -> None:
    sink = NullAuditSink()
    # Even with a malformed-shape AuditRecord, NullAuditSink never raises.
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))
    await sink.flush()


@pytest.mark.asyncio
async def test_otel_event_sink_record_never_raises_when_backend_fails() -> None:
    """Backend exception inside record_event must be swallowed."""
    class _BadObs:
        def start_span(self, *args: Any, **kwargs: Any) -> Any: ...
        async def record_llm_usage(self, *args: Any, **kwargs: Any) -> None: ...
        async def record_event(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("backend down")
        async def shutdown(self) -> None: ...

    sink = OTelEventAuditSink(_BadObs())  # type: ignore[arg-type]
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))


@pytest.mark.asyncio
async def test_jsonl_sink_record_never_raises_when_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    sink = JsonlFileAuditSink(tmp_path / "audit.jsonl", buffer_size=1)

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full (test-injected)")

    monkeypatch.setattr(sink, "_append_lines", _explode)
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))


@pytest.mark.asyncio
async def test_jsonl_sink_flush_never_raises_when_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    sink = JsonlFileAuditSink(tmp_path / "audit.jsonl", buffer_size=100)
    await sink.record(AuditRecord.now(AuditEvent.POLICY_DECISION))

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full (test-injected)")

    monkeypatch.setattr(sink, "_append_lines", _explode)
    await sink.flush()  # must not raise
```

- [ ] **Step 3.29: Run all audit tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/audit -v 2>&1 | tail -20
```

Expected: 19+ passed.

- [ ] **Step 3.30: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 270+ passing; 9 pre-existing errors unchanged.

- [ ] **Step 3.31: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/audit src/ai_core/di tests/unit/audit
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/audit src/ai_core/di tests/unit/audit
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 21.

- [ ] **Step 3.32: Commit**

```bash
git add src/ai_core/audit/ \
        src/ai_core/__init__.py \
        src/ai_core/config/settings.py \
        src/ai_core/di/module.py \
        src/ai_core/di/container.py \
        src/ai_core/tools/invoker.py \
        tests/conftest.py \
        tests/unit/audit/ \
        tests/unit/tools/test_invoker.py \
        tests/unit/test_top_level_imports.py
git commit -m "feat(audit): IAuditSink + Null/OTelEvent/Jsonl sinks; ToolInvoker records policy decisions and tool outcomes"
```

---

## Task 4 — Real health probes + async `app.health`

**Files:**
- Create: `src/ai_core/health/__init__.py`
- Create: `src/ai_core/health/interface.py` — IHealthProbe, ProbeResult, HealthStatus
- Create: `src/ai_core/health/probes.py` — 4 concrete probes
- Modify: `src/ai_core/config/settings.py` — `HealthSettings`
- Modify: `src/ai_core/di/module.py` — bind `list[IHealthProbe]`
- Modify: `src/ai_core/app/runtime.py` — `_HealthCheckRunner`, async `app.health()`, new `component_details` field
- Modify: `src/ai_core/__init__.py` — top-level exports
- Modify: `tests/unit/app/test_runtime.py` — async health tests
- Test: `tests/unit/health/` (new dir)

### 4a — Create `health/interface.py` and `health/__init__.py`

- [ ] **Step 4.1: Write failing test**

```bash
mkdir -p src/ai_core/health tests/unit/health
[ -f tests/unit/health/__init__.py ] || touch tests/unit/health/__init__.py
```

Create `tests/unit/health/test_interface.py`:

```python
"""Tests for IHealthProbe / ProbeResult / HealthStatus."""
from __future__ import annotations

import pytest

from ai_core.health import IHealthProbe, ProbeResult

pytestmark = pytest.mark.unit


def test_probe_result_is_frozen() -> None:
    result = ProbeResult(component="db", status="ok")
    with pytest.raises(Exception):
        result.status = "down"  # type: ignore[misc]


def test_probe_result_default_detail() -> None:
    result = ProbeResult(component="db", status="ok")
    assert result.detail is None


def test_probe_result_with_detail() -> None:
    result = ProbeResult(component="db", status="degraded",
                         detail="slow response (1.8s)")
    assert result.detail == "slow response (1.8s)"


def test_ihealthprobe_is_abstract() -> None:
    with pytest.raises(TypeError):
        IHealthProbe()  # type: ignore[abstract]
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/health/test_interface.py -q
```

Expected: ImportError on `ai_core.health`.

- [ ] **Step 4.3: Create `src/ai_core/health/interface.py`**

```python
"""Health-probe abstraction.

A :class:`IHealthProbe` runs a cheap reachability check against one
subsystem and returns a structured :class:`ProbeResult`. The
``HealthCheckRunner`` (in :mod:`ai_core.app.runtime`) fans out probes
in parallel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


HealthStatus = Literal["ok", "degraded", "down"]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of a single probe.

    Attributes:
        component: Subsystem name, e.g. ``"database"``, ``"opa"``, ``"model_lookup"``.
        status: ``"ok"`` (reachable + responsive), ``"degraded"`` (responding
            but with warnings), ``"down"`` (unreachable / errored).
        detail: Optional human-readable detail (latency_ms, error_code,
            response message).
    """

    component: str
    status: HealthStatus
    detail: str | None = None


class IHealthProbe(ABC):
    """One probe runs one component reachability check."""

    component: str  # class-level — name used in HealthSnapshot.components

    @abstractmethod
    async def probe(self) -> ProbeResult:
        """Run the probe. Implementations MUST NOT raise — return a
        :class:`ProbeResult` with ``status="down"`` and ``detail`` explaining
        the failure instead.
        """


__all__ = ["HealthStatus", "IHealthProbe", "ProbeResult"]
```

- [ ] **Step 4.4: Create `src/ai_core/health/__init__.py`**

```python
"""Health-probe subsystem — async parallel probes for `app.health()`."""

from __future__ import annotations

from ai_core.health.interface import HealthStatus, IHealthProbe, ProbeResult

__all__ = ["HealthStatus", "IHealthProbe", "ProbeResult"]
```

- [ ] **Step 4.5: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/health/test_interface.py -v
```

Expected: 4 passed.

### 4b — Create `health/probes.py` with 4 concrete probes

- [ ] **Step 4.6: Write failing tests for `SettingsProbe`, `OPAReachabilityProbe`, `DatabaseProbe`, `ModelLookupProbe`**

Create `tests/unit/health/test_probes.py`:

```python
"""Tests for the four shipped health probes."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_core.config.settings import AppSettings
from ai_core.health.probes import (
    DatabaseProbe,
    ModelLookupProbe,
    OPAReachabilityProbe,
    SettingsProbe,
)

pytestmark = pytest.mark.unit


# --- SettingsProbe ---

@pytest.mark.asyncio
async def test_settings_probe_always_ok() -> None:
    probe = SettingsProbe(AppSettings())
    result = await probe.probe()
    assert result.component == "settings"
    assert result.status == "ok"


# --- OPAReachabilityProbe ---

@pytest.mark.asyncio
async def test_opa_probe_ok_on_2xx() -> None:
    probe = OPAReachabilityProbe(AppSettings())
    fake_response = MagicMock(status_code=200)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()
    assert result.status == "ok"
    assert "200" in (result.detail or "")


@pytest.mark.asyncio
async def test_opa_probe_degraded_on_5xx() -> None:
    probe = OPAReachabilityProbe(AppSettings())
    fake_response = MagicMock(status_code=503)
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.return_value = fake_response
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()
    assert result.status == "degraded"


@pytest.mark.asyncio
async def test_opa_probe_down_on_connect_error() -> None:
    probe = OPAReachabilityProbe(AppSettings())
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.side_effect = httpx.ConnectError("refused")
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()
    assert result.status == "down"


@pytest.mark.asyncio
async def test_opa_probe_never_raises_on_unexpected_error() -> None:
    """Even on a non-httpx exception, the probe returns down rather than raising."""
    probe = OPAReachabilityProbe(AppSettings())
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.get.side_effect = RuntimeError("unexpected")
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await probe.probe()
    assert result.status == "down"


# --- DatabaseProbe ---

@pytest.mark.asyncio
async def test_database_probe_ok() -> None:
    fake_engine = AsyncMock()
    fake_conn = AsyncMock()
    fake_engine.connect.return_value.__aenter__.return_value = fake_conn
    fake_conn.execute = AsyncMock(return_value=None)

    probe = DatabaseProbe(fake_engine)
    result = await probe.probe()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_database_probe_down_on_connect_failure() -> None:
    fake_engine = AsyncMock()
    fake_engine.connect.side_effect = ConnectionError("db unreachable")

    probe = DatabaseProbe(fake_engine)
    result = await probe.probe()
    assert result.status == "down"
    assert "connect_failed" in (result.detail or "")


# --- ModelLookupProbe ---

@pytest.mark.asyncio
async def test_model_lookup_probe_ok_for_known_model() -> None:
    """litellm.utils.get_supported_openai_params returns a list for known models."""
    settings = AppSettings()
    settings.llm.default_model = "gpt-4o-mini"
    probe = ModelLookupProbe(settings)
    with patch(
        "litellm.utils.get_supported_openai_params",
        return_value=["max_tokens", "temperature"],
    ):
        result = await probe.probe()
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_model_lookup_probe_degraded_when_unknown() -> None:
    settings = AppSettings()
    probe = ModelLookupProbe(settings)
    with patch("litellm.utils.get_supported_openai_params", return_value=None):
        result = await probe.probe()
    assert result.status == "degraded"


@pytest.mark.asyncio
async def test_model_lookup_probe_down_on_lookup_error() -> None:
    settings = AppSettings()
    probe = ModelLookupProbe(settings)
    with patch(
        "litellm.utils.get_supported_openai_params",
        side_effect=RuntimeError("bad model"),
    ):
        result = await probe.probe()
    assert result.status == "down"
```

- [ ] **Step 4.7: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/health/test_probes.py -q
```

Expected: ImportError on `ai_core.health.probes`.

- [ ] **Step 4.8: Implement `src/ai_core/health/probes.py`**

```python
"""Concrete health probes shipped with the SDK."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import litellm.utils
from sqlalchemy import text

from ai_core.config.settings import AppSettings
from ai_core.health.interface import IHealthProbe, ProbeResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class SettingsProbe(IHealthProbe):
    """Always returns ``ok`` if settings loaded successfully."""

    component = "settings"

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    async def probe(self) -> ProbeResult:
        return ProbeResult(component=self.component, status="ok")


class OPAReachabilityProbe(IHealthProbe):
    """Sends ``GET <opa_url>/health`` to verify OPA is reachable."""

    component = "opa"

    def __init__(self, settings: AppSettings) -> None:
        self._url = str(settings.security.opa_url).rstrip("/") + "/health"
        self._timeout = settings.security.opa_request_timeout_seconds

    async def probe(self) -> ProbeResult:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._url)
            if response.status_code < 500:
                return ProbeResult(
                    component=self.component, status="ok",
                    detail=f"http_status={response.status_code}",
                )
            return ProbeResult(
                component=self.component, status="degraded",
                detail=f"http_status={response.status_code}",
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            return ProbeResult(
                component=self.component, status="down",
                detail=f"unreachable: {type(exc).__name__}",
            )
        except Exception as exc:  # noqa: BLE001 — probes never raise
            return ProbeResult(
                component=self.component, status="down",
                detail=f"error: {type(exc).__name__}",
            )


class DatabaseProbe(IHealthProbe):
    """Runs ``SELECT 1`` against the configured AsyncEngine."""

    component = "database"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def probe(self) -> ProbeResult:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return ProbeResult(component=self.component, status="ok")
        except Exception as exc:  # noqa: BLE001 — probes never raise
            return ProbeResult(
                component=self.component, status="down",
                detail=f"connect_failed: {type(exc).__name__}",
            )


class ModelLookupProbe(IHealthProbe):
    """Verifies ``litellm.utils.get_supported_openai_params`` resolves
    the configured default model."""

    component = "model_lookup"

    def __init__(self, settings: AppSettings) -> None:
        self._model = settings.llm.default_model

    async def probe(self) -> ProbeResult:
        try:
            params = await asyncio.to_thread(
                litellm.utils.get_supported_openai_params, self._model
            )
            if params is None:
                return ProbeResult(
                    component=self.component, status="degraded",
                    detail=f"model {self._model!r} not recognized by litellm",
                )
            return ProbeResult(
                component=self.component, status="ok",
                detail=f"model={self._model}",
            )
        except Exception as exc:  # noqa: BLE001 — probes never raise
            return ProbeResult(
                component=self.component, status="down",
                detail=f"lookup_error: {type(exc).__name__}",
            )


__all__ = ["DatabaseProbe", "ModelLookupProbe", "OPAReachabilityProbe", "SettingsProbe"]
```

Update `src/ai_core/health/__init__.py`:

```python
from ai_core.health.interface import HealthStatus, IHealthProbe, ProbeResult
from ai_core.health.probes import (
    DatabaseProbe,
    ModelLookupProbe,
    OPAReachabilityProbe,
    SettingsProbe,
)

__all__ = [
    "DatabaseProbe", "HealthStatus", "IHealthProbe",
    "ModelLookupProbe", "OPAReachabilityProbe", "ProbeResult",
    "SettingsProbe",
]
```

- [ ] **Step 4.9: Run tests to verify they pass**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/health/test_probes.py -v 2>&1 | tail -20
```

Expected: 10 passed.

### 4c — Add `HealthSettings` + DI binding

- [ ] **Step 4.10: Add `HealthSettings` to `src/ai_core/config/settings.py`**

```python
class HealthSettings(BaseSettings):
    """Health-probe configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    probe_timeout_seconds: float = Field(default=2.0, gt=0)


# in AppSettings:
health: HealthSettings = Field(default_factory=HealthSettings)
```

- [ ] **Step 4.11: Add `provide_health_probes` to `AgentModule`**

In `src/ai_core/di/module.py`, add:

```python
@singleton
@provider
def provide_health_probes(
    self,
    settings: AppSettings,
    engine: AsyncEngine,
) -> list[IHealthProbe]:
    """Default health-probe set. Override in a custom module to add probes."""
    from ai_core.health.probes import (  # noqa: PLC0415
        DatabaseProbe, ModelLookupProbe, OPAReachabilityProbe, SettingsProbe,
    )
    return [
        SettingsProbe(settings),
        OPAReachabilityProbe(settings),
        DatabaseProbe(engine),
        ModelLookupProbe(settings),
    ]
```

Add the import at the top of `module.py`:

```python
from ai_core.health import IHealthProbe
```

### 4d — Async `app.health` + `_HealthCheckRunner`

- [ ] **Step 4.12: Write failing tests for the new async `health()`**

In `tests/unit/app/test_runtime.py`, replace any test that calls `app.health` (sync) with `await app.health()`. Add new tests for the runner:

```python
@pytest.mark.asyncio
async def test_async_health_returns_health_snapshot(
    fake_observability, fake_policy_evaluator_factory
) -> None:
    app = AICoreApp(modules=[_override_module(fake_observability, fake_policy_evaluator_factory)])
    async with app:
        snap = await app.health()
        assert isinstance(snap, HealthSnapshot)
        assert "settings" in snap.components


@pytest.mark.asyncio
async def test_health_rolls_up_to_down_when_any_probe_down() -> None:
    """If any probe returns down, the rolled-up status is down."""
    from ai_core.health import IHealthProbe, ProbeResult
    from injector import Module, provider, singleton

    class _GoodProbe(IHealthProbe):
        component = "good"
        async def probe(self) -> ProbeResult:
            return ProbeResult(component=self.component, status="ok")

    class _BadProbe(IHealthProbe):
        component = "bad"
        async def probe(self) -> ProbeResult:
            return ProbeResult(component=self.component, status="down", detail="boom")

    class _Probes(Module):
        @singleton
        @provider
        def probes(self) -> list[IHealthProbe]:
            return [_GoodProbe(), _BadProbe()]

    app = AICoreApp(modules=[_Probes()])
    async with app:
        snap = await app.health()
    assert snap.status == "down"
    assert snap.components["good"] == "ok"
    assert snap.components["bad"] == "down"
    assert snap.component_details["bad"] == "boom"


@pytest.mark.asyncio
async def test_health_probe_timeout_marks_probe_down() -> None:
    """A probe that exceeds health.probe_timeout_seconds is marked down."""
    from ai_core.config.settings import AppSettings, HealthSettings
    from ai_core.health import IHealthProbe, ProbeResult
    from injector import Module, provider, singleton
    import asyncio

    class _SlowProbe(IHealthProbe):
        component = "slow"
        async def probe(self) -> ProbeResult:
            await asyncio.sleep(2.0)  # >> 0.05s timeout
            return ProbeResult(component=self.component, status="ok")

    class _Probes(Module):
        @singleton
        @provider
        def probes(self) -> list[IHealthProbe]:
            return [_SlowProbe()]

    settings = AppSettings()
    settings.health = HealthSettings(probe_timeout_seconds=0.05)

    app = AICoreApp(settings=settings, modules=[_Probes()])
    async with app:
        snap = await app.health()
    assert snap.components["slow"] == "down"
    assert "probe_timeout" in (snap.component_details["slow"] or "")
```

(Existing tests that asserted `snap = app.health` (sync) need to be updated to `snap = await app.health()`.)

- [ ] **Step 4.13: Run tests to verify they fail**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py -v 2>&1 | tail -15
```

Expected: failures because `app.health` is sync and `HealthSnapshot.component_details` doesn't exist yet.

- [ ] **Step 4.14: Update `HealthSnapshot` and `app.health` in `src/ai_core/app/runtime.py`**

Replace the `HealthSnapshot` definition:

```python
@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Coarse application health snapshot returned by :py:meth:`AICoreApp.health`."""

    status: HealthStatus
    components: dict[str, HealthStatus]
    component_details: dict[str, str | None]
    service_name: str
```

Add `HealthStatus` import from `ai_core.health`.

Replace the existing `health` property with an async method:

```python
    async def health(self) -> HealthSnapshot:
        """Run all configured health probes in parallel and return the snapshot."""
        if not self._entered or self._settings is None:
            return HealthSnapshot(
                status="down",
                components={},
                component_details={},
                service_name="",
            )
        from ai_core.health import IHealthProbe  # noqa: PLC0415
        runner = _HealthCheckRunner(
            self._container.get(list[IHealthProbe]),  # type: ignore[type-abstract]
            timeout_seconds=self._settings.health.probe_timeout_seconds,
        )
        results = await runner.run()
        components: dict[str, HealthStatus] = {r.component: r.status for r in results}
        details: dict[str, str | None] = {r.component: r.detail for r in results}
        if any(s == "down" for s in components.values()):
            roll_up: HealthStatus = "down"
        elif any(s == "degraded" for s in components.values()):
            roll_up = "degraded"
        else:
            roll_up = "ok"
        if self._closed:
            roll_up = "down"
        return HealthSnapshot(
            status=roll_up,
            components=components,
            component_details=details,
            service_name=self._settings.service_name,
        )
```

Add the runner class somewhere in the same file (above `AICoreApp` is fine):

```python
class _HealthCheckRunner:
    """Fans out IHealthProbe instances in parallel with a per-probe timeout."""

    def __init__(self, probes: Sequence[IHealthProbe], *,
                 timeout_seconds: float) -> None:
        self._probes = list(probes)
        self._timeout = timeout_seconds

    async def run(self) -> list[ProbeResult]:
        async def _run_one(probe: IHealthProbe) -> ProbeResult:
            try:
                return await asyncio.wait_for(probe.probe(), timeout=self._timeout)
            except asyncio.TimeoutError:
                return ProbeResult(
                    component=probe.component, status="down",
                    detail=f"probe_timeout_{self._timeout}s",
                )
            except Exception as exc:  # noqa: BLE001 — runner never raises
                return ProbeResult(
                    component=probe.component, status="down",
                    detail=f"probe_error: {type(exc).__name__}",
                )
        return await asyncio.gather(*(_run_one(p) for p in self._probes))
```

Add necessary imports at the top of `runtime.py`:

```python
import asyncio
from collections.abc import Sequence

from ai_core.health import HealthStatus, IHealthProbe, ProbeResult
```

- [ ] **Step 4.15: Update existing tests that called sync `app.health`**

In `tests/unit/app/test_runtime.py`, find any test that assigned `snap = app.health` (sync access). Convert to `snap = await app.health()` and ensure the test function is `async` with `@pytest.mark.asyncio`.

```bash
grep -n "app.health" tests/unit/app/test_runtime.py
```

Update each callsite. The Phase 2 test `test_health_components_populated_after_entry` and similar tests need updating.

- [ ] **Step 4.16: Run all app tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/app/test_runtime.py -v 2>&1 | tail -15
```

Expected: all passing.

### 4e — Top-level exports + meta-test

- [ ] **Step 4.17: Update `src/ai_core/__init__.py`**

```python
from ai_core.health import IHealthProbe, ProbeResult

__all__ = [
    ...,
    # Health
    "IHealthProbe",
    "ProbeResult",
]
```

- [ ] **Step 4.18: Write probe meta-test ("never raises")**

Create `tests/unit/health/test_never_raises.py`:

```python
"""Meta-test: every IHealthProbe implementation MUST return ProbeResult on failure."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from ai_core.config.settings import AppSettings
from ai_core.health.probes import (
    DatabaseProbe,
    ModelLookupProbe,
    OPAReachabilityProbe,
    SettingsProbe,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_settings_probe_never_raises() -> None:
    probe = SettingsProbe(AppSettings())
    result = await probe.probe()
    assert result is not None  # always returns something


@pytest.mark.asyncio
async def test_opa_probe_never_raises_on_unexpected_error() -> None:
    probe = OPAReachabilityProbe(AppSettings())
    with patch("httpx.AsyncClient", side_effect=RuntimeError("totally unexpected")):
        result = await probe.probe()
    assert result.status == "down"  # not raised


@pytest.mark.asyncio
async def test_database_probe_never_raises_on_unexpected_error() -> None:
    """A bad engine that raises on connect() must produce ProbeResult(status='down')."""
    class _BadEngine:
        def connect(self, *_: object) -> Any:
            raise RuntimeError("totally unexpected")

    probe = DatabaseProbe(_BadEngine())  # type: ignore[arg-type]
    result = await probe.probe()
    assert result.status == "down"


@pytest.mark.asyncio
async def test_model_lookup_probe_never_raises_on_unexpected_error() -> None:
    probe = ModelLookupProbe(AppSettings())
    with patch(
        "litellm.utils.get_supported_openai_params",
        side_effect=RuntimeError("library bug"),
    ):
        result = await probe.probe()
    assert result.status == "down"
```

- [ ] **Step 4.19: Run all health tests**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit/health -v 2>&1 | tail -20
```

Expected: 18+ passed (4 + 10 + 4).

- [ ] **Step 4.20: Run full unit + component suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 290+ passing.

- [ ] **Step 4.21: Lint + type-check**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src/ai_core/health src/ai_core/app/runtime.py src/ai_core/di/module.py src/ai_core/config/settings.py tests/unit/health
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src/ai_core/health src/ai_core/app/runtime.py tests/unit/health
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: clean; total ≤ 21.

- [ ] **Step 4.22: Commit**

```bash
git add src/ai_core/health/ \
        src/ai_core/__init__.py \
        src/ai_core/config/settings.py \
        src/ai_core/di/module.py \
        src/ai_core/app/runtime.py \
        tests/unit/app/test_runtime.py \
        tests/unit/health/
git commit -m "feat(health): real probes (settings/OPA/DB/model_lookup) + async app.health() with parallel runner"
```

---

## Task 5 — End-of-phase smoke gate

**Files:** none (verification only).

- [ ] **Step 5.1: Full test suite**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m pytest tests/unit tests/component -q 2>&1 | tail -10
```

Expected: 290+ pass; only pre-existing `respx`/`aiosqlite` collection errors remain.

- [ ] **Step 5.2: Lint the entire tree**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m ruff check src tests
```

Verify no NEW violations vs the post-Phase-2 state (`1855642`).

- [ ] **Step 5.3: Mypy strict — total error count must not regress**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -m mypy src 2>&1 | tail -1
```

Expected: `Found N errors` with `N <= 21`.

- [ ] **Step 5.4: Smoke against `my-eaap-app`**

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "import importlib; importlib.import_module('ai_core'); print('ai_core imported ok')"
```

Then verify the new Phase 3 surface:

```bash
/Users/admin-h26/EAAP/.venv/bin/python -c "
from ai_core import (
    AICoreApp, BaseAgent, tool,
    AgentState, new_agent_state, Tool, ToolSpec, HealthSnapshot,
    EAAPBaseException, ConfigurationError, DependencyResolutionError,
    SecretResolutionError, StorageError, PolicyDenialError, BudgetExceededError,
    LLMInvocationError, LLMTimeoutError, SchemaValidationError, ToolValidationError,
    ToolExecutionError, AgentRuntimeError, AgentRecursionLimitError, RegistryError,
    MCPTransportError,
    IAuditSink, AuditRecord, AuditEvent,
    IHealthProbe, ProbeResult,
)
print('Phase 3 canonical surface ok')
"
```

- [ ] **Step 5.5: Capture phase summary**

```bash
git log --oneline 69d1db0..HEAD
```

Expected: ~5 conventional-commit subjects (one per implementation task).

- [ ] **Step 5.6: Do NOT push automatically**

```bash
git status
echo "Suggested next step:"
echo "git push origin feat/phase-1-facade-tool-validation"
echo "gh pr create --title 'feat: Phase 3 — operability hardening + Phase 2 coherence fixes'"
```

---

## Out-of-scope reminders

For traceability, here is what is **deferred to Phase 4+** and must not creep into Phase 3:

- Anthropic prompt caching (`cache_control` headers).
- MCP connection pooling.
- Sentry / Datadog integrations (custom IAuditSink at user level).
- Per-tenant audit retention policies (backend-side concern).
- Concrete PII redactor implementations.
- `error_code` lookup registry / constants module.
- `tests/unit/security/test_opa.py` env fix (`respx` install).
- `eaap init` scaffold updates.

If a step starts pulling work from this list, stop and confirm scope with the user.
