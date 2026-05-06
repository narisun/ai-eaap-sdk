# testing_demo — the public `ai_core.testing` surface in action

Tests run with no real LLM, no DB, no OPA, and no Docker. Everything
is wired to the SDK's `ai_core.testing` fakes.

## What this demonstrates

- Activating the SDK's pytest plugin via a one-line `conftest.py`:
  ```python
  pytest_plugins = ["ai_core.testing.pytest_plugin"]
  ```
- Using `scripted_llm_factory` (a fixture from the plugin) to build a
  deterministic LLM.
- Asserting on `FakeAuditSink.records` to verify audit emission.
- Using `FakePolicyEvaluator(default_allow=False)` to drive deny-path
  tests.

## Layout

```
examples/testing_demo/
├── conftest.py            # pytest_plugins = ["ai_core.testing.pytest_plugin"]
├── pyproject.toml         # testpaths, pythonpath
├── src/
│   ├── __init__.py        # empty — marks src as a package
│   └── my_agent.py        # toy agent under test
└── tests/
    ├── __init__.py        # empty — marks tests as a package
    └── test_my_agent.py   # three tests showing the surface
```

> **Note on import path:** tests import the agent as `from src.my_agent import
> answer_question`. The demo's `pyproject.toml` adds `[".", "src"]` to
> `pythonpath`, so pytest resolves `src` relative to the demo's root
> (`examples/testing_demo/`).

## Run

```bash
.venv/bin/pytest examples/testing_demo/ -v
```

Expected: 3 passing tests in well under a second.

## What's importable

Everything in `ai_core.testing.__all__`:

| Name | Purpose |
| --- | --- |
| `FakeAuditSink` | Records `AuditRecord`s for assertion. |
| `FakeBudgetService` | Always-allow budget; records `check`/`record_usage` calls. |
| `FakeObservabilityProvider` | Records spans, events, LLM-usage. |
| `FakePolicyEvaluator` | Deterministic allow/deny + per-path overrides. |
| `FakeSecretManager` | In-memory secret resolver. |
| `ScriptedLLM` | Returns pre-built `LLMResponse`s in sequence. |
| `make_llm_response` | Builds an `LLMResponse` with sensible defaults. |

The pytest plugin registers fresh-per-test fixtures for each:
`fake_audit_sink`, `fake_observability`, `fake_budget`, plus factory
fixtures `fake_policy_evaluator_factory`,
`fake_secret_manager_factory`, and `scripted_llm_factory`.

## What to read next

- `src/ai_core/testing/__init__.py` — exact `__all__`.
- `src/ai_core/testing/pytest_plugin.py` — fixture definitions.
- `src/ai_core/testing/fakes.py` — fake class internals.
