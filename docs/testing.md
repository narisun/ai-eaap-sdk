# Testing your agent with `ai_core.testing`

The `ai_core.testing` subpackage ships test helpers — fakes for the SDK's
core protocols, a `ScriptedLLM` for canned LLM responses, and a pytest
plugin that makes them all available as fixtures.

## Install

```bash
pip install ai-eaap-sdk[testing]
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
- `fake_secret_manager_factory` — call with a `{(backend, name): value}` dict
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

## Recipe — asserting on secret lookups

The `FakeSecretManager` is keyed by `(backend, name)` tuples matching the
`SecretRef` fields on the real implementations:

```python
def test_agent_reads_api_key(fake_secret_manager_factory):
    secrets = fake_secret_manager_factory(
        {("env", "OPENAI_API_KEY"): "test-key-xyz"}
    )
    # wire secrets into your container, then assert the agent used it
```

## Recipe — observing LLM usage accounting

```python
@pytest.mark.asyncio
async def test_usage_recorded(
    scripted_llm_factory,
    fake_observability,
):
    llm = scripted_llm_factory(
        [make_llm_response("ok", prompt_tokens=5, completion_tokens=10)]
    )
    # run your service with llm + fake_observability wired in ...
    assert len(fake_observability.usage) == 1
    assert fake_observability.usage[0]["prompt_tokens"] == 5
```

## What's NOT in `ai_core.testing`

- `Container.test_mode()` — explicit per-fixture wiring is the v1 pattern;
  drop in a Phase 10 follow-up if real consumer projects ask for the
  builder.
- `RaisingLLM` / `SlowLLM` — handle these in your test files via inline
  classes if needed; the canned-response common case is `ScriptedLLM`.
- Snapshot / replay — agent run recording is not yet shipped.

## Reference

### `make_llm_response`

```python
from ai_core.testing import make_llm_response

r = make_llm_response(
    "assistant reply text",      # positional, default ""
    tool_calls=[...],            # default []
    finish_reason="stop",        # default "stop"
    prompt_tokens=10,            # default 10
    completion_tokens=20,        # default 20
    model="test-model",          # default "test-model"
)
```

`total_tokens` is computed as `prompt_tokens + completion_tokens`.

### `ScriptedLLM`

```python
from ai_core.testing import ScriptedLLM, make_llm_response

llm = ScriptedLLM(
    [make_llm_response("first"), make_llm_response("second")],
    repeat_last=False,   # raise IndexError after exhaustion (default)
)
# After all calls: llm.calls is a list of dicts with every argument passed.
```

Set `repeat_last=True` when you need an LLM that always returns the same
response regardless of how many times it is called.

## More patterns

The SDK's own `tests/contract/` directory shows how the public surface
itself is pinned via parametrized tests over `__subclasses__()`. The
`tests/integration/` directory shows how to use Testcontainers with the
real Postgres + OPA backends. Both are worth reading if you need
patterns beyond the basic fixtures.
