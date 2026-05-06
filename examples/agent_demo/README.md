# agent_demo — DI + LangGraph + memory compaction

A runnable demo of an end-to-end SDK agent. Drives a real `BaseAgent`
through the DI container and LangGraph, but uses
`ai_core.testing.ScriptedLLM` instead of a network LLM, so it runs
deterministically with no API keys.

## What this demonstrates

- `Container.build([AgentModule(settings=...), _Overrides()])` for DI.
- `BaseAgent` subclassing for a custom agent (`MathTutorAgent`).
- `ai_core.testing.ScriptedLLM` and `make_llm_response` for offline tests.
- Memory compaction forced by overriding `TokenCounter` via DI, demonstrating how the compaction path works.
- The full async `Container` lifecycle (`async with`).

## Prerequisites

```bash
uv sync
```

No API keys required — the LLM is scripted.

## Run

```bash
uv run python examples/agent_demo/run.py
```

You'll see two panels:

1. **DEMO 1** — single-turn: the agent answers "What is 2+2?".
2. **DEMO 2** — compaction: a stub `TokenCounter` (always reports above the threshold) forces a summarisation pass (one extra LLM call) before the agent responds, regardless of actual history length.

## What to read next

- `src/ai_core/agents/base.py` — `BaseAgent` definition and LangGraph wiring.
- `src/ai_core/agents/memory.py` — `MemoryManager` and the compaction trigger.
- `src/ai_core/testing/__init__.py` — every fake importable from `ai_core.testing`.
