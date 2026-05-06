# Phase 10 — Examples + DX Polish: Design

**Status:** Draft
**Date:** 2026-05-06
**Branch:** `feat/phase-10-examples-dx-polish`
**Predecessor:** Phase 9 (PR #8, merged at `9e34951`)

---

## 1. Architecture

Phase 10 ships three deliverables, in priority order:

1. **A runnable examples directory.** Four self-contained demos under `examples/`, each with its own README, that exercise the SDK's main surfaces end-to-end. Replaces the single undocumented `examples/math_tutor/` and fills gaps the top-level README points to but doesn't ship code for.
2. **An auto-generated settings reference.** A script that introspects the Pydantic settings schema and writes `docs/settings.md`, plus a drift test so the doc stays current.
3. **Phase 9 follow-ups.** Two minor cleanups carried over from the Phase 9 review (M1: stale filter for `FakeAuditSink` in a contract test; M2: two import lines that still reference the pre-extraction `tests.conftest` path).

This is a pure DX phase. **No SDK source changes.** If we find a bug while writing an example, we work around it in the example (with a comment explaining why) and file it for Phase 11.

### Module layout

```
examples/
├── agent_demo/              # renamed from math_tutor; LangGraph agent + tools
│   ├── README.md
│   └── run.py
├── mcp_server_demo/         # FastMCP server + SDK connection-factory client
│   ├── README.md
│   ├── run_client.py
│   └── server.py
├── fastapi_integration/     # OPA-protected FastAPI app
│   ├── README.md
│   └── app.py
└── testing_demo/            # public ai_core.testing surface in action
    ├── README.md
    ├── conftest.py
    ├── src/my_agent.py
    └── tests/test_my_agent.py

scripts/generate_settings_doc.py    # introspects AppSettings → docs/settings.md
docs/settings.md                    # committed generator output
tests/unit/config/test_settings_doc_drift.py    # drift gate
```

### Out of scope

Deferred explicitly to later phases:

- Phase 4 cost/latency closure (budget enforcement gaps, latency SLO wiring).
- Production robustness primitives (circuit breakers beyond what's already in place, dead-letter queues).
- Audit sink and redaction polish.
- A full API reference site (e.g., MkDocs/Sphinx).
- `Container.test_mode()` ergonomic helper.
- Sentry SDK v3 adoption (still not GA on PyPI).

---

## 2. Components & data flow

Six tasks, one component each. Each task produces self-contained changes that ship independently.

### Task 1 — `examples/agent_demo/` (rename + polish)

- **Files.** Rename `examples/math_tutor/run.py` → `examples/agent_demo/run.py`. New `examples/agent_demo/README.md`.
- **Polish.** Top-of-file docstring linking to the README. Prerequisite check on LLM env vars at startup. `if __name__ == "__main__"` guard if missing. Swap any direct `Container()` construction to the public `Container.build([...])` pattern shown in the top-level README.
- **README.** One-paragraph "what this shows" (LangGraph agent + tool calling + audit sink). Prerequisites (`uv sync`, `OPENAI_API_KEY` or `LITELLM_PROXY_URL`). Run command. Expected output snippet. "Read next" pointer to `src/ai_core/agents/`.
- **Data flow.** `uv run python examples/agent_demo/run.py` → script builds Container → instantiates agent → loops on stdin → agent calls calculator tool → audit sink prints to stdout.

### Task 2 — `examples/mcp_server_demo/` (new)

- **Files.** `server.py` (FastMCP server with two tools: `echo`, `current_time`), `run_client.py` (a script that uses the SDK's connection factory to connect, list tools, and invoke one), `README.md`.
- **`server.py`.** Minimal FastMCP setup with `@mcp.tool()` decorators for the two tools. Stdio transport (simplest for users to reason about; no port allocation).
- **`run_client.py`.** Uses `MCPServerSpec` + `PoolingMCPConnectionFactory` (the SDK's actual public surface today) to open a connection, list available tools, and invoke `echo`. **Not an agent.** The README explicitly notes that agent-as-tool-source integration is on the roadmap and points users to the connection-factory abstraction available now.
- **README.** Two-process model (server in one terminal, client in another). What the SDK ships today (transports + connection factory) vs. what's coming (agent-side tool-source registration). How to add your own tool.
- **Data flow.** Terminal A: `python server.py` (FastMCP listens on stdio). Terminal B: `python run_client.py` (factory opens connection, lists tools, invokes `echo`, prints result).

### Task 3 — `examples/fastapi_integration/` (new)

- **Files.** `app.py`, `README.md`. Reuses `src/ai_core/cli/templates/init/policies/api.rego` — does not ship a duplicate policy.
- **`app.py`.** FastAPI app with one OPA-protected endpoint, `GET /users/{user_id}/profile`, using the SDK's `OPAAuthorization` dependency. Handler returns a stub profile dict. **No audit middleware** — the SDK does not ship a FastAPI-level audit middleware today; the README points users to `agent_demo` to see audit on the agent/tool path.
- **README.** How to start OPA locally (uses the existing rego file via `docker run` snippet). Curl commands demonstrating allow path (JWT `sub` matches `user_id` path param → 200) and deny path (mismatch → 403). Pointer to the rego file. Note: the example focuses on authorization; audit lives on the agent path.
- **Data flow.** `curl` (with bearer token) → FastAPI → `OPAAuthorization.requires(action="profile.read")` dep extracts JWT, calls `IPolicyEvaluator.evaluate`, which POSTs to OPA `/v1/data/eaap/api/allow` → `allow=true` → handler runs.

### Task 4 — `examples/testing_demo/` (new)

- **Files.** `src/my_agent.py` (toy agent under test), `tests/test_my_agent.py`, `conftest.py`, `README.md`.
- **`conftest.py`.** Single line: `pytest_plugins = ["ai_core.testing.pytest_plugin"]`. This is the documented way to activate the SDK's plugin and the example exists primarily to demonstrate that line.
- **`tests/test_my_agent.py`.** Three tests:
  - (a) `ScriptedLLM` + `make_llm_response` to drive a deterministic two-turn conversation.
  - (b) `FakeAuditSink` to assert events were recorded (asserting on `sink.records`).
  - (c) `FakePolicyEvaluator(default_allow=False)` to assert a tool call is denied and the expected `PolicyDenialError` is raised.
- **README.** Explains the public testing surface; points to `ai_core.testing.__all__` so users know what's importable.
- **Data flow.** `pytest examples/testing_demo/` → conftest activates plugin → fixtures wire fakes → tests run with no real LLM/DB/OPA.

### Task 5 — Settings auto-generator + Phase 9 follow-ups (bundled)

These are bundled because all three are "DX hardening" with no runtime impact.

- **New: `scripts/generate_settings_doc.py`.** Introspects `AppSettings` via Pydantic v2's `model_fields` walk. Emits `docs/settings.md` with one section per nested settings group (DatabaseSettings, VectorDBSettings, StorageSettings, LLMSettings, BudgetSettings, ObservabilitySettings, SecuritySettings, AgentSettings, AuditSettings, HealthSettings, MCPSettings, AppSettings). Each section is a markdown table: `field | type | default | env var | description`.
- **New: `docs/settings.md`.** Committed generator output. Users reading the repo on GitHub see it without running anything.
- **New: `tests/unit/config/test_settings_doc_drift.py`.** Invokes the generator into a tempfile and diffs against the committed `docs/settings.md`. Failure message: a single line instructing `uv run python scripts/generate_settings_doc.py`.
- **Phase 9 M1 fix.** `tests/contract/test_audit_invariants.py` — the discovery filter excludes test fakes by checking `__module__.startswith("tests.")`, but Phase 9 moved `FakeAuditSink` to `ai_core.testing.fakes`. The filter now incorrectly *includes* `FakeAuditSink` as a "production sink" under test. Fix: extend the exclusion to also match `__module__.startswith("ai_core.testing.")`.
- **Phase 9 M2 fix.** Migrate two import lines off the pre-extraction path:
  - `tests/unit/llm/test_litellm_client.py:29` — `from tests.conftest import Fake*` → `from ai_core.testing import Fake*`.
  - `tests/unit/tools/test_invoker.py:28` — same migration.
- **Data flow.** Developer adds a settings field → CI runs drift test → fails → developer regenerates → commits the regenerated doc → clean.

### Task 6 — Smoke gate

- **No new code paths.** Verifies all five preceding tasks land working.
- **`make examples` target** (or `scripts/run_examples.sh`) exercises each demo with a 10-second timeout and asserts exit code 0.
- **Skip-not-fail** when a demo's prerequisites are missing: `mcp_server_demo` skips when FastMCP isn't installed. `fastapi_integration` runs an import-and-construct smoke check (no Docker needed); the full request flow is exercised manually per the README.
- **Drift test passes; full `uv run pytest` passes** including the migrated Phase 9 imports.

---

## 3. Error handling, testing, constraints

### Error handling

- **Missing prerequisites in examples.** Every `run.py` / `app.py` / `server.py` checks for required env vars at startup and exits with a clear "set $X to run this demo, see README" message. No silent failures, no stack traces for missing config.
- **MCP demo two-process coordination.** `run_client.py` retries the MCP connection 3× with 200ms backoff before failing with a "is `server.py` running in another terminal?" hint. Uses a small inline retry loop (the SDK's tenacity wiring is internal to `LiteLLMClient` and not exposed for reuse).
- **FastAPI demo OPA unreachable.** The SDK's `OPAAuthorization` already raises a typed error; the demo lets it surface so users see the real diagnostic. README documents the failure mode.
- **Settings generator.** Fails loud on Pydantic schema introspection errors — no fallback "best effort" output. Drift test failure message is a single line.
- **No new error classes.** Examples reuse the existing `ai_core.errors.ErrorCode` registry where they need to raise.

### Testing strategy

- **Examples are not unit-tested.** Their correctness gate is Task 6's smoke runner (run-to-exit-zero). Rationale: testing the test-doubles in `testing_demo` via more test-doubles is circular; the *real run* is what we want green.
- **`testing_demo/tests/`** are real tests pytest collects when run from that subdirectory; CI runs them via the smoke gate.
- **Drift test** is a normal unit test, runs in the standard `uv run pytest` flow.
- **Phase 9 M1** verified by running the contract suite — after the filter fix, `test_audit_invariants` should not pick up `FakeAuditSink` as a production sink (currently misclassifying it because the filter doesn't cover the new module path).
- **Phase 9 M2** verified by `uv run pytest tests/unit/llm/test_litellm_client.py tests/unit/tools/test_invoker.py` after the import migration.
- **No new fixtures, no new conftests** outside `examples/testing_demo/conftest.py`.

### Constraints

- **No SDK source changes.** Phase 10 is pure DX: examples, scripts, docs, and two test-file edits. Bugs found during example writing are filed for Phase 11; the example documents a workaround in a code comment with the *why*.
- **No new runtime dependencies.** Examples must run on the existing `pyproject.toml` extras. `scripts/generate_settings_doc.py` uses only stdlib + Pydantic v2 (already a dep).
- **Examples pinned to the modern public API.** No `Container()` direct instantiation, no private imports, no `tests.conftest` references — every example uses what `__all__` exports.
- **`docs/settings.md` is committed.** Generator output is a regular file in the repo; CI verifies it's not stale.
- **Smoke gate budget: 60s total wall-time.** Each example gets 10s; demos with missing prerequisites (e.g. FastMCP not installed) skip rather than fail.
