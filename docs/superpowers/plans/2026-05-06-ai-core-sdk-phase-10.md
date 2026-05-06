# Phase 10 — Examples + DX Polish: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship four runnable example demos, an auto-generated settings reference with drift gate, and two carry-over Phase 9 review follow-ups — all without changing SDK source.

**Architecture:** Pure DX phase. New top-level `examples/` directory with four demo subdirectories (each self-contained with its own README), a `scripts/generate_settings_doc.py` introspecting `AppSettings`, a committed `docs/settings.md`, a drift test, and a smoke runner that exits zero on every demo. Two trivial test-file edits migrate stale `tests.conftest` imports onto the public `ai_core.testing` surface.

**Tech Stack:** Python 3.11+, Pydantic v2, FastAPI, FastMCP, OPA, pytest, ruff, `rich` (already a dep, used by math_tutor). Toolchain: `uv run pytest`, `uv run ruff check`.

---

## File map

| Path | New / Modified | Purpose |
| --- | --- | --- |
| `examples/agent_demo/run.py` | Renamed from `examples/math_tutor/run.py` + edited | Polished agent demo on modern API |
| `examples/agent_demo/README.md` | New | Demo-specific docs |
| `examples/math_tutor/` | Removed | Old location |
| `examples/mcp_server_demo/server.py` | New | FastMCP server with two tools |
| `examples/mcp_server_demo/run_client.py` | New | Connects via `PoolingMCPConnectionFactory`, lists tools, invokes one |
| `examples/mcp_server_demo/README.md` | New | Two-process model + roadmap note |
| `examples/fastapi_integration/app.py` | New | OPA-protected FastAPI app |
| `examples/fastapi_integration/README.md` | New | OPA setup + curl examples |
| `examples/testing_demo/conftest.py` | New | Activates `ai_core.testing.pytest_plugin` |
| `examples/testing_demo/src/my_agent.py` | New | Toy agent under test |
| `examples/testing_demo/tests/test_my_agent.py` | New | Three tests using public testing surface |
| `examples/testing_demo/README.md` | New | Tour of `ai_core.testing` |
| `scripts/generate_settings_doc.py` | New | Introspects `AppSettings` → markdown |
| `docs/settings.md` | New | Generator output, committed |
| `tests/unit/config/test_settings_doc_drift.py` | New | Drift gate |
| `tests/contract/test_audit_invariants.py:47` | Modified (M1) | Filter excludes `ai_core.testing.*` modules |
| `tests/unit/llm/test_litellm_client.py:29` | Modified (M2) | Import migration |
| `tests/unit/tools/test_invoker.py:28` | Modified (M2) | Import migration |
| `scripts/run_examples.sh` | New | Smoke gate |

---

## Task 1: Polish `examples/agent_demo/`

**Why:** The current `examples/math_tutor/run.py` predates Phase 9 — it inlines its own `ScriptedLLM` and `LLMResponse` plumbing. Users reading it learn the wrong API. We rename it, add a README, and swap the inline fakes for the public `ai_core.testing` surface so the demo *teaches* the modern API.

**Files:**
- Create: `examples/agent_demo/README.md`
- Move + Modify: `examples/math_tutor/run.py` → `examples/agent_demo/run.py`
- Delete: `examples/math_tutor/` (entire directory)

- [ ] **Step 1: Move the source file**

```bash
git mv examples/math_tutor/run.py examples/agent_demo/run.py
```

Expected: file is now at the new path; original directory is empty.

- [ ] **Step 2: Remove any leftover files in the old directory**

```bash
rmdir examples/math_tutor
```

If the directory has other files (`__init__.py`, etc.), `rmdir` will fail. In that case run `git rm -r examples/math_tutor`.

- [ ] **Step 3: Verify the file still runs as-is before refactoring**

Run: `uv run python examples/agent_demo/run.py`

Expected: prints two demo panels (single-turn and compaction). If it fails, the move broke something — fix before proceeding.

- [ ] **Step 4: Replace the inline ScriptedLLM imports with the public surface**

In `examples/agent_demo/run.py`, find the import block at lines ~50-54:

```python
from ai_core.agents import BaseAgent, TokenCounter  # noqa: E402
from ai_core.agents.memory import to_openai_message  # noqa: E402
from ai_core.config.settings import AppSettings  # noqa: E402
from ai_core.di import AgentModule, Container  # noqa: E402
from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMUsage  # noqa: E402
```

Replace with:

```python
from ai_core.agents import BaseAgent, TokenCounter  # noqa: E402
from ai_core.agents.memory import to_openai_message  # noqa: E402
from ai_core.config.settings import AppSettings  # noqa: E402
from ai_core.di import AgentModule, Container  # noqa: E402
from ai_core.di.interfaces import ILLMClient  # noqa: E402
from ai_core.testing import ScriptedLLM, make_llm_response  # noqa: E402
```

`LLMResponse` and `LLMUsage` are no longer needed at the top level.

- [ ] **Step 5: Delete the inline ScriptedLLM class**

In `examples/agent_demo/run.py`, find the inline `class ScriptedLLM(ILLMClient):` definition (lines ~74-110 — starts with the class declaration, ends just before `class _ForceCompactCounter:`). Delete the entire class.

- [ ] **Step 6: Update the two construction sites of ScriptedLLM**

In `demo_single_turn` (search for `llm = ScriptedLLM(`):

Old:
```python
    llm = ScriptedLLM(["Two plus two equals four."])
```

New:
```python
    llm = ScriptedLLM([make_llm_response("Two plus two equals four.")])
```

In `demo_compaction` (the other `llm = ScriptedLLM(` site):

Old:
```python
    llm = ScriptedLLM(
        [
            "User asked about TASK-42 earlier; deadline question is pending.",
            "The deadline is Friday at 5pm.",
        ]
    )
```

New:
```python
    llm = ScriptedLLM(
        [
            make_llm_response("User asked about TASK-42 earlier; deadline question is pending."),
            make_llm_response("The deadline is Friday at 5pm."),
        ]
    )
```

- [ ] **Step 7: Update the `_print_calls` helper to use the new ScriptedLLM call shape**

`ai_core.testing.ScriptedLLM` records `self.calls` as `list[dict[str, Any]]` whose entries include `messages`, `tenant_id`, `agent_id`, `tools`, `temperature`, etc. — keys differ slightly from the inline version. The inline `_print_calls` accesses `call['messages']`, `call['tenant_id']`, `call['agent_id']`, `call['temperature']`. The public ScriptedLLM uses the same keys, so no change is required. **Read the body of `_print_calls` to confirm**, then move on.

- [ ] **Step 8: Update the run-command in the docstring**

Find the docstring run instruction (line ~13-15):

Old:
```
Run from the repo root::

    PYTHONPATH=src python examples/math_tutor/run.py
```

New:
```
Run from the repo root::

    uv run python examples/agent_demo/run.py
```

- [ ] **Step 9: Verify the polished demo still runs end-to-end**

Run: `uv run python examples/agent_demo/run.py`

Expected: same two demo panels print successfully; no traceback.

- [ ] **Step 10: Write the README**

Create `examples/agent_demo/README.md`:

```markdown
# agent_demo — DI + LangGraph + memory compaction

A runnable demo of an end-to-end SDK agent. Drives a real `BaseAgent`
through the DI container and LangGraph, but uses
`ai_core.testing.ScriptedLLM` instead of a network LLM, so it runs
deterministically with no API keys.

## What this demonstrates

- `Container.build([AgentModule(settings=...), _Overrides()])` for DI.
- `BaseAgent` subclassing for a custom agent (`MathTutorAgent`).
- `ai_core.testing.ScriptedLLM` and `make_llm_response` for offline tests.
- Memory compaction triggered by a long conversation history.
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
2. **DEMO 2** — compaction: a five-message history triggers summarisation
   (one extra LLM call) before the agent responds.

## What to read next

- `src/ai_core/agents/base.py` — `BaseAgent` definition and LangGraph wiring.
- `src/ai_core/agents/memory.py` — `MemoryManager` and the compaction trigger.
- `src/ai_core/testing/__init__.py` — every fake importable from `ai_core.testing`.
```

- [ ] **Step 11: Run linters**

Run: `uv run ruff check examples/agent_demo/`

Expected: no errors. Fix any reported issues (most likely unused imports from Step 4).

- [ ] **Step 12: Commit**

```bash
git add examples/agent_demo/ examples/math_tutor/
git commit -m "feat(examples): rename math_tutor → agent_demo and adopt ai_core.testing API

- Move examples/math_tutor/ → examples/agent_demo/
- Remove inline ScriptedLLM; use ai_core.testing.ScriptedLLM + make_llm_response
- Add README explaining what the demo shows
- Update docstring run command (uv run)"
```

---

## Task 2: Build `examples/mcp_server_demo/`

**Why:** The SDK ships `MCPServerSpec` + `PoolingMCPConnectionFactory` (transport layer). There is **no** agent-side MCP-as-tool-source adapter today. This demo shows users the *actual* surface they can use now: stand up a FastMCP server, connect to it via the SDK's connection factory, list and invoke tools. The README documents the roadmap gap.

**Files:**
- Create: `examples/mcp_server_demo/server.py`
- Create: `examples/mcp_server_demo/run_client.py`
- Create: `examples/mcp_server_demo/README.md`

- [ ] **Step 1: Confirm FastMCP is available**

Run: `uv run python -c "import fastmcp; print(fastmcp.__version__)"`

Expected: prints a version. If `ImportError`, run `uv sync` (fastmcp is a core dependency in `pyproject.toml`; there is no `mcp` extra).

- [ ] **Step 2: Write `server.py`**

Create `examples/mcp_server_demo/server.py`:

```python
"""Minimal FastMCP server exposing two trivial tools over stdio.

Run this in one terminal:

    uv run python examples/mcp_server_demo/server.py

Then run the client (in a second terminal):

    uv run python examples/mcp_server_demo/run_client.py
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastmcp import FastMCP

mcp = FastMCP("ai-core-sdk-mcp-demo")


@mcp.tool()
def echo(text: str) -> str:
    """Return the input string verbatim."""
    return text


@mcp.tool()
def current_time() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 3: Write `run_client.py`**

Create `examples/mcp_server_demo/run_client.py`:

```python
"""Connect to the demo MCP server via the SDK's connection factory.

The SDK ships `PoolingMCPConnectionFactory` (transport layer), but does NOT
yet ship an agent-side adapter that registers a remote MCP server as a tool
source. This script demonstrates the surface that's available today:
open a connection, list tools, invoke one.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ai_core.mcp import MCPServerSpec, PoolingMCPConnectionFactory

console = Console()


def _server_spec() -> MCPServerSpec:
    """Spawn the demo server as a stdio subprocess."""
    server_path = Path(__file__).parent / "server.py"
    return MCPServerSpec(
        component_id="mcp-demo",
        transport="stdio",
        target=sys.executable,
        args=(str(server_path),),
    )


async def main() -> None:
    factory = PoolingMCPConnectionFactory(pool_enabled=False)
    spec = _server_spec()

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with factory.open(spec) as client:
                tools = await client.list_tools()
                console.print(f"[bold]Connected.[/bold] Server exposes {len(tools)} tool(s):")
                table = Table()
                table.add_column("name", style="cyan")
                table.add_column("description")
                for t in tools:
                    table.add_row(t.name, (t.description or "").splitlines()[0])
                console.print(table)

                result = await client.call_tool("echo", {"text": "hello from the SDK"})
                console.print(f"[bold green]echo result:[/bold green] {result}")
                return
        except Exception as exc:  # noqa: BLE001 — retry on any transport failure
            last_exc = exc
            await asyncio.sleep(0.2)
    raise SystemExit(
        f"Failed to connect to MCP server after 3 attempts: {last_exc!r}\n"
        f"Is `python examples/mcp_server_demo/server.py` running in another terminal?"
    )


if __name__ == "__main__":
    asyncio.run(main())
```

> **Note on the retry loop:** the spec calls for a small inline retry loop rather than tenacity, because the SDK's tenacity wiring lives inside `LiteLLMClient` and is not exposed for reuse.

- [ ] **Step 4: Write the README**

Create `examples/mcp_server_demo/README.md`:

```markdown
# mcp_server_demo — FastMCP server + SDK connection factory

Demonstrates the SDK's MCP transport surface today: a FastMCP server
plus a client that uses `PoolingMCPConnectionFactory` to connect, list
tools, and invoke one.

## What this demonstrates

- Standing up a FastMCP server with `@mcp.tool()` decorators.
- Using `MCPServerSpec` + `PoolingMCPConnectionFactory` from
  `ai_core.mcp` to open a connection.
- Listing and invoking tools through the FastMCP client.

## What's not (yet) shown

The SDK does **not** yet ship an agent-side adapter that registers a
remote MCP server as a `ToolInvoker` tool source. That integration is
on the roadmap. For now the connection factory is the public surface
you can use to bridge agents and MCP servers (you'd write the bridge
yourself).

## Prerequisites

```bash
uv sync
```

## Run

In one terminal:

```bash
uv run python examples/mcp_server_demo/server.py
```

In a second terminal:

```bash
uv run python examples/mcp_server_demo/run_client.py
```

The client prints the list of tools the server advertises and the
result of calling `echo`.

## Add your own tool

In `server.py`:

```python
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b
```

Restart the server, rerun the client, and the new tool will appear in
the table.

## What to read next

- `src/ai_core/mcp/transports.py` — `MCPServerSpec` and
  `PoolingMCPConnectionFactory`.
- FastMCP docs: https://github.com/jlowin/fastmcp
```

- [ ] **Step 5: Manual smoke test**

Open two terminals.

Terminal A: `uv run python examples/mcp_server_demo/server.py`

Expected: the server starts and waits silently on stdin (FastMCP stdio mode).

Terminal B: `uv run python examples/mcp_server_demo/run_client.py`

Wait — actually the client *spawns* the server itself via `MCPServerSpec(transport="stdio", target=sys.executable, args=(str(server_path),))`. So you only need to run the client; **no two-terminal coordination required**. (The README's two-terminal narrative is for users who want to inspect the server independently.)

Run: `uv run python examples/mcp_server_demo/run_client.py`

Expected: prints "Connected. Server exposes 2 tool(s)" with `echo` and `current_time` listed; prints `echo result: hello from the SDK`. Exit code 0.

- [ ] **Step 6: Run linters**

Run: `uv run ruff check examples/mcp_server_demo/`

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add examples/mcp_server_demo/
git commit -m "feat(examples): mcp_server_demo — FastMCP server + SDK connection factory

Demonstrates the actual MCP surface SDK ships today: MCPServerSpec +
PoolingMCPConnectionFactory. README explicitly notes that
agent-as-tool-source integration is on the roadmap."
```

---

## Task 3: Build `examples/fastapi_integration/`

**Why:** The README's FastAPI section points users at `OPAAuthorization`, but the SDK doesn't ship a runnable example. This demo wires the dependency to a single endpoint, demonstrates the allow/deny path with the existing `api.rego` policy, and is the canonical "I want to put OPA in front of an HTTP API" reference.

**Files:**
- Create: `examples/fastapi_integration/app.py`
- Create: `examples/fastapi_integration/README.md`

The demo reuses `src/ai_core/cli/templates/init/policies/api.rego` (which already encodes `allow if input.user == input.request.path_params.user_id`).

- [ ] **Step 1: Write `app.py`**

Create `examples/fastapi_integration/app.py`:

```python
"""FastAPI app demonstrating OPA-backed authorization via the SDK.

The endpoint enforces `data.eaap.api.allow` from the SDK's starter
api.rego policy: a JWT subject (`sub`) must equal the `user_id` path
parameter, otherwise OPA returns deny → FastAPI returns 403.

Run:

    uv run python examples/fastapi_integration/app.py

See README.md for OPA setup and curl examples.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI
from injector import Module, provider, singleton

from ai_core.config.settings import AppSettings, Environment
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import IPolicyEvaluator
from ai_core.di.module import ProductionSecurityModule
from ai_core.security import AuthorizedPrincipal, OPAAuthorization
from ai_core.security.jwt import HS256JWTVerifier, JWTVerifier


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(
            f"environment variable {name} is required — set it before running this demo. "
            f"See examples/fastapi_integration/README.md."
        )
    return value


def build_app() -> FastAPI:
    jwt_secret = _require_env("DEMO_JWT_SECRET")
    opa_url = os.environ.get("EAAP_SECURITY__OPA_URL", "http://localhost:8181")
    os.environ["EAAP_SECURITY__OPA_URL"] = opa_url
    os.environ.setdefault("EAAP_SECURITY__JWT_AUDIENCE", "ai-core-sdk-demo")
    os.environ.setdefault("EAAP_SECURITY__JWT_ISSUER", "ai-core-sdk-demo")

    settings = AppSettings(service_name="fastapi-demo", environment=Environment.LOCAL)

    class _DemoSecurityOverrides(Module):
        @singleton
        @provider
        def provide_jwt_verifier(self, settings_: AppSettings) -> JWTVerifier:
            return HS256JWTVerifier(jwt_secret, settings_)

    container = Container.build(
        [
            AgentModule(settings=settings),
            ProductionSecurityModule(),
            _DemoSecurityOverrides(),
        ]
    )
    authz = OPAAuthorization(container, decision_path="eaap/api/allow")

    app = FastAPI(title="ai-core-sdk FastAPI integration demo")

    @app.get("/users/{user_id}/profile")
    async def read_profile(
        user_id: str,
        principal: AuthorizedPrincipal = Depends(
            authz.requires(action="profile.read", resource="profile")
        ),
    ) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "subject": principal.subject,
            "email": principal.claims.get("email", "(unknown)"),
        }

    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=8000)
```

- [ ] **Step 2: Write the README**

Create `examples/fastapi_integration/README.md`:

````markdown
# fastapi_integration — OPA-protected FastAPI endpoint

A minimal FastAPI app with a single OPA-protected endpoint, using the
SDK's `OPAAuthorization` dependency and the SDK's starter `api.rego`
policy.

## What this demonstrates

- Wiring `OPAAuthorization` into a FastAPI handler via `Depends(...)`.
- Reusing the SDK's starter `api.rego` policy (no copy/paste).
- The full request flow: JWT bearer → `HS256JWTVerifier` → OPA → handler.

## What's *not* shown

The SDK does **not** ship a FastAPI request-scoped audit middleware
today. To see audit on the agent/tool path, run `examples/agent_demo/`.

## Prerequisites

```bash
uv sync
```

You'll need Docker to run OPA locally.

## Step 1 — Start OPA

In one terminal, from the repo root:

```bash
docker run --rm -p 8181:8181 \
    -v "$(pwd)/src/ai_core/cli/templates/init/policies:/policies" \
    openpolicyagent/opa:0.66.0 \
    run --server --addr 0.0.0.0:8181 /policies
```

OPA loads `api.rego` and `agent.rego` and serves on
`http://localhost:8181`.

## Step 2 — Start the FastAPI app

In a second terminal:

```bash
export DEMO_JWT_SECRET=dev-secret-change-me
uv run python examples/fastapi_integration/app.py
```

## Step 3 — Mint two JWTs and call the endpoint

In a third terminal:

```bash
# Allowed: token sub matches user_id path param.
ALLOW=$(uv run python -c "
import jwt, time
print(jwt.encode(
  {'sub': 'alice', 'iss': 'ai-core-sdk-demo', 'aud': 'ai-core-sdk-demo',
   'exp': int(time.time()) + 600},
  'dev-secret-change-me', algorithm='HS256'))")

curl -s -H "Authorization: Bearer $ALLOW" \
    http://localhost:8000/users/alice/profile
# {"user_id":"alice","subject":"alice","email":"(unknown)"}

# Denied: token sub does not match user_id path param.
curl -i -H "Authorization: Bearer $ALLOW" \
    http://localhost:8000/users/bob/profile
# HTTP/1.1 403 Forbidden
```

## How it works

1. FastAPI receives the request and resolves `Depends(authz.requires(...))`.
2. The dependency extracts the bearer token and verifies it via
   `HS256JWTVerifier`.
3. It builds the OPA input doc:
   `{user, action, resource, claims, request: {path_params, ...}}`.
4. It POSTs to OPA `/v1/data/eaap/api/allow`.
5. The rego policy evaluates `input.user == input.request.path_params.user_id`.
6. On `allow=true` the handler runs; on `false` the dependency raises 403.

## What to read next

- `src/ai_core/security/fastapi_dep.py` — `OPAAuthorization` source.
- `src/ai_core/cli/templates/init/policies/api.rego` — the demo policy.
- `src/ai_core/di/module.py` — `ProductionSecurityModule`.
````

- [ ] **Step 3: Manual smoke test**

You'll need three terminals.

Terminal A — start OPA:

```bash
docker run --rm -p 8181:8181 \
    -v "$(pwd)/src/ai_core/cli/templates/init/policies:/policies" \
    openpolicyagent/opa:0.66.0 \
    run --server --addr 0.0.0.0:8181 /policies
```

Terminal B — start the app:

```bash
export DEMO_JWT_SECRET=dev-secret-change-me
uv run python examples/fastapi_integration/app.py
```

Terminal C — exercise the endpoint with the curl commands from the README.

Expected: allow path returns 200 with the profile JSON; deny path returns 403.

- [ ] **Step 4: Run linters**

Run: `uv run ruff check examples/fastapi_integration/`

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add examples/fastapi_integration/
git commit -m "feat(examples): fastapi_integration — OPA-protected endpoint

Single GET /users/{user_id}/profile guarded by OPAAuthorization,
reusing the SDK's starter api.rego policy. README walks through OPA
setup and demonstrates allow/deny paths via curl."
```

---

## Task 4: Build `examples/testing_demo/`

**Why:** Phase 9 introduced the public `ai_core.testing` surface (5 fakes + ScriptedLLM + pytest plugin), but there's no canonical example. This demo *is* the documentation: a tiny project showing how to activate the plugin, drive an agent with `ScriptedLLM`, and assert on `FakeAuditSink` and `FakePolicyEvaluator`.

**Files:**
- Create: `examples/testing_demo/conftest.py`
- Create: `examples/testing_demo/src/__init__.py` (empty)
- Create: `examples/testing_demo/src/my_agent.py`
- Create: `examples/testing_demo/tests/__init__.py` (empty)
- Create: `examples/testing_demo/tests/test_my_agent.py`
- Create: `examples/testing_demo/pyproject.toml` (minimal, makes `src/` importable)
- Create: `examples/testing_demo/README.md`

- [ ] **Step 1: Write `conftest.py`**

Create `examples/testing_demo/conftest.py`:

```python
"""Activate the SDK's pytest plugin so its fixtures (fake_audit_sink,
scripted_llm_factory, etc.) are available to every test in this dir."""

pytest_plugins = ["ai_core.testing.pytest_plugin"]
```

- [ ] **Step 2: Write the toy agent**

Create `examples/testing_demo/src/__init__.py` (empty file):

```python
```

Create `examples/testing_demo/src/my_agent.py`:

```python
"""Toy agent under test in this demo.

This isn't a real BaseAgent — it's the smallest possible function that
exercises the SDK's testing surface end-to-end:

- It calls an `ILLMClient.complete()`.
- It records an audit event.
- It checks an `IPolicyEvaluator.evaluate()` decision.
"""
from __future__ import annotations

from ai_core.audit import AuditEvent, AuditRecord
from ai_core.di.interfaces import IAuditSink, ILLMClient, IPolicyEvaluator
from ai_core.exceptions import PolicyDenialError


async def answer_question(
    question: str,
    *,
    llm: ILLMClient,
    audit: IAuditSink,
    policy: IPolicyEvaluator,
    tenant_id: str = "demo-tenant",
) -> str:
    decision = await policy.evaluate(
        decision_path="demo.allow",
        input={"action": "answer", "question": question},
    )
    if not decision.allowed:
        raise PolicyDenialError(
            "policy denied", details={"reason": decision.reason or "no reason"}
        )

    response = await llm.complete(
        model=None,
        messages=[{"role": "user", "content": question}],
        tenant_id=tenant_id,
        agent_id="demo-agent",
    )

    await audit.record(
        AuditRecord.now(
            AuditEvent.TOOL_INVOCATION_COMPLETED,
            tool_name="answer_question",
            tool_version=1,
            agent_id="demo-agent",
            tenant_id=tenant_id,
            payload={"input": {"question": question}, "output": response.content},
        )
    )

    return response.content
```

The `AuditRecord.now(...)` classmethod is the canonical constructor (see `tests/contract/test_audit_invariants.py:54-60`); it auto-fills `timestamp`. The unused `datetime` imports above can be removed if no other code-path uses them. Read `src/ai_core/audit/__init__.py` if the signature differs.

- [ ] **Step 3: Write the test file (with all three test cases)**

Create `examples/testing_demo/tests/__init__.py` (empty):

```python
```

Create `examples/testing_demo/tests/test_my_agent.py`:

```python
"""Tests for the demo agent using ai_core.testing's public surface.

This file showcases the three big use-cases:

1. ScriptedLLM + make_llm_response — drive a deterministic LLM exchange.
2. FakeAuditSink — assert audit events were emitted.
3. FakePolicyEvaluator(default_allow=False) — assert deny path raises.
"""
from __future__ import annotations

import pytest

from ai_core.exceptions import PolicyDenialError
from ai_core.testing import (
    FakeAuditSink,
    FakePolicyEvaluator,
    make_llm_response,
)
from examples.testing_demo.src.my_agent import answer_question


@pytest.mark.asyncio
async def test_happy_path(scripted_llm_factory: object) -> None:
    """ScriptedLLM returns the canned response; agent records audit; returns content."""
    llm = scripted_llm_factory(  # type: ignore[operator]
        [make_llm_response("The answer is 42.")]
    )
    audit = FakeAuditSink()
    policy = FakePolicyEvaluator(default_allow=True)

    result = await answer_question(
        "What's the meaning of life?",
        llm=llm,
        audit=audit,
        policy=policy,
    )

    assert result == "The answer is 42."
    assert len(llm.calls) == 1
    assert llm.calls[0]["agent_id"] == "demo-agent"


@pytest.mark.asyncio
async def test_audit_records_one_event(scripted_llm_factory: object) -> None:
    """FakeAuditSink captures one TOOL_INVOCATION_COMPLETED record."""
    llm = scripted_llm_factory([make_llm_response("ok")])  # type: ignore[operator]
    audit = FakeAuditSink()
    policy = FakePolicyEvaluator(default_allow=True)

    await answer_question("ping", llm=llm, audit=audit, policy=policy)

    assert len(audit.records) == 1
    record = audit.records[0]
    assert record.tool_name == "answer_question"
    assert record.payload["input"]["question"] == "ping"


@pytest.mark.asyncio
async def test_deny_path_raises(scripted_llm_factory: object) -> None:
    """FakePolicyEvaluator with default_allow=False causes a PolicyDenialError."""
    llm = scripted_llm_factory([make_llm_response("never reached")])  # type: ignore[operator]
    audit = FakeAuditSink()
    policy = FakePolicyEvaluator(default_allow=False, reason="demo denial")

    with pytest.raises(PolicyDenialError):
        await answer_question("forbidden", llm=llm, audit=audit, policy=policy)

    # LLM was not called because policy denied first.
    assert len(llm.calls) == 0
    # No audit event recorded on the deny path.
    assert len(audit.records) == 0
```

- [ ] **Step 4: Make `src/` importable from tests**

Create `examples/testing_demo/pyproject.toml` (minimal — declares the package layout for pytest's rootdir/conftest discovery):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = [".", "src"]
```

This lets the tests use `from examples.testing_demo.src.my_agent import answer_question` when run from the demo directory.

> **Alternative:** if the import path proves clunky, change the test import to `from src.my_agent import answer_question` and adjust `pythonpath` accordingly. Either form must run cleanly under `uv run pytest examples/testing_demo/`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest examples/testing_demo/ -v`

Expected: 3 tests pass. If `AuditRecord` constructor doesn't match, fix `my_agent.py` to use the actual constructor signature (read `src/ai_core/audit/__init__.py`).

- [ ] **Step 6: Write the README**

Create `examples/testing_demo/README.md`:

````markdown
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
├── src/my_agent.py        # toy agent under test
└── tests/test_my_agent.py # three tests showing the surface
```

## Run

```bash
uv run pytest examples/testing_demo/ -v
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
````

- [ ] **Step 7: Run linters**

Run: `uv run ruff check examples/testing_demo/`

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add examples/testing_demo/
git commit -m "feat(examples): testing_demo — ai_core.testing surface in action

Three pytest tests that exercise ScriptedLLM, FakeAuditSink, and
FakePolicyEvaluator end-to-end against a toy agent. Demonstrates the
one-line pytest_plugins activation."
```

---

## Task 5: Settings auto-generator + Phase 9 follow-ups

**Why:** Bundling these because all are "DX hardening" with no runtime impact and they share one PR boundary in spirit.

This task has four sub-deliverables, each with its own commit:

- 5a. `scripts/generate_settings_doc.py` + initial `docs/settings.md`
- 5b. `tests/unit/config/test_settings_doc_drift.py` (drift gate)
- 5c. Phase 9 M1 — fix the contract test filter for `FakeAuditSink`
- 5d. Phase 9 M2 — migrate two `tests.conftest` imports

### 5a — Settings generator

**Files:**
- Create: `scripts/generate_settings_doc.py`
- Create: `docs/settings.md`

- [ ] **Step 1: Write the generator script**

Create `scripts/generate_settings_doc.py`:

```python
"""Generate docs/settings.md from the AppSettings Pydantic schema.

Walks AppSettings.model_fields, recursing into nested BaseSettings
groups (database, llm, audit, ...). Each group gets its own section
with a markdown table: field | type | default | env var | description.

Run:

    uv run python scripts/generate_settings_doc.py

The output is committed at docs/settings.md and gated by
tests/unit/config/test_settings_doc_drift.py.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from ai_core.config.settings import AppSettings

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "settings.md"
ENV_PREFIX = "EAAP_"
ENV_DELIM = "__"


def _format_type(annotation: Any) -> str:
    """Render a Pydantic field annotation as readable markdown."""
    if annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    args = ", ".join(_format_type(a) for a in get_args(annotation))
    return f"{getattr(origin, '__name__', str(origin))}[{args}]"


def _format_default(field: FieldInfo) -> str:
    if field.default_factory is not None:
        return "*(factory)*"
    default = field.default
    if default is None:
        return "`None`"
    if isinstance(default, str) and not default:
        return "`\"\"`"
    return f"`{default!r}`"


def _env_name(group_path: tuple[str, ...], field_name: str) -> str:
    """Build the EAAP_<GROUP>__<FIELD> env var name (uppercase)."""
    parts = (*group_path, field_name)
    return ENV_PREFIX + ENV_DELIM.join(p.upper() for p in parts)


def _render_table(model_cls: type[BaseSettings], group_path: tuple[str, ...]) -> str:
    rows: list[str] = []
    for name, field in model_cls.model_fields.items():
        if isinstance(field.default, type) and issubclass(field.default, BaseSettings):
            continue  # skip — rendered in its own section
        annotation = _format_type(field.annotation)
        default = _format_default(field)
        env = _env_name(group_path, name)
        description = (field.description or "").replace("|", "\\|").replace("\n", " ")
        rows.append(f"| `{name}` | `{annotation}` | {default} | `{env}` | {description} |")

    if not rows:
        return "_No direct fields — see nested groups below._\n"
    return (
        "| Field | Type | Default | Env var | Description |\n"
        "| --- | --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )


def _nested_groups(model_cls: type[BaseSettings]) -> list[tuple[str, type[BaseSettings]]]:
    """Discover BaseSettings-typed nested fields and return (name, type) pairs."""
    out: list[tuple[str, type[BaseSettings]]] = []
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseSettings):
                out.append((name, annotation))
        except TypeError:
            continue
    return out


def render() -> str:
    out = io.StringIO()
    out.write(
        "# Settings reference\n\n"
        "Auto-generated from `src/ai_core/config/settings.py`. "
        "Do not edit by hand — run `uv run python scripts/generate_settings_doc.py` "
        "to regenerate.\n\n"
        "Environment variable names use the prefix `EAAP_` and `__` as the "
        "nested-group delimiter (e.g. `EAAP_DATABASE__DSN`).\n\n"
    )

    out.write("## AppSettings\n\n")
    out.write(_render_table(AppSettings, ()))
    out.write("\n")

    for name, group_cls in _nested_groups(AppSettings):
        out.write(f"## {group_cls.__name__} (`AppSettings.{name}`)\n\n")
        if group_cls.__doc__:
            first_line = group_cls.__doc__.strip().splitlines()[0]
            out.write(f"{first_line}\n\n")
        out.write(_render_table(group_cls, (name,)))
        out.write("\n")

    return out.getvalue()


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    content = render()
    if "--check" in argv:
        # Print to stdout for CI diff use-cases.
        sys.stdout.write(content)
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    sys.stdout.write(f"wrote {OUTPUT.relative_to(REPO_ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the generator**

Run: `uv run python scripts/generate_settings_doc.py`

Expected: prints `wrote docs/settings.md`. The file now exists with one section per settings group.

- [ ] **Step 3: Eyeball the output**

Run: `head -40 docs/settings.md`

Expected: header + AppSettings table + first nested group section. If the formatting looks off (e.g., types render as `<class 'foo'>` instead of `foo`), revisit `_format_type`.

- [ ] **Step 4: Run linters on the new script**

Run: `uv run ruff check scripts/generate_settings_doc.py`

Expected: no errors.

- [ ] **Step 5: Commit 5a**

```bash
git add scripts/generate_settings_doc.py docs/settings.md
git commit -m "feat(scripts): generate_settings_doc.py + docs/settings.md

Introspects AppSettings.model_fields and emits a markdown reference
with one section per nested settings group. Output is committed so
GitHub readers can see it without running anything."
```

### 5b — Drift test

**Files:**
- Create: `tests/unit/config/test_settings_doc_drift.py`
- Create: `tests/unit/config/__init__.py` (if it does not already exist; check first)

- [ ] **Step 1: Check whether the config test directory exists**

Run: `ls tests/unit/config/ 2>/dev/null || echo "missing"`

If it prints `missing`, create the directory and an empty `__init__.py`:

```bash
mkdir -p tests/unit/config
touch tests/unit/config/__init__.py
```

- [ ] **Step 2: Write the drift test (this is a failing test first)**

Create `tests/unit/config/test_settings_doc_drift.py`:

```python
"""Drift gate: docs/settings.md must equal the generator's output.

Failure means somebody changed AppSettings without regenerating the
reference. Fix: run `uv run python scripts/generate_settings_doc.py`
and commit the regenerated file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_settings_doc import render

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED = REPO_ROOT / "docs" / "settings.md"


def test_settings_doc_is_not_stale() -> None:
    expected = render()
    actual = COMMITTED.read_text(encoding="utf-8")
    assert actual == expected, (
        "docs/settings.md is stale; run "
        "`uv run python scripts/generate_settings_doc.py` and commit."
    )
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/unit/config/test_settings_doc_drift.py -v`

Expected: PASS. (The committed `docs/settings.md` was just generated, so it should match exactly.)

If it fails: regenerate and re-run (`uv run python scripts/generate_settings_doc.py`). If it still fails, the generator output is non-deterministic — fix that before proceeding.

- [ ] **Step 4: Verify the drift detection works**

Manually corrupt `docs/settings.md`:

```bash
echo " " >> docs/settings.md
```

Run: `uv run pytest tests/unit/config/test_settings_doc_drift.py -v`

Expected: FAIL with the regeneration message.

Restore: `uv run python scripts/generate_settings_doc.py`

Re-run: `uv run pytest tests/unit/config/test_settings_doc_drift.py -v` — PASS.

- [ ] **Step 5: Make sure `scripts/` is importable as a package**

The drift test imports `from scripts.generate_settings_doc import render`. If pytest can't find `scripts`, add an empty `scripts/__init__.py`:

```bash
test -f scripts/__init__.py || touch scripts/__init__.py
```

- [ ] **Step 6: Commit 5b**

```bash
git add tests/unit/config/ scripts/__init__.py
git commit -m "test(config): drift gate for docs/settings.md

Diffs committed docs/settings.md against the generator's output.
Failure message instructs how to regenerate."
```

### 5c — Phase 9 M1: contract test filter

**Files:**
- Modify: `tests/contract/test_audit_invariants.py:47`

- [ ] **Step 1: Reproduce the issue**

Run: `uv run pytest tests/contract/test_audit_invariants.py -v 2>&1 | head -40`

Look at the parametrize ids printed near the top of the output. If `FakeAuditSink` appears in the list, the filter is currently incorrectly classifying it as a production sink. (If it doesn't appear, M1 may already be implicitly resolved — verify before continuing.)

- [ ] **Step 2: Edit the filter**

In `tests/contract/test_audit_invariants.py`, find lines 41-51:

```python
    # Filter out test-fixture sinks (those defined in tests/* modules).
    # FakeAuditSink in tests/conftest.py is a test helper, not a production sink.
    return sorted(
        (
            c for c in seen
            if not inspect.isabstract(c)
            and not c.__module__.startswith("tests.")
            and c.__module__ != "conftest"  # pytest-loaded conftest.py shows up as 'conftest'
        ),
        key=lambda c: c.__qualname__,
    )
```

Replace with:

```python
    # Filter out test-fixture sinks. Phase 9 moved the public fakes to
    # ai_core.testing, so we exclude that module path too.
    return sorted(
        (
            c for c in seen
            if not inspect.isabstract(c)
            and not c.__module__.startswith("tests.")
            and not c.__module__.startswith("ai_core.testing.")
            and c.__module__ != "conftest"  # pytest-loaded conftest.py shows up as 'conftest'
        ),
        key=lambda c: c.__qualname__,
    )
```

- [ ] **Step 3: Re-run the contract test**

Run: `uv run pytest tests/contract/test_audit_invariants.py -v`

Expected: all tests pass; `FakeAuditSink` is no longer in the parametrize ids.

- [ ] **Step 4: Commit 5c**

```bash
git add tests/contract/test_audit_invariants.py
git commit -m "fix(contract): exclude ai_core.testing fakes from production-sink scan

Phase 9 moved FakeAuditSink to ai_core.testing.fakes; the filter
previously excluded only tests.* modules, causing FakeAuditSink to be
misclassified as a production sink under test."
```

### 5d — Phase 9 M2: import migration

**Files:**
- Modify: `tests/unit/llm/test_litellm_client.py:29`
- Modify: `tests/unit/tools/test_invoker.py:28`

- [ ] **Step 1: Edit `test_litellm_client.py`**

In `tests/unit/llm/test_litellm_client.py`, line 29:

Old:
```python
from tests.conftest import FakeBudgetService
```

New:
```python
from ai_core.testing import FakeBudgetService
```

- [ ] **Step 2: Edit `test_invoker.py`**

In `tests/unit/tools/test_invoker.py`, line 28 (inside the `if TYPE_CHECKING:` block):

Old:
```python
    from tests.conftest import FakeAuditSink, FakeObservabilityProvider, FakePolicyEvaluator
```

New:
```python
    from ai_core.testing import FakeAuditSink, FakeObservabilityProvider, FakePolicyEvaluator
```

- [ ] **Step 3: Run the affected tests**

Run: `uv run pytest tests/unit/llm/test_litellm_client.py tests/unit/tools/test_invoker.py -v`

Expected: same pass/fail counts as before the change. Imports resolve from the public surface.

- [ ] **Step 4: Run the full suite to confirm nothing else broke**

Run: `uv run pytest`

Expected: all green.

- [ ] **Step 5: Commit 5d**

```bash
git add tests/unit/llm/test_litellm_client.py tests/unit/tools/test_invoker.py
git commit -m "refactor(tests): import fakes from ai_core.testing, not tests.conftest

Phase 9 carry-over (M2): tests should consume the public testing
surface, not the back-compat shim in tests.conftest."
```

---

## Task 6: Smoke gate

**Why:** Without a runner, the four examples will rot silently. This task adds a single shell script that exercises every demo with a 10s timeout each and asserts exit code 0. Docker-dependent demos (FastAPI) skip when Docker is absent. CI calls this script.

**Files:**
- Create: `scripts/run_examples.sh`

- [ ] **Step 1: Write the smoke gate**

Create `scripts/run_examples.sh` (executable):

```bash
#!/usr/bin/env bash
# Smoke gate for the examples/ directory.
# Exits non-zero if any non-skipped demo fails to complete in 10s.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failures=0
ran=0
skipped=0

run_demo() {
    local name="$1"; shift
    local cmd=("$@")
    echo "===== ${name} ====="
    if timeout 10 "${cmd[@]}"; then
        echo "  ✓ ${name}"
        ran=$((ran + 1))
    else
        local rc=$?
        echo "  ✗ ${name} (exit ${rc})"
        failures=$((failures + 1))
    fi
}

skip_demo() {
    local name="$1"; local reason="$2"
    echo "===== ${name} ====="
    echo "  ↷ skipped — ${reason}"
    skipped=$((skipped + 1))
}

# --- agent_demo: scripted LLM, no network, always runs.
run_demo agent_demo uv run python examples/agent_demo/run.py

# --- testing_demo: pytest tests; always runs.
run_demo testing_demo uv run pytest examples/testing_demo/ -q

# --- mcp_server_demo: needs fastmcp installed.
if uv run python -c "import fastmcp" 2>/dev/null; then
    run_demo mcp_server_demo uv run python examples/mcp_server_demo/run_client.py
else
    skip_demo mcp_server_demo "fastmcp not installed (run \`uv sync\`)"
fi

# --- fastapi_integration: full demo needs OPA running, but the smoke
# gate just verifies the app constructs (DI wires, deps resolve). That
# does not need Docker, so this branch always runs. Export vars so
# they propagate to the `uv run` invocation inside `run_demo`.
export DEMO_JWT_SECRET="dev-secret-smoke-test"
export PYTHONPATH="examples/fastapi_integration${PYTHONPATH:+:$PYTHONPATH}"
run_demo fastapi_integration_import \
    uv run python -c "from app import build_app; build_app()"

echo
echo "summary: ${ran} ran, ${skipped} skipped, ${failures} failed"
exit "${failures}"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/run_examples.sh
```

- [ ] **Step 3: Run the smoke gate locally**

Run: `scripts/run_examples.sh`

Expected: at least `agent_demo` and `testing_demo` run with `✓`; the others either run or skip cleanly. Exit code 0. Total runtime well under 60s.

- [ ] **Step 4: Run the full pytest suite one more time**

Run: `uv run pytest`

Expected: all green, including the new drift test.

- [ ] **Step 5: Run linters across everything Phase 10 touched**

Run: `uv run ruff check examples/ scripts/ tests/unit/config/ tests/contract/test_audit_invariants.py tests/unit/llm/test_litellm_client.py tests/unit/tools/test_invoker.py`

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_examples.sh
git commit -m "feat(scripts): run_examples.sh — smoke gate for examples/

Runs each demo with a 10s timeout. Skips fastapi_integration when
Docker is absent and mcp_server_demo when fastmcp is missing, but
fails loud on any non-skipped failure."
```

---

## Definition of done

- All six tasks committed with green linters and tests.
- `scripts/run_examples.sh` exits 0 in a fresh checkout with `uv sync` (Docker optional).
- `uv run pytest` is green; `uv run ruff check` is green.
- `docs/settings.md` exists, is committed, and matches the generator output.
- The two `tests.conftest` imports are gone from the test suite.
- The contract test filter no longer includes `FakeAuditSink`.
- No source files under `src/ai_core/` were modified.
