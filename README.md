# ai-core-sdk

The foundational SDK for the **Enterprise Agentic AI Platform (EAAP)**.

> **Do NOT run `pip install ai-core-sdk`.** That distribution name on PyPI
> belongs to SAP's AI Core client SDK — a completely unrelated package. Until
> this project is published under a non-conflicting name, install **from
> source** (instructions below). This SDK has not been published to PyPI.

`ai-core-sdk` is a typed, async-first Python toolkit for building agentic AI
applications that have to clear an enterprise bar — observable, policy-aware,
budget-aware, schema-versioned, and testable in isolation. It bundles
LangGraph orchestration, LiteLLM proxying, FastMCP integration, OpenTelemetry
+ LangFuse observability, OPA-based authorization, and a Typer-based CLI for
project scaffolding under a single dependency-injection seam.

The SDK is opinionated: all of these concerns are first-class and wired by
default, but every concrete implementation is bound to an ABC and trivially
overridable through DI.

---

## Core ideas

| Idea | What it means in this codebase |
|---|---|
| **The DI container is the only seam** | Concrete classes are bound to interfaces inside `AgentModule`. Application code depends on the ABCs (`ILLMClient`, `IStorageProvider`, `IPolicyEvaluator`, …) and resolves them through `Container.get(...)`. There are no module-level singletons and no `os.environ` reads scattered through the code. |
| **Async-first** | Every I/O surface — DB, LLM, OPA, MCP, observability — is `async`. Sync paths are explicit and rare; the LangGraph adapter, for example, deliberately leaves sync methods unimplemented and documents it. |
| **Interfaces over implementations** | Every external integration sits behind a small ABC defined in `ai_core.di.interfaces`. Hosts can swap an implementation (real S3 → in-memory fake, OPA → JSON-file evaluator) without touching calling code. |
| **Settings are typed and centralised** | One `AppSettings` Pydantic model aggregates nested groups (`database`, `llm`, `budget`, `observability`, `security`, `agent`, …) populated from `EAAP_*` environment variables with `__` as the nested delimiter. Secrets are resolved through `ISecretManager`, never read directly from `os.environ`. |
| **Observability is non-optional** | Every LLM call records prompt/completion tokens, latency, and cost; every agent invocation opens an OTel span and a LangFuse trace. The default provider degrades gracefully when no collector or LangFuse keys are configured, so the SDK is safe to import in tests and local dev. |
| **Fail-closed security defaults** | OPA evaluator defaults to `fail_closed=True` (network or HTTP errors → deny). The agent guardrail blocks all tool calls on policy-evaluator errors and emits a self-explanatory denial message the agent can re-plan from. |
| **Versioned contracts at every Agent ↔ Tool boundary** | `SchemaRegistry` keys input/output Pydantic models by `(name, version)`. The `validate_tool` decorator enforces both directions at runtime; `eaap schema export` emits JSON Schemas for cross-language clients and docs. |
| **Memory compaction preserves Essential Entities** | When prompt tokens cross the configured threshold, a compaction node summarises the conversation into a single system message + the trailing user/assistant pair, *while preserving every key listed under `agent.essential_entity_keys`* and any host-defined entity keys already present in state. |
| **Budgeting is mandatory** | Every LLM call passes through `IBudgetService.check()` *before* spending money and `IBudgetService.record_usage()` after. The default `InMemoryBudgetService` enforces per-`(tenant, agent)` daily token + USD limits; production hosts swap in Redis/DB backends through DI. |
| **Custom exception hierarchy** | All SDK errors derive from `EAAPBaseException` and carry a structured `details` mapping that flows into spans + LangFuse metadata. Operators correlate failures by domain (`PolicyDenialError`, `BudgetExceededError`, `CheckpointError`, …) without parsing strings. |

---

## Critical principles

These rules are enforced by code review and reflect every design decision in
the SDK:

1. **No globals.** Every dependency arrives via constructor injection. Tests
   build a fresh `Container` per case; singletons are scoped *to a container*,
   never to the process.
2. **Async I/O.** No blocking calls in async paths. SQLAlchemy uses
   `AsyncEngine`; HTTP uses `httpx.AsyncClient`; LLM uses `litellm.acompletion`.
3. **Validate at boundaries, trust internally.** Pydantic v2 models validate
   incoming JSON at every Agent ↔ MCP / API ↔ user boundary. Internal helpers
   trust their typed inputs.
4. **Errors carry structured context.** Every `EAAPBaseException` subclass
   accepts a `details: dict` that is logged, attached to OTel attributes, and
   sent to LangFuse — never just a string message.
5. **Retry only on transient failures.** Tenacity retries 429 / 5xx / connect
   / timeout. Auth, validation, and budget errors surface immediately.
6. **Document every public method.** Google-style docstrings cover Args /
   Returns / Raises. Comments inside method bodies are reserved for *why*,
   never *what*.
7. **Concrete bindings live in `AgentModule`.** Adding a new external
   integration means: define its ABC in `di/interfaces.py`, write the concrete,
   bind it in `module.py`. No exceptions.
8. **Tests can swap any binding.** `Container.override(*modules)` returns a
   *new* container with extra bindings layered on top — the original is
   untouched. Last binding wins.
9. **Production-grade defaults that degrade gracefully.** `RealObservabilityProvider`
   without OTel/LangFuse credentials → in-process spans only. `OPAPolicyEvaluator`
   with `fail_closed=False` → returns `allowed=True` with a `reason` so audit logs
   show why. `EnvSecretManager` with a missing var → `SecretResolutionError` with
   the var name in `details`.
10. **The DI seam is also the test seam.** Every fake the test suite needs is
    a `Module` subclass, not a monkey-patch.

---

## Architecture map

```
src/ai_core/
├── config/           AppSettings + ISecretManager (Pydantic v2)
├── di/               Container + AgentModule + IObservabilityProvider, ILLMClient, ...
├── exceptions.py     EAAPBaseException + domain subclasses
├── observability/    NoOp + Real (OTel + LangFuse)
├── security/         OPAPolicyEvaluator, JWT verifiers, FastAPI dep, GuardrailNode
├── schema/           SchemaRegistry + validate_tool decorator + JSON-Schema export
├── persistence/      Async SQLAlchemy engine + PostgresCheckpointSaver + LangGraphCheckpointSaver
├── llm/              LiteLLMClient (Tenacity + budgeting + observability) + InMemoryBudgetService
├── agents/           BaseAgent + AgentState + MemoryManager + compaction
├── mcp/              ComponentRegistry + FastMCP transport handlers
└── cli/              `eaap` Typer app + Jinja2 templates for init / generate / schema export
```

---

## Quick start

### Install

> **Heads-up — name collision on PyPI.** The PyPI distribution
> `ai-core-sdk` is currently SAP's AI Core client (it pulls in
> `ai-api-client-sdk`, `pyhumps`, `aenum`). Running
> `pip install ai-core-sdk` will install *that* package, not this one,
> and the `eaap` command will not be on your PATH. Always install from
> source until this project ships under a non-conflicting distribution
> name (likely `eaap-core-sdk`).

Clone (or otherwise obtain the source), then install in editable mode
into your project's virtual environment:

```bash
# From the directory that contains the ai-core-sdk/ source tree:

# Optional but recommended: a fresh venv (Python 3.11+).
python3 -m venv .venv
source .venv/bin/activate

# If you previously ran `pip install ai-core-sdk` and got SAP's package,
# remove it first so the editable install can take its place cleanly:
pip uninstall -y ai-core-sdk ai-api-client-sdk

# Editable install. The `eaap` CLI lands on PATH; source changes are
# picked up immediately without re-installing.
pip install -e ai-core-sdk

# Verify
eaap --help
```

For development on the SDK itself (running the test suite, ruff, mypy):

```bash
pip install -e "ai-core-sdk[dev]"
pytest -q                     # 148 tests
```

### Scaffold a project

```bash
eaap init my-eaap-app
cd my-eaap-app
cp .env.example .env
docker compose up -d            # Postgres + OPA + OTel collector + Jaeger

# The scaffolded pyproject.toml intentionally omits the ai-core-sdk
# dependency (PyPI name collision — see above). Install it from source
# alongside the generated app:
pip install -e /path/to/ai-core-sdk
pip install -e ".[dev]"

python -m my_eaap_app.main      # FastAPI app on :8000
```

`eaap init` writes a runnable project tree:

- `pyproject.toml`
- `docker-compose.yaml` (Postgres 16, OPA, OTel collector, Jaeger UI on `:16686`)
- `otel-collector-config.yaml`
- `policies/eaap.rego` (sample API + tool-call policies)
- `.env.example`, `.gitignore`, `README.md`
- `src/<package>/main.py` — FastAPI app that wires `OPAAuthorization` against a real route

### Configure

Settings come from `EAAP_*` environment variables with `__` as the nested
delimiter:

```bash
EAAP_ENVIRONMENT=staging
EAAP_DATABASE__DSN=postgresql+asyncpg://user:pass@host/db
EAAP_LLM__DEFAULT_MODEL=bedrock/anthropic.claude-sonnet-4-6
EAAP_OBSERVABILITY__OTEL_ENDPOINT=http://otel-collector:4317
EAAP_OBSERVABILITY__LANGFUSE_PUBLIC_KEY=pk-...
EAAP_OBSERVABILITY__LANGFUSE_SECRET_KEY=sk-...
EAAP_SECURITY__OPA_URL=http://opa:8181
EAAP_SECURITY__FAIL_CLOSED=true
EAAP_BUDGET__DEFAULT_DAILY_TOKEN_LIMIT=2000000
```

Every nested settings group lives at the same depth, so adding a new one
follows the same pattern.

---

## Building an Agent

### 1. Generate the boilerplate

```bash
eaap generate agent support_triage
```

This writes `src/agents/support_triage.py` and `tests/unit/agents/test_support_triage.py`.

### 2. Subclass `BaseAgent`

```python
# src/agents/support_triage.py
from collections.abc import Mapping, Sequence
from typing import Any

from ai_core.agents import BaseAgent


class SupportTriageAgent(BaseAgent):
    agent_id: str = "support-triage"

    def system_prompt(self) -> str:
        return (
            "You are a support triage agent. Classify incoming tickets, "
            "extract structured fields, and call the registered tools to "
            "create or update tickets. Stop as soon as the task is done."
        )

    def tools(self) -> Sequence[Mapping[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Open a new support ticket.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title":    {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "high"]},
                        },
                        "required": ["title"],
                    },
                },
            },
        ]
```

`BaseAgent` ships with two pre-wired LangGraph nodes:

- `agent` — calls the LLM with `system_prompt() + state.messages`, records
  usage, returns the assistant message (with any `tool_calls`).
- `compact` — runs ahead of `agent` whenever
  `MemoryManager.should_compact()` returns `True`. Summarises history,
  preserves Essential Entities, returns a fresh state with a single summary
  system message + the trailing user/assistant pair.

Override `extend_graph(graph)` to add tool execution nodes, the
`GuardrailNode`, or any host-specific logic before the graph is compiled.

### 3. Wire it through DI

```python
from ai_core.di import Container, AgentModule
from agents.support_triage import SupportTriageAgent

container = Container.build([AgentModule()])
agent = container.get(SupportTriageAgent)         # all deps auto-wired
```

`SupportTriageAgent`'s constructor receives `AppSettings`, `ILLMClient`,
`MemoryManager`, and `IObservabilityProvider` automatically because each one
is bound in `AgentModule` and the subclass inherits `@inject` from
`BaseAgent`.

### 4. Invoke

```python
final_state = await agent.ainvoke(
    messages=[{"role": "user", "content": "Production database is down."}],
    essential={"user_id": "u-1", "task_id": "T-42"},
    tenant_id="acme",
    thread_id="conv-7",        # enables LangGraph checkpointing
)
print(final_state["messages"][-1]["content"])
```

The `essential` keys land in `state.essential_entities` and survive every
compaction. `tenant_id` flows into budget enforcement and tracing.

### 5. Add a Guardrail

```python
from ai_core.security import GuardrailNode

class SupportTriageAgent(BaseAgent):
    ...
    def extend_graph(self, graph) -> None:
        guard = GuardrailNode(self._policy, agent_id=self.agent_id)
        graph.add_node("guardrail", guard.run)
        # Edge wiring depends on your tool execution node — see the
        # `extend_graph` docs for the canonical pattern.
```

The guardrail evaluates every tool call in the latest assistant message
against OPA at `eaap/agent/tool_call/allow`. Allowed calls pass through
unchanged; denied calls produce a `system` message explaining the
denial — the agent re-plans on the next turn.

### 6. Persist conversations across runs

```python
from ai_core.persistence import LangGraphCheckpointSaver

saver = container.get(LangGraphCheckpointSaver)   # async-only
graph = agent.compile(checkpointer=saver)
```

`LangGraphCheckpointSaver` implements LangGraph's `BaseCheckpointSaver` over
the same Postgres database the SDK already manages.

---

## Building an MCP Server

### 1. Generate

```bash
eaap generate mcp ticketing
```

Writes `src/mcp_servers/ticketing.py` (FastMCP server with one example tool)
and `tests/unit/mcp_servers/test_ticketing.py`.

### 2. Define Pydantic contracts

```python
from pydantic import BaseModel, Field
from fastmcp import FastMCP

server = FastMCP("ticketing")


class CreateTicketIn(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = "low"


class CreateTicketOut(BaseModel):
    ticket_id: str
    href: str


@server.tool()
async def create_ticket(payload: CreateTicketIn) -> CreateTicketOut:
    ...
```

### 3. Register the contracts (optional but recommended)

```python
from ai_core.schema import SchemaRegistry

def register_schemas(registry: SchemaRegistry) -> None:
    registry.register(
        "create_ticket", 1,
        input_schema=CreateTicketIn,
        output_schema=CreateTicketOut,
        description="Open a support ticket.",
    )
```

Now the same models can power runtime validation **and** be exported as
JSON Schema for documentation or cross-language clients:

```bash
eaap schema export --module-path src/contracts.py --out ./schemas
```

`schemas/create_ticket.v1.input.json` lands with EAAP provenance fields
(`x-eaap-name`, `x-eaap-version`, `x-eaap-kind`, `$id`) layered on top of
Pydantic's emitted JSON Schema.

### 4. Decorate the tool with the registry

```python
@server.tool()
@registry.validate_tool("create_ticket", version=1)
async def create_ticket(payload: CreateTicketIn) -> CreateTicketOut:
    ticket_id = await ticketing_backend.open(payload.title, payload.priority)
    return CreateTicketOut(ticket_id=ticket_id, href=f"/tickets/{ticket_id}")
```

The decorator parses the incoming dict into `CreateTicketIn`, calls the tool,
and validates the return value against `CreateTicketOut`. Schema-violation
errors become `SchemaValidationError` carrying Pydantic's full error list in
`details` so observability can surface exactly which field failed.

### 5. Register the server with the SDK

```python
from ai_core.mcp import ComponentRegistry, MCPServerSpec, FastMCPConnectionFactory

registry = container.get(ComponentRegistry)
factory  = container.get(FastMCPConnectionFactory)

await registry.register(
    server,                                    # any IComponent
    component_type="mcp_server",
    metadata={"team": "ticketing"},
)

spec = MCPServerSpec(
    component_id="ticketing",
    transport="stdio",                         # or "http" / "sse"
    target="python",
    args=("-m", "mcp_servers.ticketing"),
)

async with factory.open(spec) as client:
    result = await client.call_tool("create_ticket", {"title": "outage"})
```

`ComponentRegistry` is async-safe, supports `health_check_all()` across
every registered component, and is the rendezvous point agents use to
discover and call tools at runtime.

---

## FastAPI integration

```python
from fastapi import Depends, FastAPI
from ai_core.di import Container, AgentModule
from ai_core.security import AuthorizedPrincipal, OPAAuthorization

container = Container.build([AgentModule()])
authz     = OPAAuthorization(container)

app = FastAPI()

@app.get("/projects/{project_id}")
async def read_project(
    project_id: str,
    principal: AuthorizedPrincipal = Depends(
        authz.requires(action="project.read", resource="project")
    ),
) -> dict[str, str]:
    return {"viewer": principal.subject, "project_id": project_id}
```

The dependency:

1. extracts the bearer token from `Authorization`;
2. verifies it via the bound `JWTVerifier`
   (`UnverifiedJWTDecoder` by default — appropriate when an upstream gateway
   has already validated the signature; swap in `HS256JWTVerifier` for in-service
   verification);
3. submits `{user, action, resource, claims, request}` to OPA;
4. returns an `AuthorizedPrincipal` carrying the JWT claims plus any
   obligations the policy attached.

---

## Testing

The DI seam is the test seam. Override any binding with a `Module`:

```python
from injector import Module, provider, singleton
from ai_core.di import Container, AgentModule
from ai_core.di.interfaces import ILLMClient

class _FakeLLMModule(Module):
    @singleton
    @provider
    def provide_llm(self) -> ILLMClient:
        return FakeLLM(canned_response="ok")

container = Container.build([AgentModule(), _FakeLLMModule()])
agent     = container.get(MyAgent)         # uses FakeLLM, real everything else
```

Containers are independent — singletons are scoped to a container, never
to the process. The provided test suite (126 tests) exercises every module
through this pattern; concrete examples live under `tests/unit/`.

---

## CLI reference

```
eaap init <name> [--path PATH] [--force]
eaap generate agent <name> [--path src --package agents] [--force]
eaap generate mcp   <name> [--path src --package mcp_servers] [--force]
eaap schema export --module-path PATH [--callable register]
                   [--out ./schemas-export] [--indent 2] [--no-overwrite]
```

Run `eaap --help` for the full surface.

---

## Module-by-module pointers

- **Settings** → `ai_core.config.AppSettings`. Nested submodels for `database`,
  `vector_db`, `storage`, `llm`, `budget`, `observability`, `security`, `agent`.
- **DI** → `ai_core.di.Container.build([AgentModule()])`. Override with
  `container.override(*modules)`.
- **LLM** → `ai_core.llm.LiteLLMClient` (LiteLLM + Tenacity + budget + observability).
- **Memory** → `ai_core.agents.MemoryManager` with `TokenCounter` Protocol.
- **Persistence** → `ai_core.persistence.PostgresCheckpointSaver` (SDK-level)
  + `ai_core.persistence.LangGraphCheckpointSaver` (LangGraph-native).
- **Observability** → `ai_core.observability.RealObservabilityProvider` (OTel + LangFuse)
  with NoOp opt-out.
- **Security** → `ai_core.security.OPAPolicyEvaluator`, `JWTVerifier`,
  `OPAAuthorization` FastAPI dep, `GuardrailNode`.
- **Schema** → `ai_core.schema.SchemaRegistry` + `validate_tool` decorator +
  `export_schemas` helper.
- **MCP** → `ai_core.mcp.ComponentRegistry`, `FastMCPConnectionFactory`
  (stdio / http / sse).

---

## License

See `LICENSE`.
