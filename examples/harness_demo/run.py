"""Runnable demo: a real :class:`HarnessAgent` wrapping a real
:class:`BaseAgent` with one tool, recording every LLM call and tool
dispatch into a structured :class:`Trace`.

Demonstrates:

* the wrapped agent producing a tool_call → the harness capturing the
  LLM request/response,
* the tool dispatch (with validation, audit, observability spans) →
  the harness capturing the dispatch outcome,
* the wrapped agent's final answer → the harness capturing the second
  LLM call,
* the captured :class:`Trace` exposed via :attr:`HarnessAgent.last_trace`,
  ready for ``trace.model_dump_json()`` persistence.

Only the LLM is scripted — DI, LangGraph, observability, and the
ToolInvoker pipeline all run through production code paths.

Run from the repo root::

    .venv/bin/python examples/harness_demo/run.py
"""

from __future__ import annotations

import asyncio
import os

from injector import Module, provider, singleton
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

# Stub DSN so the lazy AsyncEngine never tries to dial a real database.
os.environ.setdefault(
    "EAAP_DATABASE__DSN",
    "postgresql+asyncpg://demo:demo@localhost:5432/demo",
)
os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")

from ai_core.agents import BaseAgent, HarnessAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response
from ai_core.tools import tool

console = Console()


# ---------------------------------------------------------------------------
# Tool + agent under test
# ---------------------------------------------------------------------------
class _LookupIn(BaseModel):
    customer_id: str


class _LookupOut(BaseModel):
    name: str
    plan: str


@tool(name="lookup_customer", version=1)
async def lookup_customer(payload: _LookupIn) -> _LookupOut:
    """Return a fake customer record."""
    return _LookupOut(name=f"Customer {payload.customer_id}", plan="enterprise")


class SupportAgent(BaseAgent):
    """Answers support questions with one customer lookup."""

    agent_id = "support-agent"

    def system_prompt(self) -> str:
        return "Look up the customer, then answer the user."

    def tools(self):
        return [lookup_customer]


class SupportHarness(HarnessAgent):
    """Captures everything :class:`SupportAgent` does for replay/eval."""

    agent_id = "support-harness"

    def wrapped_agent(self) -> type[BaseAgent]:
        return SupportAgent


# ---------------------------------------------------------------------------
# Container + helpers
# ---------------------------------------------------------------------------
def build_container(*, llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="harness-demo", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
async def main() -> None:
    console.print(Rule("[bold green]Harness demo — capture-only tracing[/bold green]"))

    # Scripted LLM:
    #   1. tool_call('lookup_customer', {customer_id: "C-42"})
    #   2. final answer mentioning the looked-up plan
    llm = ScriptedLLM([
        make_llm_response(
            text="Looking that up.",
            tool_calls=[{
                "id": "tc-1",
                "function": {
                    "name": "lookup_customer",
                    "arguments": '{"customer_id": "C-42"}',
                },
            }],
            finish_reason="tool_calls",
        ),
        make_llm_response(
            text="Customer C-42 is on the enterprise plan.",
        ),
    ])

    container = build_container(llm=llm)
    async with container as c:
        harness = c.get(SupportHarness)
        final_state = await harness.ainvoke(
            messages=[{"role": "user", "content": "Tell me about C-42."}],
            essential={"user_id": "u-1"},
            tenant_id="acme",
        )

    last = final_state["messages"][-1]
    final_content = (
        getattr(last, "content", None)
        or (last.get("content") if isinstance(last, dict) else "")
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    trace = harness.last_trace
    assert trace is not None
    summary.add_row("Wrapped agent:", trace.wrapped_agent)
    summary.add_row("Trace events:", str(len(trace.events)))
    summary.add_row("Final answer:", final_content[:200])
    console.print(Panel(summary, title="Outcome", border_style="green"))

    # Show the captured event sequence.
    console.print(Rule("[bold]Captured events[/bold]"))
    table = Table(show_lines=False, expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("kind", style="cyan", width=6)
    table.add_column("detail", overflow="fold")
    for i, event in enumerate(trace.events, start=1):
        if event.kind == "llm" and event.llm is not None:
            tc_count = len(event.llm.response_tool_calls)
            text = event.llm.response_content[:80]
            detail = (
                f"{event.llm.agent_id} → "
                f"content={text!r}, tool_calls={tc_count}"
            )
        elif event.kind == "tool" and event.tool is not None:
            detail = (
                f"{event.tool.tool} v{event.tool.tool_version} "
                f"args={event.tool.raw_args} → outcome={event.tool.outcome}"
            )
        else:
            detail = "(empty)"
        table.add_row(str(i), event.kind, detail)
    console.print(table)

    # Show the JSON-serialised trace (truncated for legibility).
    console.print(Rule("[bold]Trace JSON (truncated)[/bold]"))
    payload = trace.model_dump_json(indent=2)
    snippet = payload if len(payload) < 1500 else payload[:1500] + "\n  ..."
    console.print(Syntax(snippet, "json", theme="ansi_dark"))

    console.print()
    console.print(
        "[bold green]Done.[/bold green] One HarnessAgent wrapped a real "
        "BaseAgent, captured every LLM + tool call as the agent ran, and "
        "exposed the trace as a Pydantic model ready to persist."
    )


if __name__ == "__main__":
    asyncio.run(main())
