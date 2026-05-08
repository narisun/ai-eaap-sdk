"""Runnable demo: a real :class:`SupervisorAgent` driven by a scripted LLM.

Two child agents (a triage classifier and a research lookup), one
supervisor coordinating them. The supervisor's LLM emits ``tool_call``
deltas to delegate work; each child runs through a real LangGraph,
real DI, real ToolInvoker. Only the LLM is scripted.

Run from the repo root::

    uv run python examples/supervisor_demo/run.py

Setting ``EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV=false`` silences the
OTel ConsoleSpanExporter that is otherwise on in ``EAAP_ENVIRONMENT=local``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

from injector import Module, provider, singleton
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# Persistence is bound by default; point it at a stub DSN so the lazy engine
# never tries to dial a real database during import.
os.environ.setdefault(
    "EAAP_DATABASE__DSN",
    "postgresql+asyncpg://demo:demo@localhost:5432/demo",
)
# Demo runs in local mode but we don't want the OTel ConsoleSpanExporter
# spamming the output.
os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")

from ai_core.agents import BaseAgent, SupervisorAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response

console = Console()


# ---------------------------------------------------------------------------
# Children + supervisor
# ---------------------------------------------------------------------------
class TriageAgent(BaseAgent):
    """Classifies a support ticket into one short category."""

    agent_id = "triage"

    def system_prompt(self) -> str:
        return "You classify support tickets in one short sentence."


class ResearchAgent(BaseAgent):
    """Looks up the customer's last order id."""

    agent_id = "research"

    def system_prompt(self) -> str:
        return "You answer with the customer's last order id."


class SupportSupervisor(SupervisorAgent):
    agent_id = "support-supervisor"

    def system_prompt(self) -> str:
        return (
            "You coordinate a support team. "
            "Use `triage` first to classify, "
            "then `research` for context, "
            "then write the final answer."
        )

    def children(self) -> Mapping[str, type[BaseAgent]]:
        return {"triage": TriageAgent, "research": ResearchAgent}


# ---------------------------------------------------------------------------
# Container construction
# ---------------------------------------------------------------------------
def build_container(*, llm: ILLMClient) -> Container:
    settings = AppSettings(
        service_name="supervisor-demo", environment="local",
    )

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _print_call_log(llm: ScriptedLLM) -> None:
    console.print(Rule("[bold]LLM call log[/bold]"))
    table = Table(show_lines=False, expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("agent_id", style="cyan", width=22)
    table.add_column("first user msg", overflow="fold")
    for i, call in enumerate(llm.calls, start=1):
        # The user message right after the system prompt.
        msgs = call["messages"]
        first_user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        table.add_row(str(i), call.get("agent_id") or "?", first_user[:120])
    console.print(table)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
async def main() -> None:
    console.print(Rule("[bold green]Supervisor demo — multi-agent orchestration[/bold green]"))

    # Scripted LLM responses (what the supervisor and each child say):
    #   1. supervisor: tool_call('triage', {...})
    #   2. triage:    "billing-question"
    #   3. supervisor: tool_call('research', {...})
    #   4. research:  "order-12345"
    #   5. supervisor: final assistant message stitching the answer together
    llm = ScriptedLLM([
        make_llm_response(
            text="Delegating to triage.",
            tool_calls=[{
                "id": "tc-1",
                "function": {
                    "name": "triage",
                    "arguments": (
                        '{"task": "Classify this ticket about a missing order."}'
                    ),
                },
            }],
            finish_reason="tool_calls",
        ),
        make_llm_response(text="billing-question"),
        make_llm_response(
            text="Now researching customer history.",
            tool_calls=[{
                "id": "tc-2",
                "function": {
                    "name": "research",
                    "arguments": '{"task": "Find the customer\'s last order id."}',
                },
            }],
            finish_reason="tool_calls",
        ),
        make_llm_response(text="order-12345"),
        make_llm_response(
            text=(
                "Hi! I've classified your request as billing-question. "
                "Your most recent order is order-12345 — let me check its "
                "delivery status next."
            ),
        ),
    ])

    container = build_container(llm=llm)
    async with container as c:
        supervisor = c.get(SupportSupervisor)
        final_state = await supervisor.ainvoke(
            messages=[{
                "role": "user",
                "content": "I'm missing an order; can you help?",
            }],
            essential={"user_id": "u-1", "task_id": "T-42"},
            tenant_id="acme",
        )

    last_msg = final_state["messages"][-1]
    final_content = (
        getattr(last_msg, "content", None)
        or (last_msg.get("content") if isinstance(last_msg, dict) else "")
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Total LLM calls:", str(len(llm.calls)))
    summary.add_row(
        "Agents involved:",
        ", ".join(sorted({c.get("agent_id") or "?" for c in llm.calls})),
    )
    summary.add_row("Final answer:", final_content[:200])
    console.print(Panel(summary, title="Outcome", border_style="green"))

    _print_call_log(llm)
    console.print()
    console.print(
        "[bold green]Done.[/bold green] One supervisor delegated to two "
        "children through real LangGraph + DI + ToolInvoker — only the LLM "
        "was scripted."
    )


if __name__ == "__main__":
    asyncio.run(main())
