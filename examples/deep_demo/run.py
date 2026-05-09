"""Runnable demo: a real :class:`DeepAgent` orchestrating two real
sub-agents through hierarchical decomposition + virtual filesystem.

Demonstrates:

* the deep agent emitting `_decompose` to declare a multi-step plan
  with per-step sub-agent assignments,
* `_dispatch` calls running real sub-agents (Researcher, Writer) in
  isolated contexts; results land in the dispatch's return value and in
  ``scratchpad["dispatch_history"]``,
* `_write_file` persisting research notes between dispatches via
  ``scratchpad["files"]``,
* the deep agent's final answer assembled from the file + dispatch
  results.

Only the LLM is scripted — DI, LangGraph orchestration (for the deep
agent and each sub-agent), the ToolInvoker pipeline (validation, audit,
observability), and the synthetic-tool side-channels all run through
production code paths.

Run from the repo root::

    .venv/bin/python examples/deep_demo/run.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

from injector import Module, provider, singleton
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

from ai_core.agents import BaseAgent, DeepAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response

console = Console()


# ---------------------------------------------------------------------------
# Sub-agents + deep agent under test
# ---------------------------------------------------------------------------
class Researcher(BaseAgent):
    """Researches a topic in one short sentence."""

    agent_id = "researcher"

    def system_prompt(self) -> str:
        return "Research the topic in one short sentence."


class Writer(BaseAgent):
    """Drafts a paragraph from supplied notes."""

    agent_id = "writer"

    def system_prompt(self) -> str:
        return "Write a brief paragraph from the supplied notes."


class BriefAuthor(DeepAgent):
    """Authors a short brief by orchestrating a researcher and a writer."""

    agent_id = "brief-author"

    def base_system_prompt(self) -> str:
        return (
            "You author short briefs by orchestrating a researcher and a "
            "writer. Use the virtual filesystem to share notes between them."
        )

    def sub_agents(self) -> Mapping[str, type[BaseAgent]]:
        return {"researcher": Researcher, "writer": Writer}


# ---------------------------------------------------------------------------
# Container + helpers
# ---------------------------------------------------------------------------
def build_container(*, llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="deep-demo", environment="local")

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
    console.print(Rule("[bold green]Deep agent demo — plan + dispatch + filesystem[/bold green]"))

    plan_args = (
        '{"goal": "Author a brief about widgets.", '
        '"steps": ['
        '{"id": "step-1", "description": "Research widgets", "sub_agent": "researcher"}, '
        '{"id": "step-2", "description": "Draft the brief", "sub_agent": "writer"}'
        ']}'
    )

    llm = ScriptedLLM([
        # 1. _decompose
        make_llm_response(
            text="Planning the work.",
            tool_calls=[{
                "id": "tc-plan",
                "function": {"name": "_decompose", "arguments": plan_args},
            }],
            finish_reason="tool_calls",
        ),
        # 2. dispatch researcher
        make_llm_response(
            text="Delegating research.",
            tool_calls=[{
                "id": "tc-d1",
                "function": {
                    "name": "_dispatch",
                    "arguments": (
                        '{"sub_agent": "researcher", '
                        '"task": "Research widgets briefly.", '
                        '"step_id": "step-1"}'
                    ),
                },
            }],
            finish_reason="tool_calls",
        ),
        # 3. researcher's final answer
        make_llm_response(
            text="Widgets are small mechanical components used in many machines.",
        ),
        # 4. _write_file
        make_llm_response(
            text="Saving research notes.",
            tool_calls=[{
                "id": "tc-w",
                "function": {
                    "name": "_write_file",
                    "arguments": (
                        '{"path": "research.md", '
                        '"content": "Widgets are small mechanical components."}'
                    ),
                },
            }],
            finish_reason="tool_calls",
        ),
        # 5. dispatch writer
        make_llm_response(
            text="Delegating draft.",
            tool_calls=[{
                "id": "tc-d2",
                "function": {
                    "name": "_dispatch",
                    "arguments": (
                        '{"sub_agent": "writer", '
                        '"task": "Write a paragraph about widgets.", '
                        '"context": "See research.md.", '
                        '"step_id": "step-2"}'
                    ),
                },
            }],
            finish_reason="tool_calls",
        ),
        # 6. writer's final draft
        make_llm_response(
            text="Widgets are compact mechanical components powering many devices.",
        ),
        # 7. final answer
        make_llm_response(
            text="Brief: Widgets are compact mechanical components powering many devices.",
        ),
    ])

    container = build_container(llm=llm)
    async with container as c:
        deep = c.get(BriefAuthor)
        final_state = await deep.ainvoke(
            messages=[{"role": "user", "content": "Author a brief about widgets."}],
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
    summary.add_row("Total LLM calls:", str(len(llm.calls)))
    summary.add_row(
        "Sub-agent dispatches:",
        str(len((final_state.get("scratchpad") or {}).get("dispatch_history") or [])),
    )
    summary.add_row(
        "Files in scratch:",
        str(len((final_state.get("scratchpad") or {}).get("files") or {})),
    )
    summary.add_row("Final answer:", final_content[:200])
    console.print(Panel(summary, title="Outcome", border_style="green"))

    # LLM call log.
    console.print(Rule("[bold]LLM call log[/bold]"))
    table = Table(show_lines=False, expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("agent_id", style="cyan", width=18)
    table.add_column("first user msg / tool call", overflow="fold")
    for i, call in enumerate(llm.calls, start=1):
        first_user = next(
            (m["content"] for m in call["messages"] if m.get("role") == "user"),
            "",
        )
        table.add_row(str(i), call.get("agent_id") or "?", first_user[:120])
    console.print(table)

    # Dispatch history.
    console.print(Rule("[bold]Dispatch history[/bold]"))
    history = (final_state.get("scratchpad") or {}).get("dispatch_history") or []
    h_table = Table(show_lines=False, expand=True)
    h_table.add_column("step_id", width=10)
    h_table.add_column("sub_agent", style="cyan", width=12)
    h_table.add_column("result", overflow="fold")
    for entry in history:
        h_table.add_row(
            entry.get("step_id") or "-",
            entry["sub_agent"],
            entry["result"][:120],
        )
    console.print(h_table)

    # Files.
    console.print(Rule("[bold]Virtual filesystem[/bold]"))
    files = (final_state.get("scratchpad") or {}).get("files") or {}
    if files:
        for path, content in sorted(files.items()):
            console.print(Panel(
                Syntax(content, "markdown", theme="ansi_dark"),
                title=path, border_style="blue",
            ))
    else:
        console.print("(empty)")

    console.print()
    console.print(
        "[bold green]Done.[/bold green] One DeepAgent decomposed the task, "
        "dispatched two real sub-agents, persisted notes between them via "
        "the virtual filesystem, and assembled the final brief — all "
        "through the production SDK pipeline."
    )


if __name__ == "__main__":
    asyncio.run(main())
