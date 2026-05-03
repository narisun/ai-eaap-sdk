"""Runnable demo: a real :class:`BaseAgent` driven by a scripted LLM.

This script exercises the full SDK stack end-to-end without contacting
any real LLM provider:

* DI container with :class:`AgentModule` overridden to inject a fake
  :class:`ILLMClient`.
* Real :class:`BaseAgent.compile` producing a real LangGraph.
* Real :class:`MemoryManager` running its summarization chain (also
  routed through the fake LLM).
* Real :class:`Container` async lifecycle (``async with``).

Run from the repo root::

    PYTHONPATH=src python examples/math_tutor/run.py

The script prints a step-by-step trace of two scenarios:

1. Single-turn happy path.
2. Long-history conversation that triggers memory compaction.

Setting ``EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV=false`` silences
the OTel ConsoleSpanExporter that is otherwise on by default in
``EAAP_ENVIRONMENT=local``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from typing import Any

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
# spamming the output — the panels below already convey what we need.
os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")

from ai_core.agents import BaseAgent, TokenCounter  # noqa: E402
from ai_core.agents.memory import to_openai_message  # noqa: E402
from ai_core.config.settings import AppSettings  # noqa: E402
from ai_core.di import AgentModule, Container  # noqa: E402
from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMUsage  # noqa: E402

console = Console()


# ---------------------------------------------------------------------------
# Agent + fakes
# ---------------------------------------------------------------------------
class MathTutorAgent(BaseAgent):
    """Trivial tutor agent — used only to demo the SDK plumbing."""

    agent_id: str = "math-tutor"

    def system_prompt(self) -> str:
        return (
            "You are a friendly math tutor. Answer questions about basic "
            "arithmetic in one short sentence."
        )


class ScriptedLLM(ILLMClient):
    """Returns canned responses in registration order; records every call."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": [to_openai_message(m) for m in messages],
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "temperature": temperature,
            }
        )
        content = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return LLMResponse(
            model="scripted/test",
            content=content,
            tool_calls=[],
            usage=LLMUsage(prompt_tokens=42, completion_tokens=12, total_tokens=54),
            raw={},
        )


class _ForceCompactCounter:
    """A :class:`TokenCounter` that always reports far above the threshold."""

    def count(self, messages: Sequence[Any], *, model: str) -> int:
        return 1_000_000


# ---------------------------------------------------------------------------
# Container construction
# ---------------------------------------------------------------------------
def build_container(
    *,
    llm: ILLMClient,
    counter: TokenCounter | None = None,
) -> Container:
    settings = AppSettings(service_name="math-tutor-demo", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

        if counter is not None:
            @singleton
            @provider
            def provide_counter(self) -> TokenCounter:
                return counter  # type: ignore[return-value]

    return Container.build([AgentModule(settings=settings), _Overrides()])


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------
def _msg_role(m: Any) -> str:
    return to_openai_message(m).get("role", "?")


def _msg_content(m: Any) -> str:
    return to_openai_message(m).get("content", "")


def _print_history(label: str, messages: Sequence[Any]) -> None:
    table = Table(title=label, show_lines=False, expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("role", style="cyan", width=10)
    table.add_column("content", overflow="fold")
    for i, m in enumerate(messages):
        table.add_row(str(i), _msg_role(m), _msg_content(m))
    console.print(table)


def _print_calls(label: str, llm: ScriptedLLM) -> None:
    console.print(Rule(f"[bold]{label}[/bold]"))
    for i, call in enumerate(llm.calls, start=1):
        console.print(
            f"[dim]call {i}/{len(llm.calls)} — tenant={call['tenant_id']!r} "
            f"agent={call['agent_id']!r} temperature={call['temperature']}[/dim]"
        )
        for j, m in enumerate(call["messages"]):
            console.print(f"  [{j}] [cyan]{m.get('role')}[/cyan]: {m.get('content', '')[:120]}")


# ---------------------------------------------------------------------------
# Scenario 1: Single-turn happy path
# ---------------------------------------------------------------------------
async def demo_single_turn() -> None:
    console.print(Rule("[bold green]DEMO 1 — Single-turn agent invocation[/bold green]"))

    llm = ScriptedLLM(["Two plus two equals four."])
    container = build_container(llm=llm)

    async with container as c:
        agent = c.get(MathTutorAgent)
        final_state = await agent.ainvoke(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            essential={"user_id": "u-demo", "task_id": "T-1"},
            tenant_id="demo-tenant",
        )

    _print_history("Final messages in AgentState", final_state["messages"])

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("LLM calls made:", str(len(llm.calls)))
    summary.add_row("token_count:", str(final_state.get("token_count", 0)))
    summary.add_row("compaction_count:", str(final_state.get("compaction_count", 0)))
    summary.add_row("essential_entities:", str(dict(final_state.get("essential_entities", {}))))
    console.print(Panel(summary, title="Outcome", border_style="green"))

    _print_calls("LLM call log (single-turn)", llm)


# ---------------------------------------------------------------------------
# Scenario 2: Compaction triggered by long history
# ---------------------------------------------------------------------------
async def demo_compaction() -> None:
    console.print()
    console.print(
        Rule(
            "[bold magenta]DEMO 2 — Memory compaction triggered before agent turn"
            "[/bold magenta]"
        )
    )

    # The compaction node fires its own LLM call FIRST (summarisation),
    # then the agent node fires the second LLM call (post-compaction reply).
    llm = ScriptedLLM(
        [
            "User asked about TASK-42 earlier; deadline question is pending.",
            "The deadline is Friday at 5pm.",
        ]
    )
    container = build_container(llm=llm, counter=_ForceCompactCounter())

    long_history = [
        {"role": "user", "content": "I'm working on TASK-42"},
        {"role": "assistant", "content": "Got it — high priority?"},
        {"role": "user", "content": "Yes, customer-facing."},
        {"role": "assistant", "content": "Understood. Anything else?"},
        {"role": "user", "content": "What was the deadline?"},
    ]
    _print_history("Initial messages (5 turns)", long_history)

    async with container as c:
        agent = c.get(MathTutorAgent)
        final_state = await agent.ainvoke(
            messages=long_history,
            essential={"user_id": "u-demo", "task_id": "TASK-42"},
            tenant_id="demo-tenant",
        )

    _print_history("Final messages after compaction + agent turn", final_state["messages"])

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("LLM calls made:", f"{len(llm.calls)} (1× summary + 1× agent turn)")
    summary.add_row("compaction_count:", str(final_state.get("compaction_count", 0)))
    summary.add_row("summary stored:", final_state.get("summary", "(none)"))
    summary.add_row(
        "essential_entities:", str(dict(final_state.get("essential_entities", {})))
    )
    console.print(Panel(summary, title="Outcome", border_style="magenta"))

    _print_calls("LLM call log (compaction + agent)", llm)


async def main() -> None:
    await demo_single_turn()
    await demo_compaction()
    console.print()
    console.print(
        "[bold green]Done.[/bold green] Both scenarios executed against real "
        "LangGraph + DI + MemoryManager — only the LLM was scripted."
    )


if __name__ == "__main__":
    asyncio.run(main())
