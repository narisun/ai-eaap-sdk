"""Runnable demo: a real :class:`VerifierAgent` wrapping a real
:class:`BaseAgent`, both driven by a scripted LLM through the full SDK
stack.

Demonstrates:

* the wrapped agent producing an initial answer that fails verification,
* the verifier injecting feedback into the wrapped agent for retry,
* the wrapped agent revising its answer to satisfy the rubric,
* the verifier passing on attempt 2 and surfacing the verdict in
  ``state.metadata["last_verdict"]`` plus the verdict history in
  ``state.scratchpad["verifications"]``.

Only the LLM is scripted — DI, LangGraph, observability spans, budget
binding, and the verifier control loop all run through production code
paths.

Run from the repo root::

    uv run python examples/verifier_demo/run.py
"""

from __future__ import annotations

import asyncio
import os

from injector import Module, provider, singleton
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# Stub DSN so the lazy AsyncEngine never tries to dial a real database.
os.environ.setdefault(
    "EAAP_DATABASE__DSN",
    "postgresql+asyncpg://demo:demo@localhost:5432/demo",
)
os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")

from ai_core.agents import BaseAgent, Verdict, VerifierAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response

console = Console()


# ---------------------------------------------------------------------------
# Wrapped agent + verifier
# ---------------------------------------------------------------------------
class FactualAnswerer(BaseAgent):
    """Answers factual questions in one short sentence."""

    agent_id = "factual-answerer"

    def system_prompt(self) -> str:
        return "Answer the user's question in one short sentence."


class CitationVerifier(VerifierAgent):
    """Verifies the wrapped agent included a citation marker."""

    agent_id = "citation-verifier"
    max_retries = 1

    def wrapped_agent(self) -> type[BaseAgent]:
        return FactualAnswerer

    def verification_prompt(self) -> str:
        return (
            "Pass only if the answer contains the literal token "
            "'[cited]' as a citation marker."
        )


# ---------------------------------------------------------------------------
# Container + helpers
# ---------------------------------------------------------------------------
def build_container(*, llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="verifier-demo", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


def _verdict_tool_call(verdict: Verdict) -> dict:
    return {
        "id": "tc-verdict",
        "function": {
            "name": "_record_verdict",
            "arguments": verdict.model_dump_json(),
        },
    }


def _print_calls(llm: ScriptedLLM) -> None:
    console.print(Rule("[bold]LLM call log[/bold]"))
    table = Table(show_lines=False, expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("agent_id", style="cyan", width=22)
    table.add_column("first user msg", overflow="fold")
    for i, call in enumerate(llm.calls, start=1):
        first_user = next(
            (m["content"] for m in call["messages"] if m.get("role") == "user"),
            "",
        )
        table.add_row(str(i), call.get("agent_id") or "?", first_user[:120])
    console.print(table)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
async def main() -> None:
    console.print(Rule("[bold green]Verifier demo — strict citation check[/bold green]"))

    # Scripted LLM responses:
    #   1. wrapped (attempt 1) → "Paris is the capital of France." (no [cited])
    #   2. verifier round 1   → Verdict(passed=False, feedback=...)
    #   3. wrapped (attempt 2) → "Paris is the capital of France [cited]." (after seeing feedback)
    #   4. verifier round 2   → Verdict(passed=True)
    llm = ScriptedLLM([
        make_llm_response(text="Paris is the capital of France."),
        make_llm_response(
            text="",
            tool_calls=[_verdict_tool_call(Verdict(
                passed=False,
                feedback="Add the citation marker '[cited]' to your answer.",
            ))],
            finish_reason="tool_calls",
        ),
        make_llm_response(text="Paris is the capital of France [cited]."),
        make_llm_response(
            text="",
            tool_calls=[_verdict_tool_call(Verdict(
                passed=True,
                feedback="Includes the required citation marker.",
                score=1.0,
            ))],
            finish_reason="tool_calls",
        ),
    ])

    container = build_container(llm=llm)
    async with container as c:
        verifier = c.get(CitationVerifier)
        final_state = await verifier.ainvoke(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
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
    history = (final_state.get("scratchpad") or {}).get("verifications") or []
    summary.add_row("Verification rounds:", str(len(history)))
    final_verdict = (final_state.get("metadata") or {}).get("last_verdict") or {}
    summary.add_row(
        "Final verdict:",
        f"passed={final_verdict.get('passed')} — {final_verdict.get('feedback')}",
    )
    summary.add_row("Final answer:", final_content[:200])
    console.print(Panel(summary, title="Outcome", border_style="green"))

    # Show the verdict history.
    console.print(Rule("[bold]Verdict history[/bold]"))
    h_table = Table(show_lines=False, expand=True)
    h_table.add_column("attempt", justify="right", width=8)
    h_table.add_column("passed", style="cyan", width=8)
    h_table.add_column("feedback", overflow="fold")
    for entry in history:
        v = entry["verdict"]
        h_table.add_row(
            str(entry["attempt"]),
            "✓" if v["passed"] else "✗",
            v["feedback"],
        )
    console.print(h_table)

    _print_calls(llm)
    console.print()
    console.print(
        "[bold green]Done.[/bold green] One VerifierAgent wrapped a real "
        "BaseAgent, ran a real LangGraph for each attempt, and surfaced "
        "the final verdict in state — only the LLM was scripted."
    )


if __name__ == "__main__":
    asyncio.run(main())
