"""Runnable demo: a real :class:`PlanningAgent` driven by a scripted LLM.

Plan-and-execute loop end-to-end:

  1. The LLM emits a `_make_plan` tool call to declare the plan.
  2. The plan lands in `state.scratchpad["plans"]`; the system prompt
     on the next turn shows the plan with status markers.
  3. The LLM executes each pending step using the user-declared work
     tool (`lookup_customer`).
  4. After all steps, the LLM emits the final answer.

Only the LLM is scripted — DI, LangGraph orchestration, ToolInvoker
dispatch, observability spans, and the plan-capture mechanism all run
through the production code paths.

Run from the repo root::

    uv run python examples/planner_demo/run.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

from injector import Module, provider, singleton
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# Persistence is bound by default; point it at a stub DSN so the lazy
# engine never tries to dial a real database.
os.environ.setdefault(
    "EAAP_DATABASE__DSN",
    "postgresql+asyncpg://demo:demo@localhost:5432/demo",
)
os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")

from ai_core.agents import PlanningAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response
from ai_core.tools import tool

console = Console()


# ---------------------------------------------------------------------------
# Work tool + planner
# ---------------------------------------------------------------------------
class _LookupInput(BaseModel):
    customer_id: str


class _LookupOutput(BaseModel):
    info: str


@tool(name="lookup_customer", version=1, opa_path=None)
async def lookup_customer(payload: _LookupInput) -> _LookupOutput:
    """Test work tool — returns canned data."""
    canned = {"C-1": "loyal customer / 12 orders / VIP tier"}
    return _LookupOutput(info=canned.get(payload.customer_id, "no record"))


class CustomerResearchPlanner(PlanningAgent):
    """Plans how to answer a customer-research question, then executes."""

    agent_id = "customer-research-planner"
    max_replans = 3

    def base_system_prompt(self) -> str:
        return (
            "You are a customer-research analyst. Decompose the user's "
            "question into discrete steps and execute them with the "
            "available work tools."
        )

    def work_tools(self) -> Sequence[Any]:
        return [lookup_customer]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
def build_container(*, llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="planner-demo", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------
def _print_calls(llm: ScriptedLLM) -> None:
    console.print(Rule("[bold]LLM call log[/bold]"))
    table = Table(show_lines=False, expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("system mode", style="cyan", width=22)
    table.add_column("LLM emitted", overflow="fold")
    for i, call in enumerate(llm.calls, start=1):
        sys_msg = next(
            (m["content"] for m in call["messages"] if m.get("role") == "system"),
            "",
        )
        if "[Plan complete]" in sys_msg:
            mode = "Plan complete"
        elif "[Replan cap reached]" in sys_msg:
            mode = "Replan cap reached"
        elif "[Execution mode]" in sys_msg:
            mode = "Execution mode"
        elif "[Planning mode]" in sys_msg:
            mode = "Planning mode"
        else:
            mode = "?"
        table.add_row(str(i), mode, "(emitted on turn — see scripted responses)")
    console.print(table)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
async def main() -> None:
    console.print(Rule("[bold green]Planner demo — plan-and-execute loop[/bold green]"))

    plan_args_json = (
        '{"goal": "Profile customer C-1\'s loyalty status.", '
        '"steps": ['
        '{"id": "step-1", "description": "Look up customer C-1 by id."}, '
        '{"id": "step-2", "description": "Note loyalty tier and order count."}'
        ']}'
    )

    llm = ScriptedLLM([
        # 1. Declare the plan.
        make_llm_response(
            text="Planning the customer profile.",
            tool_calls=[{
                "id": "tc-plan",
                "function": {"name": "_make_plan", "arguments": plan_args_json},
            }],
            finish_reason="tool_calls",
        ),
        # 2. Execute step-1: lookup_customer.
        make_llm_response(
            text="Looking up C-1.",
            tool_calls=[{
                "id": "tc-1",
                "function": {
                    "name": "lookup_customer",
                    "arguments": '{"customer_id": "C-1"}',
                },
            }],
            finish_reason="tool_calls",
        ),
        # 3. Execute step-2: confirm status.
        make_llm_response(
            text="Confirming with another lookup.",
            tool_calls=[{
                "id": "tc-2",
                "function": {
                    "name": "lookup_customer",
                    "arguments": '{"customer_id": "C-1"}',
                },
            }],
            finish_reason="tool_calls",
        ),
        # 4. Final answer.
        make_llm_response(
            text=(
                "Customer C-1 is a loyal VIP customer with 12 orders. "
                "Their loyalty tier is verified."
            ),
        ),
    ])

    container = build_container(llm=llm)
    async with container as c:
        planner = c.get(CustomerResearchPlanner)
        final_state = await planner.ainvoke(
            messages=[{
                "role": "user",
                "content": "Tell me about customer C-1's loyalty status.",
            }],
            essential={"user_id": "u-1"},
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
    plans = (final_state.get("scratchpad") or {}).get("plans") or []
    summary.add_row("Plan revisions:", str(len(plans)))
    if plans:
        last_plan = plans[-1]
        summary.add_row("Goal:", last_plan["goal"])
        summary.add_row("Steps in plan:", str(len(last_plan["steps"])))
    summary.add_row("Final answer:", final_content[:200])
    console.print(Panel(summary, title="Outcome", border_style="green"))

    _print_calls(llm)
    console.print()
    console.print(
        "[bold green]Done.[/bold green] One PlanningAgent ran a real "
        "plan-and-execute loop through real LangGraph + DI + ToolInvoker — "
        "only the LLM was scripted."
    )


if __name__ == "__main__":
    asyncio.run(main())
