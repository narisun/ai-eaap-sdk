"""End-to-end component test: a real :class:`PlanningAgent` driven by
a scripted LLM through the full SDK stack.

Verifies the plan-and-execute loop:

1. Supervisor LLM emits ``_make_plan`` (turn 1) → plan captured into
   ``state.scratchpad``.
2. LLM executes step 1 via a real user tool (turn 2) → tool dispatch
   runs, result returned.
3. LLM executes step 2 via the same user tool (turn 3).
4. LLM emits a final assistant message (turn 4) → graph terminates.

The system prompt at each turn carries the live plan status, so
this also exercises the dynamic-system-prompt mechanism.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import pytest
from injector import Module, provider, singleton
from pydantic import BaseModel

from ai_core.agents import PlanningAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response
from ai_core.tools import tool

pytestmark = pytest.mark.component

os.environ.setdefault(
    "EAAP_DATABASE__DSN",
    "postgresql+asyncpg://demo:demo@localhost:5432/demo",
)
os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")


# ---------------------------------------------------------------------------
# Test agent + tool
# ---------------------------------------------------------------------------
class _LookupInput(BaseModel):
    customer_id: str


class _LookupOutput(BaseModel):
    info: str


@tool(name="lookup_customer", version=1, opa_path=None)
async def _lookup_customer(payload: _LookupInput) -> _LookupOutput:
    """Test work tool — returns canned data."""
    return _LookupOutput(info=f"customer={payload.customer_id}/loyal")


class ResearchPlanner(PlanningAgent):
    agent_id = "research-planner"
    max_replans = 3

    def base_system_prompt(self) -> str:
        return "You are a research assistant who answers questions about customers."

    def work_tools(self) -> Sequence[Any]:
        return [_lookup_customer]


# ---------------------------------------------------------------------------
# DI wiring
# ---------------------------------------------------------------------------
def _build_container(llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="planner-test", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_planner_makes_plan_then_executes_steps_then_finalises() -> None:
    """Full plan-and-execute path: 4 LLM turns, plan captured, two real
    tool dispatches, final answer."""
    plan_args_json = (
        '{"goal": "Profile customer C-1\'s loyalty status.", '
        '"steps": ['
        '{"id": "step-1", "description": "Look up the customer record."}, '
        '{"id": "step-2", "description": "Note loyalty status from the record."}'
        ']}'
    )

    llm = ScriptedLLM([
        # 1. _make_plan
        make_llm_response(
            text="Planning the work.",
            tool_calls=[{
                "id": "tc-plan",
                "function": {
                    "name": "_make_plan",
                    "arguments": plan_args_json,
                },
            }],
            finish_reason="tool_calls",
        ),
        # 2. Execute step-1: lookup_customer
        make_llm_response(
            text="Looking up customer.",
            tool_calls=[{
                "id": "tc-1",
                "function": {
                    "name": "lookup_customer",
                    "arguments": '{"customer_id": "C-1"}',
                },
            }],
            finish_reason="tool_calls",
        ),
        # 3. Execute step-2: another lookup (same tool, demonstrates that
        #    work tools are reusable; in a real setup step-2 might be a
        #    different tool, but ScriptedLLM doesn't care).
        make_llm_response(
            text="Confirming loyalty.",
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
            text="Customer C-1 is a loyal customer.",
        ),
    ])

    container = _build_container(llm)
    async with container as c:
        planner = c.get(ResearchPlanner)
        final_state = await planner.ainvoke(
            messages=[{"role": "user", "content": "What's the loyalty status of C-1?"}],
            tenant_id="acme",
        )

    # Exactly four LLM turns; the planner doesn't call extra ones because
    # the LLM emitted no tool_calls on turn 4.
    assert len(llm.calls) == 4

    # All calls were made by the planner (no sub-agents in this test).
    assert all(c["agent_id"] == "research-planner" for c in llm.calls)

    # Plan landed in scratchpad with revision count 1.
    scratchpad = final_state.get("scratchpad") or {}
    plans = scratchpad.get("plans") or []
    assert len(plans) == 1
    assert plans[0]["goal"] == "Profile customer C-1's loyalty status."
    assert len(plans[0]["steps"]) == 2
    assert scratchpad.get("replan_count") == 1

    # The system prompt visible on each LLM call shows the right mode.
    # Turn 1 is initial mode (no plan yet); turn 2-4 are executing mode.
    sys_prompts = [
        next(
            (m for m in c["messages"] if m.get("role") == "system"),
            {},
        ).get("content", "")
        for c in llm.calls
    ]
    assert "[Planning mode]" in sys_prompts[0]
    assert "[Execution mode]" in sys_prompts[1]
    assert "[Execution mode]" in sys_prompts[2]
    assert "[Execution mode]" in sys_prompts[3]

    # Final assistant message is what the LLM produced on turn 4.
    last_msg = final_state["messages"][-1]
    last_content = (
        getattr(last_msg, "content", None)
        or last_msg.get("content", "")
    )
    assert "loyal customer" in last_content
