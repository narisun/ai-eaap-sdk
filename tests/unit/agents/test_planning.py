"""Unit tests for :class:`PlanningAgent`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ai_core.agents import (
    AgentRuntime,
    AgentState,
    Plan,
    PlanAck,
    PlanningAgent,
    PlanStep,
)
from ai_core.agents._resolver import AgentResolver
from ai_core.agents.tool_errors import DefaultToolErrorRenderer
from ai_core.audit.null import NullAuditSink
from ai_core.config.settings import AppSettings
from ai_core.testing import FakeObservabilityProvider
from ai_core.tools import tool
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.registrar import ToolRegistrar
from ai_core.tools.resolver import DefaultToolResolver
from ai_core.tools.spec import ToolSpec

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class _SearchInput(BaseModel):
    query: str


class _SearchOutput(BaseModel):
    items: list[str]


@tool(name="search", version=1, opa_path=None)
async def _search(payload: _SearchInput) -> _SearchOutput:
    return _SearchOutput(items=[f"hit-for-{payload.query}"])


class _ResearchPlanner(PlanningAgent):
    agent_id = "research-planner"

    def base_system_prompt(self) -> str:
        return "You are a research assistant."

    def work_tools(self) -> Sequence[Any]:
        return [_search]


class _NoToolsPlanner(PlanningAgent):
    agent_id = "no-tools-planner"

    def base_system_prompt(self) -> str:
        return "You only plan; no work tools."


def _make_runtime() -> AgentRuntime:
    """Build a runtime that doesn't need DI — direct field assignment."""
    invoker = ToolInvoker(
        observability=FakeObservabilityProvider(),
        audit=NullAuditSink(),
    )

    class _StubResolver(AgentResolver):
        def __init__(self) -> None:
            pass

        def resolve(self, cls):  # type: ignore[override]
            return MagicMock()

    return AgentRuntime(
        agent_settings=AppSettings(service_name="t", environment="local").agent,
        llm=MagicMock(),
        memory=MagicMock(),
        observability=FakeObservabilityProvider(),
        tool_invoker=invoker,
        mcp_factory=MagicMock(),
        tool_error_renderer=DefaultToolErrorRenderer(),
        tool_resolver=DefaultToolResolver(MagicMock(), invoker),
        tool_registrar=ToolRegistrar(invoker),
        agent_resolver=_StubResolver(),
    )


# ---------------------------------------------------------------------------
# tools() advertisement
# ---------------------------------------------------------------------------
def test_tools_prepends_make_plan_synthetic_tool() -> None:
    """The synthetic _make_plan tool comes first; user tools follow."""
    planner = _ResearchPlanner(_make_runtime())
    tools = planner.tools()

    assert len(tools) == 2
    plan_spec = tools[0]
    assert isinstance(plan_spec, ToolSpec)
    assert plan_spec.name == "_make_plan"
    assert plan_spec.input_model is Plan
    assert plan_spec.output_model is PlanAck
    assert plan_spec.opa_path is None  # planner-internal — not OPA-gated

    # Second tool is the user-declared one.
    user_spec = tools[1]
    assert isinstance(user_spec, ToolSpec)
    assert user_spec.name == "search"


def test_tools_with_no_work_tools_returns_only_make_plan() -> None:
    planner = _NoToolsPlanner(_make_runtime())
    tools = planner.tools()
    assert len(tools) == 1
    spec = tools[0]
    assert isinstance(spec, ToolSpec)
    assert spec.name == "_make_plan"


# ---------------------------------------------------------------------------
# system_prompt() decoration based on plan state
# ---------------------------------------------------------------------------
def test_system_prompt_initial_mode_when_no_state() -> None:
    """Without state, the addendum is the initial-plan instructions."""
    planner = _ResearchPlanner(_make_runtime())
    prompt = planner.system_prompt()
    assert "[Planning mode]" in prompt
    assert "_make_plan" in prompt
    # base_system_prompt() flows through.
    assert "research assistant" in prompt


def test_system_prompt_initial_mode_when_state_has_no_plan() -> None:
    planner = _ResearchPlanner(_make_runtime())
    planner._current_state = {"scratchpad": {}}
    prompt = planner.system_prompt()
    assert "[Planning mode]" in prompt
    assert "[Execution mode]" not in prompt


def test_system_prompt_executing_mode_when_plan_present() -> None:
    planner = _ResearchPlanner(_make_runtime())
    plan = Plan(
        goal="Find the customer's last order",
        steps=[
            PlanStep(id="step-1", description="Look up customer record"),
            PlanStep(
                id="step-2",
                description="Get most recent order id",
                status="done",
                result="order-12345",
            ),
        ],
    )
    planner._current_state = {
        "scratchpad": {"plans": [plan.model_dump()], "replan_count": 1},
    }
    prompt = planner.system_prompt()
    assert "[Execution mode]" in prompt
    # Plan rendered with status markers.
    assert "[ ] step-1" in prompt
    assert "[x] step-2" in prompt
    assert "order-12345" in prompt
    assert "1 of 3 revisions used" in prompt


def test_system_prompt_done_mode_when_all_steps_done() -> None:
    planner = _ResearchPlanner(_make_runtime())
    plan = Plan(
        goal="Trivial",
        steps=[PlanStep(id="step-1", description="Look up", status="done", result="ok")],
    )
    planner._current_state = {
        "scratchpad": {"plans": [plan.model_dump()], "replan_count": 1},
    }
    prompt = planner.system_prompt()
    assert "[Plan complete]" in prompt


def test_system_prompt_max_replans_message_when_cap_hit() -> None:
    """When revision_count >= max_replans and plan still has pending steps,
    the LLM gets the 'finalize without further revisions' nudge."""
    planner = _ResearchPlanner(_make_runtime())
    planner.max_replans = 2
    plan = Plan(
        goal="Hard task",
        steps=[
            PlanStep(id="step-1", description="Pending", status="pending"),
        ],
    )
    planner._current_state = {
        "scratchpad": {
            "plans": [plan.model_dump(), plan.model_dump()],  # 2 revisions
            "replan_count": 2,
        },
    }
    prompt = planner.system_prompt()
    assert "[Replan cap reached]" in prompt
    assert "Do NOT call `_make_plan` again" in prompt


# ---------------------------------------------------------------------------
# Plan capture via _make_plan tool handler
# ---------------------------------------------------------------------------
def test_make_plan_handler_stashes_pending_plan() -> None:
    planner = _ResearchPlanner(_make_runtime())
    plan = Plan(
        goal="Test",
        steps=[PlanStep(id="step-1", description="x")],
    )

    ack = planner._on_plan_submitted(plan)
    assert isinstance(ack, PlanAck)
    assert ack.accepted is True
    assert ack.revision == 1  # first revision
    assert ack.step_count == 1
    assert ack.note is None
    assert planner._pending_plan is plan


def test_make_plan_handler_warns_when_past_replan_cap() -> None:
    """Submitting a plan past the cap returns a note flagging it."""
    planner = _ResearchPlanner(_make_runtime())
    planner.max_replans = 1
    # Existing state: already at revision 1 (cap = 1)
    planner._current_state = {
        "scratchpad": {
            "plans": [Plan(
                goal="prev",
                steps=[PlanStep(id="step-1", description="prev")],
            ).model_dump()],
            "replan_count": 1,
        },
    }
    new_plan = Plan(goal="new", steps=[PlanStep(id="step-1", description="new")])
    ack = planner._on_plan_submitted(new_plan)
    assert ack.revision == 2
    assert ack.note is not None
    assert "past the configured replan cap" in ack.note


@pytest.mark.asyncio
async def test_tool_node_merges_pending_plan_into_scratchpad() -> None:
    """The override of _tool_node captures _pending_plan into state.scratchpad."""
    planner = _ResearchPlanner(_make_runtime())

    # Simulate the LLM already having called _make_plan — we pre-set
    # _pending_plan as if super()._tool_node ran the handler.
    plan = Plan(goal="g", steps=[PlanStep(id="step-1", description="d")])

    async def _stub_super_tool_node(self, state):  # type: ignore[no-untyped-def]
        # Mimic super()._tool_node: simulate that the handler set _pending_plan.
        self._pending_plan = plan
        return {"messages": [{"role": "tool", "content": "ok"}]}

    # Patch BaseAgent._tool_node by hijacking super() lookup via attribute.
    from unittest.mock import patch
    with patch(
        "ai_core.agents.base.BaseAgent._tool_node", _stub_super_tool_node,
    ):
        new_state = await planner._tool_node({"scratchpad": {}})

    assert "scratchpad" in new_state
    scratchpad = new_state["scratchpad"]
    assert "plans" in scratchpad
    assert len(scratchpad["plans"]) == 1
    assert scratchpad["plans"][0]["goal"] == "g"
    assert scratchpad["replan_count"] == 1


@pytest.mark.asyncio
async def test_tool_node_appends_to_existing_plan_history() -> None:
    """Subsequent _make_plan calls append; replan_count grows."""
    planner = _ResearchPlanner(_make_runtime())
    prev_plan = Plan(
        goal="prev", steps=[PlanStep(id="s", description="prev")]
    )
    new_plan = Plan(
        goal="new", steps=[PlanStep(id="s", description="new")]
    )

    async def _stub_super_tool_node(self, state):  # type: ignore[no-untyped-def]
        self._pending_plan = new_plan
        return {"messages": []}

    from unittest.mock import patch
    initial_state: AgentState = {
        "scratchpad": {"plans": [prev_plan.model_dump()], "replan_count": 1},
    }
    with patch(
        "ai_core.agents.base.BaseAgent._tool_node", _stub_super_tool_node,
    ):
        new_state = await planner._tool_node(initial_state)

    plans = new_state["scratchpad"]["plans"]
    assert len(plans) == 2
    assert plans[0]["goal"] == "prev"
    assert plans[1]["goal"] == "new"
    assert new_state["scratchpad"]["replan_count"] == 2


# ---------------------------------------------------------------------------
# Plan-status helpers
# ---------------------------------------------------------------------------
def test_render_plan_marks_step_status_visually() -> None:
    plan = Plan(
        goal="Multi-status",
        steps=[
            PlanStep(id="s1", description="done step", status="done", result="ok"),
            PlanStep(id="s2", description="pending step"),
            PlanStep(id="s3", description="failed step", status="failed", notes="why"),
            PlanStep(id="s4", description="active", status="in_progress"),
        ],
    )
    rendered = PlanningAgent._render_plan(plan)
    assert "[x] s1" in rendered
    assert "[ ] s2" in rendered
    assert "[!] s3" in rendered
    assert "[~] s4" in rendered
    assert "result: ok" in rendered
    assert "notes: why" in rendered


def test_all_steps_done_helper() -> None:
    p1 = Plan(goal="g", steps=[PlanStep(id="x", description="d", status="done")])
    p2 = Plan(goal="g", steps=[PlanStep(id="x", description="d")])
    p3 = Plan(goal="g", steps=[
        PlanStep(id="x", description="d", status="done"),
        PlanStep(id="y", description="d", status="failed"),
    ])
    assert PlanningAgent._all_steps_done(p1) is True
    assert PlanningAgent._all_steps_done(p2) is False
    assert PlanningAgent._all_steps_done(p3) is False  # failed != done
