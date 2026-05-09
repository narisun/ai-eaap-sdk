"""Plan-and-execute agent primitive.

:class:`PlanningAgent` runs a two-mode loop:

* **Plan mode** — on the first turn (and any turn after the LLM
  decides to re-plan), the LLM emits a ``_make_plan`` tool call
  declaring the goal and discrete steps it intends to follow.
* **Execute mode** — on subsequent turns, the LLM executes one
  pending step at a time using the user-declared work tools
  (search, DB lookups, HTTP calls). Each turn sees the live plan
  status in its system prompt.

The two-mode loop is driven entirely by what the LLM emits. There
is no custom graph routing — :class:`PlanningAgent` decorates the
system prompt dynamically based on plan state and lets the LLM
pick which tool to invoke next. This composes with the v1 tool
loop (validation, OPA, audit, observability) and the
:class:`SupervisorAgent` pattern (a planner can be a child of a
supervisor; a supervisor's child can itself be a planner).

Plan history is preserved in :attr:`AgentState.scratchpad` so the
LLM can see what didn't work on previous attempts and re-plan
informedly. ``max_replans`` caps the revision count; when reached,
the system prompt nudges the LLM to finalize with whatever data
it has rather than generating yet another plan.

Why plan-as-tool-call
=====================
The synthetic ``_make_plan`` tool takes a :class:`Plan` payload via
Pydantic. This:

* reuses 100 % of the v1 tool plumbing (input validation, span
  emission, audit recording),
* gives modern LLMs a familiar, well-supported surface
  (structured tool-call output is well-trained),
* keeps re-planning a single LLM-driven action — the LLM calls
  ``_make_plan`` again and we automatically capture the new plan
  and increment the revision counter.

Implicit step status
====================
:class:`PlanningAgent` does not provide a separate ``_mark_step``
tool. Step status is tracked implicitly: the next pending step is
whichever has ``status == "pending"`` first; the rendered plan in
the system prompt makes ordering visible. If the LLM wants to
reorder, skip, or revise, it calls ``_make_plan`` again with the
updated step list. This keeps the API minimal and the LLM-side
contract obvious.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from injector import inject
from pydantic import BaseModel, Field

from ai_core.agents.base import BaseAgent
from ai_core.agents.runtime import AgentRuntime
from ai_core.agents.state import AgentState
from ai_core.tools.spec import ToolHandler, ToolSpec

if TYPE_CHECKING:
    from ai_core.tools.spec import Tool


# ---------------------------------------------------------------------------
# Plan data model
# ---------------------------------------------------------------------------
StepStatus = Literal["pending", "in_progress", "done", "failed"]


class PlanStep(BaseModel):
    """A single step in a structured plan.

    Attributes:
        id: Stable identifier for the step (e.g. ``"step-1"``). The
            LLM is encouraged to use ``"step-N"`` for monotonic IDs
            so re-plans can refer to specific steps.
        description: Human-readable description of the step's intent.
        status: Lifecycle marker — ``pending`` initially, transitions
            to ``done`` / ``failed`` as the LLM works through them.
        result: Free-form text the LLM populates after a step
            completes. Optional; the LLM may carry results in
            message history instead.
        notes: Free-form notes (e.g. why a step failed, blockers,
            context that the next replan should consider).
    """

    id: str
    description: str
    status: StepStatus = "pending"
    result: str | None = None
    notes: str | None = None


class Plan(BaseModel):
    """A goal plus a sequence of discrete steps to reach it.

    The LLM produces an instance of this model when calling the
    synthetic ``_make_plan`` tool. The model is intentionally
    minimal — anything not captured by ``goal`` and ``steps``
    flows through the regular message history.
    """

    goal: str = Field(..., description="The user-level goal in one short sentence.")
    steps: list[PlanStep] = Field(
        ..., description="Ordered sequence of steps to reach the goal.",
    )


class PlanAck(BaseModel):
    """Acknowledgement returned by ``_make_plan`` after a plan is captured.

    The LLM sees this on its next turn as the ``_make_plan`` tool's
    output, confirming the plan was accepted and providing the new
    revision number.
    """

    accepted: bool
    revision: int
    step_count: int
    note: str | None = None


# ---------------------------------------------------------------------------
# Default planning instructions
# ---------------------------------------------------------------------------
_PLAN_INSTRUCTIONS_INITIAL = """\
[Planning mode]
Before doing any work, call the `_make_plan` tool with a Plan that
breaks the user's request into discrete steps. Each step should be
small enough to execute in a single tool call. Use stable ids
("step-1", "step-2", ...). After your plan is accepted, you'll be
asked to execute the steps one at a time."""

_PLAN_INSTRUCTIONS_EXECUTING = """\
[Execution mode]
Your plan is below. On each turn, execute the next pending step by
calling the appropriate work tool. Keep step execution focused —
one pending step per turn. If a step fails or you discover the plan
needs revision, call `_make_plan` again with an updated Plan; you
have a limited number of revisions, so revise meaningfully.

Current plan ({revision_n} of {max_replans} revisions used):"""

_PLAN_INSTRUCTIONS_DONE = """\
[Plan complete]
Every step in your latest plan is marked `done`. Either emit your
final answer to the user or call `_make_plan` to revise if you
believe more work is needed."""

_PLAN_INSTRUCTIONS_MAX_REPLANS = """\
[Replan cap reached]
You have reached the configured maximum number of plan revisions.
Do NOT call `_make_plan` again. Finalize your answer to the user
based on what your previous steps produced."""


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------
class PlanningAgent(BaseAgent):
    """Plan-and-execute agent. Subclass and provide:

    * :meth:`base_system_prompt` — the user-facing role / persona
    * :meth:`tools` (optional) — work tools (search, DB, etc.).
      :class:`PlanningAgent` automatically prepends a synthetic
      ``_make_plan`` tool to whatever the subclass returns.

    The LLM declares its plan via the ``_make_plan`` tool call,
    then executes steps one at a time using the work tools. The
    plan and a history of revisions are persisted in
    :attr:`AgentState.scratchpad`.
    """

    #: Cap on plan revisions. The LLM gets a clear "max reached"
    #: nudge in its system prompt once the cap is hit so it stops
    #: trying to re-plan.
    max_replans: int = 3

    @inject
    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__(runtime)
        # Stash for ``system_prompt()`` and the plan-capture handler so
        # they can read / write per-turn state. Reset around each
        # ``_agent_node`` and ``_tool_node`` call (LangGraph runs nodes
        # serially within a single graph execution).
        self._current_state: AgentState | None = None
        self._pending_plan: Plan | None = None

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------
    @abstractmethod
    def base_system_prompt(self) -> str:
        """Return the user-facing system prompt for the agent.

        :class:`PlanningAgent` decorates this with plan-mode
        instructions and live plan status before sending to the LLM.
        """

    def planning_instructions(self) -> tuple[str, str, str, str]:
        """Hook for customising the four plan-state messages.

        Returns a 4-tuple of templates corresponding to:

        1. Initial (no plan yet) — used on the very first turn.
        2. Executing (plan in progress) — must contain
           ``{revision_n}`` and ``{max_replans}`` placeholders.
        3. Done (all steps complete).
        4. Replan cap reached.

        Override to adjust tone, force stricter behaviour, or
        localize.
        """
        return (
            _PLAN_INSTRUCTIONS_INITIAL,
            _PLAN_INSTRUCTIONS_EXECUTING,
            _PLAN_INSTRUCTIONS_DONE,
            _PLAN_INSTRUCTIONS_MAX_REPLANS,
        )

    # ------------------------------------------------------------------
    # System prompt — decorated with plan-mode + live status
    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        """Combine :meth:`base_system_prompt` with planning instructions.

        Reads ``self._current_state`` (set by overridden
        :meth:`_agent_node` and :meth:`_tool_node`) so the LLM sees
        the live plan status on every turn.
        """
        base = self.base_system_prompt()
        addendum = self._planning_addendum()
        return f"{base}\n\n{addendum}"

    def _planning_addendum(self) -> str:
        instr_initial, instr_exec, instr_done, instr_capped = (
            self.planning_instructions()
        )
        plan, revision = self._current_plan_and_revision()
        if plan is None:
            return instr_initial
        if revision >= self.max_replans and not self._all_steps_done(plan):
            return instr_capped + "\n\n" + self._render_plan(plan)
        if self._all_steps_done(plan):
            return instr_done + "\n\n" + self._render_plan(plan)
        return (
            instr_exec.format(revision_n=revision, max_replans=self.max_replans)
            + "\n"
            + self._render_plan(plan)
        )

    @staticmethod
    def _all_steps_done(plan: Plan) -> bool:
        return all(step.status == "done" for step in plan.steps)

    @staticmethod
    def _render_plan(plan: Plan) -> str:
        """Return a compact, LLM-friendly rendering of a plan."""
        lines = [f"Goal: {plan.goal}"]
        for step in plan.steps:
            mark = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "done": "[x]",
                "failed": "[!]",
            }[step.status]
            line = f"  {mark} {step.id}: {step.description}"
            if step.result:
                line += f"\n      result: {step.result}"
            if step.notes:
                line += f"\n      notes: {step.notes}"
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # tools() — prepend synthetic _make_plan to the subclass's tools
    # ------------------------------------------------------------------
    def tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Return the synthetic ``_make_plan`` tool plus user tools.

        Subclasses override :meth:`work_tools` to declare the
        actual tools they need; :class:`PlanningAgent` always
        prepends ``_make_plan``. Subclasses that need to extend the
        full tool list (rather than just adding work tools) can
        override :meth:`tools` directly and call ``super().tools()``.
        """
        from typing import cast
        plan_tool = self._build_make_plan_tool()
        user_tools = list(self.work_tools())
        # Cast for the same Tool-Protocol-vs-frozen-ToolSpec mismatch
        # we hit on SupervisorAgent.tools(); structurally compatible.
        return cast("list[Tool | Mapping[str, Any]]", [plan_tool, *user_tools])

    def work_tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Subclass-provided work tools (search, DB, HTTP, etc.).

        Override this rather than :meth:`tools` so the synthetic
        ``_make_plan`` tool is preserved.
        """
        return ()

    def _build_make_plan_tool(self) -> ToolSpec:
        """Synthetic tool for plan declaration / revision."""

        async def _handler(payload: BaseModel) -> BaseModel:
            assert isinstance(payload, Plan), (
                f"Plan handler expected Plan, got {type(payload).__name__}"
            )
            return self._on_plan_submitted(payload)

        return ToolSpec(
            name="_make_plan",
            version=1,
            description=(
                "Declare or revise the plan you intend to follow. Call "
                "this once at the start of work, and again whenever a "
                "step fails or your approach changes."
            ),
            input_model=Plan,
            output_model=PlanAck,
            handler=_cast_handler(_handler),
            opa_path=None,
        )

    def _on_plan_submitted(self, plan: Plan) -> PlanAck:
        """Stash the plan for ``_tool_node`` to merge into state.

        Returns a :class:`PlanAck` that the LLM sees as the tool's
        output on its next turn (confirms acceptance + revision
        count).
        """
        self._pending_plan = plan
        # Determine the upcoming revision number from current state.
        _, current_revision = self._current_plan_and_revision()
        upcoming = current_revision + 1
        note: str | None = None
        if upcoming > self.max_replans:
            note = (
                "This plan is past the configured replan cap; the "
                "system prompt on your next turn will instruct you to "
                "finalize without further revisions."
            )
        return PlanAck(
            accepted=True,
            revision=upcoming,
            step_count=len(plan.steps),
            note=note,
        )

    # ------------------------------------------------------------------
    # Graph nodes — stash state so system_prompt() and the plan
    # handler can read it.
    # ------------------------------------------------------------------
    async def _agent_node(self, state: AgentState) -> AgentState:
        self._current_state = state
        try:
            return await super()._agent_node(state)
        finally:
            self._current_state = None

    async def _tool_node(self, state: AgentState) -> AgentState:
        self._current_state = state
        self._pending_plan = None
        try:
            new_state = await super()._tool_node(state)
        finally:
            # Read the side-channel before clearing it. mypy's flow
            # analysis can't see that super()._tool_node may invoke the
            # _make_plan handler synchronously (which writes to
            # self._pending_plan); explicit annotation widens the type
            # back to Plan | None so the if-branch below isn't pruned.
            captured_plan: Plan | None = self._pending_plan
            self._current_state = None
            self._pending_plan = None
        if captured_plan is not None:
            # Merge the captured plan into scratchpad. History is
            # preserved so the LLM can see prior plans on
            # subsequent turns and re-plan informedly.
            current_scratchpad = dict(state.get("scratchpad") or {})
            plans_history: list[dict[str, Any]] = list(
                current_scratchpad.get("plans") or [],
            )
            plans_history.append(captured_plan.model_dump())
            current_scratchpad["plans"] = plans_history
            current_scratchpad["replan_count"] = len(plans_history)
            new_state["scratchpad"] = current_scratchpad
        return new_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _current_plan_and_revision(self) -> tuple[Plan | None, int]:
        """Read the latest plan + revision count from current state."""
        if self._current_state is None:
            return None, 0
        scratchpad = self._current_state.get("scratchpad") or {}
        plans = scratchpad.get("plans") or []
        if not plans:
            return None, 0
        return Plan.model_validate(plans[-1]), len(plans)


# ---------------------------------------------------------------------------
# Type-erased handler cast (mirrors SupervisorAgent's helper).
# ---------------------------------------------------------------------------
from collections.abc import Awaitable, Callable  # noqa: E402


def _cast_handler(
    fn: Callable[[BaseModel], Awaitable[BaseModel]],
) -> ToolHandler:
    """Cast the closure to :data:`ToolHandler` for ToolSpec.

    Both types are ``Callable[[BaseModel], Awaitable[BaseModel]]`` so
    the cast is a no-op at runtime; the helper exists to keep the
    pyright/mypy noise localised.
    """
    return fn


__all__ = ["Plan", "PlanAck", "PlanStep", "PlanningAgent", "StepStatus"]
