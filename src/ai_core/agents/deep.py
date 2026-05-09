"""Hierarchical planning + sub-agent dispatch + virtual filesystem.

:class:`DeepAgent` synthesises three primitives the SDK already ships:

* :class:`PlanningAgent`'s plan-as-tool-call pattern (the LLM declares a
  structured plan via a synthetic tool, plan history is persisted in
  ``state.scratchpad``),
* :class:`SupervisorAgent`'s children-as-tools pattern (the LLM delegates
  by emitting tool_calls; the SDK's existing ToolInvoker dispatches each
  call through validation + OPA + audit + observability),
* a virtual filesystem keyed by ``state.scratchpad["files"]`` that
  survives across sub-agent dispatches so the deep agent can persist
  intermediate artefacts (research notes, draft text, structured
  records).

What's different from the other primitives
==========================================
* The plan's steps reference *which sub-agent runs them*, not just "do
  the next pending step in-context". The deep agent decomposes work
  vertically, then dispatches each step to a specialist.
* Each ``_dispatch`` call runs a fresh :class:`BaseAgent` invocation
  with its own :class:`AgentState` — no message bleed-over between
  steps. Sub-agents communicate via the shared filesystem and via the
  string return value of the dispatch (which the deep agent's LLM sees
  on its next turn).
* Verification is **not** bundled. Hosts compose with
  :class:`VerifierAgent` either by wrapping individual sub-agents or by
  wrapping the whole :class:`DeepAgent`.

Why a separate ``_decompose`` tool (not ``_make_plan``)
=======================================================
:class:`PlanningAgent` registers a synthetic ``_make_plan`` tool with
schema :class:`Plan`. :class:`DeepAgent`'s plan adds a per-step
``sub_agent`` field; registering a different schema under the same
``(name, version)`` pair raises
:class:`ai_core.exceptions.SchemaValidationError`. Using a distinct
tool name (``_decompose``) keeps the registries cleanly separated and
lets a single host run both primitives in the same process.

Composition
===========
* a :class:`DeepAgent`'s sub-agents can themselves be
  :class:`SupervisorAgent`, :class:`PlanningAgent`, or another
  :class:`DeepAgent` — recursion is bounded per-instance by
  :attr:`AgentSettings.max_recursion_depth`.
* a :class:`VerifierAgent` may wrap a sub-agent (verify each step) or
  wrap the whole :class:`DeepAgent` (verify the final answer).
* a :class:`HarnessAgent` wraps a :class:`DeepAgent` to record every
  LLM + tool dispatch — including each sub-agent's own LLM calls,
  because they run through the same runtime's invoker.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from injector import inject
from pydantic import BaseModel, Field

from ai_core.agents.base import BaseAgent
from ai_core.agents.planning import StepStatus
from ai_core.agents.runtime import AgentRuntime
from ai_core.agents.state import AgentState
from ai_core.exceptions import RegistryError
from ai_core.tools.spec import ToolHandler, ToolSpec

if TYPE_CHECKING:
    from ai_core.tools.spec import Tool


# ---------------------------------------------------------------------------
# Plan data model
# ---------------------------------------------------------------------------
class DeepPlanStep(BaseModel):
    """A single step in a deep-agent plan.

    Attributes:
        id: Stable identifier (e.g. ``"step-1"``). Useful when the LLM
            references a specific step in a revised plan.
        description: Human-readable description of the step's intent.
        sub_agent: Name of the sub-agent that runs this step. ``None``
            means "deep agent handles in its own context" (e.g. a final
            assembly step that doesn't need delegation). The name must
            match a key returned by :meth:`DeepAgent.sub_agents`.
        status: Lifecycle marker — ``pending`` initially, transitions
            to ``done`` / ``failed`` as the LLM works through the
            steps. The LLM revises status by calling ``_decompose``
            again with the updated step list.
        result: Free-form text the LLM populates after the step
            completes. Optional.
        notes: Free-form notes (blockers, why a step was reordered).
    """

    id: str
    description: str
    sub_agent: str | None = None
    status: StepStatus = "pending"
    result: str | None = None
    notes: str | None = None


class DeepPlan(BaseModel):
    """Top-level plan for a deep-agent run."""

    goal: str = Field(..., description="The user-level goal in one short sentence.")
    steps: list[DeepPlanStep] = Field(
        ..., description="Ordered sequence of steps with optional sub-agent assignment.",
    )


class _PlanAck(BaseModel):
    """Acknowledgement returned by ``_decompose`` after a plan is captured."""

    accepted: bool
    revision: int
    step_count: int


# ---------------------------------------------------------------------------
# Dispatch + filesystem tool I/O models
# ---------------------------------------------------------------------------
class _DispatchIn(BaseModel):
    """Input schema for the synthetic ``_dispatch`` tool."""

    sub_agent: str = Field(..., description="Name of the sub-agent to run.")
    task: str = Field(..., description="The task to delegate, in plain text.")
    context: str | None = Field(
        default=None,
        description="Optional extra context for the sub-agent (e.g. a "
        "summary of relevant files already in scratch).",
    )
    step_id: str | None = Field(
        default=None,
        description="Optional plan step id this dispatch fulfils. The "
        "deep agent links the dispatch result to the step in scratchpad.",
    )


class _DispatchOut(BaseModel):
    """Output schema for ``_dispatch``: the sub-agent's final assistant message."""

    sub_agent: str
    result: str


class _FileWriteIn(BaseModel):
    """Input schema for ``_write_file``."""

    path: str = Field(..., description="Path inside the virtual filesystem.")
    content: str = Field(..., description="UTF-8 text content. Overwrites existing.")


class _FileWriteAck(BaseModel):
    """Acknowledgement: how many characters landed at ``path``."""

    path: str
    chars_written: int


class _FileReadIn(BaseModel):
    """Input schema for ``_read_file``."""

    path: str


class _FileReadOut(BaseModel):
    """Output schema for ``_read_file``."""

    path: str
    content: str
    found: bool


class _FileListIn(BaseModel):
    """Empty input schema for ``_list_files``.

    Pydantic requires a model class even for tools that take no arguments;
    the LLM emits ``{}`` and the registrar accepts it.
    """


class _FileListOut(BaseModel):
    """Output schema for ``_list_files``: the current set of paths."""

    paths: list[str]


# ---------------------------------------------------------------------------
# System prompt scaffolding
# ---------------------------------------------------------------------------
_DEEP_INSTRUCTIONS_INITIAL = """\
[Deep-agent: planning mode]
Decompose the user's request into discrete steps and call the
`_decompose` tool with a DeepPlan. For each step assign a `sub_agent`
from the available roster (or leave `sub_agent=null` for steps you
will handle in your own context). After the plan is accepted, you'll
execute each step by calling `_dispatch` for the next pending one.

Available sub-agents:
{sub_agent_lines}"""

_DEEP_INSTRUCTIONS_EXECUTING = """\
[Deep-agent: execution mode]
Your plan is below. On each turn, advance one pending step:
- For a step with `sub_agent` set, call `_dispatch(sub_agent=..., task=..., step_id=...)`.
- For a step with `sub_agent=null`, do the work directly in your reply
  (or use `_write_file` / `_read_file` to manage shared state).
Use `_write_file` / `_read_file` / `_list_files` for durable scratch
that survives across sub-agent dispatches. If the plan needs revision
(failed steps, new context), call `_decompose` again with an updated
DeepPlan.

Available sub-agents:
{sub_agent_lines}

Current plan ({revision_n} of {max_replans} revisions used):
{plan_render}

Files in scratch ({file_count}):
{file_list}

Recent dispatches:
{dispatch_history_render}"""

_DEEP_INSTRUCTIONS_DONE = """\
[Deep-agent: plan complete]
Every step in your plan is marked `done`. Either emit your final
answer to the user (assembling from files / dispatch results as
needed), or call `_decompose` again if you believe more work is
required.

Files in scratch:
{file_list}"""

_DEEP_INSTRUCTIONS_MAX_REPLANS = """\
[Deep-agent: replan cap reached]
You have reached the configured maximum number of plan revisions. Do
NOT call `_decompose` again. Finalize your answer based on what your
sub-agent dispatches and files have produced so far."""


# ---------------------------------------------------------------------------
# DeepAgent
# ---------------------------------------------------------------------------
class DeepAgent(BaseAgent):
    """Hierarchical planning + sub-agent dispatch + virtual filesystem.

    Subclass and provide:

    * :meth:`base_system_prompt` — the persona / role
    * :meth:`sub_agents` — the named map of available sub-agent
      classes, parallel to :meth:`SupervisorAgent.children`

    The deep agent automatically prepends synthetic
    ``_decompose`` / ``_dispatch`` / ``_write_file`` / ``_read_file`` /
    ``_list_files`` tools to whatever :meth:`work_tools` returns, and
    wires them into ``state.scratchpad`` keys:

    * ``scratchpad["plan"]`` — the most recent :class:`DeepPlan` (dict)
    * ``scratchpad["plan_history"]`` — list of all revisions
    * ``scratchpad["files"]`` — the virtual filesystem (``dict[str, str]``)
    * ``scratchpad["dispatch_history"]`` — list of dispatch records
    """

    #: Cap on plan revisions; mirrors :class:`PlanningAgent.max_replans`.
    max_replans: int = 3

    @inject
    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__(runtime)
        # Per-call stash so the synthetic tool handlers can mutate
        # state. LangGraph runs nodes serially within one graph
        # invocation, so a single instance variable is safe.
        self._current_state: AgentState | None = None
        # Side-channels populated by tool handlers in `_tool_node`,
        # drained back into state after super delegate returns.
        self._pending_plan: DeepPlan | None = None
        self._pending_files: dict[str, str] = {}
        self._pending_dispatches: list[dict[str, Any]] = []
        # Cache of resolved sub-agent instances; same intent as
        # SupervisorAgent's child cache (graph compiles at most once
        # per sub-agent across a single deep-agent invocation).
        self._sub_agent_instances: dict[str, BaseAgent] = {}

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------
    @abstractmethod
    def base_system_prompt(self) -> str:
        """Return the user-facing system prompt for the deep agent.

        :class:`DeepAgent` decorates this with the planning / execution
        instructions and live state on every turn.
        """

    @abstractmethod
    def sub_agents(self) -> Mapping[str, type[BaseAgent]]:
        """Return the map of sub-agent name → :class:`BaseAgent` subclass.

        The plan's :attr:`DeepPlanStep.sub_agent` field references these
        names; ``_dispatch`` looks them up here.
        """

    def work_tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Subclass-provided work tools (in addition to the synthetic ones).

        Override to expose tools the deep agent itself uses directly
        (without delegating to a sub-agent). Default: ``()`` — every
        action goes through ``_dispatch`` or the file tools.
        """
        return ()

    # ------------------------------------------------------------------
    # System prompt — decorated with live state
    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        base = self.base_system_prompt()
        addendum = self._render_addendum()
        return f"{base}\n\n{addendum}"

    def _render_addendum(self) -> str:
        plan, revision = self._current_plan_and_revision()
        sub_agent_lines = self._render_sub_agent_lines()
        if plan is None:
            return _DEEP_INSTRUCTIONS_INITIAL.format(
                sub_agent_lines=sub_agent_lines,
            )
        files = self._current_files()
        if revision >= self.max_replans and not self._all_steps_done(plan):
            return _DEEP_INSTRUCTIONS_MAX_REPLANS
        if self._all_steps_done(plan):
            return _DEEP_INSTRUCTIONS_DONE.format(
                file_list=self._render_file_list(files),
            )
        return _DEEP_INSTRUCTIONS_EXECUTING.format(
            sub_agent_lines=sub_agent_lines,
            revision_n=revision,
            max_replans=self.max_replans,
            plan_render=self._render_plan(plan),
            file_count=len(files),
            file_list=self._render_file_list(files),
            dispatch_history_render=self._render_dispatch_history(),
        )

    def _render_sub_agent_lines(self) -> str:
        roster = self.sub_agents()
        if not roster:
            return "  (none — every step must run inline with sub_agent=null)"
        lines = []
        for name, cls in roster.items():
            doc = (
                cls.__doc__.strip().splitlines()[0]
                if cls.__doc__ else "(no description)"
            )
            lines.append(f"  - {name}: {doc}")
        return "\n".join(lines)

    @staticmethod
    def _all_steps_done(plan: DeepPlan) -> bool:
        return all(step.status == "done" for step in plan.steps)

    @staticmethod
    def _render_plan(plan: DeepPlan) -> str:
        lines = [f"Goal: {plan.goal}"]
        for step in plan.steps:
            mark = {
                "pending": "[ ]", "in_progress": "[~]",
                "done": "[x]", "failed": "[!]",
            }[step.status]
            sa = f" → {step.sub_agent}" if step.sub_agent else " (inline)"
            line = f"  {mark} {step.id}{sa}: {step.description}"
            if step.result:
                line += f"\n      result: {step.result}"
            if step.notes:
                line += f"\n      notes: {step.notes}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _render_file_list(files: Mapping[str, str]) -> str:
        if not files:
            return "  (empty)"
        return "\n".join(
            f"  - {path} ({len(content)} chars)"
            for path, content in files.items()
        )

    def _render_dispatch_history(self) -> str:
        if self._current_state is None:
            return "  (none)"
        history = (self._current_state.get("scratchpad") or {}).get(
            "dispatch_history", [],
        )
        if not history:
            return "  (none yet)"
        recent = history[-5:]
        lines = []
        for entry in recent:
            lines.append(
                f"  - step={entry.get('step_id') or '-'} "
                f"sub_agent={entry['sub_agent']}: "
                f"{entry['result'][:80]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # tools() — synthetic + work tools
    # ------------------------------------------------------------------
    def tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Return ``[_decompose, _dispatch, _write_file, _read_file,
        _list_files, *work_tools()]``.

        Subclasses override :meth:`work_tools` rather than :meth:`tools`
        so the synthetic primitives stay registered automatically.
        """
        from typing import cast
        synthetic = [
            self._build_decompose_tool(),
            self._build_dispatch_tool(),
            self._build_write_file_tool(),
            self._build_read_file_tool(),
            self._build_list_files_tool(),
        ]
        # Cast for the same Tool-Protocol-vs-frozen-ToolSpec mismatch
        # we hit on SupervisorAgent.tools() / PlanningAgent.tools();
        # the runtime types are structurally compatible.
        return cast(
            "list[Tool | Mapping[str, Any]]",
            [*synthetic, *self.work_tools()],
        )

    def _build_decompose_tool(self) -> ToolSpec:
        async def _handler(payload: BaseModel) -> BaseModel:
            assert isinstance(payload, DeepPlan)
            return self._on_plan_submitted(payload)

        return ToolSpec(
            name="_decompose",
            version=1,
            description=(
                "Declare or revise the deep agent's plan. Each step "
                "may name a sub_agent that handles it; leave "
                "sub_agent=null for steps the deep agent runs inline. "
                "Call this once at the start, and again whenever a step "
                "fails or your approach changes."
            ),
            input_model=DeepPlan,
            output_model=_PlanAck,
            handler=_cast_handler(_handler),
            opa_path=None,
        )

    def _build_dispatch_tool(self) -> ToolSpec:
        async def _handler(payload: BaseModel) -> BaseModel:
            assert isinstance(payload, _DispatchIn)
            return await self._on_dispatch(payload)

        return ToolSpec(
            name="_dispatch",
            version=1,
            description=(
                "Run a sub-agent on a sub-task and return its final "
                "assistant message. Each dispatch runs the sub-agent in "
                "an isolated context (fresh message history); use "
                "_write_file beforehand to share state across dispatches."
            ),
            input_model=_DispatchIn,
            output_model=_DispatchOut,
            handler=_cast_handler(_handler),
            opa_path=None,
        )

    def _build_write_file_tool(self) -> ToolSpec:
        async def _handler(payload: BaseModel) -> BaseModel:
            assert isinstance(payload, _FileWriteIn)
            return self._on_write_file(payload)

        return ToolSpec(
            name="_write_file",
            version=1,
            description=(
                "Write text content to a path in the virtual filesystem. "
                "Overwrites any existing content at the path. The "
                "filesystem persists across sub-agent dispatches within "
                "this run."
            ),
            input_model=_FileWriteIn,
            output_model=_FileWriteAck,
            handler=_cast_handler(_handler),
            opa_path=None,
        )

    def _build_read_file_tool(self) -> ToolSpec:
        async def _handler(payload: BaseModel) -> BaseModel:
            assert isinstance(payload, _FileReadIn)
            return self._on_read_file(payload)

        return ToolSpec(
            name="_read_file",
            version=1,
            description=(
                "Read text content from a path in the virtual filesystem. "
                "Returns found=false with empty content when the path "
                "does not exist."
            ),
            input_model=_FileReadIn,
            output_model=_FileReadOut,
            handler=_cast_handler(_handler),
            opa_path=None,
        )

    def _build_list_files_tool(self) -> ToolSpec:
        async def _handler(payload: BaseModel) -> BaseModel:
            assert isinstance(payload, _FileListIn)
            return self._on_list_files()

        return ToolSpec(
            name="_list_files",
            version=1,
            description="List paths currently present in the virtual filesystem.",
            input_model=_FileListIn,
            output_model=_FileListOut,
            handler=_cast_handler(_handler),
            opa_path=None,
        )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------
    def _on_plan_submitted(self, plan: DeepPlan) -> _PlanAck:
        self._pending_plan = plan
        _, current_revision = self._current_plan_and_revision()
        upcoming = current_revision + 1
        return _PlanAck(
            accepted=True,
            revision=upcoming,
            step_count=len(plan.steps),
        )

    async def _on_dispatch(self, payload: _DispatchIn) -> _DispatchOut:
        roster = self.sub_agents()
        sub_cls = roster.get(payload.sub_agent)
        if sub_cls is None:
            available = ", ".join(sorted(roster.keys())) or "(none)"
            raise RegistryError(
                f"Unknown sub_agent {payload.sub_agent!r}. "
                f"Available: {available}.",
                details={
                    "sub_agent": payload.sub_agent,
                    "available": list(roster.keys()),
                },
            )
        sub = self._resolve_sub_agent(payload.sub_agent, sub_cls)

        # Build the user message for the sub-agent. The dispatch context
        # (if provided) goes after the task as a separate paragraph so
        # the sub-agent's prompt structure stays predictable.
        user_content = payload.task
        if payload.context:
            user_content = f"{payload.task}\n\nContext: {payload.context}"

        state = self._current_state or {}
        essentials = dict(state.get("essential_entities") or {})
        tenant_id = str(essentials.get("tenant_id") or "") or None
        thread_id = essentials.get("thread_id") or None

        sub_essentials = {
            **essentials,
            "delegated_by": self.agent_id,
            "delegation_target": payload.sub_agent,
        }

        sub_state = await sub.ainvoke(
            messages=[{"role": "user", "content": user_content}],
            essential=sub_essentials,
            tenant_id=tenant_id,
            thread_id=str(thread_id) if thread_id is not None else None,
        )

        # Extract the sub-agent's final assistant message.
        result_text = self._extract_last_assistant_text(sub_state)

        # Record the dispatch in the side-channel; _tool_node will
        # merge it into state.scratchpad after super delegate returns.
        self._pending_dispatches.append({
            "step_id": payload.step_id,
            "sub_agent": payload.sub_agent,
            "task": payload.task,
            "result": result_text,
        })
        return _DispatchOut(sub_agent=payload.sub_agent, result=result_text)

    def _on_write_file(self, payload: _FileWriteIn) -> _FileWriteAck:
        self._pending_files[payload.path] = payload.content
        return _FileWriteAck(
            path=payload.path,
            chars_written=len(payload.content),
        )

    def _on_read_file(self, payload: _FileReadIn) -> _FileReadOut:
        # Read sees both the persisted state and any writes already
        # queued in this same _tool_node call (e.g. write-then-read in a
        # single LLM turn).
        files = self._current_files()
        merged = {**files, **self._pending_files}
        if payload.path in merged:
            return _FileReadOut(path=payload.path, content=merged[payload.path], found=True)
        return _FileReadOut(path=payload.path, content="", found=False)

    def _on_list_files(self) -> _FileListOut:
        files = self._current_files()
        merged = {**files, **self._pending_files}
        return _FileListOut(paths=sorted(merged.keys()))

    # ------------------------------------------------------------------
    # Graph nodes — stash state + drain side-channels back into it
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
        self._pending_files = {}
        self._pending_dispatches = []
        try:
            new_state = await super()._tool_node(state)
        finally:
            # Capture side-channels before clearing them. Mirrors the
            # PlanningAgent pattern with explicit type annotations to
            # keep mypy's flow analysis from pruning the merge branches.
            captured_plan: DeepPlan | None = self._pending_plan
            captured_files: dict[str, str] = dict(self._pending_files)
            captured_dispatches: list[dict[str, Any]] = list(self._pending_dispatches)
            self._current_state = None
            self._pending_plan = None
            self._pending_files = {}
            self._pending_dispatches = []

        if captured_plan is None and not captured_files and not captured_dispatches:
            return new_state

        scratchpad = dict(state.get("scratchpad") or {})
        if captured_plan is not None:
            scratchpad["plan"] = captured_plan.model_dump()
            history: list[dict[str, Any]] = list(scratchpad.get("plan_history") or [])
            history.append(captured_plan.model_dump())
            scratchpad["plan_history"] = history
            scratchpad["replan_count"] = len(history)
        if captured_files:
            files: dict[str, str] = dict(scratchpad.get("files") or {})
            files.update(captured_files)
            scratchpad["files"] = files
        if captured_dispatches:
            dispatch_history: list[dict[str, Any]] = list(
                scratchpad.get("dispatch_history") or [],
            )
            dispatch_history.extend(captured_dispatches)
            scratchpad["dispatch_history"] = dispatch_history
        new_state["scratchpad"] = scratchpad
        return new_state

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def _current_plan_and_revision(self) -> tuple[DeepPlan | None, int]:
        if self._current_state is None:
            return None, 0
        scratchpad = self._current_state.get("scratchpad") or {}
        plan_raw = scratchpad.get("plan")
        if plan_raw is None:
            return None, 0
        revision = int(scratchpad.get("replan_count") or 1)
        return DeepPlan.model_validate(plan_raw), revision

    def _current_files(self) -> Mapping[str, str]:
        if self._current_state is None:
            return {}
        return dict((self._current_state.get("scratchpad") or {}).get("files") or {})

    # ------------------------------------------------------------------
    # Sub-agent resolution + caching
    # ------------------------------------------------------------------
    def _resolve_sub_agent(
        self, name: str, cls: type[BaseAgent],
    ) -> BaseAgent:
        existing = self._sub_agent_instances.get(name)
        if existing is not None:
            return existing
        instance = self._runtime.agent_resolver.resolve(cls)
        if not isinstance(instance, BaseAgent):
            raise RegistryError(
                f"Sub-agent {name!r} resolved to {type(instance).__name__}; "
                f"expected a BaseAgent subclass.",
                details={"sub_agent": name, "resolved_type": type(instance).__name__},
            )
        self._sub_agent_instances[name] = instance
        return instance

    @staticmethod
    def _extract_last_assistant_text(state: AgentState) -> str:
        """Pull the last assistant message from a sub-agent's final state.

        Handles both LangChain ``AIMessage`` (``type == "ai"``) and
        plain dicts; mirrors :meth:`VerifierAgent._extract_last_assistant_text`.
        """
        messages: list[Any] = list(state.get("messages") or [])
        for msg in reversed(messages):
            if isinstance(msg, Mapping):
                if msg.get("role") == "assistant":
                    return str(msg.get("content") or "")
                continue
            if getattr(msg, "type", None) == "ai":
                return str(getattr(msg, "content", "") or "")
        return ""


# ---------------------------------------------------------------------------
# Type-erased handler cast (mirrors SupervisorAgent / PlanningAgent helpers)
# ---------------------------------------------------------------------------
def _cast_handler(
    fn: Callable[[BaseModel], Awaitable[BaseModel]],
) -> ToolHandler:
    """Cast the closure to :data:`ToolHandler` for ``ToolSpec``.

    Both types are ``Callable[[BaseModel], Awaitable[BaseModel]]`` so
    the cast is a no-op at runtime; the helper localises the
    pyright/mypy noise.
    """
    return fn


__all__ = ["DeepAgent", "DeepPlan", "DeepPlanStep"]
