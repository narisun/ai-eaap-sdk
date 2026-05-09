"""Unit tests for :class:`DeepAgent`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from ai_core.agents import (
    AgentRuntime,
    AgentState,
    BaseAgent,
    DeepAgent,
    DeepPlan,
    DeepPlanStep,
)
from ai_core.agents._resolver import AgentResolver
from ai_core.agents.tool_errors import DefaultToolErrorRenderer
from ai_core.audit.null import NullAuditSink
from ai_core.config.settings import AppSettings
from ai_core.exceptions import RegistryError
from ai_core.testing import FakeObservabilityProvider
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.registrar import ToolRegistrar
from ai_core.tools.resolver import DefaultToolResolver
from ai_core.tools.spec import ToolSpec

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeSubAgent(BaseAgent):
    """A sub-agent that records dispatch calls and returns scripted replies."""

    agent_id = "fake-sub"

    def __init__(
        self,
        runtime: AgentRuntime,
        replies: Sequence[str] = ("default reply",),
    ) -> None:
        super().__init__(runtime)
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def system_prompt(self) -> str:
        return "fake sub"

    async def ainvoke(  # type: ignore[override]
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        essential: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        self.calls.append({
            "messages": list(messages),
            "essential": dict(essential or {}),
            "tenant_id": tenant_id,
            "thread_id": thread_id,
        })
        idx = min(len(self.calls) - 1, len(self._replies) - 1)
        return {
            "messages": [AIMessage(content=self._replies[idx])],
            "essential_entities": {},
            "scratchpad": {},
            "metadata": {},
            "token_count": 0,
            "compaction_count": 0,
            "summary": "",
        }


class _Researcher(_FakeSubAgent):
    agent_id = "researcher"


class _Writer(_FakeSubAgent):
    agent_id = "writer"


class _DemoDeep(DeepAgent):
    agent_id = "demo-deep"

    def base_system_prompt(self) -> str:
        return "You orchestrate research and writing."

    def sub_agents(self) -> Mapping[str, type[BaseAgent]]:
        return {"researcher": _Researcher, "writer": _Writer}


class _NoSubsDeep(DeepAgent):
    agent_id = "no-subs-deep"

    def base_system_prompt(self) -> str:
        return "Solo deep agent."

    def sub_agents(self) -> Mapping[str, type[BaseAgent]]:
        return {}


# ---------------------------------------------------------------------------
# Runtime helper
# ---------------------------------------------------------------------------
def _make_runtime(*, sub_agent: BaseAgent | None = None) -> AgentRuntime:
    """Build a runtime with a stub resolver returning ``sub_agent`` for any class."""
    invoker = ToolInvoker(
        observability=FakeObservabilityProvider(),
        audit=NullAuditSink(),
    )

    class _StubResolver(AgentResolver):
        def __init__(self) -> None:
            pass

        def resolve(self, cls):  # type: ignore[override]
            return sub_agent if sub_agent is not None else MagicMock()

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
def test_tools_prepends_five_synthetic_tools_before_work_tools() -> None:
    """tools() advertises _decompose, _dispatch, _write_file, _read_file,
    _list_files in order, then any work_tools."""
    deep = _DemoDeep(_make_runtime())
    tools = deep.tools()

    assert len(tools) == 5
    names = [t.name for t in tools if isinstance(t, ToolSpec)]
    assert names == [
        "_decompose", "_dispatch", "_write_file", "_read_file", "_list_files",
    ]


def test_synthetic_tools_use_correct_io_models() -> None:
    deep = _DemoDeep(_make_runtime())
    tools = deep.tools()
    by_name: dict[str, ToolSpec] = {
        t.name: t for t in tools if isinstance(t, ToolSpec)
    }

    assert by_name["_decompose"].input_model is DeepPlan
    assert by_name["_decompose"].opa_path is None
    # The other synthetic tools' models are private; we just check
    # they're set to *some* Pydantic class (smoke).
    for name in ("_dispatch", "_write_file", "_read_file", "_list_files"):
        spec = by_name[name]
        assert spec.input_model is not None
        assert spec.output_model is not None


# ---------------------------------------------------------------------------
# system_prompt() decoration
# ---------------------------------------------------------------------------
def test_system_prompt_initial_mode_lists_sub_agents() -> None:
    deep = _DemoDeep(_make_runtime())
    prompt = deep.system_prompt()
    assert "[Deep-agent: planning mode]" in prompt
    assert "researcher" in prompt
    assert "writer" in prompt
    assert "orchestrate research and writing" in prompt


def test_system_prompt_initial_mode_reports_empty_roster() -> None:
    deep = _NoSubsDeep(_make_runtime())
    prompt = deep.system_prompt()
    assert "[Deep-agent: planning mode]" in prompt
    assert "(none — every step must run inline" in prompt


def test_system_prompt_executing_mode_renders_plan_and_files() -> None:
    deep = _DemoDeep(_make_runtime())
    plan = DeepPlan(
        goal="Write a brief on widgets",
        steps=[
            DeepPlanStep(
                id="step-1", description="Research", sub_agent="researcher",
                status="done", result="found 5 sources",
            ),
            DeepPlanStep(
                id="step-2", description="Draft", sub_agent="writer",
            ),
            DeepPlanStep(
                id="step-3", description="Final review", sub_agent=None,
            ),
        ],
    )
    deep._current_state = {
        "scratchpad": {
            "plan": plan.model_dump(),
            "replan_count": 1,
            "files": {"notes.md": "5 sources"},
        },
    }
    prompt = deep.system_prompt()

    assert "[Deep-agent: execution mode]" in prompt
    assert "[x] step-1 → researcher" in prompt
    assert "[ ] step-2 → writer" in prompt
    assert "[ ] step-3 (inline)" in prompt
    assert "1 of 3 revisions used" in prompt
    assert "notes.md" in prompt
    assert "found 5 sources" in prompt


def test_system_prompt_done_mode_when_all_steps_done() -> None:
    deep = _DemoDeep(_make_runtime())
    plan = DeepPlan(
        goal="Simple",
        steps=[DeepPlanStep(
            id="s1", description="d", sub_agent="researcher",
            status="done", result="ok",
        )],
    )
    deep._current_state = {
        "scratchpad": {"plan": plan.model_dump(), "replan_count": 1},
    }
    prompt = deep.system_prompt()
    assert "[Deep-agent: plan complete]" in prompt


def test_system_prompt_max_replans_message_when_cap_hit() -> None:
    deep = _DemoDeep(_make_runtime())
    deep.max_replans = 2
    plan = DeepPlan(
        goal="Hard",
        steps=[DeepPlanStep(
            id="s1", description="pending", sub_agent="researcher",
        )],
    )
    deep._current_state = {
        "scratchpad": {
            "plan": plan.model_dump(),
            "replan_count": 2,
        },
    }
    prompt = deep.system_prompt()
    assert "[Deep-agent: replan cap reached]" in prompt
    assert "NOT call `_decompose` again" in prompt


# ---------------------------------------------------------------------------
# _decompose handler
# ---------------------------------------------------------------------------
def test_decompose_handler_stashes_pending_plan() -> None:
    deep = _DemoDeep(_make_runtime())
    plan = DeepPlan(
        goal="g",
        steps=[DeepPlanStep(id="s1", description="d", sub_agent="researcher")],
    )

    ack = deep._on_plan_submitted(plan)
    assert ack.accepted is True
    assert ack.revision == 1
    assert ack.step_count == 1
    assert deep._pending_plan is plan


# ---------------------------------------------------------------------------
# _dispatch handler
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dispatch_invokes_named_sub_agent_and_records_history() -> None:
    fake_sub = _Researcher(_make_runtime(), replies=["found 3 sources"])
    deep = _DemoDeep(_make_runtime(sub_agent=fake_sub))
    deep._current_state = {
        "essential_entities": {"tenant_id": "acme", "user_id": "u1"},
        "scratchpad": {},
    }
    from ai_core.agents.deep import _DispatchIn

    out = await deep._on_dispatch(_DispatchIn(
        sub_agent="researcher",
        task="Find sources on widgets",
        context="Focus on 2024.",
        step_id="step-1",
    ))

    assert out.sub_agent == "researcher"
    assert out.result == "found 3 sources"
    # Sub-agent was called with the rendered user message + propagated essentials.
    assert len(fake_sub.calls) == 1
    call = fake_sub.calls[0]
    assert call["tenant_id"] == "acme"
    assert call["essential"]["tenant_id"] == "acme"
    assert call["essential"]["user_id"] == "u1"
    assert call["essential"]["delegated_by"] == "demo-deep"
    assert call["essential"]["delegation_target"] == "researcher"
    assert call["messages"][0]["content"] == (
        "Find sources on widgets\n\nContext: Focus on 2024."
    )
    # Recorded in the side-channel for _tool_node to drain.
    assert len(deep._pending_dispatches) == 1
    rec = deep._pending_dispatches[0]
    assert rec["step_id"] == "step-1"
    assert rec["sub_agent"] == "researcher"
    assert rec["result"] == "found 3 sources"


@pytest.mark.asyncio
async def test_dispatch_unknown_sub_agent_raises_registry_error() -> None:
    deep = _DemoDeep(_make_runtime())
    deep._current_state = {"essential_entities": {}, "scratchpad": {}}
    from ai_core.agents.deep import _DispatchIn

    with pytest.raises(RegistryError) as ei:
        await deep._on_dispatch(_DispatchIn(
            sub_agent="ghost", task="impossible",
        ))

    assert "Unknown sub_agent" in ei.value.message
    assert "researcher" in ei.value.message  # available list surfaced
    assert "writer" in ei.value.message
    assert ei.value.details["sub_agent"] == "ghost"
    assert set(ei.value.details["available"]) == {"researcher", "writer"}


@pytest.mark.asyncio
async def test_dispatch_caches_sub_agent_instance_across_calls() -> None:
    """Same sub-agent instance reused across multiple dispatches in a run."""
    fake_sub = _Researcher(_make_runtime(), replies=["one", "two"])
    deep = _DemoDeep(_make_runtime(sub_agent=fake_sub))
    deep._current_state = {"essential_entities": {}, "scratchpad": {}}
    from ai_core.agents.deep import _DispatchIn

    await deep._on_dispatch(_DispatchIn(sub_agent="researcher", task="t1"))
    await deep._on_dispatch(_DispatchIn(sub_agent="researcher", task="t2"))

    assert deep._sub_agent_instances["researcher"] is fake_sub
    # _resolve_sub_agent only called the resolver once (cache hit on call 2).
    assert len(fake_sub.calls) == 2


# ---------------------------------------------------------------------------
# File handlers
# ---------------------------------------------------------------------------
def test_write_file_records_in_pending_files() -> None:
    deep = _DemoDeep(_make_runtime())
    from ai_core.agents.deep import _FileWriteIn

    ack = deep._on_write_file(_FileWriteIn(path="notes.md", content="hello"))
    assert ack.path == "notes.md"
    assert ack.chars_written == 5
    assert deep._pending_files["notes.md"] == "hello"


def test_read_file_returns_persisted_content() -> None:
    deep = _DemoDeep(_make_runtime())
    deep._current_state = {
        "scratchpad": {"files": {"a.txt": "alpha"}},
    }
    from ai_core.agents.deep import _FileReadIn

    out = deep._on_read_file(_FileReadIn(path="a.txt"))
    assert out.found is True
    assert out.content == "alpha"


def test_read_file_sees_in_flight_writes() -> None:
    """A write earlier in the same _tool_node call is visible to a
    subsequent read in the same call (write-then-read pattern)."""
    deep = _DemoDeep(_make_runtime())
    deep._current_state = {"scratchpad": {"files": {}}}
    from ai_core.agents.deep import _FileReadIn, _FileWriteIn

    deep._on_write_file(_FileWriteIn(path="draft.md", content="in-progress"))
    out = deep._on_read_file(_FileReadIn(path="draft.md"))
    assert out.found is True
    assert out.content == "in-progress"


def test_read_missing_file_returns_not_found() -> None:
    deep = _DemoDeep(_make_runtime())
    deep._current_state = {"scratchpad": {}}
    from ai_core.agents.deep import _FileReadIn

    out = deep._on_read_file(_FileReadIn(path="ghost.md"))
    assert out.found is False
    assert out.content == ""


def test_list_files_merges_persisted_and_pending() -> None:
    deep = _DemoDeep(_make_runtime())
    deep._current_state = {
        "scratchpad": {"files": {"old.md": "x"}},
    }
    from ai_core.agents.deep import _FileWriteIn

    deep._on_write_file(_FileWriteIn(path="new.md", content="y"))
    out = deep._on_list_files()
    assert out.paths == ["new.md", "old.md"]


# ---------------------------------------------------------------------------
# _tool_node merges side-channels into scratchpad
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_node_merges_plan_into_scratchpad() -> None:
    deep = _DemoDeep(_make_runtime())
    plan = DeepPlan(
        goal="g",
        steps=[DeepPlanStep(id="s1", description="d", sub_agent="researcher")],
    )

    async def _stub_super(self, state):  # type: ignore[no-untyped-def]
        self._pending_plan = plan
        return {"messages": [{"role": "tool", "content": "ok"}]}

    with patch("ai_core.agents.base.BaseAgent._tool_node", _stub_super):
        new_state = await deep._tool_node({"scratchpad": {}})

    sp = new_state["scratchpad"]
    assert sp["plan"]["goal"] == "g"
    assert len(sp["plan_history"]) == 1
    assert sp["replan_count"] == 1


@pytest.mark.asyncio
async def test_tool_node_merges_files_and_dispatches_into_scratchpad() -> None:
    deep = _DemoDeep(_make_runtime())

    async def _stub_super(self, state):  # type: ignore[no-untyped-def]
        self._pending_files = {"notes.md": "n", "draft.md": "d"}
        self._pending_dispatches = [{
            "step_id": "step-1", "sub_agent": "researcher",
            "task": "research", "result": "done",
        }]
        return {"messages": []}

    initial: AgentState = {
        "scratchpad": {
            "files": {"old.md": "old"},
            "dispatch_history": [{
                "step_id": "step-0", "sub_agent": "writer",
                "task": "earlier", "result": "earlier-result",
            }],
        },
    }
    with patch("ai_core.agents.base.BaseAgent._tool_node", _stub_super):
        new_state = await deep._tool_node(initial)

    sp = new_state["scratchpad"]
    # Existing file preserved; new files added.
    assert sp["files"] == {"old.md": "old", "notes.md": "n", "draft.md": "d"}
    # Dispatch history appended (not replaced).
    assert len(sp["dispatch_history"]) == 2
    assert sp["dispatch_history"][0]["step_id"] == "step-0"
    assert sp["dispatch_history"][1]["step_id"] == "step-1"


@pytest.mark.asyncio
async def test_tool_node_no_op_when_no_side_channels_populated() -> None:
    """When the LLM emits a non-synthetic tool call (e.g. a work tool),
    the override leaves scratchpad alone."""
    deep = _DemoDeep(_make_runtime())

    async def _stub_super(self, state):  # type: ignore[no-untyped-def]
        # Do not populate _pending_plan / _pending_files / _pending_dispatches.
        return {"messages": [{"role": "tool", "content": "result"}]}

    initial: AgentState = {"scratchpad": {"untouched": "yes"}}
    with patch("ai_core.agents.base.BaseAgent._tool_node", _stub_super):
        new_state = await deep._tool_node(initial)

    # super delegate's return preserved as-is — no scratchpad mutation.
    assert "scratchpad" not in new_state or new_state["scratchpad"] == initial["scratchpad"]


# ---------------------------------------------------------------------------
# Plan rendering helper
# ---------------------------------------------------------------------------
def test_render_plan_marks_status_and_sub_agent() -> None:
    plan = DeepPlan(
        goal="Multi-status",
        steps=[
            DeepPlanStep(
                id="s1", description="research", sub_agent="researcher",
                status="done", result="3 sources",
            ),
            DeepPlanStep(id="s2", description="draft", sub_agent="writer"),
            DeepPlanStep(
                id="s3", description="failed", sub_agent="researcher",
                status="failed", notes="API down",
            ),
            DeepPlanStep(id="s4", description="inline review", sub_agent=None),
        ],
    )
    rendered = DeepAgent._render_plan(plan)
    assert "[x] s1 → researcher" in rendered
    assert "[ ] s2 → writer" in rendered
    assert "[!] s3 → researcher" in rendered
    assert "[ ] s4 (inline)" in rendered
    assert "result: 3 sources" in rendered
    assert "notes: API down" in rendered


def test_all_steps_done_helper() -> None:
    p1 = DeepPlan(goal="g", steps=[
        DeepPlanStep(id="x", description="d", status="done"),
    ])
    p2 = DeepPlan(goal="g", steps=[
        DeepPlanStep(id="x", description="d"),
    ])
    p3 = DeepPlan(goal="g", steps=[
        DeepPlanStep(id="x", description="d", status="done"),
        DeepPlanStep(id="y", description="d", status="failed"),
    ])
    assert DeepAgent._all_steps_done(p1) is True
    assert DeepAgent._all_steps_done(p2) is False
    assert DeepAgent._all_steps_done(p3) is False
