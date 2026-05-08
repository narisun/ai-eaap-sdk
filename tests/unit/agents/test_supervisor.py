"""Unit tests for :class:`SupervisorAgent` — the multi-agent primitive."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ai_core.agents import (
    AgentRuntime,
    AgentState,
    BaseAgent,
    SupervisorAgent,
    TaskInput,
    TaskOutput,
)
from ai_core.agents._resolver import AgentResolver
from ai_core.agents.tool_errors import DefaultToolErrorRenderer
from ai_core.audit.null import NullAuditSink
from ai_core.config.settings import AppSettings
from ai_core.testing import (
    FakeObservabilityProvider,
)
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.registrar import ToolRegistrar
from ai_core.tools.resolver import DefaultToolResolver
from ai_core.tools.spec import ToolSpec

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _CapturingChild(BaseAgent):
    """A child agent that records every invocation and returns a scripted reply.

    Bypasses the LangGraph runtime entirely so unit tests stay fast and
    deterministic; the supervisor's tool dispatch only cares that
    ``ainvoke`` returns an ``AgentState``-shaped mapping with the right
    last assistant message.
    """

    agent_id = "_capture-child"

    def __init__(self, runtime: AgentRuntime, reply: str = "child reply") -> None:
        super().__init__(runtime)
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    def system_prompt(self) -> str:
        return "test child"

    async def ainvoke(  # type: ignore[override]
        self,
        *,
        messages,
        essential=None,
        tenant_id=None,
        thread_id=None,
    ):
        self.calls.append({
            "messages": list(messages),
            "essential": dict(essential or {}),
            "tenant_id": tenant_id,
            "thread_id": thread_id,
        })
        # Return a state shape the supervisor's render_child_output knows how to read.
        return {
            "messages": [
                {"role": "assistant", "content": self._reply},
            ],
            "essential_entities": {},
            "token_count": 0,
            "compaction_count": 0,
            "summary": "",
            "metadata": {},
        }


class _SupportSupervisor(SupervisorAgent):
    agent_id = "support-supervisor"

    def system_prompt(self) -> str:
        return "Coordinate the support team."

    def children(self) -> Mapping[str, type[BaseAgent]]:
        return {
            "triage": _CapturingChild,
            "research": _CapturingChild,
        }


class _TypedRequest(BaseModel):
    customer_id: str
    topic: str


class _TypedReply(BaseModel):
    summary: str
    confidence: float


class _TypedSupervisor(SupervisorAgent):
    """Supervisor with a typed contract on the 'analyst' child."""

    agent_id = "typed-supervisor"

    def system_prompt(self) -> str:
        return "Use analyst with structured input."

    def children(self) -> Mapping[str, type[BaseAgent]]:
        return {"analyst": _CapturingChild}

    def child_input_schema(self, name: str) -> type[BaseModel]:
        return _TypedRequest if name == "analyst" else super().child_input_schema(name)

    def child_output_schema(self, name: str) -> type[BaseModel]:
        return _TypedReply if name == "analyst" else super().child_output_schema(name)

    def render_child_input(self, name: str, payload: BaseModel) -> str:
        if isinstance(payload, _TypedRequest):
            return f"customer={payload.customer_id} topic={payload.topic}"
        return super().render_child_input(name, payload)

    def render_child_output(
        self, name: str, child_state: AgentState,
    ) -> BaseModel:
        # In a real impl the child would emit structured output and we'd
        # parse it; for the test, fabricate a TypedReply directly.
        if name == "analyst":
            return _TypedReply(summary="ok", confidence=0.95)
        return super().render_child_output(name, child_state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_runtime(child_factory) -> AgentRuntime:
    """Build a runtime where ``agent_resolver.resolve(cls)`` returns ``child_factory(cls)``.

    Avoids standing up a full DI container — the supervisor only ever
    calls ``runtime.agent_resolver.resolve(child_cls)`` and we want to
    control exactly what comes back.
    """
    invoker = ToolInvoker(
        observability=FakeObservabilityProvider(),
        audit=NullAuditSink(),
    )

    class _StubResolver(AgentResolver):
        def __init__(self) -> None:  # skip the Container dep
            pass

        def resolve(self, cls):  # type: ignore[override]
            return child_factory(cls)

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
def test_tools_returns_one_toolspec_per_child() -> None:
    """Each entry in children() becomes a ToolSpec with the matching name."""
    cache: dict[type, _CapturingChild] = {}

    def _factory(cls):
        cache.setdefault(cls, _CapturingChild(MagicMock(spec=AgentRuntime)))
        return cache[cls]

    sup = _SupportSupervisor(_make_runtime(_factory))
    tools = sup.tools()

    assert len(tools) == 2
    names = sorted(t.name for t in tools if isinstance(t, ToolSpec))
    assert names == ["research", "triage"]


def test_default_child_input_schema_is_task_input() -> None:
    """No override → the supervisor's tool advertisement uses TaskInput."""

    def _factory(cls):
        return _CapturingChild(MagicMock(spec=AgentRuntime))

    sup = _SupportSupervisor(_make_runtime(_factory))
    triage_tool = next(t for t in sup.tools() if isinstance(t, ToolSpec) and t.name == "triage")
    assert triage_tool.input_model is TaskInput
    assert triage_tool.output_model is TaskOutput


def test_typed_child_schemas_replace_defaults() -> None:
    """Per-child overrides surface in the synthesised ToolSpec."""

    def _factory(cls):
        return _CapturingChild(MagicMock(spec=AgentRuntime))

    sup = _TypedSupervisor(_make_runtime(_factory))
    analyst_tool = next(
        t for t in sup.tools() if isinstance(t, ToolSpec) and t.name == "analyst"
    )
    assert analyst_tool.input_model is _TypedRequest
    assert analyst_tool.output_model is _TypedReply


# ---------------------------------------------------------------------------
# Dispatch behaviour
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dispatch_invokes_child_with_rendered_input() -> None:
    """The supervisor's child handler calls child.ainvoke with a user message
    derived from render_child_input."""
    captured: list[_CapturingChild] = []

    def _factory(cls):
        instance = _CapturingChild(MagicMock(spec=AgentRuntime), reply="triaged")
        captured.append(instance)
        return instance

    sup = _SupportSupervisor(_make_runtime(_factory))
    sup._current_state = {  # would normally be set by _tool_node
        "essential_entities": {"tenant_id": "acme", "user_id": "u-1"},
    }

    payload = TaskInput(task="Classify this ticket", context="VIP customer")
    result = await sup._dispatch_to_child("triage", _CapturingChild, payload)

    assert isinstance(result, TaskOutput)
    assert result.result == "triaged"
    assert len(captured) == 1
    call = captured[0].calls[0]
    assert call["messages"][0]["content"] == "Classify this ticket\n\nContext: VIP customer"
    # Essential entities propagate plus delegation metadata.
    assert call["essential"]["tenant_id"] == "acme"
    assert call["essential"]["user_id"] == "u-1"
    assert call["essential"]["delegated_by"] == "support-supervisor"
    assert call["essential"]["delegation_target"] == "triage"
    assert call["tenant_id"] == "acme"


@pytest.mark.asyncio
async def test_dispatch_caches_child_instance_across_calls() -> None:
    """Resolving the same child twice reuses the same instance."""
    counter = {"calls": 0}

    def _factory(cls):
        counter["calls"] += 1
        return _CapturingChild(MagicMock(spec=AgentRuntime))

    sup = _SupportSupervisor(_make_runtime(_factory))
    sup._current_state = {"essential_entities": {}}

    await sup._dispatch_to_child("triage", _CapturingChild, TaskInput(task="a"))
    await sup._dispatch_to_child("triage", _CapturingChild, TaskInput(task="b"))

    # Resolved once, not twice.
    assert counter["calls"] == 1


@pytest.mark.asyncio
async def test_render_child_output_extracts_last_assistant_message() -> None:
    """Default render uses the child's last assistant message text."""

    def _factory(cls):
        return _CapturingChild(MagicMock(spec=AgentRuntime), reply="final answer")

    sup = _SupportSupervisor(_make_runtime(_factory))
    sup._current_state = {"essential_entities": {}}

    out = await sup._dispatch_to_child("triage", _CapturingChild, TaskInput(task="x"))
    assert isinstance(out, TaskOutput)
    assert out.result == "final answer"


@pytest.mark.asyncio
async def test_typed_supervisor_dispatch_uses_typed_render() -> None:
    """A supervisor that overrides render_child_output returns its custom shape."""

    def _factory(cls):
        return _CapturingChild(MagicMock(spec=AgentRuntime))

    sup = _TypedSupervisor(_make_runtime(_factory))
    sup._current_state = {"essential_entities": {}}

    out = await sup._dispatch_to_child(
        "analyst",
        _CapturingChild,
        _TypedRequest(customer_id="c-1", topic="billing"),
    )
    assert isinstance(out, _TypedReply)
    assert out.summary == "ok"
    assert out.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Validation hardening
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolver_returning_non_baseagent_raises_registry_error() -> None:
    """Defensive: misconfigured DI binding for a child triggers a clear error."""

    def _factory(cls):
        return "not an agent"  # type: ignore[return-value]

    sup = _SupportSupervisor(_make_runtime(_factory))
    sup._current_state = {"essential_entities": {}}

    from ai_core.exceptions import RegistryError
    with pytest.raises(RegistryError) as ei:
        await sup._dispatch_to_child("triage", _CapturingChild, TaskInput(task="x"))
    assert "expected a BaseAgent subclass" in ei.value.message


# ---------------------------------------------------------------------------
# End-to-end with ScriptedLLM (covers _tool_node state stash)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_supervisor_end_to_end_dispatches_via_tool_node() -> None:
    """Full path: scripted supervisor LLM emits a tool_call → _tool_node
    runs → child handler invoked → result rendered as ToolMessage.

    Verifies the state-stash mechanism on _tool_node so the synthetic
    child handlers can read essential_entities."""
    from langchain_core.messages import AIMessage

    def _factory(cls):
        return _CapturingChild(MagicMock(spec=AgentRuntime), reply="triage done")

    runtime = _make_runtime(_factory)
    sup = _SupportSupervisor(runtime)

    state: AgentState = {
        "essential_entities": {"tenant_id": "acme", "user_id": "u-1"},
        "messages": [
            AIMessage(
                content="Delegating",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "triage",
                        "args": {"task": "Look at this"},
                    },
                ],
            ),
        ],
    }

    # Pre-populate the supervisor's child cache so we don't depend on the
    # real ToolInvoker (which would also need a real registry / spec).
    # We're testing _tool_node specifically — stub one call out via
    # the tool_invoker's invoke method.
    invoker_calls: list[Any] = []

    async def _invoke_stub(spec, args, *, agent_id=None, tenant_id=None, principal=None):
        # Drive the spec's handler directly to exercise the
        # _dispatch_to_child closure (which reads sup._current_state).
        invoker_calls.append({
            "spec_name": spec.name,
            "args": dict(args),
            "agent_id": agent_id,
            "tenant_id": tenant_id,
        })
        payload = spec.input_model.model_validate(dict(args))
        out = await spec.handler(payload)
        return out.model_dump(mode="json")

    runtime.tool_invoker.invoke = _invoke_stub  # type: ignore[method-assign]

    new_state = await sup._tool_node(state)

    assert len(invoker_calls) == 1
    assert invoker_calls[0]["spec_name"] == "triage"
    assert invoker_calls[0]["tenant_id"] == "acme"
    # The tool message in the resulting state carries the rendered child output.
    appended = list(new_state.get("messages") or [])
    assert len(appended) == 1
    msg = appended[0]
    assert "triage done" in msg.content
