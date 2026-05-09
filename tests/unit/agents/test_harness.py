"""Unit tests for :class:`HarnessAgent` and its capturing wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ai_core.agents import (
    AgentRuntime,
    AgentState,
    BaseAgent,
    HarnessAgent,
    LLMCallRecord,
    ToolDispatchRecord,
    Trace,
    TraceEvent,
)
from ai_core.agents.harness import _CapturingLLMClient, _CapturingToolInvoker
from ai_core.agents.tool_errors import DefaultToolErrorRenderer
from ai_core.audit.null import NullAuditSink
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import LLMResponse, LLMUsage
from ai_core.exceptions import ToolExecutionError
from ai_core.testing import FakeObservabilityProvider
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.registrar import ToolRegistrar
from ai_core.tools.resolver import DefaultToolResolver
from ai_core.tools.spec import ToolSpec

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Trace event sink helper
# ---------------------------------------------------------------------------
class _Sink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def __call__(self, event: TraceEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeLLM:
    """Minimal ILLMClient stand-in that records inputs and returns scripted output."""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
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
        self.calls.append({
            "model": model,
            "messages": list(messages),
            "tools": list(tools) if tools else None,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
        })
        return self._response

    async def astream(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class _ScriptedInvoker:
    """Minimal IToolInvoker stand-in returning a scripted result or raising."""

    def __init__(
        self,
        result: Mapping[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[dict[str, Any]] = []
        self.registrations: list[str] = []

    async def invoke(
        self,
        spec: ToolSpec,
        raw_args: Mapping[str, Any],
        *,
        principal: Mapping[str, Any] | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append({
            "spec": spec.name,
            "raw_args": dict(raw_args),
            "agent_id": agent_id,
            "tenant_id": tenant_id,
        })
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result

    def register(self, spec: ToolSpec) -> None:
        self.registrations.append(spec.name)


class _EchoIn(BaseModel):
    text: str


class _EchoOut(BaseModel):
    text: str


def _make_spec(name: str = "echo") -> ToolSpec:
    async def _handler(payload: BaseModel) -> BaseModel:
        assert isinstance(payload, _EchoIn)
        return _EchoOut(text=payload.text)

    return ToolSpec(
        name=name,
        version=1,
        description="Echo the input text.",
        input_model=_EchoIn,
        output_model=_EchoOut,
        handler=_handler,
        opa_path=None,
    )


# ---------------------------------------------------------------------------
# _CapturingLLMClient unit
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_capturing_llm_client_records_complete_call() -> None:
    response = LLMResponse(
        model="real-model",
        content="hello",
        tool_calls=[{"id": "tc-1", "function": {"name": "x", "arguments": "{}"}}],
        usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.01),
        raw={},
        finish_reason="stop",
    )
    real = _FakeLLM(response)
    sink = _Sink()
    wrapper = _CapturingLLMClient(real, sink, t0_perf=0.0)

    out = await wrapper.complete(
        model="requested-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
        tenant_id="t1",
        agent_id="a1",
    )

    # Real client got the call through.
    assert out is response
    assert real.calls[0]["model"] == "requested-model"

    # Exactly one trace event with kind="llm".
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.kind == "llm"
    assert event.tool is None
    assert event.llm is not None
    rec = event.llm
    assert rec.model == "requested-model"
    assert rec.messages == [{"role": "user", "content": "hi"}]
    assert rec.tools == [{"type": "function", "function": {"name": "x"}}]
    assert rec.tenant_id == "t1"
    assert rec.agent_id == "a1"
    assert rec.response_model == "real-model"
    assert rec.response_content == "hello"
    assert len(rec.response_tool_calls) == 1
    assert rec.finish_reason == "stop"
    assert rec.prompt_tokens == 10
    assert rec.completion_tokens == 20
    assert rec.total_tokens == 30
    assert rec.cost_usd == 0.01
    assert rec.timestamp_ms >= 0.0


@pytest.mark.asyncio
async def test_capturing_llm_client_handles_no_tools_no_finish_reason() -> None:
    response = LLMResponse(
        model="m",
        content="ok",
        tool_calls=[],
        usage=LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        raw={},
        finish_reason=None,
    )
    sink = _Sink()
    wrapper = _CapturingLLMClient(_FakeLLM(response), sink, t0_perf=0.0)

    await wrapper.complete(model=None, messages=[])
    assert sink.events[0].llm is not None
    rec = sink.events[0].llm
    assert rec.tools is None
    assert rec.finish_reason is None
    assert rec.cost_usd is None


# ---------------------------------------------------------------------------
# _CapturingToolInvoker unit
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_capturing_tool_invoker_records_successful_dispatch() -> None:
    spec = _make_spec("echo")
    real = _ScriptedInvoker(result={"text": "out"})
    sink = _Sink()
    wrapper = _CapturingToolInvoker(real, sink, t0_perf=0.0)

    out = await wrapper.invoke(
        spec, {"text": "in"}, agent_id="a1", tenant_id="t1",
    )

    # Real invoker got the call.
    assert real.calls[0]["raw_args"] == {"text": "in"}
    assert dict(out) == {"text": "out"}

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.kind == "tool"
    assert event.llm is None
    assert event.tool is not None
    rec = event.tool
    assert rec.tool == "echo"
    assert rec.tool_version == 1
    assert rec.raw_args == {"text": "in"}
    assert rec.agent_id == "a1"
    assert rec.tenant_id == "t1"
    assert rec.outcome == "ok"
    assert rec.result == {"text": "out"}
    assert rec.error_type is None
    assert rec.error_message is None
    assert rec.timestamp_ms >= 0.0
    assert rec.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_capturing_tool_invoker_records_failure_and_reraises() -> None:
    spec = _make_spec("boom")
    err = ToolExecutionError("kaboom", details={})
    real = _ScriptedInvoker(raises=err)
    sink = _Sink()
    wrapper = _CapturingToolInvoker(real, sink, t0_perf=0.0)

    with pytest.raises(ToolExecutionError):
        await wrapper.invoke(spec, {"text": "in"}, agent_id="a1")

    assert len(sink.events) == 1
    rec = sink.events[0].tool
    assert rec is not None
    assert rec.outcome == "error"
    assert rec.error_type == "ToolExecutionError"
    assert "kaboom" in (rec.error_message or "")
    assert rec.result is None


def test_capturing_tool_invoker_register_delegates() -> None:
    spec = _make_spec("echo")
    real = _ScriptedInvoker(result={"text": "x"})
    sink = _Sink()
    wrapper = _CapturingToolInvoker(real, sink, t0_perf=0.0)

    wrapper.register(spec)
    assert real.registrations == ["echo"]
    # Registration is not a runtime event — no trace events recorded.
    assert sink.events == []


# ---------------------------------------------------------------------------
# HarnessAgent integration (with imperative wrapped agent)
# ---------------------------------------------------------------------------
class _ImperativeChild(BaseAgent):
    """A child that calls runtime.llm + runtime.tool_invoker imperatively.

    Skips the LangGraph; lets us assert the harness wires its capturing
    wrappers into the runtime correctly without standing up a full graph
    in unit tests.
    """

    agent_id = "imperative-child"

    def system_prompt(self) -> str:
        return "imperative child"

    async def ainvoke(  # type: ignore[override]
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        essential: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        # One LLM call.
        await self._runtime.llm.complete(
            model=None,
            messages=list(messages),
            tenant_id=tenant_id,
            agent_id=self.agent_id,
        )
        # One tool dispatch.
        await self._runtime.tool_invoker.invoke(
            _make_spec("echo"),
            {"text": "from-child"},
            agent_id=self.agent_id,
            tenant_id=tenant_id,
        )
        return {
            "messages": [{"role": "assistant", "content": "child-done"}],
            "essential_entities": {},
            "scratchpad": {},
            "metadata": {},
            "token_count": 0,
            "compaction_count": 0,
            "summary": "",
        }


class _BoomChild(_ImperativeChild):
    """Like _ImperativeChild but raises after the LLM call."""

    agent_id = "boom-child"

    async def ainvoke(  # type: ignore[override]
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        essential: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        await self._runtime.llm.complete(
            model=None, messages=list(messages), agent_id=self.agent_id,
        )
        raise RuntimeError("child-failed")


class _ChildHarness(HarnessAgent):
    agent_id = "child-harness"

    def wrapped_agent(self) -> type[BaseAgent]:
        return _ImperativeChild


class _BoomHarness(HarnessAgent):
    agent_id = "boom-harness"

    def wrapped_agent(self) -> type[BaseAgent]:
        return _BoomChild


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_runtime(*, llm: Any, invoker: Any) -> AgentRuntime:
    return AgentRuntime(
        agent_settings=AppSettings(service_name="t", environment="local").agent,
        llm=llm,
        memory=MagicMock(),
        observability=FakeObservabilityProvider(),
        tool_invoker=invoker,
        mcp_factory=MagicMock(),
        tool_error_renderer=DefaultToolErrorRenderer(),
        tool_resolver=DefaultToolResolver(MagicMock(), invoker),
        tool_registrar=ToolRegistrar(invoker),
        agent_resolver=MagicMock(),
    )


def _llm_response(text: str = "ok") -> LLMResponse:
    return LLMResponse(
        model="m",
        content=text,
        tool_calls=[],
        usage=LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        raw={},
        finish_reason="stop",
    )


# ---------------------------------------------------------------------------
# HarnessAgent end-to-end (no graph) tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_harness_captures_llm_and_tool_in_order() -> None:
    real_llm = _FakeLLM(_llm_response("hello"))
    real_invoker = _ScriptedInvoker(result={"text": "from-tool"})
    runtime = _make_runtime(llm=real_llm, invoker=real_invoker)
    harness = _ChildHarness(runtime)

    state = await harness.ainvoke(
        messages=[{"role": "user", "content": "go"}],
        tenant_id="acme",
    )

    # Wrapped agent ran and returned its state.
    last = state["messages"][-1]
    assert last["content"] == "child-done"

    # last_trace populated.
    trace = harness.last_trace
    assert trace is not None
    assert trace.agent_id == "child-harness"
    assert trace.wrapped_agent == "_ImperativeChild"
    assert trace.started_at is not None
    assert trace.finished_at is not None
    assert trace.finished_at >= trace.started_at

    # Two events: first LLM, then tool.
    assert len(trace.events) == 2
    assert trace.events[0].kind == "llm"
    assert trace.events[1].kind == "tool"

    # LLM event details.
    llm_rec = trace.events[0].llm
    assert llm_rec is not None
    assert llm_rec.agent_id == "imperative-child"
    assert llm_rec.tenant_id == "acme"
    assert llm_rec.response_content == "hello"

    # Tool event details.
    tool_rec = trace.events[1].tool
    assert tool_rec is not None
    assert tool_rec.tool == "echo"
    assert tool_rec.outcome == "ok"
    assert tool_rec.result == {"text": "from-tool"}
    assert tool_rec.tenant_id == "acme"

    # Real services were not bypassed — wrappers delegated through.
    assert len(real_llm.calls) == 1
    assert len(real_invoker.calls) == 1


@pytest.mark.asyncio
async def test_harness_populates_last_trace_when_wrapped_agent_raises() -> None:
    """Even when the wrapped agent raises, the partial trace must be
    captured so callers can inspect what ran before the failure."""
    real_llm = _FakeLLM(_llm_response("partial"))
    real_invoker = _ScriptedInvoker(result={"text": "unused"})
    runtime = _make_runtime(llm=real_llm, invoker=real_invoker)
    harness = _BoomHarness(runtime)

    with pytest.raises(RuntimeError, match="child-failed"):
        await harness.ainvoke(messages=[{"role": "user", "content": "go"}])

    trace = harness.last_trace
    assert trace is not None
    assert trace.finished_at is not None
    # The single LLM call before the raise was captured.
    assert len(trace.events) == 1
    assert trace.events[0].kind == "llm"


@pytest.mark.asyncio
async def test_harness_does_not_mutate_original_runtime() -> None:
    """``dataclasses.replace`` produces a fresh runtime; the original
    binding for llm/tool_invoker on the harness's own runtime is
    unchanged after invocation."""
    real_llm = _FakeLLM(_llm_response())
    real_invoker = _ScriptedInvoker(result={"text": "x"})
    runtime = _make_runtime(llm=real_llm, invoker=real_invoker)
    harness = _ChildHarness(runtime)

    await harness.ainvoke(messages=[{"role": "user", "content": "go"}])

    # Harness's own runtime still points at the original (un-wrapped)
    # services.
    assert harness._runtime.llm is real_llm
    assert harness._runtime.tool_invoker is real_invoker


@pytest.mark.asyncio
async def test_trace_round_trips_through_model_dump_json() -> None:
    """The Trace data model must serialize cleanly so hosts can persist it."""
    real_llm = _FakeLLM(_llm_response())
    real_invoker = _ScriptedInvoker(result={"text": "ok"})
    runtime = _make_runtime(llm=real_llm, invoker=real_invoker)
    harness = _ChildHarness(runtime)

    await harness.ainvoke(messages=[{"role": "user", "content": "go"}])
    assert harness.last_trace is not None
    payload = harness.last_trace.model_dump_json()

    parsed = Trace.model_validate_json(payload)
    assert parsed.agent_id == harness.last_trace.agent_id
    assert len(parsed.events) == len(harness.last_trace.events)
    assert parsed.events[0].kind == harness.last_trace.events[0].kind


@pytest.mark.asyncio
async def test_harness_constructs_fresh_wrapped_per_invocation() -> None:
    """The harness builds a new wrapped instance every call so capture is
    bound to the per-run customised runtime, not a long-lived shared one."""
    real_llm = _FakeLLM(_llm_response())
    real_invoker = _ScriptedInvoker(result={"text": "x"})
    runtime = _make_runtime(llm=real_llm, invoker=real_invoker)
    harness = _ChildHarness(runtime)

    constructed: list[BaseAgent] = []
    original = harness._construct_wrapped

    def _record(cls: type[BaseAgent], rt: AgentRuntime) -> BaseAgent:
        instance = original(cls, rt)
        constructed.append(instance)
        return instance

    harness._construct_wrapped = _record  # type: ignore[method-assign]

    await harness.ainvoke(messages=[{"role": "user", "content": "1"}])
    await harness.ainvoke(messages=[{"role": "user", "content": "2"}])

    assert len(constructed) == 2
    assert constructed[0] is not constructed[1]


# ---------------------------------------------------------------------------
# Trace data-model unit
# ---------------------------------------------------------------------------
def test_trace_event_records_are_pydantic() -> None:
    """Smoke check: the public records are constructible directly."""
    llm = LLMCallRecord(
        model=None,
        messages=[],
        tools=None,
        tenant_id=None,
        agent_id=None,
        response_model="m",
        response_content="",
        response_tool_calls=[],
        finish_reason=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cost_usd=None,
        timestamp_ms=0.0,
    )
    tool = ToolDispatchRecord(
        tool="t",
        tool_version=1,
        raw_args={},
        agent_id=None,
        tenant_id=None,
        outcome="ok",
        result={},
        timestamp_ms=0.0,
        latency_ms=0.0,
    )
    assert llm.response_model == "m"
    assert tool.tool == "t"
    # Default invoker registers nothing on the bare ToolInvoker.
    invoker = ToolInvoker(observability=FakeObservabilityProvider(), audit=NullAuditSink())
    assert invoker is not None
