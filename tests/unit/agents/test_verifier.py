"""Unit tests for :class:`VerifierAgent`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from ai_core.agents import (
    AgentRuntime,
    AgentState,
    BaseAgent,
    Verdict,
    VerifierAgent,
)
from ai_core.agents._resolver import AgentResolver
from ai_core.agents.tool_errors import DefaultToolErrorRenderer
from ai_core.audit.null import NullAuditSink
from ai_core.config.settings import AppSettings
from ai_core.di.interfaces import LLMResponse, LLMUsage
from ai_core.exceptions import AgentRuntimeError, RegistryError
from ai_core.testing import FakeObservabilityProvider
from ai_core.tools.invoker import ToolInvoker
from ai_core.tools.registrar import ToolRegistrar
from ai_core.tools.resolver import DefaultToolResolver

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _CapturingChild(BaseAgent):
    """A child that returns scripted final-state messages and records calls."""

    agent_id = "_wrapped-child"

    def __init__(
        self,
        runtime: AgentRuntime,
        replies: Sequence[str] = ("default reply",),
    ) -> None:
        super().__init__(runtime)
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def system_prompt(self) -> str:
        return "fake child"

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
        reply = self._replies[idx]
        return {
            "messages": [AIMessage(content=reply)],
            "essential_entities": {},
            "scratchpad": {"child_marker": "set-by-child"},
            "metadata": {"child_meta": "set-by-child"},
            "token_count": 0,
            "compaction_count": 0,
            "summary": "",
        }


class _StubLLM:
    """Scripted ILLMClient: returns a sequence of LLMResponses for verification."""

    def __init__(self, verdicts: Sequence[Verdict]) -> None:
        self._responses = [self._verdict_to_response(v) for v in verdicts]
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _verdict_to_response(verdict: Verdict) -> LLMResponse:
        """Wrap a Verdict in an LLMResponse with a `_record_verdict` tool_call."""
        tool_call = {
            "id": "tc-verdict",
            "function": {
                "name": "_record_verdict",
                "arguments": verdict.model_dump_json(),
            },
        }
        return LLMResponse(
            model="fake",
            content="",
            tool_calls=[tool_call],
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.0),
            raw={},
            finish_reason="tool_calls",
        )

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
            "messages": [dict(m) for m in messages],
            "tools": list(tools) if tools else None,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
        })
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]

    async def astream(  # type: ignore[override]
        self, **kwargs,
    ):
        # Not used by the verifier — it only calls complete().
        raise NotImplementedError


class _AnswerVerifier(VerifierAgent):
    agent_id = "answer-verifier"

    def wrapped_agent(self) -> type[BaseAgent]:
        return _CapturingChild

    def verification_prompt(self) -> str:
        return "Pass only if the answer mentions the word 'verified'."


class _LenientAnswerVerifier(_AnswerVerifier):
    """Same as the strict verifier but with strict=False — used for the
    'fail past max_retries returns last attempt' test."""

    strict = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_runtime(
    *,
    wrapped: BaseAgent,
    verifier_llm,
) -> AgentRuntime:
    """Build a runtime where the wrapped child is pre-bound and the LLM
    used by the verifier is the supplied stub."""
    invoker = ToolInvoker(
        observability=FakeObservabilityProvider(),
        audit=NullAuditSink(),
    )

    class _StubResolver(AgentResolver):
        def __init__(self) -> None:
            pass

        def resolve(self, cls):  # type: ignore[override]
            return wrapped

    return AgentRuntime(
        agent_settings=AppSettings(service_name="t", environment="local").agent,
        llm=verifier_llm,
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
# Pass on first try
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verifier_returns_state_when_first_attempt_passes() -> None:
    """Wrapped agent runs once; verdict is `passed=True`; state returns."""
    child = _CapturingChild(MagicMock(spec=AgentRuntime), replies=["the answer is verified"])
    llm = _StubLLM([
        Verdict(passed=True, feedback="Looks great."),
    ])
    verifier = _AnswerVerifier(_make_runtime(wrapped=child, verifier_llm=llm))

    state = await verifier.ainvoke(
        messages=[{"role": "user", "content": "Tell me something."}],
        tenant_id="acme",
    )

    # Wrapped agent invoked exactly once.
    assert len(child.calls) == 1
    # Verifier's LLM invoked once (the verification call).
    assert len(llm.calls) == 1
    # Final verdict mirrored to metadata.
    assert state["metadata"]["last_verdict"]["passed"] is True
    assert state["metadata"]["last_verdict"]["feedback"] == "Looks great."
    # Verdict history accumulated in scratchpad.
    assert len(state["scratchpad"]["verifications"]) == 1
    assert state["scratchpad"]["verifications"][0]["attempt"] == 0
    # Wrapped agent's existing scratchpad / metadata fields preserved.
    assert state["scratchpad"]["child_marker"] == "set-by-child"
    assert state["metadata"]["child_meta"] == "set-by-child"


# ---------------------------------------------------------------------------
# Fail then pass on retry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verifier_retries_with_feedback_when_first_attempt_fails() -> None:
    """First answer fails; verifier feeds back; child produces a better
    answer that passes verification on attempt 2."""
    child = _CapturingChild(
        MagicMock(spec=AgentRuntime),
        replies=["draft answer", "the answer is verified now"],
    )
    llm = _StubLLM([
        Verdict(passed=False, feedback="Add the word 'verified'."),
        Verdict(passed=True, feedback="Now correct."),
    ])
    verifier = _AnswerVerifier(_make_runtime(wrapped=child, verifier_llm=llm))

    state = await verifier.ainvoke(
        messages=[{"role": "user", "content": "Tell me something."}],
    )

    # Two child invocations + two verification calls.
    assert len(child.calls) == 2
    assert len(llm.calls) == 2

    # Attempt 1's messages: just the original user request.
    attempt1_msgs = child.calls[0]["messages"]
    assert len(attempt1_msgs) == 1
    assert attempt1_msgs[0]["content"] == "Tell me something."

    # Attempt 2's messages: original + previous answer (assistant) +
    # verifier feedback (user).
    attempt2_msgs = child.calls[1]["messages"]
    assert len(attempt2_msgs) == 3
    assert attempt2_msgs[0]["content"] == "Tell me something."
    assert attempt2_msgs[1]["role"] == "assistant"
    assert attempt2_msgs[1]["content"] == "draft answer"
    assert attempt2_msgs[2]["role"] == "user"
    assert "Add the word 'verified'." in attempt2_msgs[2]["content"]
    assert "rejected by the verifier" in attempt2_msgs[2]["content"]

    # Final verdict is the second one.
    assert state["metadata"]["last_verdict"]["passed"] is True
    # Verdict history records both attempts.
    history = state["scratchpad"]["verifications"]
    assert len(history) == 2
    assert history[0]["attempt"] == 0
    assert history[0]["verdict"]["passed"] is False
    assert history[1]["attempt"] == 1
    assert history[1]["verdict"]["passed"] is True


# ---------------------------------------------------------------------------
# Strict-mode: raise after max_retries
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_strict_verifier_raises_after_max_retries() -> None:
    """All attempts fail; strict=True (default) → AgentRuntimeError."""
    child = _CapturingChild(
        MagicMock(spec=AgentRuntime),
        replies=["bad", "still bad", "still bad too"],
    )
    llm = _StubLLM([
        Verdict(passed=False, feedback="No."),
        Verdict(passed=False, feedback="Still no."),
        Verdict(passed=False, feedback="Final no."),
    ])
    verifier = _AnswerVerifier(_make_runtime(wrapped=child, verifier_llm=llm))
    verifier.max_retries = 2  # 1 initial + 2 retries = 3 total attempts

    with pytest.raises(AgentRuntimeError) as ei:
        await verifier.ainvoke(
            messages=[{"role": "user", "content": "Tell me."}],
        )

    assert "Verification failed after max_retries" in ei.value.message
    assert ei.value.details["agent_id"] == "answer-verifier"
    assert ei.value.details["wrapped_agent"] == "_CapturingChild"
    assert ei.value.details["max_retries"] == 2
    assert ei.value.details["last_verdict"]["passed"] is False
    # Three child invocations were made.
    assert len(child.calls) == 3


# ---------------------------------------------------------------------------
# Lenient mode: return last attempt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_lenient_verifier_returns_last_attempt_after_max_retries() -> None:
    """All attempts fail; strict=False → return the last state with verdict."""
    child = _CapturingChild(
        MagicMock(spec=AgentRuntime),
        replies=["bad", "still bad"],
    )
    llm = _StubLLM([
        Verdict(passed=False, feedback="No."),
        Verdict(passed=False, feedback="Still no."),
    ])
    verifier = _LenientAnswerVerifier(_make_runtime(wrapped=child, verifier_llm=llm))
    verifier.max_retries = 1  # 1 initial + 1 retry = 2 total

    state = await verifier.ainvoke(
        messages=[{"role": "user", "content": "Tell me."}],
    )
    # No raise — caller gets the last attempt + verdict in metadata.
    assert state["metadata"]["last_verdict"]["passed"] is False
    assert state["metadata"]["last_verdict"]["feedback"] == "Still no."
    assert len(child.calls) == 2


# ---------------------------------------------------------------------------
# Wrapped-agent caching
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verifier_caches_wrapped_instance_across_retries() -> None:
    """Same wrapped instance reused on retry — graph compiles once, etc."""
    child = _CapturingChild(
        MagicMock(spec=AgentRuntime),
        replies=["bad", "verified ok"],
    )
    llm = _StubLLM([
        Verdict(passed=False, feedback="Try again."),
        Verdict(passed=True, feedback="Good."),
    ])
    verifier = _AnswerVerifier(_make_runtime(wrapped=child, verifier_llm=llm))
    await verifier.ainvoke(messages=[{"role": "user", "content": "x"}])

    # Cached instance survives across attempts.
    assert verifier._wrapped_instance is child


# ---------------------------------------------------------------------------
# Verdict parsing edge cases
# ---------------------------------------------------------------------------
def test_parse_verdict_handles_no_tool_call() -> None:
    """If the verifier LLM emits no tool_call, default to passed=False."""
    response = LLMResponse(
        model="fake",
        content="oops",
        tool_calls=[],
        usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        raw={},
        finish_reason="stop",
    )
    verdict = VerifierAgent._parse_verdict_or_default(response)
    assert verdict.passed is False
    assert "no structured verdict" in verdict.feedback


def test_parse_verdict_handles_unparseable_arguments() -> None:
    """Malformed JSON in tool_call args → defensive Verdict(passed=False)."""
    response = LLMResponse(
        model="fake",
        content="",
        tool_calls=[{
            "id": "tc",
            "function": {"name": "_record_verdict", "arguments": "not-json"},
        }],
        usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        raw={},
        finish_reason="tool_calls",
    )
    verdict = VerifierAgent._parse_verdict_or_default(response)
    assert verdict.passed is False
    assert "unparseable verdict" in verdict.feedback


# ---------------------------------------------------------------------------
# wrapped_agent resolver hardening
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolver_returning_non_baseagent_raises_registry_error() -> None:
    """Defensive: misconfigured DI binding for the wrapped class triggers a
    clear error rather than silent dispatch to a non-agent object."""

    class _BadResolver(AgentResolver):
        def __init__(self) -> None:
            pass

        def resolve(self, cls):  # type: ignore[override]
            return "not an agent"  # type: ignore[return-value]

    invoker = ToolInvoker(
        observability=FakeObservabilityProvider(),
        audit=NullAuditSink(),
    )
    runtime = AgentRuntime(
        agent_settings=AppSettings(service_name="t", environment="local").agent,
        llm=_StubLLM([Verdict(passed=True, feedback="ok")]),
        memory=MagicMock(),
        observability=FakeObservabilityProvider(),
        tool_invoker=invoker,
        mcp_factory=MagicMock(),
        tool_error_renderer=DefaultToolErrorRenderer(),
        tool_resolver=DefaultToolResolver(MagicMock(), invoker),
        tool_registrar=ToolRegistrar(invoker),
        agent_resolver=_BadResolver(),
    )
    verifier = _AnswerVerifier(runtime)

    with pytest.raises(RegistryError) as ei:
        await verifier.ainvoke(messages=[{"role": "user", "content": "x"}])
    assert "expected a BaseAgent subclass" in ei.value.message


# ---------------------------------------------------------------------------
# Verification call shape
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verification_call_advertises_record_verdict_tool() -> None:
    """The verifier LLM call must offer the `_record_verdict` tool with
    a Pydantic-derived schema so the LLM can produce structured output."""
    child = _CapturingChild(
        MagicMock(spec=AgentRuntime),
        replies=["candidate"],
    )
    llm = _StubLLM([Verdict(passed=True, feedback="ok")])
    verifier = _AnswerVerifier(_make_runtime(wrapped=child, verifier_llm=llm))
    await verifier.ainvoke(messages=[{"role": "user", "content": "go"}])

    assert len(llm.calls) == 1
    tools = llm.calls[0]["tools"]
    assert tools is not None
    assert len(tools) == 1
    spec = tools[0]
    assert spec["function"]["name"] == "_record_verdict"
    # The schema includes the Verdict fields.
    params = spec["function"]["parameters"]
    assert "passed" in params["properties"]
    assert "feedback" in params["properties"]


@pytest.mark.asyncio
async def test_verification_call_includes_rubric_and_candidate() -> None:
    """The verifier's user message must surface both the rubric and
    the candidate answer so the LLM has the context to judge."""
    child = _CapturingChild(
        MagicMock(spec=AgentRuntime),
        replies=["candidate text"],
    )
    llm = _StubLLM([Verdict(passed=True, feedback="ok")])
    verifier = _AnswerVerifier(_make_runtime(wrapped=child, verifier_llm=llm))
    await verifier.ainvoke(messages=[{"role": "user", "content": "go"}])

    msgs = llm.calls[0]["messages"]
    user_msg = next(m for m in msgs if m.get("role") == "user")
    assert "Pass only if the answer mentions the word 'verified'." in user_msg["content"]
    assert "candidate text" in user_msg["content"]


# ---------------------------------------------------------------------------
# Empty messages edge case
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_extract_last_assistant_text_handles_empty_state() -> None:
    """When the wrapped agent returns no assistant message, candidate text
    is empty (defensive — verifier should still run, likely fail)."""
    text = VerifierAgent._extract_last_assistant_text({"messages": []})
    assert text == ""

    text = VerifierAgent._extract_last_assistant_text({})
    assert text == ""


@pytest.mark.asyncio
async def test_extract_last_assistant_text_handles_dict_message() -> None:
    """Plain dict messages with role='assistant' are recognised."""
    state: AgentState = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    text = VerifierAgent._extract_last_assistant_text(state)
    assert text == "hello"


@pytest.mark.asyncio
async def test_extract_last_assistant_text_handles_aimessage() -> None:
    """LangChain AIMessage shapes (type='ai') are recognised."""
    state: AgentState = {
        "messages": [AIMessage(content="from AI")],
    }
    text = VerifierAgent._extract_last_assistant_text(state)
    assert text == "from AI"
