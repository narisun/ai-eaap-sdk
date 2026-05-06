"""End-to-end component test: a real :class:`BaseAgent` driven by a scripted LLM.

This exercises the full async stack:

* DI container with default :class:`AgentModule` bindings overridden to
  inject a deterministic LLM and (where needed) a deterministic
  :class:`TokenCounter`.
* Real :class:`BaseAgent.compile` producing a real LangGraph.
* Real :meth:`BaseAgent.ainvoke` flowing through the agent and
  compaction nodes.
* Real :class:`MemoryManager` running its summarization chain.
* Real OTel Baggage propagation, recursion-limit config wiring, etc.

Only the LLM (and optionally the token counter) is faked — every other
component is the production binding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from injector import Module, provider, singleton
from langchain_core.messages import RemoveMessage

from ai_core.agents import BaseAgent, TokenCounter
from ai_core.agents.memory import to_openai_message
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient  # noqa: TC001
from ai_core.testing import ScriptedLLM, make_llm_response


pytestmark = pytest.mark.component


# ---------------------------------------------------------------------------
# Test agent + fakes
# ---------------------------------------------------------------------------
class MathTutorAgent(BaseAgent):
    """Trivial agent — used only to exercise the SDK plumbing."""

    agent_id: str = "math-tutor"

    def system_prompt(self) -> str:
        return "You are a friendly math tutor. Answer in one sentence."


class _ForceCompactCounter:
    """Token counter that always reports over-threshold to trigger compaction."""

    def __init__(self) -> None:
        self.calls = 0

    def count(self, messages: Sequence[Any], *, model: str) -> int:
        self.calls += 1
        return 1_000_000


def _msg_to_dict(m: Any) -> dict[str, Any]:
    """Local convenience — delegates to the SDK's normaliser."""
    return to_openai_message(m)


def _msg_content(m: Any) -> str:
    if isinstance(m, Mapping):
        return str(m.get("content", ""))
    return str(getattr(m, "content", ""))


def _build_container(
    *,
    llm: ILLMClient,
    counter: TokenCounter | None = None,
    settings: AppSettings | None = None,
) -> Container:
    settings = settings or AppSettings(service_name="math-tutor-test")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

        if counter is not None:
            @singleton
            @provider
            def provide_counter(self) -> TokenCounter:
                return counter  # type: ignore[return-value]

    return Container.build([AgentModule(settings=settings), _Overrides()])


# ---------------------------------------------------------------------------
# Scenario 1: Single-turn happy path
# ---------------------------------------------------------------------------
async def test_single_turn_run_produces_expected_state() -> None:
    llm = ScriptedLLM([
        make_llm_response("Two plus two equals four.", prompt_tokens=42, completion_tokens=12),
    ])
    container = _build_container(llm=llm)

    async with container as c:
        agent = c.get(MathTutorAgent)
        final_state = await agent.ainvoke(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            essential={"user_id": "u-demo", "task_id": "T-1"},
            tenant_id="demo-tenant",
        )

    # The LLM was called exactly once: the agent turn.
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["tenant_id"] == "demo-tenant"
    assert call["agent_id"] == "math-tutor"
    # The system prompt sits at the head of the prompt to the LLM.
    assert call["messages"][0]["role"] == "system"
    assert "math tutor" in call["messages"][0]["content"]
    # Followed by the user turn.
    assert call["messages"][1]["role"] == "user"
    assert call["messages"][1]["content"] == "What is 2+2?"

    # Final state contains the original user message + the new assistant reply.
    assert len(final_state["messages"]) == 2
    assert _msg_content(final_state["messages"][0]) == "What is 2+2?"
    assert _msg_content(final_state["messages"][1]) == "Two plus two equals four."

    # Essential entities passed in survived intact.
    essentials = dict(final_state["essential_entities"])
    assert essentials["user_id"] == "u-demo"
    assert essentials["task_id"] == "T-1"
    assert essentials["tenant_id"] == "demo-tenant"

    # Token count comes from the LLM response.
    assert final_state["token_count"] == 54
    # No compaction in a single-turn run.
    assert final_state.get("compaction_count", 0) == 0


# ---------------------------------------------------------------------------
# Scenario 2: Compaction triggered before the agent turn
# ---------------------------------------------------------------------------
async def test_compaction_runs_before_agent_when_threshold_exceeded() -> None:
    # Two LLM calls expected: (1) summarisation, (2) post-compaction agent turn.
    llm = ScriptedLLM([
        make_llm_response("User asked about TASK-42 earlier; deadline question pending."),
        make_llm_response("The deadline is Friday at 5pm."),
    ])
    counter = _ForceCompactCounter()
    container = _build_container(llm=llm, counter=counter)

    long_history = [
        {"role": "user", "content": "I'm working on TASK-42"},
        {"role": "assistant", "content": "Got it — high priority?"},
        {"role": "user", "content": "Yes, customer-facing."},
        {"role": "assistant", "content": "Understood. Anything else?"},
        {"role": "user", "content": "What was the deadline?"},
    ]

    async with container as c:
        agent = c.get(MathTutorAgent)
        final_state = await agent.ainvoke(
            messages=long_history,
            essential={"user_id": "u-demo", "task_id": "TASK-42"},
            tenant_id="demo-tenant",
        )

    # 1 summarisation call + 1 agent turn call.
    assert len(llm.calls) == 2
    summarisation_call, agent_turn_call = llm.calls

    # The summarisation prompt explicitly asks for Essential Entity preservation.
    assert any(
        "ESSENTIAL ENTITIES" in m["content"]
        for m in summarisation_call["messages"]
    )
    assert any("TASK-42" in m["content"] for m in summarisation_call["messages"])

    # The post-compaction agent turn sees the conversation summary instead of
    # the full original history — the older turns were evicted by the reducer.
    agent_turn_msgs = agent_turn_call["messages"]
    assert agent_turn_msgs[0]["role"] == "system"  # system prompt
    # The summary system message is at index 1 (right after system prompt).
    assert "[Conversation summary]" in agent_turn_msgs[1]["content"]
    # No RemoveMessage markers leaked into the prompt.
    assert not any(isinstance(m, RemoveMessage) for m in agent_turn_msgs)

    # Final state recorded the compaction.
    assert final_state["compaction_count"] == 1
    assert (
        final_state["summary"]
        == "User asked about TASK-42 earlier; deadline question pending."
    )

    # Essential Entities — both configured and host-provided — survived.
    essentials = dict(final_state["essential_entities"])
    assert essentials["user_id"] == "u-demo"
    assert essentials["task_id"] == "TASK-42"
    assert essentials["tenant_id"] == "demo-tenant"

    # The new assistant reply landed in the final history.
    assert any(
        _msg_content(m) == "The deadline is Friday at 5pm."
        for m in final_state["messages"]
    )


# ---------------------------------------------------------------------------
# Scenario 3: Compile is idempotent + recursion limit reaches the graph
# ---------------------------------------------------------------------------
async def test_recursion_limit_propagated_into_compiled_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = ScriptedLLM([make_llm_response("ok")])
    settings = AppSettings(
        service_name="math-tutor-test",
        agent={"max_recursion_depth": 11},  # type: ignore[arg-type]
    )
    container = _build_container(llm=llm, settings=settings)

    async with container as c:
        agent = c.get(MathTutorAgent)
        compiled = agent.compile()
        # compile() is idempotent — second call returns the same graph.
        assert agent.compile() is compiled

        captured: dict[str, Any] = {}
        original_ainvoke = compiled.ainvoke

        async def _spy(state: Any, config: Any | None = None) -> Any:
            captured["config"] = config
            return await original_ainvoke(state, config=config)

        monkeypatch.setattr(compiled, "ainvoke", _spy)
        await agent.ainvoke(messages=[{"role": "user", "content": "hi"}])

    assert captured["config"]["recursion_limit"] == 11
