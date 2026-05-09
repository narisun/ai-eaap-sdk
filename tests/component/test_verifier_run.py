"""End-to-end component test: a real :class:`VerifierAgent` wrapping a
real :class:`BaseAgent`, both driven by the same scripted LLM, through
the full SDK stack.

Covers:

* the wrapped agent producing a draft → verifier rejecting → wrapped
  agent revising → verifier passing,
* full DI + LangGraph runtime for the wrapped agent (real graph
  compilation, real `_agent_node` / `_tool_node` paths),
* verifier control loop dispatching the wrapped agent twice and
  issuing two verification LLM calls,
* verdict history and final verdict landing in
  ``state.scratchpad["verifications"]`` and
  ``state.metadata["last_verdict"]`` respectively.

If this passes, VerifierAgent works against the production code paths.
"""

from __future__ import annotations

import os

import pytest
from injector import Module, provider, singleton

from ai_core.agents import BaseAgent, Verdict, VerifierAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response

pytestmark = pytest.mark.component

os.environ.setdefault(
    "EAAP_DATABASE__DSN",
    "postgresql+asyncpg://demo:demo@localhost:5432/demo",
)
os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")


# ---------------------------------------------------------------------------
# Test agents
# ---------------------------------------------------------------------------
class FactualAnswerer(BaseAgent):
    """The wrapped agent — answers user questions in one short sentence."""

    agent_id = "factual-answerer"

    def system_prompt(self) -> str:
        return "Answer the user's question in one short sentence."


class CitationVerifier(VerifierAgent):
    """A verifier that requires the answer to mention 'cited'."""

    agent_id = "citation-verifier"
    max_retries = 1  # 1 initial + 1 retry = 2 total attempts

    def wrapped_agent(self) -> type[BaseAgent]:
        return FactualAnswerer

    def verification_prompt(self) -> str:
        return "Pass only if the answer contains the word 'cited'."


# ---------------------------------------------------------------------------
# DI wiring
# ---------------------------------------------------------------------------
def _build_container(llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="verifier-test", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


# ---------------------------------------------------------------------------
# Verdict tool-call helpers
# ---------------------------------------------------------------------------
def _verdict_tool_call(verdict: Verdict) -> dict:
    return {
        "id": "tc-verdict",
        "function": {
            "name": "_record_verdict",
            "arguments": verdict.model_dump_json(),
        },
    }


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verifier_rejects_then_accepts_after_retry_with_feedback() -> None:
    """Round 1: wrapped agent's first answer doesn't include 'cited' →
    verifier fails it. Round 2: wrapped agent revises (with the verifier
    feedback as an injected user message) → produces an answer that
    includes 'cited' → verifier passes.

    Scripted LLM call sequence:
      1. wrapped (attempt 1) → "Paris is the capital of France."
      2. verifier (round 1) → Verdict(passed=False, feedback="Add the word 'cited'.")
      3. wrapped (attempt 2) → "Paris is the capital of France [cited]."
      4. verifier (round 2) → Verdict(passed=True, feedback="Now correct.")
    """
    llm = ScriptedLLM([
        # 1. Wrapped attempt 1.
        make_llm_response(text="Paris is the capital of France."),
        # 2. Verifier round 1 — reject.
        make_llm_response(
            text="",
            tool_calls=[_verdict_tool_call(
                Verdict(passed=False, feedback="Please cite a source. Add the word 'cited'."),
            )],
            finish_reason="tool_calls",
        ),
        # 3. Wrapped attempt 2 — revised.
        make_llm_response(text="Paris is the capital of France [cited]."),
        # 4. Verifier round 2 — accept.
        make_llm_response(
            text="",
            tool_calls=[_verdict_tool_call(
                Verdict(passed=True, feedback="Looks correct now."),
            )],
            finish_reason="tool_calls",
        ),
    ])

    container = _build_container(llm)
    async with container as c:
        verifier = c.get(CitationVerifier)
        final_state = await verifier.ainvoke(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            tenant_id="acme",
        )

    # Four LLM calls total.
    assert len(llm.calls) == 4

    # The wrapped agent's calls are agent_id="factual-answerer".
    # The verifier's calls are agent_id="citation-verifier".
    agent_ids = [c["agent_id"] for c in llm.calls]
    assert agent_ids == [
        "factual-answerer",
        "citation-verifier",
        "factual-answerer",
        "citation-verifier",
    ]

    # Final verdict mirrored to metadata.
    assert final_state["metadata"]["last_verdict"]["passed"] is True
    assert final_state["metadata"]["last_verdict"]["feedback"] == "Looks correct now."

    # Verdict history accumulated in scratchpad — both attempts recorded.
    history = final_state["scratchpad"]["verifications"]
    assert len(history) == 2
    assert history[0]["attempt"] == 0
    assert history[0]["verdict"]["passed"] is False
    assert history[1]["attempt"] == 1
    assert history[1]["verdict"]["passed"] is True

    # Verifier feedback was actually injected into the wrapped agent's
    # second invocation. We can see it in the prompt the wrapped agent
    # sent on attempt 2: it should contain the feedback as a user
    # message after the previous assistant draft.
    wrapped_attempt2 = llm.calls[2]
    msg_contents = [
        m.get("content", "") for m in wrapped_attempt2["messages"]
        if m.get("role") == "user"
    ]
    assert any(
        "rejected by the verifier" in c and "cite a source" in c
        for c in msg_contents
    )

    # The wrapped agent's final assistant message is what the user sees;
    # not directly returned from the verifier (verifier returns the full
    # state including messages).
    last = final_state["messages"][-1]
    last_content = (
        getattr(last, "content", None) or last.get("content", "")
    )
    assert "[cited]" in last_content


@pytest.mark.asyncio
async def test_strict_verifier_raises_when_max_retries_exhausted() -> None:
    """When the wrapped agent never satisfies the rubric within
    max_retries+1 attempts, the strict verifier raises."""
    from ai_core.exceptions import AgentRuntimeError

    llm = ScriptedLLM([
        # Two wrapped attempts, two verifier rejections — never passing.
        make_llm_response(text="answer-1 (no key word)"),
        make_llm_response(
            text="",
            tool_calls=[_verdict_tool_call(
                Verdict(passed=False, feedback="Missing 'cited'."),
            )],
            finish_reason="tool_calls",
        ),
        make_llm_response(text="answer-2 (still no key word)"),
        make_llm_response(
            text="",
            tool_calls=[_verdict_tool_call(
                Verdict(passed=False, feedback="Still missing 'cited'."),
            )],
            finish_reason="tool_calls",
        ),
    ])

    container = _build_container(llm)
    async with container as c:
        verifier = c.get(CitationVerifier)
        with pytest.raises(AgentRuntimeError) as ei:
            await verifier.ainvoke(
                messages=[{"role": "user", "content": "What is X?"}],
                tenant_id="acme",
            )

    assert "Verification failed after max_retries" in ei.value.message
    assert ei.value.details["max_retries"] == 1
    assert ei.value.details["wrapped_agent"] == "FactualAnswerer"
    # Both verdicts are in the failure details.
    assert ei.value.details["last_verdict"]["passed"] is False
