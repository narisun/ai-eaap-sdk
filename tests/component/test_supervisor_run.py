"""End-to-end component test: a real :class:`SupervisorAgent` driven by
a scripted LLM, dispatching to a real child agent (also driven by a
scripted LLM), through the full SDK stack.

This is the smallest test that proves the supervisor primitive works
against the production code paths:

* real :class:`Container` with :class:`AgentModule` overridden to inject
  a scripted LLM,
* real :class:`SupervisorAgent` compiling a real LangGraph,
* real child :class:`BaseAgent` compiling its own LangGraph,
* real :class:`ToolInvoker` dispatching the synthetic child-tool, with
  validation, audit, and observability spans firing for free,
* real :class:`AgentResolver` looking up the child class through DI.

If this passes, the supervisor pattern works end-to-end. If it fails,
the pattern is broken at an integration boundary that unit tests can't
see.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import pytest
from injector import Module, provider, singleton

from ai_core.agents import BaseAgent, SupervisorAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import (
    ScriptedLLM,
    make_llm_response,
)

pytestmark = pytest.mark.component

# Stub DSN so the lazy AsyncEngine never tries to dial a real database
# during the test (component tests run without testcontainers).
os.environ.setdefault(
    "EAAP_DATABASE__DSN",
    "postgresql+asyncpg://demo:demo@localhost:5432/demo",
)


# ---------------------------------------------------------------------------
# Test agents
# ---------------------------------------------------------------------------
class TriageAgent(BaseAgent):
    agent_id = "triage"

    def system_prompt(self) -> str:
        return "You classify support tickets in one short sentence."


class ResearchAgent(BaseAgent):
    agent_id = "research"

    def system_prompt(self) -> str:
        return "You answer with the customer's last order id."


class SupportSupervisor(SupervisorAgent):
    agent_id = "support-supervisor"

    def system_prompt(self) -> str:
        return (
            "You coordinate support specialists. Use `triage` first, "
            "then `research`, then write a final answer."
        )

    def children(self) -> Mapping[str, type[BaseAgent]]:
        return {"triage": TriageAgent, "research": ResearchAgent}


# ---------------------------------------------------------------------------
# DI wiring
# ---------------------------------------------------------------------------
def _build_container(llm: ILLMClient) -> Container:
    """Build a real container with a scripted LLM override.

    Every other binding (observability, audit, policy, MCP factory,
    agent resolver, etc.) is the production default from
    :class:`AgentModule`. Observability is not overridden — the
    default :class:`RealObservabilityProvider` degrades to in-process
    spans when no OTel/LangFuse credentials are configured, which is
    exactly what we want for a component test.
    """
    settings = AppSettings(service_name="supervisor-test", environment="local")
    # Silence the OTel ConsoleSpanExporter so test output stays clean.
    os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")

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
async def test_supervisor_dispatches_to_two_children_then_finishes() -> None:
    """The supervisor LLM emits two tool_calls in sequence (triage, then
    research), each child returns a scripted reply, and the supervisor's
    final turn produces the answer to the user.

    Scripted LLM call sequence:
      1. supervisor: tool_call('triage', {task: '...'})
      2. triage:    final assistant message ('billing-question')
      3. supervisor: tool_call('research', {task: '...'})
      4. research:  final assistant message ('order-12345')
      5. supervisor: final assistant message (answer)
    """
    llm = ScriptedLLM([
        # 1. supervisor → call triage
        make_llm_response(
            text="Delegating to triage.",
            tool_calls=[{
                "id": "tc-1",
                "function": {
                    "name": "triage",
                    "arguments": '{"task": "Classify this ticket about a missing order."}',
                },
            }],
            finish_reason="tool_calls",
        ),
        # 2. triage child → final reply
        make_llm_response(text="billing-question"),
        # 3. supervisor → call research
        make_llm_response(
            text="Now researching.",
            tool_calls=[{
                "id": "tc-2",
                "function": {
                    "name": "research",
                    "arguments": '{"task": "Find the customer\'s last order id."}',
                },
            }],
            finish_reason="tool_calls",
        ),
        # 4. research child → final reply
        make_llm_response(text="order-12345"),
        # 5. supervisor → final answer
        make_llm_response(
            text=(
                "Your missing order is order-12345. "
                "I've classified your ticket as billing-question."
            ),
        ),
    ])

    container = _build_container(llm)
    async with container as c:
        supervisor = c.get(SupportSupervisor)
        final_state = await supervisor.ainvoke(
            messages=[{
                "role": "user",
                "content": "I'm missing an order; please help.",
            }],
            essential={"user_id": "u-1", "task_id": "T-1"},
            tenant_id="acme",
        )

    # Total of 5 LLM calls — supervisor turn 1, triage, supervisor turn 2,
    # research, supervisor final.
    assert len(llm.calls) == 5

    # Each call carries the agent_id of the agent that made it.
    agent_ids_in_order = [c["agent_id"] for c in llm.calls]
    assert agent_ids_in_order == [
        "support-supervisor",
        "triage",
        "support-supervisor",
        "research",
        "support-supervisor",
    ]

    # tenant_id propagates from the supervisor invocation through to
    # every child LLM call.
    assert all(c["tenant_id"] == "acme" for c in llm.calls)

    # Final assistant content includes both pieces of information.
    last_message = final_state["messages"][-1]
    last_content = getattr(last_message, "content", None) or last_message.get("content", "")
    assert "order-12345" in last_content
    assert "billing-question" in last_content


@pytest.mark.asyncio
async def test_supervisor_caches_child_instance_across_dispatches() -> None:
    """Within one supervisor.ainvoke run, repeated tool_calls to the
    same child reuse the same instance (graph compiles at most once).
    """
    llm = ScriptedLLM([
        # 1. supervisor → first triage call
        make_llm_response(
            text="t1",
            tool_calls=[{
                "id": "a",
                "function": {"name": "triage", "arguments": '{"task": "first"}'},
            }],
            finish_reason="tool_calls",
        ),
        # 2. triage reply
        make_llm_response(text="ok-1"),
        # 3. supervisor → second triage call (same child)
        make_llm_response(
            text="t2",
            tool_calls=[{
                "id": "b",
                "function": {"name": "triage", "arguments": '{"task": "second"}'},
            }],
            finish_reason="tool_calls",
        ),
        # 4. triage reply again
        make_llm_response(text="ok-2"),
        # 5. supervisor final
        make_llm_response(text="done"),
    ])

    container = _build_container(llm)
    async with container as c:
        sup = c.get(SupportSupervisor)
        await sup.ainvoke(
            messages=[{"role": "user", "content": "hello"}],
            tenant_id="acme",
        )

        # The supervisor's child cache should hold one entry for 'triage'
        # (re-resolution via the resolver would have produced a fresh
        # instance; the cache asserts we didn't).
        assert "triage" in sup._child_instances
        assert len(sup._child_instances) == 1
