"""End-to-end component test: a real :class:`DeepAgent` driven by a
scripted LLM through the full SDK stack, dispatching to two real
sub-agents and using the virtual filesystem.

Verifies the deep-agent loop:

1. Top-level LLM emits ``_decompose`` (turn 1) → plan with two
   sub-agent steps captured in ``scratchpad["plan"]``.
2. LLM emits ``_dispatch(researcher, ...)`` (turn 2) → real sub-agent
   ``ainvoke`` runs through its own LangGraph; result lands in the
   ``_dispatch`` tool's return value and ``scratchpad["dispatch_history"]``.
3. LLM emits ``_write_file`` (turn 3) → file lands in
   ``scratchpad["files"]``.
4. LLM emits ``_dispatch(writer, ...)`` (turn 4) → second real
   sub-agent run.
5. LLM emits final assistant message (turn 5) → graph terminates.

This proves DeepAgent works end-to-end against the production code
paths: real DI container, real LangGraph compilation, real
ToolInvoker dispatch with audit + observability + validation.
"""

from __future__ import annotations

import os

import pytest
from injector import Module, provider, singleton

from ai_core.agents import BaseAgent, DeepAgent
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
# Sub-agents
# ---------------------------------------------------------------------------
class Researcher(BaseAgent):
    """Researches a topic and returns a one-sentence summary."""

    agent_id = "researcher"

    def system_prompt(self) -> str:
        return "Research the topic in one short sentence."


class Writer(BaseAgent):
    """Writes a brief paragraph from supplied notes."""

    agent_id = "writer"

    def system_prompt(self) -> str:
        return "Write a brief paragraph from the supplied notes."


class BriefAuthor(DeepAgent):
    agent_id = "brief-author"

    def base_system_prompt(self) -> str:
        return (
            "You author short briefs by orchestrating a researcher and "
            "a writer. Use the virtual filesystem to share notes between "
            "them."
        )

    def sub_agents(self):
        return {"researcher": Researcher, "writer": Writer}


# ---------------------------------------------------------------------------
# DI wiring
# ---------------------------------------------------------------------------
def _build_container(llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="deep-test", environment="local")

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
async def test_deep_agent_decomposes_dispatches_and_writes_through_di() -> None:
    plan_args = (
        '{"goal": "Author a brief about widgets.", '
        '"steps": ['
        '{"id": "step-1", "description": "Research widgets", "sub_agent": "researcher"}, '
        '{"id": "step-2", "description": "Draft the brief", "sub_agent": "writer"}'
        ']}'
    )

    llm = ScriptedLLM([
        # 1. Top-level: _decompose
        make_llm_response(
            text="Planning.",
            tool_calls=[{
                "id": "tc-plan",
                "function": {"name": "_decompose", "arguments": plan_args},
            }],
            finish_reason="tool_calls",
        ),
        # 2. Top-level: _dispatch researcher
        make_llm_response(
            text="Delegating research.",
            tool_calls=[{
                "id": "tc-disp1",
                "function": {
                    "name": "_dispatch",
                    "arguments": (
                        '{"sub_agent": "researcher", '
                        '"task": "Research widgets briefly.", '
                        '"step_id": "step-1"}'
                    ),
                },
            }],
            finish_reason="tool_calls",
        ),
        # 3. Researcher: produces final answer for its dispatch
        make_llm_response(
            text="Widgets are small mechanical components.",
        ),
        # 4. Top-level: _write_file
        make_llm_response(
            text="Saving research notes.",
            tool_calls=[{
                "id": "tc-write",
                "function": {
                    "name": "_write_file",
                    "arguments": (
                        '{"path": "research.md", '
                        '"content": "Widgets are small mechanical components."}'
                    ),
                },
            }],
            finish_reason="tool_calls",
        ),
        # 5. Top-level: _dispatch writer
        make_llm_response(
            text="Delegating draft.",
            tool_calls=[{
                "id": "tc-disp2",
                "function": {
                    "name": "_dispatch",
                    "arguments": (
                        '{"sub_agent": "writer", '
                        '"task": "Write a paragraph about widgets.", '
                        '"context": "See research.md.", '
                        '"step_id": "step-2"}'
                    ),
                },
            }],
            finish_reason="tool_calls",
        ),
        # 6. Writer: final draft
        make_llm_response(
            text="Widgets are small components used in many machines.",
        ),
        # 7. Top-level: final answer
        make_llm_response(
            text="Brief on widgets is ready.",
        ),
    ])

    container = _build_container(llm)
    async with container as c:
        deep = c.get(BriefAuthor)
        final_state = await deep.ainvoke(
            messages=[{"role": "user", "content": "Author a brief about widgets."}],
            tenant_id="acme",
        )

    # 7 LLM calls total: 5 by the deep agent, 1 each by the two sub-agents.
    assert len(llm.calls) == 7

    agent_ids = [c["agent_id"] for c in llm.calls]
    assert agent_ids == [
        "brief-author",   # 1. _decompose
        "brief-author",   # 2. _dispatch researcher
        "researcher",     # 3. researcher final
        "brief-author",   # 4. _write_file
        "brief-author",   # 5. _dispatch writer
        "writer",         # 6. writer final
        "brief-author",   # 7. final answer
    ]

    # tenant_id propagates from top-level into the sub-agent dispatches.
    assert all(c["tenant_id"] == "acme" for c in llm.calls)

    # Plan captured in scratchpad.
    scratchpad = final_state.get("scratchpad") or {}
    plan = scratchpad.get("plan") or {}
    assert plan.get("goal") == "Author a brief about widgets."
    assert len(plan.get("steps", [])) == 2
    assert plan["steps"][0]["sub_agent"] == "researcher"
    assert plan["steps"][1]["sub_agent"] == "writer"

    # Files persisted across dispatches.
    files = scratchpad.get("files") or {}
    assert "research.md" in files
    assert "Widgets" in files["research.md"]

    # Dispatch history records both dispatches with their sub-agent + step_id.
    history = scratchpad.get("dispatch_history") or []
    assert len(history) == 2
    assert history[0]["sub_agent"] == "researcher"
    assert history[0]["step_id"] == "step-1"
    assert history[0]["result"].startswith("Widgets are small mechanical")
    assert history[1]["sub_agent"] == "writer"
    assert history[1]["step_id"] == "step-2"
    assert history[1]["result"].startswith("Widgets are small components")

    # Final answer flows back to the user.
    last = final_state["messages"][-1]
    last_content = getattr(last, "content", None) or last.get("content", "")
    assert "Brief on widgets" in last_content
