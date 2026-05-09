"""End-to-end component test: a real :class:`HarnessAgent` wrapping a
real :class:`BaseAgent` that runs through a full LangGraph + tool loop.

Covers:

* the harness constructing a customised :class:`AgentRuntime` with
  capturing wrappers for ``llm`` and ``tool_invoker``,
* a real wrapped agent compiling its LangGraph and dispatching a tool
  call through :class:`ToolInvoker` (validation, audit, observability
  spans all firing for free),
* the captured :class:`Trace` containing both the LLM call and the
  tool dispatch in chronological order,
* the trace round-tripping through ``model_dump_json``.

If this passes, HarnessAgent works end-to-end against the production
code paths.
"""

from __future__ import annotations

import os

import pytest
from injector import Module, provider, singleton
from pydantic import BaseModel

from ai_core.agents import BaseAgent, HarnessAgent, Trace
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.testing import ScriptedLLM, make_llm_response
from ai_core.tools import tool

pytestmark = pytest.mark.component

os.environ.setdefault(
    "EAAP_DATABASE__DSN",
    "postgresql+asyncpg://demo:demo@localhost:5432/demo",
)
os.environ.setdefault("EAAP_OBSERVABILITY__CONSOLE_EXPORT_IN_DEV", "false")


# ---------------------------------------------------------------------------
# Tool + agent under test
# ---------------------------------------------------------------------------
class _LookupIn(BaseModel):
    customer_id: str


class _LookupOut(BaseModel):
    name: str
    plan: str


@tool(name="lookup_customer", version=1)
async def _lookup_customer(payload: _LookupIn) -> _LookupOut:
    """Return a fake customer record."""
    return _LookupOut(name=f"Customer {payload.customer_id}", plan="enterprise")


class _SupportAgent(BaseAgent):
    agent_id = "support-agent"

    def system_prompt(self) -> str:
        return "Look up the customer and answer the user."

    def tools(self):
        return [_lookup_customer]


class _SupportHarness(HarnessAgent):
    agent_id = "support-harness"

    def wrapped_agent(self) -> type[BaseAgent]:
        return _SupportAgent


# ---------------------------------------------------------------------------
# DI wiring
# ---------------------------------------------------------------------------
def _build_container(llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="harness-test", environment="local")

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
async def test_harness_captures_full_agent_run_through_di() -> None:
    """Wrapped agent: turn 1 emits a tool_call → tool dispatches →
    turn 2 emits a final answer. The harness's :class:`Trace` records:

    - LLM call #1 (the tool-call request)
    - tool dispatch (lookup_customer)
    - LLM call #2 (the final answer)
    """
    llm = ScriptedLLM([
        make_llm_response(
            text="Looking that up.",
            tool_calls=[{
                "id": "tc-1",
                "function": {
                    "name": "lookup_customer",
                    "arguments": '{"customer_id": "C-42"}',
                },
            }],
            finish_reason="tool_calls",
        ),
        make_llm_response(
            text="Customer C-42 is on the enterprise plan.",
        ),
    ])

    container = _build_container(llm)
    async with container as c:
        harness = c.get(_SupportHarness)
        final_state = await harness.ainvoke(
            messages=[{"role": "user", "content": "Tell me about C-42."}],
            tenant_id="acme",
        )

    # The wrapped agent ran end-to-end.
    last = final_state["messages"][-1]
    last_content = (
        getattr(last, "content", None) or last.get("content", "")
    )
    assert "enterprise plan" in last_content

    # Two LLM calls, both attributed to the wrapped agent.
    assert len(llm.calls) == 2
    assert all(c["agent_id"] == "support-agent" for c in llm.calls)

    # The harness recorded a trace.
    trace = harness.last_trace
    assert trace is not None
    assert trace.agent_id == "support-harness"
    assert trace.wrapped_agent == "_SupportAgent"
    assert trace.finished_at is not None

    # Three events in order: LLM, tool, LLM.
    kinds = [event.kind for event in trace.events]
    assert kinds == ["llm", "tool", "llm"]

    # First LLM call advertised the tool.
    first_llm = trace.events[0].llm
    assert first_llm is not None
    assert first_llm.tools is not None
    tool_names = [t["function"]["name"] for t in first_llm.tools]
    assert "lookup_customer" in tool_names
    # And it returned a tool_call.
    assert len(first_llm.response_tool_calls) == 1

    # The captured tool dispatch.
    tool_event = trace.events[1].tool
    assert tool_event is not None
    assert tool_event.tool == "lookup_customer"
    assert tool_event.outcome == "ok"
    assert tool_event.raw_args == {"customer_id": "C-42"}
    assert tool_event.result is not None
    assert tool_event.result["plan"] == "enterprise"
    assert tool_event.tenant_id == "acme"
    assert tool_event.agent_id == "support-agent"

    # Final LLM call.
    final_llm = trace.events[2].llm
    assert final_llm is not None
    assert "enterprise plan" in final_llm.response_content

    # Trace round-trips through JSON cleanly.
    payload = trace.model_dump_json()
    reparsed = Trace.model_validate_json(payload)
    assert [e.kind for e in reparsed.events] == kinds
