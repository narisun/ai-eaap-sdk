"""End-to-end: agent.list_prompts() and get_prompt() against real FastMCP server.

Spawns examples/mcp_server_demo/server.py via PoolingMCPConnectionFactory.
Exercises the application-invoked prompt path — fetches summarize_text,
splices messages, runs ainvoke().
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from injector import Module, provider, singleton

from ai_core.agents import BaseAgent
from ai_core.audit import IAuditSink
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.mcp import MCPPrompt, MCPServerSpec
from ai_core.testing import FakeAuditSink, ScriptedLLM, make_llm_response

pytestmark = pytest.mark.integration

pytest.importorskip("fastmcp")

DEMO_SERVER = Path(__file__).resolve().parents[3] / "examples" / "mcp_server_demo" / "server.py"


class _PromptIntegrationAgent(BaseAgent):
    agent_id = "prompt-int-agent"

    def system_prompt(self) -> str:
        return "Test agent."

    def mcp_servers(self):
        return [
            MCPServerSpec(
                component_id="demo-srv",
                transport="stdio",
                target=sys.executable,
                args=(str(DEMO_SERVER),),
            ),
        ]


def _build_container(llm: ILLMClient, audit_sink: IAuditSink) -> Container:
    settings = AppSettings(service_name="prompt-int-test", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

        @singleton
        @provider
        def provide_audit_sink(self) -> IAuditSink:
            return audit_sink

    return Container.build([AgentModule(settings=settings), _Overrides()])


async def test_list_prompts_finds_summarize_text() -> None:
    """agent.list_prompts() returns the demo server's prompts as MCPPrompt instances."""
    llm = ScriptedLLM([make_llm_response("noop")])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_PromptIntegrationAgent)
        prompts = await agent.list_prompts()

    names = [p.name for p in prompts]
    assert "summarize_text" in names
    assert all(isinstance(p, MCPPrompt) for p in prompts)


async def test_get_prompt_fetches_templated_messages() -> None:
    """agent.get_prompt('summarize_text', {'text': '...'}) returns templated messages."""
    llm = ScriptedLLM([make_llm_response("noop")])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_PromptIntegrationAgent)
        messages = await agent.get_prompt(
            "summarize_text",
            arguments={"text": "Hello, world."},
        )

    assert len(messages) >= 1
    assert any("Hello, world." in m.content for m in messages)


async def test_prompt_splice_into_ainvoke_runs_agent() -> None:
    """End-to-end: app fetches prompt → splices into ainvoke → agent responds."""
    # The agent's response after consuming the templated prompt.
    llm = ScriptedLLM([make_llm_response("This is the summary.")])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_PromptIntegrationAgent)
        templated = await agent.get_prompt(
            "summarize_text",
            arguments={"text": "MCP is a protocol."},
        )
        seed = [{"role": m.role, "content": m.content} for m in templated]
        final = await agent.ainvoke(messages=seed, tenant_id="t")

    last = final["messages"][-1]
    assert "summary" in str(getattr(last, "content", "")).lower()
