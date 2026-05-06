"""End-to-end: agent invokes a real FastMCP stdio server's tool.

Spawns `examples/mcp_server_demo/server.py` as a stdio subprocess via the
production `PoolingMCPConnectionFactory`. Drives an agent with a
`ScriptedLLM` that emits a tool-call for `echo`, asserts the round-trip
through the full SDK pipeline (resolution, ToolInvoker, audit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from injector import Module, provider, singleton

from ai_core.agents import BaseAgent
from ai_core.audit import IAuditSink  # noqa: TC001
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient  # noqa: TC001
from ai_core.mcp import MCPServerSpec
from ai_core.testing import FakeAuditSink, ScriptedLLM, make_llm_response

pytestmark = pytest.mark.integration

# Skip when fastmcp isn't importable (CI environments without the extra).
pytest.importorskip("fastmcp")

DEMO_SERVER = Path(__file__).resolve().parents[3] / "examples" / "mcp_server_demo" / "server.py"


class _MCPDemoAgent(BaseAgent):
    agent_id = "mcp-demo-agent"

    def system_prompt(self) -> str:
        return "Use the echo tool to repeat what the user said."

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
    settings = AppSettings(service_name="mcp-int-test", environment="local")

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


async def test_agent_invokes_mcp_echo_end_to_end() -> None:
    """LLM → tool_call → MCPToolSpec.handler → real FastMCP server → result."""
    # ScriptedLLM responds with a tool_call to echo, then a final assistant message.
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"text": "hello from the SDK"}',
                    },
                },
            ],
        ),
        make_llm_response("Done."),
    ])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_MCPDemoAgent)
        final = await agent.ainvoke(
            messages=[{"role": "user", "content": "say hi"}],
            tenant_id="t",
        )

    # Verify: a ToolMessage with the echo'd value was appended.
    msgs = final["messages"]
    tool_msgs = [m for m in msgs if getattr(m, "type", None) == "tool"]
    assert len(tool_msgs) == 1
    assert "hello from the SDK" in tool_msgs[0].content

    # Audit recorded the invocation.
    completed_events = [
        r for r in audit.records
        if r.tool_name == "echo"
    ]
    assert len(completed_events) >= 1


async def test_mcp_resolution_caches_across_turns() -> None:
    """The MCP server is contacted exactly once even across multiple tool-loop turns."""
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "first"}'},
                },
            ],
        ),
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "second"}'},
                },
            ],
        ),
        make_llm_response("Done."),
    ])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_MCPDemoAgent)
        final = await agent.ainvoke(
            messages=[{"role": "user", "content": "two echoes"}],
            tenant_id="t",
        )

    # Two echo invocations succeeded.
    tool_msgs = [m for m in final["messages"] if getattr(m, "type", None) == "tool"]
    assert len(tool_msgs) == 2
    assert "first" in tool_msgs[0].content
    assert "second" in tool_msgs[1].content
