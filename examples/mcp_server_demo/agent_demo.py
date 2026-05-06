"""Agent that uses the demo MCP server's tools end-to-end.

Drives a real `BaseAgent` with `mcp_servers()` declaring the local
`server.py` (spawned as a stdio subprocess by the SDK's connection
factory). A `ScriptedLLM` emits a tool_call for `echo`; the agent
runs through the full SDK pipeline (resolution, ToolInvoker, audit)
and prints the result.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from injector import Module, provider, singleton
from rich.console import Console
from rich.panel import Panel

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient  # noqa: TC001
from ai_core.mcp import MCPServerSpec
from ai_core.testing import ScriptedLLM, make_llm_response

if TYPE_CHECKING:
    from collections.abc import Sequence

console = Console()
SERVER_PATH = Path(__file__).parent / "server.py"


class MCPAgent(BaseAgent):
    """Trivial agent that uses the demo server's tools."""

    agent_id: str = "mcp-demo-agent"

    def system_prompt(self) -> str:
        return "Repeat the user's message using the echo tool."

    def mcp_servers(self) -> Sequence[MCPServerSpec]:
        return [
            MCPServerSpec(
                component_id="mcp-demo",
                transport="stdio",
                target=sys.executable,
                args=(str(SERVER_PATH),),
            ),
        ]


def build_container(llm: ILLMClient) -> Container:
    settings = AppSettings(service_name="mcp-agent-demo", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


def _render_tool_message(msg: object) -> str:
    """Unwrap MCPToolSpec's `{"value": ...}` envelope when rendering for the demo.

    The SDK's `_MCPPassthroughOutput` wraps tool results so they fit the
    `output_model.model_validate(...)` contract. Unwrap here so the user-facing
    output shows the actual tool result, not the wrapper.
    """
    raw = str(getattr(msg, "content", ""))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict) and "value" in parsed and len(parsed) == 1:
        return str(parsed["value"])
    return raw


async def main() -> None:
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
        make_llm_response("Echoed successfully."),
    ])
    container = build_container(llm)
    async with container as c:
        agent = c.get(MCPAgent)
        final = await agent.ainvoke(
            messages=[{"role": "user", "content": "say hi"}],
            tenant_id="demo",
        )

    tool_msgs = [m for m in final["messages"] if getattr(m, "type", None) == "tool"]
    rendered = "\n".join(_render_tool_message(m) for m in tool_msgs) or "(no tool messages)"
    console.print(Panel.fit(
        rendered,
        title="MCP tool output",
        border_style="green",
    ))
    console.print(
        "[bold green]Done.[/bold green] Agent invoked the remote MCP echo tool "
        "through the full SDK pipeline (resolution + ToolInvoker + audit)."
    )


if __name__ == "__main__":
    asyncio.run(main())
