"""Agent that uses the demo MCP server's documentation resource end-to-end.

Drives a real `BaseAgent` with `mcp_servers()` declaring the local
`server.py` (spawned as a stdio subprocess by the SDK's connection
factory). A `ScriptedLLM` emits a tool_call for the `documentation`
resource; the agent runs through the full SDK pipeline (resolution,
ToolInvoker, audit) and prints the result via `unwrap_mcp_tool_message`.
"""
from __future__ import annotations

import asyncio
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
        return "Read the documentation resource to learn what this server provides."

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
    """Render a ToolMessage's content using the SDK's unwrap helper."""
    from ai_core.mcp import unwrap_mcp_tool_message  # noqa: PLC0415
    raw = str(getattr(msg, "content", ""))
    unwrapped = unwrap_mcp_tool_message(raw)
    return unwrapped if isinstance(unwrapped, str) else str(unwrapped)


async def main() -> None:
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "documentation",
                        "arguments": "{}",
                    },
                },
            ],
        ),
        make_llm_response("Read the docs."),
    ])
    container = build_container(llm)
    async with container as c:
        agent = c.get(MCPAgent)
        final = await agent.ainvoke(
            messages=[{"role": "user", "content": "what does this server provide?"}],
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
        "[bold green]Done.[/bold green] Agent read the documentation resource "
        "through the full SDK pipeline (resolution + ToolInvoker + audit)."
    )


if __name__ == "__main__":
    asyncio.run(main())
