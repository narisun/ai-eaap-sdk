"""Application-invoked prompt demo.

Demonstrates the second integration path Phase 12 ships: the application
fetches a prompt template from the MCP server, splices the resulting
messages into ainvoke(messages=...), and runs the agent.

Run from the repo root::

    uv run python examples/mcp_server_demo/prompt_demo.py
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


class _PromptDemoAgent(BaseAgent):
    agent_id: str = "prompt-demo-agent"

    def system_prompt(self) -> str:
        return "Respond concisely."

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
    settings = AppSettings(service_name="prompt-demo", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


async def main() -> None:
    # Scripted final assistant response — what the LLM would say after
    # consuming the templated prompt.
    llm = ScriptedLLM([make_llm_response("This is a one-sentence summary.")])
    container = build_container(llm)

    async with container as c:
        agent = c.get(_PromptDemoAgent)

        # Step 1: discover available prompts.
        prompts = await agent.list_prompts()
        console.print(Panel.fit(
            "\n".join(f"- {p.name}: {p.description or '(no description)'}" for p in prompts),
            title="Available prompts",
            border_style="cyan",
        ))

        # Step 2: fetch a templated prompt.
        templated_messages = await agent.get_prompt(
            "summarize_text",
            arguments={"text": "MCP is a protocol for tool/resource/prompt servers."},
        )

        # Step 3: splice the templated messages into ainvoke().
        # Convert to the openai-format dicts ainvoke expects.
        seed_messages = [
            {"role": m.role, "content": m.content}
            for m in templated_messages
        ]
        final = await agent.ainvoke(messages=seed_messages, tenant_id="demo")

    last = final["messages"][-1]
    console.print(Panel.fit(
        str(getattr(last, "content", "(empty)")),
        title="Agent response",
        border_style="green",
    ))
    console.print(
        "[bold green]Done.[/bold green] App fetched the summarize_text prompt, "
        "spliced its messages into ainvoke(), and the agent ran."
    )


if __name__ == "__main__":
    asyncio.run(main())
