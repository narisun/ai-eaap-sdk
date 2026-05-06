"""Connect to the demo MCP server via the SDK's connection factory.

The SDK ships `PoolingMCPConnectionFactory` (transport layer), but does NOT
yet ship an agent-side adapter that registers a remote MCP server as a tool
source. This script demonstrates the surface that's available today:
open a connection, list tools, invoke one.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# FastMCPConnectionFactory is an alias for PoolingMCPConnectionFactory.
# The Pooling-prefixed name isn't in ai_core.mcp.__all__ today; the alias
# is the canonical import path until that's reconciled (Phase 11+).
from ai_core.mcp import FastMCPConnectionFactory, MCPServerSpec

console = Console()


def _server_spec() -> MCPServerSpec:
    """Spawn the demo server as a stdio subprocess."""
    server_path = Path(__file__).parent / "server.py"
    return MCPServerSpec(
        component_id="mcp-demo",
        transport="stdio",
        target=sys.executable,
        args=(str(server_path),),
    )


async def main() -> None:
    factory = FastMCPConnectionFactory(pool_enabled=False)
    spec = _server_spec()

    last_exc: Exception | None = None
    for _attempt in range(3):
        try:
            async with factory.open(spec) as client:
                tools = await client.list_tools()
                console.print(f"[bold]Connected.[/bold] Server exposes {len(tools)} tool(s):")
                table = Table()
                table.add_column("name", style="cyan")
                table.add_column("description")
                for t in tools:
                    table.add_row(t.name, (t.description or "").splitlines()[0])
                console.print(table)

                result = await client.call_tool("echo", {"text": "hello from the SDK"})
                console.print(f"[bold green]echo result:[/bold green] {result}")
                return
        except Exception as exc:  # retry on any transport failure
            last_exc = exc
            await asyncio.sleep(0.2)
    raise SystemExit(
        f"Failed to connect to MCP server after 3 attempts: {last_exc!r}\n"
        f"Is `python examples/mcp_server_demo/server.py` running in another terminal?"
    )


if __name__ == "__main__":
    asyncio.run(main())
