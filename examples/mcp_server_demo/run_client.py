"""Connect to the demo MCP server via the SDK's connection factory.

The SDK ships `FastMCPConnectionFactory` (transport layer), but does NOT
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

from ai_core.exceptions import MCPTransportError

# FastMCPConnectionFactory is the public alias for PoolingMCPConnectionFactory
# (see src/ai_core/mcp/transports.py).
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
                console.print(f"[bold green]echo result:[/bold green] {result.data}")
                return
        except MCPTransportError as exc:
            last_exc = exc
            await asyncio.sleep(0.2)
    raise SystemExit(
        f"Failed to connect to MCP server after 3 attempts: {last_exc!r}\n"
        f"The client spawns server.py as a subprocess — check that "
        f"'{sys.executable}' can execute "
        f"'{Path(__file__).parent / 'server.py'}'."
    )


if __name__ == "__main__":
    asyncio.run(main())
