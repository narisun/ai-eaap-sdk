"""Minimal FastMCP server exposing two trivial tools over stdio.

Run this in one terminal:

    uv run python examples/mcp_server_demo/server.py

Then run the client (in a second terminal):

    uv run python examples/mcp_server_demo/run_client.py
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastmcp import FastMCP

mcp = FastMCP("ai-core-sdk-mcp-demo")


@mcp.tool()
def echo(text: str) -> str:
    """Return the input string verbatim."""
    return text


@mcp.tool()
def current_time() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


if __name__ == "__main__":
    mcp.run()
