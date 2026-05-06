"""Minimal FastMCP server exposing two trivial tools over stdio.

Normally you don't run this file directly — `run_client.py` spawns it
as a stdio subprocess. Run it standalone only when you want to inspect
the server independently (e.g. with the FastMCP CLI):

    uv run python examples/mcp_server_demo/server.py
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
