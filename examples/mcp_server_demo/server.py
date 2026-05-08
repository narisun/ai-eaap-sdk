"""Minimal FastMCP server exposing two trivial tools over stdio.

Normally you don't run this file directly — `run_client.py` spawns it
as a stdio subprocess. Run it standalone only when you want to inspect
the server independently (e.g. with the FastMCP CLI):

    uv run python examples/mcp_server_demo/server.py
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastmcp import FastMCP

mcp = FastMCP("ai-eaap-sdk-mcp-demo")


@mcp.tool()
def echo(text: str) -> str:
    """Return the input string verbatim."""
    return text


@mcp.tool()
def current_time() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


@mcp.resource("mcp-demo://documentation")
def documentation() -> str:
    """Project documentation — exposed as an MCP resource."""
    return (
        "This is the demo MCP server's documentation.\n\n"
        "It exposes:\n"
        "  - echo(text): repeat a string\n"
        "  - current_time(): UTC ISO-8601 timestamp\n"
        "  - documentation: this resource\n"
        "  - summarize_text(text): a prompt template"
    )


@mcp.prompt()
def summarize_text(text: str) -> str:
    """Generate a summarization prompt for a given text."""
    return f"Please summarize the following text in one sentence:\n\n{text}"


if __name__ == "__main__":
    mcp.run()
