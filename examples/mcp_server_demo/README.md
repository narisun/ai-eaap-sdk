# mcp_server_demo — FastMCP server + SDK connection factory

Demonstrates the SDK's MCP transport surface today: a FastMCP server
plus a client that uses `FastMCPConnectionFactory` to connect, list
tools, and invoke one.

## What this demonstrates

- Standing up a FastMCP server with `@mcp.tool()` decorators.
- Using `MCPServerSpec` + `FastMCPConnectionFactory` from
  `ai_core.mcp` to open a connection.
- Listing and invoking tools through the FastMCP client.

## What's not (yet) shown

The SDK does **not** yet ship an agent-side adapter that registers a
remote MCP server as a `ToolInvoker` tool source. That integration is
on the roadmap. For now the connection factory is the public surface
you can use to bridge agents and MCP servers (you'd write the bridge
yourself).

## Prerequisites

```bash
uv sync
```

`fastmcp` is a core SDK dependency — no extras needed.

## Run

```bash
uv run python examples/mcp_server_demo/run_client.py
```

The client auto-spawns `server.py` as a stdio subprocess and tears it down
when the connection closes. Output:

```
Connected. Server exposes 2 tool(s):
  echo         — Return the input string verbatim.
  current_time — Return the current UTC time as an ISO-8601 string.
echo result: hello from the SDK
```

### Inspecting the server manually

To poke at the server independently (e.g. with the FastMCP CLI), run it
in a separate terminal:

```bash
uv run python examples/mcp_server_demo/server.py
```

This is purely for experimentation — `run_client.py` always spawns its
own server process and ignores any other instance.

## Add your own tool

In `server.py`:

```python
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b
```

Restart the server, rerun the client, and the new tool will appear in
the table.

## What to read next

- `src/ai_core/mcp/transports.py` — `MCPServerSpec` and
  `PoolingMCPConnectionFactory`.
- FastMCP docs: https://github.com/jlowin/fastmcp
