# mcp_server_demo — FastMCP server + SDK connection factory

Demonstrates the SDK's MCP transport surface today: a FastMCP server
plus a client that uses `FastMCPConnectionFactory` to connect, list
tools, and invoke one.

## What this demonstrates

- Standing up a FastMCP server with `@mcp.tool()` decorators.
- Using `MCPServerSpec` + `FastMCPConnectionFactory` from
  `ai_core.mcp` to open a connection.
- Listing and invoking tools through the FastMCP client.
- Agent-side resolution: `BaseAgent.mcp_servers()` makes MCP tools
  available to `ToolInvoker` without any hand-written bridge.

## What's also shown (Phase 11)

- Agent-side adapter: `agent_demo.py` runs an agent that declares this
  server via `mcp_servers()` and uses its tools through the standard
  `ToolInvoker` pipeline (validation, OPA, audit, observability).

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

### Run as an agent

```bash
uv run python examples/mcp_server_demo/agent_demo.py
```

The agent declares this server via `mcp_servers()`, the SDK resolves
its tools on the first turn, and `ToolInvoker` dispatches the call
through the same pipeline as local `@tool`s.

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
