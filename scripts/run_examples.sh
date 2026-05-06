#!/usr/bin/env bash
# Smoke gate for the examples/ directory.
# Exits non-zero if any non-skipped demo fails to complete in 10s.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failures=0
ran=0
skipped=0

run_demo() {
    local name="$1"; shift
    local cmd=("$@")
    echo "===== ${name} ====="
    if timeout 10 "${cmd[@]}"; then
        echo "  ✓ ${name}"
        ran=$((ran + 1))
    else
        local rc=$?
        echo "  ✗ ${name} (exit ${rc})"
        failures=$((failures + 1))
    fi
}

skip_demo() {
    local name="$1"; local reason="$2"
    echo "===== ${name} ====="
    echo "  ↷ skipped — ${reason}"
    skipped=$((skipped + 1))
}

# --- agent_demo: scripted LLM, no network, always runs.
run_demo agent_demo uv run python examples/agent_demo/run.py

# --- testing_demo: pytest tests; always runs.
run_demo testing_demo uv run pytest examples/testing_demo/ -q

# --- mcp_server_demo: needs fastmcp installed.
if uv run python -c "import fastmcp" 2>/dev/null; then
    run_demo mcp_server_demo uv run python examples/mcp_server_demo/run_client.py
else
    skip_demo mcp_server_demo "fastmcp not installed (run \`uv sync\`)"
fi

# --- fastapi_integration: full demo needs OPA running, but the smoke
# gate just verifies the app constructs (DI wires, deps resolve). That
# does not need Docker, so this branch always runs. Export vars so
# they propagate to the `uv run` invocation inside `run_demo`.
export DEMO_JWT_SECRET="dev-secret-smoke-test"
export PYTHONPATH="examples/fastapi_integration${PYTHONPATH:+:$PYTHONPATH}"
run_demo fastapi_integration_import \
    uv run python -c "from app import build_app; build_app()"

echo
echo "summary: ${ran} ran, ${skipped} skipped, ${failures} failed"
exit "${failures}"
