"""Smoke tests for the ``eaap`` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_core.cli.main import app


pytestmark = pytest.mark.unit


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# eaap init
# ---------------------------------------------------------------------------
def test_init_scaffolds_full_project_tree(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "my-app", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    project = tmp_path / "my-app"
    assert (project / "pyproject.toml").is_file()
    assert (project / "docker-compose.yaml").is_file()
    assert (project / "otel-collector-config.yaml").is_file()
    assert (project / "policies" / "eaap.rego").is_file()
    assert (project / ".env.example").is_file()
    assert (project / ".gitignore").is_file()
    assert (project / "README.md").is_file()
    assert (project / "src" / "my_app" / "__init__.py").is_file()
    assert (project / "src" / "my_app" / "main.py").is_file()


def test_init_renders_project_specific_values(runner: CliRunner, tmp_path: Path) -> None:
    runner.invoke(app, ["init", "my-app", "--path", str(tmp_path)])
    project = tmp_path / "my-app"

    pyproject = (project / "pyproject.toml").read_text()
    assert 'name = "my-app"' in pyproject
    assert 'packages = ["src/my_app"]' in pyproject

    main_py = (project / "src" / "my_app" / "main.py").read_text()
    assert 'title="my-app"' in main_py


def test_init_compose_includes_postgres_opa_otel(runner: CliRunner, tmp_path: Path) -> None:
    runner.invoke(app, ["init", "my-app", "--path", str(tmp_path)])
    compose = (tmp_path / "my-app" / "docker-compose.yaml").read_text()
    assert "postgres:16" in compose
    assert "openpolicyagent/opa" in compose
    assert "otel/opentelemetry-collector" in compose


def test_init_rejects_invalid_name(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "BadName", "--path", str(tmp_path)])
    assert result.exit_code != 0
    assert "lowercase" in result.output


def test_init_refuses_non_empty_target_without_force(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "my-app"
    target.mkdir()
    (target / "existing.txt").write_text("hi")
    result = runner.invoke(app, ["init", "my-app", "--path", str(tmp_path)])
    assert result.exit_code != 0


def test_init_force_overrides_non_empty_target(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "my-app"
    target.mkdir()
    (target / "stale.txt").write_text("old")
    result = runner.invoke(
        app, ["init", "my-app", "--path", str(tmp_path), "--force"]
    )
    assert result.exit_code == 0, result.output
    assert (target / "pyproject.toml").is_file()


# ---------------------------------------------------------------------------
# eaap generate agent
# ---------------------------------------------------------------------------
def test_generate_agent_writes_module_and_test(runner: CliRunner, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    result = runner.invoke(
        app,
        [
            "generate", "agent", "support_triage",
            "--path", str(src),
            "--package", "agents",
        ],
    )
    assert result.exit_code == 0, result.output
    agent_file = src / "agents" / "support_triage.py"
    test_file = tmp_path / "tests" / "unit" / "agents" / "test_support_triage.py"
    assert agent_file.is_file()
    assert test_file.is_file()

    body = agent_file.read_text()
    assert "class SupportTriageAgent(BaseAgent):" in body
    assert 'agent_id: str = "support-triage"' in body


def test_generate_agent_rejects_invalid_identifier(
    runner: CliRunner, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, ["generate", "agent", "BadName", "--path", str(tmp_path / "src")]
    )
    assert result.exit_code != 0
    assert "identifier" in result.output


# ---------------------------------------------------------------------------
# eaap generate mcp
# ---------------------------------------------------------------------------
def test_generate_mcp_writes_server_and_test(runner: CliRunner, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    result = runner.invoke(
        app,
        [
            "generate", "mcp", "ticketing",
            "--path", str(src),
            "--package", "mcp_servers",
        ],
    )
    assert result.exit_code == 0, result.output
    server_file = src / "mcp_servers" / "ticketing.py"
    test_file = tmp_path / "tests" / "unit" / "mcp_servers" / "test_ticketing.py"
    assert server_file.is_file()
    assert test_file.is_file()
    body = server_file.read_text()
    assert "from fastmcp import FastMCP" in body
    assert "class TicketingEchoIn(BaseModel):" in body
    assert "ticketing_echo" in body
