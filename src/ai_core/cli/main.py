"""``eaap`` CLI — project & component scaffolding.

Subcommands:

* ``eaap init NAME [--path PATH]`` — scaffold a new EAAP project at
  ``PATH/NAME`` containing pyproject, docker-compose (Postgres, OPA,
  OTel collector), .env template, and a runnable ``main.py``.
* ``eaap generate agent NAME [--path PATH]`` — render a LangGraph agent
  class plus a unit test stub.
* ``eaap generate mcp NAME [--path PATH]`` — render a FastMCP server
  with one example tool, plus a unit test stub.

Templates live under :mod:`ai_core.cli.templates` and are loaded via
:func:`importlib.resources` so the CLI works after installation.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any

import typer
from jinja2 import Environment, StrictUndefined
from rich.console import Console

from ai_core.cli.scaffold import (
    AGENT_TEMPLATES,
    INIT_TEMPLATE_PACKAGE,
    MCP_TEMPLATES,
    iter_template_files,
    render_template,
)

console = Console()
app = typer.Typer(help="Enterprise Agentic AI Platform — scaffolding CLI", no_args_is_help=True)
generate_app = typer.Typer(help="Generate a new SDK component", no_args_is_help=True)
app.add_typer(generate_app, name="generate")


_IDENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_project_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise typer.BadParameter(
            "Project name must start with a lowercase letter and contain only "
            "lowercase letters, digits, and dashes."
        )
    return name


def _validate_python_identifier(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise typer.BadParameter(
            "Name must be a valid Python identifier: start with a lowercase "
            "letter and contain only lowercase letters, digits, and underscores."
        )
    return name


def _to_class_name(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_file(target: Path, content: str, *, force: bool) -> bool:
    if target.exists() and not force:
        console.print(f"[yellow]skip[/yellow]   {target} (exists; use --force to overwrite)")
        return False
    _ensure_dir(target.parent)
    target.write_text(content, encoding="utf-8")
    console.print(f"[green]wrote[/green]  {target}")
    return True


def _jinja_env() -> Environment:
    return Environment(
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )


# ---------------------------------------------------------------------------
# eaap init
# ---------------------------------------------------------------------------
@app.command()
def init(
    name: str = typer.Argument(..., help="Project name (kebab-case, e.g. 'my-eaap-app')."),
    path: Path = typer.Option(
        Path.cwd(), "--path", "-p", help="Directory under which the project will be created."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files."),
) -> None:
    """Scaffold a new EAAP project at ``PATH/NAME``."""
    project_name = _validate_project_name(name)
    package = project_name.replace("-", "_")
    target = path / project_name

    if target.exists() and any(target.iterdir()) and not force:
        raise typer.BadParameter(
            f"Directory {target} is not empty — use --force to overwrite."
        )

    env = _jinja_env()
    context: dict[str, Any] = {
        "project_name": project_name,
        "project_package": package,
        "service_name": project_name,
    }

    written = 0
    for rel_path, body in iter_template_files(INIT_TEMPLATE_PACKAGE):
        # Filename rewrites:
        #   - PROJECT placeholder for the project's snake-case package name
        #   - _DOT_ prefix becomes a leading "." (workaround for the fact that
        #     dotfiles can't be reliably packaged inside a Python distribution)
        #   - .j2 suffix is stripped after rendering
        rewritten = rel_path.replace("PROJECT", package).replace("_DOT_", ".")
        if rewritten.endswith(".j2"):
            rewritten = rewritten[: -len(".j2")]
        rendered_body = render_template(env, body, context)
        if _write_file(target / rewritten, rendered_body, force=force):
            written += 1

    console.print(
        f"\n[bold green]✓[/bold green] Scaffolded {project_name} "
        f"({written} files) at {target}\n"
        f"Next steps:\n"
        f"  cd {project_name}\n"
        f"  cp .env.example .env\n"
        f"  docker compose up -d\n"
        f"  python -m {package}.main"
    )


# ---------------------------------------------------------------------------
# eaap generate agent
# ---------------------------------------------------------------------------
@generate_app.command("agent")
def generate_agent(
    name: str = typer.Argument(..., help="Agent name (snake_case, e.g. 'support_triage')."),
    path: Path = typer.Option(
        Path("src"), "--path", "-p", help="Source directory under which to write files."
    ),
    package: str = typer.Option(
        "agents", "--package", help="Sub-package (relative to --path) to write the agent into."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files."),
) -> None:
    """Generate a LangGraph agent class with pre-wired DI and a unit test."""
    snake = _validate_python_identifier(name)
    class_name = _to_class_name(snake) + "Agent"

    env = _jinja_env()
    context: dict[str, Any] = {
        "agent_name": snake,
        "agent_id": snake.replace("_", "-"),
        "class_name": class_name,
        "module_path": f"{package.replace('/', '.')}.{snake}",
    }

    src_dir = path / package
    test_dir = path.parent / "tests" / "unit" / package

    written = 0
    for rel_path, body in AGENT_TEMPLATES.items():
        rendered = render_template(env, body, context)
        rendered_filename = render_template(env, rel_path, context)
        target = (test_dir if rendered_filename.startswith("test_") else src_dir) / rendered_filename
        if _write_file(target, rendered, force=force):
            written += 1

    console.print(
        f"\n[bold green]✓[/bold green] Generated agent '{class_name}' "
        f"({written} files). Wire it via DI:\n"
        f"  container = Container.build([AgentModule()])\n"
        f"  agent = container.get({class_name})\n"
    )


# ---------------------------------------------------------------------------
# eaap generate mcp
# ---------------------------------------------------------------------------
@generate_app.command("mcp")
def generate_mcp(
    name: str = typer.Argument(..., help="Server name (snake_case, e.g. 'ticketing')."),
    path: Path = typer.Option(
        Path("src"), "--path", "-p", help="Source directory under which to write files."
    ),
    package: str = typer.Option(
        "mcp_servers", "--package", help="Sub-package (relative to --path) to write the server into."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files."),
) -> None:
    """Generate a FastMCP server with example tool definitions."""
    snake = _validate_python_identifier(name)
    class_prefix = _to_class_name(snake)

    env = _jinja_env()
    context: dict[str, Any] = {
        "server_name": snake,
        "class_prefix": class_prefix,
        "tool_name": f"{snake}_echo",
        "module_path": f"{package.replace('/', '.')}.{snake}",
    }

    src_dir = path / package
    test_dir = path.parent / "tests" / "unit" / package

    written = 0
    for rel_path, body in MCP_TEMPLATES.items():
        rendered = render_template(env, body, context)
        rendered_filename = render_template(env, rel_path, context)
        target = (test_dir if rendered_filename.startswith("test_") else src_dir) / rendered_filename
        if _write_file(target, rendered, force=force):
            written += 1

    console.print(
        f"\n[bold green]✓[/bold green] Generated MCP server '{snake}' "
        f"({written} files). Run it with:\n"
        f"  python -m {package}.{snake}\n"
    )


def main() -> None:
    """Console entry point used by ``[project.scripts] eaap = ai_core.cli.main:app``."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
