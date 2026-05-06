"""Generate docs/settings.md from the AppSettings Pydantic schema.

Walks AppSettings.model_fields, recursing into nested BaseSettings
groups (database, llm, audit, ...). Each group gets its own section
with a markdown table: field | type | default | env var | description.

Run modes:

* `python scripts/generate_settings_doc.py` — regenerate `docs/settings.md`.
* `python scripts/generate_settings_doc.py --check` — exit 1 if the
  committed file differs from what would be generated; useful as a
  CI gate alongside the pytest drift test.

The output is committed at docs/settings.md and gated by
tests/unit/config/test_settings_doc_drift.py.
"""
from __future__ import annotations

import enum
import io
import sys
from pathlib import Path
from typing import TYPE_CHECKING, get_args, get_origin

from pydantic_settings import BaseSettings

from ai_core.config.settings import AppSettings

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "settings.md"
ENV_PREFIX = "EAAP_"
ENV_DELIM = "__"


def _format_type(annotation: object) -> str:
    """Render a Pydantic field annotation as readable markdown."""
    if annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    args = ", ".join(
        _format_type(a) if isinstance(a, type) else repr(a)
        for a in get_args(annotation)
    )
    return f"{getattr(origin, '__name__', str(origin))}[{args}]"


def _format_default(field: FieldInfo) -> str:
    if field.default_factory is not None:
        return "*(factory)*"
    default = field.default
    if default is None:
        return "`None`"
    if isinstance(default, str) and not default:
        return "`\"\"`"
    if isinstance(default, enum.Enum):
        return f"`{default.value!r}`"
    return f"`{default!r}`"


def _env_name(group_path: tuple[str, ...], field_name: str) -> str:
    """Build the EAAP_<GROUP>__<FIELD> env var name (uppercase)."""
    parts = (*group_path, field_name)
    return ENV_PREFIX + ENV_DELIM.join(p.upper() for p in parts)


def _render_table(model_cls: type[BaseSettings], group_path: tuple[str, ...]) -> str:
    rows: list[str] = []
    for name, field in model_cls.model_fields.items():
        try:
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseSettings):
                continue  # skip — rendered in its own section
        except TypeError:
            pass
        annotation = _format_type(field.annotation)
        default = _format_default(field)
        env = _env_name(group_path, name)
        description = (field.description or "").replace("|", "\\|").replace("\n", " ")
        rows.append(f"| `{name}` | `{annotation}` | {default} | `{env}` | {description} |")

    if not rows:
        return "_No direct fields — see nested groups below._\n"
    return (
        "| Field | Type | Default | Env var | Description |\n"
        "| --- | --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )


def _nested_groups(model_cls: type[BaseSettings]) -> list[tuple[str, type[BaseSettings]]]:
    """Discover BaseSettings-typed nested fields and return (name, type) pairs."""
    out: list[tuple[str, type[BaseSettings]]] = []
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseSettings):
                out.append((name, annotation))
        except TypeError:
            continue
    return out


def render() -> str:
    out = io.StringIO()
    out.write(
        "# Settings reference\n\n"
        "Auto-generated from `src/ai_core/config/settings.py`. "
        "Do not edit by hand — run `uv run python scripts/generate_settings_doc.py` "
        "to regenerate.\n\n"
        "Environment variable names use the prefix `EAAP_` and `__` as the "
        "nested-group delimiter (e.g. `EAAP_DATABASE__DSN`).\n\n"
    )

    out.write("## AppSettings\n\n")
    out.write(_render_table(AppSettings, ()))
    out.write("\n")

    for name, group_cls in _nested_groups(AppSettings):
        out.write(f"## {group_cls.__name__} (`AppSettings.{name}`)\n\n")
        if group_cls.__doc__:
            first_line = group_cls.__doc__.strip().splitlines()[0]
            out.write(f"{first_line}\n\n")
        out.write(_render_table(group_cls, (name,)))
        out.write("\n")

    return out.getvalue()


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    content = render()
    if "--check" in argv:
        if not OUTPUT.exists():
            sys.stderr.write(
                f"{OUTPUT.relative_to(REPO_ROOT)} does not exist; "
                "run `python scripts/generate_settings_doc.py` first\n"
            )
            return 1
        committed = OUTPUT.read_text(encoding="utf-8")
        if committed == content:
            return 0
        sys.stderr.write(
            f"{OUTPUT.relative_to(REPO_ROOT)} is stale; "
            "run `python scripts/generate_settings_doc.py` and commit.\n"
        )
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    sys.stdout.write(f"wrote {OUTPUT.relative_to(REPO_ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
