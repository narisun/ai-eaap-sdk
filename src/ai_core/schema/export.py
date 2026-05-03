"""Export :class:`SchemaRegistry` records to JSON Schema files.

Used by ``eaap schema export`` and importable directly so host code can
generate documentation / SDK stubs / cross-language clients in any
build pipeline.

File naming
-----------
For each registered record, two files are written::

    <out>/<name>.v<version>.input.json
    <out>/<name>.v<version>.output.json

Names are slugged (``/`` and other separators rejected at registration
time) so they're safe as file names on any platform.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_core.schema.registry import SchemaRegistry


def export_schemas(
    registry: SchemaRegistry,
    out_dir: Path,
    *,
    indent: int = 2,
    overwrite: bool = True,
) -> list[Path]:
    """Write JSON Schema files for every record in ``registry``.

    Args:
        registry: Populated :class:`SchemaRegistry`.
        out_dir: Output directory (created if absent).
        indent: ``json.dump`` indentation. ``0`` produces compact output.
        overwrite: When ``False`` and a target file exists, the file is
            left untouched and skipped from the return list.

    Returns:
        Paths of every file written, in deterministic order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record in registry.iter_records():
        for kind, model in (
            ("input", record.input_schema),
            ("output", record.output_schema),
        ):
            target = out_dir / f"{record.name}.v{record.version}.{kind}.json"
            if target.exists() and not overwrite:
                continue
            schema = model.model_json_schema()
            schema = _decorate_with_provenance(schema, record=record, kind=kind)
            target.write_text(
                json.dumps(schema, indent=indent or None, sort_keys=True),
                encoding="utf-8",
            )
            written.append(target)
    return written


def _decorate_with_provenance(
    schema: dict[str, object],
    *,
    record: object,
    kind: str,
) -> dict[str, object]:
    """Attach SDK-specific metadata to the exported schema."""
    rec = getattr(record, "name", "?")
    ver = getattr(record, "version", "?")
    desc = getattr(record, "description", "") or ""
    decorated: dict[str, object] = {
        "$id": f"eaap://schema/{rec}/v{ver}/{kind}",
        "x-eaap-name": rec,
        "x-eaap-version": ver,
        "x-eaap-kind": kind,
    }
    if desc:
        decorated["x-eaap-description"] = desc
    decorated.update(schema)
    return decorated


__all__ = ["export_schemas"]
