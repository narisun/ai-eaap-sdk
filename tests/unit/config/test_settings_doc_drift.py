"""Drift gate: docs/settings.md must equal the generator's output.

Failure means somebody changed AppSettings without regenerating the
reference. Fix: run `uv run python scripts/generate_settings_doc.py`
and commit the regenerated file.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from scripts.generate_settings_doc import render

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED = REPO_ROOT / "docs" / "settings.md"


def test_settings_doc_is_not_stale() -> None:
    expected = render()
    actual = COMMITTED.read_text(encoding="utf-8")
    assert actual == expected, (
        "docs/settings.md is stale; run "
        "`uv run python scripts/generate_settings_doc.py` and commit."
    )
