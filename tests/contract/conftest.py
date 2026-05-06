"""Pytest configuration for contract tests.

Contract tests pin SDK promises (public surface, never-raise contracts,
error_code mirroring, container lifecycle) at Phase 6's end. They run
in-process with no infrastructure — the Docker-conditional integration
tests live under tests/integration/.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Tag every test in this directory with pytest.mark.contract."""
    for item in items:
        item.add_marker(pytest.mark.contract)
