"""Pytest configuration for contract tests.

Contract tests pin SDK promises (public surface, never-raise contracts,
error_code mirroring, container lifecycle) at Phase 6's end. They run
in-process with no infrastructure — the Docker-conditional integration
tests live under tests/integration/.

Each test file in this directory declares::

    pytestmark = pytest.mark.contract

at module top, which scopes the marker correctly without affecting
tests outside this directory (a directory-level conftest hook would
receive ALL session-collected items and mis-tag siblings).
"""
from __future__ import annotations
