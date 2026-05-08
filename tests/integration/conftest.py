"""Session-scoped Testcontainers fixtures with Docker-availability gating.

Every test in tests/integration/ skips automatically when Docker is
unavailable. When Docker is up, fixtures spin up Postgres + OPA once
per session and tests share them.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from testcontainers.core.container import DockerContainer
    from testcontainers.postgres import PostgresContainer

# Locate the eaap init starter policies — used by OPA fixture (Task 3).
POLICIES_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "ai_core" / "cli" / "templates" / "init" / "policies"
)
OPA_IMAGE = "openpolicyagent/opa:0.66.0"
POSTGRES_IMAGE = "postgres:16"


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Probe the Docker socket; return False if unreachable."""
    try:
        from testcontainers.core.docker_client import DockerClient
        DockerClient().client.ping()
    except Exception:  # broad catch is intentional — any failure means Docker is unusable
        return False
    return True


@pytest.fixture(scope="session")
def postgres_container(
    docker_available: bool,
) -> Iterator[PostgresContainer]:
    if not docker_available:
        pytest.skip("Docker not available — integration tests skipped")
    from testcontainers.postgres import PostgresContainer
    with PostgresContainer(POSTGRES_IMAGE) as pg:
        yield pg


@pytest.fixture(scope="session")
def opa_container(
    docker_available: bool,
) -> Iterator[DockerContainer]:
    if not docker_available:
        pytest.skip("Docker not available — integration tests skipped")
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.wait_strategies import LogMessageWaitStrategy
    container = (
        DockerContainer(OPA_IMAGE)
        .with_command("run --server --addr 0.0.0.0:8181 /policies")
        .with_volume_mapping(str(POLICIES_DIR), "/policies", "ro")
        .with_exposed_ports(8181)
        # OPA 0.59+ emits "Initializing server." then binds the listener.
        # Older releases emitted "Server started" — neither phrase is a
        # documented contract, so we match the modern one.
        .waiting_for(LogMessageWaitStrategy("Initializing server"))
    )
    with container as opa:
        yield opa


# Each test file in this directory declares::
#
#     pytestmark = pytest.mark.integration
#
# at module top, which scopes the marker correctly without affecting tests
# outside this directory (a directory-level conftest hook would receive ALL
# session-collected items and mis-tag siblings — see tests/contract/conftest.py
# for the same lesson).
