"""Shared pytest fixtures for the SDK test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ai_core.config.settings import AppSettings, get_settings


@pytest.fixture
def clear_settings_cache() -> Iterator[None]:
    """Reset the lru_cache on :func:`get_settings` around a test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fresh_settings() -> AppSettings:
    """Return a freshly-constructed :class:`AppSettings` (no env mutation)."""
    return AppSettings()
