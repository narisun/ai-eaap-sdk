"""Smoke tests for :mod:`ai_core.config.secrets`."""

from __future__ import annotations

import pytest

from ai_core.config.secrets import EnvSecretManager, SecretRef
from ai_core.exceptions import SecretResolutionError

pytestmark = pytest.mark.unit


async def test_env_secret_manager_resolves_present_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_TOKEN", "shhh")
    mgr = EnvSecretManager()
    value = await mgr.resolve(SecretRef(backend="env", name="MY_TOKEN"))
    assert value == "shhh"


async def test_env_secret_manager_raises_for_missing_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    mgr = EnvSecretManager()
    with pytest.raises(SecretResolutionError) as ei:
        await mgr.resolve(SecretRef(backend="env", name="DOES_NOT_EXIST"))
    assert ei.value.details["name"] == "DOES_NOT_EXIST"


async def test_env_secret_manager_rejects_other_backend() -> None:
    mgr = EnvSecretManager()
    with pytest.raises(SecretResolutionError):
        await mgr.resolve(SecretRef(backend="aws-sm", name="arn:aws:..."))


async def test_resolve_optional_returns_none() -> None:
    mgr = EnvSecretManager()
    assert await mgr.resolve_optional(None) is None


def test_secret_ref_str_representation() -> None:
    ref = SecretRef(backend="env", name="API_KEY")
    assert str(ref) == "secret://env/API_KEY"
    versioned = SecretRef(backend="aws-sm", name="prod/db", version="v3")
    assert str(versioned) == "secret://aws-sm/prod/db@v3"
