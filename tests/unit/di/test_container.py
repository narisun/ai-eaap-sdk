"""Smoke tests for :mod:`ai_core.di.container` and :mod:`ai_core.di.module`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

import pytest
from injector import Module, provider, singleton

from ai_core.config.secrets import EnvSecretManager, ISecretManager, SecretRef
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container, IStorageProvider
from ai_core.exceptions import DependencyResolutionError


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeSecretManager(ISecretManager):
    async def resolve(self, ref: SecretRef) -> str:
        return f"fake:{ref.name}"


class FakeStorageProvider(IStorageProvider):
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def put_object(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        self.store[key] = body
        return f"fake://{key}"

    async def get_object(self, key: str) -> bytes:
        return self.store[key]

    async def delete_object(self, key: str) -> None:
        self.store.pop(key, None)

    async def list_objects(self, prefix: str) -> AsyncIterator[str]:
        for k in list(self.store):
            if k.startswith(prefix):
                yield k


class FakeStorageModule(Module):
    @singleton
    @provider
    def provide_storage(self) -> IStorageProvider:
        return FakeStorageProvider()


class FakeSecretModule(Module):
    @singleton
    @provider
    def provide_secrets(self) -> ISecretManager:
        return FakeSecretManager()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_default_container_resolves_settings_and_secrets() -> None:
    container = Container.build()

    settings = container.get(AppSettings)
    secrets = container.get(ISecretManager)

    assert isinstance(settings, AppSettings)
    assert isinstance(secrets, EnvSecretManager)


def test_settings_singleton_within_container() -> None:
    container = Container.build()
    a = container.get(AppSettings)
    b = container.get(AppSettings)
    assert a is b


def test_distinct_containers_do_not_share_singletons() -> None:
    c1 = Container.build([AgentModule(settings=AppSettings(service_name="svc-a"))])
    c2 = Container.build([AgentModule(settings=AppSettings(service_name="svc-b"))])
    assert c1.get(AppSettings).service_name == "svc-a"
    assert c2.get(AppSettings).service_name == "svc-b"


def test_constructor_injection_of_settings() -> None:
    custom = AppSettings(service_name="injected-svc")
    container = Container.build([AgentModule(settings=custom)])
    assert container.get(AppSettings) is custom


def test_constructor_injection_of_secret_manager() -> None:
    fake = FakeSecretManager()
    container = Container.build([AgentModule(secret_manager=fake)])
    assert container.get(ISecretManager) is fake


def test_override_returns_new_container_with_extra_bindings() -> None:
    base = Container.build()
    overridden = base.override(FakeStorageModule())

    # Original container still cannot resolve IStorageProvider.
    assert IStorageProvider not in base
    # Override container can.
    storage = overridden.get(IStorageProvider)
    assert isinstance(storage, FakeStorageProvider)


def test_override_last_binding_wins() -> None:
    fake1 = FakeSecretManager()

    class Mod1(Module):
        @singleton
        @provider
        def p(self) -> ISecretManager:
            return fake1

    fake2 = FakeSecretManager()

    class Mod2(Module):
        @singleton
        @provider
        def p(self) -> ISecretManager:
            return fake2

    container = Container.build([AgentModule(), Mod1(), Mod2()])
    assert container.get(ISecretManager) is fake2


def test_unresolvable_binding_raises_eaap_error() -> None:
    container = Container.build()
    with pytest.raises(DependencyResolutionError) as ei:
        container.get(IStorageProvider)
    assert "IStorageProvider" in str(ei.value.details["interface"])
