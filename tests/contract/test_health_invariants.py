"""IHealthProbe concretes must never raise from probe(); they return a
ProbeResult (potentially with status='down' or 'error') instead.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

# Force-import the probes module so __subclasses__() finds the concretes.
import ai_core.health.probes  # noqa: F401
from ai_core.config.settings import AppSettings, SecuritySettings
from ai_core.health import IHealthProbe, ProbeResult

pytestmark = pytest.mark.contract


def _all_concrete_probes() -> list[type[IHealthProbe]]:
    seen: set[type[IHealthProbe]] = set()
    stack: list[type[IHealthProbe]] = list(IHealthProbe.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    return sorted(
        (c for c in seen if not inspect.isabstract(c)),
        key=lambda c: c.__qualname__,
    )


def _construct_probe_with_failing_dependency(
    probe_cls: type[IHealthProbe],
) -> IHealthProbe:
    name = probe_cls.__qualname__
    if name == "OPAReachabilityProbe":
        # Point at a host:port nothing's listening on.
        settings = AppSettings()
        settings.security = SecuritySettings(opa_url="http://127.0.0.1:1")
        return probe_cls(settings)  # type: ignore[call-arg]
    if name == "DatabaseProbe":
        # Engine that fails on connect.
        engine = create_async_engine(
            "postgresql+asyncpg://x:y@127.0.0.1:1/x", connect_args={"timeout": 1}
        )
        return probe_cls(engine)  # type: ignore[call-arg]
    if name == "ModelLookupProbe":
        # Settings with a model that won't resolve.
        settings = AppSettings()
        return probe_cls(settings)  # type: ignore[call-arg]
    pytest.skip(f"No fault-injection harness defined for {name}")


@pytest.mark.parametrize(
    "probe_cls", _all_concrete_probes(), ids=lambda c: c.__qualname__
)
def test_health_probe_never_raises(probe_cls: type[IHealthProbe]) -> None:
    probe = _construct_probe_with_failing_dependency(probe_cls)
    # Must NOT raise — must return a ProbeResult (even with status="down"/"error").
    result = asyncio.run(probe.probe())
    assert isinstance(result, ProbeResult)


def test_at_least_three_concrete_probes_exist() -> None:
    probes = _all_concrete_probes()
    assert len(probes) >= 3, (
        f"Expected >=3 concrete IHealthProbe subclasses, found "
        f"{[c.__qualname__ for c in probes]}"
    )
