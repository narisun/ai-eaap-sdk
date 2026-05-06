"""Tests for ai_core.testing.pytest_plugin via pytester."""
from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

# Activate the plugin under test for these specific tests.
pytestmark = pytest.mark.unit


def test_plugin_provides_fake_audit_sink_fixture(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(
        'pytest_plugins = ["ai_core.testing.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        from ai_core.testing import FakeAuditSink

        def test_uses_fake_audit_sink(fake_audit_sink):
            assert isinstance(fake_audit_sink, FakeAuditSink)
            assert fake_audit_sink.records == []
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_plugin_provides_fake_observability_fixture(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(
        'pytest_plugins = ["ai_core.testing.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        from ai_core.testing import FakeObservabilityProvider

        def test_uses_fake_observability(fake_observability):
            assert isinstance(fake_observability, FakeObservabilityProvider)
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_plugin_provides_scripted_llm_factory_fixture(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(
        'pytest_plugins = ["ai_core.testing.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        import pytest

        from ai_core.testing import ScriptedLLM, make_llm_response


        @pytest.mark.asyncio
        async def test_uses_factory(scripted_llm_factory):
            llm = scripted_llm_factory([make_llm_response("hello")])
            assert isinstance(llm, ScriptedLLM)
            out = await llm.complete(model=None, messages=[])
            assert out.content == "hello"
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_plugin_factory_fixtures_accept_kwargs(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(
        'pytest_plugins = ["ai_core.testing.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        from ai_core.testing import FakePolicyEvaluator, FakeSecretManager


        def test_factories(
            fake_policy_evaluator_factory, fake_secret_manager_factory
        ):
            policy = fake_policy_evaluator_factory(default_allow=False)
            assert isinstance(policy, FakePolicyEvaluator)

            secrets = fake_secret_manager_factory({("backend", "key"): "value"})
            assert isinstance(secrets, FakeSecretManager)
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
