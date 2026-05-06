"""Activate the SDK's pytest plugin so its fixtures (fake_audit_sink,
scripted_llm_factory, etc.) are available to every test in this dir."""

pytest_plugins = ["ai_core.testing.pytest_plugin"]
