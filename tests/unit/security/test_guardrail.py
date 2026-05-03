"""Unit tests for :class:`ai_core.security.guardrail.GuardrailNode`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from ai_core.agents.state import new_agent_state
from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision
from ai_core.exceptions import PolicyDenialError
from ai_core.security.guardrail import GuardrailNode


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakePolicy(IPolicyEvaluator):
    def __init__(
        self,
        *,
        allow: bool | dict[str, bool] = True,
        raises: bool = False,
    ) -> None:
        self._allow = allow
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def evaluate(
        self,
        *,
        decision_path: str,
        input: Mapping[str, Any],
    ) -> PolicyDecision:
        self.calls.append({"decision_path": decision_path, "input": dict(input)})
        if self._raises:
            raise PolicyDenialError("opa down")
        if isinstance(self._allow, dict):
            tool_name = input.get("tool", {}).get("name", "")
            allowed = self._allow.get(tool_name, False)
            return PolicyDecision(
                allowed=allowed,
                obligations={},
                reason=None if allowed else f"{tool_name}-not-permitted",
            )
        return PolicyDecision(allowed=self._allow, obligations={}, reason=None)


def _state_with_tool_calls(tool_calls: list[dict[str, Any]]) -> Any:
    return new_agent_state(
        initial_messages=[
            {"role": "user", "content": "do the thing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
        ],
        essential={"user_id": "u-1", "tenant_id": "t-9", "session_id": "s-3"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
async def test_no_tool_calls_passes_through_unchanged() -> None:
    policy = FakePolicy()
    node = GuardrailNode(policy, agent_id="a-1")
    state = new_agent_state(
        initial_messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    result = await node.run(state)
    assert result is state
    assert policy.calls == []


async def test_all_calls_allowed_passes_through() -> None:
    policy = FakePolicy(allow=True)
    node = GuardrailNode(policy, agent_id="a-1")
    state = _state_with_tool_calls(
        [
            {"id": "c1", "function": {"name": "search", "arguments": "{}"}},
            {"id": "c2", "function": {"name": "fetch", "arguments": "{}"}},
        ]
    )
    result = await node.run(state)
    assert result is state
    assert len(policy.calls) == 2
    paths = {call["decision_path"] for call in policy.calls}
    assert paths == {"eaap/agent/tool_call/allow"}


async def test_partial_denial_returns_denial_message() -> None:
    policy = FakePolicy(allow={"search": True, "delete_everything": False})
    node = GuardrailNode(policy, agent_id="a-1")
    state = _state_with_tool_calls(
        [
            {"id": "c1", "function": {"name": "search", "arguments": "{}"}},
            {"id": "c2", "function": {"name": "delete_everything", "arguments": "{}"}},
        ]
    )
    result = await node.run(state)
    assert result is not state
    msgs = result["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert msgs[0]["name"] == "guardrail"
    assert "delete_everything" in msgs[0]["content"]
    assert "delete_everything-not-permitted" in msgs[0]["content"]
    # Allowed call is NOT mentioned in the denial bullets.
    bullets = msgs[0]["content"].split("\n")
    deny_bullets = [b for b in bullets if b.startswith("- ")]
    assert len(deny_bullets) == 1


async def test_fail_closed_treats_evaluator_error_as_deny() -> None:
    policy = FakePolicy(raises=True)
    node = GuardrailNode(policy, agent_id="a-1")
    state = _state_with_tool_calls(
        [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}]
    )
    result = await node.run(state)
    assert result is not state
    msg = result["messages"][0]
    assert "policy-evaluator-error" in msg["content"]


async def test_essentials_forwarded_to_policy_input() -> None:
    policy = FakePolicy(allow=True)
    node = GuardrailNode(policy, agent_id="a-1")
    await node.run(
        _state_with_tool_calls([{"id": "c1", "function": {"name": "search", "arguments": "{}"}}])
    )
    assert policy.calls[0]["input"]["user"] == "u-1"
    assert policy.calls[0]["input"]["tenant_id"] == "t-9"
    assert policy.calls[0]["input"]["session_id"] == "s-3"
    assert policy.calls[0]["input"]["agent_id"] == "a-1"
    assert policy.calls[0]["input"]["action"] == "tool.invoke"


async def test_alternate_call_shape_handled() -> None:
    """Some providers serialise tool calls as flat ``{name, arguments}``."""
    policy = FakePolicy(allow=True)
    node = GuardrailNode(policy, agent_id="a-1")
    await node.run(_state_with_tool_calls([{"id": "c1", "name": "search", "arguments": "{}"}]))
    assert policy.calls[0]["input"]["tool"]["name"] == "search"


async def test_custom_decision_path() -> None:
    policy = FakePolicy(allow=True)
    node = GuardrailNode(policy, agent_id="a-1", decision_path="custom/path")
    await node.run(_state_with_tool_calls([{"id": "c1", "function": {"name": "x", "arguments": "{}"}}]))
    assert policy.calls[0]["decision_path"] == "custom/path"
