"""Tests for the demo agent using ai_core.testing's public surface.

This file showcases the three big use-cases:

1. ScriptedLLM + make_llm_response — drive a deterministic LLM exchange.
2. FakeAuditSink — assert audit events were emitted.
3. FakePolicyEvaluator(default_allow=False) — assert deny path raises.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src.my_agent import answer_question

from ai_core.exceptions import PolicyDenialError
from ai_core.testing import make_llm_response

if TYPE_CHECKING:
    from collections.abc import Callable

    from ai_core.testing import (
        FakeAuditSink,
        FakePolicyEvaluator,
        ScriptedLLM,
    )


async def test_happy_path(
    scripted_llm_factory: Callable[..., ScriptedLLM],
    fake_audit_sink: FakeAuditSink,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """ScriptedLLM returns the canned response; agent records audit; returns content."""
    llm = scripted_llm_factory([make_llm_response("The answer is 42.")])
    audit = fake_audit_sink
    policy = fake_policy_evaluator_factory(default_allow=True)

    result = await answer_question(
        "What's the meaning of life?",
        llm=llm,
        audit=audit,
        policy=policy,
    )

    assert result == "The answer is 42."
    assert len(llm.calls) == 1
    assert llm.calls[0]["agent_id"] == "demo-agent"


async def test_audit_records_one_event(
    scripted_llm_factory: Callable[..., ScriptedLLM],
    fake_audit_sink: FakeAuditSink,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """FakeAuditSink captures one TOOL_INVOCATION_COMPLETED record."""
    llm = scripted_llm_factory([make_llm_response("ok")])
    audit = fake_audit_sink
    policy = fake_policy_evaluator_factory(default_allow=True)

    await answer_question("ping", llm=llm, audit=audit, policy=policy)

    assert len(audit.records) == 1
    record = audit.records[0]
    assert record.tool_name == "answer_question"
    assert record.payload["input"]["question"] == "ping"


async def test_deny_path_raises(
    scripted_llm_factory: Callable[..., ScriptedLLM],
    fake_audit_sink: FakeAuditSink,
    fake_policy_evaluator_factory: Callable[..., FakePolicyEvaluator],
) -> None:
    """FakePolicyEvaluator with default_allow=False causes a PolicyDenialError."""
    llm = scripted_llm_factory([make_llm_response("never reached")])
    audit = fake_audit_sink
    policy = fake_policy_evaluator_factory(default_allow=False, reason="demo denial")

    with pytest.raises(PolicyDenialError):
        await answer_question("forbidden", llm=llm, audit=audit, policy=policy)

    # LLM was not called because policy denied first.
    assert len(llm.calls) == 0
    # No audit event recorded on the deny path.
    assert len(audit.records) == 0
