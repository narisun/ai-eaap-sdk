"""Toy agent under test in this demo.

This isn't a real BaseAgent — it's the smallest possible function that
exercises the SDK's testing surface end-to-end:

- It calls an `ILLMClient.complete()`.
- It records an audit event.
- It checks an `IPolicyEvaluator.evaluate()` decision.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ai_core.audit import AuditEvent, AuditRecord
from ai_core.exceptions import PolicyDenialError

if TYPE_CHECKING:
    from ai_core.audit import IAuditSink
    from ai_core.di.interfaces import ILLMClient, IPolicyEvaluator


async def answer_question(
    question: str,
    *,
    llm: ILLMClient,
    audit: IAuditSink,
    policy: IPolicyEvaluator,
    tenant_id: str = "demo-tenant",
) -> str:
    decision = await policy.evaluate(
        decision_path="demo.allow",
        input={"action": "answer", "question": question},
    )
    if not decision.allowed:
        raise PolicyDenialError(
            "policy denied", details={"reason": decision.reason or "no reason"}
        )

    response = await llm.complete(
        model=None,
        messages=[{"role": "user", "content": question}],
        tenant_id=tenant_id,
        agent_id="demo-agent",
    )

    await audit.record(
        AuditRecord.now(
            AuditEvent.TOOL_INVOCATION_COMPLETED,
            tool_name="answer_question",
            tool_version=1,
            agent_id="demo-agent",
            tenant_id=tenant_id,
            payload={"input": {"question": question}, "output": response.content},
        )
    )

    return response.content
