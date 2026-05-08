"""LangGraph guardrail node — intercepts tool calls, enforces OPA policy.

The :class:`GuardrailNode` is invoked by the LangGraph runtime after
the agent emits an assistant message containing one or more
``tool_calls``. It evaluates each call against the OPA policy and:

* allows the message through unchanged when **every** call is permitted;
* replaces the assistant message with a denial-explanation when one or
  more calls are denied (the agent then re-plans on the next turn);
* blocks all calls (fail-closed) on policy-evaluator failure.

Wiring (from a :class:`BaseAgent` subclass)::

    from ai_core.security import GuardrailNode

    class MyAgent(BaseAgent):
        def system_prompt(self) -> str: ...

        def extend_graph(self, graph) -> None:
            guard = GuardrailNode(self._policy, agent_id=self.agent_id)
            graph.add_node("guardrail", guard.run)
            # plus host-defined edges for the tool execution node.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_core.agents.state import AgentState
from ai_core.di.interfaces import IPolicyEvaluator, PolicyDecision
from ai_core.exceptions import PolicyDenialError


@dataclass(slots=True, frozen=True)
class GuardrailDecision:
    """Outcome of a guardrail evaluation for a single tool call.

    Attributes:
        tool_name: Name of the called tool.
        call_id: Provider-supplied id of the tool call.
        allowed: Whether OPA allowed the call.
        reason: Optional human-readable reason on denial.
    """

    tool_name: str
    call_id: str
    allowed: bool
    reason: str | None = None


class GuardrailNode:
    """LangGraph node that authorises every tool call before execution.

    Args:
        policy: Policy evaluator (typically :class:`OPAPolicyEvaluator`).
        agent_id: Logical agent identifier — passed to OPA for context.
        decision_path: Optional override for the OPA decision path.
    """

    DEFAULT_DECISION_PATH = "eaap/agent/tool_call/allow"

    def __init__(
        self,
        policy: IPolicyEvaluator,
        *,
        agent_id: str,
        decision_path: str | None = None,
    ) -> None:
        self._policy = policy
        self._agent_id = agent_id
        self._decision_path = decision_path or self.DEFAULT_DECISION_PATH

    async def run(self, state: AgentState) -> AgentState:
        """Evaluate every tool call in the latest assistant message.

        Args:
            state: Current agent state.

        Returns:
            * The original state (unchanged) if every call is allowed.
            * A state mutation with a system-level denial message if at
              least one call is denied — the agent's next turn will see
              the explanation and re-plan.
        """
        messages = list(state.get("messages") or [])
        last = _last_assistant_message(messages)
        if last is None:
            return state
        tool_calls: Sequence[Mapping[str, Any]] = last.get("tool_calls") or ()
        if not tool_calls:
            return state

        essentials = state.get("essential_entities") or {}
        decisions = [
            await self._evaluate_one(call, essentials=essentials)
            for call in tool_calls
        ]

        denied = [d for d in decisions if not d.allowed]
        if not denied:
            return state

        denial_text = _format_denial_message(denied)
        return AgentState(
            messages=[
                {
                    "role": "system",
                    "content": denial_text,
                    "name": "guardrail",
                }
            ],
        )

    # ------------------------------------------------------------------
    # Per-call evaluation
    # ------------------------------------------------------------------
    async def _evaluate_one(
        self,
        call: Mapping[str, Any],
        *,
        essentials: Mapping[str, Any],
    ) -> GuardrailDecision:
        tool_name, call_id, arguments = _extract_call_metadata(call)

        opa_input: dict[str, Any] = {
            "agent_id": self._agent_id,
            "user": str(essentials.get("user_id") or ""),
            "tenant_id": str(essentials.get("tenant_id") or ""),
            "session_id": str(essentials.get("session_id") or ""),
            "action": "tool.invoke",
            "tool": {"name": tool_name, "arguments": arguments},
            "call_id": call_id,
        }

        try:
            decision: PolicyDecision = await self._policy.evaluate(
                decision_path=self._decision_path,
                input=opa_input,
            )
        except PolicyDenialError as exc:
            return GuardrailDecision(
                tool_name=tool_name,
                call_id=call_id,
                allowed=False,
                reason=f"policy-evaluator-error: {exc.message}",
            )

        return GuardrailDecision(
            tool_name=tool_name,
            call_id=call_id,
            allowed=decision.allowed,
            reason=decision.reason if not decision.allowed else None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _last_assistant_message(messages: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg
    return None


def _extract_call_metadata(call: Mapping[str, Any]) -> tuple[str, str, Any]:
    """Pull ``(name, call_id, arguments)`` out of an OpenAI-style tool call."""
    call_id = str(call.get("id") or "")
    function = call.get("function") or {}
    if isinstance(function, Mapping):
        name = str(function.get("name") or call.get("name") or "")
        arguments = function.get("arguments")
    else:
        name = str(call.get("name") or "")
        arguments = call.get("arguments")
    return name, call_id, arguments


def _format_denial_message(decisions: Sequence[GuardrailDecision]) -> str:
    bullets = "\n".join(
        f"- {d.tool_name} (id={d.call_id}): {d.reason or 'denied by policy'}"
        for d in decisions
    )
    return (
        "[guardrail] The following tool call(s) were denied by policy and were "
        "NOT executed. Reconsider your plan:\n" + bullets
    )


__all__ = ["GuardrailDecision", "GuardrailNode"]
