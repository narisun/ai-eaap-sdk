"""Output-verification primitive.

:class:`VerifierAgent` wraps another :class:`BaseAgent`. After the
wrapped agent produces a final answer, the verifier issues a separate
LLM call against a host-supplied rubric and gets back a structured
:class:`Verdict`. On failure, the wrapped agent re-runs with the
verdict's feedback injected as a new user message; up to
``max_retries`` retries.

Why direct LLM call instead of synthetic tool
=============================================
Verification is a control-flow decision, not a tool dispatch. There is
no policy to enforce on "verifying" itself (the wrapped agent's tool
calls are already audited), no observability gap (the verifier emits
its own ``agent.verify`` span), and no validation gain from going
through :class:`ToolInvoker`. A direct ``runtime.llm.complete`` call
with a ``_record_verdict`` tool advertisement keeps the path narrow
and the verdict structure forced via the existing tool-call shape
that modern LLMs handle cleanly.

Why composition (wrap one child) instead of inheritance
=======================================================
A verifier wraps exactly **one** agent. Multi-agent verification is
the supervisor's job (with verifiers as children, or a verifier
wrapping a supervisor — both compose naturally). Keeping the wrapper
single-target keeps the retry loop and feedback injection
unambiguous.

Strictness
==========
``strict=True`` (default) raises :class:`AgentRuntimeError` when
verification fails after ``max_retries``. Production use cases want
the failure to surface explicitly rather than silently leak an
unverified answer. ``strict=False`` returns the last attempt with the
final verdict in ``state.metadata["last_verdict"]`` so callers may
inspect and decide.

Composition with other primitives
=================================
* a :class:`SupervisorAgent` can declare a :class:`VerifierAgent` as
  one of its children — the supervisor delegates a sub-task to the
  verifier, which wraps a downstream specialist
* a :class:`VerifierAgent` can wrap a :class:`SupervisorAgent` — the
  whole multi-agent flow gets a final verification pass before
  surfacing to the user
* :class:`VerifierAgent` can wrap a :class:`PlanningAgent` so the
  plan-and-execute final answer is gated on verification
* verifiers can stack (wrap a verifier in a verifier) for layered
  critique, though usually a single verifier with a comprehensive
  rubric is more effective than chains
"""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from injector import inject
from pydantic import BaseModel, Field, ValidationError

from ai_core.agents.base import BaseAgent
from ai_core.agents.runtime import AgentRuntime
from ai_core.agents.state import AgentState
from ai_core.exceptions import AgentRuntimeError, RegistryError
from ai_core.observability.logging import bind_context, get_logger, unbind_context

if TYPE_CHECKING:
    from ai_core.di.interfaces import LLMResponse
    from ai_core.tools.spec import Tool

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Verdict data model
# ---------------------------------------------------------------------------
class Verdict(BaseModel):
    """Structured verdict the verifier LLM emits via ``_record_verdict``.

    Attributes:
        passed: ``True`` when the candidate answer satisfies every
            rubric requirement; ``False`` otherwise.
        feedback: Human-readable summary suitable for injection back
            into the wrapped agent as retry feedback. On ``passed=False``
            this becomes the next ``user`` turn the wrapped agent sees.
        issues: Optional list of specific issues. Surface for
            dashboards / eval; not injected into retry messages by
            default.
        score: Optional 0.0-1.0 confidence — useful when the verifier
            LLM emits a graded judgement. Hosts that don't need
            scoring can ignore.
    """

    passed: bool
    feedback: str
    issues: list[str] = Field(default_factory=list)
    score: float | None = None


# ---------------------------------------------------------------------------
# Default verifier system prompt
# ---------------------------------------------------------------------------
_DEFAULT_VERIFIER_SYSTEM_PROMPT = """\
You are a strict answer-quality verifier.

You will receive a verification rubric and a candidate answer. Your
job is to evaluate the answer against the rubric and call the
`_record_verdict` tool with your judgement.

Be objective and specific:
- Set `passed=true` only if the candidate answer satisfies every
  rubric requirement.
- When `passed=false`, write `feedback` that the upstream agent can
  act on — concrete, actionable, and short. The feedback will be
  injected verbatim as the next user message the agent sees.
- Use `issues` for any structured list of problems (optional).
- `score` is optional — populate it only when the rubric explicitly
  asks for grading."""


# ---------------------------------------------------------------------------
# VerifierAgent
# ---------------------------------------------------------------------------
class VerifierAgent(BaseAgent):
    """Wraps a child agent; verifies its final answer; retries on failure.

    Subclass and provide:

    * :meth:`wrapped_agent` — the :class:`BaseAgent` subclass to wrap
    * :meth:`verification_prompt` — the rubric the verifier LLM uses

    The verifier inherits the SDK's cross-cutting concerns:

    * each retry attempt opens a fresh ``agent.ainvoke`` span on the
      wrapped agent (nested under the verifier's own
      ``agent.ainvoke`` span);
    * the verifier's verification call opens an ``agent.verify`` span;
    * each retry's LLM calls go through :class:`IBudgetService` so
      both the wrapped agent's budget and the verifier's own budget
      are enforced separately;
    * verdict history accumulates in
      ``state.scratchpad["verifications"]`` for replay / eval, and
      the final verdict is mirrored to ``state.metadata["last_verdict"]``
      for quick reads.

    Termination:

    * ``Verdict.passed=True`` → return the wrapped state immediately
    * after ``max_retries`` failed attempts:
        * ``strict=True`` (default) → raise
          :class:`AgentRuntimeError` with the final verdict in
          ``details``
        * ``strict=False`` → return the last attempt; caller inspects
          ``state.metadata["last_verdict"]``
    """

    #: Number of retries beyond the first attempt. Total wrapped-agent
    #: invocations = ``max_retries + 1``.
    max_retries: int = 2

    #: When ``True``, raise :class:`AgentRuntimeError` on final
    #: failure; when ``False``, return the last attempt with the final
    #: verdict in ``state.metadata``.
    strict: bool = True

    @inject
    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__(runtime)
        # Cache the wrapped instance so its compiled graph + per-instance
        # state survive across retries (avoids recompiling on every
        # attempt). One verifier instance ⇒ one wrapped instance.
        self._wrapped_instance: BaseAgent | None = None

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------
    @abstractmethod
    def wrapped_agent(self) -> type[BaseAgent]:
        """Return the :class:`BaseAgent` subclass this verifier wraps."""

    @abstractmethod
    def verification_prompt(self) -> str:
        """Return the rubric the verifier LLM evaluates against.

        The rubric is shown to the verifier LLM as part of a fixed
        scaffolding ("Rubric: <prompt>; Candidate answer: <text>;
        Call _record_verdict ..."). Hosts override
        :meth:`build_verification_messages` for full control of the
        prompt structure.
        """

    def system_prompt(self) -> str:
        """System prompt for the verifier's internal LLM call.

        Default reproduces the strict-judge persona and wires the
        ``_record_verdict`` tool. Override for custom verifier
        personas (e.g. localised prompts, lighter-touch judges).
        """
        return _DEFAULT_VERIFIER_SYSTEM_PROMPT

    def build_verification_messages(
        self, candidate_answer: str,
    ) -> list[Mapping[str, Any]]:
        """Build the message list passed to the verifier LLM.

        Hosts override for non-default prompt structures (multi-turn
        verification, in-context examples, custom rubric formats).
        """
        return [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": (
                    f"## Rubric\n{self.verification_prompt()}\n\n"
                    f"## Candidate answer\n{candidate_answer}\n\n"
                    "## Instructions\n"
                    "Evaluate the candidate answer against the rubric. "
                    "Call `_record_verdict` with your verdict. Set "
                    "`passed=true` only if every rubric requirement is "
                    "met. When `passed=false`, write specific actionable "
                    "feedback the upstream agent can use to revise."
                ),
            },
        ]

    def render_retry_feedback(self, verdict: Verdict) -> str:
        """Render the user message injected into the wrapped agent on retry.

        Default surfaces the verdict's feedback verbatim with a brief
        framing. Override to enrich the feedback (e.g. include the
        ``issues`` list or score).
        """
        return (
            "Your previous answer was rejected by the verifier.\n\n"
            f"Feedback: {verdict.feedback}\n\n"
            "Please revise."
        )

    # ------------------------------------------------------------------
    # tools() — verifier exposes no tools at the BaseAgent level
    # ------------------------------------------------------------------
    def tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Return ``()`` — VerifierAgent does not run a LangGraph.

        The verification LLM call is made imperatively in
        :meth:`ainvoke`, so the BaseAgent tool plumbing is unused
        here.
        """
        return ()

    # ------------------------------------------------------------------
    # ainvoke — verifier control loop
    # ------------------------------------------------------------------
    async def ainvoke(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        essential: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        """Invoke the wrapped agent, verify, and retry on failure.

        Overrides :meth:`BaseAgent.ainvoke` because the verifier's
        control flow is not a LangGraph — it's a deterministic loop
        around the wrapped agent's own ``ainvoke``.
        """
        log_token = bind_context(
            agent_id=self.agent_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
        )
        try:
            wrapped = self._resolve_wrapped()
            wrapped_cls_name = type(wrapped).__name__

            attributes = {
                "agent.id": self.agent_id,
                "agent.tenant_id": tenant_id or "",
                "verify.target": wrapped_cls_name,
                "verify.max_retries": self.max_retries,
            }

            verdict_history: list[dict[str, Any]] = []
            last_state: AgentState | None = None
            last_verdict: Verdict | None = None
            original_messages = list(messages)
            attempt = 0

            async with self._runtime.observability.start_span(
                "agent.ainvoke", attributes=attributes,
            ):
                while attempt <= self.max_retries:
                    # Build messages for this attempt. On retries, append
                    # the previous final answer + verifier feedback so the
                    # wrapped agent sees the full chain of attempts.
                    messages_for_child = list(original_messages)
                    if attempt > 0 and last_state is not None and last_verdict is not None:
                        prev_answer = self._extract_last_assistant_text(last_state)
                        if prev_answer:
                            messages_for_child.append({
                                "role": "assistant", "content": prev_answer,
                            })
                        messages_for_child.append({
                            "role": "user",
                            "content": self.render_retry_feedback(last_verdict),
                        })

                    last_state = await wrapped.ainvoke(
                        messages=messages_for_child,
                        essential={
                            **(essential or {}),
                            "verified_by": self.agent_id,
                            "verify_attempt": attempt,
                        },
                        tenant_id=tenant_id,
                        thread_id=thread_id,
                    )

                    last_verdict = await self._verify(last_state, tenant_id=tenant_id)
                    verdict_history.append({
                        "attempt": attempt,
                        "verdict": last_verdict.model_dump(),
                    })

                    if last_verdict.passed:
                        break
                    attempt += 1

            assert last_state is not None and last_verdict is not None

            # Stash verdict history + final verdict into state. Existing
            # scratchpad / metadata are preserved.
            scratchpad = dict(last_state.get("scratchpad") or {})
            existing = list(scratchpad.get("verifications") or [])
            existing.extend(verdict_history)
            scratchpad["verifications"] = existing
            last_state["scratchpad"] = scratchpad

            metadata = dict(last_state.get("metadata") or {})
            metadata["last_verdict"] = last_verdict.model_dump()
            last_state["metadata"] = metadata

            if not last_verdict.passed and self.strict:
                raise AgentRuntimeError(
                    "Verification failed after max_retries",
                    details={
                        "agent_id": self.agent_id,
                        "wrapped_agent": wrapped_cls_name,
                        "max_retries": self.max_retries,
                        "last_verdict": last_verdict.model_dump(),
                    },
                )
            return last_state
        finally:
            unbind_context(log_token)

    # ------------------------------------------------------------------
    # Internal — verification call
    # ------------------------------------------------------------------
    async def _verify(
        self,
        child_state: AgentState,
        *,
        tenant_id: str | None,
    ) -> Verdict:
        """Issue the verification LLM call and parse the verdict.

        Wrapped in its own ``agent.verify`` span so dashboards can
        attribute verifier latency / cost separately from the wrapped
        agent's own LLM calls. Also runs through the verifier's
        ``IBudgetService`` binding (same tenant, separate agent id).
        """
        candidate = self._extract_last_assistant_text(child_state)
        verdict_tool = {
            "type": "function",
            "function": {
                "name": "_record_verdict",
                "description": (
                    "Record your structured verdict on the candidate "
                    "answer. Call this exactly once."
                ),
                "parameters": Verdict.model_json_schema(),
            },
        }
        attrs = {
            "agent.id": self.agent_id,
            "verify.candidate_chars": len(candidate),
        }
        async with self._runtime.observability.start_span(
            "agent.verify", attributes=attrs,
        ):
            response = await self._runtime.llm.complete(
                model=None,
                messages=self.build_verification_messages(candidate),
                tools=[verdict_tool],
                tenant_id=tenant_id,
                agent_id=self.agent_id,
            )
        return self._parse_verdict_or_default(response)

    @staticmethod
    def _extract_last_assistant_text(state: AgentState) -> str:
        """Pull the last assistant message's text out of a state.

        Handles both LangChain ``AIMessage`` and plain dict shapes that
        may appear in the messages list. ``AgentState`` types
        ``messages`` as ``list[dict[str, Any]]``, but LangGraph's
        ``add_messages`` reducer promotes dicts to LangChain
        ``BaseMessage`` subclasses at runtime — the loop is typed as
        ``Any`` so mypy doesn't rule out the AIMessage branch as
        unreachable.
        """
        messages: list[Any] = list(state.get("messages") or [])
        for msg in reversed(messages):
            if isinstance(msg, Mapping):
                if msg.get("role") == "assistant":
                    return str(msg.get("content") or "")
                continue
            # LangChain BaseMessage subclasses (AIMessage, etc.) carry
            # ``type`` rather than ``role``. The agent node always
            # appends an AIMessage, so ``type == "ai"`` is the marker.
            if getattr(msg, "type", None) == "ai":
                return str(getattr(msg, "content", "") or "")
        return ""

    @staticmethod
    def _parse_verdict_or_default(response: LLMResponse) -> Verdict:
        """Extract ``Verdict`` from the verifier LLM response.

        Defensive: if the verifier LLM produced no ``_record_verdict``
        tool_call (or emitted unparseable JSON), fall back to a
        ``passed=False`` verdict so the retry loop kicks in. This
        protects against verifier-side glitches without silently
        accepting bad answers.
        """
        if not response.tool_calls:
            return Verdict(
                passed=False,
                feedback=(
                    "Verifier produced no structured verdict; treating as "
                    "fail. (LLM did not invoke `_record_verdict`.)"
                ),
            )
        # ``tool_calls`` is typed as ``Sequence[Mapping[str, Any]]`` so we
        # can read fields directly. JSONDecodeError / ValidationError /
        # AttributeError on an unexpected runtime shape all funnel into
        # the defensive except branch below.
        tc = response.tool_calls[0]
        args_raw = tc.get("function", {}).get("arguments", "")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            return Verdict.model_validate(args)
        except (json.JSONDecodeError, ValidationError, TypeError, AttributeError) as exc:
            _logger.warning(
                "verifier.parse_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return Verdict(
                passed=False,
                feedback=(
                    f"Verifier produced unparseable verdict "
                    f"({type(exc).__name__}); treating as fail."
                ),
            )

    # ------------------------------------------------------------------
    # Internal — wrapped-agent resolution + caching
    # ------------------------------------------------------------------
    def _resolve_wrapped(self) -> BaseAgent:
        if self._wrapped_instance is not None:
            return self._wrapped_instance
        wrapped_cls = self.wrapped_agent()
        instance = self._runtime.agent_resolver.resolve(wrapped_cls)
        if not isinstance(instance, BaseAgent):
            raise RegistryError(
                f"Wrapped agent {wrapped_cls.__name__!r} resolved to "
                f"{type(instance).__name__}; expected a BaseAgent subclass.",
                details={
                    "wrapped_agent": wrapped_cls.__name__,
                    "resolved_type": type(instance).__name__,
                },
            )
        self._wrapped_instance = instance
        return instance


__all__ = ["Verdict", "VerifierAgent"]
