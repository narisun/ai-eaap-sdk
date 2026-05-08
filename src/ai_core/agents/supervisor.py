"""Multi-agent supervisor primitive.

:class:`SupervisorAgent` coordinates a set of child :class:`BaseAgent`
instances. Children are exposed to the supervisor's LLM as tools; the
LLM emits ``tool_calls`` to delegate work, the supervisor's existing
:class:`ToolInvoker` dispatches each call, and the child agent's
:meth:`BaseAgent.ainvoke` runs to produce a result that the LLM sees as
the tool's return value on its next turn.

Why children-as-tools
=====================
This design reuses 100 % of the v1 tool-dispatch plumbing:

* OPA policy: each child invocation can be gated by ``opa_path``;
* observability: each delegation opens a ``tool.invoke`` span nested
  under the supervisor's ``agent.ainvoke`` span, and the child's own
  ``agent.ainvoke`` span nests under that;
* audit: ``IAuditSink`` records each delegation;
* error handling: a child that fails surfaces as a ``ToolMessage`` via
  :class:`IToolErrorRenderer`, so the supervisor LLM can re-plan;
* budgeting: the child's LLM calls flow through its own
  ``IBudgetService.check`` independently — same tenant pool, separate
  agent budget;
* validation: each child's input is validated via Pydantic before
  dispatch.

Modern LLMs are excellent at tool-selection, which makes routing
quality high without a custom protocol.

Default contract
================
Each child gets a default tool schema::

    class TaskInput(BaseModel):
        task: str
        context: str | None = None

    class TaskOutput(BaseModel):
        result: str

The supervisor renders the validated ``TaskInput`` into a single user
message for the child, runs ``child.ainvoke``, and returns
``TaskOutput(result=<last assistant message text>)``. Hosts override
:meth:`SupervisorAgent.child_input_schema` /
:meth:`SupervisorAgent.child_output_schema` /
:meth:`SupervisorAgent.render_child_input` /
:meth:`SupervisorAgent.render_child_output` to give individual children
typed contracts (e.g. ``customer_id: str, topic: str``).

Inheritance
===========
Subclasses must implement :meth:`children` (returning the name → class
map). Optional overrides:

* :meth:`tools` — extend with non-child tools. Default returns just
  the auto-generated child tools; ``super().tools()`` exposes them.
* :meth:`child_input_schema` / :meth:`child_output_schema` — typed
  contracts per child.
* :meth:`render_child_input` / :meth:`render_child_output` — custom
  marshalling.

Recursion safety
================
Each agent (the supervisor and each child) has its own
``max_recursion_depth`` from :class:`AgentSettings`, so a confused
supervisor that loops 25 times — each delegating to a child that loops
25 times — is bounded at ``25 * 25`` LLM calls in the worst case.
Nested supervisors compose the same way.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from injector import inject
from pydantic import BaseModel

from ai_core.agents.base import BaseAgent
from ai_core.agents.runtime import AgentRuntime
from ai_core.agents.state import AgentState
from ai_core.exceptions import RegistryError
from ai_core.tools.spec import ToolHandler, ToolSpec

if TYPE_CHECKING:
    from ai_core.tools.spec import Tool


# ---------------------------------------------------------------------------
# Default child contract
# ---------------------------------------------------------------------------
class TaskInput(BaseModel):
    """Default input schema for child delegation.

    Captures the LLM's free-text intent. Hosts override
    :meth:`SupervisorAgent.child_input_schema` to give a child a typed
    contract instead.
    """

    task: str
    context: str | None = None


class TaskOutput(BaseModel):
    """Default output schema for child delegation.

    The supervisor renders the child's final assistant message into
    ``result``. Hosts override :meth:`SupervisorAgent.child_output_schema`
    to surface structured fields the supervisor LLM should reason over.
    """

    result: str


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------
class SupervisorAgent(BaseAgent):
    """Orchestrates child :class:`BaseAgent` instances via LLM tool-calls.

    Subclasses declare children::

        class SupportSupervisor(SupervisorAgent):
            agent_id = "support-supervisor"

            def children(self) -> Mapping[str, type[BaseAgent]]:
                return {
                    "triage": TriageAgent,
                    "research": ResearchAgent,
                }

            def system_prompt(self) -> str:
                return (
                    "You coordinate a support team. Use `triage` first "
                    "to classify, then `research`, then respond."
                )

    Children are resolved through ``runtime.agent_resolver`` on first
    invocation and cached on the supervisor instance for the lifetime
    of the run, so each child's graph compiles at most once.
    """

    #: When ``None`` (default), child invocations skip OPA policy
    #: enforcement — the children themselves enforce policy on their own
    #: tool calls. Subclasses that want supervisor-level delegation
    #: policies set this to a decision path like ``"eaap/agent/delegate/allow"``
    #: and OPA receives ``{supervisor: ..., child: ..., payload: ...}``
    #: as the input document.
    delegation_opa_path: str | None = None

    @inject
    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__(runtime)
        self._child_instances: dict[str, BaseAgent] = {}
        # Stash for the current state during _tool_node so the synthetic
        # child-tool handlers can read essential_entities + tenant_id from
        # the supervisor's state without threading it through ToolInvoker.
        self._current_state: AgentState | None = None

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------
    @abstractmethod
    def children(self) -> Mapping[str, type[BaseAgent]]:
        """Return the map of tool name → child agent class.

        The tool name is what the supervisor's LLM emits in its
        ``tool_calls``; the class is resolved via the DI container so
        each child gets its own :class:`AgentRuntime`, observability
        span, budget binding, and policy evaluator.
        """

    def child_input_schema(self, name: str) -> type[BaseModel]:
        """Pydantic input schema the supervisor's LLM uses for child ``name``.

        Default: :class:`TaskInput` (free-text task + optional context).
        Override per-child for typed contracts.
        """
        del name  # unused in default impl; subclasses may switch on name
        return TaskInput

    def child_output_schema(self, name: str) -> type[BaseModel]:
        """Pydantic output schema for child ``name``.

        The supervisor's :meth:`render_child_output` produces an instance
        of this class; the result is JSON-serialised and fed back to the
        supervisor LLM as the tool's return value.

        Default: :class:`TaskOutput` (single ``result`` field).
        """
        del name  # unused in default impl
        return TaskOutput

    def render_child_input(
        self, name: str, payload: BaseModel,
    ) -> str:
        """Convert a validated input payload into a user message for the child.

        The default handles :class:`TaskInput` specially (renders
        ``task`` + optional ``context``); for any other model it
        falls back to a JSON dump. Override to format custom input
        models more naturally.
        """
        del name
        if isinstance(payload, TaskInput):
            if payload.context:
                return f"{payload.task}\n\nContext: {payload.context}"
            return payload.task
        return payload.model_dump_json()

    def render_child_output(
        self, name: str, child_state: AgentState,
    ) -> BaseModel:
        """Convert the child's final state into the output payload.

        Default: extracts the last assistant message content from
        ``child_state["messages"]`` and wraps it in :class:`TaskOutput`.
        Override when the child's final state carries structured fields
        the supervisor LLM should see directly.
        """
        del name
        messages = list(child_state.get("messages") or [])
        last = messages[-1] if messages else None
        text = ""
        if last is not None:
            content = getattr(last, "content", None)
            if content is None and isinstance(last, Mapping):
                content = last.get("content")
            text = str(content or "")
        return TaskOutput(result=text)

    # ------------------------------------------------------------------
    # Auto-generated tools surface
    # ------------------------------------------------------------------
    def tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Return one :class:`ToolSpec` per declared child.

        Subclasses that need to expose additional non-child tools
        override this method and concatenate ``super().tools()`` with
        their own list.
        """
        # ToolSpec is a frozen dataclass and structurally satisfies the
        # Tool Protocol, but mypy's nominal-typing check sees frozen
        # ``name``/``version`` as read-only and the Protocol's settable
        # variables as a contradiction. Cast at the boundary; runtime
        # behaviour is unchanged.
        from typing import cast
        specs = [
            self._build_child_tool(name, child_cls)
            for name, child_cls in self.children().items()
        ]
        return cast("list[Tool | Mapping[str, Any]]", specs)

    def _build_child_tool(
        self, name: str, child_cls: type[BaseAgent],
    ) -> ToolSpec:
        """Wrap a child agent as a :class:`ToolSpec`.

        Built directly (not via :func:`ai_core.tools.make_tool`) because
        the host-supplied input/output schemas come from
        :meth:`child_input_schema` / :meth:`child_output_schema` rather
        than from handler type hints — :class:`make_tool` infers schemas
        from annotations and would force the closure to declare them
        statically, which is the wrong abstraction for per-supervisor
        runtime overrides.
        """
        input_model = self.child_input_schema(name)
        output_model = self.child_output_schema(name)

        async def _handler(payload: BaseModel) -> BaseModel:
            return await self._dispatch_to_child(name, child_cls, payload)

        description = (
            child_cls.__doc__.strip().splitlines()[0]
            if child_cls.__doc__
            else f"Delegate work to the {name!r} sub-agent."
        )
        return ToolSpec(
            name=name,
            version=1,
            description=description,
            input_model=input_model,
            output_model=output_model,
            handler=_cast_handler(_handler),
            opa_path=self.delegation_opa_path,
        )

    # ------------------------------------------------------------------
    # Tool dispatch — override BaseAgent._tool_node to stash state
    # ------------------------------------------------------------------
    async def _tool_node(self, state: AgentState) -> AgentState:
        """Stash the current state so child handlers can read it, then delegate.

        The handlers built by :meth:`_build_child_tool` are pure
        ``(payload) -> result`` callables — they can't see graph state
        directly. Stashing on ``self._current_state`` is the simplest
        thread-safe path because each agent invocation owns one
        ``_tool_node`` call at a time (LangGraph runs nodes serially
        inside a single graph execution).
        """
        self._current_state = state
        try:
            return await super()._tool_node(state)
        finally:
            self._current_state = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _dispatch_to_child(
        self,
        name: str,
        child_cls: type[BaseAgent],
        payload: BaseModel,
    ) -> BaseModel:
        """Resolve, invoke, and render the result of a single child call."""
        child = self._resolve_child(name, child_cls)
        user_message = self.render_child_input(name, payload)

        state = self._current_state or {}
        essentials = dict(state.get("essential_entities") or {})
        tenant_id = str(essentials.get("tenant_id") or "") or None
        thread_id = essentials.get("thread_id") or None

        # Tag the child's essentials with delegation metadata so the
        # observability + audit pipeline can associate child spans /
        # records with the supervising agent.
        child_essentials = {
            **essentials,
            "delegated_by": self.agent_id,
            "delegation_target": name,
        }

        child_state = await child.ainvoke(
            messages=[{"role": "user", "content": user_message}],
            essential=child_essentials,
            tenant_id=tenant_id,
            thread_id=str(thread_id) if thread_id is not None else None,
        )
        return self.render_child_output(name, child_state)

    def _resolve_child(
        self, name: str, child_cls: type[BaseAgent],
    ) -> BaseAgent:
        """Resolve and cache a child instance.

        First-call: container resolution; cache the instance for the
        lifetime of this supervisor instance so subsequent invocations
        reuse the compiled graph and any per-instance state.
        """
        existing = self._child_instances.get(name)
        if existing is not None:
            return existing
        instance = self._runtime.agent_resolver.resolve(child_cls)
        if not isinstance(instance, BaseAgent):
            raise RegistryError(
                f"Child agent {name!r} resolved to {type(instance).__name__}; "
                f"expected a BaseAgent subclass.",
                details={"child": name, "resolved_type": type(instance).__name__},
            )
        self._child_instances[name] = instance
        return instance


# ---------------------------------------------------------------------------
# Type-erased handler cast.
# ---------------------------------------------------------------------------
def _cast_handler(
    fn: Callable[[BaseModel], Awaitable[BaseModel]],
) -> ToolHandler:
    """Cast the closure to the :data:`ToolHandler` alias for ToolSpec.

    Both types are ``Callable[[BaseModel], Awaitable[BaseModel]]`` so this
    is a no-op at runtime; the helper exists to keep the cast localized
    and documented.
    """
    return fn


__all__ = ["SupervisorAgent", "TaskInput", "TaskOutput"]
