"""Replay-grade tracing primitive (capture-only).

:class:`HarnessAgent` wraps another :class:`BaseAgent` and records every
LLM call and tool dispatch the wrapped agent makes into a structured
:class:`Trace`. Hosts inspect :attr:`HarnessAgent.last_trace` after a
run, persist via ``trace.model_dump_json()``, and feed the result to
dashboards, evals, or replay machinery.

Why composition + structural wrapping
=====================================
The harness ships zero new policy. It captures by replacing two fields
of the wrapped agent's :class:`AgentRuntime`:

* ``runtime.llm`` is swapped for :class:`_CapturingLLMClient`, which
  satisfies :class:`ILLMClient` structurally and records each
  ``complete()`` call.
* ``runtime.tool_invoker`` is swapped for :class:`_CapturingToolInvoker`,
  which satisfies :class:`IToolInvoker` and records each ``invoke()``
  call (including failures, with the exception type and message).

The wrapped agent is constructed with this customised runtime, runs its
normal LangGraph, and never knows it's being observed. Every other
collaborator (memory, observability, audit, MCP factory, agent resolver)
is the original singleton from the harness's own runtime — there is no
shadow container, no parallel DI graph.

Scope
=====
**Capture only.** Replay (deterministic re-execution of a recorded
:class:`Trace`) is deferred to a future slice. The :class:`Trace` data
model is the public surface that future replay logic builds on; it is
intentionally JSON-serialisable so traces can be persisted, indexed,
and replayed across SDK versions without coupling to in-memory types.

Caveats
=======
* :class:`MemoryManager` compaction uses the separate
  :class:`ICompactionLLM` DI binding (typically aliased to the same
  :class:`ILLMClient` instance, but the binding is distinct). The
  harness wraps :attr:`AgentRuntime.llm` — the request-path LLM — so
  compaction-LLM calls are **not** captured in v1. Hosts that need
  compaction-LLM capture override the :class:`ICompactionLLM` binding
  in their test container.
* The wrapped agent is constructed with ``wrapped_cls(custom_runtime)``
  by default. Subclasses with extra ``@inject`` constructor parameters
  override :meth:`HarnessAgent._construct_wrapped` to supply them.
* :meth:`ILLMClient.astream` is delegated transparently but **not**
  captured — the SDK's default agent loop uses ``complete()``.
"""

from __future__ import annotations

import dataclasses
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from injector import inject
from pydantic import BaseModel, Field

from ai_core.agents.base import BaseAgent
from ai_core.agents.runtime import AgentRuntime
from ai_core.agents.state import AgentState
from ai_core.di.interfaces import ILLMClient, LLMResponse, LLMStreamChunk
from ai_core.observability.logging import bind_context, get_logger, unbind_context

if TYPE_CHECKING:
    from collections.abc import Callable

    from ai_core.tools.invoker import IToolInvoker
    from ai_core.tools.spec import Tool, ToolSpec

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Trace data model
# ---------------------------------------------------------------------------
class LLMCallRecord(BaseModel):
    """One captured ``ILLMClient.complete`` call.

    Captures both request and response in JSON-friendly shape so the
    record round-trips through ``model_dump_json`` cleanly.

    Attributes:
        model: Model string the agent requested (``None`` means default).
        messages: Request messages, copied as a list of plain dicts.
        tools: Request tool definitions (or ``None`` when no tools were
            advertised).
        tenant_id: Tenant id propagated to the LLM client.
        agent_id: Agent id propagated to the LLM client.
        response_model: Resolved model id from the response.
        response_content: Response assistant content.
        response_tool_calls: Response tool-call payloads, copied to plain
            dicts.
        finish_reason: Response finish reason, if reported.
        prompt_tokens / completion_tokens / total_tokens: Usage counters.
        cost_usd: Cost reported by the LLM client, if any.
        timestamp_ms: Milliseconds since the harness's capture started.
    """

    model: str | None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    tenant_id: str | None
    agent_id: str | None
    response_model: str
    response_content: str
    response_tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    timestamp_ms: float


class ToolDispatchRecord(BaseModel):
    """One captured ``IToolInvoker.invoke`` call.

    Records both successful dispatches (with the validated result) and
    failed dispatches (with the exception type and message). Failures
    propagate to the caller after recording so the wrapped agent's
    error-recovery paths fire normally.

    Attributes:
        tool: Tool name.
        tool_version: Tool version.
        raw_args: Args passed to the invoker, copied to a plain dict.
        agent_id / tenant_id: Identity context propagated to the invoker.
        outcome: ``"ok"`` for a successful dispatch, ``"error"`` for a
            raised exception.
        result: Validated output dict (only on ``"ok"``).
        error_type: Exception class name (only on ``"error"``).
        error_message: Stringified exception (only on ``"error"``).
        timestamp_ms: Milliseconds since capture started, at dispatch entry.
        latency_ms: Wall-clock duration of the dispatch.
    """

    tool: str
    tool_version: int
    raw_args: dict[str, Any]
    agent_id: str | None
    tenant_id: str | None
    outcome: Literal["ok", "error"]
    result: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    timestamp_ms: float
    latency_ms: float


class TraceEvent(BaseModel):
    """One ordered entry in the captured trace.

    Discriminated by :attr:`kind` — exactly one of :attr:`llm` or
    :attr:`tool` is populated per event. The flat shape (rather than a
    Pydantic discriminated union) keeps the JSON output stable for
    consumers that scan events without library support.
    """

    kind: Literal["llm", "tool"]
    llm: LLMCallRecord | None = None
    tool: ToolDispatchRecord | None = None


class Trace(BaseModel):
    """A single :class:`HarnessAgent.ainvoke` capture.

    Attributes:
        agent_id: Identifier of the harness that produced the trace.
        wrapped_agent: Class name of the wrapped agent.
        started_at: UTC timestamp when capture began.
        finished_at: UTC timestamp when capture ended; ``None`` if the
            wrapped run is still in flight (the harness sets this in a
            ``finally`` block, so traces always have it after
            :meth:`HarnessAgent.ainvoke` returns or raises).
        events: Chronologically ordered list of captured events.
    """

    agent_id: str
    wrapped_agent: str
    started_at: datetime
    finished_at: datetime | None = None
    events: list[TraceEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Capturing wrappers
# ---------------------------------------------------------------------------
class _CapturingLLMClient(ILLMClient):
    """Wraps an :class:`ILLMClient` to record every ``complete()`` call.

    ``astream()`` is delegated transparently without capture — the v1
    agent loop only uses ``complete()``. Recording streams cleanly is
    deferred to the same future slice that ships replay.
    """

    def __init__(
        self,
        real: ILLMClient,
        append: Callable[[TraceEvent], None],
        t0_perf: float,
    ) -> None:
        self._real = real
        self._append = append
        self._t0 = t0_perf

    async def complete(
        self,
        *,
        model: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        timestamp_ms = (time.perf_counter() - self._t0) * 1000.0
        response = await self._real.complete(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            tenant_id=tenant_id,
            agent_id=agent_id,
            extra=extra,
        )
        record = LLMCallRecord(
            model=model,
            messages=[dict(m) for m in messages],
            tools=[dict(t) for t in tools] if tools is not None else None,
            tenant_id=tenant_id,
            agent_id=agent_id,
            response_model=response.model,
            response_content=response.content,
            response_tool_calls=[dict(tc) for tc in response.tool_calls],
            finish_reason=response.finish_reason,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost_usd=response.usage.cost_usd,
            timestamp_ms=timestamp_ms,
        )
        self._append(TraceEvent(kind="llm", llm=record))
        return response

    async def astream(
        self,
        *,
        model: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        return await self._real.astream(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            tenant_id=tenant_id,
            agent_id=agent_id,
            extra=extra,
        )


class _CapturingToolInvoker:
    """Wraps an :class:`IToolInvoker` to record every ``invoke()`` call.

    ``register()`` is delegated without capture — registration is a
    schema-registry side effect, not a runtime event. Failures during
    ``invoke`` are recorded with their exception type and re-raised so
    the wrapped agent's error-recovery paths (e.g. the renderer that
    turns ``ToolValidationError`` into a ``ToolMessage``) fire normally.
    """

    def __init__(
        self,
        real: IToolInvoker,
        append: Callable[[TraceEvent], None],
        t0_perf: float,
    ) -> None:
        self._real = real
        self._append = append
        self._t0 = t0_perf

    async def invoke(
        self,
        spec: ToolSpec,
        raw_args: Mapping[str, Any],
        *,
        principal: Mapping[str, Any] | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Mapping[str, Any]:
        timestamp_ms = (time.perf_counter() - self._t0) * 1000.0
        started = time.perf_counter()
        try:
            result = await self._real.invoke(
                spec,
                raw_args,
                principal=principal,
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._append(TraceEvent(
                kind="tool",
                tool=ToolDispatchRecord(
                    tool=spec.name,
                    tool_version=spec.version,
                    raw_args=dict(raw_args),
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    outcome="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    timestamp_ms=timestamp_ms,
                    latency_ms=latency_ms,
                ),
            ))
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._append(TraceEvent(
            kind="tool",
            tool=ToolDispatchRecord(
                tool=spec.name,
                tool_version=spec.version,
                raw_args=dict(raw_args),
                agent_id=agent_id,
                tenant_id=tenant_id,
                outcome="ok",
                result=dict(result),
                timestamp_ms=timestamp_ms,
                latency_ms=latency_ms,
            ),
        ))
        return result

    def register(self, spec: ToolSpec) -> None:
        self._real.register(spec)


# ---------------------------------------------------------------------------
# HarnessAgent
# ---------------------------------------------------------------------------
class HarnessAgent(BaseAgent):
    """Wraps a child agent; records every LLM + tool-dispatch event.

    Subclass and provide:

    * :meth:`wrapped_agent` — the :class:`BaseAgent` subclass to wrap.

    Override hooks:

    * :meth:`_construct_wrapped` — when the wrapped class's
      ``__init__`` takes more than just ``runtime``.

    Usage::

        class MyHarness(HarnessAgent):
            agent_id = "my-harness"
            def wrapped_agent(self) -> type[BaseAgent]:
                return MyAgent

        async with AICoreApp() as app:
            harness = app.agent(MyHarness)
            final_state = await harness.ainvoke(
                messages=[{"role": "user", "content": "..."}],
                tenant_id="acme",
            )
            trace = harness.last_trace
            print(trace.model_dump_json(indent=2))
    """

    @inject
    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__(runtime)
        #: The most recent capture. ``None`` until the first
        #: :meth:`ainvoke` completes (or raises). Always populated in a
        #: ``finally`` block, so callers can inspect partial traces on
        #: failure.
        self.last_trace: Trace | None = None

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------
    @abstractmethod
    def wrapped_agent(self) -> type[BaseAgent]:
        """Return the :class:`BaseAgent` subclass this harness wraps."""

    def system_prompt(self) -> str:
        """Unused: the harness bypasses the LangGraph in :meth:`ainvoke`.

        Required because :class:`BaseAgent` declares ``system_prompt`` as
        abstract; the harness overrides :meth:`ainvoke` so this value
        never reaches an LLM.
        """
        return ""

    def tools(self) -> Sequence[Tool | Mapping[str, Any]]:
        """Return ``()`` — the harness has no tools of its own."""
        return ()

    # ------------------------------------------------------------------
    # ainvoke — capture loop
    # ------------------------------------------------------------------
    async def ainvoke(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        essential: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        """Run the wrapped agent under capture and return its final state.

        The wrapped agent runs its normal :class:`BaseAgent.ainvoke`
        with a customised :class:`AgentRuntime` whose ``llm`` and
        ``tool_invoker`` are wrapping decorators. Every other
        collaborator (memory, observability, audit, MCP, agent resolver)
        is shared with this harness's own runtime so the capture is
        non-intrusive.
        """
        log_token = bind_context(
            agent_id=self.agent_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
        )
        try:
            wrapped_cls = self.wrapped_agent()
            t0_perf = time.perf_counter()
            trace = Trace(
                agent_id=self.agent_id,
                wrapped_agent=wrapped_cls.__name__,
                started_at=datetime.now(UTC),
                events=[],
            )

            def _append(event: TraceEvent) -> None:
                trace.events.append(event)

            capturing_llm = _CapturingLLMClient(
                self._runtime.llm, _append, t0_perf,
            )
            capturing_invoker = _CapturingToolInvoker(
                self._runtime.tool_invoker, _append, t0_perf,
            )
            custom_runtime = dataclasses.replace(
                self._runtime,
                llm=capturing_llm,
                tool_invoker=capturing_invoker,
            )

            attributes = {
                "agent.id": self.agent_id,
                "agent.tenant_id": tenant_id or "",
                "harness.target": wrapped_cls.__name__,
            }
            try:
                async with self._runtime.observability.start_span(
                    "agent.ainvoke", attributes=attributes,
                ):
                    wrapped = self._construct_wrapped(wrapped_cls, custom_runtime)
                    return await wrapped.ainvoke(
                        messages=messages,
                        essential=essential,
                        tenant_id=tenant_id,
                        thread_id=thread_id,
                    )
            finally:
                trace.finished_at = datetime.now(UTC)
                self.last_trace = trace
        finally:
            unbind_context(log_token)

    # ------------------------------------------------------------------
    # Construction hook
    # ------------------------------------------------------------------
    def _construct_wrapped(
        self,
        cls: type[BaseAgent],
        runtime: AgentRuntime,
    ) -> BaseAgent:
        """Build a fresh wrapped instance bound to the capturing ``runtime``.

        Default: ``cls(runtime)``. The harness needs a wrapped instance
        whose ``_runtime`` is the capturing one — using
        :attr:`AgentRuntime.agent_resolver` would return a DI-resolved
        instance carrying the *real* runtime, defeating capture.

        Override when the wrapped class declares additional
        ``@inject`` constructor parameters; resolve them from
        ``self._runtime.agent_resolver`` (a thin wrapper over the
        container) or pass them explicitly.
        """
        return cls(runtime)


__all__ = [
    "HarnessAgent",
    "LLMCallRecord",
    "ToolDispatchRecord",
    "Trace",
    "TraceEvent",
]
