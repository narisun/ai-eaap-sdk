"""``ToolInvoker`` — runs the schema → OPA → span → handler pipeline.

The invoker is constructed once at app boot (via the DI container) and
holds references to:

* an :class:`IObservabilityProvider` for spans + events,
* an optional :class:`IPolicyEvaluator` for OPA enforcement,
* an optional :class:`SchemaRegistry` for cross-tool discovery.

It is stateless w.r.t. specs — each :py:meth:`invoke` call takes the spec
as a parameter so the same invoker handles every tool in the application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ai_core.exceptions import (
    PolicyDenialError,
    SchemaValidationError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ai_core.di.interfaces import IObservabilityProvider, IPolicyEvaluator
    from ai_core.schema.registry import SchemaRegistry
    from ai_core.tools.spec import ToolSpec

_logger = get_logger(__name__)


class ToolInvoker:
    """Runs ``ToolSpec`` instances through the SDK's standard tool-call pipeline.

    Pipeline (all steps wrapped in a single ``tool.invoke`` span):

    1. Validate ``raw_args`` -> ``spec.input_model`` (``ToolValidationError`` on fail).
    2. If ``spec.opa_path`` and a policy evaluator is wired, evaluate;
       deny -> ``PolicyDenialError``.
    3. ``await spec.handler(payload)``; raise -> ``ToolExecutionError`` chained.
    4. Validate output via ``spec.output_model.model_validate`` (``ToolValidationError``).

    After the span closes cleanly, emit a ``"tool.completed"`` event and return
    ``output.model_dump(mode="json")``.
    """

    def __init__(
        self,
        *,
        observability: IObservabilityProvider,
        policy: IPolicyEvaluator | None = None,
        registry: SchemaRegistry | None = None,
    ) -> None:
        self._observability = observability
        self._policy = policy
        self._registry = registry

    def register(self, spec: ToolSpec) -> None:
        """Register a spec with the underlying :class:`SchemaRegistry`. Idempotent.

        No-op when this invoker was constructed without a registry.
        """
        if self._registry is None:
            return
        try:
            self._registry.register(
                spec.name,
                spec.version,
                input_schema=spec.input_model,
                output_schema=spec.output_model,
                description=spec.description,
            )
        except SchemaValidationError:
            # Already registered with the same (name, version): treat as idempotent.
            existing = self._registry.get(spec.name, version=spec.version)
            if (
                existing.input_schema is not spec.input_model
                or existing.output_schema is not spec.output_model
            ):
                raise

    async def invoke(
        self,
        spec: ToolSpec,
        raw_args: Mapping[str, Any],
        *,
        principal: Mapping[str, Any] | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Mapping[str, Any]:
        attrs: dict[str, Any] = {
            "tool.name": spec.name,
            "tool.version": spec.version,
            "agent_id": agent_id or "",
            "tenant_id": tenant_id or "",
        }
        async with self._observability.start_span("tool.invoke", attributes=attrs):
            # ----- 1. Input validation --------------------------------------------
            try:
                payload = spec.input_model.model_validate(dict(raw_args))
            except ValidationError as exc:
                raise ToolValidationError(
                    f"Tool '{spec.name}' v{spec.version} input failed validation.",
                    details={
                        "tool": spec.name,
                        "version": spec.version,
                        "side": "input",
                        "errors": exc.errors(),
                    },
                    cause=exc,
                ) from exc

            # ----- 2. OPA enforcement ---------------------------------------------
            if spec.opa_path is not None and self._policy is not None:
                decision = await self._policy.evaluate(
                    decision_path=spec.opa_path,
                    input={
                        "tool": spec.name,
                        "version": spec.version,
                        "payload": payload.model_dump(),
                        "user": dict(principal or {}),
                        "agent_id": agent_id,
                        "tenant_id": tenant_id,
                    },
                )
                if not decision.allowed:
                    raise PolicyDenialError(
                        f"Tool '{spec.name}' v{spec.version} denied by policy: "
                        f"{decision.reason or 'no reason provided'}",
                        details={
                            "tool": spec.name,
                            "version": spec.version,
                            "reason": decision.reason,
                            "agent_id": agent_id,
                            "tenant_id": tenant_id,
                        },
                    )

            # ----- 3+4. Handler call ----------------------------------------------
            try:
                result: Any = await spec.handler(payload)
            except Exception as exc:  # wrap as ToolExecutionError
                raise ToolExecutionError(
                    f"Tool '{spec.name}' v{spec.version} failed: {exc}",
                    details={
                        "tool": spec.name,
                        "version": spec.version,
                        "agent_id": agent_id,
                        "tenant_id": tenant_id,
                    },
                    cause=exc,
                ) from exc

            # ----- 5. Output validation -------------------------------------------
            try:
                validated: BaseModel = spec.output_model.model_validate(result)
            except ValidationError as exc:
                _logger.warning(
                    "tool.output_validation_failed",
                    tool_name=spec.name, tool_version=spec.version,
                    agent_id=agent_id, tenant_id=tenant_id,
                )
                raise ToolValidationError(
                    f"Tool '{spec.name}' v{spec.version} returned invalid data.",
                    details={
                        "tool": spec.name,
                        "version": spec.version,
                        "side": "output",
                        "errors": exc.errors(),
                    },
                    cause=exc,
                ) from exc

        # ----- 6. Completion event (outside span — span already closed cleanly) ---
        await self._observability.record_event("tool.completed", attributes=attrs)
        return validated.model_dump(mode="json")


__all__ = ["ToolInvoker"]
