"""Programmatic :class:`ToolSpec` factory for DI-aware tool authoring.

The :func:`@tool <ai_core.tools.tool>` decorator is the happy-path API:
it consumes a module-level async function, runs definition-time
validation (single Pydantic arg, async handler, Pydantic return type),
and returns a :class:`ToolSpec`. By design it refuses to decorate
methods or closures, which means a tool whose handler needs DI-resolved
state (a database session, an HTTP client, a feature-flag service)
cannot be expressed.

:func:`make_tool` is the escape hatch. It accepts any async callable
matching ``(payload: BaseModel) -> Awaitable[BaseModel]`` — including
bound methods, lambdas, partials, and closures — and returns the same
:class:`ToolSpec` shape. The expected usage pattern is to build the
spec inside the host's agent ``__init__`` so the bound method's
``self`` reference and the agent's DI-resolved collaborators are
captured naturally::

    class CustomerLookupService:
        def __init__(self, db: DBSession) -> None:
            self._db = db

        async def lookup(self, p: CustomerIn) -> CustomerOut:
            row = await self._db.fetch_one(...)
            return CustomerOut(...)

    class SupportAgent(BaseAgent):
        @inject
        def __init__(
            self, runtime: AgentRuntime, svc: CustomerLookupService,
        ) -> None:
            super().__init__(runtime)
            self._tools = (
                make_tool(
                    name="customer_lookup",
                    version=1,
                    handler=svc.lookup,
                    description="Find a customer by external id",
                ),
            )

        def tools(self):
            return self._tools

Validation differences vs ``@tool``
------------------------------------
* Methods, closures, partials, and lambdas are accepted (the whole
  point).
* The handler must still be async.
* The handler must still take exactly one Pydantic-typed positional
  argument and return a Pydantic-typed value.
* Type hints are resolved via :func:`typing.get_type_hints`. If the
  handler's annotations reference symbols only available under
  ``TYPE_CHECKING``, a ``TypeError`` is raised at construction time —
  the same behaviour as the decorator.

The returned :class:`ToolSpec` is interchangeable with one produced by
``@tool``: both flow through :class:`ToolInvoker` identically and
register with the schema registry the same way.
"""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ai_core.tools.spec import ToolHandler, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def make_tool(
    *,
    name: str,
    version: int,
    handler: Callable[..., Awaitable[BaseModel]],
    description: str = "",
    opa_path: str | None = "eaap/agent/tool_call/allow",
) -> ToolSpec:
    """Build a :class:`ToolSpec` from any async ``(payload) -> result`` callable.

    Args:
        name: Logical tool identifier.
        version: Positive integer schema version (must be ``>= 1``).
        handler: An async callable accepting a single Pydantic-typed
            positional argument and returning a Pydantic-typed value.
            Bound methods, closures, and lambdas are all acceptable.
        description: Human-readable description shown to the LLM and
            dashboards. If omitted and ``handler`` has a docstring, the
            docstring is used.
        opa_path: OPA decision path consulted before the handler runs.
            Pass ``None`` to skip policy enforcement.

    Returns:
        A :class:`ToolSpec` ready to be returned from
        :meth:`BaseAgent.tools` or registered explicitly.

    Raises:
        ValueError: If ``name`` is empty or ``version`` is less than 1.
        TypeError: If ``handler`` is not async, has the wrong arity, or
            has missing/non-Pydantic type annotations.
    """
    if not name:
        raise ValueError("make_tool(): name must be non-empty")
    if version < 1:
        raise ValueError("make_tool(): version must be >= 1")
    if not callable(handler):
        raise TypeError(
            f"make_tool() handler must be callable; got {type(handler).__name__}"
        )
    if not inspect.iscoroutinefunction(_underlying(handler)):
        raise TypeError(
            f"make_tool() handler must be async; "
            f"{getattr(handler, '__qualname__', handler)!r} is sync."
        )

    input_type, output_type = _resolve_payload_types(handler)
    resolved_description = description or (inspect.getdoc(handler) or "").strip()

    cast_handler: ToolHandler = typing.cast("ToolHandler", handler)
    return ToolSpec(
        name=name,
        version=version,
        description=resolved_description,
        input_model=input_type,
        output_model=output_type,
        handler=cast_handler,
        opa_path=opa_path,
    )


def _underlying(handler: Callable[..., object]) -> Callable[..., object]:
    """Return the underlying function for a bound method or partial.

    :func:`inspect.iscoroutinefunction` returns ``False`` for some
    wrappers (functools.partial in particular) even when the wrapped
    function is a coroutine. Unwrap one level so the async check holds
    for realistic factory usage.
    """
    inner = getattr(handler, "func", None)  # functools.partial
    if inner is not None:
        return _underlying(inner)
    return getattr(handler, "__func__", handler)  # bound method -> function


def _resolve_payload_types(
    handler: Callable[..., object],
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Validate handler signature and return ``(input_model, output_model)``.

    Mirrors the decorator's validation but accepts bound methods (skips
    the implicit ``self`` parameter), partials (signature reflects bound
    args), and closures.
    """
    underlying = _underlying(handler)
    # Inspect ``handler`` (not ``underlying``) so functools.partial's bound
    # arguments are excluded from the visible parameter list. Bound methods
    # already resolve self correctly because ``signature`` understands
    # ``__self__``.
    sig = inspect.signature(handler)
    all_params = list(sig.parameters.values())

    # Drop self/cls if present (bound methods, classmethods).
    if all_params and all_params[0].name in ("self", "cls"):
        all_params = all_params[1:]

    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in all_params):
        raise TypeError(
            f"make_tool() handler must not use *args. "
            f"Tools accept a single payload object."
        )
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in all_params):
        raise TypeError(
            f"make_tool() handler must not use **kwargs. "
            f"Tools accept a single payload object."
        )
    kw_only = [p for p in all_params if p.kind is inspect.Parameter.KEYWORD_ONLY]
    if kw_only:
        raise TypeError(
            f"make_tool() handler must not use keyword-only parameters "
            f"(found: {', '.join(p.name for p in kw_only)})."
        )
    positional = [
        p
        for p in all_params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) != 1:
        raise TypeError(
            f"make_tool() handler must take exactly one positional payload; "
            f"got {len(positional)}."
        )

    try:
        hints = typing.get_type_hints(underlying)
    except (NameError, AttributeError, TypeError) as exc:
        raise TypeError(
            f"make_tool() could not resolve type hints for "
            f"{getattr(handler, '__qualname__', handler)!r}: {exc}. "
            "Tool input/output models must be importable at runtime "
            "(not gated behind TYPE_CHECKING)."
        ) from exc

    param_name = positional[0].name
    input_type = hints.get(param_name)
    if not (isinstance(input_type, type) and issubclass(input_type, BaseModel)):
        raise TypeError(
            f"make_tool() handler parameter {param_name!r} must be annotated "
            f"with a Pydantic BaseModel subclass; got {input_type!r}."
        )
    output_type = hints.get("return")
    if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
        raise TypeError(
            f"make_tool() handler return annotation must be a Pydantic "
            f"BaseModel subclass; got {output_type!r}."
        )
    return input_type, output_type


__all__ = ["make_tool"]
