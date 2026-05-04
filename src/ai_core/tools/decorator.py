"""``@tool`` decorator — converts an async Pydantic-typed function into a ToolSpec.

The decorator enforces a small contract at definition time so that runtime
invocation never has to ask "is this even shaped like a tool?":

* the decorated callable must be ``async``,
* it must accept exactly one positional parameter typed as a Pydantic
  ``BaseModel`` subclass,
* its return annotation must be a Pydantic ``BaseModel`` subclass.

Definition-time has zero dependency on DI, observability, or OPA — those
concerns are wired in by :class:`ai_core.tools.invoker.ToolInvoker`.
"""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ai_core.tools.spec import ToolHandler, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Callable


def tool(
    *,
    name: str,
    version: int = 1,
    description: str | None = None,
    opa_path: str | None = "eaap/agent/tool_call/allow",
) -> Callable[[ToolHandler], ToolSpec]:
    """Decorate an async function to expose it as a SDK tool.

    Args:
        name: Logical tool identifier.
        version: Positive integer schema version (must be ``>= 1``).
        description: Optional human-readable description. If omitted, the
            decorated function's docstring is used (or ``""`` if absent).
        opa_path: OPA decision path consulted before the handler runs. Pass
            ``None`` to skip policy enforcement.

    Returns:
        A decorator that consumes the function and returns a :class:`ToolSpec`.

    Raises:
        TypeError: If the function is not async, is not single-arg, or its
            input/return annotations are not Pydantic ``BaseModel`` subclasses.
        ValueError: If ``name`` is empty or ``version`` is less than 1.
    """
    if not name:
        raise ValueError("tool(): name must be non-empty")
    if version < 1:
        raise ValueError("tool(): version must be >= 1")

    def decorate(fn: ToolHandler) -> ToolSpec:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@tool requires an async function; '{fn.__name__}' is sync."
            )

        sig = inspect.signature(fn)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(params) != 1:
            raise TypeError(
                f"@tool requires exactly one positional parameter; "
                f"'{fn.__name__}' has {len(params)}."
            )

        # Resolve string annotations (e.g. when `from __future__ import annotations`
        # is in use in the user's module).
        try:
            hints = typing.get_type_hints(fn)
        except Exception as exc:
            raise TypeError(
                f"@tool could not resolve type hints for '{fn.__name__}': {exc}"
            ) from exc

        param_name = params[0].name
        input_type = hints.get(param_name)
        if not (isinstance(input_type, type) and issubclass(input_type, BaseModel)):
            raise TypeError(
                f"@tool '{fn.__name__}' parameter '{param_name}' must be "
                f"annotated with a Pydantic BaseModel subclass; got {input_type!r}."
            )

        output_type = hints.get("return")
        if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
            raise TypeError(
                f"@tool '{fn.__name__}' return annotation must be a Pydantic "
                f"BaseModel subclass; got {output_type!r}."
            )

        resolved_description = description
        if resolved_description is None:
            doc = inspect.getdoc(fn) or ""
            resolved_description = doc.strip()

        return ToolSpec(
            name=name,
            version=version,
            description=resolved_description,
            input_model=input_type,
            output_model=output_type,
            handler=fn,
            opa_path=opa_path,
        )

    return decorate


__all__ = ["tool"]
