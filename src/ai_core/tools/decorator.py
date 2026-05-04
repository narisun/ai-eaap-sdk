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
import sys
import typing
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from ai_core.tools.spec import ToolHandler, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

InModel = TypeVar("InModel", bound=BaseModel)
OutModel = TypeVar("OutModel", bound=BaseModel)


def tool(
    *,
    name: str,
    version: int = 1,
    description: str | None = None,
    opa_path: str | None = "eaap/agent/tool_call/allow",
) -> Callable[[Callable[[InModel], Awaitable[OutModel]]], ToolSpec]:
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

    def decorate(fn: Callable[[InModel], Awaitable[OutModel]]) -> ToolSpec:
        fn_any = typing.cast("Callable[..., object]", fn)

        if not inspect.iscoroutinefunction(fn_any):
            raise TypeError(
                f"@tool requires an async function; '{fn_any.__name__}' is sync. "
                f"Define it with `async def {fn_any.__name__}(...)`."
            )

        sig = inspect.signature(fn_any)
        all_params = list(sig.parameters.values())

        # Detect self/cls (methods are not supported).
        if all_params and all_params[0].name in ("self", "cls"):
            raise TypeError(
                f"@tool '{fn_any.__name__}' appears to be a method (first parameter "
                f"is '{all_params[0].name}'). Tools must be module-level functions, "
                f"not methods or classmethods."
            )

        # Reject *args.
        var_pos = [
            p for p in all_params
            if p.kind is inspect.Parameter.VAR_POSITIONAL
        ]
        if var_pos:
            raise TypeError(
                f"@tool '{fn_any.__name__}' must not use *args. "
                f"Tools accept a single payload object: "
                f"`async def {fn_any.__name__}(payload: MyInModel) -> MyOutModel`."
            )

        # Reject **kwargs.
        var_kw = [
            p for p in all_params
            if p.kind is inspect.Parameter.VAR_KEYWORD
        ]
        if var_kw:
            raise TypeError(
                f"@tool '{fn_any.__name__}' must not use **kwargs. "
                f"Tools accept a single payload object: "
                f"`async def {fn_any.__name__}(payload: MyInModel) -> MyOutModel`."
            )

        # Reject keyword-only params.
        kw_only = [
            p for p in all_params
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]
        if kw_only:
            raise TypeError(
                f"@tool '{fn_any.__name__}' must not use keyword-only parameters "
                f"(found: {', '.join(p.name for p in kw_only)}). "
                f"Tools accept a single positional payload object: "
                f"`async def {fn_any.__name__}(payload: MyInModel) -> MyOutModel`."
            )

        # Now count only positional params (POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD).
        params = [
            p for p in all_params
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(params) != 1:
            raise TypeError(
                f"@tool '{fn_any.__name__}' must take exactly one positional "
                f"Pydantic-typed parameter; got {len(params)}. "
                f"Tools accept a single payload object: "
                f"`async def {fn_any.__name__}(payload: MyInModel) -> MyOutModel`."
            )

        # Resolve string annotations (e.g. when `from __future__ import annotations`
        # is in use in the user's module).
        # We try progressively richer namespaces so that tools defined inside
        # functions (common in tests) can reference locally-scoped models.
        localns: dict[str, object] | None = None
        try:
            # Frame 0 = decorate (this function); frame 1 = call site where
            # @tool(...) is applied. Merge locals + globals so that models
            # defined inside a function body (common in tests) are visible.
            frame = sys._getframe(1)  # private but stable CPython API
            localns = {**frame.f_globals, **frame.f_locals}
        except (AttributeError, ValueError):
            pass

        try:
            hints = typing.get_type_hints(fn_any, localns=localns)
        except (NameError, AttributeError, TypeError) as exc:
            raise TypeError(
                f"@tool could not resolve type hints for '{fn_any.__name__}': {exc}. "
                f"This often means the annotation references a symbol imported only under "
                f"`TYPE_CHECKING`; tool input/output models must be importable at runtime."
            ) from exc

        param_name = params[0].name
        input_type = hints.get(param_name)
        if not (isinstance(input_type, type) and issubclass(input_type, BaseModel)):
            raise TypeError(
                f"@tool '{fn_any.__name__}' parameter '{param_name}' must be annotated with a "
                f"Pydantic BaseModel subclass; got {input_type!r}. "
                f"Define `class {param_name.title()}Model(BaseModel): ...` and annotate "
                f"`{param_name}: {param_name.title()}Model`."
            )

        output_type = hints.get("return")
        if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
            raise TypeError(
                f"@tool '{fn_any.__name__}' return annotation must be a Pydantic BaseModel "
                f"subclass; got {output_type!r}. "
                f"Define `class {fn_any.__name__.title()}Out(BaseModel): ...` and annotate "
                f"`-> {fn_any.__name__.title()}Out`."
            )

        resolved_description = description
        if resolved_description is None:
            doc = inspect.getdoc(fn_any) or ""
            resolved_description = doc.strip()

        handler: ToolHandler = typing.cast("ToolHandler", fn_any)
        return ToolSpec(
            name=name,
            version=version,
            description=resolved_description,
            input_model=input_type,
            output_model=output_type,
            handler=handler,
            opa_path=opa_path,
        )

    return decorate


__all__ = ["tool"]
