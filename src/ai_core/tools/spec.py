"""Immutable tool descriptor + structural Protocol.

A :class:`ToolSpec` is what the ``@tool`` decorator produces. It carries
everything needed to (1) advertise the tool to an LLM via OpenAI's
function-calling schema, (2) validate a call's input/output, and (3) route
the call through OPA for authorisation.

ToolSpec deliberately does **not** depend on DI or observability. Those
concerns live in :class:`ai_core.tools.invoker.ToolInvoker`, which is
constructed once at app startup and passed each spec at invoke time.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

ToolHandler = Callable[[BaseModel], Awaitable[BaseModel]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable description of a tool available to an agent.

    Attributes:
        name: Logical tool identifier (must match across versions for upgrades).
        version: Positive integer — incremented on breaking schema changes.
        description: Human-readable description shown to the LLM and dashboards.
        input_model: Pydantic model used to validate raw arguments.
        output_model: Pydantic model used to validate the handler's return.
        handler: The async callable implementing the tool's behaviour.
        opa_path: Decision path for OPA. ``None`` skips policy enforcement
            (suitable for read-only or system tools).
    """

    name: str
    version: int
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    opa_path: str | None

    def openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


@runtime_checkable
class Tool(Protocol):
    """Structural type for anything an agent can advertise to its LLM."""

    name: str
    version: int

    def openai_schema(self) -> dict[str, Any]: ...


__all__ = ["Tool", "ToolHandler", "ToolSpec"]
