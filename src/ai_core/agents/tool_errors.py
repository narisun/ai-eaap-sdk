"""Pluggable rendering of tool-dispatch errors into LangChain ``ToolMessage``s.

Background
----------
When an LLM-issued tool call fails — bad input shape, OPA denial, handler
exception, or an unknown name — :class:`BaseAgent._tool_node` cannot raise:
LangGraph would short-circuit the run and the LLM would get no signal. The
node has to fabricate a ``ToolMessage`` whose ``content`` is what the LLM
sees on its next turn. The English text on that message is policy: some
products want strict failure (no recovery message), some want localized
strings, some want stack-trace breadcrumbs in dev environments only.

Pre-v1 the strings were inlined in :class:`BaseAgent._tool_node`; this
module extracts that policy behind :class:`IToolErrorRenderer` so hosts
can override it via DI.

The default :class:`DefaultToolErrorRenderer` reproduces the pre-v1
English exactly, so behaviour is unchanged unless a host installs a
replacement binding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from langchain_core.messages import ToolMessage

from ai_core.exceptions import (
    PolicyDenialError,
    ToolExecutionError,
    ToolValidationError,
)


@runtime_checkable
class IToolErrorRenderer(Protocol):
    """Renders tool-dispatch failures into ``ToolMessage`` instances.

    Each method MUST return a ``ToolMessage``; raising would short-circuit
    the LangGraph turn and starve the LLM of feedback.
    """

    def render_parse_error(
        self, *, tool_name: str, tool_call_id: str, raw: str
    ) -> ToolMessage:
        """Render an unparseable JSON argument blob from the LLM."""
        ...

    def render_unknown_tool(
        self, *, tool_name: str, tool_call_id: str
    ) -> ToolMessage:
        """Render a call referencing a tool that no agent has registered."""
        ...

    def render_validation_error(
        self, *, tool_name: str, tool_call_id: str, error: ToolValidationError
    ) -> ToolMessage:
        """Render an input-validation failure raised by the invoker."""
        ...

    def render_policy_denial(
        self, *, tool_name: str, tool_call_id: str, error: PolicyDenialError
    ) -> ToolMessage:
        """Render an OPA / authorization denial."""
        ...

    def render_execution_error(
        self, *, tool_name: str, tool_call_id: str, error: ToolExecutionError
    ) -> ToolMessage:
        """Render a handler exception wrapped by the invoker."""
        ...


class DefaultToolErrorRenderer:
    """v1 default — preserves pre-v1 behaviour byte-for-byte.

    Hosts that want strict-failure semantics, localized text, or
    redaction of error details replace this binding with their own
    :class:`IToolErrorRenderer` implementation.
    """

    def render_parse_error(
        self, *, tool_name: str, tool_call_id: str, raw: str
    ) -> ToolMessage:
        return ToolMessage(
            content=f"Tool '{tool_name}' arguments were not valid JSON: {raw!r}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def render_unknown_tool(
        self, *, tool_name: str, tool_call_id: str
    ) -> ToolMessage:
        return ToolMessage(
            content=f"Unknown tool '{tool_name}'.",
            tool_call_id=tool_call_id,
            name=tool_name or "",
        )

    def render_validation_error(
        self, *, tool_name: str, tool_call_id: str, error: ToolValidationError
    ) -> ToolMessage:
        first_err = (
            error.details.get("errors", [{}])[0] if error.details.get("errors") else {}
        )
        msg = first_err.get("msg") if isinstance(first_err, Mapping) else None
        return ToolMessage(
            content=f"Validation failed for '{tool_name}': {msg or error.message}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def render_policy_denial(
        self, *, tool_name: str, tool_call_id: str, error: PolicyDenialError
    ) -> ToolMessage:
        reason = error.details.get("reason") or error.message
        return ToolMessage(
            content=f"Tool '{tool_name}' denied by policy: {reason}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def render_execution_error(
        self, *, tool_name: str, tool_call_id: str, error: ToolExecutionError
    ) -> ToolMessage:
        return ToolMessage(
            content=f"Tool '{tool_name}' failed: {error.message}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )


__all__ = ["DefaultToolErrorRenderer", "IToolErrorRenderer"]
