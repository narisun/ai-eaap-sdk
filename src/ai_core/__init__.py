"""ai_core — Enterprise Agentic AI Platform (EAAP) core SDK.

Curated public surface — these names are the documented entry points.
Power-user surface (Container, AgentModule, the I* interfaces) lives in
:mod:`ai_core.di.*`; per-subsystem detail (config, schema, security, mcp)
lives in its respective subpackage.

"Hello, agent" example::

    from ai_core import AICoreApp, BaseAgent, tool
    from pydantic import BaseModel

    class HiIn(BaseModel): name: str
    class HiOut(BaseModel): greeting: str

    @tool(name="say_hi", version=1)
    async def say_hi(p: HiIn) -> HiOut:
        return HiOut(greeting=f"Hi {p.name}!")

    class Greeter(BaseAgent):
        agent_id = "greeter"
        def system_prompt(self) -> str: return "You greet people."
        def tools(self): return [say_hi]

    async def main() -> None:
        async with AICoreApp() as app:
            agent = app.agent(Greeter)
            ...

"""

from __future__ import annotations

from ai_core.agents import AgentState, BaseAgent, new_agent_state
from ai_core.app import AICoreApp, HealthSnapshot
from ai_core.audit import AuditEvent, AuditRecord, IAuditSink
from ai_core.exceptions import (
    AgentRecursionLimitError,
    AgentRuntimeError,
    BudgetExceededError,
    ConfigurationError,
    DependencyResolutionError,
    EAAPBaseException,
    ErrorCode,
    LLMInvocationError,
    LLMTimeoutError,
    MCPTransportError,
    PolicyDenialError,
    RegistryError,
    SchemaValidationError,
    SecretResolutionError,
    StorageError,
    ToolExecutionError,
    ToolValidationError,
)
from ai_core.health import IHealthProbe, ProbeResult
from ai_core.tools import Tool, ToolSpec, tool

__all__ = [
    "AICoreApp",
    "AgentRecursionLimitError",
    "AgentRuntimeError",
    "AgentState",
    "AuditEvent",
    "AuditRecord",
    "BaseAgent",
    "BudgetExceededError",
    "ConfigurationError",
    "DependencyResolutionError",
    "EAAPBaseException",
    "ErrorCode",
    "HealthSnapshot",
    "IAuditSink",
    # Health
    "IHealthProbe",
    "LLMInvocationError",
    "LLMTimeoutError",
    "MCPTransportError",
    "PolicyDenialError",
    "ProbeResult",
    "RegistryError",
    "SchemaValidationError",
    "SecretResolutionError",
    "StorageError",
    "Tool",
    "ToolExecutionError",
    "ToolSpec",
    "ToolValidationError",
    "new_agent_state",
    "tool",
]

__version__ = "0.1.0"
