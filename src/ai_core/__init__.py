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

from ai_core.agents import (
    AgentRuntime,
    AgentState,
    BaseAgent,
    Plan,
    PlanAck,
    PlanningAgent,
    PlanStep,
    SupervisorAgent,
    TaskInput,
    TaskOutput,
    new_agent_state,
)
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
from ai_core.tools import Tool, ToolSpec, make_tool, tool

__all__ = [
    "AICoreApp",
    "AgentRecursionLimitError",
    "AgentRuntime",
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
    "IHealthProbe",
    "LLMInvocationError",
    "LLMTimeoutError",
    "MCPTransportError",
    "Plan",
    "PlanAck",
    "PlanStep",
    "PlanningAgent",
    "PolicyDenialError",
    "ProbeResult",
    "RegistryError",
    "SchemaValidationError",
    "SecretResolutionError",
    "StorageError",
    "SupervisorAgent",
    "TaskInput",
    "TaskOutput",
    "Tool",
    "ToolExecutionError",
    "ToolSpec",
    "ToolValidationError",
    "make_tool",
    "new_agent_state",
    "tool",
]

# Derive ``__version__`` from the installed package metadata so the
# literal in pyproject.toml is the single source of truth. A hardcoded
# value here drifts every release; importlib.metadata lookup never does.
# Falls back to "0.0.0+unknown" when running directly from a source
# checkout that has not been installed (rare; mostly affects ad-hoc
# `python src/...` invocations during local development).
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("ai-eaap-sdk")
except PackageNotFoundError:  # pragma: no cover — only hit in uninstalled checkouts
    __version__ = "0.0.0+unknown"
