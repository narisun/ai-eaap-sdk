# supervisor_demo

A runnable demonstration of the v1.1 `SupervisorAgent` primitive: one
supervisor coordinating two child agents through scripted LLM
responses.

## What it shows

* `SupportSupervisor` (`SupervisorAgent` subclass) with two children:
  `TriageAgent` and `ResearchAgent`.
* The supervisor's LLM emits `tool_call` deltas to delegate work; each
  child is exposed as a tool.
* Each child invocation goes through the real SDK pipeline:
  - DI-resolved through `runtime.agent_resolver`
  - `ToolInvoker` validates the input payload
  - Child runs its own LangGraph (compaction, agent node, tool loop)
  - Child's last assistant message is rendered back as the supervisor's
    tool result
* Cross-cutting concerns auto-apply: each child has its own
  observability span, budget binding, policy evaluator. Spans nest:
  supervisor span ⊃ child span ⊃ child LLM span.

## How to run

```bash
uv run python examples/supervisor_demo/run.py
```

Expected output: a table listing five LLM calls (supervisor → triage →
supervisor → research → supervisor) and a final stitched answer that
references both children's outputs.

## How to extend

1. **Add a typed contract for a child:**

   ```python
   class _AnalystRequest(BaseModel):
       customer_id: str
       topic: str

   class SupportSupervisor(SupervisorAgent):
       def child_input_schema(self, name: str) -> type[BaseModel]:
           return _AnalystRequest if name == "analyst" else super().child_input_schema(name)
   ```

   The supervisor's LLM will see the typed schema and emit structured
   tool_call args matching it.

2. **Add a structured output:**

   ```python
   class _AnalystReply(BaseModel):
       summary: str
       confidence: float

   def child_output_schema(self, name: str) -> type[BaseModel]:
       return _AnalystReply if name == "analyst" else super().child_output_schema(name)

   def render_child_output(self, name: str, child_state: AgentState) -> BaseModel:
       # parse child_state into _AnalystReply
       ...
   ```

3. **Nest a sub-supervisor:** a child can itself be a `SupervisorAgent`.
   OTel baggage and the recursion-limit guard handle nesting naturally.

## Connecting a real LLM

Replace the `ScriptedLLM` in `build_container()` with a real `ILLMClient`
binding (typically via `LiteLLMModule`):

```python
from ai_core.llm import LiteLLMModule

container = Container.build([
    AgentModule(settings=settings),
    LiteLLMModule(),
])
```

Set the LLM environment variables (`EAAP_LLM__DEFAULT_MODEL`, the
provider's API key) and the demo will work against the real model.
