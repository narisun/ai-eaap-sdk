# Phase 12 — MCP Resources + Prompts: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend agent ↔ MCP integration with read-only resources (LLM-callable), application-invoked prompts, and an `unwrap_mcp_tool_message` helper that closes Phase 11's `{"value": ...}` pedagogical wart.

**Architecture:** Resources become parameter-less tools via a new `MCPResourceSpec(MCPToolSpec)` subclass; the lazy resolver from Phase 11 grows a parallel `resolve_mcp_resources()`; `BaseAgent._all_tools()` merges both into the existing tool dispatch path. Prompts get a separate application-facing API (`agent.list_prompts()` / `agent.get_prompt()`) — no LLM exposure, no audit, no OPA. `unwrap_mcp_tool_message` is a free function in `ai_core.mcp.tools`.

**Tech Stack:** Python 3.11+, Pydantic v2, FastMCP 3.2.4 (already a core dep), `mcp.shared.exceptions.McpError` (FastMCP transitive). Toolchain: `uv run pytest`, `uv run ruff check`. (`uv` may not be on every dev machine — substitute `.venv/bin/python` and `.venv/bin/ruff` as needed.)

---

## File map

| Path | New / Modified | Purpose |
| --- | --- | --- |
| `src/ai_core/mcp/tools.py` | Modified | Add `MCPResourceSpec` subclass + `unwrap_mcp_tool_message` helper |
| `src/ai_core/mcp/resolver.py` | Modified | Add `_is_method_not_found` predicate + `resolve_mcp_resources` |
| `src/ai_core/mcp/prompts.py` | New | `MCPPrompt`, `MCPPromptArgument`, `MCPPromptMessage` frozen dataclasses |
| `src/ai_core/mcp/__init__.py` | Modified | Export `MCPResourceSpec`, `MCPPrompt`, `MCPPromptArgument`, `MCPPromptMessage`, `resolve_mcp_resources`, `unwrap_mcp_tool_message` |
| `src/ai_core/agents/base.py` | Modified | Extend `_all_tools()` to merge resources; add `list_prompts()` + `get_prompt()` |
| `tests/unit/mcp/test_tools.py` | Modified | Add tests for `MCPResourceSpec` and `unwrap_mcp_tool_message` |
| `tests/unit/mcp/test_resolver.py` | Modified | Add tests for `resolve_mcp_resources` and `_is_method_not_found` |
| `tests/unit/mcp/test_prompts.py` | New | Tests for prompt types |
| `tests/unit/agents/test_base_mcp.py` | Modified | Add tests for `_all_tools()` resource merging, `list_prompts()`, `get_prompt()` |
| `tests/integration/mcp/test_agent_uses_mcp.py` | Modified | Add resource-read end-to-end test |
| `tests/integration/mcp/test_prompts.py` | New | End-to-end prompt fetch + splice test |
| `examples/mcp_server_demo/server.py` | Modified | Add `documentation` resource + `summarize_text` prompt |
| `examples/mcp_server_demo/agent_demo.py` | Modified | Show resource being read; use `unwrap_mcp_tool_message` |
| `examples/mcp_server_demo/prompt_demo.py` | New | Application-invoked prompt → splice → ainvoke |
| `examples/mcp_server_demo/README.md` | Modified | Document resource + prompt demos |
| `scripts/run_examples.sh` | Modified | Add `mcp_prompt_demo` smoke entry |

---

## Task 1: `unwrap_mcp_tool_message` helper

**Why:** Closes Phase 11's `{"value": ...}` pedagogical wart. Tiny, foundational. Doesn't depend on anything new — can ship as a pure helper.

**Files:**
- Modify: `src/ai_core/mcp/tools.py` (add the function + add `json` import)
- Modify: `src/ai_core/mcp/__init__.py` (export)
- Test: `tests/unit/mcp/test_tools.py` (extend)

- [ ] **Step 1: Write the failing tests**

In `tests/unit/mcp/test_tools.py`, add at the end of the file:

```python
# ---------------------------------------------------------------------------
# unwrap_mcp_tool_message tests (Phase 12)
# ---------------------------------------------------------------------------
from ai_core.mcp.tools import unwrap_mcp_tool_message


def test_unwrap_returns_value_for_single_key_envelope() -> None:
    """{'value': X} → X."""
    assert unwrap_mcp_tool_message('{"value": "hello"}') == "hello"
    assert unwrap_mcp_tool_message('{"value": 42}') == 42
    assert unwrap_mcp_tool_message('{"value": [1, 2]}') == [1, 2]
    assert unwrap_mcp_tool_message('{"value": null}') is None


def test_unwrap_returns_parsed_dict_when_multiple_keys() -> None:
    """A dict with more than just 'value' is returned as-is (parsed)."""
    raw = '{"value": "x", "meta": "y"}'
    assert unwrap_mcp_tool_message(raw) == {"value": "x", "meta": "y"}


def test_unwrap_returns_parsed_dict_when_no_value_key() -> None:
    """A dict without a 'value' key is returned parsed."""
    raw = '{"foo": "bar"}'
    assert unwrap_mcp_tool_message(raw) == {"foo": "bar"}


def test_unwrap_returns_parsed_non_dict() -> None:
    """JSON list / scalar at the top level is returned parsed."""
    assert unwrap_mcp_tool_message('[1, 2, 3]') == [1, 2, 3]
    assert unwrap_mcp_tool_message('"plain"') == "plain"
    assert unwrap_mcp_tool_message('42') == 42


def test_unwrap_returns_raw_string_when_not_json() -> None:
    """Non-JSON string is returned verbatim."""
    assert unwrap_mcp_tool_message("not json") == "not json"
    assert unwrap_mcp_tool_message("") == ""


def test_unwrap_raises_typeerror_on_non_string() -> None:
    """Non-string input is rejected."""
    with pytest.raises(TypeError):
        unwrap_mcp_tool_message(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        unwrap_mcp_tool_message(None)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/pytest tests/unit/mcp/test_tools.py -v -k unwrap`

Expected: FAIL with `ImportError: cannot import name 'unwrap_mcp_tool_message' from 'ai_core.mcp.tools'`.

- [ ] **Step 3: Add `import json` to `src/ai_core/mcp/tools.py`**

In `src/ai_core/mcp/tools.py`, find the imports block. Add `import json` to the stdlib imports (alongside `import copy`).

- [ ] **Step 4: Implement `unwrap_mcp_tool_message`**

In `src/ai_core/mcp/tools.py`, add the function at the end of the file (above `__all__`):

```python
def unwrap_mcp_tool_message(content: str) -> Any:
    """Unwrap MCPToolSpec's {"value": ...} envelope from a ToolMessage.content string.

    Returns the inner value when content is a JSON object with exactly one key
    "value" (the standard MCPToolSpec/MCPResourceSpec envelope). Otherwise
    returns the parsed JSON, or the raw string if it's not JSON at all.

    Use this to display MCP tool results cleanly in user-facing UIs without
    re-implementing the unwrap pattern inline.

    Args:
        content: The `ToolMessage.content` string from an MCP tool dispatch.

    Returns:
        The unwrapped value, parsed JSON, or raw string.

    Raises:
        TypeError: If `content` is not a string.
    """
    if not isinstance(content, str):
        raise TypeError(
            f"unwrap_mcp_tool_message expected str, got {type(content).__name__}"
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, dict) and set(parsed.keys()) == {"value"}:
        return parsed["value"]
    return parsed
```

- [ ] **Step 5: Update `__all__` in `src/ai_core/mcp/tools.py`**

Replace the existing `__all__ = ["MCPToolSpec"]` line with:

```python
__all__ = ["MCPToolSpec", "unwrap_mcp_tool_message"]
```

- [ ] **Step 6: Update `src/ai_core/mcp/__init__.py` to re-export**

In `src/ai_core/mcp/__init__.py`, update the import from `.tools` to bring in `unwrap_mcp_tool_message` and add it to `__all__` (keeping alphabetical order):

```python
from ai_core.mcp.tools import MCPToolSpec, unwrap_mcp_tool_message
```

In the existing `__all__`, add `"unwrap_mcp_tool_message"` in alphabetical order (it goes after `resolve_mcp_tools`).

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/mcp/test_tools.py -v -k unwrap`

Expected: 6 PASSED.

- [ ] **Step 8: Run the full mcp suite — no regressions**

Run: `.venv/bin/pytest tests/unit/mcp/ -v`

Expected: all green (Phase 11's count + 6 new = 42+ passed).

- [ ] **Step 9: Lint**

Run: `.venv/bin/ruff check src/ai_core/mcp/tools.py tests/unit/mcp/test_tools.py src/ai_core/mcp/__init__.py`

Expected: no new errors (any pre-existing violations stay unchanged).

- [ ] **Step 10: Commit**

```bash
git add src/ai_core/mcp/tools.py src/ai_core/mcp/__init__.py tests/unit/mcp/test_tools.py
git commit -m "feat(mcp): unwrap_mcp_tool_message helper

Closes Phase 11's pedagogical wart: callers no longer need to inline
the {\"value\": ...} envelope unwrap when displaying MCP tool results.
Returns the inner value when content is a single-key envelope, else
returns parsed JSON or raw string verbatim."
```

---

## Task 2: `MCPResourceSpec` class

**Why:** The type that bridges MCP resources into `ToolInvoker` as parameter-less tools. Subclass of Phase 11's `MCPToolSpec` — adds one field (the URI), overrides `openai_schema()` to drop the parameters block.

**Files:**
- Modify: `src/ai_core/mcp/tools.py` (add `MCPResourceSpec` subclass)
- Modify: `src/ai_core/mcp/__init__.py` (export)
- Test: `tests/unit/mcp/test_tools.py` (extend)

- [ ] **Step 1: Write the failing tests**

In `tests/unit/mcp/test_tools.py`, add after the existing tests (and before the `unwrap_*` block from Task 1):

```python
# ---------------------------------------------------------------------------
# MCPResourceSpec tests (Phase 12)
# ---------------------------------------------------------------------------
from ai_core.mcp.tools import MCPResourceSpec


def _resource_spec_factory(**overrides) -> MCPResourceSpec:
    """Build an MCPResourceSpec with sensible defaults for testing."""
    server = MCPServerSpec(
        component_id="test-server",
        transport="stdio",
        target="/bin/true",
        opa_decision_path=overrides.pop("opa_decision_path", None),
    )

    async def _noop_handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
        return _MCPPassthroughOutput(value="ok")

    return MCPResourceSpec(
        name=overrides.pop("name", "documentation"),
        version=1,
        description=overrides.pop("description", "Project docs"),
        input_model=_MCPPassthroughInput,
        output_model=_MCPPassthroughOutput,
        handler=_noop_handler,
        opa_path=overrides.pop("opa_path", None),
        mcp_server_spec=server,
        mcp_input_schema={"type": "object", "properties": {}},
        mcp_resource_uri=overrides.pop("mcp_resource_uri", "mcp-demo://docs"),
    )


def test_mcp_resource_spec_is_a_mcp_tool_spec() -> None:
    """MCPResourceSpec subclasses MCPToolSpec → existing isinstance checks find it."""
    from ai_core.tools.spec import ToolSpec

    spec = _resource_spec_factory()
    assert isinstance(spec, MCPResourceSpec)
    assert isinstance(spec, MCPToolSpec)  # parent
    assert isinstance(spec, ToolSpec)  # grandparent


def test_resource_openai_schema_has_no_parameters() -> None:
    """Resources take no args; openai_schema returns a parameter-less shape."""
    spec = _resource_spec_factory(name="docs", description="Project docs")

    schema = spec.openai_schema()

    assert schema == {
        "type": "function",
        "function": {
            "name": "docs",
            "description": "Project docs",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_resource_uri_is_stored_as_string() -> None:
    """mcp_resource_uri is plain str, not pydantic AnyUrl (matches MCPToolSpec field-type style)."""
    spec = _resource_spec_factory(mcp_resource_uri="mcp-demo://docs")
    assert isinstance(spec.mcp_resource_uri, str)
    assert spec.mcp_resource_uri == "mcp-demo://docs"


def test_resource_opa_path_propagates() -> None:
    """When the resolver passes server.opa_decision_path → spec.opa_path, it round-trips."""
    spec = _resource_spec_factory(opa_path="mcp.test-server.allow")
    assert spec.opa_path == "mcp.test-server.allow"


def test_resource_spec_is_frozen() -> None:
    """MCPResourceSpec inherits frozen=True from ToolSpec."""
    from dataclasses import FrozenInstanceError

    spec = _resource_spec_factory()
    with pytest.raises(FrozenInstanceError):
        spec.mcp_resource_uri = "different://uri"  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/pytest tests/unit/mcp/test_tools.py -v -k resource`

Expected: FAIL with `ImportError: cannot import name 'MCPResourceSpec'`.

- [ ] **Step 3: Implement `MCPResourceSpec` in `src/ai_core/mcp/tools.py`**

In `src/ai_core/mcp/tools.py`, add the class after `MCPToolSpec` (and before any `__all__`):

```python
@dataclass(frozen=True, slots=True)
class MCPResourceSpec(MCPToolSpec):
    """An MCP resource exposed as a parameter-less read-only tool.

    Phase 12 maps each resource the server advertises via `list_resources()`
    to one `MCPResourceSpec`. The handler closure (built by the resolver)
    hardcodes `mcp_resource_uri` and dispatches via `client.read_resource(uri)`.

    Attributes:
        mcp_resource_uri: The resource's URI (stored as plain str; FastMCP
            returns `pydantic.AnyUrl` which the resolver casts via `str()`
            on the way in).
    """

    mcp_resource_uri: str

    def openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema with no parameters."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }
```

- [ ] **Step 4: Update `__all__` in `src/ai_core/mcp/tools.py`**

Replace:
```python
__all__ = ["MCPToolSpec", "unwrap_mcp_tool_message"]
```

with:

```python
__all__ = ["MCPResourceSpec", "MCPToolSpec", "unwrap_mcp_tool_message"]
```

- [ ] **Step 5: Update `src/ai_core/mcp/__init__.py` to re-export**

Update the import line:

```python
from ai_core.mcp.tools import MCPResourceSpec, MCPToolSpec, unwrap_mcp_tool_message
```

Add `"MCPResourceSpec"` to `__all__` in alphabetical order (between `MCPPromptMessage` (added in Task 5) — wait, that's not yet present — between `MCPServerSpec` and `MCPToolSpec` for now).

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/unit/mcp/test_tools.py -v -k resource`

Expected: 5 PASSED.

- [ ] **Step 7: Lint**

Run: `.venv/bin/ruff check src/ai_core/mcp/tools.py tests/unit/mcp/test_tools.py`

Expected: no new errors.

- [ ] **Step 8: Commit**

```bash
git add src/ai_core/mcp/tools.py src/ai_core/mcp/__init__.py tests/unit/mcp/test_tools.py
git commit -m "feat(mcp): MCPResourceSpec — frozen subclass for resource-as-tool

Inherits all Phase 11 plumbing (passthrough Pydantic, ToolInvoker
integration). Adds mcp_resource_uri (plain str) and overrides
openai_schema() to return zero-parameter shape — resources take no
args, the URI is hardcoded in the handler closure built by the resolver."
```

---

## Task 3: `resolve_mcp_resources` resolver + `_is_method_not_found` predicate

**Why:** The async helper that turns each declared MCP server's `list_resources()` output into `MCPResourceSpec` instances with closure handlers. Servers that don't expose resources raise `McpError` with code `-32601` (method not found) — we silently skip those.

**Files:**
- Modify: `src/ai_core/mcp/resolver.py` (add `_is_method_not_found`, `resolve_mcp_resources`, `_build_mcp_resource_spec`)
- Modify: `src/ai_core/mcp/__init__.py` (export `resolve_mcp_resources`)
- Test: `tests/unit/mcp/test_resolver.py` (extend)

- [ ] **Step 1: Verify FastMCP's `McpError` shape**

Quick sanity probe:

```bash
.venv/bin/python -c "
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
err = McpError(ErrorData(code=-32601, message='Method not found'))
print('attrs:', [a for a in dir(err) if not a.startswith('_')])
print('error.code:', err.error.code)
print('error.message:', err.error.message)
"
```

Expected: prints `attrs: ['add_note', 'args', 'error', 'with_traceback']`, `error.code: -32601`, `error.message: Method not found`.

This confirms the predicate shape: `isinstance(exc, McpError) and exc.error.code == -32601`.

- [ ] **Step 2: Write the failing tests**

In `tests/unit/mcp/test_resolver.py`, add at the end:

```python
# ---------------------------------------------------------------------------
# resolve_mcp_resources tests (Phase 12)
# ---------------------------------------------------------------------------
from ai_core.mcp.resolver import _is_method_not_found, resolve_mcp_resources
from ai_core.mcp.tools import MCPResourceSpec


@dataclass(frozen=True)
class _FakeFastMCPResource:
    name: str
    description: str | None
    uri: str


@dataclass(frozen=True)
class _FakeReadContents:
    """Stand-in for TextResourceContents."""
    text: str | None = None
    blob: bytes | None = None


class _FakeMCPResourceClient:
    """Fake client supporting list_resources + read_resource."""

    def __init__(
        self,
        resources: list[_FakeFastMCPResource],
        read_results: dict[str, list[_FakeReadContents]] | None = None,
        list_resources_raises: BaseException | None = None,
    ) -> None:
        self._resources = resources
        self._read_results = read_results or {}
        self._list_raises = list_resources_raises
        self.read_resource_invocations: list[str] = []

    async def list_resources(self) -> list[_FakeFastMCPResource]:
        if self._list_raises is not None:
            raise self._list_raises
        return list(self._resources)

    async def read_resource(self, uri: str) -> list[_FakeReadContents]:
        self.read_resource_invocations.append(uri)
        return self._read_results.get(uri, [_FakeReadContents(text=f"contents of {uri}")])


def _make_method_not_found_error() -> Exception:
    """Build a real McpError with code -32601."""
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData
    return McpError(ErrorData(code=-32601, message="Method not found"))


# ---- _is_method_not_found ------------------------------------------------------

def test_is_method_not_found_true_on_minus_32601() -> None:
    """The predicate accepts a real McpError with code -32601."""
    exc = _make_method_not_found_error()
    assert _is_method_not_found(exc) is True


def test_is_method_not_found_false_on_other_codes() -> None:
    """The predicate rejects McpErrors with different codes."""
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData
    exc = McpError(ErrorData(code=-32602, message="Invalid params"))
    assert _is_method_not_found(exc) is False


def test_is_method_not_found_false_on_unrelated_exception() -> None:
    """Non-McpError exceptions are not method-not-found."""
    assert _is_method_not_found(ValueError("nope")) is False
    assert _is_method_not_found(RuntimeError("transport gone")) is False


# ---- resolve_mcp_resources -----------------------------------------------------

async def test_resolves_one_server_one_resource() -> None:
    server = MCPServerSpec(
        component_id="docs-svc", transport="stdio", target="/bin/true",
    )
    fake_resource = _FakeFastMCPResource(
        name="documentation", description="Project docs",
        uri="mcp-demo://docs",
    )
    client = _FakeMCPResourceClient(resources=[fake_resource])
    factory = _FakeFactory({"docs-svc": client})

    specs = await resolve_mcp_resources([server], factory)

    assert len(specs) == 1
    assert isinstance(specs[0], MCPResourceSpec)
    assert specs[0].name == "documentation"
    assert specs[0].description == "Project docs"
    assert specs[0].mcp_resource_uri == "mcp-demo://docs"
    assert specs[0].mcp_input_schema == {"type": "object", "properties": {}}


async def test_resolve_resources_silently_skips_method_not_found() -> None:
    """Servers that don't expose resources raise McpError(-32601); we skip them."""
    server = MCPServerSpec(
        component_id="tool-only", transport="stdio", target="/bin/true",
    )
    client = _FakeMCPResourceClient(
        resources=[],
        list_resources_raises=_make_method_not_found_error(),
    )
    factory = _FakeFactory({"tool-only": client})

    specs = await resolve_mcp_resources([server], factory)

    assert specs == []


async def test_resolve_resources_propagates_other_mcp_errors() -> None:
    """Non-method-not-found McpErrors propagate (e.g., -32602 invalid params)."""
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    server = MCPServerSpec(
        component_id="broken", transport="stdio", target="/bin/true",
    )
    client = _FakeMCPResourceClient(
        resources=[],
        list_resources_raises=McpError(ErrorData(code=-32602, message="invalid params")),
    )
    factory = _FakeFactory({"broken": client})

    with pytest.raises(McpError):
        await resolve_mcp_resources([server], factory)


async def test_resolve_resources_opa_path_propagates() -> None:
    """server.opa_decision_path → spec.opa_path."""
    server = MCPServerSpec(
        component_id="docs", transport="stdio", target="/bin/true",
        opa_decision_path="mcp.docs.allow",
    )
    fake_resource = _FakeFastMCPResource(name="readme", description=None, uri="mcp://readme")
    factory = _FakeFactory({"docs": _FakeMCPResourceClient(resources=[fake_resource])})

    specs = await resolve_mcp_resources([server], factory)

    assert specs[0].opa_path == "mcp.docs.allow"


async def test_resolve_resources_conflict_across_servers_raises() -> None:
    """Two servers exposing the same resource name → RegistryError."""
    server_a = MCPServerSpec(component_id="a", transport="stdio", target="/bin/true")
    server_b = MCPServerSpec(component_id="b", transport="stdio", target="/bin/true")
    factory = _FakeFactory({
        "a": _FakeMCPResourceClient(resources=[
            _FakeFastMCPResource(name="docs", description=None, uri="mcp://a/docs"),
        ]),
        "b": _FakeMCPResourceClient(resources=[
            _FakeFastMCPResource(name="docs", description=None, uri="mcp://b/docs"),
        ]),
    })

    with pytest.raises(RegistryError) as excinfo:
        await resolve_mcp_resources([server_a, server_b], factory)

    assert "docs" in str(excinfo.value)


async def test_resource_handler_returns_text_content() -> None:
    """The handler concatenates TextResourceContents and wraps in _MCPPassthroughOutput."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    client = _FakeMCPResourceClient(
        resources=[_FakeFastMCPResource(name="readme", description=None, uri="mcp://readme")],
        read_results={"mcp://readme": [_FakeReadContents(text="line one"), _FakeReadContents(text="line two")]},
    )
    factory = _FakeFactory({"svc": client})

    specs = await resolve_mcp_resources([server], factory)
    payload = _MCPPassthroughInput.model_validate({})
    result = await specs[0].handler(payload)

    assert isinstance(result, _MCPPassthroughOutput)
    assert result.value == "line one\nline two"
    assert client.read_resource_invocations == ["mcp://readme"]


async def test_resource_handler_substitutes_binary_with_placeholder() -> None:
    """BlobResourceContents become a placeholder with a logged warning."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    client = _FakeMCPResourceClient(
        resources=[_FakeFastMCPResource(name="image", description=None, uri="mcp://image")],
        read_results={"mcp://image": [
            _FakeReadContents(text=None, blob=b"\x89PNG..."),
            _FakeReadContents(text="caption"),
            _FakeReadContents(text=None, blob=b"more bytes"),
        ]},
    )
    factory = _FakeFactory({"svc": client})

    specs = await resolve_mcp_resources([server], factory)
    payload = _MCPPassthroughInput.model_validate({})
    result = await specs[0].handler(payload)

    # Two binary blocks → one placeholder; one text → "caption".
    # Order: text content emitted as encountered; placeholder appended after the loop.
    assert "caption" in result.value
    assert "<binary content suppressed: 2 block(s)>" in result.value
```

- [ ] **Step 3: Run tests to verify failure**

Run: `.venv/bin/pytest tests/unit/mcp/test_resolver.py -v -k 'resource or method_not_found'`

Expected: FAIL with `ImportError: cannot import name '_is_method_not_found'`.

- [ ] **Step 4: Implement `_is_method_not_found` in `src/ai_core/mcp/resolver.py`**

In `src/ai_core/mcp/resolver.py`, add this helper at the top of the module (after imports, before `resolve_mcp_tools`):

```python
def _is_method_not_found(exc: BaseException) -> bool:
    """Return True if `exc` is an McpError signaling JSON-RPC method-not-found.

    Used by Phase 12's resolve_mcp_resources and by BaseAgent's prompt API to
    silently skip servers that don't advertise the resources/prompts methods.

    Centralized here so a single line changes if FastMCP's exception shape evolves.
    """
    from mcp.shared.exceptions import McpError  # noqa: PLC0415 — defer FastMCP import
    return isinstance(exc, McpError) and exc.error.code == -32601
```

- [ ] **Step 5: Implement `_build_mcp_resource_spec` and `resolve_mcp_resources`**

In `src/ai_core/mcp/resolver.py`, add at the end of the module (before `__all__`):

```python
async def resolve_mcp_resources(
    servers: Sequence[MCPServerSpec],
    factory: IMCPConnectionFactory,
) -> list[MCPResourceSpec]:
    """Discover resources on each server and return them as MCPResourceSpec instances.

    Each resource becomes a parameter-less read-only tool. Handler closures capture
    the URI and dispatch via `client.read_resource(uri)`. Same conflict-detection
    semantics as `resolve_mcp_tools` — duplicate names across servers raise
    RegistryError.

    Servers that don't expose resources (list_resources raises McpError with
    code -32601) are silently skipped. Other errors propagate.

    Args:
        servers: MCP server specs the agent declared via `mcp_servers()`.
        factory: Connection factory (typically the DI-bound `PoolingMCPConnectionFactory`).

    Returns:
        One `MCPResourceSpec` per discovered resource. Empty when no servers
        expose any.

    Raises:
        MCPTransportError: When a server is unreachable.
        RegistryError: When two servers expose resources with the same name.
        McpError: For any non-method-not-found protocol error.
    """
    seen_names: set[str] = set()
    out: list[MCPResourceSpec] = []
    for spec in servers:
        async with factory.open(spec) as client:
            try:
                resources = await client.list_resources()
            except Exception as exc:  # noqa: BLE001 — we re-raise unless it's method-not-found
                if _is_method_not_found(exc):
                    continue
                raise
        for fastmcp_resource in resources:
            name = fastmcp_resource.name
            if name in seen_names:
                raise RegistryError(
                    f"MCP resource name {name!r} appears in multiple servers",
                    details={"name": name, "server": spec.component_id},
                )
            seen_names.add(name)
            out.append(_build_mcp_resource_spec(spec, fastmcp_resource, factory))
    return out


def _build_mcp_resource_spec(
    server: MCPServerSpec,
    fastmcp_resource: Any,  # noqa: ANN401 — FastMCP resource shape is duck-typed
    factory: IMCPConnectionFactory,
) -> MCPResourceSpec:
    """Construct an MCPResourceSpec wrapping a closure handler that reads the resource.

    The closure captures `uri`, `component_id`, `server`, and `factory` as locals
    of THIS function — the standard factory-function pattern that prevents
    Python's late-binding closure footgun.

    Args:
        server: The MCP server this resource was discovered on. Captured by the
            handler closure to pass to `factory.open()` on each call.
        fastmcp_resource: The FastMCP `Resource` object returned by
            `client.list_resources()` (duck-typed: only `.name`, `.description`,
            `.uri` are accessed).
        factory: The connection factory; captured by the closure to open a fresh
            pooled connection per invocation.

    Returns:
        A frozen `MCPResourceSpec` ready to register with `ToolInvoker`.
    """
    name = fastmcp_resource.name
    description = fastmcp_resource.description or ""
    uri = str(fastmcp_resource.uri)
    component_id = server.component_id

    async def _handler(payload: _MCPPassthroughInput) -> _MCPPassthroughOutput:
        # payload is empty (resources take no args); ignored.
        async with factory.open(server) as client:
            contents = await client.read_resource(uri)
        text_parts: list[str] = []
        binary_count = 0
        for c in contents:
            if getattr(c, "text", None) is not None:
                text_parts.append(c.text)
            else:
                binary_count += 1
        if binary_count:
            _logger.warning(
                "mcp.resource.binary_suppressed",
                uri=uri, server=component_id, count=binary_count,
            )
            text_parts.append(f"<binary content suppressed: {binary_count} block(s)>")
        return _MCPPassthroughOutput(value="\n".join(text_parts))

    return MCPResourceSpec(
        name=name,
        version=1,
        description=description,
        input_model=_MCPPassthroughInput,
        output_model=_MCPPassthroughOutput,
        handler=_handler,
        opa_path=server.opa_decision_path,
        mcp_server_spec=server,
        mcp_input_schema={"type": "object", "properties": {}},
        mcp_resource_uri=uri,
    )
```

- [ ] **Step 6: Add the necessary imports + logger to `src/ai_core/mcp/resolver.py`**

At the top of the module, add `MCPResourceSpec` to the existing `from ai_core.mcp.tools import` line, and add a logger if not already present:

```python
from ai_core.mcp.tools import (
    MCPResourceSpec,
    MCPToolSpec,
    _MCPPassthroughInput,
    _MCPPassthroughOutput,
)

from ai_core.observability.logging import get_logger

_logger = get_logger(__name__)
```

(Place the logger import near the other module-level imports; place `_logger = ...` near the top of module-level code.)

- [ ] **Step 7: Update `__all__` in `src/ai_core/mcp/resolver.py`**

Replace:
```python
__all__ = ["resolve_mcp_tools"]
```

with:

```python
__all__ = ["resolve_mcp_resources", "resolve_mcp_tools"]
```

- [ ] **Step 8: Update `src/ai_core/mcp/__init__.py` to re-export**

Update the resolver import line:

```python
from ai_core.mcp.resolver import resolve_mcp_resources, resolve_mcp_tools
```

Add `"resolve_mcp_resources"` to `__all__` in alphabetical order (it comes before `resolve_mcp_tools`).

- [ ] **Step 9: Run tests**

Run: `.venv/bin/pytest tests/unit/mcp/test_resolver.py -v -k 'resource or method_not_found'`

Expected: 10 PASSED.

- [ ] **Step 10: Run the full mcp suite — no regressions**

Run: `.venv/bin/pytest tests/unit/mcp/ -v`

Expected: all green.

- [ ] **Step 11: Lint**

Run: `.venv/bin/ruff check src/ai_core/mcp/resolver.py tests/unit/mcp/test_resolver.py src/ai_core/mcp/__init__.py`

Expected: no new errors.

- [ ] **Step 12: Commit**

```bash
git add src/ai_core/mcp/resolver.py src/ai_core/mcp/__init__.py tests/unit/mcp/test_resolver.py
git commit -m "feat(mcp): resolve_mcp_resources — async resource-to-MCPResourceSpec resolver

Parallel to resolve_mcp_tools. For each declared MCPServerSpec:
opens a pooled connection, calls list_resources(), synthesizes one
MCPResourceSpec per advertised resource. Each spec carries a closure
handler that reads the resource on call and maps text/binary contents.

Servers that don't expose resources (McpError -32601 method-not-found)
are silently skipped. Centralized _is_method_not_found predicate is
used by both resolve_mcp_resources and BaseAgent's prompt API."
```

---

## Task 4: `BaseAgent._all_tools()` extension to merge resources

**Why:** Now that `resolve_mcp_resources` exists, `BaseAgent._all_tools()` needs to call both resolvers, merge the results, run conflict detection across the merged set, and register everything with `ToolInvoker`. The local-vs-MCP conflict check now distinguishes "resource" vs "tool" in the error message.

**Files:**
- Modify: `src/ai_core/agents/base.py` (extend `_all_tools()`)
- Test: `tests/unit/agents/test_base_mcp.py` (extend)

- [ ] **Step 1: Write the failing tests**

In `tests/unit/agents/test_base_mcp.py`, add at the end (after the existing `_AgentMCPOnly` fixtures):

```python
# ---------------------------------------------------------------------------
# _all_tools() resource merging (Phase 12)
# ---------------------------------------------------------------------------
from ai_core.mcp.tools import MCPResourceSpec


@dataclass(frozen=True)
class _FakeFastMCPResource:
    name: str
    description: str | None
    uri: str


class _FakeMCPClientWithResources:
    """Fake client supporting list_tools, list_resources, read_resource, call_tool."""

    def __init__(
        self,
        tools: list[_FakeFastMCPTool] | None = None,
        resources: list[_FakeFastMCPResource] | None = None,
    ) -> None:
        self._tools = tools or []
        self._resources = resources or []

    async def list_tools(self) -> list[_FakeFastMCPTool]:
        return list(self._tools)

    async def list_resources(self) -> list[_FakeFastMCPResource]:
        return list(self._resources)

    async def read_resource(self, uri: str) -> list:
        from dataclasses import dataclass
        @dataclass
        class _Text:
            text: str
            blob: bytes | None = None
        return [_Text(text=f"contents of {uri}")]

    async def call_tool(self, name: str, args: dict[str, Any], **_: Any):
        return _FakeCallToolResult(is_error=False, data=f"called {name}", content=[])


class _FakeFactoryWithResources:
    def __init__(self, clients: dict[str, _FakeMCPClientWithResources]) -> None:
        self._clients = clients
        self.open_count = 0

    def open(self, spec: MCPServerSpec):
        self.open_count += 1

        @asynccontextmanager
        async def _cm():
            yield self._clients[spec.component_id]

        return _cm()


async def test_all_tools_merges_resources_with_tools() -> None:
    """_all_tools() returns local + MCP tools + MCP resources."""
    factory = _FakeFactoryWithResources({
        "svc": _FakeMCPClientWithResources(
            tools=[_FakeFastMCPTool("remote_tool", "tool desc", {})],
            resources=[_FakeFastMCPResource("docs", "Project docs", "mcp://docs")],
        ),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    all_tools = await agent._all_tools()

    names = sorted(t.name for t in all_tools)
    assert names == ["docs", "local_echo", "remote_tool"]


async def test_all_tools_resource_is_mcp_resource_spec() -> None:
    """The resolved resource is an MCPResourceSpec, registered with the invoker."""
    factory = _FakeFactoryWithResources({
        "svc": _FakeMCPClientWithResources(
            resources=[_FakeFastMCPResource("docs", "d", "mcp://docs")],
        ),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    all_tools = await agent._all_tools()

    resource_specs = [t for t in all_tools if isinstance(t, MCPResourceSpec)]
    assert len(resource_specs) == 1
    assert resource_specs[0].name == "docs"
    assert resource_specs[0].mcp_resource_uri == "mcp://docs"


async def test_resource_vs_local_tool_conflict_raises_with_kind() -> None:
    """An MCP resource named the same as a local tool → RegistryError saying 'resource'."""
    factory = _FakeFactoryWithResources({
        "svc": _FakeMCPClientWithResources(
            resources=[_FakeFastMCPResource("local_echo", "d", "mcp://x")],
        ),
    })
    agent = _build_agent(_AgentWithMCP, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    with pytest.raises(RegistryError) as excinfo:
        await agent._all_tools()

    assert "local_echo" in str(excinfo.value)
    assert "resource" in str(excinfo.value)


async def test_tool_and_resource_with_same_name_conflict_raises() -> None:
    """An MCP tool and an MCP resource sharing a name → RegistryError."""
    factory = _FakeFactoryWithResources({
        "svc": _FakeMCPClientWithResources(
            tools=[_FakeFastMCPTool("ambiguous", "t", {})],
            resources=[_FakeFastMCPResource("ambiguous", "r", "mcp://x")],
        ),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (
        MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true"),
    )

    with pytest.raises(RegistryError) as excinfo:
        await agent._all_tools()

    assert "ambiguous" in str(excinfo.value)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/pytest tests/unit/agents/test_base_mcp.py -v -k 'resource or merge or ambiguous'`

Expected: FAIL — `_all_tools()` currently only resolves tools; resources aren't merged in.

- [ ] **Step 3: Update `src/ai_core/agents/base.py`'s `_all_tools()` method**

Find the `_all_tools` method in `src/ai_core/agents/base.py` (around line 398-428 based on Phase 11). Replace the entire method body with:

```python
    async def _all_tools(self) -> list[Tool | Mapping[str, Any]]:
        """Return the merged list of local + resolved MCP tools + resources.

        Lazily resolves MCP servers on the first call; caches per-instance.
        Concurrent first-turn callers serialize on `_mcp_resolution_lock`.

        Raises:
            MCPTransportError: When a declared MCP server is unreachable.
            RegistryError: When MCP names conflict with each other or with
                local @tool names. Conflicts span tools-vs-tools, tools-vs-resources,
                and any of those vs local @tools.
        """
        if self._mcp_resolved is None:
            async with self._mcp_resolution_lock:
                if self._mcp_resolved is None:
                    servers = list(self.mcp_servers())
                    if servers:
                        tools_resolved = await resolve_mcp_tools(servers, self._mcp_factory)
                        resources_resolved = await resolve_mcp_resources(servers, self._mcp_factory)
                        resolved: list[MCPToolSpec] = list(tools_resolved) + list(resources_resolved)
                    else:
                        resolved = []
                    local_names = {
                        t.name for t in self.tools() if isinstance(t, ToolSpec)
                    }
                    mcp_names_seen: set[str] = set()
                    for mcp_spec in resolved:
                        if mcp_spec.name in local_names:
                            kind = "resource" if isinstance(mcp_spec, MCPResourceSpec) else "tool"
                            raise RegistryError(
                                f"MCP {kind} name {mcp_spec.name!r} conflicts with a local tool",
                                details={"tool": mcp_spec.name},
                            )
                        if mcp_spec.name in mcp_names_seen:
                            raise RegistryError(
                                f"MCP name {mcp_spec.name!r} appears in both tools and resources "
                                f"on declared servers",
                                details={"name": mcp_spec.name},
                            )
                        mcp_names_seen.add(mcp_spec.name)
                        self._tool_invoker.register(mcp_spec)
                    self._mcp_resolved = resolved
        return list(self.tools()) + list(self._mcp_resolved)
```

- [ ] **Step 4: Update imports in `src/ai_core/agents/base.py`**

Find the imports block. Update:

```python
from ai_core.mcp.resolver import resolve_mcp_tools
from ai_core.mcp.tools import MCPToolSpec
```

to:

```python
from ai_core.mcp.resolver import resolve_mcp_resources, resolve_mcp_tools
from ai_core.mcp.tools import MCPResourceSpec, MCPToolSpec
```

- [ ] **Step 5: Run the new tests**

Run: `.venv/bin/pytest tests/unit/agents/test_base_mcp.py -v -k 'resource or merge or ambiguous'`

Expected: 4 PASSED.

- [ ] **Step 6: Run the broader agent + mcp suites — no regressions**

Run: `.venv/bin/pytest tests/unit/agents/ tests/unit/mcp/ tests/unit/tools/ -q`

Expected: all green.

- [ ] **Step 7: Lint**

Run: `.venv/bin/ruff check src/ai_core/agents/base.py tests/unit/agents/test_base_mcp.py`

Expected: no new errors.

- [ ] **Step 8: Commit**

```bash
git add src/ai_core/agents/base.py tests/unit/agents/test_base_mcp.py
git commit -m "feat(agents): _all_tools() merges MCP resources alongside tools

Calls resolve_mcp_resources after resolve_mcp_tools, merges into a
single list, and runs the conflict check across the merged name set:
local @tool vs MCP tool, local @tool vs MCP resource, MCP tool vs
MCP resource (within or across servers). Conflict messages distinguish
'tool' vs 'resource' so the offending side is clear."
```

---

## Task 5: Prompt types + `BaseAgent.list_prompts()` + `BaseAgent.get_prompt()`

**Why:** Application-invoked prompt fetching. Three new dataclasses (frozen) live in their own module; two new async methods on `BaseAgent` use them.

**Files:**
- Create: `src/ai_core/mcp/prompts.py` (new types)
- Test: `tests/unit/mcp/test_prompts.py` (new)
- Modify: `src/ai_core/mcp/__init__.py` (export prompt types)
- Modify: `src/ai_core/agents/base.py` (add `list_prompts` + `get_prompt`)
- Modify: `tests/unit/agents/test_base_mcp.py` (extend)

- [ ] **Step 1: Write tests for the new types in `tests/unit/mcp/test_prompts.py`**

Create `tests/unit/mcp/test_prompts.py`:

```python
"""Unit tests for MCP prompt types (Phase 12)."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ai_core.mcp import MCPServerSpec
from ai_core.mcp.prompts import (
    MCPPrompt,
    MCPPromptArgument,
    MCPPromptMessage,
)

pytestmark = pytest.mark.unit


def _server() -> MCPServerSpec:
    return MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")


def test_prompt_argument_is_frozen() -> None:
    arg = MCPPromptArgument(name="text", description="The text", required=True)
    with pytest.raises(FrozenInstanceError):
        arg.required = False  # type: ignore[misc]


def test_prompt_is_frozen_with_arguments_tuple() -> None:
    """MCPPrompt holds arguments as a tuple (frozen, hashable)."""
    args = (
        MCPPromptArgument(name="text", description="Input", required=True),
        MCPPromptArgument(name="locale", description=None, required=False),
    )
    prompt = MCPPrompt(
        name="summarize",
        description="Summarize the text",
        arguments=args,
        mcp_server_spec=_server(),
    )

    assert prompt.name == "summarize"
    assert prompt.arguments[0].name == "text"
    assert prompt.arguments[1].required is False
    with pytest.raises(FrozenInstanceError):
        prompt.name = "different"  # type: ignore[misc]


def test_prompt_message_role_and_content() -> None:
    msg = MCPPromptMessage(role="user", content="Hello, world.")
    assert msg.role == "user"
    assert msg.content == "Hello, world."
    with pytest.raises(FrozenInstanceError):
        msg.content = "different"  # type: ignore[misc]


def test_prompt_with_no_arguments() -> None:
    """A prompt with no arguments uses an empty tuple."""
    prompt = MCPPrompt(
        name="ping",
        description=None,
        arguments=(),
        mcp_server_spec=_server(),
    )
    assert prompt.arguments == ()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/pytest tests/unit/mcp/test_prompts.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ai_core.mcp.prompts'`.

- [ ] **Step 3: Implement `src/ai_core/mcp/prompts.py`**

Create `src/ai_core/mcp/prompts.py`:

```python
"""MCP prompt types — used by BaseAgent.list_prompts() / get_prompt().

Phase 12 exposes MCP prompts as application-invoked helpers (not LLM-callable
tools). The application fetches a prompt by name, splices the resulting
messages into ainvoke(messages=...), and runs the agent.

All types are frozen dataclasses:
- MCPPromptArgument: one argument declaration on a prompt template.
- MCPPrompt: a prompt template with its origin server tagged.
- MCPPromptMessage: one message after argument substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_core.mcp.transports import MCPServerSpec


@dataclass(frozen=True, slots=True)
class MCPPromptArgument:
    """One argument declaration on an MCP prompt template.

    Mirrors mcp.types.PromptArgument but kept SDK-internal so the public
    API doesn't depend on the FastMCP/mcp Python types directly.

    Attributes:
        name: The argument's name (used as a key when calling get_prompt).
        description: Human-readable description (may be None).
        required: Whether the argument must be provided.
    """

    name: str
    description: str | None
    required: bool


@dataclass(frozen=True, slots=True)
class MCPPrompt:
    """A prompt template advertised by an MCP server.

    Attributes:
        name: The prompt's name; unique across declared servers (a conflict
            during list_prompts raises RegistryError).
        description: Human-readable description (may be None).
        arguments: Tuple of MCPPromptArgument declarations (may be empty).
        mcp_server_spec: Which server this prompt came from. Useful for
            passing back to get_prompt(name, args, server=spec.component_id)
            to skip the cross-server search.
    """

    name: str
    description: str | None
    arguments: tuple[MCPPromptArgument, ...]
    mcp_server_spec: "MCPServerSpec"


@dataclass(frozen=True, slots=True)
class MCPPromptMessage:
    """One message from a fetched prompt template (after argument substitution).

    Attributes:
        role: Either "user" or "assistant" (matches MCP protocol values).
        content: Text content. Binary content blocks (images, etc.) from the
            FastMCP message are dropped in v1; only TextContent.text is preserved.
    """

    role: str
    content: str


__all__ = ["MCPPrompt", "MCPPromptArgument", "MCPPromptMessage"]
```

- [ ] **Step 4: Update `src/ai_core/mcp/__init__.py` to export prompt types**

Add to the imports section:

```python
from ai_core.mcp.prompts import MCPPrompt, MCPPromptArgument, MCPPromptMessage
```

Update `__all__` to include the three new names in alphabetical order. Final shape (full list, alphabetized):

```python
__all__ = [
    "ComponentRegistry",
    "FastMCPConnectionFactory",
    "IMCPConnectionFactory",
    "MCPPrompt",
    "MCPPromptArgument",
    "MCPPromptMessage",
    "MCPResourceSpec",
    "MCPServerSpec",
    "MCPToolSpec",
    "MCPTransport",
    "RegisteredComponent",
    "resolve_mcp_resources",
    "resolve_mcp_tools",
    "unwrap_mcp_tool_message",
]
```

- [ ] **Step 5: Run prompt-type tests**

Run: `.venv/bin/pytest tests/unit/mcp/test_prompts.py -v`

Expected: 4 PASSED.

- [ ] **Step 6: Write failing tests for `list_prompts()` and `get_prompt()` in `tests/unit/agents/test_base_mcp.py`**

Add at the end of the file:

```python
# ---------------------------------------------------------------------------
# list_prompts() / get_prompt() tests (Phase 12)
# ---------------------------------------------------------------------------
from ai_core.mcp.prompts import MCPPrompt, MCPPromptArgument, MCPPromptMessage


@dataclass(frozen=True)
class _FakeFastMCPPromptArg:
    name: str
    description: str | None
    required: bool


@dataclass(frozen=True)
class _FakeFastMCPPrompt:
    name: str
    description: str | None
    arguments: list[_FakeFastMCPPromptArg]


@dataclass(frozen=True)
class _FakeTextContent:
    """Stand-in for mcp.types.TextContent."""
    text: str
    type: str = "text"


@dataclass(frozen=True)
class _FakeFastMCPPromptMessage:
    role: str
    content: object  # may be _FakeTextContent or other


@dataclass(frozen=True)
class _FakeGetPromptResult:
    messages: list[_FakeFastMCPPromptMessage]


class _FakeMCPClientWithPrompts:
    """Fake client supporting list_prompts + get_prompt + the tool/resource methods."""

    def __init__(
        self,
        prompts: list[_FakeFastMCPPrompt] | None = None,
        get_prompt_results: dict[str, _FakeGetPromptResult] | None = None,
        list_prompts_raises: BaseException | None = None,
    ) -> None:
        self._prompts = prompts or []
        self._results = get_prompt_results or {}
        self._list_raises = list_prompts_raises
        self.get_prompt_invocations: list[tuple[str, dict]] = []

    async def list_prompts(self) -> list[_FakeFastMCPPrompt]:
        if self._list_raises is not None:
            raise self._list_raises
        return list(self._prompts)

    async def get_prompt(self, name: str, arguments: dict) -> _FakeGetPromptResult:
        self.get_prompt_invocations.append((name, dict(arguments)))
        if name in self._results:
            return self._results[name]
        return _FakeGetPromptResult(messages=[
            _FakeFastMCPPromptMessage(
                role="user",
                content=_FakeTextContent(text=f"prompt {name}"),
            ),
        ])

    # Methods for tool/resource compatibility (not exercised by these tests):
    async def list_tools(self) -> list:
        return []

    async def list_resources(self) -> list:
        return []


class _FakePromptFactory:
    def __init__(self, clients: dict[str, _FakeMCPClientWithPrompts]) -> None:
        self._clients = clients

    def open(self, spec: MCPServerSpec):
        @asynccontextmanager
        async def _cm():
            yield self._clients[spec.component_id]

        return _cm()


# ---- list_prompts ----

async def test_list_prompts_returns_typed_mcp_prompts() -> None:
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    factory = _FakePromptFactory({
        "svc": _FakeMCPClientWithPrompts(prompts=[
            _FakeFastMCPPrompt(
                name="summarize",
                description="Summarize text",
                arguments=[
                    _FakeFastMCPPromptArg(name="text", description="Input", required=True),
                ],
            ),
        ]),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    prompts = await agent.list_prompts()

    assert len(prompts) == 1
    assert isinstance(prompts[0], MCPPrompt)
    assert prompts[0].name == "summarize"
    assert prompts[0].description == "Summarize text"
    assert prompts[0].arguments == (
        MCPPromptArgument(name="text", description="Input", required=True),
    )
    assert prompts[0].mcp_server_spec.component_id == "svc"


async def test_list_prompts_silently_skips_method_not_found() -> None:
    """Servers that don't expose prompts → empty result for that server."""
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    factory = _FakePromptFactory({
        "svc": _FakeMCPClientWithPrompts(
            prompts=[],
            list_prompts_raises=McpError(ErrorData(code=-32601, message="Method not found")),
        ),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    prompts = await agent.list_prompts()

    assert prompts == []


async def test_list_prompts_cross_server_conflict_raises() -> None:
    """Two servers exposing same prompt name → RegistryError."""
    server_a = MCPServerSpec(component_id="a", transport="stdio", target="/bin/true")
    server_b = MCPServerSpec(component_id="b", transport="stdio", target="/bin/true")
    factory = _FakePromptFactory({
        "a": _FakeMCPClientWithPrompts(prompts=[
            _FakeFastMCPPrompt(name="summarize", description=None, arguments=[]),
        ]),
        "b": _FakeMCPClientWithPrompts(prompts=[
            _FakeFastMCPPrompt(name="summarize", description=None, arguments=[]),
        ]),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server_a, server_b)

    with pytest.raises(RegistryError) as excinfo:
        await agent.list_prompts()

    assert "summarize" in str(excinfo.value)


# ---- get_prompt ----

async def test_get_prompt_returns_typed_messages() -> None:
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    client = _FakeMCPClientWithPrompts(
        prompts=[_FakeFastMCPPrompt(name="summarize", description=None, arguments=[])],
        get_prompt_results={
            "summarize": _FakeGetPromptResult(messages=[
                _FakeFastMCPPromptMessage(
                    role="user",
                    content=_FakeTextContent(text="Please summarize:"),
                ),
                _FakeFastMCPPromptMessage(
                    role="assistant",
                    content=_FakeTextContent(text="Sure, what's the input?"),
                ),
            ]),
        },
    )
    factory = _FakePromptFactory({"svc": client})
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    messages = await agent.get_prompt("summarize", {"text": "hello"})

    assert messages == [
        MCPPromptMessage(role="user", content="Please summarize:"),
        MCPPromptMessage(role="assistant", content="Sure, what's the input?"),
    ]
    assert client.get_prompt_invocations == [("summarize", {"text": "hello"})]


async def test_get_prompt_with_explicit_server_skips_search() -> None:
    """When server= is provided, only that server is queried."""
    server_a = MCPServerSpec(component_id="a", transport="stdio", target="/bin/true")
    server_b = MCPServerSpec(component_id="b", transport="stdio", target="/bin/true")

    client_a = _FakeMCPClientWithPrompts(prompts=[
        _FakeFastMCPPrompt(name="x", description=None, arguments=[]),
    ])
    client_b = _FakeMCPClientWithPrompts(prompts=[
        _FakeFastMCPPrompt(name="x", description=None, arguments=[]),
    ])
    factory = _FakePromptFactory({"a": client_a, "b": client_b})
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server_a, server_b)

    await agent.get_prompt("x", server="b")

    assert client_a.get_prompt_invocations == []
    assert len(client_b.get_prompt_invocations) == 1


async def test_get_prompt_not_found_raises_registry_error() -> None:
    """If no server has the prompt, RegistryError with hint."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")
    factory = _FakePromptFactory({
        "svc": _FakeMCPClientWithPrompts(prompts=[
            _FakeFastMCPPrompt(name="other", description=None, arguments=[]),
        ]),
    })
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    with pytest.raises(RegistryError) as excinfo:
        await agent.get_prompt("missing")

    assert "missing" in str(excinfo.value)


async def test_get_prompt_drops_non_text_content() -> None:
    """Non-TextContent message bodies become empty content (binary suppressed)."""
    server = MCPServerSpec(component_id="svc", transport="stdio", target="/bin/true")

    @dataclass(frozen=True)
    class _FakeImage:
        type: str = "image"

    client = _FakeMCPClientWithPrompts(
        prompts=[_FakeFastMCPPrompt(name="x", description=None, arguments=[])],
        get_prompt_results={
            "x": _FakeGetPromptResult(messages=[
                _FakeFastMCPPromptMessage(role="user", content=_FakeImage()),
                _FakeFastMCPPromptMessage(role="assistant", content=_FakeTextContent(text="text reply")),
            ]),
        },
    )
    factory = _FakePromptFactory({"svc": client})
    agent = _build_agent(_AgentMCPOnly, factory)
    agent._mcp_servers = (server,)

    messages = await agent.get_prompt("x")

    assert messages[0].content == ""  # image suppressed
    assert messages[1].content == "text reply"
```

- [ ] **Step 7: Run tests to verify failure**

Run: `.venv/bin/pytest tests/unit/agents/test_base_mcp.py -v -k 'prompt'`

Expected: FAIL — `agent.list_prompts` and `agent.get_prompt` don't exist yet.

- [ ] **Step 8: Implement `list_prompts` and `get_prompt` on `BaseAgent`**

In `src/ai_core/agents/base.py`, add the two methods. Place them after `_all_tools()` (or anywhere logical in the public API section).

First update imports:

```python
from ai_core.mcp.prompts import MCPPrompt, MCPPromptArgument, MCPPromptMessage
from ai_core.mcp.resolver import (
    _is_method_not_found,
    resolve_mcp_resources,
    resolve_mcp_tools,
)
```

(Adjust based on your existing import order. Note `_is_method_not_found` is private to `resolver.py` but used here — since both modules ship as part of the SDK, this internal coupling is acceptable. Alternatively, expose it via the resolver module's public surface or move to a shared `_internal.py` — but for v1 the leading underscore + cross-module use mirrors the existing pattern of `_MCPPassthroughInput`/`_MCPPassthroughOutput` in `tools.py`.)

Add the methods:

```python
    async def list_prompts(self) -> list[MCPPrompt]:
        """List all prompts across declared MCP servers.

        Fetched fresh each call (no cache — application-invoked patterns vary,
        consistency matters more than throughput).

        Returns:
            One MCPPrompt per discovered prompt, with its origin server tagged.

        Raises:
            RegistryError: When two servers expose prompts with the same name.
            MCPTransportError: Propagated from the connection factory.
        """
        out: list[MCPPrompt] = []
        seen_names: set[str] = set()
        for server in self.mcp_servers():
            async with self._mcp_factory.open(server) as client:
                try:
                    prompts = await client.list_prompts()
                except Exception as exc:  # noqa: BLE001 — narrow via predicate
                    if _is_method_not_found(exc):
                        continue
                    raise
            for p in prompts:
                if p.name in seen_names:
                    raise RegistryError(
                        f"MCP prompt name {p.name!r} appears in multiple servers",
                        details={"name": p.name, "server": server.component_id},
                    )
                seen_names.add(p.name)
                out.append(_to_mcp_prompt(p, server))
        return out

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        server: str | None = None,
    ) -> list[MCPPromptMessage]:
        """Fetch a templated prompt's messages by name.

        Args:
            name: The prompt's name (must match what `list_prompts()` returned).
            arguments: Argument dict to substitute into the template.
            server: Optional component_id of the server known to host the prompt.
                When omitted, the agent searches across declared servers (one
                list_prompts call each until found).

        Returns:
            List of MCPPromptMessage instances, ready to splice into
            ainvoke(messages=...).

        Raises:
            RegistryError: When the prompt is not found on any declared server.
            MCPTransportError: Propagated from the connection factory.
        """
        for srv in self.mcp_servers():
            if server is not None and srv.component_id != server:
                continue
            async with self._mcp_factory.open(srv) as client:
                try:
                    prompts = await client.list_prompts()
                except Exception as exc:  # noqa: BLE001 — narrow via predicate
                    if _is_method_not_found(exc):
                        continue
                    raise
                if not any(p.name == name for p in prompts):
                    continue
                result = await client.get_prompt(name, dict(arguments or {}))
            return [_to_mcp_prompt_message(m) for m in result.messages]
        raise RegistryError(
            f"MCP prompt {name!r} not found in any declared server",
            details={"name": name, "hint_server": server},
        )
```

- [ ] **Step 9: Add the two `_to_mcp_prompt*` mapping helpers**

At the bottom of `src/ai_core/agents/base.py` (or in a clearly-marked private-helpers section), add:

```python
def _to_mcp_prompt(fastmcp_prompt: Any, server: MCPServerSpec) -> MCPPrompt:
    """Map a FastMCP `Prompt` to our typed MCPPrompt."""
    args = tuple(
        MCPPromptArgument(
            name=a.name,
            description=getattr(a, "description", None),
            required=getattr(a, "required", False),
        )
        for a in (getattr(fastmcp_prompt, "arguments", None) or [])
    )
    return MCPPrompt(
        name=fastmcp_prompt.name,
        description=getattr(fastmcp_prompt, "description", None),
        arguments=args,
        mcp_server_spec=server,
    )


def _to_mcp_prompt_message(fastmcp_msg: Any) -> MCPPromptMessage:
    """Map a FastMCP `PromptMessage` to our typed MCPPromptMessage.

    PromptMessage.content is a union of TextContent | ImageContent | … .
    v1 only handles TextContent; other types yield empty content.
    """
    content_obj = getattr(fastmcp_msg, "content", None)
    text = ""
    if content_obj is not None and getattr(content_obj, "type", None) == "text":
        text = getattr(content_obj, "text", "") or ""
    return MCPPromptMessage(role=str(fastmcp_msg.role), content=text)
```

- [ ] **Step 10: Run prompt method tests**

Run: `.venv/bin/pytest tests/unit/agents/test_base_mcp.py -v -k prompt`

Expected: 7 PASSED.

- [ ] **Step 11: Run the broader unit suite**

Run: `.venv/bin/pytest tests/unit/ -q`

Expected: all green.

- [ ] **Step 12: Lint**

Run: `.venv/bin/ruff check src/ai_core/mcp/prompts.py src/ai_core/agents/base.py tests/unit/mcp/test_prompts.py tests/unit/agents/test_base_mcp.py src/ai_core/mcp/__init__.py`

Expected: no new errors.

- [ ] **Step 13: Commit**

```bash
git add src/ai_core/mcp/prompts.py src/ai_core/mcp/__init__.py src/ai_core/agents/base.py tests/unit/mcp/test_prompts.py tests/unit/agents/test_base_mcp.py
git commit -m "feat(agents): list_prompts() + get_prompt() async helpers

MCP prompts exposed as application-invoked async helpers on BaseAgent —
not LLM-callable, no audit, no OPA. New types in ai_core.mcp.prompts:
MCPPrompt, MCPPromptArgument, MCPPromptMessage (frozen dataclasses).

list_prompts() merges across declared servers with cross-server name
conflict detection. get_prompt(name, args, *, server=None) finds the
prompt on the first matching server (or only the named server if
server= is given) and returns typed messages ready to splice into
ainvoke(messages=...).

Method-not-found protocol errors (-32601) skip silently; other errors
propagate. Centralized _is_method_not_found predicate lives in
resolver.py and is re-used here."
```

---

## Task 6: Integration tests + examples + smoke gate

**Why:** Phase 11's pattern: end-to-end test against a real FastMCP subprocess + a runnable example + smoke gate hookup. This task closes the loop and ships the user-facing demo.

**Files:**
- Modify: `examples/mcp_server_demo/server.py` (add resource + prompt)
- Modify: `examples/mcp_server_demo/agent_demo.py` (call resource; use unwrap helper)
- Create: `examples/mcp_server_demo/prompt_demo.py` (app-invoked prompt → ainvoke)
- Modify: `examples/mcp_server_demo/README.md` (document the new demos)
- Modify: `tests/integration/mcp/test_agent_uses_mcp.py` (add resource read test)
- Create: `tests/integration/mcp/test_prompts.py` (end-to-end prompt fetch)
- Modify: `scripts/run_examples.sh` (add `mcp_prompt_demo`)

- [ ] **Step 1: Extend `examples/mcp_server_demo/server.py`**

Read the current file first to understand the existing structure. Then add the resource and prompt declarations after the existing `@mcp.tool()` decorators:

```python
@mcp.resource("mcp-demo://documentation")
def documentation() -> str:
    """Project documentation — exposed as an MCP resource."""
    return (
        "This is the demo MCP server's documentation.\n\n"
        "It exposes:\n"
        "  - echo(text): repeat a string\n"
        "  - current_time(): UTC ISO-8601 timestamp\n"
        "  - documentation: this resource\n"
        "  - summarize_text(text): a prompt template"
    )


@mcp.prompt()
def summarize_text(text: str) -> str:
    """Generate a summarization prompt for a given text."""
    return f"Please summarize the following text in one sentence:\n\n{text}"
```

- [ ] **Step 2: Verify the server starts cleanly with the new declarations**

Run: `.venv/bin/python examples/mcp_server_demo/server.py &` (background) — give it a moment, then `kill %1`. Or quickly probe it via the existing run_client.py (which already lists tools but doesn't list resources/prompts — that's fine for now).

Alternatively run a one-shot probe:

```bash
.venv/bin/python -c "
import asyncio, sys
from pathlib import Path
from ai_core.mcp import MCPServerSpec, FastMCPConnectionFactory

async def main():
    factory = FastMCPConnectionFactory(pool_enabled=False)
    spec = MCPServerSpec(
        component_id='probe',
        transport='stdio',
        target=sys.executable,
        args=(str(Path('examples/mcp_server_demo/server.py').resolve()),),
    )
    async with factory.open(spec) as client:
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        print('resources:', [r.name for r in resources])
        print('prompts:', [p.name for p in prompts])

asyncio.run(main())
"
```

Expected: `resources: ['documentation']`, `prompts: ['summarize_text']`.

- [ ] **Step 3: Update `examples/mcp_server_demo/agent_demo.py`**

Read the current file. Find the ScriptedLLM construction in `main()`. Replace it so the LLM emits a tool call to `documentation` (the resource we just added) instead of (or in addition to) `echo`:

Replace:

```python
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"text": "hello from the SDK"}',
                    },
                },
            ],
        ),
        make_llm_response("Echoed successfully."),
    ])
```

with:

```python
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "documentation",
                        "arguments": "{}",
                    },
                },
            ],
        ),
        make_llm_response("Read the docs."),
    ])
```

Update the system prompt accordingly:

```python
    def system_prompt(self) -> str:
        return "Read the documentation resource to learn what this server provides."
```

The existing `_render_tool_message` helper already uses `unwrap_mcp_tool_message` — but if it's still inlining the unwrap logic, replace that body with:

```python
def _render_tool_message(msg: object) -> str:
    """Render a ToolMessage's content using the SDK's unwrap helper."""
    from ai_core.mcp import unwrap_mcp_tool_message  # noqa: PLC0415
    raw = str(getattr(msg, "content", ""))
    unwrapped = unwrap_mcp_tool_message(raw)
    return unwrapped if isinstance(unwrapped, str) else str(unwrapped)
```

(If the file already inlines the equivalent logic from the Phase 11 fix-up, replace it with this `unwrap_mcp_tool_message`-based form to demonstrate the new helper.)

- [ ] **Step 4: Verify the updated agent_demo runs end-to-end**

Run: `.venv/bin/python examples/mcp_server_demo/agent_demo.py`

Expected: panel prints the documentation text (multi-line, starting with "This is the demo MCP server's documentation.").

- [ ] **Step 5: Create `examples/mcp_server_demo/prompt_demo.py`**

Create the new file:

```python
"""Application-invoked prompt demo.

Demonstrates the second integration path Phase 12 ships: the application
fetches a prompt template from the MCP server, splices the resulting
messages into ainvoke(messages=...), and runs the agent.

Run from the repo root::

    uv run python examples/mcp_server_demo/prompt_demo.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from injector import Module, provider, singleton
from rich.console import Console
from rich.panel import Panel

from ai_core.agents import BaseAgent
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.mcp import MCPServerSpec
from ai_core.testing import ScriptedLLM, make_llm_response

if TYPE_CHECKING:
    from ai_core.di.interfaces import ILLMClient  # noqa: TC001

console = Console()
SERVER_PATH = Path(__file__).parent / "server.py"


class _PromptDemoAgent(BaseAgent):
    agent_id: str = "prompt-demo-agent"

    def system_prompt(self) -> str:
        return "Respond concisely."

    def mcp_servers(self):
        return [
            MCPServerSpec(
                component_id="mcp-demo",
                transport="stdio",
                target=sys.executable,
                args=(str(SERVER_PATH),),
            ),
        ]


def build_container(llm) -> Container:  # noqa: ANN001 — ILLMClient at runtime
    settings = AppSettings(service_name="prompt-demo", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self):  # noqa: ANN201 — ILLMClient at runtime
            return llm

    return Container.build([AgentModule(settings=settings), _Overrides()])


async def main() -> None:
    # Scripted final assistant response — what the LLM would say after
    # consuming the templated prompt.
    llm = ScriptedLLM([make_llm_response("This is a one-sentence summary.")])
    container = build_container(llm)

    async with container as c:
        agent = c.get(_PromptDemoAgent)

        # Step 1: discover available prompts.
        prompts = await agent.list_prompts()
        console.print(Panel.fit(
            "\n".join(f"- {p.name}: {p.description or '(no description)'}" for p in prompts),
            title="Available prompts",
            border_style="cyan",
        ))

        # Step 2: fetch a templated prompt.
        templated_messages = await agent.get_prompt(
            "summarize_text",
            arguments={"text": "MCP is a protocol for tool/resource/prompt servers."},
        )

        # Step 3: splice the templated messages into ainvoke().
        # Convert to the openai-format dicts ainvoke expects.
        seed_messages = [
            {"role": m.role, "content": m.content}
            for m in templated_messages
        ]
        final = await agent.ainvoke(messages=seed_messages, tenant_id="demo")

    last = final["messages"][-1]
    console.print(Panel.fit(
        str(getattr(last, "content", "(empty)")),
        title="Agent response",
        border_style="green",
    ))
    console.print(
        "[bold green]Done.[/bold green] App fetched the summarize_text prompt, "
        "spliced its messages into ainvoke(), and the agent ran."
    )


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: Run the prompt demo manually**

Run: `.venv/bin/python examples/mcp_server_demo/prompt_demo.py`

Expected:
- "Available prompts" panel listing `summarize_text`.
- "Agent response" panel containing "This is a one-sentence summary."
- "Done." line.

- [ ] **Step 7: Update `examples/mcp_server_demo/README.md`**

Read the current README. Find the "What's also shown (Phase 11)" section. Add a new subsection after it:

```markdown
## What's also shown (Phase 12)

- **Resource as tool**: the `documentation` resource is exposed to the LLM
  as a parameter-less read-only tool. `agent_demo.py` shows the LLM calling
  it; the result flows through the standard `ToolInvoker` pipeline.
- **Prompt as application helper**: `prompt_demo.py` shows the application-side
  flow — `agent.list_prompts()` enumerates available prompts; `agent.get_prompt(...)`
  fetches a templated prompt's messages; the application splices them into
  `agent.ainvoke(messages=...)`.
- **`unwrap_mcp_tool_message` helper**: the `agent_demo.py` panel uses this
  to display the unwrapped tool result (no more `{"value": ...}` envelope visible).
```

In the "Run" section, add a new subsection after "Run as an agent":

```markdown
### Run the prompt-demo

```bash
uv run python examples/mcp_server_demo/prompt_demo.py
```

The application discovers the server's `summarize_text` prompt, fetches it
with sample text, and runs the agent over the templated messages.
```

- [ ] **Step 8: Add an integration test for the resource path**

In `tests/integration/mcp/test_agent_uses_mcp.py`, add a new test at the end of the file:

```python
async def test_agent_reads_documentation_resource_end_to_end() -> None:
    """LLM calls the documentation resource → resolver runs read_resource → ToolMessage carries text."""
    llm = ScriptedLLM([
        make_llm_response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "documentation",
                        "arguments": "{}",
                    },
                },
            ],
        ),
        make_llm_response("Done."),
    ])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_MCPDemoAgent)
        final = await agent.ainvoke(
            messages=[{"role": "user", "content": "tell me about this server"}],
            tenant_id="t",
        )

    msgs = final["messages"]
    tool_msgs = [m for m in msgs if getattr(m, "type", None) == "tool"]
    assert len(tool_msgs) == 1
    # The documentation text must appear inside the tool message content.
    assert "demo MCP server" in tool_msgs[0].content.lower()
    # Audit recorded the resource invocation under the resource's name.
    completed = [r for r in audit.records if r.tool_name == "documentation"]
    assert len(completed) >= 1
```

- [ ] **Step 9: Create the prompt integration test**

Create `tests/integration/mcp/test_prompts.py`:

```python
"""End-to-end: agent.list_prompts() and get_prompt() against real FastMCP server.

Spawns examples/mcp_server_demo/server.py via PoolingMCPConnectionFactory.
Exercises the application-invoked prompt path — fetches summarize_text,
splices messages, runs ainvoke().
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from injector import Module, provider, singleton

from ai_core.agents import BaseAgent
from ai_core.audit import IAuditSink
from ai_core.config.settings import AppSettings
from ai_core.di import AgentModule, Container
from ai_core.di.interfaces import ILLMClient
from ai_core.mcp import MCPPrompt, MCPServerSpec
from ai_core.testing import FakeAuditSink, ScriptedLLM, make_llm_response

pytestmark = pytest.mark.integration

pytest.importorskip("fastmcp")

DEMO_SERVER = Path(__file__).resolve().parents[3] / "examples" / "mcp_server_demo" / "server.py"


class _PromptIntegrationAgent(BaseAgent):
    agent_id = "prompt-int-agent"

    def system_prompt(self) -> str:
        return "Test agent."

    def mcp_servers(self):
        return [
            MCPServerSpec(
                component_id="demo-srv",
                transport="stdio",
                target=sys.executable,
                args=(str(DEMO_SERVER),),
            ),
        ]


def _build_container(llm: ILLMClient, audit_sink: IAuditSink) -> Container:
    settings = AppSettings(service_name="prompt-int-test", environment="local")

    class _Overrides(Module):
        @singleton
        @provider
        def provide_llm(self) -> ILLMClient:
            return llm

        @singleton
        @provider
        def provide_audit_sink(self) -> IAuditSink:
            return audit_sink

    return Container.build([AgentModule(settings=settings), _Overrides()])


async def test_list_prompts_finds_summarize_text() -> None:
    """agent.list_prompts() returns the demo server's prompts as MCPPrompt instances."""
    llm = ScriptedLLM([make_llm_response("noop")])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_PromptIntegrationAgent)
        prompts = await agent.list_prompts()

    names = [p.name for p in prompts]
    assert "summarize_text" in names
    assert all(isinstance(p, MCPPrompt) for p in prompts)


async def test_get_prompt_fetches_templated_messages() -> None:
    """agent.get_prompt('summarize_text', {'text': '...'}) returns templated messages."""
    llm = ScriptedLLM([make_llm_response("noop")])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_PromptIntegrationAgent)
        messages = await agent.get_prompt(
            "summarize_text",
            arguments={"text": "Hello, world."},
        )

    assert len(messages) >= 1
    assert any("Hello, world." in m.content for m in messages)


async def test_prompt_splice_into_ainvoke_runs_agent() -> None:
    """End-to-end: app fetches prompt → splices into ainvoke → agent responds."""
    # The agent's response after consuming the templated prompt.
    llm = ScriptedLLM([make_llm_response("This is the summary.")])
    audit = FakeAuditSink()
    container = _build_container(llm, audit)

    async with container as c:
        agent = c.get(_PromptIntegrationAgent)
        templated = await agent.get_prompt(
            "summarize_text",
            arguments={"text": "MCP is a protocol."},
        )
        seed = [{"role": m.role, "content": m.content} for m in templated]
        final = await agent.ainvoke(messages=seed, tenant_id="t")

    last = final["messages"][-1]
    assert "summary" in str(getattr(last, "content", "")).lower()
```

- [ ] **Step 10: Run the new integration tests**

Run: `.venv/bin/pytest tests/integration/mcp/ -v`

Expected: all integration tests pass (the original 2 + the 3 new prompt tests + the 1 new resource test = 6 passed).

- [ ] **Step 11: Update `scripts/run_examples.sh`**

Find the `mcp_server_demo` block. Add a new line for `mcp_prompt_demo` in both the `if` and `else` branches:

In the `if` branch:

```bash
    run_demo mcp_prompt_demo uv run python examples/mcp_server_demo/prompt_demo.py
```

In the `else` branch:

```bash
    skip_demo mcp_prompt_demo "fastmcp not installed (run \`uv sync\`)"
```

The block should now look like:

```bash
# --- mcp_server_demo: needs fastmcp installed.
if uv run python -c "import fastmcp" 2>/dev/null; then
    run_demo mcp_server_demo uv run python examples/mcp_server_demo/run_client.py
    run_demo mcp_agent_demo uv run python examples/mcp_server_demo/agent_demo.py
    run_demo mcp_prompt_demo uv run python examples/mcp_server_demo/prompt_demo.py
else
    skip_demo mcp_server_demo "fastmcp not installed (run \`uv sync\`)"
    skip_demo mcp_agent_demo "fastmcp not installed (run \`uv sync\`)"
    skip_demo mcp_prompt_demo "fastmcp not installed (run \`uv sync\`)"
fi
```

- [ ] **Step 12: Run the smoke gate**

Run: `bash scripts/run_examples.sh`

Expected: all demos run; summary shows non-zero `ran` and zero `failed`. (If `uv` isn't on PATH on the dev machine, all demos fail with exit 127 — that's the smoke gate doing its job; just confirm structural correctness via `bash -n scripts/run_examples.sh`.)

- [ ] **Step 13: Final lint sweep across everything Phase 12 touched**

Run: `.venv/bin/ruff check src/ai_core/mcp/ src/ai_core/agents/base.py tests/unit/mcp/ tests/unit/agents/test_base_mcp.py tests/integration/mcp/ examples/mcp_server_demo/`

Expected: no new errors (any pre-existing baseline violations are unchanged).

- [ ] **Step 14: Final pytest sweep**

Run: `.venv/bin/pytest -q`

Expected: all green (~545 passed = Phase 11 baseline + Phase 12 additions; 6 Docker-gated skipped).

- [ ] **Step 15: Commit**

```bash
git add examples/mcp_server_demo/server.py examples/mcp_server_demo/agent_demo.py examples/mcp_server_demo/prompt_demo.py examples/mcp_server_demo/README.md tests/integration/mcp/test_agent_uses_mcp.py tests/integration/mcp/test_prompts.py scripts/run_examples.sh
git commit -m "feat(examples): MCP resource + prompt demos end-to-end

- server.py: adds documentation resource + summarize_text prompt
- agent_demo.py: LLM reads the documentation resource through the
  standard ToolInvoker pipeline; uses unwrap_mcp_tool_message to
  display the result cleanly
- prompt_demo.py (new): app-invoked path — list_prompts(),
  get_prompt(), splice messages into ainvoke()
- README updated to document both demos
- Integration test extended for resource read; new test_prompts.py
  for the prompt path
- Smoke gate runs the new demos alongside the existing ones"
```

---

## Definition of done

- All six tasks committed; six green commits between BASE (post-Phase-11 main) and HEAD.
- `uv run pytest` is green; integration tests run when FastMCP is importable.
- `uv run ruff check src/ai_core/mcp/ src/ai_core/agents/base.py tests/ examples/` is clean (no new violations vs. baseline).
- `bash scripts/run_examples.sh` exercises `mcp_agent_demo` (reads documentation resource) and `mcp_prompt_demo` (app-invoked prompt) and exits 0 (when `uv` and `fastmcp` are present).
- `git diff main -- src/ai_core/` shows changes only in `mcp/` (tools.py, resolver.py, prompts.py new, __init__.py) and `agents/base.py`.
- `examples/mcp_server_demo/agent_demo.py` runs cleanly and shows the documentation text (no `{"value": ...}` wrapper visible to the user).
- `examples/mcp_server_demo/prompt_demo.py` runs cleanly and shows both panels (available prompts + agent response).
- Phase 11's behavior is unchanged: existing tests pass without modification; agents that don't override `mcp_servers()` see no behavior change.
