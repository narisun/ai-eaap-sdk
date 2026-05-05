# Decision document used by `GuardrailNode` to authorise agent tool calls.
# Default decision path: data.eaap.agent.tool_call.allow
package eaap.agent.tool_call

# Default-deny: every tool call must be explicitly allowed.
default allow := false

# Allow agents to call any tool that isn't on the deny list.
allow if {
    not denied_tool
}

# Deny-list of tools the agent must never invoke without HITL approval.
denied_tool if {
    input.tool.name == "delete_everything"
}
