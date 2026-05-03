package eaap.api

# Default-deny: every request must be explicitly allowed.
default allow := false

# Allow authenticated users to read their own profile.
allow if {
    input.action == "profile.read"
    input.user != ""
    input.user == input.request.path_params.user_id
}


package eaap.agent.tool_call

default allow := false

# Allow agents to call any tool that isn't on the deny list.
allow if {
    not denied_tool
}

denied_tool if {
    input.tool.name == "delete_everything"
}
