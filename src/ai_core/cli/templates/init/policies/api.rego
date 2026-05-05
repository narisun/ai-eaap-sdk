# Decision document used by the FastAPI `OPAAuthorization` dependency.
# Default decision path: data.eaap.api.allow
package eaap.api

# Default-deny: every request must be explicitly allowed.
default allow := false

# Allow authenticated users to read their own profile.
allow if {
    input.action == "profile.read"
    input.user != ""
    input.user == input.request.path_params.user_id
}
