# fastapi_integration — OPA-protected FastAPI endpoint

A minimal FastAPI app with a single OPA-protected endpoint, using the
SDK's `OPAAuthorization` dependency and the SDK's starter `api.rego`
policy.

## What this demonstrates

- Wiring `OPAAuthorization` into a FastAPI handler via `Depends(...)`.
- Reusing the SDK's starter `api.rego` policy (no copy/paste).
- The full request flow: JWT bearer → `HS256JWTVerifier` → OPA → handler.

## What's *not* shown

The SDK does **not** ship a FastAPI request-scoped audit middleware
today. To see audit on the agent/tool path, run `examples/agent_demo/`.

## Prerequisites

```bash
uv sync
```

You'll need Docker to run OPA locally.

## Step 1 — Start OPA

In one terminal, from the repo root:

```bash
docker run --rm -p 8181:8181 \
    -v "$(pwd)/src/ai_core/cli/templates/init/policies:/policies" \
    openpolicyagent/opa:0.66.0 \
    run --server --addr 0.0.0.0:8181 /policies
```

OPA loads `api.rego` and `agent.rego` and serves on
`http://localhost:8181`.

## Step 2 — Start the FastAPI app

In a second terminal:

```bash
export DEMO_JWT_SECRET=dev-secret-change-me
uv run python examples/fastapi_integration/app.py
```

## Step 3 — Mint two JWTs and call the endpoint

In a third terminal:

```bash
# Allowed: token sub matches user_id path param.
ALLOW=$(uv run python -c "
import jwt, time
print(jwt.encode(
  {'sub': 'alice', 'iss': 'ai-core-sdk-demo', 'aud': 'ai-core-sdk-demo',
   'exp': int(time.time()) + 600},
  'dev-secret-change-me', algorithm='HS256'))")

curl -s -H "Authorization: Bearer $ALLOW" \
    http://localhost:8000/users/alice/profile
# {"user_id":"alice","subject":"alice","email":"(unknown)"}

# Denied: token sub does not match user_id path param.
curl -i -H "Authorization: Bearer $ALLOW" \
    http://localhost:8000/users/bob/profile
# HTTP/1.1 403 Forbidden
```

## How it works

1. FastAPI receives the request and resolves `Depends(authz.requires(...))`.
2. The dependency extracts the bearer token and verifies it via
   `HS256JWTVerifier`.
3. It builds the OPA input doc:
   `{user, action, resource, claims, request: {path_params, ...}}`.
4. It POSTs to OPA `/v1/data/eaap/api/allow`.
5. The rego policy evaluates `input.user == input.request.path_params.user_id`.
6. On `allow=true` the handler runs; on `false` the dependency raises 403.

## What to read next

- `src/ai_core/security/fastapi_dep.py` — `OPAAuthorization` source.
- `src/ai_core/cli/templates/init/policies/api.rego` — the demo policy.
- `src/ai_core/di/module.py` — `ProductionSecurityModule`.
