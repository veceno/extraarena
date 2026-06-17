# Secure Admin MCP

This document describes the local server-side MCP control plane for ExtraArena administration.

## Status

The MCP surface is opt-in. It is disabled unless `MCP_ENABLED=true` is configured.

The implementation intentionally does not proxy arbitrary admin HTTP routes. Every exposed action is an allowlisted capability from `web/admin_capabilities.py`, executed through typed adapters in `web/mcp_admin_tools.py`.

## Configuration

Required for production-like use:

- `MCP_ENABLED=true`
- `MCP_TOKEN_SECRET=<separate strong secret>`
- `MCP_ALLOWED_ORIGINS=https://your-ops-origin.example`

Optional path controls:

- `MCP_ENDPOINT_PATH=/admin/mcp`
- `MCP_SESSION_PATH=/api/admin/mcp/session`
- `MCP_TOKEN_TTL_SECONDS=900`

`MCP_ENDPOINT_PATH` and `MCP_SESSION_PATH` must be different. `MCP_SESSION_PATH` must remain under `/api/admin/` so the existing centralized admin auth middleware protects token bootstrap. Do not put production hostnames or `extraShop` assumptions into MCP clients; call the session route and use the returned `endpoint`.

Outside development, `MCP_TOKEN_SECRET` must be separate from `JWT_SECRET` and `ADMIN_SESSION_SECRET`, non-default, and at least 32 characters. `MCP_ALLOWED_ORIGINS` must be an explicit allowlist.

## Auth Flow

1. Authenticate as an ExtraArena admin using the existing admin boundary.
2. `POST MCP_SESSION_PATH` with regular admin auth.
3. The response contains:
   - `endpoint`
   - short-lived MCP token
   - `expires_in`
   - granted scopes
4. MCP clients call `POST endpoint` with `Authorization: Bearer <mcp-token>`.

The MCP endpoint rejects:

- query `_auth`
- cookies/admin session cookies
- dev `user_id` fallback
- normal game JWTs
- non-allowlisted browser `Origin`

## Tool Safety

Tool descriptors include scope, safety level, audit policy, dry-run policy, and idempotency policy. Mutating tools require:

- the required scope
- an idempotency key
- a dry-run call first
- a one-time confirmation token from that dry run
- universal MCP audit logging

Idempotency keys are scoped by admin user and tool before hashing, so the same client-provided key cannot collide across operators or capabilities. Completed idempotency records are replayed before confirmation consumption, and active `in_progress` records block duplicate execution.

Rate limits are scoped per admin user and tool, not per token JTI, so rotating the short-lived MCP token does not reset a tool bucket.

The gateway fails closed when MCP persistence is unavailable for audit, confirmation, or idempotency.

## Persistence

`Database.initialize()` ensures:

- `mcp_tool_calls`
- `mcp_confirmations`
- `mcp_idempotency_keys`
- `mcp_rate_limits`

Raw tokens are not stored. JTI, confirmation tokens, idempotency keys, and rate-limit subjects are SHA-256 hashed. Arguments and results are stored through sanitized JSON summaries.

## Protocol Shape

The gateway implements the JSON-RPC subset needed by MCP clients:

- `initialize`
- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`

Streaming/session extensions are intentionally left for a later iteration; the current control plane is request/response and local-server safe.
