# Secure Admin MCP

This document describes the local server-side MCP control plane for ExtraArena administration.

## Status

The MCP surface is opt-out. It is enabled by default (`MCP_ENABLED=true`). Operators that want to fully disable it must set `MCP_ENABLED=false`; in non-development environments the server still enforces a non-default `MCP_TOKEN_SECRET` and an explicit `MCP_ALLOWED_ORIGINS` allowlist before allowing the surface to stay on.

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

## Audit (2026-06-25)

Checked against `web/mcp_routes.py`, `web/mcp_admin_tools.py`, `web/mcp_auth.py`, `web/admin_capabilities.py`, `infrastructure/config.py`, `infrastructure/database.py`, `infrastructure/moderation.py`.

Fixed:
- `docs/SECURE_ADMIN_MCP.md:7` — corrected "opt-in / disabled unless `MCP_ENABLED=true`" to "opt-out / enabled by default (`MCP_ENABLED=true`)", matching `infrastructure/config.py:369,585` where the default and env fallback are both `True` (only an explicit `MCP_ENABLED=false` disables it).

Verified accurate (no change needed):
- Endpoint paths: `DEFAULT_MCP_ENDPOINT_PATH="/admin/mcp"`, `DEFAULT_MCP_SESSION_PATH="/api/admin/mcp/session"` (`infrastructure/config.py:32-33`).
- TTL default `15 * 60 = 900` (`infrastructure/config.py:34`) matches the `MCP_TOKEN_TTL_SECONDS=900` example.
- Production guard: `MIN_PRODUCTION_MCP_TOKEN_SECRET_LENGTH=32` and the secret must differ from `JWT_SECRET`/`ADMIN_SESSION_SECRET`/`DEFAULT_MCP_TOKEN_SECRET` (`infrastructure/config.py:38,649-660`).
- Auth flow rejects query `_auth`, cookies, and non-bearer auth (`web/mcp_routes.py:94-103`); `verify_mcp_token` requires `mcp:admin` scope (`web/mcp_auth.py:86-143`).
- Session response shape `{status, endpoint, token, expires_in, scopes}` (`web/mcp_routes.py:576-582`).
- Tool descriptor fields `safetyLevel`, `requiredScope`, `auditPolicy`, `idempotencyRequired`, `dryRunRequired` (`web/admin_capabilities.py:54-62`).
- Idempotency scoped by `(admin_user_id, tool_name)`; rate-limit scope `mcp:<tool>` keyed on `subject=admin_user_id` (`web/mcp_routes.py:186-198`).
- Rate-limit constants `MCP_RATE_LIMIT_REQUESTS=120`, `MCP_RATE_LIMIT_WINDOW_SECONDS=60` (`web/mcp_routes.py:29-30`).
- Persistence tables `mcp_tool_calls`, `mcp_confirmations`, `mcp_idempotency_keys`, `mcp_rate_limits` ensured in `Database.initialize()` via `_ensure_mcp_admin_tables` (`infrastructure/database.py:17433-17439`); raw tokens/JTI hashed with SHA-256, sanitized JSON for args/results (`web/mcp_routes.py:47-52`, `web/mcp_admin_tools.py:131-140`).
- JSON-RPC methods `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read` (`web/mcp_routes.py:623-644`); server metadata `MCP_PROTOCOL_VERSION="2025-06-18"`, `MCP_SERVER_NAME="extraarena-admin"` (`web/mcp_routes.py:27-28`).
- Fail-closed on missing `record_mcp_tool_call`, `create_mcp_confirmation`, `reserve_mcp_idempotency_key` (`web/mcp_routes.py:215-217, 241-243, 298-301`).

Not verified: live runtime behavior of the moderation pipeline in `infrastructure/moderation.py` — file exists and exports `moderate_content` / `check_rate_limit` but the doc does not reference it, so no claim to update.
