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

## Android Release Control Plane

Android release automation uses two deliberately separate planes:

- MCP carries typed metadata and lifecycle commands.
- The APK/AAB bytes use an authenticated resumable HTTP upload. Binary content is never base64-encoded into MCP JSON.

Capabilities and scopes:

- `admin.android.releases.list` and `admin.android.releases.read` require `admin:android_releases:read`.
- `admin.android.releases.upload`, `admin.android.releases.publish`, and `admin.android.releases.retire` require `admin:android_releases:write` plus the normal dry-run, confirmation, idempotency, and audit controls.
- Publish and retire are critical operations. Upload prepare/finalize/abort are high-risk operations.

Upload sequence:

1. Call `admin.android.releases.upload` with `action=prepare`, metadata, exact byte size and SHA-256, first as a dry run and then as a confirmed apply.
2. The apply result contains bounded `chunk_bytes` and an `upload_url`, but no second credential. MCP responses, audit rows, confirmations, and idempotency results never contain an upload token.
3. Stream each bounded chunk with `PATCH upload_url`, the existing `Authorization: Bearer <MCP token>`, and the exact `Upload-Offset`. The server verifies `mcp:admin` plus `admin:android_releases:write` specifically on this route and rejects gaps, overlaps, oversized chunks, and bytes beyond the declared size. The browser ExtraAdmin flow instead receives an expiring upload-only ticket and sends it in `X-Android-Upload-Token`.
4. Call the same MCP upload capability with `action=finalize`. Finalization re-reads the file, checks its size and SHA-256, and runs configured `aapt2` and `apksigner` verification.
5. Read the staged release, dry-run publish, review the effective latest version and required floor, then apply with a fresh idempotency key. Pass `store_release_confirmed=false` for direct. For RuStore, independently verify that the exact version is already available to users in RuStore Console before passing `store_release_confirmed=true`.

Direct-channel publication fails closed unless the artifact is an APK, package metadata matches `ANDROID_RELEASE_PACKAGE_NAME`, signature verification succeeds, and its certificate matches `ANDROID_DIRECT_SIGNING_CERT_SHA256`. A separate optional `ANDROID_RUSTORE_SIGNING_CERT_SHA256` pin applies only to RuStore APKs; the direct pin is never applied across channels. A staged `(channel, version_code, version_name)` release may retain both one APK and one AAB; uploading the missing artifact kind attaches it to that same release rather than consuming a second version. Published/retired releases, a conflicting version name, and duplicate artifact kinds are rejected. AAB files are retained as staging/storage artifacts for an external console workflow and are explicitly unpublishable by the V1 release service; V1 exposes no AAB export/download endpoint.

RuStore publication also fails closed without the explicit store-rollout confirmation. ExtraAdmin disables auto-publish for RuStore and requires the operator to type `RUSTORE LIVE <version_code>` for manual publication. This guard must be satisfied for optional releases too, and especially before advancing the required floor.

Public direct-APK links are always emitted as absolute HTTPS URLs so already-installed bootstrap clients can consume them. `ANDROID_RELEASE_PUBLIC_BASE_URL` may set a dedicated trusted origin; otherwise the service uses configured `PUBLIC_BASE_URL` or `WEBAPP_URL`. It never builds update URLs from the inbound `Host` header.

Mandatory releases advance `min_supported_version_code` monotonically. Optional releases advance only the latest version. For example, after required v10 followed by optional v11, a v9 client is still blocked but downloads v11, while a v10 client may skip v11. Retiring or superseding a release never lowers the floor.

Release storage is configured with `ANDROID_RELEASE_STORAGE_DIR`; use a dedicated persistent mount. Public clients read the no-store manifest at `/api/mobile/client-version` (or `/api/mobile/android/releases/manifest`) and download immutable verified APKs from `/api/mobile/android/releases/{release_id}/apk` with Range support.
Expired resumable uploads are marked expired and their app-owned `.part` files are removed at server startup and every 15 minutes. Cleanup uses the same per-upload cross-process lock as append/finalize/abort and never follows a path outside the configured release storage root.

## Protocol Shape

The gateway implements the JSON-RPC subset needed by MCP clients:

- `initialize`
- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`

The MCP transport itself remains request/response. Large Android artifacts use the separate scoped resumable HTTP path described above rather than an MCP streaming extension.

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
