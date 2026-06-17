# MCP Admin Capability Gaps

Date: 2026-06-08

This is a read-only integration plan for extending the ExtraArena admin MCP
surface without changing `web/admin_capabilities.py` or `web/mcp_admin_tools.py`.
It maps requested ExtraAdmin sections to stable HTTP handlers and Database
methods, then proposes safe capability schemas and adapter behavior.

## Ground Rules

- Do not expose raw HTTP, raw SQL, or a generic admin proxy.
- Mutating tools must follow the existing MCP gateway contract:
  `dry_run`, one-time `confirmation_token`, scoped `idempotency_key`, audit,
  and rate limiting.
- Keep schemas narrower than the HTTP handlers where the HTTP surface accepts
  loose JSON.
- Prefer existing Database methods over duplicating SQL in MCP adapters.
- Return JSON-safe objects only; cap read payloads that can grow large.

Existing MCP coverage already includes runtime read/patch, player search/detail,
shop product/set read, shop set create, ExtraPass season draft/patch, ExtraPass
reward import, ExtraPass reset preview/execute, ExtraPass entitlement, match
mode read, push status, analytics overview, player notes, and resource grants.

## Stable Backing Map

| Section | Stable HTTP handlers | Stable DB/service methods | Current MCP gap |
| --- | --- | --- | --- |
| Squads | `GET /api/admin/squads/analytics`, `GET /api/admin/squads/list`, `GET /api/admin/squads/{clan_id}`, `POST /api/admin/squads/...` in `web/server.py:14549` | `get_admin_squads_analytics`, `search_admin_squads`, `resolve_clan_identifier`, `get_admin_squad_detail`, `admin_update_squad`, `admin_adjust_squad_balance`, `admin_squad_member_action`, `get_squad_runtime_config`, `process_weekly_squad_cbrp` | No squad MCP tools |
| Match mode toggles | `GET/POST /api/admin/match-modes` in `web/server.py:11471` | `get_match_mode_overrides`, `set_match_mode_enabled`, `is_match_mode_enabled` | Read exists; write missing |
| Push app-update | `POST /api/admin/push/app-update` in `web/server.py:11906` | `count_push_devices`, `get_push_devices_for_broadcast`, `mark_push_device_error`, `build_android_push_payload`, `send_android_broadcast` | Status read exists; broadcast missing |
| Rewards track CRUD | `GET/POST /api/admin/rewards/tracks...` in `web/server.py:12045` | `get_all_reward_tracks`, `get_reward_track_by_id`, `create_reward_track`, `update_reward_track`, `delete_reward_track`, `replace_reward_tracks` | Season import exists; generic CRUD missing |
| Analytics/export | `GET /api/admin/analytics/*` in `web/server.py:10893` | `get_admin_*_analytics`, `export_train_v2_battle_dataset` | Only overview read exists |
| TPS | `GET /api/admin/tps` in `web/server.py:11986` | `tps_monitor.get_statistics()` | Missing read tool |
| Config | `GET /api/admin/configs`, `GET/POST /api/admin/runtime-config` in `web/server.py:11420` | `get_runtime_config`, `set_runtime_config`, `get_squad_runtime_config`, `set_game_setting` | Runtime exists; summary and squad config gaps |
| Cards | `GET/POST /api/admin/cards...` in `web/server.py:5653` | `get_cards_list`, `create_card`, `add_all_cards_to_user`, `delete_all_user_cards` | Missing; create is high-risk |
| Items | `GET/POST /api/admin/items...` in `web/server.py:5745` | No `Database.create_item` or `Database.get_items_list` found | Do not add yet |
| Debug | `POST /api/admin/debug/add-key` in `web/server.py:8129` | Direct SQL only | Do not add raw debug tool |

## Common Mutating Fields

Every mutating schema below should include these fields:

```json
{
  "dry_run": {"type": "boolean"},
  "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 128},
  "confirmation_token": {"type": "string", "minLength": 16, "maxLength": 256},
  "reason": {"type": "string", "maxLength": 500}
}
```

`dry_run` is required. `confirmation_token` is optional in schema because the
gateway requires it only for the apply call.

## P1 - Ready To Integrate

### 1. `admin.match_modes.availability.set`

Purpose: toggle one known match mode via DB override.

Scope: `admin:match_modes:write`

Safety: high, mutating, `request_and_result`, dry-run and idempotency required.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "mode_id": {
      "type": "string",
      "enum": [
        "classic",
        "friendly",
        "training",
        "extra_arena:blitzkrieg",
        "extra_arena:spellstorm",
        "extra_arena:sudden_death",
        "extra_arena:powermax"
      ]
    },
    "enabled": {"type": "boolean"},
    "dry_run": {"type": "boolean"},
    "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 128},
    "confirmation_token": {"type": "string", "minLength": 16, "maxLength": 256},
    "reason": {"type": "string", "maxLength": 500}
  },
  "required": ["mode_id", "enabled", "dry_run"]
}
```

Adapter behavior:

- Build the same known-mode set used by `admin_match_modes_handler`.
- Read current state with `get_match_mode_overrides`.
- Dry-run returns `current`, `candidate`, and the full projected modes list.
- Apply calls `set_match_mode_enabled(mode_id, enabled)`.
- Re-read overrides and return `{dry_run:false, mode_id, enabled, modes}`.
- Reject unknown mode ids before hitting the DB.

### 2. `admin.push.app_update.broadcast`

Purpose: send the required Android app-update broadcast.

Scope: `admin:push:write`

Safety: critical, mutating, `request_and_result`, dry-run and idempotency
required.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "title": {"type": "string", "maxLength": 120},
    "body": {"type": "string", "maxLength": 500},
    "url": {
      "type": "string",
      "enum": ["https://t.me/extraarenamobile", "https://apk.laveqox.ru"]
    },
    "limit": {"type": "integer", "minimum": 1, "maximum": 50000},
    "dry_run": {"type": "boolean"},
    "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 128},
    "confirmation_token": {"type": "string", "minLength": 16, "maxLength": 256},
    "reason": {"type": "string", "maxLength": 500}
  },
  "required": ["dry_run"]
}
```

Adapter behavior:

- Get `push_sender` from `app`; return `push_sender_unavailable` if absent.
- Build payload through `build_android_push_payload("app_update",
  "app_update_required", payload)`, not by hand.
- Count devices with `count_push_devices(platform="android")`.
- Dry-run returns `devices`, `limit`, and sanitized payload data.
- Apply rejects unconfigured senders with `push_sender_not_configured`.
- Apply calls `send_android_broadcast(db=db, push_sender=sender, payload=payload,
  platform="android", limit=limit)`.
- Return `devices`, `sent`, and `failed`; never return device tokens.

### 3. `admin.rewards.tracks.read`

Purpose: read generic reward track rows, with optional filters.

Scope: `admin:seasons:read`

Safety: medium, read-only, audit metadata.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "track_type": {"type": "string", "minLength": 1, "maxLength": 80},
    "active_only": {"type": "boolean"},
    "limit": {"type": "integer", "minimum": 1, "maximum": 2000}
  }
}
```

Adapter behavior:

- Call `get_all_reward_tracks()`.
- Filter in memory by `track_type` and `is_active` when requested.
- Return `{tracks, total, truncated}`.
- Default limit should be 1000; cap at 2000 to avoid bloated MCP responses.

### 4. `admin.rewards.tracks.create`

Purpose: create or upsert one reward track row.

Scope: `admin:seasons:write`

Safety: high, mutating, `request_and_result`, dry-run and idempotency required.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "track_type": {"type": "string", "minLength": 1, "maxLength": 80},
    "position": {"type": "integer", "minimum": 1, "maximum": 100000},
    "reward_type": {"type": "string", "enum": ["coins", "gems", "keys", "card", "case"]},
    "reward_amount": {"type": "integer", "minimum": 0, "maximum": 10000000},
    "reward_meta": {"type": "object", "additionalProperties": true},
    "extra_pass_required": {"type": "boolean"},
    "dry_run": {"type": "boolean"},
    "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 128},
    "confirmation_token": {"type": "string", "minLength": 16, "maxLength": 256},
    "reason": {"type": "string", "maxLength": 500}
  },
  "required": ["track_type", "position", "reward_type", "reward_amount", "dry_run"]
}
```

Adapter behavior:

- Reject unsupported reward types before DB calls.
- For `case`, require `reward_amount` from 1 to 5.
- Load `get_seasons()` and apply the same ExtraPass track scope check as
  `_admin_reward_track_scope_error`: if the track belongs to a configured
  season lane, `position` must be inside that lane's start/end range.
- Dry-run returns normalized row and any existing row at the same
  `(track_type, position, reward_type)` if discoverable.
- Apply calls `create_reward_track(...)` and returns the DB row.

### 5. `admin.rewards.tracks.patch`

Purpose: update allowlisted fields on one reward track row.

Scope: `admin:seasons:write`

Safety: high, mutating, `request_and_result`, dry-run and idempotency required.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "id": {"type": "integer", "minimum": 1},
    "patch": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "track_type": {"type": "string", "minLength": 1, "maxLength": 80},
        "position": {"type": "integer", "minimum": 1, "maximum": 100000},
        "reward_type": {"type": "string", "enum": ["coins", "gems", "keys", "card", "case"]},
        "reward_amount": {"type": "integer", "minimum": 0, "maximum": 10000000},
        "reward_meta": {"type": "object", "additionalProperties": true},
        "extra_pass_required": {"type": "boolean"},
        "is_active": {"type": "boolean"}
      }
    },
    "dry_run": {"type": "boolean"},
    "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 128},
    "confirmation_token": {"type": "string", "minLength": 16, "maxLength": 256},
    "reason": {"type": "string", "maxLength": 500}
  },
  "required": ["id", "patch", "dry_run"]
}
```

Adapter behavior:

- Load existing row with `get_reward_track_by_id`; reject `reward_not_found`.
- Merge existing row plus patch, then run the same reward type and ExtraPass
  scope validation as create.
- Dry-run returns `current`, `patch`, and `candidate`.
- Apply calls `update_reward_track(id, **patch)`.

### 6. `admin.rewards.tracks.delete`

Purpose: soft-delete one reward track row.

Scope: `admin:seasons:write`

Safety: high, mutating, `request_and_result`, dry-run and idempotency required.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "id": {"type": "integer", "minimum": 1},
    "dry_run": {"type": "boolean"},
    "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 128},
    "confirmation_token": {"type": "string", "minLength": 16, "maxLength": 256},
    "reason": {"type": "string", "maxLength": 500}
  },
  "required": ["id", "dry_run"]
}
```

Adapter behavior:

- Load existing row first; reject missing rows.
- Dry-run returns the row that would be deactivated.
- Apply calls `delete_reward_track(id)` and returns `{deleted:true}` or
  `reward_not_found`.

### 7. `admin.squads.analytics.read`

Purpose: read admin squad analytics.

Scope: `admin:squads:read`

Safety: medium, read-only, audit metadata.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "days": {"type": "integer", "minimum": 1, "maximum": 365}
  }
}
```

Adapter behavior:

- Call `get_admin_squads_analytics(days=days)`.
- Return the same empty fallback shape used by ExtraAdmin if the method fails
  only if the caller needs dashboard resilience; otherwise fail closed.
- Do not include raw member contact data.

### 8. `admin.squads.search`

Purpose: list/search squads for ops triage.

Scope: `admin:squads:read`

Safety: medium, read-only, audit metadata.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "query": {"type": "string", "maxLength": 80},
    "filter": {"type": "string", "enum": ["all", "open", "closed", "boost", "full"]},
    "sort": {"type": "string", "enum": ["cbrp", "members", "treasury", "created", "rank"]},
    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
    "offset": {"type": "integer", "minimum": 0, "maximum": 1000000}
  }
}
```

Adapter behavior:

- Call `search_admin_squads(query, filter_type, sort, limit, offset)`.
- Normalize `filter` to DB parameter `filter_type`.
- Return `{total, squads}`.

### 9. `admin.squads.detail.read`

Purpose: read one squad by numeric id or public id.

Scope: `admin:squads:read`

Safety: medium, read-only, audit metadata.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "clan_id": {"type": "string", "minLength": 1, "maxLength": 80}
  },
  "required": ["clan_id"]
}
```

Adapter behavior:

- Resolve with `resolve_clan_identifier(clan_id)`.
- Reject `clan_not_found`.
- Call `get_admin_squad_detail(int(clan["id"]))`.
- Return clan, members, pending requests, activity, CBRP events, upgrades, and
  recent purchases/snapshots.

### 10. `admin.analytics.report.read`

Purpose: read any existing ExtraAdmin analytics section through one typed tool.

Scope: `admin:analytics:read`

Safety: medium, read-only, audit metadata.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "section": {
      "type": "string",
      "enum": [
        "overview",
        "revenue",
        "players",
        "battles",
        "economy",
        "cards",
        "heroes",
        "retention",
        "onboarding",
        "battle_actions"
      ]
    },
    "days": {"type": "integer", "minimum": 1, "maximum": 365}
  },
  "required": ["section"]
}
```

Adapter behavior:

- Dispatch by enum to the existing DB methods.
- For `economy`, avoid raw SQL in the MCP adapter if possible; add a
  `Database.get_admin_economy_analytics(days)` helper first or keep economy
  out of the enum until then.
- Return `{section, days, data}`.

### 11. `admin.analytics.dataset.export.read`

Purpose: export a bounded Train V2 battle-action dataset over MCP.

Scope: `admin:analytics:export`

Safety: medium, read-only, audit policy `request`.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "days": {"type": "integer", "minimum": 1, "maximum": 365},
    "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
    "include_players": {"type": "boolean"}
  }
}
```

Adapter behavior:

- Default `include_players` to false.
- Call `export_train_v2_battle_dataset(days, limit, include_players)`.
- Return structured JSON `{format, days, rows, samples, truncated:false}` rather
  than NDJSON text.
- Cap MCP `limit` lower than the HTTP export max to prevent huge JSON-RPC
  responses.

## P2 - Useful, But Needs Narrow Guards

### `admin.squads.config.read`

Read `get_squad_runtime_config()` with scope `admin:squads:read`.

### `admin.squads.config.scalar.set`

Set only scalar squad settings first. Do not accept full replacement of
`squad_upgrades`, `squad_personal_rewards`, or nested `squad_rewards` until they
have dedicated validators.

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "key": {
      "type": "string",
      "enum": [
        "squad_creation_policy",
        "squad_weekly_cbrp_enabled",
        "squad_seasonal_cbrp_enabled",
        "squad_clan_boost_token_multiplier",
        "squad_creator_passive_tax_pct",
        "squad_weekly_delta_divisor",
        "squad_weekly_personal_tokens_divisor",
        "squad_weekly_treasury_tokens_divisor",
        "squad_seasonal_cbrp_divisor",
        "squad_seasonal_personal_tokens_divisor",
        "squad_seasonal_treasury_tokens_divisor"
      ]
    },
    "value_bool": {"type": "boolean"},
    "value_number": {"type": "number", "minimum": 0, "maximum": 1000000},
    "value_string": {"type": "string", "maxLength": 80},
    "dry_run": {"type": "boolean"},
    "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 128},
    "confirmation_token": {"type": "string", "minLength": 16, "maxLength": 256},
    "reason": {"type": "string", "maxLength": 500}
  },
  "required": ["key", "dry_run"]
}
```

Adapter behavior:

- Validate that exactly one of `value_bool`, `value_number`, or `value_string`
  is present and matches the selected key's expected type.
- Dry-run returns current value and candidate value.
- Apply calls `set_game_setting(key, value, "Updated from MCP squad config")`
  and returns `get_squad_runtime_config()`.

### `admin.squads.update`

Use `admin_update_squad(admin_user_id, clan_id, fields, reason)`.

Keep schema allowlist to non-destructive fields first:
`name`, `tag`, `description`, `type`, `min_trophies`, `max_members`,
`has_boost`, `avatar_url`, `banner_url`. Defer `owner_id`, `rank`, `trophies`,
`cbrp`, and `treasury_tokens` to separate tools because they alter ownership or
economy state.

### `admin.squads.balance.adjust`

Use `admin_adjust_squad_balance(admin_user_id, clan_id, resource, amount,
reason)`.

Schema should require `resource` enum `cbrp|treasury_tokens`, `amount` between
`-1000000` and `1000000`, and non-empty `reason`.

### `admin.squads.member.action`

Use `admin_squad_member_action(...)`.

Schema should allow actions `add`, `kick`, `promote`, `demote`, and
`set_tokens`. Defer `transfer` to its own critical tool because it changes
ownership.

### `admin.runtime.tps.read`

Read `tps_monitor.get_statistics()` with no args or optional
`include_history:boolean`. This is read-only and low risk, but it has no DB
method, so the adapter must handle missing `tps_monitor` gracefully.

### `admin.configs.summary.read`

Read the same blocks as `/api/admin/configs`: match modes, promocodes count,
reward tracks, active season, shop sets, ruble products, runtime config, cards,
and squad analytics. Keep it read-only, medium safety, and include per-section
errors instead of failing the whole tool.

### `admin.cards.list`

Use `get_cards_list()`. Optional filters: `rarity`, `query`, `limit`, `offset`.
This is safe read-only and useful for reward-track validation.

### `admin.cards.create`

Use `create_card(...)`, but keep this P2/P3 unless product confirms DB-created
cards are the source of truth. The repo also has `cards.json`, card art assets,
and training/card-shape expectations. A MCP card-create tool can drift from
those assets if it only inserts into the DB.

Minimal safe schema if enabled:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "name": {"type": "string", "minLength": 1, "maxLength": 120},
    "description": {"type": "string", "maxLength": 1000},
    "rarity": {
      "type": "string",
      "enum": ["common", "rare", "superrare", "epic", "legendary", "mythic", "divine", "limited", "start"]
    },
    "power": {"type": "integer", "minimum": 0, "maximum": 100000},
    "mana_cost": {"type": "integer", "minimum": 0, "maximum": 20},
    "base_attack": {"type": "integer", "minimum": 0, "maximum": 100000},
    "base_hp": {"type": "integer", "minimum": 1, "maximum": 100000},
    "card_type": {"type": "string", "enum": ["warrior", "spell", "potion"]},
    "dry_run": {"type": "boolean"},
    "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 128},
    "confirmation_token": {"type": "string", "minLength": 16, "maxLength": 256},
    "reason": {"type": "string", "maxLength": 500}
  },
  "required": ["name", "rarity", "power", "dry_run"]
}
```

## P3 - Defer Or Do Not Add Yet

### Items section

Do not expose item MCP tools yet. ExtraAdmin has `admin_items_create_handler`
and `admin_items_list_handler`, but `Database.create_item` and
`Database.get_items_list` were not found. Adding MCP on top would expose a
known-failing path.

Recommended prerequisite:

- Decide whether "items" means legacy gameplay items or current
  `cosmetic_items`.
- Add stable DB methods and tests first.

### Debug add-key

Do not expose `debug_add_key_handler` as MCP. It uses direct SQL against the
current admin user only. The existing `admin.players.resource.grant` MCP tool
already covers granting keys to a target user through the audited economy path.

### Squad create/delete/upgrade/weekly process

These have HTTP and DB backing, but they are high-blast operations:

- create: `create_clan` plus private activity logging
- delete: `delete_clan`
- upgrade: `buy_clan_upgrade` or direct `execute` for set-level mode
- weekly process: `process_weekly_squad_cbrp` can affect many squads

Keep them out of the first MCP expansion unless each gets a separate critical
tool with strict confirmation, idempotency, and preview output.

## Suggested Integration Order

1. Add `admin.match_modes.availability.set` and tests. This is the smallest
   missing write surface and uses stable DB methods.
2. Add `admin.push.app_update.broadcast` with dry-run tests and fake sender
   tests.
3. Add `admin.rewards.tracks.read/create/patch/delete` after extracting or
   duplicating the ExtraPass scope validation.
4. Add squad read tools: analytics, search, detail, config read.
5. Add analytics report and bounded dataset export.
6. Add TPS/config summary/cards list.
7. Consider squad writes and card create only after the narrow validators are
   in place.

## Test Targets For Future Integration

- `tests/test_mcp_admin_gateway.py`: registry visibility, schema rejection,
  dry-run/confirmation/idempotency, sanitized error behavior.
- `tests/test_admin_management_workflows.py`: parity with ExtraAdmin HTTP
  handlers, especially reward-track scope errors and squad fallbacks.
- `tests/test_push_notifications.py`: app-update payload and broadcast error
  handling.
- `tests/test_match_modes.py`: catalog and mode availability invariants.
- `tests/test_squad_backend_invariants.py`: membership and CBRP safety
  invariants before enabling squad mutators.
