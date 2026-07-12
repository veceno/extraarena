# CasesFix — particle-drop bug fix + real-time case config

Worktree branch: `glm-5.2/CasesFix`. Designed via ultracode workflow (3 architects → synthesis → 3 adversarial verifiers → final vetted plan). All verifiers PASS_WITH_FIXES; their must-fix edits are incorporated below.

## Context

Two problems in one change-set:

**Problem A (bug):** "Particles for some cards don't drop from cases." Particles are the per-card upgrade currency granted when a case rolls a card the user already owns (a duplicate). Value = `int(BASE_PARTICLES_BY_RARITY[rarity] * TIER_PARTICLES_MULTIPLIER[tier])`, truncated. With current HEAD values every non-`limited` cell is ≥1, but:
- `BASE_PARTICLES_BY_RARITY["limited"] = 0` (`infrastructure/case_config.py:171`) → limited duplicates **always** grant 0 particles (the 0-reward is appended then silently no-op'd by `Database.add_particles_to_card`'s `if particles <= 0` guard at `database.py:17140`).
- The `int()` truncation is **fragile**: any future lowering of multipliers re-introduces 0-particle cells (already happened once: `960a5e8e` made `int(2×0.38)=0`; fixed by `3cb63560`+`a4544544`). The user will lower case chances via the new surface, so a guard is mandatory.

**Problem B (feature):** The limited-cards drop toggle (`LIMITED_EVENT_ACTIVE` + `LIMITED_EVENT_PROBABILITY`, `case_config.py:205-206`) is restart-only because `case_system` imports them at module load (`case_system.py:25-26`) and reads them as module globals (`case_system.py:136`). User wants BOTH flag + probability real-time, AND ALL case chances real-time, via BOTH `/extraShop/admin` UI AND a typed ExtraAdmin MCP capability — no restart.

## Architecture (chosen: "single-blob-injected" with structured deep-merge)

- ONE JSONB row in the existing `game_settings` table, key `"case_config"`, holding the full case-config blob. Self-contained typed wrapper `get_case_config`/`set_case_config` mirrors `get_runtime_config`/`set_runtime_config`.
- `admin.case_config.read` + `admin.case_config.patch` MCP capability pair mirrors `admin.runtime.config.read`/`patch`.
- `/api/admin/case-config` GET+POST route mirrors `/api/admin/runtime-config`.
- A Case-config panel in `extraShop/admin.html` mirrors the Runtime-availability panel.
- `case_system` roll functions gain an optional `case_config: dict|None = None` param (default → module globals = backward-compatible, no DB access for tests/standalone). The server reads the live blob ONCE per handler via `_case_config_safe(db)` and injects it; roll functions never touch the DB directly.

### Key design points (verifier must-fixes incorporated)
1. **Structured deep-merge, NOT shallow.** `merge_case_config_patch(current, patch)` merges per-tier (tier-keyed sub-dicts: replace only the tiers present in the patch, keep others) and per-rarity (`base_particles_by_rarity`, `start_rarity_replacement`: replace only the rarities present, keep others). This prevents a partial patch like `{"base_particles_by_rarity":{"limited":100}}` from zeroing common/rare/divine — which would re-introduce the 0-particle bug. Shared helper used by both `Database.set_case_config` and the `GatewayMCPDB` test stub.
2. **Canonical tier-key representation = INT in Python.** JSONB stores string keys; `_coerce_tier_keys` converts str→int on read; `_normalize_case_config_patch` coerces patch tier keys to int; `validate_case_config` requires int tier keys 1..5. No mixed int/string hazard.
3. **`resolve_case_config(None)` returns LIVE references** to `case_config` module-global dict objects (NOT copies) — load-bearing for `test_t5_case_rewards_do_not_generate_removed_limited_shards`, which does `monkeypatch.setitem(case_system.TIER_REWARDS_COUNT, 5, {...})`. Pinned by an identity test + inline comment.
4. **`simulate_case_tap_results` keeps the explicit `if case_config is not None` branch** (3-arg vs 4-arg `roll_tier_upgrade` call) — required for the 3-param fake in `test_simulate_case_tap_results_uses_server_rolls`. Do-not-collapse comment + tests for both branches.
5. **`_case_config_safe(db)` fetched ONCE per handler** at the top (after auth gate), reused for all roll calls in that handler (matches `_runtime_config_safe` single-fetch-per-request). `getattr`-first form → test DB mocks without `get_case_config` return `None` silently.
6. **`fill_case_config_defaults` deep-fills missing tier keys within each tier-keyed sub-dict** from defaults (mirrors `_merge_runtime_defaults` filling keys inside `feature_availability`), not just top-level keys.
7. **Migration:** `_ensure_game_settings_table` seeds the `case_config` row `ON CONFLICT DO NOTHING`; `_merge_case_config_defaults` fills newly-introduced top-level + tier keys without clobbering admin edits; idempotent.

## Particle fix (Problem A)
1. `infrastructure/case_config.py:171` — `"limited": 0` → `"limited": 150` (above `divine`=100; limited is the highest rarity ordinal, so a duplicate should grant particles like every other rarity; 0.15% T5-event-only drops → marginal particle impact negligible). **This is the only value change.**
2. `infrastructure/case_system.py:260` — replace `return int(base_particles * multiplier)` with:
   ```python
   amount = int(base_particles * multiplier)
   if base_particles > 0 and amount < 1:
       amount = 1
   return amount
   ```
   Guard fires only for base>0 (admin explicit zero respected); defends against `int()` truncation when admin later lowers `tier_particles_multiplier` live. T5-common-jackpot early-return (`:251-252`, returns 125) stays unchanged.

## Stored blob shape
`game_settings` row `case_config` = JSONB blob with keys: `tier_rarity_probabilities`, `tier_particles_multiplier`, `base_particles_by_rarity` (incl `limited:100`), `tier_rewards_count` (tuples serialize as `[a,b]`), `start_rarity_replacement`, `max_rarity_by_tier`, `t5_common_jackpot_particles`, `tier_upgrade_chances`, `limited_event_active`, `limited_event_probability`. Defaults flow from current module constants.

## MCP capability
Two new capabilities in `web/admin_capabilities.py` `ADMIN_CAPABILITIES`, mirroring `admin.runtime.config.read`/`patch`:
- `admin.case_config.read` — read-only, scope `admin:runtime:read`, safety low, audit metadata, adapter `adapter_read_case_config`.
- `admin.case_config.patch` — mutating, scope `admin:runtime:write`, safety high, audit `request_and_result`, `dry_run_required=True`, `idempotency_required=True`, schema `{patch: _case_config_patch_schema(), **_mutating_controls()}`, adapter `adapter_patch_case_config`.
- Adapters in `web/mcp_admin_tools.py`: `adapter_read_case_config` → `db.get_case_config`; `adapter_patch_case_config` → `_normalize_case_config_patch` (coerces tier keys to int, validates ranges/sums) → dry-run preview or `db.set_case_config` (structured deep-merge + `validate_case_config`). Registered in `ADAPTERS`. Existing `mcp_routes.py` machinery enforces dry_run/confirmation/idempotency — no new MCP plumbing.

## Admin panel UI (`extraShop/admin.html`)
New `<section class="panel-card">` in configs-view after the Runtime-availability card: a limited-event toggle + probability number input + "Save event toggle" button (the one-click live lever), plus a JSON `<textarea>` preloaded with the full blob (partial-patch semantics: only keys present are sent) + "Apply JSON patch" button + status line. JS: `caseConfig` state, `renderCaseConfig()`, `toggleCaseLimitedEvent()`, `saveCaseConfigPatch()`, `saveCaseConfig()` → POST `/api/admin/case-config`. Mirrors the existing `toggleMaintenance`/`saveRuntimeConfig` pattern.

## HTTP endpoint
New `/api/admin/case-config` GET+POST handler `admin_case_config_handler` (admin-gated) near `admin_runtime_config_handler`; GET → `db.get_case_config`, POST `{patch}` → `db.set_case_config` (deep-merge), 400 on `ValueError`. Registered after the runtime-config routes. Also extend `admin_configs_summary_handler` to include `case_config` (feeds `loadConfigs()`).

## Files
- `infrastructure/case_config.py` — limited base fix + blob helpers (`CASE_CONFIG_KEY`, `build_default_case_config`, `merge_case_config_patch`, `fill_case_config_defaults`, `resolve_case_config`, `validate_case_config`, `_coerce_tier_keys`, field-category sets).
- `infrastructure/case_system.py` — `>=1` particle guard + thread `case_config` through `select_rarity`/`check_start_rarity_replacement`/`get_available_rarities_for_tier`/`calculate_particles_for_duplicate`/`roll_tier_upgrade`/`simulate_case_tap_results`/`_generate_single_case_rewards`/`generate_case_rewards`/`process_case_opening`.
- `infrastructure/database.py` — `get_case_config`/`set_case_config` + `_merge_case_config_defaults` + `_ensure_game_settings_table` seeding.
- `web/server.py` — `_case_config_safe`, inject once per handler at all case call sites, `admin_case_config_handler` + route registration, extend `admin_configs_summary_handler`.
- `web/admin_capabilities.py` — `_case_config_patch_schema` + the two new capabilities.
- `web/mcp_admin_tools.py` — `_normalize_case_config_patch` + the two adapters + `ADAPTERS` registration.
- `extraShop/admin.html` — Case-config panel + JS.
- `tests/test_case_system.py` — db-mock `get_case_config→None` + ~15 new regression tests.
- `tests/test_mcp_admin_gateway.py` — `GatewayMCPDB.set_case_config` (mirrors real deep-merge) + coverage-gate update + MCP + HTTP integration tests.

## Verification
1. `python -m pytest tests/test_case_system.py -x` — existing green + new particle/resolve/merge/fill/config tests.
2. `python -m pytest tests/test_mcp_admin_gateway.py -x` — coverage gate updated, schema-invariant test covers new cap, dry-run/confirm/idempotency e2e, validation-rejection, partial-patch preservation, HTTP endpoint.
3. `python -m pytest -x` — no regressions vs the 32/1967 baseline.
4. Live MCP dry-run: `admin.case_config.patch` dry_run=True patch `{'limited_event_active':True}` → confirmation_token; apply dry_run=False + token + idempotency_key → merged; `admin.case_config.read` returns updated blob.
5. Partial-patch safety: patch `{'base_particles_by_rarity':{'limited':100}}` → read confirms common/rare/divine still present (not zeroed).
6. Admin panel: open `/extraShop/admin`, reload configs → Case configuration panel shows limited OFF + prob 0.0015; toggle ON + Save → `admin.case_config.read` shows `limited_event_active` True **without restart**.
7. Case-open smoke: limited_event_active True + prob 1.0, open T5 case with owned limited card → `rewards['particles']` has a limited entry ≥1 (was 0). Default config, T1 case with owned common → ≥1.
8. Restart test: change `limited_event_probability` via MCP, do NOT restart, open a case → new probability in effect.
9. Migration: fresh DB (no `case_config` row) → seeded; `get_case_config` returns defaults; `_merge_case_config_defaults` idempotent.

## Open questions (defaults chosen; confirm to override)
- Limited base value **100** (matches divine). Higher (120/150) would make limited duplicates strictly more rewarding than divine.
- `>=1` guard fires only for base>0 (admin explicit zero stays 0).
- Dedicated `admin.case_config.read` cap + summary extension (both).
- Scope `admin:runtime:write` reused for `admin.case_config.patch`.
- Tier-probability sum tolerance ±0.02 (gross-error guard; `select_rarity` normalizes anyway).
- `additional_properties=True` for tier-keyed sub-dicts in the patch JSON schema (dynamic tier keys).