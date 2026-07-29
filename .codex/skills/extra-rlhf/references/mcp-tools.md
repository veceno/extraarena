# ExtraRLHF MCP — Tool Reference

The `extra-rlhf` MCP server (stdio JSON-RPC 2.0) exposes headless arena tools
and a private cross-contour dataset toolbox. Launch with the pinned Python 3.13
interpreter (see `../INSTALL.md`). All tools are called via `tools/call` with
`arguments`.

Conventions: `match_id` / `group_id` / `battle_id` / `agent_name` are strings.
`p1` = the left/human-side actor; `p2` = the opponent (bot/RL model). `pid` ids
are returned in `start_series.player_ids`.

Wire contract: successful and failed `tools/call` results contain JSON text in
`content[0].text`, the same object in `structuredContent`, and `isError`.
Prefer `structuredContent`; do not expect a non-standard `content[].data`.

## Series lifecycle (orchestration)

| Tool | Args | Returns | Notes |
|---|---|---|---|
| `start_series` | `spec` (object) | `{group_id, match_id, battle_id, battles_planned, player_ids, opponent, p1_actor_type, battle_tag, agent_name, p1_model, p2_model, is_ended, winner_id, policy_warnings, degraded}` | Creates a series + plays battle 0 headless. Auto-advances bot turns; for `p1_actor_type="rl"` plays the whole battle to game_over. |
| `next_battle` | `group_id` | `{match_id, battle_id, ...}` or `{status:"series_complete"}` | Advance to next battle in the series. |
| `finish_series` | `group_id` | full `manifest` | Explicit close: finalizes manifest + releases the agent codename + frees the match. |
| `list_active_series` | `{}` | `{count, agents:[{agent_name,group_id,status,battles N/M,wins,losses,draws,p1_actor_type,opponent_model,current_match_id}], by_model:[{model,groups,wins,losses,draws}]}` | Fleet view. Reaps completed series first (self-heal). |
| `get_agent_status` | `agent_name` | `{agent_name,busy,group_id,current_match_id,battles_finished,battles_planned,wins,losses,draws,decks,opponent_model,p1_actor_type,status}` | Aggregate from the group manifest. Reaps + re-reads. |
| `get_battle_group_status` | `group_id` | group status | Reaps first. |
| `get_battle_group_manifest` | `group_id` | full `manifest.json` | |
| `list_battle_groups` | `{}` | `{groups:[...]}` | All groups (active + completed). |
| `list_models` | `{}` | `{models:[{name,path,kind,description,extra}]}` | Registry + built-in baselines. |
| `register_custom_model` | `name, path, kind` | `{registered, name, path, kind, detected_kind}` | In-memory add. `path` must resolve under the models dir (no traversal). `kind`: `auto|action_onnx|legacy_onnx|v5|random|greedy_face|end_turn`. |
| `list_preset_decks` | `{}` | `{presets:[], note}` | Headless has no DB → empty; use `deck_strategy="random_arenaenv"` or `"custom"`. |

## Player (one battle, p1 = human/llm)

| Tool | Args | Returns | Notes |
|---|---|---|---|
| `get_match_status` | `match_id` | `{match_id,group_id,battle_id,agent_name,turn,is_ended,winner_id,current_player_id,is_my_turn,p1_actor_type,battle_tag}` | Lightweight poll (no full state). Reaps completed series. |
| `get_state` | `match_id`, `compact?`, `history_limit?` | actor-perspective state: nested `player`/`opponent`, top-level legal actions/history/turn/result | Use `compact=true,history_limit=8` for LLM play; it keeps every decision field while bounding history. |
| `get_legal_actions` | `match_id` | `{legal_actions:[{legal_action_index,...}], is_my_turn}` | Empty + `is_my_turn=false` when it is the opponent's turn. |
| `submit_action` | `match_id`, `legal_action_index` (preferred) or `action`, `compact_response?`, `history_limit?` | `{result:{success,game_over,winner_id}, state, sound_events}` or `{error}` | LLMs should use the index from the latest state and `compact_response=true`. **Rejected** for `p1_actor_type="rl"`. Ordinary actions auto-run p2; `mana_draw` does not pass. |
| `advance_bot` | `match_id` | `{status:"ok", is_ended, winner_id}` or `{status:"not_bot_turn"}` | Runs one opponent turn. For `p1_actor_type="rl"` also steps the p1 RL model when it's p1's turn. |
| `surrender` | `match_id` | `{result:{game_over,winner_id}, state}` | **Rejected** with `surrender_unavailable_for_rl_p1` for rl p1. |
| `get_action_history` | `match_id, limit?` | `{actions:[{turn,actor,kind,action_dict,ok}], count}` | Replay without re-fetching full state. |

### `action` payload shapes (`submit_action`)

```
{"type":"play_card", "hand_index":N, "target_position":N?, "target_id":id?, "target_is_hero":bool?}
{"type":"play_card", "card_id_from_hand":N, ...}            # alt id form
{"type":"attack", "attacker_id":id, "target_id":id, "target_is_hero":bool?}
{"type":"end_turn"}
{"type":"mana_draw"}                                          # draw a card for mana; does NOT pass the turn
```

`id` fields accept int or string (hero ids are strings). `hand_index` is the
0-based position in `get_state.player.hand`. `target_is_hero=true` targets the hero
face; otherwise `target_id` is a board unit.

## Dataset & trace (V5-style omniscient offline trace)

| Tool | Args | Returns | Notes |
|---|---|---|---|
| `get_v5_dataset_summary` | `group_id` | structural counts, accepted/rejected action counts, catalog provenance, degraded battles, per-group semi-synthetic quality, per-contour readiness | A trace row is a policy target only when `accepted is True`; rejected actions remain audit evidence. |
| `list_v5_groups` | `battle_tag?, limit?` | groups plus pooled quality over the returned selection | Use a large enough limit for the campaign gate; filter e.g. `llm-vs-rl`. |
| `get_v5_trace` | `group_id, battle_id, what:"meta"\|"turns"\|"actions", offset?, limit?` | paginated `{data, rows_count, total_rows, offset, limit, truncated}` | `limit` is bounded; do not pull unbounded traces into an LLM context. |
| `validate_v5_traces` | `group_id` | `{checked, ok, broken, degraded, training_ready, training_ready_scope, v5_policy_training_ready, metronome_training_ready, timestamp_training_ready, metronome_observed_labels, current_catalog_hash, current_card_count}` | Deep action/state/terminal continuity plus current catalog hash/card-count gate and separate contour readiness. |
| `get_dataset` | `group_id` | `{dataset_jsonl, dataset_rows, per_battle_jsonl}` | Legacy per-action NDJSON. |
| `list_battle_manifests` | `group_id` | `{battles:[...]}` | Per-battle manifest entries. |
| `download_battle_logs` | `group_id, format:"json"\|"zip"` | `{path, size}` | Archive path under sessions dir. |

For headless groups, `training_ready` is a backward-compatible alias for
`v5_policy_training_ready`, and `training_ready_scope` is
`v5_policy_only`. It requires at least one accepted target, no broken/degraded
battle, valid legal-action targets, consistent pre/post-state continuity,
terminal agreement, and matching catalog provenance. It can gate policy
training and is a valid source for separately eligible Nemesis Lite rows.

Metronome and TimeStamp are independent gates. Require
`metronome_training_ready=true` / `timestamp_training_ready=true` and the
corresponding observed production labels before training them. Headless
wall-clock/CPU duration, LLM response latency and synthetic actor delays are
not human timing labels. Structural validity is also not behavioral quality:
LLM campaigns must pass the pooled quality gate described in the L1 skill.

## Private training-data administration

All paths are confined to `--datasets-dir`; absolute paths are allowed only
when they still resolve inside that root. Symlinks and `..` escapes are
rejected.

| Tool | Args | Returns / gate |
|---|---|---|
| `get_training_data_status` | `{}` | Datasets root, inventory by kind, headless counts, production/salt readiness, and the explicit causal ReturnClock blocker. Never returns DSNs or secrets. |
| `list_training_exports` | `kind?`, `limit?` | Bounded inventory of `returnclock`, `returnclock_split`, `nemesis`, `nemesis_split`, `v5_export`, `v5_materialized` artifacts. |
| `inspect_training_export` | `path` | Kind, size, checksum, mode and bounded header/manifest metadata; no dataset rows. |
| `validate_training_export` | `path` | Deep schema/privacy/count/provenance diagnostics, contour summary and `training_ready`. Run after every export/materialization/split input. |
| `export_v5_training_dataset` | `output`, `days?=30`, `limit_battles?=1000`, `overwrite?=false` | Opt-in, read-only production export of complete terminal V5 bundles; players use side pseudonyms and battle/match IDs use export-local opaque aliases; private `0600`; a fresh file is promoted by same-filesystem replace after validation. |
| `materialize_v5_training_dataset` | `input_path`, `output_dir`, `group_id?`, `overwrite?=false` | Validated `rlhf_v5_storage_v1` directory with manifest and canonical V5/Metronome/TimeStamp traces. |
| `export_nemesis_training_dataset` | exactly one of `input_path` / `group_id`, plus `output`, `overwrite?=false` | Extracts canonical terminal `meta.nemesis_record` rows from a V5 transport export or completed headless group; fixed side pseudonyms plus export-local opaque record IDs, no raw-player switch. |
| `split_nemesis_training_dataset` | `source`, `output_dir`, `train_fraction?=0.70`, `validation_fraction?=0.15`, `overwrite?=false` | Always publishes a Lite deck-grouped assignment. It additionally publishes all three Standard assignments (player-disjoint, chronological, deck-grouped) only when Standard gates pass: >=6 players, three pairwise-disjoint human-human battles, three matchup groups and three cutoff cohorts. Otherwise it remains Lite-ready and returns `standard_readiness_blockers`. The full cross-edge fingerprint ledger stays in the private manifest; MCP responses expose exact counts plus a bounded sample. Aliases never enter features. |
| `export_returnclock_training_dataset` | `output`, `start?`, `end?`, `horizon_hours?=168`, `safety_lag_minutes?=10`, `min_analytics_version?=2`, `limit?=50000`, `overwrite?=false` | Opt-in read-only export. The HMAC secret comes only from server env; future `end`, truncation, or schema/privacy failure aborts publication. Keyset pages are capped at 50k and the fail-closed ceiling is 1m rows per stream. |
| `split_returnclock_training_dataset` | `source`, `output_dir`, `train_fraction?=0.70`, `validation_fraction?=0.15`, `overwrite?=false` | Writes an organic-only manifest + train/validation/test JSONL, grouped by user and temporal by first cutoff; records treated/boundary exclusions and fails on leakage. |

Production reads require `RLHF_ENABLE_PRODUCTION_DATASETS=1` (or
`--enable-production-datasets`). ReturnClock additionally requires
`RETURNCLOCK_DATASET_SALT` and non-secret
`RETURNCLOCK_DATASET_SALT_KEY_ID`; the salt value must never be an MCP argument.
The exporter uses a repeatable read and fail-closed limits. When `end` is
omitted, the ingestion safety lag chooses the event-time boundary; an explicit
historical `end` is used as-is. The exporter records a later
`ingested_before` row-creation watermark so late status updates cannot erase
an assignment. Its output is pseudonymized—not anonymous. Export and split
currently materialize the selected bounded window in memory, so size large
windows against available RAM.
Raw ReturnClock readiness already mirrors splitter preconditions: at least
three organic user groups with three strictly ordered first-cutoff cohorts.

### Training-ready sequence

1. `get_training_data_status`; resolve only blockers relevant to the contour.
2. Export to a new path (`overwrite=false` by default).
3. `inspect_training_export` and `validate_training_export`; require `ok` and
   the readiness field for the contour being trained. For headless V5 policy,
   that field is `v5_policy_training_ready`, not either timing-model field.
4. Production V5 transport: materialize, then validate the directory again
   (headless groups are already canonical). Nemesis: invoke
   `split_nemesis_training_dataset` and preserve every assignment it
   materializes. Lite-only inputs are valid Lite handoffs; require
   `training_ready_standard=true` before expecting all three Standard views.
   No single split is claimed to cover deck, player and temporal
   generalization. ReturnClock: invoke the dedicated grouped-temporal splitter
   and preserve the split manifest.
5. Record checksum, format/version, catalog/weights provenance, ReturnClock key
   id, exclusions/sample weights, and split manifest with the training run.

Do not train from a partially written file, an artifact with `include_players`,
a mismatched catalog, a degraded V5 battle, a Nemesis row not eligible for the
chosen variant, or a ReturnClock row field outside `header.feature_columns`.
Do not infer Metronome/TimeStamp readiness from a headless
`training_ready=true`.

For directory bundles, use a new versioned destination with
`overwrite=false`. Fresh-path publication is temp-directory plus
same-filesystem rename; overwrite has rollback on handled errors but is not
crash-atomic under `SIGKILL`.

## `start_series` `spec` fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `battles_planned` | int | 1 | Series length. |
| `p1_actor_type` | `human\|llm\|rl` | `llm` | p1 kind. `llm` = MCP model plays via `submit_action`; `rl` = our model auto-plays (model-vs-model, no `submit_action`); `human` = browser. |
| `p2_model` | str or `{name,path,kind}` | `random` | Opponent. String = registry name; object = explicit. |
| `p2_model_path` / `p2_model_kind` | str | — | Flat form for custom opponent by path. |
| `p1_model`, `p1_model_path`, `p1_model_kind` | str | — | p1 RL model (only when `p1_actor_type="rl"`). |
| `agent_name` | str | auto | Codename to pin the series to (see `concepts.md`). Auto-assigned from the pool if omitted. |
| `seed` | int | 0 | Deterministic deck/order seed. Each battle uses `seed + index*1009`. |
| `starting_player` | `random\|p1\|p2` | `random` | |
| `deck_strategy_p1` / `deck_strategy_p2` | `random_arenaenv\|custom\|preset` | `random_arenaenv` | Deck source. `custom` needs `custom_deck_p1`/`p2` (card id lists). |
| `custom_deck_p1` / `custom_deck_p2` | [int] | — | Card ids for `custom`. |
| `p1_deck_source` / `p2_deck_source` | `{type:"imported",preset_number:N}` | — | Preset deck (needs prod DB; headless returns empty). |
| `difficulty` | str | ignored | Legacy; models always play `argmax` (max difficulty). |

Response `p1_model`/`p2_model` are `{name,kind,path,weights_hash,weights_version}`;
`degraded=true` + `policy_warnings` signal a fallback (for example a model-load
error → `end_turn`).
