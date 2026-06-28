# ExtraRLHF MCP — Tool Reference

The `extra-rlhf` MCP server (stdio JSON-RPC 2.0) exposes **25 tools** over the
`rlhf_env` arena. Launch: `python3 -m rlhf_env.mcp_server` (see `../INSTALL.md`
to register in a client). All tools are called via `tools/call` with `arguments`.

Conventions: `match_id` / `group_id` / `battle_id` / `agent_name` are strings.
`p1` = the left/human-side actor; `p2` = the opponent (bot/RL model). `pid` ids
are returned in `start_series.player_ids`.

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
| `get_state` | `match_id` | actor-perspective full state (hand, board, hp, mana, deck, action_history, turn) | Same shape as prod `/api/battle/state`. |
| `get_legal_actions` | `match_id` | `{legal_actions, is_my_turn}` | Empty + `is_my_turn=false` when it's the opponent's turn. |
| `submit_action` | `match_id, action` | `{result:{success,game_over,winner_id}, state, sound_events}` or `{error}` | **Rejected** with `submit_action_unavailable_for_rl_p1` when `p1_actor_type="rl"` (model-vs-model p1 is driven by `advance_bot`). After `play_card`/`attack`/`end_turn` the bot turn auto-runs; after `mana_draw` it does **not**. |
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
0-based position in `get_state.hand`. `target_is_hero=true` targets the hero
face; otherwise `target_id` is a board unit.

## Dataset & trace (V5-style omniscient offline trace)

| Tool | Args | Returns | Notes |
|---|---|---|---|
| `get_v5_dataset_summary` | `group_id` | `{battles_finished, v5_trace_ok_count, battle_tag_distribution, turns_total, actions_total}` | Dataset readiness for a group. |
| `list_v5_groups` | `battle_tag?, limit?` | `{groups:[{group_id,battles_finished,battle_tag,v5_trace_ok_count}]}` | Filter by tag (e.g. `llm-vs-rl`). |
| `get_v5_trace` | `group_id, battle_id, what:"meta"\|"turns"\|"actions"` | `{data, rows_count}` | Raw trace contents. |
| `validate_v5_traces` | `group_id` | `{checked, ok, broken:[{battle_id, issues:[]}]}` | Integrity check (non-null fields, decision_source, actor↔source mapping, turns count). |
| `get_dataset` | `group_id` | `{dataset_jsonl, dataset_rows, per_battle_jsonl}` | Legacy per-action NDJSON. |
| `list_battle_manifests` | `group_id` | `{battles:[...]}` | Per-battle manifest entries. |
| `download_battle_logs` | `group_id, format:"json"\|"zip"` | `{path, size}` | Archive path under sessions dir. |

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
| `starting_player` | `p1\|p2` | `p1` | |
| `deck_strategy_p1` / `deck_strategy_p2` | `random_arenaenv\|custom\|preset` | `random_arenaenv` | Deck source. `custom` needs `custom_deck_p1`/`p2` (card id lists). |
| `custom_deck_p1` / `custom_deck_p2` | [int] | — | Card ids for `custom`. |
| `p1_deck_source` / `p2_deck_source` | `{type:"imported",preset_number:N}` | — | Preset deck (needs prod DB; headless returns empty). |
| `difficulty` | str | ignored | Legacy; models always play `argmax` (max difficulty). |

Response `p1_model`/`p2_model` are `{name,kind,path,weights_hash,weights_version}`;
`degraded=true` + `policy_warnings` signal a fallback (e.g. v5-stub → end_turn).