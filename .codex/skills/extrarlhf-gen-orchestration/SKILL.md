---
name: extrarlhf-gen-orchestration
description: "Use when generating semi-synthetic training data: plan and dispatch a fleet of battle series across codename agents, pick models/opponents/decks/seeds, monitor progress, validate trace integrity, and ship the dataset. Spawns extrarlhf-player agents for human/llm p1 battles; uses rl-vs-* auto-play for model-vs-model. Model-version agnostic."
---

# ExtraRLHF — Level 1: Data-Generation Orchestrator

You turn a **campaign spec** ("N battles of match-kind K vs model M, diverse
seeds/decks") into a validated on-disk dataset. You manage a **fleet of
codename agents**, each running a series. You do **not** play battles yourself
— for `human`/`llm` p1 you spawn **Level 2** (`extrarlhf-player`) agents; for
`rl` p1 the battles auto-play (model-vs-model, no L2 needed). Keep MiniMax M3
campaigns bounded to roughly 4–8 concurrent workers until the quality smoke
passes; nested `claude -p` per action is prohibited.

## Your tools (level-1 set)

- **Create/advance/close:** `start_series`, `next_battle`, `finish_series`.
- **Fleet monitor:** `list_active_series`, `get_agent_status`,
  `get_battle_group_status`, `list_battle_groups`.
- **Dataset/integrity:** `get_battle_group_manifest`, `get_dataset`,
  `list_battle_manifests`, `download_battle_logs`, `get_v5_dataset_summary`,
  `list_v5_groups`, `validate_v5_traces`.
- **Models/decks:** `list_models`, `register_custom_model`, `list_preset_decks`.

## Anatomy of a `start_series` spec

```jsonc
{
  "battles_planned": 10,
  "p1_actor_type": "llm",                 // human | llm | rl
  "p2_model": "extra-lr-current",          // registry name, OR {name,path,kind}, OR use p2_model_path+kind
  "p2_model_path": "/abs/ai/models/foo.onnx",  // explicit (bypasses auto-detect)
  "p2_model_kind": "action_onnx",          // action_onnx | legacy_onnx | v5 | random | greedy_face | end_turn
  "p1_model": "...", "p1_model_path": "...", "p1_model_kind": "...",  // only for p1_actor_type="rl"
  "agent_name": "Veceno",                  // pin a codename; omit for auto-assign
  "seed": 101,                             // battle i uses seed + i*1009
  "starting_player": "p1",                 // p1 | p2
  "deck_strategy_p1": "random_arenaenv",   // random_arenaenv | custom | preset
  "custom_deck_p1": [..card ids..]         // for "custom"
}
```
`start_series` returns `group_id`, `match_id`, `battle_tag`, `agent_name`,
`p1_model`/`p2_model` (`{name,kind,path,weights_hash}`), `is_ended`, `winner_id`,
`degraded`, `policy_warnings`. **Always check `degraded`** — if true, the
opponent fell back to a safe policy and the trace is not what you asked for.

## Match-kind matrix (pick for data diversity)

| Goal | `p1_actor_type` | p2 | `battle_tag` | L2 needed? |
|---|---|---|---|---|
| LLM semi-synthetic vs real model | `llm` | onnx model | `llm-vs-rl` | **yes** (L2 plays p1) |
| Model-vs-model self/peer play | `rl` | onnx model | `rl-vs-rl` | no (auto-play) |
| Cheap rollouts / sanity | `llm` or `rl` | `random`/`greedy_face`/`end_turn` | `*-vs-bot` | yes for llm; no for rl |
| Human reference data | `human` | onnx model | `human-vs-rl` | yes (human/L2) |

`battle_tag` depends only on p2's kind. Slice the dataset with
`list_v5_groups battle_tag=llm-vs-rl`. High-value: `*-vs-rl`; cheap: `*-vs-bot`.

## Fleet pattern (many agents in parallel)

1. **Plan codenames**: assign distinct names from the pool
   (`Veceno`, `Mentalist`, `Pvwell`, `Sinaf`, `Movi`, `Ilya`, `Oguzok`,
   `Milita`, `dranik`, `sukunyata`, `absolute`, then card names). One series
   per agent. Omit `agent_name` for auto-assign if you don't care.
2. **Dispatch**: assign every `llm` L2 worker the full lifecycle. Its persistent
   MCP session calls `start_series`, plays, and finishes/advances its own group.
   Do **not** create a match in L1 and hand its process-local `match_id` to a
   different MCP process.
   For `rl` p1, the battle auto-plays to game_over inside `start_series`.
3. **Advance multi-battle series**: `next_battle(group_id)` until
   `series_complete`. (For `llm` p1, each new `match_id` → a fresh L2 agent.)
4. **Monitor**: `list_active_series` for the fleet view (battles N/M, wins/
   losses, by-model grouping). `get_agent_status(name)` for one agent.
5. **Close**: series auto-finalize on completion; call `finish_series` to
   explicitly release an agent early, or just let the read-path reap free it.

**Re-claim is automatic**: a completed series' codename is freed on the next
`get_agent_status`/`get_match_status`/`list_active_series` (self-healing reap).
No mid-series release — a 1/3 series stays busy.

## Validation gate (before declaring the dataset ready)

For every campaign group:
1. `validate_v5_traces group_id=<gid>` → require `ok == checked`, `broken == []`.
2. `get_v5_dataset_summary group_id=<gid>` → confirm `actions_total` and
   `battle_tag_distribution` match the campaign plan.
3. Cross-check `degraded`: `get_v5_dataset_summary.battles[]` and the manifest
   must contain no degraded/policy warnings; check
   `v5/meta.json bot_policy.weights_hash`
   == `sha256(<onnx>)[:16]` for the opponent (proves the real model played).
4. Require `accepted_training_rows > 0` and treat
   `rejected_audit_rows > 0` as an investigation blocker, not as targets.
5. For LLM campaigns call
   `list_v5_groups battle_tag=<campaign-tag> limit=10000` and require pooled
   quality: `llm_battles>=50`, zero rejected/error decisions, zero degraded
   battles and `p1_win_rate_wilson_lower_95 > 0.03`. Per-group quality is
   diagnostic; a one-battle group cannot pass alone. Report
   end-turn-with-attack/play and mana-draw counters.
6. `download_battle_logs group_id=<gid> format=zip` → ship the archive path to
   the trainer (Level 0).

If a group is broken or degraded, re-run it (new seed) or drop it — do not ship.

## Model selection

- **By name** (registry): `p2_model: "extra-lr-current"` — uses
  `list_models` entries; auto-detects kind (needs layer A for onnx).
- **By path+kind** (custom / when layer A absent): `p2_model_path` +
  `p2_model_kind` — bypasses auto-detect; the explicit way to play a specific
  onnx checkpoint.
- **Nested**: `p2_model: {"name":"foo","path":"...","kind":"action_onnx"}`.
- **Register at runtime**: `register_custom_model name path kind` then reference
  by name (in-memory; lives until process restart).

Adapter kinds: `action_onnx`/`v4` (action-conditioned), `legacy_onnx` (V3-style),
`v5` (implemented 7128/601 + mana-draw-head adapter), `random`/`greedy_face`/
`end_turn` (baselines). See `../extra-rlhf/references/concepts.md`.

## Decks

`random_arenaenv` (default, balanced random), `custom` (`custom_deck_p1`/`p2`
card-id lists), `preset` (needs prod DB — headless returns empty; use the other
two). For targeted matchups, `custom` lets you fix both decks.

## Patterns vs anti-patterns
- ✅ Many short series across codenames (parallel, isolated, re-claimable).
- ✅ Varied seeds/decks/starting_player for diversity.
- ✅ Validate every group before shipping.
- ❌ One giant `battles_planned=1000` series (no parallelism, one crash loses all).
- ❌ Ignoring `degraded` — a fallback trace is silently wrong.
- ❌ Treating structural validity as behavioral quality. A timeout followed by
  a silent `end_turn` must invalidate the battle.
- ❌ Shipping LLM rows directly into the default Phase-C bridge. Phase C is
  human-only unless the accepted decision sources are explicitly changed
  **after** the campaign quality gate passes.
- ❌ `submit_action` on an `rl` p1 series (rejected) — rl auto-plays; use L2 only
  for human/llm p1.

## Handoff to Level 0
Report: campaign spec, list of `group_id`s, per-group
`{battles_finished, battle_tag_distribution, actions_total, validate_ok}`,
`download_battle_logs` archive path. Flag any degraded/broken groups.

## See also
- `../extra-rlhf/SKILL.md` — umbrella + setup
- `../extra-rlhf/references/mcp-tools.md` (full tool args/returns),
  `data-format.md`, `concepts.md`
- `../extrarlhf-player/SKILL.md` (L2) — for `human`/`llm` p1 battles
- `../extrarlhf-pipeline-orchestration/SKILL.md` (L0) — your caller
