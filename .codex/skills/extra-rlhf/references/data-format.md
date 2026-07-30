# ExtraRLHF — On-disk Data Format

All artifacts are written under the sessions dir (`RLHF_SESSIONS_DIR`, default
`rlhf_env/sessions/`). Files-only — no DB. Everything is JSON / NDJSON and safe
to inspect, version, or ship to an offline trainer.

```
sessions/
├── agents_index.json                       # codename registry (cross-process)
└── <group_id>/
    ├── manifest.json                        # series manifest + results
    ├── summary.json                         # written on finalize()
    ├── catalog.json                         # V5 battle catalog (group-level)
    ├── battles/
    │   ├── b_<battle_id>.json               # battle_log: actions[], result, models, decks
    │   ├── b_<battle_id>.jsonl              # per-action analytics NDJSON
    │   └── <battle_id>/v5/
    │       ├── meta.json                    # omniscient per-battle metadata
    │       ├── turns.jsonl                  # per-turn snapshots
    │       └── actions.jsonl                # per-action omniscient trace (training data)
    └── ... (one battles/ block per battle)
```

## `manifest.json` (series-level)

```
{
  "group_id": "...",
  "agent_name": "Veceno",                 // codename pinned to this series
  "created_at": "...", "finished_at": "...|null",
  "spec": { "p2_model": "...", "p1_actor_type": "rl|llm|human", "seed": N, ... },
  "results": {
    "battles_planned": N, "battles_finished": M,
    "p1_wins": .., "p2_wins": .., "draws": ..
  },
  "battles_results": [
    { "battle_id": "...", "agent_name": "...", "p1_actor_type": "...",
      "battle_tag": "rl-vs-bot|rl-vs-rl|llm-vs-bot|llm-vs-rl|human-vs-bot|human-vs-rl",
      "winner_user_id": .., "status": "...", "turns": .., "degraded": false }
  ]
}
```

Auto-finalizes when `battles_finished >= battles_planned` (sets `finished_at`,
writes `summary.json`). `agent_name` flows top-level + per-battle + into
`v5/meta.json`.

## `battles/b_<bid>.json` (battle_log)

```
{
  "battle_id": "...", "group_id": "...",
  "models": {
    "p1": { "name": "...", "kind": "llm|rl|human|action_onnx|legacy_onnx|v5|random|...",
            "is_human": bool },
    "p2": { "name": "...", "kind": "...", "is_human": false }
  },
  "decks": { "p1": [...], "p2": [...] },
  "actions": [ { "turn": N, "actor": 1|2, "kind": "play_card|attack|end_turn|mana_draw",
                 "action": {...}, "ok": bool, "ts_ms": .. } ],
  "result": { "winner_user_id": .., "status": "P1_WIN|P2_WIN|DRAW|...", "turns": .. }
}
```

`models.p1.kind` records the real adapter kind (for rl p1) or the actor type
(`llm`/`human`). `name`/`is_human` reflect the actor type for human/llm.

## `battles/<bid>/v5/` — omniscient offline training trace

This is the **training-data surface**: a full-information per-action trace
recorded for every battle, regardless of actor type. "V5" here is the storage
layout name; the trace itself is model-version-agnostic.

### `meta.json`
```
{
  "battle_id": "...", "group_id": "...", "agent_name": "...",
  "p1_is_bot": bool,                // true iff p1_actor_type == "rl"
  "p1_actor_type": "rl|llm|human",
  "battle_tag": "rl-vs-rl|llm-vs-bot|...",
  "bot_policy":   { "name": "...", "kind": "...", "path": "...",
                    "weights_hash": "<sha256[:16]>", "weights_version": .. },  // p2
  "p1_policy": { ... } | null,      // p1 RL model (only for rl p1)
  "winner_user_id": .., "status": "...", "turns": ..,
  "models": { "p1": {...}, "p2": {...} }
}
```
`weights_hash` = `sha256(<onnx file>)[:16]` — proves which checkpoint actually
played (guards against silent fallback). Compare it to the file on disk to
verify no degradation.

### `actions.jsonl` (one row per action — the training rows)
```
{
  "turn": N, "actor_player": 1|2,
  "decision_source": "human|llm|bot|rl",   // who chose this action
  "human_decision_time_ms": 1842|null,      // server-observed; populated only for human
  "legal_actions": [ {...}, ... ],          // full legal set at decision point
  "legal_action_index": K,                  // which legal action was taken
  "action_json": { "type": "...", ... },    // the action (matches legal_actions[K])
  "pre_state":  { ... full omniscient state ... },
  "post_state": { ... },
  "deltas":     { ... per-action diff ... },
  "accepted": bool, "timestamp_ms": ..
}
```
`human_decision_time_ms` measures the observed interval from exposing an
actionable browser state until the next action request arrives. It is reset
after every action and when control returns from the bot. The value therefore
includes normal UI/network latency, which is intentional for human-pacing
modelling. It is always `null` for `llm`, `rl`, and `bot` rows.

Invariants (enforced by `validate_v5_traces`): all fields non-null;
`action_json.type == legal_actions[legal_action_index].type`;
`actor_player`↔`decision_source` consistent (1↔human|llm|rl, 2↔bot);
`post_state` of row N == `pre_state` of row N+1 (hero hp continuity);
row count == `battle_log.actions` length.

### `turns.jsonl`
One row per turn: `{turn, current_player, state_snapshot, ...}`. Row count ==
`meta.turns` == manifest `turns`.

## `agents_index.json` (codename registry, cross-process)

```
{
  "Veceno":   { "group_id": "...", "claimed_at": "...", "finished": false, "released_at": null },
  "Mentalist":{ "group_id": "...", "claimed_at": "...", "finished": true,  "released_at": "..." }
}
```
`finished=true` (not deleted) keeps history queryable (`get_agent_status`
shows `busy=false, status=completed`) while freeing the name for re-claim.
Atomic (tmp+rename) + `fcntl.flock` for cross-process safety. The registry
self-heals: a busy entry whose group manifest is finalized is marked finished
on the next `is_busy`/`claim`/`status` read, even after a process crash.

## Determinism

`seed` controls deck construction and the engine RNG for a series; battle `i`
uses `seed + i*1009`. The engine (`core.engine.ArenaEnvironment`) is pure:
`(state_in, action) → (state_out, ok)`. Replaying the same seed + action stream
reproduces the trace byte-for-byte (modulo wall-clock `timestamp_ms`).
