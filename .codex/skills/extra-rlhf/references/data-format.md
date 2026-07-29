# ExtraRLHF — On-disk Data Format

Headless battle artifacts are written beneath `RLHF_SESSIONS_DIR` (default
`rlhf_env/sessions/`). Private training exports are written beneath the
separate `RLHF_DATASETS_DIR` (default `datasets/`). Everything is JSON/NDJSON,
but privacy-safe does not mean publishable: ReturnClock and production-derived
artifacts must remain in access-controlled training storage.

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
  "human_decision_time_ms": 1842|null,      // server-observed; human rows only
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
after every action and when control returns from the bot. It intentionally
includes ordinary UI/network latency and is always `null` for `llm`, `rl` and
`bot` rows. Metronome training uses this timing only together with the
corresponding pre-action state. CPU time spent advancing a headless engine,
LLM response latency and injected/synthetic delays are not valid human labels.

Invariants (enforced by `validate_v5_traces`): required action/state fields are
present;
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

## Private cross-contour exports (`datasets/`)

All MCP paths below are relative to or contained by `datasets_dir`; symlinks and
path traversal are rejected. Files use mode `0600`. A fresh destination is
built in a sibling temporary path and exposed by same-filesystem rename.
Overwrite can roll back ordinary caught failures, but it is not crash-atomic
across `SIGKILL` or power loss; training handoffs use a new versioned path with
`overwrite=false`, validate it, and only then promote an external pointer.

### Production V5 transport and materialized directory

`export_v5_training_dataset` writes
`extraarena_v5_dataset_export_v1`: a header followed by one complete terminal
V5 battle bundle per line. Identities are fixed side pseudonyms (`p1=1`,
`p2=2`), while every structurally declared `battle_id`/`match_id` is replaced
by one export-local `record_<hex>` alias
(`record_id_scheme=random_per_export_record_ids_v1`). This also removes IDs
such as `tutorial-<raw-user-id>`; raw player or record IDs cannot be requested.
Use
`materialize_v5_training_dataset` to convert it into the canonical
`rlhf_v5_storage_v1` group directory with `manifest.json` and per-battle
`v5/{meta,turns,actions}.jsonl`.

Materialization validates the V5 state/action contract plus attached
Metronome/TimeStamp fields before publishing. A transport file is not a
trainer input until `validate_training_export` passes for the specific
training contour.

Headless `validate_v5_traces` reports readiness dimensions separately:

- `v5_policy_training_ready` gates accepted V5 policy targets;
- `training_ready` is a backward-compatible alias for that policy gate and
  `training_ready_scope` is `v5_policy_only`;
- `metronome_training_ready` requires observed, uncensored human
  decision-time labels aligned with pre-action states;
- `timestamp_training_ready` requires real production battle-time labels, not
  headless CPU/wall-clock duration.

A policy-ready headless group may also feed Nemesis Lite after the Nemesis
eligibility and split-group checks. It does not, by itself, make either timing
sub-model ready.

#### TimeStamp feature/label boundary

TimeStamp inputs are restricted to the prebattle deck or deck pair,
`starting_player`, and explicitly approved features whose timestamps prove they
were available before battle start. `duration_seconds`, `turns`, `finished_at`,
and any derived values are target labels or audit metadata only and are
forbidden as model inputs. The stored `timestamp_features` object currently
co-locates input descriptors and labels for audit, so a loader must project an
explicit input allowlist—passing that object or all of `meta` wholesale is a
target-leakage failure.

### Nemesis

`export_nemesis_training_dataset` accepts exactly one source: a V5 transport
`input_path` or a completed headless `group_id`. It extracts one terminal
`meta.nemesis_record` per battle and writes
`extraarena_nemesis_dataset_export_v1`. Each battle has one shared base:

- `features.base`: deck pair, starting player, ruleset/catalog provenance and
  side metadata used by Nemesis Lite;
- optional `features.extended`: the profile/history snapshot captured at the
  feature cutoff for standard Nemesis;
- `quality.eligible_lite`, `quality.eligible_standard`, `sample_weight` and
  explicit exclusion reasons;
- terminal `label` and a stable deck-pair `provenance.split_group`.

Human-vs-human can train standard Nemesis when both cutoff snapshots exist.
Human-vs-bot and model-vs-model are Lite training data. Human-vs-bot keeps its
masked extended/provenance payload for audit and future domain-aware research,
but the current canonical Standard trainer must not consume it.
Exports always use side pseudonyms and keep exactly one record per battle—there
are not separate Lite and standard copies. Pseudonymized Nemesis exports use
the same export-local opaque record-ID scheme. Their grouping-only
`privacy.player_group_aliases` retain export-local player identity for grouped
standard splits without entering model features. Validators reject native
`battle_id`/`match_id` values or missing group aliases under a pseudonymized
header.

Run `split_nemesis_training_dataset` before training. It always materializes
the Lite deck-pair-grouped train/validation/test view. When the Standard
preconditions pass, the same `extraarena_nemesis_split_v1` package also
contains three Standard views:

- standard player-disjoint train/validation/test;
- standard chronological train/validation/test;
- standard deck-pair-grouped train/validation/test.

The player-disjoint view deterministically partitions export-local player
aliases and excludes cross-partition battles; the manifest records every such
exclusion and the validator recomputes both assignments and leakage checks.
Standard materialization requires at least six distinct players, three
pairwise-disjoint human-human battles, three distinct matchup groups and three
distinct cutoff cohorts. If those gates do not pass, the package remains
Lite-ready and returns explicit `standard_readiness_blockers`; it does not
silently promote human-bot data.
The source export therefore keeps `training_ready_standard=false` until this
artifact exists. Lite three-way holdout readiness still requires at least
three distinct `provenance.split_group` values. Because player aliases are
export-local, do not concatenate overlapping exports and assume identity
continuity; create one versioned source window (or perform an explicit overlap
audit) before splitting.

### ReturnClock

`export_returnclock_training_dataset` writes
`extraarena_returnclock_dataset_v1`:

1. a header with exact `feature_columns`, quality summary, survival horizon,
   meaningful-session rule and `pseudonymization_key_id`;
2. chronological examples with `features` (cutoff-safe only), `label`
   (event/censoring), `post_cutoff` (treatment/exposure; forbidden to the
   estimator), `user_id_hash` (grouping only), and
   `prediction_cutoff_at` (temporal split/audit only).

A session is meaningful iff
`(duration_seconds >= 120 AND screen_count >= 2) OR battles_completed > 0 OR
cases_opened > 0`.

`user_id_hash` is HMAC-derived and pseudonymous. The secret is read only from
the server environment; the non-secret key id tracks rotation. Never concatenate
exports made under different key ids as though their user groups were stable.

Use `split_returnclock_training_dataset` to create train/validation/test JSONL.
It groups by user and orders groups by first cutoff, then runs a leakage gate.
Do not use a random row split. The raw export may contain treated intervals for
audit, but the published split files are organic-only and every row must satisfy
`post_cutoff.organic_candidate=true`; the manifest records the exact
`training_filter` and excluded treated count. Natural-return training and
evaluation consume only these split files, never the mixed raw export. Causal
send-time training remains blocked until randomized no-send/control assignments
exist.

Production extraction reads sessions, decisions, and delivery events with
keyset pagination (pages of at most 50,000; ceiling 1,000,000 rows per stream)
inside one repeatable-read snapshot. Exclusive `end_at` bounds event
time/censoring. A later `ingested_before` independently bounds session,
decision, and delivery row creation; this admits late pre-boundary events
without erasing assignments whose status changed later. Apply the safety lag
when `end` is omitted; use an explicit historical `end` as-is.
If any stream reaches the ceiling, stop the handoff and choose an audited
complete window—do not concatenate independently censored exports or claim
completeness. Export and split currently materialize the selected bounded
window in memory, so size it against available RAM.

## Determinism

`seed` controls deck construction and the engine RNG for a series; battle `i`
uses `seed + i*1009`. The engine (`core.engine.ArenaEnvironment`) is pure:
`(state_in, action) → (state_out, ok)`. Replaying the same seed + action stream
reproduces the trace byte-for-byte (modulo wall-clock `timestamp_ms`).
