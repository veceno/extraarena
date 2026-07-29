# Production battle dataset: V5 contract

This document describes the server-side dataset produced by live ExtraArena
battles. It is a private, full-information training surface; it is not the
state returned to a browser.

## Scope

The production collector records both supported human cohorts:

- `human-vs-bot`: a real player against a game bot, including every V5-family
  model;
- `human-vs-human`: two real players in the same match.

A human seat taken over by a replacement bot remains a human seat in battle
metadata, but the takeover actions are explicitly marked as automated. Timeout
actions follow the same rule. They must never become human behavior labels.

The deterministic onboarding tutorial is excluded. It rebuilds and silently
replays a scripted graph between steps, so treating it as an ordinary
human-vs-bot trajectory would corrupt both action provenance and TimeStamp
labels. A temporary unpersisted game bot may have a negative internal user ID;
that seat is still valid and is pseudonymized exactly like a persisted bot.

The storage schema is `rlhf_v5_storage_v1`. It is shared with the ExtraRLHF
offline bridge, so a production export does not need a V3/V4 conversion step.

## Lifecycle and durability

1. When a match engine is initialized, an in-memory journal is created with
   status `ongoing`. The private journal contains both hands, both ordered
   decks, card levels and the structured V5 history tape.
2. The actual battle start is anchored when all required clients report
   `client_ready`. This anchor, rather than engine construction or queue time,
   is the start of the TimeStamp label.
3. A checkpoint is upserted after readiness and after each attempted action.
   Action identity is `(battle_id, seq)`, so retrying a checkpoint is
   idempotent and cannot duplicate a training row.
4. Every accepted or rejected attempt is finalized with both its pre-state and
   post-state before the checkpoint is durable.
5. A normal result or surrender seals the whole battle as `p1_win`, `p2_win`,
   `draw` or `stalemate`. Only a complete terminal battle is exportable.
6. A server reload, abandoned engine or other non-result teardown marks the
   journal `aborted`. Stale `ongoing` journals are marked aborted by
   maintenance. Aborted and ongoing battles are never exported for training.

Database retention is whole-battle retention. Pruning a journal removes its
action rows through the battle foreign key; it does not truncate trajectories
at an arbitrary action limit.

## Canonical files

The materialized dataset has one group manifest and one V5 directory per
battle:

```text
<group>/
├── manifest.json
└── battles/
    └── <battle_id>/
        └── v5/
            ├── meta.json
            ├── turns.jsonl
            └── actions.jsonl
```

`manifest.json` lists every complete battle in `battle_ids` and
`battles_results`. Each result has `v5_trace_ok: true`, `v5_dir` and
`v5_meta_path`, which are the discovery fields used by the offline loaders.

`meta.json` contains the terminal result, actor types, model provenance,
initial decks, TimeStamp features and labels, and sparse `control_events` for
AFK/reconnect/surrender transitions. `turns.jsonl` contains a full omniscient
snapshot at each turn boundary. `actions.jsonl` contains the ordered
action/audit rows with contiguous `seq` values starting at one.

The authoritative history input for V5 is `v5_history_events` in every state
snapshot. The older free-form `history` field is retained for compatibility
but must not replace it.

## Action and training-label semantics

Every attempted action records:

- `action_json`, `action_native`, the complete `legal_actions` set and
  `legal_action_index`;
- `pre_state`, `post_state`, `deltas`, `accepted` and `error`;
- `actor_user_id`, `actor_player`, `decision_source` and `control_source`;
- a monotonic `timestamp_ms` and the timing fields described below.

`decision_source` is the stable chooser taxonomy:
`human | llm | bot | rl`. `control_source` explains live control, including
`human`, `bot`, `replacement_bot` and `timeout`.

Rejected rows are intentionally preserved. They expose stale UI requests,
out-of-turn attempts, invalid targets and other integration defects. A
rejected row must have `accepted: false`, a non-empty `error`, and identical
pre/post state.

Behavior-cloning and offline-RL targets use only rows where:

```python
row.get("decision_source") == "human" and row.get("accepted") is True
```

The identity check is deliberate. Truthy strings, missing values and rejected
attempts are not labels. Automated replacement/timeout actions are audit and
environment data, not demonstrations of the player.

## Metronome timing

Human decision time is measured from delivery of an actionable state to
arrival of the next request at the server:

- `human_decision_time_ms`: usable label in the current training window;
- `human_decision_time_raw_ms`: raw observed duration when available;
- `decision_time_censored`: whether the label is unsafe to train on;
- `decision_censor_reason`: for example `not_observed`,
  `outside_training_window`, disconnect or reconnect.

Only an uncensored human row has a non-null training label. The current label
window is 100–25,000 ms. Delivery gaps, reconnects and observations outside
the window stay in the audit trail as censored rows rather than being silently
clamped.

Automated actions additionally record:

- `metronome_prediction_ms`: Metronome V1 prediction;
- `metronome_applied_ms`: delay actually applied by the game;
- `metronome_fallback_used`: whether runtime fallback replaced the prediction.

Timeout end-turns use `decision_source: "bot"` and
`control_source: "timeout"`. They are not Metronome predictions and are not
human timing labels.

## TimeStamp labels

TimeStamp is battle-level, not action-level. The label is
`meta.duration_seconds`, measured monotonically from the `client_ready` anchor
to terminal sealing. The audit fields are:

- `meta.started_at`, `meta.finished_at`, `meta.duration_seconds`;
- `meta.start_metadata.client_ready_anchored`;
- `meta.timestamp_features`, including both deck sizes, starting side, final
  turn count and duration;
- `meta.p1_deck` and `meta.p2_deck`, with card IDs and levels.

The mono model may consume one deck plus population context. The duo model may
consume both concrete decks. The same terminal duration is the label for both.
Queueing, prebattle readiness and abandoned matches are excluded.

## Export and materialization

The admin endpoint is:

```text
GET /api/admin/analytics/dataset/export
    ?days=30
    &limit_battles=1000
    &include_players=0
```

The response is `application/x-ndjson`:

1. a `record_type: "header"` record with format
   `extraarena_v5_dataset_export_v1`;
2. one `record_type: "battle"` record per complete terminal battle, containing
   the whole `meta`, `turns` and `actions` bundle.

`limit_battles` limits whole trajectories, never individual action rows. The
legacy query name `limit` remains an alias but has the same whole-battle
meaning.

Materialize an export for the offline pipeline with:

```bash
python scripts/materialize_v5_dataset_export.py \
  /secure/path/extraarena_v5_dataset_20260728_120000.jsonl \
  /secure/path/phase-c-human \
  --group-id phase-c-human-20260728
```

The materializer rejects:

- a wrong export/storage version;
- unsafe or duplicate battle IDs;
- nonterminal, empty or incomplete bundles;
- missing Metronome/TimeStamp fields;
- a TimeStamp clock not anchored to `client_ready`;
- any battle that fails the canonical
  `rlhf_env.components.v5_trace_validate.validate_v5_trace` checks.

It builds and validates a sibling temporary directory, then renames it into
place. An existing destination is left untouched on validation failure. Use
`--overwrite` only when replacing a previously materialized copy; replacement
also occurs only after the new copy passes all gates.

The materialized production export does not synthesize a legacy compact
`battle_log` file. Its manifest therefore records validation scope
`v5_trace_without_legacy_battle_log`; state/action/timing invariants are
validated, while legacy battle-log correspondence is intentionally not
claimed.

## Privacy

The default export uses side pseudonyms (`p1 = 1`, `p2 = 2`) throughout nested
metadata, turns and actions. This is sufficient for actor continuity,
winner/side resolution and training.

`include_players=1` exports raw player identifiers and should be used only for
an explicitly authorized investigation. Such files are sensitive operational
artifacts: keep them outside the repository, restrict access, and delete them
after the investigation. Human-readable usernames, contact data and client
tokens do not belong in either export mode.

Model training should normally use the pseudonymized export. Rejected audit
rows may be retained with it because they contain game state, not an extra
identity surface.
