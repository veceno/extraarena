---
name: extrarlhf-pipeline-orchestration
description: "Use when running the full Extra-LR training pipeline as the top-level model-manager: plan collection, export/inspect/validate/split V5, Nemesis or ReturnClock data, hand immutable artifacts to a trainer, benchmark candidates, and promote models. Delegates battle generation to extrarlhf-gen-orchestration and play to extrarlhf-player."
---

# ExtraRLHF — Level 0: Pipeline Orchestrator

You are the **top-level model-manager** running the Extra-LR training pipeline.
You do **not** play battles yourself and do **not** baby-sit individual series —
you direct the lower levels and make go/no-go decisions between phases. This
skill is model-version agnostic: "current model" and "candidate" are abstract.

## The pipeline (a cycle, not a line)

```
        ┌─────────────────────────────────────────────────────────┐
        │  1. DATA  — campaign: collect semi-synthetic traces     │
        │            (delegate to extrarlhf-gen-orchestration)     │
        └─────────────────────────────────────────────────────────┘
                              │ dataset ready (validate_v5_traces clean)
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │  2. SHIP  — export → validate → split/materialize        │
        │            immutable artifact + manifest/checksum        │
        └─────────────────────────────────────────────────────────┘
                              │ offline train → checkpoint .onnx + sidecar
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │  3. EVAL  — benchmark candidate vs current (model-vs-    │
        │            model series, rl-vs-rl) + sanity (vs baselines)│
        └─────────────────────────────────────────────────────────┘
                              │ win-rate + trace integrity pass
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │  4. PROMOTE — register_custom_model; retire old; loop    │
        └─────────────────────────────────────────────────────────┘
```

## Your tools (level-0 set)

- **Fleet / status:** `list_active_series` (fleet + by-model), `get_agent_status`,
  `list_battle_groups`, `get_battle_group_status`.
- **Dataset readiness:** `get_training_data_status`, `list_training_exports`,
  `inspect_training_export`, `validate_training_export`; V5 summary/trace
  tools; contour export/materialize/split tools.
- **Model registry:** `list_models`, `register_custom_model`.
- You **delegate** series creation/advance to **Level 1** (`extrarlhf-gen-orchestration`)
  and single-battle play to **Level 2** (`extrarlhf-player`).

## Phase playbooks

### Phase 1 — Data campaign (delegate to L1)
Decide the campaign spec, then hand it to L1:
- **Opponent mix**: which model(s) to play against (current prod model + a
  baseline for sanity). More `*-vs-rl` (real opponent) = higher-value traces;
  some `*-vs-bot` for cheap coverage.
- **Actor mix**: `p1_actor_type="llm"` (LLM sub-agents play → `llm-vs-rl`,
  semi-synthetic) and/or `"rl"` (model-vs-model → `rl-vs-rl`, self-play style).
- **Volume**: `battles_planned` per series × number of codename agents.
- **Diversity**: vary `seed`, `starting_player`, `deck_strategy` (mostly
  `random_arenaenv`; `custom` for targeted matchups).

Tell L1: "Collect `<N>` battles of `llm-vs-rl` vs `<model>` across agents
`<codenames>` with varied seeds. Validate traces clean. Report group_ids + a
`download_battle_logs` archive." L1 returns group_ids + integrity status.

For Extra-LR V5 **Phase C**, the frozen default is different: collect browser
`human-vs-rl` series against the selected V5 checkpoint. The web process owns
those matches; the collection driver only observes completed groups from the
shared sessions directory. It must never launch/advance a human battle through
a separate MCP process.

### Phase 2 — Export, validate, split/materialize, hand off

Start with `get_training_data_status`. Local operations work without production
access; only intentional production V5/ReturnClock reads require the explicit
opt-in and private server environment described by the umbrella skill.

For **V5 policy**:

1. `validate_v5_traces` on every headless group: require
   `v5_policy_training_ready=true`, `training_ready_scope="v5_policy_only"`,
   no `broken` and no `degraded`. The generic `training_ready` field is only a
   backward-compatible alias for this policy gate.
2. Headless groups can supply accepted V5 action targets and, after its own
   eligibility checks, Nemesis Lite examples. Train policy only from rows with
   `accepted is True`; rejected rows remain audit evidence.
3. For production human data, `export_v5_training_dataset`, then
   `materialize_v5_training_dataset` and `validate_training_export` on the
   directory. Require catalog/card-count and weights provenance in either
   route.

For **Metronome / TimeStamp**, use their independent readiness fields. A clean
headless policy trace is not sufficient: without real observed production
labels, `metronome_training_ready` and `timestamp_training_ready` must remain
false. Preserve human Metronome timing aligned with its pre-action state and
TimeStamp battle labels aligned with the battle they describe. Never substitute
headless CPU/wall-clock duration, LLM latency, or synthetic actor delay for
human labels. For TimeStamp, construct inputs only from the prebattle deck or
deck pair, `starting_player`, and explicitly approved prebattle features.
`duration_seconds`, `turns`, `finished_at`, and derivatives are labels/audit
only and must never enter the input tensor. Fail the handoff if a loader passes
all of `timestamp_features` or `meta` instead of an explicit allowlist.

For **Nemesis**, call `export_nemesis_training_dataset` with exactly one of a
validated V5 transport `input_path` or a completed headless `group_id`. Lite
uses `features.base`; standard uses base + optional `features.extended`.
Respect `eligible_lite`,
`eligible_standard`, `sample_weight`, exclusions and
`provenance.split_group`. Then call `split_nemesis_training_dataset`: keep the
Lite deck-grouped assignment. Require `training_ready_standard=true` before
expecting all three Standard views (player-disjoint primary, chronological and
deck-grouped evaluation); a Lite-only headless export is still a valid Lite
handoff. Standard needs at least six players, three pairwise-disjoint
human-human battles, three matchup groups and three cutoff cohorts. The
player-disjoint view
assigns each export-local player alias once and records every excluded
cross-partition battle fingerprint; aliases remain grouping metadata and never
model inputs. Human-bot records are Lite data under the current canonical
trainer; their masked extended payload is audit/future-research metadata, not
Standard training input. Do not create two copies of the same source battle.

For **ReturnClock**, export with a cutoff-safe horizon and safety lag, validate,
then call `split_returnclock_training_dataset`. Require a leakage-clean
grouped-by-user, temporal train/validation/test manifest. Feed the estimator
only `header.feature_columns`; keep `post_cutoff`, `user_id_hash` and
`prediction_cutoff_at` out of model features. The raw export is mixed audit
material; train/evaluate the natural-return baseline only from the organic-only
split files, where every row has `organic_candidate=true` and the manifest
accounts for excluded treated rows. Do not train a causal notification policy
until randomized no-send/control assignments exist.

ReturnClock production reads must stay within one repeatable-read snapshot:
keyset pagination uses pages up to 50,000 and a ceiling of 1,000,000 rows per
raw stream. Treat `end_at` as the exclusive event/censoring boundary and
require the separately recorded `ingested_before` row-creation watermark
(`user_sessions.created_at`, decision/delivery-event `created_at`). The safety
lag applies when `end` is omitted; an explicit historical `end` is used as-is.
Later mutable status updates must not erase earlier assignments.
Treat a stream that reaches the ceiling as incomplete; do not stitch separately
censored exports. Current extraction and splitting materialize bounded windows
in memory, so size large campaigns explicitly instead of assuming streaming
training handoff.

Every handoff must record artifact path, SHA-256, format/version, validation
summary, split/materialization manifest, catalog/weights provenance, exclusion
counts, and ReturnClock pseudonymization key id where applicable. The trainer
itself remains offline/out of environment scope.

Use a new versioned output with `overwrite=false`. Fresh-path publication is a
temp-build plus same-filesystem rename; overwrite rollback handles ordinary
exceptions but is not crash-atomic, so an overwritten path is not a durable
promotion protocol.

Structural cleanliness is necessary but not sufficient for semi-synthetic LLM
data. Require at least 50 battles, no rejected/fallback decisions, and a 95%
Wilson lower bound for p1 win rate above 3%. The default Phase-C bridge remains
human-only; LLM input needs an explicit decision-source opt-in after this gate.

Structural cleanliness is necessary but not sufficient for semi-synthetic
LLM data. Require the summary quality gate: at least 50 battles, no rejected or
fallback decisions, and the 95% Wilson lower bound for p1 win rate above 3%.
The default Phase-C replay bridge still accepts only `decision_source=human`.
LLM input requires an explicit `accepted_decision_sources=("llm",)` opt-in
after that quality gate passes.

### Phase 3 — Eval (candidate vs current)
Run a **model-vs-model** benchmark series: `p1_actor_type="rl"`,
`p1_model_path=<candidate.onnx>`, `p1_model_kind=<kind>`,
`p2_model=<current prod model>` (or symmetric, both directions, multiple seeds).
This is `rl-vs-rl` — auto-plays, no L2 needed. Also run a sanity series vs
`random`/`greedy_face` (`rl-vs-bot`) — candidate must dominate baselines.
Decision rule (tune to your bar): promote if candidate win-rate vs current
exceeds threshold AND no `degraded` AND `validate_v5_traces` clean.

### Phase 4 — Promote
`register_custom_model name=<new name> path=<candidate.onnx> kind=<kind>` →
confirm it appears in `list_models`. Optionally retire the old model from the
registry (or just stop referencing it). Loop to Phase 1 with the new current
model as the opponent.

## Decision heuristics

- **"Is there enough data?"** → use contour summaries and class/split coverage,
  not file size alone. For V5, sum accepted training rows and inspect tag/deck/
  starting-player balance.
- **"Is the data clean?"** → deep-validate every source and final artifact.
  Investigate any broken, degraded, privacy, catalog, truncation, eligibility
  or leakage issue; re-export/re-run rather than weakening the gate.
- **"Candidate better?"** → eval series win-rate + trace integrity; never
  promote a candidate that ever played `degraded`.
- **"Fleet stuck?"** → `list_active_series`; a busy agent on a completed series
  self-heals on read — call `get_agent_status` once to trigger reap. If a series
  is genuinely hung, `finish_series` to release the agent.

## Anti-patterns
- Don't `submit_action` on an `rl-vs-*` match — it's rejected (rl p1 is
  auto-played via `advance_bot`). Use L2 only for `human`/`llm` p1.
- Don't trust a trace without checking `degraded` and `weights_hash`.
- Don't treat an export call as a handoff: inspect, validate and split or
  materialize it first.
- Don't pass raw salts, DSNs or raw-player switches through MCP.
- Don't randomly split ReturnClock rows or Nemesis deck-pair groups.
- Don't run one giant series — run many short series across codenames for
  parallelism + isolation (a crash affects one agent, not the campaign).

## See also
- `../extra-rlhf/SKILL.md` — umbrella + setup
- `../extra-rlhf/references/mcp-tools.md`, `concepts.md`, `data-format.md`
- `../extrarlhf-gen-orchestration/SKILL.md` (L1), `../extrarlhf-player/SKILL.md` (L2)
