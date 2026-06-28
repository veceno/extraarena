---
name: extrarlhf-pipeline-orchestration
description: "Use when running the full Extra-LR training pipeline as the top-level model-manager: plan data-collection campaigns, hand traces to the offline trainer, benchmark a candidate checkpoint against the current model, and promote/deregister models. Delegates data generation to extrarlhf-gen-orchestration and play to extrarlhf-player. Model-version agnostic."
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
        │  2. TRAIN — hand the dataset to the offline trainer      │
        │            (out of env scope; you produce the artifact)  │
        └─────────────────────────────────────────────────────────┘
                              │ new checkpoint .onnx + sidecar
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
- **Dataset readiness:** `list_v5_groups` (filter by `battle_tag`),
  `get_v5_dataset_summary`, `validate_v5_traces` (integrity gate),
  `get_battle_group_manifest`, `download_battle_logs`.
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

### Phase 2 — Train (out of env)
You don't run the trainer. You produce the dataset artifact:
- `validate_v5_traces` on every campaign group → all `ok`, no `broken`.
- `get_v5_dataset_summary` to confirm row count + `battle_tag_distribution`.
- `download_battle_logs group_id=<gid> format=zip` → hand the path to the
  offline trainer. The trainer consumes `battles/<bid>/v5/actions.jsonl`
  (see `../extra-rlhf/references/data-format.md`).
Gate: do not train on a group with `broken` traces or `degraded` battles
(check `weights_hash` matches the intended opponent).

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

- **"Is there enough data?"** → `list_v5_groups` + sum `actions_total` via
  `get_v5_dataset_summary` per group. If below target or tag-skewed, dispatch
  another L1 campaign.
- **"Is the data clean?"** → `validate_v5_traces` every group; investigate any
  `broken` (non-null fields, actor↔source mapping, turns count). Re-run the
  battle or drop the group.
- **"Candidate better?"** → eval series win-rate + trace integrity; never
  promote a candidate that ever played `degraded`.
- **"Fleet stuck?"** → `list_active_series`; a busy agent on a completed series
  self-heals on read — call `get_agent_status` once to trigger reap. If a series
  is genuinely hung, `finish_series` to release the agent.

## Anti-patterns
- Don't `submit_action` on an `rl-vs-*` match — it's rejected (rl p1 is
  auto-played via `advance_bot`). Use L2 only for `human`/`llm` p1.
- Don't trust a trace without checking `degraded` and `weights_hash`.
- Don't run one giant series — run many short series across codenames for
  parallelism + isolation (a crash affects one agent, not the campaign).

## See also
- `../extra-rlhf/SKILL.md` — umbrella + setup
- `../extra-rlhf/references/mcp-tools.md`, `concepts.md`, `data-format.md`
- `../extrarlhf-gen-orchestration/SKILL.md` (L1), `../extrarlhf-player/SKILL.md` (L2)