# ExtraRLHF — Core Concepts

Universal vocabulary for the `rlhf_env` orchestration layer. Model-version
agnostic — applies to any checkpoint (legacy/action-conditioned/future), any
adapter kind, any actor mix.

## Actor types (`p1_actor_type` / decision source)

| `p1_actor_type` | p1 is… | p1 driven by | `submit_action`? | `decision_source` for p1 |
|---|---|---|---|---|
| `human` | browser player | WS / MCP `submit_action` | yes | `human` |
| `llm` | an LLM sub-agent | MCP `submit_action` (player tools) | yes | `llm` |
| `rl` | our trained model | `advance_bot` / auto-play (`match.p1_policy`) | **no** (rejected) | `rl` |

p2 is always a bot/RL model: `decision_source="bot"` for baseline policies
(random/greedy_face/end_turn), `"bot"` for onnx opponents too (the side, not
the policy class). `actor_player` (1/2) maps to `decision_source`:
1↔{human,llm,rl}, 2↔bot.

## `battle_tag` — what kind of match this is

`"{p1_actor_type}-vs-{p2_side}"` where `p2_side` is `bot` for built-in
baselines (`random`/`greedy_face`/`end_turn`/empty) and `rl` for real model
adapters (`action_onnx`/`legacy_onnx`/`v5`/custom onnx). The tag depends **only
on p2's kind**, not on p1's actor type.

Full matrix:

| p1 \ p2 | baseline | model (onnx/v5/custom) |
|---|---|---|
| `human` | `human-vs-bot` | `human-vs-rl` |
| `llm` | `llm-vs-bot` | `llm-vs-rl` |
| `rl` | `rl-vs-bot` | `rl-vs-rl` |

Use the tag to slice the dataset (`list_v5_groups battle_tag=...`): `llm-vs-rl`
and `rl-vs-rl` are the high-value semi-synthetic traces (real opponent, no
human-noise); `*-vs-bot` is cheap rollouts/sanity.

## Adapter registry (model resolution)

`policy_adapters.AdapterRegistry` is the single extension point. Resolution in
`build(spec)`:
1. **Baseline by name** (highest priority): `random`/`greedy_face`/`end_turn`
   win even if a `path` is present (no file needed).
2. **Registry resolve** name→`{path,kind}` if `path` absent.
3. **Auto-detect** kind via registered detectors (LIFO) when `kind` is
   `auto`/absent and `path` is set; a layer-A fallback (`ai.model_benchmark`)
   may classify onnx. Without layer A, detect returns None → explicit
   `ValueError` (not a crash).
4. `kind="unknown"` → `ValueError`.
5. Factory by kind; resolved `path`/`kind` written back into `spec`.

Built-in kinds: `random`, `greedy_face`, `end_turn`, `legacy_onnx` (V3-style),
`action_onnx`/`v4` (action-conditioned), `v5` (implemented: observation 7128,
601 action candidates, separate value and mana-draw heads). V5 sidecars are
detected before the generic V4/action-conditioned detector. Add a model without editing if/elif:
`default_registry().register("mykind", factory)` / `register_detector(fn)`.
MCP exposes `register_custom_model` for runtime add by path+kind.

When layer A (`ai.model_benchmark`, gitignored) is absent, onnx auto-detect is
None; pass `p2_model_path` + `p2_model_kind` explicitly to play onnx models.

## Agent codenames

Series are pinned to a "playing" sub-agent codename so an orchestrator can
track a chain of battles per agent. Pool (priority): fixed list
(`Veceno, Mentalist, Pvwell, Sinaf, Movi, Ilya, Oguzok, Milita, dranik,
sukunyata, absolute`) → card names from `ai/cards.json` → random
`Agent-<hex>` fallback when exhausted.

- `start_series` with `agent_name` claims it (False if busy); omit for auto-assign.
- A name is **released** (`finished=true`) when the series completes — via
  `next_battle` natural completion, `finish_series`, or **self-healing reap**
  from MCP read-paths (`get_match_status`/`get_agent_status`/
  `get_battle_group_status`/`list_active_series`).
- **No mid-series release**: reap fires only when the current battle `is_ended`
  AND `battles_finished >= battles_planned`. A 1/3 series stays busy.
- Cross-process: a new process recovers via `AgentRegistry._self_heal_locked`
  reading the group manifest on disk → marks finished if finalized.

This bounds the codename pool: even if a client crashes after the last battle,
the name is freed on the next status read.

## `degraded` / `policy_warnings` (silent-fallback guard)

If a policy can't be built (missing model, invalid sidecar, onnx load
fail), the match falls back to a safe policy (e.g. `end_turn`) and
`start_series` returns `degraded=true` + `policy_warnings`. **Always check
`degraded`** before trusting a trace — a degraded match's `decision_source` is
not the requested model. `v5/meta.json bot_policy.weights_hash` lets you
verify post-hoc that the real checkpoint played (sha256[:16] of the onnx file).
Warnings/degraded are also persisted per battle in `manifest.json`, so an
offline collector can reject fallback data after the process exits.

## Process ownership

MCP stdio and the web arena each own an independent in-memory match manager.
They share trace files only when configured with the same absolute sessions
directory. A live `match_id` is therefore valid only in the process/session
that created it. An L2 worker must own start→play→finish; a Phase-C human
collector observes completed web groups from disk instead of trying to drive
them through MCP.

## Determinism & replay

`seed` → deck build + engine RNG; battle `i` → `seed + i*1009`. Pure engine:
same seed + same action stream ⇒ identical trace (except `timestamp_ms`).
Replay a battle by feeding `actions.jsonl` `action_json`/`legal_action_index`
back through the engine.

## Orchestration levels (who uses what)

- **Level 0 — Pipeline orchestrator** (`extrarlhf-pipeline-orchestration`):
  runs the full train loop. Uses level 1 to generate data, level 2 to play,
  `register_custom_model` to promote checkpoints, `list_v5_groups`/
  `validate_v5_traces` to gate training.
- **Level 1 — Data-gen orchestrator** (`extrarlhf-gen-orchestration`):
  plans + dispatches series (fleet of codename agents), monitors, validates
  traces, hands the dataset to the trainer.
- **Level 2 — Player sub-agent** (`extrarlhf-player`): plays **one** battle as
  p1 (human/llm) via player tools. Composable: level 1 spawns many level-2
  agents in parallel.

See each skill's SKILL.md for the playbook.
