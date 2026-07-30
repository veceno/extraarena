# Handoff — Extra-LR-V5-Max pipeline

**Companion to:** [`2026-06-27-extra-lr-v5-pipeline-design.md`](./2026-06-27-extra-lr-v5-pipeline-design.md)
**Spec commit:** `0f9c906b`
**Created:** 2026-06-27
**Purpose:** let a future session (or the user) resume V5-Max work after the user finishes (a) the new card mechanics and (b) the RLHF env. This doc is self-contained enough to resume **without re-running the repo audit**.

---

## 1. One-line status

Design spec **written + committed + self-reviewed**, **awaiting user review + open-question closure**, then → `superpowers:writing-plans` to produce the implementation plan. The user is currently finishing **new mechanics** and **the RLHF env**; nothing trains until those land and a few prerequisites are verified.

> **Project memory (cross-session recall):** this work is also recorded as a memory entry — `~/.claude/projects/-Users-laveqox-Documents-ExtraArenaRaS/memory/extra-lr-v5-pipeline.md` — indexed under the “RLHF training environment” section of that project’s `MEMORY.md`. Future sessions auto-load `MEMORY.md`, so the V5-Max pointer surfaces on resume even before this handoff is opened.

**2026-07-05 override:** this handoff is historical for the initial design. Phase A's
first bootstrap is now random-heavy Rust ArenaEnv PPO
(`TrainV3.5/scripts/run_phaseA_random_bootstrap.py`, target 98%+ vs random), not
semi-synthetic ExtraRLHF LLM/V4Max distillation, not V4Max warm-start by default,
and not mandatory pilot→BC first.
Later Block-B league and Block-C human-vs-preV5 remain in the pipeline.

**2026-07-17 auxiliary-model override:** D8 is superseded. The pipeline now
confirms **ExtraLR Assembler V1**, **ExtraLR CardOptimum V1**, **ExtraLR
Metronome V1**, and **ExtraLR TimeStamp V1 Mono**; TimeStamp Duo is optional.
Post-audit code already contains heuristic/data-plumbing precursors for
Assembler and CardOptimum (`assembler` and legacy `desirerer`), but not trained
production artifacts. Metronome and TimeStamp are new. The companion design
spec is authoritative for their contracts, datasets, migration, and gates.

---

## 2. Decision ledger (frozen — do not re-derive)

| # | Decision |
|---|---|
| D1 | RLHF data → **offline-bridge** (reconstruct `GameState` from `v5_trace.pre_state` → offline-PPO replay). |
| D2 | **Hybrid stack**: Rust kernel (TrainV3) rollouts/self-play/league/acceptance; Python offline-bridge + replay consumer; rlhf_env (8090) for fresh human collection. |
| D3 | Phase A seed = **random-heavy Rust ArenaEnv PPO bootstrap**. The older pilot→BC idea is optional/deferred; semi-synthetic LLM/V4Max distillation is disabled for Phase A. |
| γ | **Frozen v1 codec**: `classic_obs_v1`/`classic_actions_v1`/`classic_card_shape_v1` byte-frozen → V4-orig ONNX runs unchanged (pristine, zero adapter). V5 uses new `v5_card_shape_v1` + `encode_observation_v5` + **parallel `mana_draw` binary head** (not a 602nd candidate). V5 601-scorer + base-1456 path **warm-starts from V4-Max** (`update_1190.npz`). |
| D6 | **V4’ fine-tune removed.** V4 frozen. Benchmark = best self-snapshot + human-QA (+ decisive H2H vs V4-orig as asymmetric legacy ref). |
| D7 | **C1 archive replay dropped.** RLHF loop = C2 (collect) → C3 (replay). |
| D8 (superseded) | ~~Sub-models deferred.~~ Superseded by D14 on 2026-07-17. |
| D9 | C-phase mechanism = **AWAC/CRR offline-PPO replay** (not pure BC / DPO). |
| D10 | **Acceptance stays on Rust** (TrainV3 `gauntlet_v5`/`run_v5_acceptance`); rule agents not ported. `model_benchmark` keeps legacy-H2H role only (no V5 adapter). |
| D11 | Encoder = **omniscient always** (server-side bot). No perspective/aux split. |
| D12 | History **mandatory**, window 20. |
| D13 | **Early-stop K=2** in the RLHF loop. |
| D14 | **Four auxiliary models confirmed:** ExtraLR Assembler V1, CardOptimum V1, Metronome V1, and TimeStamp V1 Mono; TimeStamp Duo optional. Separate datasets/artifacts/gates; no-assist V5 acceptance remains independent. |
| — | V5-Lite **out of scope** (separate TrainV2-arch effort). |

Pipeline: `Block -1 (freeze + Rust parity) → 0 (foundation) → A (random-heavy Rust PPO bootstrap) → B (league) → C (C2→C3, early-stop K=2) → D (league-2) → E1 (tournament + ship)`.

---

## 3. Current state — who does what

**Done (this session):**
- Repo audit (8-agent workflow) → context map (ephemeral tmp output, not relied on; grounded refs are in §6 of this doc + the spec).
- Spec written, self-reviewed, committed (`0f9c906b`).

**User is doing now (blocks training):**
1. **Finish all new mechanics** — card rebalance + 6 new cards (47–52) + mechanic changes + `mana_draw` (commit `9f20d7a4` already added `ManaDrawAction`; NewCards2606 worktree has cards 47–52 + 5 new mechanic families).
2. **Update the RLHF env** to the new ruleset so collected traces encode consistently with the training codec.

**Blocked on user (cannot start Block -1/0 until resolved):**
- Ruleset must be **frozen** (rebalance + 7 cards + mechanic changes + mana_draw locked).
- Rust ArenaEnv must reach **parity** with the frozen ruleset (see Open Question Q1).

**Not started (resume targets):**
- Closing open questions Q1–Q5 (§5).
- `superpowers:writing-plans` → implementation plan.
- Any code (codec/encoder/bridge/tests) — **no implementation has started**.

---

## 4. Resume procedure (when the user returns)

1. Read the **spec** (`2026-06-27-extra-lr-v5-pipeline-design.md`) and this handoff.
2. Confirm with the user: are new mechanics + RLHF env **done and the ruleset frozen**? If no, stop and wait.
3. Close **Open Questions** §5 (verifications + a couple of decisions). Most are read-and-confirm; Q3 may need a small probe.
4. Re-confirm the user still approves the spec as-is (the spec self-review already passed; user review was the open gate).
5. Invoke `superpowers:writing-plans` to produce the implementation plan. **Do not** invoke any other implementation skill — writing-plans is the terminal brainstorming step.
6. Then (later, separate) execution: Block 0 foundation first (v5_card_shape_v1, encode_observation_v5, mana_draw head, warm-start loader, offline-bridge), gated by its tests (spec §6).

Resume prompt the user can paste:
> "Continuing Extra-LR-V5-Max. Read `docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-handoff.md` and the sibling design doc. Mechanics + RLHF env are done and the ruleset is frozen. Close the open questions, confirm the spec, then run superpowers:writing-plans to produce the implementation plan."

---

## 5. Open questions — how to close each

- **Q1 — Rust ArenaEnv parity.** Verify `TrainV3` Rust kernel implements `mana_draw` + the 6 new cards (47–52) + the 5 new mechanic families identically to Python `core/engine.py`. *How:* compare Rust rollout outputs vs Python `core/` on a fixed seeded scenario suite covering mana_draw cost-scaling (2/4/6…), each new card, each new mechanic (`aoe_silence`, `team_wide_shield`, `rebirth`, `crime_and_punishment`, `target_ally_max_hp_plus[_universal]`). If Rust lags → scope a port task into Block 0. **Cannot start V5 training without this.**
- **Q2 — `HAND_CAP` value.** Confirm the ruleset max hand size to parameterize the lifted `_NUM_HAND` (currently hard-capped at 4). Quick: read `core/`/engine constant or ask the user.
- **Q3 — V4 warm-start fidelity.** Does V5’s fused architecture allow a clean 601-scorer + base-1456 weight transfer from V4-Max, or only partial? *How:* instantiate V5 policy, load `update_1190.npz` into the matching layers, forward-pass a frozen-obs subset, compare logits to V4-Max ONNX on the base-1456 portion (within export tolerance). If partial, document which layers transfer.
- **Q4 — mana_draw usage baseline.** Define the human-baseline measurement precisely (over the pilot battles): `mana_draw_count / eligible_turns` distribution. Needed for the A-gate `[0.5×, 1.5×]` band and the E1 acceptance band.
- **Q5 — V4-orig blind-lane bias risk.** V4-orig never draws and is a Phase-B lane → the learner could over-fit to “opponent never draws.” *Mitigation:* keep V4-orig lane weight modest and self-snapshot prevalence high; monitor the learner’s mana_draw usage doesn’t collapse when facing V4-orig lanes. Decide thresholds at plan time.

---

## 6. Grounded file map (from the audit — durable, no re-run needed)

**Training arch:**
- `ai/train_v2/train_ppo.py` — TrainV2 PPO (PPOConfig, `_parse_opponent_mix`:153, `_run_eval`:1296 vs Random/EndTurn).
- `ai/train_v2/model_mlx.py:37` — `ActionConditionedPolicy` (V4 net; obs 1456 → 256 → 601 scorer).
- `TrainV3/python/train_v3/v5_policy.py:29` — V5 split-encoder (base 1456 + global 32 + private + history 20×144=2880) → 601 scorer.
- `TrainV3/python/train_v3/reward_v5.py` — V5 reward shaping formulas (copied per-action into traces).
- `TrainV3/scripts/run_v5_acceptance.py:41` — acceptance thresholds.
- `TrainV3/scripts/gauntlet_v5.py:8` — exploit-lane roster.
- `TrainV3/scripts/run_phase26_noassist_easy_gate.py` — Phase 26/A (the stall; `step_rewards=learner+opponent`:490, `opponent_mix` 0.55 random).

**Codec / encoder (FROZEN under γ — do not modify):**
- `ai/train_v2/classic_actions_v1.py:46` — `MAX_CANDIDATE_ACTIONS=601`, `build_action_mask`, `encode_action_features`, `_fill_preview_delta`:596 (preview channels 142–170; reusable for the confirmed CardOptimum track).
- `ai/train_v2/classic_obs_v1.py:28` — `encode_observation` (1456; `_NUM_HAND=4` cap, zone-summary deck, ignores history).
- `ai/train_v2/classic_card_shape_v1.py:106` — `encode_card_shape` (64-float; **overlap bug** index 47).

**Engine / mechanics (user is finishing):**
- `core/engine.py` — `mana_draw` handling, `_cleanup_dead_units` (NewCards2606 worktree), targeting.
- `core/effects.py` — 5 new mechanic families + parsers (worktree).
- `core/actions.py:76` — `ManaDrawAction`.
- `core/state.py:20` — `MECHANICS_LIST` (to extend in v5 fork, not in frozen `classic_*`).
- `core/card_scaling.py` — per-level scaling (worktree +18).
- `web/server.py` — `POST /api/battle/mana-draw` (human-only plumbing).

**Data / RLHF env:**
- `rlhf_env/components/v5_trace.py` — `V5TraceRecorder` (omniscient; `meta.json`/`turns.jsonl`/`actions.jsonl`; `visibility='omniscient_offline_only'`; `catalog_ids` is a LIST ATTR). Surrender synthetic row via `record_terminal`.
- `rlhf_env/components/analytics.py` — `AnalyticsRecorder` (prod-schema NDJSON; matches `infrastructure/database.py::export_train_v2_battle_dataset`).
- `rlhf_env/components/match_runner.py` — battle loop (`execute_human_action`, `run_bot_turn`, Ничья-fix ordering `state_changed` before `_finalize`).
- `rlhf_env/components/policy_registry.py:94` — `scan`; uncommitted `_derive_kind_from_sidecar` (classifies ONNX without the benchmark inspector).
- `rlhf_env/components/policy_factory.py` — `build_policy` (`BOT_MAX_DIFFICULTY='max'`, argmax; integrity `RuntimeError` on rule-based fallback).

**Auxiliary-model precursors / new work:**
- `TrainV3.5/python/train_v3/aux_models.py` — current heuristic Assembler +
  `desirerer` dataset/scorer plumbing. `extra-sublr-desirerer-v1` migrates to
  the public ExtraLR CardOptimum V1 concept; its current first-deck-card /
  immediate-reward label is smoke-only, not a valid final target.
- `TrainV3.5/scripts/run_phase5_aux_models.py` — builds/evaluates the two
  precursor datasets; it does not train the confirmed learned artifacts.
- Metronome consumes accepted human `human_decision_time_ms` rows after a
  human-visible feature projection and censor filtering.
- TimeStamp Mono/Duo require new completed-battle duration datasets; simulator
  pretraining plus human calibration is mandatory because real C2 volume is
  expected to be 100–300 battles (500 optimistic).

**Benchmark harness (legacy role only for V5):**
- `ai/model_benchmark/policies.py:24` — `BenchmarkPolicy` Protocol; `ActionConditionedOnnxPolicy`:61 (V4 adapter; temperature/sample native); `LegacyOnnxPolicy`:128 + `legacy_codec.py` (V3 adapter).
- `ai/model_benchmark/runner.py`, `config.py`, `scenarios.py`, `reporting.py`.

**Bridge (new, to build):** `ai/train_v2/offline_dataset_loader.py` (does not exist yet — confirmed by grep; only readers of v5 traces today are a writer-verification test).

**Worktrees:** `.claude/worktrees/NewCards2606` (branch `worktree-NewCards2606`, HEAD `9f20d7a4`) — cards 47–52 + 5 mechanic families, uncommitted.

---

## 7. Git / repo state notes
- Spec committed as `0f9c906b` on `main` (only the spec file; the user’s unrelated uncommitted work — `policy_registry.py`, `requirements.txt`, `tests/test_policy_factory.py`, `tests/test_extra_pass_claiming.py`, `web/server.py`, plus untracked DesignAssets/docs/`rlhf_env/tools/` — was **left untouched**).
- V5-relevant uncommitted mod already present: `rlhf_env/components/policy_registry.py` `_derive_kind_from_sidecar` (needed to classify V4/V5 ONNX sidecars without the benchmark inspector).

---

## 8. Out-of-scope reminders
- **V5-Lite** — separate TrainV2-architecture effort (new cards/mechanics); not this pipeline.
- **Auxiliary models are no longer out of scope.** Assembler, CardOptimum,
  Metronome, and TimeStamp Mono are confirmed; TimeStamp Duo remains optional.
- Do **not** modify frozen `classic_obs_v1` / `classic_actions_v1` / `classic_card_shape_v1` — V4-orig ONNX depends on them byte-identically. Bugfixes + new flags go in the new `v5_*` files.
- Do **not** invoke implementation skills after spec approval — only `superpowers:writing-plans`.
