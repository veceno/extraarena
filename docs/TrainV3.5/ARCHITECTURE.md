# TrainV3.5 — V5-Max Training Pipeline: Architecture & Full Section Documentation

> **Scope.** This document is the complete reference for the `TrainV3.5/` section: what it
> is, how the V5-Max pipeline is structured end-to-end, every module's role, the Rust↔Python
> boundary, the observation/reward/action contracts, the league/gate/tournament/ship layers,
> the prod wiring, the tests, and the invariants that must not be broken.
> For the step-by-step *how to train a model* recipe, see **`TRAINING_GUIDE.md`** (the sibling
> document). The legacy `TrainV3.5/README.md` covers only the V4-era Rust-boundary prep +
> the golden-kernel CLI; this file covers the V5-Max pipeline that was built on top of it.

---

## 1. What TrainV3.5 is

`TrainV3.5/` is the **training-only workspace for the fifth-generation Extra-LR bot
("V5-Max")**. The production bot/runtime stays Python-first and authoritative
(`core/engine.py`, `ai/bot_brain.py`, web/server). TrainV3.5 owns:

- a **Rust rollout-acceleration kernel** (`rust/trainv3_core/`, `trainv3_core::kernel` +
  `trainv3_core::worker`) reached through a `ctypes` FFI, and
- a **Python MLX training stack** (`python/train_v3/`) that runs PPO / BC / AWAC-CRR
  over the Rust environment and ships the trained policy into production as an ONNX.

The full pipeline implemented here is the **V5-Max pipeline**, organised into blocks:

```
Block -1  ->  Block 0  ->  Block A  ->  Block B  ->  Block C  ->  Block D  ->  Block E1
(Rust       (V5 card-   (random-     (League on    (RLHF loop   (League-2     (Tournament
 ArenaEnv   shape 73 +   heavy Rust   Rust         C2->C3,      consolidation  + ship, the
 parity     foundation   ArenaEnv     ArenaEnv)    AWAC/CRR     + D->E1       FINAL stage)
 port)      warm-start)  PPO, A-gate)              offline      handoff)
                                                  replay)
```

As of 2026-07-05, Block A's first bootstrap is **teacher-free random-heavy Rust
ArenaEnv PPO** (`scripts/run_phaseA_random_bootstrap.py`, target 98%+ vs random).
The earlier Phase-A semi-synthetic ExtraRLHF/LLM/V4Max distillation path is disabled;
V4Max warm-start is not used by default for this first bootstrap. Block-B league
and Block-C human-vs-preV5 still follow afterward.

Each block has a `BLOCK_<X>_PLAN.md` + `BLOCK_<X>_COMPLETION.md` pair in `TrainV3.5/`
(the plan is the design, the completion log records what shipped + the test/commit
results). The field-test proof that the whole chain runs live is
`TrainV3.5/BLOCK_E1_FIELDTEST.md` (8/8 smoke steps passed on a real trained garbage model).

### Hard invariants (must not break)

1. **Prod must not import `TrainV3.5` and must not import `rlhf_env`.** The live
   inference path (`ai/bot_brain.py:_get_action_v5`) lazy-imports ONLY
   `ai.train_v2.*` (the vendored V5 encoder copies) + `core.*` — never `train_v3`,
   never `rlhf_env`. `block_e1_runner.py` is TrainV3.5-side only.
2. **Frozen-classic Python is byte-frozen for V4-orig.** The V4 ONNX path
   (`_get_action_train_v2_classic`, `ai/train_v2/export_onnx.py`, the V4 session dict)
   is byte-unchanged; V5 wiring is **additive**.
3. **`core/state.py` is not modified.** `core/engine.py` is the oracle; Rust mirrors it.
4. **Rust `*.rs`/`*.toml` are not touched by Python-side blocks.** Blocks A→E1 made
   ZERO `.rs`/`.toml` edits (Rust golden_kernel: 47 tests green throughout). The
   `cargo build` only compiles existing code.
5. **Promotion-by-loss guard.** Internal training metrics (PPO loss / KL / entropy)
   are MONITORING-ONLY and are NEVER consulted by any promotion/ship decision — only
   external-bench H2H score rates decide promotion (`a_gate.select_promotion`,
   `e1_tournament.select_e1_winner`).
6. **ONNX fallback guard (SPEC design.md:174).** A malformed V5 ONNX (NaN/inf logits or
   no legal candidate) raises `RuntimeError` on the live path — it is **last-resort prod
   safety, NOT a silent rule-based fallback**. The `RuntimeError` propagates out of
   `_get_action_v5`; only *other* unexpected exceptions degrade to `_legal_fallback`.

---

## 2. Directory layout

```
TrainV3.5/
|-- Cargo.toml                       # workspace root (rust/)
|-- Cargo.lock
|-- rust/trainv3_core/                # the Rust kernel (.rs; 20 files)
|   |-- Cargo.toml
|   `-- src/                          # kernel.rs, worker.rs, obs.rs, reward.rs, ...
|-- python/train_v3/                  # the Python MLX training stack (84 .py)
|   |-- __init__.py
|   |-- (training core, env/ffi, league drivers, gates, ship, ...)  -- see §4
|   `-- tests/                        # 24 test_*.py
|-- scripts/                          # runnable experiment/phase scripts (~38)
|-- runs/                             # gitignored training-run output (checkpoints, manifests)
|-- target/                           # cargo build output (gitignored)
|-- BLOCK_*_PLAN.md / BLOCK_*_COMPLETION.md   # per-block design + delivery logs
|-- BLOCK_E1_FIELDTEST.md             # the live end-to-end proof
`-- README.md                          # legacy V4-era Rust-boundary prep doc
```

`TrainV3.5/` IS tracked in git in this worktree (root `.gitignore` lists `TrainV3/`,
which does NOT match the `TrainV3.5/` name). `runs/` and `target/` are gitignored.

---

## 3. The Rust↔Python boundary

The Rust kernel (`trainv3_core`) is a **batched training rollout worker**, deliberately
Python-oracle driven: Python generates golden traces from the authoritative `core/engine.py`,
Rust must match the trace hashes before it is trusted as an acceleration path.

- **`rust/trainv3_core/src/kernel.rs`** — parses a Python golden-trace snapshot and rebuilds
  the hot training tensors in one coarse call (legal action mask, no-preview candidate action
  features, `classic_obs_v1`, V5 private-info/history observation, V5 reward components) +
  `apply_action(action_id)` transitions (end_turn, warrior/potion play, targeted damage,
  taunt-constrained attacks, unit combat cleanup, simple deathrattle AOE, basic attacks).
- **`rust/trainv3_core/src/worker.rs`** — `BatchedRolloutWorker`: owns many mirrored env
  states, applies a batch of action ids, maintains V5 history, returns flat tensor buffers
  (observations, masks, action features, rewards, termination flags). Supports
  `reset()` / selective `reset_indices([...])` / `step_auto_reset(action_ids)` for
  vectorized training loops; emits `episode_returns`/`episode_lengths` + terminal obs.

**Build:** `cargo build --release` in `TrainV3.5/rust/trainv3_core` →
`target/release/libtrainv3_core.dylib` (darwin). Override location with
`TRAINV3_CORE_LIB=/path/to/libtrainv3_core.dylib`.

**Python FFI:** `train_v3.rust_ffi` (`rust_ffi.py`, ~2178 lines) is the `ctypes` bridge.
- `resolve_library_path(path=None) -> Path` (`rust_ffi.py:102`): explicit arg →
  `TRAINV3_CORE_LIB` env → `default_library_candidates()` (looks under
  `TrainV3.5/target/release/`). Cached.
- **`RustBatchWorker`** (`rust_ffi.py:684`): the main class. Constructors:
  - `from_trace_file(cls, path, *, env_count, ...)` (`:719`) — from a golden trace JSON.
  - `from_trace_files(cls, paths, *, env_count, ...)` (`:834`) — a pool of traces.
  - **`from_live(cls, *, seed, env_count, max_turns=120, ...)`** (`:1227`) — **the live
    self-play constructor (Block A4)**. Builds an init-only golden trace, default modes
    `v5_only` / `legal_only` / `none` / `none` (the live training hot path).
- FFI methods: `encode`, `reset`, `reset_indices`, `step`, `step_auto_reset`,
  `current_actor_ids`, `select_rule_actions` (rule-based opponents, dispatch by agent
  code), `advance_rule_until_actor` (fast-forward rule opponents until the learner acts),
  `truncated`, `mana_draw_legal`, `step_mana_draw` (the parallel mana_draw FFI),
  `hero_hp` (`(env_count,4)` = p1/p2 hp+max), `rollout_action_tape`, `arrays`.
- Module-level compute helpers: `compute_rust_gae_returns` (`:143`),
  `compute_rust_selected_local_indices`, `compute_rust_prepare_ppo_batch` (fused
  GAE+selected-local), `compute_rust_pad_legal_actions`, `compute_rust_*_argmax_actions`
  (padded/compact/dense), `compute_rust_pack_legal_action_rows`, etc.

---

## 4. Python module inventory (`python/train_v3/`)

Modules are grouped by pipeline role. Unless noted, a module is **importable only
(no CLI)**; the few with a CLI are flagged. File:line citations are to this worktree.

### 4.1 Contracts / observation / reward (the V5 surface)

| File | Role |
|---|---|
| `contracts.py` | Frozen dim constants: `OBS_V1_DIM=1456`, `V5_GLOBAL_DIM=32`, `PRIVATE_INFO_DIM=2400`, `HISTORY_DIM=3240`, **`OBS_V5_DIM=7128`**, `ACTION_FEATURE_DIM=171`, `MAX_CANDIDATE_ACTIONS=601`, `CARD_SHAPE_DIM=64` (frozen V4), `CARD_SHAPE_DIM_V5=73`. `InfoModeV5` (adaptive_strength, hand/deck-known flags, draw-assist) + `AssistModeV5` (assembler/desirerer/teacher). |
| `obs_v5.py` | `encode_observation_v5(state, player_id, *, info_mode, assist_mode, history_events) -> float32[7128]`. Layout: `[0:1456]` V1 base · `[1456:1488]` V5 global (32, incl. normalized `mana_draw_count_this_turn`) · `[1488:3888]` private info (32 slots × 75) · `[3888:7128]` history (20 events × 162). Unknown zones zeroed per `InfoModeV5`. |
| `reward_v5.py` | `V5RewardSnapshot`, `V5RewardWeights`, `reward_snapshot_v5`, `compute_reward_components_v5`, `compute_weighted_reward_v5` (base reward + HP-potential/board-power shaping, informed/draw-assist penalty multipliers, clipped to `±max_shaping_abs`). |
| `mana_draw_head_v5.py` | The parallel binary mana_draw head. `mana_draw_cost(count)=MANA_DRAW_BASE*(count+1)`; `mana_draw_legal_mask(state, pid)` — byte-faithful mirror of `core/engine.py`'s ManaDrawAction emission; `select_includes_mana_draw(mana_draw_logit, best_candidate_logit, mana_draw_legal)` — legal-mask DOMINATES: illegal → False. **mana_draw is NOT a 602nd candidate; the 601 space is frozen.** |
| `v5_inference_guard.py` | `_assert_v5_logits_finite_legal(logits, legal_mask) -> int` (SPEC :174). Pure-numpy (PROD-VENDORABLE: no MLX/torch/onnxruntime/train_v3 imports). Raises `RuntimeError` on non-finite logits OR no legal candidate; on success returns the finite-legal argmax. Never silently returns 0. |
| `env_v5.py` | `TrainV3ClassicEnv` + `TrainV3EnvConfig` (seed, verify_mask, placement_mode, info_mode, assist_mode, reward_weights, history_limit=20). Wraps the V4 `ClassicRLEnv`; `observe/action_mask/action_features/step` add V5 obs + reward shaping + history events. |
| `golden_trace.py` | **Has a CLI** (`_main()` `:640`). Python-oracle golden-trace builder (`build_golden_trace` `:363`) capturing RNG outcomes per step for Rust parity. `--seed --steps --placement-mode --p1/p2-deck-ids --adaptive-strength --action-ids --mana-draw-steps --max-turns ...` (prints JSON trace). |
| `trace_factory_v5.py` | `V5TraceScenario`, `generate_v5_trace_pool` (grid over seeds×strengths×visibility×draw-assist×assist×level), `load_v5_trace_pool_manifest`, `resolve_v5_trace_paths`, `select_v5_trace_paths_for_mode` (manifest_cycle vs league_schedule). |
| `v5_artifacts.py` | JSON manifest schemas + frozen dataclasses: `TracePoolManifest`, `LeagueRunManifest`, `AuxDatasetManifest`; `write_manifest_json`/`read_manifest_json` (stable `manifest_id = "v_"+sha256[:16]`). |

### 4.2 Policy / model / warm-start / BC

| File | Role |
|---|---|
| `v5_policy.py` | `V5ActionConditionedPolicy(nn.Module)` (`policy_kind="v5_split_encoder"`): 4 split encoders (base 1456 / global 32 / private 2400 / history 3240) → `state_fuser` → 3 heads: `candidate_scorer` (601 logits), `value_head`, **`mana_draw_head` (parallel binary)**. `__call__(obs, action_features, mana_draw_legal=None) -> (candidate_logits(B,601), value(B,), mana_draw_logit(B,))`. `create_v5_policy(*, policy_kind, obs_dim=7128, action_feature_dim=171, hidden_dim=256, action_hidden_dim=128)` (`:155`); `"baseline_mlp"` → the V4 model. Checkpoint I/O is delegated to `ai.train_v2.model_mlx`. |
| `warm_start_v5.py` | V4-Max → V5 **PARTIAL** warm-start (Q3). `load_v4_max_into_v5(policy, npz_path=None) -> dict` (`:176`): faithful params copied exactly (base_encoder.layers.0, action_encoder), shape-compat-disconnected copied by shape (state_fuser.layers.2, candidate_scorer, value_head — 601 logits will NOT match V4), fresh V5 layers untouched (global/private/history encoders, state_fuser.layers.0, mana_draw_head). `resolve_v4_max_npz_path` (`:111`): explicit → `V4_MAX_NPZ_PATH` env → walk-up search. |
| `bc_dataset.py` | (A1) BC dataset builder. `BCTransition`, `resolve_v5_tcode(pre_state, actor, action_native, *, mask, strict)` — **`action_native` MUST be the engine's `BaseAction.to_dict()`, NOT a `decode_action` output** (the self-referential trap). `build_bc_dataset(group_dir, *, info_mode, assist_mode, max_battles, strict)` — filters to `decision_source=='human'` only. |
| `bc_train.py` | (A2) BC fine-tune loop. `collate_bc_batch`, `compute_bc_loss` (masked candidate CE + mana_draw BCE over `mana_draw_legal` rows), `train_bc(...)`, `prepare_bc_policy(npz_path, *, freeze_faithful=True)` (warm-start with a skip-gate if V4-Max npz absent), `run_bc_training(...) -> (policy, report)` (writes a `bc_seed` checkpoint). Freeze-faithful idiom: zero-grad on frozen params ⇒ byte-identical preservation. |
| `aux_models.py` | Training-only aux heads: deck-assembler + draw-desirerer dataset builders + baseline scorers (`DeckMatchupEvaluator`, `DrawDesirerer`, `DrawAssistController`). `build_assembler_rows_*`, `build_desirerer_rows_from_v5_trace`, jsonl save/load with manifests. |
| `llm_teacher.py` | Offline OpenAI-compatible teacher labeling for V5 hard states (preference distillation). `OpenAICompatibleTeacherConfig`, `OpenAICompatibleTeacherClient.label_state(...) -> TeacherPreferenceRow` (disk cache + retries). |
| `v4_orig_temp_spectrum.py` | (B2) Three policy-opponents from ONE frozen V4 ONNX: `V4_ORIG_ARGMAX` (w0.40), `V4_ORIG_T07` (sample T0.7, w0.20), `V4_ORIG_T12` (sample T1.2, w0.15). `make_v4_temp_select_fn`, `TempV4Opponent`, `build_v4_temp_spectrum_opponents(session_or_path)`. Exposes the mana_draw-collapse monitor hook. |
| `curriculum.py` | (B4) Per-lane learner-loss curriculum reweight. `extract_lane_outcomes(rollout)`, `CurriculumReweighter(window_n).update(...).reweight(mix, *, cap=0.25)` — boost only lanes with `loss_rate>0.5`, max factor `1+cap` (1.25×), renormalize to 1.0. PURE (returns new list). |

### 4.3 Training core (PPO / rollout / policy)

| File | Role |
|---|---|
| `train_v5_adaptive.py` | **The main training CLI** (`main()` `:96`). Loads/generates a V5 trace pool, runs `train_rust_ppo_trace_files`, derives assembler+desirerer aux datasets, prints JSON result. `run_v5_adaptive_training_pipeline(...)` (`:29`), `create_v5_default_model_optimizer(...)` (`:153`). CLI flags in §TRAINING_GUIDE. |
| `rust_trainer.py` | `RustPPOTrainingConfig` (frozen dataclass, `:23`) + `train_rust_ppo_trace_files(paths, model, optimizer, config, *, library_path=None)` (`:92`) — the per-update PPO loop: select traces → open vec env → `collect_rust_vec_rollout` → `prepare_rust_ppo_batch` → `train_rust_ppo_minibatch` → checkpoint + metrics + league manifest. `_save_checkpoint` (`:802`) writes `trainv3_rust_legal_update_{update:04d}.npz` via `ai.train_v2.model_mlx.save_checkpoint`. Per-update league metadata (`opponent_mix`, `v5_mode`, `adaptive_strength`) via `_v5_league_update_metadata` (`:535`). |
| `rust_ppo.py` | `RustPPOBatch`/`RustPPOEvaluation`, `prepare_rust_ppo_batch(...)` (GAE/returns; backend choices `advantage_backend`/`selected_local_backend`/`prepare_backend`), `train_rust_ppo_minibatch(...)` (the MLX PPO optimizer step; legal-only compact path), `train_dense_rust_ppo_minibatch`, `evaluate_rust_ppo_batch`, `evaluate_dense_rust_ppo_batch`. Standard clipped-surrogate PPO loss. |
| `rust_collector.py` | `RustTransitionBatch`, `RustLegalActionFeatures`, `collect_rust_vec_rollout(env, policy, *, steps, ...)` (drives a `RustVecEnv` under a Python policy fn returning actions / `(actions, values, log_probs)` / dict), `transition_batch_from_action_tape_rollout`. |
| `rust_vec_env.py` | `RustVecEnv` (thin adapter over `RustBatchWorker`): `reset`/`step`, `from_trace_file`/`from_trace_files`, `current_actor_ids`, `select_rule_actions`, `advance_rule_until_actor`. Returns `RustVecEnvReset`/`RustVecEnvStep`. |
| `rust_rollout.py` | `RustTraceRolloutRunner` (replays a golden-trace action script), `benchmark_trace_file` (Python→Rust FFI throughput). |
| `rust_live_self_play.py` | (A4) **The live self-play entry** (spec-faithful alternative to the trace pool). `run_live_self_play_update(config, learner_policy, opponent_policies=None, *, seed, library_path, model, optimizer, ...)` (`:1002`) — the operational tournament runner. `collect_rust_live_rollout` (`:548`). Dispatch split: 7 rule-agent identities via `RustBatchWorker.select_rule_actions` + 4 policy-opponent identities (end_turn/greedy_face/self/v4max) via the Python loop + Block-B `v4-orig-*`. `default_opponent_policies`, `ArgmaxRandomLearner`, `LiveRolloutBatch`. PPO step is MLX-gated (skipped if `model`/`optimizer` is None). |
| `rust_policy.py` | Collector-facing argmax policy builders: `make_padded_legal_argmax_policy`, `make_compact_legal_argmax_policy`, `make_dense_argmax_policy`; score/pad helpers. |
| `rust_benchmark.py` | Rust pipeline speed benchmarks; `benchmark_trainv3_speed_report(...)` the umbrella. |
| `ppo_phaseA_config.py` | (A3) `PhaseAPPOConfig` (`:308`) — standalone superset of `RustPPOTrainingConfig` carrying the 5 Phase-A root-cause fixes: learner-only reward (#1), `max_turns=120` (#2), pinned `entropy_coef=0.01` (#3) + `epochs=6` (#4), graduated `opponent_mix` (#5), second-start oversampling (D-A10). `build_phase_a_opponent_mix_string`, `reward_attribution`, `is_decisive_state`, `second_start_oversampling_scheme`, `to_rust_ppo_config`. |
| `snapshot_pool.py` | (B1) Bounded self-snapshot pool (~6 rolling + 2 anchors: seed, best-ever), FIFO eviction, manifest round-trip. `maybe_update_best_ever` (strict H2H improvement — promotion-by-loss guard), `load_as_self_prev_opponent_select_fn`, `self_snapshot_prevalence_weight` (D-B5 hybrid). |

### 4.4 League drivers + gates (Blocks A / B / D)

All importable only — a league is run by instantiating the driver and calling `.run(n_updates)`.

| File | Role |
|---|---|
| `a_gate.py` | (A5) The A-gate + promotion selector. `compute_score_rate`, `GateOutcome`, `ManaDrawBaseline`/`record_mana_draw_baseline` (Q4 baseline B, invalidated if engine constants change), the 4 gate checks (`check_no_assist_gate` ≥0.55, `check_exploit_resistance_gate` ≥0.50, `check_mana_draw_band` [0.5B,1.5B], `check_h2h_trending`), `evaluate_a_gate -> AGateResult`. `select_promotion(CandidateExternalBench, internal, *, current_best_h2h_score_rate, h2h_promotion_threshold=0.5)` — promote iff A-gate passes AND H2H > 0.5; **`internal` metrics recorded but NEVER read** (the guard). `GameRunner` Protocol (`play(opponent_kind, *, seed) -> GameResult`), `GauntletOutcomes`, `play_gauntlet`. |
| `block_b_gate.py` | (B6) Block-B promotion gate (extends A5, does NOT re-apply Phase-A exits). `block_b_aggregate(h2h, gauntlet, mana_draw_in_band, p1_p2_gap) -> float` ∈ [0,3]; `evaluate_block_b_gate(...) -> BlockBGateResult` (monotone over `n_snap`=5, 4 component gates). |
| `block_b_league_driver.py` | (B8) `BlockBLeagueDriver.run(n_updates) -> BlockBLeagueManifest`. Per update: `_build_reweighted_mix` (B3 mix + collapse boost + B4 curriculum cap 0.25 + `_merge_self_snapshot_split`) → `run_live_self_play_update` (A4 Option 1) → `curriculum.update` → every `snapshot_cadence` a `_snapshot_step` (B1 add → B6 gate → B1 best-ever → B7 plateau). **Early-exits to C2 when B7 `exit_fires`** (`exited_to_c2=True`). |
| `block_b_opponent_mix.py` | (B3) Block-B opponent mix (frozen non-self 0.95; v4-orig 0.75 = 0.40/0.20/0.15, exploit 0.15, tail 0.05; self-snapshot grown 0→0.05, cap `_MAX_SELF_SHARE=0.95`). `build_block_b_opponent_mix(pool, *, collapse_boost)`, `collapse_reweight_boost`. |
| `block_d_league_driver.py` | (D2) `BlockDLeagueDriver(BlockBLeagueDriver)` overrides `_build_reweighted_mix` (D1 mix) + `run()` (copies B8 loop inline; `BlockDLeagueManifest` with `exited_to_e1` + `candidate_paths` + `aggregate_history`). `exit_mode="fixed_schedule"` (default, B7 exit suppressed) or `"plateau"`. Emits the **D→E1 handoff**. |
| `block_d_opponent_mix.py` | (D1) Consolidation mix (D-D1: self+v5_snapshot 0.50, V4-orig 0.30, exploit 0.15, tail 0.05; always sums to 1.0 incl. degenerate pure-self-play where `self_snapshot_weight=1.0` is forced). NO new dispatch identities → NO A4 edit. |
| `c_to_d_handoff.py` | (D3) `E1CandidateSet` (frozen: `post_d_path`/`post_c3_best_path`/`post_b_path`, only mutation path `with_post_d`), `build_block_d_seed_pool` (fresh pool seeded from post-C best-ever), `thread_e1_candidates`. |
| `second_start_parity.py` | (B5) `BlockBGameRunner` Protocol (`play(opponent_kind, *, seed, candidate_side) -> BlockBGameResult`), `play_side_stratified_gauntlet`, `SecondStartParityLoop` (rolling p1/p2 score rates, `gap()`/`breach()`>0.12, `gap_for_promotion()`, `oversampling_scheme()` reusing A3). |
| `exit_to_c2.py` | (B7) `detect_h2h_plateau(h2h_scores, *, dominance_target, K_snap=10, min_gain=0.01, below_target_exits=True, best_checkpoint_path) -> ExitToC2Verdict`. Inverse of A5 trending. Default reading fires exit when plateau AND below target (→C2); flipped reading fires when plateau AND at/above target (→E1). |
| `league_v5.py` | `V5LeagueConfig`, `V5EpisodeModes`, `parse_v5_opponent_mix` (`"name:weight,..."`), `sample_v5_episode_modes` (deterministic), `compare_adaptive_strength_monotonicity`. |
| `gauntlet_v5.py` | `EXPLOIT_AGENT_KINDS` (7), `V5GauntletConfig`, `build_default_exploit_gauntlet`. |
| `opponents_v5.py` | Phase-9 broad opponent runtime (`V5OpponentLane`, `build_phase9_broad_opponent_lanes`, `prepare_phase9_broad_opponent_environment`, lane probes). |

### 4.5 Block C (RLHF loop + offline replay)

| File | Role |
|---|---|
| `offline_replay_bridge.py` | (C2) `build_offline_replay_batch(policy_fn, *, group_dirs, transitions, info_mode=omniscient, assist_mode, max_battles, strict) -> OfflineReplayBatch`. Consumes fresh HUMAN v5_trace rows (`battles/<bid>/v5/{meta,turns,actions}.jsonl`) with an **omniscient** `InfoModeV5` (enemy hand/deck/order known) so obs match the deploy encoder; resolves 601-tcode via `resolve_v5_tcode` (engine `action_native`); computes `old_log_prob`+`value` at bridge time; GAE per-game episodes. `make_policy_fn_from_checkpoint(checkpoint_path, ...)`. |
| `awac_crr_replay.py` | (C3) **Hybrid AWAC×PPO-clip** offline-PPO replay. `awac_crr_loss` (pure-numpy core; the **BLOCKER sign-fix**: `A` is the surrogate multiplier, NOT a sign-flipped `log pi` multiplier), `awac_weight`, `evaluate_awac_dense_batch` (MLX; retains mana_draw_logit), `train_awac_crr_replay(...)` (warm-start via `load_v4_max_into_v5`, freeze-faithful + value_head trainable, writes checkpoint), `AwacCrrReplay.run(...) -> AwacCrrMetrics` (MONITORING-ONLY — no promote field). |
| `c_loop_driver.py` | (C4) `CLoopDriver.run(n_iterations) -> CLoopManifest`. Per iteration: C2 collect → C3 replay → measure (external bench) → B6 promote → B1 snapshot → **D-C6 stall-counter** (exit to D at `stall >= k_stall=2`). `A5MeasurementRunner` (real measurement adapter). Injectable Protocols; MLX/Rust/rlhf_env are USER-wired, fakes for tests. |
| `exit_to_c2.py` | (B7, also the C2 entry signal) — see §4.4. |

### 4.6 Block E1 (export / tournament / ship / runner) + prod wiring

| File | Role |
|---|---|
| `export_onnx_v5.py` | (E1) `export_v5_checkpoint_to_onnx(checkpoint_path, output_path, *, opset=17, placement_mode=None) -> str` — **Has a CLI** (`python3 -m train_v3.export_onnx_v5 --checkpoint <npz> --output <onnx> [--opset] [--placement-mode]`). 3-output ONNX head (logits/value/mana_draw_logit), split-encoder mirror; sidecar `.onnx.json` with `model_version="v5_split_encoder_onnx_v1"`, `mana_draw_head:true`, `format:"v5"`, `obs_dim:7128`. |
| `e1_tournament.py` | (E3) `run_e1_tournament(config, *, game_runner, candidate_loader, side_runner=None) -> list[E1CandidateReport]`, `select_e1_winner(reports) -> Optional[E1CandidateReport]` (None = NO-SHIP), `E1TournamentConfig`, `E1CandidateReport.passed()` (threshold table: `h2h_vs_v4_orig`≥0.70, `h2h_vs_random/end_turn`≥0.95, `h2h_vs_self_snapshot_floor`≥0.52 + trending, no_assist, exploit_resistance, mana_draw band, p1_p2_gap≤0.12, throughput≥12000, entropy≥0.70, max_abs_kl≤0.12, no_bonus p1/p2/second≥0.70). `make_default_candidate_loader(policy)`. |
| `e1_human_qa_panel.py` | (E4) `run_e1_human_qa_panel(...) -> dict` — **SOFT** gate (E-E8); NEVER aborts ship. `McpCollectionClient` + `ReviewerScorecardClient`/`JsonScorecardClient` Protocols. |
| `e1_ship.py` | (E5) `ship_v5_winner(winner_report, *, onnx_export_fn, bundle_config) -> ShipResult` — GATED on `winner_report.passed()` (raises RuntimeError on None/False = NO-SHIP). Steps: export ONNX → `build_release_bundle` → `register_v5_kind_detector` (LIFO, ahead of V4) → verify prod wiring (`extra-lr-v5-max` in `BOT_MODEL_PROFILES`, 4 top tiers `brain_profile=="extra-lr-v5-max"`, derived `obs_dim==7128`). `ShipResult` (marker, prod_profile_key, onnx/sidecar/bundle/manifest paths, `fallback_guard_verified`). |
| `block_e1_runner.py` | (E-E12) **Has a CLI** (`python3 -m train_v3.block_e1_runner --manifest <json> --candidate-dir <dir> --output-dir <dir> --mana-draw-count <int> --eligible-turns <int> [--min-reviewers] [--min-battles] [--no-bonus-benchmark] [--skip-panel] [--battles-per-series]`). `run_e1_pipeline(manifest, *, game_runner, candidate_loader, c2_client, scorecard_client, mana_draw_baseline, bundle_config, min_reviewers, min_battles, run_panel=True, onnx_export_fn=...) -> Optional[ShipResult]` — INJECTABLE core; NO-SHIP paths return None (not raise). `build_e1_candidate_set_from_manifest`, `write_candidate_json`. 4 `build_production_*` stubs raise NotImplementedError (real RUN USER-wired). |

**Prod wiring (committed source, NOT call-time mutation):**

- `ai/bot_brain.py` — **additive** V5 branch (V4 byte-unchanged). `_V5_FORMAT="v5"`; relaxed
  gate `if profile_format not in (_TRAIN_V2_FORMAT, _V5_FORMAT)`; `_validate_v5_contract`
  (obs_dim 7128, 3-tuple, mana_draw_head); the `v5` elif in `get_action`; `_get_action_v5`
  (`:628`) — lazy-imports `ai.train_v2.{obs_v5,mana_draw_head_v5,classic_actions_1,v5_inference_guard}`
  + `core.actions.ManaDrawAction`; `session.run(["logits","value","mana_draw_logit"], ...)`;
  `except RuntimeError: raise` BEFORE generic `except Exception: return _legal_fallback`
  (SPEC :174); wires `mana_draw_legal_mask` + `select_includes_mana_draw`.
- `ai/train_v2/` — **vendored** V5 live-path copies: `obs_v5.py`, `v5_contracts.py` (renamed
  from `contracts.py`; the one `from .contracts`→`from ai.train_v2.v5_contracts` rewrite is
  the only intentional divergence), `mana_draw_head_v5.py`, `v5_inference_guard.py`.
  `test_vendored_obs_v5_byte_faithful_to_train_v3` guards byte-faithfulness.
- `infrastructure/config.py` — `extra-lr-v5-max` profile (format v5, obs_dim 7128,
  action_feature_dim 171, max_candidate_actions 601, mana_draw_head True,
  placement_mode append_only, verify_mask False). Retarget ONLY 4 top tiers
  (`tier_hard_4500`, `tier_hard_plus_6000`, `tier_max_minus_7500`, `tier_max_9000`)
  `brain_profile`→`extra-lr-v5-max`; 8 non-top tiers stay `extra-lr-v4-{micro,lite,opti}`.
  `BOT_DIFFICULTY_PROFILES` derives automatically (NO edit).
- `e1_ship.register_v5_kind_detector` — LIFO: inserts `v5_detector` at registry index 0,
  AHEAD of `_sidecar_kind_detector` (a V5 sidecar also satisfies the V4 detector's
  `inputs`/`action_feature_dim` OR-branches → would misclassify as `action_onnx` without
  the V5 detector first). Idempotent. Does NOT re-register the taken V5 factory slot.

### 4.7 Scripts, fixtures, tests

**Runnable scripts (`TrainV3.5/scripts/`):** V5 acceptance/benchmark —
`run_v5_acceptance.py`, `run_v5_adaptive_strength_benchmark.py`,
`run_v5_vs_v4max_benchmark.py` (each argparse; see TRAINING_GUIDE for flags). Phase
experiment scripts — `run_phase1_foundation.py` … `run_phase37_lane_pairwise_repair.py`
(each a V5 phase experiment; most have a `main()`/argparse, some run at import). Status
helpers — `phaseN_status.py`, `phase25_training_dashboard.py`, `select_phase1_checkpoint.py`,
`audit_phase1_actions.py`, `run_frontier_llm_showmatches.py`.

**Fixture generators (`python/train_v3/gen_phase*_fixtures.py`,
`regen_action_fixtures.py`, `regen_obs5_fixtures.py`, `gen_e2e_oracle_fixture.py`):**
rebuild golden-trace fixtures for specific mechanics (rebirth, cleave, freeze,
consume_ally, mana_drain, cast_random_spell, …). Importable (call `gen_*()`/`main()`).

**Tests (`python/train_v3/tests/`, 24 files):** one per component —
`test_a_gate`, `test_block_b_gate`, `test_block_b_league_driver`, `test_block_b_opponent_mix`,
`test_block_d_league_driver`, `test_block_d_opponent_mix`, `test_block_e1_runner`,
`test_c_loop_driver`, `test_c_to_d_handoff`, `test_curriculum`, `test_e1_human_qa_panel`,
`test_e1_ship` (+ prod wiring), `test_e1_tournament`, `test_exit_to_c2`, `test_export_onnx_v5`
(+ guard), `test_offline_replay_bridge`, `test_ppo_phaseA_config`, `test_rust_live_self_play`,
`test_second_start_parity`, `test_snapshot_pool`, `test_v4_orig_temp_spectrum`,
`test_awac_crr_replay`, `test_bc_dataset`, `test_bc_train`.

---

## 5. The V5 observation, action, and reward contracts

These three contracts are the load-bearing interface between training, Rust, and prod.

### Observation (`obs_v5.encode_observation_v5`) — 7128 floats
```
[   0:1456]  V1 frozen base (encode_observation)            -- the V4 observation
[1456:1488]  V5 global (32): adaptive_strength, hand/deck-known flags,
             history_fill_ratio, draw_assist_enabled/strength, assist channels,
             mana_draw_count_this_turn (normalized by 5.0)
[1488:3888]  private info (32 slots x 75 = 2400): [occupied, card_id_norm, card_shape_v5(73)]
             zones: own hand(4), own deck(12), enemy hand(4), enemy deck(12);
             unknown zones zeroed per InfoModeV5
[3888:7128]  history (20 events x 162 = 3240): right-aligned recent events;
             each event = 16 metadata floats + source card shape(73) + target card shape(73)
```
`CARD_SHAPE_DIM_V5=73` (grown, disjoint from frozen V4 `CARD_SHAPE_DIM=64`) — the Block-0
decision (`CARD_SHAPE_DIM_V5=73`).

### Action — 601 candidate logits + a parallel binary mana_draw head
- `MAX_CANDIDATE_ACTIONS=601` (frozen, V4-inherited), `ACTION_FEATURE_DIM=171`.
- The policy emits `(candidate_logits(B,601), value(B,), mana_draw_logit(B,))`.
- **mana_draw is a PARALLEL BINARY HEAD, not a 602nd candidate.** When
  `select_includes_mana_draw(mana_draw_logit, best_candidate_logit, mana_draw_legal)` fires
  AND mana_draw is legal, the engine's `ManaDrawAction` index in `legal_actions` is returned.
  The 601 candidate space is never modified.
- Placement mode `append_only` (mana_draw slot appended after the 601), `verify_mask=False`
  in the live V5 profile.

### Reward (`reward_v5.compute_weighted_reward_v5`)
`base_reward + shaping`, where shaping = `hp_potential_delta`·w + `board_power_delta`·w/normalizer
− `board_under_0_7_penalty` − `own_board_wiped_penalty`, scaled by an `informed_multiplier`
(1.0 + penalties if enemy private info known / draw-assist enabled), clipped to
`±max_shaping_abs` (0.06). The learner-only reward attribution (`ppo_phaseA_config.reward_attribution`)
zeros rewards where the actor ≠ the learner (fix #1).

---

## 6. How the blocks compose (data flow)

```
                         ┌─────────────────────────────────────────────────────┐
   Block -1 (Rust ArenaEnv parity) ──┐
   Block 0  (V5 card-shape 73, warm-start, offline-bridge loader)
                         │
                         ▼
   Block A:  bc_dataset(A1) -> bc_train(A2) -> PhaseAPPOConfig(A3) -> rust_live_self_play(A4)
             -> a_gate(A5)  [Pilot/BC/redesigned PPO + A-gate promotion]
                         │  (A5 select_promotion: A-gate passed AND H2H>0.5)
                         ▼
   Block B:  snapshot_pool(B1) -> v4_orig_temp_spectrum(B2) -> opponent_mix(B3)
             -> curriculum(B4) -> second_start_parity(B5) -> block_b_gate(B6)
             -> exit_to_c2(B7) -> block_b_league_driver(B8)
                         │  (B7 plateau-below-dominance fires exit_to_c2=True)
                         ▼
   Block C:  c2 collection (rlhf_env human battles) -> offline_replay_bridge(C2)
             -> awac_crr_replay(C3, Hybrid AWACxPPO-clip) -> c_loop_driver(C4)
             [C2->C3 loop; D-C6 stall>=2 exits to D]
                         │  (CLoopManifest.best_ever_path)
                         ▼
   Block D:  c_to_d_handoff(D3, E1CandidateSet) -> block_d_opponent_mix(D1)
             -> block_d_league_driver(D2)
                         │  (BlockDLeagueManifest: exited_to_e1 + candidate_paths)
                         ▼
   Block E1: build_e1_candidate_set_from_manifest -> run_e1_tournament(E3)
             -> select_e1_winner  (None = NO-SHIP)
             -> run_e1_human_qa_panel(E4, SOFT) -> write_candidate_json
             -> ship_v5_winner(E5: export ONNX + bundle + LIFO V5 detector + verify)
                         │
                         ▼
   PROD:    infrastructure/config.py extra-lr-v5-max profile (4 top tiers)
          + ai/bot_brain.py BerserkInference._get_action_v5 (vendored ai/train_v2 encoders)
          + ONNX fallback guard (SPEC :174, RuntimeError last-resort)
```

**Operational vs in-worktree split (load-bearing):** the in-worktree code is the
importable machinery; the operational RUNs (A-gate gauntlet, B/D league, C loop, E1
tournament) are **USER-run** — they wire the real `RustBatchWorker` (via `worker_factory`),
MLX `model`/`optimizer`, a real `rlhf_env` MCP client (port 8090), and the v4-max ONNX.
`block_e1_runner.build_production_*` stubs raise `NotImplementedError` precisely because
the operational factories are USER-supplied. The field-test (`BLOCK_E1_FIELDTEST.md`)
proved the machinery runs live by injecting fakes for those operational pieces.

---

## 7. Key decisions (the E*-style pins that shaped the pipeline)

- **Block 0:** `CARD_SHAPE_DIM_V5=73` (grow, disjoint from frozen 64); `HAND_CAP=4` (no
  hand lift); PARTIAL V4→V5 warm-start (faithful + shape-compat copied, fresh V5 layers).
- **Block A:** 5 root-cause fixes (learner-only reward, max_turns=120, pinned entropy 0.01
  + epochs 6, graduated opponent_mix, second-start oversampling); A-gate is a NEW build,
  not wiring; `rust_live_self_play` is the missing live entry.
- **Block B:** snapshot pool with 2 immutable anchors; v4-orig temp spectrum (3 opponents
  from 1 ONNX); frozen non-self share 0.95; curriculum cap 0.25 (D-B8); p1/p2 gap 0.12.
- **Block C:** D-C0 build in worktree `rlhf_env`; D-C1 Hybrid AWAC×PPO-clip (the BLOCKER
  sign-fix: `A` as multiplier); D-C4 5k mana_draw actions; D-C6 aggregate-stall exit (K=2);
  omniscient bridge InfoModeV5; value_head trainable.
- **Block D:** D-D1 consolidation self/peer ~0.50; D-D2 fresh pool from post-C; D-D3 fixed
  short schedule; D-D4 curriculum OFF (no per-lane-loss reweight); NO new dispatch identities.
- **Block E1:** E-E1 NEW `export_onnx_v5.py`; E-E2 extend `BerserkInference` (additive, V4
  byte-unchanged); E-E3 1e-4 measure-then-pin; E-E5 hard gate trending AND latest≥0.52;
  E-E6 no_bonus self-snapshot≥0.70 each + V4-max advisory; E-E7 sanity gates≥0.95; E-E8
  Hybrid (component + USER-run, SOFT gate); E-E11 max-only `extra-lr-v5-max`;
  E-E12 Hybrid (thin runner + USER-run); **E-E13 LIFO V5 detector load-bearing**; E-E14
  faithful-layer bonus.

---

## 8. Local checks

```bash
# From the worktree root:
cargo build --release                           # in TrainV3.5/rust/trainv3_core
export TRAINV3_CORE_LIB="$PWD/TrainV3.5/target/release/libtrainv3_core.dylib"
export PYTHONPATH="$PWD:$PWD/TrainV3.5/python"

# Rust golden kernel:
cargo test --manifest-path TrainV3.5/rust/trainv3_core/Cargo.toml

# Python component tests:
python3 -m pytest TrainV3.5/python/train_v3/tests/

# A single component, e.g. the E1 runner:
python3 -m pytest TrainV3.5/python/train_v3/tests/test_block_e1_runner.py
```
Baseline on `main`: the full Python suite is green; the Rust `golden_kernel` is 47/0.
See `MEMORY.md` (the project memory) for the per-block commit/test tallies.

---

## 9. Further reading

- `TRAINING_GUIDE.md` (sibling) — the step-by-step "how to train a V5 model" recipe.
- `TrainV3.5/README.md` — legacy V4-era Rust-boundary prep + golden-kernel CLI.
- `TrainV3.5/BLOCK_*_PLAN.md` / `BLOCK_*_COMPLETION.md` — per-block design + delivery.
- `TrainV3.5/BLOCK_E1_FIELDTEST.md` — the live end-to-end proof (8/8 smoke steps).
- `docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-design.md` — the V5-Max design spec.
