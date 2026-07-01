# Block A — V5 Pilot → BC → Redesigned PPO — IN-WORKTREE COMPLETION LOG

**Branch:** `glm-5.2/TrainV3.5Prep` (worktree `.claude/worktrees/glm-TrainV3.5Prep`)
**Scope:** the V5-Max Block A in-worktree components — BC dataset + BC training loop + Phase-A PPO config (5 root-cause fixes) + the MISSING live-self-play entry point + the A-gate/promotion/Q4 measurement.
**Status:** ✅ **ALL 5 IN-WORKTREE COMPONENTS COMPLETE** (2026-07-01). Each done + independently re-verified green (source-vs-source); ultracode workflow (implementer + 4 refute-by-default verifiers + fix) PASS with 0 blocker/major on each. The operational steps (pilot deploy → BC run → PPO training run → A-gate measurement) are USER-run (D-A3=yes/soon) — the code is built + synthetic-tested regardless.
**Pipeline position:** `Block -1 ✅ → 0 ✅ → A ✅ (in-worktree, this) → [operational steps] → B → C → D → E1`.

---

## 1. Commits (chronological)

| Commit | Component | Summary |
|---|---|---|
| `152d2350` | A1 | `bc_dataset.py` (601-tcode resolver via decode_action matching vs ENGINE-sourced action_native [source-vs-source, forked helper — not the legacy self-referential one] + decision_source=='human' filter + BCTransition) + additive `offline_dataset_loader.py` (action_native + pre_state_snapshot + meta.decision_source, regression-free 6/6) |
| `0fd9a1bb` | A2 | `bc_train.py` (greenfield BC training loop; masked CE on 601 candidates + mana_draw BCE; freeze_faithful=True freezes Q3 FAITHFUL layers byte-identical; MLX; BC-seed checkpoint via model_mlx.save_checkpoint; skip-gate on npz absence) |
| `9c37815c` | A3 | `ppo_phaseA_config.py` (PhaseAPPOConfig standalone superset of RustPPOTrainingConfig + 5 root-cause fixes; pure-python no MLX/FFI; to_rust_ppo_config lazy for A4) |
| `70c9df46` | A4 | `rust_live_self_play.py` (THE MISSING live-self-play entry point D-A8=build new; composes existing FFI primitives; dispatch split 6 rule-agent + 4 policy-opponent grounded in worker.rs; max_turns FFI + learner-only reward + second-start oversampling) + additive FFI accessors (+203 rust_ffi.py / +26 ffi.rs / +16 worker.rs, 0 deletions) |
| `30f96ff1` | A5 | `a_gate.py` (NEW BUILD not wiring; A-gate 4 criteria AND + external-bench promotion selector + promotion-by-loss guard [internal metrics never read] + Q4 mana_draw measurement with HAND_CAP/MANA_DRAW_BASE invalidation guard; injectable GameRunner) |

Combined suite (final, independently re-confirmed): **python 111 (62 A5 + 27 A4 + 6 A3 + 10 A2 + 6 A1) + Block-0 regression (warm_start 15 + offline_bridge 6) + cargo 157 (110 lib + 47 int, 0 failed).**

---

## 2. Components (what was built)

### A1. `bc_dataset.py` (601-tcode resolver + human filter) — `152d2350`
BC dataset builder consuming the offline_dataset_loader PUBLIC API (Block 0 c5). Resolves the V5 601-tcode Block 0 deferred.
- `resolve_v5_tcode(pre_state, actor, action_native)`: build append_only 601-mask (FROZEN `build_action_mask`), enumerate `np.flatnonzero(mask)`, decode each via FROZEN `decode_action`, value-equality match `.to_dict()` against the ENGINE-sourced `action_native` (v5_trace.py:481 `legal[legal_index].to_dict()` — the engine's own BaseAction, NOT a decode_action output → TRUE codec-vs-engine source-vs-source). The legacy `test_train_v2_offline_bridge._write_real_trace` sources `action_native=decode_action(...).to_dict()` (codec under test) = self-referential decode-vs-decode (cannot catch a codec regression) — A1 FORKS the test helper sourcing `action_native=legal_raw[idx].to_dict()` from `env._env.get_legal_actions` (engine oracle).
- `build_bc_dataset`: filters `decision_source=='human'` BEFORE tcode resolution (pilot deploys placeholder BOT vs humans → traces contain both; bot/rl excluded — CRITICAL FIX B); mana_draw row → `target_tcode=None`+`is_mana_draw=True` (BC targets the parallel head, NOT a 601 slot); terminal rows (surrender/draw/stalemate) → no 601 target. `action_features` REBUILT with `placement_mode='append_only'` (the loader's `'full'` over-includes warriors at non-append positions the engine does NOT offer — `core/engine.py:1260` emits warriors only at `position=len(board)`).
- `BCTransition{obs(7128), action_features(601,171), target_tcode(0..600|None), is_mana_draw, mana_draw_legal, legal_mask(601), reward, terminal, meta}`.
- Loader additive extension (regression-free 6/6): `OfflineTransition` gains `action_native` + `pre_state_snapshot` + `meta['decision_source']` (single source of truth — BC does NOT re-read actions.jsonl).
- Tests `test_bc_dataset.py` (TRACKED): 6 source-vs-source (live ClassicRLEnv=oracle): tcode round-trip over a REAL battle (end_turn+play_card warrior+potion+attack+mana_draw, action_native=legal_raw[idx].to_dict() ENGINE oracle, warrior+potion pinned), resolve_v5_tcode_unit, mana_draw_row_targets_head, legal_mask_matches_action_features, orphan_and_terminal_skip (surrender+draw+stalemate all three), decision_source_human_filter (mixed human+bot+rl → only human emit).

### A2. `bc_train.py` (BC training loop, greenfield) — `0fd9a1bb`
No prior BC infra existed. 3-stage: prepare_bc_policy (load V5ActionConditionedPolicy → warm_start_v5.load_v4_max_into_v5 READ-ONLY, SKIP-GATED on npz absence → fresh-init fallback, no crash) → train_bc gradient loop (nn.value_and_grad → _zero_frozen_grads → mlx.optimizers.Adam.update → mx.eval) → run_bc_training (write BC-seed checkpoint via `ai.train_v2.model_mlx.save_checkpoint`, SAME signature as `rust_trainer._save_checkpoint:838` → PPO A4 resumes via load_checkpoint).
- Loss = candidate_CE + mana_draw_bce_weight*mana_draw_BCE (weight 1.0 default, D-A9 plain CE BC; AWAC/CRR reserved for Block C). value_head OFF.
- Masked candidate CE: `mx.where(legal_mask, logits, -1e9)` (numerically-stable -inf; exp(-1e9)→0 so softmax over legal unaffected + logsumexp finite for all-False rows — avoids the NaN a real -inf produces); target=target_tcode (-1 for None); valid=target>=0; CE mean over valid rows only; mana_draw+terminal rows contribute nothing.
- mana_draw BCE: forward called with `mana_draw_legal=None` so head returns RAW logit (BCE on -inf-gated logit would NaN); `md_p=clip(sigmoid,1e-7,1-1e-7)`; mean over mana_draw_legal=True rows ONLY (mirrors `select_includes_mana_draw:116`); illegal rows masked + divide-by-zero guarded.
- freeze_faithful=True DEFAULT freezes via zeroing grads each step (MLX Adam zero-grad → EXACTLY zero update → BYTE-IDENTICAL across BC steps, np.array_equal verified): FAITHFUL (base_encoder.layers.0 + action_encoder — Q3 set warm_start_v5:67-73) + state_fuser.layers.2 (shape-compat, not in BC move set) + value_head (OFF). TRAINABLE = candidate_scorer + mana_draw_head + global/private/history encoders + state_fuser.layers.0. freeze_faithful=False ablation unfreezes FAITHFUL+state_fuser.layers.2. Param-name sets confirmed EXACTLY match live policy tree via nn.utils.tree_flatten (declared==live, partition covers all no overlap).
- Tests `test_bc_train.py` (TRACKED): 10 (4 spec + 6 extra): loss_decreases, loss_decreases_fresh, faithful_layers_preserved_after_bc (np.array_equal before vs after all 4 FAITHFUL arrays), freeze_faithful_false_unfreezes, mana_draw_head_learns_signal (directional: sigmoid increases for is_mana_draw=True rows), mana_draw_bce_only_on_legal_rows, skip_if_no_npz, skip_if_no_npz_via_env, checkpoint_round_trip (param key set unchanged + np.array_equal every param + forward allclose atol=1e-6 before vs after reload into fresh policy), run_bc_training_writes_checkpoint.

### A3. `ppo_phaseA_config.py` (Phase-A PPO config + 5 root-cause fixes) — `9c37815c`
PhaseAPPOConfig STANDALONE frozen @dataclass (verified field-name SUPERSET of RustPPOTrainingConfig `rust_trainer.py:23-71`: 47/47 base fields + 8 A3-only = 55). Standalone NOT inherited so the module imports WITHOUT the rust_trainer→rust_ffi chain (pure-python at import; rust_ffi lazy-load, no .dylib opens); `to_rust_ppo_config()` lazily imports RustPPOTrainingConfig for A4.
- Fix #1 learner-only reward (TRAINER-side): `learner_only_reward=True` + `reward_attribution(step_rewards, actor_ids, learner_actor)` → `np.where(actors==learner, rewards, 0.0)` (reward_v5.py NOT imported/edited — already per-side at :40; legacy bug source run_phase26:490 READ-ONLY, mirrored at trainer-config level only).
- Fix #2 max_turns FFI PLUMBING (the verify-caught no-op gap): `max_turns=PHASE_A_MAX_TURNS=120` + `build_trace_env_config(...)` pure-python dict for trace_factory_v5 to write into trace env_config (alongside level_handicap at :101) so kernel.rs:660 reads 120 not 80, + `LIVE_MAX_TURNS_THREADING_NOTE` for the live constructor + `decisive_early_end` flag + `is_decisive_state(snapshot, threshold=0.60)` predicate (win-margin, duck-typed to V5RewardSnapshot shape).
- Fix #3 entropy_coef PINNED 0.01 (NOT 0.035 phase26 override).
- Fix #4 epochs PINNED 6 (NOT 1/3).
- Fix #5 opponent_mix + second-start: `build_phase_a_opponent_mix_string()` canonical via PHASE_A_OPPONENT_NAME_ALIASES (display→canonical: legal_random→random, self_prev→self, v4-orig-argmax→v4max; other 7 identity) + `validate_phase_a_opponent_mix()` read-only parse_v5_opponent_mix + `second_start_oversampling_scheme(p1_rate,p2_rate)` gap-weighted (breach |p1-p2|>0.12, shift=min(0.5,gap-0.12)).
- VALIDATION vs DISPATCH (the gap): A3 VALIDATES the mix is well-formed + matches spec weights (read-only); runtime dispatch is A4 job. Test asserts DISPLAY names REJECTED by parser (pytest.raises), proving validation-only.
- Tests `test_ppo_phaseA_config.py` (TRACKED, pure-python — audit mlx modules loaded=[] + rust_ffi _LIBRARY_CACHE empty): 6 tests.

### A4. `rust_live_self_play.py` (THE MISSING live-self-play entry point) — `70c9df46`
D-A8=build new (spec-faithful). V5 training previously ran ONLY on the golden-trace pool (train_rust_ppo_trace_files via trace_factory_v5; RustBatchWorker only had from_trace_file/from_trace_files). A4 builds the live-self-play runner on the Rust ArenaEnv COMPOSING existing FFI primitives (no game logic reinvented; only additive FFI accessors exposed).
- `run_live_self_play_update` = top-level: build worker (RustBatchWorker.from_live, threads max_turns=120) → sample opponents (weighted graduated mix) + learner sides (D-A10 second-start oversampling) → collect_rust_live_rollout → prepare_rust_ppo_batch (reused unchanged, RustTransitionBatch format rust_collector.py:25-49) → optional train_rust_ppo_minibatch. Core loop = lock-step per-action batch loop (mixed rule+policy opponents in SAME batch; advance_rule_until_actor only for pure-rule batches, exposed as fast_forward_rule_opponent_turns which raises if any policy opponent).
- Per batch step: read arrays/current_actor_ids/mana_draw_legal; build per-env action_ids (LEARNER→learner policy + mana_draw head when legal; RULE-AGENT→select_rule_actions with integer code; POLICY-OPPONENT→opponent_policies[identity].select); step_mana_draw; A3 reward_attribution ZEROES opponent-actor steps (NOT zero-sum negation); record learner transitions; decisive-early-end via hero_hp + A3 is_decisive_state; reset on terminal/truncated. Returns LiveRolloutBatch (transitions + mana_draw_legal/taken + learner_actor_ids + opponent_identities + dispatch_log).
- **DISPATCH SPLIT (grounded in source, plan BLOCK_A_PLAN.md:404-419 AGREEs with worker.rs; source=oracle):** 6 RULE-AGENT (Rust integer codes via select_rule_action_for_state worker.rs:1285): random=0 (select_deterministic_legal_random_action), face_rush=1, board_control=2, greedy_trade=3, stall=4, anti_draw_greed=6 (codes 5 PunishEmptyBoard + 7 AntiHandLeakOverfit exist in worker.rs:1252 but NOT in Phase-A graduated set; unknown raises worker.rs:1297); 4 POLICY-OPPONENT (Python loop mirroring rollout_worker.py:211-227): end_turn (EndTurnPolicy action_id 0), greedy_face (GreedyFacePolicy), self (SelfPrevOpponent wraps select_fn), v4max (V4MaxOpponent). Canonical aliases via PHASE_A_OPPONENT_NAME_ALIASES.
- FFI accessors added (additive, +203 rust_ffi.py / +26 ffi.rs / +16 worker.rs, 0 deletions): RustBatchWorker.from_live (live constructor composing from_trace_file with init-only GoldenTrace; threads max_turns=120→env_config→KernelConfig.max_turns kernel.rs:660 NOT serde-80 :624; defensive assert rust_ffi.py:1288), truncated, mana_draw_legal, step_mana_draw (kernel.rs:788), hero_hp, + current_actor_ids/select_rule_actions/arrays/reset_indices exposed. diagnostic_mode 'auto'→'none' resolved on live path. Reward: out.rewards=ACTING player's reward (kernel.rs:799 compute_trainv2_reward); A3 reward_attribution zeroes opponent steps.
- Tests `test_rust_live_self_play.py` (TRACKED): 27 tests BOTH layers. (1) FakeWorker composition (deterministic): TestDispatchSplit 5, TestCompositionDispatch 10 parametrized per identity (rule→select_rule_actions called with real code + rule action stepped NOT sentinel; policy→opponent_policies[ident].select called + sentinel 999 stepped), TestLearnerOnlyReward 2, TestDecisiveEarlyEnd 2, TestMaxTurnsThreading 2, TestSecondStartOversampling 2. (2) Gated real-FFI smoke 4 (TestRealFFISmoke, skip if lib missing; all 4 run+pass on this host — libtrainv3_core.dylib loads: test_one_ppo_update_seeded_arena_completes obs (4,2,7128)+learner_step_counts [4,4], test_max_turns_threaded_into_worker truncation s12>s6, test_learner_only_reward_on_real_run).

### A5. `a_gate.py` (A-gate + promotion selector + Q4 measurement, NEW BUILD) — `30f96ff1`
NEW BUILD (not wiring). Existing acceptance infra confirmed dead: V5GauntletConfig.no_assist_min_score_rate=0.45 (gauntlet_v5.py:42) + exploit_resistance_min_score_rate=0.42 (:46) DEAD (grep 0 reads); run_v5_acceptance.py plays NO games (reads pre-computed winrates :320-322 + checks config flags :488; :16 broken TrainV3 path); compare_adaptive_strength_monotonicity (league_v5.py:146) synthetic proxy; mana_draw band + H2H trending zero infra. A5 builds REAL measurement+gating.
- compute_score_rate=(wins+0.5*draws)/total (mirrors run_phase1_runtime_acceptance_bench.py:704,717). 4 named gate checks + evaluate_a_gate AND-aggregate (AGateResult.passed=all 4; failed_criteria() lists failures):
  (1) no_assist score rate>=0.55 (RAISED from dead 0.45; opponent EXPLOIT_AGENT_KINDS roster, candidate no-assist/hidden; 0.54 fails/0.55 passes).
  (2) exploit_resistance>=0.50 (RAISED from dead 0.42; opponent gauntlet_v5.py:8 EXPLOIT_AGENT_KINDS=face_rush/board_control/greedy_trade/stall/punish_empty_board/anti_draw_greed/anti_hand_leak_overfit; 0.49 fails/0.50 passes).
  (3) mana_draw usage in [0.5x,1.5x] of baseline B (inclusive both edges).
  (4) H2H vs best self-snapshot trending up over >=5 snapshots (DEFAULT_H2H_MIN_SNAPSHOTS=5 D-A5; most recent 5 non-decreasing within tolerance 0.0 strict; dip fails, plateau/equal passes [documented: non-decreasing not strictly-increasing]; insufficient fails).
- PROMOTION SELECTOR external-bench-only (design.md:112 D-lesson, gap #7): select_promotion promotes iff candidate PASSES A-gate AND STRICTLY beats current best on H2H (h2h_vs_best_score_rate>H2H_PROMOTION_THRESHOLD=0.5; tie=0.5=no promo; first snapshot no prior best→promote iff A-gate passes). **THE PROMOTION-BY-LOSS GUARD (load-bearing):** decision NEVER consults CandidateInternalMetrics (ppo_loss/approx_kl/entropy) — the deliberate ABSENCE of any read of internal fields IS the guard; lower-loss-but-A-gate-failing NOT promoted (reason=a_gate_failed); identical-external-different-internal same decision; internal metrics echoed back MONITORING ONLY.
- Q4 MEASUREMENT: B=mana_draw_count/eligible_turns (eligible_turns=turns where mana_draw legal: hand not full + mana sufficient for MANA_DRAW_BASE*(count+1) core/engine.py:784); record_mana_draw_baseline(count,eligible_turns,hand_cap=ENGINE_HAND_CAP=4 core/engine.py:44, mana_draw_base=ENGINE_MANA_DRAW_BASE=2 core/engine.py:59). DEPENDENCY GUARD (BLOCK_A_PLAN.md:532-538): is_baseline_valid False when HAND_CAP/MANA_DRAW_BASE changed → check_mana_draw_band invalidated=True + gate FAILS even if rate in-band; evaluate_a_gate threads current constants so invalidation propagates.
- Operational: GameResult/GameRunner Protocol + play_gauntlet + run_no_assist_gauntlet/run_exploit_resistance_gauntlet wrappers; GameRunner INJECTABLE (production wires A4 rust_live_self_play run_live_self_play_update/collect_rust_live_rollout; tests inject _FakeGameRunner) so unit-testable without MLX/Rust. build_a_gate_gauntlet_config overrides dead fields 0.45→0.55/0.42→0.50 (belt+suspenders). has_mlx_or_rust() skip-gate probes mlx+rust_ffi lazily.
- Tests `test_a_gate.py` (TRACKED): 62 synthetic tests (4 criteria+boundaries+each-fails-independently+mana_draw band within/outside+invalidation guard propagates+H2H trending up/regressing/plateau/insufficient+promotion selector+PROMOTION-BY-LOSS GUARD+GameRunner injection).

---

## 3. Validation (per-component ultracode: implementer + 4 refute-by-default verifiers + fix)

Each component ran an ultracode workflow: single max-effort implementer → 4 refute-by-default verifiers (parallel) → fix stage. Verdicts:

| Component | Implementer | Verifiers (findings) | blockerCount | Binding gate |
|---|---|---|---|---|
| A1 | 601-tcode source-vs-source + human filter | 4 PASS (2 minor resolved: terminal coverage, warrior+potion pin; 1 pre-existing Block-0 note) | 0 | decode_action(...)==ENGINE-sourced action_native round-trip (true codec-vs-engine) + human-only filter |
| A2 | masked CE + mana_draw BCE + freeze_faithful | 4 PASS (0 findings) | 0 | faithful layers BYTE-IDENTICAL (np.array_equal) after BC; mana_draw head directional signal; checkpoint round-trip |
| A3 | standalone superset + 5 fixes | 4 PASS (3 minor doc-accuracy nits, no code impact) | 0 | entropy==0.01/epochs==6 pins + max_turns=120 trace env_config plumbing (no-op guard) + opponent_mix validation-only (display names rejected) |
| A4 | compose FFI + dispatch split | 4 PASS (all minor: unused import, report wording) | 0 | dispatch split grounded in worker.rs:1285 + max_turns threaded into KernelConfig + learner-only reward + real-FFI smoke (27 tests, lib loads) |
| A5 | NEW gate + promotion guard + Q4 | 4 PASS (1 minor: H2H non-decreasing vs strictly-increasing, documented) | 0 | promotion-by-loss guard (internal metrics never read) + mana_draw band + HAND_CAP/MANA_DRAW_BASE invalidation + 4-criteria AND + boundary thresholds |

**Source-vs-source discipline held throughout** (Block -1/0 lesson): live engine/Python oracle = oracle, V5 code = UUT. A1 sources action_native from the ENGINE oracle (legal_raw[idx].to_dict(), NOT decode_action output) — the forked helper avoids the legacy self-referential trap. A4 grounds the dispatch split in worker.rs:1285 (source = oracle; plan AGREEs, no discrepancy). A5 confirms the dead fields via grep (0 reads) before building NEW.

---

## 4. Key architecture decisions (do not re-derive)

- **D-A8 = BUILD NEW LIVE SELF-PLAY** (USER decision). The spec's "self-play PPO on Rust ArenaEnv" did NOT exist; A4 builds it composing existing FFI primitives (RustBatchWorker.from_live + step + advance_rule_until_actor + select_rule_actions + the policy-opponent loop). Trace-pool fallback rejected (would degrade the 4 policy-opponent identities, 40% weight, to rule-agents-only).
- **D-A3 = YES/SOON** (USER decision). Pilot runnable — rlhf_env available, ruleset frozen, humans reachable. This unblocks the operational path; user runs the live pilot in parallel. The 5 in-worktree components are built + synthetic-tested regardless.
- **D-A9 = plain CE BC** (spec-literal). AWAC/CRR/offline-RL reserved for Block C; A2 uses masked CE + mana_draw BCE only.
- **D-A10 = gap-weighted p1/p2 oversampling** (second-start). A3 second_start_oversampling_scheme + A4 applies it.
- **Dispatch split = 6 rule-agent (codes 0-6) + 4 policy-opponent** (end_turn/greedy_face/self/v4max), grounded in worker.rs:1252-1297. Codes 5/7 exist in Rust but are NOT in the Phase-A graduated set.
- **freeze_faithful** (A2): freeze Q3 FAITHFUL layers (base_encoder.layers.0 + action_encoder) byte-identical so BC doesn't destroy warm start; only candidate_scorer + mana_draw_head + fresh layers move. Enforced via zeroing grads (MLX Adam zero-grad → exactly zero update).
- **Promotion = external-bench ONLY** (A5, D-lesson design.md:112): internal PPO loss/KL/entropy are MONITORING-ONLY; the promotion-by-loss guard is the deliberate absence of any read of internal fields. Promotion requires A-gate pass + strict H2H beat.
- **max_turns=120 FFI plumbing** (A3/A4): trace-pool writes env_config['max_turns']=120 (kernel.rs:660 reads 120 not serde-80 :624); live constructor threads max_turns into KernelConfig. The no-op gap (was a verify catch in planning) is fixed + guarded by tests.
- **mana_draw = parallel binary head (NOT 602nd candidate; 601 frozen).** A1 mana_draw rows → target_tcode=None + is_mana_draw=True (BC targets the head). A2 BCE only on legal rows. A4 step_mana_draw (kernel.rs:788).

---

## 5. Frozen-classic guard

Held throughout Block A. `classic_obs_v1` / `classic_actions_v1` / `classic_card_shape_v1` / `classic_rl_env.py` / `reward_v5.py` BYTE-LOCKED (never modified — read-only mirror/consume). `v5_trace.py` NOT imported into any BC/BC-train/live-self-play/gate code path (data-contract read-only; the bridge READS emitted JSON/JSONL). `core/state.py` NOT modified. `warm_start_v5.py` consumed READ-ONLY (git diff empty). `gauntlet_v5.py` / `league_v5.py` / `run_v5_acceptance.py` NOT modified (A5 imports gauntlet_v5 read-only for EXPLOIT_AGENT_KINDS; builds NEW gate logic, does NOT wire the dead no_assist_min_score_rate/exploit_resistance_min_score_rate fields). The only non-new-file modifications: A1 additive `offline_dataset_loader.py` (regression-free 6/6) + A4 additive FFI accessors (rust_ffi.py +203 / ffi.rs +26 / worker.rs +16, 0 deletions — confirmed no cargo regression, 157 green). Each component's verify phase confirmed `git diff` clean for frozen-classic + read-only sources. No TrainV3.5 import into prod paths.

---

## 6. Latent finding (uncovered by A1, flagged for Block C)

The loader's `OfflineTransition.action_features` is built with `placement_mode='full'` → OVER-INCLUDES warrior PlayCard candidates at non-append positions the engine does NOT offer (`core/engine.py:1260` emits warriors ONLY at `position=len(board)`). The loader's 6 Block-0 tests don't check action_features against the engine legal set (they check obs/reward/orphans/terminal/mana_draw_legal), so it passed. A1 works around it correctly by REBUILDING `action_features` with `placement_mode='append_only'` (engine-faithful). **Follow-up for Block C (AWAC/CRR consumers):** fix the loader to use `append_only` so `OfflineTransition.action_features` + its legal mask are engine-faithful. Does NOT affect A1/A2 (BC rebuilds its own).

---

## 7. Handoff — operational steps (USER runs, per D-A3=yes/soon)

The 5 in-worktree components are built + synthetic-tested. The operational steps are NOT autonomously doable; the user runs them:

1. **A.pilot deploy+collect** — deploy the placeholder policy (V4-orig ONNX, D-A1) vs humans in rlhf_env (port 8090), collect ~1-3k fresh battles (D-A2 scale ~2k). Records v5 traces (the A1 input) + measures Q4 baseline B (mana_draw_count/eligible_turns).
2. **A.BC run** — A1 bc_dataset consumes the pilot (human-only) → A2 bc_train fine-tunes (warm-started from V4-Max, freeze_faithful) → BC-seed checkpoint.
3. **A.PPO training run** — A3 config + A4 live-self-play (compute ~Medium 5-10k updates, D-A4) → short redesigned PPO with the 5 root-cause fixes. Snapshots N=5 (D-A5).
4. **A-gate measurement** — A5 measures the candidate vs the A-gate (no_assist≥0.55, exploit_resistance≥0.50, mana_draw∈[0.5×,1.5×]B, H2H trending) + the external-bench promotion selector. Candidate passes → promote to best-self-snapshot → Block B.

Q4 (mana_draw usage baseline B) measured over the pilot (operational; needs pilot data). A5 builds the measurement tooling now.

---

## 8. Session artifacts (workflow scripts)

Block A orchestration scripts (A1-A5 ultracode workflows: implementer + 4 refute-by-default verifiers + fix) persisted under the session workflows directory. Plan: `BLOCK_A_PLAN.md`. This completion log: `BLOCK_A_COMPLETION.md`.

See project-memory `extra-lr-v5-blocka-plan.md` (Block A plan + per-component execution status + decisions), `extra-lr-v5-block0-plan.md` (Block 0 complete), `extra-lr-v5-blockminus1-port-progress.md` (pipeline state), `extra-lr-v5-pipeline.md` (V5-Max design + decision ledger).