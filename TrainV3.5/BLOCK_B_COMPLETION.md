# Block B — V5 League on Rust ArenaEnv — IN-WORKTREE COMPLETION LOG

**Branch:** `glm-5.2/TrainV3.5Prep` (worktree `.claude/worktrees/glm-TrainV3.5Prep`)
**Scope:** the V5-Max Block B in-worktree components — snapshot pool (B1) + V4-orig temperature spectrum (B2) + Block-B opponent mix (B3) + per-lane curriculum (B4) + second-start parity loop (B5) + Block-B external-bench promotion gate (B6) + plateau/exit→C2 detector (B7) + the multi-update live league driver (B8). 8 components, dependency-ordered, synthetic-testable via A4 FakeWorker + A5 _FakeGameRunner.
**Status:** ✅ **ALL 8 IN-WORKTREE COMPONENTS COMPLETE** (2026-07-01). Each done + independently re-verified green (source-vs-source); ultracode workflow (implementer + 4 refute-by-default verifiers + fix) PASS / PASS-WITH-FINDINGS with 0 blocker/major remaining on each. The league RUN is USER-run after A-gate PASS (D-B12=not-yet) — the code is built + synthetic-tested regardless.
**Pipeline position:** `Block -1 ✅ → 0 ✅ → A ✅ → [A operational steps → A-gate PASS] → B ✅ (in-worktree, this) → [Block B league RUN] → C → D → E1`.

---

## 1. Commits (chronological)

| Commit | Component | Summary |
|---|---|---|
| `cbfaa647` | B1 | `snapshot_pool.py` (SnapshotEntry NO internal metrics = promotion-by-loss guard + SnapshotPool; FIFO non-anchors + 2 immutable anchors [seed, best-ever]; load-as-SelfPrevOpponent select_fn A4:279; self_snapshot_prevalence_weight) |
| `f2e651d6` | B2 | `v4_orig_temp_spectrum.py` (3 frozen identities from 1 V4 ONNX: argmax 0.40 / t07=0.7 0.20 / t12=1.2 0.15; TempV4Opponent adapter reproduces onnx_policy.py:16 OnnxActionPolicy math from packed OpponentCtx; mana_draw-collapse monitor hook exposed) |
| `2a9da617` | B3 | `block_b_opponent_mix.py` (BLOCK_B_IDENTITIES superset 11 + parse/build/collapse_reweight_boost) + B1 FIX (frozen_non_self_share 0.75→0.95, prevalence cap 0.25→0.05) + A4 uncomment `punish_empty_board`:5 (D-B10) |
| `279588b4` | B4 | `curriculum.py` (extract_lane_outcomes adapter [outcomes from transitions.rewards sign, NOT dispatch_log which carries no win/loss] + CurriculumReweighter rolling per-identity loss + reweight cap 0.25 D-B8) |
| `60801578` | B5 | `second_start_parity.py` (continuous p1/p2 parity loop closing A4 sample_learner_sides open loop; BlockBGameResult side-stratification composes A5 GameResult; oversampling reuses A3 scheme; gap_for_promotion exposes to B6) |
| `cd71fc3e` | B6 | `block_b_gate.py` (Block-B external-bench promotion gate EXTENDS A5: H2H-vs-best + gauntlet + mana_draw band + p1_p2 gap, monotone ≥ N_snap=5 D-B1; BlockBGateResult frozen; does NOT re-apply A-gate no_assist/exploit_resistance) |
| `778495fe` | B7 | `exit_to_c2.py` (plateau detector INVERSE of A5 check_h2h_trending; detect_h2h_plateau run-length K_snap=10 D-B2; D-B3=below: exit when plateau AND h2h<0.55; flippable) |
| `db9569a5` | B8 | `block_b_league_driver.py` (multi-update LIVE league driver composing A4 run_live_self_play_update + B1-B7; minimal ADDITIVE A4 extension closes v4-orig-* dispatch gap; LIVE path not trace-pool; continues A3 frozen hyperparams D-B7) |

Combined suite (final, independently re-confirmed after B8): **python 197 passed / 1 skipped** (the MLX/Rust skip-gate `test_skip_if_no_mlx_or_rust`), 0 failed. B8+A4 = 40 passed. Cargo 157 unchanged (no Rust edit in Block B).

---

## 2. Components (what was built)

### B1. `snapshot_pool.py` (snapshot pool + 2 anchors) — `cbfaa647`
ZERO existing infra (A5 tracks a single current_best; `rust_trainer._save_checkpoint:802` linear only). B1 builds the league pool.
- `SnapshotEntry{path, policy_ref, identity, is_anchor, metrics}` — NO `ppo_loss`/`approx_kl`/`entropy` fields = **promotion-by-loss guard** (the deliberate absence of internal fields; league promotion never consults training internals).
- `SnapshotPool`: capacity ~6 non-anchors + 2 immutable anchors (seed, best-ever); FIFO eviction of non-anchors ONLY (anchors never evicted); `set_seed_anchor` / `add_snapshot` / `maybe_update_best_ever`; `load_as_self_prev_opponent(select_fn)` → `SelfPrevOpponent` (A4:279) so a stored snapshot dispatches as the `"self"` policy-opponent identity; `self_snapshot_prevalence_weight()` → the share of self-snapshots in the mix (B3 D-B5 hybrid, grown 0→cap as pool fills).
- **B3 FIX (spec-literal):** `frozen_non_self_share` 0.75→0.95 (B1's 0.75 counted only V4-orig, ignoring also-frozen exploit 0.15 + tail 0.05; spec-literal frozen non-self=0.95, residual self-snapshot=0.05); prevalence cap 0.25→0.05 + saturation test + regression guard `weights[6]<0.10`.
- Tests: 7/7 green.

### B2. `v4_orig_temp_spectrum.py` (3 identities from 1 V4 ONNX) — `f2e651d6`
- 3 frozen identities from ONE V4 ONNX: `v4-orig-argmax` 0.40 / `v4-orig-t07` (T=0.7) 0.20 / `v4-orig-t12` (T=1.2) 0.15 (D-B6 temps). `V4_ORIG_TEMP_ALIASES` + `V4_ORIG_TEMP_WEIGHTS`.
- `TempV4Opponent` adapter reproduces `ai/train_v2/onnx_policy.py:16 OnnxActionPolicy` math from packed `OpponentCtx` (mlogits=`np.where` :90, argmax :101 OR `scaled/temp` :93-99, legal-fallback :103-106) — env-based `select_action` vs A4 `PolicyOpponent.select(env_idx, ctx)` packed arrays → adapter, NOT pure pass-through.
- mana_draw-collapse monitor hook exposed (the wiring/decision logic deferred to B8 driver).
- Tests: 9 passed / 1 skipped (V4-ONNX skip-gate on npz absence).

### B3. `block_b_opponent_mix.py` (Block-B mix + alias) — `2a9da617`
- `BLOCK_B_IDENTITIES` superset (11 names: self, v5_snapshot, v4-orig-argmax, v4-orig-t07, v4-orig-t12, stall, anti_draw_greed, punish_empty_board, greedy_face, random, end_turn).
- `parse_block_b_opponent_mix` mirrors `league_v5.parse_v5_opponent_mix:43-60` (raise on unknown, skip weight≤0, default self:1.0) but ACCEPTS `BLOCK_B_IDENTITIES` — v4-orig-* do NOT route through `parse_v5_opponent_mix` (frozen-classic NOT edited; v4-orig-* absent from `V5_OPPONENT_KINDS` so B3 owns its validator). Alias covers v4-orig-* only (`punish_empty_board` parses natively via `*EXPLOIT_AGENT_KINDS`).
- `build_block_b_opponent_mix(pool, collapse_boost=1.0)` D-B5 hybrid: `self_snapshot_weight = pool.self_snapshot_prevalence_weight()` (grown 0→0.05); `non_self_budget = 1 - self` in frozen RATIOS (V4-orig 0.40/0.20/0.15 + exploit 0.05 + tail); always sums 1.0.
- `collapse_reweight_boost(factor)` entry point (the mana_draw-collapse monitor logic wired by B8).
- A4 uncomment `rust_live_self_play.py:143` `"punish_empty_board": 5` (D-B10; zero Rust change, worker.rs:1258 PunishEmptyBoard pre-existing). Stale prose refreshed (6→7 rule, 10→11 graduated; PHASE_A mix stays 10).
- Tests: 47/47 green; full train_v3 140 passed/1 skipped at this point.

### B4. `curriculum.py` (per-lane loss tracker + reweight) — `279588b4`
- `LaneOutcome{identity, outcome∈{win,loss,draw}}` (frozen). `extract_lane_outcomes(rollout)` adapter: per env i, `identity=rollout.opponent_identities[i]`, `outcome` by sign of `transitions.rewards[:,i].sum()` — **CORRECTS the plan's loose "from dispatch_log" wording** (A4 `dispatch_log:765-773` entries carry NO win/loss field).
- `CurriculumReweighter(window_n)`: `deque(maxlen)` rolling per-identity win/loss/draw; `per_lane_loss_rate = losses/(wins+losses)` (draws excluded, 0.5 neutral default); `reweight(mix, cap=0.25)` boosts ONLY `loss_rate>0.5` lanes by factor `1.0+min(loss_rate-0.5, cap)` (max 1.25×, D-B8); 100%-beaten + no-data lanes factor 1.0; renormalize sum 1.0; pure.
- Scope: per-lane-loss ONLY — the mana_draw-collapse monitor is NOT in B4 (wiring is B8 driver). B4 PRODUCES the mix; B8 wires it into A4 `sample_opponent_identities`.
- Tests: 10/10 green.

### B5. `second_start_parity.py` (continuous p1/p2 parity loop) — `60801578`
- Closes A4 `sample_learner_sides:490` open loop (A4 accepts p1/p2 rates as INPUTS, never measures; B5 MEASURES over a rolling window of side-stratified gauntlet games + feeds back via A3 scheme + exposes gap to B6).
- `BlockBGameResult` (frozen) COMPOSES A5 `a_gate.GameResult:685` + `candidate_side` ("p1"/"p2", isinstance guard, A5 frozen unchanged). `BlockBGameRunner` Protocol `play(opponent_kind, *, seed, candidate_side)` ADDITIVE over A5 `GameRunner:711`. `play_side_stratified_gauntlet` plays each opponent from BOTH sides (mirrors A5 `play_gauntlet:753`).
- `SecondStartParityLoop(window_n)`: `deque(maxlen)`; p1/p2_score_rate measured SEPARATELY per side `((wins+0.5*draws)/total)`; `gap=abs(p1-p2)`; breach `>0.12`; `oversampling_scheme()` REUSES A3 `second_start_oversampling_scheme:258` (lower-rate side oversampled); `gap_for_promotion()` exposes to B6 (B5 produces, B6 consumes, NO B5→B6 dep). `BLOCK_B_GAP_THRESHOLD=PHASE_A_P1_P2_GAP_THRESHOLD` single-sourced.
- Dead `gauntlet_v5.p1_p2_max_score_gap:43` NOT wired (AST guard `test_does_not_wire_dead_field`).
- Tests: 12/12 green.

### B6. `block_b_gate.py` (Block-B external-bench promotion gate) — `cd71fc3e`
EXTENDS A5 (design.md:121: promote iff FULL 4-component aggregate improves monotonically ≥ N_snap=5 D-B1).
- `BlockBGateResult` (frozen): `h2h_vs_best`, `gauntlet`, `mana_draw_band`, `p1_p2_gap` (all A5 `GateOutcome`) + `passed` + `monotone_aggregate_history` + `n_snap`; NO `no_assist`/`exploit_resistance` fields (A-gate = Phase-A EXIT design.md:114, NOT league promotion design.md:121 — regression guard `test_does_not_reapply_a_gate_no_assist`).
- `block_b_aggregate = h2h_rate + gauntlet_rate + (1 if in_band else 0) + max(0, 1-gap/0.12)` (gap>0.12 clamps parity→0 → dips aggregate → no promote).
- `evaluate_block_b_gate`: `mana_draw_band` REUSES A5 `check_mana_draw_band:309`; `p1_p2_gap` passed iff ≤0.12 (B5 `gap_for_promotion`); `passed = len≥n_snap AND last n_snap aggregates non-decreasing within tolerance AND all 4 pass most-recent`; first-snapshot → `passed=False` (seed-anchor); accepts `internal_metrics` but NEVER reads it (promotion-by-loss guard inherited A5 `select_promotion:607`).
- REUSES A5 `play_gauntlet:753` via `measure_gauntlet_rate`. Does NOT call `check_no_assist_gate`/`check_exploit_resistance_gate`/`evaluate_a_gate` (regression guard).
- Tests: 13/13 green.

### B7. `exit_to_c2.py` (plateau detector + exit→C2) — `778495fe`
- `ExitToC2Verdict` (frozen): `exit_fires, plateau, below_target, current_h2h, dominance_target, k_snap, min_gain, plateau_run_length, best_checkpoint_path, reason, details`.
- `detect_h2h_plateau(h2h_scores, *, dominance_target, K_snap=10, min_gain=0.01, below_target_exits=True, best_checkpoint_path=None)`: run-length plateau (per-step gain≤min_gain→run++, gain>min_gain→reset, `plateau=run≥K_snap`); `below_target=scores[-1]<dominance_target`. **DEFAULT D-B3=below: `exit_fires = plateau AND below_target`** (plateau at/above → reason `'dominant_plateau_e1_path'` → Block E1 ship, NOT C2). FLIPPABLE via `below_target_exits=False` (at/above reading) WITHOUT changing the default. Insufficient (`len<K_snap+1`)→no exit; still-improving→`plateau=False`. Carries B1 best-ever anchor path (C2 deploy candidate).
- K_snap=10 (D-B2 ~2× N_snap=5), min_gain=0.01, dominance_target=0.55 (D-B3). INVERSE of A5 `check_h2h_trending:374` (trending-UP).
- Tests: 10/10 green; full train_v3 185 passed/1 skipped at this point.

### B8. `block_b_league_driver.py` (multi-update LIVE league driver) — `db9569a5`
FINAL Block B component. Composes A4 `run_live_self_play_update` into the league loop.
- `BlockBLeagueDriver(config, *, snapshot_cadence=2000 [D-B4], n_snap=5 [D-B1], k_snap=10 [D-B2], dominance_target=0.55 [D-B3], pool, game_runner, learner_policy, opponent_policies_factory, curriculum, parity, seed, worker_factory)`.
- `run(n_updates)` per update: (a) `build_block_b_opponent_mix(pool)` (B3 D-B5 hybrid); (b) mana_draw-collapse monitor `_collapse_boost_for`: learner mana_draw rate from last rollout's `mana_draw_taken`/`mana_draw_legal` vs A5 band low edge (`MANA_DRAW_BAND_LOW=0.5*B`, a_gate.py:103) → B3 `collapse_reweight_boost(factor)`; (c) `CurriculumReweighter.reweight(mix, cap=0.25)` (B4 D-B8); (d) B5 `SecondStartParityLoop` p1/p2 rates → A4 `sample_learner_sides`; (e) A4 `run_live_self_play_update(opponent_mix_parsed=mix)` — **LIVE path, NOT `train_rust_ppo_trace_files`**; (f) `curriculum.update(extract_lane_outcomes(rollout))`; (g) every `snapshot_cadence` updates: B1 pool-add (seed anchor first via `set_seed_anchor`, then `add_snapshot` + `maybe_update_best_ever`) → external-bench gauntlet (`AsA5GameRunner` adapter wraps `BlockBGameRunner` for A5 `play_gauntlet`) → B6 `evaluate_block_b_gate` → B7 `detect_h2h_plateau` → emit exit→C2 if fires (carries `pool.best_ever.path`). Returns `BlockBLeagueManifest` (metrics list, snapshot_history, promotion_decisions, h2h_history, exit_verdict; reuses `train_rust_ppo_trace_files:92` + `_save_checkpoint:802` as STRUCTURAL template only).
- Continues A3 frozen hyperparams D-B7 (entropy=0.01/epochs=6/max_turns=120/learner_only_reward/decisive_early_end, unchanged — `test_continues_a_hyperparams`).
- MLX/Rust skip-gate: PPO step is MLX-gated INSIDE A4 (`model/optimizer None` → stops after `prepare_rust_ppo_batch`); the league loop (collect+curriculum+parity+snapshot+gate+plateau) runs WITHOUT MLX via `FakeWorker` + `_FakeGameRunner`.
- **KEY: minimal ADDITIVE A4 extension closes the v4-orig-* dispatch gap** (deferred from B2/B3). `rust_live_self_play.py`: `BLOCK_B_POLICY_OPPONENT_KINDS = frozenset({v4-orig-argmax, v4-orig-t07, v4-orig-t12})` (:169-171) SEPARATE from `POLICY_OPPONENT_KINDS`; `resolve_opponent_dispatch` (:212-213) additively returns `(POLICY_DISPATCH, None)` for those (AFTER the `POLICY_OPPONENT_KINDS` check, before the raise); `run_live_self_play_update` (:1015) optional `opponent_mix_parsed` param bypasses `parse_v5_opponent_mix` (which rejects v4-orig-*) so the Block-B mix passes `sample_opponent_identities`. `v5_snapshot` merged into the dispatchable `"self"` identity (`_merge_self_snapshot_split:662-692`) keeping the frozenset plan-faithful to the 3 v4-orig-*. B2 `TempV4Opponent` wired in `opponent_policies` for v4-orig-*. A4 counts UNCHANGED (`POLICY_OPPONENT_KINDS=4` / `PHASE_A_IDENTITIES=11` / `RULE_AGENT_CODES=7`, `test_does_not_break_a4_dispatch_counts`).
- Tests: 12 (6 plan tests + `test_v4_orig_dispatches_via_block_b_extension` + `test_v4_orig_mix_runs_through_a4_live_path` + `test_does_not_break_a4_dispatch_counts` + `test_mana_draw_collapse_monitor_wires_b3_boost` + `test_does_not_edit_frozen_classic`). B8+A4 40 passed; full train_v3 197 passed/1 skipped.

---

## 3. Validation (per-component ultracode: implementer + 4 refute-by-default verifiers + fix)

Each component ran an ultracode workflow: single max-effort implementer → 4 refute-by-default verifiers (parallel) → fix stage. Verdicts:

| Component | Implementer | Verifiers (findings) | blockerCount | Binding gate |
|---|---|---|---|---|
| B1 | pool + 2 anchors + SelfPrevOpponent | 4 PASS (B1 frozen-share bug caught + fixed IN B3, see below) | 0 | anchor immutability + FIFO non-anchors + promotion-by-loss (no internal fields) |
| B2 | 3 identities + TempV4Opponent adapter | PASS-WITH-FINDINGS (monitor hook deferred, documented) | 0 | adapter reproduces OnnxActionPolicy math from packed ctx; V4-ONNX skip-gate |
| B3 | mix + parse/build/collapse + A4 uncomment + B1 fix | 4 PASS (0 blocker/major; B1 frozen-share 0.75→0.95 fix merged) | 0 | D-B5 hybrid sums 1.0; v4-orig-* NOT routed through frozen parse_v5_opponent_mix; punish_empty_board native parse |
| B4 | extract_lane_outcomes + CurriculumReweighter | 4 PASS (corrected plan's loose "from dispatch_log" wording) | 0 | outcomes from transitions.rewards sign (dispatch_log carries no win/loss); reweight cap 0.25; loop closed |
| B5 | parity loop + side-stratification | 4 PASS (dead field NOT wired, AST guard) | 0 | closes A4 sample_learner_sides open loop; reuses A3 oversampling scheme; gap exposes to B6 |
| B6 | Block-B gate extends A5 | 4 PASS | 0 | 4-component aggregate monotone ≥ N_snap=5; does NOT re-apply A-gate no_assist/exploit_resistance; promotion-by-loss guard inherited |
| B7 | plateau detector + exit→C2 | 4 PASS (D-B3 below-vs-above spec ambiguity resolved = below, flippable) | 0 | run-length plateau K_snap=10; exit = plateau AND below 0.55; carries best-ever path |
| B8 | LIVE league driver + additive A4 extension | PASS-WITH-FINDINGS (1 major test-dynamism assertion strengthened + 2 minors: MANA_DRAW_BAND_LOW import coupling, stray temp removed) | 0 | v4-orig-* dispatch gap closed additively (A4 counts unchanged); LIVE path not trace-pool; loop closed; A3 hyperparams unchanged |

**Source-vs-source discipline held throughout** (Block -1/0/A lesson): live engine/A4/A5 = oracle, V5/training code = UUT. B2 reproduces `OnnxActionPolicy` math from the real `ai/train_v2/onnx_policy.py:16` (source = oracle). B4 derives lane outcomes from `transitions.rewards` (A4 contract), NOT the plan's loose "dispatch_log" (which carries no win/loss). B6 confirms the dead `no_assist_min_score_rate`/`exploit_resistance_min_score_rate` fields are A-gate (Phase-A EXIT), NOT league promotion, before building NEW. B8 grounds the A4 dispatch mechanism (`resolve_opponent_dispatch` + `collect_rust_live_rollout:663` + `run_live_self_play_update:1023`) and the `PHASE_A_IDENTITIES`-hardcoded invariant (so the additive `BLOCK_B_POLICY_OPPONENT_KINDS` does NOT change the A4 identity-count tests).

---

## 4. Key architecture decisions (do not re-derive)

- **D-B3 = Below + progression** (USER). Exit→C2 fires when the candidate plateaus AND `h2h_vs_best < ~0.55` (still-weak → C2 human data); a plateau AT/ABOVE target → Block E1 ship path (`reason='dominant_plateau_e1_path'`), NOT C2. B7 flippable via `below_target_exits=False` without changing the default.
- **D-B5 = Hybrid** (USER). Frozen non-self = V4-orig 0.40/0.20/0.15 + exploit 0.05 + tail = 0.95 absolute (spec-literal design.md:118); self-snapshot = 0.05 residual grown as the pool fills (B1 prevalence); mana_draw-collapse monitor reweights on out-of-band vs V4-orig lanes (B3 `collapse_reweight_boost` wired by B8).
- **D-B8 = Adaptive cap 25%/update** (USER). B4 `CurriculumReweighter.reweight(mix, cap=0.25)` boosts only loss_rate>0.5 lanes, max 1.25×, renormalized.
- **D-B12 = Not yet** (USER). Build B1-B8 in-worktree with synthetic tests; the league RUN is gated on A-gate PASS (USER-run after A.pilot→A.BC→A.PPO→A-gate).
- **D-B7 = freeze all A hyperparams** (spec-literal). entropy=0.01/epochs=6/max_turns=120/learner_only_reward/decisive_early_end unchanged (B8 `test_continues_a_hyperparams`).
- **D-B1 N_snap=5, D-B2 K_snap≈10 (~2×N_snap), D-B4 ~2000/~6+2/FIFO, D-B6 t07=0.7/t12=1.2, D-B10 enable punish_empty_board code 5 (additive A4 uncomment).**
- **v4-orig-* = PYTHON policy opponents (B2 TempV4Opponent), NOT Rust rule codes.** worker.rs unchanged in Block B. Dispatch via the additive `BLOCK_B_POLICY_OPPONENT_KINDS` + `opponent_policies` (A4 :700 `opponent_policies[identity].select(i, ctx)`).
- **Promotion = external-bench ONLY** (inherited from A5, design.md:112/121): internal PPO loss/KL/entropy are MONITORING-ONLY; B1 `SnapshotEntry` + B6 `BlockBGateResult` carry NO internal fields = the promotion-by-loss guard.
- **`v5_snapshot` merges into `"self"`** (B8 `_merge_self_snapshot_split`): both self-play roles route through the same Python policy-opponent loop (`"self"` ∈ `POLICY_OPPONENT_KINDS`), keeping `BLOCK_B_POLICY_OPPONENT_KINDS` exactly the 3 v4-orig-* (plan-faithful).

---

## 5. Frozen-classic guard

Held throughout Block B. `classic_obs_v1` / `classic_actions_v1` / `classic_card_shape_v1` / `classic_rl_env.py` / `reward_v5.py` BYTE-LOCKED (never modified). `v5_trace.py` NOT imported into any Block-B code path. `core/state.py` NOT modified. `league_v5.py` / `gauntlet_v5.py` / `opponents_v5.py` consumed READ-ONLY (B3 aliases live in B3, NOT `V5_OPPONENT_KINDS`; B5 dead `p1_p2_max_score_gap:43` NOT wired). `warm_start_v5.py` consumed READ-ONLY. A5 `a_gate.py` + A3 `ppo_phaseA_config.py` READ-ONLY (B6/B5/B8 reuse `check_mana_draw_band`/`play_gauntlet`/`second_start_oversampling_scheme`/`MANA_DRAW_BAND_LOW` without edit). `rust_trainer.py` READ-ONLY (`_save_checkpoint:802` + `train_rust_ppo_trace_files:92` reused as STRUCTURAL template only).
- **The only non-new-file modifications in Block B:** (1) B3's A4 uncomment `rust_live_self_play.py:143` `"punish_empty_board": 5` (D-B10, zero Rust change); (2) B8's minimal ADDITIVE A4 extension (`BLOCK_B_POLICY_OPPONENT_KINDS` :169-171 + `resolve_opponent_dispatch` :212-213 + `opponent_mix_parsed` :1015 — purely additive, NO removed line touches `POLICY_OPPONENT_KINDS`/`PHASE_A_IDENTITIES`/`RULE_AGENT_CODES`, confirmed via `git diff` grep + `test_does_not_break_a4_dispatch_counts`); (3) test-only reconciliation of two B4/B5 frozen-guard tests (`test_curriculum::test_does_not_edit_a4_sampler` + `test_second_start_parity::test_does_not_edit_a5_a3_a4`) now permit ONLY the B8 additive A4 extension — a STRONGER guard than empty-diff (assert the diff contains `BLOCK_B_POLICY_OPPONENT_KINDS` + `opponent_mix_parsed` and no removed line touches frozen constants). A5/A3 empty-diff guards intact. NO Rust edit. No TrainV3.5 import into prod paths.

---

## 6. Latent finding (uncovered by A1, re-flagged for Block C)

The loader's `OfflineTransition.action_features` is built with `placement_mode='full'` → OVER-INCLUDES warrior PlayCard candidates at non-append positions the engine does NOT offer (`core/engine.py:1260` emits warriors ONLY at `position=len(board)`). A1 works around it by REBUILDING with `placement_mode='append_only'`. **Follow-up for Block C (AWAC/CRR consumers):** fix the loader to use `append_only` so `OfflineTransition.action_features` + its legal mask are engine-faithful. Does NOT affect Block B (B8 drives the LIVE path, not the offline loader).

---

## 7. Handoff — operational steps (USER runs, per D-B12=not-yet)

The 8 in-worktree components are built + synthetic-tested. The league RUN is NOT autonomously doable; the user runs it after A-gate PASS:

1. **Prerequisite (Block A operational steps, USER):** A.pilot deploy+collect → A.BC run → A.PPO training run → A-gate measurement → candidate PASSES A-gate → promote to best-self-snapshot.
2. **Block B league RUN** — B8 `BlockBLeagueDriver.run(n_updates)` (compute ~Medium 20-40k updates, D-B11) on the A-gate-passing candidate: per-update B3 mix + B4 curriculum + B5 parity + A4 live PPO; every ~2000 updates B1 snapshot → B6 promotion → B7 plateau/exit check. Snapshots fill the pool; self-snapshot prevalence grows (D-B5 hybrid); mana_draw-collapse monitor reweights on out-of-band.
3. **Exit→C2** — when B7 `exit_fires` (plateau AND `h2h_vs_best<0.55`), the driver emits the exit signal carrying the B1 best-ever checkpoint path → Block C (offline RL: AWAC/CRR on human data + the plateaued policy).

A-gate PASS is the gate to the league RUN (D-B12). Block B code is built regardless.

---

## 8. Session artifacts (workflow scripts)

Block B orchestration scripts (B1-B8 ultracode workflows: implementer + 4 refute-by-default verifiers + fix) persisted under the session workflows directory (ASCII-only `.claude_tmp_b{N}.js`, `node --check`, invoked via `scriptPath`, deleted before each commit — DURABLE stalling-plan-writer fix from the Block B plan phase). Plan: `BLOCK_B_PLAN.md`. This completion log: `BLOCK_B_COMPLETION.md`.

See project-memory `extra-lr-v5-blockb-plan.md` (Block B plan + per-component execution status + decisions), `extra-lr-v5-blocka-plan.md` (Block A foundation), `extra-lr-v5-block0-plan.md` + `extra-lr-v5-blockminus1-port-progress.md` (pipeline state), `extra-lr-v5-pipeline.md` (V5-Max design + decision ledger).