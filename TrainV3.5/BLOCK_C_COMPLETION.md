# Block C — V5 RLHF loop (C2→C3, AWAC/CRR offline-PPO replay, K=2 early-stop) — IN-WORKTREE COMPLETION LOG

**Branch:** `glm-5.2/TrainV3.5Prep` (worktree `.claude/worktrees/glm-TrainV3.5Prep`)
**Scope:** the V5-Max Block C in-worktree components — loader append-only fix (C0) + rlhf deploy/collection adapter (C1) + fresh-human→dense-Transition offline bridge (C2) + Hybrid AWAC×PPO-clip offline-PPO replay (C3) + the C-loop driver (C4: collect→replay→measure→B6 promote→B1 pool→D-C6 K=2 stall→exit→D). 5 components, dependency-ordered, synthetic-testable; the operational rlhf_env RUN is USER-run per D-C0/D-C9 — the in-worktree code is built + synthetic-tested regardless.
**Status:** ✅ **ALL 5 IN-WORKTREE COMPONENTS COMPLETE** (2026-07-02). Each done + independently re-verified green via the ultracode workflow (implementer + 4 refute-by-default verifiers + fix) PASS with 0 blocker/major remaining on each.
**Pipeline position:** `Block -1 ✅ → 0 ✅ → A ✅ → B ✅ → C ✅ (in-worktree, this) → [C-loop RUN: C2 real-human collect → C3 replay → K=2 stall → exit→D] → D → E1`.

---

## 1. Commits (chronological)

| Commit | Component | Summary |
|---|---|---|
| `113d722c` | plan | `BLOCK_C_PLAN.md` — RLHF loop C2→C3 AWAC/CRR replay (solo-draft + verify-only, PASS-WITH-FINDINGS: 1 blocker + 5 major + 8 minor merged) |
| `0c2bad36` | C0 | `loader_append_only_fix` — engine-faithful `action_features` (D-C5); the append-only latent-loader branch produces 601-tcode action_features matching the dense evaluator's contract |
| `4bbd1bb6` | C1 | `v5_rlhf_adapter` + `c2_collection_driver` (D-C0=build in worktree rlhf_env now / D-C4=5k mana_draw ACTIONS or 5k battles / D-C8=B1 best-ever anchor argmax); D11 omniscient deploy + fresh-human collection with `decision_source='human'` |
| `bdd856a5` | C2 | `offline_replay_bridge.py` — fresh-human `v5_trace` rows → dense `RustTransitionBatch` (601-tcode via READ-ONLY `resolve_v5_tcode`, D11 omniscient, D-C7 human rows only, D-C10 old_log_prob=current policy at bridge time, GAE per-game episode boundaries) |
| `0321b022` | C3 | `awac_crr_replay.py` — Hybrid AWAC×PPO-clip offline-PPO replay (core ML: pure-numpy loss + MLX evaluator/trainer; BLOCKER loss-math sign-fix; A2 freeze_faithful + value_head trainable D-C3; warm-start via `load_v4_max_into_v5` Q3 PARTIAL; skip-gate on mlx/npz absence) |
| `51657dfd` | C4 | `c_loop_driver.py` — the C-loop (C2 collect → C3 replay → A5/B5 measure → B6 promote → B1 pool → D-C6 K=2 stall → exit→D); `CLoopManifest` mirrors `BlockBLeagueManifest`; injectable Protocol collaborators; MLX/Rust NOT imported at module top |

Combined suite (final, independently re-confirmed after C4): **python 236 passed / 1 skipped** (the MLX/Rust skip-gate `test_skip_if_no_mlx_or_rust` — the pre-existing V4-ONNX worktree skip-gate), 0 failed. Cargo 157 unchanged (no Rust edit in Block C).

---

## 2. Components (what was built)

### C0. loader_append_only_fix — engine-faithful action_features (D-C5) — `0c2bad36`
The append-only latent-loader branch must produce 601-tcode `action_features` matching the dense evaluator's contract so C2/C3 can consume append-only-loaded rows. Enabling fix; no new public surface beyond the corrected feature tensor. Tests green.

### C1. `v5_rlhf_adapter` + `c2_collection_driver` — `4bbd1bb6`
- `V5RlhfAdapter` + `_factory_v5_real` 2-arg factory (`(spec, registry)`, the AdapterRegistry contract) registered into rlhf_env; **D11 omniscient** `InfoModeV5(enemy_hand_known=True, enemy_deck_known=True, enemy_deck_order_known=True)` enforced (default is self-visible, `contracts.py:46-47`); history parity (`state.history`, NOT `state.action_history`).
- `c2_collection_driver`: deploys the current best V5 vs humans in rlhf_env, collects fresh preV5-vs-human battles, tags `decision_source='human'`, stops on **D-C4 = 5k mana_draw ACTIONS (rows with `action_type=='mana_draw' AND decision_source=='human'`) OR 5k battles whichever first**. Built in the worktree's tracked `rlhf_env/components/` per D-C0 (rlhf_env/components/*.py ARE tracked in this worktree).
- D-C8 = B1 best-ever anchor argmax (the deployed checkpoint is the argmax of `pool.best_ever`).

### C2. `offline_replay_bridge.py` — fresh-human v5_trace → dense RustTransitionBatch — `bdd856a5` (716 lines)
`build_offline_replay_batch(policy_fn, *, group_dirs=None, transitions=None, info_mode=None, assist_mode=None, max_battles=None, strict=False) -> OfflineReplayBatch`.
- **D11** `_omniscient_info_mode()` = `InfoModeV5(enemy_hand_known=True, enemy_deck_known=True, enemy_deck_order_known=True)`.
- **D-C7** filters `decision_source=='human'` rows only.
- **601-tcode** via READ-ONLY `resolve_v5_tcode(pre_state, actor, t.action_native, mask=append_only_mask, strict=False)` (import from A1 `bc_dataset.py:172`); `action_native` from the loader's ENGINE-sourced field (`offline_dataset_loader.py:796`), NOT `decode_action`.
- **D-C10** `old_log_prob = current policy at bridge time` (NOT human behavior policy): `policy_fn(obs, action_features) -> (logits, values, mana_draw_logit)` called ONCE; `old_log_prob = log(softmax(where(mask==1, logits, -1e9))[target_tcode] + 1e-10)` (mirrors `rust_ppo.py:760-765`); `value = values[i]` for EVERY obs; mana_draw/terminal `old_log_prob=0` but value populated.
- **GAE episode boundaries (load-bearing):** batch organized as `(steps, env_count=num_games)` — each battle one env, padded to max human-actions-per-game — NOT flattened `(total_rows, env_count=1)` (would leak value across game boundaries). `terminated=True` on each game's last real human action + padded steps; `bootstrap_values = V(next_obs)` of each game's final real transition, shape `(num_games,)`.
- Emits `RustTransitionBatch` (dense `action_mask (s,e,601)` + `action_features (s,e,601,171)`) + flat legal-action tape step-major C order; padded rows carry a DECOUPLED `[0]` dummy legal id (the dense evaluator ignores `legal_action_ids`, reads `action_mask=zeros` → zero policy-loss). Parallel arrays: `is_mana_draw`, `mana_draw_legal`, `target_tcodes` (int32, -1 for mana_draw/terminal/padded).
- Optional `make_policy_fn_from_checkpoint` (A2 skip-gate on file existence, lazy MLX).
- Tests: 11 SYNTHETIC tests green. **Tautology fix applied** at `test_per_game_terminated_and_bootstrap` (was `assert bool(x) is False or bool(x) is True` — always-True; replaced with `assert bool(out.batch.terminated[step, env]) is False` for intermediate real steps — a regression to `terminated=True` on all real steps would now be caught).

### C3. `awac_crr_replay.py` — Hybrid AWAC×PPO-clip offline-PPO replay — `0321b022` (935 lines)
3 layers.
- **(A1) pure-numpy `awac_crr_loss(model_outputs, batch_fields, *, clip_epsilon, value_coef, entropy_coef, lambda_awac, awac_clamp, mana_draw_bce_weight) -> (total_loss, metrics)` + `awac_weight(A, *, lambda_awac, awac_clamp) = exp(clamp(A/lambda, -C, C))`** — PRIMARY no-mlx regression guard (mlx lazy-imported inside the evaluator/trainer only). **BLOCKER loss-math EXACT (the C3 blocker fix — the incoherent draft `(1+log π)·w·log π` DECREASED log π for high-advantage via a sign flip when log π<-1; ELIMINATED):**
  ```
  masked = where(mask==1, logits, -1e9); probs = softmax(masked)
  new_log_probs = log(gather(probs, actions) + 1e-10)
  ratios = exp(new_log_probs - old_log_probs)
  A = advantages                  # surrogate multiplier (NOT log π)
  w = exp(clamp(A / lambda_awac, -awac_clamp, awac_clamp))   # clamp BEFORE exp
  surr1 = ratios * A; surr2 = clip(ratios, 1-eps, 1+eps) * A
  valid_policy_mask = (target_tcodes >= 0) & (~is_padded)
  policy_loss = -sum(w * min(surr1, surr2) * valid_policy_mask) / max(sum, 1)
  valid_value_mask = ~is_padded
  value_loss = value_coef * sum((returns - values)^2 * valid_value_mask) / max(sum, 1)
  mana_draw BCE (retain mana_draw_logit; A2 pattern bc_train.py:332-338) over mana_draw_legal
  entropy over valid rows
  total = policy_loss + value_loss - entropy_coef*entropy + mana_draw_bce_weight*mana_draw_bce
  ```
  Sign correctness: positive A + ratio>1 → `policy_loss` DECREASES → gradient INCREASES log π. (Folk "AWR × PPO-clip", not a named algo — per `mlx-onnx-v5-brief`.)
- **(A2) MLX `evaluate_awac_dense_batch(...)`** mirrors `evaluate_dense_rust_ppo_batch:734-797` BUT **RETAINS `mana_draw_logit=_out[2]`** (the dense template `:759` drops it); model forward passes `mana_draw_legal=None` (raw head logit, `bc_train.py:306-308`).
- **(A3) `train_awac_crr_replay(checkpoint_path_or_model, offline_replay_batch, *, epochs, minibatch_size, clip_epsilon, value_coef, entropy_coef, lambda_awac, awac_clamp, mana_draw_bce_weight, max_grad_norm, lr, freeze_faithful=True, train_value_head=True, seed, hidden_dim, save_checkpoint_path, padded_mask)`** mirrors `_train_rust_ppo_minibatch_with_evaluator:193`. **C3-SPECIFIC loop slices BOTH the `RustPPOBatch` minibatch (via `_take_flat_rows`) AND the C2 parallel arrays (`is_mana_draw`/`mana_draw_legal`/`target_tcodes`/`is_padded`) in lockstep — `rust_ppo.py` NOT modified.** First calls `prepare_rust_ppo_batch(gamma=0.99, gae_lambda=0.95, bootstrap_values)` for GAE (D-C2 GAE value bootstrap). A2 `freeze_faithful` + `_zero_frozen_grads` (byte-identical frozen params) + `mlx.optimizers.Adam.update` + `_clip_grads`; value_head TRAINABLE (D-C3, `value_coef=0.5`). Warm-start via `load_v4_max_into_v5` (Q3 PARTIAL). **SKIP-GATE: returns `{status:'skipped'}` NO crash when mlx/npz absent; a model instance bypasses the npz gate.** `is_padded` derived from `action_mask.sum(-1)==0` OR `padded_mask` kwarg.
- **(A4) `AwacCrrReplay.run(...) -> AwacCrrMetrics`** — MONITORING-ONLY (NO promote/score field); `clip_fraction`/`approx_kl` over valid_policy rows.
- Tests: 15 tests green. `test_blocker_sign_fix` FALSIFIES the incoherent draft (`policy_loss(ratio=2,A>0) < policy_loss(ratio=1,A>0)` AND a log_pi-multiplier form does NOT satisfy the direction).

### C4. `c_loop_driver.py` — the C-loop — `51657dfd` (661 lines)
- `CLoopManifest` mirrors `BlockBLeagueManifest`: `iteration_metrics`, `snapshot_history`, `promotion_decisions`, `aggregate_history`, `best_ever_path`, `exited_to_D`, `exit_verdict`, `n_iterations_run`, `stall_count` + `to_dict`.
- `CLoopDriver`: injectable Protocol collaborators (`CollectionDriverProtocol`, `ReplayProtocol`, `GameRunnerProtocol`, `CheckpointNamerProtocol`) + concrete `SnapshotPool`. MLX/Rust NOT imported at module top.
- Per-iteration: (a) **C2 collect** → `CollectionOutcome` (`group_dirs` → `build_offline_replay_batch` lazy import); (b) **C3 replay** `.run` → `AwacCrrMetrics`-shaped (`new_checkpoint_path=candidate`); (c) **measure** → h2h/gauntlet (A5 `play_gauntlet` via `measure_gauntlet_rate`, NOT custom) / mana_draw / p1_p2_gap (B5); (d) **B6 promote** `evaluate_block_b_gate` (NOT A5 a_gate); `self._aggregate_history = list(gate.monotone_aggregate_history)` NO double-append; (e) **B1** `SnapshotEntry` + `set_seed_anchor` on FIRST else `add_snapshot` + `maybe_update_best_ever` (strict H2H, D-C8); (f) **D-C6 K=2 stall-counter** on `update_number>=2 AND len>=2`: GAIN = `current > prior + monotone_tolerance` → `stall=0` else `stall++`; NOT on iteration 1 NOR on skips; DECOUPLED from B6 promote; **exit→D at `stall >= k_stall (K=2)`** sets `exited_to_D` + `exit_verdict` + `best_ever_path = pool.best_ever.path` + break.
- `aggregate_history` FRESH-SEEDED at `run()` entry (`[]`). Does NOT import/call B7 `detect_h2h_plateau`/`exit_to_c2` (B7 = B→C2 handoff, NOT C→D exit — do not conflate). Skip-gates: C2/C3 skip → record skipped, loop CONTINUES, no stall increment. `A5MeasurementRunner` real adapter (USER-run wiring, lazy imports).
- Tests: 13 SYNTHETIC tests green (`FakeCollectionDriver`, `FakeAwacReplay`, `FakeGameRunner`, real `SnapshotPool`, `_FakeNamer`).

---

## 3. Cross-cutting constraints honored

- **Prod runtime must not import TrainV3.5** — all C components live under `TrainV3.5/python/train_v3/`; the only prod-touching artifact is `rlhf_env/components/v5_rlhf_adapter.py` (C1), which is the rlhf_env adapter surface (D-C0), NOT a prod import of TrainV3.5.
- **Frozen-classic byte-frozen** — `classic_obs_v1/classic_actions_v1/classic_card_shape_v1/classic_rl_env.py/reward_v5.py` NOT touched; C2 imports `resolve_v5_tcode` READ-ONLY from A1.
- **`v5_trace.py` NOT modified + NOT imported into the loader/BC/training code path.**
- **`core/state.py` NOT modified**; `league_v5.py`/`gauntlet_v5.py`/`opponents_v5.py` consumed READ-ONLY.
- **`rust_ffi.py` NOT touched** (no Rust edit in Block C → cargo 157 unchanged, no re-confirmation needed).
- **`rust_ppo.py` NOT modified** — C3 slices the `RustPPOBatch` minibatch + C2 parallel arrays in lockstep rather than editing the dense evaluator.
- **DURABLE workflow-script rules** — all ultracode scripts ASCII-only (`.claude_tmp_c*.js`), NO function declarations (trip the Workflow parser), invoked via `scriptPath`, deleted before each commit.

---

## 4. Spec-level minors flagged for downstream (NOT fixed; documented)

1. **C2 bootstrap_values accepted-but-unused under `terminated=True` on last real step** — C3 may set `truncated=True` for non-terminal last human actions so GAE bootstraps `V(next_obs)`; the bridge provides `bootstrap_values` with correct shape/value regardless.
2. **C3 warm-start uses `load_v4_max_into_v5` (V4-Max npz) per spec (Q3 PARTIAL)** — a V5 BC-seed checkpoint uses different key names → path-based warm-start of a V5 checkpoint transfers only shape-compatible params; C4/user should confirm whether to warm-start from a BC-seed V5 via `model_mlx.load_checkpoint` instead.
3. **C3 `is_padded` via `action_mask.sum(-1)==0`** (overridable via `padded_mask` kwarg).
4. **C3 `approx_kl`/`clip_fraction` over valid_policy rows** (consistent masked-AWAC deviation).
5. **C4 `SnapshotPool` typed concrete** (mirrors B8, duck-typeable via the Protocol collaborators).

---

## 5. Operational steps (USER-run, per D-C0 + D-C9)

1. Register the V5 adapter factory into rlhf_env (`rlhf_env/components/policy_adapters.py`) — C1 built the adapter; the registration wiring is USER-run.
2. **C2 real-human collection** — rlhf web @ 8090, deploy best post-B V5 vs humans, collect fresh battles until D-C4 floor (5k mana_draw ACTIONS or 5k battles).
3. **C3 replay run** — `train_awac_crr_replay` on the C2 bridge batch; Medium compute (~20–40k replay updates/iteration, D-C9).
4. **C-loop until K=2 stall** → exit→D carrying `pool.best_ever.path`.

The in-worktree code is built + synthetic-tested regardless of the operational RUN.

---

## 6. Transition to Block D

Block C exits to **Block D — League-2 (consolidation)** (`design.md:130-131`): short self-play among {best post-C checkpoint, post-B/seed anchors, V4-orig spectrum, exploit-lanes} to settle post-C and prevent overfit to the last replay batch; promotion by external bench. Open Block-B question #10 (pool coexistence with Block D league-2, `design.md:131`) is resolved at Block-D planning time: whether Block D reuses the SAME snapshot pool or a fresh one.

Block D plan next — drafted via ultracode (solo-draft + verify-only, the durable plan-writer pattern).