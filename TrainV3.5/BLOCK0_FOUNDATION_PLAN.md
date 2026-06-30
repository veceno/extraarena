# Block 0 — V5 Foundation — Implementation Plan

**Produced:** 2026-06-30 by `block0-plan` ultracode workflow (CloseQ → Plan → 4-way adversarial Verify → Finalize). PASS-WITH-FINDINGS; all 5 blocker/major findings merged into the final plan.
**Pipeline position:** `Block -1 ✅ → 0 (this) → A → B → C → D → E1`.
**Gating decision (USER, resolved 2026-06-30):** CARD_SHAPE_DIM_V5 = **73 (Grow, disjoint)** — append 5 new mechanic one-hots at [64..69) + 4 magnitude scalars at [69..73); fix the index-47 overlap; cascade derived dims through `contracts.py` + Rust mirror; warm-start base_encoder.layers.0 (1456→256) unaffected.

---

## Open questions closed (Q1 was closed by Block -1)

- **Q2 ✅ closed** — `HAND_CAP = 4` (`core/engine.py:44`, single source of truth; all draw paths funnel through `draw_one_from_deck` with the cap guard; `mana_draw` subject to the same cap at `engine.py:781,1344`). **No hand lift** — `_NUM_HAND=4` already correct. Spec §0.88 "hand lifted" is misleading → corrected to "keep _NUM_HAND=4".
- **Q3 ✅ closed PARTIAL** — V5 split-encoder cannot reproduce V4-Max 1456→256→601 (fresh `state_fuser.layers.0 (544→256)` + extra SiLU between base projection and 2nd linear; SiLU nonlinear → no identity init). Warm-start = **partial name+shape remap**:
  - **Faithful** (reproduce V4 exactly): `base_encoder.layers.0` (256,1456) ← V4 `state_encoder.layers.0`; `action_encoder` (128,171) ← V4 `action_encoder`.
  - **Shape-compatible, semantically disconnected** (loads, but logits won't match V4): `state_fuser.layers.2` (256,256) ← V4 `state_encoder.layers.2`; `candidate_scorer` (1,384); `value_head` (1,256).
  - **Fresh init** (no V4 counterpart): `global_encoder.layers.0` (32,32), `private_encoder.layers.0` (2112→2400 w/ dim-73,128), `history_encoder.layers.0` (2880→3240,128), `state_fuser.layers.0` (544,256).
  - V4-Max npz located in **main repo only** (`ai/train_v2/runs/m4_balanced_from_0950_20260522_144431/checkpoints/update_1190.npz`, 5.5MB, **gitignored — absent from worktrees**); ONNX at `ai/models/extra-lr-v4-max.onnx` (1.85MB, present). MLX **not importable in this worktree** → warm-start tests run in the TrainV3.5 training env with npz provisioned, else skip-gate.
- **Q4 ⏳ open_deferred** — baseline `B = mana_draw_count / eligible_turns` unmeasurable until Phase A pilot (no battles yet). Plan embeds the exact measurement spec + A-gate `[0.5×,1.5×]` + E1 band as placeholders. Hard dependency: `B` invalidated if `HAND_CAP` or `MANA_DRAW_BASE` change.
- **Q5 ✅ decision** — league composition: self_snapshot 0.55 / v4_orig_argmax 0.15 / v4_orig_t07 0.07 / v4_orig_t12 0.03 / exploit(stall, anti_draw_greed, punish_empty_board) 0.05 each / tail(greedy_face 0.03, legal_random 0.01, end_turn 0.01). mana_draw-collapse monitor: `md_rate(V4-lane)/md_rate(self-snap-lane) ≥ 0.5`; breach ladder (2 windows → reweight self 0.65/V4 0.15 + C2 top-u trigger).

---

## Components (dependency-ordered)

### 1. `v5_card_shape_v1.py` (NEW) + Rust mirror `card_shape_v5.rs` + `contracts.py` derivation
- **file:** `ai/train_v2/v5_card_shape_v1.py` (new fork); Rust mirror `TrainV3.5/rust/trainv3_core/src/card_shape_v5.rs` (currently delegates to classic at `:115-116`, `CARD_SHAPE_DIM_V5=64` at `:103` — must update to 73); `TrainV3.5/python/train_v3/contracts.py` dim derivation.
- **purpose:** V5 card-shape forked from frozen `classic_card_shape_v1` with the index-47 overlap FIXED via disjoint regions + 5 new mechanic one-hots + 4 magnitude scalars. `CARD_SHAPE_DIM_V5 = 73`.
- **layout (decision grow→73):** `[0..47)` classic 64 base; `[47..64)` classic 17 scalars (kept); `[64..69)` 5 NEW one-hots (aoe_silence, team_wide_shield, rebirth, crime_and_punishment, target_ally_max_hp_plus); `[69..73)` 4 NEW magnitude scalars. Overlap fixed (disjoint).
- **cascade:** `PRIVATE_INFO_DIM = (OWN_HAND_SLOTS+OWN_DECK_SLOTS+ENEMY_HAND_SLOTS+ENEMY_DECK_SLOTS)*(1+1+CARD_SHAPE_DIM_V5)` = `32*75 = 2400`; `HISTORY_EVENT_DIM = SOURCE_CARD_OFFSET + CARD_SHAPE_DIM_V5*2 = 16+146 = 162` → `HISTORY_DIM = 20*162 = 3240`; `OBS_V5_DIM = 1456+32+2400+3240 = 7128`. Derive all from `CARD_SHAPE_DIM_V5` (named offsets, not literals).
- **depends_on:** none.
- **tests:** `test_overlap_fixed_no_index_collision` (card w/ desk_freeze+damage_5 encodes BOTH at disjoint indices — negates the Rust `card_shape_v5.rs:283` overlap-inherited assertion); `test_five_new_mechanic_one_hots`; `test_magnitude_scalars`; Rust↔Python parity for the 73-dim vector across the 50-card catalog.
- **acceptance:** PREREQUISITE GATE — layout decision resolved (grow→73). 73-dim vector byte-matches Python↔Rust across all 50 cards; overlap test green; cascade dims consistent in `contracts.py` + `v5_policy.py` + `obs_v5.py` + Rust `v5.rs`.
- **frozen_guard:** `classic_card_shape_v1.py` never modified (separate file). `core/state.py:MECHANICS_LIST` stays 34 (extending to 39 would overflow the frozen classic one-hot region — the 5 new families are V5-fork-only flags).

### 2. `obs_v5.py` (EXISTS, fully implemented — EXTEND in place)
- **file:** `TrainV3.5/python/train_v3/obs_v5.py` (`encode_observation_v5` at `:31`, calls frozen `encode_observation` at `:42`, `_encode_globals_v5` `:55-75`, `_encode_private_info`/`_encode_zone` `:78-95`, `_encode_history`/`_encode_one_event` `:98-128`); Rust mirror `v5.rs`.
- **purpose:** base 1456 (frozen, warm-startable) ⊕ global 32 ⊕ private ⊕ history 20×144. Use `v5_card_shape_v1` for per-card slots.
- **depends_on:** `v5_card_shape_v1`.
- **tests:** `test_history_window_20_reads_action_history`; `test_hand_shape_invariant_0_to_4` (Q2 — shape invariant for hands 0..4, 5th-card→graveyard parity `engine.py:187-196`); `test_own_deck_encoded_per_card` (V5 private block encodes `me.deck` per-card via `PRIVATE_CARD_SLOT_DIM` slots, NOT the classic zone-summary — spec §6.184); `test_mana_draw_count_this_turn_global_channel` at `dst[15]` from `me.mana_draw_count_this_turn`; `test_omniscient_both_hands_decks_encoded` (deck = 12 slots per `contracts.py:15 OWN_DECK_SLOTS=12`, not 9).
- **acceptance:** `OBS_V5_DIM` stable at 7128 OR updated consistently with component-1 dim decision across `contracts.py` + Rust `v5.rs` + `v5_policy.py`; mana_draw_count global channel present + populated; per-card own-deck encoded (not zone-summary).
- **frozen_guard:** `classic_obs_v1.py` NOT modified — base-1456 path calls frozen `encode_observation` at `obs_v5.py:42`. `_NUM_HAND=4` (Q2).

### 3. `mana_draw_head_v5.py` (NEW) + `v5_policy.py` extend
- **file:** `TrainV3.5/python/train_v3/mana_draw_head_v5.py` (new mask+selection); `v5_policy.py:29` `V5ActionConditionedPolicy` extended (`:65-78` currently has NO mana_draw_head — add `nn.Linear(hidden_dim,1)` alongside `value_head` at `:78`).
- **purpose:** parallel binary mana_draw head (decision γ / spec §0.89 — NOT a 602nd candidate; 601 codec frozen). `Linear(hidden_dim,1)` on the fused state_emb.
- **depends_on:** `encode_observation_v5`.
- **tests:** `test_mana_draw_legal_mask_hand_full_unset` (hand≥4 → mask 0.0, parity `engine.py:781-782`); `test_mana_draw_legal_mask_insufficient_mana_unset` (mana < `2*(count+1)` → 0.0, `engine.py:785-786`); `test_mana_draw_legal_mask_...`; mask byte-parity with `golden_trace.py:523` predicate.
- **acceptance:** head outputs finite logit for any state_emb; mask matches `engine.py:1343-1347` byte-parity; selection deterministically picks mana_draw iff legal-and-preferred; `V5ActionConditionedPolicy.__call__` returns (candidate_logits, value, mana_draw_logit).
- **frozen_guard:** `classic_actions_v1.py` NOT modified (`MAX_CANDIDATE_ACTIONS=601` frozen `:46`; `ManaDrawAction` is `core/actions.py:76`, outside the 601 space).

### 4. `warm_start_v5.py` (NEW)
- **file:** `TrainV3.5/python/train_v3/warm_start_v5.py` (new; Q3 confirmed no existing warm_start/load_v4/transfer).
- **purpose:** load V4-Max `update_1190.npz` into `V5ActionConditionedPolicy` per Q3 PARTIAL (name+shape remap, `strict=False`).
- **depends_on:** `mana_draw head`.
- **tests:** `test_v4_max_npz_present_or_skip` (**blocker-fix prerequisite** — if `V4_MAX_NPZ_PATH` env unset AND default main-repo path absent, SKIP with reason 'gitignored, not provisioned'); `test_faithful_layer_equality` (EXACT match `base_encoder.layers.0` + `action_encoder`); non-parity assertion (documents full logits differ — prevents false confidence); `strict=False` load with zero silent key drops; npz key/shapes dumped.
- **acceptance:** faithful-layer equality green; non-parity assertion green; `strict=False` succeeds; spec §6.188 full-forward-pass-parity RELAXED per Q3 (binding gate = faithful-layer equality, not full-logit match).
- **frozen_guard:** V4-Max npz + ONNX read-only (loader never writes). `classic_*` untouched. Warm-start = "head start" not parity (Q3).
- **env note:** MLX not importable in this worktree + npz gitignored/absent → component-4 tests run in the TrainV3.5 training env with npz provisioned; skip-gate here.

### 5. `offline_dataset_loader.py` (NEW)
- **file:** `ai/train_v2/offline_dataset_loader.py` (new; handoff §6:120 + confirmed absent). Trace schema at **`rlhf_env/components/v5_trace.py`** (NOT under `ai/train_v2/` — the bridge READS emitted JSON `actions.jsonl`/`meta.json`/`turns.jsonl`, does NOT import `v5_trace`).
- **purpose:** D1/D9 offline-bridge: iterate `sessions/<gid>/catalog.json` + `battles/<bid>/v5/{meta,actions,turns}.jsonl`; skip orphans (`meta.status != 'finished'`); reconstruct `GameState` from `pre_state`; re-encode via `encode_observation_v5`; produce offline-PPO replay (AWAC/CRR) tuples.
- **depends_on:** `encode_observation_v5`, `mana_draw head`.
- **tests:** `test_bridge_round_trip` (record a battle via `v5_trace`, reconstruct from `pre_state`, re-encode, assert reconstructed post-state obs == next row's `pre_state` obs within tolerance — canonical integrity gate spec §6.185); orphans skipped; surrender synthetic row → terminal transition; offline reward matches online `ClassicRLEnv` reward.
- **acceptance:** round-trip obs == next-row pre_state obs within tolerance; orphans skipped; surrender row terminal; reward byte-matches `classic_rl_env.py:383-423` (`_compute_reward`: -0.05 invalid; ±1.0 terminal win/loss, 0.0 draw; +0.02*enemy_hp_delta if>0; -0.01*own_hp_delta if>0; +0.03*enemy_killed; -0.02*own_killed; +min(0.02, 0.005*mana_spent)).
- **frozen_guard:** `classic_*` untouched; `classic_rl_env.py` NOT modified (bridge MIRRORS `_compute_reward` formula, does not edit it). `v5_trace.py` is a data-contract dependency (read JSON), not a code import.

---

## Execution order
`v5_card_shape_v1` → `encode_observation_v5` → `mana_draw head` → `warm-start loader` → `offline-bridge`.

## Open risks (residual, tracked)
1. npz provenance — gitignored/absent in worktree; skip-gate + `V4_MAX_NPZ_PATH` env (component 4).
2. v5_trace cross-package coupling — bridge reads JSON from `rlhf_env/components/v5_trace.py` (component 5).
3. MLX not importable in this worktree — warm-start + ONNX↔MLX export tests (spec §6.189) run in TrainV3.5 training env (component 4).
4. Frozen-classic 5-slot vs 7-slot obs divergence (documented `card_shape_v5.rs:38-53`) — 5 old golden fixtures stale; inherited, not a Block 0 blocker.
5. `HISTORY_EVENT_DIM` must be DERIVED from `CARD_SHAPE_DIM_V5` (named-offset layout), not a literal 144/162 — the dim-73 cascade depends on this.

## Q resolutions
Q2 → no hand lift (`_NUM_HAND=4`). Q3 → partial warm-start (faithful base_encoder+action_encoder only; spec §6.188 full-parity relaxed). Q4 → deferred to Phase A (placeholder + measurement spec). Q5 → league composition + collapse-monitor decided.

See `docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-{design,handoff}.md` for the source spec + decision ledger, and `BLOCK_MINUS1_COMPLETION.md` for the Block -1 closeout.