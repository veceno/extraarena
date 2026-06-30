# Block 0 — V5 Foundation — COMPLETION LOG

**Branch:** `glm-5.2/TrainV3.5Prep` (worktree `.claude/worktrees/glm-TrainV3.5Prep`)
**Scope:** the V5-Max foundation layer — fork the card-shape codec (disjoint 73-dim), thread the mana_draw-count obs channel, add the parallel mana_draw head, warm-start from V4-Max (partial), and the offline-bridge from recorded v5 traces to AWAC/CRR tuples.
**Status:** ✅ **COMPLETE** (2026-07-01). All 5 components done + independently re-verified green (source-vs-source); 4-way adversarial verify on each component PASS with 0 blocker/major.
**Pipeline position:** `Block -1 ✅ → 0 ✅ (this) → A → B → C → D → E1`.

---

## 1. Commits (chronological)

| Commit | Component | Summary |
|---|---|---|
| `c6020806` | 1 | `v5_card_shape_v1.py` (73-dim disjoint fork) + `contracts.py` cascade (OBS_V5_DIM=7128) + Rust `card_shape_v5.rs` mirror + `kernel.rs` V5-obs wiring + 24 golden_trace fixtures regen (Python oracle, V1 base-1456 hash unchanged) |
| `b17e1440` | 2 | `encode_observation_v5` mana_draw_count_this_turn global channel at `global[15]` (MANA_DRAW_COUNT_NORMALIZER=5.0) + Rust `kernel.rs` mirror (no fixture regen — count=0 in all compared fixtures) |
| `d0137faa` | 3 | `mana_draw_head_v5.py` (pure-python mask+selection, byte-parity `engine.py` get_legal_actions + `golden_trace.py:522-523`) + `v5_policy.py` 3-tuple `__call__` (parallel binary head, NOT 602nd candidate) + 3 callers arity-robust |
| `fcef0282` | 4 | `warm_start_v5.py` V4-Max `update_1190.npz` → V5 partial (Q3: faithful base_encoder+action_encoder, shape-compat-disconnected state_fuser.2/candidate_scorer/value_head, fresh rest+mana_draw_head; strict=False via tree_flatten/unflatten) |
| `34744a8b` | 5 | `offline_dataset_loader.py` (snapshot→GameState deserializer + reward mirror of `classic_rl_env.py:383-421` + manifest.json battle discovery + AWAC/CRR tuples; skip orphans; surrender-terminal) |

Suite (final): **python 114 + cargo 157 (110 lib + 47 int), all green.**

---

## 2. Components (what was built)

### 1. `v5_card_shape_v1.py` (73-dim disjoint fork) — `c6020806`
`CARD_SHAPE_DIM_V5 = 73` (USER decision 2026-06-30, "Grow, disjoint"):
- `[0..47)` classic base, `[47..64)` classic 17 scalars (byte-identical via `_encode_mechanics_cached` import), `[64..69)` 5 NEW mechanic one-hots (aoe_silence, team_wide_shield, rebirth, crime_and_punishment, target_ally_max_hp_plus), `[69]` desk_freeze overlap-fix one-hot, `[70..73)` 3 magnitude scalars (rebirth_N/10, crime_and_punishment_N/10, target_ally_max_hp_plus_N/10; aoe_silence+team_wide_shield have no N → grounded count = 3, compensated by the index-69 one-hot to keep dim 73). Classic clobbers desk_freeze at 47 with a damage scalar; V5 re-encodes it disjoint at 69, so `[0:64)` final-values stay byte-identical to classic.
- Cascade in `contracts.py` (named offsets, not literals): `PRIVATE_INFO_DIM = 32*(1+1+73) = 2400`, `HISTORY_EVENT_DIM = 16+146 = 162` → `HISTORY_DIM = 3240`, `OBS_V5_DIM = 1456+32+2400+3240 = 7128`.
- Rust mirror `card_shape_v5.rs` byte-parity; `kernel.rs` wires the V5 obs path to `card.shape_v5()` (wiring, NOT game logic). 24 golden_trace fixtures regenerated from the PYTHON oracle (`golden_trace.py`) — only `obs_v5_dim` (6480→7128) + `obs_v5_sha256` changed; the V1 base-1456 hash (`obs_sha256_f32_le`) UNCHANGED across all 24 → frozen base preserved; Rust `golden_kernel.rs` matches the Python oracle (source-vs-source, NOT self-referential). 2 stale-literal blockers (6480 in `train_v5_adaptive.py:166` + `trace_factory_v5.py:442`) found by verify + fixed.

### 2. `encode_observation_v5` mana_draw_count channel — `b17e1440`
Audit found 5/6 spec §6.184 features already implemented (base-1456 frozen delegation at `obs_v5.py:54`; per-card own-deck 12×75 NOT zone-summary; history window 20 right-aligned; omniscient both hands(4)+decks(12); `_NUM_HAND=4` per Q2). The ONLY missing feature was the `mana_draw_count_this_turn` global channel at `global[15]` → EXTENDED in place: `MANA_DRAW_COUNT_NORMALIZER=5.0` (grounded: max_mana 10 / MANA_DRAW_BASE 2), `dst[15] = min(me.mana_draw_count_this_turn/5.0, 1.0)`. Python↔Rust byte-parity. NO fixture regen (count=0 in all obs_v5-comparing fixtures → no-op; 442 hashes re-encoded, 0 mismatch). **Convention:** the python obs_v5 test `tests/test_train_v3_obs_v5.py` is gitignored (`.gitignore:53 tests/test_train_v3_*.py` — train_v3 tests local-only); durability covered by tracked Rust `kernel.rs` channel tests + `golden_kernel.rs` V5-obs parity. The component-1 test (`test_train_v2_*`) IS tracked (v5_card_shape under tracked `ai/train_v2/`).

### 3. `mana_draw_head_v5.py` (parallel binary head) — `d0137faa`
Pure-Python (no mlx). `mana_draw_legal_mask(state, player_id)` byte-parity with `engine.py` get_legal_actions mana_draw emission (game-over/wrong-turn/unknown-player/hand_full :781 / insufficient_mana :785 / emit :1347) + `golden_trace.py:522-523` predicate (source-vs-source, 17 parametrized cases). `select_includes_mana_draw(mana_draw_logit, best_candidate_logit, mana_draw_legal)` (§6.186: legal AND strict-greater than the best 601-candidate; ties favor the candidate — documented decision, spec §186 doesn't pin it). `v5_policy.py:85` adds `mana_draw_head = nn.Linear(hidden,1)`; `__call__` → 3-tuple `(candidate_logits, value, mana_draw_logit)`, illegal→-inf gating. **601 candidate path frozen (NO 602nd candidate).** 3 callers updated for the 3-tuple arity (`rust_policy`/`rust_ppo`/`rust_benchmark`). Baseline `ai/train_v2/model_mlx.py` untouched. MLX 0.30.1 importable+functional in this worktree (corrects the Q3 "MLX not importable" note) — MLX-head tests RAN.

### 4. `warm_start_v5.py` (V4-Max → V5 partial) — `fcef0282`
`load_v4_max_into_v5(policy, npz_path=None)` + `resolve_v4_max_npz_path()`. Q3 PARTIAL verdict: V5 split-encoder CANNOT reproduce V4-Max 1456→256→601 (fresh `state_fuser.layers.0 (544,256)` + extra SiLU; SiLU nonlinear → no identity init). Transfer map verified against the REAL `v5_policy.py` param tree (`nn.utils.tree_flatten`) + REAL npz keys (`np.load` allow_pickle=False, 33 keys = 10 weight + 22 `_opt_` Adam-state + 1 `__meta__`; meta `model_version=classic_action_conditioned_mlx_v1`, obs_dim=1456, max_candidate_actions=601, update=1190):
- **FAITHFUL** (EXACT, binding gate): `base_encoder.layers.0` (256,1456) ← V4 `state_encoder.layers.0`; `action_encoder` (128,171) ← V4 `action_encoder` (`np.array_equal` asserted).
- **SHAPE-COMPAT-DISCONNECTED** (copy by shape, inputs differ → 601 logits NOT match): `state_fuser.layers.2` (256,256) ← `state_encoder.layers.2`; `candidate_scorer` (1,384); `value_head` (1,256).
- **FRESH** (default MLX init untouched): `global_encoder.layers.0` (32,32), `private_encoder.layers.0` (128,2400), `history_encoder.layers.0` (128,3240), `state_fuser.layers.0` (256,544), `mana_draw_head` (1,256). Component-1 dim grow (PRIVATE_INFO_DIM=2400, HISTORY_DIM=3240) reflected in fresh shapes.

22 `_opt_` Adam-state keys documented-not-loaded (warm-start = policy transfer, not optimizer restore). npz path resolution: explicit arg → `V4_MAX_NPZ_PATH` env → walk-up candidate search (reaches main-repo checkout; npz gitignored in worktrees) → clear RuntimeError. **Non-parity:** with a frozen obs where the V5 base-1456 prefix == V4 obs (so the faithful base_encoder sees identical input) + identical action_features, the V5 601 logits diverge from V4-Max by `max_abs_diff=65.20` (`allclose atol=1e-3=False`); divergence is the fresh fused downstream path, NOT a failed faithful copy (`base_encoder.layers[0](base) == V4 state_encoder.layers[0](base)` EXACTLY). Binding gate = FAITHFUL-LAYER EQUALITY (green); spec §6.188 full-forward-parity RELAXED (green via the non-parity assertion — prevents false confidence that warm-start = parity).

### 5. `offline_dataset_loader.py` (recorded v5 traces → AWAC/CRR) — `34744a8b`
NEW — no deserialize helper existed (grep-confirmed; `GameState`/`PlayerState`/`CardInstance` are plain `@dataclass` with NO `from_dict`). Three pieces:
- **(a) snapshot→GameState deserializer** `reconstruct_gamestate(snapshot)`. `reconstruct_card` mirrors `arena_engine._snapshot_card:914-936` (type/card_type→CardType enum, mechanics, atk/hp/max_hp/mana_cost/is_ready/is_frozen/level; DROPPED `base_*`/`skip_count`/`instant_kill_used`/`simplified_levelup` defaulted — `obs_v5` reads only current values). `reconstruct_player` mirrors `v5_trace._snapshot_player:274-296` (replacement_status→ReplacementStatus, mana_draw_count_this_turn). `reconstruct_gamestate` mirrors `v5_trace._snapshot_state:298-327` (action_history list→`deque(maxlen=100)` of `tuple[str,str]`; status→GameStatus; `pending_mana_drain_by_player`/`sudden_death_*` left at defaults — HARMLESS for obs_v5 which reads only p1/p2/turn_number/current_turn_owner_id/history per `obs_v5.py:43-67`; bridge does NOT re-step the engine).
- **(b) reward mirror** `compute_offline_reward(actor_id, pre, post, accepted, status)` mirrors `classic_rl_env._compute_reward:383-421` BYTE-FOR-BYTE (not accepted→-0.05; p1_win/p2_win→±1.0; draw/stalemate→0.0; else shaped +0.02*enemy_hp_delta -0.01*own_hp_delta +0.03*enemy_killed -0.02*own_killed +min(0.02,0.005*mana_spent)). pre/post are `_snapshot`-shape dicts built directly from the v5 state snapshot (`reward_view_from_snapshot`). `classic_rl_env.py` NOT modified (frozen-classic guard: bridge MIRRORS, does not edit).
- **(c) offline loader** `load_offline_dataset`/`iter_offline_transitions`. Battle discovery via `manifest.json` `battles_results` (`manifest.py:141-168`); for each battle with `v5_trace_ok True`, resolve `v5_dir`/`v5_meta_path`, read `meta.json`, SKIP orphans (meta.status not in {p1_win,p2_win,draw,stalemate} — NO `'finished'` value per `v5_trace_validate.py:51`). Per `actions.jsonl` row: reconstruct pre_state → `encode_observation_v5(pre_gs, actor, info_mode, assist_mode, history_events=pre_snap['history'])`; `encode_action_features` (601,171) with `build_action_mask` mirroring `get_legal_actions` directly (NO engine call) + `include_preview=False` (NO `ArenaEnvironment` wrap needed); reward via the mirror with per-row resolved status; reconstruct post_state → next_obs; terminal = resolved_status terminal; `mana_draw_legal = mana_draw_head_v5.mana_draw_legal_mask`. Emits `OfflineTransition{obs, action_features, action_tcode_or_index, reward, next_obs, terminal, mana_draw_legal, meta}`. **Action representation:** `action_tcode_or_index` = the RECORDED `legal_action_index` (index into `get_legal_actions_raw` 0..N-1, `v5_trace.py:471-475`) — present in the row, NO engine re-run. The V5 601-tcode computation via the `classic_actions_v1` codec is DEFERRED to a later block (Block 0 does NOT bind action-identity correctness; documented choice). DATA-CONTRACT: the bridge READS emitted JSON/JSONL; does NOT import `v5_trace` recorder code (avoids prod-rlhf coupling); `_TERMINAL_STATUSES` re-declared mirroring `v5_trace_validate.py:48,51`.

---

## 3. Validation (per-component 4-way adversarial, source-vs-source)

Each component ran an ultracode workflow: single max-effort implementer → 4 refute-by-default verifiers (parallel) → fix stage. Verdicts:

| Component | Implementer | Verifiers (findings) | Fix | Binding gate |
|---|---|---|---|---|
| 1 | 10 layers mapped | 4 PASS (2 NEEDS-REVIEW → blockers fixed: stale 6480 literal) | fixed | 73-dim byte-parity Python↔Rust across 50 cards; V1 base-1456 hash unchanged |
| 2 | count channel wired | 4 PASS (0 blocker) | none | mana_draw_count channel populated + Python↔Rust parity |
| 3 | 3-tuple head | 4 PASS (0 blocker, minor cosmetic) | none | mask byte-parity `engine.py:781/785/1347` + `golden_trace.py:522-523` |
| 4 | 15 tests | 4 PASS (0 blocker/major) | none | faithful-layer EXACT equality (np.array_equal); non-parity max_abs_diff=65.20 |
| 5 | 6 gates | 4 PASS (0 blocker/major) | none | round-trip obs integrity (spec §6.185) + source-vs-source correctness vs live ClassicRLEnv + reward byte-match |

**Source-vs-source discipline (Block -1 lesson) held throughout:** the live engine / Python oracle is the oracle; the V5 code is the UUT. Self-referential fixture regen (re-encode from the UUT's own output) was avoided — for component 1 the regen used the Python `golden_trace.py` oracle + Rust `golden_kernel.rs` re-verifies (V1 base hash unchanged proves frozen base preserved); for component 5 the correctness gate uses the REAL `V5TraceRecorder._snapshot_state` serializer against live `ClassicRLEnv` states (5 distinct steps), then asserts `obs(reconstructed) == obs(live)` within tolerance — NOT `obs(reconstructed) == obs(reconstructed)`.

---

## 4. Key architecture decisions (do not re-derive)

- **`CARD_SHAPE_DIM_V5 = 73` (Grow, disjoint) — USER decision.** Append 5 new mechanic one-hots at [64..69) + 1 desk_freeze overlap-fix at [69] + 3 magnitude scalars at [70..73); fix the index-47 overlap; cascade derived dims through `contracts.py` (named offsets). Warm-start `base_encoder.layers.0 (1456→256)` unaffected (base obs frozen; card_shape feeds private/history, not base).
- **Q2 ✅ NO hand lift.** `HAND_CAP = 4` (`core/engine.py:44`, single source; all draw paths through `draw_one_from_deck` cap-guard; mana_draw same cap). `_NUM_HAND=4` already correct.
- **Q3 ✅ PARTIAL warm-start.** Faithful = `base_encoder.layers.0` + `action_encoder`; shape-compat-disconnected = `state_fuser.layers.2` + `candidate_scorer` + `value_head`; fresh = global/private/history encoders + `state_fuser.layers.0` + `mana_draw_head`. Spec §6.188 full-forward-parity RELAXED; binding gate = faithful-layer equality. Warm-start = "head start", not parity.
- **Q4 ⏳ open_deferred to Phase A.** Baseline `B = mana_draw_count / eligible_turns` unmeasurable until the Phase A pilot (no battles yet). Plan embeds the measurement spec + A-gate `[0.5×,1.5×]` + E1 band placeholders. Hard dependency: `B` invalidated if `HAND_CAP` or `MANA_DRAW_BASE` change.
- **Q5 ✅ league composition.** self_snapshot 0.55 / v4_orig_argmax 0.15 / v4_orig_t07 0.07 / v4_orig_t12 0.03 / exploit(stall, anti_draw_greed, punish_empty_board) 0.05 each / tail(greedy_face 0.03, legal_random 0.01, end_turn 0.01). mana_draw-collapse monitor: `md_rate(V4-lane)/md_rate(self-snap-lane) ≥ 0.5`; breach ladder (2 windows → reweight self 0.65/V4 0.15 + C2 top-u).
- **mana_draw = parallel binary head (NOT a 602nd candidate; 601 frozen).** mask byte-parity with `engine.py` get_legal_actions + `golden_trace.py:522-523`. Tie-break (strict-greater, ties favor candidate) is a documented decision (spec §186 doesn't pin it).
- **Offline-bridge data-contract:** iterate `manifest.json` (battle index), NOT `catalog.json` (which is card-data, `rlhf_catalog_v1`). meta.status is enum-name lowercased: terminal = `p1_win`/`p2_win`/`draw`/`stalemate` (NO `'finished'`); orphan = `'ongoing'`/missing → skip. `_TERMINAL_TYPES = {surrender,draw,stalemate}` discriminates terminal rows (NOT empty legal_actions). The snapshot OMITS `pending_mana_drain_by_player`/`sudden_death_*` — HARMLESS for `encode_observation_v5` (reads only p1/p2/turn_number/current_turn_owner_id/history); the bridge does NOT re-step the engine.

---

## 5. Frozen-classic guard

Held throughout Block 0. `classic_obs_v1` / `classic_actions_v1` / `classic_card_shape_v1` BYTE-LOCKED (never modified — V4-orig ONNX). `classic_rl_env.py` NOT modified (the reward mirror copies the formula, does not edit it). `v5_trace.py` NOT modified + NOT imported into the loader code path (the bridge READS emitted JSON/JSONL). `core/state.py` NOT modified (the deserializer builds `core.state` dataclasses, lives in the loader file). V4-Max npz + ONNX read-only (the warm-start loader never writes them). `ai/train_v2/model_mlx.py` (V4 baseline) unchanged. Each component's verify phase confirmed `git diff` empty for frozen-classic + read-only sources.

---

## 6. Handoff to Block A

Per `docs/superpowers/specs/2026-06-27-extra-lr-v5-pipeline-{design,handoff}.md`:

- **Block 0 (foundation) — CLOSED by this work.** ✅ All 5 components green.
- **Q4 (mana_draw usage baseline) — OPEN, deferred to Phase A pilot.** Define `mana_draw_count / eligible_turns` over pilot battles; A-gate `[0.5×,1.5×]`; E1 band. Hard dependency on `HAND_CAP` / `MANA_DRAW_BASE`.
- **Next: Block A** — pilot → BC → short redesign Phase A. Run a pilot with the warm-started V5 policy (component 4) + the Rust ArenaEnv (Block -1) + the offline-bridge (component 5) to measure `B` + the league composition (Q5) + the mana_draw-collapse monitor, then the BC (behavioral cloning) → short redesign loop.

---

## 7. Session artifacts (workflow scripts)

Block 0 orchestration scripts persisted under the session workflows directory (component-1..5 ultracode workflows: implementer + 4 refute-by-default verifiers + fix). Committed plan: `BLOCK0_FOUNDATION_PLAN.md`. This completion log: `BLOCK0_FOUNDATION_COMPLETION.md`.

See project-memory `extra-lr-v5-block0-plan.md` (Block 0 plan + per-component execution status), `extra-lr-v5-blockminus1-port-progress.md` (Block -1 closeout + pipeline state), and `extra-lr-v5-blockminus1-parity-audit.md` (the 42-gap audit + 8-phase plan).