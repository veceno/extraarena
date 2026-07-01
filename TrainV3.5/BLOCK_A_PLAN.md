# BLOCK A — Phase-A Pilot + BC + Redesigned PPO (FINAL PLAN)

V5-Max pipeline position: Block -1 done → Block 0 done → **Block A (this plan)**.

This plan merges ALL blocker/major findings from the four adversarial reviews. No
finding is weakened: where a verifier found a real gap (live self-play PPO absent,
max_turns FFI plumbing missing, opponent_mix not dispatchable for 4/10 identities,
A-gate is NEW build not wiring, promotion selector absent, BC test
self-referential, decision_source filter missing), the plan creates the missing
code or corrects the design.

---

## 1. Operational vs in-worktree split (up front)

The central discipline: **what can be built + tested now (synthetic, in-worktree)**
vs **what the user must run live (operational)**.

### IN-WORKTREE-BUILDABLE (I do these now, synthetic-testable)

1. **A.pilot deploy plumbing** — point rlhf_env at the placeholder identity
   (greedy_face already registered at `rlhf_env/components/policy_registry.py:44` +
   `policy_factory.py:128-130`; V4-orig ONNX deploys via argmax,
   `BOT_MAX_DIFFICULTY='max'`); ensure the v5_trace recorder
   (`rlhf_env/components/v5_trace.py:92 V5TraceRecorder`, `:298 _snapshot_state`,
   `:481 action_native = legal[legal_index].to_dict()`, `:496-500 decision_source +
   legal_action_index + action_native`) is on for pilot battles; post-rebalance
   ruleset-verification harness; pilot-manifest builder that enumerates collected
   `battles/<bid>/v5/{actions.jsonl,meta.json,manifest.json}` for the bridge.
   All synthetic-testable now.
2. **A.BC pipeline code** (A1 bc_dataset + A2 bc_train) — NEW pure-Python BC infra
   (grep-confirmed absent: no `behavior_clone`/`bc_train`/`bc_dataset`/`clone_loss`
   anywhere in `TrainV3.5/` or `ai/train_v2/`). Consumes the offline_dataset_loader
   PUBLIC API (`ai/train_v2/offline_dataset_loader.py:772-781 __all__`:
   `reconstruct_gamestate`, `reward_view_from_snapshot`, `compute_offline_reward`,
   `iter_offline_transitions`). The real BC RUN needs collected pilot data
   (operational); the pipeline code + tests are buildable now on synthetic
   pilot-style v5_trace directories.
3. **A.PPO config + live-self-play trainer** (A3 ppo_phaseA_config + A4
   rust_live_self_play) — the 5 root-cause fixes are config/plumbing changes against
   `RustPPOTrainingConfig` (`TrainV3.5/python/train_v3/rust_trainer.py:23-71`) +
   a learner-only reward attribution path + a NEW live-self-play Rust ArenaEnv
   entry point (CONFIRMED ABSENT: `rust_ffi.py` only has `from_trace_file` at
   `:719` + `from_trace_files` at `:834`; `rust_trainer.py:486` uses
   `RustVecEnv.from_trace_files`; `train_v5_adaptive.py:53` calls
   `train_rust_ppo_trace_files` = trace-pool replay, NOT self-play. The live FFI
   primitives DO exist: `step` `:1024`, `current_actor_ids` `:1044`,
   `select_rule_actions` `:1057`, `advance_rule_until_actor` `:1078`. Building the
   trainer that composes them is in-worktree Rust+Python, testable on seeded
   arena scenarios). The actual training RUN needs compute (operational, D-A4).
4. **A-gate + promotion selector + Q4 measurement tooling** (A5) — NEW code
   (verifier-confirmed: the 4 A-gate criteria are NOT existing wiring — see A5
   rationale). The Q4 mana_draw-usage band measurement script is buildable on
   synthetic pilot-style traces now; the real-data measurement needs the pilot.

### OPERATIONAL-NEEDS-USER (user runs these live)

1. **A.pilot data collection** — deploy the placeholder against real human players in
   the live rlhf_env (port 8090), collect ~1-3k fresh human battles post-rebalance.
   Requires the live server + wall-clock + human throughput. CANNOT be done
   autonomously in a worktree. Blocked on ruleset freeze + RLHF env update (handoff
   §3) and on D-A1/D-A2/D-A3.
2. **A.BC real run** — execute the BC pipeline on the collected pilot (needs A.pilot
   done + D-A4 compute).
3. **A.PPO real run** — execute the redesigned short PPO loop, snapshot, external-bench
   promote (needs A.pilot for BC seed + A.BC run + D-A4 compute).
4. **A-gate mana_draw-usage baseline measurement** (Q4) — the [0.5x,1.5x] band is
   defined over the pilot battles; the measurement script is in-worktree-buildable
   but running it on real data requires A.pilot done.

---

## 2. User decisions (from CloseQ — present these to the user)

These are the genuine decisions only the user can make. Design decisions I can
default are listed in §3 per-component.

| ID | Question | Default | Options |
|----|----------|---------|---------|
| **D-A1** | Which placeholder identity does A.pilot deploy against humans? | V4-orig ONNX (frozen codec, blind to mana_draw/new cards but playable, argmax via `BOT_MAX_DIFFICULTY='max'`) | V4-orig ONNX / greedy_face heuristic (already registered) / Both (A/B split: V4-orig primary, greedy_face fallback if ONNX deploy blocked) |
| **D-A2** | Pilot scale: how many fresh human battles before A.BC? | ~2k (mid) | ~1k (min, fastest) / ~2k / ~3k (spec upper, best BC coverage) |
| **D-A3** | Can/will the user run the live rlhf_env pilot now? | Not yet — confirm ruleset frozen + env updated first | Yes now / Not yet (blocked on ruleset freeze + Rust parity Q1) / Yes but on staging/dev 8082 first |
| **D-A4** | PPO compute budget / wall-clock for the short A.PPO run? | Medium (~5-10k updates, ~1 GPU-day) | Small (<=2k, hours) / Medium / Large (>10k, multi-GPU) |
| **D-A5** | N_snapshots for A-gate external-H2H trending-up? | 5 snapshots | 3 / 5 / >=7 (conservative) |
| **D-A8** | Does A.PPO use trace-pool trainer (no Rust change) or a NEW live-self-play Rust ArenaEnv entry point? | Build a NEW live-rollout RustVecEnv + learner-only reward | Extend trace-pool with synthetic self-play traces (no Rust change) / **Build NEW live-rollout** (matches spec "self-play PPO, Rust ArenaEnv") / Hybrid: trace-pool warm-up then live self-play |

**Why D-A8 is a user decision:** the spec (`design.md:101`, task brief) calls for
"self-play PPO (Rust ArenaEnv)" which does NOT exist and must be built (confirmed:
`rust_ffi.py` has no `from_live`/`from_scenario`; `rust_trainer.py:486` uses
`from_trace_files`; `train_v5_adaptive.py:53` drives the trace-pool path). Building
live self-play is materially larger (new FFI constructor + opponent dispatch +
reward wiring) but matches the spec; extending the trace-pool is cheaper but is
not true self-play. The user should confirm spec intent vs the pragmatic
approximation. This plan defaults to **D-A8 = build live self-play** (spec-literal)
and includes the trace-pool fallback path documented as the approximation.

**Frozen / non-decisions (do NOT re-litigate, handoff §2):** gamma frozen v1 codec;
entropy_coef=0.01 (spec-fixed); A-gate thresholds 0.55/0.50/[0.5x,1.5x]
(spec-fixed); `classic_obs_v1`/`classic_actions_v1`/`classic_card_shape_v1`/
`classic_rl_env.py` byte-locked (V5 work in `v5_*` files + NEW files; classic
mirrored, not edited).

---

## 3. In-worktree components (dependency-ordered)

### A1 — `bc_dataset.py` — BC dataset builder + 601-tcode resolver + human-only filter

**File:** `TrainV3.5/python/train_v3/bc_dataset.py` (NEW)

**Purpose:** Consume recorded pilot v5 traces via the offline_dataset_loader
PUBLIC API and emit BC training tuples. RESOLVES the V5 601-tcode that Block 0
component 5 explicitly deferred
(`ai/train_v2/offline_dataset_loader.py:71-87,144-147`: "legal_action_index stores
the RECORDED legal_action_index; the V5 601-tcode is deferred to a later block").

**Resolution method — decode_action matching (engine = oracle, decode_action =
UUT, source-vs-source):** reconstruct `pre_state` via
`reconstruct_gamestate(snapshot)`; build the legal 601-candidate mask
`build_action_mask(pre_state, actor, verify_mask=False, placement_mode='append_only')`
(`classic_actions_v1.py:188`, the frozen codec the policy scores, FROZEN —
read-only); enumerate `legal_action_ids = np.flatnonzero(mask)`; decode each
candidate via `decode_action(pre_state, actor, candidate_id)`
(`classic_actions_v1.py:70`, FROZEN — read-only); value-equality match
`decode_action(candidate).to_dict()` against the **ENGINE-sourced action_native**
(see test-strategy fix below — NOT the codec-sourced value the legacy test helper
uses).

**CRITICAL FIX (verifier finding 4a — self-referential test):** the action_native
must be sourced from the ENGINE oracle, not the codec under test. In production
`v5_trace.py:481` writes `action_native = legal[legal_index].to_dict()` where `legal
= engine.get_legal_actions_raw(...)` — the engine's own `BaseAction`, INDEPENDENT
of `decode_action`. The legacy test helper
(`tests/test_train_v2_offline_bridge.py:231` `chosen = decode_action(...)` then
`:252 action_native = chosen.to_dict()`) sources action_native from the codec under
test, making its round-trip test decode_action-vs-decode_action (cannot detect a
codec-vs-engine regression). A1 must NOT reuse that helper verbatim — it forks a
new helper that sources `action_native = legal_raw[legal_index].to_dict()` (the
engine's BaseAction, same source as `v5_trace.py:481`) so the round-trip assertion
`decode_action(pre_state, actor, resolved_tcode).to_dict() == action_native` is a
true source-vs-source check.

**CRITICAL FIX (verifier finding 4b — decision_source filter):** the pilot deploys
a placeholder BOT against humans, so recorded v5 traces contain BOTH human actions
(`decision_source='human'`) AND placeholder-bot actions (`decision_source in
{'bot','rl','llm'}` — confirmed `rlhf_env/components/match_runner.py:360
decision_source = "rl" if is_p1_rl else "bot"`, and `v5_trace.py:496
"decision_source": decision_source`). The spec is explicit: "Human actions are the
target; placeholder identity doesn't change that (D3)." BUT
`offline_dataset_loader.py` does NOT surface `decision_source` (grep-confirmed:
the yielded `OfflineTransition.meta` at `:169` includes battle_id/seq/action_type/
actor_user_id/turn_number/status but OMITS decision_source, even though the raw row
carries it at `v5_trace.py:496`). Therefore A1 MUST filter to
`decision_source=='human'` before emitting BCTransition, by reading the raw
`actions.jsonl` row's `decision_source` field directly (the loader's
`iter_offline_transitions` reads the same rows — A1 reads them in parallel to
gate, OR the loader's meta is extended additively to include decision_source; the
additive meta extension does NOT touch frozen classic_* or v5_trace.py).

**Output:** `BCTransition{obs, action_features(601,171), target_tcode (0..600 or
None), is_mana_draw (bool), mana_draw_legal, legal_mask(601,), reward, terminal,
meta}`. For `action_type=='mana_draw'`: `target_tcode=None` +
`is_mana_draw=True` (BC targets the mana_draw head, not a 601 slot —
`mana_draw_head_v5.py:4-6` ManaDrawAction is OUTSIDE the 601 space,
`:116 select_includes_mana_draw` legal mask dominates). Terminal rows
(surrender/draw/stalemate per `offline_dataset_loader.py:119-125
_TERMINMINAL_TYPES`) carry no 601 target.

**Depends on:** Block 0 component 5 (offline_dataset_loader — DONE, PUBLIC API);
`classic_actions_v1.py` (FROZEN, read-only decode_action/build_action_mask);
`obs_v5.encode_observation_v5`; `mana_draw_head_v5.mana_draw_legal_mask`;
`core.actions` (PlayCardAction/AttackAction/EndTurnAction/ManaDrawAction
value-equality).

**Acceptance:**
- 601-tcode round-trips: `decode_action(pre_state, actor, tcode) == action_native`
  (ENGINE-sourced) for all non-mana_draw rows across a synthetic battle exercising
  end_turn + play_card (warrior+potion) + attack + mana_draw.
- mana_draw rows flagged `is_mana_draw=True`, `target_tcode=None`.
- **decision_source=='human' filter:** rows with `decision_source in
  {'bot','rl','llm'}` are EXCLUDED from the BC dataset; only human rows emit
  BCTransition. (verifier finding 4b)
- No import of v5_trace recorder code (data-contract read-only); frozen-classic
  guard held; `classic_actions_v1` / `classic_rl_env` / `v5_trace.py` NOT
  modified.

**Tests:** `TrainV3.5/python/train_v3/tests/test_bc_dataset.py` (NEW)
1. `test_tcode_matches_recorded_action` — drive a REAL `ClassicRLEnv` battle,
   serialize with the REAL `V5TraceRecorder._snapshot_state` bound to a fake engine,
   BUT source `action_native = legal_raw[legal_index].to_dict()` (engine oracle,
   NOT `decode_action(...).to_dict()` — fork the helper, do not reuse
   `test_train_v2_offline_bridge.py:_write_real_trace` verbatim). Build the BC
   dataset, assert `decode_action(pre_state, actor, resolved_tcode).to_dict() ==
   action_native` for every non-mana_draw row. (verifier finding 4a — true
   source-vs-source)
2. `test_mana_draw_row_targets_head` — a row where the human took ManaDrawAction
   yields `is_mana_draw=True`, `target_tcode=None`.
3. `test_legal_mask_matches_action_features` — `BCTransition.legal_mask ==
   np.flatnonzero(action_features has nonzero source-card channel)` consistency.
4. `test_orphan_and_terminal_skip` — orphan (`meta.status` ongoing) +
   surrender-terminal rows handled per `_TERMINAL_TYPES`.
5. `test_decision_source_human_filter` — a synthetic battle with mixed
   `decision_source` rows (human + bot + rl) → only `decision_source=='human'` rows
   emit BCTransition; bot/rl rows excluded. (verifier finding 4b)

Synthetic data only — live `ClassicRLEnv` is the oracle (source-vs-source).

---

### A2 — `bc_train.py` — BC training loop (warm_start_v5 → BC fine-tune)

**File:** `TrainV3.5/python/train_v3/bc_train.py` (NEW)

**Purpose:** Behavior-cloning training loop (NO BC/imitation infra exists —
grep-confirmed). Sequence: load `V5ActionConditionedPolicy` (`v5_policy.py:29`) →
`warm_start_v5.load_v4_max_into_v5` (`warm_start_v5.py:176`, Block 0 component 4)
→ BC fine-tune on the A1 pilot dataset. **Loss = supervised cross-entropy on the
601-candidate logits MASKED to legal candidates** (target =
`BCTransition.target_tcode`; CE over `legal_action_ids` only, illegal slots masked
to -inf via `legal_mask`) **PLUS BCE on the mana_draw head** (target = 1.0 if
`is_mana_draw` else 0.0; BCE applied ONLY on rows where `mana_draw_legal` is True —
illegal rows the head is -inf and the target is forced 0/not-trained, mirroring
`mana_draw_head_v5.select_includes_mana_draw` at `mana_draw_head_v5.py:116`).
Optional value-head fine-tune on `BCTransition.reward` (OFF by default — BC is
policy-focused). **Objective = plain cross-entropy BC (D-A9 spec-literal,
design.md:100; AWAC/CRR reserved for C-phase, handoff D9).** MLX (runs in the
TrainV3.5 training env; skip-gated in the worktree where MLX is unimportable).
Emits a BC-seed checkpoint (npz via `ai.train_v2.model_mlx.save_checkpoint`,
same format as `rust_trainer._save_checkpoint` at `rust_trainer.py:838`)
consumable by A.PPO.

**freeze_faithful option (default True):** the Q3 FAITHFUL layers
(`warm_start_v5.py:67-73` `base_encoder.layers.0` + `action_encoder`) are frozen
during BC (learning-rate 0 on those layers) so BC does NOT destroy the warm start.
Only `candidate_scorer` + `mana_draw_head` + fresh V5 layers move.

**Depends on:** A1 (bc_dataset); `warm_start_v5.py` (Block 0 component 4 — DONE,
consumed read-only); `v5_policy.V5ActionConditionedPolicy`;
`model_mlx.save_checkpoint`; `V4_MAX_NPZ_PATH` env / main-repo npz (gitignored —
skip-gate if absent).

**Acceptance:**
- BC loss decreases on synthetic pilot (mean CE+BCE strictly down over 20 steps).
- FAITHFUL layers preserved byte-match when `freeze_faithful=True` (warm-start not
  destroyed).
- mana_draw head gradient flows (logit on legal-mana_draw states > illegal states
  after BC on a 50%-draw synthetic set).
- Checkpoint round-trip: loads back into `V5ActionConditionedPolicy` with identical
  forward.
- classic_* untouched; `warm_start_v5.py` NOT modified (consumed read-only).

**Tests:** `TrainV3.5/python/train_v3/tests/test_bc_train.py` (NEW)
1. `test_loss_decreases` — synthetic 200-transition pilot, 20 BC steps, mean
   CE+BCE strictly decreases.
2. `test_faithful_layers_preserved_after_bc` — after BC with
   `freeze_faithful=True`, `base_encoder.layers.0` + `action_encoder` byte-match
   the V4-Max source.
3. `test_mana_draw_head_learns_signal` — synthetic 50%-draw dataset → after BC
   mana_draw_logit on legal-mana_draw states > illegal states.
4. `test_skip_if_no_mlx` — skip-gate when mlx unimportable (worktree).

Synthetic data via the REAL `V5TraceRecorder` serializer pattern (forked,
engine-sourced action_native per A1 fix).

---

### A3 — `ppo_phaseA_config.py` — redesigned Phase-A PPO config + learner-only reward attribution

**File:** `TrainV3.5/python/train_v3/ppo_phaseA_config.py` (NEW)

**Purpose:** Phase-A PPO config dataclass + helpers encoding the 5 root-cause
fixes (`design.md:103-109`). Superset of `RustPPOTrainingConfig`
(`rust_trainer.py:23-71`) consumed by both the trace-pool trainer AND the new
live-self-play trainer (A4).

**Fix mapping (all 5 with explicit config field + regression test):**

1. **LEARNER-ONLY reward** — root cause at
   `TrainV3.5/scripts/run_phase26_noassist_easy_gate.py:490
   step_rewards = learner_rewards + opponent_rewards` (verifier-confirmed).
   **CLARIFICATION (verifier finding 2c):** `reward_v5.py` is ALREADY per-side
   (`reward_v5.py:40 reward_snapshot_v5` takes a `player_id`; `:55
   compute_reward_components_v5` computes per-side deltas for ONE player; `:109
   compute_weighted_reward_v5` shapes a single base_reward). There is NO
   learner+opponent summation in `reward_v5.py` — the bug is ONLY in the legacy
   phase26 script. **Fix #1 is a TRAINER-side attribution change** (A4 records
   only learner-actor step rewards, zeros opponent-actor steps), NOT a
   `reward_v5.py` edit. `reward_v5.py` is consumed read-only (frozen-classic
   guard). The config carries `learner_only_reward=True` + a
   `reward_attribution(step_rewards, actor_ids, learner_actor)` helper that zeroes
   opponent-actor rewards. For the trace-pool path this is a no-op
   (`collect_rust_vec_rollout` at `rust_collector.py:218` records `step.rewards`
   already learner-attributed per the Rust env).
2. **max_turns >= 120 + decisive-state early-end** (D-A6 default 120 +
   win-margin threshold). **PROVENANCE CORRECTION (verifier finding 4c):** the
   "was 80" attribution to a legacy phase script is UNVERIFIABLE —
   `run_phase26_noassist_easy_gate.py` has NO `max_turns`/`horizon` field (only
   `steps_per_update=16` at `:73`); the "80" is the Rust `KernelConfig` serde
   default (`kernel.rs:624`), NOT a phase-script value. Fix is unaffected: spec
   mandates >=120 (`design.md:106`). **FFI PLUMBING FIX (verifier finding 2b —
   major):** the Python→Rust path for max_turns is currently a NO-OP:
   `rust_ffi.py:719 from_trace_file` and `:834 from_trace_files` have NO
   max_turns parameter; `trace_factory_v5.py` does NOT write `max_turns` into
   `env_config` (only `level_handicap` at `:101`). Thus generated traces fall back
   to the Rust serde default 80. **A3 specifies the plumbing:** (a) trace-pool
   path — `trace_factory_v5.py` MUST write `env_config['max_turns'] = 120` into
   generated traces (alongside `level_handicap`); (b) live path — the new
   `from_live`/`from_scenario` constructor (A4) MUST accept a `max_turns` argument
   and set `config.max_turns` before `KernelConfig` construction
   (`kernel.rs:660 from_trace_config`). A test asserts the constructed
   `KernelConfig.max_turns == 120` (Rust side) AND trace `env_config` carries
   `max_turns`. Without this, fix #2 never takes effect.
3. **entropy_coef = 0.01** (already `RustPPOTrainingConfig` default at
   `rust_trainer.py:38`; phase26 overrode to 0.035 at `:80`) — pin 0.01.
4. **epochs = 6** (D-A7 mid-band; phase26 used 1 at `:292`; `rust_trainer.py:34`
   default is 3) — override to 6.
5. **graduated opponent_mix** (`design.md:111`): legal_random(random) 0.10,
   end_turn 0.05, greedy_face 0.10, face_rush 0.10, board_control 0.10,
   greedy_trade 0.10, stall 0.10, anti_draw_greed 0.10, self_prev(self/v5_snapshot)
   0.10, v4-orig-argmax(v4max) 0.15 — replaces the `legal_random:0.55` random
   majority at `run_phase26:53` (DEFAULT_OPPONENT_MIX) + `:836
   sample_agent_codes`. **VALIDATION (verifier-confirmed):** all 10 names parse via
   `league_v5.parse_v5_opponent_mix` (`league_v5.py:43`) because
   `V5_OPPONENT_KINDS` (`:12-21`) includes self/v5_snapshot/v4max/random/
   greedy_face/end_turn/llm_teacher + `*EXPLOIT_AGENT_KINDS`
   (`gauntlet_v5.py:8-16`). **BUT validation != runtime dispatch — see A4 blocker
   fix.** Default D-A11: use spec weights verbatim, monitor Q5 blind-lane bias
   (handoff §5) rather than pre-emptively reweighting.

Plus **second-start oversampling** (D-A10: gap-weighted p1/p2 init split, mirrors
Block B `design.md:119-120`).

**Depends on:** `rust_trainer.RustPPOTrainingConfig` (`:23`); `reward_v5` (read-
only, learner-only attribution is in the trainer not here);
`league_v5.parse_v5_opponent_mix` (`:43`) + `V5_OPPONENT_KINDS` (`:12`);
`gauntlet_v5.EXPLOIT_AGENT_KINDS` (`:8`).

**Acceptance:**
- All 5 root causes have an explicit config field + a regression test.
- Learner-only reward path is a NEW trainer-side path (does NOT edit run_phase26
  or `classic_rl_env.py` or `reward_v5.py` — frozen-classic guard; verifier
  finding 2c honored).
- opponent_mix validates against `league_v5.V5_OPPONENT_KINDS` +
  `gauntlet_v5.EXPLOIT_AGENT_KINDS` to exactly the 10 spec weights summing to 1.0.
- entropy_coef=0.01 / epochs=6 pinned (fail if drifted).
- **max_turns plumbing specified** (trace-pool writes env_config; live constructor
  threads max_turns into KernelConfig) — verifier finding 2b.

**Tests:** `TrainV3.5/python/train_v3/tests/test_ppo_phaseA_config.py` (NEW)
1. `test_learner_only_reward_zeroes_opponent` — given a reward tape with
   learner+opponent actor ids, attribution returns ONLY learner-actor rewards
   (regression guard for `run_phase26:490`).
2. `test_opponent_mix_parses_to_spec_weights` — graduated mix parses to exactly
   the 10 spec weights summing to 1.0.
3. `test_max_turns_and_decisive_early_end` — config validates `max_turns>=120`
   and the decisive-state predicate (win-margin>threshold → terminal) is a pure
   function of a state snapshot. **PLUS** `test_max_turns_threaded_to_rust` —
   trace `env_config` carries `max_turns=120`; constructed `KernelConfig.max_turns
   == 120`. (verifier finding 2b)
4. `test_entropy_and_epochs_pinned` — entropy_coef==0.01, epochs==6.
5. `test_second_start_oversampling_scheme` — p1/p2 init split is gap-weighted.
Pure-Python, no MLX, no Rust FFI (max_turns threading tested via the env_config
dict + a mocked constructor signature).

---

### A4 — `rust_live_self_play.py` — live-self-play trainer (THE MISSING ENTRY POINT) + policy-opponent dispatch + max_turns FFI

**File:** `TrainV3.5/python/train_v3/rust_live_self_play.py` (NEW)

**Purpose:** THE MISSING LIVE-SELF-PLAY ENTRY POINT (D-A8 default = build live).
Composes the EXISTING Rust FFI primitives that NO trainer currently wires:
`RustBatchWorker.step` (`rust_ffi.py:1024`) for learner turns +
`advance_rule_until_actor` (`:1078`) for opponent turns +
`current_actor_ids` (`:1044`) to know whose turn + `select_rule_actions`
(`:1057`) to sample the opponent_mix. The legacy phase26 composed exactly these
(`run_phase26:475-499`) but summed learner+opponent rewards (the bug). This
trainer: (a) initializes a fresh Rust arena per env (NEW
`RustVecEnv.from_scenario`/`from_live` constructor — `RustBatchWorker` currently
only `from_trace_file`/`from_trace_files` at `rust_ffi.py:719,834`; **additive —
does NOT modify existing constructors, frozen-classic guard: classic_rl_env.py
untouched, rust_ffi.py is NOT frozen-classic**); (b) per step: if learner to act →
policy selects 601-candidate + mana_draw head, step, record LEARNER reward only;
else → opponent dispatch (see below), record ZERO reward for opponent steps
(learner-only, fix #1); (c) enforce max_turns>=120 + decisive-state early-end
(threaded via the new constructor — A3 plumbing fix #2); (d) PPO update via
`rust_ppo.prepare_rust_ppo_batch` + `train_rust_ppo_minibatch`
(`rust_trainer.py:348,360`); (e) snapshot for external-bench promotion (A5).

**BLOCKER FIX (verifier finding 2a — opponent_mix dispatch):** Fix #5 is NOT
fully dispatchable via the Rust `select_rule_actions` mechanism alone.
`select_rule_action_for_state` (worker.rs:1269-1288) only handles integer codes
0-7 (0=legal_random, 1=FaceRush, 2=BoardControl, 3=GreedyTrade, 4=Stall,
5=PunishEmptyBoard, 6=AntiDrawGreed, 7=AntiHandLeakOverfit). The spec mix
(`design.md:111`) includes **4 identities with NO rule-agent code** that would hit
`worker.rs:1281 unknown rule agent code`: `end_turn` (0.05), `greedy_face`
(0.10), `self_prev`/self/v5_snapshot (0.10), `v4-orig-argmax`/v4max (0.15) —
**40% of the mix weight**. These are POLICY opponents, not rule agents. The
legacy phase26 mix worked because it used ONLY rule-agent codes
(`run_phase26:45-54 RULE_AGENT_CODES`: legal_random/face_rush/board_control/
greedy_trade/stall/punish_empty_board/anti_draw_greed/anti_hand_leak_overfit, all
0-7). The new mix is genuinely broader.

**A4 splits opponent dispatch into two paths:**
- **(a) Rule-agent identities** (legal_random, face_rush, board_control,
  greedy_trade, stall, anti_draw_greed — codes 0,1,2,3,4,6) via
  `advance_rule_until_actor(select_rule_actions(agent_codes 0-7))`.
- **(b) POLICY-opponent identities** (end_turn, greedy_face, self_prev/v5_snapshot,
  v4-orig-argmax/v4max) via a **Python-side opponent loop** that, when
  `current_actor_ids != learner`, calls `env.step` with the opponent policy's
  selected action. Mirrors legacy `ai/train_v2/rollout_worker.py:211-227
  _get_opponent_policy` + `_auto_play_until_learner` (`:230`):
  - `end_turn` — Python heuristic policy that always emits EndTurnAction.
  - `greedy_face` — Python heuristic (already registered in policy_registry;
    reuse the heuristic as an in-loop policy).
  - `self_prev` — a frozen V5 snapshot policy (load a prior A.PPO snapshot as the
    opponent policy fn).
  - `v4-orig-argmax` — the V4 ONNX argmax policy (`opponents_v5.py:23,85` marks
    `v4max` execution_kind='offline_v4max_teacher'; reuse the argmax policy fn).
  - Reward attribution: ZERO for all opponent steps (learner-only, fix #1).

  Alternatively, `end_turn`+`greedy_face` may be ported to new Rust rule-agent
  codes (8,9) in `worker.rs:1236-1247` and a policy-opponent FFI path added for
  self_prev/v4-orig-argmax. The plan describes the Python-side path as the default
  (smaller Rust surface change); the Rust-port alternative is documented.

**max_turns FFI threading (verifier finding 2b):** the new
`from_scenario`/`from_live` constructor accepts `max_turns` and sets
`config.max_turns` before `BatchedRolloutWorker` construction
(`kernel.rs:660`). Test asserts `KernelConfig.max_turns==120`.

**D-A8 trace-pool fallback (documented approximation):** if D-A8 = trace-pool, A4
is a synthetic-self-play-trace generator + `train_rust_ppo_trace_files` run with
A3 config (smaller scope, documented as NOT true self-play). The opponent_mix
dispatch issue still applies to generated traces — the generator must emit the
correct agent_codes per identity (rule-agent codes 0-7 for the 6 rule identities;
the 4 policy identities require the generator to also produce the opponent's
action via the Python policy and write it as a trace row, or the trace-pool path
degrades to rule-agents-only and the 4 policy identities are deferred — a
documented limitation the user accepts by choosing trace-pool).

**Depends on:** A3 (ppo_phaseA_config); `rust_ffi.RustBatchWorker` (`:684` —
step/current_actor_ids/select_rule_actions/advance_rule_until_actor);
`rust_collector.collect_rust_vec_rollout` (`:52`) + `RustTransitionBatch`
(`:25`); `rust_ppo.prepare_rust_ppo_batch` + `train_rust_ppo_minibatch`;
`v5_policy` (policy fn) + `mana_draw_head_v5.select_includes_mana_draw`
(`:116`); `league_v5.parse_v5_opponent_mix`; `opponents_v5` (v4max policy);
D-A8 decision.

**Acceptance:**
- Live-self-play trainer runs one finite PPO update on a seeded arena.
- Learner-only reward confirmed (opponent steps zero — verifier finding 2a/b).
- opponent_mix sampled per spec; **both rule-agent AND policy-opponent paths
  exercised** (all 10 identities dispatchable, not just the 6 rule-agents —
  verifier finding 2a blocker fixed).
- max_turns>=120 enforced; `KernelConfig.max_turns==120` confirmed (verifier
  finding 2b).
- NO modification to existing `rust_ffi.py` constructors beyond ADDING
  `from_scenario`/`from_live` (additive; frozen-classic guard:
  `classic_rl_env.py` untouched).
- If D-A8 = trace-pool: synthetic-self-play-trace generator + `train_rust_ppo_trace_files`
  run with A3 config; opponent_mix dispatch limitation documented.

**Tests:** `TrainV3.5/python/train_v3/tests/test_rust_live_self_play.py` (NEW)
1. `test_one_ppo_update_seeded_arena` — run one finite PPO update on a seeded
   arena, assert it completes.
2. `test_opponent_steps_zero_reward` — opponent-actor steps record ZERO reward
   (learner-only, regression guard for `run_phase26:490`).
3. `test_all_ten_identities_dispatch` — for each of the 10 opponent_mix
   identities, assert the trainer dispatches correctly (rule-agent path for the
   6 rule identities; policy-opponent path for end_turn/greedy_face/self_prev/
   v4-orig-argmax). (verifier finding 2a blocker)
4. `test_max_turns_threaded` — constructed `KernelConfig.max_turns==120`.
5. `test_decisive_state_early_end` — a state with win-margin>threshold terminates
   early.
6. `test_skip_if_no_rust_ffi` — skip-gate when Rust extension unbuildable in
   worktree.

---

### A5 — `a_gate.py` — A-gate + promotion selector + Q4 measurement (NEW BUILD, not wiring)

**File:** `TrainV3.5/python/train_v3/a_gate.py` (NEW)

**Purpose:** **THIS IS NEW CODE, NOT WIRING** (verifier finding 3a blocker + 3c
major). Three of the four A-gate criteria have NO existing infrastructure; the
fourth (H2H trending) likewise does not exist. The plan promotes the A-gate from
operational_split prose to a NAMED COMPONENT with file/acceptance/tests
(verifier finding 3c).

**Verifier-confirmed gaps (why this is NEW, not wiring):**
- `V5GauntletConfig.no_assist_min_score_rate` (`gauntlet_v5.py:42 = 0.45`) and
  `exploit_resistance_min_score_rate` (`:46 = 0.42`) are **DEAD FIELDS** —
  grep-confirmed never read anywhere outside their definition.
- `run_v5_acceptance.py` plays NO games: it reads pre-computed winrates from a
  benchmark JSON (`:320-322`) + checks config FLAGS (`:488
  candidate_no_assist_hidden_mode`), NOT a score rate. The cited `:41
  --min-no-bonus-p1` (default 0.75) is a v4max no-bonus benchmark, NOT the
  no_assist/exploit_resistance thresholds (verifier finding 2d + 3a).
- The REAL score-rate harness is
  `TrainV3.5/scripts/run_phase1_runtime_acceptance_bench.py:629-634` with
  SEPARATE hardcoded thresholds (`0.45`/`0.42` at `:51-52
  random_min_score_rate`/`scenario_min_score_rate`) and has NO `no_assist`
  lane — never cited by the prior plan.
- mana_draw-usage band [0.5x,1.5x]: ZERO infrastructure anywhere (grep-confirmed
  empty in `gauntlet_v5.py`/`run_v5_acceptance.py`/`league_v5.py`).
- H2H vs best-self-snapshot trending: ZERO infrastructure.
  `compare_adaptive_strength_monotonicity` (`league_v5.py:146`) is a SYNTHETIC
  FORMULA comparing `evaluate_adaptive_strength_proxy` floats (`:125-143`, 0.25 vs
  1.0), NOT H2H games vs a self-snapshot (verifier finding 3a).
- `run_v5_acceptance.py:16` is broken in this worktree: `sys.path.insert(0, str(ROOT
  / "TrainV3" / "python"))` but the worktree contains `TrainV3.5/` not `TrainV3/`
  (verifier finding 2e). A5 must NOT rely on that script's path setup; it imports
  `V5GauntletConfig` + `build_default_exploit_gauntlet` directly from `train_v3`
  via the TrainV3.5 path, with a smoke import test.

**A5 builds:**
1. **no_assist score-rate gate (>=0.55)** — a V5 gauntlet game-runner that plays
   candidate vs no-assist exploit-lane opponents and computes a real score rate
   (wins + 0.5*draws)/total, mirroring
   `run_phase1_runtime_acceptance_bench.py:704 score_rate` + `:717`. The gate
   threshold is RAISED from the current `0.45` (`gauntlet_v5.py:42`,
   `run_phase1_runtime_acceptance_bench.py:51`) to the spec **0.55**
   (`design.md:114`). Override `V5GauntletConfig.no_assist_min_score_rate`
   0.45→0.55 at construction (verifier finding 2d + 3a: cite `gauntlet_v5.py:42`
   as the real field, NOT `run_v5_acceptance.py:41`).
2. **exploit_resistance score-rate gate (>=0.50)** — same game-runner vs the
   exploit-lane roster (`gauntlet_v5.py:8 EXPLOIT_AGENT_KINDS`); threshold RAISED
   from `0.42` (`gauntlet_v5.py:46`, `run_phase1_runtime_acceptance_bench.py:52`)
   to the spec **0.50** (`design.md:114`). Override
   `V5GauntletConfig.exploit_resistance_min_score_rate` 0.42→0.50.
3. **mana_draw-usage band gate [0.5x,1.5x]** — NEW measurement: compute
   `mana_draw_count / eligible_turns` distribution over the pilot battles
   (handoff Q4, `handoff.md:80`), compare candidate usage to the human baseline;
   gate passes when candidate usage ∈ [0.5×, 1.5×] human baseline. The
   measurement script is in-worktree-buildable (synthetic-testable on pilot-style
   v5_trace dirs); the real-data run needs A.pilot done (operational). (verifier
   finding 3d — allocated as A5 sub-task.)
4. **H2H vs best-self-snapshot trending (>= N snapshots, D-A5 default 5)** — NEW:
   run real H2H games between the candidate and the best prior self-snapshot (NOT
   the synthetic `compare_adaptive_strength_monotonicity` formula); track score
   rate over N snapshots; gate passes when trending up. (verifier finding 3a).
5. **Promotion selector** (verifier finding 3b major) — chooses the best snapshot
   by EXTERNAL-BENCH H2H vs best-self-snapshot. **Explicit guard: promotion is NOT
   driven by PPO loss/KL/entropy** (monitoring-only, `design.md:112`); a regression
   test asserts promotion is independent of PPO loss/KL/entropy values. Wired
   into the A4 snapshot cadence.

**Depends on:** A4 (snapshot cadence); `gauntlet_v5.V5GauntletConfig` (`:42,46`),
`build_default_exploit_gauntlet`, `EXPLOIT_AGENT_KINDS` (`:8`);
`league_v5.parse_v5_opponent_mix`; `run_phase1_runtime_acceptance_bench.py:704,717
score_rate` (mirrored, not imported from the broken-path script); D-A5
(N_snapshots).

**Acceptance:**
- no_assist score-rate gate enforces >=0.55 (raised from dead-field 0.45).
- exploit_resistance score-rate gate enforces >=0.50 (raised from dead-field
  0.42).
- mana_draw-usage band gate computes the [0.5x,1.5x] check on synthetic pilot
  traces; real-data run deferred to A.pilot (operational).
- H2H trending tracker runs real games vs best self-snapshot over N snapshots
  (D-A5); passes when trending up.
- **Promotion selector chooses by external-bench H2H only; regression test
  asserts promotion is independent of PPO loss/KL/entropy.** (verifier finding
  3b)
- A-gate emits a single Phase-A pass/fail.
- Does NOT rely on `run_v5_acceptance.py` path setup (broken in TrainV3.5
  worktree, verifier finding 2e).

**Tests:** `TrainV3.5/python/train_v3/tests/test_a_gate.py` (NEW)
1. `test_no_assist_gate_threshold_055` — gate fails at 0.50, passes at 0.55.
2. `test_exploit_resistance_gate_threshold_050` — gate fails at 0.45, passes at
   0.50.
3. `test_mana_draw_band_synthetic` — synthetic pilot traces with known
   mana_draw_count/eligible_turns → band check emits the right pass/fail for
   [0.5x,1.5x].
4. `test_h2h_trending_up` — synthetic snapshots with monotonically improving H2H
   score → trending gate passes; non-monotone → fails.
5. `test_promotion_independent_of_ppo_loss` — two snapshots with identical
   external-bench H2H but different PPO loss/KL/entropy → same promotion
   decision. (verifier finding 3b)
6. `test_smoke_import_train_v3` — A5 imports `V5GauntletConfig` +
   `build_default_exploit_gauntlet` directly from `train_v3` via the TrainV3.5
   path (NOT via `run_v5_acceptance.py`'s broken `TrainV3` path). (verifier
   finding 2e)
7. `test_skip_if_no_mlx_or_rust` — skip-gate when MLX/Rust unbuildable.

---

## 4. A.pilot deploy plumbing (in-worktree-buildable, supports the operational pilot)

Not a numbered component (config + verification scripts), but required for the
operational pilot and buildable now:

- **Placeholder identity wiring** — greedy_face already registered
  (`policy_registry.py:44`, `policy_factory.py:128-130`); V4-orig ONNX deploys via
  policy_factory argmax (`BOT_MAX_DIFFICULTY='max'`, `policy_adapters.py:164`
  BerserkPolicy argmax). Remaining: a config switch selecting D-A1's choice.
- **v5_trace recorder ON for pilot battles** — `v5_trace.py:92 V5TraceRecorder`
  is the existing omniscient offline-only recorder; verify it is enabled in the
  pilot rlhf_env config.
- **Post-rebalance ruleset-verification harness** — synthetic-testable
  confirmation that mana_draw + new cards are legal in the current engine.
- **Pilot-manifest builder** — enumerates collected
  `battles/<bid>/v5/{actions.jsonl,meta.json,manifest.json}` for the
  offline_dataset_loader + A1 BC dataset.

**Acceptance:** greedy_face + V4-orig-argmax both deploy against a seeded arena;
v5_trace records `decision_source` + `action_native` + `legal_action_index`
(`v5_trace.py:496-500`); manifest builder emits a list consumable by
`iter_offline_transitions` (`offline_dataset_loader.py:627`).

---

## 5. Test strategy (synthetic + source-vs-source)

- **Source-vs-source discipline (Block -1/0 lesson, verifier finding 4a):** the
  live engine = oracle, V5 code = UUT. The A1 round-trip test sources
  `action_native` from the ENGINE oracle (`legal_raw[legal_index].to_dict()`, same
  source as `v5_trace.py:481`), NOT from `decode_action` (the codec under test).
  The legacy `test_train_v2_offline_bridge.py:_write_real_trace` helper is FORKED,
  not reused verbatim, precisely because it sources action_native from the codec
  (`:231 chosen = decode_action(...)` then `:252 action_native = chosen.to_dict()`)
  — that would make the test decode_action-vs-decode_action and unable to detect a
  codec-vs-engine regression.
- **Synthetic data only** for all in-worktree tests: fabricated v5_trace
  directories (actions.jsonl + meta.json + manifest.json) exercising end_turn +
  play_card (warrior+potion) + attack + mana_draw, with mixed decision_source
  rows. The real `ClassicRLEnv` is the oracle for the source-vs-source check.
- **Frozen-classic guard:** no test or component edits
  `classic_obs_v1`/`classic_actions_v1`/`classic_card_shape_v1`/`classic_rl_env.py`
  /`v5_trace.py`/`warm_start_v5.py`/`reward_v5.py`/`run_phase26*.py`. All are
  consumed read-only or mirrored. V5 work is in `v5_*` files + the NEW
  `bc_dataset.py`/`bc_train.py`/`ppo_phaseA_config.py`/`rust_live_self_play.py`/
  `a_gate.py`. `rust_ffi.py` is NOT frozen-classic — the A4 constructors are
  ADDITIVE (no existing constructor modified).
- **Skip-gates:** MLX tests skip when mlx unimportable (worktree); Rust FFI tests
  skip when the extension is unbuildable.

---

## 6. Operational steps (user runs these, in order)

1. **Resolve user decisions D-A1..D-A5 + D-A8** (§2). D-A8 in particular determines
   A4 scope (live self-play build vs trace-pool approximation).
2. **Freeze ruleset + update RLHF env** (handoff §3 blocks 1-2; D-A3). Training
   cannot start until the post-rebalance ruleset is frozen and Rust ArenaEnv
   reaches parity (Q1).
3. **A.pilot data collection** — deploy the placeholder (D-A1) against real humans
   in rlhf_env (8090, or staging 8082 first per D-A3), collect ~1-3k battles
   (D-A2). The in-worktree deploy plumbing (§4) makes this possible.
4. **A.BC real run** — execute the A2 BC pipeline on the collected pilot (needs
   A.pilot done + D-A4 compute). Produces the V5-seed checkpoint.
5. **A.PPO real run** — execute the A4 redesigned short PPO loop (needs A.BC seed
   + D-A4 compute), snapshot per cadence, A5 promotion selector picks the best.
6. **A-gate mana_draw-usage baseline measurement** (Q4) on the collected pilot —
   feeds the A5 band gate. (operational; the measurement script is built in A5.)

---

## 7. A-gate summary (exit Phase A, `design.md:114`)

| Criterion | Threshold | Source | Built in |
|-----------|-----------|--------|----------|
| no_assist score rate | >= 0.55 (RAISED from dead-field 0.45) | `design.md:114`; field `gauntlet_v5.py:42` | A5 (NEW game-runner) |
| exploit_resistance score rate | >= 0.50 (RAISED from dead-field 0.42) | `design.md:114`; field `gauntlet_v5.py:46` | A5 (NEW game-runner) |
| mana_draw usage ∈ [0.5×, 1.5×] human baseline | [0.5, 1.5] ratio | `design.md:114`, handoff Q4 | A5 (NEW measurement; real-data operational) |
| external H2H vs best self-snapshot trending up | >= N snapshots (D-A5 default 5) | `design.md:114` | A5 (NEW H2H tracker) |
| Promotion selector | external-bench H2H only; NOT PPO loss/KL/entropy | `design.md:112` | A5 (NEW selector + guard) |

---

## 8. Open questions / unresolved

1. **D-A8 (live self-play vs trace-pool approximation)** — the spec calls for
   "self-play PPO (Rust ArenaEnv)" which does not exist and must be built. This is
   the largest in-worktree build for Block A. The user should confirm spec intent
   (live self-play, default) vs the pragmatic trace-pool approximation (smaller,
   but NOT true self-play and degrades opponent_mix to rule-agents-only for the 4
   policy identities). The plan defaults to live self-play (spec-literal) and
   documents the trace-pool fallback.
2. **D-A3 (ruleset freeze + RLHF env update status)** — the hard operational gate
   for A.pilot. Only the user knows whether the post-rebalance ruleset is frozen
   and Rust ArenaEnv reaches parity (Q1).
3. **Q5 (V4-orig blind-lane bias, handoff §5)** — the v4-orig-argmax opponent
   (0.15 of the mix) is blind to mana_draw/new cards; risk of blind-lane bias. The
   plan defaults to monitoring (D-A11), not pre-emptive reweighting. Confirm
   during A.PPO.
4. **decisive-state early-end threshold** (D-A6) — the win-margin threshold for
   early termination is a design default; the user may want a specific value.
5. **max_turns "was 80" provenance** — UNVERIFIABLE from any phase script
   (verifier finding 4c); the "80" is the Rust `KernelConfig` serde default
   (`kernel.rs:624`), not a phase-script value. The fix (>=120) is unaffected.
   Cited correctly as the Rust default, not a legacy override.
6. **rlhf-components-untracked memory note is stale for this worktree** (verifier
   finding 1b): `git ls-files rlhf_env/components/` lists all 18 files as tracked
   here (glm-TrainV3.5Prep). Edits are git-recoverable in this worktree; the
   "back up before bulk edits" caution applies only to the original repo path.

---

## 9. Dependency graph

```
Block 0 (DONE: v5_card_shape_v1, encode_observation_v5, mana_draw_head_v5,
         warm_start_v5, offline_dataset_loader)
   |
   +-- A1 bc_dataset.py (601-tcode resolver + human-only filter)
   |     depends: offline_dataset_loader (PUBLIC API), classic_actions_v1 (FROZEN RO)
   |     |
   |     +-- A2 bc_train.py (warm_start → BC fine-tune)
   |           depends: A1, warm_start_v5 (RO), v5_policy, model_mlx
   |
   +-- A3 ppo_phaseA_config.py (5 root-cause fixes + max_turns plumbing spec)
   |     depends: rust_trainer.RustPPOTrainingConfig, reward_v5 (RO), league_v5, gauntlet_v5
   |     |
   |     +-- A4 rust_live_self_play.py (live-self-play trainer + policy-opponent dispatch + max_turns FFI)
   |           depends: A3, rust_ffi.RustBatchWorker, rust_collector, rust_ppo, v5_policy, opponents_v5
   |           |
   |           +-- A5 a_gate.py (A-gate NEW build + promotion selector + Q4 measurement)
   |                 depends: A4 (snapshot cadence), gauntlet_v5, league_v5, D-A5
   |
   +-- A.pilot deploy plumbing (§4; supports operational pilot)
         depends: policy_registry, policy_factory, v5_trace (RO), match_runner
```

---

## 10. Frozen-classic guard (preserved, in effect)

- Prod runtime must NOT import TrainV3/TrainV3.5 (`golden_trace.py` is training
  tooling only).
- Frozen-classic files BYTE-LOCKED: `classic_obs_v1.py`, `classic_actions_v1.py`,
  `classic_card_shape_v1.py`, `classic_rl_env.py` (all in `ai/train_v2/`). V5 work
  in `v5_*` files + NEW files. `classic_rl_env.py` NOT modified (mirror reward
  formula, don't edit). `reward_v5.py` consumed read-only (verifier finding 2c —
  it is already per-side; fix #1 is a trainer-side attribution, NOT a reward_v5
  edit). `warm_start_v5.py` consumed read-only. `v5_trace.py` consumed read-only
  (data-contract). `run_phase26*.py` NOT edited (the bug is documented, the fix is
  a NEW path in A4).
- `rust_ffi.py` is NOT frozen-classic — A4 constructors are ADDITIVE (no existing
  `from_trace_file`/`from_trace_files` modified).
- Worktree discipline: run cargo from
  `/Users/laveqox/Documents/ExtraArenaRaS/.claude/worktrees/glm-TrainV3.5Prep/TrainV3.5`,
  python from parent; never cd to original repo root.