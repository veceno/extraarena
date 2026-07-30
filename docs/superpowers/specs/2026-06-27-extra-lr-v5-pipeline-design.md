# Extra-LR-V5-Max — Training Pipeline Design

**Status:** design (pending implementation plan)
**Date:** 2026-06-27
**Scope:** V5-Max only. V5-Lite is **out of scope** (separate TrainV2-architecture effort with the new cards/mechanics).
**Author:** brainstormed 2026-06-26..27 with the user; grounded in repo audit (workflow `explore-v5-context`, 8 agents).

---

## 2026-07-05 Phase-A update

The first Phase-A seed path is updated: **do not use semi-synthetic ExtraRLHF
LLM/V4Max distillation for Phase A**. The attempted `llm-vs-v4max` lane produced weak
LLM play and burns model limits; V4-Max itself is also not a valid teacher for the new
ruleset/mechanics. Current Phase A starts with direct **random-heavy Rust ArenaEnv PPO**
using `train_v3.ppo_phaseA_config.build_phase_a_random_bootstrap_config()` and
`TrainV3.5/scripts/run_phaseA_random_bootstrap.py`. Phase A fresh-random bootstrap
does **not** V4-warm-start by default; the V4 warm-start loader remains an explicit
diagnostic/legacy opt-in, not the first bootstrap path.

Scope of this update is narrow: later **Block B league**, **Block C human-vs-preV5**,
offline replay/repair, and final tournament gates remain in the pipeline.

---

## 2026-07-17 auxiliary-model update

Decision D8 is superseded: the auxiliary-model lane is now a confirmed part of
the V5 programme. Four separately trained artifacts are planned:

- **ExtraLR Assembler V1** — ranks/builds a deck from an allowed card pool for a
  specified opponent deck;
- **ExtraLR CardOptimum V1** — ranks candidate cards by their expected utility
  in the current battle state;
- **ExtraLR Metronome V1** — models the distribution of human decision latency
  from the human-visible state and legal-action context;
- **ExtraLR TimeStamp V1 Mono / Duo** — predicts battle-duration distributions
  from one user deck (Mono) or an explicit deck-vs-deck matchup (Duo, optional).

Assembler and CardOptimum were present in the post-audit code as early
training-only scaffolding (`assembler` and `desirerer`). That scaffolding is not
yet a learned production model. `desirerer` is the legacy internal name for the
CardOptimum concept and must be migrated without silently changing old artifact
schemas. Metronome and TimeStamp are new tracks.

---

## 0. Context & root-cause recap

Two training stacks coexist (must not be confused):

- **TrainV2** (`ai/train_v2/`) — pure-Python MLX PPO, online-only, feedforward MLP, obs_dim=1456, `ActionConditionedPolicy` (`model_mlx.py:37`), 601-action candidate scorer, eval only vs `RandomLegalPolicy`/`EndTurnPolicy` (`train_ppo.py:1296`). V4-Max is a TrainV2-class model (`extra-lr-v4-max.onnx`, source `update_1190.npz`).
- **TrainV3** (`TrainV3/`) — Rust rollout kernel + FFI; the V5 policy lives here: split-encoder `v5_policy.py` (base OBS_V1_DIM=1456 + global V5_GLOBAL_DIM=32 + private + history HISTORY_DIM=20×144=2880) fused → 601-candidate scorer. Reward shaping in `reward_v5.py`. Acceptance in `run_v5_acceptance.py` + `gauntlet_v5.py` (exploit lanes `stall/anti_draw_greed/punish_empty_board/anti_hand_leak_overfit`, Rust rule agents). **Prod runtime must not import TrainV3** (`TrainV3/README.md:5`).

**Why V3/Phase-26 (“Phase A”) stalled** (all grounded in code):

1. `opponent_mix` 0.55 `legal_random` — majority opponent carries little strategic signal.
2. Reward contamination: `step_rewards = learner_rewards + opponent_rewards` (`run_phase26:490`).
3. Sparse terminal ±1.0 vs `max_turns=80` truncation to draw=0.
4. `entropy_coef=0.035`, `epochs=1`, `clip=0.16` — policy never commits to a decisive win.
5. Internal PPO loss/KL/entropy diverge from external bench (Phase33/34 docstrings) — promotion must be by **external** bench, not internal metrics.

**V5-Max fixes all five** by design (see Block A/B).

The user’s earlier “Phase A” proposal (95% random / 5% end_turn) replicated exactly these root causes; the redesign below supersedes it.

---

## 1. Key decisions (frozen)

| # | Decision |
|---|---|
| D1 | **RLHF data → offline-bridge.** Reconstruct `GameState` from `v5_trace.pre_state`; feed offline-PPO replay. No online-only assumption. |
| D2 | **Hybrid stack.** Rust kernel (TrainV3) for rollouts / self-play / league / acceptance. Python layer for offline-bridge + offline-PPO replay consumer. `rlhf_env` (port 8090) for fresh human-vs-bot collection. |
| D3 | **Phase A seed = random-heavy Rust ArenaEnv PPO bootstrap.** The previous fresh-pilot/BC seed idea is not the current first-phase path; LLM/V4Max semi-synthetic distillation is disabled for Phase A. |
| D4 | **V5-Lite out of scope.** |
| D5 (superseded) | ~~Adapt V4 → V4’ (fine-tune) as the 70-80% benchmark.~~ Superseded by D6/γ. |
| γ | **Frozen v1 codec.** `classic_obs_v1` (1456), `classic_actions_v1` (601), `classic_card_shape_v1` stay **byte-frozen** — V4-orig ONNX runs unchanged forever (pristine legacy, zero adapter). V5 uses a **new** `v5_card_shape_v1` + `encode_observation_v5` + a **parallel `mana_draw` binary head** (not a 602nd candidate). The V5 601-scorer + base-1456 path **warm-starts from V4-Max weights** (`update_1190.npz`). |
| D6 | **V4’ fine-tune removed.** V4 stays frozen. Strength benchmark = **best self-snapshot + human-QA** (+ decisive H2H vs V4-orig as an asymmetric legacy reference, not the bar). |
| D7 | **C1 (archive replay) dropped** — archive is pre-rebalance, no mana_draw. RLHF loop starts at C2 (fresh collection). |
| D8 (superseded) | ~~Sub-models deferred.~~ Superseded on 2026-07-17 by D14. |
| D9 | **C-phase mechanism = offline-PPO replay** on human traces (AWAC/CRR-style), not pure BC or DPO. |
| D10 | **Acceptance stays on Rust** (TrainV3 `gauntlet_v5`/`run_v5_acceptance`). Exploit-lane rule agents train preV5 in the Rust ArenaEnv — **not ported**. `model_benchmark` (Python) keeps its current legacy-H2H role only; no `V5Policy` adapter, no porting. |
| D11 | **Encoder = omniscient, always.** Server-side bot has full `GameState`; train and deploy both omniscient (own + opponent hand/deck per-card). No perspective/aux split, no InfoMode. Consequence: V5-Max is permanently server-side (non-transferable to perspective-restricted clients) and has an information edge over human opponents in RLHF-Arena (acceptable given the “субъективно сложнее” goal). |
| D12 | **History mandatory**, window 20 (matches the existing split-encoder stub `HISTORY_DIM=20×144`). |
| D13 | **Early-stop K=2** in the RLHF loop. |
| D14 | **Four auxiliary models confirmed.** Train and ship ExtraLR Assembler V1, CardOptimum V1, Metronome V1, and TimeStamp V1 Mono; TimeStamp Duo is optional. They are separate artifacts with separate datasets/gates and do not weaken the no-assist V5 acceptance lane. |

---

## 2. Architecture

```
Block -1  Freeze ruleset (rebalance + 7 cards + mechanic changes + mana_draw)
          + Rust ArenaEnv parity with frozen ruleset  ← verify/port (prerequisite)
Block 0   Foundation: v5_card_shape_v1 + encode_observation_v5 + mana_draw head
          + V4-warm-start loader + offline-bridge (+bugfixes in v5_* only; classic_* frozen)
Block A   Random-heavy Rust ArenaEnv PPO bootstrap (teacher-free, target 98%+ vs random)
          → A-gate → hand off to league
Block B   League (self-snapshots + V4-orig t-spectrum + exploit-lanes + tail) on Rust
          → external-bench promote → trend toward self-snapshot domination
Block C   C2 (deploy best-vs-people → collect 100-300 real battles; 500 optimistic)
          → C3 (offline-PPO replay, AWAC/CRR) → promote; early-stop K=2
Block D   League-2 (post-C consolidation)
Block E1  Tournament (post-D / post-C3 / post-B) → pick best (Rust gauntlet + human-QA) → ship

Aux track Assembler + CardOptimum: simulator/counterfactual datasets from B/C/D states
          Metronome + TimeStamp: human timing/duration labels from C2
          → independent train/eval/export gates; never required for no-assist V5
```

**Unified promotion mechanism = external-bench gate** (Rust `gauntlet_v5` + `run_v5_acceptance`), retargeted to **best self-snapshot** as the strength reference. Internal PPO loss/KL/entropy are monitoring only (D-lesson from Phase33/34).

**Three roles for V4-orig (frozen):** (1) warm-start source for V5’s 601-scorer (D6/γ); (2) legacy/asymmetric H2H reference in acceptance (decisive-win expected, not the bar); (3) a temperature-spectrum sparring lane in Phase B/D (argmax-heavy). V4-orig is blind to mana_draw + new mechanics in roles 2–3 — by design.

---

## 3. Blocks in detail

### Block -1 — Prerequisites
- **Freeze ruleset**: the card rebalance + 6 new cards (47–52) + mechanic changes + mana_draw must be locked before any V5 training starts.
- **Rust ArenaEnv parity**: verify (and port if needed) that `TrainV3` Rust ArenaEnv implements the frozen ruleset identically to Python `core/engine.py` — mana_draw action, 6 new cards (47–52), 5 new mechanic families (`aoe_silence`, `team_wide_shield`, `rebirth`, `crime_and_punishment`, `target_ally_max_hp_plus[_universal]`). Without this, V5 cannot train or be accepted under new rules on Rust.
- **rlhf_env codec sync**: rlhf_env stays in sync with the frozen ruleset/codec so collected traces encode consistently with training.

### Block 0 — Foundation
**0.1 V5 codec/encoder (new; `classic_*` frozen — do not touch):**
- `v5_card_shape_v1.py`: a forked card-shape with the bug fixes + new mechanic flags:
  - fix the flag/scalar **overlap** (current `classic_card_shape_v1` writes flags `14:48` then scalars `47:64` → index 47 collides, silently dropping one mechanic flag per card);
  - register the **5 new mechanic families** as flags;
  - fix the **magnitude regex** (`^aoe_damage_(\d+)` anchored misses `deathrattle_aoe_damage_2` → card 34 Крипер loses damage-2); prefix-agnostic match + a `deathrattle_aoe_damage` channel.
- `encode_observation_v5.py` (new): base-1456 (unchanged `classic_obs_v1` subset, warm-startable) ⊕ **history window 20** (read `state.action_history`/`state.history`, currently ignored) ⊕ global extras incl. `mana_draw_count_this_turn` channel ⊕ new-mechanic channels ⊕ **per-card own deck** (9 unique cards, not zone-summary) ⊕ **hand lifted to `HAND_CAP`** (current `_NUM_HAND=4` cap) ⊕ **omniscient opponent hand/deck per-card**.
- **parallel `mana_draw` binary head** (new) + its legal-mask (unset when `hand_full` / insufficient mana). mana_draw is **not** a 602nd candidate — the 601 candidate codec stays shared/frozen.
- **V4-warm-start loader**: init the V5 601-candidate scorer + base-1456 path from `update_1190.npz`; new heads/encoders (history, mana_draw, new-mechanic channels) train from scratch.

**0.2 Offline-bridge (`ai/train_v2/offline_dataset_loader.py`, new):**
- iterate `sessions/<gid>/catalog.json` + `battles/<bid>/v5/{meta.json,actions.jsonl}`; **skip orphans** (`meta.status != 'finished'`);
- reconstruct `GameState` from `pre_state` + catalog — **not from `rng_seed`** (`core/effects.py` uses global `random`, not byte-reproducible);
- produce V5-format `Transition`s (obs via `encode_observation_v5`; action = 601 candidate id + mana_draw-bit; **reward copied from the per-action reward-deltas in `actions.jsonl`** = `reward_v5.py` formulas → offline shape matches online; `done`/`truncated` from row metadata incl. synthetic surrender row; `value`/`log_prob` from the current policy);
- **human rows** (`decision_source='human'`) feed **offline-PPO replay (C3)**. (BC-seed in Block A uses the **fresh pilot**, not the archive — D7.)

### Block A — Seed
- **A.random-bootstrap (current)**: train directly from fresh init in Rust ArenaEnv with
  `TrainV3.5/scripts/run_phaseA_random_bootstrap.py`. Opponent mix:
  `legal_random 0.70, end_turn 0.05, greedy_face 0.10, face_rush 0.05,
  board_control 0.05, greedy_trade 0.05`. No LLM teacher, no V4Max teacher, no
  V4Max warm-start by default, no previous-self snapshot in the first bootstrap.
  Target: **98%+ vs random** before handing off to broader league pressure.
- **A.pilot/A.BC (deferred/optional)**: human pilot and BC are not the default first
  phase anymore. Human data belongs primarily to Block C (`human-vs-preV5`) once the
  preV5 candidate exists.
- **A.PPO fixes retained:**

| Root cause | Fix |
|---|---|
| `step_rewards = learner + opponent` | `learner_rewards` only (own-side credit) |
| `max_turns=80` truncation | raise to ≥120 (or decisive-state early-end) |
| `entropy_coef=0.035` | 0.01; explore via action-mask |
| `epochs=1` | 4–10 |
| 0.55 random majority | graduated mix (below), exploit-lanes present |

  The older graduated `opponent_mix` remains a later/harder A/B curriculum option:
  `legal_random 0.10, end_turn 0.05, greedy_face 0.10, face_rush 0.10,
  board_control 0.10, greedy_trade 0.10, stall 0.10, anti_draw_greed 0.10,
  self_prev 0.10, v4-orig-argmax 0.15`.
- Promotion by **external bench only** (Rust gauntlet, retargeted). Second-start oversampling (human-as-p2 episodes + targeted init).

**A-gate (exit Phase A):** `no_assist ≥ 0.55`, `exploit_resistance ≥ 0.50`, `mana_draw usage ∈ [0.5×, 1.5×]` human baseline, external H2H (vs best self-snapshot) trending up ≥ N snapshots.

### Block B — League (Rust ArenaEnv)
- online-PPO self-play + league, continuing A’s fixed hyperparams + decontaminated reward.
- **Composition**: self-snapshots (snapshot every ~2000 PPO updates; pool ~6 + 2 anchors [seed, best-ever]); **V4-orig temperature spectrum** `v4-orig-argmax 0.40 / v4-orig-t07 0.20 / v4-orig-t12 0.15` (one frozen model, three identities, argmax-heavy); exploit-lanes low-weight continuous `stall/anti_draw_greed/punish_empty_board` ~0.05 each; tail `greedy_face 0.03, legal_random 0.01, end_turn 0.01`.
- **Curriculum**: oversample the lane the learner is losing to.
- **Second-start parity** enforced continuously (`p1_p2_score_gap ≤ 0.12` acceptance; oversample p2-init on breach).
- **Promote** iff external bench (H2H vs best self-snapshot + gauntlet + mana_draw band + p1_p2 gap) improves monotonically ≥ N_snap.
- **Exit → C2** when H2H vs best self-snapshot **plateaus** (no gain K_snap) below the dominance target.

### Block C — RLHF loop (C2 → C3, early-stop K=2)
- **C2**: deploy current best V5 vs humans in rlhf_env (synced to frozen
  ruleset/codec). Plan for **100–300 real human battles**; 500 is the optimistic
  cap. Supplement the main-policy state coverage with semi-synthetic battles,
  but never relabel synthetic latency as human latency. `decision_source='human'`
  rows retain full omniscient pre/post for the V5 replay bridge and a separate
  human-visible projection for Metronome.
- **C3**: offline-bridge builds V5 Transitions from fresh human rows → **AWAC/CRR offline-PPO replay** (advantage-weighted regression; no `π_behavior` needed; importance ratios PPO-clipped; value bootstrapped from current value-head; reward copied per-action). External-bench promote.
- **Early-stop**: no external-bench gain this iteration → `stall++`; **K=2 → exit to D**.
- Archive replay (the old C1) is **dropped** (D7).

### Block D — League-2 (consolidation)
Short self-play among {best post-C checkpoint, post-B/seed anchors, V4-orig spectrum, exploit-lanes} to settle post-C and prevent overfit to the last replay batch. Promotion by external bench.

### Block E1 — Tournament + ship
- Candidates: post-D, post-C3 (best C-iteration), post-B (if C added nothing).
- **Eval** on the **Rust ArenaEnv gauntlet** (retargeted to best self-snapshot) + a **human-QA panel** in rlhf_env (operationalizes “субъективно сложнее для человека”).
- **Final acceptance thresholds** (adapt `run_v5_acceptance`/`gauntlet_v5`):

Two distinct notions are measured separately:

- **Absolute strength anchor** — vs frozen V4-orig (asymmetric: V5 advantaged, since V4-orig is blind to mana_draw + new mechanics). This carries the user’s original “70-80% vs V4-Max” intent, retargeted to V4-orig.
- **Progression signal** — vs the immediately-prior best self-snapshot. A strong candidate wins only mildly (~0.52–0.55) over its own previous best (beating your own best by 70% is ill-defined); monotone improvement here, detected by the plateau rule, is the promotion driver.

| Metric | Final |
|---|---|
| H2H vs V4-orig (absolute strength, asymmetric) | **≥ 0.70** (user’s original 70-80% band) |
| H2H vs Random | ~1.0 (sanity; user’s “100% vs Random”) |
| H2H vs end_turn | ~1.0 (sanity) |
| H2H vs best self-snapshot (progression) | monotone improvement, ≥ ~0.52–0.55 over prior best |
| no_bonus p1/p2/second (vs best self-snapshot) | ≥ 0.70 each |
| no_assist_min_score_rate | ≥ 0.55 → calibrate up |
| exploit_resistance_min_score_rate | ≥ 0.50 |
| p1_p2_max_score_gap | ≤ 0.12 |
| mana_draw usage-rate | ∈ [0.5×, 1.5×] human baseline |
| min_entropy / max_abs_kl | ≥ 0.70 / ≤ 0.12 |
| min_e2e_tps | ≥ 12000 |
| human-QA difficulty | subjective bar (panel) — “субъективно сложнее для человека” |

- **Ship**: export best → ONNX + sidecar; register via `_derive_kind_from_sidecar`; deploy to prod arena (argmax); marker `extra-lr-v5-max`. Snapshots/temperature variants feed the next iteration’s league pool.

### Auxiliary-model lane — confirmed deliverables

These are four independent small models, not extra heads whose success is
allowed to mask a weak V5 policy. Each artifact gets its own immutable input
schema, dataset manifest, catalog/ruleset hash, train/validation split, export
parity check, and acceptance report.

| Model | Contract | Authoritative labels | Primary acceptance gate |
|---|---|---|---|
| **ExtraLR Assembler V1** | Input: allowed card pool (including levels/variants), opponent deck, ruleset/catalog identity. Output: ranked legal deck candidates and the selected deck with an expected matchup score. | Controlled simulator matchup matrix. Evaluate each candidate with paired seeds, both starting-player directions, and enough repeats to attach uncertainty to win rate. | Low held-out ranking regret and a statistically positive win-rate lift of the selected deck over random/search baselines on unseen pools and opponent decks. |
| **ExtraLR CardOptimum V1** | Input: pinned information-mode battle state plus candidate-card pool. Output: per-card expected utility/ranking, not an unchecked forced draw. | Counterfactual branching from the same state: evaluate every candidate under matched RNG/continuation policies and train on expected return/advantage. Human or Minimax traces may supply hard states, but the observed top-deck card is not itself a valid optimality label. | Top-k ranking/regret on held-out states and positive downstream value versus heuristic draw selection. |
| **ExtraLR Metronome V1** | Input: human-visible state, legal-action set/complexity, turn/action context. Output: a latency distribution (quantiles or log-time distribution) from which runtime samples a delay. It never chooses the action. | `human_decision_time_ms` from accepted human decisions only. Exclude bot/LLM rows, reconnect/background idle, loading time, and censored/abandoned turns; retain censor metadata instead of treating timeouts as normal decisions. | Held-out log-time error plus calibrated p50/p90 coverage by action type and complexity bucket. Runtime clamps only safety tails, not every decision to a hard-coded 3–6 s band. |
| **ExtraLR TimeStamp V1 Mono / Duo** | Mono input: user deck plus ruleset, starting-player and opponent-population/model context. Duo input: both decks plus the same context. Output: expected duration and uncertainty/quantiles in seconds and turns. | Completed-battle duration/turn labels. Human battles calibrate wall-clock time; large simulator/semi-synthetic corpora provide matchup coverage. Abandoned and truncated battles are censored, not ordinary completions. | MAE/median-AE and quantile coverage on unseen decks. Duo must also pass a held-out unseen deck-pair split; otherwise only Mono ships. |

**Current-code migration rule.** `extra-sublr-assembler-v1` is the precursor
schema for ExtraLR Assembler V1. `extra-sublr-desirerer-v1` is the precursor
schema for ExtraLR CardOptimum V1. Existing manifests remain readable under
their legacy labels, but new artifacts use the confirmed public names and a
versioned migration field. The current heuristic `DeckMatchupEvaluator` and
`DrawDesirerer` are baselines only. In particular, the current desirerer label
(`candidate_card_id = deck_ids[0]` with immediate step reward) is unsuitable as
the final CardOptimum training target and must be replaced by counterfactual
candidate evaluation.

**Human-data sufficiency rule.** Metronome trains on decision rows, so 100–300
battles can yield thousands of examples, but splits should be by pseudonymous
participant/session rather than random rows. TimeStamp trains on battle rows;
100–300 real battles are calibration data, not enough deck-pair coverage by
themselves. Mono/Duo therefore use simulation pretraining followed by human
calibration. The collector must persist a non-PII participant/session key;
the fixed arena-side `user_id=1000` is not sufficient for leakage-safe splits.

**Grokking-oriented evaluation.** For Assembler, CardOptimum, and TimeStamp,
keep a compositional holdout of unseen card-pool/deck/deck-pair combinations
and continue training beyond interpolation while monitoring held-out regret.
Any delayed generalization is evidence only if it repeats across seeds and is
not caused by leakage between near-duplicate matchups.

---

## 4. Data flow

- **Online**: Rust ArenaEnv rollouts → online PPO → snapshot → Rust gauntlet eval → promote.
- **Offline**: `v5_trace` (omniscient; `rlhf_env/components/v5_trace.py`) → Python offline-bridge (reconstruct `GameState` from `pre_state`) → V5 `Transition`s → AWAC/CRR offline-PPO replay → promoted checkpoint.
- **Human**: rlhf_env deploy best-vs-people → `v5_trace` rows `decision_source='human'` → bridge → C3 replay.
- **Assembler**: legal card pools + controlled deck-vs-deck simulator grid → aggregate matchup rows with uncertainty → Assembler train/eval.
- **CardOptimum**: hard states from B/C/D + matched counterfactual candidate branches → card-utility ranking rows → CardOptimum train/eval.
- **Metronome**: accepted human rows + `human_decision_time_ms` + human-visible state projection → latency model. Synthetic/LLM timings are forbidden as human labels.
- **TimeStamp**: completed battles → Mono/Duo duration rows; simulator pretraining → real-human calibration.
- mana_draw-capable human and model traces may train the main V5 policy, but
  human traces remain the authoritative source for human timing behaviour.

---

## 5. Error handling / integrity
- **Skip orphan battles** (`meta.status != 'finished'`).
- **Reconstruct `GameState` from `pre_state`**, never from `rng_seed` (non-reproducible).
- **ONNX fallback guard**: `policy_factory` already raises `RuntimeError` if a model silently falls back to rule-based (V4-orig unchanged).
- **Export-parity tolerance**: the existing test uses `LOGIT_TOL=1e-3` (Metal-vs-CPU float32 drift); a real V5 export bug <1e-3 slips — **tighten** V5 export-validation tolerance.
- **Warm-start parity**: assert V5 601-scorer + base-1456 path forward-pass on a frozen-obs subset is bit-close to V4-Max immediately after load (sanity), before any training.
- **rlhf_env codec sync**: traces are only consumable if rlhf_env encodes under the frozen ruleset — enforced by Block -1.
- **Distribution shift**: archive dropped (stale); pilot + C2 are post-rebalance by construction.

---

## 6. Testing (before training starts)
- `v5_card_shape_v1`: overlap fixed (no index collision); 5 new mechanic flags present; magnitude regex matches `deathrattle_aoe_damage_2` (card 34 damage-2 recovered); cards 47–52 encode correctly.
- `encode_observation_v5`: history window 20 reads `state.action_history`; hand up to `HAND_CAP`; per-card own deck (9); global `mana_draw_count` channel; omniscient both hands/decks.
- **Bridge round-trip**: record a battle via `v5_trace`, reconstruct `GameState` from `pre_state`, re-encode, assert the reconstructed post-state matches the next row’s `pre_state`; orphans skipped; surrender synthetic row handled.
- **mana_draw parallel head**: legal-mask correct (hand_full / insufficient mana unset); selection combines 601-candidate-best vs mana_draw logit correctly.
- **Rust ArenaEnv parity**: mana_draw + 6 new cards (47–52) + 5 mechanic families behave identically to Python `core/engine` (parity tests).
- **V4 warm-start**: load `update_1190.npz` into V5 601-scorer + base path; forward-pass parity on base-1456 subset.
- **V5 ONNX↔MLX export** within tightened tolerance.
- **Assembler:** paired-seed/starting-player matchup labels, legal-deck
  constraints, unseen-pool and unseen-opponent-deck evaluation.
- **CardOptimum:** all candidate cards branched from an identical pre-state;
  no deck-order label leakage; top-k regret reproducible across continuation
  seeds.
- **Metronome:** human-only timing labels, human-visible feature projection,
  reconnect/background censoring, calibrated sampled-delay distribution.
- **TimeStamp:** completed/censored separation, Mono population context,
  optional Duo unseen-pair split, seconds/turns export parity.

---

## 7. Open questions (to resolve during implementation planning)
- Rust ArenaEnv parity gap: does the Rust kernel already implement mana_draw + 6 new cards (47–52) + 5 mechanic families, or must it be ported? (Block -1 verification.)
- `HAND_CAP` exact value (ruleset) — confirm before lifting `_NUM_HAND`.
- V4-warm-start transfer fidelity: does V5’s fused architecture allow a clean 601-scorer + base-1456 weight transfer, or only partial? (Affects how much V5 inherits.)
- mana_draw usage baseline: define the human-baseline measurement precisely (over pilot battles).
- Whether V4-orig as a **blind** Phase-B lane helps or biases the learner (V4-orig never draws → learner must not over-fit to “opponent never draws”); mitigate via self-snapshot prevalence.
