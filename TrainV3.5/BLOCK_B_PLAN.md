# BLOCK B — League (Rust ArenaEnv) (FINAL PLAN)

V5-Max pipeline position: Block -1 done → Block 0 done → Block A in-worktree
COMPLETE (A1-A5, commits 152d2350..07896c5a) → **Block B (this plan)** → C → D → E1.

This plan is grounded in the CloseQ synthesis (spec-grounding design.md:116-122 +
code-surface map of A3/A4/A5/league_v5/gauntlet_v5/rust_trainer) and merges the
adversarial verify findings. Block B **builds on** A3 (`ppo_phaseA_config.py`),
A4 (`rust_live_self_play.py`), A5 (`a_gate.py`) — it does NOT reinvent the
live-self-play runner, the gate, the promotion selector, or the PPO config. Block
B ADDS the **league layer**: a self-snapshot pool, a V4-orig temperature spectrum,
curriculum oversampling, a continuous second-start parity loop, a Block-B
external-bench promotion gate (extending A5), and an exit-to-C2 plateau detector,
all orchestrated by a multi-update live league driver over A4's one-update
`run_live_self_play_update`.

---

## 1. Operational vs in-worktree split (up front)

The central discipline (mirrors Block A §1): **what can be built + synthetic-tested
now (in-worktree)** vs **what the user must run live (operational)**.

### IN-WORKTREE-BUILDABLE (I do these now, synthetic-testable, no MLX/Rust required
for unit tests via the A4 FakeWorker + A5 `_FakeGameRunner` injection patterns)

1. **B1 snapshot_pool.py** — bounded pool (~6 rolling + 2 anchors [seed, best-ever])
   of V5 checkpoints with metadata (update#, H2H-vs-best, path, p1_p2 gap), FIFO
   eviction of non-anchors, anchor immutability, load-as-`SelfPrevOpponent`
   `select_fn` + league-opponent. ZERO existing infra (A5 tracks a single
   `current_best`, `rust_trainer._save_checkpoint` is linear cadence only).
   Synthetic-testable with fake checkpoints + a `_FakeCheckpointStore`.
2. **B2 v4_orig_temp_spectrum.py** — three policy-opponents from ONE frozen V4
   ONNX model: `v4-orig-argmax` (temp=0, weight 0.40), `v4-orig-t07` (temp=0.7,
   0.20), `v4-orig-t12` (temp=1.2, 0.15). Wraps `ai/train_v2/onnx_policy.py:16
   OnnxActionPolicy` which ALREADY exposes a `temperature` param (validated >0 at
   `:30-31`; `scaled = mlogits / self._temperature` at `:93`; `mode='argmax'|'sample'`
   at `:21`) — Block B wraps, does NOT port. **Adapter required:** `OnnxActionPolicy
   .select_action(env, player_id)` is env-based (reads legal actions from a Python
   env via `build_action_mask`), whereas A4's policy-opponent loop calls
   `PolicyOpponent.select(env_idx, ctx: OpponentCtx)` with packed legal_action_ids
   arrays (`rust_live_self_play.py:678-695`); B2 builds a thin adapter from
   `OpponentCtx`'s packed arrays to `OnnxActionPolicy`'s forward (mlogits), NOT a
   pure pass-through wrap. Adds canonical-name aliases (`v4-orig-t07`/`v4-orig-t12`
   NOT in `league_v5.V5_OPPONENT_KINDS` — `punish_empty_board` IS already in
   `V5_OPPONENT_KINDS` via `*EXPLOIT_AGENT_KINDS`, so it needs NO parse alias; see
   B3 for the alias vs dispatch split). NOTE: the stale path
   `ai/model_benchmark/policies.py:69 ActionConditionedOnnxPolicy` in
   `handoff.md:117` is a gitignored/absent directory (`.gitignore:24
   ai/model_benchmark/`) with zero definitions anywhere in the repo (grep-confirmed)
   — corrected here so it does NOT re-propagate to Block C/D. Synthetic-testable: a
   `_FakeOnnxPolicy` exercising the temperature path
   (scaled = mlogits / self._temperature at `onnx_policy.py:93`) without a real ONNX.
3. **B3 block_b_opponent_mix.py** — the Block-B mix composition
   (self-snapshots-from-pool + V4-orig spectrum 0.40/0.20/0.15 + exploit-lanes
   stall/anti_draw_greed/punish_empty_board ~0.05 each + tail
   greedy_face 0.03 / legal_random 0.01 / end_turn 0.01), the
   `punish_empty_board` (Rust code 5) dispatch enable, tail/exploit reweighting
   from Phase-A values, and the alias extension so `v4-orig-t07`/`v4-orig-t12`
   parse. Validates via `league_v5.parse_v5_opponent_mix`. Synthetic-testable.
4. **B4 curriculum.py** — per-lane learner loss tracker (from A4
   `LiveRolloutBatch.dispatch_log` + per-identity outcome aggregation) +
   adaptive reweight policy (oversample the lane the learner is losing to,
   proportional to the loss margin, capped ~25%/update per D-B8) + injection
   hook into A4 `sample_opponent_identities`. A4's sampler is STATIC; B4 makes it
   DYNAMIC. Synthetic-testable with fabricated dispatch logs.
5. **B5 second_start_parity.py** — continuous p1/p2 score-gap tracker over a
   rolling window of league games + feedback into A4 `sample_learner_sides` +
   `p1_p2_score_gap <= 0.12` acceptance + p2-oversample-on-breach. The MECHANISM
   already exists (A3 `second_start_oversampling_scheme` + A4 `sample_learner_sides`
   accept `p1_score_rate`/`p2_score_rate` but never measure/close the loop); B5
   CLOSES THE LOOP. Builds NEW measurement (A5 pattern — the dead
   `gauntlet_v5.p1_p2_max_score_gap=0.12` field has zero consumers in
   `gauntlet_v5.py` itself; the same-named key in
   `run_phase1_runtime_acceptance_bench.py:50,639` is a separate local dict, NOT
   the field). Plays the candidate from BOTH p1 and p2 (or records
   `candidate_side` via a `BlockBGameResult` composing the frozen `GameResult`) so
   p1/p2 score rates are separately measurable. Synthetic-testable.
6. **B6 block_b_gate.py** — Block-B external-bench promotion gate. EXTENDS A5
   (does NOT rewrite): composes A5's mana_draw band + H2H-vs-best + gauntlet,
   ADDS `p1_p2_gap <= 0.12` (from B5) + **monotonic improvement >= N_snap** of the
   full external-bench aggregate (H2H vs best self-snapshot + gauntlet +
   mana_draw band + p1_p2 gap). Distinct from the A-gate 4-criteria (which are
   Phase-A EXIT, NOT league promotion — open_question #11): Block B does NOT
   re-apply `no_assist`/`exploit_resistance` thresholds. Promotion-by-loss guard
   inherited from A5 (internal ppo_loss/approx_kl/entropy NEVER read).
   Synthetic-testable with fabricated external-bench series.
7. **B7 exit_to_c2.py** — plateau detector: "H2H vs best self-snapshot plateaus
   (no gain K_snap) below the dominance target" (design.md:122 verbatim —
   genuinely ambiguous, see D-B3 3a). The INVERSE of A5's `check_h2h_trending`
   (which detects trending UP). `detect_h2h_plateau(h2h_scores, K_snap, min_gain,
   dominance_target)`. Emits the exit→C2 signal. **Default reading (D-B3 3a =
   "below", spec-literal):** exit when plateau AND candidate BELOW dominance target
   (still weak — C2 human data breaks the plateau); tests FLIP if user picks
   "at/above" (dominant). Synthetic-testable.
8. **B8 block_b_league_driver.py** — the multi-update live league driver. Composes
   A4 `run_live_self_play_update` (ONE update) into a loop: snapshot every ~2000
   updates (B1 pool-add on top of `rust_trainer._save_checkpoint`), curriculum
   reweight each update (B4), parity feedback each update (B5), promotion call
   per snapshot (B6), plateau/exit check per snapshot (B7). Reuses the trace-pool
   `train_rust_ppo_trace_files` loop scaffolding (metrics list, `checkpoint_every`,
   league manifest) as a STRUCTURAL TEMPLATE only — B8 drives the LIVE path via
   A4, not the trace-pool replay. Synthetic-testable with A4 `FakeWorker` +
   `_FakeGameRunner` (no MLX/Rust for unit tests; real-FFI smoke gated).

### OPERATIONAL-NEEDS-USER (user runs these live)

1. **Block B league training run** — execute B8 on compute (needs A-gate PASS so
   the A-gate-passed policy is the inaugural best-self-snapshot / seed anchor;
   needs D-B11 compute). CANNOT be done autonomously in a worktree.
2. **External-bench gauntlet runs** — B6/B7 read real H2H-vs-best-self-snapshot +
   gauntlet + mana_draw band + p1_p2 gap measured by playing real Rust ArenaEnv
   games via the production `GameRunner` adapter (A5 Protocol, wired to A4). The
   measurement CODE is in-worktree; the RUN needs the trained snapshots (needs the
   league run).
3. **Q4 mana_draw baseline B** — A5 built the measurement; the real-data B is
   operational (needs A.pilot, which precedes Block B). Block B consumes B
   read-only for the band gate.
4. **Exit→C2 decision** — when B7 detects plateau below dominance target, the user
   takes the Block-B-best checkpoint into C2 (deploy vs humans, collect ~3-5k
   fresh preV5-vs-human battles, design.md:125).

**Execution gate (in effect, like A's D-A3):** Block B can be PLANNED now (like A
was planned before the pilot ran), but the in-worktree components EXECUTE
synthetic-testable now while the league RUN is gated on A-gate PASS (D-B12). The
in-worktree code is built regardless of the A-gate status.

---

## 2. User decisions (genuine — present these to the user via AskUserQuestion
AFTER this plan is verified)

These are the decisions only the user can make. Design defaults I can set are
listed per-component in §3. The recommendations are spec-literal unless flagged.

| ID | Question | Default | Options |
|----|----------|---------|---------|
| **D-B1** | `N_snap` — the promotion monotone-improvement window (design.md:121 "improves monotonically >= N_snap"). Spec names but does NOT pin it. How many snapshots of monotone external-bench improvement before a candidate promotes? | 5 (reuse A5 `DEFAULT_H2H_MIN_SNAPSHOTS`, cross-block consistency) | 3 (aggressive) / 5 (reuse A-gate) / >=7 (conservative) |
| **D-B2** | `K_snap` — the plateau no-gain window for exit→C2 (design.md:122 "no gain K_snap"). Spec names but does NOT pin it. (The plateau gain-tolerance `min_gain` — the H2H gain below which a snapshot counts as "no gain" in `detect_h2h_plateau(h2h_scores, K_snap, min_gain, dominance_target)` — is an IMPLEMENTER design default, NOT a user decision: default `min_gain = 0.01` H2H score-rate, small enough that genuine improvement clears it while noise does not. Surfaced here for visibility; not a separate user decision.) | 2×N_snap (~10) — patient enough to distinguish a true plateau from a noisy dip, bounded so B does not run forever | K_snap = N_snap (~5, exit as soon as one promotion window stalls) / 2×N_snap (~10) / >=3×N_snap (~15, very patient) |
| **D-B3** | **THE Block-B termination signal — TWO sub-questions (the below-vs-above semantic is PRIMARY and load-bearing).** design.md:122 verbatim: "Exit → C2 when H2H vs best self-snapshot plateaus (no gain K_snap) below the dominance target." **(3a, PRIMARY) Below-vs-above:** does Block B exit→C2 when it plateaus while STILL WEAK (H2H-vs-best BELOW the dominance target → self-play exhausted before reaching dominance → C2 human data is the next lever to break the plateau) — the LITERAL reading of "below" — OR when it plateaus while DOMINANT (H2H-vs-best AT/ABOVE the dominance target → self-play reached dominance and stalled → C2 to collect human data for the next phase)? These are OPPOSITE exit conditions; the plan's default + B7 tests follow the LITERAL "below" (still-weak) reading and FLIP if the user picks "at/above." **(3b) Which dominance notion:** progression ~0.52-0.55 (design.md:148) / absolute vs V4-orig >=0.70 (design.md:145, Block E1 final-acceptance band) / both. | Default 3a = **below (still-weak) — spec-literal**: plateau AND candidate BELOW dominance target → exit (self-play stuck below dominance; C2 human data breaks the plateau). Default 3b = progression ~0.52-0.55 (design.md:148 "a strong candidate wins only mildly over its own previous best"; design.md:141 "the plateau rule is the promotion driver"). | 3a: Below (still-weak) — spec-literal / At-or-above (dominant) — colloquial reading /  3b: Progression (~0.52-0.55) / Absolute vs V4-orig (>=0.70) / Both (plateau AND progression>=0.55 AND vs-V4-orig>=0.70) |
| **D-B4** | Snapshot cadence + pool size + eviction (design.md:118, "~" approximate). | ~2000 updates / pool ~6 + 2 anchors [seed, best-ever] / FIFO non-anchors, never evict anchors — spec-literal | Spec default (2000/6+2/FIFO) / Smaller-faster (1000/4+2) / Larger-stabler (4000/8+2) |
| **D-B5** | **Q5 V4-orig blind-lane bias — THE dominant spec-internal conflict** (design.md:198, handoff.md:81). The frozen V4-orig spectrum weights total 0.75 (argmax 0.40 + t07 0.20 + t12 0.15), yet Q5 says "keep V4-orig lane weight MODEST + self-snapshot prevalence HIGH." 0.75 is not modest. The self-snapshot weight is also NOT pinned (if V4-orig 0.75 + exploit 0.15 + tail 0.05 = 0.95, self-snapshots get 0.05 — contradicts Q5). | Hybrid: keep V4-orig spectrum weights FROZEN (spec-literal) but grow self-snapshot prevalence as the pool fills (self-snapshot / `v5_snapshot` weight rises when more snapshots exist), with a mana_draw-collapse monitor that triggers reweighting if the learner's mana_draw usage drops out of band vs V4-orig lanes | Accept spec weights verbatim (V4-orig 0.75) + MONITOR collapse only / Reweight V4-orig DOWN (e.g. 0.20/0.10/0.075) + boost self-snapshot to ~0.40 (honors Q5, deviates from frozen weights) / **Hybrid** (freeze weights + grow self-snapshot prevalence + collapse monitor) |
| **D-B6** | V4-orig temperature values for t07/t12 (design.md:118). Spec names identities with weights 0.20/0.15 but does NOT explicitly state the temperatures. (Low-stakes: the label names `t07`/`t12` encode 0.7/1.2 by convention, so the default is label-dictated; this is a borderline implementer-default the user confirms only if they want different values — `OnnxActionPolicy` at `ai/train_v2/onnx_policy.py:16` supports temperature natively at `:22`.) | t07 = temp 0.7, t12 = temp 1.2 (label convention) | Yes (0.7 / 1.2) / t07-t12 are just labels, actual temps differ (specify) |
| **D-B7** | Does Block B freeze A's PPO hyperparams EXACTLY (entropy=0.01, epochs=6, max_turns=120, learner_only_reward, decisive_early_end) per "continuing A's fixed hyperparams" (design.md:117), or retune any for the league phase? | Freeze all A hyperparams verbatim — spec-literal (any retune risks re-introducing the 5 root causes A fixed) | Freeze all verbatim / Freeze except max_turns (raise for longer league games) / Freeze except entropy_coef (lower for late-league exploitation) / Retune epochs/minibatch |
| **D-B8** | Curriculum oversample-losing-lane reweight aggressiveness (design.md:119). How fast does the mix reweight toward the lane the learner is losing to each update? | Adaptive (shift proportional to the per-lane loss margin, capped ~25%/update) — matches "oversample the lane the learner is losing to" without destabilizing the mix | Gentle (max 10%/update) / Moderate (max 25%) / Aggressive (cap at spec weight ceiling, full shift) / Adaptive (proportional, capped ~25%) |
| **D-B9** | p1_p2 parity enforcement. The `gauntlet_v5.V5GauntletConfig.p1_p2_max_score_gap=0.12` field (`gauntlet_v5.py:43`) has ZERO consumers in `gauntlet_v5.py` itself (the same-named key in `TrainV3.5/scripts/run_phase1_runtime_acceptance_bench.py:50,639` is a SEPARATE local THRESHOLDS dict, NOT an import of the `gauntlet_v5` field — so the field is genuinely dead). Build a NEW live measurement + feedback loop, or wire the existing config field? | Build NEW live measurement (A5 pattern — A5 established build-new-over-wire-dead for the sibling dead fields) | Build new (A5 pattern) / Wire the existing dead field + add a consumer / Build new but write result into V5GauntletConfig for observability |
| **D-B10** | `punish_empty_board` (Rust rule code 5) is in the Block-B exploit-lane continuous mix (~0.05, design.md:118) but was DOCUMENTED-but-EXCLUDED from Phase A (A4 `RULE_AGENT_CODES:143` comment; it exists at `worker.rs:1258`). Confirm enabling it? | Enable (code 5) — spec design.md:118 explicitly lists it; exists in worker.rs; uncommenting is additive, zero Rust change. `anti_hand_leak_overfit` (code 7) is NOT in the spec mix — leave excluded. | Enable punish_empty_board (code 5) / Keep excluded (use only stall/anti_draw_greed) / Enable + also enable anti_hand_leak_overfit (code 7) |
| **D-B11** | Block B compute budget / wall-clock for the league run (operational, like D-A4). | Medium (~20-40k updates, ~2-4 GPU-days, ~10-20 snapshots — enough for the pool to fill at 2000-cadence + a plateau window) | Small (~10k, hours-1 GPU-day, ~5 snapshots) / Medium (~20-40k) / Large (>40k, multi-GPU-day, full pool + plateau detection) |
| **D-B12** | Can the user run the Block B league run NOW (operational gate, like D-A3)? Block B execution requires A-gate PASS (the A-gate-passed policy is the inaugural best-self-snapshot / seed anchor). Is the A-gate passed / are A.pilot+A.BC+A.PPO+A-gate done? | Not yet — Block B is PLANNED now, executes after A-gate PASS (mirrors how Block A was planned before the pilot ran) | Yes (A-gate passed, league can start now) / Not yet (A operational steps still in progress) / Yes but on staging/dev first (validate the league driver on dev before the full run) |

**DECISIONS CONFIRMED (user, 2026-07-01):** D-B3 = **Below + progression**
(spec-literal "below": exit→C2 when plateau AND `h2h_vs_best < ~0.55` — self-play
stalled while still weak → C2 human data breaks the plateau; B7 tests configured
for this reading). D-B5 = **Hybrid** (V4-orig 0.40/0.20/0.15 frozen; self-snapshot
share = residual grown as the pool fills; mana_draw-collapse monitor triggers
reweight on out-of-band vs V4-orig lanes). D-B8 = **Adaptive, cap 25%/update**
(curriculum shift ∝ per-lane loss margin, capped). D-B12 = **Not yet** (B1-B8
built in-worktree now with synthetic tests; the league RUN is gated on A-gate
PASS — A.pilot → A.BC → A.PPO → A-gate, user-run). The remaining decisions
proceed on their recommended (spec-literal) defaults: D-B1 N_snap=5, D-B2
K_snap≈2×N_snap (~10), D-B4 ~2000 cadence / pool ~6 + 2 anchors / FIFO
non-anchors, D-B6 t07=0.7 / t12=1.2, D-B7 freeze all A hyperparams verbatim,
D-B9 build NEW live p1/p2 measurement (A5 pattern, do not wire the dead field),
D-B10 enable `punish_empty_board` (Rust code 5, additive uncomment
`rust_live_self_play.py:143`), D-B11 medium compute (~20-40k updates) for the
operational run.

**Why D-B5 is THE user decision (do not resolve as an implementer choice):** the
spec internally contradicts itself — design.md:118 freezes V4-orig at 0.75 total
weight, while design.md:198/handoff.md:81 (Q5) says "keep V4-orig lane weight
MODEST + self-snapshot prevalence HIGH." 0.75 is not modest. Compounding this, the
self-snapshot weight is NOT pinned by the spec (design.md:118 lists self-snapshots
FIRST but gives no weight; only V4-orig/exploit/tail weights are frozen). A naive
sum (V4-orig 0.75 + exploit 0.15 + tail 0.05 = 0.95) leaves self-snapshots 0.05 —
directly contradicting Q5. This is the single biggest spec-internal tension Block B
must resolve with the user; the plan surfaces it prominently and defaults to the
Hybrid (freeze the frozen weights, grow self-snapshot prevalence as the pool fills,
add a mana_draw-collapse monitor) so neither the frozen weights nor Q5 is
pre-emptively violated.

**Frozen / non-decisions (do NOT re-litigate):** A's 5 root-cause fixes (carried
verbatim per D-B7); V4-orig = ONE frozen model (V4 byte-frozen, gamma frozen v1
codec); the frozen-classic set (`classic_obs_v1`/`classic_actions_v1`/
`classic_card_shape_v1`/`classic_rl_env.py`/`reward_v5.py` byte-locked); A-gate
thresholds 0.55/0.50/[0.5×,1.5×] (Phase-A exit, NOT league promotion); p1_p2 gap
0.12 (spec-pinned).

**"Decontaminated reward" (design.md:117, undefined term):** the plan defines it
operationally as **A's `reward_v5` formulas with A3 learner-only attribution,
unchanged** — already in A3/A4 (`reward_attribution` zeroes opponent-actor steps;
`reward_v5.py` is already per-side at `:40`, consumed read-only). No new code.

---

## 3. In-worktree components (dependency-ordered)

### B1 — `snapshot_pool.py` — self-snapshot pool manager (NEW — zero existing infra)

**File:** `TrainV3.5/python/train_v3/snapshot_pool.py` (NEW)

**Purpose:** a bounded pool (~6 rolling self-snapshots + 2 anchors [seed,
best-ever]) of V5 checkpoints tracked with metadata (`update#`, H2H-vs-best score,
`path`, `p1_p2 gap`, promotion-eligible flag). Snapshot cadence ~2000 updates (set
via B8 `checkpoint_every` + a pool-add hook on top of
`rust_trainer._save_checkpoint` at `rust_trainer.py:802`). Anchor management: the
**seed** anchor = the A-gate-passed BC/PPO seed (first promoted, immutable); the
**best-ever** anchor = highest external-bench H2H-vs-best score (replaces on
improvement, never evicted). Non-anchor FIFO eviction at `len(pool) > ~6`. Load
snapshots as A4 `SelfPrevOpponent` `select_fn` (`rust_live_self_play.py:279` —
defaults learner-argmax, production wires a prior checkpoint) AND as league
opponents (the `v5_snapshot` identity, `opponents_v5.py:109` marks
`requires_checkpoint_pool=True` — the spec-level marker for THIS gap). Physical
storage reuses `rust_trainer` checkpoint npz
(`trainv3_rust_legal_update_{update:04d}.npz`).

**Why NEW (verifier-anticipated):** A5 `select_promotion` tracks a SINGLE
`current_best`, not a pool; `rust_trainer._save_checkpoint` is linear cadence
only (`_should_checkpoint` at `:795`, `checkpoint_every` default 1); no rolling
eviction, no anchors, no best-ever. B1 is the single biggest new build in Block B.

**Depends on:** A4 `SelfPrevOpponent` (`:279`); `rust_trainer._save_checkpoint`
(`:802`, read-only reuse) + `model_mlx.save_checkpoint` format; A5
`select_promotion` (feeds `best-ever`); the A-gate-passed seed (operational,
D-B12).

**Acceptance:**
- Pool holds <= ~6 non-anchors + 2 anchors; FIFO evicts the oldest non-anchor on
  overflow; anchors (`seed`, `best-ever`) are NEVER evicted.
- `best-ever` updates only on strict external-bench H2H improvement (promotion-by-
  loss guard inherited from A5).
- `seed` anchor is immutable after first promotion.
- A snapshot loads back into A4 `SelfPrevOpponent` as a `select_fn` and produces
  a deterministic argmax action (pure self-play when wired to a prior snapshot).
- Pool manifest round-trips to disk + back (paths + metadata).
- No edit to `rust_trainer.py`/`warm_start_v5.py`/`v5_trace.py` (read-only reuse).

**Tests:** `TrainV3.5/python/train_v3/tests/test_snapshot_pool.py` (NEW)
1. `test_fifo_eviction_keeps_anchors` — overflow evicts oldest non-anchor; seed +
   best-ever retained.
2. `test_best_ever_updates_on_strict_improvement` — ties do NOT replace best-ever
   (mirrors A5 `H2H_PROMOTION_THRESHOLD=0.5` strict beat).
3. `test_seed_anchor_immutable` — second promotion does not overwrite seed.
4. `test_load_snapshot_as_self_prev_opponent` — a pool snapshot wired into
   `SelfPrevOpponent.select_fn` yields a deterministic argmax.
5. `test_manifest_roundtrip` — pool manifest writes + reloads with paths +
   metadata intact.
6. `test_pool_grows_self_snapshot_prevalence` (D-B5 hybrid support) — the mix
   weight available to self-snapshots is a function of pool size (prevalence rises
   as the pool fills).

---

### B2 — `v4_orig_temp_spectrum.py` — V4-orig temperature spectrum (3 identities, 1 model)

**File:** `TrainV3.5/python/train_v3/v4_orig_temp_spectrum.py` (NEW)

**Purpose:** three policy-opponents from ONE frozen V4 ONNX model —
`v4-orig-argmax` (temp=0 / argmax, weight 0.40), `v4-orig-t07` (temp=0.7, weight
0.20), `v4-orig-t12` (temp=1.2, weight 0.15). The underlying
`ai/train_v2/onnx_policy.py:16 OnnxActionPolicy` ALREADY exposes a `temperature`
param (validated >0 at `:30-31`; `scaled = mlogits / self._temperature` at `:93`;
`mode='argmax'|'sample'` at `:21`). A4's `V4MaxOpponent`
(`rust_live_self_play.py:297`) is argmax-only and does NOT expose temperature —
B2 wraps the ONNX policy with a temperature param, NOT a port. **Adapter (not a
pure wrap):** `OnnxActionPolicy.select_action(env, player_id)` is env-based
(reads legal actions from a Python env via `build_action_mask` +
`encode_action_features`, `onnx_policy.py:62-78`), whereas A4's policy-opponent
loop calls `PolicyOpponent.select(env_idx, ctx: OpponentCtx)` with packed
legal_action_ids arrays (`rust_live_self_play.py:678-695`); B2 builds a thin
adapter from `OpponentCtx`'s packed arrays to `OnnxActionPolicy`'s forward
(mlogits) — "wraps, does NOT port" understates this adapter, acknowledged here.
Registers three named identities + an alias layer so `v4-orig-t07`/`v4-orig-t12`
are canonical (they are NOT in `league_v5.V5_OPPONENT_KINDS` at `:12` — see B3 for
the alias resolution). NOTE: the stale path
`ai/model_benchmark/policies.py:69 ActionConditionedOnnxPolicy` (transcribed from
`handoff.md:117`) is gitignored (`.gitignore:24`) and ABSENT on disk with zero
definitions anywhere in the repo (grep-confirmed); the corrected anchor is
`ai/train_v2/onnx_policy.py:16 OnnxActionPolicy` (NOT gitignored, verifiable
in-worktree).

**Q5 bias applies (design.md:198):** V4-orig is blind to mana_draw + new cards;
the learner could over-fit "opponent never draws." B2 exposes the
mana_draw-usage-vs-V4-orig-lanes monitor hook consumed by the D-B5 hybrid
reweight policy (B3).

**Depends on:** `ai/train_v2/onnx_policy.py:16 OnnxActionPolicy`
(temperature, read-only reuse — NOT gitignored, verifiable in-worktree); A4
`V4MaxOpponent` (`:297`, the argmax instance to extend); A4 `PolicyOpponent` /
`OpponentCtx` packed-array interface (`rust_live_self_play.py:678-695`, the adapter
target); the frozen V4 ONNX path (same `V4_MAX_NPZ_PATH` env / gitignored npz as
A2/A4 — skip-gate if absent). NOT the stale `ai/model_benchmark/policies.py`
(gitignored at `.gitignore:24`, absent on disk, zero definitions — do NOT cite).

**Acceptance:**
- Three identities built from one ONNX: argmax (temp=0), t07 (temp=0.7), t12
  (temp=1.2) per D-B6.
- temp>1 yields a more-random selection than argmax on a `_FakeOnnxPolicy` with a
  known logit distribution (entropy strictly increases with temperature).
- Weights 0.40/0.20/0.15 carried (frozen unless D-B5 hybrid reweight triggers).
- Skip-gate when the V4 ONNX is absent (worktree).
- No edit to `ai/train_v2/onnx_policy.py` (read-only reuse) or `V4MaxOpponent`
  (extend additively or wrap). The B2 adapter lives in the NEW
  `v4_orig_temp_spectrum.py` (does NOT edit `onnx_policy.py` or A4).

**Tests:** `TrainV3.5/python/train_v3/tests/test_v4_orig_temp_spectrum.py` (NEW)
1. `test_three_identities_from_one_model` — one `_FakeOnnxPolicy` shared by three
   identities with temps 0 / 0.7 / 1.2.
2. `test_higher_temperature_more_random` — entropy(argmax) < entropy(t07) <
   entropy(t12) on a fixed logit set.
3. `test_argmax_identity_is_deterministic` — temp=0 yields argmax.
4. `test_weights_frozen` — the 0.40/0.20/0.15 weights are the spectrum config.
5. `test_skip_if_no_v4_onnx` — skip-gate when the V4 npz is absent.

---

### B3 — `block_b_opponent_mix.py` — Block-B mix composition + aliases + punish_empty_board

**File:** `TrainV3.5/python/train_v3/block_b_opponent_mix.py` (NEW)

**Purpose:** the full Block-B opponent mix = **self-snapshots** (from B1 pool,
dynamic weight per D-B5 hybrid) + **V4-orig spectrum** (B2, 0.40/0.20/0.15) +
**exploit-lanes continuous** (stall/anti_draw_greed/punish_empty_board ~0.05 each)
+ **tail** (greedy_face 0.03, legal_random 0.01, end_turn 0.01). Two DISTINCT gaps
are closed here (do NOT conflate them):

- **(a) Parse-side alias map (alias gap — for `v4-orig-t07`/`v4-orig-t12` ONLY):**
  `league_v5.parse_v5_opponent_mix` (`:43`) rejects names not in
  `V5_OPPONENT_KINDS` (`:12`). `v4-orig-t07`/`v4-orig-t12` are genuinely ABSENT
  from `V5_OPPONENT_KINDS` (only `v4max` is present, `league_v5.py:15`), so they
  need an additive alias map (mirrors A3 `PHASE_A_OPPONENT_NAME_ALIASES` at
  `ppo_phaseA_config.py:102`) resolving `v4-orig-t07`→temp-spectrum-t07,
  `v4-orig-t12`→temp-spectrum-t12. `punish_empty_board` does NOT need this alias —
  it is ALREADY in `V5_OPPONENT_KINDS` via `*EXPLOIT_AGENT_KINDS`
  (`gauntlet_v5.py:13`, `league_v5.py:20`), so `parse_v5_opponent_mix` already
  accepts it.
- **(b) Dispatch-enable for `punish_empty_board` (dispatch gap — A4 EDIT):**
  `punish_empty_board` PARSES but does NOT DISPATCH — A4 `resolve_opponent_dispatch`
  (`rust_live_self_play.py:166-182`) raises `ValueError` because Rust rule code 5
  is commented out in `RULE_AGENT_CODES` (`rust_live_self_play.py:143`). Enabling
  it requires an ADDITIVE EDIT to the A4-built file `rust_live_self_play.py`
  (uncomment line 143: `"punish_empty_board": 5`). This is an edit to an A4 file
  (NOT frozen-classic; A4 built it — see §10), zero Rust change
  (`worker.rs:1258 ExploitAgentKind::PunishEmptyBoard` already exists), gated on
  D-B10. Alternatively B8 routes `punish_empty_board` envs through a Block-B
  extended dispatch resolver before calling A4 — the plan takes the additive-edit
  path (simplest, A4-owned file).

Reweights tail/exploit from Phase-A values (Phase A: 0.10/0.05/0.10 → Block B tail
0.03/0.01/0.01). The mix weight accounting resolves the D-B5 self-snapshot-weight
ambiguity: the frozen weights (V4-orig 0.75 + exploit 0.15 + tail 0.05 = 0.95) are
the NON-self-snapshot share; the self-snapshot share is the residual grown as the
pool fills (D-B5 hybrid), NOT a fixed 0.05.

**Depends on:** B1 (pool identities: `self_prev`/`v5_snapshot`), B2 (V4-orig temp
identities); A4 `sample_opponent_identities` (`:471`, the sampler B3 feeds);
`league_v5.parse_v5_opponent_mix` (`:43`, validation); `gauntlet_v5.EXPLOIT_AGENT_KINDS`
(`:8`, includes `punish_empty_board`); A4 `rust_live_self_play.py:143`
(additive uncomment of `punish_empty_board: 5` — an A4-file edit, see §10).

**Acceptance:**
- The Block-B mix parses via `league_v5.parse_v5_opponent_mix` with the alias
  extension (all identities canonical — `v4-orig-t07`/`v4-orig-t12` via the B3
  alias map; `punish_empty_board` already parses, no alias needed).
- `punish_empty_board` dispatches to Rust code 5 after the additive uncomment of
  `rust_live_self_play.py:143` — verified by
  `resolve_opponent_dispatch('punish_empty_board') == (RULE_DISPATCH, 5)`.
- The mix accounts to 1.0 with the self-snapshot share = residual grown by pool
  size (D-B5 hybrid), V4-orig 0.40/0.20/0.15 frozen, exploit 0.05×3, tail
  0.03/0.01/0.01.
- The alias layer is additive to a Block-B alias map (does NOT edit
  `league_v5.py`'s `V5_OPPONENT_KINDS` — A3/A5 read-only pattern; the alias map
  lives in B3 and covers ONLY `v4-orig-t07`/`v4-orig-t12`).
- A mana_draw-collapse monitor hook is exposed (D-B5): when the learner's
  mana_draw usage vs V4-orig lanes drops out of the A5 band, the reweight policy
  can boost self-snapshot prevalence.

**Tests:** `TrainV3.5/python/train_v3/tests/test_block_b_opponent_mix.py` (NEW)
1. `test_mix_parses_with_aliases` — the full Block-B mix string parses via
   `parse_v5_opponent_mix` with the B3 alias map (`v4-orig-t07`/`v4-orig-t12`
   alias-resolved; `punish_empty_board` parses natively, no alias).
2. `test_punish_empty_board_dispatches_code5` — after the additive uncomment of
   `rust_live_self_play.py:143`, `resolve_opponent_dispatch('punish_empty_board')`
   == `(RULE_DISPATCH, 5)`.
3. `test_weights_account_to_one` — self-snapshot residual + V4-orig + exploit +
   tail = 1.0.
4. `test_self_snapshot_prevalence_grows_with_pool` (D-B5) — larger pool → larger
   self-snapshot share; V4-orig weights unchanged.
5. `test_tail_reweighted_from_phase_a` — greedy_face 0.03 / legal_random 0.01 /
   end_turn 0.01 (NOT the Phase-A 0.10/0.05/0.10).
6. `test_alias_map_covers_only_t07_t12` — the B3 alias map resolves
   `v4-orig-t07`/`v4-orig-t12` and does NOT alias `punish_empty_board` (it parses
   natively); does NOT edit `league_v5.V5_OPPONENT_KINDS`.

---

### B4 — `curriculum.py` — oversample-losing-lane dynamic reweight

**File:** `TrainV3.5/python/train_v3/curriculum.py` (NEW)

**Purpose:** per-lane learner loss tracker + adaptive reweight (design.md:119
"oversample the lane the learner is losing to"). A4
`sample_opponent_identities` (`:471`) samples from a STATIC parsed mix each
update; B4 makes it DYNAMIC: aggregate per-identity learner win/loss from the A4
`LiveRolloutBatch` (`:415` class; `dispatch_log` field at `:438` carries
`opponent_identities` (`:430`) + `learner_actor_ids` (`:428`)), compute per-lane
loss rates over a rolling window, and reweight the mix toward losing lanes
proportional to the loss margin, capped ~25%/update (D-B8). Injects the
reweighted mix into `sample_opponent_identities` (hook the `mix` arg).
V5LeagueConfig has NO curriculum field (`league_v5.py:25`); B4 is the new
component.

**Depends on:** A4 `LiveRolloutBatch` (`:415`; `dispatch_log` `:438`,
`opponent_identities` `:430`, `learner_actor_ids` `:428`); B3 (the mix B4
reweights); A4 `sample_opponent_identities` (`:471`).

**Acceptance:**
- Per-lane loss rates computed from `dispatch_log` over a rolling window.
- The lane with the highest learner loss rate gets the largest oversample boost.
- Reweight is capped at ~25%/update (D-B8 adaptive) — the mix does not collapse to
  a single lane in one step.
- A lane the learner beats 100% gets NO oversample boost.
- The reweighted mix still accounts to 1.0.

**Tests:** `TrainV3.5/python/train_v3/tests/test_curriculum.py` (NEW)
1. `test_losing_lane_oversampled` — fabricated dispatch log with learner losing to
   `stall` → `stall` weight rises next update.
2. `test_winning_lane_not_oversampled` — a lane the learner beats 100% gets no
   boost.
3. `test_reweight_capped` — a single update never shifts more than the cap (~25%)
   toward one lane.
4. `test_reweighted_mix_accounts_to_one` — after reweight, weights sum to 1.0.
5. `test_rolling_window` — only the last N updates of dispatch log inform the
   reweight (stale losses age out).

---

### B5 — `second_start_parity.py` — continuous p1/p2 parity loop (closes A4's open loop)

**File:** `TrainV3.5/python/train_v3/second_start_parity.py` (NEW)

**Purpose:** continuous second-start parity (design.md:120 "p1_p2_score_gap <=
0.12 acceptance; oversample p2-init on breach"). The MECHANISM already exists: A3
`second_start_oversampling_scheme` (`ppo_phaseA_config.py:258`,
`PHASE_A_P1_P2_GAP_THRESHOLD=0.12` at `:120`) + A4 `sample_learner_sides`
(`:490`) implement the gap-weighted p1/p2 split — but A4 accepts
`p1_score_rate`/`p2_score_rate` as INPUTS and never MEASURES them. B5 CLOSES THE
LOOP: measure p1/p2 score rates over a rolling window of league games each update
(via the production `GameRunner` adapter, A5 Protocol) + feed back into
`sample_learner_sides` + the `p1_p2_score_gap <= 0.12` ACCEPTANCE gate (feeds B6
promotion — B5 PRODUCES the gap, B6 CONSUMES it; B5 does NOT depend on B6). Builds
NEW measurement (A5 pattern — the dead `gauntlet_v5.p1_p2_max_score_gap=0.12` at
`:43` has ZERO consumers in `gauntlet_v5.py` itself; the same-named key in
`TrainV3.5/scripts/run_phase1_runtime_acceptance_bench.py:50,639` is a SEPARATE
local THRESHOLDS dict, NOT an import of the `gauntlet_v5` field — so the field is
genuinely dead; do NOT wire it). The measurement source (open_question: gauntlet
games vs the external bench, or live-self-play episode outcomes) is resolved
here: **the external-bench gauntlet games** (B6 plays them anyway; p1/p2 score
rates are a byproduct, no extra games). **Side-stratification required
(finding):** A5 `GameResult` (`a_gate.py:685-708`) records `outcome`/
`mana_draw_count`/`eligible_turns`/`opponent` but NOT the candidate's starting
side, and `play_gauntlet` (`:753`) leaves the candidate side to the wired
`game_runner`. To measure p1_score_rate and p2_score_rate SEPARATELY, B5 must
either (a) have the gauntlet play the candidate from BOTH p1 and p2 (a
side-stratified gauntlet subset), or (b) extend the `GameResult` view with a
`candidate_side` field (additive, in B5/B6 — A5 `GameResult` is a frozen dataclass,
so compose/wrap a `BlockBGameResult` rather than mutate). B5 owns this
side-stratified run unless B6 already plays both sides (it does not by default).

**Depends on:** A3 `second_start_oversampling_scheme` (`:258`); A4
`sample_learner_sides` (`:490`); A5 `GameRunner` Protocol (`:711`) + the production
adapter (A4-wired). Does NOT depend on B6 — B6 depends on B5 (the gap flows
B5→B6).

**Acceptance:**
- p1/p2 score rates measured SEPARATELY (the candidate is played from BOTH p1 and
  p2, or `candidate_side` is recorded per game via a `BlockBGameResult` that
  composes A5 `GameResult` without mutating the frozen dataclass) over a rolling
  window of gauntlet games.
- On `p1_p2_score_gap > 0.12`, p2-init is oversampled next update (breach →
  rebalance).
- The gap is exposed as a promotion input to B6 (continuous parity is a promotion
  criterion, design.md:121).
- A perfectly balanced p1/p2 (gap=0) triggers no oversample change.
- Builds NEW (the dead `gauntlet_v5.p1_p2_max_score_gap` field is NOT wired — A5
  pattern; the same-named key in `run_phase1_runtime_acceptance_bench.py:50,639` is
  a separate local dict, NOT the `gauntlet_v5` field).

**Tests:** `TrainV3.5/python/train_v3/tests/test_second_start_parity.py` (NEW)
1. `test_breach_oversamples_p2` — gap>0.12 → p2-init weight rises.
2. `test_balanced_no_change` — gap<=0.12 → no oversample change.
3. `test_gap_feeds_promotion` — the measured gap is exposed to B6.
4. `test_rolling_window` — only recent gauntlet games inform the rate.
5. `test_does_not_wire_dead_field` — the measurement is built new; the dead
   `gauntlet_v5.p1_p2_max_score_gap` field is NOT consumed (regression guard for
   the A5 pattern).
6. `test_measures_p1_and_p2_separately` — p1_score_rate and p2_score_rate are
   measured from side-stratified gauntlet games (or recorded `candidate_side`),
   not a single aggregate that assumes a fixed candidate side.

---

### B6 — `block_b_gate.py` — Block-B external-bench promotion gate (EXTENDS A5)

**File:** `TrainV3.5/python/train_v3/block_b_gate.py` (NEW)

**Purpose:** Block-B external-bench promotion (design.md:121 "Promote iff external
bench (H2H vs best self-snapshot + gauntlet + mana_draw band + p1_p2 gap) improves
monotonically >= N_snap"). EXTENDS A5 (does NOT rewrite `a_gate.py`): composes A5's
`check_mana_draw_band` (`:309`), A5's H2H-vs-best (`check_h2h_trending` `:374` is
trending-up; B6 needs monotone IMPROVEMENT of the full aggregate — distinct),
A5's gauntlet runner (`play_gauntlet` `:753`), and ADDS `p1_p2_gap <= 0.12` (from
B5) + **monotonic improvement >= N_snap** of the 4-component external-bench
aggregate. Distinct from the A-gate 4-criteria (open_question #11/#12): the A-gate
(`no_assist`/`exploit_resistance`/`mana_draw`/`H2H-trending`, design.md:114) is the
PHASE-A EXIT gate; Block-B promotion is a DIFFERENT external-bench gate — Block B
does NOT re-apply `no_assist`/`exploit_resistance` (those are Phase-A exit, not
league promotion). Promotion-by-loss guard inherited from A5 (internal
`ppo_loss`/`approx_kl`/`entropy` NEVER read, `CandidateInternalMetrics` monitoring-
only). New `BlockBGateResult` (A5 `AGateResult` is frozen-dataclass; do not mutate
it — compose or wrap).

**Depends on:** A5 `evaluate_a_gate` components (`:484`), `check_mana_draw_band`
(`:309`), `check_h2h_trending` (`:374`), `play_gauntlet` (`:753`),
`select_promotion` (`:607`, promotion-by-loss guard); B1 (best-ever anchor + the
candidate), B5 (p1_p2 gap); D-B1 (N_snap).

**Acceptance:**
- Promotion requires monotone improvement of the FULL external-bench aggregate
  (H2H-vs-best + gauntlet + mana_draw band + p1_p2 gap) over N_snap consecutive
  snapshots (D-B1 default 5) — NOT just H2H non-decreasing.
- `p1_p2_gap <= 0.12` is a required component (continuous parity).
- The A-gate `no_assist`/`exploit_resistance` thresholds are NOT re-applied
  (regression guard: a candidate passing Block-B promotion does NOT require
  A-gate `no_assist` re-pass).
- Promotion-by-loss guard: promotion is independent of internal
  `ppo_loss`/`approx_kl`/`entropy` (regression test, mirrors A5).
- First-snapshot case handled (no prior best → seed anchor, A5 pattern
  `current_best_h2h_score_rate=None`).

**Tests:** `TrainV3.5/python/train_v3/tests/test_block_b_gate.py` (NEW)
1. `test_promotion_requires_monotone_improvement_over_N_snap` — 5 improving
   snapshots → promote; one dip in the window → no promote.
2. `test_p1_p2_gap_required` — gap>0.12 → no promote even if other components
   improve.
3. `test_does_not_reapply_a_gate_no_assist` — a candidate NOT passing A-gate
   `no_assist` CAN still promote via Block-B external-bench (regression guard for
   open_question #11).
4. `test_promotion_independent_of_ppo_loss` — two candidates, identical
   external-bench, different PPO loss/KL/entropy → same decision (inherited A5
   guard).
5. `test_first_snapshot_seed` — no prior best → seed anchor, no plateau yet.
6. `test_composes_a5_band_and_gauntlet` — A5 `check_mana_draw_band` +
   `play_gauntlet` are reused (not rewritten).

---

### B7 — `exit_to_c2.py` — plateau detector + exit→C2 signal

**File:** `TrainV3.5/python/train_v3/exit_to_c2.py` (NEW)

**Purpose:** detect "H2H vs best self-snapshot plateaus (no gain K_snap) below the
dominance target" (design.md:122, verbatim) → emit the exit→C2 signal. The INVERSE
of A5 `check_h2h_trending` (`:374`, which detects trending UP):
`detect_h2h_plateau(h2h_scores, K_snap, min_gain, dominance_target)` — no gain >
`min_gain` over `K_snap` consecutive snapshots (the plateau) AND the candidate is
BELOW the `dominance_target` over prior best (D-B3 default 3a = the LITERAL
"below" reading: still weak — self-play exhausted before reaching dominance; C2
human data is the next lever to break the plateau). C2 = deploy best V5 vs humans
in rlhf_env, collect ~3-5k fresh preV5-vs-human battles (design.md:125). The
dominance-target notion (D-B3) distinguishes the two readings:
- **Default 3a "below" (still-weak) — spec-literal:** plateau AND candidate BELOW
  dominance target → exit (self-play stuck below dominance; C2 human data breaks
  the plateau). A plateau AT/ABOVE the dominance target (dominant) does NOT fire
  this exit — a dominant plateau is handled via the Block E1 ship path, NOT C2.
- **Alternative 3a "at/above" (dominant) — colloquial reading:** plateau AND
  candidate AT/ABOVE dominance target → exit (self-play reached dominance and
  stalled; C2 collects human data for the next phase). A plateau BELOW target
  does NOT exit (keep training).

**The plan's B7 acceptance + tests follow the DEFAULT "below" reading; if the
user picks "at/above" (D-B3 3a), tests 1 and 3 FLIP** (noted per-test below). The
`min_gain` plateau gain-tolerance is an IMPLEMENTER default (D-B2): `min_gain =
0.01` H2H score-rate — small enough that genuine improvement clears it while
noise does not; not a user decision.

**Depends on:** B6 (H2H-vs-best series), A5 `check_h2h_trending` (`:374`,
inverse-logic reference); D-B2 (K_snap + `min_gain` implementer default),
D-B3 (3a below-vs-above + 3b dominance notion).

**Acceptance:**
- Plateau = no H2H gain > `min_gain` (implementer default 0.01) over `K_snap`
  consecutive snapshots (D-B2 default ~10).
- **DEFAULT reading (D-B3 3a = "below", spec-literal):** exit fires when plateau
  AND candidate is BELOW the dominance target (D-B3 3b) — still weak; self-play
  exhausted before reaching dominance; C2 human data is the next lever. A
  plateau AT/ABOVE the dominance target (dominant) does NOT fire this exit
  (dominant plateau → Block E1 ship path, not C2).
- A still-improving H2H series does NOT fire exit.
- The exit signal carries the Block-B-best checkpoint path (the C2 deploy
  candidate).
- IF the user picks D-B3 3a = "at/above" (dominant), the exit condition FLIPS:
  exit fires when plateau AND candidate AT/ABOVE the dominance target; a plateau
  below target does NOT exit (keep training). The B7 tests flip accordingly
  (noted per-test).

**Tests:** `TrainV3.5/python/train_v3/tests/test_exit_to_c2.py` (NEW)
1. `test_plateau_below_target_fires_exit` — flat H2H BELOW dominance target for
   K_snap → exit (DEFAULT reading; self-play stuck below dominance). [FLIPS to
   `test_plateau_at_or_above_target_fires_exit` if D-B3 3a = "at/above".]
2. `test_still_improving_no_exit` — rising H2H → no exit.
3. `test_plateau_at_or_above_target_no_exit` — flat H2H AT/ABOVE dominance target
   → no exit (DEFAULT reading: dominant plateau handled via E1, not C2).
   [FLIPS to `test_plateau_below_target_no_exit` if D-B3 3a = "at/above".]
4. `test_exit_carries_best_checkpoint` — the exit signal references the best-ever
   anchor path.
5. `test_k_snap_window` — plateau must persist the full K_snap window (a single
   flat snapshot does not exit).
6. `test_min_gain_tolerance` — a sub-`min_gain` (0.01) uptick counts as "no gain"
   (plateau); an above-`min_gain` uptick resets the plateau window.

---

### B8 — `block_b_league_driver.py` — multi-update live league driver (composes A4)

**File:** `TrainV3.5/python/train_v3/block_b_league_driver.py` (NEW)

**Purpose:** the multi-update live league driver. Composes A4
`run_live_self_play_update` (`:970`, ONE update) into the league loop: per update
— curriculum reweight (B4) + parity feedback (B5) feed A4
`sample_opponent_identities`/`sample_learner_sides`; A4 runs one PPO update; every
~2000 updates (D-B4) snapshot via B1 pool-add on top of
`rust_trainer._save_checkpoint`; per snapshot — run the external-bench gauntlet
(production `GameRunner` adapter, A5 Protocol wired to A4) → B6 promotion
decision → B1 best-ever update → B7 plateau/exit check → emit exit→C2 if plateau.
Reuses the trace-pool `train_rust_ppo_trace_files` (`rust_trainer.py:92`) loop
scaffolding (metrics list, `checkpoint_every`, league manifest) as a STRUCTURAL
TEMPLATE only — B8 drives the LIVE path via A4, NOT trace-pool replay. Continues
A's frozen hyperparams (D-B7): entropy=0.01, epochs=6, max_turns=120,
learner_only_reward, decisive_early_end (all in A3 `PhaseAPPOConfig`, unchanged).

**Depends on:** A4 `run_live_self_play_update` + `collect_rust_live_rollout`
(`:970`/`:516`); A3 `PhaseAPPOConfig` (continued, D-B7); A5 `GameRunner` Protocol
(`:711`) + production adapter (A4-wired); B1-B7; `rust_trainer._save_checkpoint`
(`:802`); D-B4 (cadence/pool), D-B11 (compute).

**Acceptance:**
- The driver runs N updates on a seeded arena (synthetic; A4 `FakeWorker`),
  snapshots at the ~2000 cadence, fills the pool, and emits a promotion decision +
  plateau check per snapshot.
- Curriculum (B4) + parity (B5) feedback is threaded each update (the loop is
  closed, not static).
- Continues A's frozen hyperparams unchanged (regression guard: entropy/epochs/
  max_turns match A3).
- The driver reuses the A4 live path (does NOT call `train_rust_ppo_trace_files`
  replay — that's the documented approximation, not the default).
- Skip-gate when MLX/Rust unbuildable (worktree); unit tests use `FakeWorker` +
  `_FakeGameRunner`.

**Tests:** `TrainV3.5/python/train_v3/tests/test_block_b_league_driver.py` (NEW)
1. `test_runs_n_updates_and_snapshots` — N synthetic updates, snapshots at the
   cadence, pool fills.
2. `test_curriculum_and_parity_threaded` — B4/B5 feedback changes the sampled mix
   + sides across updates (loop is closed).
3. `test_promotion_and_plateau_per_snapshot` — each snapshot triggers a B6
   promotion decision + B7 plateau check.
4. `test_continues_a_hyperparams` — entropy/epochs/max_turns match A3
   (regression guard).
5. `test_uses_live_path_not_trace_pool` — the driver calls A4
   `run_live_self_play_update`, NOT `train_rust_ppo_trace_files`.
6. `test_skip_if_no_mlx_or_rust` — skip-gate in the worktree.

---

## 4. League-run support (in-worktree-buildable, supports the operational run)

Not numbered components (config + adapters), required for the operational league
run and buildable now:

- **Production `GameRunner` adapter** — A5 `GameRunner` Protocol (`:711`) is
  injectable; production wires A4 `run_live_self_play_update`/`collect_rust_live_rollout`.
  The adapter plays real Rust ArenaEnv games for the external-bench gauntlet (B6),
  p1/p2 measurement (B5 — side-stratified: candidate played from BOTH p1 and p2, or
  `candidate_side` recorded via a `BlockBGameResult` composing the frozen
  `GameResult`), H2H-vs-best (B6/B7). A5 built the Protocol + synthetic tests; B8
  wires the production adapter (MLX/Rust-gated).
- **Block-B config** — `PhaseBPPOConfig` or a Block-B extension of A3
  `PhaseAPPOConfig` carrying the Block-B mix (B3), cadence/pool (D-B4),
  N_snap/K_snap/dominance-target (D-B1/B2/B3), curriculum rate (D-B8), and the
  continued A hyperparams (D-B7). Pure-python, synthetic-testable.
- **League manifest writer** — mirrors `rust_trainer`'s league manifest, recording
  per-snapshot metadata (update#, H2H-vs-best, p1_p2 gap, promotion-eligible) for
  the B1 pool + the C2 handoff.

**Acceptance:** the adapter plays a seeded gauntlet game end-to-end via A4 (gated);
the config validates the Block-B mix (B3) + carries N_snap/K_snap/dominance-target;
the manifest round-trips.

---

## 5. Test strategy (synthetic + source-vs-source)

- **Source-vs-source discipline (Block -1/0/A1 lesson, BLOCK_A_PLAN §5):** the live
  Rust engine = oracle, V5/training code = UUT; avoid self-referential fixture
  regen. The B2 V4-temperature tests use a `_FakeOnnxPolicy` with a KNOWN logit
  distribution (entropy is a pure function of temperature on fixed logits) — not
  the policy under test scoring its own outputs. The B1 pool round-trip uses
  real `model_mlx.save_checkpoint` format (mirrors A2/A4 checkpoint tests).
- **Synthetic data only** for all in-worktree tests: fabricated A4
  `LiveRolloutBatch` dispatch logs (B4), fabricated external-bench series (B6/B7),
  fabricated gauntlet `GameResult`s via A5 `_FakeGameRunner` (B5/B6/B8). No real
  ONNX, no real Rust FFI, no real MLX required for unit tests.
- **Injection patterns (reuse A4/A5):** A4 `FakeWorker` (rust_live_self_play tests)
  for the live-self-play path; A5 `_FakeGameRunner` (a_gate tests) for the
  gauntlet/external-bench path. B8 composes both.
- **Frozen-classic guard:** no test or component edits
  `classic_obs_v1`/`classic_actions_v1`/`classic_card_shape_v1`/`classic_rl_env.py`/
  `reward_v5.py`/`v5_trace.py`/`warm_start_v5.py`/`run_phase26*.py`/
  `run_v5_acceptance.py`/`core/state.py` (B5/B8 measure via the `GameRunner`
  adapter, never edit engine state). `league_v5.py`/`gauntlet_v5.py`/
  `opponents_v5.py` consumed READ-ONLY (A5 build-new-over-wire-dead pattern; the B3
  alias map covers ONLY `v4-orig-t07`/`v4-orig-t12` and lives in B3, not in
  `V5_OPPONENT_KINDS`; `punish_empty_board` already parses natively). `ai/train_v2/
  onnx_policy.py` consumed read-only (B2 wraps, NOT edits — the adapter lives in the
  NEW `v4_orig_temp_spectrum.py`). `rust_ffi.py` is NOT frozen-classic (A4 additive
  accessors already present; Block B likely composes existing — any new accessor is
  additive). `rust_live_self_play.py` (A4-built) takes ONE additive uncomment
  (`:143` `punish_empty_board: 5`, D-B10) — see §10. V5 work in `v5_*` files + NEW
  `block_b_*.py`/`snapshot_pool.py`/`v4_orig_temp_spectrum.py`/`curriculum.py`/
  `second_start_parity.py`/`block_b_gate.py`/`exit_to_c2.py`/
  `block_b_league_driver.py`.
- **Skip-gates:** MLX tests skip when mlx unimportable (worktree); Rust FFI tests
  skip when the extension is unbuildable; V4-ONNX tests skip when the npz is absent.

---

## 6. Operational steps (user runs these, in order)

1. **Resolve user decisions D-B1..D-B12** (§2). D-B5 (Q5 V4-orig bias) is the
   dominant tension; D-B12 (A-gate status) is the execution gate.
2. **A-gate PASS** (the execution gate) — A.pilot deploy+collect → A.BC run →
   A.PPO training run → A-gate measurement must complete and PASS (design.md:114)
   so the A-gate-passed policy is the inaugural best-self-snapshot / seed anchor.
   Block B in-worktree components are built regardless; the league RUN waits here.
3. **Block B league training run** — execute B8 on compute (D-B11), snapshot per
   cadence (D-B4), curriculum + parity continuous, B6 promotion per snapshot, B7
   plateau/exit per snapshot.
4. **External-bench gauntlet runs** — per snapshot, play the gauntlet + H2H-vs-best
   + measure p1/p2 gap + mana_draw band (the production `GameRunner` adapter, B8-
   wired) → B6 promotion decision + B7 plateau check.
5. **Exit→C2** — when B7 fires (plateau below dominance target, D-B3), take the
   Block-B-best checkpoint into C2 (deploy vs humans, collect ~3-5k fresh
   preV5-vs-human battles, design.md:125).

---

## 7. Block-B exit summary (gate to Block C / C2, design.md:122-125)

| Signal | Threshold | Source | Built in |
|--------|-----------|--------|----------|
| External-bench monotone promotion | full aggregate (H2H-vs-best + gauntlet + mana_draw band + p1_p2 gap) improves monotonically >= N_snap (D-B1) | `design.md:121` | B6 (extends A5) |
| p1_p2 parity (continuous) | gap <= 0.12 acceptance; oversample p2 on breach | `design.md:120` | B5 (closes A4 loop) |
| Plateau / exit→C2 | no H2H-vs-best gain > min_gain over K_snap (D-B2) AND candidate BELOW dominance target (D-B3 3a default "below", spec-literal) — FLIPS to AT/ABOVE if user picks the colloquial reading | `design.md:122` | B7 (inverse of A5 trending) |
| Promotion-by-loss guard | internal ppo_loss/approx_kl/entropy NEVER read | `design.md:112` | B6 (inherited A5) |
| Best-ever anchor | highest external-bench H2H, never evicted | `design.md:118` | B1 (NEW pool) |

**Exit→C2:** on B7 plateau-below-dominance-target, the Block-B-best checkpoint
deploys vs humans in rlhf_env (C2, design.md:125), collecting ~3-5k fresh
preV5-vs-human battles. The final self-snapshot pool feeds Block D league-2 +
the next iteration's league pool (design.md:131/158).

---

## 8. Open questions / unresolved

1. **D-B5 (Q5 V4-orig blind-lane bias)** — THE dominant spec-internal conflict
   (0.75 frozen total vs "modest"). The plan defaults to Hybrid (freeze weights +
   grow self-snapshot prevalence + mana_draw-collapse monitor) but the user must
   confirm; this is not an implementer choice.
2. **D-B1/B2/B3 (N_snap / K_snap / dominance target)** — all named by the spec,
   none pinned. The plan defaults to 5 / ~10 / progression ~0.52-0.55
   (design.md:148); user confirms. **D-B3 3a (below-vs-above) is the load-bearing
   sub-question:** design.md:122 "plateaus ... below the dominance target" is
   genuinely ambiguous (plateau-while-weak → exit for human data, vs
   plateau-while-dominant → exit because self-play exhausted). The plan defaults
   to the LITERAL "below" (still-weak) reading and the B7 tests FLIP if the user
   picks the "at/above" (dominant) reading. The `min_gain` plateau gain-tolerance
   (default 0.01) is an implementer default, NOT a user decision (D-B2).
3. **A-gate vs Block-B promotion distinction (open_question #11/#12)** — confirmed
   in §3/B6: Block B does NOT re-apply the A-gate `no_assist`/`exploit_resistance`
   thresholds (Phase-A exit, not league promotion). A regression guard
   (`test_does_not_reapply_a_gate_no_assist`) enforces it.
4. **Self-snapshot weight (D-B5 second arm)** — the spec does not pin a
   self-snapshot weight; the naive sum leaves 0.05, contradicting Q5. Resolved via
   the D-B5 hybrid (self-snapshot share = residual grown by pool size).
5. **"Decontaminated reward" (design.md:117, undefined)** — defined operationally
   as A's `reward_v5` formulas with A3 learner-only attribution, unchanged. No new
   code; flagged as a spec-term gap.
6. **p1/p2 measurement source (open_question)** — resolved in B5: the external-
   bench gauntlet games (B6 plays them; p1/p2 rates are a byproduct, no extra
   games), NOT separate live-self-play episode outcomes. **Side-stratification
   required:** A5 `GameResult` does not record the candidate's starting side, so
   B5 plays the candidate from BOTH p1 and p2 (or records `candidate_side` via a
   `BlockBGameResult` that composes the frozen `GameResult`).
7. **v4-orig-t07/t12 canonical names + punish_empty_board dispatch (open_question)**
   — resolved in B3 as TWO distinct gaps: (a) an additive alias map for
   `v4-orig-t07`/`v4-orig-t12` ONLY (genuinely absent from `V5_OPPONENT_KINDS`;
   mirrors A3 `PHASE_A_OPPONENT_NAME_ALIASES`; does NOT edit
   `league_v5.V5_OPPONENT_KINDS`); (b) an additive uncomment of
   `rust_live_self_play.py:143` to dispatch `punish_empty_board` to Rust code 5
   (it already PARSES via `*EXPLOIT_AGENT_KINDS`, so no alias needed — the gap is
   dispatch-only; this is an edit to an A4-built file, see §10).
8. **Block-B gate composition vs A5 (open_question)** — resolved in B6: a NEW
   `BlockBGateResult` composes A5's components (mana_draw band, gauntlet, H2H) +
   adds p1_p2 gap + monotone-improvement-over-N_snap; A5 `AGateResult` is
   frozen-dataclass (not mutated).
9. **D-B7 (freeze A hyperparams)** — the plan freezes A's 5 fixes verbatim
   (spec-literal "continuing A's fixed hyperparams"); any retune risks
   re-introducing the root causes A fixed. User confirms.
10. **Pool coexistence with Block D league-2 (design.md:131)** — the Block-B pool
    feeds Block D + the next iteration; whether Block D reuses the SAME pool or a
    fresh one is a Block-D planning question, deferred.

---

## 9. Dependency graph

```
Block A in-worktree COMPLETE (A3 ppo_phaseA_config, A4 rust_live_self_play, A5 a_gate)
   |
   +-- B1 snapshot_pool.py (self-snapshot pool ~6 + 2 anchors, FIFO, best-ever)
   |     depends: A4 SelfPrevOpponent (:279), rust_trainer._save_checkpoint (:802 RO), A5 select_promotion
   |
   +-- B2 v4_orig_temp_spectrum.py (3 identities from 1 frozen V4 ONNX: argmax/t07/t12 — SIBLING of B1, no pool dep)
   |     depends: ai/train_v2/onnx_policy.py:16 OnnxActionPolicy (temperature, RO), A4 V4MaxOpponent (:297), A4 PolicyOpponent/OpponentCtx (:678-695 adapter target)
   |
   +-- B3 block_b_opponent_mix.py (mix composition + aliases + punish_empty_board code 5 + tail/exploit reweight)
   |     depends: B1 (pool identities), B2 (V4-orig identities), A4 sample_opponent_identities (:471), league_v5.parse_v5_opponent_mix (:43)
   |
   +-- B4 curriculum.py (per-lane loss tracker + adaptive reweight, oversample losing lane)
   |     depends: A4 LiveRolloutBatch (:415 class, :430 opponent_identities, :438 dispatch_log, :428 learner_actor_ids), B3 (mix reweighted), A4 sample_opponent_identities (:471)
   |
   +-- B5 second_start_parity.py (continuous p1/p2 gap loop, closes A4 sample_learner_sides — PRODUCES the gap B6 consumes)
   |     depends: A3 second_start_oversampling_scheme (:258), A4 sample_learner_sides (:490), A5 GameRunner Protocol (:711). NOTE: B5 does NOT depend on B6; B6 depends on B5 (gap flows B5→B6).
   |
   +-- B6 block_b_gate.py (external-bench monotone promotion, EXTENDS A5 + p1_p2 gap + N_snap)
   |     depends: A5 evaluate_a_gate (:484)/check_mana_draw_band (:309)/check_h2h_trending (:374)/play_gauntlet (:753)/select_promotion (:607), B1 (best-ever), B5 (p1_p2 gap — consumed here)
   |
   +-- B7 exit_to_c2.py (plateau detector, inverse of A5 trending, K_snap + dominance target)
   |     depends: B6 (H2H-vs-best series), A5 check_h2h_trending (:374 inverse ref)
   |
   +-- B8 block_b_league_driver.py (multi-update loop composes A4 run_live_self_play_update)
   |     depends: A4 (:970/:516), A3 PhaseAPPOConfig (continued), A5 GameRunner + production adapter, B1-B7, rust_trainer._save_checkpoint (:802)
   |
   +-- League-run support (§4): production GameRunner adapter, PhaseBPPOConfig, league manifest writer
         depends: A4/A5 Protocol, B3 mix, D-B1/B2/B3/B4/B7/B8
```

---

## 10. Frozen-classic guard (preserved, in effect — carries from Block A)

- Prod runtime must NOT import TrainV3/TrainV3.5 (`golden_trace.py` is training
  tooling only).
- Frozen-classic files BYTE-LOCKED: `classic_obs_v1.py`, `classic_actions_v1.py`,
  `classic_card_shape_v1.py`, `classic_rl_env.py` (all in `ai/train_v2/`). V5 work
  in `v5_*` files + NEW `block_b_*.py`/`snapshot_pool.py`/`v4_orig_temp_spectrum.py`/
  `curriculum.py`/`second_start_parity.py`/`block_b_gate.py`/`exit_to_c2.py`/
  `block_b_league_driver.py`. `classic_rl_env.py` NOT modified. `reward_v5.py`
  consumed read-only (already per-side; learner-only attribution is in A3/A4, NOT
  a reward_v5 edit). `warm_start_v5.py` consumed read-only. `v5_trace.py` consumed
  read-only (data-contract; NOT imported into the training path). **`core/state.py`
  NOT modified** (B5/B8 measure via the `GameRunner` adapter + the A4 live path,
  never edit engine state; carries the workflow constraint forward — Block A had
  the same omission).
- `league_v5.py`/`gauntlet_v5.py`/`opponents_v5.py`/`run_v5_acceptance.py` consumed
  READ-ONLY (A5 build-new-over-wire-dead pattern; Block B continues it — the dead
  `gauntlet_v5.p1_p2_max_score_gap=0.12` field is NOT wired; the same-named key in
  `TrainV3.5/scripts/run_phase1_runtime_acceptance_bench.py:50,639` is a separate
  local dict, NOT the field; B5 builds new). The B3 alias map lives in B3, NOT in
  `league_v5.V5_OPPONENT_KINDS`.
- `rust_trainer.py` consumed read-only (`_save_checkpoint` reused; the loop
  scaffolding used as a structural template only — B8 drives the LIVE A4 path, not
  trace-pool replay).
- **`rust_live_self_play.py` (A4-built, NOT frozen-classic) — ONE ADDITIVE EDIT in
  Block B:** uncomment `RULE_AGENT_CODES` line 143 (`"punish_empty_board": 5`) to
  enable the `punish_empty_board` dispatch (D-B10, B3b). This is an additive
  uncomment of a documented-but-excluded Rust rule code (zero Rust change;
  `worker.rs:1258 ExploitAgentKind::PunishEmptyBoard` already exists). No other
  A4 line is modified. `ai/train_v2/onnx_policy.py` (B2 dependency) consumed
  read-only (wrap, NOT edit). `ai/model_benchmark/policies.py` is gitignored
  (`.gitignore:24`) and ABSENT — do NOT cite or edit (stale handoff.md path).
- `rust_ffi.py` is NOT frozen-classic — A4's additive accessors already present;
  Block B likely composes existing accessors. Any new accessor is ADDITIVE (no
  existing constructor modified), with the 157 cargo tests re-confirmed green (as
  in A4).
- Worktree discipline: run cargo from
  `/Users/laveqox/Documents/ExtraArenaRaS/.claude/worktrees/glm-TrainV3.5Prep/TrainV3.5`,
  python from parent; never cd to original repo root. `TrainV3.5/` untracked at
  repo root but tracked in this worktree (expected, do NOT fix git tracking).
- Source-vs-source: live Rust engine = oracle, V5/training code = UUT; avoid
  self-referential fixture regen.