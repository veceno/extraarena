# TrainV3.5 — V5-Max Model Training Guide

> **Scope.** This is the *точечное руководство по обучению модели* — a focused,
> runnable recipe for training a V5 ("V5-Max") model on the TrainV3.5 architecture.
> It assumes you have read `ARCHITECTURE.md` (the sibling full-section doc) for the
> vocabulary (V5-Max pipeline blocks, 7128-dim observation, 601 candidates + parallel
> mana_draw head, the Rust↔Python FFI, the prod wiring). This file is the "how do I
> actually train one" counterpart: prerequisites → the entries → a worked config →
> checkpoints → export → ship → prod-load → a concrete end-to-end smoke → troubleshooting.

---

## 1. Prerequisites

Run everything from the worktree root:
`/Users/laveqox/Documents/ExtraArenaRaS/.claude/worktrees/glm-TrainV3.5Prep`.
**Never `cd` to the original repo root** — the worktree is a separate checkout.

### 1.1 Build the Rust kernel (FFI)
```bash
cargo build --release --manifest-path TrainV3.5/rust/trainv3_core/Cargo.toml
```
This produces `TrainV3.5/target/release/libtrainv3_core.dylib` (darwin). The Python
training loop talks to it through `ctypes` (`train_v3.rust_ffi.RustBatchWorker`). If you
build elsewhere, point the loader at it:
```bash
export TRAINV3_CORE_LIB="$PWD/TrainV3.5/target/release/libtrainv3_core.dylib"
```
(`rust_ffi.resolve_library_path` checks the explicit arg first, then `TRAINV3_CORE_LIB`,
then `default_library_candidates()` under `TrainV3.5/target/release/`.)

### 1.2 Python path + deps
```bash
export PYTHONPATH="$PWD:$PWD/TrainV3.5/python"
```
`$PWD` gives access to `core.*`, `ai.*` (the production oracle + MLX model code under
`ai/train_v2/`). `$PWD/TrainV3.5/python` gives access to `train_v3.*`. Use `python3`.
Training uses **MLX** (`ai/train_v2/model_mlx.py` for checkpoint I/O) + the FFI; install
per the repo's Python deps if not already present.

### 1.3 V4-Max warm-start checkpoint (optional but recommended)
`train_v3.warm_start_v5.load_v4_max_into_v5(policy, npz_path=None)` warm-starts a V5
policy from a V4-Max `.npz`. Resolution (`warm_start_v5.resolve_v4_max_npz_path`):
explicit `npz_path` → `V4_MAX_NPZ_PATH` env → walk-up search. Without it, BC/PPO will
start from a fresh init (still works; just slower to converge). The warm-start is
**PARTIAL** (Block-0 decision): faithful V4 params copied exactly, shape-compat params
copied by shape, fresh V5 layers (global/private/history encoders, mana_draw head)
left at init.

> **Frozen-classic note.** The V4-orig ONNX + the V4 session dict are byte-frozen; V5
> wiring is purely additive and never mutates them. You are training a NEW V5 model,
> not touching V4.

### 1.4 (Block C only) rlhf_env running
The Block-C RLHF loop reads fresh **human** v5_trace rows from `rlhf_env`'s battle
artifacts (`battles/<bid>/v5/{meta,turns,actions}.jsonl`) via an MCP client on
**port 8090**. Only needed for the C2→C3 offline-replay loop; the basic PPO path
(Blocks A/B/D/E1) does not need rlhf_env.

---

## 2. The training entries (which one do I call?)

| Goal | Entry | How |
|---|---|---|
| **Plain PPO on a trace pool** (the simplest full train) | `train_v3.train_v5_adaptive.main` | CLI: `python3 -m train_v3.train_v5_adaptive ...` |
| **Live self-play PPO** (Block A4 — the operational tournament runner) | `train_v3.rust_live_self_play.run_live_self_play_update` | programmatic (importable) |
| **BC fine-tune** (Block A2) | `train_v3.bc_train.run_bc_training` | programmatic |
| **League run — Block B** | `train_v3.block_b_league_driver.BlockBLeagueDriver.run` | programmatic |
| **League run — Block D** | `train_v3.block_d_league_driver.BlockDLeagueDriver.run` | programmatic |
| **RLHF offline replay — Block C** | `train_v3.c_loop_driver.CLoopDriver.run` | programmatic (needs rlhf_env) |
| **Tournament + ship — Block E1** | `train_v3.block_e1_runner.main` | CLI: `python3 -m train_v3.block_e1_runner ...` |

The recommended first end-to-end run is the **`train_v5_adaptive` CLI** (it wraps the
PPO trace-pool path end-to-end and also derives the aux datasets). The worked example
in §6 used exactly this entry on a tiny garbage config.

---

## 3. The main PPO entry — `train_v3.train_v5_adaptive`

CLI (`main()` at `train_v5_adaptive.py:96`). Exact flags + defaults:

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--config` | `Path` | (none) | trace-pool manifest config (JSON) to drive the run |
| `--output-dir` | `Path` | (required-ish) | where checkpoints + metrics + league manifest land |
| `--trace-pool-dir` | `Path` | (none) | prebuilt trace pool directory |
| `--trace-manifest-path` | `Path` | (none) | path to a trace-pool manifest JSON |
| `--aux-output-dir` | `Path` | (none) | where aux (assembler/desirerer) datasets are written |
| `--hidden-dim` | `int` | `256` | policy hidden dim |
| `--action-hidden-dim` | `int` | `128` | action-branch hidden dim |
| `--learning-rate` | `float` | `3e-4` | optimizer LR |
| `--policy-kind` | `str` | — | `v5_split_encoder` (the V5 model) or `baseline_mlp` (V4) |
| `--allow-empty-aux` | flag | False | don't fail if aux dataset ends up empty |

Internally it calls `run_v5_adaptive_training_pipeline(...)` (`:29`) and
`create_v5_default_model_optimizer(...)` (`:153`), which:
1. load (or generate) a V5 trace pool,
2. build a `V5ActionConditionedPolicy` via `create_v5_policy(policy_kind="v5_split_encoder", ...)`,
3. warm-start from V4-Max if available,
4. call `train_rust_ppo_trace_files(trace_paths, model, optimizer, config)`,
5. derive the deck-assembler + draw-desirerer aux datasets,
6. print a JSON result + write the per-update league manifest.

### 3.1 The PPO config — `RustPPOTrainingConfig`

`train_v3.rust_trainer.RustPPOTrainingConfig` (frozen dataclass, `rust_trainer.py:23`).
The PPO-relevant fields + defaults:

| Field | Default | Meaning |
|---|---|---|
| `run_name` | (required) | run identifier |
| `model_name` | `"extra-lr-v5-adaptive"` | model name in artifacts |
| `updates` | `1` | number of PPO updates |
| `env_count` | `16` | parallel vec-env count |
| `steps_per_update` | `30` | env steps collected per update |
| `gamma` | `0.99` | discount |
| `gae_lambda` | `0.95` | GAE lambda |
| `epochs` | `3` | PPO epochs per update |
| `minibatch_size` | `256` | minibatch size |
| `clip_epsilon` | `0.2` | PPO clip |
| `value_coef` | `0.5` | value loss coef |
| `entropy_coef` | `0.01` | entropy coef (Phase-A pins this) |
| `observation_mode` | `"v5_only"` | obs encoder mode |
| `action_features_mode` | `"legal_only"` | candidate-action features mode |
| `policy_scoring_backend` | `"padded"` | backend for scoring logits |
| `checkpoint_dir` | (required) | where `.npz` checkpoints are written |
| `checkpoint_every` | `1` | checkpoint cadence (updates) |
| `metrics_path` | (required) | JSON metrics output path |
| `seed` | (required) | RNG seed |

> For the **Phase-A hardened config**, use `train_v3.ppo_phaseA_config.PhaseAPPOConfig`
> (`:308`) — a superset of `RustPPOTrainingConfig` carrying the 5 Phase-A root-cause
> fixes (learner-only reward, `max_turns=120`, pinned `entropy_coef=0.01`, `epochs=6`,
> graduated opponent_mix, second-start oversampling). `to_rust_ppo_config()` converts it
> back to the plain config. Prefer `PhaseAPPOConfig` for real training; the plain config
> is for smoke tests.

### 3.2 What gets written

`rust_trainer._save_checkpoint` (`:802`) writes, each `checkpoint_every` updates:
```
{checkpoint_dir}/trainv3_rust_legal_update_{update:04d}.npz
```
via `ai.train_v2.model_mlx.save_checkpoint`. The `.npz` is the V5 policy weights
(split encoders + candidate_scorer + value_head + mana_draw_head). Each update also
appends a metrics JSON line + a per-update league manifest carrying
`opponent_mix`, `v5_mode`, `adaptive_strength` (`_v5_league_update_metadata`, `:535`).

---

## 4. The live self-play entry — Block A4 (`rust_live_self_play`)

For a real league run you don't replay a fixed trace pool — you roll out live against a
mix of opponents on the Rust `ArenaEnv`. The entry is
`run_live_self_play_update(config, learner_policy, opponent_policies=None, *, seed,
library_path, model, optimizer, ...)` (`rust_live_self_play.py:1002`).

- It builds a `RustBatchWorker.from_live(seed=..., env_count=..., max_turns=120, ...)`
  (the live self-play constructor, default modes `v5_only`/`legal_only`).
- Opponent dispatch is split: **7 rule-agent identities** via
  `RustBatchWorker.select_rule_actions` (fast, FFI-side), **4 policy-opponent identities**
  (end_turn / greedy_face / self / v4max) via the Python loop, plus the Block-B
  `v4-orig-*` temperature-spectrum opponents.
- The PPO optimizer step is **MLX-gated**: skipped if `model`/`optimizer` is `None`
  (so the same function serves both "collect rollouts for evaluation" and "train").
- `collect_rust_live_rollout` (`:548`) is the per-update collector.

This is the function the Block-B and Block-D league drivers call every update
(`block_b_league_driver.BlockBLeagueDriver.run` → `_build_reweighted_mix` →
`run_live_self_play_update`).

---

## 5. League drivers (Block B / Block D)

A "league" = a loop that, every update, builds an opponent mix, runs live self-play,
optionally snapshots the learner, and runs a promotion gate. Both drivers are
**importable only** (no CLI) — you instantiate and call `.run(n_updates)`.

**Block B** (`block_b_league_driver.py`):
```python
from train_v3.block_b_league_driver import BlockBLeagueDriver
driver = BlockBLeagueDriver(
    config=<RustPPOTrainingConfig or PhaseAPPOConfig>,
    worker_factory=lambda: RustBatchWorker.from_live(...),
    model=<MLX V5 policy>, optimizer=<MLX optimizer>,
    snapshot_pool=<SnapshotPool>, gauntlet_runner=<GameRunner>, ...,
)
manifest = driver.run(n_updates=...)
# manifest: BlockBLeagueManifest — exits to C2 when B7 plateau-below-dominance fires
```
Per update: `_build_reweighted_mix` (B3 frozen-non-self 0.95 mix + B4 curriculum cap 0.25
+ self-snapshot split) → `run_live_self_play_update` → `curriculum.update` → every
`snapshot_cadence`: `_snapshot_step` (B1 add → B6 gate → B1 best-ever → B7 plateau check).

**Block D** (`block_d_league_driver.py`): `BlockDLeagueDriver(BlockBLeagueDriver)`
overrides `_build_reweighted_mix` (D1 consolidation mix: self+v5_snapshot 0.50,
V4-orig 0.30, exploit 0.15, tail 0.05) and `run()` (copies the B8 loop inline).
`exit_mode="fixed_schedule"` (default, B7 exit suppressed) or `"plateau"`. Emits a
`BlockDLeagueManifest` with **`exited_to_e1` + `candidate_paths` + `aggregate_history`**
— the **D→E1 handoff** you feed to the E1 runner.

---

## 6. A worked end-to-end smoke (the field-test recipe)

This is the actual 8-step smoke from `TrainV3.5/BLOCK_E1_FIELDTEST.md` — it proves the
whole chain runs live. Quality is irrelevant; the point is to exercise every machine.

```bash
# 0. Build Rust
cargo build --release --manifest-path TrainV3.5/rust/trainv3_core/Cargo.toml
export TRAINV3_CORE_LIB="$PWD/TrainV3.5/target/release/libtrainv3_core.dylib"
export PYTHONPATH="$PWD:$PWD/TrainV3.5/python"
```

```python
# 1. Bootstrap a garbage V5 policy + checkpoint (obs_dim 7128 verified)
from train_v3.v5_policy import create_v5_policy
from ai.train_v2.model_mlx import save_checkpoint
policy = create_v5_policy(policy_kind="v5_split_encoder")   # default dims
save_checkpoint(policy, "/tmp/v5_garbage_bootstrap.npz")
```

```python
# 2. REAL PPO TRAIN on the Rust ArenaEnv via FFI + MLX
#    (the heart of "обучи мусорную модельку")
from train_v3.rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_files
config = RustPPOTrainingConfig(
    run_name="smoke_v5_garbage",
    updates=1, env_count=8, steps_per_update=30, epochs=3,
    observation_mode="v5_only", action_features_mode="legal_only",
    checkpoint_dir="TrainV3.5/runs/smoke_v5_garbage/checkpoints",
    metrics_path="TrainV3.5/runs/smoke_v5_garbage/metrics.json",
    seed=0,
)
# (or just: python3 -m train_v3.train_v5_adaptive --output-dir ... --policy-kind v5_split_encoder)
# -> writes checkpoints/trainv3_rust_legal_update_0001.npz
```

```python
# 3. E1 EXPORT + E2 GUARD
from train_v3.export_onnx_v5 import export_v5_checkpoint_to_onnx
from train_v3.v5_inference_guard import _assert_v5_logits_finite_legal
onnx_path = export_v5_checkpoint_to_onnx(
    "TrainV3.5/runs/smoke_v5_garbage/checkpoints/trainv3_rust_legal_update_0001.npz",
    "TrainV3.5/runs/smoke_v5_garbage/garbage.onnx",
    opset=17, placement_mode="append_only",
)
# sidecar .onnx.json: model_version="v5_split_encoder_onnx_v1",
#                     mana_draw_head=True, format="v5", obs_dim=7128
# run onnxruntime on a real TrainV3ClassicEnv obs -> logits(1,601)/value/mana_draw_logit
# _assert_v5_logits_finite_legal -> a legal candidate int
```

```python
# 4. E3 TOURNAMENT  (use a real GameRunner for a real run; fake here)
from train_v3.e1_tournament import run_e1_tournament, select_e1_winner, E1TournamentConfig
reports = run_e1_tournament(cfg, game_runner=<GameRunner>, candidate_loader=<loader>)
winner = select_e1_winner(reports)   # None when no candidate passes -> NO-SHIP (healthy for garbage)
```

```python
# 5. E5 SHIP  (only when winner is not None AND winner.passed())
from train_v3.e1_ship import ship_v5_winner, ShipResult
result = ship_v5_winner(
    winner_report=<a passing E1CandidateReport>,
    onnx_export_fn=export_v5_checkpoint_to_onnx,
    bundle_config=<ReleaseBundleConfig with candidate.json + profile_overlay.json + acceptance_gate.json>,
)
# ShipResult: marker="extra-lr-v5-max", onnx+sidecar+bundle+manifest paths,
#             fallback_guard_verified=True, 4 top tiers retargeted,
#             LIFO V5 detector at registry index 0.
# ship_v5_winner raises RuntimeError on None / passed=False  (NO-SHIP is an error at ship time)
```

```python
# 6. E-E12 RUNNER (composition; returns None for NO-SHIP, does NOT raise)
from train_v3.block_e1_runner import run_e1_pipeline
ship = run_e1_pipeline(
    manifest=<BlockDLeagueManifest with candidate_paths>,
    game_runner=<GameRunner>, candidate_loader=<loader>,
    c2_client=<rlhf_env MCP client or None>, scorecard_client=<ReviewerScorecardClient or None>,
    mana_draw_baseline=<ManaDrawBaseline>,
    bundle_config=<ReleaseBundleConfig>,
    min_reviewers=3, min_battles=10, run_panel=True,
)
```

```python
# 7. PROD-LOAD  (ai.bot_brain.BerserkInference with the V5 profile)
from ai.bot_brain import BerserkInference
profiles = {"extra-lr-v5-max": {
    "model_path": "<garbage.onnx>", "format": "v5",
    "obs_dim": 7128, "action_feature_dim": 171, "max_candidate_actions": 601,
    "mana_draw_head": True, "placement_mode": "append_only", "verify_mask": False,
    "temperature_range": [0.0, 0.0], "selection": "argmax",
}}
brain = BerserkInference(profiles=profiles)
action = brain.get_action(game_state=<real GameState>, player_id=1)
# -> a legal action index; encode_observation_v5 + 3-output session.run + guard +
#    mana_draw parallel-binary-head wiring all run on the live hot path.
```

All 8 steps passed in the field test. Artifacts land in gitignored
`TrainV3.5/runs/smoke_v5_garbage/`.

---

## 7. Export → Ship → Prod-deploy (the real path)

1. **Export.** `export_v5_checkpoint_to_onnx(checkpoint, output, *, opset=17,
   placement_mode=None)` (or CLI `python3 -m train_v3.export_onnx_v5 --checkpoint <npz>
   --output <onnx> [--opset 17] [--placement-mode append_only]`). 3-output ONNX
   (logits / value / mana_draw_logit) + a `.onnx.json` sidecar.
2. **Tournament.** `run_e1_tournament` against the v4-max gauntlet + exploit agents
   (real `rust_live_self_play` GameRunner, USER-operational per E-E4/E-E12). The
   threshold table (`E1CandidateReport.passed()`) is the gate — see `ARCHITECTURE.md` §4.6.
3. **Ship.** `ship_v5_winner(winner_report, *, onnx_export_fn, bundle_config)` — raises
   `RuntimeError` if `winner_report` is `None` or `.passed()` is False. Builds the release
   bundle, registers the LIFO V5 kind detector, verifies the committed prod wiring.
4. **Prod-deploy.** Load `BERSERK_BRAIN` with the `extra-lr-v5-max` profile
   (`infrastructure/config.py`); the ONNX fallback guard (`ai/bot_brain.py:_get_action_v5`,
   SPEC design.md:174) is **last-resort** — a malformed V5 ONNX raises `RuntimeError` on
   the live path, it is NOT a silent rule-based fallback.

> The operational RUN (D-league → E1 tournament with real factories → prod deploy) is
> **USER-run**. `block_e1_runner.build_production_*` stubs raise `NotImplementedError`
> precisely because the real GameRunner / model loader / rlhf_env client / scorecard
> client are USER-supplied.

---

## 8. Troubleshooting

- **`resolve_library_path` can't find `libtrainv3_core.dylib`** → run `cargo build
  --release` in `TrainV3.5/rust/trainv3_core`, then `export TRAINV3_CORE_LIB=...` to the
  produced `.dylib`.
- **Export tensor-size mismatch (`1456 vs <hidden_dim>`) when exporting a trained
  checkpoint** → the exporter's torch reconstruction expects the **default** architecture
  dims. Train with the default `--hidden-dim 256 --action-hidden-dim 128` (or, for a pure
  smoke, export the bootstrap default-dim checkpoint). Future fix: read dims from
  checkpoint metadata (not yet done — flagged in the field-test log).
- **`ship_v5_winner` raises `RuntimeError: NO-SHIP`** → the candidate did not pass the E1
  tournament gate. This is correct behavior: the gate is a hard external-bench gate, not a
  loss-based one. Inspect the `E1CandidateReport` to see which of the ~15 criteria failed.
- **`build_release_bundle` raises `FileNotFoundError: candidate.json`** → you skipped
  `write_candidate_json`. The runner does this for you; if calling `ship_v5_winner`
  directly, write the candidate.json first (`block_e1_runner.write_candidate_json`).
- **V5 ONNX misclassified as `action_onnx` in prod** → the LIFO V5 kind detector
  (`e1_ship.register_v5_kind_detector`) must be registered BEFORE the V4 sidecar
  detector. It is idempotent; re-registering is a no-op.
- **Tests run from a worktree report `FAIL: not found` for `rlhf_env/sessions`** → path
  artifact: HTTP smoke scripts check CWD while the 8090 server runs from the original
  repo. Validate Block-C code with in-process pytest (`test_*.py` via `_v5_helpers` /
  `MCPServer._tool`), not the HTTP smoke scripts. (See project memory
  `rlhf-smoke-scripts-worktree-path-mismatch`.)
- **Promotion seemed to follow the training loss** → it didn't. `a_gate.select_promotion`
  and `e1_tournament.select_e1_winner` record internal metrics but **never read them**;
  only external-bench H2H score rates decide promotion (the promotion-by-loss guard).

---

## 9. Where to look next

- `ARCHITECTURE.md` — full module inventory, contracts, prod wiring, invariants.
- `TrainV3.5/BLOCK_E1_FIELDTEST.md` — the live end-to-end proof this guide's §6 is drawn from.
- `TrainV3.5/BLOCK_*_PLAN.md` / `BLOCK_*_COMPLETION.md` — per-block design + delivery logs.
- `TrainV3.5/python/train_v3/tests/` — 24 component tests; the fastest way to see each
  entry's exact call shape is to read the corresponding `test_<component>.py`.