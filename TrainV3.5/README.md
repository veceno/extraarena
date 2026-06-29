# TrainV3 Environment Preparation

TrainV3 is a training-only workspace for fifth-generation Extra-LR bot environment work.

The production bot/runtime remains Python-first and authoritative. Rust code in this directory is for rollout
acceleration, parity experiments, and future V5 environment research only. Production bot modules such as
`ai/bot_brain.py`, `core/engine.py`, and web/server runtime code must not import from `TrainV3`.

## Current Scope

- Keep Python `core/engine.py` and TrainV2 codecs as the oracle.
- Reduce low-risk Python bottlenecks in the current TrainV2 training environment.
- Prepare a Rust-owned coarse-grained training worker boundary.
- Provide golden trace tooling so future Rust behavior can be compared against Python state/mask/feature hashes.
- Define the V5 environment observation surface: adaptive-strength flags, optional private hand/deck identity information,
  action history tape, and reward component metadata.

## Rust Boundary

The intended Rust ownership is a batched training rollout kernel:

- compact training state mirror;
- action-id based `step(action_id)`;
- action masks and candidate action features;
- observation/private-info/history tensor assembly;
- reward components;
- coarse FFI that returns tensors and compact summaries.

The boundary should avoid per-card Python object transfer and should not expose Rust as production battle authority.

The first implemented Rust kernel slice is `trainv3_core::kernel`: it parses Python golden-trace snapshots and
rebuilds the hot training tensors in one coarse call:

- legal action mask;
- no-preview candidate action features;
- `classic_obs_v1`;
- V5 private-info/history observation;
- V5 reward components for pre/post snapshots.
- `apply_action(action_id)` transitions for parity-driven rollout work:
  `end_turn`, no-target warrior/potion play, targeted damage, taunt-constrained attacks, unit combat cleanup,
  basic deathrattle AOE cleanup, simple damage/heal/draw effects, and basic attacks.

This is deliberately still Python-oracle driven: Python generates traces from the authoritative environment, and Rust
must match the trace hashes before it can be trusted as an acceleration path.

The first batched boundary is `BatchedRolloutWorker` in `trainv3_core::worker`. It owns many mirrored env states,
applies a batch of action ids, maintains V5 history internally, and returns flat tensor buffers for observations,
masks, action features, rewards, and termination flags.

Python can access this boundary through the training-only `ctypes` wrapper `train_v3.rust_ffi.RustBatchWorker`.
Build the Rust dynamic library first, then point `TRAINV3_CORE_LIB` at it.
The wrapper caches successful library path resolution and loaded `ctypes.CDLL` handles, so repeated Rust helper calls
reuse the same prototype setup instead of re-checking/reloading the dynamic library on every GAE, legal-padding, or
argmax call.

For script-level training experiments, `train_v3.rust_rollout.RustTraceRolloutRunner` wraps that FFI worker and
replays a golden-trace action script across many mirrored Rust env states without spawning the Rust CLI process.
The worker now supports both full-batch `reset()` and selective `reset_indices([...])`, which is the shape needed
for vectorized training loops where only completed env slots should be restarted.
For PPO-style rollout collection, `step_auto_reset(action_ids)` applies one action per env slot and automatically
resets slots that reached terminal status. Its returned `terminated` flags describe the transition that just
finished, while `reset_flags` marks which observations have already been replaced with reset observations.
The same output includes `episode_returns` and `episode_lengths`: for terminal auto-reset slots those are the final
episode stats, while subsequent `encode()` calls show the reset slot's counters back at zero.
`terminal_observation_v1` / `terminal_observation_v5` preserve the final pre-reset observation for slots where
`terminal_observation_valid` is true; non-terminal slots receive zero-filled terminal observation rows.

## Layout

```text
TrainV3/
|-- Cargo.toml
|-- rust/trainv3_core/
|   |-- Cargo.toml
|   `-- src/
|-- python/train_v3/
|   |-- __init__.py
|   |-- contracts.py
|   |-- env_v5.py
|   `-- golden_trace.py
`-- README.md
```

## Local Checks

```bash
cargo test --manifest-path TrainV3/Cargo.toml
python3 -m pytest tests/test_train_v3_golden_trace.py tests/test_train_v3_env_v5.py
```

Build the Rust dynamic library for Python FFI:

```bash
cargo build --release --manifest-path TrainV3/Cargo.toml
```

On macOS, the library is usually:

```bash
export TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib
```

Then smoke-test the Python FFI wrapper:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib \
  python3 -m pytest tests/test_train_v3_rust_ffi.py
```

Smoke-test the Python rollout runner and get an FFI-level throughput number:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
from train_v3.rust_rollout import benchmark_trace_file

print(benchmark_trace_file(
    "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
    env_count=64,
    iterations=1000,
))
PY
```

`benchmark_trace_file(..., reuse_worker=True, coarse_rollout=True)` is the default path and keeps one Rust worker alive,
calling `reset()` between scripted rollouts. Pass `reuse_worker=False` only when you explicitly want to measure worker
construction and trace parsing overhead. The default coarse mode sends the whole scripted action tape through one FFI
call; Rust applies all `(steps, env_count)` action ids internally and returns one time-major tensor batch. The benchmark
runner precomputes that repeated env action tape once, so repeated `replay_coarse()` calls do not allocate it again. The
benchmark also defaults to the PPO trainer's lean output contract:
`action_features_mode="legal_only", observation_mode="v5_only", action_mask_mode="legal_only",
terminal_observation_mode="none"`. Pass `coarse_rollout=False` when you explicitly want the older per-action FFI loop as
a debug oracle or overhead comparison; the runner reuses cached per-action vectors for scripted trace ids, so that path
does not allocate `np.full(env_count, action_id)` on every step. Pass `action_features_mode="dense_and_legal",
observation_mode="v1_and_v5", action_mask_mode="dense", terminal_observation_mode="full"` when you need the full parity
tensor surface.

For vectorized rollout experiments, call `RustBatchWorker.step(action_ids)` with one action id per env slot, then
call `reset_indices(done_indices)` for the subset that should restart. The returned arrays are contiguous Rust-owned
views unless `copy=True` is passed.

For scripted PPO batch parity work, `RustBatchWorker.rollout_action_tape_pre_step(action_ids)` records the observation,
mask, and legal feature rows before each action is applied, while rewards and done flags still describe the resulting
transition. Pass that output plus the same action tape to
`train_v3.rust_collector.transition_batch_from_action_tape_rollout(...)` to build a `RustTransitionBatch` that can feed
`prepare_rust_ppo_batch` once policy values and log-probabilities are attached.
The Rust `BatchTensorOutput` now preallocates the fixed-size V5 observation buffer and pre-step selected-local tape for
the full `(steps * env_count)` output, avoiding geometric `Vec` growth while filling coarse rollout batches.

If the collector does not need terminal observations yet, use `RustBatchWorker.step_auto_reset(action_ids)` to keep
episode restarts inside Rust and avoid an extra Python-side reset pass. The current coarse boundary returns rewards,
`terminated`, `reset_flags`, `episode_returns`, `episode_lengths`, observations, masks, and candidate action features
as flat contiguous buffers. It also returns zero-filled terminal observation buffers plus validity flags, so callers
can keep the reset observation as `next_obs` while still logging or bootstrapping from the final pre-reset state.

For collector integration without writing FFI glue directly, use `train_v3.rust_vec_env.RustVecEnv`. It wraps the same
worker with `reset()` and `step(action_ids)` methods, returning observation batches, rewards, done flags, masks,
candidate action features, and per-env `infos` with `episode` and `terminal_observation` entries when available.

For PPO-style fixed-horizon collection, `train_v3.rust_collector.collect_rust_vec_rollout(env, policy, steps=N)` sits
one layer higher. The policy receives `(observations, action_mask, action_features)` and returns action ids, optionally
with value estimates and log-probabilities. The collector returns tensors shaped `(steps, env_count, ...)` for
pre-step observations, masks, candidate action features, action ids, rewards, done flags, reset flags, reset-aware
`next_observations`, terminal observations, and episode return/length summaries.
The collector now writes global legal-action offsets while appending each Rust-provided legal tape slice, avoiding a
separate Python `cumsum` pass over `(steps, env_count)` counts after rollout collection.
When the policy returns values, log-probs, and selected-local indices from the first step, the collector allocates those
stat tensors lazily without a full-buffer `nan`/`-1` fill pass before overwriting every row.
For scripted golden-trace rollouts where each mirrored env takes the same action on a step,
`RustTraceRolloutRunner.replay_coarse(...)` uses `trainv3_worker_rollout_broadcast_action_ids` and sends only the
1D step action ids to Rust, avoiding the Python `np.repeat` that used to materialize a full `(steps, env_count)` action
tape. `RustBatchWorker.rollout_action_tape(...)` and `rollout_action_tape_pre_step(...)` also accept those 1D scripted
ids directly and route them to the Rust broadcast kernels; pass a 2D array only for true per-env action tapes.
`transition_batch_from_action_tape_rollout(...)` accepts the same 1D scripted actions as a broadcast view.

To reduce rollout-batch memory, pass `action_features_dtype="float16"` to `RustBatchWorker.from_trace_file(...)`,
`RustVecEnv.from_trace_file(...)`, `RustTraceRolloutRunner.from_trace_file(...)`, or `benchmark_trace_file(...)`.
In this mode Rust keeps half-precision FFI feature buffers and Python exposes `action_features` and
`legal_action_features` as `np.float16`, while observations, masks, rewards, and terminal observations remain
`np.float32` / bool / integer tensors. `RustPPOTrainingConfig` exposes the same `action_features_dtype` switch, but keeps
`"float32"` as the default because local smoke timings showed half precision reducing collector bytes while slowing the
current MLX update path. Use `"float16"` when rollout transfer/storage pressure dominates on the target setup.

For v5-only training, pass `observation_mode="v5_only"` to `RustBatchWorker` or `RustVecEnv`. Rust still encodes
`observation_v5`, whose prefix contains the v1 features, but it omits the separate `observation_v1` and
`terminal_observation_v1` FFI buffers. `RustPPOTrainingConfig` defaults to this lean observation mode when training
from `observation_v5`.

For policies that only score legal candidates, pass `action_features_mode="legal_only"` to `RustBatchWorker` or
`RustVecEnv`. Rust still returns the dense legal mask, but omits the dense `(env_count, 601, 171)` action-feature
tensor and exposes only:

- `legal_action_counts`: legal candidate count per env;
- `legal_action_offsets`: flat legal-tape offset per env, emitted by Rust with the counts;
- `legal_action_ids`: flat action ids for those legal candidates;
- `legal_action_features`: flat `(sum(legal_action_counts), 171)` feature rows.

`collect_rust_vec_rollout(..., use_compact_action_features=True, store_dense_action_features=False)` sends a
`RustLegalActionFeatures` object to the policy and stores the flat legal feature tape plus per-step offsets, without
allocating a dense action-feature rollout batch.

If the policy and PPO update consume only those compact legal tensors, also pass `action_mask_mode="legal_only"` and
`store_dense_action_mask=False`. This omits the dense `(env_count, 601)` mask buffer while preserving
`legal_action_counts` and `legal_action_ids`; `RustPPOTrainingConfig` uses this lean mask mode by default.

For PPO runs that do not bootstrap or log terminal observations, pass `terminal_observation_mode="none"` and
`store_terminal_observations=False`. Rust still returns `terminated`, `reset_flags`, episode return/length summaries,
and terminal-valid flags, but omits the large zero-filled terminal observation buffers; `RustPPOTrainingConfig` defaults
to this mode.

If PPO preparation only needs pre-step observations and the action/reward tape, pass
`store_next_observations=False` to skip the second `(steps, env_count, obs_dim)` Python buffer. The trainer defaults
to this no-next-observation storage path because `prepare_rust_ppo_batch` does not consume `next_observations`.

For pure PPO updates that do not log per-step reset/episode diagnostics, pass `store_truncated=False`,
`store_reset_flags=False`, and `store_episode_stats=False`. This skips Python-side storage for the all-false
`truncated` tape plus `reset_flags`, `terminal_observation_valid`, `episode_returns`, and `episode_lengths`;
`RustPPOTrainingConfig` defaults to this lean diagnostic-storage path and resolves `diagnostic_mode="auto"` to a
Rust worker `diagnostic_mode="none"`, so reset/episode diagnostics are not filled in Rust output buffers either.
For example, a local `env_count=8`, `steps=12`
collector smoke skipped 1056 bytes of diagnostic arrays per rollout (`11 * steps * env_count`) while preserving PPO
preparation. `collect_rust_vec_rollout(..., store_truncated=False)` also calls `RustVecEnv.step(...,
include_truncated=False)`, so the all-false `truncated` array is not stored in the lean path. When callers do request
`include_truncated=True`, `RustVecEnv` reuses a read-only all-false vector instead of allocating one per env step.
The pre-step action-tape converter defaults `store_truncated=False` because Rust rollouts do not emit truncated flags and
the all-false tape is synthetic; it also exposes `store_reset_flags=False` and `store_episode_stats=False` switches, and
the collector/coarse/update benchmark helpers create Rust workers with `diagnostic_mode="none"` where those diagnostics
are not consumed. They report the skipped diagnostic bytes so the measured fast path matches the trainer's lean PPO
payload.
When diagnostics are enabled and callers use `copy=False`, the Python FFI wrapper views Rust `u8` flag buffers
(`terminated`, `reset_flags`, `terminal_observation_valid`) as NumPy `bool_` arrays without an `astype` copy; `copy=True`
still materializes independent arrays. The same zero-copy bool view is used for `compute_rust_pad_legal_actions(...)`
`legal_mask`, so the padded legal policy path no longer copies the Rust `u8` mask before handing it to MLX.
When a rollout batch has `truncated=None`, Rust GAE and fused PPO preparation now accept a null truncated pointer and treat
it as all-false, so `prepare_rust_ppo_batch(...)` and minibatch slicing no longer materialize that synthetic tape in the
default trainer path. When `terminated` or non-null `truncated` tapes are contiguous NumPy `bool_` arrays, the FFI also
views them as Rust `u8` flag buffers without copying before `trainv3_compute_gae` or `trainv3_prepare_ppo_batch`.
When `bootstrap_values=None`, the same GAE and fused prepare wrappers pass a null bootstrap pointer and Rust treats the
final-step bootstrap values as zeros; `prepare_rust_ppo_batch(..., advantage_backend="rust")` preserves that `None`
through both separate and fused Rust prepare modes, avoiding a per-prepare Python zero-vector allocation in the lean path.

If the rollout consumer reads episode summaries from tensor arrays instead of Gym-style info dicts, pass
`RustVecEnv.step(..., include_infos=False)` or `collect_rust_vec_rollout(..., store_infos=False)`. This avoids building
and copying per-env Python dictionaries in the hot path; `RustPPOTrainingConfig` defaults to no info storage.

`prepare_rust_ppo_batch(..., advantage_backend="rust")` computes PPO GAE advantages and returns through the standalone
Rust FFI kernel `trainv3_compute_gae`, using the Python implementation as the parity oracle. `RustPPOTrainingConfig`
defaults to this Rust advantage backend so training avoids Python loops over `(steps, env_count)` during PPO
preparation; pass `advantage_backend="python"` when debugging the oracle path directly. The synthetic
`benchmark_rust_gae_prepare(...)` batch now builds its action-id tape as `uintp` and reuses that memory for
`legal_action_ids`, so the measured Rust prepare modes do not include an artificial `int64` to `uintp` copy.

`prepare_rust_ppo_batch(..., selected_local_backend="rust")` also precomputes the selected action's local index inside
each compact legal-action row through `trainv3_select_local_indices`. Legal-only PPO evaluation reuses that cache instead
of scanning the legal action id tape in Python on every full-batch and minibatch loss evaluation.

Padded and compact argmax collector policies now also return `selected_local_indices`, and
`collect_rust_vec_rollout(...)` stores them on `RustTransitionBatch`. For experiments that want to reuse the policy's
already-known local index instead of recomputing it during batch preparation, use
`prepare_rust_ppo_batch(..., selected_local_backend="provided", prepare_backend="separate")`. The
`benchmark_rust_gae_prepare(...)` helper reports this as `rust_provided`. Pre-step action-tape rollouts now also emit
`selected_local_indices` from the Rust worker in the same mask pass that encodes legal ids/features, so
`transition_batch_from_action_tape_rollout(...)` can feed the same provided-index prepare path without a second legal-id
scan in Rust or Python. The pre-step coarse benchmark defaults to `selected_local_backend="provided"` and
`prepare_backend="separate"` for that reason.
The action-id tape stays in NumPy `uintp` form through `transition_batch_from_action_tape_rollout(...)`,
`collect_rust_vec_rollout(...)`, and `prepare_rust_ppo_batch(...)`, so pre-step coarse rollouts no longer widen/copy
those ids to `int64` before the legal-only PPO path consumes the provided selected-local indices.
The selected-local validators on that provided path also compare against the existing `uintp` legal-count arrays directly,
avoiding an extra `uintp -> int64` counts copy during transition construction and PPO batch preparation.

`prepare_rust_ppo_batch(..., prepare_backend="rust_fused")` combines Rust GAE/returns and selected-local index
preparation in one FFI call through `trainv3_prepare_ppo_batch`. The generic helper keeps `prepare_backend="separate"`
as its compatibility default, and `RustPPOTrainingConfig` now also defaults to the separate
`selected_local_backend="provided"` path because the trainer's padded/compact policies already return that index. A local
smoke benchmark at `steps=512`, `env_count=64`, `iterations=160` measured `rust_provided` prepare at about 121.6M rows/s
and fused Rust prepare at about 104.6M rows/s, both with zero selected-index diff against the Python reference.

`score_padded_legal_actions(..., padding_backend="rust")` and `make_padded_legal_argmax_policy(...)` use
`trainv3_pad_legal_actions` to expand compact legal ids/features into the padded `(batch, max_legal, action_dim)` tensor
and mask before MLX model scoring. This keeps the model-facing legal-only path compact while avoiding a Python loop over
env rows for each policy/evaluation call. Padded and compact legal scorers now keep contiguous `legal_action_counts`
as NumPy `uintp`, and `evaluate_rust_ppo_batch(...)` passes `legal_action_counts`/`legal_action_offsets` through as
`uintp` views, so Rust padding and row-index kernels do not pay a `uintp -> int64 -> uintp` conversion in the default
legal-only PPO path. If a PPO batch lacks cached `selected_local_indices`, evaluation now computes that mapping with
`trainv3_selected_local_indices` instead of scanning legal ids in Python.

`make_padded_legal_argmax_policy(..., selection_backend="rust")` then uses `trainv3_padded_argmax_actions` to select
action ids and log-probs from the padded logits in Rust. This removes the per-env Python loop from policy-driven rollout
collection; `RustPPOTrainingConfig` defaults to this Rust selection backend.

`make_compact_legal_argmax_policy(..., selection_backend="rust")` keeps rollout policy scoring on the flat compact legal
rows and uses `trainv3_compact_argmax_actions` for action/log-prob selection. `score_compact_legal_actions` also exposes
`row_index_backend="rust"` via `trainv3_repeat_row_indices` and now uses it by default, so compact scoring avoids NumPy
row-index expansion before MLX gathers state embeddings. Set `RustPPOTrainingConfig.policy_scoring_backend="compact"` to
benchmark or use this no-padding rollout policy path; the trainer default remains `"padded"` unless the compact scorer
wins on the target hardware/model shape.

Dense policy collector benchmarks can also pass `make_dense_argmax_policy(..., selection_backend="rust")`, which uses
`trainv3_dense_argmax_actions` to select masked 601-action logits and log-probs without a Python loop. If the dense mask
is already a contiguous NumPy `bool_` or `uint8` array, the FFI views it as the Rust `u8` mask buffer without allocating
the intermediate `(mask > 0)` array; the dense policy wrapper preserves those bool/u8 masks into the Rust selector
instead of widening them to `float32` first.

`train_rust_ppo_minibatch(..., legal_row_pack_backend="auto")` uses a contiguous-slice fast path for the trainer's
default `shuffle=False` minibatches, avoiding a legal-row pack call when the selected rows map to one contiguous legal
tape slice. For shuffled or non-contiguous minibatches, `auto` falls back to `trainv3_pack_legal_action_rows`, so
variable-length legal ids/features are still compacted in Rust instead of a Python loop. The internal `_take_flat_rows`
helper also defaults to this `auto` route, so direct minibatch slicing no longer falls back to Python packing unless
callers explicitly request `legal_row_pack_backend="python"`. A local smoke benchmark on
512-row minibatches measured contiguous `auto` at about 66.5k row-take calls/s versus about 6.9k calls/s for the Rust
packing path. The trainer also reuses one pre-flattened PPO batch view across minibatch slicing, avoiding repeated
`RustPPOBatch.flatten()` reshape/dict setup for each optimizer step. For the default unshuffled `auto` path it also uses
a direct row-range slice, so contiguous minibatches do not allocate or re-check index arrays. A local synthetic
row-slicing microbenchmark measured about 72.6k contiguous `_take_flat_rows` calls/s with the reused flat view versus
about 62.2k calls/s without it. The direct range path removes the remaining index-array overhead from the trainer loop;
the same synthetic setup measured about 99.6k row-range slices/s versus about 77.2k pre-flattened index slices/s. The
default no-shuffle range path now normalizes contiguous legal offsets through `trainv3_normalize_legal_offsets`, so it
also avoids the Python `cumsum` allocation that previously rebuilt zero-based minibatch offsets. PPO updates now plan
those contiguous minibatches once per update call and reuse the planned legal tape views across epochs, so repeated
epochs do not redo row-range slicing or Rust offset normalization for the same unshuffled minibatches. Update metrics
report `contiguous_minibatch_plan`, `planned_minibatches`, `planned_minibatch_reuses`, and
`planned_legal_action_rows` to make the fast path visible in benchmarks and trainer checkpoint metadata.
The contiguous-index detector itself now checks adjacent deltas instead of allocating an expected `np.arange` buffer.

PPO loss evaluation gathers selected action probabilities directly from the legal/dense probability matrix instead of
allocating a one-hot matrix and summing over candidates. A local MLX smoke benchmark on a `(4096, 32)` probability matrix
measured direct gather at about 11.1k calls/s versus about 6.6k calls/s for the one-hot extraction path.

`train_rust_ppo_minibatch(..., full_batch_eval=False)` skips the diagnostic full-batch loss pass before and after each
PPO update. Minibatch losses are still reported, but `loss_before` / `loss_after` are `None`. The low-level updater keeps
`full_batch_eval=True` by default for compatibility with benchmarks, while `RustPPOTrainingConfig` defaults to `False`
so real trainer runs avoid two extra full-batch model forward passes per update. A local post-warmup smoke run at
`updates=3`, `env_count=4`, `steps_per_update=4`, `epochs=2`, `minibatch_size=4` measured about 0.0114s/update with
full-batch eval disabled versus about 0.0179s/update with it enabled.

To avoid replaying one identical initial state in every env slot, use `RustBatchWorker.from_trace_files([...])` or
`RustVecEnv.from_trace_files([...])`. Rust parses the trace pool once, cycles the initial snapshots across env slots,
and still returns one coarse-grained contiguous tensor batch. Pass `reset_pool_mode="cycle"` to rotate the next pooled
snapshot into an env slot whenever it resets; the default `RustPPOTrainingConfig.trace_pool_reset_mode` uses that mode
for multi-trace training pools.

`train_rust_ppo_trace_files(...)` opens the Rust trace env once per training run and reuses it across updates. Each
rollout still starts from a worker reset inside `collect_rust_vec_rollout(...)`, but trace parsing and worker allocation
are no longer repeated for every update; result and metric payloads expose `trace_env_reuse=True` plus
`trace_env_open_seconds`. A local two-trace smoke run with 5 updates opened the env once in about 0.0024s; the old
per-update lifecycle would pay that open/parse cost every update. The trainer result, checkpoint metadata, and metrics
JSONL now also include a final `type="summary"` speed row with aggregate collect/prepare/update seconds, train-loop
env-transitions/s, wall-clock env-transitions/s, and phase time fractions so end-to-end training runs expose one
representative speed report instead of only per-update timings.

`collect_rust_vec_rollout(...)` stores compact legal ids/features through an amortized contiguous legal-tape builder
instead of appending one copied array per step and concatenating at the end. A synthetic `steps=512`, `legal_rows=96`,
`feature_dim=171` microbenchmark measured the builder at about 2.2x the old chunk/concatenate throughput for that
legal-tape assembly slice.

Compare dense and legal-only collector footprints with:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
from train_v3.rust_benchmark import benchmark_rust_vec_collector_modes

print(benchmark_rust_vec_collector_modes(
    "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
    env_count=16,
    steps=30,
    iterations=5,
))
PY
```

For MLX policies that only need legal candidate scores, `train_v3.rust_policy.score_compact_legal_actions(...)`
reuses the current `ActionConditionedPolicy` layers but scores only the flat legal rows. If a regular batch shape is
more useful for the backend, `score_padded_legal_actions(...)` scores `(env_count, max_legal, 171)` instead of
`(env_count, 601, 171)`. Compare dense 601-candidate inference with flat and padded legal-row inference using:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
from ai.train_v2.model_mlx import ActionConditionedPolicy
from train_v3.rust_benchmark import benchmark_compact_legal_policy_inference

model = ActionConditionedPolicy(obs_dim=1456, hidden_dim=64, action_hidden_dim=32)
print(benchmark_compact_legal_policy_inference(
    "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
    model,
    env_count=16,
    iterations=100,
))
PY
```

For PPO-style collection without dense action features, use the padded legal policy adapter:

```python
from ai.train_v2.model_mlx import ActionConditionedPolicy
import mlx.optimizers as optim
from train_v3.rust_collector import collect_rust_vec_rollout
from train_v3.rust_policy import make_padded_legal_argmax_policy
from train_v3.rust_ppo import (
    evaluate_rust_ppo_batch,
    prepare_rust_ppo_batch,
    train_rust_ppo_minibatch,
)
from train_v3.rust_vec_env import RustVecEnv

model = ActionConditionedPolicy(obs_dim=6480, hidden_dim=64, action_hidden_dim=32)
optimizer = optim.Adam(learning_rate=3e-4)
with RustVecEnv.from_trace_file(
    "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
    env_count=16,
    observation_key="observation_v5",
    action_features_mode="legal_only",
) as env:
    batch = collect_rust_vec_rollout(
        env,
        make_padded_legal_argmax_policy(model),
        steps=30,
        use_compact_action_features=True,
        store_dense_action_features=False,
    )
ppo_batch = prepare_rust_ppo_batch(
    batch,
    gamma=0.99,
    gae_lambda=0.95,
    advantage_backend="rust",
    selected_local_backend="rust",
)
flat = ppo_batch.flatten()
evaluation = evaluate_rust_ppo_batch(model, ppo_batch)
metrics = train_rust_ppo_minibatch(
    model,
    optimizer,
    ppo_batch,
    epochs=3,
    minibatch_size=256,
)
```

To compare that legal-only policy collector path against the dense model path:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
from ai.train_v2.model_mlx import ActionConditionedPolicy
from train_v3.rust_benchmark import benchmark_rust_vec_policy_collector_modes

model = ActionConditionedPolicy(obs_dim=6480, hidden_dim=64, action_hidden_dim=32)
print(benchmark_rust_vec_policy_collector_modes(
    "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
    model,
    env_count=16,
    steps=30,
    iterations=3,
))
PY
```

To compare the regular stepwise legal-only collector against the pre-step coarse action-tape batch bridge without MLX
policy/update cost:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
from train_v3.rust_benchmark import benchmark_rust_pre_step_action_tape_batch_modes

print(benchmark_rust_pre_step_action_tape_batch_modes(
    "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
    env_count=32,
    iterations=1000,
))
PY
```

The stepwise side of that comparison creates a contiguous per-step action vector without building a full repeated
`(steps, env_count)` action tape and uses the Rust selected-local mapper instead of a Python `flatnonzero` scan, while
the coarse side sends the 1D scripted action ids through the Rust broadcast pre-step rollout kernel.

To isolate Python-loop versus Rust-FFI PPO GAE preparation:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
from train_v3.rust_benchmark import benchmark_rust_gae_prepare

print(benchmark_rust_gae_prepare(
    steps=128,
    env_count=64,
    iterations=100,
))
PY
```

To compare rollout plus PPO optimizer updates for the dense and legal-only paths:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
import mlx.optimizers as optim
from ai.train_v2.model_mlx import ActionConditionedPolicy
from train_v3.rust_benchmark import benchmark_rust_ppo_update_modes

print(benchmark_rust_ppo_update_modes(
    "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
    model_factory=lambda: ActionConditionedPolicy(obs_dim=6480, hidden_dim=64, action_hidden_dim=32),
    optimizer_factory=lambda: optim.Adam(learning_rate=3e-4),
    env_count=16,
    steps=30,
    iterations=3,
    epochs=3,
    minibatch_size=256,
))
PY
```

To produce one JSON-safe speed report across collector, coarse action-tape, GAE/prepare, policy inference, policy
collector, and PPO update sections:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
import mlx.optimizers as optim
from ai.train_v2.model_mlx import ActionConditionedPolicy
from train_v3.rust_benchmark import benchmark_trainv3_speed_report

report = benchmark_trainv3_speed_report(
    "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
    model_factory=lambda: ActionConditionedPolicy(obs_dim=6480, hidden_dim=64, action_hidden_dim=32),
    optimizer_factory=lambda: optim.Adam(learning_rate=3e-4),
    env_count=16,
    steps=30,
    iterations=3,
    epochs=3,
    minibatch_size=256,
    output_path="TrainV3/rust_speed_report.json",
)
print(report["summary"])
PY
```

The PPO update path caches model-independent padded legal-action features and masks for unshuffled contiguous
minibatches only when `epochs > 1`, then reuses them across PPO epochs while still recomputing logits/values after each
optimizer step. Single-epoch updates keep the planned padded-cost metrics but report the cache as disabled. The update
metrics report padded legal-action expansion and cache savings:
`planned_padded_action_rows`, `planned_padding_waste_rows`, `planned_padding_expansion_ratio`,
`planned_padded_total_bytes`, `planned_recomputed_padded_total_bytes`, `padded_cache_builds`, `padded_cache_hits`,
`padded_cache_reuses`, `padded_cache_saved_padded_total_bytes`, `padded_cache_reuse_fraction`,
`padded_cache_hit_build_ratio`, and `padded_cache_saved_recomputed_fraction`. In the suite summary these are exposed
with `legal_planned_*` and `legal_padded_cache_*` prefixes so compact legal rows, padded expansion, and cache savings
can be compared in one report. `legal_epochs` and `legal_planned_minibatches` are included so the summary alone can
prove `saved_builds == planned_minibatches * (epochs - 1)` for representative cache-reuse runs.
The summary also includes `bottleneck_ranking`, `next_optimization_target`, and `next_optimization_action`, which rank
remaining sections by inverse measured speedup so the next optimization target is chosen from the report data.
`gae_prepare_scale` records rows per prepare call, the raw fused-Rust speedup, and a scale hint. Tiny synthetic GAE
runs automatically add a `gae_prepare_representative` section at a representative row count; the summary exposes
`gae_prepare_representative_scale` and uses that speedup for `bottleneck_ranking` when it is present.
`policy_collector` summary breaks dense/compact/padded collector modes into throughput, policy-byte savings, and
policy-call time fractions, so ranking can recommend `profile_policy_collector_policy_scoring` when model scoring
dominates the remaining collector cost.
The padded collector policy skips invalid-logit masking before Rust selection because selection and log-probability
calculation already respect per-env legal counts; public padded scoring keeps masking enabled by default.
Benchmark-only padded collector policies also emit `policy_padding_seconds`, `policy_model_seconds`, and
`policy_selection_seconds` so the report can split the remaining policy-scoring bottleneck before choosing the next
model-forward optimization.
For the standard linear `candidate_scorer`, padded legal scoring computes state/action scorer terms separately instead
of materializing the broadcasted `[state, action]` joint tensor; custom scorers fall back to the generic concat path.
Trainer configs may set `policy_scoring_backend="auto"` to resolve tiny smoke/debug rollouts to compact legal-row
scoring while keeping normal representative rollouts on padded scoring; metrics include
`resolved_policy_scoring_backend` so the chosen path is explicit.

For a minimal legal-only training run with metrics and checkpoint metadata:

```bash
TRAINV3_CORE_LIB=TrainV3/target/release/libtrainv3_core.dylib PYTHONPATH=TrainV3/python \
python3 - <<'PY'
import mlx.optimizers as optim
from ai.train_v2.model_mlx import ActionConditionedPolicy
from train_v3.rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_files

model = ActionConditionedPolicy(obs_dim=6480, hidden_dim=64, action_hidden_dim=32)
optimizer = optim.Adam(learning_rate=3e-4)
result = train_rust_ppo_trace_files(
    [
        "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json",
        "TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_seed123.json",
    ],
    model,
    optimizer,
    RustPPOTrainingConfig(
        updates=2,
        env_count=16,
        steps_per_update=30,
        checkpoint_dir="TrainV3/runs/smoke/checkpoints",
        metrics_path="TrainV3/runs/smoke/metrics.jsonl",
    ),
)
print(result["checkpoint_path"])
PY
```

## Golden Kernel CLI

Generate or refresh the parity fixture:

```bash
PYTHONPATH=TrainV3/python python3 -m train_v3.golden_trace --seed 123 --steps 3 --verify-mask false --choose first \
  > TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_seed123.json
```

Verify Rust tensor parity against that Python oracle trace:

```bash
cargo run --release --manifest-path TrainV3/Cargo.toml --bin trainv3_kernel -- verify \
  TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_seed123.json
```

Generate a scripted transition fixture:

```bash
PYTHONPATH=TrainV3/python python3 -m train_v3.golden_trace --seed 123 --steps 5 --verify-mask false \
  --p1-deck-ids 1,37,38,40,41,42,27,28,29 \
  --p2-deck-ids 1,37,38,40,41,42,27,28,29 \
  --action-ids 1,0,1,0,552 \
  > TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json
```

Measure the Rust tensor kernel:

```bash
cargo run --release --manifest-path TrainV3/Cargo.toml --bin trainv3_kernel -- bench \
  TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_seed123.json 20000
```

Measure scripted Rust transition + post-tensor encoding:

```bash
cargo run --release --manifest-path TrainV3/Cargo.toml --bin trainv3_kernel -- bench-step \
  TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json 20000
```

Measure batched worker-style rollout over many mirrored envs. This keeps one worker alive and resets it between
iterations, matching the intended vectorized training boundary more closely than rebuilding the worker every rollout:

```bash
cargo run --release --manifest-path TrainV3/Cargo.toml --bin trainv3_kernel -- bench-batch \
  TrainV3/rust/trainv3_core/tests/fixtures/golden_trace_scripted_basic.json 1000 64
```

Additional combat fixtures currently cover:

- `golden_trace_targeted_potion.json`
- `golden_trace_taunt_attack.json`
- `golden_trace_attack_cleanup.json`
- `golden_trace_deathrattle_cleanup.json`
