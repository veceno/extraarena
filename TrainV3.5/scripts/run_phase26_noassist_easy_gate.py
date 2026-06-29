#!/usr/bin/env python3
"""Phase A/26: easy no-assist runtime-opponent gate for V5.

This layer is intentionally narrow: V5 acts only on its own turns, while Rust
rule/exploit agents play the opponent side until the turn returns to V5. The
trace pool supplies clean no-assist initial states, not opponent behavior.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3" / "python"))

from train_v3.rust_collector import RustLegalActionFeatures, RustTransitionBatch
from train_v3.rust_policy import make_padded_legal_argmax_policy
from train_v3.rust_ppo import prepare_rust_ppo_batch, train_rust_ppo_minibatch
from train_v3.rust_vec_env import RustVecEnv
from train_v3.trace_factory_v5 import V5TraceScenario, generate_v5_trace_pool, resolve_v5_trace_paths
from train_v3.v5_artifacts import read_manifest_json, write_manifest_json
from train_v3.v5_policy import create_v5_policy


DEFAULT_DECK_POOL: tuple[tuple[int, ...], ...] = (
    (1, 37, 38, 40, 41, 42, 27, 28, 29),
    (1, 40, 43, 29, 31, 25, 27, 37, 28),
    (1, 32, 26, 16, 40, 33, 46, 28, 29),
    (1, 17, 23, 19, 8, 42, 28, 44, 43),
    (1, 32, 18, 45, 28, 16, 46, 38, 36),
    (1, 20, 42, 31, 21, 13, 27, 28, 38),
    (1, 41, 45, 31, 21, 40, 39, 43, 15),
)

RULE_AGENT_CODES: dict[str, int] = {
    "legal_random": 0,
    "face_rush": 1,
    "board_control": 2,
    "greedy_trade": 3,
    "stall": 4,
    "punish_empty_board": 5,
    "anti_draw_greed": 6,
    "anti_hand_leak_overfit": 7,
}
RULE_AGENT_NAMES: dict[int, str] = {code: name for name, code in RULE_AGENT_CODES.items()}

DEFAULT_OPPONENT_MIX = (
    "legal_random:0.55,"
    "face_rush:0.14,"
    "board_control:0.12,"
    "greedy_trade:0.10,"
    "stall:0.04,"
    "punish_empty_board:0.03,"
    "anti_draw_greed:0.01,"
    "anti_hand_leak_overfit:0.01"
)


def main() -> int:
    run_name = _env_str("PHASE26_RUN_NAME", "phase26_noassist_easy_gate")
    phase_label = _env_str("PHASE26_PHASE", "phase_a_noassist_easy_gate")
    env_count = _env_int("PHASE26_ENV_COUNT", 8192)
    steps_per_update = _env_int("PHASE26_STEPS_PER_UPDATE", 16)
    updates = _env_int("PHASE26_UPDATES", 1000)
    minibatch_size = _env_int("PHASE26_MINIBATCH_SIZE", 8192)
    checkpoint_every = _env_int("PHASE26_CHECKPOINT_EVERY", 25)
    hidden_dim = _env_int("PHASE26_HIDDEN_DIM", 256)
    action_hidden_dim = _env_int("PHASE26_ACTION_HIDDEN_DIM", 128)
    learning_rate = _env_float("PHASE26_LR", 2.0e-4)
    entropy_coef = _env_float("PHASE26_ENTROPY_COEF", 0.035)
    clip_epsilon = _env_float("PHASE26_CLIP_EPSILON", 0.16)
    max_grad_norm = _env_optional_float("PHASE26_MAX_GRAD_NORM", 0.5)
    seed = _env_int("PHASE26_SEED", 26001)
    trace_seed_count = _env_int("PHASE26_TRACE_SEED_COUNT", 8)
    deck_pool = _env_deck_pool("PHASE26_DECK_POOL", DEFAULT_DECK_POOL)
    deck_pair_mode = _env_str("PHASE26_DECK_PAIR_MODE", "cycle")
    opponent_mix = parse_opponent_mix(_env_str("PHASE26_OPPONENT_MIX", DEFAULT_OPPONENT_MIX))
    max_opponent_actions = _env_int("PHASE26_MAX_OPPONENT_ACTIONS", 96)
    policy_padding_mode = _env_str("PHASE26_POLICY_PADDING_MODE", "single")
    ppo_minibatch_plan = _env_str("PHASE26_PPO_MINIBATCH_PLAN", "contiguous")
    resume_checkpoint = _env_path("PHASE26_RESUME_CHECKPOINT")
    resume_optimizer_policy = _env_str("PHASE26_RESUME_OPTIMIZER_POLICY", "force_lr").lower()
    if resume_optimizer_policy not in {"restore", "force_lr", "reset"}:
        raise ValueError("PHASE26_RESUME_OPTIMIZER_POLICY must be restore, force_lr, or reset")
    reuse_trace_manifest_path = _env_path("PHASE26_TRACE_MANIFEST_PATH")
    out_root = Path(os.environ.get("PHASE26_OUT_ROOT", ROOT / "TrainV3" / "runs")).resolve()
    library_path = Path(
        os.environ.get("TRAINV3_CORE_LIB", ROOT / "TrainV3" / "target" / "release" / "libtrainv3_core.dylib")
    ).resolve()

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = run_dir / "trace_pool"
    manifest_path = reuse_trace_manifest_path or (run_dir / "trace_manifest.json")
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    latest_phase_file = out_root / "latest_phase26_noassist_easy_gate_run.txt"
    latest_general_file = out_root / "latest_trainv3_training_run.txt"
    latest_phase_file.write_text(str(run_dir) + "\n", encoding="utf-8")
    latest_general_file.write_text(str(run_dir) + "\n", encoding="utf-8")

    deck_pairs = build_deck_pairs(deck_pool, mode=deck_pair_mode)
    config_snapshot = {
        "phase": phase_label,
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_name": "extra-lr-v5-adaptive",
        "env_count": env_count,
        "steps_per_update": steps_per_update,
        "updates": updates,
        "minibatch_size": minibatch_size,
        "checkpoint_every": checkpoint_every,
        "hidden_dim": hidden_dim,
        "action_hidden_dim": action_hidden_dim,
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "clip_epsilon": clip_epsilon,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "trace_seed_count": trace_seed_count,
        "trace_manifest_path": str(manifest_path),
        "trace_manifest_reused": reuse_trace_manifest_path is not None,
        "adaptive_strength": 1.0,
        "assist_policy": "off",
        "draw_assist_policy": "off",
        "private_info_policy": "enemy_hidden_only",
        "runtime_opponent_mode": "rust_rule_fast_forward",
        "runtime_opponents": opponent_mix,
        "max_opponent_actions": max_opponent_actions,
        "deck_pair_mode": deck_pair_mode,
        "deck_pair_count": len(deck_pairs),
        "deck_pool": [list(deck) for deck in deck_pool],
        "policy_padding_mode": policy_padding_mode,
        "ppo_minibatch_plan": ppo_minibatch_plan,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_optimizer_policy": resume_optimizer_policy,
        "library_path": str(library_path),
        "clean_room_noassist": True,
        "contaminated_prior_data_excluded": True,
        "v4_1_included": False,
    }
    (run_dir / "phase26_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE26_RUN_DIR", run_dir, flush=True)
    print("PHASE26_CONFIG", json.dumps(config_snapshot, sort_keys=True), flush=True)

    if reuse_trace_manifest_path is None:
        manifest = generate_v5_trace_pool(
            build_trace_scenarios(
                seed=seed,
                trace_seed_count=trace_seed_count,
                deck_pairs=deck_pairs,
            ),
            trace_dir,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        write_manifest_json(manifest, manifest_path)
    manifest_data = read_manifest_json(manifest_path)
    trace_paths = resolve_v5_trace_paths(manifest_path)
    print(
        "PHASE26_TRACE_MANIFEST",
        manifest_path,
        manifest_data["manifest_id"],
        "traces",
        len(trace_paths),
        flush=True,
    )

    model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=hidden_dim,
        action_hidden_dim=action_hidden_dim,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    resume_metadata: dict[str, Any] = {}
    optimizer_restored = False
    optimizer_lr_before_resume = optimizer_learning_rate(optimizer)
    if resume_checkpoint is not None:
        from ai.train_v2.model_mlx import load_checkpoint

        if resume_optimizer_policy == "reset":
            loaded = load_checkpoint(str(resume_checkpoint), model, optimizer=None)
            optimizer = optim.Adam(learning_rate=learning_rate)
            optimizer_restored = False
        else:
            loaded = load_checkpoint(str(resume_checkpoint), model, optimizer=optimizer)
            optimizer_restored = bool(loaded.get("optimizer_restored", False))
            if resume_optimizer_policy == "force_lr" and optimizer_restored:
                force_optimizer_learning_rate(optimizer, learning_rate)
        resume_metadata = dict(loaded.get("metadata", {}))
        print(
            "PHASE26_RESUME",
            json.dumps(
                {
                    "checkpoint": str(resume_checkpoint),
                    "optimizer_policy": resume_optimizer_policy,
                    "optimizer_restored": optimizer_restored,
                    "optimizer_lr_before_resume": optimizer_lr_before_resume,
                    "optimizer_lr_after_resume": optimizer_learning_rate(optimizer),
                    "source_update": resume_metadata.get("update"),
                    "source_run_name": resume_metadata.get("run_name"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    config_snapshot["optimizer_learning_rate_after_resume"] = optimizer_learning_rate(optimizer)
    (run_dir / "phase26_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    policy = make_padded_legal_argmax_policy(
        model,
        padding_backend="rust",
        selection_backend="rust",
        padding_mode=policy_padding_mode,
        profile_policy=False,
        library_path=library_path,
    )

    started = time.perf_counter()
    total_transitions = 0
    checkpoint_path = ""
    metrics: list[dict[str, Any]] = []
    with RustVecEnv.from_trace_files(
        trace_paths,
        env_count=env_count,
        library_path=library_path,
        observation_key="observation_v5",
        auto_reset=True,
        action_features_dtype="float32",
        action_features_mode="legal_only",
        reset_pool_mode="cycle",
        observation_mode="v5_only",
        action_mask_mode="legal_only",
        terminal_observation_mode="none",
        diagnostic_mode="full",
    ) as env:
        for update in range(1, updates + 1):
            agent_codes = sample_agent_codes(env_count, opponent_mix, seed=seed, update=update)
            collect_t0 = time.perf_counter()
            transitions, rollout_metrics = collect_phase26_rollout(
                env,
                policy,
                steps=steps_per_update,
                agent_codes=agent_codes,
                max_opponent_actions=max_opponent_actions,
                salt=(seed * 1_000_003 + update * 9_176),
            )
            collect_seconds = time.perf_counter() - collect_t0

            prepare_t0 = time.perf_counter()
            ppo_batch = prepare_rust_ppo_batch(
                transitions,
                gamma=0.99,
                gae_lambda=0.95,
                advantage_backend="rust",
                selected_local_backend="provided",
                prepare_backend="separate",
                library_path=library_path,
            )
            prepare_seconds = time.perf_counter() - prepare_t0

            update_t0 = time.perf_counter()
            update_metrics = train_rust_ppo_minibatch(
                model,
                optimizer,
                ppo_batch,
                epochs=1,
                minibatch_size=minibatch_size,
                clip_epsilon=clip_epsilon,
                value_coef=0.5,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                shuffle=False,
                seed=seed + update,
                legal_row_pack_backend="auto",
                full_batch_eval=False,
                minibatch_plan=ppo_minibatch_plan,
                library_path=library_path,
            )
            update_seconds = time.perf_counter() - update_t0

            env_transitions = env_count * steps_per_update
            total_transitions += env_transitions
            metric = {
                "update": update,
                "phase": phase_label,
                "run_name": run_name,
                "model_name": "extra-lr-v5-adaptive",
                "trace_manifest_id": manifest_data["manifest_id"],
                "v5_mode": {
                    "adaptive_strength": 1.0,
                    "own_hand_identity_known": True,
                    "own_deck_known": True,
                    "enemy_hand_known": False,
                    "enemy_deck_known": False,
                    "enemy_deck_order_known": False,
                    "draw_assist_enabled": False,
                    "draw_assist_strength": 0.0,
                },
                "assist_mode": {
                    "assembler_enabled": False,
                    "assembler_strength": 0.0,
                    "desirerer_enabled": False,
                    "desirerer_strength": 0.0,
                    "teacher_hint_available": False,
                    "assist_profile_id": 0,
                },
                "runtime_opponent_mode": "rust_rule_fast_forward",
                "runtime_opponents": opponent_mix,
                "selected_trace_count": len(trace_paths),
                "env_transitions": env_transitions,
                "total_env_transitions": total_transitions,
                "collect_seconds": collect_seconds,
                "prepare_seconds": prepare_seconds,
                "update_seconds": update_seconds,
                "policy_padding_mode": policy_padding_mode,
                "ppo_minibatch_plan": ppo_minibatch_plan,
                "diagnostic_mode": "full",
                **rollout_metrics,
                **update_metrics,
            }
            append_jsonl(metrics_path, metric)
            metrics.append(metric)
            print("PHASE26_METRIC", json.dumps(compact_metric(metric), sort_keys=True), flush=True)

            if checkpoint_every > 0 and update % checkpoint_every == 0:
                checkpoint_path = save_phase26_checkpoint(
                    run_dir,
                    model,
                    optimizer,
                    update=update,
                    total_env_transitions=total_transitions,
                    config=config_snapshot,
                    metric=metric,
                    trace_manifest_id=str(manifest_data["manifest_id"]),
                    resume_metadata=resume_metadata,
                    optimizer_restored=optimizer_restored,
                )

    elapsed = time.perf_counter() - started
    summary = {
        "type": "summary",
        "status": "ok",
        "phase": phase_label,
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "trace_manifest_id": manifest_data["manifest_id"],
        "updates": updates,
        "env_count": env_count,
        "steps_per_update": steps_per_update,
        "total_env_transitions": total_transitions,
        "elapsed_seconds": elapsed,
        "end_to_end_transitions_per_second": total_transitions / elapsed if elapsed > 0 else 0.0,
        "checkpoint_path": checkpoint_path,
        "metrics_path": str(metrics_path),
        "runtime_opponents": opponent_mix,
        "clean_room_noassist": True,
        "max_rss_mb": rss_mb(),
    }
    append_jsonl(metrics_path, summary)
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE26_RUN_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def collect_phase26_rollout(
    env: RustVecEnv,
    policy: Any,
    *,
    steps: int,
    agent_codes: np.ndarray,
    max_opponent_actions: int,
    salt: int,
) -> tuple[RustTransitionBatch, dict[str, Any]]:
    reset = env.reset(copy=False, include_infos=False)
    learner_actor_ids = env.current_actor_ids()
    env_count = int(env.env_count)
    obs_shape = reset.observations.shape[1:]
    observations = np.empty((steps, env_count, *obs_shape), dtype=reset.observations.dtype)
    legal_action_counts = np.empty((steps, env_count), dtype=reset.legal_action_counts.dtype)
    legal_action_offsets = np.empty((steps, env_count), dtype=reset.legal_action_offsets.dtype)
    legal_tape = _LegalTape(
        ids_dtype=reset.legal_action_ids.dtype,
        features_dtype=reset.legal_action_features.dtype,
        feature_shape=reset.legal_action_features.shape[1:],
        initial_capacity=int(reset.legal_action_ids.shape[0]) * steps,
    )
    actions = np.empty((steps, env_count), dtype=np.uintp)
    rewards = np.empty((steps, env_count), dtype=np.float32)
    learner_reward_tape = np.empty((steps, env_count), dtype=np.float32)
    opponent_reward_tape = np.empty((steps, env_count), dtype=np.float32)
    terminated = np.empty((steps, env_count), dtype=np.bool_)
    reset_flags = np.empty((steps, env_count), dtype=np.bool_)
    opponent_action_count_tape = np.empty((steps, env_count), dtype=np.uint16)
    values = np.empty((steps, env_count), dtype=np.float32)
    log_probs = np.empty((steps, env_count), dtype=np.float32)
    selected_local_indices = np.empty((steps, env_count), dtype=np.int32)

    current_obs = reset.observations
    current_counts = reset.legal_action_counts
    current_offsets = reset.legal_action_offsets
    current_ids = reset.legal_action_ids
    current_features = reset.legal_action_features
    total_opponent_actions = 0
    terminal_count = 0
    reset_count = 0
    learner_reward_sum = 0.0
    opponent_reward_sum = 0.0

    for step_idx in range(steps):
        observations[step_idx] = current_obs
        legal_action_counts[step_idx] = current_counts
        legal_action_offsets[step_idx] = current_offsets + legal_tape.size
        legal_tape.append(current_ids, current_features)
        policy_features = RustLegalActionFeatures(
            counts=current_counts,
            offsets=current_offsets,
            ids=current_ids,
            features=current_features,
        )
        policy_out = policy(current_obs, None, policy_features)
        action_ids, step_values, step_log_probs, step_selected = parse_policy_output(policy_out)
        action_ids = validate_vector(action_ids, env_count, np.uintp, "actions")
        actions[step_idx] = action_ids
        values[step_idx] = validate_vector(step_values, env_count, np.float32, "values")
        log_probs[step_idx] = validate_vector(step_log_probs, env_count, np.float32, "log_probs")
        selected_local_indices[step_idx] = validate_vector(
            step_selected,
            env_count,
            np.int32,
            "selected_local_indices",
        )

        learner_step = env.step(action_ids, copy=False, include_infos=False, include_truncated=False)
        learner_rewards = np.array(learner_step.rewards, dtype=np.float32, copy=True)
        learner_terminal = np.array(learner_step.terminated, dtype=np.bool_, copy=True)
        learner_reset = (
            learner_terminal.copy()
            if learner_step.reset_flags is None
            else np.array(learner_step.reset_flags, dtype=np.bool_, copy=True)
        )
        if learner_reset.any():
            actor_after_learner = env.current_actor_ids()
            learner_actor_ids[learner_reset] = actor_after_learner[learner_reset]

        opponent_raw = env.advance_rule_until_actor(
            learner_actor_ids,
            agent_codes,
            max_actions_per_env=max_opponent_actions,
            salt=salt + step_idx * 65_537,
            copy=False,
        )
        opponent_rewards = np.array(opponent_raw["rule_learner_rewards"], dtype=np.float32, copy=True)
        opponent_terminal = np.array(opponent_raw["rule_terminated"], dtype=np.bool_, copy=True)
        opponent_reset = np.array(opponent_raw["rule_reset_flags"], dtype=np.bool_, copy=True)
        opponent_counts = np.array(opponent_raw["rule_action_counts"], dtype=np.uintp, copy=True)
        if opponent_reset.any():
            actor_after_opponent = env.current_actor_ids()
            learner_actor_ids[opponent_reset] = actor_after_opponent[opponent_reset]

        step_rewards = learner_rewards + opponent_rewards
        step_terminated = learner_terminal | opponent_terminal
        step_reset = learner_reset | opponent_reset
        rewards[step_idx] = step_rewards
        learner_reward_tape[step_idx] = learner_rewards
        opponent_reward_tape[step_idx] = opponent_rewards
        terminated[step_idx] = step_terminated
        reset_flags[step_idx] = step_reset
        opponent_action_count_tape[step_idx] = np.minimum(opponent_counts, np.iinfo(np.uint16).max)
        total_opponent_actions += int(opponent_counts.sum())
        terminal_count += int(step_terminated.sum())
        reset_count += int(step_reset.sum())
        learner_reward_sum += float(learner_rewards.sum())
        opponent_reward_sum += float(opponent_rewards.sum())

        current_obs = opponent_raw["observation_v5"]
        current_counts = opponent_raw["legal_action_counts"]
        current_offsets = opponent_raw["legal_action_offsets"]
        current_ids = opponent_raw["legal_action_ids"]
        current_features = opponent_raw["legal_action_features"]

    legal_ids, legal_features = legal_tape.finish()
    transition_batch = RustTransitionBatch(
        observations=observations,
        next_observations=None,
        action_mask=None,
        action_features=None,
        legal_action_counts=legal_action_counts,
        legal_action_offsets=legal_action_offsets,
        legal_action_ids=legal_ids,
        legal_action_features=legal_features,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=None,
        reset_flags=reset_flags,
        terminal_observations=None,
        terminal_observation_valid=None,
        episode_returns=None,
        episode_lengths=None,
        infos=None,
        values=values,
        log_probs=log_probs,
        selected_local_indices=selected_local_indices,
    )
    rows = env_count * steps
    lane_metrics = build_lane_rollout_metrics(
        agent_codes=agent_codes,
        rewards=rewards,
        learner_rewards=learner_reward_tape,
        opponent_rewards=opponent_reward_tape,
        terminated=terminated,
        opponent_action_counts=opponent_action_count_tape,
    )
    rollout_metrics = {
        "rollout_rows": rows,
        "runtime_opponent_actions": total_opponent_actions,
        "runtime_opponent_actions_per_transition": total_opponent_actions / rows,
        "terminal_rate": terminal_count / rows,
        "reset_rate": reset_count / rows,
        "mean_reward": float(rewards.mean()),
        "mean_learner_action_reward": learner_reward_sum / rows,
        "mean_opponent_response_reward": opponent_reward_sum / rows,
        "mean_legal_actions": float(legal_action_counts.mean()),
        "stored_legal_feature_bytes": int(legal_features.nbytes),
        "runtime_opponent_lane_metrics": lane_metrics,
    }
    return transition_batch, rollout_metrics


def build_lane_rollout_metrics(
    *,
    agent_codes: np.ndarray,
    rewards: np.ndarray,
    learner_rewards: np.ndarray,
    opponent_rewards: np.ndarray,
    terminated: np.ndarray,
    opponent_action_counts: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Cheap per-lane telemetry for repair runs.

    `agent_codes` is fixed for the env slots during one update, so these
    aggregates are vectorized over the rollout matrix and do not touch the Rust
    step hot path.
    """
    codes = np.asarray(agent_codes, dtype=np.uint32)
    flat_rewards = np.asarray(rewards, dtype=np.float32)
    flat_learner_rewards = np.asarray(learner_rewards, dtype=np.float32)
    flat_opponent_rewards = np.asarray(opponent_rewards, dtype=np.float32)
    flat_terminal = np.asarray(terminated, dtype=np.bool_)
    flat_opponent_counts = np.asarray(opponent_action_counts, dtype=np.float32)
    if flat_rewards.ndim != 2 or flat_rewards.shape[1] != codes.shape[0]:
        raise ValueError("rewards must have shape (steps, env_count) for lane metrics")
    for name, arr in (
        ("learner_rewards", flat_learner_rewards),
        ("opponent_rewards", flat_opponent_rewards),
        ("terminated", flat_terminal),
        ("opponent_action_counts", flat_opponent_counts),
    ):
        if arr.shape != flat_rewards.shape:
            raise ValueError(f"{name} must match rewards shape for lane metrics")
    out: dict[str, dict[str, float | int]] = {}
    for code in sorted(int(value) for value in np.unique(codes)):
        name = RULE_AGENT_NAMES.get(code, f"unknown_{code}")
        mask = codes == code
        if not bool(mask.any()):
            continue
        lane_rewards = flat_rewards[:, mask]
        lane_learner_rewards = flat_learner_rewards[:, mask]
        lane_opponent_rewards = flat_opponent_rewards[:, mask]
        lane_terminal = flat_terminal[:, mask]
        lane_opponent_counts = flat_opponent_counts[:, mask]
        transitions = int(lane_rewards.size)
        env_slots = int(mask.sum())
        out[name] = {
            "env_slots": env_slots,
            "transitions": transitions,
            "mean_reward": float(lane_rewards.mean()) if transitions else 0.0,
            "mean_learner_action_reward": float(lane_learner_rewards.mean()) if transitions else 0.0,
            "mean_opponent_response_reward": float(lane_opponent_rewards.mean()) if transitions else 0.0,
            "opponent_actions_per_transition": float(lane_opponent_counts.mean()) if transitions else 0.0,
            "terminal_rate": float(lane_terminal.mean()) if transitions else 0.0,
        }
    return out


class _LegalTape:
    def __init__(
        self,
        *,
        ids_dtype: np.dtype,
        features_dtype: np.dtype,
        feature_shape: tuple[int, ...],
        initial_capacity: int,
    ):
        self._ids_dtype = np.dtype(ids_dtype)
        self._features_dtype = np.dtype(features_dtype)
        self._feature_shape = tuple(feature_shape)
        self._ids = np.empty((max(1, int(initial_capacity)),), dtype=self._ids_dtype)
        self._features = np.empty((max(1, int(initial_capacity)), *self._feature_shape), dtype=self._features_dtype)
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def append(self, ids: Any, features: Any) -> None:
        ids_arr = np.asarray(ids, dtype=self._ids_dtype)
        features_arr = np.asarray(features, dtype=self._features_dtype)
        end = self._size + int(ids_arr.shape[0])
        if end > self._ids.shape[0]:
            new_capacity = max(end, self._ids.shape[0] * 2)
            new_ids = np.empty((new_capacity,), dtype=self._ids_dtype)
            new_features = np.empty((new_capacity, *self._feature_shape), dtype=self._features_dtype)
            new_ids[: self._size] = self._ids[: self._size]
            new_features[: self._size] = self._features[: self._size]
            self._ids = new_ids
            self._features = new_features
        self._ids[self._size:end] = ids_arr
        self._features[self._size:end] = features_arr
        self._size = end

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        return self._ids[: self._size], self._features[: self._size]


def parse_policy_output(policy_out: Any) -> tuple[Any, Any, Any, Any]:
    if not isinstance(policy_out, dict):
        raise ValueError("Phase26 policy must return a dict")
    return (
        policy_out["actions"],
        policy_out["values"],
        policy_out["log_probs"],
        policy_out["selected_local_indices"],
    )


def validate_vector(value: Any, env_count: int, dtype: Any, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=dtype)
    if out.shape != (env_count,):
        raise ValueError(f"{name} must have shape ({env_count},), got {out.shape}")
    return out


def optimizer_learning_rate(optimizer: Any) -> float | None:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict) or "learning_rate" not in state:
        return None
    try:
        return float(np.asarray(state["learning_rate"]).item())
    except Exception:
        return None


def force_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict):
        raise ValueError("optimizer does not expose a mutable state dict")
    state["learning_rate"] = mx.array(float(learning_rate), dtype=mx.float32)


def save_phase26_checkpoint(
    run_dir: Path,
    model: Any,
    optimizer: Any,
    *,
    update: int,
    total_env_transitions: int,
    config: dict[str, Any],
    metric: dict[str, Any],
    trace_manifest_id: str,
    resume_metadata: dict[str, Any],
    optimizer_restored: bool,
) -> str:
    from ai.train_v2.model_mlx import save_checkpoint

    checkpoint_path = run_dir / "checkpoints" / f"trainv3_rust_legal_update_{update:04d}.npz"
    metadata = {
        "trainv3_phase": config.get("phase", "phase_a_noassist_easy_gate"),
        "trainv3_rust_legal_only": True,
        "run_name": config["run_name"],
        "model_name": config["model_name"],
        "trace_manifest_id": trace_manifest_id,
        "rollout_mode": "rust_rule_opponent_macro_step",
        "runtime_opponents": config["runtime_opponents"],
        "update": update,
        "total_env_transitions": total_env_transitions,
        "obs_dim": int(getattr(model, "obs_dim", 6480)),
        "action_feature_dim": int(getattr(model, "action_feature_dim", 171)),
        "max_candidate_actions": 601,
        "config": config,
        "last_metrics": metric,
        "resume_source_update": resume_metadata.get("update", 0),
        "resume_optimizer_restored": optimizer_restored,
        "optimizer_learning_rate": optimizer_learning_rate(optimizer),
    }
    save_checkpoint(str(checkpoint_path), model, optimizer=optimizer, metadata=metadata)
    return str(checkpoint_path)


def build_trace_scenarios(
    *,
    seed: int,
    trace_seed_count: int,
    deck_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> list[V5TraceScenario]:
    seeds = tuple(int(seed) + 126_000 + idx * 17 for idx in range(int(trace_seed_count)))
    visibility = (
        {
            "own_hand_identity_known": True,
            "own_deck_known": True,
            "enemy_hand_known": False,
            "enemy_deck_known": False,
            "enemy_deck_order_known": False,
        },
    )
    draw = ({"draw_assist_enabled": False, "draw_assist_strength": 0.0},)
    assist = (
        {
            "assembler_enabled": False,
            "assembler_strength": 0.0,
            "desirerer_enabled": False,
            "desirerer_strength": 0.0,
            "teacher_hint_available": False,
            "assist_profile_id": 0,
        },
    )
    level_modes = ({"p1_level": 1, "p2_level": 1, "label": "equal_l1"},)
    scenarios: list[V5TraceScenario] = []
    for pair_idx, (p1_deck, p2_deck) in enumerate(deck_pairs):
        for choose in ("first", "last"):
            scenarios.append(
                V5TraceScenario(
                    scenario_key=f"phase26_easy_noassist_pair_{pair_idx:03d}_{choose}",
                    seeds=seeds,
                    steps=12,
                    p1_deck_ids=tuple(p1_deck),
                    p2_deck_ids=tuple(p2_deck),
                    adaptive_strengths=(1.0,),
                    visibility_modes=visibility,
                    draw_assist_modes=draw,
                    assist_modes=assist,
                    level_modes=level_modes,
                    choose=choose,
                )
            )
    return scenarios


def build_deck_pairs(
    deck_pool: tuple[tuple[int, ...], ...],
    *,
    mode: str,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    pool = tuple(tuple(deck) for deck in deck_pool)
    if not pool:
        raise ValueError("deck_pool must not be empty")
    mode = str(mode).strip().lower()
    if mode == "mirror":
        return tuple((deck, deck) for deck in pool)
    if mode == "cycle":
        pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for idx, deck in enumerate(pool):
            other = pool[(idx + 1) % len(pool)]
            pairs.append((deck, deck))
            pairs.append((deck, other))
            pairs.append((other, deck))
        return tuple(dedupe_pairs(pairs))
    if mode == "ordered":
        return tuple((p1, p2) for p1 in pool for p2 in pool)
    raise ValueError("deck_pair_mode must be mirror, cycle, or ordered")


def dedupe_pairs(
    pairs: Iterable[tuple[tuple[int, ...], tuple[int, ...]]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return tuple(out)


def parse_opponent_mix(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, value = chunk.partition(":")
        key = name.strip()
        if key not in RULE_AGENT_CODES:
            raise ValueError(f"unknown Phase26 opponent {key!r}")
        weight = float(value.strip() or "1")
        if weight < 0:
            raise ValueError("opponent weights must be non-negative")
        weights[key] = weights.get(key, 0.0) + weight
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("opponent mix must have positive total weight")
    return {key: value / total for key, value in sorted(weights.items())}


def sample_agent_codes(env_count: int, opponent_mix: dict[str, float], *, seed: int, update: int) -> np.ndarray:
    names = tuple(opponent_mix)
    probabilities = np.array([opponent_mix[name] for name in names], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(int(seed) + int(update) * 1009)
    selected = rng.choice(len(names), size=int(env_count), p=probabilities)
    codes = np.array([RULE_AGENT_CODES[name] for name in names], dtype=np.uint32)
    return codes[selected]


def compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "update",
        "collect_seconds",
        "prepare_seconds",
        "update_seconds",
        "runtime_opponent_actions_per_transition",
        "terminal_rate",
        "mean_reward",
        "entropy",
        "approx_kl",
        "loss",
        "mean_legal_actions",
    )
    return {key: metric.get(key) for key in keys if key in metric}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _env_deck_pool(name: str, default: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    decks: list[tuple[int, ...]] = []
    for chunk in raw.split(";"):
        deck = tuple(int(part.strip()) for part in chunk.split(",") if part.strip())
        if len(deck) < 2:
            raise ValueError(f"{name} contains a deck with fewer than two ids")
        decks.append(deck)
    if not decks:
        raise ValueError(f"{name} must contain at least one deck")
    return tuple(decks)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value is None or not value.strip() else value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default if value is None or not value.strip() else value)


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if raw > 10_000_000 else raw / 1024


if __name__ == "__main__":
    raise SystemExit(main())
