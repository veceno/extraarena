"""Benchmark helpers for training-facing Rust TrainV3 rollout paths."""
from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .rust_collector import (
    RustLegalActionFeatures,
    RustTransitionBatch,
    collect_rust_vec_rollout,
    transition_batch_from_action_tape_rollout,
)
from .rust_ffi import RustBatchWorker, compute_rust_selected_local_indices
from .rust_policy import (
    make_compact_legal_argmax_policy,
    make_dense_argmax_policy,
    make_padded_legal_argmax_policy,
    score_compact_legal_actions,
    score_padded_legal_actions,
)
from .rust_ppo import (
    prepare_rust_ppo_batch,
    train_dense_rust_ppo_minibatch,
    train_rust_ppo_minibatch,
)
from .rust_vec_env import RustVecEnv


def benchmark_compact_legal_policy_inference(
    path: str | Path,
    model,
    *,
    env_count: int,
    iterations: int,
    library_path: str | Path | None = None,
    observation_key: str = "observation_v1",
) -> dict[str, dict[str, float | int | str]]:
    """Compare dense 601-candidate model inference with compact legal-row inference."""
    import mlx.core as mx

    env_count = int(env_count)
    iterations = int(iterations)
    if env_count <= 0:
        raise ValueError("env_count must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    with RustBatchWorker.from_trace_file(path, env_count=env_count, library_path=library_path) as worker:
        batch = worker.encode(copy=True)

    if observation_key not in batch:
        raise ValueError(f"unknown observation_key {observation_key!r}")

    obs = mx.array(batch[observation_key])
    dense_features = mx.array(batch["action_features"])
    legal_features = mx.array(batch["legal_action_features"])
    legal_counts = batch["legal_action_counts"]
    legal_offsets = batch["legal_action_offsets"]
    legal_ids = batch["legal_action_ids"]
    dense_rows = env_count * batch["action_features"].shape[1]
    legal_rows = int(legal_counts.sum())
    padded_rows = env_count * int(legal_counts.max(initial=0))

    dense_start = time.perf_counter()
    for _ in range(iterations):
        _out = model(obs, dense_features)
        # V5 returns (logits, value, mana_draw_logit); baseline returns
        # (logits, value). The dense benchmark scores the 601 candidates only,
        # so drop any 3rd element. Indexing is robust to both arities.
        logits, values = _out[0], _out[1]
        mx.eval(logits, values)
    dense_elapsed = time.perf_counter() - dense_start

    compact_start = time.perf_counter()
    for _ in range(iterations):
        scores = score_compact_legal_actions(
            model,
            obs,
            legal_counts,
            legal_features,
            library_path=library_path,
        )
        mx.eval(scores.legal_logits, scores.values)
    compact_elapsed = time.perf_counter() - compact_start

    padded_start = time.perf_counter()
    for _ in range(iterations):
        padded_scores = score_padded_legal_actions(
            model,
            obs,
            legal_counts,
            legal_features,
            legal_action_offsets=legal_offsets,
            legal_action_ids=legal_ids,
            padding_backend="rust",
            library_path=library_path,
        )
        mx.eval(padded_scores.padded_logits, padded_scores.values)
    padded_elapsed = time.perf_counter() - padded_start

    return {
        "dense_float32": _policy_inference_stats(
            "dense_float32",
            env_count=env_count,
            iterations=iterations,
            scored_rows_per_iteration=dense_rows,
            elapsed_seconds=dense_elapsed,
            policy_scoring_backend="dense",
        ),
        "legal_only_float32": _policy_inference_stats(
            "legal_only_float32",
            env_count=env_count,
            iterations=iterations,
            scored_rows_per_iteration=legal_rows,
            elapsed_seconds=compact_elapsed,
            policy_scoring_backend="compact",
            row_index_backend="rust",
        ),
        "legal_padded_float32": _policy_inference_stats(
            "legal_padded_float32",
            env_count=env_count,
            iterations=iterations,
            scored_rows_per_iteration=padded_rows,
            elapsed_seconds=padded_elapsed,
            policy_scoring_backend="padded",
            padding_backend="rust",
        ),
    }


def benchmark_rust_vec_collector_modes(
    path: str | Path,
    *,
    env_count: int,
    steps: int,
    iterations: int,
    library_path: str | Path | None = None,
) -> dict[str, dict[str, float | int | str]]:
    """Compare dense and compact legal-only Rust collector paths.

    The benchmark deliberately uses a deterministic first-legal policy. It is
    meant to isolate rollout boundary overhead and stored batch footprint, not
    model inference speed.
    """
    env_count = int(env_count)
    steps = int(steps)
    iterations = int(iterations)
    if env_count <= 0:
        raise ValueError("env_count must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    trace_path = Path(path)
    dense = _benchmark_one_mode(
        trace_path,
        env_count=env_count,
        steps=steps,
        iterations=iterations,
        library_path=library_path,
        mode_name="dense_float32",
        action_features_mode="dense_and_legal",
        action_mask_mode="dense",
        use_compact_action_features=False,
        store_dense_action_features=True,
        policy=_first_legal_from_dense,
    )
    compact = _benchmark_one_mode(
        trace_path,
        env_count=env_count,
        steps=steps,
        iterations=iterations,
        library_path=library_path,
        mode_name="legal_only_float32",
        action_features_mode="legal_only",
        action_mask_mode="legal_only",
        use_compact_action_features=True,
        store_dense_action_features=False,
        policy=_first_legal_from_compact,
    )
    return {
        "dense_float32": dense,
        "legal_only_float32": compact,
    }


def benchmark_rust_pre_step_action_tape_batch_modes(
    path: str | Path,
    *,
    env_count: int,
    iterations: int,
    advantage_backend: str = "rust",
    selected_local_backend: str = "provided",
    prepare_backend: str = "separate",
    library_path: str | Path | None = None,
) -> dict[str, dict[str, float | int | str | bool]]:
    """Compare stepwise VecEnv collection with pre-step coarse action-tape batching."""
    env_count = int(env_count)
    iterations = int(iterations)
    if env_count <= 0:
        raise ValueError("env_count must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    trace_path = Path(path)
    action_ids = np.asarray(_action_ids_from_trace(trace_path), dtype=np.uintp)
    if action_ids.size == 0:
        raise ValueError("trace must contain at least one action")
    stat_shape = (int(action_ids.size), env_count)
    values = np.zeros(stat_shape, dtype=np.float32)
    log_probs = np.zeros(stat_shape, dtype=np.float32)

    stepwise = _benchmark_stepwise_scripted_action_tape_batch(
        trace_path,
        action_ids=action_ids,
        values=values,
        log_probs=log_probs,
        env_count=env_count,
        iterations=iterations,
        advantage_backend=advantage_backend,
        selected_local_backend=selected_local_backend,
        prepare_backend=prepare_backend,
        library_path=library_path,
    )
    coarse = _benchmark_pre_step_action_tape_batch(
        trace_path,
        action_ids=action_ids,
        values=values,
        log_probs=log_probs,
        env_count=env_count,
        iterations=iterations,
        advantage_backend=advantage_backend,
        selected_local_backend=selected_local_backend,
        prepare_backend=prepare_backend,
        library_path=library_path,
    )
    return {
        "stepwise_collector_legal_only": stepwise,
        "pre_step_action_tape_legal_only": coarse,
    }


def benchmark_rust_gae_prepare(
    *,
    steps: int,
    env_count: int,
    iterations: int,
    library_path: str | Path | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    selected_local_backend: str = "python",
) -> dict[str, dict[str, float | int | str]]:
    """Compare Python-loop, separate Rust, provided-index, and fused Rust PPO preparation."""
    steps = int(steps)
    env_count = int(env_count)
    iterations = int(iterations)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if env_count <= 0:
        raise ValueError("env_count must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    transitions = _synthetic_gae_transition_batch(steps, env_count)
    reference = prepare_rust_ppo_batch(
        transitions,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantages=normalize_advantages,
        advantage_backend="python",
        selected_local_backend=selected_local_backend,
    )
    transitions_with_selected = replace(transitions, selected_local_indices=reference.selected_local_indices)
    stats: dict[str, dict[str, float | int | str]] = {}
    modes = [
        ("python", transitions, "python", selected_local_backend, "separate"),
        ("rust", transitions, "rust", selected_local_backend, "separate"),
        ("rust_provided", transitions_with_selected, "rust", "provided", "separate"),
        ("rust_fused", transitions, "rust", "rust", "rust_fused"),
    ]
    for mode_name, mode_transitions, advantage_backend, local_backend, prepare_backend in modes:
        elapsed = 0.0
        last_batch = None
        for _ in range(iterations):
            start = time.perf_counter()
            last_batch = prepare_rust_ppo_batch(
                mode_transitions,
                gamma=gamma,
                gae_lambda=gae_lambda,
                normalize_advantages=normalize_advantages,
                advantage_backend=advantage_backend,
                selected_local_backend=local_backend,
                prepare_backend=prepare_backend,
                library_path=library_path,
            )
            elapsed += time.perf_counter() - start
        if last_batch is None:
            raise RuntimeError("benchmark did not prepare a PPO batch")
        rows = steps * env_count * iterations
        rows_per_prepare_call = steps * env_count
        selected_abs_diff = (
            0
            if last_batch.selected_local_indices is None or reference.selected_local_indices is None
            else int(np.max(np.abs(last_batch.selected_local_indices - reference.selected_local_indices)))
        )
        stats[mode_name] = {
            "advantage_backend": advantage_backend,
            "selected_local_backend": local_backend,
            "prepare_backend": prepare_backend,
            "steps": steps,
            "env_count": env_count,
            "iterations": iterations,
            "prepare_calls": iterations,
            "rows": rows,
            "total_rows": rows,
            "rows_per_prepare_call": rows_per_prepare_call,
            "elapsed_seconds": elapsed,
            "rows_per_second": float("inf") if elapsed <= 0 else rows / elapsed,
            "max_advantage_abs_diff": float(np.max(np.abs(last_batch.advantages - reference.advantages))),
            "max_return_abs_diff": float(np.max(np.abs(last_batch.returns - reference.returns))),
            "max_selected_local_abs_diff": selected_abs_diff,
        }
    return stats


def benchmark_rust_vec_policy_collector_modes(
    path: str | Path,
    model,
    *,
    env_count: int,
    steps: int,
    iterations: int,
    library_path: str | Path | None = None,
    observation_key: str = "observation_v5",
    policy_selection_backend: str = "rust",
) -> dict[str, dict[str, float | int | str | bool]]:
    """Compare dense and legal-only Rust collector paths including model policy calls."""
    env_count = int(env_count)
    steps = int(steps)
    iterations = int(iterations)
    if env_count <= 0:
        raise ValueError("env_count must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    trace_path = Path(path)
    dense = _benchmark_policy_collector_one_mode(
        trace_path,
        env_count=env_count,
        steps=steps,
        iterations=iterations,
        library_path=library_path,
        observation_key=observation_key,
        mode_name="dense_model_float32",
        action_features_mode="dense_and_legal",
        action_mask_mode="dense",
        use_compact_action_features=False,
        store_dense_action_features=True,
        policy=make_dense_argmax_policy(
            model,
            selection_backend=policy_selection_backend,
            library_path=library_path,
        ),
        policy_scoring_backend="dense",
        policy_selection_backend=policy_selection_backend,
    )
    legal = _benchmark_policy_collector_one_mode(
        trace_path,
        env_count=env_count,
        steps=steps,
        iterations=iterations,
        library_path=library_path,
        observation_key=observation_key,
        mode_name="legal_compact_model_float32",
        action_features_mode="legal_only",
        action_mask_mode="legal_only",
        use_compact_action_features=True,
        store_dense_action_features=False,
        policy=make_compact_legal_argmax_policy(
            model,
            selection_backend=policy_selection_backend,
            library_path=library_path,
        ),
        policy_scoring_backend="compact",
        policy_selection_backend=policy_selection_backend,
    )
    padded = _benchmark_policy_collector_one_mode(
        trace_path,
        env_count=env_count,
        steps=steps,
        iterations=iterations,
        library_path=library_path,
        observation_key=observation_key,
        mode_name="legal_padded_model_float32",
        action_features_mode="legal_only",
        action_mask_mode="legal_only",
        use_compact_action_features=True,
        store_dense_action_features=False,
        policy=make_padded_legal_argmax_policy(
            model,
            selection_backend=policy_selection_backend,
            profile_policy=True,
            library_path=library_path,
        ),
        policy_scoring_backend="padded",
        policy_selection_backend=policy_selection_backend,
    )
    return {
        "dense_model_float32": dense,
        "legal_compact_model_float32": legal,
        "legal_padded_model_float32": padded,
    }


def benchmark_rust_ppo_update_modes(
    path: str | Path,
    *,
    model_factory,
    optimizer_factory,
    env_count: int,
    steps: int,
    iterations: int,
    epochs: int = 1,
    minibatch_size: int = 256,
    library_path: str | Path | None = None,
    observation_key: str = "observation_v5",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    advantage_backend: str = "rust",
    selected_local_backend: str = "rust",
    prepare_backend: str = "rust_fused",
    legal_row_pack_backend: str = "auto",
    policy_selection_backend: str = "rust",
) -> dict[str, dict[str, float | int | str | bool]]:
    """Compare dense and compact legal-only Rust PPO update paths."""
    env_count = int(env_count)
    steps = int(steps)
    iterations = int(iterations)
    epochs = int(epochs)
    minibatch_size = int(minibatch_size)
    if env_count <= 0:
        raise ValueError("env_count must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")

    trace_path = Path(path)
    dense = _benchmark_ppo_update_one_mode(
        trace_path,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        env_count=env_count,
        steps=steps,
        iterations=iterations,
        epochs=epochs,
        minibatch_size=minibatch_size,
        library_path=library_path,
        observation_key=observation_key,
        mode_name="dense_model_update_float32",
        action_features_mode="dense_and_legal",
        action_mask_mode="dense",
        use_compact_action_features=False,
        store_dense_action_features=True,
        policy_factory=lambda model: make_dense_argmax_policy(
            model,
            selection_backend=policy_selection_backend,
            library_path=library_path,
        ),
        train_fn=train_dense_rust_ppo_minibatch,
        gamma=gamma,
        gae_lambda=gae_lambda,
        advantage_backend=advantage_backend,
        selected_local_backend=selected_local_backend,
        prepare_backend=prepare_backend,
        legal_row_pack_backend=legal_row_pack_backend,
        policy_selection_backend=policy_selection_backend,
    )
    legal = _benchmark_ppo_update_one_mode(
        trace_path,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        env_count=env_count,
        steps=steps,
        iterations=iterations,
        epochs=epochs,
        minibatch_size=minibatch_size,
        library_path=library_path,
        observation_key=observation_key,
        mode_name="legal_padded_update_float32",
        action_features_mode="legal_only",
        action_mask_mode="legal_only",
        use_compact_action_features=True,
        store_dense_action_features=False,
        policy_factory=lambda model: make_padded_legal_argmax_policy(
            model,
            selection_backend=policy_selection_backend,
            library_path=library_path,
        ),
        train_fn=train_rust_ppo_minibatch,
        gamma=gamma,
        gae_lambda=gae_lambda,
        advantage_backend=advantage_backend,
        selected_local_backend=selected_local_backend,
        prepare_backend=prepare_backend,
        legal_row_pack_backend=legal_row_pack_backend,
        policy_selection_backend=policy_selection_backend,
    )
    return {
        "dense_model_update_float32": dense,
        "legal_padded_update_float32": legal,
    }


def benchmark_trainv3_speed_report(
    path: str | Path,
    *,
    env_count: int,
    steps: int,
    iterations: int,
    model_factory=None,
    optimizer_factory=None,
    epochs: int = 1,
    minibatch_size: int = 256,
    library_path: str | Path | None = None,
    observation_key: str = "observation_v5",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    policy_selection_backend: str = "rust",
    representative_gae_rows_per_prepare_call: int | None = 4096,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the TrainV3 Rust benchmark suite and return one JSON-safe speed report."""
    trace_path = str(path)
    env_count = int(env_count)
    steps = int(steps)
    iterations = int(iterations)
    epochs = int(epochs)
    minibatch_size = int(minibatch_size)
    if env_count <= 0:
        raise ValueError("env_count must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if (
        representative_gae_rows_per_prepare_call is not None
        and int(representative_gae_rows_per_prepare_call) <= 0
    ):
        raise ValueError("representative_gae_rows_per_prepare_call must be positive or None")

    sections: dict[str, dict[str, Any]] = {}
    sections["collector"] = _ok_section(
        benchmark_rust_vec_collector_modes(
            path,
            env_count=env_count,
            steps=steps,
            iterations=iterations,
            library_path=library_path,
        )
    )
    sections["pre_step_action_tape"] = _ok_section(
        benchmark_rust_pre_step_action_tape_batch_modes(
            path,
            env_count=env_count,
            iterations=iterations,
            library_path=library_path,
        )
    )
    gae_prepare_modes = benchmark_rust_gae_prepare(
        steps=steps,
        env_count=env_count,
        iterations=iterations,
        library_path=library_path,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    sections["gae_prepare"] = _ok_section(gae_prepare_modes)
    gae_initial_scale = _gae_prepare_scale_summary(
        gae_prepare_modes.get("python"),
        gae_prepare_modes.get("rust_fused"),
    )
    representative_gae_dims = _representative_gae_dimensions(
        gae_initial_scale,
        min_rows_per_prepare_call=representative_gae_rows_per_prepare_call,
    )
    if representative_gae_dims is not None:
        representative_gae_steps, representative_gae_env_count = representative_gae_dims
        sections["gae_prepare_representative"] = _ok_section(
            benchmark_rust_gae_prepare(
                steps=representative_gae_steps,
                env_count=representative_gae_env_count,
                iterations=iterations,
                library_path=library_path,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
        )

    if model_factory is None:
        sections["policy_inference"] = _skipped_section("model_factory is required")
        sections["policy_collector"] = _skipped_section("model_factory is required")
    else:
        sections["policy_inference"] = _ok_section(
            benchmark_compact_legal_policy_inference(
                path,
                model_factory(),
                env_count=env_count,
                iterations=iterations,
                library_path=library_path,
                observation_key=observation_key,
            )
        )
        sections["policy_collector"] = _ok_section(
            benchmark_rust_vec_policy_collector_modes(
                path,
                model_factory(),
                env_count=env_count,
                steps=steps,
                iterations=iterations,
                library_path=library_path,
                observation_key=observation_key,
                policy_selection_backend=policy_selection_backend,
            )
        )

    if model_factory is None or optimizer_factory is None:
        missing = []
        if model_factory is None:
            missing.append("model_factory")
        if optimizer_factory is None:
            missing.append("optimizer_factory")
        sections["ppo_update"] = _skipped_section(" and ".join(missing) + " required")
    else:
        sections["ppo_update"] = _ok_section(
            benchmark_rust_ppo_update_modes(
                path,
                model_factory=model_factory,
                optimizer_factory=optimizer_factory,
                env_count=env_count,
                steps=steps,
                iterations=iterations,
                epochs=epochs,
                minibatch_size=minibatch_size,
                library_path=library_path,
                observation_key=observation_key,
                gamma=gamma,
                gae_lambda=gae_lambda,
                policy_selection_backend=policy_selection_backend,
            )
        )

    report = {
        "version": 1,
        "trace_path": trace_path,
        "env_count": env_count,
        "steps": steps,
        "iterations": iterations,
        "epochs": epochs,
        "minibatch_size": minibatch_size,
        "sections": sections,
        "summary": _trainv3_speed_report_summary(sections),
    }
    report = _jsonable(report)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _benchmark_ppo_update_one_mode(
    path: Path,
    *,
    model_factory,
    optimizer_factory,
    env_count: int,
    steps: int,
    iterations: int,
    epochs: int,
    minibatch_size: int,
    library_path: str | Path | None,
    observation_key: str,
    mode_name: str,
    action_features_mode: str,
    action_mask_mode: str,
    use_compact_action_features: bool,
    store_dense_action_features: bool,
    policy_factory,
    train_fn,
    gamma: float,
    gae_lambda: float,
    advantage_backend: str,
    selected_local_backend: str,
    prepare_backend: str,
    legal_row_pack_backend: str,
    policy_selection_backend: str,
) -> dict[str, float | int | str | bool]:
    model = model_factory()
    optimizer = optimizer_factory()
    collect_seconds = 0.0
    prepare_seconds = 0.0
    update_seconds = 0.0
    ppo_updates = 0
    rows = 0
    planned_minibatches = 0
    planned_minibatch_reuses = 0
    planned_legal_action_rows = 0
    planned_padded_action_rows = 0
    planned_padding_waste_rows = 0
    planned_padded_feature_bytes = 0
    planned_padded_mask_bytes = 0
    planned_padded_id_bytes = 0
    planned_padded_total_bytes = 0
    planned_reused_padded_action_rows = 0
    planned_recomputed_padded_total_bytes = 0
    padded_cache_enabled = True
    padded_cache_builds = 0
    padded_cache_hits = 0
    padded_cache_reuses = 0
    padded_cache_bytes = 0
    padded_cache_saved_builds = 0
    padded_cache_saved_padded_total_bytes = 0
    contiguous_minibatch_plan = True
    metric_history: dict[str, list[float]] = {
        "loss_before": [],
        "loss_after": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clip_fraction": [],
        "approx_kl": [],
    }
    last_batch: RustTransitionBatch | None = None
    last_metrics: dict[str, float | int | bool | str] | None = None

    for _ in range(iterations):
        t_collect0 = time.perf_counter()
        with RustVecEnv.from_trace_file(
            path,
            env_count=env_count,
            library_path=library_path,
            observation_key=observation_key,
            action_features_mode=action_features_mode,
            observation_mode=_observation_mode_for_key(observation_key),
            action_mask_mode=action_mask_mode,
            terminal_observation_mode="none",
        ) as env:
            last_batch = collect_rust_vec_rollout(
                env,
                policy_factory(model),
                steps=steps,
                use_compact_action_features=use_compact_action_features,
                store_dense_action_features=store_dense_action_features,
                store_dense_action_mask=action_mask_mode == "dense",
                store_terminal_observations=False,
                store_next_observations=False,
                store_infos=False,
                store_truncated=False,
                store_reset_flags=False,
                store_episode_stats=False,
            )
        collect_seconds += time.perf_counter() - t_collect0

        t_prepare0 = time.perf_counter()
        ppo_batch = prepare_rust_ppo_batch(
            last_batch,
            gamma=gamma,
            gae_lambda=gae_lambda,
            advantage_backend=advantage_backend,
            selected_local_backend=selected_local_backend,
            prepare_backend=prepare_backend,
            library_path=library_path,
        )
        prepare_seconds += time.perf_counter() - t_prepare0

        t_update0 = time.perf_counter()
        last_metrics = train_fn(
            model,
            optimizer,
            ppo_batch,
            epochs=epochs,
            minibatch_size=minibatch_size,
            shuffle=False,
            legal_row_pack_backend=legal_row_pack_backend,
            library_path=library_path,
        )
        update_seconds += time.perf_counter() - t_update0
        ppo_updates += int(last_metrics["updates"])
        rows += int(last_metrics["rows"])
        contiguous_minibatch_plan = contiguous_minibatch_plan and bool(
            last_metrics.get("contiguous_minibatch_plan", False)
        )
        planned_minibatches += int(last_metrics.get("planned_minibatches", 0))
        planned_minibatch_reuses += int(last_metrics.get("planned_minibatch_reuses", 0))
        planned_legal_action_rows += int(last_metrics.get("planned_legal_action_rows", 0))
        planned_padded_action_rows += int(last_metrics.get("planned_padded_action_rows", 0))
        planned_padding_waste_rows += int(last_metrics.get("planned_padding_waste_rows", 0))
        planned_padded_feature_bytes += int(last_metrics.get("planned_padded_feature_bytes", 0))
        planned_padded_mask_bytes += int(last_metrics.get("planned_padded_mask_bytes", 0))
        planned_padded_id_bytes += int(last_metrics.get("planned_padded_id_bytes", 0))
        planned_padded_total_bytes += int(last_metrics.get("planned_padded_total_bytes", 0))
        planned_reused_padded_action_rows += int(last_metrics.get("planned_reused_padded_action_rows", 0))
        planned_recomputed_padded_total_bytes += int(
            last_metrics.get("planned_recomputed_padded_total_bytes", 0)
        )
        padded_cache_enabled = padded_cache_enabled and bool(
            last_metrics.get("padded_cache_enabled", False)
        )
        padded_cache_builds += int(last_metrics.get("padded_cache_builds", 0))
        padded_cache_hits += int(last_metrics.get("padded_cache_hits", 0))
        padded_cache_reuses += int(last_metrics.get("padded_cache_reuses", 0))
        padded_cache_bytes += int(last_metrics.get("padded_cache_bytes", 0))
        padded_cache_saved_builds += int(last_metrics.get("padded_cache_saved_builds", 0))
        padded_cache_saved_padded_total_bytes += int(
            last_metrics.get("padded_cache_saved_padded_total_bytes", 0)
        )
        for key in metric_history:
            metric_history[key].append(float(last_metrics[key]))

    if last_batch is None or last_metrics is None:
        raise RuntimeError("benchmark did not run")

    env_transitions = env_count * steps * iterations
    total_seconds = collect_seconds + prepare_seconds + update_seconds
    stats: dict[str, float | int | str | bool] = {
        "mode": mode_name,
        "env_count": env_count,
        "steps": steps,
        "iterations": iterations,
        "epochs": epochs,
        "minibatch_size": minibatch_size,
        "advantage_backend": advantage_backend,
        "selected_local_backend": selected_local_backend,
        "prepare_backend": prepare_backend,
        "legal_row_pack_backend": legal_row_pack_backend,
        "policy_selection_backend": policy_selection_backend,
        "collect_seconds": collect_seconds,
        "prepare_seconds": prepare_seconds,
        "update_seconds": update_seconds,
        "total_seconds": total_seconds,
        "env_transitions": env_transitions,
        "env_transitions_per_second": float("inf") if total_seconds <= 0 else env_transitions / total_seconds,
        "ppo_updates": ppo_updates,
        "ppo_updates_per_second": float("inf") if update_seconds <= 0 else ppo_updates / update_seconds,
        "rows": rows,
        "dense_action_features": bool(last_metrics["dense_action_features"]),
        "contiguous_minibatch_plan": contiguous_minibatch_plan,
        "planned_minibatches": planned_minibatches,
        "planned_minibatch_reuses": planned_minibatch_reuses,
        "planned_legal_action_rows": planned_legal_action_rows,
        "planned_padded_action_rows": planned_padded_action_rows,
        "planned_padding_waste_rows": planned_padding_waste_rows,
        "planned_padding_expansion_ratio": _ratio(
            planned_padded_action_rows,
            planned_legal_action_rows,
        ),
        "planned_padded_feature_bytes": planned_padded_feature_bytes,
        "planned_padded_mask_bytes": planned_padded_mask_bytes,
        "planned_padded_id_bytes": planned_padded_id_bytes,
        "planned_padded_total_bytes": planned_padded_total_bytes,
        "planned_reused_padded_action_rows": planned_reused_padded_action_rows,
        "planned_recomputed_padded_total_bytes": planned_recomputed_padded_total_bytes,
        "padded_cache_enabled": padded_cache_enabled,
        "padded_cache_builds": padded_cache_builds,
        "padded_cache_hits": padded_cache_hits,
        "padded_cache_reuses": padded_cache_reuses,
        "padded_cache_bytes": padded_cache_bytes,
        "padded_cache_saved_builds": padded_cache_saved_builds,
        "padded_cache_saved_padded_total_bytes": padded_cache_saved_padded_total_bytes,
        **_batch_storage_stats(last_batch),
    }
    for key, values in metric_history.items():
        stats[key] = float(np.mean(values))
    return stats


def _benchmark_stepwise_scripted_action_tape_batch(
    path: Path,
    *,
    action_ids: np.ndarray,
    values: np.ndarray,
    log_probs: np.ndarray,
    env_count: int,
    iterations: int,
    advantage_backend: str,
    selected_local_backend: str,
    prepare_backend: str,
    library_path: str | Path | None,
) -> dict[str, float | int | str | bool]:
    collect_seconds = 0.0
    prepare_seconds = 0.0
    last_batch: RustTransitionBatch | None = None
    with RustVecEnv.from_trace_file(
        path,
        env_count=env_count,
        library_path=library_path,
        observation_key="observation_v5",
        auto_reset=False,
        action_features_mode="legal_only",
        observation_mode="v5_only",
        action_mask_mode="legal_only",
        terminal_observation_mode="none",
        diagnostic_mode="none",
    ) as env:
        for _ in range(iterations):
            t_collect0 = time.perf_counter()
            env.reset(copy=False, include_infos=False)
            last_batch = collect_rust_vec_rollout(
                env,
                _scripted_action_tape_policy(action_ids, values, log_probs, library_path=library_path),
                steps=int(action_ids.size),
                use_compact_action_features=True,
                store_dense_action_features=False,
                store_dense_action_mask=False,
                store_terminal_observations=False,
                store_next_observations=False,
                store_infos=False,
                store_truncated=False,
                store_reset_flags=False,
                store_episode_stats=False,
            )
            collect_seconds += time.perf_counter() - t_collect0

            t_prepare0 = time.perf_counter()
            prepare_rust_ppo_batch(
                last_batch,
                normalize_advantages=False,
                advantage_backend=advantage_backend,
                selected_local_backend=selected_local_backend,
                prepare_backend=prepare_backend,
                library_path=library_path,
            )
            prepare_seconds += time.perf_counter() - t_prepare0

        if last_batch is None:
            raise RuntimeError("benchmark did not collect a stepwise batch")
        return _scripted_batch_stats(
            "stepwise_collector_legal_only",
            batch=last_batch,
            env_count=env_count,
            steps=int(action_ids.size),
            iterations=iterations,
            advantage_backend=advantage_backend,
            selected_local_backend=selected_local_backend,
            prepare_backend=prepare_backend,
            diagnostic_mode="none",
            collect_seconds=collect_seconds,
            prepare_seconds=prepare_seconds,
        )


def _benchmark_pre_step_action_tape_batch(
    path: Path,
    *,
    action_ids: np.ndarray,
    values: np.ndarray,
    log_probs: np.ndarray,
    env_count: int,
    iterations: int,
    advantage_backend: str,
    selected_local_backend: str,
    prepare_backend: str,
    library_path: str | Path | None,
) -> dict[str, float | int | str | bool]:
    action_ids = np.asarray(action_ids, dtype=np.uintp)
    if action_ids.ndim != 1:
        raise ValueError(f"action_ids must be 1D, got shape {action_ids.shape}")
    collect_seconds = 0.0
    prepare_seconds = 0.0
    last_batch: RustTransitionBatch | None = None
    with RustBatchWorker.from_trace_file(
        path,
        env_count=env_count,
        library_path=library_path,
        action_features_mode="legal_only",
        observation_mode="v5_only",
        action_mask_mode="legal_only",
        terminal_observation_mode="none",
        diagnostic_mode="none",
    ) as worker:
        for _ in range(iterations):
            t_collect0 = time.perf_counter()
            worker.reset(copy=False)
            rollout = worker.rollout_action_tape_pre_step(action_ids, copy=False)
            last_batch = transition_batch_from_action_tape_rollout(
                rollout,
                action_ids,
                values=values,
                log_probs=log_probs,
                store_truncated=False,
                store_reset_flags=False,
                store_episode_stats=False,
            )
            collect_seconds += time.perf_counter() - t_collect0

            t_prepare0 = time.perf_counter()
            prepare_rust_ppo_batch(
                last_batch,
                normalize_advantages=False,
                advantage_backend=advantage_backend,
                selected_local_backend=selected_local_backend,
                prepare_backend=prepare_backend,
                library_path=library_path,
            )
            prepare_seconds += time.perf_counter() - t_prepare0

        if last_batch is None:
            raise RuntimeError("benchmark did not collect a coarse action-tape batch")
        return _scripted_batch_stats(
            "pre_step_action_tape_legal_only",
            batch=last_batch,
            env_count=env_count,
            steps=int(action_ids.shape[0]),
            iterations=iterations,
            advantage_backend=advantage_backend,
            selected_local_backend=selected_local_backend,
            prepare_backend=prepare_backend,
            diagnostic_mode="none",
            collect_seconds=collect_seconds,
            prepare_seconds=prepare_seconds,
        )


def _scripted_batch_stats(
    mode_name: str,
    *,
    batch: RustTransitionBatch,
    env_count: int,
    steps: int,
    iterations: int,
    advantage_backend: str,
    selected_local_backend: str,
    prepare_backend: str,
    diagnostic_mode: str,
    collect_seconds: float,
    prepare_seconds: float,
) -> dict[str, float | int | str | bool]:
    rows = env_count * steps
    env_transitions = rows * iterations
    total_seconds = collect_seconds + prepare_seconds
    return {
        "mode": mode_name,
        "env_count": env_count,
        "steps": steps,
        "iterations": iterations,
        "advantage_backend": advantage_backend,
        "selected_local_backend": selected_local_backend,
        "prepare_backend": prepare_backend,
        "diagnostic_mode": diagnostic_mode,
        "collect_seconds": collect_seconds,
        "prepare_seconds": prepare_seconds,
        "total_seconds": total_seconds,
        "env_transitions": env_transitions,
        "env_transitions_per_second": float("inf") if total_seconds <= 0 else env_transitions / total_seconds,
        "rows": rows * iterations,
        **_batch_storage_stats(batch),
        "values_present": batch.values is not None,
        "log_probs_present": batch.log_probs is not None,
    }


def _scripted_action_tape_policy(
    action_ids: np.ndarray,
    values: np.ndarray,
    log_probs: np.ndarray,
    *,
    library_path: str | Path | None = None,
):
    action_ids = np.asarray(action_ids, dtype=np.uintp)
    values = np.asarray(values, dtype=np.float32)
    log_probs = np.asarray(log_probs, dtype=np.float32)
    if action_ids.ndim != 1:
        raise ValueError(f"action_ids must be 1D, got shape {action_ids.shape}")
    if values.shape != log_probs.shape:
        raise ValueError("values and log_probs must have matching shapes")
    if values.ndim != 2 or values.shape[0] != action_ids.shape[0]:
        raise ValueError("values/log_probs must have shape (steps, env_count)")
    env_count = int(values.shape[1])
    state = {"step_idx": 0}

    def policy(obs: np.ndarray, _action_mask: np.ndarray | None, action_features: RustLegalActionFeatures):
        if not isinstance(action_features, RustLegalActionFeatures):
            raise ValueError("scripted action-tape policy requires compact legal action features")
        step_idx = state["step_idx"]
        if step_idx >= int(action_ids.size):
            raise ValueError("scripted action-tape policy exhausted action ids")
        state["step_idx"] += 1
        actions = np.empty(env_count, dtype=np.uintp)
        actions.fill(action_ids[step_idx])
        return {
            "actions": actions,
            "values": values[step_idx],
            "log_probs": log_probs[step_idx],
            "selected_local_indices": _selected_local_for_scripted_actions(
                actions,
                action_features,
                library_path=library_path,
            ),
        }

    return policy


def _selected_local_for_scripted_actions(
    actions: np.ndarray,
    features: RustLegalActionFeatures,
    *,
    library_path: str | Path | None = None,
) -> np.ndarray:
    return compute_rust_selected_local_indices(
        actions,
        features.counts,
        features.offsets,
        features.ids,
        library_path=library_path,
    )


def _batch_storage_stats(batch: RustTransitionBatch) -> dict[str, int]:
    dense_feature_bytes = 0 if batch.action_features is None else int(batch.action_features.nbytes)
    dense_mask_bytes = 0 if batch.action_mask is None else int(batch.action_mask.nbytes)
    terminal_observation_bytes = (
        0 if batch.terminal_observations is None else int(batch.terminal_observations.nbytes)
    )
    next_observation_bytes = 0 if batch.next_observations is None else int(batch.next_observations.nbytes)
    truncated_bytes = 0 if batch.truncated is None else int(batch.truncated.nbytes)
    reset_flag_bytes = 0 if batch.reset_flags is None else int(batch.reset_flags.nbytes)
    episode_stat_bytes = 0
    if batch.terminal_observation_valid is not None:
        episode_stat_bytes += int(batch.terminal_observation_valid.nbytes)
    if batch.episode_returns is not None:
        episode_stat_bytes += int(batch.episode_returns.nbytes)
    if batch.episode_lengths is not None:
        episode_stat_bytes += int(batch.episode_lengths.nbytes)
    info_dicts = 0 if batch.infos is None else sum(len(step_infos) for step_infos in batch.infos)
    legal_feature_bytes = int(
        batch.legal_action_counts.nbytes
        + batch.legal_action_offsets.nbytes
        + batch.legal_action_ids.nbytes
        + batch.legal_action_features.nbytes
    )
    return {
        "stored_dense_feature_bytes": dense_feature_bytes,
        "stored_dense_mask_bytes": dense_mask_bytes,
        "stored_terminal_observation_bytes": terminal_observation_bytes,
        "stored_next_observation_bytes": next_observation_bytes,
        "stored_truncated_bytes": truncated_bytes,
        "stored_reset_flag_bytes": reset_flag_bytes,
        "stored_episode_stat_bytes": episode_stat_bytes,
        "stored_diagnostic_bytes": truncated_bytes + reset_flag_bytes + episode_stat_bytes,
        "stored_info_dicts": info_dicts,
        "stored_legal_feature_bytes": legal_feature_bytes,
        "stored_total_feature_bytes": dense_feature_bytes + legal_feature_bytes,
        "stored_total_policy_bytes": dense_feature_bytes + dense_mask_bytes + legal_feature_bytes,
    }


def _ok_section(modes: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "modes": _jsonable(modes)}


def _skipped_section(reason: str) -> dict[str, str]:
    return {"status": "skipped", "reason": reason}


def _mode(section: dict[str, Any], name: str) -> dict[str, Any] | None:
    if section.get("status") != "ok":
        return None
    modes = section.get("modes")
    if not isinstance(modes, dict):
        return None
    mode = modes.get(name)
    return mode if isinstance(mode, dict) else None


def _ratio(numerator: Any, denominator: Any) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    return 0.0 if denominator <= 0.0 else numerator / denominator


def _speed_ratio(accelerated: dict[str, Any] | None, baseline: dict[str, Any] | None, key: str) -> float | None:
    if accelerated is None or baseline is None or key not in accelerated or key not in baseline:
        return None
    return _ratio(accelerated[key], baseline[key])


def _time_speedup(baseline: dict[str, Any], accelerated: dict[str, Any], key: str) -> float:
    return _ratio(baseline.get(key, 0.0), accelerated.get(key, 0.0))


def _trainv3_speed_report_summary(sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    collector_dense = _mode(sections.get("collector", {}), "dense_float32")
    collector_legal = _mode(sections.get("collector", {}), "legal_only_float32")
    stepwise = _mode(sections.get("pre_step_action_tape", {}), "stepwise_collector_legal_only")
    coarse = _mode(sections.get("pre_step_action_tape", {}), "pre_step_action_tape_legal_only")
    gae_python = _mode(sections.get("gae_prepare", {}), "python")
    gae_fused = _mode(sections.get("gae_prepare", {}), "rust_fused")
    gae_representative_python = _mode(sections.get("gae_prepare_representative", {}), "python")
    gae_representative_fused = _mode(
        sections.get("gae_prepare_representative", {}),
        "rust_fused",
    )
    policy_dense = _mode(sections.get("policy_inference", {}), "dense_float32")
    policy_padded = _mode(sections.get("policy_inference", {}), "legal_padded_float32")
    policy_collector_dense = _mode(sections.get("policy_collector", {}), "dense_model_float32")
    policy_collector_compact = _mode(sections.get("policy_collector", {}), "legal_compact_model_float32")
    policy_collector_padded = _mode(sections.get("policy_collector", {}), "legal_padded_model_float32")
    ppo_dense = _mode(sections.get("ppo_update", {}), "dense_model_update_float32")
    ppo_legal = _mode(sections.get("ppo_update", {}), "legal_padded_update_float32")

    skipped = any(section.get("status") == "skipped" for section in sections.values())
    ppo_update = (
        None
        if ppo_dense is None or ppo_legal is None
        else _ppo_update_speedup_summary(ppo_dense, ppo_legal)
    )
    gae_prepare_scale = _gae_prepare_scale_summary(gae_python, gae_fused)
    gae_prepare_representative_scale = _gae_prepare_scale_summary(
        gae_representative_python,
        gae_representative_fused,
    )
    gae_effective_python = gae_representative_python or gae_python
    gae_effective_fused = gae_representative_fused or gae_fused
    gae_effective_scale = gae_prepare_representative_scale or gae_prepare_scale
    policy_collector = (
        None
        if policy_collector_dense is None or policy_collector_compact is None or policy_collector_padded is None
        else _policy_collector_speedup_summary(
            policy_collector_dense,
            policy_collector_compact,
            policy_collector_padded,
        )
    )
    ranking = _speed_report_bottleneck_ranking(
        collector_speedup=_speed_ratio(
            collector_legal,
            collector_dense,
            "env_transitions_per_second",
        ),
        pre_step_speedup=_speed_ratio(
            coarse,
            stepwise,
            "env_transitions_per_second",
        ),
        gae_speedup=_speed_ratio(
            gae_effective_fused,
            gae_effective_python,
            "rows_per_second",
        ),
        policy_inference_speedup=_speed_ratio(
            policy_padded,
            policy_dense,
            "forward_passes_per_second",
        ),
        policy_collector_speedup=_speed_ratio(
            policy_collector_padded,
            policy_collector_dense,
            "env_transitions_per_second",
        ),
        ppo_update_speedup=None if ppo_update is None else ppo_update["update_seconds_speedup"],
        policy_collector_next_action=(
            None if policy_collector is None else policy_collector["recommended_next_action"]
        ),
        gae_next_action=None if gae_effective_scale is None else gae_effective_scale["next_action"],
    )
    return {
        "model_dependent_sections_skipped": skipped,
        "collector_env_transitions_per_second_speedup": _speed_ratio(
            collector_legal,
            collector_dense,
            "env_transitions_per_second",
        ),
        "pre_step_action_tape_env_transitions_per_second_speedup": _speed_ratio(
            coarse,
            stepwise,
            "env_transitions_per_second",
        ),
        "gae_rust_fused_rows_per_second_speedup": _speed_ratio(
            gae_effective_fused,
            gae_effective_python,
            "rows_per_second",
        ),
        "gae_prepare_scale": gae_prepare_scale,
        "gae_prepare_representative_scale": gae_prepare_representative_scale,
        "policy_padded_forward_passes_per_second_speedup": _speed_ratio(
            policy_padded,
            policy_dense,
            "forward_passes_per_second",
        ),
        "policy_collector_padded_env_transitions_per_second_speedup": _speed_ratio(
            policy_collector_padded,
            policy_collector_dense,
            "env_transitions_per_second",
        ),
        "policy_collector": policy_collector,
        "ppo_update": (
            ppo_update
        ),
        "bottleneck_ranking": ranking,
        "next_optimization_target": None if not ranking else ranking[0]["name"],
        "next_optimization_action": None if not ranking else ranking[0]["next_action"],
    }


def _policy_collector_speedup_summary(
    dense: dict[str, Any],
    compact: dict[str, Any],
    padded: dict[str, Any],
) -> dict[str, Any]:
    dense_policy_bytes = int(dense.get("stored_total_policy_bytes", 0))
    compact_policy_bytes = int(compact.get("stored_total_policy_bytes", 0))
    padded_policy_bytes = int(padded.get("stored_total_policy_bytes", 0))
    padded_policy_bytes_saved = dense_policy_bytes - padded_policy_bytes
    padded_padding_seconds = float(padded.get("policy_padding_seconds", 0.0) or 0.0)
    padded_model_seconds = float(padded.get("policy_model_seconds", 0.0) or 0.0)
    padded_selection_seconds = float(padded.get("policy_selection_seconds", 0.0) or 0.0)
    padded_profiled_seconds = padded_padding_seconds + padded_model_seconds + padded_selection_seconds
    padded_policy_seconds = float(padded.get("policy_seconds", 0.0) or 0.0)
    padded_unprofiled_seconds = max(0.0, padded_policy_seconds - padded_profiled_seconds)
    padded_policy_fraction = _ratio(
        padded_policy_seconds,
        padded.get("elapsed_seconds", 0.0),
    )
    recommended_next_action = (
        "profile_policy_collector_policy_scoring"
        if padded_policy_fraction >= 0.5
        else "profile_policy_collector_env_step"
    )
    return {
        "baseline_mode": "dense_model_float32",
        "compact_mode": "legal_compact_model_float32",
        "accelerated_mode": "legal_padded_model_float32",
        "same_compact_env_transitions": dense.get("env_transitions") == compact.get("env_transitions"),
        "same_padded_env_transitions": dense.get("env_transitions") == padded.get("env_transitions"),
        "compact_env_transitions_per_second_speedup": _speed_ratio(
            compact,
            dense,
            "env_transitions_per_second",
        ),
        "padded_env_transitions_per_second_speedup": _speed_ratio(
            padded,
            dense,
            "env_transitions_per_second",
        ),
        "padded_vs_compact_env_transitions_per_second_speedup": _speed_ratio(
            padded,
            compact,
            "env_transitions_per_second",
        ),
        "dense_policy_seconds_fraction": _ratio(
            dense.get("policy_seconds", 0.0),
            dense.get("elapsed_seconds", 0.0),
        ),
        "compact_policy_seconds_fraction": _ratio(
            compact.get("policy_seconds", 0.0),
            compact.get("elapsed_seconds", 0.0),
        ),
        "padded_policy_seconds_fraction": padded_policy_fraction,
        "padded_vs_compact_policy_seconds_speedup": _time_speedup(
            compact,
            padded,
            "policy_seconds",
        ),
        "padded_policy_padding_seconds_fraction": _ratio(
            padded_padding_seconds,
            padded.get("elapsed_seconds", 0.0),
        ),
        "padded_policy_model_seconds_fraction": _ratio(
            padded_model_seconds,
            padded.get("elapsed_seconds", 0.0),
        ),
        "padded_policy_selection_seconds_fraction": _ratio(
            padded_selection_seconds,
            padded.get("elapsed_seconds", 0.0),
        ),
        "padded_policy_profiled_seconds_fraction": _ratio(
            padded_profiled_seconds,
            padded.get("elapsed_seconds", 0.0),
        ),
        "padded_policy_unprofiled_seconds_fraction": _ratio(
            padded_unprofiled_seconds,
            padded.get("elapsed_seconds", 0.0),
        ),
        "dense_policy_bytes": dense_policy_bytes,
        "compact_policy_bytes": compact_policy_bytes,
        "padded_policy_bytes": padded_policy_bytes,
        "compact_policy_bytes_saved": dense_policy_bytes - compact_policy_bytes,
        "padded_policy_bytes_saved": padded_policy_bytes_saved,
        "padded_policy_bytes_reduction_fraction": _ratio(
            padded_policy_bytes_saved,
            dense_policy_bytes,
        ),
        "recommended_next_action": recommended_next_action,
    }


def _representative_gae_dimensions(
    gae_prepare_scale: dict[str, Any] | None,
    *,
    min_rows_per_prepare_call: int | None,
) -> tuple[int, int] | None:
    if min_rows_per_prepare_call is None or gae_prepare_scale is None:
        return None
    if gae_prepare_scale.get("benchmark_scale") != "tiny":
        return None

    min_rows = int(min_rows_per_prepare_call)
    current_rows = int(gae_prepare_scale.get("rows_per_prepare_call", 0) or 0)
    if current_rows >= min_rows:
        return None

    current_steps = int(gae_prepare_scale.get("steps", 0) or 0)
    current_env_count = int(gae_prepare_scale.get("env_count", 0) or 0)
    balanced_axis = int(math.ceil(math.sqrt(float(min_rows))))
    representative_steps = max(current_steps, balanced_axis)
    representative_env_count = max(current_env_count, int(math.ceil(min_rows / representative_steps)))
    return representative_steps, representative_env_count


def _gae_prepare_scale_summary(
    baseline: dict[str, Any] | None,
    accelerated: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if baseline is None or accelerated is None:
        return None

    prepare_calls = int(accelerated.get("prepare_calls", baseline.get("prepare_calls", 0)) or 0)
    steps = int(accelerated.get("steps", baseline.get("steps", 0)) or 0)
    env_count = int(accelerated.get("env_count", baseline.get("env_count", 0)) or 0)
    rows = int(accelerated.get("total_rows", accelerated.get("rows", baseline.get("rows", 0))) or 0)
    rows_per_prepare_call = int(
        accelerated.get(
            "rows_per_prepare_call",
            (steps * env_count) if steps > 0 and env_count > 0 else 0,
        )
        or 0
    )
    if rows_per_prepare_call <= 0 and rows > 0 and prepare_calls > 0:
        rows_per_prepare_call = rows // prepare_calls

    representative_rows_per_prepare_call = 4096
    speedup = _speed_ratio(accelerated, baseline, "rows_per_second")
    benchmark_scale = (
        "unknown"
        if rows_per_prepare_call <= 0
        else (
            "tiny"
            if rows_per_prepare_call < representative_rows_per_prepare_call
            else "representative"
        )
    )
    next_action = (
        "rerun_gae_prepare_at_representative_scale"
        if benchmark_scale == "tiny"
        else "optimize_gae_prepare"
    )
    scale_hint = {
        "tiny": "ffi_overhead_dominated_possible",
        "representative": "representative_problem_size",
        "unknown": "missing_problem_size",
    }[benchmark_scale]
    return {
        "benchmark_scale": benchmark_scale,
        "scale_hint": scale_hint,
        "representative_rows_per_prepare_call": representative_rows_per_prepare_call,
        "steps": steps,
        "env_count": env_count,
        "iterations": int(accelerated.get("iterations", baseline.get("iterations", 0)) or 0),
        "prepare_calls": prepare_calls,
        "rows": rows,
        "total_rows": rows,
        "rows_per_prepare_call": rows_per_prepare_call,
        "python_rows_per_second": float(baseline.get("rows_per_second", 0.0) or 0.0),
        "rust_fused_rows_per_second": float(accelerated.get("rows_per_second", 0.0) or 0.0),
        "rust_fused_rows_per_second_speedup": speedup,
        "next_action": next_action,
    }


def _ppo_update_speedup_summary(dense: dict[str, Any], legal: dict[str, Any]) -> dict[str, Any]:
    dense_policy_bytes = int(dense.get("stored_total_policy_bytes", 0))
    legal_policy_bytes = int(legal.get("stored_total_policy_bytes", 0))
    policy_bytes_saved = dense_policy_bytes - legal_policy_bytes
    legal_padded_cache_builds = int(legal.get("padded_cache_builds", 0))
    legal_padded_cache_hits = int(legal.get("padded_cache_hits", 0))
    legal_padded_cache_reuses = int(legal.get("padded_cache_reuses", 0))
    legal_planned_recomputed_padded_total_bytes = int(
        legal.get("planned_recomputed_padded_total_bytes", 0)
    )
    legal_padded_cache_saved_padded_total_bytes = int(
        legal.get("padded_cache_saved_padded_total_bytes", 0)
    )
    return {
        "baseline_mode": "dense_model_update_float32",
        "accelerated_mode": "legal_padded_update_float32",
        "same_env_transitions": dense.get("env_transitions") == legal.get("env_transitions"),
        "same_rows": dense.get("rows") == legal.get("rows"),
        "same_ppo_updates": dense.get("ppo_updates") == legal.get("ppo_updates"),
        "same_epochs": dense.get("epochs") == legal.get("epochs"),
        "same_minibatch_size": dense.get("minibatch_size") == legal.get("minibatch_size"),
        "total_seconds_speedup": _time_speedup(dense, legal, "total_seconds"),
        "collect_seconds_speedup": _time_speedup(dense, legal, "collect_seconds"),
        "prepare_seconds_speedup": _time_speedup(dense, legal, "prepare_seconds"),
        "update_seconds_speedup": _time_speedup(dense, legal, "update_seconds"),
        "env_transitions_per_second_speedup": _speed_ratio(
            legal,
            dense,
            "env_transitions_per_second",
        ),
        "ppo_updates_per_second_speedup": _speed_ratio(
            legal,
            dense,
            "ppo_updates_per_second",
        ),
        "dense_policy_bytes": dense_policy_bytes,
        "legal_policy_bytes": legal_policy_bytes,
        "policy_bytes_saved": policy_bytes_saved,
        "policy_bytes_reduction_fraction": _ratio(policy_bytes_saved, dense_policy_bytes),
        "dense_feature_bytes_saved": int(dense.get("stored_dense_feature_bytes", 0))
        - int(legal.get("stored_dense_feature_bytes", 0)),
        "legal_epochs": int(legal.get("epochs", 0)),
        "legal_contiguous_minibatch_plan": bool(legal.get("contiguous_minibatch_plan", False)),
        "legal_planned_minibatches": int(legal.get("planned_minibatches", 0)),
        "legal_planned_minibatch_reuses": int(legal.get("planned_minibatch_reuses", 0)),
        "legal_planned_legal_action_rows": int(legal.get("planned_legal_action_rows", 0)),
        "legal_planned_padded_action_rows": int(legal.get("planned_padded_action_rows", 0)),
        "legal_planned_padding_waste_rows": int(legal.get("planned_padding_waste_rows", 0)),
        "legal_planned_padding_expansion_ratio": float(
            legal.get("planned_padding_expansion_ratio", 0.0)
        ),
        "legal_planned_padded_feature_bytes": int(legal.get("planned_padded_feature_bytes", 0)),
        "legal_planned_padded_mask_bytes": int(legal.get("planned_padded_mask_bytes", 0)),
        "legal_planned_padded_id_bytes": int(legal.get("planned_padded_id_bytes", 0)),
        "legal_planned_padded_total_bytes": int(legal.get("planned_padded_total_bytes", 0)),
        "legal_planned_reused_padded_action_rows": int(
            legal.get("planned_reused_padded_action_rows", 0)
        ),
        "legal_planned_recomputed_padded_total_bytes": legal_planned_recomputed_padded_total_bytes,
        "legal_padded_cache_enabled": bool(legal.get("padded_cache_enabled", False)),
        "legal_padded_cache_builds": legal_padded_cache_builds,
        "legal_padded_cache_hits": legal_padded_cache_hits,
        "legal_padded_cache_reuses": legal_padded_cache_reuses,
        "legal_padded_cache_bytes": int(legal.get("padded_cache_bytes", 0)),
        "legal_padded_cache_saved_builds": int(legal.get("padded_cache_saved_builds", 0)),
        "legal_padded_cache_saved_padded_total_bytes": legal_padded_cache_saved_padded_total_bytes,
        "legal_padded_cache_reuse_fraction": _ratio(
            legal_padded_cache_reuses,
            legal_padded_cache_hits,
        ),
        "legal_padded_cache_hit_build_ratio": _ratio(
            legal_padded_cache_hits,
            legal_padded_cache_builds,
        ),
        "legal_padded_cache_saved_recomputed_fraction": _ratio(
            legal_padded_cache_saved_padded_total_bytes,
            legal_planned_recomputed_padded_total_bytes,
        ),
        "advantage_backend": legal.get("advantage_backend"),
        "selected_local_backend": legal.get("selected_local_backend"),
        "prepare_backend": legal.get("prepare_backend"),
        "legal_row_pack_backend": legal.get("legal_row_pack_backend"),
        "policy_selection_backend": legal.get("policy_selection_backend"),
    }


def _speed_report_bottleneck_ranking(
    *,
    collector_speedup: float | None,
    pre_step_speedup: float | None,
    gae_speedup: float | None,
    policy_inference_speedup: float | None,
    policy_collector_speedup: float | None,
    ppo_update_speedup: float | None,
    policy_collector_next_action: str | None = None,
    gae_next_action: str | None = None,
) -> list[dict[str, float | str]]:
    candidates = [
        ("ppo_update", "phase", ppo_update_speedup, 0, None),
        ("policy_collector", "section", policy_collector_speedup, 1, policy_collector_next_action),
        ("policy_inference", "section", policy_inference_speedup, 2, None),
        ("collector", "section", collector_speedup, 3, None),
        ("pre_step_action_tape", "section", pre_step_speedup, 4, None),
        ("gae_prepare", "section", gae_speedup, 5, gae_next_action),
    ]
    ranking: list[dict[str, float | str]] = []
    for name, kind, speedup, priority, next_action in candidates:
        if speedup is None or speedup <= 0:
            continue
        ranking.append(
            {
                "name": name,
                "kind": kind,
                "score": _ratio(1.0, speedup),
                "speedup": float(speedup),
                "next_action": next_action or f"optimize_{name}",
                "_priority": priority,
            }
        )
    ranking.sort(key=lambda item: (-float(item["score"]), int(item["_priority"])))
    for item in ranking:
        del item["_priority"]
    return ranking


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _synthetic_gae_transition_batch(steps: int, env_count: int) -> RustTransitionBatch:
    row_count = steps * env_count
    observations = np.zeros((steps, env_count, 1), dtype=np.float32)
    grid = np.arange(row_count, dtype=np.float32).reshape(steps, env_count)
    rewards = (np.sin(grid * 0.17) * 0.5 + np.cos(grid * 0.07) * 0.25).astype(np.float32)
    values = (np.cos(grid * 0.11) * 0.4).astype(np.float32)
    log_probs = (-0.2 - (grid % 13) * 0.01).astype(np.float32)
    actions = np.arange(row_count, dtype=np.uintp).reshape(steps, env_count) % np.uintp(601)
    terminated = np.zeros((steps, env_count), dtype=np.bool_)
    truncated = np.zeros((steps, env_count), dtype=np.bool_)
    if steps > 2:
        terminated[steps // 2, ::2] = True
    if steps > 3 and env_count > 1:
        truncated[steps - 2, 1::2] = True
    legal_action_counts = np.ones((steps, env_count), dtype=np.uintp)
    legal_action_offsets = np.arange(row_count, dtype=np.uintp).reshape(steps, env_count)
    legal_action_ids = actions.reshape(-1)
    legal_action_features = np.zeros((row_count, 1), dtype=np.float32)
    return RustTransitionBatch(
        observations=observations,
        next_observations=None,
        action_mask=None,
        action_features=None,
        legal_action_counts=legal_action_counts,
        legal_action_offsets=legal_action_offsets,
        legal_action_ids=legal_action_ids,
        legal_action_features=legal_action_features,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        reset_flags=np.zeros((steps, env_count), dtype=np.bool_),
        terminal_observations=None,
        terminal_observation_valid=np.zeros((steps, env_count), dtype=np.bool_),
        episode_returns=np.zeros((steps, env_count), dtype=np.float32),
        episode_lengths=np.zeros((steps, env_count), dtype=np.int32),
        infos=None,
        values=values,
        log_probs=log_probs,
    )


def _policy_inference_stats(
    mode_name: str,
    *,
    env_count: int,
    iterations: int,
    scored_rows_per_iteration: int,
    elapsed_seconds: float,
    policy_scoring_backend: str,
    padding_backend: str | None = None,
    row_index_backend: str | None = None,
) -> dict[str, float | int | str]:
    stats: dict[str, float | int | str] = {
        "mode": mode_name,
        "policy_scoring_backend": policy_scoring_backend,
        "env_count": env_count,
        "iterations": iterations,
        "elapsed_seconds": elapsed_seconds,
        "forward_passes_per_second": float("inf") if elapsed_seconds <= 0 else iterations / elapsed_seconds,
        "scored_action_rows_per_iteration": scored_rows_per_iteration,
        "total_scored_action_rows": scored_rows_per_iteration * iterations,
        "scored_action_rows_per_second": (
            float("inf")
            if elapsed_seconds <= 0
            else (scored_rows_per_iteration * iterations) / elapsed_seconds
        ),
    }
    if padding_backend is not None:
        stats["padding_backend"] = padding_backend
    if row_index_backend is not None:
        stats["row_index_backend"] = row_index_backend
    return stats


def _benchmark_policy_collector_one_mode(
    path: Path,
    *,
    env_count: int,
    steps: int,
    iterations: int,
    library_path: str | Path | None,
    observation_key: str,
    mode_name: str,
    action_features_mode: str,
    action_mask_mode: str,
    use_compact_action_features: bool,
    store_dense_action_features: bool,
    policy,
    policy_scoring_backend: str,
    policy_selection_backend: str,
) -> dict[str, float | int | str | bool]:
    start = time.perf_counter()
    last_batch: RustTransitionBatch | None = None
    policy_seconds = 0.0
    env_step_seconds = 0.0
    policy_profile: dict[str, float] = {}
    for _ in range(iterations):
        with RustVecEnv.from_trace_file(
            path,
            env_count=env_count,
            library_path=library_path,
            observation_key=observation_key,
            action_features_mode=action_features_mode,
            observation_mode=_observation_mode_for_key(observation_key),
            action_mask_mode=action_mask_mode,
            terminal_observation_mode="none",
            diagnostic_mode="none",
        ) as env:
            last_batch = collect_rust_vec_rollout(
                env,
                policy,
                steps=steps,
                use_compact_action_features=use_compact_action_features,
                store_dense_action_features=store_dense_action_features,
                store_dense_action_mask=action_mask_mode == "dense",
                store_terminal_observations=False,
                store_next_observations=False,
                store_infos=False,
                store_truncated=False,
                store_reset_flags=False,
                store_episode_stats=False,
            )
            policy_seconds += last_batch.policy_seconds
            env_step_seconds += last_batch.env_step_seconds
            if last_batch.policy_profile is not None:
                for key, value in last_batch.policy_profile.items():
                    policy_profile[str(key)] = policy_profile.get(str(key), 0.0) + float(value)
    elapsed = time.perf_counter() - start
    if last_batch is None:
        raise RuntimeError("benchmark did not collect a batch")

    env_transitions = env_count * steps * iterations
    collector_overhead_seconds = max(0.0, elapsed - policy_seconds - env_step_seconds)
    return {
        "mode": mode_name,
        "policy_scoring_backend": policy_scoring_backend,
        "policy_selection_backend": policy_selection_backend,
        "env_count": env_count,
        "steps": steps,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "policy_seconds": policy_seconds,
        "env_step_seconds": env_step_seconds,
        "collector_overhead_seconds": collector_overhead_seconds,
        "policy_seconds_fraction": _ratio(policy_seconds, elapsed),
        "env_step_seconds_fraction": _ratio(env_step_seconds, elapsed),
        "collector_overhead_seconds_fraction": _ratio(collector_overhead_seconds, elapsed),
        **policy_profile,
        "env_transitions": env_transitions,
        "env_transitions_per_second": float("inf") if elapsed <= 0 else env_transitions / elapsed,
        **_batch_storage_stats(last_batch),
        "values_present": last_batch.values is not None,
        "log_probs_present": last_batch.log_probs is not None,
    }


def _benchmark_one_mode(
    path: Path,
    *,
    env_count: int,
    steps: int,
    iterations: int,
    library_path: str | Path | None,
    mode_name: str,
    action_features_mode: str,
    action_mask_mode: str,
    use_compact_action_features: bool,
    store_dense_action_features: bool,
    policy,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    last_batch: RustTransitionBatch | None = None
    for _ in range(iterations):
        with RustVecEnv.from_trace_file(
            path,
            env_count=env_count,
            library_path=library_path,
            action_features_mode=action_features_mode,
            observation_mode="v5_only",
            action_mask_mode=action_mask_mode,
            terminal_observation_mode="none",
            diagnostic_mode="none",
        ) as env:
            last_batch = collect_rust_vec_rollout(
                env,
                policy,
                steps=steps,
                use_compact_action_features=use_compact_action_features,
                store_dense_action_features=store_dense_action_features,
                store_dense_action_mask=action_mask_mode == "dense",
                store_terminal_observations=False,
                store_next_observations=False,
                store_infos=False,
                store_truncated=False,
                store_reset_flags=False,
                store_episode_stats=False,
            )
    elapsed = time.perf_counter() - start
    if last_batch is None:
        raise RuntimeError("benchmark did not collect a batch")

    env_transitions = env_count * steps * iterations
    return {
        "mode": mode_name,
        "env_count": env_count,
        "steps": steps,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "env_transitions": env_transitions,
        "env_transitions_per_second": float("inf") if elapsed <= 0 else env_transitions / elapsed,
        **_batch_storage_stats(last_batch),
    }


def _first_legal_from_dense(_obs: np.ndarray, action_mask: np.ndarray, _action_features: np.ndarray) -> np.ndarray:
    return np.argmax(action_mask > 0, axis=1).astype(np.uintp)


def _first_legal_from_compact(
    _obs: np.ndarray,
    _action_mask: np.ndarray,
    action_features: RustLegalActionFeatures,
) -> np.ndarray:
    actions = np.empty(action_features.counts.shape[0], dtype=np.uintp)
    offset = 0
    for env_idx, count in enumerate(action_features.counts.tolist()):
        if count <= 0:
            raise ValueError(f"env {env_idx} has no legal actions")
        actions[env_idx] = action_features.ids[offset]
        offset += int(count)
    return actions


def _observation_mode_for_key(observation_key: str) -> str:
    return "v5_only" if observation_key == "observation_v5" else "v1_and_v5"


def _action_ids_from_trace(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"trace {path} does not contain a steps list")
    action_ids: list[int] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict) or "action_id" not in step:
            raise ValueError(f"trace {path} step {idx} does not contain action_id")
        action_ids.append(int(step["action_id"]))
    return action_ids


__all__ = [
    "benchmark_compact_legal_policy_inference",
    "benchmark_rust_gae_prepare",
    "benchmark_rust_pre_step_action_tape_batch_modes",
    "benchmark_rust_ppo_update_modes",
    "benchmark_trainv3_speed_report",
    "benchmark_rust_vec_policy_collector_modes",
    "benchmark_rust_vec_collector_modes",
]
