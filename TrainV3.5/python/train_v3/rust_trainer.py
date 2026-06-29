"""Minimal PPO training harness for Rust-backed TrainV3 rollouts."""
from __future__ import annotations

import json
import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .league_v5 import V5LeagueConfig, sample_v5_episode_modes
from .contracts import AssistModeV5
from .rust_collector import RustTransitionBatch, collect_rust_vec_rollout
from .rust_policy import make_compact_legal_argmax_policy, make_padded_legal_argmax_policy
from .rust_ppo import prepare_rust_ppo_batch, train_rust_ppo_minibatch
from .rust_vec_env import RustVecEnv
from .trace_factory_v5 import load_v5_trace_pool_manifest, select_v5_trace_paths_for_mode
from .v5_artifacts import LeagueRunManifest, read_manifest_json, write_manifest_json


@dataclass(frozen=True)
class RustPPOTrainingConfig:
    run_name: str | None = None
    model_name: str = "extra-lr-v5-adaptive"
    v5_league_config: V5LeagueConfig | None = None
    curriculum_metadata: dict[str, Any] = field(default_factory=dict)
    updates: int = 1
    env_count: int = 16
    steps_per_update: int = 30
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 3
    minibatch_size: int = 256
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float | None = None
    observation_key: str = "observation_v5"
    action_features_dtype: str = "float32"
    observation_mode: str = "v5_only"
    action_features_mode: str = "legal_only"
    action_mask_mode: str = "legal_only"
    terminal_observation_mode: str = "none"
    store_next_observations: bool = False
    store_infos: bool = False
    store_truncated: bool = False
    store_reset_flags: bool = False
    store_episode_stats: bool = False
    diagnostic_mode: str = "auto"
    advantage_backend: str = "rust"
    selected_local_backend: str = "provided"
    prepare_backend: str = "separate"
    legal_row_pack_backend: str = "auto"
    full_batch_eval: bool = False
    policy_scoring_backend: str = "padded"
    policy_selection_backend: str = "rust"
    policy_padding_mode: str = "single"
    policy_bucket_max_padding_ratio: float = 1.35
    policy_bucket_min_rows: int = 2048
    ppo_minibatch_plan: str = "contiguous"
    log_selected_trace_paths: bool = True
    trace_pool_reset_mode: str = "cycle"
    v5_runtime_mode_source: str = "manifest_cycle"
    trace_manifest_path: str | Path | None = None
    league_manifest_path: str | Path | None = None
    checkpoint_dir: str | Path | None = None
    checkpoint_every: int = 1
    metrics_path: str | Path | None = None
    seed: int | None = None


def train_rust_ppo_trace_file(
    path: str | Path,
    model: Any,
    optimizer: Any,
    config: RustPPOTrainingConfig,
    *,
    library_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a small legal-only PPO loop against a Rust golden-trace initial state."""
    return train_rust_ppo_trace_files(
        [path],
        model,
        optimizer,
        config,
        library_path=library_path,
    )


def train_rust_ppo_trace_files(
    paths: list[str | Path] | tuple[str | Path, ...],
    model: Any,
    optimizer: Any,
    config: RustPPOTrainingConfig,
    *,
    library_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a small legal-only PPO loop against a pool of Rust golden-trace initial states."""
    _validate_config(config)
    trace_input = _load_trace_input(paths, config)
    trace_manifest_id = trace_input["trace_manifest_id"]
    if not trace_input["all_trace_paths"]:
        raise ValueError("paths must contain at least one trace file")
    total_env_transitions = 0
    metrics: list[dict[str, Any]] = []
    checkpoint_path = ""
    t_total0 = time.perf_counter()
    trace_env_reuse = config.v5_runtime_mode_source == "manifest_cycle"
    trace_env_open_seconds = 0.0
    resolved_policy_scoring_backend = _resolve_policy_scoring_backend(config)
    policy = _make_legal_policy(model, config, library_path=library_path)
    diagnostic_mode = _resolve_diagnostic_mode(config)
    league_updates: list[dict[str, Any]] = []

    shared_env_cm = None
    shared_env = None
    if config.v5_runtime_mode_source == "manifest_cycle":
        trace_paths, trace_selection = _select_trace_paths_for_update(trace_input, config, 1)
        t_env_open0 = time.perf_counter()
        shared_env_cm = _open_trace_env(
            trace_paths,
            config,
            library_path=library_path,
        )
        trace_env_open_seconds += time.perf_counter() - t_env_open0

    try:
        if shared_env_cm is not None:
            shared_env = shared_env_cm.__enter__()

        for update in range(1, int(config.updates) + 1):
            v5_update = _v5_league_update_metadata(config, update)
            trace_paths, trace_selection = _select_trace_paths_for_update(
                trace_input,
                config,
                update,
                v5_update=v5_update,
            )
            v5_update = dict(v5_update)
            v5_update.setdefault("assist_mode", AssistModeV5().to_dict())
            v5_update["trace_selection"] = trace_selection
            league_updates.append(v5_update)

            if shared_env is None:
                t_env_open0 = time.perf_counter()
                env_cm = _open_trace_env(trace_paths, config, library_path=library_path)
                trace_env_open_seconds += time.perf_counter() - t_env_open0
                with env_cm as env:
                    collect_seconds, prepare_seconds, update_seconds, update_metrics, transitions = _run_training_update(
                        env,
                        policy,
                        model,
                        optimizer,
                        config,
                        update,
                        library_path=library_path,
                    )
            else:
                collect_seconds, prepare_seconds, update_seconds, update_metrics, transitions = _run_training_update(
                    shared_env,
                    policy,
                    model,
                    optimizer,
                    config,
                    update,
                    library_path=library_path,
                )

            env_transitions = int(config.env_count * config.steps_per_update)
            total_env_transitions += env_transitions
            metric = {
                "update": update,
                "run_name": config.run_name,
                "model_name": config.model_name,
                "trace_manifest_id": trace_manifest_id,
                "v5_mode": v5_update["v5_mode"],
                "assist_mode": v5_update["assist_mode"],
                "v5_opponent_mix": v5_update["opponent_mix"],
                "v5_curriculum_metadata": dict(config.curriculum_metadata),
                "trace_selection": trace_selection,
                "selected_trace_count": trace_selection["selected_trace_count"],
                "selected_trace_subset_sha256": trace_selection["selected_trace_subset_sha256"],
                "env_transitions": env_transitions,
                "total_env_transitions": total_env_transitions,
                "trace_env_reuse": trace_env_reuse,
                "collect_seconds": collect_seconds,
                "prepare_seconds": prepare_seconds,
                "update_seconds": update_seconds,
                "stored_dense_feature_bytes": _dense_feature_bytes(transitions),
                "stored_dense_mask_bytes": _dense_mask_bytes(transitions),
                "stored_terminal_observation_bytes": _terminal_observation_bytes(transitions),
                "stored_next_observation_bytes": _next_observation_bytes(transitions),
                "stored_truncated_bytes": _truncated_bytes(transitions),
                "stored_reset_flag_bytes": _reset_flag_bytes(transitions),
                "stored_episode_stat_bytes": _episode_stat_bytes(transitions),
                "stored_info_dicts": _info_dict_count(transitions),
                "stored_legal_feature_bytes": _legal_feature_bytes(transitions),
                "action_features_dtype": config.action_features_dtype,
                "advantage_backend": config.advantage_backend,
                "selected_local_backend": config.selected_local_backend,
                "prepare_backend": config.prepare_backend,
                "legal_row_pack_backend": config.legal_row_pack_backend,
                "full_batch_eval": config.full_batch_eval,
                "policy_scoring_backend": config.policy_scoring_backend,
                "resolved_policy_scoring_backend": resolved_policy_scoring_backend,
                "policy_selection_backend": config.policy_selection_backend,
                "policy_padding_mode": config.policy_padding_mode,
                "policy_bucket_max_padding_ratio": config.policy_bucket_max_padding_ratio,
                "policy_bucket_min_rows": config.policy_bucket_min_rows,
                "store_truncated": config.store_truncated,
                "store_reset_flags": config.store_reset_flags,
                "store_episode_stats": config.store_episode_stats,
                "diagnostic_mode": diagnostic_mode,
                "ppo_updates": int(update_metrics["updates"]),
                **update_metrics,
            }
            if config.log_selected_trace_paths:
                metric["selected_trace_paths"] = trace_selection.get("selected_trace_paths", [])
            _assert_metric_finite(metric)
            metrics.append(metric)
            if config.metrics_path is not None:
                _append_jsonl(config.metrics_path, metric)

            if _should_checkpoint(config, update):
                checkpoint_v5_league = _v5_league_summary(
                    config,
                    league_updates,
                    trace_manifest_id=trace_manifest_id,
                )
                checkpoint_summary = _summarize_training_speed(
                    metrics,
                    total_env_transitions=total_env_transitions,
                    elapsed_seconds=time.perf_counter() - t_total0,
                )
                checkpoint_summary["v5_league"] = checkpoint_v5_league
                checkpoint_path = _save_checkpoint(
                    model,
                    optimizer,
                    config,
                    update=update,
                    total_env_transitions=total_env_transitions,
                    metric=metric,
                    speed_summary=checkpoint_summary,
                    trace_pool_size=len(trace_paths),
                    trace_env_reuse=trace_env_reuse,
                    trace_manifest_id=trace_manifest_id,
                    v5_league=checkpoint_v5_league,
                )
    finally:
        if shared_env_cm is not None:
            shared_env_cm.__exit__(None, None, None)

    elapsed_seconds = time.perf_counter() - t_total0
    v5_league = _v5_league_summary(config, league_updates, trace_manifest_id=trace_manifest_id)
    speed_summary = _summarize_training_speed(
        metrics,
        total_env_transitions=total_env_transitions,
        elapsed_seconds=elapsed_seconds,
    )
    speed_summary["trace_manifest_id"] = trace_manifest_id
    speed_summary["v5_league"] = v5_league
    if config.metrics_path is not None:
        _append_jsonl(config.metrics_path, speed_summary)

    league_manifest, league_manifest_path = _write_league_run_manifest(
        config,
        v5_league=v5_league,
        trace_manifest_id=trace_manifest_id,
        checkpoint_path=checkpoint_path,
    )

    result = {
        "run_name": config.run_name,
        "model_name": config.model_name,
        "trace_manifest_id": trace_manifest_id,
        "v5_league": v5_league,
        "updates": int(config.updates),
        "total_env_transitions": total_env_transitions,
        "legal_only": True,
        "rollout_mode": "rust_vec_legal_only",
        "action_features_dtype": config.action_features_dtype,
        "advantage_backend": config.advantage_backend,
        "selected_local_backend": config.selected_local_backend,
        "prepare_backend": config.prepare_backend,
        "legal_row_pack_backend": config.legal_row_pack_backend,
        "full_batch_eval": config.full_batch_eval,
        "policy_scoring_backend": config.policy_scoring_backend,
        "resolved_policy_scoring_backend": resolved_policy_scoring_backend,
        "policy_selection_backend": config.policy_selection_backend,
        "trace_pool_size": int(metrics[-1].get("selected_trace_count", 0)) if metrics else len(trace_input["all_trace_paths"]),
        "trace_pool_total_size": len(trace_input["all_trace_paths"]),
        "selected_trace_paths": metrics[-1].get("selected_trace_paths", []) if metrics else [],
        "selected_trace_subsets": [metric["trace_selection"] for metric in metrics],
        "trace_pool_reset_mode": config.trace_pool_reset_mode,
        "trace_env_reuse": trace_env_reuse,
        "trace_env_open_seconds": trace_env_open_seconds,
        "store_truncated": config.store_truncated,
        "store_reset_flags": config.store_reset_flags,
        "store_episode_stats": config.store_episode_stats,
        "diagnostic_mode": diagnostic_mode,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "league_manifest": league_manifest,
        "league_manifest_path": league_manifest_path,
        "speed_summary": speed_summary,
        "elapsed_seconds": elapsed_seconds,
    }
    result.update(
        {
            key: value
            for key, value in speed_summary.items()
            if key not in {"type", "updates", "total_env_transitions", "elapsed_seconds"}
        }
    )
    return result


def _run_training_update(
    env: RustVecEnv,
    policy: Any,
    model: Any,
    optimizer: Any,
    config: RustPPOTrainingConfig,
    update: int,
    *,
    library_path: str | Path | None,
) -> tuple[float, float, float, dict[str, Any], RustTransitionBatch]:
    t_collect0 = time.perf_counter()
    transitions = collect_rust_vec_rollout(
        env,
        policy,
        steps=config.steps_per_update,
        use_compact_action_features=True,
        store_dense_action_features=False,
        store_dense_action_mask=config.action_mask_mode == "dense",
        store_terminal_observations=config.terminal_observation_mode == "full",
        store_next_observations=config.store_next_observations,
        store_infos=config.store_infos,
        store_truncated=config.store_truncated,
        store_reset_flags=config.store_reset_flags,
        store_episode_stats=config.store_episode_stats,
    )
    collect_seconds = time.perf_counter() - t_collect0

    t_prepare0 = time.perf_counter()
    ppo_batch = prepare_rust_ppo_batch(
        transitions,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        advantage_backend=config.advantage_backend,
        selected_local_backend=config.selected_local_backend,
        prepare_backend=config.prepare_backend,
        library_path=library_path,
    )
    prepare_seconds = time.perf_counter() - t_prepare0

    t_update0 = time.perf_counter()
    update_metrics = train_rust_ppo_minibatch(
        model,
        optimizer,
        ppo_batch,
        epochs=config.epochs,
        minibatch_size=config.minibatch_size,
        clip_epsilon=config.clip_epsilon,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        shuffle=False,
        seed=None if config.seed is None else config.seed + update,
        legal_row_pack_backend=config.legal_row_pack_backend,
        full_batch_eval=config.full_batch_eval,
        minibatch_plan=config.ppo_minibatch_plan,
        library_path=library_path,
    )
    update_seconds = time.perf_counter() - t_update0
    return collect_seconds, prepare_seconds, update_seconds, update_metrics, transitions


def _load_trace_input(
    paths: list[str | Path] | tuple[str | Path, ...],
    config: RustPPOTrainingConfig,
) -> dict[str, Any]:
    if config.trace_manifest_path is None:
        trace_paths = [Path(path) for path in paths]
        return {
            "manifest": None,
            "trace_manifest_id": None,
            "all_trace_paths": trace_paths,
        }

    manifest = load_v5_trace_pool_manifest(config.trace_manifest_path)
    all_trace_paths = select_v5_trace_paths_for_mode(
        manifest,
        _v5_league_update_metadata(config, 1)["v5_mode"],
        runtime_mode_source="manifest_cycle",
    )
    return {
        "manifest": manifest,
        "trace_manifest_id": str(manifest["manifest_id"]),
        "all_trace_paths": all_trace_paths,
    }


def _select_trace_paths_for_update(
    trace_input: dict[str, Any],
    config: RustPPOTrainingConfig,
    update: int,
    *,
    v5_update: dict[str, Any] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    manifest = trace_input.get("manifest")
    if manifest is None:
        trace_paths = list(trace_input["all_trace_paths"])
    else:
        mode = (v5_update or _v5_league_update_metadata(config, update))["v5_mode"]
        trace_paths = select_v5_trace_paths_for_mode(
            manifest,
            mode,
            assist_mode=(v5_update or _v5_league_update_metadata(config, update)).get("assist_mode"),
            runtime_mode_source=config.v5_runtime_mode_source,
        )
    if not trace_paths:
        raise ValueError("selected trace subset must not be empty")
    selection = _trace_selection_metadata(
        trace_paths,
        update=update,
        trace_manifest_id=trace_input.get("trace_manifest_id"),
        runtime_mode_source=config.v5_runtime_mode_source,
        include_paths=config.log_selected_trace_paths,
    )
    return trace_paths, selection


def _trace_selection_metadata(
    trace_paths: list[Path],
    *,
    update: int,
    trace_manifest_id: str | None,
    runtime_mode_source: str,
    include_paths: bool = True,
) -> dict[str, Any]:
    selected = [str(path) for path in trace_paths]
    digest_payload = {
        "runtime_mode_source": runtime_mode_source,
        "trace_manifest_id": trace_manifest_id,
        "selected_trace_paths": selected,
    }
    subset_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    out = {
        "update": int(update),
        "runtime_mode_source": runtime_mode_source,
        "trace_manifest_id": trace_manifest_id,
        "selected_trace_count": len(selected),
        "selected_trace_subset_sha256": subset_digest,
    }
    if include_paths:
        out["selected_trace_paths"] = selected
    return out


def _open_trace_env(
    trace_paths: list[Path],
    config: RustPPOTrainingConfig,
    *,
    library_path: str | Path | None,
) -> RustVecEnv:
    if len(trace_paths) == 1:
        diagnostic_mode = _resolve_diagnostic_mode(config)
        return RustVecEnv.from_trace_file(
            trace_paths[0],
            env_count=config.env_count,
            library_path=library_path,
            observation_key=config.observation_key,
            action_features_dtype=config.action_features_dtype,
            action_features_mode=config.action_features_mode,
            observation_mode=config.observation_mode,
            action_mask_mode=config.action_mask_mode,
            terminal_observation_mode=config.terminal_observation_mode,
            diagnostic_mode=diagnostic_mode,
        )
    diagnostic_mode = _resolve_diagnostic_mode(config)
    return RustVecEnv.from_trace_files(
        trace_paths,
        env_count=config.env_count,
        library_path=library_path,
        observation_key=config.observation_key,
        action_features_dtype=config.action_features_dtype,
        action_features_mode=config.action_features_mode,
        reset_pool_mode=config.trace_pool_reset_mode,
        observation_mode=config.observation_mode,
        action_mask_mode=config.action_mask_mode,
        terminal_observation_mode=config.terminal_observation_mode,
        diagnostic_mode=diagnostic_mode,
    )


def _make_legal_policy(
    model: Any,
    config: RustPPOTrainingConfig,
    *,
    library_path: str | Path | None,
):
    policy_scoring_backend = _resolve_policy_scoring_backend(config)
    if policy_scoring_backend == "compact":
        return make_compact_legal_argmax_policy(
            model,
            selection_backend=config.policy_selection_backend,
            library_path=library_path,
        )
    if policy_scoring_backend == "padded":
        return make_padded_legal_argmax_policy(
            model,
            selection_backend=config.policy_selection_backend,
            padding_mode=config.policy_padding_mode,
            bucket_max_padding_ratio=config.policy_bucket_max_padding_ratio,
            bucket_min_rows=config.policy_bucket_min_rows,
            library_path=library_path,
        )
    raise ValueError("policy_scoring_backend must be compact, padded, or auto")


def _resolve_policy_scoring_backend(config: RustPPOTrainingConfig) -> str:
    if config.policy_scoring_backend in {"compact", "padded"}:
        return config.policy_scoring_backend
    if config.policy_scoring_backend != "auto":
        raise ValueError("policy_scoring_backend must be compact, padded, or auto")
    rollout_rows = int(config.env_count) * int(config.steps_per_update)
    return "compact" if rollout_rows <= 8 else "padded"


def _v5_league_update_metadata(config: RustPPOTrainingConfig, update: int) -> dict[str, Any]:
    league_config = config.v5_league_config or V5LeagueConfig()
    seed = 0 if config.seed is None else int(config.seed)
    modes = sample_v5_episode_modes(league_config, seed=seed, update=int(update))
    info = modes.info_mode
    assist = modes.assist_mode
    return {
        "update": int(update),
        "v5_mode": {
            "adaptive_strength": info.clipped_strength(),
            "own_hand_identity_known": bool(info.own_hand_identity_known),
            "own_deck_known": bool(info.own_deck_known),
            "enemy_hand_known": bool(info.enemy_hand_known),
            "enemy_deck_known": bool(info.enemy_deck_known),
            "enemy_deck_order_known": bool(info.enemy_deck_order_known),
            "draw_assist_enabled": bool(info.draw_assist_enabled),
            "draw_assist_strength": info.clipped_draw_assist_strength(),
        },
        "assist_mode": assist.to_dict(),
        "opponent_mix": [(str(name), float(weight)) for name, weight in modes.opponent_mix],
    }


def _v5_league_summary(
    config: RustPPOTrainingConfig,
    updates: list[dict[str, Any]],
    *,
    trace_manifest_id: str | None = None,
) -> dict[str, Any]:
    league_config = config.v5_league_config or V5LeagueConfig()
    return _jsonable(
        {
            "run_name": config.run_name,
            "model_name": config.model_name,
            "trace_manifest_id": trace_manifest_id,
            "config": asdict(league_config),
            "curriculum_metadata": dict(config.curriculum_metadata),
            "updates": updates,
        }
    )


def _validate_config(config: RustPPOTrainingConfig) -> None:
    if int(config.updates) <= 0:
        raise ValueError("updates must be positive")
    if int(config.env_count) <= 0:
        raise ValueError("env_count must be positive")
    if int(config.steps_per_update) <= 0:
        raise ValueError("steps_per_update must be positive")
    if int(config.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if int(config.minibatch_size) <= 0:
        raise ValueError("minibatch_size must be positive")
    if config.action_features_mode != "legal_only":
        raise ValueError("RustPPOTrainingConfig currently supports action_features_mode='legal_only' only")
    if config.action_features_dtype not in {"float32", "float16"}:
        raise ValueError("action_features_dtype must be float32 or float16")
    if config.trace_pool_reset_mode not in {"fixed", "cycle"}:
        raise ValueError("trace_pool_reset_mode must be fixed or cycle")
    if config.v5_runtime_mode_source not in {"manifest_cycle", "league_schedule"}:
        raise ValueError("v5_runtime_mode_source must be manifest_cycle or league_schedule")
    if config.observation_mode not in {"v1_and_v5", "v5_only"}:
        raise ValueError("observation_mode must be v1_and_v5 or v5_only")
    if config.observation_key == "observation_v1" and config.observation_mode == "v5_only":
        raise ValueError("observation_key='observation_v1' requires observation_mode='v1_and_v5'")
    if config.action_mask_mode not in {"dense", "legal_only"}:
        raise ValueError("action_mask_mode must be dense or legal_only")
    if config.action_features_mode != "legal_only" and config.action_mask_mode == "legal_only":
        raise ValueError("action_mask_mode='legal_only' requires action_features_mode='legal_only'")
    if config.terminal_observation_mode not in {"full", "none"}:
        raise ValueError("terminal_observation_mode must be full or none")
    if config.diagnostic_mode not in {"auto", "full", "none"}:
        raise ValueError("diagnostic_mode must be auto, full, or none")
    if config.diagnostic_mode == "none" and (
        config.store_infos or config.store_reset_flags or config.store_episode_stats
    ):
        raise ValueError("diagnostic_mode='none' cannot store infos, reset flags, or episode stats")
    if config.advantage_backend not in {"python", "rust"}:
        raise ValueError("advantage_backend must be python or rust")
    if config.selected_local_backend not in {"python", "rust", "provided"}:
        raise ValueError("selected_local_backend must be python, rust, or provided")
    if config.prepare_backend not in {"separate", "rust_fused"}:
        raise ValueError("prepare_backend must be separate or rust_fused")
    if config.prepare_backend == "rust_fused" and (
        config.advantage_backend != "rust" or config.selected_local_backend != "rust"
    ):
        raise ValueError("prepare_backend='rust_fused' requires rust advantage and selected-local backends")
    if config.legal_row_pack_backend not in {"python", "rust", "auto"}:
        raise ValueError("legal_row_pack_backend must be python, rust, or auto")
    if config.policy_scoring_backend not in {"compact", "padded", "auto"}:
        raise ValueError("policy_scoring_backend must be compact, padded, or auto")
    if config.policy_selection_backend not in {"python", "rust"}:
        raise ValueError("policy_selection_backend must be python or rust")
    if config.policy_padding_mode not in {"single", "bucketed"}:
        raise ValueError("policy_padding_mode must be single or bucketed")
    if float(config.policy_bucket_max_padding_ratio) < 1.0:
        raise ValueError("policy_bucket_max_padding_ratio must be >= 1.0")
    if int(config.policy_bucket_min_rows) <= 0:
        raise ValueError("policy_bucket_min_rows must be positive")
    if config.ppo_minibatch_plan not in {"contiguous", "legal_count_sorted"}:
        raise ValueError("ppo_minibatch_plan must be contiguous or legal_count_sorted")
    if not str(config.model_name):
        raise ValueError("model_name must not be empty")
    if config.v5_league_config is not None:
        # Force parser validation for opponent names and weights early.
        sample_v5_episode_modes(config.v5_league_config, seed=0 if config.seed is None else int(config.seed), update=1)


def _resolve_diagnostic_mode(config: RustPPOTrainingConfig) -> str:
    if config.diagnostic_mode != "auto":
        return config.diagnostic_mode
    if config.store_infos or config.store_reset_flags or config.store_episode_stats:
        return "full"
    return "none"


def _dense_feature_bytes(batch: RustTransitionBatch) -> int:
    return 0 if batch.action_features is None else int(batch.action_features.nbytes)


def _dense_mask_bytes(batch: RustTransitionBatch) -> int:
    return 0 if batch.action_mask is None else int(batch.action_mask.nbytes)


def _terminal_observation_bytes(batch: RustTransitionBatch) -> int:
    return 0 if batch.terminal_observations is None else int(batch.terminal_observations.nbytes)


def _next_observation_bytes(batch: RustTransitionBatch) -> int:
    return 0 if batch.next_observations is None else int(batch.next_observations.nbytes)


def _truncated_bytes(batch: RustTransitionBatch) -> int:
    return 0 if batch.truncated is None else int(batch.truncated.nbytes)


def _reset_flag_bytes(batch: RustTransitionBatch) -> int:
    return 0 if batch.reset_flags is None else int(batch.reset_flags.nbytes)


def _episode_stat_bytes(batch: RustTransitionBatch) -> int:
    total = 0
    if batch.terminal_observation_valid is not None:
        total += int(batch.terminal_observation_valid.nbytes)
    if batch.episode_returns is not None:
        total += int(batch.episode_returns.nbytes)
    if batch.episode_lengths is not None:
        total += int(batch.episode_lengths.nbytes)
    return total


def _info_dict_count(batch: RustTransitionBatch) -> int:
    return 0 if batch.infos is None else sum(len(step_infos) for step_infos in batch.infos)


def _legal_feature_bytes(batch: RustTransitionBatch) -> int:
    return int(
        batch.legal_action_counts.nbytes
        + batch.legal_action_offsets.nbytes
        + batch.legal_action_ids.nbytes
        + batch.legal_action_features.nbytes
    )


def _summarize_training_speed(
    metrics: list[dict[str, Any]],
    *,
    total_env_transitions: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    total_collect_seconds = sum(float(metric.get("collect_seconds", 0.0)) for metric in metrics)
    total_prepare_seconds = sum(float(metric.get("prepare_seconds", 0.0)) for metric in metrics)
    total_update_seconds = sum(float(metric.get("update_seconds", 0.0)) for metric in metrics)
    total_train_loop_seconds = total_collect_seconds + total_prepare_seconds + total_update_seconds
    total_planned_minibatches = sum(int(metric.get("planned_minibatches", 0)) for metric in metrics)
    total_planned_minibatch_reuses = sum(int(metric.get("planned_minibatch_reuses", 0)) for metric in metrics)
    total_planned_legal_action_rows = sum(int(metric.get("planned_legal_action_rows", 0)) for metric in metrics)
    total_planned_padded_action_rows = sum(int(metric.get("planned_padded_action_rows", 0)) for metric in metrics)
    total_planned_padding_waste_rows = sum(int(metric.get("planned_padding_waste_rows", 0)) for metric in metrics)
    total_planned_padded_total_bytes = sum(int(metric.get("planned_padded_total_bytes", 0)) for metric in metrics)
    total_planned_recomputed_padded_total_bytes = sum(
        int(metric.get("planned_recomputed_padded_total_bytes", 0))
        for metric in metrics
    )
    padded_cache_enabled_updates = sum(1 for metric in metrics if bool(metric.get("padded_cache_enabled", False)))
    total_padded_cache_builds = sum(int(metric.get("padded_cache_builds", 0)) for metric in metrics)
    total_padded_cache_hits = sum(int(metric.get("padded_cache_hits", 0)) for metric in metrics)
    total_padded_cache_reuses = sum(int(metric.get("padded_cache_reuses", 0)) for metric in metrics)
    total_padded_cache_bytes = sum(int(metric.get("padded_cache_bytes", 0)) for metric in metrics)
    total_padded_cache_saved_builds = sum(int(metric.get("padded_cache_saved_builds", 0)) for metric in metrics)
    total_padded_cache_saved_padded_total_bytes = sum(
        int(metric.get("padded_cache_saved_padded_total_bytes", 0))
        for metric in metrics
    )
    policy_scoring_backend = None if not metrics else metrics[0].get("policy_scoring_backend")
    resolved_policy_scoring_backend = (
        None if not metrics else metrics[0].get("resolved_policy_scoring_backend")
    )

    def rate(numerator: int | float, denominator: float) -> float:
        return 0.0 if denominator <= 0.0 else float(numerator) / float(denominator)

    summary = {
        "type": "summary",
        "updates": len(metrics),
        "total_env_transitions": int(total_env_transitions),
        "elapsed_seconds": float(elapsed_seconds),
        "total_collect_seconds": total_collect_seconds,
        "total_prepare_seconds": total_prepare_seconds,
        "total_update_seconds": total_update_seconds,
        "total_train_loop_seconds": total_train_loop_seconds,
        "env_transitions_per_second": rate(total_env_transitions, float(elapsed_seconds)),
        "train_loop_env_transitions_per_second": rate(total_env_transitions, total_train_loop_seconds),
        "collect_time_fraction": rate(total_collect_seconds, total_train_loop_seconds),
        "prepare_time_fraction": rate(total_prepare_seconds, total_train_loop_seconds),
        "update_time_fraction": rate(total_update_seconds, total_train_loop_seconds),
        "policy_scoring_backend": policy_scoring_backend,
        "resolved_policy_scoring_backend": resolved_policy_scoring_backend,
        "padded_cache_enabled_updates": padded_cache_enabled_updates,
        "total_planned_minibatches": total_planned_minibatches,
        "total_planned_minibatch_reuses": total_planned_minibatch_reuses,
        "total_planned_legal_action_rows": total_planned_legal_action_rows,
        "total_planned_padded_action_rows": total_planned_padded_action_rows,
        "total_planned_padding_waste_rows": total_planned_padding_waste_rows,
        "planned_padding_expansion_ratio": rate(
            total_planned_padded_action_rows,
            total_planned_legal_action_rows,
        ),
        "total_planned_padded_total_bytes": total_planned_padded_total_bytes,
        "total_planned_recomputed_padded_total_bytes": total_planned_recomputed_padded_total_bytes,
        "total_padded_cache_builds": total_padded_cache_builds,
        "total_padded_cache_hits": total_padded_cache_hits,
        "total_padded_cache_reuses": total_padded_cache_reuses,
        "total_padded_cache_bytes": total_padded_cache_bytes,
        "total_padded_cache_saved_builds": total_padded_cache_saved_builds,
        "total_padded_cache_saved_padded_total_bytes": total_padded_cache_saved_padded_total_bytes,
        "padded_cache_reuse_fraction": rate(
            total_padded_cache_reuses,
            total_padded_cache_hits,
        ),
        "padded_cache_hit_build_ratio": rate(
            total_padded_cache_hits,
            total_padded_cache_builds,
        ),
        "padded_cache_saved_recomputed_fraction": rate(
            total_padded_cache_saved_padded_total_bytes,
            total_planned_recomputed_padded_total_bytes,
        ),
    }
    _assert_metric_finite(summary)
    return summary


def _append_jsonl(path: str | Path, metric: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(metric), sort_keys=True) + "\n")


def _should_checkpoint(config: RustPPOTrainingConfig, update: int) -> bool:
    if config.checkpoint_dir is None:
        return False
    checkpoint_every = int(config.checkpoint_every)
    return checkpoint_every > 0 and update % checkpoint_every == 0


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    config: RustPPOTrainingConfig,
    *,
    update: int,
    total_env_transitions: int,
    metric: dict[str, Any],
    speed_summary: dict[str, Any],
    trace_pool_size: int,
    trace_env_reuse: bool,
    trace_manifest_id: str | None,
    v5_league: dict[str, Any],
) -> str:
    from ai.train_v2.model_mlx import save_checkpoint

    checkpoint_dir = Path(config.checkpoint_dir)  # type: ignore[arg-type]
    checkpoint_path = checkpoint_dir / f"trainv3_rust_legal_update_{update:04d}.npz"
    metadata = {
        "trainv3_rust_legal_only": True,
        "run_name": config.run_name,
        "model_name": config.model_name,
        "trace_manifest_id": trace_manifest_id,
        "v5_league": _jsonable(v5_league),
        "rollout_mode": "rust_vec_legal_only",
        "trace_pool_size": trace_pool_size,
        "trace_env_reuse": trace_env_reuse,
        "update": update,
        "total_env_transitions": total_env_transitions,
        "obs_dim": int(getattr(model, "obs_dim", 0) or 0),
        "action_feature_dim": int(getattr(model, "action_feature_dim", 171)),
        "max_candidate_actions": 601,
        "config": _config_to_dict(config),
        "last_metrics": _jsonable(metric),
        "speed_summary": _jsonable(speed_summary),
    }
    save_checkpoint(str(checkpoint_path), model, optimizer=optimizer, metadata=metadata)
    return str(checkpoint_path)


def _write_league_run_manifest(
    config: RustPPOTrainingConfig,
    *,
    v5_league: dict[str, Any],
    trace_manifest_id: str | None,
    checkpoint_path: str,
) -> tuple[dict[str, Any] | None, str]:
    if config.run_name is None and trace_manifest_id is None:
        return None, ""

    manifest_path = _resolve_league_manifest_path(config, trace_manifest_id=trace_manifest_id)
    manifest = LeagueRunManifest(
        run_name=str(config.run_name or ""),
        model_name=str(config.model_name),
        trace_manifest_id=str(trace_manifest_id or ""),
        config=_jsonable(v5_league),
        curriculum={
            "curriculum_metadata": dict(config.curriculum_metadata),
            "updates": v5_league.get("updates", []),
        },
        metrics_path=config.metrics_path,
        checkpoint_path=checkpoint_path or None,
    )
    written_path = write_manifest_json(manifest, manifest_path)
    return read_manifest_json(written_path), str(written_path)


def _resolve_league_manifest_path(
    config: RustPPOTrainingConfig,
    *,
    trace_manifest_id: str | None,
) -> Path:
    if config.league_manifest_path is not None:
        return Path(config.league_manifest_path)
    if config.metrics_path is not None:
        metrics_path = Path(config.metrics_path)
        return metrics_path.with_name(f"{metrics_path.stem}.league_manifest.json")
    slug = _league_manifest_slug(config.run_name, trace_manifest_id)
    if config.checkpoint_dir is not None:
        return Path(config.checkpoint_dir) / f"{slug}_league_manifest.json"
    return Path(f"{slug}_league_manifest.json")


def _league_manifest_slug(run_name: str | None, trace_manifest_id: str | None) -> str:
    raw = str(run_name or trace_manifest_id or "league_run")
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return slug.strip("._-") or "league_run"


def _config_to_dict(config: RustPPOTrainingConfig) -> dict[str, Any]:
    return _jsonable(asdict(config))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _assert_metric_finite(metric: dict[str, Any]) -> None:
    _assert_json_metric_finite(metric, "metric")


def _assert_json_metric_finite(value: Any, path: str) -> None:
    if isinstance(value, (int, float, np.generic)) and not np.isfinite(value):
        raise ValueError(f"{path} is non-finite: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_metric_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _assert_json_metric_finite(item, f"{path}[{idx}]")


__all__ = ["RustPPOTrainingConfig", "train_rust_ppo_trace_file", "train_rust_ppo_trace_files"]
