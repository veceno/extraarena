"""Phase 9 broad-opponent V5 training runner helpers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .league_v5 import V5LeagueConfig
from .opponents_v5 import assert_phase9_broad_environment_ready
from .trace_factory_v5 import V5TraceScenario, generate_v5_trace_pool
from .v5_artifacts import read_manifest_json, write_manifest_json


def build_phase9_league_config(
    opponent_environment: dict[str, Any],
    *,
    adaptive_strengths: tuple[float, ...],
) -> V5LeagueConfig:
    ready = assert_phase9_broad_environment_ready(opponent_environment)
    return V5LeagueConfig(
        adaptive_strengths=tuple(float(value) for value in adaptive_strengths),
        mixed_visibility_rate=0.55,
        enemy_private_info_rate=0.35,
        draw_assist_rate=0.40,
        draw_assist_min_strength=0.75,
        teacher_start_update=0,
        opponent_mix=str(ready["opponent_mix"]),
        assist_modes=_phase9_assist_modes(),
    )


def phase9_curriculum_metadata(
    *,
    opponent_environment: dict[str, Any],
    opponent_environment_path: str | Path,
    resume_checkpoint: str | Path | None,
    rust_exploit_gauntlet_path: str | Path | None,
    v4max_smoke_path: str | Path | None,
) -> dict[str, Any]:
    ready = assert_phase9_broad_environment_ready(opponent_environment)
    lanes = ready["lanes"]
    return {
        "phase": "phase9_broad_opponent_blend",
        "machine": "macbook_pro_m4_pro_24gb",
        "target_v4max_score": 0.75,
        "v4_1_included": False,
        "phase8_rejected_do_not_resume": True,
        "resume_checkpoint": str(resume_checkpoint or ""),
        "opponent_environment_path": str(opponent_environment_path),
        "rust_exploit_gauntlet_path": str(rust_exploit_gauntlet_path or ""),
        "v4max_smoke_path": str(v4max_smoke_path or ""),
        "opponent_lane_count": len(lanes),
        "opponent_lanes": [lane["kind"] for lane in lanes],
        "online_opponent_router_enabled": False,
        "opponent_execution_contract": {
            "legacy_opponent_mix_is_metadata_only": True,
            "broad_opponent_environment_ready": True,
            "rust_exploit_lanes_executed": rust_exploit_gauntlet_path is not None,
            "v4max_teacher_labels_offline": True,
            "v4max_h2h_smoke_executed": v4max_smoke_path is not None,
            "llm_teacher_offline_labels_only": True,
            "v5_self_and_snapshot_policy_control": True,
            "training_trace_pool_uses_real_lane_artifacts": True,
        },
        "assist_matrix": [
            "no_assist",
            "assembler_only",
            "desirerer_only",
            "assembler_desirerer_teacher_hint",
        ],
        "private_info_matrix": ["enemy_hidden", "enemy_private_known"],
        "level_handicap_matrix": ["equal_l1", "p1_l1_vs_p2_l2", "p1_l2_vs_p2_l1"],
    }


def build_phase9_trace_scenarios(
    *,
    seed: int,
    trace_seed_count: int,
    adaptive_strengths: tuple[float, ...],
) -> list[V5TraceScenario]:
    seeds = tuple(int(seed) + 90_000 + idx * 17 for idx in range(int(trace_seed_count)))
    visibility_modes = (
        {
            "own_hand_identity_known": True,
            "own_deck_known": True,
            "enemy_hand_known": False,
            "enemy_deck_known": False,
            "enemy_deck_order_known": False,
        },
        {
            "own_hand_identity_known": True,
            "own_deck_known": True,
            "enemy_hand_known": True,
            "enemy_deck_known": True,
            "enemy_deck_order_known": False,
        },
    )
    draw_assist_modes = _phase9_draw_assist_modes(adaptive_strengths)
    return [
        V5TraceScenario(
            scenario_key="phase9_broad_foundation_mix",
            seeds=seeds,
            steps=14,
            adaptive_strengths=adaptive_strengths,
            visibility_modes=visibility_modes,
            draw_assist_modes=draw_assist_modes,
            assist_modes=_phase9_assist_modes(),
            level_modes=_phase9_level_modes(),
            choose="first",
        ),
        V5TraceScenario(
            scenario_key="phase9_broad_reverse_choice_mix",
            seeds=seeds,
            steps=14,
            adaptive_strengths=adaptive_strengths,
            visibility_modes=visibility_modes,
            draw_assist_modes=draw_assist_modes,
            assist_modes=_phase9_assist_modes(),
            level_modes=_phase9_level_modes(),
            choose="last",
        ),
    ]


def run_phase9_broad_training(root: str | Path | None = None) -> dict[str, Any]:
    import mlx.optimizers as optim

    from .rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_files
    from .v5_policy import create_v5_policy

    root_path = Path(root).resolve() if root is not None else Path.cwd().resolve()
    run_name = _env_str("PHASE9_RUN_NAME", "phase9_broad_opponent_blend")
    env_count = _env_int("PHASE9_ENV_COUNT", 24576)
    steps_per_update = _env_int("PHASE9_STEPS_PER_UPDATE", 32)
    updates = _env_int("PHASE9_UPDATES", 5000)
    minibatch_size = _env_int("PHASE9_MINIBATCH_SIZE", 8192)
    checkpoint_every = _env_int("PHASE9_CHECKPOINT_EVERY", 100)
    hidden_dim = _env_int("PHASE9_HIDDEN_DIM", 256)
    action_hidden_dim = _env_int("PHASE9_ACTION_HIDDEN_DIM", 128)
    learning_rate = _env_float("PHASE9_LR", 0.00016)
    entropy_coef = _env_float("PHASE9_ENTROPY_COEF", 0.028)
    max_grad_norm = _env_optional_float("PHASE9_MAX_GRAD_NORM", 0.5)
    seed = _env_int("PHASE9_SEED", 99001)
    trace_seed_count = _env_int("PHASE9_TRACE_SEED_COUNT", 24)
    strengths = _env_float_tuple("PHASE9_ADAPTIVE_STRENGTHS", (0.25, 0.5, 0.75, 1.0))
    policy_padding_mode = _env_str("PHASE9_POLICY_PADDING_MODE", "bucketed")
    policy_bucket_max_padding_ratio = _env_float("PHASE9_POLICY_BUCKET_MAX_PADDING_RATIO", 1.35)
    policy_bucket_min_rows = _env_int("PHASE9_POLICY_BUCKET_MIN_ROWS", 2048)
    ppo_minibatch_plan = _env_str("PHASE9_PPO_MINIBATCH_PLAN", "contiguous")
    log_selected_trace_paths = _env_bool("PHASE9_LOG_SELECTED_TRACE_PATHS", False)
    out_root = Path(os.environ.get("PHASE9_OUT_ROOT", root_path / "TrainV3" / "runs")).resolve()
    reuse_trace_manifest_path = _env_path("PHASE9_TRACE_MANIFEST_PATH", None)
    library_path = Path(
        os.environ.get("TRAINV3_CORE_LIB", root_path / "TrainV3" / "target" / "release" / "libtrainv3_core.dylib")
    ).resolve()
    resume_checkpoint = _env_path(
        "PHASE9_RESUME_CHECKPOINT",
        root_path
        / "TrainV3"
        / "runs"
        / "phase4_mixed_assist_private_refresh_after_handicap_20260606_145409"
        / "checkpoints"
        / "trainv3_rust_legal_update_1000.npz",
    )
    opponent_environment_path = _phase9_environment_path(root_path)
    opponent_environment = assert_phase9_broad_environment_ready(read_manifest_json(opponent_environment_path))

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    trace_dir = run_dir / "trace_pool"
    manifest_path = run_dir / "trace_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_phase9_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    config_snapshot = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "opponent_environment_path": str(opponent_environment_path),
        "env_count": env_count,
        "steps_per_update": steps_per_update,
        "updates": updates,
        "minibatch_size": minibatch_size,
        "checkpoint_every": checkpoint_every,
        "hidden_dim": hidden_dim,
        "action_hidden_dim": action_hidden_dim,
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "max_grad_norm": max_grad_norm,
        "adaptive_strengths": list(strengths),
        "seed": seed,
        "trace_seed_count": trace_seed_count,
        "trace_manifest_path": str(reuse_trace_manifest_path) if reuse_trace_manifest_path is not None else str(manifest_path),
        "trace_manifest_reused": reuse_trace_manifest_path is not None,
        "policy_padding_mode": policy_padding_mode,
        "policy_bucket_max_padding_ratio": policy_bucket_max_padding_ratio,
        "policy_bucket_min_rows": policy_bucket_min_rows,
        "ppo_minibatch_plan": ppo_minibatch_plan,
        "log_selected_trace_paths": log_selected_trace_paths,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "library_path": str(library_path),
        "phase8_rejected_do_not_resume": True,
        "v4_1_included": False,
    }
    (run_dir / "phase9_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE9_RUN_DIR", run_dir, flush=True)
    print("PHASE9_CONFIG", json.dumps(config_snapshot, sort_keys=True), flush=True)

    if reuse_trace_manifest_path is not None:
        manifest_path = reuse_trace_manifest_path
    else:
        manifest = generate_v5_trace_pool(
            build_phase9_trace_scenarios(
                seed=seed,
                trace_seed_count=trace_seed_count,
                adaptive_strengths=strengths,
            ),
            trace_dir,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        write_manifest_json(manifest, manifest_path)
    loaded_manifest = read_manifest_json(manifest_path)
    print(
        "PHASE9_TRACE_MANIFEST",
        manifest_path,
        loaded_manifest["manifest_id"],
        "traces",
        len(loaded_manifest["traces"]),
        flush=True,
    )

    rust_exploit_gauntlet_path = _run_rust_exploit_gauntlet(
        manifest_path=manifest_path,
        run_dir=run_dir,
        root=root_path,
        limit=_env_int("PHASE9_GAUNTLET_TRACE_LIMIT", 32),
        steps=_env_int("PHASE9_GAUNTLET_STEPS", 12),
    )

    v4max_smoke_path = None
    if _env_bool("PHASE9_RUN_V4MAX_SMOKE", False):
        v4max_smoke_path = _run_v4max_smoke(
            root=root_path,
            run_dir=run_dir,
            checkpoint=resume_checkpoint,
            games=_env_int("PHASE9_V4MAX_SMOKE_GAMES", 2),
            seed=seed + 777,
        )

    model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=hidden_dim,
        action_hidden_dim=action_hidden_dim,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    resume_metadata: dict[str, Any] = {}
    optimizer_restored = False
    if resume_checkpoint is not None:
        from ai.train_v2.model_mlx import load_checkpoint

        loaded = load_checkpoint(str(resume_checkpoint), model, optimizer=optimizer)
        resume_metadata = dict(loaded.get("metadata", {}))
        optimizer_restored = bool(loaded.get("optimizer_restored", False))
        print(
            "PHASE9_RESUME",
            json.dumps(
                {
                    "checkpoint": str(resume_checkpoint),
                    "optimizer_restored": optimizer_restored,
                    "source_update": resume_metadata.get("update"),
                    "source_run_name": resume_metadata.get("run_name"),
                    "source_model_name": resume_metadata.get("model_name"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    curriculum = phase9_curriculum_metadata(
        opponent_environment=opponent_environment,
        opponent_environment_path=opponent_environment_path,
        resume_checkpoint=resume_checkpoint,
        rust_exploit_gauntlet_path=rust_exploit_gauntlet_path,
        v4max_smoke_path=v4max_smoke_path,
    )
    curriculum["resume_source_update"] = resume_metadata.get("update", 0)
    curriculum["resume_optimizer_restored"] = optimizer_restored

    train_config = RustPPOTrainingConfig(
        run_name=run_name,
        v5_league_config=build_phase9_league_config(opponent_environment, adaptive_strengths=strengths),
        curriculum_metadata=curriculum,
        updates=updates,
        env_count=env_count,
        steps_per_update=steps_per_update,
        epochs=1,
        minibatch_size=minibatch_size,
        clip_epsilon=0.16,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        checkpoint_dir=run_dir / "checkpoints",
        checkpoint_every=checkpoint_every,
        metrics_path=run_dir / "metrics.jsonl",
        league_manifest_path=run_dir / "league_manifest.json",
        trace_manifest_path=manifest_path,
        v5_runtime_mode_source="league_schedule",
        policy_scoring_backend="padded",
        policy_selection_backend="rust",
        policy_padding_mode=policy_padding_mode,
        policy_bucket_max_padding_ratio=policy_bucket_max_padding_ratio,
        policy_bucket_min_rows=policy_bucket_min_rows,
        ppo_minibatch_plan=ppo_minibatch_plan,
        log_selected_trace_paths=log_selected_trace_paths,
        full_batch_eval=False,
        diagnostic_mode="none",
        seed=seed,
    )

    started = time.perf_counter()
    result = train_rust_ppo_trace_files([], model, optimizer, train_config, library_path=library_path)
    elapsed = time.perf_counter() - started
    metrics = result["metrics"]
    collect_tps = [int(item["env_transitions"]) / float(item["collect_seconds"]) for item in metrics]
    summary = {
        "status": "ok",
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "trace_manifest_id": result["trace_manifest_id"],
        "opponent_environment_path": str(opponent_environment_path),
        "updates": updates,
        "env_count": env_count,
        "steps_per_update": steps_per_update,
        "total_env_transitions": int(result["total_env_transitions"]),
        "elapsed_seconds": elapsed,
        "end_to_end_transitions_per_second": int(result["total_env_transitions"]) / elapsed,
        "mean_collect_transitions_per_second": sum(collect_tps) / len(collect_tps),
        "max_abs_approx_kl": max(abs(float(item["approx_kl"])) for item in metrics),
        "min_entropy": min(float(item["entropy"]) for item in metrics),
        "last_loss": float(metrics[-1]["loss"]),
        "last_approx_kl": float(metrics[-1]["approx_kl"]),
        "last_entropy": float(metrics[-1]["entropy"]),
        "checkpoint_path": result["checkpoint_path"],
        "league_manifest_path": result["league_manifest_path"],
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "rust_exploit_gauntlet_path": str(rust_exploit_gauntlet_path),
        "v4max_smoke_path": str(v4max_smoke_path or ""),
        "max_rss_mb": _rss_mb(),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE9_RUN_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _phase9_assist_modes() -> tuple[dict[str, Any], ...]:
    return (
        {
            "assembler_enabled": False,
            "assembler_strength": 0.0,
            "desirerer_enabled": False,
            "desirerer_strength": 0.0,
            "teacher_hint_available": False,
            "assist_profile_id": 0,
            "weight": 2.0,
        },
        {
            "assembler_enabled": True,
            "assembler_strength": 1.0,
            "desirerer_enabled": False,
            "desirerer_strength": 0.0,
            "teacher_hint_available": False,
            "assist_profile_id": 4,
            "weight": 0.8,
        },
        {
            "assembler_enabled": False,
            "assembler_strength": 0.0,
            "desirerer_enabled": True,
            "desirerer_strength": 1.0,
            "teacher_hint_available": False,
            "assist_profile_id": 8,
            "weight": 0.8,
        },
        {
            "assembler_enabled": True,
            "assembler_strength": 1.0,
            "desirerer_enabled": True,
            "desirerer_strength": 1.0,
            "teacher_hint_available": True,
            "assist_profile_id": 15,
            "weight": 1.0,
        },
    )


def _phase9_draw_assist_modes(adaptive_strengths: Iterable[float]) -> tuple[dict[str, Any], ...]:
    enabled_strengths = sorted(
        {
            round(float(strength), 6)
            for strength in adaptive_strengths
            if float(strength) >= 0.75
        }
    )
    return (
        {"draw_assist_enabled": False, "draw_assist_strength": 0.0},
        *(
            {"draw_assist_enabled": True, "draw_assist_strength": float(strength)}
            for strength in enabled_strengths
        ),
    )


def _phase9_level_modes() -> tuple[dict[str, Any], ...]:
    raw = os.environ.get("PHASE9_LEVEL_MODES_JSON")
    if raw is not None and raw.strip():
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("PHASE9_LEVEL_MODES_JSON must be a non-empty JSON list")
        return tuple(_normalize_level_mode(item) for item in parsed)
    return (
        {"p1_level": 1, "p2_level": 1, "label": "equal_l1"},
        {"p1_level": 1, "p2_level": 2, "label": "p1_l1_vs_p2_l2"},
        {"p1_level": 2, "p2_level": 1, "label": "p1_l2_vs_p2_l1"},
    )


def _normalize_level_mode(value: Any) -> dict[str, int | str]:
    if not isinstance(value, dict):
        raise ValueError("each level mode must be an object")
    p1 = max(1, min(10, int(value.get("p1_level", 1))))
    p2 = max(1, min(10, int(value.get("p2_level", 1))))
    label = str(value.get("label") or f"p1_l{p1}_vs_p2_l{p2}")
    return {"p1_level": p1, "p2_level": p2, "label": label}


def _phase9_environment_path(root: Path) -> Path:
    explicit = os.environ.get("PHASE9_OPPONENT_ENVIRONMENT")
    if explicit and explicit.strip():
        return Path(explicit).expanduser().resolve()
    latest = root / "TrainV3" / "runs" / "latest_phase9_opponent_environment.txt"
    run_dir = Path(latest.read_text(encoding="utf-8").strip()).resolve()
    return run_dir / "phase9_opponent_environment.json"


def _run_rust_exploit_gauntlet(
    *,
    manifest_path: Path,
    run_dir: Path,
    root: Path,
    limit: int,
    steps: int,
) -> Path:
    out = run_dir / "rust_exploit_gauntlet.json"
    cmd = [
        "cargo",
        "run",
        "--manifest-path",
        str(root / "TrainV3" / "Cargo.toml"),
        "--quiet",
        "--bin",
        "trainv3_kernel",
        "--",
        "gauntlet-manifest",
        str(manifest_path),
        str(int(limit)),
        str(int(steps)),
    ]
    completed = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    if not payload.get("ok") or int(payload.get("executed_steps", 0)) <= 0:
        raise RuntimeError("Rust exploit gauntlet did not execute cleanly")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _run_v4max_smoke(
    *,
    root: Path,
    run_dir: Path,
    checkpoint: Path | None,
    games: int,
    seed: int,
) -> Path:
    if checkpoint is None:
        raise ValueError("V4-max smoke requires a V5 checkpoint")
    out_dir = run_dir / "v4max_smoke"
    cmd = [
        sys.executable,
        str(root / "TrainV3" / "scripts" / "run_v5_vs_v4max_benchmark.py"),
        "--games",
        str(int(games)),
        "--seed",
        str(int(seed)),
        "--max-steps",
        "160",
        "--start-mode",
        "both",
        "--v5-checkpoint",
        str(checkpoint),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=root, check=True)
    return out_dir / "v5_s1_assist_vs_v4max.json"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value is None or not value.strip() else value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default if value is None or not value.strip() else value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(default)
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return parsed


def _env_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default.resolve() if default is not None else None
    return Path(value).expanduser().resolve()


def _rss_mb() -> float:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(completed.stdout.strip() or "0") / 1024.0
    except Exception:
        return 0.0


__all__ = [
    "build_phase9_league_config",
    "build_phase9_trace_scenarios",
    "phase9_curriculum_metadata",
    "run_phase9_broad_training",
]
