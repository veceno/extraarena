#!/usr/bin/env python3
"""Run Extra-LR V5 Phase 1 foundation training from a tmux-friendly entrypoint."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import mlx.optimizers as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3" / "python"))

from train_v3.league_v5 import V5LeagueConfig
from train_v3.rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_files
from train_v3.trace_factory_v5 import V5TraceScenario, generate_v5_trace_pool
from train_v3.v5_artifacts import read_manifest_json, write_manifest_json
from train_v3.v5_policy import create_v5_policy


def main() -> int:
    run_name = _env_str("PHASE1_RUN_NAME", "phase1_foundation_noassist")
    env_count = _env_int("PHASE1_ENV_COUNT", 4096)
    steps_per_update = _env_int("PHASE1_STEPS_PER_UPDATE", 32)
    updates = _env_int("PHASE1_UPDATES", 1000)
    minibatch_size = _env_int("PHASE1_MINIBATCH_SIZE", 4096)
    checkpoint_every = _env_int("PHASE1_CHECKPOINT_EVERY", 100)
    hidden_dim = _env_int("PHASE1_HIDDEN_DIM", 256)
    action_hidden_dim = _env_int("PHASE1_ACTION_HIDDEN_DIM", 128)
    learning_rate = _env_float("PHASE1_LR", 1.0e-3)
    entropy_coef = _env_float("PHASE1_ENTROPY_COEF", 0.01)
    max_grad_norm = _env_optional_float("PHASE1_MAX_GRAD_NORM")
    strengths = _env_float_tuple("PHASE1_ADAPTIVE_STRENGTHS", (0.25, 0.5, 0.75, 1.0))
    seed = _env_int("PHASE1_SEED", 11001)
    trace_seed_count = _env_int("PHASE1_TRACE_SEED_COUNT", 64)
    resume_checkpoint = _env_path("PHASE1_RESUME_CHECKPOINT")
    enemy_private_known = _env_bool("PHASE1_ENEMY_PRIVATE_KNOWN", False)
    draw_assist_enabled = _env_bool("PHASE1_DRAW_ASSIST_ENABLED", False)
    draw_assist_min_strength = _env_float("PHASE1_DRAW_ASSIST_MIN_STRENGTH", 0.75)
    assembler_enabled = _env_bool("PHASE1_ASSEMBLER_ENABLED", False)
    assembler_strength = _env_float("PHASE1_ASSEMBLER_STRENGTH", 0.0)
    desirerer_enabled = _env_bool("PHASE1_DESIRERER_ENABLED", False)
    desirerer_strength = _env_float("PHASE1_DESIRERER_STRENGTH", 0.0)
    assist_profile_id = _env_int("PHASE1_ASSIST_PROFILE_ID", 0)
    out_root = Path(os.environ.get("PHASE1_OUT_ROOT", ROOT / "TrainV3" / "runs")).resolve()
    library_path = Path(
        os.environ.get("TRAINV3_CORE_LIB", ROOT / "TrainV3" / "target" / "release" / "libtrainv3_core.dylib")
    ).resolve()

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    trace_dir = run_dir / "trace_pool"
    manifest_path = run_dir / "trace_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
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
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "enemy_private_known": enemy_private_known,
        "draw_assist_enabled": draw_assist_enabled,
        "draw_assist_min_strength": draw_assist_min_strength,
        "assembler_enabled": assembler_enabled,
        "assembler_strength": assembler_strength,
        "desirerer_enabled": desirerer_enabled,
        "desirerer_strength": desirerer_strength,
        "assist_profile_id": assist_profile_id,
        "library_path": str(library_path),
    }
    (run_dir / "phase1_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_root / "latest_phase1_foundation_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    print("PHASE1_RUN_DIR", run_dir, flush=True)
    print("PHASE1_CONFIG", json.dumps(config_snapshot, sort_keys=True), flush=True)

    manifest = generate_v5_trace_pool(
        [
            _foundation_scenario(
                trace_seed_count,
                strengths=strengths,
                enemy_private_known=enemy_private_known,
                draw_assist_modes=_draw_assist_modes(
                    strengths,
                    enabled=draw_assist_enabled,
                    min_strength=draw_assist_min_strength,
                ),
                assist_mode={
                    "assembler_enabled": assembler_enabled,
                    "assembler_strength": assembler_strength,
                    "desirerer_enabled": desirerer_enabled,
                    "desirerer_strength": desirerer_strength,
                    "teacher_hint_available": False,
                    "assist_profile_id": assist_profile_id,
                },
            )
        ],
        trace_dir,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    write_manifest_json(manifest, manifest_path)
    loaded_manifest = read_manifest_json(manifest_path)
    print(
        "PHASE1_TRACE_MANIFEST",
        manifest_path,
        loaded_manifest["manifest_id"],
        "traces",
        len(loaded_manifest["traces"]),
        flush=True,
    )

    model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=hidden_dim,
        action_hidden_dim=action_hidden_dim,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    resume_metadata = {}
    optimizer_restored = False
    if resume_checkpoint is not None:
        from ai.train_v2.model_mlx import load_checkpoint

        loaded = load_checkpoint(str(resume_checkpoint), model, optimizer=optimizer)
        resume_metadata = dict(loaded.get("metadata", {}))
        optimizer_restored = bool(loaded.get("optimizer_restored", False))
        print(
            "PHASE1_RESUME",
            json.dumps(
                {
                    "checkpoint": str(resume_checkpoint),
                    "optimizer_restored": optimizer_restored,
                    "source_update": resume_metadata.get("update"),
                    "source_total_env_transitions": resume_metadata.get("total_env_transitions"),
                    "source_run_name": resume_metadata.get("run_name"),
                    "source_model_name": resume_metadata.get("model_name"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    train_config = RustPPOTrainingConfig(
        run_name=run_name,
        v5_league_config=V5LeagueConfig(
            adaptive_strengths=strengths,
            mixed_visibility_rate=1.0 if enemy_private_known else 0.0,
            enemy_private_info_rate=1.0 if enemy_private_known else 0.0,
            draw_assist_rate=1.0 if draw_assist_enabled else 0.0,
            draw_assist_min_strength=draw_assist_min_strength,
            teacher_start_update=999999,
            opponent_mix="self:1.0,v5_snapshot:0.35,random:0.03,llm_teacher:0.02",
            assist_modes=(
                {
                    "assembler_enabled": assembler_enabled,
                    "assembler_strength": assembler_strength,
                    "desirerer_enabled": desirerer_enabled,
                    "desirerer_strength": desirerer_strength,
                    "teacher_hint_available": False,
                    "assist_profile_id": assist_profile_id,
                    "weight": 1.0,
                },
            ),
        ),
        curriculum_metadata={
            "phase": "phase1_foundation",
            "machine": "macbook_pro_m4_pro_24gb",
            "assist_policy": "assist_enabled" if (assembler_enabled or desirerer_enabled or draw_assist_enabled) else "no_assist",
            "private_info_policy": "enemy_known" if enemy_private_known else "enemy_hidden",
            "teacher_hint_policy": "off",
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
            "resume_source_update": resume_metadata.get("update", 0),
            "resume_optimizer_restored": optimizer_restored,
        },
        updates=updates,
        env_count=env_count,
        steps_per_update=steps_per_update,
        epochs=1,
        minibatch_size=minibatch_size,
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
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_source_update": resume_metadata.get("update", 0),
        "resume_optimizer_restored": optimizer_restored,
        "trace_manifest_id": result["trace_manifest_id"],
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
        "dense_bytes_any": max(int(item["stored_dense_feature_bytes"]) for item in metrics),
        "next_observation_bytes_any": max(int(item["stored_next_observation_bytes"]) for item in metrics),
        "terminal_observation_bytes_any": max(int(item["stored_terminal_observation_bytes"]) for item in metrics),
        "checkpoint_path": result["checkpoint_path"],
        "league_manifest_path": result["league_manifest_path"],
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "max_rss_mb": _rss_mb(),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE1_RUN_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _foundation_scenario(
    seed_count: int,
    *,
    strengths: tuple[float, ...],
    enemy_private_known: bool,
    draw_assist_modes: tuple[dict[str, float | bool], ...],
    assist_mode: dict[str, float | int | bool],
) -> V5TraceScenario:
    seeds = tuple(10_000 + idx * 17 for idx in range(int(seed_count)))
    return V5TraceScenario(
        scenario_key="phase1_foundation_noassist",
        seeds=seeds,
        steps=10,
        adaptive_strengths=strengths,
        visibility_modes=(
            {
                "own_hand_identity_known": True,
                "own_deck_known": True,
                "enemy_hand_known": bool(enemy_private_known),
                "enemy_deck_known": bool(enemy_private_known),
                "enemy_deck_order_known": False,
            },
        ),
        draw_assist_modes=draw_assist_modes,
        assist_modes=(assist_mode,),
    )


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value is None or not value.strip() else value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default if value is None or not value.strip() else value)


def _env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return float(value)


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(default)
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return parsed


def _draw_assist_modes(
    strengths: tuple[float, ...],
    *,
    enabled: bool,
    min_strength: float,
) -> tuple[dict[str, float | bool], ...]:
    if not enabled:
        return ({"draw_assist_enabled": False, "draw_assist_strength": 0.0},)
    modes: list[dict[str, float | bool]] = [{"draw_assist_enabled": False, "draw_assist_strength": 0.0}]
    for strength in sorted({float(value) for value in strengths if float(value) >= float(min_strength)}):
        modes.append({"draw_assist_enabled": True, "draw_assist_strength": strength})
    return tuple(modes)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if raw > 10_000_000 else raw / 1024


if __name__ == "__main__":
    raise SystemExit(main())
