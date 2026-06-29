#!/usr/bin/env python3
"""Run Extra-LR V5 Phase 2 mixed assist/private-info matrix training."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import mlx.optimizers as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3" / "python"))

from train_v3.league_v5 import V5LeagueConfig
from train_v3.rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_files
from train_v3.trace_factory_v5 import V5TraceScenario, generate_v5_trace_pool
from train_v3.v5_artifacts import read_manifest_json, write_manifest_json
from train_v3.v5_policy import create_v5_policy


DEFAULT_RESUME = (
    ROOT
    / "TrainV3"
    / "runs"
    / "phase1_noassist_refresh_after_assist_20260604_184324"
    / "checkpoints"
    / "trainv3_rust_legal_update_1000.npz"
)


def main() -> int:
    run_name = _env_str("PHASE2_RUN_NAME", "phase2_mixed_assist_private_matrix")
    env_count = _env_int("PHASE2_ENV_COUNT", 4096)
    steps_per_update = _env_int("PHASE2_STEPS_PER_UPDATE", 32)
    updates = _env_int("PHASE2_UPDATES", 1000)
    minibatch_size = _env_int("PHASE2_MINIBATCH_SIZE", 4096)
    checkpoint_every = _env_int("PHASE2_CHECKPOINT_EVERY", 100)
    hidden_dim = _env_int("PHASE2_HIDDEN_DIM", 256)
    action_hidden_dim = _env_int("PHASE2_ACTION_HIDDEN_DIM", 128)
    learning_rate = _env_float("PHASE2_LR", 0.0004)
    entropy_coef = _env_float("PHASE2_ENTROPY_COEF", 0.02)
    max_grad_norm = _env_optional_float("PHASE2_MAX_GRAD_NORM", 0.5)
    strengths = _env_float_tuple("PHASE2_ADAPTIVE_STRENGTHS", (0.25, 0.5, 0.75, 1.0))
    seed = _env_int("PHASE2_SEED", 22001)
    trace_seed_count = _env_int("PHASE2_TRACE_SEED_COUNT", 64)
    resume_checkpoint = _env_path("PHASE2_RESUME_CHECKPOINT", DEFAULT_RESUME)
    no_assist_weight = _env_float("PHASE2_NO_ASSIST_WEIGHT", 0.55)
    assembler_weight = _env_float("PHASE2_ASSEMBLER_WEIGHT", 0.15)
    desirerer_weight = _env_float("PHASE2_DESIRERER_WEIGHT", 0.15)
    both_assist_weight = _env_float("PHASE2_BOTH_ASSIST_WEIGHT", 0.15)
    private_info_rate = _env_float("PHASE2_PRIVATE_INFO_RATE", 0.35)
    draw_assist_rate = _env_float("PHASE2_DRAW_ASSIST_RATE", 0.30)
    draw_assist_min_strength = _env_float("PHASE2_DRAW_ASSIST_MIN_STRENGTH", 0.75)
    out_root = Path(os.environ.get("PHASE2_OUT_ROOT", ROOT / "TrainV3" / "runs")).resolve()
    library_path = Path(
        os.environ.get("TRAINV3_CORE_LIB", ROOT / "TrainV3" / "target" / "release" / "libtrainv3_core.dylib")
    ).resolve()

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    trace_dir = run_dir / "trace_pool"
    manifest_path = run_dir / "trace_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_phase2_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    assist_modes = _assist_modes(
        no_assist_weight=no_assist_weight,
        assembler_weight=assembler_weight,
        desirerer_weight=desirerer_weight,
        both_assist_weight=both_assist_weight,
    )
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
        "private_info_rate": private_info_rate,
        "draw_assist_rate": draw_assist_rate,
        "draw_assist_min_strength": draw_assist_min_strength,
        "assist_modes": assist_modes,
        "library_path": str(library_path),
    }
    (run_dir / "phase2_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE2_RUN_DIR", run_dir, flush=True)
    print("PHASE2_CONFIG", json.dumps(config_snapshot, sort_keys=True), flush=True)

    manifest = generate_v5_trace_pool(
        [
            V5TraceScenario(
                scenario_key="phase2_mixed_assist_private_matrix",
                seeds=tuple(30_000 + idx * 19 for idx in range(trace_seed_count)),
                steps=12,
                adaptive_strengths=strengths,
                visibility_modes=(
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
                ),
                draw_assist_modes=_draw_assist_modes(strengths, min_strength=draw_assist_min_strength),
                assist_modes=tuple(assist_modes),
            )
        ],
        trace_dir,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    write_manifest_json(manifest, manifest_path)
    loaded_manifest = read_manifest_json(manifest_path)
    print(
        "PHASE2_TRACE_MANIFEST",
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
    resume_metadata: dict[str, Any] = {}
    optimizer_restored = False
    if resume_checkpoint is not None:
        from ai.train_v2.model_mlx import load_checkpoint

        loaded = load_checkpoint(str(resume_checkpoint), model, optimizer=optimizer)
        resume_metadata = dict(loaded.get("metadata", {}))
        optimizer_restored = bool(loaded.get("optimizer_restored", False))
        print(
            "PHASE2_RESUME",
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
            mixed_visibility_rate=1.0,
            enemy_private_info_rate=private_info_rate,
            draw_assist_rate=draw_assist_rate,
            draw_assist_min_strength=draw_assist_min_strength,
            teacher_start_update=999999,
            opponent_mix="self:1.0,v5_snapshot:0.5,random:0.02,llm_teacher:0.02",
            assist_modes=tuple(assist_modes),
        ),
        curriculum_metadata={
            "phase": "phase2_mixed_assist_private_matrix",
            "machine": "macbook_pro_m4_pro_24gb",
            "assist_policy": "mixed_no_assist_majority",
            "private_info_policy": "mixed_enemy_hidden_known",
            "teacher_hint_policy": "off",
            "level_handicap_policy": "deferred_until_rust_first_trace_contract",
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
    print("PHASE2_RUN_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _assist_modes(
    *,
    no_assist_weight: float,
    assembler_weight: float,
    desirerer_weight: float,
    both_assist_weight: float,
) -> list[dict[str, float | int | bool]]:
    return [
        {
            "assembler_enabled": False,
            "assembler_strength": 0.0,
            "desirerer_enabled": False,
            "desirerer_strength": 0.0,
            "teacher_hint_available": False,
            "assist_profile_id": 0,
            "weight": float(no_assist_weight),
        },
        {
            "assembler_enabled": True,
            "assembler_strength": 0.8,
            "desirerer_enabled": False,
            "desirerer_strength": 0.0,
            "teacher_hint_available": False,
            "assist_profile_id": 1,
            "weight": float(assembler_weight),
        },
        {
            "assembler_enabled": False,
            "assembler_strength": 0.0,
            "desirerer_enabled": True,
            "desirerer_strength": 0.8,
            "teacher_hint_available": False,
            "assist_profile_id": 2,
            "weight": float(desirerer_weight),
        },
        {
            "assembler_enabled": True,
            "assembler_strength": 1.0,
            "desirerer_enabled": True,
            "desirerer_strength": 1.0,
            "teacher_hint_available": False,
            "assist_profile_id": 3,
            "weight": float(both_assist_weight),
        },
    ]


def _draw_assist_modes(
    strengths: tuple[float, ...],
    *,
    min_strength: float,
) -> tuple[dict[str, float | bool], ...]:
    modes: list[dict[str, float | bool]] = [{"draw_assist_enabled": False, "draw_assist_strength": 0.0}]
    for strength in sorted({float(value) for value in strengths if float(value) >= float(min_strength)}):
        modes.append({"draw_assist_enabled": True, "draw_assist_strength": strength})
    return tuple(modes)


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
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if raw > 10_000_000 else raw / 1024


if __name__ == "__main__":
    raise SystemExit(main())
