from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.train_v2.train_ppo import TRAIN_PRESETS, make_config_from_preset, estimate_update_memory
from ai.train_v2.experiment import ExperimentConfig, run_experiment


@dataclass
class NightRunConfig:
    name: str = "night_fast"
    output_dir: str = "ai/train_v2/runs"
    preset: str = "m4_night"
    seed: int = 42
    include_preview_features: bool = False
    export_onnx: bool = False
    eval_games: int = 4
    eval_max_steps: int = 200
    dry_run: bool = False
    max_expected_hours: float | None = None
    rollout_workers: int | None = None
    verify_mask: bool | None = None
    placement_mode: str | None = None


def preflight_night_run(config: NightRunConfig) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    # Preset check
    preset = config.preset
    if preset not in TRAIN_PRESETS:
        errors.append(f"Unknown preset: {preset}. Available: {list(TRAIN_PRESETS)}")

    # MLX import check
    try:
        import mlx.core as mx  # noqa: F401
    except Exception as exc:
        errors.append(f"MLX import failed: {exc}")

    # ClassicRLEnv / cards.json check
    try:
        from ai.train_v2.classic_rl_env import ClassicRLEnv
        _env = ClassicRLEnv(seed=config.seed)
    except Exception as exc:
        errors.append(f"ClassicRLEnv init failed: {exc}")

    # Output dir writable check
    out = Path(config.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / f"_night_preflight_test_{os.getpid()}.tmp"
        tmp.write_text("ok")
        tmp.unlink()
    except Exception as exc:
        errors.append(f"Output dir not writable: {exc}")

    # Warnings
    if config.include_preview_features:
        warnings.append("include_preview_features=True: may increase memory and compute")
    if config.export_onnx:
        warnings.append("export_onnx=True: eval/export may dominate total runtime")

    # Estimates (only if preset known)
    estimated_transitions = 0
    estimated_action_feature_mb = 0.0
    estimated_per_update_mb = 0.0
    workers = config.rollout_workers
    if preset in TRAIN_PRESETS:
        ppo_cfg = make_config_from_preset(
            preset,
            rollout_workers=config.rollout_workers,
            verify_mask=config.verify_mask,
            placement_mode=config.placement_mode,
        )
        workers = ppo_cfg.rollout_workers
        p = TRAIN_PRESETS[preset]
        updates = p.get("total_updates", 0)
        eps = p.get("episodes_per_update", 0)
        steps = p.get("max_steps_per_episode", 0)
        estimated_transitions = updates * eps * steps
        # action_features: 601 * 171 floats, 4 bytes each
        bytes_per_transition = 601 * 171 * 4
        estimated_action_feature_mb = (estimated_transitions * bytes_per_transition) / (1024 * 1024)

        mem = estimate_update_memory(ppo_cfg)
        estimated_per_update_mb = mem["rough_peak_mb"]

        if estimated_per_update_mb > 10_000:
            warnings.append(f"Estimated per-update RAM {estimated_per_update_mb:.0f} MB > 10 GB; high risk of swapping")
        elif estimated_per_update_mb > 8_000:
            warnings.append(f"Estimated per-update RAM {estimated_per_update_mb:.0f} MB > 8 GB; reduce workers or steps")

        if config.max_expected_hours is not None and estimated_transitions > 0:
            # very rough: assume ~100 transitions/sec on MLX/MPS as ballpark
            rough_hours = estimated_transitions / (100.0 * 3600.0)
            if rough_hours > config.max_expected_hours:
                warnings.append(
                    f"Estimated rough runtime ({rough_hours:.1f}h) exceeds max_expected_hours ({config.max_expected_hours}h)"
                )

    return {
        "ok": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
        "preset": TRAIN_PRESETS.get(preset) if preset in TRAIN_PRESETS else None,
        "estimated_transitions": estimated_transitions,
        "estimated_action_feature_mb": round(estimated_action_feature_mb, 2),
        "estimated_per_update_mb": round(estimated_per_update_mb, 2),
        "rollout_workers": workers if workers is not None else 1,
    }


def build_night_experiment_config(config: NightRunConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name=config.name,
        output_dir=config.output_dir,
        seed=config.seed,
        preset=config.preset,
        include_preview_features=config.include_preview_features,
        eval_games=config.eval_games,
        eval_max_steps=config.eval_max_steps,
        export_onnx=config.export_onnx,
        rollout_workers=config.rollout_workers,
        verify_mask=config.verify_mask,
        placement_mode=config.placement_mode,
    )


def run_night(config: NightRunConfig) -> dict:
    preflight = preflight_night_run(config)
    if not preflight["ok"]:
        raise RuntimeError(
            "Night run preflight failed:\n" + "\n".join(preflight["errors"])
        )

    exp_cfg = build_night_experiment_config(config)

    if config.dry_run:
        return {
            "version": "train_v2_night_run_v2",
            "dry_run": True,
            "preflight": preflight,
            "planned_experiment_config": {
                "name": exp_cfg.name,
                "output_dir": exp_cfg.output_dir,
                "seed": exp_cfg.seed,
                "preset": exp_cfg.preset,
                "include_preview_features": exp_cfg.include_preview_features,
                "eval_games": exp_cfg.eval_games,
                "eval_max_steps": exp_cfg.eval_max_steps,
                "export_onnx": exp_cfg.export_onnx,
                "rollout_workers": exp_cfg.rollout_workers,
                "verify_mask": exp_cfg.verify_mask,
                "placement_mode": exp_cfg.placement_mode,
            },
        }

    summary = run_experiment(exp_cfg)
    run_dir = Path(summary["run_dir"])

    night_summary = {
        "version": "train_v2_night_run_v2",
        "preflight": preflight,
        "experiment": summary,
        "morning_commands": {
            "monitor": (
                f"python3 -m ai.train_v2.monitor --run {run_dir}"
            ),
            "panel": (
                f"python3 -m ai.train_v2.operator panel "
                f"--runs-dir {config.output_dir} "
                f"--releases-dir {Path(config.output_dir).parent / 'releases'}"
            ),
            "resume": (
                f"python3 -m ai.train_v2.experiment "
                f"--preset {config.preset} "
                f"--resume-checkpoint <CHECKPOINT> "
                f"--name {config.name}_resume "
                f"--output-dir {config.output_dir}"
            ),
            "export": (
                f"python3 -m ai.train_v2.export_onnx "
                f"--checkpoint <CHECKPOINT> --output <ONNX>"
            ),
        },
    }

    summary_path = run_dir / "night_run_summary.json"
    summary_path.write_text(
        json.dumps(night_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return night_summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="TrainV2 overnight training launcher")
    parser.add_argument("--name", default="night_fast")
    parser.add_argument("--preset", default="m4_night", choices=list(TRAIN_PRESETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="ai/train_v2/runs")
    parser.add_argument("--include-preview-features", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--eval-max-steps", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-expected-hours", type=float, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--verify-mask", default=None, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--placement-mode", default=None, choices=["append_only", "full"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = NightRunConfig(
        name=args.name,
        output_dir=args.output_dir,
        preset=args.preset,
        seed=args.seed,
        include_preview_features=args.include_preview_features,
        export_onnx=args.export_onnx,
        eval_games=args.eval_games,
        eval_max_steps=args.eval_max_steps,
        dry_run=args.dry_run,
        max_expected_hours=args.max_expected_hours,
        rollout_workers=args.rollout_workers,
        verify_mask=args.verify_mask,
        placement_mode=args.placement_mode,
    )

    try:
        result = run_night(config)
    except RuntimeError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"Night run preflight: FAILED")
            print(str(exc))
        raise SystemExit(1)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        pf = result["preflight"]
        if result.get("dry_run"):
            print("Night run preflight: OK (dry run)")
            print(f"Preset: {config.preset}")
            print(f"Workers: {pf['rollout_workers']}")
            print(f"Estimated transitions: {pf['estimated_transitions']}")
            print(f"Action feature batch estimate: {pf['estimated_action_feature_mb']:.1f} MB")
            print(f"Per-update memory estimate: {pf['estimated_per_update_mb']:.1f} MB")
            if pf["warnings"]:
                print(f"Warnings: {'; '.join(pf['warnings'])}")
            print("Command:")
            print(
                f"python3 -m ai.train_v2.night_run "
                f"--name {config.name} --preset {config.preset} "
                f"--seed {config.seed} --output-dir {config.output_dir}"
            )
            if config.rollout_workers is not None:
                print(f"  --rollout-workers {config.rollout_workers}")
            if config.verify_mask is not None:
                print(f"  --verify-mask {config.verify_mask}")
            if config.placement_mode is not None:
                print(f"  --placement-mode {config.placement_mode}")
            if config.include_preview_features:
                print("  --include-preview-features")
            if config.export_onnx:
                print("  --export-onnx")
        else:
            exp = result["experiment"]
            print("Night run complete")
            print(f"Run dir: {exp['run_dir']}")
            print(f"Checkpoint: {exp.get('checkpoint_path', 'N/A')}")
            print("Morning:")
            for k, v in result["morning_commands"].items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    _main()
