from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ai.train_v2.experiment import ExperimentConfig, run_experiment
from ai.train_v2.report import load_run_report, format_report_markdown
from ai.train_v2.run_index import build_run_index, save_run_index
from ai.train_v2.leaderboard import build_leaderboard, save_leaderboard


@dataclass
class SuiteConfig:
    name: str = "suite"
    output_dir: str = "ai/train_v2/runs"
    seeds: list[int] = field(default_factory=lambda: [42])
    presets: list[str] = field(default_factory=lambda: ["smoke"])
    include_preview_variants: bool = False
    updates: int | None = None
    episodes_per_update: int | None = None
    max_steps: int | None = None
    eval_games: int = 2
    eval_max_steps: int = 100
    export_onnx: bool = True
    build_leaderboard: bool = True
    leaderboard_games: int = 4
    leaderboard_max_steps: int = 100
    continue_on_error: bool = True
    promote_best: bool = True
    candidate_dir: str | None = None


def promote_best_candidate(
    *,
    best_row: dict,
    suite_dir: str,
    candidate_dir: str | None = None,
) -> dict | None:
    onnx_src = best_row.get("onnx_path")
    if not onnx_src:
        return None

    cdir = Path(candidate_dir) if candidate_dir else Path(suite_dir) / "candidates"
    ts = time.strftime("%Y%m%d_%H%M%S")
    subdir = cdir / f"{best_row['model_name']}_{ts}"
    subdir.mkdir(parents=True, exist_ok=True)

    candidate_onnx = subdir / Path(onnx_src).name
    shutil.copy2(onnx_src, candidate_onnx)

    sidecar_src = onnx_src + ".json"
    if Path(sidecar_src).is_file():
        shutil.copy2(sidecar_src, subdir / Path(sidecar_src).name)

    run_dir = Path(onnx_src).parent.parent
    report_src = run_dir / "report.md"
    if report_src.is_file():
        shutil.copy2(report_src, subdir / "report.md")

    suite_lb = Path(suite_dir) / "leaderboard.json"
    if suite_lb.is_file():
        shutil.copy2(suite_lb, subdir / "leaderboard.json")

    meta = {
        "source_onnx": onnx_src,
        "candidate_onnx": str(candidate_onnx),
        "model_name": best_row["model_name"],
        "score": best_row["score"],
        "source_run_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (subdir / "candidate.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "candidate_dir": str(subdir),
        "candidate_onnx": str(candidate_onnx),
        "candidate_meta": str(subdir / "candidate.json"),
    }


def run_suite(config: SuiteConfig) -> dict:
    base = Path(config.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_dir = base / f"{config.name}_{timestamp}"
    if suite_dir.exists():
        suite_dir = base / f"{config.name}_{timestamp}_{os.getpid()}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    variants = [False]
    if config.include_preview_variants:
        variants = [False, True]

    run_summaries: list[dict] = []
    run_statuses: list[dict] = []

    for preset in config.presets:
        for seed in config.seeds:
            for preview in variants:
                suffix = "preview" if preview else "fast"
                exp_name = f"{preset}_seed{seed}_{suffix}"

                exp_cfg = ExperimentConfig(
                    name=exp_name,
                    output_dir=str(suite_dir),
                    seed=seed,
                    preset=preset,
                    include_preview_features=preview,
                    eval_games=config.eval_games,
                    eval_max_steps=config.eval_max_steps,
                    export_onnx=config.export_onnx,
                )

                if config.updates is not None:
                    exp_cfg.updates = config.updates
                if config.episodes_per_update is not None:
                    exp_cfg.episodes_per_update = config.episodes_per_update
                if config.max_steps is not None:
                    exp_cfg.max_steps = config.max_steps

                try:
                    summary = run_experiment(exp_cfg)
                    run_statuses.append({
                        "status": "ok",
                        "preset": preset,
                        "seed": seed,
                        "variant": suffix,
                        "run_dir": summary["run_dir"],
                        "checkpoint_path": summary.get("checkpoint_path"),
                        "onnx_path": summary.get("onnx_path"),
                        "steps": summary.get("train", {}).get("steps"),
                    })
                    run_summaries.append(summary)
                except Exception:
                    if not config.continue_on_error:
                        raise
                    tb = traceback.format_exc()
                    run_statuses.append({
                        "status": "error",
                        "preset": preset,
                        "seed": seed,
                        "variant": suffix,
                        "run_dir": None,
                        "expected_run_name": exp_name,
                        "error": str(sys.exc_info()[1]),
                        "traceback": tb,
                    })

    # Build run index
    index = build_run_index(str(suite_dir))
    run_index_path = str(suite_dir / "run_index.json")
    save_run_index(index, run_index_path)

    # Build leaderboard
    leaderboard_path: str | None = None
    leaderboard: dict | None = None
    if config.export_onnx and config.build_leaderboard:
        has_onnx = any(s.get("onnx_path") for s in run_statuses if s["status"] == "ok")
        if has_onnx:
            seeds = list(range(config.seeds[0], config.seeds[0] + config.leaderboard_games))
            leaderboard = build_leaderboard(
                paths=[str(suite_dir)],
                seeds=seeds,
                opponents=["random", "end_turn", "greedy_face"],
                max_steps=config.leaderboard_max_steps,
            )
            leaderboard_path = str(suite_dir / "leaderboard.json")
            save_leaderboard(leaderboard, leaderboard_path)

    # Generate per-run markdown reports
    reports: list[str] = []
    for status in run_statuses:
        run_dir = status.get("run_dir")
        if run_dir is None:
            continue
        run_dir_path = Path(run_dir)
        if not run_dir_path.exists():
            run_dir_path.mkdir(parents=True, exist_ok=True)
        report = load_run_report(str(run_dir_path))
        md = format_report_markdown(report)
        report_path = run_dir_path / "report.md"
        report_path.write_text(md, encoding="utf-8")
        reports.append(str(report_path))

    best = None
    if leaderboard is not None:
        best = leaderboard.get("best")

    candidate_result = None
    if config.promote_best and best is not None:
        candidate_result = promote_best_candidate(
            best_row=best,
            suite_dir=str(suite_dir),
            candidate_dir=config.candidate_dir,
        )

    ok_runs = sum(1 for s in run_statuses if s["status"] == "ok")
    failed_runs = sum(1 for s in run_statuses if s["status"] == "error")
    runs_with_ckpt = sum(
        1 for s in run_statuses if s["status"] == "ok" and s.get("checkpoint_path")
    )
    runs_with_onnx = sum(
        1 for s in run_statuses if s["status"] == "ok" and s.get("onnx_path")
    )

    health = {
        "total_runs": len(run_statuses),
        "ok_runs": ok_runs,
        "failed_runs": failed_runs,
        "runs_with_checkpoint": runs_with_ckpt,
        "runs_with_onnx": runs_with_onnx,
        "skipped_updates_total": sum(
            r.get("skipped_updates", 0) for r in index.get("rows", [])
        ),
    }

    # Compact suite_summary.json
    compact = {
        "suite_dir": str(suite_dir),
        "config": asdict(config),
        "run_count": len(run_statuses),
        "run_statuses": run_statuses,
        "health": health,
        "runs": [
            {
                "run_dir": s.get("run_dir"),
                "checkpoint_path": s.get("checkpoint_path"),
                "onnx_path": s.get("onnx_path"),
                "steps": s.get("steps"),
            }
            for s in run_statuses
        ],
        "run_index_path": run_index_path,
        "leaderboard_path": leaderboard_path,
        "best": best,
        "candidate": candidate_result,
        "reports": reports,
    }

    summary_path = suite_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "suite_dir": str(suite_dir),
        "config": asdict(config),
        "runs": run_summaries,
        "run_statuses": run_statuses,
        "health": health,
        "run_index_path": run_index_path,
        "leaderboard_path": leaderboard_path,
        "reports": reports,
        "best": best,
        "candidate": candidate_result,
    }


def _main():
    parser = argparse.ArgumentParser(description="Run a TrainV2 experiment suite")
    parser.add_argument("--name", default="suite")
    parser.add_argument("--output-dir", default="ai/train_v2/runs")
    parser.add_argument("--presets", nargs="+", default=["smoke"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--include-preview-variants", action="store_true")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=2)
    parser.add_argument("--eval-max-steps", type=int, default=100)
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--no-leaderboard", action="store_true")
    parser.add_argument("--leaderboard-games", type=int, default=4)
    parser.add_argument("--leaderboard-max-steps", type=int, default=100)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-promote-best", action="store_true")
    parser.add_argument("--candidate-dir", type=str, default=None)
    args = parser.parse_args()

    suite_cfg = SuiteConfig(
        name=args.name,
        output_dir=args.output_dir,
        seeds=args.seeds,
        presets=args.presets,
        include_preview_variants=args.include_preview_variants,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        max_steps=args.max_steps,
        eval_games=args.eval_games,
        eval_max_steps=args.eval_max_steps,
        export_onnx=not args.no_export_onnx,
        build_leaderboard=not args.no_leaderboard,
        leaderboard_games=args.leaderboard_games,
        leaderboard_max_steps=args.leaderboard_max_steps,
        continue_on_error=not args.fail_fast,
        promote_best=not args.no_promote_best,
        candidate_dir=args.candidate_dir,
    )

    result = run_suite(suite_cfg)

    health = result["health"]
    print(f"Suite: {result['config']['name']}")
    print(f"Suite dir: {result['suite_dir']}")
    print(f"Runs: {health['ok_runs']} ok / {health['failed_runs']} failed")
    if result["best"]:
        best = result["best"]
        print(f"Best: {best['model_name']} (score={best['score']:.3f})")
    print(f"Index: {result['run_index_path']}")
    if result["leaderboard_path"]:
        print(f"Leaderboard: {result['leaderboard_path']}")
    if result.get("candidate"):
        print(f"Candidate: {result['candidate']['candidate_dir']}")


if __name__ == "__main__":
    _main()
