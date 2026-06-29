#!/usr/bin/env python3
"""Benchmark whether the V5 AdaptiveStrength dial changes runtime strength."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAINV3_PYTHON = ROOT / "TrainV3" / "python"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, TRAINV3_PYTHON, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_v5_vs_v4max_benchmark import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    DEFAULT_V4_MAX,
    DEFAULT_V5_CHECKPOINT,
    BenchmarkConfig,
    run_benchmark,
)
from train_v3.league_v5 import compare_adaptive_strength_monotonicity  # noqa: E402


PHASE10_V5_CHECKPOINT = (
    ROOT
    / "TrainV3"
    / "runs"
    / "phase10_v4max_distill_round2_from_15020_20260609_1324"
    / "extra_lr_v5_phase10_v4max_distill_61571_states.npz"
)


def _default_v5_checkpoint() -> Path:
    return PHASE10_V5_CHECKPOINT if PHASE10_V5_CHECKPOINT.exists() else DEFAULT_V5_CHECKPOINT


def _parse_strengths(raw: str) -> tuple[float, ...]:
    strengths: list[float] = []
    for part in str(raw).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if not 0.0 <= value <= 1.0:
            raise ValueError("strength values must be in [0, 1]")
        strengths.append(value)
    if not strengths:
        raise ValueError("at least one strength is required")
    return tuple(strengths)


def _safe_strength_name(value: float) -> str:
    return f"s{float(value):.3f}".replace(".", "_")


def _preset_kwargs(preset: str) -> dict[str, Any]:
    if preset == "none":
        return {
            "second_start_search": False,
        }
    if preset == "base-search":
        return {
            "second_start_search": True,
            "search_candidates": 8,
            "search_depth_plies": 24,
        }
    if preset in {"guard-roll5", "guard-roll5-adaptive"}:
        data = {
            "second_start_search": True,
            "search_candidates": 8,
            "search_depth_plies": 24,
            "search_hp_weight": 11.0,
            "search_board_power_weight": 1.5,
            "search_attack_weight": 2.5,
            "search_board_count_weight": 1.5,
            "search_empty_board_penalty": 12.0,
            "search_board_disadvantage_penalty": 30.0,
            "search_board_disadvantage_ratio": 0.8,
            "search_rollout_empty_board_penalty": 5.0,
            "search_rollout_board_disadvantage_penalty": 10.0,
            "search_rollout_board_disadvantage_ratio": 0.8,
        }
        if preset == "guard-roll5-adaptive":
            data["adaptive_strength_runtime"] = True
        return data
    raise ValueError(f"unknown search preset: {preset}")


def _row_from_result(strength: float, result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    mode_strength = float(result["modes"]["info_mode"]["adaptive_strength"])
    return {
        "adaptive_strength": float(strength),
        "encoded_adaptive_strength": mode_strength,
        "games": int(summary["games"]),
        "overall_pct": round(float(summary["v5_winrate"]) * 100.0, 2),
        "score_pct": round(float(summary["v5_score_rate"]) * 100.0, 2),
        "first_pct": round(float(summary["v5_first_winrate"]) * 100.0, 2),
        "second_pct": round(float(summary["v5_second_winrate"]) * 100.0, 2),
        "p1_pct": round(float(summary["v5_p1_winrate"]) * 100.0, 2),
        "p2_pct": round(float(summary["v5_p2_winrate"]) * 100.0, 2),
        "avg_hp_margin": round(float(summary["avg_v5_hp_margin"]), 2),
        "invalid_actions": int(summary["invalid_actions"]),
        "draw_assist_uses": int(summary["draw_assist_uses"]),
        "search_rerank_uses": int(summary.get("search_rerank_uses", 0)),
    }


def _monotonicity(rows: list[dict[str, Any]], *, tolerance_pct: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["adaptive_strength"]))
    metrics = ("score_pct", "first_pct", "second_pct", "avg_hp_margin")
    checks: list[dict[str, Any]] = []
    for metric in metrics:
        for lower, higher in zip(ordered, ordered[1:]):
            margin = float(higher[metric]) - float(lower[metric])
            checks.append(
                {
                    "metric": metric,
                    "lower_strength": float(lower["adaptive_strength"]),
                    "higher_strength": float(higher["adaptive_strength"]),
                    "lower_value": float(lower[metric]),
                    "higher_value": float(higher[metric]),
                    "margin": round(margin, 4),
                    "passes": bool(margin >= -float(tolerance_pct)),
                }
            )
    return {
        "tolerance_pct": float(tolerance_pct),
        "passes": all(bool(check["passes"]) for check in checks),
        "checks": checks,
    }


def _effectiveness(
    rows: list[dict[str, Any]],
    *,
    min_score_range_pct: float,
    min_second_range_pct: float,
    min_hp_range: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "passes": False,
            "reason": "no rows",
        }
    score_values = [float(row["score_pct"]) for row in rows]
    second_values = [float(row["second_pct"]) for row in rows]
    hp_values = [float(row["avg_hp_margin"]) for row in rows]
    score_range = max(score_values) - min(score_values)
    second_range = max(second_values) - min(second_values)
    hp_range = max(hp_values) - min(hp_values)
    passes = (
        score_range >= float(min_score_range_pct)
        or second_range >= float(min_second_range_pct)
        or hp_range >= float(min_hp_range)
    )
    return {
        "passes": bool(passes),
        "score_range_pct": round(score_range, 4),
        "second_range_pct": round(second_range, 4),
        "hp_margin_range": round(hp_range, 4),
        "min_score_range_pct": float(min_score_range_pct),
        "min_second_range_pct": float(min_second_range_pct),
        "min_hp_range": float(min_hp_range),
        "reason": "range threshold met" if passes else "all measured runtime metrics are effectively flat",
    }


def run_adaptive_strength_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    strengths = _parse_strengths(args.strengths)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "TrainV3" / "runs" / f"adaptive_strength_benchmark_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = _preset_kwargs(args.search_preset)
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for strength in strengths:
        run_dir = output_dir / _safe_strength_name(strength)
        config = BenchmarkConfig(
            v4_model_path=args.v4_model,
            v5_checkpoint_path=args.v5_checkpoint,
            assembler_dataset_path=args.assembler_dataset,
            output_dir=run_dir,
            games=int(args.games),
            seed=int(args.seed),
            max_steps=int(args.max_steps),
            start_mode=args.start_mode,
            adaptive_strength=float(strength),
            draw_assist_strength=float(args.draw_assist_strength),
            assembler_strength=float(args.assembler_strength),
            desirerer_strength=float(args.desirerer_strength),
            **preset,
        )
        result = run_benchmark(config)
        row = _row_from_result(strength, result)
        rows.append(row)
        results.append(
            {
                "adaptive_strength": float(strength),
                "output_json": str(run_dir / "v5_s1_assist_vs_v4max.json"),
                "summary": row,
            }
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    proxy_pairs = []
    ordered_strengths = tuple(sorted(strengths))
    for lower, higher in zip(ordered_strengths, ordered_strengths[1:]):
        proxy_pairs.append(
            compare_adaptive_strength_monotonicity(
                lower_strength=lower,
                higher_strength=higher,
                seeds=tuple(range(int(args.proxy_seeds))),
                scenarios_per_seed=int(args.proxy_scenarios),
            )
        )

    report = {
        "schema": "extra_lr_v5_adaptive_strength_benchmark_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "v4_model": str(args.v4_model),
            "v5_checkpoint": str(args.v5_checkpoint),
            "assembler_dataset": str(args.assembler_dataset) if args.assembler_dataset is not None else None,
            "strengths": list(strengths),
            "games": int(args.games),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "start_mode": args.start_mode,
            "search_preset": args.search_preset,
            "draw_assist_strength": float(args.draw_assist_strength),
            "assembler_strength": float(args.assembler_strength),
            "desirerer_strength": float(args.desirerer_strength),
            "monotonic_tolerance_pct": float(args.monotonic_tolerance_pct),
            "min_effective_score_range_pct": float(args.min_effective_score_range_pct),
            "min_effective_second_range_pct": float(args.min_effective_second_range_pct),
            "min_effective_hp_range": float(args.min_effective_hp_range),
        },
        "rows": sorted(rows, key=lambda row: float(row["adaptive_strength"])),
        "monotonicity": _monotonicity(rows, tolerance_pct=float(args.monotonic_tolerance_pct)),
        "effectiveness": _effectiveness(
            rows,
            min_score_range_pct=float(args.min_effective_score_range_pct),
            min_second_range_pct=float(args.min_effective_second_range_pct),
            min_hp_range=float(args.min_effective_hp_range),
        ),
        "proxy_monotonicity": proxy_pairs,
        "results": results,
    }
    report_path = output_dir / "adaptive_strength_benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {report_path}")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a V5 AdaptiveStrength matrix benchmark against V4 max")
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument("--v5-checkpoint", type=Path, default=_default_v5_checkpoint())
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strengths", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261200)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--start-mode", choices=["both", "p1", "p2"], default="both")
    parser.add_argument(
        "--search-preset",
        choices=["none", "base-search", "guard-roll5", "guard-roll5-adaptive"],
        default="guard-roll5-adaptive",
    )
    parser.add_argument("--draw-assist-strength", type=float, default=1.0)
    parser.add_argument("--assembler-strength", type=float, default=1.0)
    parser.add_argument("--desirerer-strength", type=float, default=1.0)
    parser.add_argument("--monotonic-tolerance-pct", type=float, default=3.0)
    parser.add_argument("--min-effective-score-range-pct", type=float, default=5.0)
    parser.add_argument("--min-effective-second-range-pct", type=float, default=5.0)
    parser.add_argument("--min-effective-hp-range", type=float, default=1.0)
    parser.add_argument("--proxy-seeds", type=int, default=4)
    parser.add_argument("--proxy-scenarios", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    report = run_adaptive_strength_benchmark(_parse_args(argv))
    print("\nAdaptiveStrength benchmark")
    for row in report["rows"]:
        print(
            f"s={row['adaptive_strength']:.2f} "
            f"score={row['score_pct']:.1f}% "
            f"first={row['first_pct']:.1f}% "
            f"second={row['second_pct']:.1f}% "
            f"hp={row['avg_hp_margin']:.2f}"
        )
    print(f"monotonic={report['monotonicity']['passes']}")
    print(f"effective={report['effectiveness']['passes']} ({report['effectiveness']['reason']})")


if __name__ == "__main__":
    main()
