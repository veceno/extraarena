#!/usr/bin/env python3
"""Select the best Phase 1 checkpoint from diagnostic bench reports."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


LANES = (
    "legal_random",
    "face_rush",
    "anti_draw_greed",
    "board_control",
    "greedy_trade",
    "punish_empty_board",
    "anti_hand_leak_overfit",
    "stall",
)
SCENARIO_LANES = tuple(lane for lane in LANES if lane != "legal_random")
CONTROL_FAMILY = (
    "anti_draw_greed",
    "board_control",
    "greedy_trade",
    "punish_empty_board",
    "anti_hand_leak_overfit",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Bench JSON paths or glob patterns.")
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    reports = load_reports(expand_inputs(args.paths, args.run_dir))
    if not reports:
        raise SystemExit("no Phase 1 bench reports found")
    ranked = sorted(
        ((score_report(report), report) for report in reports),
        key=lambda item: item[0]["composite"],
        reverse=True,
    )
    result = {
        "schema": "extra_lr_v5_phase1_checkpoint_selection_v1",
        "reports": [item[0] for item in ranked],
        "best": ranked[0][0],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


def expand_inputs(patterns: list[str], run_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(value) for value in glob.glob(pattern)]
        paths.extend(matched or [Path(pattern)])
    for run_dir in run_dirs:
        paths.extend(Path(run_dir).glob("phase1_acceptance_runtime_bench_*.json"))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path.resolve()
    return list(unique.values())


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or path.suffix != ".json":
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if report.get("schema") != "extra_lr_v5_phase1_acceptance_runtime_bench_v1":
            continue
        report["_report_path"] = str(path)
        reports.append(report)
    return reports


def score_report(report: dict[str, Any]) -> dict[str, Any]:
    by_lane = report.get("by_lane") or {}
    overall = report.get("overall") or {}
    lane_scores = {lane: float((by_lane.get(lane) or {}).get("score_rate", 0.0)) for lane in LANES}
    lane_hp = {lane: float((by_lane.get(lane) or {}).get("avg_hp_margin", 0.0)) for lane in LANES}
    scenario_min = min((lane_scores[lane] for lane in SCENARIO_LANES), default=0.0)
    control_mean = sum(lane_scores[lane] for lane in CONTROL_FAMILY) / len(CONTROL_FAMILY)
    hard_gate_penalty = 0.0
    hard_gate_penalty += max(0.0, 0.42 - lane_scores["face_rush"]) * 1.5
    hard_gate_penalty += max(0.0, 0.95 - lane_scores["legal_random"]) * 1.0
    hard_gate_penalty += max(0.0, 0.95 - lane_scores["stall"]) * 1.0
    hard_gate_penalty += max(0.0, float(overall.get("p1_p2_score_gap", 1.0)) - 0.12) * 1.0
    composite = (
        float(overall.get("score_rate", 0.0)) * 0.25
        + scenario_min * 0.30
        + control_mean * 0.25
        + lane_scores["face_rush"] * 0.12
        + min(lane_scores["legal_random"], lane_scores["stall"]) * 0.08
        - hard_gate_penalty
    )
    return {
        "report_path": report.get("_report_path", ""),
        "checkpoint": report.get("checkpoint", ""),
        "checkpoint_update": int(report.get("checkpoint_update", -1)),
        "phase1_pass": bool((report.get("acceptance") or {}).get("phase1_pass", False)),
        "composite": float(composite),
        "overall_score_rate": float(overall.get("score_rate", 0.0)),
        "p1_score_rate": float(overall.get("p1_score_rate", 0.0)),
        "p2_score_rate": float(overall.get("p2_score_rate", 0.0)),
        "p1_p2_score_gap": float(overall.get("p1_p2_score_gap", 0.0)),
        "scenario_min_score_rate": float(scenario_min),
        "control_family_mean_score_rate": float(control_mean),
        "lane_scores": lane_scores,
        "lane_avg_hp_margin": lane_hp,
    }


if __name__ == "__main__":
    raise SystemExit(main())
