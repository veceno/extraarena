#!/usr/bin/env python3
"""Prepare the broad real-opponent environment manifest for V5 Phase 9."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3" / "python"))

from train_v3.opponents_v5 import prepare_phase9_broad_opponent_environment


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Prepare TrainV3 V5 Phase 9 broad opponent lanes.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3" / "runs")
    parser.add_argument("--run-name", default="phase9_broad_opponent_environment")
    parser.add_argument("--no-probes", action="store_true")
    args = parser.parse_args(argv)

    run_id = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir.resolve() / run_id
    manifest_path = run_dir / "phase9_opponent_environment.json"
    manifest = prepare_phase9_broad_opponent_environment(
        output_path=manifest_path,
        run_probes=not args.no_probes,
        root=ROOT,
    )
    latest_path = args.output_dir.resolve() / "latest_phase9_opponent_environment.txt"
    latest_path.write_text(str(run_dir) + "\n", encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "lane_count": len(manifest["lanes"]),
        "probe_counts": _probe_counts(manifest["lanes"]),
        "opponent_mix": manifest["opponent_mix"],
        "v4_1_included": manifest["v4_1_included"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _probe_counts(lanes: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(lanes, list):
        return counts
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        status = str(lane.get("probe_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


if __name__ == "__main__":
    main()
