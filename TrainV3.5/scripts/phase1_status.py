#!/usr/bin/env python3
"""Print a compact status line for a TrainV3 Phase 1 foundation run."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATEST = ROOT / "TrainV3" / "runs" / "latest_phase1_foundation_run.txt"


def main(argv: list[str]) -> int:
    run_dir = _resolve_run_dir(argv[1] if len(argv) > 1 else None)
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "phase1_config.json"
    summary_path = run_dir / "run_summary.json"
    config = _read_json(config_path) if config_path.exists() else {}
    updates = _read_updates(metrics_path)
    target = int(config.get("updates", 0) or 0)
    if not updates:
        print(f"run={run_dir} updates=0/{target} metrics={metrics_path}")
        return 0

    last = updates[-1]
    done = int(last["update"])
    transitions = int(last["total_env_transitions"])
    collect_tps = int(last["env_transitions"]) / float(last["collect_seconds"])
    speed_summary = _read_summary_row(metrics_path)
    checkpoint = _latest_checkpoint(run_dir)
    payload = {
        "run": str(run_dir),
        "update": done,
        "target_updates": target,
        "progress": None if target <= 0 else round(done / target, 4),
        "total_transitions": transitions,
        "collect_tps": round(collect_tps, 1),
        "loss": round(float(last["loss"]), 6),
        "approx_kl": round(float(last["approx_kl"]), 6),
        "entropy": round(float(last["entropy"]), 6),
        "dense_bytes": int(last.get("stored_dense_feature_bytes", 0)),
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "finished": summary_path.exists(),
    }
    if speed_summary:
        payload["e2e_tps"] = round(float(speed_summary.get("env_transitions_per_second", 0.0)), 1)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _resolve_run_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    if DEFAULT_LATEST.exists():
        return Path(DEFAULT_LATEST.read_text(encoding="utf-8").strip()).resolve()
    candidates = sorted((ROOT / "TrainV3" / "runs").glob("phase1_foundation_noassist_*"))
    if not candidates:
        raise SystemExit("no Phase 1 run found; pass run_dir explicitly")
    return candidates[-1].resolve()


def _read_updates(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        return []
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("update"):
            rows.append(row)
    return rows


def _read_summary_row(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    summary = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "summary":
            summary = row
    return summary


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.npz"))
    return checkpoints[-1] if checkpoints else None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
