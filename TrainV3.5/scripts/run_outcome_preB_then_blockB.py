#!/usr/bin/env python3
"""Collect terminal-outcome recovery data and gate it before long Block B."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.13/bin/python3")
OUTCOME_RUNNER = ROOT / "TrainV3.5" / "scripts" / "run_preB_outcome_recovery.py"
PAIRWISE_RUNNER = ROOT / "TrainV3.5" / "scripts" / "run_preB_counterfactual_recovery.py"
for path in (ROOT, ROOT / "TrainV3.5" / "python", ROOT / "TrainV3.5" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_nightly_preB_then_blockB import (  # noqa: E402
    _ensure_benchmark,
    _launch_block_b,
    _run,
)


REQUIRED = {"extra-lr-v4-max", "greedy_face", "random", "OnlyVersusRandomBiggest"}


def _rows_gate(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    games_per_scenario: int,
) -> dict[str, Any]:
    missing = sorted((REQUIRED - candidate.keys()) | (REQUIRED - baseline.keys()))
    if missing:
        return {"accepted": False, "missing": missing}
    v4 = candidate["extra-lr-v4-max"]
    base_v4 = baseline["extra-lr-v4-max"]
    second_games = 2 * int(games_per_scenario)
    total_v4_games = 4 * int(games_per_scenario)
    required_second_wins = 1 if games_per_scenario <= 8 else 2
    checks = {
        "v4_second_improved": v4["second_score_rate"]
        >= base_v4["second_score_rate"] + required_second_wins / second_games,
        "v4_overall_improved": v4["score_rate"]
        >= base_v4["score_rate"] + required_second_wins / total_v4_games,
        "v4_first_preserved": v4["first_score_rate"] >= base_v4["first_score_rate"],
        "greedy_guard": candidate["greedy_face"]["score_rate"]
        >= baseline["greedy_face"]["score_rate"] - (0.125 if games_per_scenario <= 8 else 0.03125),
        "random_acceptance": candidate["random"]["score_rate"] == 1.0,
        "random_big_guard": candidate["OnlyVersusRandomBiggest"]["score_rate"]
        >= baseline["OnlyVersusRandomBiggest"]["score_rate"] - (0.03125 if games_per_scenario <= 8 else 0.0078125),
        "no_invalid_actions": all(int(candidate[name]["invalid_actions"]) == 0 for name in REQUIRED),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "candidate": {name: candidate[name] for name in sorted(REQUIRED)},
        "baseline": {name: baseline[name] for name in sorted(REQUIRED)},
        "games_per_scenario": games_per_scenario,
    }


def _benchmark_gate(
    *,
    checkpoint: Path,
    attempt_dir: Path,
    source: Path,
    root: Path,
    seed: int,
    games: int,
    log_path: Path,
) -> dict[str, Any]:
    baseline, baseline_raw = _ensure_benchmark(
        checkpoint=source,
        benchmark_dir=root / f"warm_model_benchmark_g{games}",
        seed=seed,
        log_path=log_path,
        games_per_scenario=games,
    )
    candidate, candidate_raw = _ensure_benchmark(
        checkpoint=checkpoint,
        benchmark_dir=attempt_dir / f"model_benchmark_g{games}",
        seed=seed,
        log_path=log_path,
        games_per_scenario=games,
    )
    result = _rows_gate(candidate, baseline, games_per_scenario=games)
    result["no_errors"] = (
        int(baseline_raw.get("error_count", -1)) == 0
        and int(candidate_raw.get("error_count", -1)) == 0
    )
    result["accepted"] = bool(result["accepted"] and result["no_errors"])
    (attempt_dir / f"acceptance_g{games}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _primary_checkpoint(*, source: Path, root: Path, log_path: Path) -> Path:
    attempt = root / "outcome_primary"
    summary_path = attempt / "preB_outcome_summary.json"
    if not summary_path.exists():
        _run(
            [
                str(PYTHON), str(OUTCOME_RUNNER),
                "--base-checkpoint", str(source),
                "--output-dir", str(attempt),
                "--games", "512",
                "--anchor-games", "128",
                "--max-states-per-loss", "8",
                "--max-candidate-actions", "12",
                "--ranked-candidates", "6",
                "--continuation-policy", "v4_teacher",
                "--min-pairs", "250",
                "--epochs", "80",
                "--batch-size", "256",
                "--learning-rate", "4e-4",
                "--action-pair-coef", "1.5",
                "--draw-bce-coef", "0.02",
                "--recovery-policy-kl-coef", "2.0",
                "--recovery-draw-kl-coef", "8.0",
                "--anchor-policy-kl-coef", "4.0",
                "--anchor-draw-kl-coef", "12.0",
            ],
            log_path=log_path,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return Path(summary["summary"]["checkpoint_path"])


def _strong_checkpoint(*, source: Path, root: Path, log_path: Path) -> Path:
    primary = root / "outcome_primary"
    attempt = root / "outcome_strong"
    summary_path = attempt / "preB_summary.json"
    if not summary_path.exists():
        _run(
            [
                str(PYTHON), str(PAIRWISE_RUNNER),
                "--base-checkpoint", str(source),
                "--output-dir", str(attempt),
                "--dataset-path", str(primary / "preB_outcome_dataset.npz"),
                "--epochs", "120",
                "--batch-size", "256",
                "--learning-rate", "5e-4",
                "--action-pair-coef", "2.0",
                "--draw-bce-coef", "0.02",
                "--recovery-policy-kl-coef", "1.5",
                "--recovery-draw-kl-coef", "8.0",
                "--anchor-policy-kl-coef", "3.0",
                "--anchor-draw-kl-coef", "12.0",
                "--min-pairs", "250",
                "--freeze-mana-draw-recovery",
            ],
            log_path=log_path,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return Path(summary["summary"]["checkpoint_path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-seed", type=int, default=71401001)
    args = parser.parse_args(argv)
    source = args.source_checkpoint.resolve()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "outcome_supervisor.log"
    status_path = root / "outcome_status.json"
    status: dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_checkpoint": str(source),
        "attempts": [],
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    for name, build in (("outcome_primary", _primary_checkpoint), ("outcome_strong", _strong_checkpoint)):
        record: dict[str, Any] = {"name": name, "status": "running"}
        status["attempts"].append(record)
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        try:
            checkpoint = build(source=source, root=root, log_path=log_path)
            quick = _benchmark_gate(
                checkpoint=checkpoint,
                attempt_dir=root / name,
                source=source,
                root=root,
                seed=int(args.benchmark_seed),
                games=8,
                log_path=log_path,
            )
            record["quick_gate"] = quick
            if not quick["accepted"]:
                record["status"] = "rejected_quick"
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                continue
            full = _benchmark_gate(
                checkpoint=checkpoint,
                attempt_dir=root / name,
                source=source,
                root=root,
                seed=int(args.benchmark_seed),
                games=32,
                log_path=log_path,
            )
            record["full_gate"] = full
            record["status"] = "accepted" if full["accepted"] else "rejected_full"
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            if full["accepted"]:
                status["status"] = "blockB_running"
                status["accepted_checkpoint"] = str(checkpoint)
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                _launch_block_b(
                    checkpoint=checkpoint,
                    output_dir=root / "blockB_25k",
                    log_path=log_path,
                )
                status["status"] = "complete"
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                return 0
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    status["status"] = "no_candidate_accepted"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
