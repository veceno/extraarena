#!/usr/bin/env python3
"""Run an objective pre-B recovery gate and conditionally start long Block B.

The gate is deliberately based on the local ``/ai/model_benchmark`` utility:
mirror decks, level-4 cards, both initiative sides, and a fixed seed.  A failed
candidate is never fed to PPO.  Instead, the supervisor returns to the accepted
warm checkpoint and launches a larger counterfactual recovery attempt so an
unattended machine keeps doing useful work without poisoning Block B.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.13/bin/python3")
BENCHMARK_RUNNER = ROOT / "TrainV3.5" / "runs" / "run_model_benchmark_v5_current.py"
PREB_RUNNER = ROOT / "TrainV3.5" / "scripts" / "run_preB_counterfactual_recovery.py"
BLOCKB_RUNNER = ROOT / "TrainV3.5" / "scripts" / "run_blockB_league.py"
BENCHMARK_GAMES_PER_SCENARIO = 32

def _run(command: list[str], *, log_path: Path) -> None:
    printable = " ".join(command)
    print(f"NIGHTLY_COMMAND {printable}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nNIGHTLY_COMMAND {printable}\n")
        log.flush()
        child_env = os.environ.copy()
        python_paths = [str(ROOT), str(ROOT / "TrainV3.5" / "python")]
        if child_env.get("PYTHONPATH"):
            python_paths.append(child_env["PYTHONPATH"])
        child_env["PYTHONPATH"] = os.pathsep.join(python_paths)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def _preb_command(
    *,
    source: Path,
    output_dir: Path,
    games: int,
    anchor_games: int,
    depth: int,
    candidates: int,
    min_margin: float,
    max_pairs: int,
    epochs: int,
    learning_rate: float,
    action_coef: float,
    greedy_fraction: float,
    seed: int,
) -> list[str]:
    return [
        str(PYTHON),
        str(PREB_RUNNER),
        "--base-checkpoint", str(source),
        "--output-dir", str(output_dir),
        "--games", str(games),
        "--anchor-games", str(anchor_games),
        "--max-steps", "180",
        "--seed", str(seed),
        "--search-candidates", str(candidates),
        "--search-depth-plies", str(depth),
        "--min-score-margin", str(min_margin),
        "--max-pairs", str(max_pairs),
        "--epochs", str(epochs),
        "--batch-size", "256",
        "--learning-rate", str(learning_rate),
        "--ranking-margin", "0.5",
        "--action-pair-coef", str(action_coef),
        "--draw-bce-coef", "0.2",
        "--recovery-policy-kl-coef", "1.5",
        "--recovery-draw-kl-coef", "3.0",
        "--anchor-policy-kl-coef", "3.0",
        "--anchor-draw-kl-coef", "6.0",
        "--min-pairs", "500",
        "--greedy-face-fraction", str(greedy_fraction),
    ]


def _checkpoint_from_preb(output_dir: Path) -> Path:
    summary = json.loads((output_dir / "preB_summary.json").read_text(encoding="utf-8"))
    checkpoint = Path(summary["summary"]["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _ensure_benchmark(
    *,
    checkpoint: Path,
    benchmark_dir: Path,
    seed: int,
    log_path: Path,
    games_per_scenario: int = BENCHMARK_GAMES_PER_SCENARIO,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    h2h_path = benchmark_dir / "v5_h2h.json"
    raw_path = benchmark_dir / "raw.json"
    if not h2h_path.exists() or not raw_path.exists():
        _run(
            [
                str(PYTHON),
                str(BENCHMARK_RUNNER),
                "--checkpoint", str(checkpoint),
                "--output-dir", str(benchmark_dir),
                "--games-per-scenario", str(games_per_scenario),
                "--seed", str(seed),
            ],
            log_path=log_path,
        )
    rows = json.loads((benchmark_dir / "v5_h2h.json").read_text(encoding="utf-8"))
    even = {row["opponent"]: row for row in rows if row["scenario"] == "even"}
    raw = json.loads((benchmark_dir / "raw.json").read_text(encoding="utf-8"))
    return even, raw


def _benchmark_and_gate(
    *,
    checkpoint: Path,
    output_dir: Path,
    seed: int,
    log_path: Path,
    baseline_even: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    benchmark_dir = output_dir / "model_benchmark_g32"
    even, raw = _ensure_benchmark(
        checkpoint=checkpoint,
        benchmark_dir=benchmark_dir,
        seed=seed,
        log_path=log_path,
    )
    required = {"extra-lr-v4-max", "greedy_face", "random", "OnlyVersusRandomBiggest"}
    missing = sorted((required - even.keys()) | (required - baseline_even.keys()))
    v4 = even.get("extra-lr-v4-max", {})
    base_v4 = baseline_even.get("extra-lr-v4-max", {})
    greedy = even.get("greedy_face", {})
    base_greedy = baseline_even.get("greedy_face", {})
    random_big = even.get("OnlyVersusRandomBiggest", {})
    base_random_big = baseline_even.get("OnlyVersusRandomBiggest", {})
    checks = {
        # Two additional second-start wins out of 64: more than a one-game
        # fluctuation, while first-start is required to remain byte-equivalent.
        "v4_second_improved": bool(not missing and v4["second_score_rate"] >= base_v4["second_score_rate"] + 2.0 / 64.0),
        "v4_overall_improved": bool(not missing and v4["score_rate"] >= base_v4["score_rate"] + 2.0 / 128.0),
        "v4_first_preserved": bool(not missing and v4["first_score_rate"] >= base_v4["first_score_rate"]),
        "greedy_second_guard": bool(not missing and greedy["second_score_rate"] >= base_greedy["second_score_rate"] - 1.0 / 16.0),
        "random_acceptance": bool(not missing and even["random"]["score_rate"] == 1.0),
        "random_big_guard": bool(not missing and random_big["score_rate"] >= base_random_big["score_rate"] - 1.0 / 128.0),
        "no_invalid_actions": bool(not missing and all(int(even[name]["invalid_actions"]) == 0 for name in required)),
        "no_errors": int(raw.get("error_count", -1)) == 0,
        "required_rows_present": not missing,
    }
    result = {
        "accepted": all(checks.values()),
        "checks": checks,
        "missing": missing,
        "objective": "/ai/model_benchmark",
        "seed": seed,
        "warm_baseline": {name: baseline_even.get(name) for name in sorted(required)},
        "even": {name: even.get(name) for name in sorted(required)},
        "checkpoint": str(checkpoint),
    }
    (output_dir / "acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("NIGHTLY_GATE", json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def _launch_block_b(*, checkpoint: Path, output_dir: Path, log_path: Path) -> None:
    opponent_mix = (
        "v4-orig-argmax:0.45,v4-orig-t07:0.15,v4-orig-t12:0.10,"
        "self:0.10,greedy_face:0.10,anti_draw_greed:0.03,"
        "punish_empty_board:0.04,stall:0.03"
    )
    _run(
        [
            str(PYTHON),
            str(BLOCKB_RUNNER),
            "--run-name", "blockB_after_objective_preB_gate",
            "--source-checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--updates", "25000",
            "--env-count", "128",
            "--steps-per-update", "24",
            "--epochs", "6",
            "--minibatch-size", "2048",
            "--learning-rate", "1e-4",
            "--lr-warmup-updates", "250",
            "--lr-final-scale", "0.1",
            "--checkpoint-every", "250",
            "--second-start-weight", "0.65",
            "--opponent-mix", opponent_mix,
        ],
        log_path=log_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71420001)
    parser.add_argument("--benchmark-seed", type=int, default=71401001)
    args = parser.parse_args(argv)
    source = args.source_checkpoint.resolve()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "nightly_supervisor.log"
    status_path = root / "nightly_status.json"

    attempts = [
        {
            "name": "round2_deep",
            "games": 512,
            "anchor_games": 128,
            "depth": 30,
            "candidates": 12,
            "min_margin": 50.0,
            "max_pairs": 8192,
            "epochs": 80,
            "learning_rate": 5e-4,
            "action_coef": 1.2,
            "greedy_fraction": 0.25,
        },
        {
            "name": "round3_expanded",
            "games": 1024,
            "anchor_games": 256,
            "depth": 50,
            "candidates": 16,
            "min_margin": 100.0,
            "max_pairs": 16384,
            "epochs": 100,
            "learning_rate": 5e-4,
            "action_coef": 1.4,
            "greedy_fraction": 0.35,
        },
    ]
    status: dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_checkpoint": str(source),
        "attempts": [],
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    baseline_even, baseline_raw = _ensure_benchmark(
        checkpoint=source,
        benchmark_dir=root / "warm_baseline_model_benchmark_g32",
        seed=int(args.benchmark_seed),
        log_path=log_path,
    )
    if int(baseline_raw.get("error_count", -1)) != 0:
        raise RuntimeError("warm baseline benchmark contains errors")

    for index, spec in enumerate(attempts):
        attempt_dir = root / spec["name"]
        record: dict[str, Any] = {"name": spec["name"], "status": "running"}
        status["attempts"].append(record)
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        try:
            summary_path = attempt_dir / "preB_summary.json"
            if summary_path.exists():
                checkpoint = _checkpoint_from_preb(attempt_dir)
                print(
                    f"NIGHTLY_RESUME_PREB {spec['name']} checkpoint={checkpoint}",
                    flush=True,
                )
            else:
                _run(
                    _preb_command(
                        source=source,
                        output_dir=attempt_dir,
                        games=spec["games"],
                        anchor_games=spec["anchor_games"],
                        depth=spec["depth"],
                        candidates=spec["candidates"],
                        min_margin=spec["min_margin"],
                        max_pairs=spec["max_pairs"],
                        epochs=spec["epochs"],
                        learning_rate=spec["learning_rate"],
                        action_coef=spec["action_coef"],
                        greedy_fraction=spec["greedy_fraction"],
                        seed=int(args.seed) + index * 1_000_000,
                    ),
                    log_path=log_path,
                )
                checkpoint = _checkpoint_from_preb(attempt_dir)
            gate = _benchmark_and_gate(
                checkpoint=checkpoint,
                output_dir=attempt_dir,
                seed=int(args.benchmark_seed),
                log_path=log_path,
                baseline_even=baseline_even,
            )
            record.update({"status": "accepted" if gate["accepted"] else "rejected", "gate": gate})
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            if gate["accepted"]:
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
            print(f"NIGHTLY_ATTEMPT_FAILED {record['name']} {record['error']}", flush=True)
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    status["status"] = "no_candidate_accepted"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
