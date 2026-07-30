#!/usr/bin/env python3
"""Screen every Block-D snapshot, select conservatively, then run full benches."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "TrainV3.5/runs/run_model_benchmark_v5_current.py"
ANALYZER = ROOT / "TrainV3.5/scripts/analyze_v5_aux_ablation.py"
GENERATOR = ROOT / "TrainV3.5/scripts/generate_aux_synthetic_v1.py"
DEFAULT_PHASE_C = (
    ROOT
    / "TrainV3.5/runs/phase_c_main_u29250_h299_luna10_paddingfix_20260727"
    / "checkpoints/extra_lr_v5_phaseC_candidate_h299.npz"
)
DEFAULT_POST_B = (
    ROOT
    / "TrainV3.5/runs/blockB_from_phaseA_p2accepted100_parallel_20260714_210400"
    / "checkpoints/extra_lr_v5_blockB_league_update_29250.npz"
)
DEFAULT_AUX = (
    ROOT
    / "TrainV3.5/runs/phase_c_aux_v1_u29250_h299_projectionfix_20260727/models"
)
BASELINES = {"random", "greedy_face", "end_turn"}
PHASE_C_OPPONENT = "extra-lr-v5-phaseC-anchor-h299"
POST_B_OPPONENT = "extra-lr-v5-postB-preV5-u29250"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:100]


def _run(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _score(row: dict[str, Any], candidate: str) -> float:
    if row.get("winner_name") == candidate:
        return 1.0
    if row.get("draw"):
        return 0.5
    return 0.0


def _screen_summary(raw_path: Path) -> dict[str, Any]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = payload["results"]
    candidates = {str(row["focal_model"]) for row in rows}
    if len(candidates) != 1:
        raise ValueError(f"ambiguous candidate names: {sorted(candidates)}")
    candidate = candidates.pop()
    if (
        int(payload.get("error_count", 0)) != 0
        or any(row["status"] not in {"p1_win", "p2_win", "draw"} for row in rows)
        or any(bool(row.get("timed_out")) for row in rows)
        or any(bool(row.get("truncated")) for row in rows)
    ):
        valid = False
    else:
        valid = not any(
            int(row["invalid_actions"].get(candidate, 0)) for row in rows
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["opponent_model"]), []).append(row)

    def rate(names: set[str]) -> float:
        selected = [
            row
            for name, opponent_rows in grouped.items()
            if name in names
            for row in opponent_rows
        ]
        return (
            sum(_score(row, candidate) for row in selected) / len(selected)
            if selected
            else 0.0
        )

    learned_classic = {
        name
        for name in grouped
        if name not in BASELINES
        and name not in {PHASE_C_OPPONENT, POST_B_OPPONENT}
    }
    per_opponent = {
        name: rate({name}) for name in sorted(grouped)
    }
    phase_c = per_opponent.get(PHASE_C_OPPONENT, 0.0)
    post_b = per_opponent.get(POST_B_OPPONENT, 0.0)
    classic = rate(learned_classic)
    baseline = rate(BASELINES & set(grouped))
    return {
        "candidate": candidate,
        "valid": valid,
        "battles": len(rows),
        "phase_c_score_rate": phase_c,
        "post_b_score_rate": post_b,
        "learned_classic_score_rate": classic,
        "baseline_score_rate": baseline,
        "selection_score": (
            0.45 * phase_c + 0.25 * post_b + 0.25 * classic + 0.05 * baseline
        ),
        "per_opponent": per_opponent,
    }


def _candidate_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in manifest.get("snapshots", []):
        rows.append(
            {
                "label": f"postD-u{int(snapshot['update']):05d}",
                "path": Path(snapshot["path"]).resolve(),
                "post_d": True,
            }
        )
    rows.extend(
        [
            {
                "label": "postD-final",
                "path": Path(manifest["final_checkpoint"]).resolve(),
                "post_d": True,
            },
            {
                "label": "phaseC-h299",
                "path": Path(manifest["source_checkpoint"]).resolve(),
                "post_d": False,
            },
        ]
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        digest = _sha256(row["path"])
        if digest in seen:
            continue
        seen.add(digest)
        unique.append({**row, "sha256": digest})
    return unique


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "blockD_ultra_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise RuntimeError("Block-D manifest is not terminal ok")
    output = run_dir / "post_training_benchmarks"
    screen_root = output / "screen"
    output.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(manifest)
    summaries = []
    for row in candidates:
        candidate_dir = screen_root / _safe(row["label"])
        raw_path = candidate_dir / "none/raw.json"
        if not raw_path.exists():
            _run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--checkpoint",
                    str(row["path"]),
                    "--candidate-name",
                    f"extra-lr-v5-{row['label']}",
                    "--base-checkpoint",
                    str(args.phase_c_checkpoint.resolve()),
                    "--base-name",
                    PHASE_C_OPPONENT,
                    "--extra-v5-opponent",
                    f"{POST_B_OPPONENT}={args.post_b_checkpoint.resolve()}",
                    "--aux-dir",
                    str(args.aux_dir.resolve()),
                    "--output-dir",
                    str(candidate_dir),
                    "--mode",
                    "none",
                    "--seeds-per-opponent",
                    str(int(args.screen_seeds)),
                    "--seed",
                    str(int(args.seed)),
                    "--max-steps",
                    "1000",
                    "--max-turns",
                    "240",
                ],
                log_path=output / "post_bench.log",
            )
        summaries.append(
            {
                **row,
                "path": str(row["path"]),
                "screen": _screen_summary(raw_path),
            }
        )

    source = next(row for row in summaries if row["label"] == "phaseC-h299")
    source_screen = source["screen"]
    eligible = []
    for row in summaries:
        screen = row["screen"]
        regressions = {
            "phase_c": screen["phase_c_score_rate"] < 0.50,
            "post_b": (
                screen["post_b_score_rate"]
                < source_screen["post_b_score_rate"] - 0.03
            ),
            "learned_classic": (
                screen["learned_classic_score_rate"]
                < source_screen["learned_classic_score_rate"] - 0.03
            ),
        }
        row["regression_gates"] = regressions
        row["selection_eligible"] = bool(screen["valid"]) and not any(
            regressions.values()
        )
        if row["selection_eligible"]:
            eligible.append(row)
    winner = max(
        eligible or [source],
        key=lambda row: (
            float(row["screen"]["selection_score"]),
            bool(row["post_d"]),
        ),
    )
    selection = {
        "schema": "extra_lr_v5_ultra_snapshot_screen_v1",
        "screen_seeds_per_opponent": int(args.screen_seeds),
        "selection_rule": {
            "phase_c_floor": 0.50,
            "post_b_max_regression": 0.03,
            "learned_classic_max_regression": 0.03,
            "score_weights": {
                "phase_c": 0.45,
                "post_b": 0.25,
                "learned_classic": 0.25,
                "baseline": 0.05,
            },
        },
        "candidates": summaries,
        "winner": winner,
        "post_d_selected": bool(winner["post_d"]),
    }
    (output / "screening_summary.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    full_dir = output / "winner_full_4way"
    full_analysis = full_dir / "ablation_analysis.json"
    if not full_analysis.exists():
        _run(
            [
                sys.executable,
                str(RUNNER),
                "--checkpoint",
                str(winner["path"]),
                "--candidate-name",
                f"extra-lr-v5-ultra-selected-{_safe(winner['label'])}",
                "--base-checkpoint",
                str(args.phase_c_checkpoint.resolve()),
                "--base-name",
                PHASE_C_OPPONENT,
                "--extra-v5-opponent",
                f"{POST_B_OPPONENT}={args.post_b_checkpoint.resolve()}",
                "--aux-dir",
                str(args.aux_dir.resolve()),
                "--output-dir",
                str(full_dir),
                "--mode",
                "all",
                "--seeds-per-opponent",
                str(int(args.full_seeds)),
                "--seed",
                str(int(args.seed) + 1_000_000),
                "--max-steps",
                "1000",
                "--max-turns",
                "240",
            ],
            log_path=output / "post_bench.log",
        )
        _run(
            [sys.executable, str(ANALYZER), str(full_dir)],
            log_path=output / "post_bench.log",
        )

    aux_output = run_dir / "aux_authoritative_postD"
    aux_manifest = aux_output / "dataset_manifest.json"
    if bool(args.generate_aux) and not aux_manifest.exists():
        _run(
            [
                sys.executable,
                str(GENERATOR),
                "--checkpoint",
                str(winner["path"]),
                "--output",
                str(aux_output),
                "--total-battles",
                str(int(args.aux_battles)),
                "--seed",
                str(int(args.seed) + 2_000_000),
                "--branch-repeats",
                str(int(args.branch_repeats)),
                "--progress-every",
                "25",
            ],
            log_path=output / "post_bench.log",
        )
        _run(
            [
                sys.executable,
                str(GENERATOR),
                "--checkpoint",
                str(winner["path"]),
                "--output",
                str(aux_output),
                "--total-battles",
                str(int(args.aux_battles)),
                "--seed",
                str(int(args.seed) + 2_000_000),
                "--branch-repeats",
                str(int(args.branch_repeats)),
                "--merge",
            ],
            log_path=output / "post_bench.log",
        )
    result = {
        "schema": "extra_lr_v5_ultra_post_bench_result_v1",
        "status": "ok",
        "winner": winner,
        "full_ablation_dir": str(full_dir),
        "aux_authoritative_dir": (
            str(aux_output) if bool(args.generate_aux) else None
        ),
    }
    (output / "post_bench_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase-c-checkpoint", type=Path, default=DEFAULT_PHASE_C)
    parser.add_argument("--post-b-checkpoint", type=Path, default=DEFAULT_POST_B)
    parser.add_argument("--aux-dir", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--screen-seeds", type=int, default=16)
    parser.add_argument("--full-seeds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=53724001)
    parser.add_argument("--generate-aux", action="store_true")
    parser.add_argument("--aux-battles", type=int, default=10_000)
    parser.add_argument("--branch-repeats", type=int, default=4)
    args = parser.parse_args(argv)
    for path in (
        args.phase_c_checkpoint,
        args.post_b_checkpoint,
        args.aux_dir,
    ):
        if not path.exists():
            parser.error(f"required path not found: {path}")
    for name in ("screen_seeds", "full_seeds", "aux_battles", "branch_repeats"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print("ULTRA_POST_BENCH_RESULT", json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
