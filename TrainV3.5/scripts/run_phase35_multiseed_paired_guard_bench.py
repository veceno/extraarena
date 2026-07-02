#!/usr/bin/env python3
"""Paired multi-seed guard bench for V5 checkpoints.

This runner compares a baseline checkpoint and a candidate checkpoint on the
same H2H seeds. Phase1 scenario bench is deterministic for a checkpoint, so it
is computed once per checkpoint and reused across all paired H2H seed scores.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAINV3_PYTHON = ROOT / "TrainV3.5" / "python"
TRAINV3_SCRIPTS = ROOT / "TrainV3.5" / "scripts"
for path in (ROOT, TRAINV3_PYTHON, TRAINV3_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_phase33_guarded_noassist_v4_phase2 import (  # noqa: E402
    DEFAULT_TRACE_MANIFEST,
    GuardConfig,
    _jsonable,
    decide_candidate,
    run_guarded_evaluation,
    score_guarded_evaluation,
)


DEFAULT_BASE_CHECKPOINT = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase29_v4_league_validation_20260611_192030"
    / "extra_lr_v5_phase29_v4_league_11047_states.npz"
)
DEFAULT_CANDIDATE_CHECKPOINT = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase34c_pairwise_paired_accept_20260611_202032"
    / "best_checkpoint.npz"
)
DEFAULT_SEEDS = (44300, 45300, 46300, 47300, 48300)


@dataclass(frozen=True)
class Phase35Config:
    base_checkpoint: Path
    candidate_checkpoint: Path
    output_dir: Path
    seeds: tuple[int, ...]
    h2h_games: int
    min_mean_composite_delta: float
    max_lane_regression: float
    max_v4_regression: float
    require_legal_random: float
    require_stall: float
    require_face_rush: float


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Phase35Config(
        base_checkpoint=args.base_checkpoint.resolve(),
        candidate_checkpoint=args.candidate_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        seeds=_parse_seeds(args.seeds),
        h2h_games=int(args.h2h_games),
        min_mean_composite_delta=float(args.min_mean_composite_delta),
        max_lane_regression=float(args.max_lane_regression),
        max_v4_regression=float(args.max_v4_regression),
        require_legal_random=float(args.require_legal_random),
        require_stall=float(args.require_stall),
        require_face_rush=float(args.require_face_rush),
    )
    result = run_phase35(config)
    print("PHASE35_MULTI_SEED_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    return 0 if result["summary"]["status"] == "ok" else 1


def run_phase35(config: Phase35Config) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "phase35_config.json").write_text(
        json.dumps(_jsonable(asdict(config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.output_dir.parent / "latest_phase35_multiseed_bench_run.txt").write_text(
        str(config.output_dir) + "\n",
        encoding="utf-8",
    )

    first_seed = int(config.seeds[0])
    print("PHASE35_RUN_DIR", config.output_dir, flush=True)
    base_first = run_guarded_evaluation(
        checkpoint=config.base_checkpoint,
        output_dir=config.output_dir / f"seed_{first_seed}" / "base",
        h2h_games=config.h2h_games,
        h2h_seed=first_seed,
        phase1_enabled=True,
        h2h_enabled=True,
    )
    candidate_first = run_guarded_evaluation(
        checkpoint=config.candidate_checkpoint,
        output_dir=config.output_dir / f"seed_{first_seed}" / "candidate",
        h2h_games=config.h2h_games,
        h2h_seed=first_seed,
        phase1_enabled=True,
        h2h_enabled=True,
    )
    base_phase1 = base_first.get("phase1")
    candidate_phase1 = candidate_first.get("phase1")

    guard_config = _guard_config(config)
    pairs: list[dict[str, Any]] = []
    for idx, seed in enumerate(config.seeds):
        seed = int(seed)
        if idx == 0:
            base_eval = base_first
            candidate_eval = candidate_first
        else:
            base_eval = run_guarded_evaluation(
                checkpoint=config.base_checkpoint,
                output_dir=config.output_dir / f"seed_{seed}" / "base",
                h2h_games=config.h2h_games,
                h2h_seed=seed,
                phase1_enabled=False,
                h2h_enabled=True,
            )
            candidate_eval = run_guarded_evaluation(
                checkpoint=config.candidate_checkpoint,
                output_dir=config.output_dir / f"seed_{seed}" / "candidate",
                h2h_games=config.h2h_games,
                h2h_seed=seed,
                phase1_enabled=False,
                h2h_enabled=True,
            )
            base_eval = {**base_eval, "phase1": base_phase1}
            candidate_eval = {**candidate_eval, "phase1": candidate_phase1}
        base_score = score_guarded_evaluation(base_eval)
        candidate_score = score_guarded_evaluation(candidate_eval)
        decision = decide_candidate(best=base_score, candidate=candidate_score, config=guard_config)
        pair = {
            "seed": seed,
            "base_score": base_score,
            "candidate_score": candidate_score,
            "delta": {
                "composite": float(candidate_score["composite"] - base_score["composite"]),
                "v4_league_score": float(candidate_score["v4_league_score"] - base_score["v4_league_score"]),
                "phase1_score": float(candidate_score["phase1_score"] - base_score["phase1_score"]),
                "phase1_critical_min": float(
                    candidate_score["phase1_critical_min"] - base_score["phase1_critical_min"]
                ),
            },
            "decision": decision,
        }
        pairs.append(pair)
        print(
            "PHASE35_SEED_RESULT",
            json.dumps(
                {
                    "seed": seed,
                    "composite_delta": pair["delta"]["composite"],
                    "v4_delta": pair["delta"]["v4_league_score"],
                    "phase1_delta": pair["delta"]["phase1_score"],
                    "accepted": decision["accepted"],
                    "reasons": decision["reasons"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _write_partial(config, pairs)

    aggregate = aggregate_pairs(pairs)
    pass_reasons: list[str] = []
    if float(aggregate["mean_composite_delta"]) < float(config.min_mean_composite_delta):
        pass_reasons.append("mean_composite_delta_too_small")
    if int(aggregate["positive_composite_seed_count"]) < math.ceil(len(config.seeds) / 2):
        pass_reasons.append("not_enough_positive_composite_seeds")
    if int(aggregate["guard_accepted_seed_count"]) < math.ceil(len(config.seeds) / 2):
        pass_reasons.append("not_enough_guard_accepted_seeds")
    summary = {
        "status": "ok" if not pass_reasons else "needs_review",
        "run_dir": str(config.output_dir),
        "base_checkpoint": str(config.base_checkpoint),
        "candidate_checkpoint": str(config.candidate_checkpoint),
        "seeds": list(config.seeds),
        "h2h_games": int(config.h2h_games),
        "aggregate": aggregate,
        "reasons": pass_reasons,
    }
    result = {
        "summary": summary,
        "config": _jsonable(asdict(config)),
        "pairs": pairs,
    }
    (config.output_dir / "phase35_multiseed_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def aggregate_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs must not be empty")
    composite = [float(pair["delta"]["composite"]) for pair in pairs]
    v4 = [float(pair["delta"]["v4_league_score"]) for pair in pairs]
    phase1 = [float(pair["delta"]["phase1_score"]) for pair in pairs]
    return {
        "seed_count": int(len(pairs)),
        "mean_composite_delta": _mean(composite),
        "min_composite_delta": min(composite),
        "max_composite_delta": max(composite),
        "mean_v4_league_delta": _mean(v4),
        "min_v4_league_delta": min(v4),
        "max_v4_league_delta": max(v4),
        "mean_phase1_delta": _mean(phase1),
        "positive_composite_seed_count": sum(1 for value in composite if value > 0.0),
        "guard_accepted_seed_count": sum(1 for pair in pairs if bool(pair["decision"].get("accepted", False))),
        "guard_reasons": _reason_counts(pairs),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _reason_counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        for reason in pair["decision"].get("reasons", []):
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def _guard_config(config: Phase35Config) -> GuardConfig:
    return GuardConfig(
        source_checkpoint=config.base_checkpoint,
        output_dir=config.output_dir,
        trace_manifest=DEFAULT_TRACE_MANIFEST,
        core_library=ROOT / "TrainV3.5" / "target" / "release" / "libtrainv3_core.dylib",
        chunks=1,
        updates_per_chunk=1,
        env_count=1,
        steps_per_update=1,
        minibatch_size=1,
        learning_rate=1.0e-6,
        entropy_coef=0.0,
        clip_epsilon=0.05,
        max_grad_norm=0.25,
        opponent_mix="phase35_multiseed_bench",
        policy_padding_mode="bucketed",
        ppo_minibatch_plan="contiguous",
        seed=int(config.seeds[0]),
        h2h_games=config.h2h_games,
        h2h_seed=int(config.seeds[0]),
        phase1_enabled=True,
        h2h_enabled=True,
        min_composite_delta=config.min_mean_composite_delta,
        max_lane_regression=config.max_lane_regression,
        max_v4_regression=config.max_v4_regression,
        require_legal_random=config.require_legal_random,
        require_stall=config.require_stall,
        require_face_rush=config.require_face_rush,
    )


def _write_partial(config: Phase35Config, pairs: list[dict[str, Any]]) -> None:
    partial = {
        "schema": "extra_lr_v5_phase35_multiseed_partial_v1",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_checkpoint": str(config.base_checkpoint),
        "candidate_checkpoint": str(config.candidate_checkpoint),
        "pairs": pairs,
    }
    (config.output_dir / "phase35_multiseed_partial.json").write_text(
        json.dumps(_jsonable(partial), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("seeds must not be empty")
    return seeds


def _validate_config(config: Phase35Config) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {config.base_checkpoint}")
    if not config.candidate_checkpoint.exists():
        raise FileNotFoundError(f"candidate checkpoint not found: {config.candidate_checkpoint}")
    if int(config.h2h_games) <= 0:
        raise ValueError("h2h_games must be positive")
    for name in ("min_mean_composite_delta", "max_lane_regression", "max_v4_regression"):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--candidate-checkpoint", type=Path, default=DEFAULT_CANDIDATE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3.5" / "runs" / f"phase35_multiseed_bench_{stamp}")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--h2h-games", type=int, default=16)
    parser.add_argument("--min-mean-composite-delta", type=float, default=0.003)
    parser.add_argument("--max-lane-regression", type=float, default=0.06)
    parser.add_argument("--max-v4-regression", type=float, default=0.05)
    parser.add_argument("--require-legal-random", type=float, default=0.90)
    parser.add_argument("--require-stall", type=float, default=0.90)
    parser.add_argument("--require-face-rush", type=float, default=0.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
