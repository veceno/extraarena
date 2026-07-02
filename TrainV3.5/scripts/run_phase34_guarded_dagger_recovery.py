#!/usr/bin/env python3
"""Guarded DAgger/distill recovery for no-assist V5 Phase 2.

Phase33 showed that online PPO chunks can improve PPO-internal signals while
hurting the external V4/scenario bench. Phase34 generates conservative offline
candidates, evaluates them with the same external guard, and only promotes a
checkpoint when the guarded composite improves without protected regressions.
"""
from __future__ import annotations

import argparse
import json
import shutil
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

from run_phase19_noassist_conservative_second_start import (  # noqa: E402
    DEFAULT_V4_MAX,
    Phase19Config,
    run_phase19,
)
from run_phase29_v4_league_foundation import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    DEFAULT_NOASSIST_DECK_POOL,
    DEFAULT_SOURCE_CHECKPOINT as PHASE29_DEFAULT_SOURCE_CHECKPOINT,
    DEFAULT_V4_LEAGUE,
    NOASSIST_BASELINE_DECK_IDS,
    Phase29Config,
    _jsonable,
    _parse_deck_ids,
    _parse_deck_pool,
    _resolve_teacher_specs,
    run_phase29,
)
from run_phase33_guarded_noassist_v4_phase2 import (  # noqa: E402
    DEFAULT_TRACE_MANIFEST,
    GuardConfig,
    decide_candidate,
    run_guarded_evaluation,
    score_guarded_evaluation,
)


DEFAULT_SOURCE_CHECKPOINT = PHASE29_DEFAULT_SOURCE_CHECKPOINT
DEFAULT_CANDIDATES = ("v4_league_second", "v4_league_both", "pairwise_second")


@dataclass(frozen=True)
class Phase34Config:
    source_checkpoint: Path
    output_dir: Path
    rounds: int
    candidates: tuple[str, ...]
    seed: int
    h2h_games: int
    h2h_seed: int
    phase1_enabled: bool
    h2h_enabled: bool
    total_games: int
    max_steps: int
    batch_size: int
    epochs: int
    learning_rate: float
    source_kl_coef: float
    max_states: int
    v4_league: str
    noassist_deck_ids: tuple[int, ...]
    noassist_deck_pool: tuple[tuple[int, ...], ...]
    phase19_games: int
    phase19_anchor_games: int
    phase19_epochs: int
    phase19_learning_rate: float
    phase19_pairwise_coef: float
    phase19_kl_coef: float
    phase19_anchor_kl_coef: float
    phase19_min_pairs: int
    phase19_search_candidates: int
    phase19_search_depth_plies: int
    min_composite_delta: float
    max_lane_regression: float
    max_v4_regression: float
    require_legal_random: float
    require_stall: float
    require_face_rush: float
    paired_guard_evaluation: bool = True
    assembler_dataset: Path | None = DEFAULT_ASSEMBLER_DATASET


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Phase34Config(
        source_checkpoint=args.source_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        rounds=int(args.rounds),
        candidates=_parse_candidates(args.candidates),
        seed=int(args.seed),
        h2h_games=int(args.h2h_games),
        h2h_seed=int(args.h2h_seed),
        phase1_enabled=not bool(args.skip_phase1),
        h2h_enabled=not bool(args.skip_h2h),
        total_games=int(args.total_games),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        source_kl_coef=float(args.source_kl_coef),
        max_states=int(args.max_states),
        v4_league=str(args.v4_league),
        noassist_deck_ids=_parse_deck_ids(args.noassist_deck_ids),
        noassist_deck_pool=_parse_deck_pool(args.noassist_deck_pool),
        phase19_games=int(args.phase19_games),
        phase19_anchor_games=int(args.phase19_anchor_games),
        phase19_epochs=int(args.phase19_epochs),
        phase19_learning_rate=float(args.phase19_learning_rate),
        phase19_pairwise_coef=float(args.phase19_pairwise_coef),
        phase19_kl_coef=float(args.phase19_kl_coef),
        phase19_anchor_kl_coef=float(args.phase19_anchor_kl_coef),
        phase19_min_pairs=int(args.phase19_min_pairs),
        phase19_search_candidates=int(args.phase19_search_candidates),
        phase19_search_depth_plies=int(args.phase19_search_depth_plies),
        min_composite_delta=float(args.min_composite_delta),
        max_lane_regression=float(args.max_lane_regression),
        max_v4_regression=float(args.max_v4_regression),
        require_legal_random=float(args.require_legal_random),
        require_stall=float(args.require_stall),
        require_face_rush=float(args.require_face_rush),
        paired_guard_evaluation=not bool(args.no_paired_guard_evaluation),
        assembler_dataset=args.assembler_dataset.resolve() if args.assembler_dataset is not None else None,
    )
    result = run_phase34(config)
    print("PHASE34_GUARDED_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    return 0 if result["summary"]["status"] in {"ok", "no_candidate_accepted"} else 1


def run_phase34(config: Phase34Config) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "phase34_config.json").write_text(
        json.dumps(_jsonable(asdict(config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.output_dir.parent / "latest_phase34_guarded_dagger_run.txt").write_text(
        str(config.output_dir) + "\n",
        encoding="utf-8",
    )
    (config.output_dir.parent / "latest_trainv3_training_run.txt").write_text(
        str(config.output_dir) + "\n",
        encoding="utf-8",
    )

    print("PHASE34_RUN_DIR", config.output_dir, flush=True)
    best_checkpoint = config.source_checkpoint
    baseline_eval = run_guarded_evaluation(
        checkpoint=best_checkpoint,
        output_dir=config.output_dir / "baseline_eval",
        h2h_games=config.h2h_games,
        h2h_seed=config.h2h_seed,
        phase1_enabled=config.phase1_enabled,
        h2h_enabled=config.h2h_enabled,
    )
    best_score = score_guarded_evaluation(baseline_eval)
    guard_config = _phase33_incremental_guard_config(config)
    accepted: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = [
        {
            "kind": "baseline",
            "checkpoint": str(best_checkpoint),
            "evaluation": baseline_eval,
            "score": best_score,
            "accepted": True,
            "reasons": ["baseline"],
        }
    ]
    _write_state(config, best_checkpoint, best_score, records, accepted)

    for round_idx in range(1, int(config.rounds) + 1):
        for candidate_kind in config.candidates:
            candidate_dir = config.output_dir / f"round_{round_idx:02d}_{candidate_kind}"
            try:
                candidate_checkpoint = generate_candidate(
                    config,
                    kind=candidate_kind,
                    source_checkpoint=best_checkpoint,
                    output_dir=candidate_dir,
                    round_idx=round_idx,
                )
                candidate_eval = run_guarded_evaluation(
                    checkpoint=candidate_checkpoint,
                    output_dir=candidate_dir / "eval",
                    h2h_games=config.h2h_games,
                    h2h_seed=config.h2h_seed + round_idx * 10_000 + len(records) * 100,
                    phase1_enabled=config.phase1_enabled,
                    h2h_enabled=config.h2h_enabled,
                )
                candidate_score = score_guarded_evaluation(candidate_eval)
                reference_eval = None
                reference_score = best_score
                if config.paired_guard_evaluation:
                    reference_eval = run_guarded_evaluation(
                        checkpoint=best_checkpoint,
                        output_dir=candidate_dir / "eval_best_reference",
                        h2h_games=config.h2h_games,
                        h2h_seed=int(candidate_eval["h2h_seed"]),
                        phase1_enabled=config.phase1_enabled,
                        h2h_enabled=config.h2h_enabled,
                    )
                    reference_score = score_guarded_evaluation(reference_eval)
                decision = decide_candidate(best=reference_score, candidate=candidate_score, config=guard_config)
                record = {
                    "kind": candidate_kind,
                    "round": round_idx,
                    "checkpoint": str(candidate_checkpoint),
                    "evaluation": candidate_eval,
                    "score": candidate_score,
                    "paired_guard_evaluation": bool(config.paired_guard_evaluation),
                    "reference_evaluation": reference_eval,
                    "reference_score": reference_score,
                    **decision,
                }
                if decision["accepted"]:
                    best_checkpoint = candidate_checkpoint
                    best_score = candidate_score
                    accepted.append(
                        {
                            "round": round_idx,
                            "kind": candidate_kind,
                            "checkpoint": str(candidate_checkpoint),
                            "delta": float(decision["delta"]),
                        }
                    )
                    shutil.copy2(best_checkpoint, config.output_dir / "best_checkpoint.npz")
            except Exception as exc:  # Keep other candidates alive; persist failure.
                record = {
                    "kind": candidate_kind,
                    "round": round_idx,
                    "accepted": False,
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)
            _write_state(config, best_checkpoint, best_score, records, accepted)
            print("PHASE34_DECISION", json.dumps(_jsonable(record), sort_keys=True), flush=True)

    if not (config.output_dir / "best_checkpoint.npz").exists():
        shutil.copy2(best_checkpoint, config.output_dir / "best_checkpoint.npz")
    summary = {
        "status": "ok" if accepted else "no_candidate_accepted",
        "run_dir": str(config.output_dir),
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_copy": str(config.output_dir / "best_checkpoint.npz"),
        "accepted_candidates": accepted,
        "candidate_count": len(records) - 1,
        "best_composite": float(best_score["composite"]),
        "best_v4_league_score": float(best_score["v4_league_score"]),
        "best_phase1_score": float(best_score["phase1_score"]),
        "guard": _jsonable(asdict(guard_config)),
    }
    result = {
        "summary": summary,
        "config": _jsonable(asdict(config)),
        "records": records,
    }
    (config.output_dir / "phase34_guarded_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def generate_candidate(
    config: Phase34Config,
    *,
    kind: str,
    source_checkpoint: Path,
    output_dir: Path,
    round_idx: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if kind == "v4_league_second":
        return _run_v4_league_candidate(config, source_checkpoint, output_dir, round_idx, focus_start_mode="v5_second")
    if kind == "v4_league_both":
        return _run_v4_league_candidate(config, source_checkpoint, output_dir, round_idx, focus_start_mode="both")
    if kind == "pairwise_second":
        return _run_pairwise_second_candidate(config, source_checkpoint, output_dir, round_idx)
    raise ValueError(f"unknown phase34 candidate kind: {kind}")


def _run_v4_league_candidate(
    config: Phase34Config,
    source_checkpoint: Path,
    output_dir: Path,
    round_idx: int,
    *,
    focus_start_mode: str,
) -> Path:
    result = run_phase29(
        Phase29Config(
            source_checkpoint=source_checkpoint,
            output_dir=output_dir,
            teacher_specs=_resolve_teacher_specs(total_games=config.total_games, league_spec=config.v4_league),
            collection_mode="v5_on_policy",
            focus_start_mode=focus_start_mode,
            max_steps=config.max_steps,
            seed=config.seed + round_idx * 100_000 + (17 if focus_start_mode == "v5_second" else 29),
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            source_kl_coef=config.source_kl_coef,
            max_states=config.max_states,
            noassist_deck_ids=config.noassist_deck_ids,
            noassist_deck_pool=config.noassist_deck_pool,
            assembler_dataset=config.assembler_dataset,
            save_dataset=True,
        )
    )
    return Path(str(result["checkpoint_path"]))


def _run_pairwise_second_candidate(
    config: Phase34Config,
    source_checkpoint: Path,
    output_dir: Path,
    round_idx: int,
) -> Path:
    result = run_phase19(
        Phase19Config(
            base_checkpoint=source_checkpoint,
            v4_model=DEFAULT_V4_MAX,
            output_dir=output_dir,
            games=config.phase19_games,
            anchor_games=config.phase19_anchor_games,
            max_steps=config.max_steps,
            seed=config.seed + round_idx * 100_000 + 1900,
            batch_size=config.batch_size,
            epochs=config.phase19_epochs,
            learning_rate=config.phase19_learning_rate,
            noassist_deck_ids=config.noassist_deck_ids,
            search_candidates=config.phase19_search_candidates,
            search_depth_plies=config.phase19_search_depth_plies,
            pairwise_coef=config.phase19_pairwise_coef,
            kl_coef=config.phase19_kl_coef,
            anchor_kl_coef=config.phase19_anchor_kl_coef,
            min_pairs=config.phase19_min_pairs,
        )
    )
    return Path(str(result["checkpoint_path"]))


def _phase33_incremental_guard_config(config: Phase34Config) -> GuardConfig:
    return GuardConfig(
        source_checkpoint=config.source_checkpoint,
        output_dir=config.output_dir,
        trace_manifest=DEFAULT_TRACE_MANIFEST,
        core_library=ROOT / "TrainV3.5" / "target" / "release" / "libtrainv3_core.dylib",
        chunks=1,
        updates_per_chunk=1,
        env_count=1,
        steps_per_update=1,
        minibatch_size=1,
        learning_rate=max(1.0e-12, config.learning_rate),
        entropy_coef=0.0,
        clip_epsilon=0.05,
        max_grad_norm=0.25,
        opponent_mix="offline_phase34",
        policy_padding_mode="bucketed",
        ppo_minibatch_plan="contiguous",
        seed=config.seed,
        h2h_games=config.h2h_games,
        h2h_seed=config.h2h_seed,
        phase1_enabled=config.phase1_enabled,
        h2h_enabled=config.h2h_enabled,
        min_composite_delta=config.min_composite_delta,
        max_lane_regression=config.max_lane_regression,
        max_v4_regression=config.max_v4_regression,
        require_legal_random=config.require_legal_random,
        require_stall=config.require_stall,
        require_face_rush=config.require_face_rush,
    )


def _write_state(
    config: Phase34Config,
    best_checkpoint: Path,
    best_score: dict[str, Any],
    records: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> None:
    state = {
        "schema": "extra_lr_v5_phase34_guarded_state_v1",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_checkpoint": str(best_checkpoint),
        "best_score": best_score,
        "accepted_candidates": accepted,
        "records": records,
        "config": _jsonable(asdict(config)),
    }
    (config.output_dir / "phase34_guarded_state.json").write_text(
        json.dumps(_jsonable(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_candidates(raw: str) -> tuple[str, ...]:
    candidates = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not candidates:
        raise ValueError("candidates must not be empty")
    allowed = set(DEFAULT_CANDIDATES)
    unknown = [candidate for candidate in candidates if candidate not in allowed]
    if unknown:
        raise ValueError(f"unknown phase34 candidates: {unknown}; allowed={sorted(allowed)}")
    return candidates


def _validate_config(config: Phase34Config) -> None:
    if not config.source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint not found: {config.source_checkpoint}")
    if "v4.1" in str(config.source_checkpoint).lower():
        raise ValueError("V4.1 checkpoints must not be used for Phase34")
    for name in ("rounds", "total_games", "max_steps", "batch_size", "epochs", "h2h_games"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("learning_rate", "source_kl_coef", "min_composite_delta", "max_lane_regression", "max_v4_regression"):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("phase19_games", "phase19_anchor_games", "phase19_epochs", "phase19_min_pairs"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("phase19_learning_rate", "phase19_pairwise_coef", "phase19_kl_coef", "phase19_anchor_kl_coef"):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config.assembler_dataset is not None and not config.assembler_dataset.exists():
        raise FileNotFoundError(f"assembler dataset not found: {config.assembler_dataset}")
    specs = _resolve_teacher_specs(total_games=config.total_games, league_spec=config.v4_league)
    if not specs:
        raise ValueError("V4 league must include at least one teacher")
    for spec in specs:
        if "v4.1" in spec.name.lower() or "v4.1" in str(spec.model_path).lower():
            raise ValueError("V4.1 teacher models must not be used for Phase34")
        if not spec.model_path.exists():
            raise FileNotFoundError(f"V4 teacher model not found: {spec.model_path}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3.5" / "runs" / f"phase34_guarded_dagger_recovery_{stamp}")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--seed", type=int, default=34001)
    parser.add_argument("--h2h-games", type=int, default=16)
    parser.add_argument("--h2h-seed", type=int, default=34000)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-h2h", action="store_true")
    parser.add_argument("--total-games", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--source-kl-coef", type=float, default=1.5)
    parser.add_argument("--max-states", type=int, default=24000)
    parser.add_argument("--v4-league", default="")
    parser.add_argument("--noassist-deck-ids", default=",".join(str(card_id) for card_id in NOASSIST_BASELINE_DECK_IDS))
    parser.add_argument(
        "--noassist-deck-pool",
        default=";".join(",".join(str(card_id) for card_id in deck) for deck in DEFAULT_NOASSIST_DECK_POOL),
    )
    parser.add_argument("--phase19-games", type=int, default=96)
    parser.add_argument("--phase19-anchor-games", type=int, default=48)
    parser.add_argument("--phase19-epochs", type=int, default=2)
    parser.add_argument("--phase19-learning-rate", type=float, default=8.0e-6)
    parser.add_argument("--phase19-pairwise-coef", type=float, default=0.55)
    parser.add_argument("--phase19-kl-coef", type=float, default=2.5)
    parser.add_argument("--phase19-anchor-kl-coef", type=float, default=3.5)
    parser.add_argument("--phase19-min-pairs", type=int, default=12)
    parser.add_argument("--phase19-search-candidates", type=int, default=8)
    parser.add_argument("--phase19-search-depth-plies", type=int, default=8)
    parser.add_argument("--min-composite-delta", type=float, default=0.003)
    parser.add_argument("--max-lane-regression", type=float, default=0.06)
    parser.add_argument("--max-v4-regression", type=float, default=0.05)
    parser.add_argument("--require-legal-random", type=float, default=0.90)
    parser.add_argument("--require-stall", type=float, default=0.90)
    parser.add_argument("--require-face-rush", type=float, default=0.0)
    parser.add_argument("--no-paired-guard-evaluation", action="store_true")
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
