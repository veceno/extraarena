#!/usr/bin/env python3
"""Guarded no-assist Phase 2 runner for Extra-LR V5.

The runner trains in short Rust-first PPO chunks, benches each candidate, and
only advances the base checkpoint when external acceptance metrics improve.
This avoids the failure mode where PPO reward/KL look healthy while H2H or
scenario strength collapses after a few updates.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable).resolve()

DEFAULT_SOURCE_CHECKPOINT = (
    ROOT
    / "TrainV3"
    / "runs"
    / "phase29_v4_league_validation_20260611_192030"
    / "extra_lr_v5_phase29_v4_league_11047_states.npz"
)
DEFAULT_TRACE_MANIFEST = (
    ROOT
    / "TrainV3"
    / "runs"
    / "phase26_noassist_easy_gate_main_20260610_151832"
    / "trace_manifest.json"
)
DEFAULT_V4_MODELS = (
    ("v4-max", ROOT / "ai" / "models" / "extra-lr-v4-max.onnx", 0.45),
    ("v4-opti", ROOT / "ai" / "models" / "extra-lr-v4-opti.onnx", 0.25),
    ("v4-lite", ROOT / "ai" / "models" / "extra-lr-v4-lite.onnx", 0.15),
    ("v4-micro", ROOT / "ai" / "models" / "extra-lr-v4-micro.onnx", 0.15),
)
DEFAULT_OPPONENT_MIX = (
    "legal_random:0.08,"
    "face_rush:0.18,"
    "anti_draw_greed:0.18,"
    "board_control:0.18,"
    "greedy_trade:0.14,"
    "punish_empty_board:0.14,"
    "anti_hand_leak_overfit:0.07,"
    "stall:0.03"
)
CRITICAL_PHASE1_LANES = (
    "face_rush",
    "anti_draw_greed",
    "board_control",
    "greedy_trade",
    "punish_empty_board",
    "anti_hand_leak_overfit",
)


@dataclass(frozen=True)
class GuardConfig:
    source_checkpoint: Path
    output_dir: Path
    trace_manifest: Path
    core_library: Path
    chunks: int
    updates_per_chunk: int
    env_count: int
    steps_per_update: int
    minibatch_size: int
    learning_rate: float
    entropy_coef: float
    clip_epsilon: float
    max_grad_norm: float
    opponent_mix: str
    policy_padding_mode: str
    ppo_minibatch_plan: str
    seed: int
    h2h_games: int
    h2h_seed: int
    phase1_enabled: bool = True
    h2h_enabled: bool = True
    min_composite_delta: float = 0.01
    max_lane_regression: float = 0.08
    max_v4_regression: float = 0.06
    require_legal_random: float = 0.90
    require_stall: float = 0.90
    require_face_rush: float = 0.40
    require_invalid_action_pass: bool = True
    promote_rejected: bool = False
    paired_guard_evaluation: bool = True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = GuardConfig(
        source_checkpoint=args.source_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        trace_manifest=args.trace_manifest.resolve(),
        core_library=args.core_library.resolve(),
        chunks=int(args.chunks),
        updates_per_chunk=int(args.updates_per_chunk),
        env_count=int(args.env_count),
        steps_per_update=int(args.steps_per_update),
        minibatch_size=int(args.minibatch_size),
        learning_rate=float(args.learning_rate),
        entropy_coef=float(args.entropy_coef),
        clip_epsilon=float(args.clip_epsilon),
        max_grad_norm=float(args.max_grad_norm),
        opponent_mix=str(args.opponent_mix),
        policy_padding_mode=str(args.policy_padding_mode),
        ppo_minibatch_plan=str(args.ppo_minibatch_plan),
        seed=int(args.seed),
        h2h_games=int(args.h2h_games),
        h2h_seed=int(args.h2h_seed),
        phase1_enabled=not bool(args.skip_phase1),
        h2h_enabled=not bool(args.skip_h2h),
        min_composite_delta=float(args.min_composite_delta),
        max_lane_regression=float(args.max_lane_regression),
        max_v4_regression=float(args.max_v4_regression),
        require_legal_random=float(args.require_legal_random),
        require_stall=float(args.require_stall),
        require_face_rush=float(args.require_face_rush),
        promote_rejected=bool(args.promote_rejected),
        paired_guard_evaluation=not bool(args.no_paired_guard_evaluation),
    )
    result = run_guarded_phase2(config)
    print("PHASE33_GUARDED_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    return 0 if result["summary"]["status"] in {"ok", "no_candidate_accepted"} else 1


def run_guarded_phase2(config: GuardConfig) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "phase33_guard_config.json").write_text(
        json.dumps(_jsonable(asdict(config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest_file = config.output_dir.parent / "latest_phase33_guarded_noassist_v4_run.txt"
    latest_general = config.output_dir.parent / "latest_trainv3_training_run.txt"
    latest_file.write_text(str(config.output_dir) + "\n", encoding="utf-8")
    latest_general.write_text(str(config.output_dir) + "\n", encoding="utf-8")

    print("PHASE33_RUN_DIR", config.output_dir, flush=True)
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
    accepted: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = [
        {
            "kind": "baseline",
            "checkpoint": str(best_checkpoint),
            "evaluation": baseline_eval,
            "score": best_score,
            "accepted": True,
            "reasons": ["baseline"],
        }
    ]
    _write_state(config, best_checkpoint, best_score, candidates, accepted)

    for chunk_idx in range(1, int(config.chunks) + 1):
        chunk_dir = config.output_dir / f"chunk_{chunk_idx:03d}"
        candidate_checkpoint = run_training_chunk(
            config,
            chunk_dir=chunk_dir,
            resume_checkpoint=best_checkpoint,
            chunk_idx=chunk_idx,
        )
        candidate_eval = run_guarded_evaluation(
            checkpoint=candidate_checkpoint,
            output_dir=chunk_dir / "eval",
            h2h_games=config.h2h_games,
            h2h_seed=config.h2h_seed + chunk_idx * 1000,
            phase1_enabled=config.phase1_enabled,
            h2h_enabled=config.h2h_enabled,
        )
        candidate_score = score_guarded_evaluation(candidate_eval)
        reference_eval = None
        reference_score = best_score
        if config.paired_guard_evaluation:
            reference_eval = run_guarded_evaluation(
                checkpoint=best_checkpoint,
                output_dir=chunk_dir / "eval_best_reference",
                h2h_games=config.h2h_games,
                h2h_seed=int(candidate_eval["h2h_seed"]),
                phase1_enabled=config.phase1_enabled,
                h2h_enabled=config.h2h_enabled,
            )
            reference_score = score_guarded_evaluation(reference_eval)
        decision = decide_candidate(
            best=reference_score,
            candidate=candidate_score,
            config=config,
        )
        if decision["accepted"]:
            best_checkpoint = candidate_checkpoint
            best_score = candidate_score
            accepted.append({"chunk": chunk_idx, "checkpoint": str(candidate_checkpoint)})
            shutil.copy2(best_checkpoint, config.output_dir / "best_checkpoint.npz")
        elif config.promote_rejected:
            best_checkpoint = candidate_checkpoint

        record = {
            "kind": "candidate",
            "chunk": chunk_idx,
            "checkpoint": str(candidate_checkpoint),
            "evaluation": candidate_eval,
            "score": candidate_score,
            "paired_guard_evaluation": bool(config.paired_guard_evaluation),
            "reference_evaluation": reference_eval,
            "reference_score": reference_score,
            **decision,
        }
        candidates.append(record)
        _write_state(config, best_checkpoint, best_score, candidates, accepted)
        print("PHASE33_DECISION", json.dumps(_jsonable(record), sort_keys=True), flush=True)

    summary = {
        "status": "ok" if accepted else "no_candidate_accepted",
        "run_dir": str(config.output_dir),
        "best_checkpoint": str(best_checkpoint),
        "accepted_chunks": accepted,
        "candidate_count": len(candidates) - 1,
        "best_composite": best_score["composite"],
        "best_v4_league_score": best_score["v4_league_score"],
        "best_phase1_score": best_score["phase1_score"],
    }
    result = {
        "summary": summary,
        "config": _jsonable(asdict(config)),
        "candidates": candidates,
    }
    (config.output_dir / "phase33_guarded_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run_training_chunk(config: GuardConfig, *, chunk_dir: Path, resume_checkpoint: Path, chunk_idx: int) -> Path:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TRAINV3_CORE_LIB": str(config.core_library),
            "PHASE26_RUN_NAME": f"phase33_guarded_chunk_{chunk_idx:03d}",
            "PHASE26_PHASE": "phase2_guarded_noassist_v4_runtime_chunk",
            "PHASE26_OUT_ROOT": str(chunk_dir),
            "PHASE26_RESUME_CHECKPOINT": str(resume_checkpoint),
            "PHASE26_RESUME_OPTIMIZER_POLICY": "reset",
            "PHASE26_TRACE_MANIFEST_PATH": str(config.trace_manifest),
            "PHASE26_ENV_COUNT": str(config.env_count),
            "PHASE26_STEPS_PER_UPDATE": str(config.steps_per_update),
            "PHASE26_UPDATES": str(config.updates_per_chunk),
            "PHASE26_MINIBATCH_SIZE": str(config.minibatch_size),
            "PHASE26_CHECKPOINT_EVERY": str(config.updates_per_chunk),
            "PHASE26_LR": str(config.learning_rate),
            "PHASE26_ENTROPY_COEF": str(config.entropy_coef),
            "PHASE26_CLIP_EPSILON": str(config.clip_epsilon),
            "PHASE26_MAX_GRAD_NORM": str(config.max_grad_norm),
            "PHASE26_POLICY_PADDING_MODE": config.policy_padding_mode,
            "PHASE26_PPO_MINIBATCH_PLAN": config.ppo_minibatch_plan,
            "PHASE26_OPPONENT_MIX": config.opponent_mix,
            "PHASE26_SEED": str(config.seed + chunk_idx * 10_000),
        }
    )
    log_path = chunk_dir / "phase26_chunk.log"
    command = [str(PYTHON), str(ROOT / "TrainV3" / "scripts" / "run_phase26_noassist_easy_gate.py")]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"training chunk {chunk_idx} failed; see {log_path}")
    summary_paths = sorted(chunk_dir.glob("phase33_guarded_chunk_*/run_summary.json"))
    if not summary_paths:
        raise RuntimeError(f"training chunk {chunk_idx} did not write run_summary.json")
    summary = _read_json(summary_paths[-1])
    checkpoint = Path(str(summary["checkpoint_path"]))
    if not checkpoint.exists():
        raise RuntimeError(f"training chunk {chunk_idx} checkpoint not found: {checkpoint}")
    return checkpoint


def run_guarded_evaluation(
    *,
    checkpoint: Path,
    output_dir: Path,
    h2h_games: int,
    h2h_seed: int,
    phase1_enabled: bool,
    h2h_enabled: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_summary: dict[str, Any] | None = None
    h2h_summary: dict[str, Any] = {}
    if phase1_enabled:
        phase1_path = output_dir / "phase1_runtime_bench.json"
        _run_command(
            [
                str(PYTHON),
                str(ROOT / "TrainV3" / "scripts" / "run_phase1_runtime_acceptance_bench.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(phase1_path),
                "--exploit-policy",
                "state_score",
                "--legal-source",
                "classic_legal_action_ids",
            ],
            log_path=output_dir / "phase1_runtime_bench.log",
            success_path=phase1_path,
        )
        phase1_summary = summarize_phase1_report(_read_json(phase1_path))
    if h2h_enabled:
        for name, model_path, weight in DEFAULT_V4_MODELS:
            h2h_dir = output_dir / f"h2h_noassist_{name}_seed{h2h_seed}"
            report_path = h2h_dir / "v5_s1_assist_vs_v4max.json"
            _run_command(
                [
                    str(PYTHON),
                    str(ROOT / "TrainV3" / "scripts" / "run_v5_vs_v4max_benchmark.py"),
                    "--v4-model",
                    str(model_path),
                    "--v5-checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(h2h_dir),
                    "--games",
                    str(h2h_games),
                    "--seed",
                    str(h2h_seed),
                    "--max-steps",
                    "180",
                    "--no-bonuses",
                    "--no-private-info",
                    "--disable-draw-assist",
                    "--disable-assist-mode",
                    "--disable-deck-assist",
                ],
                log_path=output_dir / f"h2h_noassist_{name}.log",
                success_path=report_path,
            )
            report = _read_json(report_path)
            h2h_summary[name] = summarize_h2h_report(report, weight=weight)
    evaluation = {
        "checkpoint": str(checkpoint),
        "phase1": phase1_summary,
        "h2h": h2h_summary,
        "h2h_games": int(h2h_games),
        "h2h_seed": int(h2h_seed),
    }
    (output_dir / "guarded_evaluation_summary.json").write_text(
        json.dumps(_jsonable(evaluation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


def summarize_phase1_report(report: dict[str, Any]) -> dict[str, Any]:
    by_lane = report.get("by_lane") or {}
    lanes = {
        str(name): {
            "score_rate": float(values.get("score_rate", 0.0)),
            "p1_score_rate": float(values.get("p1_score_rate", 0.0)),
            "p2_score_rate": float(values.get("p2_score_rate", 0.0)),
            "avg_hp_margin": float(values.get("avg_hp_margin", 0.0)),
            "invalid_actions": int(values.get("invalid_actions", 0)),
        }
        for name, values in by_lane.items()
    }
    acceptance = report.get("acceptance") or {}
    critical_values = [lanes.get(name, {}).get("score_rate", 0.0) for name in CRITICAL_PHASE1_LANES]
    return {
        "acceptance": acceptance,
        "lanes": lanes,
        "critical_mean": sum(critical_values) / max(1, len(critical_values)),
        "critical_min": min(critical_values) if critical_values else 0.0,
        "legal_random_score": lanes.get("legal_random", {}).get("score_rate", 0.0),
        "stall_score": lanes.get("stall", {}).get("score_rate", 0.0),
        "invalid_action_pass": bool(acceptance.get("invalid_action_pass", False)),
    }


def summarize_h2h_report(report: dict[str, Any], *, weight: float) -> dict[str, Any]:
    summary = report.get("summary") or {}
    return {
        "weight": float(weight),
        "score_rate": float(summary.get("v5_score_rate", 0.0)),
        "p1_score_rate": float(summary.get("v5_p1_winrate", 0.0)),
        "p2_score_rate": float(summary.get("v5_p2_winrate", 0.0)),
        "first_score_rate": float(summary.get("v5_first_winrate", 0.0)),
        "second_score_rate": float(summary.get("v5_second_winrate", 0.0)),
        "avg_hp_margin": float(summary.get("avg_v5_hp_margin", 0.0)),
        "games": int(summary.get("games", 0)),
        "invalid_actions": int(summary.get("invalid_actions", 0)),
    }


def score_guarded_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    phase1 = evaluation.get("phase1") or {}
    h2h = evaluation.get("h2h") or {}
    v4_weight = sum(float(item.get("weight", 0.0)) for item in h2h.values())
    v4_league_score = (
        sum(float(item.get("score_rate", 0.0)) * float(item.get("weight", 0.0)) for item in h2h.values())
        / v4_weight
        if v4_weight > 0.0
        else 0.0
    )
    phase1_score = float(phase1.get("critical_mean", 0.0))
    legal_random = float(phase1.get("legal_random_score", 0.0))
    stall = float(phase1.get("stall_score", 0.0))
    critical_min = float(phase1.get("critical_min", 0.0))
    composite = 0.58 * v4_league_score + 0.28 * phase1_score + 0.08 * critical_min + 0.03 * legal_random + 0.03 * stall
    return {
        "composite": float(composite),
        "v4_league_score": float(v4_league_score),
        "phase1_score": float(phase1_score),
        "phase1_critical_min": float(critical_min),
        "legal_random_score": float(legal_random),
        "stall_score": float(stall),
        "phase1_lanes": phase1.get("lanes", {}),
        "h2h": h2h,
        "invalid_action_pass": bool(phase1.get("invalid_action_pass", True)),
    }


def decide_candidate(*, best: dict[str, Any], candidate: dict[str, Any], config: GuardConfig) -> dict[str, Any]:
    reasons: list[str] = []
    if config.require_invalid_action_pass and not bool(candidate.get("invalid_action_pass", False)):
        reasons.append("invalid_action_guard_failed")
    if float(candidate["legal_random_score"]) < float(config.require_legal_random):
        reasons.append("legal_random_guard_failed")
    if float(candidate["stall_score"]) < float(config.require_stall):
        reasons.append("stall_guard_failed")
    face_score = float((candidate.get("phase1_lanes") or {}).get("face_rush", {}).get("score_rate", 0.0))
    if face_score < float(config.require_face_rush):
        reasons.append("face_rush_guard_failed")
    for lane in ("face_rush", "punish_empty_board", "anti_draw_greed", "board_control"):
        old = float((best.get("phase1_lanes") or {}).get(lane, {}).get("score_rate", 0.0))
        new = float((candidate.get("phase1_lanes") or {}).get(lane, {}).get("score_rate", 0.0))
        if new + float(config.max_lane_regression) < old:
            reasons.append(f"{lane}_regressed")
    for name in ("v4-max", "v4-opti", "v4-micro"):
        old = float((best.get("h2h") or {}).get(name, {}).get("score_rate", 0.0))
        new = float((candidate.get("h2h") or {}).get(name, {}).get("score_rate", 0.0))
        if new + float(config.max_v4_regression) < old:
            reasons.append(f"{name}_regressed")
    delta = float(candidate["composite"]) - float(best["composite"])
    if delta < float(config.min_composite_delta):
        reasons.append("composite_delta_too_small")
    return {
        "accepted": not reasons,
        "delta": delta,
        "reasons": reasons,
    }


def _run_command(command: list[str], *, log_path: Path, success_path: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 and not (success_path is not None and success_path.exists()):
        raise RuntimeError(f"command failed with code {completed.returncode}: {' '.join(command)}; see {log_path}")


def _write_state(
    config: GuardConfig,
    best_checkpoint: Path,
    best_score: dict[str, Any],
    candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> None:
    state = {
        "schema": "extra_lr_v5_phase33_guarded_state_v1",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_checkpoint": str(best_checkpoint),
        "best_score": best_score,
        "accepted_chunks": accepted,
        "candidates": candidates,
        "config": _jsonable(asdict(config)),
    }
    (config.output_dir / "phase33_guarded_state.json").write_text(
        json.dumps(_jsonable(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _validate_config(config: GuardConfig) -> None:
    if not config.source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint not found: {config.source_checkpoint}")
    if not config.trace_manifest.exists():
        raise FileNotFoundError(f"trace manifest not found: {config.trace_manifest}")
    if not config.core_library.exists():
        raise FileNotFoundError(f"TrainV3 core library not found: {config.core_library}")
    for name, path, _weight in DEFAULT_V4_MODELS:
        if "v4.1" in name.lower() or "v4.1" in str(path).lower():
            raise ValueError("V4.1 must not be used in Phase33")
        if not path.exists():
            raise FileNotFoundError(f"V4 model not found: {path}")
    if int(config.chunks) <= 0:
        raise ValueError("chunks must be positive")
    if int(config.updates_per_chunk) <= 0:
        raise ValueError("updates_per_chunk must be positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3" / "runs" / f"phase33_guarded_noassist_v4_{stamp}")
    parser.add_argument("--trace-manifest", type=Path, default=DEFAULT_TRACE_MANIFEST)
    parser.add_argument("--core-library", type=Path, default=ROOT / "TrainV3" / "target" / "release" / "libtrainv3_core.dylib")
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--updates-per-chunk", type=int, default=5)
    parser.add_argument("--env-count", type=int, default=8192)
    parser.add_argument("--steps-per-update", type=int, default=12)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.006)
    parser.add_argument("--clip-epsilon", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=0.25)
    parser.add_argument("--opponent-mix", default=DEFAULT_OPPONENT_MIX)
    parser.add_argument("--policy-padding-mode", default="bucketed")
    parser.add_argument("--ppo-minibatch-plan", default="contiguous")
    parser.add_argument("--seed", type=int, default=33001)
    parser.add_argument("--h2h-games", type=int, default=16)
    parser.add_argument("--h2h-seed", type=int, default=33000)
    parser.add_argument("--min-composite-delta", type=float, default=0.01)
    parser.add_argument("--max-lane-regression", type=float, default=0.08)
    parser.add_argument("--max-v4-regression", type=float, default=0.06)
    parser.add_argument("--require-legal-random", type=float, default=0.90)
    parser.add_argument("--require-stall", type=float, default=0.90)
    parser.add_argument("--require-face-rush", type=float, default=0.40)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-h2h", action="store_true")
    parser.add_argument("--promote-rejected", action="store_true")
    parser.add_argument("--no-paired-guard-evaluation", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
