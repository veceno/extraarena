#!/usr/bin/env python3
"""Run lightweight Extra-LR V5 acceptance gates over completed training artifacts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3.5" / "python"))

from train_v3.gauntlet_v5 import V5GauntletConfig, build_default_exploit_gauntlet
from train_v3.league_v5 import compare_adaptive_strength_monotonicity


DEFAULT_CANDIDATE_RUN = ROOT / "TrainV3.5" / "runs" / "phase1_noassist_refresh_after_assist_20260604_184324"
DEFAULT_ASSIST_RUN = ROOT / "TrainV3.5" / "runs" / "phase1_private_assist_no_teacher_20260604_124056"
DEFAULT_PRIVATE_RUN = ROOT / "TrainV3.5" / "runs" / "phase1_privateinfo_noassist_20260604_095509"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--assist-run", type=Path, default=DEFAULT_ASSIST_RUN)
    parser.add_argument("--private-run", type=Path, default=DEFAULT_PRIVATE_RUN)
    parser.add_argument("--output-root", type=Path, default=ROOT / "TrainV3.5" / "runs")
    parser.add_argument("--min-e2e-tps", type=float, default=12_000.0)
    parser.add_argument("--min-collect-tps", type=float, default=14_000.0)
    parser.add_argument("--min-entropy", type=float, default=0.70)
    parser.add_argument("--min-last100-entropy", type=float, default=0.85)
    parser.add_argument("--max-abs-kl", type=float, default=0.12)
    parser.add_argument("--min-monotonic-margin", type=float, default=0.03)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--v4max-no-bonus-benchmark", type=Path)
    parser.add_argument("--min-no-bonus-p1", type=float, default=0.75)
    parser.add_argument("--max-no-bonus-p1", type=float, default=0.80)
    parser.add_argument("--min-no-bonus-p2", type=float, default=0.70)
    parser.add_argument("--max-no-bonus-p2", type=float, default=0.75)
    parser.add_argument("--min-no-bonus-second", type=float, default=0.70)
    args = parser.parse_args(argv)

    run_id = f"acceptance_v5_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.output_root.resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = None if bool(args.benchmark_only) else _load_run(args.candidate_run.resolve())
    assist = None if candidate is None else _load_optional_run(args.assist_run.resolve())
    private = None if candidate is None else _load_optional_run(args.private_run.resolve())
    thresholds = {
        "min_e2e_tps": args.min_e2e_tps,
        "min_collect_tps": args.min_collect_tps,
        "min_entropy": args.min_entropy,
        "min_last100_entropy": args.min_last100_entropy,
        "max_abs_kl": args.max_abs_kl,
        "min_monotonic_margin": args.min_monotonic_margin,
        "min_no_bonus_p1": args.min_no_bonus_p1,
        "max_no_bonus_p1": args.max_no_bonus_p1,
        "min_no_bonus_p2": args.min_no_bonus_p2,
        "max_no_bonus_p2": args.max_no_bonus_p2,
        "min_no_bonus_second": args.min_no_bonus_second,
    }

    gates: list[dict[str, Any]] = []
    if candidate is not None:
        gates.extend(_candidate_gates(candidate, thresholds))
        gates.extend(_cross_run_gates(candidate, assist, private))
    if args.v4max_no_bonus_benchmark is not None:
        gates.extend(_v4max_no_bonus_benchmark_gates(args.v4max_no_bonus_benchmark.resolve(), thresholds))
    gates.extend(_gauntlet_contract_gates(thresholds))

    passed = all(bool(gate["passed"]) for gate in gates)
    report = {
        "schema": "trainv3-v5-acceptance-report-v1",
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "pass" if passed else "fail",
        "thresholds": thresholds,
        "candidate": _run_public_summary(candidate) if candidate is not None else None,
        "assist_reference": _run_public_summary(assist) if assist is not None else None,
        "private_reference": _run_public_summary(private) if private is not None else None,
        "gates": gates,
    }
    report_path = out_dir / "acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root.resolve() / "latest_v5_acceptance_run.txt").write_text(str(out_dir) + "\n", encoding="utf-8")

    print("V5_ACCEPTANCE_REPORT", report_path)
    print(json.dumps(_console_summary(report), ensure_ascii=False, sort_keys=True))
    return 0 if passed else 2


def _load_optional_run(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.exists():
        return None
    return _load_run(run_dir)


def _load_run(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "run_summary.json")
    league = _read_json(run_dir / "league_manifest.json")
    trace = _read_json(run_dir / "trace_manifest.json")
    phase_config = _read_run_config(run_dir)
    metrics_path = Path(summary.get("metrics_path") or run_dir / "metrics.jsonl")
    metrics = _read_metrics(metrics_path)
    checkpoint_path = Path(summary.get("checkpoint_path") or league.get("checkpoint_path") or "")
    checkpoint_meta = _read_checkpoint_meta(checkpoint_path)
    return {
        "run_dir": run_dir,
        "summary": summary,
        "league": league,
        "trace": trace,
        "phase_config": phase_config,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "checkpoint_meta": checkpoint_meta,
    }


def _candidate_gates(run: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    summary = run["summary"]
    league = run["league"]
    trace = run["trace"]
    config = run["phase_config"]
    metrics = run["metrics"]
    meta = run["checkpoint_meta"]
    update_rows = [row for row in metrics if isinstance(row.get("update"), int)]
    last100 = update_rows[-100:]
    strengths = sorted({round(float(entry.get("strength", -1.0)), 3) for entry in trace.get("traces", [])})
    obs_dims = {int(entry.get("oracle", {}).get("obs_v5_dim", -1)) for entry in trace.get("traces", [])}
    trace_assist_modes = {_assist_mode_key(entry.get("assist_mode", {}), entry.get("draw_assist", {})) for entry in trace.get("traces", [])}
    trace_visibility_modes = {_visibility_key(entry.get("visibility", {})) for entry in trace.get("traces", [])}
    level_modes = config.get("level_modes", [])
    trace_level_modes = {
        str((entry.get("level_handicap") or {}).get("label", ""))
        for entry in trace.get("traces", [])
        if entry.get("level_handicap")
    }

    candidate_mode_gate = _candidate_mode_gate(
        config=config,
        trace_assist_modes=trace_assist_modes,
        trace_visibility_modes=trace_visibility_modes,
    )

    gates = [
        _gate(
            "candidate_artifacts_complete",
            bool(summary and league and trace and config and update_rows and run["checkpoint_path"].exists()),
            {
                "run_dir": str(run["run_dir"]),
                "updates": len(update_rows),
                "checkpoint": str(run["checkpoint_path"]),
            },
        ),
        _gate(
            "checkpoint_manifest_agreement",
            meta.get("model_name") == "extra-lr-v5-adaptive"
            and meta.get("run_name") == league.get("run_name") == summary.get("run_name")
            and meta.get("trace_manifest_id") == league.get("trace_manifest_id") == summary.get("trace_manifest_id")
            and str(run["checkpoint_path"]) == str(league.get("checkpoint_path")),
            {
                "meta_run_name": meta.get("run_name"),
                "league_run_name": league.get("run_name"),
                "trace_manifest_id": summary.get("trace_manifest_id"),
                "model_name": meta.get("model_name"),
            },
        ),
        _gate(
            "obs_v5_contract",
            obs_dims == {6480},
            {"obs_dims": sorted(obs_dims), "trace_count": len(trace.get("traces", []))},
        ),
        _gate(
            "mixed_strength_coverage",
            strengths == [0.25, 0.5, 0.75, 1.0],
            {"strengths": strengths},
        ),
        candidate_mode_gate,
        _gate(
            "ppo_stability",
            float(summary.get("min_entropy", 0.0)) >= thresholds["min_entropy"]
            and _mean([float(row["entropy"]) for row in last100]) >= thresholds["min_last100_entropy"]
            and float(summary.get("max_abs_approx_kl", 999.0)) <= thresholds["max_abs_kl"],
            {
                "min_entropy": summary.get("min_entropy"),
                "last100_entropy_mean": _mean([float(row["entropy"]) for row in last100]),
                "max_abs_kl": summary.get("max_abs_approx_kl"),
            },
        ),
        _gate(
            "rust_hot_path_and_storage",
            _nested(meta, "config", "advantage_backend") == "rust"
            and _nested(meta, "config", "policy_selection_backend") == "rust"
            and _nested(meta, "config", "observation_key") == "observation_v5"
            and int(summary.get("dense_bytes_any", 0) or 0) == 0
            and int(summary.get("next_observation_bytes_any", 0) or 0) == 0
            and int(summary.get("terminal_observation_bytes_any", 0) or 0) == 0
            and not bool(_nested(meta, "config", "store_next_observations"))
            and _nested(meta, "config", "terminal_observation_mode") == "none",
            {
                "advantage_backend": _nested(meta, "config", "advantage_backend"),
                "policy_selection_backend": _nested(meta, "config", "policy_selection_backend"),
                "observation_key": _nested(meta, "config", "observation_key"),
                "dense_bytes_any": summary.get("dense_bytes_any"),
                "store_next_observations": _nested(meta, "config", "store_next_observations"),
                "terminal_observation_mode": _nested(meta, "config", "terminal_observation_mode"),
            },
        ),
        _gate(
            "throughput_floor",
            float(summary.get("end_to_end_transitions_per_second", 0.0)) >= thresholds["min_e2e_tps"]
            and float(summary.get("mean_collect_transitions_per_second", 0.0)) >= thresholds["min_collect_tps"],
            {
                "e2e_tps": summary.get("end_to_end_transitions_per_second"),
                "collect_tps": summary.get("mean_collect_transitions_per_second"),
                "max_rss_mb": summary.get("max_rss_mb"),
            },
        ),
    ]
    if isinstance(level_modes, list) and level_modes:
        expected_labels = {str(mode.get("label", "")) for mode in level_modes if isinstance(mode, dict)}
        gates.append(
            _gate(
                "level_handicap_trace_coverage",
                bool(expected_labels) and expected_labels.issubset(trace_level_modes),
                {
                    "expected_labels": sorted(expected_labels),
                    "trace_labels": sorted(trace_level_modes),
                },
            )
        )
    return gates


def _cross_run_gates(
    candidate: dict[str, Any],
    assist: dict[str, Any] | None,
    private: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    if assist is not None:
        assist_config = assist["phase_config"]
        candidate_summary = candidate["summary"]
        gates.append(
            _gate(
                "assist_reference_consistency",
                _config_has_private_info(assist_config)
                and _config_has_draw_assist(assist_config)
                and _config_has_assembler(assist_config)
                and _config_has_desirerer(assist_config)
                and Path(str(assist["summary"].get("checkpoint_path", ""))).exists(),
                {
                    "assist_run": str(assist["run_dir"]),
                    "candidate_resume_checkpoint": candidate_summary.get("resume_checkpoint"),
                    "assist_checkpoint": assist["summary"].get("checkpoint_path"),
                    "private_info": _config_has_private_info(assist_config),
                    "draw_assist": _config_has_draw_assist(assist_config),
                    "assembler": _config_has_assembler(assist_config),
                    "desirerer": _config_has_desirerer(assist_config),
                },
            )
        )
    if private is not None:
        private_config = private["phase_config"]
        gates.append(
            _gate(
                "private_info_reference_consistency",
                bool(private_config.get("enemy_private_known"))
                and not bool(private_config.get("draw_assist_enabled"))
                and not bool(private_config.get("assembler_enabled"))
                and not bool(private_config.get("desirerer_enabled")),
                {
                    "private_run": str(private["run_dir"]),
                    "enemy_private_known": private_config.get("enemy_private_known"),
                    "draw_assist_enabled": private_config.get("draw_assist_enabled"),
                },
            )
        )
    return gates


def _gauntlet_contract_gates(thresholds: dict[str, float]) -> list[dict[str, Any]]:
    config = V5GauntletConfig(adaptive_strength_min_margin=thresholds["min_monotonic_margin"]).validate()
    exploit_lanes = [lane.to_dict() for lane in build_default_exploit_gauntlet()]
    monotonic = compare_adaptive_strength_monotonicity(
        lower_strength=0.25,
        higher_strength=1.0,
        seeds=(11, 17, 23, 31, 43, 59),
        scenarios_per_seed=8,
    )
    return [
        _gate(
            "gauntlet_contract_rust_first",
            config.require_rust_hot_path and all(lane["runtime"] == "rust" for lane in exploit_lanes),
            {"exploit_lanes": exploit_lanes, "gauntlet_config": config.to_dict()},
        ),
        _gate(
            "adaptive_strength_monotonicity_proxy",
            float(monotonic["mean_margin"]) >= config.adaptive_strength_min_margin
            and float(monotonic["min_pairwise_margin"]) > 0.0,
            monotonic,
        ),
    ]


def _v4max_no_bonus_benchmark_gates(path: Path, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    data = _read_json(path)
    summary = data.get("summary", {})
    config = data.get("config", {})
    modes = data.get("modes", {})
    info_mode = modes.get("info_mode", {})
    assist_mode = modes.get("assist_mode", {})
    recovery_mode = modes.get("second_start_recovery_reranker", {})
    p1 = float(summary.get("v5_p1_winrate", 0.0))
    p2 = float(summary.get("v5_p2_winrate", 0.0))
    second = float(summary.get("v5_second_winrate", 0.0))
    no_bonus_contract = (
        bool(config.get("private_info_enabled")) is False
        and bool(config.get("draw_assist_enabled")) is False
        and bool(config.get("assist_mode_enabled")) is False
        and bool(config.get("deck_assist_enabled")) is False
        and bool(info_mode.get("enemy_hand_known")) is False
        and bool(info_mode.get("enemy_deck_known")) is False
        and bool(info_mode.get("draw_assist_enabled")) is False
        and bool(assist_mode.get("assembler_enabled")) is False
        and bool(assist_mode.get("desirerer_enabled")) is False
        and int(summary.get("draw_assist_uses", 0) or 0) == 0
        and bool(config.get("second_start_search")) is False
        and bool(modes.get("second_start_search")) is False
        and int(summary.get("search_rerank_uses", 0) or 0) == 0
        and bool(recovery_mode.get("enabled")) is False
        and int(summary.get("recovery_rerank_uses", 0) or 0) == 0
    )
    return [
        _gate(
            "v4max_no_bonus_contract",
            no_bonus_contract,
            {
                "benchmark_path": str(path),
                "private_info_enabled": config.get("private_info_enabled"),
                "draw_assist_enabled": config.get("draw_assist_enabled"),
                "assist_mode_enabled": config.get("assist_mode_enabled"),
                "deck_assist_enabled": config.get("deck_assist_enabled"),
                "draw_assist_uses": summary.get("draw_assist_uses"),
                "second_start_search": config.get("second_start_search"),
                "search_rerank_uses": summary.get("search_rerank_uses"),
                "recovery_reranker": recovery_mode,
                "recovery_rerank_uses": summary.get("recovery_rerank_uses"),
                "info_mode": info_mode,
                "assist_mode": assist_mode,
            },
        ),
        _gate(
            "v4max_no_bonus_side_corridor",
            thresholds["min_no_bonus_p1"] <= p1 <= thresholds["max_no_bonus_p1"]
            and thresholds["min_no_bonus_p2"] <= p2 <= thresholds["max_no_bonus_p2"],
            {
                "v5_p1_winrate": p1,
                "v5_p2_winrate": p2,
                "p1_required": [thresholds["min_no_bonus_p1"], thresholds["max_no_bonus_p1"]],
                "p2_required": [thresholds["min_no_bonus_p2"], thresholds["max_no_bonus_p2"]],
                "overall_score_rate": summary.get("v5_score_rate"),
            },
        ),
        _gate(
            "v4max_no_bonus_second_start_floor",
            second >= thresholds["min_no_bonus_second"],
            {
                "v5_first_winrate": summary.get("v5_first_winrate"),
                "v5_second_winrate": second,
                "required_min_second": thresholds["min_no_bonus_second"],
            },
        ),
    ]


def _run_public_summary(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    summary = run["summary"]
    config = run["phase_config"]
    return {
        "run_dir": str(run["run_dir"]),
        "run_name": summary.get("run_name"),
        "checkpoint_path": summary.get("checkpoint_path"),
        "trace_manifest_id": summary.get("trace_manifest_id"),
        "updates": summary.get("updates"),
        "total_env_transitions": summary.get("total_env_transitions"),
        "e2e_tps": summary.get("end_to_end_transitions_per_second"),
        "collect_tps": summary.get("mean_collect_transitions_per_second"),
        "min_entropy": summary.get("min_entropy"),
        "max_abs_kl": summary.get("max_abs_approx_kl"),
        "adaptive_strengths": config.get("adaptive_strengths"),
        "enemy_private_known": config.get("enemy_private_known"),
        "draw_assist_enabled": config.get("draw_assist_enabled"),
        "assembler_enabled": config.get("assembler_enabled"),
        "desirerer_enabled": config.get("desirerer_enabled"),
    }


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    failed = [gate for gate in report["gates"] if not gate["passed"]]
    candidate = report.get("candidate") or {}
    return {
        "status": report["status"],
        "candidate_run": candidate.get("run_name"),
        "candidate_checkpoint": candidate.get("checkpoint_path"),
        "gate_count": len(report["gates"]),
        "failed_gates": [gate["name"] for gate in failed],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_run_config(run_dir: Path) -> dict[str, Any]:
    for name in ("phase1_config.json", "phase2_config.json", "phase3_config.json"):
        path = run_dir / name
        if path.exists():
            return _read_json(path)
    raise FileNotFoundError(f"no phase config found in {run_dir}")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_checkpoint_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return json.loads(data["__meta__"].tobytes().decode("utf-8"))


def _assist_mode_key(assist_mode: dict[str, Any], draw_assist: dict[str, Any]) -> tuple[str, str]:
    assist_on = bool(assist_mode.get("assembler_enabled")) or bool(assist_mode.get("desirerer_enabled"))
    draw_on = bool(draw_assist.get("draw_assist_enabled"))
    return ("assist_on" if assist_on else "assist_off", "draw_on" if draw_on else "draw_off")


def _visibility_key(visibility: dict[str, Any]) -> tuple[str, str]:
    enemy_known = bool(visibility.get("enemy_hand_known")) or bool(visibility.get("enemy_deck_known"))
    own_known = bool(visibility.get("own_hand_identity_known")) and bool(visibility.get("own_deck_known"))
    return ("enemy_known" if enemy_known else "enemy_hidden", "own_known" if own_known else "own_hidden")


def _candidate_mode_gate(
    *,
    config: dict[str, Any],
    trace_assist_modes: set[tuple[str, str]],
    trace_visibility_modes: set[tuple[str, str]],
) -> dict[str, Any]:
    details = {
        "trace_assist_modes": sorted(map(list, trace_assist_modes)),
        "trace_visibility_modes": sorted(map(list, trace_visibility_modes)),
    }
    mixed_assist_or_private = (
        _config_has_private_info(config)
        or _config_has_draw_assist(config)
        or _config_has_assembler(config)
        or _config_has_desirerer(config)
    )
    if mixed_assist_or_private:
        return _gate(
            "candidate_mixed_assist_private_mode",
            trace_assist_modes
            == {
                ("assist_off", "draw_off"),
                ("assist_off", "draw_on"),
                ("assist_on", "draw_off"),
                ("assist_on", "draw_on"),
            }
            and trace_visibility_modes == {("enemy_hidden", "own_known"), ("enemy_known", "own_known")},
            details,
        )
    return _gate(
        "candidate_no_assist_hidden_mode",
        not bool(config.get("enemy_private_known"))
        and not bool(config.get("draw_assist_enabled"))
        and not bool(config.get("assembler_enabled"))
        and not bool(config.get("desirerer_enabled"))
        and trace_assist_modes == {("assist_off", "draw_off")}
        and trace_visibility_modes == {("enemy_hidden", "own_known")},
        details,
    )


def _config_has_private_info(config: dict[str, Any]) -> bool:
    return bool(config.get("enemy_private_known")) or float(config.get("private_info_rate", 0.0) or 0.0) > 0.0


def _config_has_draw_assist(config: dict[str, Any]) -> bool:
    return bool(config.get("draw_assist_enabled")) or float(config.get("draw_assist_rate", 0.0) or 0.0) > 0.0


def _config_has_assembler(config: dict[str, Any]) -> bool:
    if bool(config.get("assembler_enabled")):
        return True
    return any(bool(mode.get("assembler_enabled")) for mode in _config_assist_modes(config))


def _config_has_desirerer(config: dict[str, Any]) -> bool:
    if bool(config.get("desirerer_enabled")):
        return True
    return any(bool(mode.get("desirerer_enabled")) for mode in _config_assist_modes(config))


def _config_assist_modes(config: dict[str, Any]) -> list[dict[str, Any]]:
    modes = config.get("assist_modes", [])
    if not isinstance(modes, list):
        return []
    return [mode for mode in modes if isinstance(mode, dict)]


def _gate(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
