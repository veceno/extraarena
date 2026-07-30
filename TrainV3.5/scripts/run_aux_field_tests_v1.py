#!/usr/bin/env python3
"""Run leakage-aware field checks for Metronome V1 and TimeStamp V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from train_v3.aux_inference import MetronomeV1, TimeStampDuoV1, TimeStampMonoV1


TERMINAL_STATUSES = {"p1_win", "p2_win", "draw"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_group_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(str(row["group_id"]) for row in rows)
    target = max(1, round(sum(counts.values()) * 0.10))
    ordered = sorted(
        counts,
        key=lambda group_id: hashlib.sha256(
            f"aux-split:{group_id}".encode()
        ).digest(),
    )
    result: dict[str, str] = {}
    test_rows = validation_rows = 0
    for group_id in ordered:
        if test_rows < target:
            result[group_id] = "test"
            test_rows += counts[group_id]
        elif validation_rows < target:
            result[group_id] = "validation"
            validation_rows += counts[group_id]
        else:
            result[group_id] = "train"
    return result


def _regression_metrics(
    truth: list[float],
    predicted: list[float],
) -> dict[str, float | int]:
    if not truth:
        return {"rows": 0}
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    error = estimate - actual
    return {
        "rows": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "median_ae": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_bias": float(np.mean(error)),
    }


def _metronome_metrics(
    predictions: list[dict[str, Any]],
    *,
    baseline_median_ms: float,
) -> dict[str, Any]:
    if not predictions:
        return {"rows": 0, "status": "no_eligible_rows"}
    truth = [float(row["actual_ms"]) for row in predictions]
    point = [float(row["point_ms"]) for row in predictions]
    p50 = [float(row["p50_ms"]) for row in predictions]
    p90 = [float(row["p90_ms"]) for row in predictions]
    actual = np.asarray(truth)
    metrics = {
        **_regression_metrics(truth, point),
        "log_mae": float(
            np.mean(
                np.abs(
                    np.log1p(np.asarray(point, dtype=np.float64))
                    - np.log1p(actual)
                )
            )
        ),
        "p50_coverage": float(np.mean(actual <= np.asarray(p50))),
        "p90_coverage": float(np.mean(actual <= np.asarray(p90))),
        "actual_median_ms": float(np.median(actual)),
        "actual_p90_ms": float(np.quantile(actual, 0.9)),
        "predicted_point_median_ms": float(np.median(point)),
        "predicted_p90_median_ms": float(np.median(p90)),
        "training_median_baseline_ms": baseline_median_ms,
        "training_median_baseline_mae": float(
            np.mean(np.abs(actual - baseline_median_ms))
        ),
        "hardcoded_3_6_midpoint_mae": float(np.mean(np.abs(actual - 4500.0))),
        "clamped_low_rate": float(np.mean(np.asarray(point) <= 100.0)),
        "clamped_high_rate": float(np.mean(np.asarray(point) >= 25_000.0)),
    }
    action_breakdown: dict[str, Any] = {}
    for action_type in sorted({str(row["action_type"]) for row in predictions}):
        subset = [row for row in predictions if row["action_type"] == action_type]
        action_breakdown[action_type] = _regression_metrics(
            [float(row["actual_ms"]) for row in subset],
            [float(row["point_ms"]) for row in subset],
        )
    metrics["by_action_type"] = action_breakdown
    return metrics


def _predict_metronome_dataset(
    model: MetronomeV1,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        prediction = model.predict_features(np.asarray(row["features"], dtype=np.float64))
        predictions.append(
            {
                "group_id": row["group_id"],
                "battle_id": row["battle_id"],
                "action_type": row["action_type"],
                "actual_ms": float(row["latency_ms"]),
                "point_ms": prediction["point"],
                "p50_ms": prediction["p50"],
                "p90_ms": prediction["p90"],
                "cohort": "artifact_group_holdout",
            }
        )
    return predictions


def _predict_metronome_traces(
    model: MetronomeV1,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        latency = row.get("human_decision_time_ms")
        if (
            row.get("decision_source") != "human"
            or row.get("accepted") is not True
            or not isinstance(latency, (int, float))
            or not 100.0 <= float(latency) <= 25_000.0
        ):
            continue
        prediction = model.predict_trace(row)
        predictions.append(
            {
                "group_id": row["group_id"],
                "battle_id": row["battle_id"],
                "action_type": row.get("action_type"),
                "legal_action_count": int(row.get("legal_action_count", 0) or 0),
                "actual_ms": float(latency),
                "point_ms": prediction["point"],
                "p50_ms": prediction["p50"],
                "p90_ms": prediction["p90"],
                "cohort": "fresh_out_of_freeze",
            }
        )
    return predictions


def _timestamp_prediction(
    model: TimeStampMonoV1 | TimeStampDuoV1,
    row: dict[str, Any],
) -> dict[str, float]:
    return model.predict(
        actor_deck_ids=row["mono_deck_ids"],
        opponent_deck_ids=row["opponent_deck_ids"],
        actor_levels={
            int(card_id): int(level)
            for card_id, level in row["mono_levels"].items()
        },
        opponent_levels={
            int(card_id): int(level)
            for card_id, level in row["opponent_levels"].items()
        },
        actor_starts=row["starting_player_relative"] == "first",
    )


def _timestamp_metrics(
    rows: list[dict[str, Any]],
    model: TimeStampMonoV1 | TimeStampDuoV1,
    *,
    baseline_seconds: float,
    baseline_turns: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    for row in rows:
        prediction = _timestamp_prediction(model, row)
        predictions.append(
            {
                "group_id": row["group_id"],
                "battle_id": row["battle_id"],
                "actual_turns": float(row["turns"]),
                "actual_duration_seconds": float(row["duration_seconds"]),
                **prediction,
            }
        )
    if not predictions:
        return {"rows": 0, "status": "no_completed_fresh_battles"}, []
    duration_truth = [row["actual_duration_seconds"] for row in predictions]
    duration_pred = [row["duration_seconds"] for row in predictions]
    turn_truth = [row["actual_turns"] for row in predictions]
    turn_pred = [row["turns"] for row in predictions]
    actual = np.asarray(duration_truth)
    metrics = {
        "duration": {
            **_regression_metrics(duration_truth, duration_pred),
            "p50_coverage": float(
                np.mean(
                    actual
                    <= np.asarray(
                        [row["duration_p50_seconds"] for row in predictions]
                    )
                )
            ),
            "p90_coverage": float(
                np.mean(
                    actual
                    <= np.asarray(
                        [row["duration_p90_seconds"] for row in predictions]
                    )
                )
            ),
            "baseline_seconds": baseline_seconds,
            "baseline_mae": float(np.mean(np.abs(actual - baseline_seconds))),
        },
        "turns": {
            **_regression_metrics(turn_truth, turn_pred),
            "baseline_turns": baseline_turns,
            "baseline_mae": float(
                np.mean(np.abs(np.asarray(turn_truth) - baseline_turns))
            ),
        },
    }
    return metrics, predictions


def _cluster_bootstrap_duration_delta(
    primary: list[dict[str, Any]],
    *,
    baseline_seconds: float | None = None,
    comparison: list[dict[str, Any]] | None = None,
    samples: int = 20_000,
) -> dict[str, Any]:
    if not primary:
        return {"rows": 0}
    comparison_by_battle = (
        {row["battle_id"]: row for row in comparison}
        if comparison is not None
        else None
    )
    groups = sorted({str(row["group_id"]) for row in primary})
    by_group = {
        group_id: [row for row in primary if str(row["group_id"]) == group_id]
        for group_id in groups
    }

    def delta(rows: list[dict[str, Any]]) -> float:
        primary_error = np.asarray(
            [
                abs(
                    float(row["duration_seconds"])
                    - float(row["actual_duration_seconds"])
                )
                for row in rows
            ]
        )
        if comparison_by_battle is not None:
            reference_error = np.asarray(
                [
                    abs(
                        float(comparison_by_battle[row["battle_id"]]["duration_seconds"])
                        - float(row["actual_duration_seconds"])
                    )
                    for row in rows
                ]
            )
        else:
            assert baseline_seconds is not None
            reference_error = np.asarray(
                [
                    abs(float(row["actual_duration_seconds"]) - baseline_seconds)
                    for row in rows
                ]
            )
        return float(np.mean(primary_error - reference_error))

    rng = np.random.default_rng(20260727)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_rows = [
            row
            for group_id in sampled_groups
            for row in by_group[str(group_id)]
        ]
        draws[index] = delta(sampled_rows)
    return {
        "rows": len(primary),
        "groups": len(groups),
        "delta_mae_seconds": delta(primary),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "negative_favors_primary": True,
    }


def _fresh_inventory(
    root: Path,
    frozen_keys: set[tuple[str, str]],
    *,
    weights_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trace_rows: list[dict[str, Any]] = []
    completed_battles: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    inventory = []
    for meta_path in sorted(root.glob("*/battles/*/v5/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        key = (str(meta.get("group_id")), str(meta.get("battle_id")))
        trace_path = meta_path.with_name("actions.jsonl")
        rows = _read_jsonl(trace_path)
        record = {
            "group_id": key[0],
            "battle_id": key[1],
            "status": meta.get("status"),
            "created_at": meta.get("created_at"),
            "actions": len(rows),
            "meta_sha256": _sha256(meta_path),
            "actions_sha256": _sha256(trace_path) if trace_path.is_file() else None,
        }
        inventory.append(record)
        if key in frozen_keys:
            exclusion_counts["freeze_overlap"] += 1
            continue
        if meta.get("p1_actor_type") != "human":
            exclusion_counts["not_human_p1"] += 1
            continue
        if (meta.get("bot_policy") or {}).get("kind") != "v5":
            exclusion_counts["not_v5"] += 1
            continue
        if str((meta.get("bot_policy") or {}).get("weights_hash")) != weights_hash:
            exclusion_counts["wrong_weights"] += 1
            continue
        for row in rows:
            row["group_id"] = key[0]
            row["battle_id"] = key[1]
            trace_rows.append(row)
        if meta.get("status") not in TERMINAL_STATUSES:
            exclusion_counts["timestamp_non_terminal"] += 1
            continue
        duration = float(meta.get("duration_seconds") or 0.0)
        if not 10.0 <= duration <= 1800.0:
            exclusion_counts["timestamp_idle_or_censored"] += 1
            continue
        first_state = (rows[0].get("pre_state") or {}) if rows else {}
        starting_uid = first_state.get("current_turn_owner_id")
        p1_uid = meta.get("p1_user_id")
        completed_battles.append(
            {
                "group_id": key[0],
                "battle_id": key[1],
                "mono_deck_ids": [int(card["card_id"]) for card in meta["p1_deck"]],
                "mono_levels": {
                    str(card["card_id"]): int(card["level"])
                    for card in meta["p1_deck"]
                },
                "opponent_deck_ids": [
                    int(card["card_id"]) for card in meta["p2_deck"]
                ],
                "opponent_levels": {
                    str(card["card_id"]): int(card["level"])
                    for card in meta["p2_deck"]
                },
                "starting_player_relative": (
                    "first" if starting_uid == p1_uid else "second"
                ),
                "turns": int(meta.get("turns") or 0),
                "duration_seconds": duration,
                "completed": True,
                "idle_or_censored": False,
            }
        )
    return (
        trace_rows,
        completed_battles,
        {
            "battles_scanned": len(inventory),
            "freeze_overlap": sum(
                (row["group_id"], row["battle_id"]) in frozen_keys
                for row in inventory
            ),
            "exclusions": dict(exclusion_counts),
            "files": inventory,
        },
    )


def _fmt(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _report(payload: dict[str, Any]) -> str:
    fresh = payload["metronome"]["fresh_out_of_freeze"]
    holdout = payload["metronome"]["artifact_group_holdout"]
    mono = payload["timestamp"]["artifact_group_holdout"]["mono"]
    duo = payload["timestamp"]["artifact_group_holdout"]["duo"]
    comparison = payload["timestamp"]["artifact_group_holdout"]["comparison"]
    lines = [
        "# Metronome and TimeStamp V1 field-test report",
        "",
        "## Verdict",
        "",
        f"- Metronome: **{payload['metronome']['verdict']}**.",
        f"- TimeStamp Mono: **{payload['timestamp']['mono_verdict']}**.",
        f"- TimeStamp Duo: **{payload['timestamp']['duo_verdict']}**.",
        "- Tests are shadow-only: no delay was injected and no game action was changed.",
        "",
        "## Metronome — fresh production pilot",
        "",
        (
            f"- Fresh out-of-freeze sample: {fresh.get('rows', 0)} accepted human "
            f"decisions across {fresh.get('battles', 0)} battles / "
            f"{fresh.get('groups', 0)} groups."
        ),
        (
            f"- MAE {_fmt(fresh.get('mae'))} ms; median AE "
            f"{_fmt(fresh.get('median_ae'))} ms; log-MAE "
            f"{_fmt(fresh.get('log_mae'), 3)}."
        ),
        (
            f"- Coverage: p50={_fmt(fresh.get('p50_coverage'), 3)}, "
            f"p90={_fmt(fresh.get('p90_coverage'), 3)}."
        ),
        (
            f"- Baselines: learned training median MAE "
            f"{_fmt(fresh.get('training_median_baseline_mae'))} ms; hard-coded "
            f"3–6 s midpoint MAE {_fmt(fresh.get('hardcoded_3_6_midpoint_mae'))} ms."
        ),
        "",
        "## Metronome — artifact group holdout",
        "",
        (
            f"- {holdout.get('rows', 0)} decisions; MAE "
            f"{_fmt(holdout.get('mae'))} ms; p90 coverage "
            f"{_fmt(holdout.get('p90_coverage'), 3)}."
        ),
        "- Holdout groups were excluded from the saved artifact fit.",
        "",
        "## TimeStamp — artifact group holdout",
        "",
        (
            f"- Mono: {mono['duration'].get('rows', 0)} battles; duration MAE "
            f"{_fmt(mono['duration'].get('mae'))} s vs baseline "
            f"{_fmt(mono['duration'].get('baseline_mae'))} s; turn MAE "
            f"{_fmt(mono['turns'].get('mae'), 2)}."
        ),
        (
            f"- Duo: {duo['duration'].get('rows', 0)} battles; duration MAE "
            f"{_fmt(duo['duration'].get('mae'))} s vs baseline "
            f"{_fmt(duo['duration'].get('baseline_mae'))} s; turn MAE "
            f"{_fmt(duo['turns'].get('mae'), 2)}."
        ),
        (
            f"- Duo minus baseline MAE: "
            f"{_fmt(comparison['duo_vs_baseline']['delta_mae_seconds'], 2)} s; "
            f"cluster-bootstrap 95% CI "
            f"[{_fmt(comparison['duo_vs_baseline']['ci95'][0], 2)}, "
            f"{_fmt(comparison['duo_vs_baseline']['ci95'][1], 2)}]."
        ),
        (
            f"- Fresh completed out-of-freeze battles: "
            f"{payload['timestamp']['fresh_out_of_freeze']['completed_battles']}."
        ),
        "",
        "## Field gates",
        "",
        "- Metronome full field gate: at least 500 fresh decisions from at least 10 groups.",
        "- TimeStamp live gate: at least 30 fresh completed battles; Duo also needs unseen deck-pair coverage.",
        "- Abandoned/non-terminal battles remain censored and never become ordinary TimeStamp targets.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--human-datasets-dir", type=Path, required=True)
    parser.add_argument("--freeze-selection", type=Path, required=True)
    parser.add_argument("--fresh-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-hash", default="d09ed1941aeb707e")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = args.models_dir.resolve()
    datasets = args.human_datasets_dir.resolve()
    selection = json.loads(args.freeze_selection.read_text(encoding="utf-8"))
    frozen_keys = {
        (str(row["group_id"]), str(row["battle_id"]))
        for row in selection["battle_rows"]
    }

    metronome = MetronomeV1(models / "extra_lr_metronome_v1.npz")
    mono = TimeStampMonoV1(models / "extra_lr_timestamp_v1_mono.npz")
    duo = TimeStampDuoV1(models / "extra_lr_timestamp_v1_duo.npz")

    timing_rows = [
        row
        for row in _read_jsonl(datasets / "metronome_human.jsonl")
        if not row["idle_or_censored"]
    ]
    timing_splits = _balanced_group_splits(timing_rows)
    timing_train = [
        row for row in timing_rows if timing_splits[row["group_id"]] == "train"
    ]
    timing_test = [
        row for row in timing_rows if timing_splits[row["group_id"]] == "test"
    ]
    timing_baseline = float(
        statistics.median(float(row["latency_ms"]) for row in timing_train)
    )
    holdout_metronome_predictions = _predict_metronome_dataset(
        metronome, timing_test
    )

    fresh_traces, fresh_completed, inventory = _fresh_inventory(
        args.fresh_sessions.resolve(),
        frozen_keys,
        weights_hash=args.weights_hash,
    )
    fresh_metronome_predictions = _predict_metronome_traces(
        metronome, fresh_traces
    )

    human_battles = [
        row
        for row in _read_jsonl(datasets / "timestamp_human.jsonl")
        if row["completed"] and not row["idle_or_censored"]
    ]
    battle_splits = _balanced_group_splits(human_battles)
    calibration_rows = [
        row for row in human_battles if battle_splits[row["group_id"]] != "test"
    ]
    battle_test = [
        row for row in human_battles if battle_splits[row["group_id"]] == "test"
    ]
    duration_baseline = float(
        statistics.median(float(row["duration_seconds"]) for row in calibration_rows)
    )
    turn_baseline = float(
        statistics.median(float(row["turns"]) for row in calibration_rows)
    )
    mono_holdout, mono_predictions = _timestamp_metrics(
        battle_test,
        mono,
        baseline_seconds=duration_baseline,
        baseline_turns=turn_baseline,
    )
    duo_holdout, duo_predictions = _timestamp_metrics(
        battle_test,
        duo,
        baseline_seconds=duration_baseline,
        baseline_turns=turn_baseline,
    )
    mono_fresh, mono_fresh_predictions = _timestamp_metrics(
        fresh_completed,
        mono,
        baseline_seconds=duration_baseline,
        baseline_turns=turn_baseline,
    )
    duo_fresh, duo_fresh_predictions = _timestamp_metrics(
        fresh_completed,
        duo,
        baseline_seconds=duration_baseline,
        baseline_turns=turn_baseline,
    )
    timestamp_comparison = {
        "mono_vs_baseline": _cluster_bootstrap_duration_delta(
            mono_predictions,
            baseline_seconds=duration_baseline,
        ),
        "duo_vs_baseline": _cluster_bootstrap_duration_delta(
            duo_predictions,
            baseline_seconds=duration_baseline,
        ),
        "duo_vs_mono": _cluster_bootstrap_duration_delta(
            duo_predictions,
            comparison=mono_predictions,
        ),
    }

    fresh_metronome = _metronome_metrics(
        fresh_metronome_predictions,
        baseline_median_ms=timing_baseline,
    )
    fresh_metronome["battles"] = len(
        {row["battle_id"] for row in fresh_metronome_predictions}
    )
    fresh_metronome["groups"] = len(
        {row["group_id"] for row in fresh_metronome_predictions}
    )
    holdout_metronome = _metronome_metrics(
        holdout_metronome_predictions,
        baseline_median_ms=timing_baseline,
    )

    metronome_technical_pass = (
        fresh_metronome.get("rows", 0) >= 50
        and fresh_metronome.get("mae", math.inf)
        < fresh_metronome.get("hardcoded_3_6_midpoint_mae", -math.inf)
        and 0.70 <= fresh_metronome.get("p90_coverage", 0.0) <= 0.99
        and fresh_metronome.get("clamped_high_rate", 1.0) < 0.05
    )
    mono_beats_baseline = (
        mono_holdout["duration"]["mae"]
        < mono_holdout["duration"]["baseline_mae"]
    )
    duo_beats_baseline = (
        duo_holdout["duration"]["mae"]
        < duo_holdout["duration"]["baseline_mae"]
    )
    duo_resolved_better_than_baseline = (
        duo_beats_baseline
        and timestamp_comparison["duo_vs_baseline"]["ci95"][1] < 0.0
    )

    payload = {
        "schema": "extra_lr_aux_field_test_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "shadow_only_no_gameplay_effect",
        "provenance": {
            "weights_hash": args.weights_hash,
            "freeze_battles": len(frozen_keys),
            "fresh_inventory": inventory,
            "models": {
                path.name: _sha256(path)
                for path in sorted(models.glob("extra_lr_*metronome*.npz"))
                + sorted(models.glob("extra_lr_timestamp*.npz"))
            },
        },
        "metronome": {
            "projection_contract": "raw_v5_p1_p2_to_actor_private_opponent_public_v1",
            "fresh_out_of_freeze": fresh_metronome,
            "artifact_group_holdout": holdout_metronome,
            "technical_pilot_pass": metronome_technical_pass,
            "full_field_gate": {
                "minimum_fresh_decisions": 500,
                "minimum_groups": 10,
                "passed": (
                    metronome_technical_pass
                    and fresh_metronome.get("rows", 0) >= 500
                    and fresh_metronome.get("groups", 0) >= 10
                ),
            },
            "verdict": (
                "technical pilot passed; more fresh groups required"
                if metronome_technical_pass
                else "pilot failed; do not inject delays"
            ),
        },
        "timestamp": {
            "artifact_group_holdout": {
                "mono": mono_holdout,
                "duo": duo_holdout,
                "comparison": timestamp_comparison,
            },
            "fresh_out_of_freeze": {
                "completed_battles": len(fresh_completed),
                "mono": mono_fresh,
                "duo": duo_fresh,
                "live_gate_passed": len(fresh_completed) >= 30,
            },
            "mono_verdict": (
                "holdout failed baseline; do not ship"
                if not mono_beats_baseline
                else "holdout passed; fresh live gate pending"
            ),
            "duo_verdict": (
                "holdout improvement statistically resolved; fresh live gate pending"
                if duo_resolved_better_than_baseline
                else (
                    "point estimate beats baseline narrowly, but CI crosses zero; experimental only"
                    if duo_beats_baseline
                    else "holdout failed baseline; do not ship"
                )
            ),
        },
    }

    _write_jsonl(
        output / "metronome_predictions.jsonl",
        holdout_metronome_predictions + fresh_metronome_predictions,
    )
    _write_jsonl(
        output / "timestamp_predictions.jsonl",
        [
            {**row, "model": "mono", "cohort": "artifact_group_holdout"}
            for row in mono_predictions
        ]
        + [
            {**row, "model": "duo", "cohort": "artifact_group_holdout"}
            for row in duo_predictions
        ]
        + [
            {**row, "model": "mono", "cohort": "fresh_out_of_freeze"}
            for row in mono_fresh_predictions
        ]
        + [
            {**row, "model": "duo", "cohort": "fresh_out_of_freeze"}
            for row in duo_fresh_predictions
        ],
    )
    (output / "field_test_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "FIELD_TEST_REPORT.md").write_text(
        _report(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
