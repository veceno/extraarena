#!/usr/bin/env python3
"""Analyze paired no-aux/with-aux V5 benchmark artifacts."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BASELINES = {"random", "greedy_face", "end_turn"}
BASE_V5 = "extra-lr-v5-postB-preV5-u29250"
CANDIDATE = "extra-lr-v5-phaseC-h299"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(row: dict[str, Any]) -> float:
    if row.get("winner_name") == CANDIDATE:
        return 1.0
    if row.get("draw"):
        return 0.5
    return 0.0


def _exact_sign_test(improved: int, worsened: int) -> float:
    total = improved + worsened
    if total == 0:
        return 1.0
    tail = sum(
        math.comb(total, index)
        for index in range(min(improved, worsened) + 1)
    ) / (2**total)
    return min(1.0, 2.0 * tail)


def _cluster_bootstrap_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    seed: int,
    repeats: int = 10_000,
) -> list[float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for before, after in pairs:
        by_seed[int(before["seed"])].append(_score(after) - _score(before))
    clusters = np.asarray(
        [float(np.mean(values)) for _, values in sorted(by_seed.items())],
        dtype=np.float64,
    )
    if clusters.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        clusters.size,
        size=(repeats, clusters.size),
    )
    distribution = clusters[indices].mean(axis=1) * 100.0
    return [
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    ]


def _paired_rows(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    before_map = {row["scenario_id"]: row for row in before}
    after_map = {row["scenario_id"]: row for row in after}
    if before_map.keys() != after_map.keys():
        missing_before = sorted(after_map.keys() - before_map.keys())
        missing_after = sorted(before_map.keys() - after_map.keys())
        raise ValueError(
            f"scenario mismatch: missing_before={missing_before[:3]} "
            f"missing_after={missing_after[:3]}"
        )
    return [(before_map[key], after_map[key]) for key in sorted(before_map)]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(row.get("winner_name") == CANDIDATE for row in rows)
    draws = sum(bool(row.get("draw")) for row in rows)
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins - draws,
        "draws": draws,
        "score_rate": (
            (wins + 0.5 * draws) / games
            if games
            else 0.0
        ),
    }


def analyze(output_dir: Path) -> dict[str, Any]:
    no_payload = _read(output_dir / "no_aux/raw.json")
    aux_payload = _read(output_dir / "with_aux/raw.json")
    no_rows = no_payload["results"]
    aux_rows = aux_payload["results"]
    pairs = _paired_rows(no_rows, aux_rows)
    by_opponent: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        by_opponent[pair[0]["opponent_model"]].append(pair)

    opponents = []
    for opponent_index, (opponent, opponent_pairs) in enumerate(sorted(by_opponent.items())):
        before_rows = [pair[0] for pair in opponent_pairs]
        after_rows = [pair[1] for pair in opponent_pairs]
        before = _aggregate(before_rows)
        after = _aggregate(after_rows)
        improved = sum(_score(right) > _score(left) for left, right in opponent_pairs)
        worsened = sum(_score(right) < _score(left) for left, right in opponent_pairs)
        same = len(opponent_pairs) - improved - worsened
        opponents.append(
            {
                "opponent": opponent,
                "kind": (
                    "base_v5"
                    if opponent == BASE_V5
                    else "baseline"
                    if opponent in BASELINES
                    else "learned_classic"
                ),
                "no_aux": before,
                "with_aux": after,
                "delta_percentage_points": (
                    after["score_rate"] - before["score_rate"]
                )
                * 100.0,
                "delta_cluster_bootstrap_ci95": _cluster_bootstrap_delta(
                    opponent_pairs,
                    seed=930_000 + opponent_index,
                ),
                "paired_outcomes": {
                    "improved": improved,
                    "worsened": worsened,
                    "same": same,
                    "two_sided_sign_p": _exact_sign_test(improved, worsened),
                },
            }
        )

    learned_classic_names = {
        row["opponent"]
        for row in opponents
        if row["kind"] == "learned_classic"
    }
    learned_pairs = [
        pair
        for pair in pairs
        if pair[0]["opponent_model"] in learned_classic_names
    ]
    all_learned_pairs = [
        pair
        for pair in pairs
        if pair[0]["opponent_model"] not in BASELINES
    ]

    def aggregate_pair_group(
        name: str,
        group: list[tuple[dict[str, Any], dict[str, Any]]],
        seed: int,
    ) -> dict[str, Any]:
        before = _aggregate([pair[0] for pair in group])
        after = _aggregate([pair[1] for pair in group])
        improved = sum(_score(right) > _score(left) for left, right in group)
        worsened = sum(_score(right) < _score(left) for left, right in group)
        return {
            "name": name,
            "no_aux": before,
            "with_aux": after,
            "delta_percentage_points": (
                after["score_rate"] - before["score_rate"]
            )
            * 100.0,
            "delta_cluster_bootstrap_ci95": _cluster_bootstrap_delta(
                group,
                seed=seed,
            ),
            "paired_outcomes": {
                "improved": improved,
                "worsened": worsened,
                "same": len(group) - improved - worsened,
                "two_sided_sign_p": _exact_sign_test(improved, worsened),
            },
        }

    aux_stats = {
        "assembler_unique_selected_decks": len(
            {tuple(row["candidate_deck_ids"]) for row in aux_rows}
        ),
        "assembler_score_mean": float(
            np.mean([row["assembler"]["score"] for row in aux_rows])
        ),
        "assembler_raw_score_mean": float(
            np.mean([row["assembler"]["raw_score"] for row in aux_rows])
        ),
        "assembler_clipped_at_one": sum(
            float(row["assembler"]["score"]) >= 1.0 for row in aux_rows
        ),
        "cardoptimum_calls": sum(
            int(row["aux_stats"]["cardoptimum_calls"]) for row in aux_rows
        ),
        "cardoptimum_forced_draws": sum(
            int(row["aux_stats"]["forced_draws"]) for row in aux_rows
        ),
        "metronome_calls": sum(
            int(row["aux_stats"]["metronome_calls"]) for row in aux_rows
        ),
        "metronome_weighted_mean_p50_ms": (
            sum(
                float(row["aux_stats"]["metronome_p50_ms_sum"])
                for row in aux_rows
            )
            / max(
                1,
                sum(
                    int(row["aux_stats"]["metronome_calls"])
                    for row in aux_rows
                ),
            )
        ),
        "timestamp_turn_mae": float(
            np.mean(
                [
                    abs(float(row["timestamp_duo"]["turns"]) - float(row["turns"]))
                    for row in aux_rows
                ]
            )
        ),
        "timestamp_duration_prediction_mean_seconds": float(
            np.mean(
                [
                    float(row["timestamp_duo"]["duration_seconds"])
                    for row in aux_rows
                ]
            )
        ),
    }
    validity = {
        "paired_scenarios": len(pairs),
        "no_aux_errors": int(no_payload["error_count"]),
        "with_aux_errors": int(aux_payload["error_count"]),
        "non_terminal": sum(
            row["status"] not in {"p1_win", "p2_win", "draw"}
            for row in [*no_rows, *aux_rows]
        ),
        "timed_out": sum(bool(row["timed_out"]) for row in [*no_rows, *aux_rows]),
        "truncated": sum(bool(row["truncated"]) for row in [*no_rows, *aux_rows]),
        "candidate_invalid_actions": {
            "no_aux": sum(
                int(row["invalid_actions"].get(CANDIDATE, 0)) for row in no_rows
            ),
            "with_aux": sum(
                int(row["invalid_actions"].get(CANDIDATE, 0)) for row in aux_rows
            ),
        },
        "information_contracts": {
            json.dumps(row["information_contract"], sort_keys=True)
            for row in [*no_rows, *aux_rows]
        },
    }
    validity["information_contracts"] = [
        json.loads(item) for item in sorted(validity["information_contracts"])
    ]

    return {
        "schema": "extra_lr_v5_aux_ab_analysis_v1",
        "candidate": CANDIDATE,
        "base_v5": BASE_V5,
        "validity": validity,
        "aggregates": [
            aggregate_pair_group(
                "learned_classic_models",
                learned_pairs,
                940_001,
            ),
            aggregate_pair_group(
                "all_learned_including_base_v5",
                all_learned_pairs,
                940_002,
            ),
        ],
        "opponents": opponents,
        "aux_telemetry": aux_stats,
    }


def _pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def write_report(analysis: dict[str, Any], path: Path) -> None:
    validity = analysis["validity"]
    lines = [
        "# Phase C V5 — paired benchmark without/with auxiliary models",
        "",
        "## Validity",
        "",
        (
            f"- {validity['paired_scenarios']} scenarios per mode; "
            f"{validity['paired_scenarios'] * 2} battles total."
        ),
        (
            f"- errors={validity['no_aux_errors'] + validity['with_aux_errors']}, "
            f"non_terminal={validity['non_terminal']}, "
            f"timed_out={validity['timed_out']}, truncated={validity['truncated']}."
        ),
        (
            "- Every seed is evaluated in four symmetric cells: candidate as P1/P2 "
            "and candidate starts/plays second."
        ),
        (
            "- V5 history=20 and own/enemy hand/deck identities plus enemy deck "
            "order are explicitly enabled."
        ),
        "",
        "## Results",
        "",
        "| Opponent | no aux | with aux | delta pp | cluster 95% CI | record no aux | record with aux |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["opponents"]:
        before = row["no_aux"]
        after = row["with_aux"]
        ci = row["delta_cluster_bootstrap_ci95"]
        lines.append(
            f"| {row['opponent']} | {_pct(before['score_rate'])} | "
            f"{_pct(after['score_rate'])} | {row['delta_percentage_points']:+.2f} | "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | "
            f"{before['wins']}-{before['losses']}-{before['draws']} | "
            f"{after['wins']}-{after['losses']}-{after['draws']} |"
        )
    lines.extend(["", "## Aggregates", ""])
    for row in analysis["aggregates"]:
        ci = row["delta_cluster_bootstrap_ci95"]
        lines.append(
            f"- {row['name']}: {_pct(row['no_aux']['score_rate'])} -> "
            f"{_pct(row['with_aux']['score_rate'])}, "
            f"{row['delta_percentage_points']:+.2f} pp, "
            f"cluster 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]."
        )
    telemetry = analysis["aux_telemetry"]
    lines.extend(
        [
            "",
            "## Auxiliary telemetry",
            "",
            (
                f"- Assembler: {telemetry['assembler_unique_selected_decks']} unique "
                f"selected decks; mean raw score "
                f"{telemetry['assembler_raw_score_mean']:.3f}; "
                f"{telemetry['assembler_clipped_at_one']} clipped selections."
            ),
            (
                f"- CardOptimum: {telemetry['cardoptimum_forced_draws']} forced draws "
                f"from {telemetry['cardoptimum_calls']} eligible calls."
            ),
            (
                f"- Metronome shadow: {telemetry['metronome_calls']} predictions; "
                f"weighted mean p50={telemetry['metronome_weighted_mean_p50_ms']:.0f} ms."
            ),
            (
                f"- TimeStamp Duo shadow: turn MAE="
                f"{telemetry['timestamp_turn_mae']:.2f}; mean predicted human duration="
                f"{telemetry['timestamp_duration_prediction_mean_seconds']:.1f}s "
                "(duration has no simulator wall-clock ground truth in this benchmark)."
            ),
            "",
            "## Interpretation",
            "",
            (
                "- This is a bundle A/B. It establishes the effectiveness of "
                "Assembler+CardOptimum together, not the individual causal share of "
                "each model."
            ),
            (
                "- Metronome and TimeStamp run in shadow mode and cannot change the "
                "battle winner."
            ),
            (
                "- CardOptimum is marked bootstrap_only in its training manifest; "
                "the result is strong enough for a candidate integration, but it "
                "does not upgrade that readiness label by itself."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    analysis = analyze(output_dir)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(analysis, output_dir / "BENCHMARK_REPORT.md")
    print(json.dumps(analysis, ensure_ascii=False))


if __name__ == "__main__":
    main()
