#!/usr/bin/env python3
"""Analyze the paired 2x2 Assembler/CardOptimum V5 ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


MODES = ("none", "assembler_only", "cardoptimum_only", "both")
BASELINES = {"random", "greedy_face", "end_turn"}
BASE_V5 = "extra-lr-v5-postB-preV5-u29250"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_name(payload: dict[str, Any]) -> str:
    rows = payload.get("results") or []
    if not rows:
        raise ValueError("benchmark payload has no results")
    names = {str(row["focal_model"]) for row in rows}
    if len(names) != 1:
        raise ValueError(f"ambiguous focal models: {sorted(names)}")
    return names.pop()


def _score(row: dict[str, Any], candidate: str) -> float:
    if row.get("winner_name") == candidate:
        return 1.0
    if row.get("draw"):
        return 0.5
    return 0.0


def _aggregate(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    games = len(rows)
    wins = sum(row.get("winner_name") == candidate for row in rows)
    draws = sum(bool(row.get("draw")) for row in rows)
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins - draws,
        "draws": draws,
        "score_rate": (wins + 0.5 * draws) / games if games else 0.0,
    }


def _exact_sign_test(improved: int, worsened: int) -> float:
    total = improved + worsened
    if total == 0:
        return 1.0
    tail = sum(
        math.comb(total, index)
        for index in range(min(improved, worsened) + 1)
    ) / (2**total)
    return min(1.0, 2.0 * tail)


def _mode_maps(
    payloads: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    maps = {
        mode: {str(row["scenario_id"]): row for row in payloads[mode]["results"]}
        for mode in MODES
    }
    reference = set(maps["none"])
    for mode in MODES[1:]:
        if set(maps[mode]) != reference:
            missing = sorted(reference - set(maps[mode]))[:3]
            extra = sorted(set(maps[mode]) - reference)[:3]
            raise ValueError(
                f"scenario mismatch for {mode}: missing={missing}, extra={extra}"
            )
    return maps


def _cluster_cells(
    maps: dict[str, dict[str, dict[str, Any]]],
    candidate: str,
    scenario_ids: set[str] | None = None,
) -> tuple[list[tuple[str, int]], np.ndarray]:
    selected = scenario_ids if scenario_ids is not None else set(maps["none"])
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {mode: [] for mode in MODES}
    )
    for scenario_id in sorted(selected):
        for mode in MODES:
            row = maps[mode][scenario_id]
            key = (str(row["opponent_model"]), int(row["seed"]))
            grouped[key][mode].append(_score(row, candidate))
    keys = sorted(grouped)
    cells = np.asarray(
        [
            [
                float(np.mean(grouped[key][mode]))
                for mode in MODES
            ]
            for key in keys
        ],
        dtype=np.float64,
    )
    if cells.size and any(
        len(grouped[key][mode]) != 4 for key in keys for mode in MODES
    ):
        raise ValueError("each opponent/seed/mode cluster must have four seat/start cells")
    return keys, cells


def _bootstrap_effect(
    cells: np.ndarray,
    effect: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int,
    repeats: int = 10_000,
) -> tuple[float, list[float]]:
    if cells.shape[0] == 0:
        return 0.0, [0.0, 0.0]
    cluster_effect = effect(cells)
    point = float(np.mean(cluster_effect) * 100.0)
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        cluster_effect.shape[0],
        size=(repeats, cluster_effect.shape[0]),
    )
    distribution = cluster_effect[samples].mean(axis=1) * 100.0
    return point, [
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    ]


EFFECTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "assembler_without_cardoptimum": lambda x: x[:, 1] - x[:, 0],
    "assembler_with_cardoptimum": lambda x: x[:, 3] - x[:, 2],
    "cardoptimum_without_assembler": lambda x: x[:, 2] - x[:, 0],
    "cardoptimum_with_assembler": lambda x: x[:, 3] - x[:, 1],
    "both_vs_none": lambda x: x[:, 3] - x[:, 0],
    "assembler_main_effect": lambda x: 0.5
    * ((x[:, 1] - x[:, 0]) + (x[:, 3] - x[:, 2])),
    "cardoptimum_main_effect": lambda x: 0.5
    * ((x[:, 2] - x[:, 0]) + (x[:, 3] - x[:, 1])),
    "interaction": lambda x: x[:, 3] - x[:, 1] - x[:, 2] + x[:, 0],
}


def _effect_summary(cells: np.ndarray, *, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, (name, effect) in enumerate(EFFECTS.items()):
        point, ci = _bootstrap_effect(cells, effect, seed=seed + index)
        output[name] = {
            "delta_percentage_points": point,
            "cluster_bootstrap_ci95": ci,
        }
    return output


def _paired_outcomes(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    candidate: str,
) -> dict[str, Any]:
    left_map = {str(row["scenario_id"]): row for row in left}
    right_map = {str(row["scenario_id"]): row for row in right}
    improved = sum(
        _score(right_map[key], candidate) > _score(left_map[key], candidate)
        for key in left_map
    )
    worsened = sum(
        _score(right_map[key], candidate) < _score(left_map[key], candidate)
        for key in left_map
    )
    return {
        "improved": improved,
        "worsened": worsened,
        "same": len(left_map) - improved - worsened,
        "two_sided_sign_p": _exact_sign_test(improved, worsened),
    }


def _kind(opponent: str) -> str:
    if opponent == BASE_V5:
        return "base_v5"
    if opponent in BASELINES:
        return "baseline"
    return "learned_classic"


def _validity(
    payloads: dict[str, dict[str, Any]],
    maps: dict[str, dict[str, dict[str, Any]]],
    candidate: str,
) -> dict[str, Any]:
    all_rows = [
        row for mode in MODES for row in payloads[mode]["results"]
    ]
    scenario_ids = sorted(maps["none"])
    contracts = {
        json.dumps(row["information_contract"], sort_keys=True)
        for row in all_rows
    }
    deck_mismatches = {"none_vs_cardoptimum": 0, "assembler_vs_both": 0}
    opponent_deck_mismatches = 0
    for scenario_id in scenario_ids:
        rows = {mode: maps[mode][scenario_id] for mode in MODES}
        if rows["none"]["candidate_deck_ids"] != rows["cardoptimum_only"]["candidate_deck_ids"]:
            deck_mismatches["none_vs_cardoptimum"] += 1
        if rows["assembler_only"]["candidate_deck_ids"] != rows["both"]["candidate_deck_ids"]:
            deck_mismatches["assembler_vs_both"] += 1
        if len(
            {
                tuple(rows[mode]["opponent_deck_ids"])
                for mode in MODES
            }
        ) != 1:
            opponent_deck_mismatches += 1
    return {
        "scenarios_per_mode": len(scenario_ids),
        "battles_total": len(all_rows),
        "errors_by_mode": {
            mode: int(payloads[mode]["error_count"]) for mode in MODES
        },
        "non_terminal": sum(
            row["status"] not in {"p1_win", "p2_win", "draw"}
            for row in all_rows
        ),
        "timed_out": sum(bool(row["timed_out"]) for row in all_rows),
        "truncated": sum(bool(row["truncated"]) for row in all_rows),
        "candidate_invalid_actions": {
            mode: sum(
                int(row["invalid_actions"].get(candidate, 0))
                for row in payloads[mode]["results"]
            )
            for mode in MODES
        },
        "information_contracts": [
            json.loads(item) for item in sorted(contracts)
        ],
        "candidate_deck_pair_mismatches": deck_mismatches,
        "opponent_deck_mismatches": opponent_deck_mismatches,
        "assembler_presence_errors": sum(
            (maps[mode][scenario_id].get("assembler") is None)
            != (mode not in {"assembler_only", "both"})
            for mode in MODES
            for scenario_id in scenario_ids
        ),
        "cardoptimum_call_presence_errors": sum(
            (
                int(
                    maps[mode][scenario_id]["aux_stats"].get(
                        "cardoptimum_calls", 0
                    )
                )
                > 0
            )
            and mode not in {"cardoptimum_only", "both"}
            for mode in MODES
            for scenario_id in scenario_ids
        ),
    }


def analyze(output_dir: Path) -> dict[str, Any]:
    raw_paths = {mode: output_dir / mode / "raw.json" for mode in MODES}
    payloads = {mode: _read(raw_paths[mode]) for mode in MODES}
    candidates = {_candidate_name(payloads[mode]) for mode in MODES}
    if len(candidates) != 1:
        raise ValueError(f"candidate mismatch: {sorted(candidates)}")
    candidate = candidates.pop()
    maps = _mode_maps(payloads)
    keys, cells = _cluster_cells(maps, candidate)
    opponents = sorted({key[0] for key in keys})

    per_opponent = []
    for index, opponent in enumerate(opponents):
        ids = {
            scenario_id
            for scenario_id, row in maps["none"].items()
            if row["opponent_model"] == opponent
        }
        _, opponent_cells = _cluster_cells(maps, candidate, ids)
        per_opponent.append(
            {
                "opponent": opponent,
                "kind": _kind(opponent),
                "modes": {
                    mode: _aggregate(
                        [
                            row
                            for row in payloads[mode]["results"]
                            if row["opponent_model"] == opponent
                        ],
                        candidate,
                    )
                    for mode in MODES
                },
                "effects": _effect_summary(
                    opponent_cells,
                    seed=975_000 + index * 100,
                ),
                "both_vs_none_paired_outcomes": _paired_outcomes(
                    [
                        row
                        for row in payloads["none"]["results"]
                        if row["opponent_model"] == opponent
                    ],
                    [
                        row
                        for row in payloads["both"]["results"]
                        if row["opponent_model"] == opponent
                    ],
                    candidate,
                ),
            }
        )

    aggregates = []
    groups = {
        "all_opponents": set(opponents),
        "all_learned_including_base_v5": {
            opponent for opponent in opponents if _kind(opponent) != "baseline"
        },
        "learned_classic_models": {
            opponent for opponent in opponents if _kind(opponent) == "learned_classic"
        },
        "base_v5": {BASE_V5} & set(opponents),
    }
    for index, (name, names) in enumerate(groups.items()):
        ids = {
            scenario_id
            for scenario_id, row in maps["none"].items()
            if row["opponent_model"] in names
        }
        _, group_cells = _cluster_cells(maps, candidate, ids)
        aggregates.append(
            {
                "name": name,
                "opponents": sorted(names),
                "modes": {
                    mode: _aggregate(
                        [
                            row
                            for row in payloads[mode]["results"]
                            if row["opponent_model"] in names
                        ],
                        candidate,
                    )
                    for mode in MODES
                },
                "effects": _effect_summary(
                    group_cells,
                    seed=985_000 + index * 100,
                ),
            }
        )

    return {
        "schema": "extra_lr_v5_aux_2x2_ablation_analysis_v1",
        "candidate": candidate,
        "base_v5": BASE_V5,
        "modes": list(MODES),
        "cluster_unit": ["opponent_model", "seed"],
        "cluster_count": len(keys),
        "artifacts": {
            mode: {
                "path": str(raw_paths[mode]),
                "sha256": _sha256(raw_paths[mode]),
                "bytes": raw_paths[mode].stat().st_size,
            }
            for mode in MODES
        },
        "validity": _validity(payloads, maps, candidate),
        "aggregates": aggregates,
        "opponents": per_opponent,
        "interpretation_limits": [
            (
                "Assembler measures the deployable deck replacement intervention; "
                "it is not an intrinsic ranking comparison against a random deck "
                "drawn from the identical allowed pool."
            ),
            (
                "Metronome and TimeStamp are disabled in every arm and cannot "
                "affect battle outcomes."
            ),
        ],
    }


def _pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _effect_text(effect: dict[str, Any]) -> str:
    ci = effect["cluster_bootstrap_ci95"]
    return (
        f"{effect['delta_percentage_points']:+.2f} pp "
        f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
    )


def _verdict(effect: dict[str, Any]) -> str:
    low, high = effect["cluster_bootstrap_ci95"]
    if low > 0.0:
        return "positive"
    if high < 0.0:
        return "negative"
    return "uncertain"


def write_report(analysis: dict[str, Any], path: Path) -> None:
    validity = analysis["validity"]
    lines = [
        "# V5 Assembler/CardOptimum — paired 2x2 ablation",
        "",
        "## Validity",
        "",
        (
            f"- {validity['scenarios_per_mode']} paired scenarios per arm; "
            f"{validity['battles_total']} battles total; "
            f"{analysis['cluster_count']} opponent/seed clusters."
        ),
        (
            f"- errors={sum(validity['errors_by_mode'].values())}, "
            f"non_terminal={validity['non_terminal']}, "
            f"timed_out={validity['timed_out']}, "
            f"truncated={validity['truncated']}."
        ),
        (
            "- Candidate deck pair mismatches: "
            f"{validity['candidate_deck_pair_mismatches']}; "
            f"opponent deck mismatches={validity['opponent_deck_mismatches']}."
        ),
        (
            "- All arms hold history=20, full hand/deck visibility, "
            "draw-assist=false and assist_profile_id=0 fixed."
        ),
        "",
        "## Aggregate factorial results",
        "",
        "| Group | none | Assembler | CardOptimum | both | A main | C main | interaction | both-none |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["aggregates"]:
        modes = row["modes"]
        effects = row["effects"]
        lines.append(
            f"| {row['name']} | {_pct(modes['none']['score_rate'])} | "
            f"{_pct(modes['assembler_only']['score_rate'])} | "
            f"{_pct(modes['cardoptimum_only']['score_rate'])} | "
            f"{_pct(modes['both']['score_rate'])} | "
            f"{_effect_text(effects['assembler_main_effect'])} | "
            f"{_effect_text(effects['cardoptimum_main_effect'])} | "
            f"{_effect_text(effects['interaction'])} | "
            f"{_effect_text(effects['both_vs_none'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-opponent score rate",
            "",
            "| Opponent | none | Assembler | CardOptimum | both | both-none |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["opponents"]:
        modes = row["modes"]
        lines.append(
            f"| {row['opponent']} | {_pct(modes['none']['score_rate'])} | "
            f"{_pct(modes['assembler_only']['score_rate'])} | "
            f"{_pct(modes['cardoptimum_only']['score_rate'])} | "
            f"{_pct(modes['both']['score_rate'])} | "
            f"{_effect_text(row['effects']['both_vs_none'])} |"
        )
    all_row = next(
        row for row in analysis["aggregates"] if row["name"] == "all_opponents"
    )
    effects = all_row["effects"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"- Assembler main effect: {_effect_text(effects['assembler_main_effect'])}; "
                f"verdict={_verdict(effects['assembler_main_effect'])}."
            ),
            (
                f"- CardOptimum main effect: {_effect_text(effects['cardoptimum_main_effect'])}; "
                f"verdict={_verdict(effects['cardoptimum_main_effect'])}."
            ),
            (
                f"- Interaction: {_effect_text(effects['interaction'])}; "
                f"verdict={_verdict(effects['interaction'])}."
            ),
            (
                f"- Both versus neither: {_effect_text(effects['both_vs_none'])}; "
                f"verdict={_verdict(effects['both_vs_none'])}."
            ),
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis["interpretation_limits"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    analysis = analyze(output_dir)
    (output_dir / "ablation_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(analysis, output_dir / "ABLATION_REPORT.md")
    print(json.dumps(analysis, ensure_ascii=False))


if __name__ == "__main__":
    main()
