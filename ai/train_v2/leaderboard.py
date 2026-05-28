"""
Checkpoint leaderboard for TrainV2 ONNX models.

CLI:
    python3 -m ai.train_v2.leaderboard --paths ai/train_v2/runs --games 8 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ai.train_v2.berserk_eval import (
    BerserkBrainPolicy,
    make_train_v2_berserk_brain,
    evaluate_berserk_matchup,
    compare_berserk_to_onnx_policy,
)


def discover_onnx_models(paths: list[str]) -> list[str]:
    onnx_files: set[str] = set()

    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            continue

        if p.is_file() and p.suffix == ".onnx":
            onnx_files.add(str(p.resolve()))
        elif p.is_dir():
            exported_dir = p / "exported"
            if exported_dir.is_dir():
                for onnx_file in exported_dir.glob("*.onnx"):
                    onnx_files.add(str(onnx_file.resolve()))
            else:
                for onnx_file in p.rglob("*.onnx"):
                    onnx_files.add(str(onnx_file.resolve()))

    return sorted(onnx_files)


def evaluate_onnx_model_for_leaderboard(
    onnx_path: str,
    *,
    seeds: list[int],
    opponents: list[str] | None = None,
    max_steps: int = 200,
) -> dict:
    if opponents is None:
        opponents = ["random", "end_turn", "greedy_face"]

    try:
        brain = make_train_v2_berserk_brain(onnx_path, selection="argmax")
        berserk_pol = BerserkBrainPolicy(brain, difficulty="test")

        opp_results: dict[str, Any] = {}
        for opp_name in opponents:
            opp_cls = __import__("ai.train_v2.berserk_eval", fromlist=["OPPONENT_REGISTRY"]).OPPONENT_REGISTRY[opp_name]
            opp_pol = opp_cls()
            er = evaluate_berserk_matchup(
                berserk_pol, opp_pol,
                seeds=seeds, swap_sides=True, max_steps=max_steps,
            )
            opp_results[opp_name] = {
                "winrate": er["p1_winrate"],
                "games": er["games"],
                "avg_turns": er["avg_turns"],
                "avg_steps": er["avg_steps"],
                "invalid_actions": er["invalid_actions"],
                "brain_invalid_actions": er["p1_brain_invalid_actions"],
                "latency_ms_p50": er["p1_latency_ms_p50"],
                "latency_ms_p95": er["p1_latency_ms_p95"],
                "truncations": er["truncations"],
            }

        parity = compare_berserk_to_onnx_policy(
            onnx_path, seed=seeds[0], steps=min(max_steps, 40), selection="argmax",
        )

        total_invalid = sum(r["invalid_actions"] for r in opp_results.values())
        total_brain_invalid = sum(r["brain_invalid_actions"] for r in opp_results.values())

        wr_random = opp_results.get("random", {}).get("winrate", 0.0)
        wr_greedy = opp_results.get("greedy_face", {}).get("winrate", 0.0)
        wr_end_turn = opp_results.get("end_turn", {}).get("winrate", 0.0)

        score = (
            1.0 * wr_random
            + 0.5 * wr_greedy
            + 0.25 * wr_end_turn
            - 0.05 * (total_invalid + total_brain_invalid)
            - 0.01 * parity["mismatches"]
        )

        return {
            "onnx_path": str(Path(onnx_path).resolve()),
            "model_name": Path(onnx_path).stem,
            "opponents": opp_results,
            "score": score,
            "parity_mismatches": parity["mismatches"],
            "parity_checked": parity["checked"],
        }

    except Exception as exc:
        return {
            "onnx_path": str(Path(onnx_path).resolve()),
            "model_name": Path(onnx_path).stem,
            "opponents": {},
            "score": -999.0,
            "parity_mismatches": 999999,
            "parity_checked": 0,
            "error": str(exc),
        }


def build_leaderboard(
    paths: list[str],
    *,
    seeds: list[int],
    opponents: list[str] | None = None,
    max_steps: int = 200,
) -> dict:
    if opponents is None:
        opponents = ["random", "end_turn", "greedy_face"]

    onnx_models = discover_onnx_models(paths)
    rows: list[dict] = []

    for onnx_path in onnx_models:
        result = evaluate_onnx_model_for_leaderboard(
            onnx_path, seeds=seeds, opponents=opponents, max_steps=max_steps,
        )
        opp = result.get("opponents", {})

        if result.get("error"):
            rows.append({
                "rank": 0,
                "model_name": result["model_name"],
                "onnx_path": result["onnx_path"],
                "score": result["score"],
                "wr_random": 0.0,
                "wr_end_turn": 0.0,
                "wr_greedy_face": 0.0,
                "latency_ms_p50_random": 0.0,
                "invalid_actions_total": 0,
                "brain_invalid_total": 0,
                "parity_mismatches": result["parity_mismatches"],
                "error": result["error"],
            })
            continue

        total_invalid = sum(r.get("invalid_actions", 0) for r in opp.values())
        total_brain_invalid = sum(r.get("brain_invalid_actions", 0) for r in opp.values())
        lat_random = opp.get("random", {}).get("latency_ms_p50", 0.0)

        rows.append({
            "rank": 0,
            "model_name": result["model_name"],
            "onnx_path": result["onnx_path"],
            "score": result["score"],
            "wr_random": opp.get("random", {}).get("winrate", 0.0),
            "wr_end_turn": opp.get("end_turn", {}).get("winrate", 0.0),
            "wr_greedy_face": opp.get("greedy_face", {}).get("winrate", 0.0),
            "latency_ms_p50_random": lat_random,
            "invalid_actions_total": total_invalid,
            "brain_invalid_total": total_brain_invalid,
            "parity_mismatches": result["parity_mismatches"],
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    for idx, row in enumerate(rows):
        row["rank"] = idx + 1

    return {
        "models": len(rows),
        "rows": rows,
        "best": rows[0] if rows else None,
    }


def save_leaderboard(result: dict, output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def _print_table(result: dict) -> None:
    rows = result.get("rows", [])
    if not rows:
        print("No models found.")
        return

    header = f"{'#':>3} {'score':>7} {'wr_rand':>7} {'wr_greed':>7} {'wr_et':>7} {'lat_ms':>7} {'inv':>4} {'b_inv':>5} {'p_mis':>5} {'model'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        err = " ERR!" if row.get("error") else ""
        print(
            f"{row['rank']:3d} "
            f"{row['score']:7.3f} "
            f"{row['wr_random']:7.3f} "
            f"{row['wr_greedy_face']:7.3f} "
            f"{row['wr_end_turn']:7.3f} "
            f"{row.get('latency_ms_p50_random', 0):7.1f} "
            f"{row.get('invalid_actions_total', 0):4d} "
            f"{row.get('brain_invalid_total', 0):5d} "
            f"{row.get('parity_mismatches', 0):5d} "
            f"{row['model_name']}{err}"
        )


def _main():
    parser = argparse.ArgumentParser(description="Build leaderboard for TrainV2 ONNX models")
    parser.add_argument("--paths", nargs="+", required=True, help="Paths to .onnx files or run directories")
    parser.add_argument("--games", type=int, default=8, help="Number of evaluation seeds per opponent")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--opponents", nargs="+", default=["random", "end_turn", "greedy_face"])
    parser.add_argument("--output", default=None, help="Path to save leaderboard JSON")
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.games))
    result = build_leaderboard(
        paths=args.paths, seeds=seeds,
        opponents=args.opponents, max_steps=args.max_steps,
    )

    _print_table(result)

    if result["best"]:
        print(f"\nBest: {result['best']['model_name']} (score={result['best']['score']:.3f})")

    if args.output:
        save_leaderboard(result, args.output)
        print(f"\nLeaderboard saved to {args.output}")


if __name__ == "__main__":
    _main()
