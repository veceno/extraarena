"""
Round-robin tournament for notable TrainV2 and legacy bot candidates.

Evaluation-only. It does not modify training or production configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai.bot_brain import BerserkInference
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.onnx_policy import OnnxActionPolicy
from ai.train_v2.policies import GreedyFacePolicy, RandomLegalPolicy
from ai.train_v2.shadow import LegacyBerserkPolicy


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "u0156": {"kind": "onnx", "path": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/update_0156.onnx", "stage": "league"},
    "u0251": {"kind": "onnx", "path": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/best_update_0251.onnx", "stage": "league_best"},
    "u0348": {"kind": "onnx", "path": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/update_0348.onnx", "stage": "league_runner"},
    "u0408": {"kind": "onnx", "path": "ai/train_v2/runs/m4_league_from_0065_20260521_171604/exported/update_0408.onnx", "stage": "league_late"},
    "u0700": {"kind": "onnx", "path": "ai/train_v2/runs/m4_hist_from_0251_20260521_205548/exported/best_update_0700.onnx", "stage": "hist_best"},
    "u0800": {"kind": "onnx", "path": "ai/train_v2/runs/m4_hist_from_0251_20260521_205548/exported/update_0800.onnx", "stage": "hist_runner"},
    "u0900": {"kind": "onnx", "path": "ai/train_v2/runs/m4_p2_target_from_0700_20260522_105646/exported/update_0900.onnx", "stage": "p2_target"},
    "u0950": {"kind": "onnx", "path": "ai/train_v2/runs/m4_p2_target_from_0700_20260522_105646/exported/update_0950.onnx", "stage": "p2_target"},
    "u0958": {"kind": "onnx", "path": "ai/train_v2/runs/m4_p2_target_from_0700_20260522_105646/exported/update_0958.onnx", "stage": "p2_target"},
    "u0966": {"kind": "onnx", "path": "ai/train_v2/runs/m4_p2_target_from_0700_20260522_105646/exported/update_0966.onnx", "stage": "p2_target"},
    "b1107": {"kind": "onnx", "path": "ai/train_v2/runs/m4_balanced_from_0950_20260522_144431/exported/update_1107.onnx", "stage": "balanced_from_0950"},
    "b1187": {"kind": "onnx", "path": "ai/train_v2/runs/m4_balanced_from_0950_20260522_144431/exported/update_1187.onnx", "stage": "balanced_from_0950"},
    "b1190": {"kind": "onnx", "path": "ai/train_v2/runs/m4_balanced_from_0950_20260522_144431/exported/update_1190.onnx", "stage": "balanced_from_0950"},
    "legacy_max": {"kind": "legacy", "profile": "legacy_max", "path": "ai/models/extra-lr-v3-max.onnx", "obs_dim": 997, "stage": "legacy"},
    "legacy_medium": {"kind": "legacy", "profile": "legacy_medium", "path": "ai/models/extra-lr-v3-medium.onnx", "obs_dim": 997, "stage": "legacy"},
    "legacy_random_biggest": {"kind": "legacy", "profile": "legacy_random_biggest", "path": "ai/models/OnlyVersusRandomBiggest.onnx", "obs_dim": 621, "stage": "legacy"},
    "greedy_face": {"kind": "greedy", "stage": "baseline"},
    "random": {"kind": "random", "stage": "baseline"},
}


def _make_policy(name: str, seed: int = 0):
    spec = MODEL_REGISTRY[name]
    kind = spec["kind"]
    if kind == "onnx":
        return OnnxActionPolicy(spec["path"], mode="argmax", seed=seed, verify_mask=False)
    if kind == "greedy":
        return GreedyFacePolicy()
    if kind == "random":
        return RandomLegalPolicy(seed=seed)
    if kind == "legacy":
        profile = spec["profile"]
        brain = BerserkInference(
            profiles={
                profile: {
                    "model_path": spec["path"],
                    "obs_dim": int(spec["obs_dim"]),
                    "temperature_range": (0.5, 0.5),
                    "selection": "argmax",
                }
            }
        )
        return LegacyBerserkPolicy(brain, difficulty=profile)
    raise ValueError(f"unknown policy kind: {kind}")


def _run_game(
    p1_name: str,
    p2_name: str,
    seed: int,
    max_steps: int,
    *,
    starting_player_id: int = 1,
) -> dict:
    p1 = _make_policy(p1_name, seed=seed * 2 + 1)
    p2 = _make_policy(p2_name, seed=seed * 2 + 2)
    if hasattr(p1, "reset"):
        p1.reset(seed * 3 + 1)
    if hasattr(p2, "reset"):
        p2.reset(seed * 3 + 2)

    env = ClassicRLEnv(seed=seed, verify_mask=False, placement_mode="append_only")
    env.reset(seed=seed, starting_player_id=starting_player_id)

    invalid = 0
    steps = 0
    for steps in range(1, max_steps + 1):
        cp = env.current_player_id()
        policy = p1 if cp == 1 else p2
        aid = policy.select_action(env, cp)
        _, _, terminated, truncated, info = env.step(aid)
        invalid += int(bool(info.get("invalid_action")))
        if terminated or truncated:
            break

    winner = env.winner_id()
    st = env._env.state
    return {
        "p1": p1_name,
        "p2": p2_name,
        "seed": seed,
        "starting_player_id": starting_player_id,
        "winner": winner,
        "winner_name": p1_name if winner == 1 else p2_name if winner == 2 else None,
        "steps": steps,
        "turns": st.turn_number,
        "p1_hp": st.p1.hero.hp,
        "p2_hp": st.p2.hero.hp,
        "invalid": invalid,
    }


def _score_for(name: str, game: dict) -> float:
    if game["winner_name"] == name:
        return 1.0
    if game["winner_name"] is None:
        return 0.5
    return 0.0


def _summarize(models: list[str], games: list[dict]) -> dict:
    table: dict[str, dict[str, Any]] = {
        name: {
            "model": name,
            "stage": MODEL_REGISTRY[name].get("stage", ""),
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "score": 0.0,
            "p1_games": 0,
            "p1_wins": 0,
            "p2_games": 0,
            "p2_wins": 0,
            "avg_hp_margin": 0.0,
            "invalid": 0,
        }
        for name in models
    }
    h2h: dict[str, dict[str, Any]] = defaultdict(lambda: {"games": 0, "a_score": 0.0, "a_wins": 0, "b_wins": 0, "draws": 0})

    for game in games:
        p1 = game["p1"]
        p2 = game["p2"]
        winner = game["winner_name"]

        for name, side in ((p1, "p1"), (p2, "p2")):
            row = table[name]
            row["games"] += 1
            row["score"] += _score_for(name, game)
            row["invalid"] += int(game["invalid"])
            if side == "p1":
                row["p1_games"] += 1
                row["p1_wins"] += int(winner == name)
                row["avg_hp_margin"] += game["p1_hp"] - game["p2_hp"]
            else:
                row["p2_games"] += 1
                row["p2_wins"] += int(winner == name)
                row["avg_hp_margin"] += game["p2_hp"] - game["p1_hp"]

            if winner == name:
                row["wins"] += 1
            elif winner is None:
                row["draws"] += 1
            else:
                row["losses"] += 1

        a, b = sorted((p1, p2))
        key = f"{a}__vs__{b}"
        h = h2h[key]
        h["a"] = a
        h["b"] = b
        h["games"] += 1
        a_score = _score_for(a, game)
        h["a_score"] += a_score
        h["a_wins"] += int(winner == a)
        h["b_wins"] += int(winner == b)
        h["draws"] += int(winner is None)

    rows = []
    for row in table.values():
        games_n = max(1, int(row["games"]))
        row["winrate"] = row["wins"] / games_n
        row["score_rate"] = row["score"] / games_n
        row["p1_winrate"] = row["p1_wins"] / row["p1_games"] if row["p1_games"] else 0.0
        row["p2_winrate"] = row["p2_wins"] / row["p2_games"] if row["p2_games"] else 0.0
        row["avg_hp_margin"] = row["avg_hp_margin"] / games_n
        rows.append(row)
    rows.sort(key=lambda r: (r["score_rate"], r["winrate"], r["avg_hp_margin"]), reverse=True)
    for idx, row in enumerate(rows, 1):
        row["rank"] = idx

    h2h_rows = []
    for h in h2h.values():
        h = dict(h)
        h["a_score_rate"] = h["a_score"] / h["games"] if h["games"] else 0.0
        h2h_rows.append(h)
    h2h_rows.sort(key=lambda r: (r["a"], r["b"]))

    p1_games = len(games)
    p1_wins = sum(1 for game in games if game["winner"] == 1)
    p2_wins = sum(1 for game in games if game["winner"] == 2)
    draws = sum(1 for game in games if game["winner"] is None)

    return {
        "rows": rows,
        "h2h": h2h_rows,
        "side_bias": {
            "games": p1_games,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "p1_winrate": p1_wins / p1_games if p1_games else 0.0,
            "p2_winrate": p2_wins / p1_games if p1_games else 0.0,
        },
    }


def _starting_players_for_mode(mode: str, seed: int) -> list[int]:
    if mode == "p1":
        return [1]
    if mode == "p2":
        return [2]
    if mode == "both":
        return [1, 2]
    if mode == "random":
        return [1 if seed % 2 == 0 else 2]
    raise ValueError(f"unknown start_mode: {mode}")


def run_tournament(models: list[str], *, seeds: list[int], max_steps: int, start_mode: str = "both") -> dict:
    games: list[dict] = []
    pairs_total = len(models) * (len(models) - 1) // 2
    pair_idx = 0
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pair_idx += 1
            print(f"[{pair_idx}/{pairs_total}] {a} vs {b}", flush=True)
            for seed in seeds:
                for starting_player_id in _starting_players_for_mode(start_mode, seed):
                    games.append(_run_game(a, b, seed, max_steps, starting_player_id=starting_player_id))
                    games.append(_run_game(b, a, seed, max_steps, starting_player_id=starting_player_id))
    summary = _summarize(models, games)
    return {
        "models": models,
        "seeds": seeds,
        "max_steps": max_steps,
        "start_mode": start_mode,
        "games": games,
        **summary,
    }


def _write_outputs(result: dict, output: str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "rank", "model", "stage", "games", "wins", "losses", "draws",
            "winrate", "score_rate", "p1_winrate", "p2_winrate", "avg_hp_margin", "invalid",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({key: row.get(key) for key in fields})

    h2h_path = out.with_name(out.stem + "_h2h.csv")
    with h2h_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["a", "b", "games", "a_score_rate", "a_wins", "b_wins", "draws"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["h2h"]:
            writer.writerow({key: row.get(key) for key in fields})


def _print_summary(result: dict, top: int = 20) -> None:
    print("\nLeaderboard")
    for row in result["rows"][:top]:
        print(
            f"#{row['rank']:02d} {row['model']:<22} "
            f"score={row['score_rate'] * 100:5.1f}% "
            f"wr={row['winrate'] * 100:5.1f}% "
            f"p1={row['p1_winrate'] * 100:5.1f}% "
            f"p2={row['p2_winrate'] * 100:5.1f}% "
            f"hp={row['avg_hp_margin']:6.2f}"
        )
    sb = result["side_bias"]
    print(
        f"\nSide bias: p1={sb['p1_winrate'] * 100:.1f}% "
        f"p2={sb['p2_winrate'] * 100:.1f}% draws={sb['draws']}/{sb['games']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run notable-bot TrainV2 tournament")
    parser.add_argument("--models", default="u0156,u0251,u0348,u0408,u0700,u0800,u0900,u0950,u0958,u0966,legacy_max,legacy_medium,legacy_random_biggest,greedy_face,random")
    parser.add_argument("--games", type=int, default=12, help="Seeds per unordered pair; both sides are played")
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--start-mode", default="both", choices=["both", "random", "p1", "p2"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [name for name in models if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    missing = [
        (name, spec["path"])
        for name, spec in MODEL_REGISTRY.items()
        if name in models and spec.get("path") and not Path(spec["path"]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing model files: {missing}")

    seeds = list(range(args.seed, args.seed + args.games))
    result = run_tournament(models, seeds=seeds, max_steps=args.max_steps, start_mode=args.start_mode)
    _write_outputs(result, args.output)
    _print_summary(result)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
