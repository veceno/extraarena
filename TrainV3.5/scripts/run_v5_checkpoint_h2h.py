#!/usr/bin/env python3
"""Run a side- and initiative-balanced match between two Extra-LR V5 NPZs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "TrainV3.5" / "runs"
for path in (ROOT, ROOT / "TrainV3.5" / "python", RUNS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.actions import ManaDrawAction  # noqa: E402
from run_model_benchmark_v5_current import V5NpzPolicy, _action_kind  # noqa: E402
from train_v3.contracts import AssistModeV5, InfoModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402


def _score(games: list[dict[str, Any]], name: str) -> float:
    if not games:
        return 0.0
    return (
        sum(game["winner"] == name for game in games)
        + 0.5 * sum(game["winner"] is None for game in games)
    ) / len(games)


def _summarize(games: list[dict[str, Any]], name: str) -> dict[str, Any]:
    by_seat = {f"p{seat}": [game for game in games if game["player_ids"][name] == seat] for seat in (1, 2)}
    by_start = {
        "first": [game for game in games if game["player_ids"][name] == game["starting_player_id"]],
        "second": [game for game in games if game["player_ids"][name] != game["starting_player_id"]],
    }
    wins = sum(game["winner"] == name for game in games)
    draws = sum(game["winner"] is None for game in games)
    actions = Counter()
    invalid = 0
    for game in games:
        actions.update(game["action_counts"][name])
        invalid += int(game["invalid_actions"][name])
    return {
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "losses": len(games) - wins - draws,
        "score_rate": _score(games, name),
        "first_score_rate": _score(by_start["first"], name),
        "second_score_rate": _score(by_start["second"], name),
        "p1_score_rate": _score(by_seat["p1"], name),
        "p2_score_rate": _score(by_seat["p2"], name),
        "invalid_actions": invalid,
        "mana_draw_count": int(actions["mana_draw"]),
        "action_counts": dict(actions),
    }


def _run_game(
    *, seed: int, a_name: str, b_name: str, policies: dict[str, V5NpzPolicy], a_player_id: int,
    starting_player_id: int, max_steps: int, max_turns: int,
) -> dict[str, Any]:
    b_player_id = 2 if a_player_id == 1 else 1
    player_names = {a_player_id: a_name, b_player_id: b_name}
    env = TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=seed,
            max_turns=max_turns,
            verify_mask=False,
            placement_mode="append_only",
            info_mode=InfoModeV5(),
            assist_mode=AssistModeV5(),
            history_limit=20,
        )
    )
    # Omitting decks deliberately makes ClassicRLEnv generate one seed-specific
    # default deck and clone it for both players: no checkpoint receives a deck edge.
    env.reset(p1_is_bot=True, p2_is_bot=True, starting_player_id=starting_player_id, seed=seed)
    for player_id, name in player_names.items():
        policies[name].reset(seed * 17 + player_id)
    action_counts = {a_name: Counter(), b_name: Counter()}
    invalid = {a_name: 0, b_name: 0}
    terminated = truncated = False
    steps = 0
    error = None
    try:
        while not terminated and not truncated and steps < max_steps:
            player_id = env.current_player_id()
            name = player_names[player_id]
            action = policies[name].select_action(env, player_id)
            action_counts[name][_action_kind(env, player_id, action)] += 1
            if isinstance(action, ManaDrawAction):
                _obs, _reward, terminated, truncated, info = env.step_core_action(action)
            else:
                _obs, _reward, terminated, truncated, info = env.step(int(action))
            invalid[name] += int(bool(info.get("invalid_action")))
            steps += 1
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
    winner_id = env.env.winner_id()
    return {
        "seed": seed,
        "starting_player_id": starting_player_id,
        "player_ids": {a_name: a_player_id, b_name: b_player_id},
        "winner": player_names.get(winner_id),
        "steps": steps,
        "turns": int(env.env._env.state.turn_number),
        "status": getattr(env.env._env.state.status, "value", str(env.env._env.state.status)),
        "timed_out": not terminated and not truncated and steps >= max_steps,
        "truncated": bool(truncated),
        "invalid_actions": {name: int(value) for name, value in invalid.items()},
        "action_counts": {name: dict(value) for name, value in action_counts.items()},
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128, help="Seeds; four balanced battles are played per seed.")
    parser.add_argument("--seed", type=int, default=99000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-turns", type=int, default=120)
    args = parser.parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    a_path, b_path = args.checkpoint_a.resolve(), args.checkpoint_b.resolve()
    if not a_path.is_file() or not b_path.is_file():
        raise SystemExit("both checkpoints must exist")
    a_name, b_name = a_path.stem, b_path.stem
    if a_name == b_name:
        a_name, b_name = "checkpoint_a", "checkpoint_b"
    policies = {a_name: V5NpzPolicy(a_path), b_name: V5NpzPolicy(b_path)}
    games: list[dict[str, Any]] = []
    total = args.games * 4
    for offset in range(args.games):
        seed = args.seed + offset
        for a_player_id in (1, 2):
            for starting_player_id in (1, 2):
                games.append(_run_game(
                    seed=seed, a_name=a_name, b_name=b_name, policies=policies,
                    a_player_id=a_player_id, starting_player_id=starting_player_id,
                    max_steps=args.max_steps, max_turns=args.max_turns,
                ))
        if (offset + 1) % max(1, args.games // 20) == 0 or offset + 1 == args.games:
            print(f"Progress: {(offset + 1) * 4}/{total}", flush=True)
    payload = {
        "schema": "extra_lr_v5_checkpoint_h2h_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint_a": str(a_path), "checkpoint_b": str(b_path),
        "seed": args.seed, "seed_count": args.games, "total_battles": len(games),
        "balancing": "each seed plays both policy seats and both starting players",
        "summary": {a_name: _summarize(games, a_name), b_name: _summarize(games, b_name)},
        "error_count": sum(bool(game["error"]) for game in games),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
