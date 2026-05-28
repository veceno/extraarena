"""
Rollout and evaluation harness for RL policies.
"""
from __future__ import annotations

import argparse
import random as rand_mod
from typing import Dict, List

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import (
    Policy,
    RandomLegalPolicy,
    EndTurnPolicy,
    GreedyFacePolicy,
)

_POLICY_REGISTRY: Dict[str, type] = {
    "random": RandomLegalPolicy,
    "end_turn": EndTurnPolicy,
    "greedy_face": GreedyFacePolicy,
}


def _resolve_policy(name: str) -> Policy:
    cls = _POLICY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown policy: {name}. Available: {list(_POLICY_REGISTRY)}")
    return cls()


def play_episode(
    p1_policy: Policy,
    p2_policy: Policy,
    *,
    seed: int,
    env: ClassicRLEnv | None = None,
    max_steps: int = 500,
) -> dict:
    if env is None:
        env = ClassicRLEnv()

    rand_mod.seed(seed)
    obs, info = env.reset(seed=seed)

    p1_seed = seed * 2 + 1
    p2_seed = seed * 2 + 2
    if hasattr(p1_policy, 'reset'):
        p1_policy.reset(p1_seed)
    if hasattr(p2_policy, 'reset'):
        p2_policy.reset(p2_seed)

    p1_hp_init = info["p1_hp"]
    p2_hp_init = info["p2_hp"]
    invalid_count = 0
    steps = 0

    for _step in range(max_steps):
        cp = env.current_player_id()
        policy = p1_policy if cp == 1 else p2_policy
        aid = policy.select_action(env, cp)
        obs, reward, terminated, truncated, info = env.step(aid)
        if info.get("invalid_action"):
            invalid_count += 1
        steps += 1
        if terminated or truncated:
            break

    st = env._env.state
    winner = env.winner_id()

    return {
        "winner_id": winner,
        "status": st.status.value,
        "turns": st.turn_number,
        "steps": steps,
        "truncated": st.turn_number > env._max_turns,
        "p1_hp": st.p1.hero.hp,
        "p2_hp": st.p2.hero.hp,
        "p1_reward": float(env._p1_reward),
        "p2_reward": float(env._p2_reward),
        "invalid_actions": invalid_count,
        "seed": seed,
        "p1_policy": p1_policy.name,
        "p2_policy": p2_policy.name,
    }


def evaluate_matchup(
    p1_policy: Policy,
    p2_policy: Policy,
    *,
    seeds: list[int],
) -> dict:
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_turns = 0
    total_steps = 0
    total_invalid = 0

    env = ClassicRLEnv()
    for seed in seeds:
        result = play_episode(p1_policy, p2_policy, seed=seed, env=env)
        wid = result["winner_id"]
        if wid == 1:
            p1_wins += 1
        elif wid == 2:
            p2_wins += 1
        else:
            draws += 1
        total_turns += result["turns"]
        total_steps += result["steps"]
        total_invalid += result["invalid_actions"]

    n = len(seeds)
    return {
        "games": n,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_winrate": p1_wins / n if n > 0 else 0.0,
        "avg_turns": total_turns / n if n > 0 else 0.0,
        "avg_steps": total_steps / n if n > 0 else 0.0,
        "invalid_actions": total_invalid,
    }


def _main():
    parser = argparse.ArgumentParser(description="Evaluate RL policies")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--p1", default="greedy_face", help="P1 policy name")
    parser.add_argument("--p2", default="random", help="P2 policy name")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    p1 = _resolve_policy(args.p1)
    p2 = _resolve_policy(args.p2)

    seeds = list(range(args.seed, args.seed + args.games))
    result = evaluate_matchup(p1, p2, seeds=seeds)

    print(f"Matchup: {p1.name} (P1) vs {p2.name} (P2)")
    print(f"Games: {result['games']}")
    print(f"P1 wins: {result['p1_wins']}  P2 wins: {result['p2_wins']}  Draws: {result['draws']}")
    print(f"P1 winrate: {result['p1_winrate']:.3f}")
    print(f"Avg turns: {result['avg_turns']:.1f}  Avg steps: {result['avg_steps']:.1f}")
    print(f"Invalid actions: {result['invalid_actions']}")


if __name__ == "__main__":
    _main()
