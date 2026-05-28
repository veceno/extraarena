"""
Lightweight throughput profiler for ClassicRLEnv.
Measures reset, mask, features, policy, and step — non-overlapping.
"""
from __future__ import annotations

import argparse
import random as rand_mod
import time
from typing import Dict, List

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import RandomLegalPolicy, GreedyFacePolicy, Policy

_PLAY_BASE = 1
_PLAY_STRIDE = 8 * 17
_ATTACK_BASE = _PLAY_BASE + 4 * _PLAY_STRIDE


def _select_random_from_legal(legal_ids: list[int], rng: rand_mod.Random) -> int:
    if not legal_ids:
        return 0
    return rng.choice(legal_ids)


def _select_greedy_face_from_legal(legal_ids: list[int]) -> int:
    if not legal_ids:
        return 0

    for aid in legal_ids:
        if aid >= _ATTACK_BASE:
            _, tcode = divmod(aid - _ATTACK_BASE, 8)
            if tcode == 7:
                return aid

    for aid in legal_ids:
        if 1 <= aid <= 544:
            _, tcode = divmod(aid - _PLAY_BASE, 17)
            if tcode == 0:
                return aid

    for aid in legal_ids:
        if 1 <= aid <= 544:
            return aid

    for aid in legal_ids:
        if aid >= _ATTACK_BASE:
            return aid

    return 0 if 0 in legal_ids else legal_ids[0]


def benchmark_env(
    *,
    episodes: int = 100,
    seed: int = 42,
    policy: str = "random",
    include_action_features: bool = False,
) -> dict:
    env = ClassicRLEnv(seed=seed)
    select_rng = rand_mod.Random(seed)
    use_greedy = policy == "greedy_face"

    total_steps = 0
    total_turns = 0
    reset_sec = 0.0
    mask_sec = 0.0
    features_sec = 0.0
    policy_sec = 0.0
    step_sec = 0.0

    t_start = time.perf_counter()

    for ep in range(episodes):
        t0 = time.perf_counter()
        obs, info = env.reset(seed=seed + ep)
        reset_sec += time.perf_counter() - t0

        for _inner in range(500):
            cp = env.current_player_id()

            t0 = time.perf_counter()
            mask = env.action_mask(cp)
            legal_ids = [int(i) for i, v in enumerate(mask) if v == 1.0]
            mask_sec += time.perf_counter() - t0

            if include_action_features:
                t0 = time.perf_counter()
                env.action_features(cp)
                features_sec += time.perf_counter() - t0

            t0 = time.perf_counter()
            if use_greedy:
                aid = _select_greedy_face_from_legal(legal_ids)
            else:
                aid = _select_random_from_legal(legal_ids, select_rng)
            policy_sec += time.perf_counter() - t0

            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(aid)
            step_sec += time.perf_counter() - t0
            total_steps += 1

            if terminated or truncated:
                total_turns += info["turn_number"]
                break

    elapsed = time.perf_counter() - t_start
    accounted = reset_sec + mask_sec + features_sec + policy_sec + step_sec
    overhead_sec = max(0.0, elapsed - accounted)

    return {
        "episodes": episodes,
        "steps": total_steps,
        "turns": total_turns,
        "seconds": elapsed,
        "steps_per_sec": total_steps / elapsed if elapsed > 0 else 0.0,
        "episodes_per_sec": episodes / elapsed if elapsed > 0 else 0.0,
        "avg_steps_per_episode": total_steps / episodes if episodes > 0 else 0.0,
        "avg_turns_per_episode": total_turns / episodes if episodes > 0 else 0.0,
        "reset_seconds": reset_sec,
        "mask_seconds": mask_sec,
        "features_seconds": features_sec,
        "policy_seconds": policy_sec,
        "step_seconds": step_sec,
        "overhead_seconds": overhead_sec,
        "include_action_features": include_action_features,
    }


def _main():
    parser = argparse.ArgumentParser(description="Profile TrainV2 environment throughput")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--policy", default="random", choices=["random", "greedy_face"], help="Policy to use")
    parser.add_argument("--include-action-features", action="store_true", help="Also measure action_features timing")
    args = parser.parse_args()

    result = benchmark_env(
        episodes=args.episodes,
        seed=args.seed,
        policy=args.policy,
        include_action_features=args.include_action_features,
    )

    total = result["seconds"]

    print(f"Profile: {result['episodes']} episodes, policy={args.policy}, seed={args.seed}")
    print(f"Steps: {result['steps']}  Turns: {result['turns']}")
    print(f"Elapsed: {total:.3f}s")
    print(f"Throughput: {result['steps_per_sec']:.1f} steps/s, {result['episodes_per_sec']:.2f} eps/s")
    print(f"Avg per episode: {result['avg_steps_per_episode']:.1f} steps, {result['avg_turns_per_episode']:.1f} turns")
    print(f"Timing breakdown:")
    print(f"  reset:        {result['reset_seconds']:.3f}s ({result['reset_seconds']/total*100:.1f}%)" if total > 0 else f"  reset:        {result['reset_seconds']:.3f}s")
    print(f"  mask/legal:   {result['mask_seconds']:.3f}s ({result['mask_seconds']/total*100:.1f}%)" if total > 0 else f"  mask/legal:   {result['mask_seconds']:.3f}s")
    if args.include_action_features:
        print(f"  action_feats: {result['features_seconds']:.3f}s ({result['features_seconds']/total*100:.1f}%)" if total > 0 else f"  action_feats: {result['features_seconds']:.3f}s")
    print(f"  policy:       {result['policy_seconds']:.3f}s ({result['policy_seconds']/total*100:.1f}%)" if total > 0 else f"  policy:       {result['policy_seconds']:.3f}s")
    print(f"  step:         {result['step_seconds']:.3f}s ({result['step_seconds']/total*100:.1f}%)" if total > 0 else f"  step:         {result['step_seconds']:.3f}s")
    print(f"  overhead:     {result['overhead_seconds']:.3f}s ({result['overhead_seconds']/total*100:.1f}%)" if total > 0 else f"  overhead:     {result['overhead_seconds']:.3f}s")


if __name__ == "__main__":
    _main()
