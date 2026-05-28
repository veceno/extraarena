from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import GreedyFacePolicy


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _select_profile_action(env, player_id, legal_ids, action_mode, rng, greedy_policy) -> int:
    if action_mode == "first":
        return int(legal_ids[0]) if len(legal_ids) else 0
    if action_mode == "random":
        return int(rng.choice(legal_ids)) if len(legal_ids) else 0
    if action_mode == "greedy_face":
        return greedy_policy.select_action(env, player_id)
    return 0


def profile_action_pipeline(
    *,
    episodes: int = 5,
    seed: int = 42,
    steps_per_episode: int = 50,
    action_mode: str = "first",
    warmup_steps: int = 0,
    placement_mode: str = "full",
    verify_mask: bool = True,
    use_cache: bool = False,
) -> dict:
    if action_mode not in ("first", "random", "greedy_face"):
        raise ValueError(f"Unknown action_mode: {action_mode}")

    env = ClassicRLEnv(seed=seed, verify_mask=verify_mask, placement_mode=placement_mode)

    mask_ms: list[float] = []
    full_ms: list[float] = []
    fast_ms: list[float] = []
    legal_counts: list[int] = []

    rng = np.random.default_rng(seed)
    greedy_policy = GreedyFacePolicy() if action_mode == "greedy_face" else None

    for ep in range(episodes):
        env.reset(seed=seed + ep)

        episode_done = False
        for _ in range(warmup_steps):
            cp = env.current_player_id()
            mask = env.action_mask(cp)
            legal_ids = np.flatnonzero(mask == 1.0)
            aid = _select_profile_action(env, cp, legal_ids, action_mode, rng, greedy_policy)
            _, _, terminated, truncated, _ = env.step(aid)
            if terminated or truncated:
                episode_done = True
                break

        if episode_done:
            continue

        for _ in range(steps_per_episode):
            cp = env.current_player_id()

            if not use_cache and env._cache is not None:
                env._cache.invalidate()

            t0 = time.perf_counter()
            mask = env.action_mask(cp)
            mask_ms.append((time.perf_counter() - t0) * 1000.0)

            legal_ids = np.flatnonzero(mask == 1.0)
            legal_counts.append(len(legal_ids))

            t0 = time.perf_counter()
            env.action_features(cp, include_preview=True)
            full_ms.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            env.action_features(cp, include_preview=False)
            fast_ms.append((time.perf_counter() - t0) * 1000.0)

            aid = _select_profile_action(env, cp, legal_ids, action_mode, rng, greedy_policy)
            _, _, terminated, truncated, _ = env.step(aid)
            if terminated or truncated:
                break

    full_p50 = _percentile(full_ms, 50)
    fast_p50 = _percentile(fast_ms, 50)

    return {
        "episodes": episodes,
        "steps": len(mask_ms),
        "avg_legal_actions": float(np.mean(legal_counts)) if legal_counts else 0.0,
        "mask_ms_p50": _percentile(mask_ms, 50),
        "mask_ms_p95": _percentile(mask_ms, 95),
        "features_full_ms_p50": full_p50,
        "features_full_ms_p95": _percentile(full_ms, 95),
        "features_fast_ms_p50": fast_p50,
        "features_fast_ms_p95": _percentile(fast_ms, 95),
        "preview_overhead_ms_p50": full_p50 - fast_p50,
        "fast_speedup": (full_p50 / fast_p50) if fast_p50 > 0 else 0.0,
        "action_mode": action_mode,
        "warmup_steps": warmup_steps,
        "placement_mode": placement_mode,
        "verify_mask": verify_mask,
        "use_cache": use_cache,
    }


def _apply_stable_defaults(
    args,
    *,
    stable_episodes: int = 10,
    stable_steps: int = 100,
    stable_warmup: int = 10,
):
    args.episodes = max(args.episodes, stable_episodes)
    args.steps = max(args.steps, stable_steps)
    args.warmup_steps = max(args.warmup_steps, stable_warmup)
    return args


def _main():
    parser = argparse.ArgumentParser(description="Profile TrainV2 action mask/features")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-mode",
        default="first",
        choices=["first", "random", "greedy_face"],
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stable", action="store_true")
    parser.add_argument("--placement-mode", default="full", choices=["append_only", "full"])
    parser.add_argument("--verify-mask", default=None, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    if args.stable:
        _apply_stable_defaults(args)

    verify_mask = True if args.verify_mask is None else args.verify_mask

    result = profile_action_pipeline(
        episodes=args.episodes,
        seed=args.seed,
        steps_per_episode=args.steps,
        action_mode=args.action_mode,
        warmup_steps=args.warmup_steps,
        placement_mode=args.placement_mode,
        verify_mask=verify_mask,
        use_cache=args.use_cache,
    )

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Mode: {result['action_mode']}")
        print(f"Steps: {result['steps']} | avg legal actions: {result['avg_legal_actions']:.1f}")
        print(f"mask p50/p95: {result['mask_ms_p50']:.2f} / {result['mask_ms_p95']:.2f} ms")
        print(
            f"features full p50/p95: "
            f"{result['features_full_ms_p50']:.2f} / {result['features_full_ms_p95']:.2f} ms"
        )
        print(
            f"features fast p50/p95: "
            f"{result['features_fast_ms_p50']:.2f} / {result['features_fast_ms_p95']:.2f} ms"
        )
        print(
            f"preview overhead p50: {result['preview_overhead_ms_p50']:.2f} ms "
            f"| speedup: {result['fast_speedup']:.2f}x"
        )


if __name__ == "__main__":
    _main()
