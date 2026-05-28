"""
Benchmark CLI for TrainV2 rollout performance across different worker counts.

Usage:
    python3 -m ai.train_v2.benchmark_rollout --preset smoke --workers 1,2,4,8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ai.train_v2.train_ppo import (
    PPOConfig,
    train,
    make_config_from_preset,
    TRAIN_PRESETS,
    estimate_update_memory,
)


def benchmark_rollout(
    *,
    preset: str = "smoke",
    workers_list: list[int],
    episodes_per_update: int = 8,
    max_steps: int = 100,
    updates: int = 1,
    verify_mask: bool = False,
    placement_mode: str = "append_only",
    include_preview: bool = False,
    action_features_dtype: str = "float32",
    seed: int = 42,
) -> dict:
    results: list[dict] = []

    for workers in workers_list:
        config = make_config_from_preset(preset) if preset in TRAIN_PRESETS else PPOConfig()
        config.total_updates = updates
        config.episodes_per_update = episodes_per_update
        config.max_steps_per_episode = max_steps
        config.verify_mask = verify_mask
        config.placement_mode = placement_mode
        config.include_preview_features = include_preview
        config.action_features_dtype = action_features_dtype
        config.seed = seed
        config.rollout_workers = workers
        config.checkpoint_dir = f"/tmp/_benchmark_ckpts_{workers}"
        config.eval_every_updates = 0

        mem = estimate_update_memory(config)

        t0 = time.perf_counter()
        try:
            train_result = train(config)
        except Exception as exc:
            results.append({
                "workers": workers,
                "error": str(exc),
                "transitions": 0,
                "rollout_time": 0.0,
                "update_time": 0.0,
                "total_time": 0.0,
                "transitions_per_sec": 0.0,
                "episodes": 0,
                "steps": 0,
                "action_features_dtype": action_features_dtype,
                "memory_estimate_mb": mem["total_mb"],
            })
            continue
        total_time = time.perf_counter() - t0

        transitions = train_result["steps"]
        results.append({
            "workers": workers,
            "transitions": transitions,
            "rollout_time": train_result.get("rollout_time", 0.0),
            "update_time": train_result.get("update_time", 0.0),
            "total_time": total_time,
            "transitions_per_sec": transitions / total_time if total_time > 0 else 0.0,
            "episodes": train_result["episodes"],
            "steps": train_result["steps"],
            "last_loss": train_result["last_loss"],
            "action_features_dtype": action_features_dtype,
            "memory_estimate_mb": mem["total_mb"],
        })

    return {
        "benchmark": "rollout",
        "preset": preset,
        "episodes_per_update": episodes_per_update,
        "max_steps": max_steps,
        "updates": updates,
        "verify_mask": verify_mask,
        "placement_mode": placement_mode,
        "include_preview_features": include_preview,
        "action_features_dtype": action_features_dtype,
        "seed": seed,
        "results": results,
    }


def _main():
    parser = argparse.ArgumentParser(description="Benchmark TrainV2 rollout performance")
    parser.add_argument("--preset", default="smoke", choices=list(TRAIN_PRESETS))
    parser.add_argument("--workers", default="1,2,4,8", help="Comma-separated worker counts")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--verify-mask", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--placement-mode", default="append_only", choices=["append_only", "full"])
    parser.add_argument("--action-features-dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    workers_list = [int(w.strip()) for w in args.workers.split(",") if w.strip()]

    result = benchmark_rollout(
        preset=args.preset,
        workers_list=workers_list,
        episodes_per_update=args.episodes,
        max_steps=args.max_steps,
        updates=args.updates,
        verify_mask=args.verify_mask,
        placement_mode=args.placement_mode,
        action_features_dtype=args.action_features_dtype,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Benchmark: {result['benchmark']}")
        print(f"Config: preset={result['preset']} eps={result['episodes_per_update']} steps={result['max_steps']} updates={result['updates']}")
        print(f"verify_mask={result['verify_mask']} placement={result['placement_mode']} dtype={result['action_features_dtype']}")
        print()
        print(f"{'Workers':>8} {'Transitions':>12} {'Rollout(s)':>10} {'Update(s)':>10} {'Time(s)':>10} {'Trans/sec':>12} {'Memory(MB)':>12}")
        for r in result["results"]:
            if "error" in r:
                print(f"{r['workers']:>8} ERROR: {r['error']}")
            else:
                print(
                    f"{r['workers']:>8} {r['transitions']:>12} {r.get('rollout_time', 0.0):>10.2f} "
                    f"{r.get('update_time', 0.0):>10.2f} {r['total_time']:>10.2f} "
                    f"{r['transitions_per_sec']:>12.1f} {r['memory_estimate_mb']:>12.1f}"
                )


if __name__ == "__main__":
    _main()
