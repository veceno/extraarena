"""
BerserkInference shadow eval via ClassicRLEnv — winrate, latency, decision parity.

CLI:
    python3 -m ai.train_v2.berserk_eval --onnx <path> --opponent random --games 20
"""
from __future__ import annotations

import argparse
import json
import random as rand_mod
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ai.bot_brain import BerserkInference, _legal_fallback
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.classic_actions_v1 import build_action_mask, decode_action
from ai.train_v2.policies import (
    Policy,
    RandomLegalPolicy,
    EndTurnPolicy,
    GreedyFacePolicy,
)

OPPONENT_REGISTRY: Dict[str, type] = {
    "random": RandomLegalPolicy,
    "end_turn": EndTurnPolicy,
    "greedy_face": GreedyFacePolicy,
}


# ============================================================================
# POLICY WRAPPER
# ============================================================================

class BerserkBrainPolicy(Policy):
    def __init__(self, brain: BerserkInference, *, difficulty: str):
        self._brain = brain
        self._difficulty = difficulty
        self.name = f"berserk_{difficulty}"
        self.latencies_ms: list[float] = []
        self.invalid_actions: int = 0

    def reset(self, seed: int):
        rand_mod.seed(seed)
        np.random.seed(seed)
        self.latencies_ms.clear()
        self.invalid_actions = 0

    def select_action(self, env: ClassicRLEnv, player_id: int) -> int:
        state = env.clone_state()
        legal = env._env.get_legal_actions(player_id)

        t0 = time.perf_counter()
        legal_idx = self._brain.get_action(
            state, player_id, legal, difficulty=self._difficulty
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(latency_ms)

        if legal_idx < 0 or legal_idx >= len(legal):
            self.invalid_actions += 1
            return 0

        target_action = legal[legal_idx]
        profile = self._brain.sessions.get(self._difficulty, {})
        placement_mode = profile.get("placement_mode", "append_only")
        verify_mask = bool(profile.get("verify_mask", True))
        mask = build_action_mask(
            state,
            player_id,
            verify_mask=verify_mask,
            placement_mode=placement_mode,
        )

        for aid in range(601):
            if mask[aid] != 1.0:
                continue
            decoded = decode_action(state, player_id, int(aid))
            if decoded is None:
                continue
            if BerserkInference._find_matching_legal_action_index(decoded, [target_action]) == 0:
                return int(aid)

        self.invalid_actions += 1
        return 0


# ============================================================================
# FACTORY
# ============================================================================

def make_train_v2_berserk_brain(
    onnx_path: str,
    *,
    difficulty: str = "test",
    selection: str = "argmax",
    temperature: tuple[float, float] = (1.0, 1.0),
) -> BerserkInference:
    sidecar_path = Path(str(onnx_path) + ".json")
    sidecar = {}
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar_config = sidecar.get("config") if isinstance(sidecar.get("config"), dict) else {}

    profile = {
        "model_path": onnx_path,
        "format": "train_v2_classic_v1",
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "placement_mode": sidecar.get("placement_mode", "append_only"),
        "include_preview_features": sidecar.get(
            "include_preview_features",
            sidecar_config.get("include_preview_features", True),
        ),
        "verify_mask": True,
        "temperature_range": temperature,
        "selection": selection,
    }
    return BerserkInference(profiles={difficulty: profile})


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_berserk_matchup(
    p1_policy,
    p2_policy,
    *,
    seeds: list[int],
    swap_sides: bool = True,
    max_steps: int = 500,
) -> dict:
    if not seeds:
        return _empty_berserk_result()

    env = ClassicRLEnv(seed=seeds[0])
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_turns = 0
    total_steps = 0
    total_invalid = 0
    total_truncations = 0
    p1_reward_sum = 0.0
    p2_reward_sum = 0.0
    p1_invalid = 0
    p2_invalid = 0
    all_p1_latencies: list[float] = []
    all_p2_latencies: list[float] = []
    n_games = 0

    for seed in seeds:

        def _run(pa, pb, logical_p1=True):
            nonlocal p1_wins, p2_wins, draws, total_turns, total_steps
            nonlocal total_invalid, total_truncations
            nonlocal p1_reward_sum, p2_reward_sum, p1_invalid, p2_invalid
            nonlocal all_p1_latencies, all_p2_latencies, n_games

            if hasattr(pa, "reset"):
                pa.reset(seed * 2 + 1)
            if hasattr(pb, "reset"):
                pb.reset(seed * 2 + 2)

            env.reset(seed=seed)
            invalid_count = 0
            steps = 0

            for _step in range(max_steps):
                cp = env.current_player_id()
                policy = pa if cp == 1 else pb
                aid = policy.select_action(env, cp)

                _, _, terminated, truncated, info = env.step(aid)
                if info.get("invalid_action"):
                    invalid_count += 1
                steps += 1
                if terminated or truncated:
                    break

            st = env._env.state
            wid = env.winner_id()
            truncated_flag = st.turn_number > env._max_turns

            all_p1_latencies.extend(getattr(pa, "latencies_ms", []))
            all_p2_latencies.extend(getattr(pb, "latencies_ms", []))

            if logical_p1:
                if wid == 1:
                    p1_wins += 1
                elif wid == 2:
                    p2_wins += 1
                else:
                    draws += 1
                p1_reward_sum += float(env._p1_reward)
                p2_reward_sum += float(env._p2_reward)
                p1_invalid += getattr(pa, "invalid_actions", 0)
                p2_invalid += getattr(pb, "invalid_actions", 0)
            else:
                if wid == 1:
                    p2_wins += 1
                elif wid == 2:
                    p1_wins += 1
                else:
                    draws += 1
                p1_reward_sum += float(env._p2_reward)
                p2_reward_sum += float(env._p1_reward)
                p1_invalid += getattr(pb, "invalid_actions", 0)
                p2_invalid += getattr(pa, "invalid_actions", 0)

            total_turns += st.turn_number
            total_steps += steps
            total_invalid += invalid_count
            if truncated_flag:
                total_truncations += 1
            n_games += 1

        _run(p1_policy, p2_policy, logical_p1=True)
        if swap_sides:
            _run(p2_policy, p1_policy, logical_p1=False)

    return {
        "games": n_games,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_winrate": p1_wins / n_games if n_games > 0 else 0.0,
        "avg_turns": total_turns / n_games if n_games > 0 else 0.0,
        "avg_steps": total_steps / n_games if n_games > 0 else 0.0,
        "avg_p1_reward": p1_reward_sum / n_games if n_games > 0 else 0.0,
        "avg_p2_reward": p2_reward_sum / n_games if n_games > 0 else 0.0,
        "invalid_actions": total_invalid,
        "truncations": total_truncations,
        "p1_latency_ms_p50": _percentile(all_p1_latencies, 50),
        "p1_latency_ms_p95": _percentile(all_p1_latencies, 95),
        "p2_latency_ms_p50": _percentile(all_p2_latencies, 50),
        "p2_latency_ms_p95": _percentile(all_p2_latencies, 95),
        "p1_brain_invalid_actions": p1_invalid,
        "p2_brain_invalid_actions": p2_invalid,
        "seeds": len(seeds),
    }


def _empty_berserk_result() -> dict:
    return {
        "games": 0, "p1_wins": 0, "p2_wins": 0, "draws": 0,
        "p1_winrate": 0.0, "avg_turns": 0.0, "avg_steps": 0.0,
        "avg_p1_reward": 0.0, "avg_p2_reward": 0.0,
        "invalid_actions": 0, "truncations": 0,
        "p1_latency_ms_p50": 0.0, "p1_latency_ms_p95": 0.0,
        "p2_latency_ms_p50": 0.0, "p2_latency_ms_p95": 0.0,
        "p1_brain_invalid_actions": 0, "p2_brain_invalid_actions": 0,
        "seeds": 0,
    }


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


# ============================================================================
# DECISION PARITY
# ============================================================================

def compare_berserk_to_onnx_policy(
    onnx_path: str,
    *,
    seed: int,
    steps: int = 20,
    selection: str = "argmax",
) -> dict:
    brain = make_train_v2_berserk_brain(onnx_path, selection=selection)
    berserk_pol = BerserkBrainPolicy(brain, difficulty="test")

    from ai.train_v2.onnx_policy import OnnxActionPolicy
    onnx_mode = "sample" if selection == "softmax" else "argmax"
    onnx_pol = OnnxActionPolicy(onnx_path, mode=onnx_mode)

    env = ClassicRLEnv(seed=seed)
    env.reset(seed=seed)

    checked = 0
    matches = 0
    mismatches = 0

    for _step in range(steps):
        cp = env.current_player_id()
        berserk_aid = berserk_pol.select_action(env, cp)
        onnx_aid = onnx_pol.select_action(env, cp)

        if berserk_aid == onnx_aid:
            matches += 1
        else:
            mismatches += 1
        checked += 1

        _, _, terminated, truncated, _ = env.step(onnx_aid)
        if terminated or truncated:
            break

    return {
        "checked": checked,
        "matches": matches,
        "mismatches": mismatches,
    }


# ============================================================================
# FEATURE MODE BENCHMARK
# ============================================================================

def benchmark_feature_modes(
    onnx_path: str,
    *,
    seed: int = 42,
    steps: int = 50,
) -> dict:
    brain = make_train_v2_berserk_brain(onnx_path, selection="argmax")
    env = ClassicRLEnv(seed=seed)
    env.reset(seed=seed)

    full_lats: list[float] = []
    fast_lats: list[float] = []
    brain_lats: list[float] = []

    for i in range(steps):
        cp = env.current_player_id()
        st = env.clone_state()
        legal = env._env.get_legal_actions(cp)

        t0 = time.perf_counter()
        env.action_features(cp, include_preview=True)
        full_lats.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        env.action_features(cp, include_preview=False)
        fast_lats.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        brain.get_action(st, cp, legal, difficulty="test")
        brain_lats.append((time.perf_counter() - t0) * 1000.0)

        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            env.reset(seed=seed + i + 1)

    fast_med = _percentile(fast_lats, 50)
    full_med = _percentile(full_lats, 50)

    return {
        "steps": steps,
        "full_features_ms_p50": full_med,
        "full_features_ms_p95": _percentile(full_lats, 95),
        "fast_features_ms_p50": fast_med,
        "fast_features_ms_p95": _percentile(fast_lats, 95),
        "brain_get_action_ms_p50": _percentile(brain_lats, 50),
        "brain_get_action_ms_p95": _percentile(brain_lats, 95),
        "fast_vs_full_speedup": full_med / fast_med if fast_med > 0 else 0.0,
    }


# ============================================================================
# CLI
# ============================================================================

def _main():
    parser = argparse.ArgumentParser(description="Shadow eval of BerserkInference via ClassicRLEnv")
    parser.add_argument("--onnx", required=True, help="Path to TrainV2 .onnx model")
    parser.add_argument("--opponent", default="random", choices=["random", "end_turn", "greedy_face"])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--selection", default="argmax", choices=["argmax", "softmax"])
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--parity", action="store_true", help="Run decision parity check first")
    args = parser.parse_args()

    if args.parity:
        parity = compare_berserk_to_onnx_policy(
            args.onnx, seed=args.seed, steps=min(args.max_steps, 40), selection=args.selection
        )
        print(f"Parity: {parity['checked']} checked, {parity['matches']} matches, {parity['mismatches']} mismatches")
        if args.selection == "argmax" and parity["mismatches"] != 0:
            print("WARNING: argmax parity expected 0 mismatches!")

    brain = make_train_v2_berserk_brain(args.onnx, selection=args.selection)
    berserk_pol = BerserkBrainPolicy(brain, difficulty="test")
    opp_cls = OPPONENT_REGISTRY[args.opponent]
    opp_pol = opp_cls()

    seeds = list(range(args.seed, args.seed + args.games))
    result = evaluate_berserk_matchup(
        berserk_pol, opp_pol, seeds=seeds, swap_sides=not args.no_swap, max_steps=args.max_steps
    )

    print(f"\nEval: {berserk_pol.name} vs {opp_pol.name}")
    print(f"Games: {result['games']} (seeds={result['seeds']}, swap_sides={not args.no_swap})")
    print(f"P1 wins: {result['p1_wins']}  P2 wins: {result['p2_wins']}  Draws: {result['draws']}")
    print(f"P1 winrate: {result['p1_winrate']:.3f}")
    print(f"Avg turns: {result['avg_turns']:.1f}  Avg steps: {result['avg_steps']:.1f}")
    print(f"Avg P1 reward: {result['avg_p1_reward']:.3f}  Avg P2 reward: {result['avg_p2_reward']:.3f}")
    print(f"Invalid actions: {result['invalid_actions']}  Truncations: {result['truncations']}")
    print(f"P1 brain_invalid: {result['p1_brain_invalid_actions']}  P2 brain_invalid: {result['p2_brain_invalid_actions']}")
    print(f"P1 latency: p50={result['p1_latency_ms_p50']:.1f}ms  p95={result['p1_latency_ms_p95']:.1f}ms")
    print(f"P2 latency: p50={result['p2_latency_ms_p50']:.1f}ms  p95={result['p2_latency_ms_p95']:.1f}ms")


if __name__ == "__main__":
    _main()
