"""
PPO checkpoint evaluation against baseline policies.

CLI:
    python3 -m ai.train_v2.ppo_eval --checkpoint <path> --opponent random --games 20 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random as rand_mod
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import numpy as np

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.model_mlx import (
    ActionConditionedPolicy,
    load_checkpoint,
    policy_argmax,
    sample_action,
    MODEL_VERSION,
)
from ai.train_v2.policies import (
    Policy,
    RandomLegalPolicy,
    EndTurnPolicy,
    GreedyFacePolicy,
)


class MlxPolicy(Policy):
    def __init__(self, model, *, mode="argmax", seed=0, name_override=None):
        self._model = model
        self._mode = mode
        self._seed = seed
        self._invalid_fallbacks = 0
        if name_override is not None:
            self.name = name_override
        elif mode == "sample":
            self.name = "mlx_sample"
        else:
            self.name = "mlx_argmax"

    def reset(self, seed: int):
        self._invalid_fallbacks = 0
        if self._mode == "sample":
            mx.random.seed(seed + self._seed)

    def select_action(self, env, player_id: int) -> int:
        obs = env.observe(player_id)
        mask = env.action_mask(player_id)
        af = env.action_features(player_id)

        obs_mx = mx.array(obs[None, :])
        af_mx = mx.array(af[None, :, :])

        logits, _ = self._model(obs_mx, af_mx)
        mx.eval(logits)

        if self._mode == "sample":
            aid, _ = sample_action(logits[0], mask)
        else:
            aid = policy_argmax(logits[0], mask)

        if mask[aid] != 1.0:
            self._invalid_fallbacks += 1
            legal = [int(i) for i, v in enumerate(mask) if v == 1.0]
            aid = legal[0] if legal else 0

        return aid


def load_mlx_policy(
    checkpoint_path,
    *,
    hidden_dim=None,
    action_hidden_dim=None,
    mode="argmax",
    seed=0,
) -> MlxPolicy:
    ckpt = Path(checkpoint_path)

    if hidden_dim is None or action_hidden_dim is None:
        meta = _peek_checkpoint_meta(ckpt)
        cfg = meta.get("config", {})
        hidden_dim = hidden_dim or cfg.get("hidden_dim", 256)
        action_hidden_dim = action_hidden_dim or cfg.get("action_hidden_dim", 128)

    model = ActionConditionedPolicy(
        obs_dim=1456,
        action_feature_dim=171,
        hidden_dim=hidden_dim,
        action_hidden_dim=action_hidden_dim,
    )
    mx.eval(model.parameters())

    result = load_checkpoint(str(ckpt), model)
    meta = result.get("metadata", {})

    name = f"mlx_{mode}_{ckpt.stem}"
    return MlxPolicy(model, mode=mode, seed=seed, name_override=name)


def _peek_checkpoint_meta(ckpt: Path) -> dict:
    loaded = dict(np.load(str(ckpt), allow_pickle=True))
    meta_raw = loaded.pop("__meta__", None)
    if meta_raw is not None:
        if hasattr(meta_raw, 'tobytes'):
            return json.loads(meta_raw.tobytes().decode("utf-8"))
        elif hasattr(meta_raw, 'item'):
            v = meta_raw.item()
            if hasattr(v, 'decode'):
                return json.loads(v.decode("utf-8"))
            return json.loads(str(v))
        return json.loads(str(meta_raw))
    return {}


OPPONENT_REGISTRY: Dict[str, type] = {
    "random": RandomLegalPolicy,
    "end_turn": EndTurnPolicy,
    "greedy_face": GreedyFacePolicy,
}


def evaluate_policy_matchup(
    p1_policy,
    p2_policy,
    *,
    seeds: list[int],
    swap_sides: bool = True,
    max_steps: int = 500,
) -> dict:
    if not seeds:
        return _empty_matchup_result()

    env = ClassicRLEnv(seed=seeds[0])
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_turns = 0
    total_steps = 0
    total_invalid = 0
    total_truncations = 0
    p1_fb = 0
    p2_fb = 0
    p1_reward_sum = 0.0
    p2_reward_sum = 0.0
    n_games = 0

    for seed in seeds:

        def _run(pa, pb, logical_p1=True):
            nonlocal p1_wins, p2_wins, draws, total_turns, total_steps
            nonlocal total_invalid, total_truncations, p1_fb, p2_fb
            nonlocal p1_reward_sum, p2_reward_sum, n_games

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

                _, reward, terminated, truncated, info = env.step(aid)
                if info.get("invalid_action"):
                    invalid_count += 1
                steps += 1
                if terminated or truncated:
                    break

            st = env._env.state
            wid = env.winner_id()
            truncated_flag = st.turn_number > env._max_turns

            if logical_p1:
                if wid == 1:
                    p1_wins += 1
                elif wid == 2:
                    p2_wins += 1
                else:
                    draws += 1
                p1_reward_sum += float(env._p1_reward)
                p2_reward_sum += float(env._p2_reward)
                p1_fb += getattr(pa, "_invalid_fallbacks", 0)
                p2_fb += getattr(pb, "_invalid_fallbacks", 0)
            else:
                if wid == 1:
                    p2_wins += 1
                elif wid == 2:
                    p1_wins += 1
                else:
                    draws += 1
                p1_reward_sum += float(env._p2_reward)
                p2_reward_sum += float(env._p1_reward)
                p1_fb += getattr(pb, "_invalid_fallbacks", 0)
                p2_fb += getattr(pa, "_invalid_fallbacks", 0)

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
        "p1_invalid_fallbacks": p1_fb,
        "p2_invalid_fallbacks": p2_fb,
        "seeds": len(seeds),
    }


def _empty_matchup_result():
    return {
        "games": 0, "p1_wins": 0, "p2_wins": 0, "draws": 0,
        "p1_winrate": 0.0, "avg_turns": 0.0, "avg_steps": 0.0,
        "avg_p1_reward": 0.0, "avg_p2_reward": 0.0,
        "invalid_actions": 0, "truncations": 0,
        "p1_invalid_fallbacks": 0, "p2_invalid_fallbacks": 0,
        "seeds": 0,
    }


def _main():
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint against a baseline")
    parser.add_argument("--checkpoint", required=True, help="Path to .npz checkpoint")
    parser.add_argument("--opponent", default="random", choices=["random", "end_turn", "greedy_face"])
    parser.add_argument("--games", type=int, default=20, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--no-swap", action="store_true", help="Disable side swap")
    args = parser.parse_args()

    mlx_policy = load_mlx_policy(args.checkpoint, mode=args.mode)
    opp_cls = OPPONENT_REGISTRY[args.opponent]
    opp_policy = opp_cls()

    seeds = list(range(args.seed, args.seed + args.games))
    result = evaluate_policy_matchup(
        mlx_policy, opp_policy, seeds=seeds, swap_sides=not args.no_swap
    )

    print(f"Eval: {mlx_policy.name} vs {opp_policy.name}")
    print(f"Games: {result['games']} (seeds={result['seeds']}, swap_sides={not args.no_swap})")
    print(f"P1 wins: {result['p1_wins']}  P2 wins: {result['p2_wins']}  Draws: {result['draws']}")
    print(f"P1 winrate: {result['p1_winrate']:.3f}")
    print(f"Avg turns: {result['avg_turns']:.1f}  Avg steps: {result['avg_steps']:.1f}")
    print(f"Avg P1 reward: {result['avg_p1_reward']:.3f}  Avg P2 reward: {result['avg_p2_reward']:.3f}")
    print(f"Invalid actions: {result['invalid_actions']}  Truncations: {result['truncations']}")
    print(f"P1 invalid fallbacks: {result['p1_invalid_fallbacks']}  P2: {result['p2_invalid_fallbacks']}")


if __name__ == "__main__":
    _main()
