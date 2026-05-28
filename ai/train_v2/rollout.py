"""
Stable rollout transition schema for future PPO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import Policy

ROLLOUT_VERSION = "classic_rollout_v1"


@dataclass
class Transition:
    obs: np.ndarray
    action_features: np.ndarray
    mask: np.ndarray
    action_id: int
    reward: float
    done: bool
    truncated: bool
    value_player_id: int
    next_obs: np.ndarray
    info: dict


def collect_episode(
    env: ClassicRLEnv,
    p1_policy: Policy,
    p2_policy: Policy,
    *,
    seed: int,
    include_action_features: bool = False,
    max_steps: int = 500,
) -> dict:
    if hasattr(p1_policy, "reset"):
        p1_policy.reset(seed * 2 + 1)
    if hasattr(p2_policy, "reset"):
        p2_policy.reset(seed * 2 + 2)

    obs, _info = env.reset(seed=seed)
    transitions: list[Transition] = []
    invalid_count = 0
    steps = 0

    for _step in range(max_steps):
        cp = env.current_player_id()
        mask = env.action_mask(cp)

        if include_action_features:
            af = env.action_features(cp).copy()
        else:
            af = np.zeros((0,), dtype=np.float32)

        policy = p1_policy if cp == 1 else p2_policy
        aid = policy.select_action(env, cp)

        next_obs, reward, terminated, truncated, info = env.step(aid)

        if info.get("invalid_action"):
            invalid_count += 1
        steps += 1

        transitions.append(Transition(
            obs=obs.copy(),
            action_features=af,
            mask=mask.copy(),
            action_id=aid,
            reward=reward,
            done=terminated,
            truncated=truncated,
            value_player_id=cp,
            next_obs=next_obs.copy(),
            info=info,
        ))

        obs = next_obs

        if terminated or truncated:
            break

    st = env._env.state
    summary = {
        "winner_id": env.winner_id(),
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
    }

    return {
        "version": ROLLOUT_VERSION,
        "transitions": transitions,
        "summary": summary,
    }
