"""Reward component helpers for TrainV3/V5 environment experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import InfoModeV5


@dataclass(frozen=True)
class V5RewardSnapshot:
    my_hero_hp: int
    my_hero_max_hp: int
    enemy_hero_hp: int
    enemy_hero_max_hp: int
    my_board_count: int
    enemy_board_count: int
    my_board_power: float
    enemy_board_power: float


@dataclass(frozen=True)
class V5RewardWeights:
    hp_potential_delta: float = 0.08
    board_power_delta: float = 0.015
    board_power_delta_normalizer: float = 100.0
    board_under_0_7_penalty: float = 0.008
    own_board_wiped_penalty: float = 0.015
    informed_penalty_multiplier: float = 0.35
    draw_assist_penalty_multiplier: float = 0.20
    max_shaping_abs: float = 0.06


def _board_power(board) -> float:
    return float(sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in board))


def reward_snapshot_v5(state, player_id: int) -> V5RewardSnapshot:
    me = state.p1 if state.p1.user_id == player_id else state.p2
    enemy = state.p2 if state.p1.user_id == player_id else state.p1
    return V5RewardSnapshot(
        my_hero_hp=int(me.hero.hp),
        my_hero_max_hp=max(1, int(me.hero.max_hp)),
        enemy_hero_hp=int(enemy.hero.hp),
        enemy_hero_max_hp=max(1, int(enemy.hero.max_hp)),
        my_board_count=len(me.board),
        enemy_board_count=len(enemy.board),
        my_board_power=_board_power(me.board),
        enemy_board_power=_board_power(enemy.board),
    )


def compute_reward_components_v5(pre: V5RewardSnapshot, post: V5RewardSnapshot) -> dict[str, Any]:
    """Return V5 reward-shaping components without applying weights.

    Components are deltas/potentials so training code can tune coefficients later
    without changing the environment trace contract.
    """

    def missing_ratio(hp: int, max_hp: int) -> float:
        return max(0.0, min(1.0, (max_hp - hp) / max(max_hp, 1)))

    pre_hp_potential = (
        missing_ratio(pre.enemy_hero_hp, pre.enemy_hero_max_hp)
        - missing_ratio(pre.my_hero_hp, pre.my_hero_max_hp)
    )
    post_hp_potential = (
        missing_ratio(post.enemy_hero_hp, post.enemy_hero_max_hp)
        - missing_ratio(post.my_hero_hp, post.my_hero_max_hp)
    )

    pre_board_delta = pre.my_board_power - pre.enemy_board_power
    post_board_delta = post.my_board_power - post.enemy_board_power
    ratio = post.my_board_power / max(post.enemy_board_power, 1.0)

    return {
        "hp_potential_delta": float(post_hp_potential - pre_hp_potential),
        "board_power_delta": float(post_board_delta - pre_board_delta),
        "my_board_power": float(post.my_board_power),
        "enemy_board_power": float(post.enemy_board_power),
        "board_power_ratio": float(ratio),
        "board_under_0_7": bool(post.enemy_board_power > 0 and ratio < 0.7),
        "own_board_wiped": bool(pre.my_board_count > 0 and post.my_board_count == 0),
        "my_board_count_delta": int(post.my_board_count - pre.my_board_count),
        "enemy_board_count_delta": int(post.enemy_board_count - pre.enemy_board_count),
    }


def compute_history_outcome_deltas_v5(pre: V5RewardSnapshot, post: V5RewardSnapshot) -> dict[str, Any]:
    """Stable per-action outcome features for the V5 history tape.

    These are game-state deltas, not trainer reward values, so changing reward
    weights later does not leak a moving training target into the model input.
    """

    pre_board_delta = pre.my_board_power - pre.enemy_board_power
    post_board_delta = post.my_board_power - post.enemy_board_power
    return {
        "enemy_hero_hp_delta": int(pre.enemy_hero_hp - post.enemy_hero_hp),
        "own_hero_hp_delta": int(pre.my_hero_hp - post.my_hero_hp),
        "my_board_count_delta": int(post.my_board_count - pre.my_board_count),
        "enemy_board_count_delta": int(post.enemy_board_count - pre.enemy_board_count),
        "board_power_delta": float(post_board_delta - pre_board_delta),
    }


def compute_weighted_reward_v5(
    base_reward: float,
    components: dict[str, Any],
    *,
    info_mode: InfoModeV5,
    weights: V5RewardWeights | None = None,
) -> float:
    weights = weights or V5RewardWeights()
    informed_multiplier = 1.0
    if info_mode.enemy_hand_known or info_mode.enemy_deck_known or info_mode.enemy_deck_order_known:
        informed_multiplier += weights.informed_penalty_multiplier
    if info_mode.draw_assist_enabled:
        informed_multiplier += weights.draw_assist_penalty_multiplier * info_mode.clipped_draw_assist_strength()

    shaping = 0.0
    shaping += weights.hp_potential_delta * float(components.get("hp_potential_delta", 0.0) or 0.0)
    shaping += (
        weights.board_power_delta
        * float(components.get("board_power_delta", 0.0) or 0.0)
        / max(float(weights.board_power_delta_normalizer), 1.0)
    )
    if components.get("board_under_0_7"):
        shaping -= weights.board_under_0_7_penalty * informed_multiplier
    if components.get("own_board_wiped"):
        shaping -= weights.own_board_wiped_penalty * informed_multiplier
    shaping = max(-weights.max_shaping_abs, min(weights.max_shaping_abs, shaping))
    return float(np.float32(float(base_reward) + shaping))


__all__ = [
    "V5RewardSnapshot",
    "V5RewardWeights",
    "compute_history_outcome_deltas_v5",
    "compute_reward_components_v5",
    "compute_weighted_reward_v5",
    "reward_snapshot_v5",
]
