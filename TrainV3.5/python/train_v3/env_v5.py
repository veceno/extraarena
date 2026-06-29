"""TrainV3 V5 environment wrapper around the Python TrainV2 oracle env."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai.train_v2.classic_actions_v1 import decode_action, _get_me_enemy
from ai.train_v2.classic_rl_env import ClassicRLEnv
from core.actions import AttackAction, PlayCardAction

from .contracts import AssistModeV5, InfoModeV5
from .obs_v5 import encode_observation_v5
from .reward_v5 import (
    V5RewardWeights,
    compute_history_outcome_deltas_v5,
    compute_reward_components_v5,
    compute_weighted_reward_v5,
    reward_snapshot_v5,
)


@dataclass
class TrainV3EnvConfig:
    seed: int = 42
    verify_mask: bool = False
    placement_mode: str = "append_only"
    include_legal_actions_in_info: bool = False
    info_mode: InfoModeV5 = field(default_factory=InfoModeV5)
    assist_mode: AssistModeV5 = field(default_factory=AssistModeV5)
    reward_weights: V5RewardWeights = field(default_factory=V5RewardWeights)
    history_limit: int = 20


class TrainV3ClassicEnv:
    """V5 observation/reward environment wrapper.

    The underlying battle rules are still Python TrainV2/production `ArenaEnvironment`.
    This wrapper adds V5-only observation/history/private-info/reward metadata.
    """

    def __init__(self, config: TrainV3EnvConfig | None = None):
        self.config = config or TrainV3EnvConfig()
        self.env = ClassicRLEnv(
            seed=self.config.seed,
            verify_mask=self.config.verify_mask,
            placement_mode=self.config.placement_mode,
            include_legal_actions_in_info=self.config.include_legal_actions_in_info,
        )
        self.history_events: list[dict[str, Any]] = []

    def reset(self, **kwargs):
        _obs, info = self.env.reset(**kwargs)
        self.history_events.clear()
        return self.observe(), info

    def observe(self, player_id: int | None = None):
        if player_id is None:
            player_id = self.env.current_player_id()
        return encode_observation_v5(
            self.env._env.state,
            player_id,
            info_mode=self.config.info_mode,
            assist_mode=self.config.assist_mode,
            history_events=self.history_events,
        )

    def action_mask(self, player_id: int | None = None):
        return self.env.action_mask(player_id)

    def action_features(self, player_id: int | None = None, *, include_preview: bool = False):
        return self.env.action_features(player_id, include_preview=include_preview)

    def step(self, action_id: int):
        actor_id = self.env.current_player_id()
        st = self.env._env.state
        reward_pre = reward_snapshot_v5(st, actor_id)
        event = self._build_event(actor_id, action_id)
        _obs, reward, terminated, truncated, info = self.env.step(action_id)
        reward_post = reward_snapshot_v5(self.env._env.state, actor_id)
        components = compute_reward_components_v5(reward_pre, reward_post)
        event.update(compute_history_outcome_deltas_v5(reward_pre, reward_post))
        event["turn_number"] = info["turn_number"]
        self.history_events.append(event)
        self.history_events[:] = self.history_events[-self.config.history_limit :]
        weighted_reward = compute_weighted_reward_v5(
            reward,
            components,
            info_mode=self.config.info_mode,
            weights=self.config.reward_weights,
        )
        info = dict(info)
        info["train_v3_base_reward"] = float(reward)
        info["train_v3_reward_shaping"] = float(weighted_reward - float(reward))
        info["train_v3_reward_components"] = components
        return self.observe(), weighted_reward, terminated, truncated, info

    def current_player_id(self) -> int:
        return self.env.current_player_id()

    def _build_event(self, actor_id: int, action_id: int) -> dict[str, Any]:
        state = self.env._env.state
        action = decode_action(state, actor_id, action_id)
        action_type = "unknown"
        source_card = None
        target_card = None
        me, enemy = _get_me_enemy(state, actor_id)

        if action is not None:
            payload = action.to_dict()
            action_type = str(payload.get("type", "unknown"))
            if isinstance(action, PlayCardAction):
                if 0 <= action.hand_index < len(me.hand):
                    source_card = me.hand[action.hand_index]
                target_card = _find_card_by_instance_id([me.hero, enemy.hero, *me.board, *enemy.board], action.target_id)
            elif isinstance(action, AttackAction):
                source_card = _find_card_by_instance_id(me.board, action.attacker_id)
                target_card = enemy.hero if action.target_is_hero else _find_card_by_instance_id(enemy.board, action.target_id)

        return {
            "actor_id": actor_id,
            "action_id": int(action_id),
            "action_type": action_type,
            "source_card": source_card,
            "target_card": target_card,
        }


def _find_card_by_instance_id(cards, instance_id):
    if instance_id is None:
        return None
    target = str(instance_id)
    for card in cards:
        if str(card.instance_id) == target:
            return card
    return None


__all__ = ["TrainV3ClassicEnv", "TrainV3EnvConfig"]
