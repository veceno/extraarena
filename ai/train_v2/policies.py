"""
Simple non-learning policies for rollout/evaluation.
"""
from __future__ import annotations

import random as rand_mod

_PLAY_BASE = 1
_PLAY_STRIDE = 8 * 17
_ATTACK_BASE = _PLAY_BASE + 4 * _PLAY_STRIDE


class Policy:
    name: str = "base"

    def select_action(self, env, player_id: int) -> int:
        raise NotImplementedError


class RandomLegalPolicy(Policy):
    name = "random"

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._rng = rand_mod.Random(seed)

    def reset(self, seed: int):
        self._rng = rand_mod.Random(seed)

    def select_action(self, env, player_id: int) -> int:
        legal = env.legal_action_ids(player_id)
        if not legal:
            return 0
        return self._rng.choice(legal)


class EndTurnPolicy(Policy):
    name = "end_turn"

    def select_action(self, env, player_id: int) -> int:
        legal = env.legal_action_ids(player_id)
        if 0 in legal:
            return 0
        return legal[0] if legal else 0


class GreedyFacePolicy(Policy):
    name = "greedy_face"

    def select_action(self, env, player_id: int) -> int:
        legal = env.legal_action_ids(player_id)
        if not legal:
            return 0

        action_id = self._prefer_attack_enemy_hero(legal)
        if action_id is not None:
            return action_id

        action_id = self._prefer_play_no_target(legal)
        if action_id is not None:
            return action_id

        action_id = self._prefer_play_any(legal)
        if action_id is not None:
            return action_id

        action_id = self._prefer_attack_any_enemy(legal)
        if action_id is not None:
            return action_id

        return 0 if 0 in legal else legal[0]

    def _prefer_attack_enemy_hero(self, legal):
        for aid in legal:
            if aid >= _ATTACK_BASE:
                _, tcode = divmod(aid - _ATTACK_BASE, 8)
                if tcode == 7:
                    return aid
        return None

    def _prefer_play_no_target(self, legal):
        for aid in legal:
            if 1 <= aid <= 544:
                _, tcode = divmod(aid - _PLAY_BASE, 17)
                if tcode == 0:
                    return aid
        return None

    def _prefer_play_any(self, legal):
        for aid in legal:
            if 1 <= aid <= 544:
                return aid
        return None

    def _prefer_attack_any_enemy(self, legal):
        for aid in legal:
            if aid >= _ATTACK_BASE:
                return aid
        return None
