"""
Gym-like classic RL environment using the real ArenaEnvironment + Task 01 codecs.

No card identity. Deterministic for fixed seeds. Perspective-relative observations.
Each step processes exactly one action for the current player.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import random as rand_mod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.converter import deck_from_card_ids
from core.engine import ArenaEnvironment
from core.state import GameState, GameStatus

from core.classic_setup import create_classic_game_state

from ai.train_v2.classic_obs_v1 import encode_observation
from ai.train_v2.classic_actions_v1 import (
    build_action_mask,
    decode_action,
    encode_action_features,
    MAX_CANDIDATE_ACTIONS,
    _get_me_enemy,
)
from ai.train_v2.fast_action_cache import ActionCache

ENV_VERSION = "classic_rl_env_v2"

logger = logging.getLogger(__name__)


def _normalize_card_catalog(raw_list: list) -> dict:
    """Convert JSON-loaded cards list to {card_id: data} dict, parsing mechanics strings."""
    catalog: Dict[int, dict] = {}
    for item in raw_list:
        cid = item.get("id", 0)
        if cid <= 0:
            continue
        item = dict(item)
        raw_mech = item.get("mechanics", [])
        if isinstance(raw_mech, str):
            try:
                item["mechanics"] = json.loads(raw_mech)
            except (json.JSONDecodeError, TypeError):
                item["mechanics"] = []
        catalog[cid] = item
    return catalog


def _normalize_card_catalog_dict(raw_dict: dict) -> dict:
    """Normalize mechanics strings in an already-keyed {card_id: data} dict."""
    result: Dict[int, dict] = {}
    for cid, item in raw_dict.items():
        item = dict(item)
        raw_mech = item.get("mechanics", [])
        if isinstance(raw_mech, str):
            try:
                item["mechanics"] = json.loads(raw_mech)
            except (json.JSONDecodeError, TypeError):
                item["mechanics"] = []
        result[cid] = item
    return result


def _load_cards_db() -> dict:
    """Load from cards.json, parsing mechanics strings into lists. Returns {id: data}."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "..", "..", "cards.json"),
        os.path.join(current_dir, "..", "cards.json"),
        os.path.join(current_dir, "cards.json"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return _normalize_card_catalog(raw)
            except Exception:
                pass
    return {1: {"id": 1, "card_type": "hero", "name": "Dummy", "mechanics": [], "hp": 30, "base_attack": 0, "base_hp": 30, "mana_cost": 0, "rarity": "common"}}


class ClassicRLEnv:
    def __init__(
        self,
        cards_data: dict[int, dict] | None = None,
        *,
        seed: int | None = None,
        max_turns: int = 80,
        mana_per_turn: int = 1,
        verify_mask: bool = True,
        placement_mode: str = "full",
        include_legal_actions_in_info: bool = True,
    ):
        if cards_data is None:
            self._cards_data = _load_cards_db()
        elif isinstance(cards_data, dict):
            self._cards_data = _normalize_card_catalog_dict(cards_data)
        else:
            self._cards_data = _normalize_card_catalog(cards_data)
        self._base_seed = seed
        self._max_turns = max_turns
        self._mana_per_turn = mana_per_turn
        self._verify_mask = verify_mask
        self._placement_mode = placement_mode
        self._include_legal_actions_in_info = include_legal_actions_in_info

        self._available_heroes = [c["id"] for c in self._cards_data.values() if c.get("card_type") == "hero"]
        self._available_nonhero = [c["id"] for c in self._cards_data.values() if c.get("card_type") in ("warrior", "potion")]

        self._episode_rng: rand_mod.Random = rand_mod.Random()
        self._env: ArenaEnvironment | None = None
        self._steps = 0
        self._turns = 0
        self._episode_seed: int | None = None
        self._p1_reward = 0.0
        self._p2_reward = 0.0
        self._cache: ActionCache | None = None

    # ==================================================================
    # Public API
    # ==================================================================

    def reset(
        self,
        *,
        p1_deck_ids: list[int] | None = None,
        p2_deck_ids: list[int] | None = None,
        p1_levels: dict[int, int] | None = None,
        p2_levels: dict[int, int] | None = None,
        p1_is_bot: bool = False,
        p2_is_bot: bool = True,
        starting_player_id: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict]:

        if seed is not None:
            self._episode_rng = rand_mod.Random(seed)
            self._episode_seed = seed
            rand_mod.seed(seed)
        elif self._base_seed is not None:
            self._episode_rng = rand_mod.Random(self._base_seed)
            self._episode_seed = self._base_seed
            rand_mod.seed(self._base_seed)
        else:
            self._episode_seed = None

        deck_rng = rand_mod.Random(self._episode_rng.randint(0, 2**31 - 1))
        state_rng = rand_mod.Random(self._episode_rng.randint(0, 2**31 - 1))

        if p1_deck_ids is None and p2_deck_ids is None:
            deck_ids = self._generate_default_deck(deck_rng)
            p1_deck_ids = list(deck_ids)
            p2_deck_ids = list(deck_ids)
        elif p1_deck_ids is None:
            p1_deck_ids = self._generate_default_deck(deck_rng)
        elif p2_deck_ids is None:
            p2_deck_ids = self._generate_default_deck(deck_rng)

        p1_deck = deck_from_card_ids(p1_deck_ids, self._cards_data, user_levels=p1_levels)
        p2_deck = deck_from_card_ids(p2_deck_ids, self._cards_data, user_levels=p2_levels)

        game_state = create_classic_game_state(
            1, 2,
            p1_deck, p2_deck,
            p1_is_bot=p1_is_bot,
            p2_is_bot=p2_is_bot,
            starting_player_id=starting_player_id,
            rng=state_rng,
        )
        self._env = ArenaEnvironment(game_state, mana_per_turn=self._mana_per_turn)
        self._steps = 0
        self._turns = 1
        self._p1_reward = 0.0
        self._p2_reward = 0.0
        self._cache = ActionCache(
            self._env.state, self.current_player_id(),
            verify_mask=self._verify_mask,
            placement_mode=self._placement_mode,
        )

        return self.observe(), self._make_info(action_id=-1, success=True, error="")

    def observe(self, player_id: int | None = None) -> np.ndarray:
        if player_id is None:
            player_id = self.current_player_id()
        return encode_observation(self._env.state, player_id)

    def action_mask(self, player_id: int | None = None) -> np.ndarray:
        if player_id is None:
            player_id = self.current_player_id()
        if self._cache is not None and self._cache._player_id == player_id and self._cache._state is self._env.state:
            return self._cache.mask()
        return build_action_mask(
            self._env.state, player_id,
            verify_mask=self._verify_mask,
            placement_mode=self._placement_mode,
        )

    def action_features(self, player_id: int | None = None, *, include_preview: bool = True) -> np.ndarray:
        if player_id is None:
            player_id = self.current_player_id()
        if self._cache is not None and self._cache._player_id == player_id and self._cache._state is self._env.state:
            return self._cache.features(include_preview=include_preview)
        return encode_action_features(
            self._env.state, player_id,
            include_preview=include_preview,
            verify_mask=self._verify_mask,
            placement_mode=self._placement_mode,
        )

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        st = self._env.state
        if st.status != GameStatus.ONGOING:
            return self.observe(), 0.0, True, False, self._make_info(
                action_id=action_id, success=False, error="game_over", invalid=True,
                acting_player_id=st.current_turn_owner_id,
            )

        cp = st.current_turn_owner_id
        mask = self.action_mask(cp)

        if action_id < 0 or action_id >= MAX_CANDIDATE_ACTIONS or mask[action_id] != 1.0:
            self._add_reward(cp, -0.05)
            return self.observe(), -0.05, False, False, self._make_info(
                action_id=action_id, success=False, error="illegal_action", invalid=True,
                acting_player_id=cp, legal_actions_count=int(np.count_nonzero(mask)),
            )

        action = decode_action(st, cp, action_id)
        if action is None:
            self._add_reward(cp, -0.05)
            return self.observe(), -0.05, False, False, self._make_info(
                action_id=action_id, success=False, error="invalid_action", invalid=True,
                acting_player_id=cp,
            )

        pre_snapshot = self._snapshot(cp)
        success, error = self._env.step(cp, action)

        if not success:
            self._add_reward(cp, -0.05)
            return self.observe(), -0.05, False, False, self._make_info(
                action_id=action_id, success=False, error=error, invalid=True,
                acting_player_id=cp, action=action,
            )

        self._steps += 1
        post_snapshot = self._snapshot(cp)
        reward = self._compute_reward(cp, pre_snapshot, post_snapshot, success)
        self._add_reward(cp, reward)

        terminated = st.status != GameStatus.ONGOING
        truncated = st.turn_number > self._max_turns

        if self._cache is not None:
            self._cache.set_state(self._env.state, self.current_player_id())

        info = self._make_info(
            action_id=action_id, success=True, error="",
            acting_reward=reward,
            acting_player_id=cp, action=action,
        )

        return self.observe(), reward, terminated, truncated, info

    def step_core_action(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step a production legal action directly.

        This is used by fixed production-style opponents that choose an index from
        ArenaEnvironment.get_legal_actions() rather than a TrainV2 action_id.
        """
        st = self._env.state
        if st.status != GameStatus.ONGOING:
            return self.observe(), 0.0, True, False, self._make_info(
                action_id=-1, success=False, error="game_over", invalid=True,
                acting_player_id=st.current_turn_owner_id,
            )

        cp = st.current_turn_owner_id
        pre_snapshot = self._snapshot(cp)
        success, error = self._env.step(cp, action)

        if not success:
            self._add_reward(cp, -0.05)
            return self.observe(), -0.05, False, False, self._make_info(
                action_id=-1, success=False, error=error, invalid=True,
                acting_player_id=cp, action=action,
            )

        self._steps += 1
        post_snapshot = self._snapshot(cp)
        reward = self._compute_reward(cp, pre_snapshot, post_snapshot, success)
        self._add_reward(cp, reward)

        terminated = st.status != GameStatus.ONGOING
        truncated = st.turn_number > self._max_turns

        if self._cache is not None:
            self._cache.set_state(self._env.state, self.current_player_id())

        info = self._make_info(
            action_id=-1, success=True, error="",
            acting_reward=reward,
            acting_player_id=cp, action=action,
        )
        return self.observe(), reward, terminated, truncated, info

    def legal_action_ids(self, player_id: int | None = None) -> list[int]:
        if player_id is None:
            player_id = self.current_player_id()
        if self._cache is not None and self._cache._player_id == player_id and self._cache._state is self._env.state:
            return self._cache.legal_ids()
        mask = self.action_mask(player_id)
        return [int(i) for i in np.flatnonzero(mask == 1.0)]

    def current_player_id(self) -> int:
        return self._env.state.current_turn_owner_id

    def winner_id(self) -> int | None:
        st = self._env.state
        if st.status == GameStatus.P1_WIN:
            return st.p1.user_id
        if st.status == GameStatus.P2_WIN:
            return st.p2.user_id
        return None

    def clone_state(self) -> GameState:
        return copy.deepcopy(self._env.state)

    # ==================================================================
    # Internal
    # ==================================================================

    def _generate_default_deck(self, rng: rand_mod.Random) -> list[int]:
        heroes = list(self._available_heroes)
        warriors = [cid for cid in self._available_nonhero
                     if self._cards_data[cid].get("card_type") == "warrior"]
        potions = [cid for cid in self._available_nonhero
                    if self._cards_data[cid].get("card_type") == "potion"]

        if not heroes or len(warriors) < 3:
            return [1] * 9

        hero = rng.choice(heroes)
        deck_ids = [hero]
        w_chosen = rng.sample(warriors, min(3, len(warriors)))
        deck_ids.extend(w_chosen)
        remaining_pool = [c for c in warriors if c not in w_chosen] + potions
        extra = rng.sample(remaining_pool, min(5, len(remaining_pool)))
        deck_ids.extend(extra)
        return deck_ids

    def _snapshot(self, player_id: int) -> dict:
        st = self._env.state
        me, enemy = _get_me_enemy(st, player_id)
        return {
            "my_hero_hp": me.hero.hp,
            "enemy_hero_hp": enemy.hero.hp,
            "my_board_hp": [u.hp for u in me.board],
            "enemy_board_hp": [u.hp for u in enemy.board],
            "my_mana": me.mana,
            "enemy_mana": enemy.mana,
            "opponent_id": enemy.user_id,
        }

    def _compute_reward(self, actor_id: int, pre: dict, post: dict, success: bool) -> float:
        st = self._env.state
        if not success:
            return -0.05

        if st.status == GameStatus.P1_WIN:
            return 1.0 if actor_id == st.p1.user_id else -1.0
        if st.status == GameStatus.P2_WIN:
            return 1.0 if actor_id == st.p2.user_id else -1.0
        if st.status == GameStatus.DRAW:
            return 0.0

        reward = 0.0

        enemy_hp_delta = pre["enemy_hero_hp"] - post["enemy_hero_hp"]
        if enemy_hp_delta > 0:
            reward += 0.02 * enemy_hp_delta

        own_hp_delta = pre["my_hero_hp"] - post["my_hero_hp"]
        if own_hp_delta > 0:
            reward -= 0.01 * own_hp_delta

        pre_enemy_count = len(pre["enemy_board_hp"])
        post_enemy_count = len(post["enemy_board_hp"])
        enemy_killed = pre_enemy_count - post_enemy_count
        if enemy_killed > 0:
            reward += 0.03 * enemy_killed

        pre_own_count = len(pre["my_board_hp"])
        post_own_count = len(post["my_board_hp"])
        own_killed = pre_own_count - post_own_count
        if own_killed > 0:
            reward -= 0.02 * own_killed

        mana_spent = pre["my_mana"] - post["my_mana"]
        if mana_spent > 0:
            reward += min(0.02, 0.005 * mana_spent)

        return reward

    def _add_reward(self, player_id: int, value: float):
        st = self._env.state
        if player_id == st.p1.user_id:
            self._p1_reward += value
        else:
            self._p2_reward += value

    def _make_info(self, action_id, success, error, invalid=False, acting_reward=0.0,
                   acting_player_id=None, action=None, legal_actions_count=None):
        st = self._env.state
        action_dict = None
        if action is not None:
            action_dict = action.to_dict()
        elif action_id >= 0:
            cp = st.current_turn_owner_id
            a = decode_action(st, cp, action_id)
            if a:
                action_dict = a.to_dict()

        if legal_actions_count is None:
            if self._include_legal_actions_in_info:
                legal_actions_count = len(self.legal_action_ids())
            else:
                legal_actions_count = None

        return {
            "acting_player_id": acting_player_id if acting_player_id is not None else st.current_turn_owner_id,
            "current_player_id": st.current_turn_owner_id,
            "action_id": action_id,
            "action": action_dict,
            "success": success,
            "error": error,
            "invalid_action": invalid or not success,
            "turn_number": st.turn_number,
            "status": st.status.value,
            "winner_id": self.winner_id(),
            "legal_actions": legal_actions_count,
            "p1_hp": st.p1.hero.hp,
            "p2_hp": st.p2.hero.hp,
            "p1_reward": float(self._p1_reward),
            "p2_reward": float(self._p2_reward),
            "acting_reward": float(acting_reward),
        }
