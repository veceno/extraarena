"""
Arena Environment v3.3 (Stable & Fixes)
- Исправлено: использование status вместо winner_id.
- Исправлено: поддержка зеркальных матчей.
- Исправлено: конфигурация 5 слотов / 4 карты.
"""
from __future__ import annotations

import logging
import random
import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# --- IMPORTS ---
try:
    from core.state import (
        GameState, PlayerState, CardInstance, CardType, 
        GameStatus, MECHANICS_LIST, ReplacementStatus
    )
    from core.engine import ArenaEnvironment
    from core.actions import BaseAction, PlayCardAction, AttackAction, EndTurnAction
    from core.effects import requires_target, get_taunt_targets
    from core.converter import deck_from_card_ids
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.state import (
        GameState, PlayerState, CardInstance, CardType, 
        GameStatus, MECHANICS_LIST, ReplacementStatus
    )
    from core.engine import ArenaEnvironment
    from core.actions import BaseAction, PlayCardAction, AttackAction, EndTurnAction
    from core.effects import requires_target, get_taunt_targets
    from core.converter import deck_from_card_ids

logger = logging.getLogger(__name__)

# CONSTANTS
MAX_HAND_SIZE = 4
MAX_BOARD_SIZE = 5
FEAT_GLOBAL = 6
FEAT_HERO = 2 + len(MECHANICS_LIST)
FEAT_CARD = 7 + len(MECHANICS_LIST)

OBS_SIZE = (FEAT_GLOBAL + (2 * FEAT_HERO) + (2 * MAX_BOARD_SIZE * FEAT_CARD) + (MAX_HAND_SIZE * FEAT_CARD))
TOTAL_ACTIONS = 1 + (MAX_HAND_SIZE * 17) + (MAX_BOARD_SIZE * 8)

class ArenaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-1.0, high=1000.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.engine: Optional[ArenaEnvironment] = None
        self.agent_id: int = 1
        self.steps_count = 0
        
        raw_db = self._load_cards_db()
        self.cards_db = {c['id']: c for c in raw_db}
        self.available_heroes = [c['id'] for c in raw_db if c.get('card_type') == 'hero']
        self.available_cards = [c['id'] for c in raw_db if c.get('card_type') in ['warrior', 'potion']]
        
        # Печатаем только 1 раз в главном процессе, но в воркерах тоже полезно знать
        # print(f"✅ ArenaEnv v3.3 Loaded. Cards: {len(self.cards_db)}")

    def _load_cards_db(self) -> List[Dict]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            "/Users/laveqox/Documents/ExtraArena/Game/cards.json",
            os.path.join(current_dir, "cards.json"),
            os.path.join(current_dir, "..", "cards.json"),
            os.path.join(current_dir, "train", "cards.json")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except: pass
        return [{"id": 1, "card_type": "hero", "name": "Dummy", "mechanics": [], "hp": 30}]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        options = options or {}
        
        p1_deck = options.get("p1_deck")
        p2_deck = options.get("p2_deck")
        h1_id = options.get("p1_hero")
        h2_id = options.get("p2_hero")
        
        if not p1_deck or not h1_id:
            h_pool = self.available_heroes if self.available_heroes else [1]
            c_pool = self.available_cards if self.available_cards else [32]
            
            shared_hero = random.choice(h_pool)
            shared_deck = random.choices(c_pool, k=8)
            
            h1_id = h2_id = shared_hero
            p1_deck = p2_deck = shared_deck

        try:
            game_state = self._create_match(p1_deck, p2_deck, h1_id, h2_id)
            self.engine = ArenaEnvironment(game_state)
            self.agent_id = game_state.p1.user_id
            self.steps_count = 0
            return self._get_obs(), self._get_info()
        except Exception as e:
            # Silent Fallback
            try:
                h_def = self.available_heroes[0] if self.available_heroes else 1
                c_def = self.available_cards[0] if self.available_cards else 32
                game_state = self._create_match([c_def]*8, [c_def]*8, h_def, h_def)
                self.engine = ArenaEnvironment(game_state)
                self.agent_id = game_state.p1.user_id
                self.steps_count = 0
                return self._get_obs(), self._get_info()
            except: raise e

    def _capture_stats(self) -> Dict:
        st = self.engine.state
        me = st.p1 if st.p1.user_id == self.agent_id else st.p2
        opp = st.p2 if st.p1.user_id == self.agent_id else st.p1
        return {
            "my_hp": me.hero.hp,
            "opp_hp": opp.hero.hp,
            "my_board": len(me.board),
            "opp_board": len(opp.board),
            "my_mana": me.mana
        }

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.engine.state.status != GameStatus.ONGOING:
            return self._get_obs(), 0.0, True, False, self._get_info()

        action = self._decode_action(action_idx)
        if action is None:
            return self._get_obs(), -1.0, False, False, self._get_info()

        prev_stats = self._capture_stats()
        player_id = self.engine.state.current_turn_owner_id
        success, error = self.engine.step(player_id, action)
        curr_stats = self._capture_stats()
        
        terminated = self.engine.state.status != GameStatus.ONGOING
        truncated = self.engine.state.turn_number > 60
        
        reward = self._calculate_delta_reward(success, prev_stats, curr_stats)
        
        self.steps_count += 1
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _calculate_delta_reward(self, success: bool, prev: Dict, curr: Dict) -> float:
        st = self.engine.state
        
        # FIX: Check STATUS instead of winner_id
        if st.status == GameStatus.P1_WIN:
            return 50.0 if self.agent_id == st.p1.user_id else -50.0
        if st.status == GameStatus.P2_WIN:
            return 50.0 if self.agent_id == st.p2.user_id else -50.0
            
        if not success: return -0.2

        reward = 0.0
        if prev["opp_hp"] > curr["opp_hp"]: reward += (prev["opp_hp"] - curr["opp_hp"]) * 0.5
        if prev["my_hp"] > curr["my_hp"]: reward -= (prev["my_hp"] - curr["my_hp"]) * 0.25
        if prev["opp_board"] > curr["opp_board"]: reward += (prev["opp_board"] - curr["opp_board"]) * 2.0
        if curr["my_board"] > prev["my_board"]: reward += 1.0
        if prev["my_mana"] > curr["my_mana"]: reward += 0.1
        return reward

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        st = self.engine.state
        if st.status != GameStatus.ONGOING: return mask
            
        me = st.p1 if st.p1.user_id == self.agent_id else st.p2
        opp = st.p2 if st.p1.user_id == self.agent_id else st.p1
        
        mask[0] = 1.0
        for i in range(len(me.hand)):
            card = me.hand[i]
            if me.mana < card.mana_cost: continue
            if card.card_type == CardType.WARRIOR and len(me.board) >= MAX_BOARD_SIZE: continue
            needs_target = requires_target(card.mechanics)
            is_heal = any("heal" in m or "buff" in m for m in card.mechanics)
            base_idx = 1 + (i * 17)
            if not needs_target:
                mask[base_idx + 0] = 1.0
            else:
                if not is_heal: 
                    for t_idx in range(len(opp.board)): mask[base_idx + 1 + t_idx] = 1.0
                    mask[base_idx + 8] = 1.0
                if is_heal:
                    for t_idx in range(len(me.board)): mask[base_idx + 9 + t_idx] = 1.0
                    mask[base_idx + 16] = 1.0
        
        base_attack_idx = 1 + (MAX_HAND_SIZE * 17)
        taunt_units = get_taunt_targets(opp.board)
        has_taunt = len(taunt_units) > 0
        taunt_indices = [idx for idx, u in enumerate(opp.board) if u in taunt_units]

        for i in range(len(me.board)):
            attacker = me.board[i]
            if not attacker.is_ready or attacker.is_frozen or attacker.attack <= 0: continue
            can_bypass = "bypass_taunt" in attacker.mechanics
            start = base_attack_idx + (i * 8)
            for t_idx in range(len(opp.board)):
                if has_taunt and not can_bypass:
                    if t_idx in taunt_indices: mask[start + t_idx] = 1.0
                else:
                    mask[start + t_idx] = 1.0
            if not has_taunt or can_bypass:
                mask[start + 7] = 1.0
        return mask

    def _create_match(self, d1_ids, d2_ids, h1_id, h2_id) -> GameState:
        h1_list = deck_from_card_ids([h1_id], self.cards_db)
        h2_list = deck_from_card_ids([h2_id], self.cards_db)
        
        if not h1_list or not h2_list: raise ValueError("Heroes not found")
        h1, h2 = h1_list[0], h2_list[0]
        
        deck1 = deck_from_card_ids(d1_ids, self.cards_db)
        deck2 = deck_from_card_ids(d2_ids, self.cards_db)
        random.shuffle(deck1); random.shuffle(deck2)
        
        hand1, deck1 = deck1[:3], deck1[3:]
        hand2, deck2 = deck2[:3], deck2[3:]
        
        p1 = PlayerState(1, False, ReplacementStatus.ACTIVE, h1, 1, 1, hand1, [], deck1, [], 0)
        p2 = PlayerState(2, True, ReplacementStatus.ACTIVE, h2, 1, 1, hand2, [], deck2, [], 0)
        
        for p in [p1, p2]:
            if hasattr(p.hero, 'mechanics'):
                for m in p.hero.mechanics:
                    if m.startswith("start_mana_"):
                        try: p.mana += int(m.split('_')[-1]); p.max_mana += int(m.split('_')[-1])
                        except: pass

        return GameState(p1, p2, p1.user_id, 1, [], [], GameStatus.ONGOING)

    def _get_obs(self) -> np.ndarray:
        st = self.engine.state
        me = st.p1 if st.p1.user_id == self.agent_id else st.p2
        opp = st.p2 if st.p1.user_id == self.agent_id else st.p1
        obs = [st.turn_number/50.0, 1.0 if st.current_turn_owner_id == self.agent_id else 0.0,
               me.mana/10.0, me.max_mana/10.0, opp.mana/10.0, opp.max_mana/10.0]
        obs.extend(self._vectorize_hero(me.hero))
        obs.extend(self._vectorize_hero(opp.hero))
        for i in range(MAX_BOARD_SIZE): obs.extend(self._vectorize_card(me.board[i]) if i < len(me.board) else [0.0]*FEAT_CARD)
        for i in range(MAX_BOARD_SIZE): obs.extend(self._vectorize_card(opp.board[i]) if i < len(opp.board) else [0.0]*FEAT_CARD)
        for i in range(MAX_HAND_SIZE): obs.extend(self._vectorize_card(me.hand[i]) if i < len(me.hand) else [0.0]*FEAT_CARD)
        return np.array(obs, dtype=np.float32)

    def _vectorize_hero(self, h):
        v = [h.hp/50.0, h.max_hp/50.0] + [0.0]*len(MECHANICS_LIST)
        for m in h.mechanics:
            base = m.split('_')[0]
            if base in MECHANICS_LIST: v[2 + MECHANICS_LIST.index(base)] = 1.0
        return v

    def _vectorize_card(self, c):
        v = [c.mana_cost/10.0, c.attack/20.0, c.hp/20.0, c.max_hp/20.0, float(c.is_ready), float(c.is_frozen), c.level/10.0] + [0.0]*len(MECHANICS_LIST)
        for m in c.mechanics:
            base = m.split('_')[0] 
            for known in MECHANICS_LIST:
                if m.startswith(known): base = known; break
            if base in MECHANICS_LIST: v[7 + MECHANICS_LIST.index(base)] = 1.0
        return v

    def _decode_action(self, idx):
        if idx == 0: return EndTurnAction()
        idx -= 1
        st = self.engine.state
        me = st.p1 if st.p1.user_id == self.agent_id else st.p2
        opp = st.p2 if st.p1.user_id == self.agent_id else st.p1
        
        if idx < MAX_HAND_SIZE * 17:
            h_idx, t_code = divmod(idx, 17)
            t_id = None
            if 1 <= t_code <= 7: t_id = str(opp.board[t_code-1].instance_id) if t_code-1 < len(opp.board) else None
            elif t_code == 8: t_id = str(opp.hero.instance_id)
            elif 9 <= t_code <= 15: t_id = str(me.board[t_code-9].instance_id) if t_code-9 < len(me.board) else None
            elif t_code == 16: t_id = str(me.hero.instance_id)
            return PlayCardAction(h_idx, t_id)
            
        idx -= MAX_HAND_SIZE * 17
        if idx < MAX_BOARD_SIZE * 8:
            a_idx, t_code = divmod(idx, 8)
            if a_idx >= len(me.board): return None
            a_id = str(me.board[a_idx].instance_id)
            t_id = str(opp.board[t_code].instance_id) if t_code < 7 and t_code < len(opp.board) else None
            return AttackAction(a_id, t_id, target_is_hero=(t_code == 7))
        return None

    def _get_info(self): return {"legal_actions": np.sum(self.get_action_mask())}