"""
Arena Environment v2.0 (Gymnasium Compatible)
Полностью переписанная среда для поддержки сложных механик:
- Targeted Heal (Фрирен, Юни)
- Deathrattle / Charge / Lifesteal
- Умный выбор целей (свои/чужие)
- Action Masking для обучения
"""
from __future__ import annotations

import logging
import random
import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

# Импортируем игровые структуры
# (Убедись, что core/ доступен в PYTHONPATH или запускай из корня Game/)
try:
    from core.state import (
        GameState, PlayerState, CardInstance, CardType, 
        GameStatus, MECHANICS_LIST
    )
    from core.engine import ArenaEnvironment
    from core.actions import BaseAction, PlayCardAction, AttackAction, EndTurnAction
    from core.effects import requires_target, get_taunt_targets
    from core.converter import deck_from_card_ids, card_from_db
except ImportError:
    # Fallback для случаев, если запускаем не из корня
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.state import (
        GameState, PlayerState, CardInstance, CardType, 
        GameStatus, MECHANICS_LIST
    )
    from core.engine import ArenaEnvironment
    from core.actions import BaseAction, PlayCardAction, AttackAction, EndTurnAction
    from core.effects import requires_target, get_taunt_targets
    from core.converter import deck_from_card_ids, card_from_db

logger = logging.getLogger(__name__)

# ============================================================================
# КОНФИГУРАЦИЯ СРЕДЫ
# ============================================================================
MAX_HAND_SIZE = 10
MAX_BOARD_SIZE = 7
MAX_MANA = 10

# Размеры векторов для one-hot кодирования
FEAT_GLOBAL = 6
FEAT_HERO = 2 + len(MECHANICS_LIST)
FEAT_CARD = 7 + len(MECHANICS_LIST)

# Итоговый размер наблюдения
OBS_SIZE = (
    FEAT_GLOBAL + 
    (2 * FEAT_HERO) + 
    (2 * MAX_BOARD_SIZE * FEAT_CARD) + 
    (MAX_HAND_SIZE * FEAT_CARD)
)

# Action ID mapping:
# 0: End Turn
# 1..170: Play Card (10 hand slots * 17 targets)
# 171..226: Attack (7 attacker slots * 8 targets)
TOTAL_ACTIONS = 1 + (MAX_HAND_SIZE * 17) + (MAX_BOARD_SIZE * 8)

class ArenaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        
        # Определяем пространства
        self.observation_space = spaces.Box(
            low=-1.0, high=1000.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        
        # Внутреннее состояние
        self.engine: Optional[ArenaEnvironment] = None
        self.agent_id: int = 1
        self.steps_count = 0
        self.cards_db = self._load_cards_db()

    def _load_cards_db(self) -> List[Dict]:
        """Умная загрузка cards.json из разных мест."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            "cards.json",                                     # Текущая папка запуска
            os.path.join(current_dir, "cards.json"),          # Папка ai/
            os.path.join(current_dir, "..", "cards.json"),    # Корень Game/
            os.path.join(current_dir, "train", "cards.json"), # Папка ai/train/
            "/Users/laveqox/Documents/ExtraArena/Game/cards.json" # Хардкод на крайний случай
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Ошибка чтения {path}: {e}")
        
        # Если не нашли, возвращаем пустой список (скрипт упадет дальше, но с понятной ошибкой)
        print(f"⚠️  CRITICAL: cards.json не найден! Искал в: {possible_paths}")
        return []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        p1_deck_ids = options.get("p1_deck", [32, 33, 34, 35, 30, 19, 8, 25]) if options else []
        p2_deck_ids = options.get("p2_deck", [32, 33, 34, 35, 30, 19, 8, 25]) if options else []
        p1_hero_id = options.get("p1_hero", 7) if options else 7 
        p2_hero_id = options.get("p2_hero", 4) if options else 4 

        # Создаем состояние
        game_state = self._create_match(p1_deck_ids, p2_deck_ids, p1_hero_id, p2_hero_id)
        self.engine = ArenaEnvironment(game_state)
        self.agent_id = game_state.p1.user_id
        self.steps_count = 0
        
        return self._get_obs(), self._get_info()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.engine.state.status != GameStatus.ONGOING:
            return self._get_obs(), 0.0, True, False, self._get_info()

        action = self._decode_action(action_idx)
        
        # Штраф за инвалидное действие (если маска не сработала)
        if action is None:
            return self._get_obs(), -5.0, False, False, self._get_info()

        player_id = self.engine.state.current_turn_owner_id
        success, error = self.engine.step(player_id, action)

        reward = self._calculate_reward(success, error)
        
        terminated = self.engine.state.status != GameStatus.ONGOING
        truncated = self.engine.state.turn_number > 60

        self.steps_count += 1
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def get_action_mask(self) -> np.ndarray:
        """
        Возвращает булеву маску (1=valid, 0=invalid) для всех 227 действий.
        """
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        st = self.engine.state
        
        if st.status != GameStatus.ONGOING:
            return mask
            
        me = st.p1 if st.p1.user_id == self.agent_id else st.p2
        opp = st.p2 if st.p1.user_id == self.agent_id else st.p1
        
        # 0. END TURN - Всегда можно
        mask[0] = 1.0
        
        # 1. PLAY CARD
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
                # Враги (1..8)
                if not is_heal:
                    for t_idx in range(len(opp.board)):
                        mask[base_idx + 1 + t_idx] = 1.0
                    mask[base_idx + 8] = 1.0 
                
                # Свои (9..16)
                if is_heal: 
                    for t_idx in range(len(me.board)):
                        mask[base_idx + 9 + t_idx] = 1.0
                    mask[base_idx + 16] = 1.0 
                    
        # 2. ATTACK
        base_attack_idx = 1 + (MAX_HAND_SIZE * 17)
        
        taunt_units = get_taunt_targets(opp.board)
        has_taunt = len(taunt_units) > 0
        taunt_indices = [idx for idx, u in enumerate(opp.board) if u in taunt_units]

        for i in range(len(me.board)):
            attacker = me.board[i]
            if not attacker.is_ready: continue
            if attacker.is_frozen: continue
            if attacker.attack <= 0: continue
            
            can_bypass = "bypass_taunt" in attacker.mechanics
            
            start = base_attack_idx + (i * 8)
            
            # Атака существ
            for t_idx in range(len(opp.board)):
                if has_taunt and not can_bypass:
                    if t_idx in taunt_indices:
                        mask[start + t_idx] = 1.0
                else:
                    mask[start + t_idx] = 1.0
            
            # Атака героя
            if not has_taunt or can_bypass:
                mask[start + 7] = 1.0
                
        return mask

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _create_match(self, d1_ids, d2_ids, h1_id, h2_id) -> GameState:
        """Создает реальное состояние игры через конвертер."""
        # 1. Подготовка героев
        h1 = card_from_db(h1_id, self.cards_db, level=1)
        h2 = card_from_db(h2_id, self.cards_db, level=1)
        
        # 2. Подготовка колод
        deck1 = deck_from_card_ids(d1_ids, self.cards_db)
        deck2 = deck_from_card_ids(d2_ids, self.cards_db)
        
        random.shuffle(deck1)
        random.shuffle(deck2)
        
        # 3. Раздача (по 3 карты)
        hand1, deck1 = deck1[:3], deck1[3:]
        hand2, deck2 = deck2[:3], deck2[3:]
        
        # 4. Создание игроков
        p1 = PlayerState(
            user_id=1, hero=h1, deck=deck1, hand=hand1, board=[], 
            mana=1, max_mana=1, graveyard=[]
        )
        p2 = PlayerState(
            user_id=2, hero=h2, deck=deck2, hand=hand2, board=[], 
            mana=1, max_mana=1, graveyard=[] # Обычно второму игроку дают монетку, но пока упростим
        )

        # 5. Применяем start_mana_X если есть
        for mech in h1.mechanics:
            if mech.startswith("start_mana_"):
                bonus = int(mech.split('_')[-1])
                p1.mana += bonus
                p1.max_mana += bonus
        
        for mech in h2.mechanics:
            if mech.startswith("start_mana_"):
                bonus = int(mech.split('_')[-1])
                p2.mana += bonus
                p2.max_mana += bonus

        return GameState(
            match_id=str(uuid4()),
            p1=p1, p2=p2,
            current_turn_owner_id=p1.user_id,
            turn_number=1,
            status=GameStatus.ONGOING,
            action_history=[]
        )

    def _get_obs(self) -> np.ndarray:
        st = self.engine.state
        me = st.p1 if st.p1.user_id == self.agent_id else st.p2
        opp = st.p2 if st.p1.user_id == self.agent_id else st.p1
        
        obs = []
        
        # 1. Global (6)
        obs.extend([
            st.turn_number / 50.0,
            1.0 if st.current_turn_owner_id == self.agent_id else 0.0,
            me.mana / 10.0, me.max_mana / 10.0,
            opp.mana / 10.0, opp.max_mana / 10.0
        ])
        
        # 2. Heroes
        obs.extend(self._vectorize_hero(me.hero))
        obs.extend(self._vectorize_hero(opp.hero))
        
        # 3. Board (Me)
        for i in range(MAX_BOARD_SIZE):
            if i < len(me.board): obs.extend(self._vectorize_card(me.board[i]))
            else: obs.extend([0.0] * FEAT_CARD)

        # 4. Board (Opponent)
        for i in range(MAX_BOARD_SIZE):
            if i < len(opp.board): obs.extend(self._vectorize_card(opp.board[i]))
            else: obs.extend([0.0] * FEAT_CARD)

        # 5. Hand (Me)
        for i in range(MAX_HAND_SIZE):
            if i < len(me.hand): obs.extend(self._vectorize_card(me.hand[i]))
            else: obs.extend([0.0] * FEAT_CARD)

        return np.array(obs, dtype=np.float32)

    def _vectorize_hero(self, hero: CardInstance) -> List[float]:
        vec = [hero.hp / 50.0, hero.max_hp / 50.0]
        mech_vec = [0.0] * len(MECHANICS_LIST)
        for m in hero.mechanics:
            base_m = m.split('_')[0]
            if base_m in MECHANICS_LIST:
                mech_vec[MECHANICS_LIST.index(base_m)] = 1.0
        vec.extend(mech_vec)
        return vec

    def _vectorize_card(self, card: CardInstance) -> List[float]:
        vec = [
            card.mana_cost / 10.0,
            card.attack / 20.0,
            card.hp / 20.0,
            card.max_hp / 20.0,
            1.0 if card.is_ready else 0.0,
            1.0 if card.is_frozen else 0.0,
            card.level / 10.0
        ]
        mech_vec = [0.0] * len(MECHANICS_LIST)
        for m in card.mechanics:
            base_m = m
            # Упрощенная логика: deathrattle_aoe_damage -> deathrattle
            for known in MECHANICS_LIST:
                if m.startswith(known):
                    base_m = known
                    break
            if base_m in MECHANICS_LIST:
                mech_vec[MECHANICS_LIST.index(base_m)] = 1.0
        vec.extend(mech_vec)
        return vec

    def _decode_action(self, idx: int) -> Optional[BaseAction]:
        if idx == 0: return EndTurnAction()
        idx -= 1
        
        # --- PLAY CARD ---
        if idx < MAX_HAND_SIZE * 17:
            hand_idx = idx // 17
            target_code = idx % 17
            target_id = None
            
            st = self.engine.state
            me = st.p1 if st.p1.user_id == self.agent_id else st.p2
            opp = st.p2 if st.p1.user_id == self.agent_id else st.p1
            
            if target_code == 0: pass
            elif 1 <= target_code <= 7: # Opp Unit
                u_idx = target_code - 1
                if u_idx < len(opp.board): target_id = str(opp.board[u_idx].instance_id)
                else: return None 
            elif target_code == 8: # Opp Hero
                target_id = str(opp.hero.instance_id)
            elif 9 <= target_code <= 15: # Friendly Unit
                u_idx = target_code - 9
                if u_idx < len(me.board): target_id = str(me.board[u_idx].instance_id)
                else: return None
            elif target_code == 16: # Friendly Hero
                target_id = str(me.hero.instance_id)
                
            return PlayCardAction(hand_index=hand_idx, target_id=target_id)

        idx -= MAX_HAND_SIZE * 17
        
        # --- ATTACK ---
        if idx < MAX_BOARD_SIZE * 8:
            attacker_idx = idx // 8
            target_code = idx % 8
            
            st = self.engine.state
            me = st.p1 if st.p1.user_id == self.agent_id else st.p2
            opp = st.p2 if st.p1.user_id == self.agent_id else st.p1
            
            if attacker_idx >= len(me.board): return None
            attacker_id = str(me.board[attacker_idx].instance_id)
            target_id = None
            is_hero = False
            
            if target_code < 7: # Attack Unit
                if target_code >= len(opp.board): return None
                target_id = str(opp.board[target_code].instance_id)
            else: # Attack Hero
                is_hero = True
                
            return AttackAction(attacker_id=attacker_id, target_id=target_id, target_is_hero=is_hero)
            
        return None