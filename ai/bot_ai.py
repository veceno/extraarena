"""
AI для ботов в бою.
Использует legal_actions из core/engine.ArenaEnvironment для принятия решений.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BotAI:
    """
    Rule-based AI для ботов.
    Использует legal_actions для выбора действий.
    """

    @staticmethod
    def _action_dict(action: Any) -> Dict[str, Any]:
        if isinstance(action, dict):
            return action
        to_dict = getattr(action, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def decide_action(
        legal_actions: List[Any],
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Выбрать одно действие из списка легальных.
        
        Args:
            legal_actions: Список доступных действий из engine.get_legal_actions()
            state: Опциональное состояние для более умных решений
            
        Returns:
            Выбранное действие или None если действий нет
        """
        if not legal_actions:
            logger.warning("[BOT_AI] Нет доступных действий")
            return None
        
        # Разделяем действия по типам
        play_actions = [a for a in legal_actions if BotAI._action_dict(a).get("type") == "play_card"]
        attack_actions = [a for a in legal_actions if BotAI._action_dict(a).get("type") == "attack"]
        end_turn_actions = [a for a in legal_actions if BotAI._action_dict(a).get("type") == "end_turn"]
        
        logger.debug(
            "[BOT_AI] Доступно: play=%d, attack=%d, end_turn=%d",
            len(play_actions), len(attack_actions), len(end_turn_actions)
        )
        
        # Приоритет 1: Атаки (если есть)
        if attack_actions:
            # Предпочитаем атаку героя, если доступна
            hero_attacks = [a for a in attack_actions if BotAI._action_dict(a).get("target_is_hero")]
            if hero_attacks:
                chosen = random.choice(hero_attacks)
                logger.info("[BOT_AI] Выбрана атака героя: %s", chosen)
                return chosen
            
            # Иначе атакуем случайную цель
            chosen = random.choice(attack_actions)
            logger.info("[BOT_AI] Выбрана атака существа: %s", chosen)
            return chosen
        
        # Приоритет 2: Розыгрыш карт (если есть)
        if play_actions:
            chosen = random.choice(play_actions)
            logger.info("[BOT_AI] Выбран розыгрыш карты: %s", chosen)
            return chosen
        
        # Приоритет 3: Завершение хода
        if end_turn_actions:
            chosen = end_turn_actions[0]
            logger.info("[BOT_AI] Выбрано завершение хода")
            return chosen
        
        return None

    @staticmethod
    def decide_turn(engine: Any, bot_id: int) -> List[Any]:
        """
        Спланировать весь ход бота.
        
        Args:
            engine: BattleEngine с методом get_legal_actions()
            bot_id: ID бота
            
        Returns:
            Список действий для выполнения
        """
        actions: List[Any] = []
        
        logger.info("[BOT_AI] decide_turn: bot_id=%s", bot_id)
        
        # Проверяем, не завершена ли игра
        if hasattr(engine, "is_ended") and engine.is_ended:
            logger.info("[BOT_AI] Игра завершена, бот не планирует действия")
            return []
        
        legal_actions = engine.get_legal_actions(bot_id)
        if not legal_actions:
            logger.info("[BOT_AI] Нет легальных действий, завершаем планирование")
            return []

        action = BotAI.decide_action(legal_actions)
        if action:
            actions.append(action)
        
        logger.info("[BOT_AI] Итого запланировано %d действий", len(actions))
        return actions

    @staticmethod
    def decide_single_action(engine: Any, bot_id: int) -> Optional[Dict[str, Any]]:
        """
        Выбрать одно следующее действие для бота.
        Используется для пошагового выполнения.
        
        Args:
            engine: BattleEngine
            bot_id: ID бота
            
        Returns:
            Одно действие или None
        """
        # Проверяем состояние игры
        if hasattr(engine, "is_ended") and engine.is_ended:
            return None
        
        current_player = engine.get_current_player_id()
        if current_player != bot_id:
            logger.debug("[BOT_AI] Не ход бота (current=%s, bot=%s)", current_player, bot_id)
            return None
        
        # Получаем легальные действия
        legal_actions = engine.get_legal_actions(bot_id)
        
        if not legal_actions:
            return {"type": "end_turn"}
        
        return BotAI.decide_action(legal_actions)
