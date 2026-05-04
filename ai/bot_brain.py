"""
Инференс обученных ONNX-моделей для боевых ботов с поддержкой профилей сложности.
Преобразует GameState -> action_id с маскированием и температурным сэмплированием.

Интеграция:
    1. Модели загружаются при старте сервера (web/server.py create_web_app())
    2. Боты с ID начинающимся на 810416 автоматически используют ONNX
    3. Поддержка моделей: OnlyVersusRandomBiggest, extra-lr-v3-*, Midoriya v3
    4. Температурный сэмплинг: T ∈ [0.1, 1.8] для контроля случайности
    5. Маскирование: недопустимые действия получают логит -1e9

Поддержка форматов наблюдений:
    633: Новый v4 (9 глобальных, 5 слотов доски, 4 слота руки, 39 фич/карта)
    789: Промежуточный (3 глобальных, 7 слотов доски, 4 слота руки, 39 фич/карта)
    997: Legacy Мидория v3 (3 глобальных, 7 слотов доски, 10 слотов руки, 38 фич/карта)

Трансформация 633 → 997 для Мидория v3:
    - Глобальные: 9 → 3 (берем первые 3)
    - Герои: 39 → 38 фич (обрезаем уровень, делим attack/hp/max_hp на 10)
    - Доска: 5 → 7 слотов (добавляем 2 пустых по 38 нулей на каждую сторону)
    - Рука агента: 4 → 10 слотов (добавляем 6 пустых по 38 нулей)
    - Рука врага: скрыта (380 нулей)

Зависимости:
    - onnxruntime (установить: pip install onnxruntime)
    - numpy (уже в проекте)
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from core.actions import BaseAction
from core.engine import ArenaEnvironment
from core.state import GameState, MECHANICS_LIST

logger = logging.getLogger(__name__)


class BerserkInference:
    """
    ONNX-инференс бота с поддержкой множественных профилей сложности.
    Управляет словарём сессий для разных моделей и применяет температурный сэмплинг.
    """

    def __init__(
        self,
        profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        action_dim: int = 200,
    ):
        """
        Args:
            profiles: Словарь профилей вида {difficulty: {model_path, obs_dim, temperature_range}}
            action_dim: Размерность вектора действий (200 макс. действий)
        """
        self.action_dim = action_dim
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        if profiles is None:
            # Fallback на старую модель для обратной совместимости
            current_dir = Path(__file__).parent
            default_path = current_dir / "models" / "extra-lr-v1.onnx"
            profiles = {
                "default": {
                    "model_path": str(default_path),
                    "obs_dim": 997,
                    "temperature_range": (0.5, 0.5),
                }
            }
        
        # Загружаем все профили
        for difficulty, profile in profiles.items():
            model_path = Path(profile["model_path"])
            
            # Проверяем относительный путь от корня проекта
            if not model_path.is_absolute():
                current_dir = Path(__file__).parent.parent
                model_path = current_dir / profile["model_path"]
            
            if not model_path.exists():
                logger.warning(
                    f"[BerserkInference] Модель {difficulty} не найдена: {model_path}, пропускаем"
                )
                continue
            
            try:
                session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                
                self.sessions[difficulty] = {
                    "session": session,
                    "obs_dim": profile["obs_dim"],
                    "temperature_range": profile["temperature_range"],
                }
                
                input_meta = session.get_inputs()[0]
                output_meta = session.get_outputs()[0]
                logger.info(
                    f"[BerserkInference] {difficulty}: {model_path.name}, "
                    f"input={input_meta.shape}, output={output_meta.shape}, "
                    f"T={profile['temperature_range']}"
                )
            except Exception as exc:
                logger.error(
                    f"[BerserkInference] Ошибка загрузки {difficulty}: {exc}"
                )
        
        if not self.sessions:
            raise RuntimeError("[BerserkInference] Ни одна модель не загружена")

    def get_action(
        self,
        game_state: GameState,
        player_id: int,
        legal_actions: List[BaseAction],
        difficulty: str = "medium",
    ) -> int:
        """
        Получить индекс действия с температурным сэмплированием.

        Args:
            game_state: Текущее игровое состояние
            player_id: ID бота (владелец хода)
            legal_actions: Список легальных действий из engine.get_legal_actions()
            difficulty: Профиль сложности (lite/easy/medium/hard/max)

        Returns:
            Индекс действия в списке legal_actions
        """
        if not legal_actions:
            logger.warning("[BerserkInference] Нет доступных действий, возврат 0")
            return 0
        
        # Выбираем профиль (fallback на первый доступный)
        if difficulty not in self.sessions:
            difficulty = next(iter(self.sessions.keys()))
            logger.warning(
                f"[BerserkInference] Профиль не найден, используем {difficulty}"
            )
        
        profile = self.sessions[difficulty]
        session = profile["session"]
        obs_dim = profile["obs_dim"]
        temp_min, temp_max = profile["temperature_range"]
        
        # Случайная температура из диапазона
        temperature = random.uniform(temp_min, temp_max)
        
        # Формируем вектор наблюдений с обрезкой под obs_dim
        obs = self._extract_observation(game_state, player_id, obs_dim)
        
        # Прогоняем через ONNX
        obs_input = obs.reshape(1, -1).astype(np.float32)
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: obs_input})[0]
        
        # Применяем маскирование: недоступные действия получают -1e9
        logits = logits.flatten()[: len(legal_actions)]
        mask = np.ones(len(legal_actions), dtype=np.float32)
        logits = np.where(mask > 0, logits, -1e9)
        
        # Температурный Softmax-сэмплинг: P_i = exp(L_i / T) / Σ exp(L_j / T)
        scaled_logits = logits / temperature
        
        # Защита от переполнения: вычитаем максимум перед exp
        scaled_logits = scaled_logits - np.max(scaled_logits)
        exp_logits = np.exp(scaled_logits)
        probs = exp_logits / np.sum(exp_logits)
        
        # Сэмплируем действие
        action_id = int(np.random.choice(len(legal_actions), p=probs))
        
        logger.debug(
            f"[BerserkInference] player={player_id}, difficulty={difficulty}, "
            f"T={temperature:.2f}, legal={len(legal_actions)}, action={action_id}, "
            f"prob={probs[action_id]:.3f}"
        )
        
        return action_id

    def _extract_observation(
        self, state: GameState, agent_id: int, obs_dim: int
    ) -> np.ndarray:
        """
        Извлечь вектор наблюдений из GameState с адаптацией под разные форматы.
        
        Args:
            state: Игровое состояние
            agent_id: ID агента
            obs_dim: Целевая размерность (633, 789 или 997)
            
        Форматы:
            633: Новый v4 (9 глобальных, 2 героя×39, 2 доски×5×39, рука 4×39)
            789: Промежуточный (3 глобальных, 2 героя×39, 2 доски×7×39, рука 4×39)
            997: Legacy Мидория v3 (3 глобальных, 2 героя×38, 2 доски×7×38, рука 10×38)
        """
        # Определяем, кто агент, кто оппонент
        if state.p1.user_id == agent_id:
            agent_player = state.p1
            enemy_player = state.p2
        else:
            agent_player = state.p2
            enemy_player = state.p1

        # Трансформация 633 → 997 для старых моделей Мидория v3
        if obs_dim == 997:
            obs = self._transform_633_to_997(state, agent_player, enemy_player, agent_id)
        elif obs_dim == 633:
            # Новый формат v4 - прямое кодирование
            obs = self._encode_633(state, agent_player, enemy_player, agent_id)
        else:
            # Старая логика для 789 и других форматов
            use_legacy = (obs_dim == 997)
            
            # Глобальные фичи (3)
            obs_list: List[float] = [
                float(state.turn_number),
                float(state.current_turn_owner_id == agent_id),
                float(state.status.value == "ongoing"),
            ]

            # Фичи агента и оппонента
            obs_list.extend(self._encode_player(agent_player, is_agent=True, use_legacy=use_legacy))
            obs_list.extend(self._encode_player(enemy_player, is_agent=False, use_legacy=use_legacy))

            obs = np.array(obs_list, dtype=np.float32)

            # Паддинг или обрезка до obs_dim
            if len(obs) < obs_dim:
                obs = np.pad(obs, (0, obs_dim - len(obs)), constant_values=0.0)
            else:
                obs = obs[:obs_dim]

        return obs

    def _encode_633(
        self, state: GameState, agent_player: Any, enemy_player: Any, agent_id: int
    ) -> np.ndarray:
        """
        Кодирование в новый формат v4 (633 фичи).
        
        Структура:
        - Глобальные: 9 (turn, is_my_turn, status, agent_mana, agent_max_mana, 
                         agent_hand_size, agent_deck_size, enemy_hand_size, enemy_deck_size)
        - Герой агента: 39
        - Герой врага: 39
        - Доска агента: 5×39 = 195
        - Доска врага: 5×39 = 195
        - Рука агента: 4×39 = 156
        Итого: 9 + 78 + 390 + 156 = 633
        """
        features: List[float] = []
        
        # Глобальные фичи (9)
        features.extend([
            float(state.turn_number),
            float(state.current_turn_owner_id == agent_id),
            float(state.status.value == "ongoing"),
            float(agent_player.mana),
            float(agent_player.max_mana),
            float(len(agent_player.hand)),
            float(len(agent_player.deck)),
            float(len(enemy_player.hand)),
            float(len(enemy_player.deck)),
        ])
        
        # Герой агента (39)
        features.extend(self._encode_card(agent_player.hero, use_legacy=False))
        
        # Герой врага (39)
        features.extend(self._encode_card(enemy_player.hero, use_legacy=False))
        
        # Доска агента (5 слотов × 39)
        for i in range(5):
            if i < len(agent_player.board):
                features.extend(self._encode_card(agent_player.board[i], use_legacy=False))
            else:
                features.extend([0.0] * 39)
        
        # Доска врага (5 слотов × 39)
        for i in range(5):
            if i < len(enemy_player.board):
                features.extend(self._encode_card(enemy_player.board[i], use_legacy=False))
            else:
                features.extend([0.0] * 39)
        
        # Рука агента (4 слота × 39)
        for i in range(4):
            if i < len(agent_player.hand):
                features.extend(self._encode_card(agent_player.hand[i], use_legacy=False))
            else:
                features.extend([0.0] * 39)
        
        return np.array(features, dtype=np.float32)

    def _transform_633_to_997(
        self, state: GameState, agent_player: Any, enemy_player: Any, agent_id: int
    ) -> np.ndarray:
        """
        Трансформация 633 → 997 для старых моделей Мидория v3.
        
        Маппинг:
        1. Глобальные: берем первые 3 из 9 (turn, is_my_turn, status)
        2. Статы: агент 4 фичи (mana, max_mana, hand_size, deck_size), враг 2 фичи
        3. Герои: 39 фич → 38 (обрезаем уровень, делим attack/hp/max_hp на 10)
        4. Доска: 5 слотов → 7 слотов (добавляем 2 пустых по 38 нулей на каждую сторону)
        5. Рука агента: 4 слота → 10 слотов (добавляем 6 пустых по 38 нулей)
        6. Рука врага: не кодируется (в отличие от формата 633)
        
        Структура 997:
        - Глобальные: 3
        - Статы агента: 4
        - Статы врага: 2
        - Герой агента: 38
        - Герой врага: 38
        - Доска агента: 7×38 = 266
        - Доска врага: 7×38 = 266
        - Рука агента: 10×38 = 380
        Итого: 3 + 4 + 2 + 38 + 38 + 266 + 266 + 380 = 997
        """
        features: List[float] = []
        
        # Глобальные фичи (3) - берем из первых 3 фич формата 633
        features.extend([
            float(state.turn_number),
            float(state.current_turn_owner_id == agent_id),
            float(state.status.value == "ongoing"),
        ])
        
        # Статы агента (4)
        features.extend([
            float(agent_player.mana),
            float(agent_player.max_mana),
            float(len(agent_player.hand)),
            float(len(agent_player.deck)),
        ])
        
        # Статы врага (2)
        features.extend([
            float(len(enemy_player.hand)),
            float(len(enemy_player.deck)),
        ])
        
        # Герой агента (38 фич с нормализацией)
        features.extend(self._encode_card(agent_player.hero, use_legacy=True))
        
        # Герой врага (38 фич с нормализацией)
        features.extend(self._encode_card(enemy_player.hero, use_legacy=True))
        
        # Доска агента (7 слотов × 38)
        for i in range(7):
            if i < min(5, len(agent_player.board)):
                features.extend(self._encode_card(agent_player.board[i], use_legacy=True))
            else:
                features.extend([0.0] * 38)
        
        # Доска врага (7 слотов × 38)
        for i in range(7):
            if i < min(5, len(enemy_player.board)):
                features.extend(self._encode_card(enemy_player.board[i], use_legacy=True))
            else:
                features.extend([0.0] * 38)
        
        # Рука агента (10 слотов × 38)
        for i in range(10):
            if i < min(4, len(agent_player.hand)):
                features.extend(self._encode_card(agent_player.hand[i], use_legacy=True))
            else:
                features.extend([0.0] * 38)
        
        return np.array(features, dtype=np.float32)

    def _encode_player(self, player: Any, is_agent: bool, use_legacy: bool = False) -> List[float]:
        """
        Кодирование состояния игрока: ресурсы + герой + доска.
        
        Args:
            player: Объект игрока
            is_agent: True если это агент (видна рука)
            use_legacy: True для старых моделей (997 параметров, 38 фич на карту, 10 слотов руки)
        """
        features: List[float] = []

        # Базовые ресурсы (5 фич)
        features.extend(
            [
                float(player.mana),
                float(player.max_mana),
                float(len(player.hand)),
                float(len(player.deck)),
                float(player.trophies),
            ]
        )

        # Герой
        features.extend(self._encode_card(player.hero, use_legacy=use_legacy))

        # Доска (до 7 существ)
        card_features = 38 if use_legacy else 39
        for i in range(7):
            if i < len(player.board):
                features.extend(self._encode_card(player.board[i], use_legacy=use_legacy))
            else:
                features.extend([0.0] * card_features)

        # Рука агента
        if is_agent:
            if use_legacy:
                # Старые модели: 10 слотов по 38 фич (первые 4 реальные, остальные нули)
                for i in range(10):
                    if i < min(4, len(player.hand)):
                        features.extend(self._encode_card(player.hand[i], use_legacy=True))
                    else:
                        features.extend([0.0] * 38)
            else:
                # Новые модели: 4 слота по 39 фич
                for i in range(4):
                    if i < len(player.hand):
                        features.extend(self._encode_card(player.hand[i], use_legacy=False))
                    else:
                        features.extend([0.0] * 39)
        else:
            # Оппонент: рука скрыта
            if use_legacy:
                features.extend([0.0] * 380)  # 10 слотов по 38 фич
            else:
                features.extend([0.0] * 156)  # 4 слота по 39 фич

        return features

    def _encode_card(self, card: Any, use_legacy: bool = False) -> List[float]:
        """
        Кодирование одной карты.
        
        Args:
            card: Объект карты
            use_legacy: True для старых моделей (38 фич без уровня)
        
        Returns:
            Legacy: 38 фич (5 базовых + 33 механики, нормализованные статы)
            New: 39 фич (5 базовых + 33 механики + 1 уровень)
        """
        if use_legacy:
            # Старые модели: нормализуем статы как во время обучения (делим на 10)
            features: List[float] = [
                float(card.attack) / 10.0,
                float(card.hp) / 10.0,
                float(card.max_hp) / 10.0,
                float(card.mana_cost),
                float(card.is_ready),
            ]
        else:
            # Новые модели: сырые значения
            features: List[float] = [
                float(card.attack),
                float(card.hp),
                float(card.max_hp),
                float(card.mana_cost),
                float(card.is_ready),
            ]

        # Бинарный вектор механик (33 флага)
        for mechanic in MECHANICS_LIST:
            has_mechanic = 0
            for card_mechanic in card.mechanics:
                if card_mechanic == mechanic or card_mechanic.startswith(
                    mechanic + "_"
                ):
                    has_mechanic = 1
                    break
            features.append(float(has_mechanic))

        # Для новых моделей добавляем уровень карты
        if not use_legacy:
            card_level = getattr(card, 'level', 1)
            features.append(float(card_level))

        return features

    def _create_action_mask(self, num_legal_actions: int) -> np.ndarray:
        """
        Создать маску действий: 1 для доступных, 0 для недоступных.
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)
        mask[:num_legal_actions] = 1.0
        return mask


def create_berserk_bot(
    profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> BerserkInference:
    """
    Фабрика для создания инстанса Берсерка с профилями сложности.

    Args:
        profiles: Словарь профилей {difficulty: {model_path, obs_dim, temperature_range}}

    Returns:
        Готовый к использованию BerserkInference
    """
    return BerserkInference(profiles=profiles)

