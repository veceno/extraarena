"""
Ядро боевого движка.
Здесь реализован основной игровой цикл.
"""
from __future__ import annotations

import copy
import logging
import random
import re
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from core.actions import AttackAction, BaseAction, EndTurnAction, ManaDrawAction, PlayCardAction
from core.effects import (
    apply_damage,
    apply_lifesteal,
    consume_rebirth,
    get_taunt_targets,
    has_taunt,
    is_random_battlecry_damage_card,
    parse_crime_and_punishment,
    parse_rebirth,
    process_effects,
    requires_target,
)
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from infrastructure.match_modes import ClassicParams


logger = logging.getLogger(__name__)


def scale_card_by_level(card: CardInstance, level: int) -> CardInstance:
    """Compatibility wrapper for callers that still import from core.engine."""
    from core.card_scaling import scale_card_by_level as _scale_card_by_level

    return _scale_card_by_level(card, level)


# Размер руки в classic-режиме. Раньше был захардкожен в нескольких местах
# (effect_battlecry_draw_card, _handle_end_turn, логике overdraw). Сейчас
# оставляем константой — менять на параметр ClassicParams имеет смысл только
# если появятся альтернативные размеры руки (например, 5-card arena).
HAND_CAP = 4

# Константы для No-FIFO взвешенного добора. Заменяет прежнее
# `deck.pop(0)` поведение: при выборе карты учитывается, как долго
# карта «застревала» в колоде (STUCK_BONUS) и обеспечен ли баланс
# по стоимости маны относительно текущей руки (COST_BIAS).
STUCK_BONUS = 0.5          # бонус к весу за каждый пропуск добора
COST_BIAS = 0.3            # бонус к весу при нехватке cost-бакетa в руке
CHEAP_COST_MAX = 2         # mana_cost <= CHEAP_COST_MAX -> cheap bucket
EXPENSIVE_COST_MIN = 4     # mana_cost >= EXPENSIVE_COST_MIN -> expensive bucket

# Базовая стоимость player-initiated «Добор карт» (см. ManaDrawAction /
# _handle_mana_draw). Стоимость N-го добора в рамках одного хода:
# MANA_DRAW_BASE * N (2, 4, 6, ...). Сбрасывается в начале каждого хода
# игрока (mana_draw_count_this_turn -> 0 в _handle_end_turn).
MANA_DRAW_BASE = 2


def _compute_draw_weights(player: PlayerState) -> List[float]:
    """Рассчитать веса карт в колоде для No-FIFO взвешенного добора.

    Для каждой карты в deck возвращает базовый вес 1.0 плюс бонусы:
      * STUCK_BONUS * c.skip_count — карты, которые долго не выпадали,
        получают больший вес (анти-застревание).
      * COST_BIAS — если в руке не хватает карт соответствующего
        cost-бакетa (cheap / expensive), этот бакет получает бонус.

    Args:
        player: PlayerState, для которого считаются веса.

    Returns:
        Список float-весов той же длины, что и player.deck.
    """
    cheap_in_hand = sum(1 for h in player.hand if h.mana_cost <= CHEAP_COST_MAX)
    expensive_in_hand = sum(1 for h in player.hand if h.mana_cost >= EXPENSIVE_COST_MIN)

    weights: List[float] = []
    for c in player.deck:
        base = 1.0
        stuck = c.skip_count * STUCK_BONUS
        if c.mana_cost <= CHEAP_COST_MAX:
            cost_bias = max(0, 1 - cheap_in_hand) * COST_BIAS
        elif c.mana_cost >= EXPENSIVE_COST_MIN:
            cost_bias = max(0, 1 - expensive_in_hand) * COST_BIAS
        else:
            cost_bias = 0.0
        weights.append(base + stuck + cost_bias)
    return weights


def _weighted_choice_idx(weights: List[float], rng: random.Random) -> int:
    """Выбрать индекс по взвешенному распределению.

    Args:
        weights: список положительных весов.
        rng: генератор случайных чисел (используется rng.random()).

    Returns:
        Индекс выбранного элемента. Если total <= 0 — возвращает 0.
        Если cumulative не превысил target до конца — возвращает
        len(weights) - 1.
    """
    total = sum(weights)
    if total <= 0:
        return 0
    target = rng.random() * total
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if cumulative > target:
            return i
    return len(weights) - 1


def draw_one_from_deck(
    player: PlayerState,
    *,
    overdraw_to_discard: bool,
    source: str,
    logger_obj: logging.Logger | None = None,
    rng: random.Random | None = None,
) -> bool:
    """Единая логика добора карты с учётом лимита руки и reshuffle.

    Используется в двух местах: end-of-turn (core.engine._handle_end_turn)
    и battlecry_draw_card (core.effects). До этой централизации эффект
    battlecry_draw_card просто делал `deck.pop(0) + hand.append` без
    reshuffle и без разбора overdraw — что при пустой колоде приводило
    к тихому провалу (C2: inconsistent overdraw handling) и не
    соответствовало поведению end-of-turn (M2: divergent code paths).

    Возвращает:
        True  — карта была успешно добавлена в руку (или, при
                overdraw_to_discard=True и заполненной руке, перемещена
                в сброс).
        False — карта НЕ добавлена (fatigue, или рука полна и
                overdraw_to_discard=False — top deck card остаётся в deck).

    Args:
        player: PlayerState, для которого выполняется добор.
        overdraw_to_discard: см. ClassicParams.overdraw_to_discard —
            при заполненной руке карта уходит в graveyard вместо того
            чтобы остаться в deck.
        source: метка для логирования (e.g. "end_turn", "battlecry_draw_card").
        logger_obj: опциональный логгер; если None — используется module logger.
        rng: опциональный генератор случайных чисел; если None — используется
            модульный random (для обратной совместимости со старыми вызовами).
    """
    log = logger_obj or logger
    rng = rng or random

    # Шаг 1: если колода пуста — пробуем reshuffle из graveyard.
    if not player.deck:
        if player.graveyard:
            log.info(
                "[RESHUFFLE] source=%s user_id=%s Колода закончилась! Сброс замешан обратно. (%d карт)",
                source,
                player.user_id,
                len(player.graveyard),
            )
            for card in player.graveyard:
                card.reset_to_base_state()

            # Defensive copy: перемешиваем копию, чтобы исходный порядок
            # в graveyard остался предсказуемым до момента clear().
            graveyard_cards = list(player.graveyard)
            rng.shuffle(graveyard_cards)
            player.deck = graveyard_cards
            player.graveyard.clear()
        else:
            log.info(
                "[FATIGUE] source=%s user_id=%s Не может взять карту — колода и сброс пусты",
                source,
                player.user_id,
            )
            return False

    # Шаг 2: проверяем лимит руки. Логика расходится по overdraw_to_discard.
    # Перед этим инкрементируем skip_count для всех карт в колоде — если
    # рука полна и добор невозможен, они должны «состариться» в колоде.
    for c in player.deck:
        c.skip_count += 1

    if len(player.hand) >= HAND_CAP:
        if overdraw_to_discard:
            # Взвешенный выбор карты для сброса — раньше всегда уходила top deck.
            weights = _compute_draw_weights(player)
            choice_idx = _weighted_choice_idx(weights, rng)
            overdrawn_card = player.deck.pop(choice_idx)
            # Карта, ушедшая в graveyard, тоже «покинула» колоду — её
            # skip_count сбрасывается, чтобы при следующем reshuffle она
            # стартовала с чистого состояния.
            overdrawn_card.skip_count = 0
            # После изъятия карты из колоды нужно сбросить skip_count
            # у оставшихся, чтобы при следующем доборе не было двойного
            # учёта «пропусков» этой итерации.
            for c in player.deck:
                c.skip_count = max(0, c.skip_count - 1)
            player.graveyard.append(overdrawn_card)
            log.info(
                "[OVERDRAW_DISCARD] source=%s user_id=%s Hand limit reached, card %s moved to graveyard",
                source,
                player.user_id,
                overdrawn_card.name,
            )
            return True
        log.info(
            "[OVERDRAW_SKIP] source=%s user_id=%s Hand limit reached, top deck card remains in deck: %s",
            source,
            player.user_id,
            player.deck[0].name if player.deck else "none",
        )
        return False

    # Шаг 3: чистый добор (No-FIFO weighted).
    weights = _compute_draw_weights(player)
    choice_idx = _weighted_choice_idx(weights, rng)
    drawn_card = player.deck.pop(choice_idx)
    # Только что вытянутая карта покидает колоду — её skip_count должен
    # обнулиться: «пропуски» были накоплены за время ожидания в deck,
    # и при попадании в hand они уже не нужны для anti-stuck логики.
    drawn_card.skip_count = 0
    # Компенсируем инкремент skip_count для оставшихся карт в колоде.
    for c in player.deck:
        c.skip_count = max(0, c.skip_count - 1)
    player.hand.append(drawn_card)
    return True


class ArenaEnvironment:
    """
    Безголовый игровой движок для пошаговых боев.
    Хранит состояние и обрабатывает действия игроков.
    """

    def __init__(
        self,
        state: GameState,
        mana_per_turn: int = 1,
        classic_params: ClassicParams | None = None,
        apply_start_effects: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        """
        Инициализировать среду с начальным состоянием.

        Args:
            state: Начальное игровое состояние
            mana_per_turn: Прирост маны за ход (default 1, blitz=2)
            classic_params: параметры режима (переопределяет mana_per_turn и новые флаги)
            rng: опциональный RNG для воспроизводимых симуляций; если None —
                создаётся новый random.Random(). Используется при взвешенном
                доборе карт (No-FIFO draw).
        """
        self.state = state
        if classic_params is None:
            classic_params = ClassicParams(mana_per_turn=mana_per_turn)
        self.classic_params = classic_params
        self.mana_per_turn = self.classic_params.mana_per_turn
        # Инстанс-RNG для воспроизводимого No-FIFO добора. Сохраняем отдельно
        # от модульного random, чтобы симуляции/тесты могли задать seed.
        self._rng = rng or random.Random()
        # Прокидываем параметры режима и ссылку на движок в GameState,
        # чтобы эффекты в core.effects могли учитывать правила арены
        # (overdraw_to_discard и т.п.) без жёсткой связки с классом
        # ArenaEnvironment. Оба атрибута опциональны; None означает
        # «classic defaults» — battlecry draw при пустой колоде сделает
        # reshuffle (C2 fix), но при заполненной руке оставит карту в
        # колоде, как и в default classic-режиме.
        self.state.classic_params = classic_params
        self.state.arena_engine = self
        self._ensure_base_snapshots()
        # Применяем стартовые эффекты героев (например, start_mana)
        if apply_start_effects:
            self.apply_start_game_effects()
            self._apply_start_turn_mode_effects()

    def _ensure_base_snapshots(self) -> None:
        """Зафиксировать базовые статы карт, чтобы runtime-баффы не переживали ресайкл."""
        for player in (self.state.p1, self.state.p2):
            for card in [
                player.hero,
                *player.hand,
                *player.board,
                *player.deck,
                *player.graveyard,
            ]:
                card.ensure_base_snapshot()

    def _resolve_player_pair(self, player_id: int) -> Tuple[Optional[PlayerState], Optional[PlayerState]]:
        if self.state.p1.user_id == player_id:
            return self.state.p1, self.state.p2
        if self.state.p2.user_id == player_id:
            return self.state.p2, self.state.p1
        return None, None

    def _card_requires_play_target(self, card: CardInstance) -> bool:
        """Возвращает обязательность выбора цели с учётом карт с авто-выбором цели."""
        return requires_target(card.mechanics) and not is_random_battlecry_damage_card(card)

    def step(self, player_id: int, action: BaseAction) -> Tuple[bool, str]:
        """
        Выполнить действие игрока и обновить состояние.
        
        Args:
            player_id: ID игрока, выполняющего действие
            action: Действие для выполнения
            
        Returns:
            Tuple[success, error_message]: Успешность выполнения и сообщение об ошибке
        """
        action_payload = action.to_dict()
        logger.debug(
            "[CORE] step: входящее действие player_id=%s action=%s",
            player_id,
            action_payload,
        )

        # Проверка, что игра не окончена
        if self.state.status != GameStatus.ONGOING:
            return False, "game_over"

        # Определяем игрока и противника до проверки хода, чтобы мусорный
        # current_turn_owner_id не превращал неизвестного player_id в P2.
        player, opponent = self._resolve_player_pair(player_id)
        if player is None or opponent is None:
            return False, "unknown_player"

        # Проверка, что сейчас ход этого игрока
        if self.state.current_turn_owner_id != player_id:
            return False, "not_your_turn"

        # Валидация действия
        try:
            action.validate(self.state)
            logger.debug(
                "[CORE] step: валидация успешна player_id=%s action=%s",
                player_id,
                action_payload,
            )
        except ValueError as e:
            logger.info(
                "[CORE] step: валидация провалена player_id=%s action=%s error=%s",
                player_id,
                action_payload,
                e,
            )
            return False, str(e)

        v5_history_pre = self._capture_v5_history_pre(player, opponent, action)

        # Обработка действий
        action_description: Optional[Tuple[str, str]] = None
        
        if isinstance(action, PlayCardAction):
            # Сохраняем информацию о карте ДО обработки
            card_info = None
            if 0 <= action.hand_index < len(player.hand):
                card = player.hand[action.hand_index]
                card_info = (card.name, card.card_type, card.mechanics[:])
            
            success, error = self._handle_play_card(player, opponent, action)
            if not success:
                return False, error
            # Формируем описание для action_history
            action_description = self._describe_play_card(player, opponent, card_info, action.target_id)

        elif isinstance(action, AttackAction):
            success, error = self._handle_attack(player, opponent, action)
            if not success:
                return False, error
            # Формируем описание для action_history
            action_description = self._describe_attack(player, opponent, action)

        elif isinstance(action, EndTurnAction):
            old_turn = self.state.turn_number
            self._handle_end_turn(player, opponent)
            # Формируем описание для action_history
            action_description = self._describe_end_turn(player)
            # Добавляем разделитель хода после завершения
            turn_separator = ("system", f"——— Ход №{self.state.turn_number} ———")

        elif isinstance(action, ManaDrawAction):
            success, error = self._handle_mana_draw(player, opponent)
            if not success:
                return False, error
            # Формируем описание для action_history
            action_description = self._describe_mana_draw(player)

        else:
            return False, "unknown_action"

        # Проверка состояния после действия (каскадная очистка deathrattle)
        while True:
            prev_p1_board = len(player.board)
            prev_p2_board = len(opponent.board)
            self._cleanup_dead_units(player)
            self._cleanup_dead_units(opponent)
            if len(player.board) == prev_p1_board and len(opponent.board) == prev_p2_board:
                break
        self._check_game_over()

        # Запись в историю
        self.state.history.append(action.to_dict())
        
        # Обновляем action_history. GameState.action_history — это
        # collections.deque(maxlen=ACTION_HISTORY_MAXLEN), он сам
        # вытесняет старейшие записи в O(1), поэтому ручной `list[-100:]`
        # realloc тут больше не нужен (был O(n) на каждом step).
        if action_description:
            self.state.action_history.append(action_description)
            # Если это был EndTurn, добавляем разделитель
            if isinstance(action, EndTurnAction):
                self.state.action_history.append(turn_separator)
        self.state.v5_history_events.append(
            self._build_v5_history_event(v5_history_pre, player, opponent, action)
        )

        return True, ""

    @staticmethod
    def _history_board_power(player: PlayerState) -> float:
        # Match Rust RewardSnapshotV5: surviving attack times surviving HP.
        return float(sum(max(int(card.attack), 0) * max(int(card.hp), 0) for card in player.board))

    @staticmethod
    def _v5_history_action_id(player: PlayerState, opponent: PlayerState, action: BaseAction) -> int:
        """Canonical classic_actions_v1 id for the applied object action."""
        if isinstance(action, EndTurnAction) or isinstance(action, ManaDrawAction):
            return 0
        if isinstance(action, PlayCardAction):
            position = int(action.position or 0)
            target_id = str(action.target_id) if action.target_id is not None else None
            if target_id is None:
                target_code = 0
            elif target_id == str(opponent.hero.instance_id):
                target_code = 8
            elif target_id == str(player.hero.instance_id):
                target_code = 16
            else:
                target_code = next((i + 1 for i, c in enumerate(opponent.board) if str(c.instance_id) == target_id), None)
                if target_code is None:
                    target_code = next((i + 9 for i, c in enumerate(player.board) if str(c.instance_id) == target_id), 0)
            return 1 + int(action.hand_index) * (8 * 17) + position * 17 + int(target_code)
        if isinstance(action, AttackAction):
            attacker_idx = next((i for i, c in enumerate(player.board) if str(c.instance_id) == str(action.attacker_id)), 0)
            if action.target_is_hero:
                target_code = 7
            else:
                target_code = next((i for i, c in enumerate(opponent.board) if str(c.instance_id) == str(action.target_id)), 0)
            return 545 + attacker_idx * 8 + int(target_code)
        return 0

    def _capture_v5_history_pre(
        self,
        player: PlayerState,
        opponent: PlayerState,
        action: BaseAction,
    ) -> Dict:
        """Capture the rich pre-action fields the V5 temporal encoder needs.

        UI ``action_history`` deliberately stores only text. Keeping this compact
        ring buffer at the actual engine application boundary makes production
        V5 observation history match the Rust trainer's last-20-action view.
        """
        source_card = None
        target_card = None
        if isinstance(action, PlayCardAction) and 0 <= action.hand_index < len(player.hand):
            source_card = self._snapshot_history_card(player.hand[action.hand_index])
            if action.target_id:
                target_card = self._find_unit_by_id(player.board, action.target_id)
                if target_card is None:
                    target_card = self._find_unit_by_id(opponent.board, action.target_id)
                if target_card is None and str(action.target_id) == str(player.hero.instance_id):
                    target_card = player.hero
                if target_card is None and str(action.target_id) == str(opponent.hero.instance_id):
                    target_card = opponent.hero
        elif isinstance(action, AttackAction):
            source_card = self._find_unit_by_id(player.board, action.attacker_id)
            if action.target_is_hero:
                target_card = opponent.hero
            elif action.target_id:
                target_card = self._find_unit_by_id(opponent.board, action.target_id)
        return {
            "actor_id": int(player.user_id),
            "turn_number": int(self.state.turn_number),
            "own_hero_hp": int(player.hero.hp),
            "enemy_hero_hp": int(opponent.hero.hp),
            "my_board_count": len(player.board),
            "enemy_board_count": len(opponent.board),
            "board_power_delta_base": self._history_board_power(player) - self._history_board_power(opponent),
            "source_card": None if source_card is None else self._snapshot_history_card(source_card),
            "target_card": None if target_card is None else self._snapshot_history_card(target_card),
            "action_id": self._v5_history_action_id(player, opponent, action),
        }

    @staticmethod
    def _snapshot_history_card(card: CardInstance) -> CardInstance:
        """Cheap immutable-enough snapshot for the V5 history encoder.

        The encoder only reads scalar fields and mechanics; copying that list is
        sufficient and avoids deep-copying an entire game object on every turn.
        """
        snapshot = copy.copy(card)
        snapshot.mechanics = list(card.mechanics)
        snapshot.base_mechanics = None if card.base_mechanics is None else list(card.base_mechanics)
        return snapshot

    def _build_v5_history_event(
        self,
        pre: Dict,
        player: PlayerState,
        opponent: PlayerState,
        action: BaseAction,
    ) -> Dict:
        if isinstance(action, PlayCardAction):
            action_type = "play_card"
        elif isinstance(action, AttackAction):
            action_type = "attack"
        elif isinstance(action, ManaDrawAction):
            action_type = "mana_draw"
        else:
            action_type = "end_turn"
        post_board_delta = self._history_board_power(player) - self._history_board_power(opponent)
        return {
            "actor_id": int(pre["actor_id"]),
            # The production object action does not retain frozen codec id; the
            # type/card/delta fields are the stable temporal signal. Mana draw
            # is explicitly distinguished by metadata slot 13 in obs_v5.
            "action_id": int(pre["action_id"]),
            "action_type": action_type,
            "enemy_hero_hp_delta": int(pre["enemy_hero_hp"]) - int(opponent.hero.hp),
            "own_hero_hp_delta": int(pre["own_hero_hp"]) - int(player.hero.hp),
            "my_board_count_delta": len(player.board) - int(pre["my_board_count"]),
            "enemy_board_count_delta": len(opponent.board) - int(pre["enemy_board_count"]),
            "board_power_delta": post_board_delta - float(pre["board_power_delta_base"]),
            "turn_number": int(self.state.turn_number),
            "source_card": pre["source_card"],
            "target_card": pre["target_card"],
        }

    def _handle_play_card(
        self, player: PlayerState, opponent: PlayerState, action: PlayCardAction
    ) -> Tuple[bool, str]:
        """Обработать розыгрыш карты."""
        # Проверка индекса
        if action.hand_index < 0 or action.hand_index >= len(player.hand):
            return False, "invalid_hand_index"

        card = player.hand[action.hand_index]
        card.ensure_base_snapshot()
        consumes_ally = "consume_ally" in card.mechanics

        if card.card_type == CardType.WARRIOR and len(player.board) >= 5 and not consumes_ally:
            return False, "board_full"

        # Проверка маны (с учётом spells_free для зелий)
        effective_mana_cost = 0 if (self.classic_params.spells_free and card.card_type == CardType.POTION) else card.mana_cost
        if player.mana < effective_mana_cost:
            return False, "insufficient_mana"

        # КРИТИЧНО: Проверка target_id для карт с обязательной целью
        if self._card_requires_play_target(card) and not action.target_id:
            logger.warning(
                "[CORE] play_card: карта %s требует цель, но target_id отсутствует",
                card.name,
            )
            return False, "target_required"

        target_error = self._validate_play_target(player, opponent, card, action.target_id)
        if target_error:
            return False, target_error

        consumed_unit = None
        if consumes_ally:
            consumed_unit = self._find_unit_by_id(player.board, action.target_id)
            if not consumed_unit:
                return False, "consume_target_not_found"

        # Списываем ману и убираем карту из руки только после всех preflight-проверок.
        player.mana -= effective_mana_cost
        card = player.hand.pop(action.hand_index)

        # Обработка по типу карты
        if card.card_type == CardType.WARRIOR:
            # Обработка consume_ally ПЕРЕД выставлением на доску
            if consumes_ally and consumed_unit:
                consumed_unit.ensure_base_snapshot()
                # Поглощаем союзника: добавляем его статы к карте
                card.attack += consumed_unit.attack
                card.hp += consumed_unit.hp
                card.max_hp += consumed_unit.max_hp

                # Удаляем поглощенного юнита и отправляем в сброс.
                # ВАЖНО: deathrattle съеденного юнита НЕ срабатывает —
                # это by design (consume_ally = «пожирает», не «убивает»).
                # Если в будущем нужно будет триггерить deathrattle
                # при поглощении (как Hearthstone Sylvanas/Evil Geenie),
                # сюда надо добавить явный вызов _execute_deathrattle
                # ДО удаления с доски (иначе card references ломаются).
                player.board.remove(consumed_unit)
                player.graveyard.append(consumed_unit)

                logger.debug(
                    "[CORE] %s поглотил %s: теперь %d/%d (отправлен в сброс)",
                    card.name,
                    consumed_unit.name,
                    card.attack,
                    card.hp,
                )

            # Модельный action-space шире текущей доски: наружу должна уходить
            # фактическая позиция, а не сырой индекс кандидата.
            insert_position = len(player.board)
            if action.position is not None:
                try:
                    insert_position = int(action.position)
                except (TypeError, ValueError):
                    insert_position = len(player.board)
                insert_position = max(0, min(insert_position, len(player.board)))
            action.position = insert_position
            player.board.insert(insert_position, card)

            # SUMMONING SICKNESS: Существо спит в первый ход (если нет Charge)
            if "charge" in card.mechanics:
                card.is_ready = True
                logger.debug("[CORE] %s получил Charge - готов к атаке", card.name)
            elif self.classic_params.summon_ready_on_play:
                # При summon_ready_on_play сразу готов, кроме явных запретов
                if "freeze_on_play" in card.mechanics:
                    card.is_ready = False
                    card.is_frozen = True
                    logger.debug("[CORE] %s freeze_on_play — не готов", card.name)
                else:
                    card.is_ready = True
                    logger.debug("[CORE] %s сразу готов (summon_ready_on_play)", card.name)
            else:
                card.is_ready = False  # Существо спит в первый ход
                logger.debug("[CORE] %s выставлен на доску - спит до следующего хода", card.name)

            # Обрабатываем Battlecry
            process_effects(self.state, card, player, opponent, action.target_id)

        elif card.card_type == CardType.POTION:
            # Применяем эффекты зелья
            process_effects(self.state, card, player, opponent, action.target_id)
            # Зелье не остается на доске, отправляем в сброс
            player.graveyard.append(card)

        return True, ""

    def _handle_attack(
        self, player: PlayerState, opponent: PlayerState, action: AttackAction
    ) -> Tuple[bool, str]:
        """Обработать атаку."""
        # Ищем атакующее существо
        attacker = self._find_unit_by_id(player.board, action.attacker_id)
        if not attacker:
            return False, "attacker_not_found"

        # Проверка готовности атаковать
        if not attacker.is_ready:
            return False, "unit_not_ready"

        # Вычисляем эффективную атаку с учетом аур
        effective_attack = self._apply_aura_bonuses(attacker, player)

        # Проверка атаки
        if effective_attack <= 0:
            return False, "no_attack"

        # Проверка Taunt (только если у атакующего нет bypass_taunt)
        has_bypass = "bypass_taunt" in attacker.mechanics

        if not has_bypass and has_taunt(opponent.board):
            taunt_units = get_taunt_targets(opponent.board)

            if action.target_is_hero:
                logger.debug(
                    "[CORE] Атака героя заблокирована: на доске есть Taunt юниты"
                )
                return False, "must_attack_taunt"

            # Проверяем, что цель - это Taunt существо
            target_is_taunt = False
            if action.target_id:
                for taunt_unit in taunt_units:
                    if str(taunt_unit.instance_id) == action.target_id:
                        target_is_taunt = True
                        break

            if not target_is_taunt:
                logger.debug(
                    "[CORE] Атака заблокирована: цель не является Taunt юнитом"
                )
                return False, "must_attack_taunt"
        
        # Обработка атаки
        if action.target_is_hero:
            # Атака героя
            damage_dealt = apply_damage(opponent.hero, effective_attack, attacker)
            self._last_attack_result = {
                "attacker_name": attacker.name,
                "target_name": "Герой",
                "target_is_hero": True,
                "effective_attack": effective_attack,
                "damage_dealt": damage_dealt,
                "lifesteal": "lifesteal" in attacker.mechanics,
            }
            
            # Применяем Lifesteal
            if damage_dealt > 0:
                apply_lifesteal(attacker, damage_dealt, player.hero)
            
            # Проверка instant_kill: НЕ работает против героев
            if "instant_kill" in attacker.mechanics:
                logger.debug(
                    "[CORE] INSTANT KILL не работает против героя: %s наносит только базовый урон",
                    attacker.name
                )
            else:
                logger.debug(
                    "[CORE] %s атакует героя на %d урона (базовая: %d, с аурами: %d)",
                    attacker.name,
                    effective_attack,
                    attacker.attack,
                    effective_attack,
                )
        else:
            # Атака существа
            target = self._find_unit_by_id(opponent.board, action.target_id)
            if not target:
                return False, "target_not_found"

            # Вычисляем атаку цели с учетом её аур
            target_index = opponent.board.index(target)
            target_effective_attack = self._apply_aura_bonuses(target, opponent)

            # Обмен ударами (с передачей атакующего для reflect)
            target_had_shield = "shield" in target.mechanics
            damage_dealt = apply_damage(target, effective_attack, attacker)
            self._last_attack_result = {
                "attacker_name": attacker.name,
                "target_name": target.name,
                "target_is_hero": False,
                "effective_attack": effective_attack,
                "damage_dealt": damage_dealt,
                "lifesteal": "lifesteal" in attacker.mechanics,
            }
            target_shield_blocked = (
                target_had_shield
                and "shield" not in target.mechanics
                and damage_dealt == 0
            )
            apply_damage(attacker, target_effective_attack, target)
            self._apply_attack_cleave(attacker, opponent, target_index)
            
            # Применяем Lifesteal
            if damage_dealt > 0:
                apply_lifesteal(attacker, damage_dealt, player.hero)
            
            # Проверка unit_killer: убивает каждый атакованный юнит без лимита срабатываний.
            if "unit_killer" in attacker.mechanics:
                if target_shield_blocked:
                    logger.debug(
                        "[CORE] UNIT KILLER: щит %s заблокировал убийство %s",
                        target.name,
                        attacker.name,
                    )
                elif target.hp > 0:
                    target.hp = 0
                    logger.debug("[CORE] UNIT KILLER: %s убил %s мгновенно!", attacker.name, target.name)

            # Проверка instant_kill: первый выбранный вражеский юнит за жизнь Сайтамы.
            elif "instant_kill" in attacker.mechanics:
                if not attacker.instant_kill_used:
                    attacker.instant_kill_used = True
                    if target_shield_blocked:
                        logger.debug(
                            "[CORE] INSTANT KILL: щит %s заблокировал ваншот %s",
                            target.name,
                            attacker.name,
                        )
                    elif target.hp > 0:
                        target.hp = 0
                        logger.debug("[CORE] INSTANT KILL: %s убил %s мгновенно!", attacker.name, target.name)
                else:
                    logger.debug(
                        "[CORE] INSTANT KILL: %s уже использовал ваншот",
                        attacker.name,
                    )
            else:
                logger.debug(
                    "[CORE] %s атакует %s: обмен ударами %d <-> %d",
                    attacker.name,
                    target.name,
                    attacker.attack,
                    target.attack,
                )

        # Атакующее существо становится неготовым (в текущем туре больше не атакует)
        attacker.is_ready = False

        return True, ""

    def _handle_end_turn(self, player: PlayerState, opponent: PlayerState) -> None:
        """Обработать завершение хода."""
        # Передаем ход противнику
        self.state.current_turn_owner_id = opponent.user_id
        old_turn = self.state.turn_number
        self.state.turn_number += 1

        self._apply_start_turn_mode_effects()
        if self.state.status != GameStatus.ONGOING:
            return

        # Восстанавливаем ману противнику
        opponent.max_mana = min(10, opponent.max_mana + self.mana_per_turn)
        opponent.mana = opponent.max_mana
        # Стоимость «Добор карт» обнуляется в начале каждого хода игрока:
        # opponent в этом фрейме — тот, чей ход сейчас начинается.
        opponent.mana_draw_count_this_turn = 0
        pending_mana_drain = self.state.pending_mana_drain_by_player.pop(opponent.user_id, 0)
        if pending_mana_drain > 0:
            old_mana = opponent.mana
            opponent.mana = max(0, opponent.mana - pending_mana_drain)
            logger.debug(
                "[CORE] Кража маны: игрок %s начинает ход с %d/%d маны вместо %d/%d",
                opponent.user_id,
                opponent.mana,
                opponent.max_mana,
                old_mana,
                opponent.max_mana,
            )

        # Делаем всех существ противника готовыми к атаке (пробуждаем)
        for unit in opponent.board:
            # Проверка заморозки
            if unit.is_frozen:
                # Замороженное существо остается спящим (is_asleep=True) и теряет заморозку
                unit.is_ready = False
                unit.is_frozen = False
                logger.debug(
                    "[CORE] Существо %s разморожено, но пропускает активацию (is_asleep=True)",
                    unit.name,
                )
            else:
                # Пробуждаем существо (is_asleep=False)
                unit.is_ready = True

            if "shield_refresh" in unit.mechanics and "shield" not in unit.mechanics:
                unit.mechanics.append("shield")
                logger.debug("[CORE] %s восстанавливает одноразовый щит", unit.name)

            # Проверка регенерации regen_X
            for mechanic in unit.mechanics:
                if mechanic.startswith("regen_"):
                    try:
                        regen_amount = int(mechanic[6:])  # Убираем "regen_"
                        old_hp = unit.hp
                        unit.hp = min(unit.max_hp, unit.hp + regen_amount)
                        logger.debug(
                            "[CORE] %s регенерирует %d здоровья (%d -> %d)",
                            unit.name,
                            regen_amount,
                            old_hp,
                            unit.hp,
                        )
                    except (ValueError, IndexError):
                        logger.warning("[CORE] Некорректный формат regen: %s", mechanic)

        self._apply_regen(opponent.hero)

        # ЧЕСТНЫЙ ЦИКЛ КОЛОДЫ: противник тянет 1 карту. Логика
        # reshuffle + hand cap + overdraw теперь живёт в
        # draw_one_from_deck — она же используется в
        # effect_battlecry_draw_card (C2+M2 fix: единый путь добора).
        draw_one_from_deck(
            opponent,
            overdraw_to_discard=self.classic_params.overdraw_to_discard,
            source="end_turn",
            rng=self._rng,
        )

    def _handle_mana_draw(
        self, player: PlayerState, opponent: PlayerState
    ) -> Tuple[bool, str]:
        """Добор одной карты за ману — player-initiated draw.

        Стоимость N-го добора в рамках одного хода: ``MANA_DRAW_BASE * N``
        (2, 4, 6, ...). Сбрасывается в 0 в начале каждого хода игрока
        (см. :py:meth:`_handle_end_turn`). Сам добор переиспользует
        :py:func:`draw_one_from_deck` (No-FIFO weighted) с тем же
        ``self._rng``, что и end-of-turn — это критично для детерминизма
        симуляций и RL.

        Ограничение «нельзя добрать карту, уже лежащую в руке» выполняется
        автоматически: пулы hand/deck дизъюнктны, а дубликаты card_id в
        колоде запрещены на этапе сборки колоды, поэтому карта из руки
        физически отсутствует в deck и не может быть выбрана.
        """
        # Defense-in-depth: клиент не показывает «+» при полной руке, но
        # стэйл legal_actions теоретически может дойти до сюда.
        if len(player.hand) >= HAND_CAP:
            return False, "hand_full"

        cost = MANA_DRAW_BASE * (player.mana_draw_count_this_turn + 1)
        if player.mana < cost:
            return False, "insufficient_mana"

        # Списываем ману ДО добора, чтобы при fatigue (невозможности добрать)
        # вернуть её обратно.
        player.mana -= cost
        drawn_ok = draw_one_from_deck(
            player,
            overdraw_to_discard=self.classic_params.overdraw_to_discard,
            source="mana_draw",
            rng=self._rng,
        )
        if not drawn_ok:
            # Колода и сброс пусты (fatigue) — добор невозможен, ману
            # возвращаем, счётчик не растёт.
            player.mana += cost
            return False, "no_cards_to_draw"

        player.mana_draw_count_this_turn += 1
        logger.info(
            "[MANA_DRAW] user_id=%s cost=%d mana=%d/%d count_this_turn=%d",
            player.user_id,
            cost,
            player.mana,
            player.max_mana,
            player.mana_draw_count_this_turn,
        )
        return True, ""

    def _apply_start_turn_mode_effects(self) -> None:
        """Apply mode effects that trigger at the start of the active player's turn."""
        if not self.classic_params.sudden_death_enabled:
            return
        if self.state.status != GameStatus.ONGOING:
            return

        player = self.state.p1 if self.state.p1.user_id == self.state.current_turn_owner_id else self.state.p2
        user_id = int(player.user_id)
        last_applied_turn = self.state.sudden_death_last_applied_turn_by_player.get(user_id)
        if last_applied_turn == self.state.turn_number:
            return

        turn_count = self.state.sudden_death_turns_by_player.get(user_id, 0) + 1
        self.state.sudden_death_turns_by_player[user_id] = turn_count
        self.state.sudden_death_last_applied_turn_by_player[user_id] = self.state.turn_number
        damage = (
            self.classic_params.sudden_death_damage_start
            + (turn_count - 1) * self.classic_params.sudden_death_damage_step
        )
        if damage <= 0:
            return

        player.hero.hp -= damage
        logger.info(
            "[CORE] Sudden Death: %s получает %d урона (собственный ход %d) здоровья=%d",
            player.hero.name,
            damage,
            turn_count,
            player.hero.hp,
        )
        self.state.action_history.append(
            ("system", f"Внезапная смерть: {player.hero.name} теряет {damage} здоровья")
        )
        self._check_game_over()

    def _cleanup_dead_units(self, player: PlayerState) -> None:
        """Удалить мертвых существ с доски игрока, активируя Deathrattle / Rebirth / Преступление-и-наказание."""
        # Определяем противника для Deathrattle / crime_and_punishment эффектов
        if player.user_id == self.state.p1.user_id:
            opponent = self.state.p2
            log_type = "player"
        else:
            opponent = self.state.p1
            log_type = "opponent"

        # --- Rebirth: юнит с rebirth_N при летальном уроне выживает с N HP
        # (одноразово — механика снимается). Срабатывает ДО deathrattle:
        # спасённый юнит не считается умершим, его deathrattle не активируется.
        for unit in player.board:
            if unit.hp > 0:
                continue
            rebirth_hp = parse_rebirth(unit.mechanics)
            if rebirth_hp is not None and rebirth_hp > 0:
                unit.hp = rebirth_hp
                consume_rebirth(unit)
                logger.debug(
                    "[CORE] Rebirth: %s выживает с %d HP (способность потрачена)",
                    unit.name,
                    rebirth_hp,
                )
                self.state.action_history.append(
                    (log_type, f"{unit.name} возрождается с {rebirth_hp} HP!")
                )

        # Обрабатываем Deathrattle ПЕРЕД удалением
        dead_units = [unit for unit in player.board if unit.hp <= 0]

        # Преступление и наказание: N урона герою убийцы за КАЖДУЮ погибшую
        # карту владельца. Урон игнорирует броню/ауру (прямое снятие HP) —
        # пассивная кара героя, не «атака» (без reflect/lifesteal).
        cap_damage = parse_crime_and_punishment(player.hero.mechanics)

        for unit in dead_units:
            # Проверяем наличие deathrattle_aoe_damage_X механик
            for mechanic in unit.mechanics:
                if mechanic.startswith("deathrattle_aoe_damage_"):
                    # Парсим значение урона
                    match = re.match(r"deathrattle_aoe_damage_(\d+)", mechanic)
                    if match:
                        damage = int(match.group(1))
                        logger.debug(
                            "[CORE] Deathrattle: %s наносит %d урона всем вражеским юнитам и герою",
                            unit.name,
                            damage
                        )
                        # Наносим урон живым вражеским юнитам
                        for enemy_unit in opponent.board:
                            if enemy_unit.hp > 0:
                                apply_damage(enemy_unit, damage)
                        # Наносим урон вражескому герою, если бой ещё не завершён для него
                        if opponent.hero.hp > 0:
                            apply_damage(opponent.hero, damage)

                        # Добавляем в лог
                        self.state.action_history.append((log_type, f"{unit.name} взрывается после смерти и наносит {damage} урона всем врагам!"))
                    break  # Обрабатываем только первый deathrattle

            # Преступление и наказание — за каждую погибшую карту владельца.
            if cap_damage is not None and cap_damage > 0 and opponent.hero.hp > 0:
                opponent.hero.hp = max(0, opponent.hero.hp - cap_damage)
                logger.debug(
                    "[CORE] Преступление и наказание: %s карает героя %s на %d урона за смерть %s",
                    player.hero.name,
                    opponent.hero.name,
                    cap_damage,
                    unit.name,
                )
                self.state.action_history.append(
                    (log_type, f"Преступление и наказание: {player.hero.name} наносит {cap_damage} урона герою противника за смерть {unit.name}!")
                )

            # Отправляем мертвое существо в сброс
            player.graveyard.append(unit)

        # Удаляем мертвых
        player.board = [unit for unit in player.board if unit.hp > 0]

    def _check_game_over(self) -> None:
        """Проверить условие окончания игры."""
        p1_dead = self.state.p1.hero.hp <= 0
        p2_dead = self.state.p2.hero.hp <= 0
        if p1_dead and p2_dead:
            self.state.status = GameStatus.DRAW
        elif p1_dead:
            self.state.status = GameStatus.P2_WIN
        elif p2_dead:
            self.state.status = GameStatus.P1_WIN
    
    def apply_start_game_effects(self) -> None:
        """
        Применить стартовые эффекты героев при инициализации игры.
        Обрабатывает механики типа start_mana_X (Капитулюга).
        """
        for mechanic in self.state.p1.hero.mechanics:
            if mechanic.startswith("start_mana_"):
                try:
                    mana_bonus = int(mechanic[11:])
                    self.state.p1.mana = min(10, self.state.p1.mana + mana_bonus)
                    self.state.p1.max_mana = min(10, self.state.p1.max_mana + mana_bonus)
                    logger.debug(
                        "[CORE] Герой P1 получает стартовую ману: +%d (mana=%d, max_mana=%d)",
                        mana_bonus, self.state.p1.mana, self.state.p1.max_mana,
                    )
                except ValueError:
                    logger.warning("[CORE] Некорректный формат start_mana у героя P1: %s", mechanic)

        for mechanic in self.state.p2.hero.mechanics:
            if mechanic.startswith("start_mana_"):
                try:
                    mana_bonus = int(mechanic[11:])
                    self.state.p2.mana = min(10, self.state.p2.mana + mana_bonus)
                    self.state.p2.max_mana = min(10, self.state.p2.max_mana + mana_bonus)
                    logger.debug(
                        "[CORE] Герой P2 получает стартовую ману: +%d (mana=%d, max_mana=%d)",
                        mana_bonus, self.state.p2.mana, self.state.p2.max_mana,
                    )
                except ValueError:
                    logger.warning("[CORE] Некорректный формат start_mana у героя P2: %s", mechanic)

    def _find_unit_by_id(
        self, board: List[CardInstance], unit_id: Optional[str]
    ) -> Optional[CardInstance]:
        """Найти существо на доске по ID."""
        if not unit_id:
            return None

        for unit in board:
            if str(unit.instance_id) == unit_id:
                return unit
        return None

    def _validate_play_target(
        self,
        player: PlayerState,
        opponent: PlayerState,
        card: CardInstance,
        target_id: Optional[str],
    ) -> Optional[str]:
        """Проверить, что target_id соответствует механике карты до оплаты/мутаций."""
        if not target_id:
            if "consume_ally" in card.mechanics:
                return "consume_requires_target"
            return None

        if is_random_battlecry_damage_card(card):
            return None

        if "consume_ally" in card.mechanics:
            return None if self._find_unit_by_id(player.board, target_id) else "consume_target_not_found"

        if "delete_target" in card.mechanics:
            if target_id == str(opponent.hero.instance_id):
                return "delete_target_cannot_target_hero"
            return None if self._find_unit_by_id(opponent.board, target_id) else "target_not_found"

        possible_targets = set(self._get_possible_targets(player, opponent, card.mechanics))
        if possible_targets:
            return None if target_id in possible_targets else "target_not_found"

        # Карты без таргетинга не должны принимать произвольный target_id:
        # иначе клиент может протащить id карты из руки или чужой объект.
        if requires_target(card.mechanics) or target_id:
            return "target_not_found"

        return None

    def _apply_regen(self, target: CardInstance) -> None:
        """Apply all regen_X mechanics on a unit or hero."""
        if target.hp <= 0:
            return
        for mechanic in target.mechanics:
            if not mechanic.startswith("regen_"):
                continue
            try:
                regen_amount = int(mechanic[6:])
            except (ValueError, IndexError):
                logger.warning("[CORE] Некорректный формат regen: %s", mechanic)
                continue
            old_hp = target.hp
            target.hp = min(target.max_hp, target.hp + regen_amount)
            logger.debug(
                "[CORE] %s регенерирует %d здоровья (%d -> %d)",
                target.name,
                regen_amount,
                old_hp,
                target.hp,
            )

    def _apply_attack_cleave(
        self,
        attacker: CardInstance,
        opponent: PlayerState,
        target_index: int,
    ) -> None:
        """Warrior cleave hits the neighbors of the attacked board target."""
        for mechanic in attacker.mechanics:
            match = re.match(r"cleave_(\d+)(?:_\d+)?$", mechanic)
            if not match:
                continue
            damage = int(match.group(1))
            for neighbor_index in (target_index - 1, target_index + 1):
                if 0 <= neighbor_index < len(opponent.board):
                    neighbor = opponent.board[neighbor_index]
                    if neighbor.hp > 0:
                        apply_damage(neighbor, damage, attacker)
    
    def _get_aura_bonus(self, player: PlayerState, unit: CardInstance) -> int:
        """
        Получить бонус атаки от аур на доске.
        
        Args:
            player: Владелец юнита
            unit: Юнит, для которого считается бонус
            
        Returns:
            Суммарный бонус атаки от всех аур
        """
        bonus = 0
        for aura_unit in [player.hero, *player.board]:
            # Не применяем ауру к самому себе
            if aura_unit.instance_id == unit.instance_id:
                continue
            
            for mechanic in aura_unit.mechanics:
                if mechanic.startswith("aura_atk_"):
                    try:
                        # Парсим значение ауры: aura_atk_X
                        value_part = mechanic[9:]  # Убираем "aura_atk_"
                        aura_value = int(value_part)
                        bonus += aura_value
                        logger.debug(
                            "[CORE] Аура от %s дает %s +%d атаки",
                            aura_unit.name,
                            unit.name,
                            aura_value,
                        )
                    except ValueError:
                        # Попытка распарсить диапазон X_Y (не должно попадать сюда после конвертера)
                        logger.warning(
                            "[CORE] Некорректный формат ауры %s у %s (ожидалось aura_atk_X). "
                            "Конвертер должен нормализовать механики перед передачей в движок.",
                            mechanic,
                            aura_unit.name,
                        )
                        # Попытка распарсить диапазон как fallback
                        try:
                            value_part = mechanic[9:]
                            if "_" in value_part:
                                parts = value_part.split("_")
                                # Берем первое значение как fallback
                                aura_value = int(parts[0])
                                bonus += aura_value
                                logger.debug(
                                    "[CORE] Fallback: используем минимальное значение %d для %s",
                                    aura_value,
                                    mechanic,
                                )
                        except (ValueError, IndexError):
                            logger.error(
                                "[CORE] Не удалось распарсить ауру %s, пропускаем",
                                mechanic,
                            )
        
        return bonus
    
    def _apply_aura_bonuses(self, attacker: CardInstance, player: PlayerState) -> int:
        """
        Получить эффективную атаку юнита с учетом аур.
        
        Args:
            attacker: Атакующий юнит
            player: Владелец юнита
            
        Returns:
            Эффективная атака с бонусами
        """
        base_attack = attacker.attack
        aura_bonus = self._get_aura_bonus(player, attacker)
        return base_attack + aura_bonus

    def get_preview_delta(self, action: BaseAction) -> Dict[str, int]:
        """
        Симулировать действие и вернуть изменения здоровья объектов.
        
        Args:
            action: Действие для симуляции
            
        Returns:
            Словарь {instance_id: delta_hp}, где delta_hp - изменение здоровья (отрицательное для урона)
        """
        import copy
        
        # Создаем временную копию состояния
        temp_state = copy.deepcopy(self.state)
        temp_env = ArenaEnvironment(
            temp_state,
            classic_params=self.classic_params,
            apply_start_effects=False,
        )
        
        # Сохраняем здоровья до действия
        hp_before: Dict[str, int] = {}
        
        # Собираем здоровья всех объектов
        for unit in temp_state.p1.board:
            hp_before[str(unit.instance_id)] = unit.hp
        for unit in temp_state.p2.board:
            hp_before[str(unit.instance_id)] = unit.hp
        hp_before[str(temp_state.p1.hero.instance_id)] = temp_state.p1.hero.hp
        hp_before[str(temp_state.p2.hero.instance_id)] = temp_state.p2.hero.hp
        
        # Выполняем действие в копии
        player_id = temp_state.current_turn_owner_id
        success, error = temp_env.step(player_id, action)
        
        if not success:
            # Если действие невалидно, возвращаем пустой словарь
            return {}
        
        # Собираем здоровья после действия
        hp_after: Dict[str, int] = {}
        for unit in temp_state.p1.board:
            hp_after[str(unit.instance_id)] = unit.hp
        for unit in temp_state.p2.board:
            hp_after[str(unit.instance_id)] = unit.hp
        hp_after[str(temp_state.p1.hero.instance_id)] = temp_state.p1.hero.hp
        hp_after[str(temp_state.p2.hero.instance_id)] = temp_state.p2.hero.hp
        
        # Вычисляем дельты (только для объектов с изменениями)
        delta: Dict[str, int] = {}
        for instance_id in hp_before:
            if instance_id in hp_after:
                change = hp_after[instance_id] - hp_before[instance_id]
                if change != 0:
                    delta[instance_id] = change
        
        return delta

    def get_legal_actions(self, player_id: int) -> List[BaseAction]:
        """
        КРИТИЧНО ДЛЯ RL: Получить список всех возможных действий для игрока.
        
        Args:
            player_id: ID игрока
            
        Returns:
            Список всех валидных действий
        """
        actions: List[BaseAction] = []

        # Проверка, что игра не окончена
        if self.state.status != GameStatus.ONGOING:
            return actions

        # Проверка, что сейчас ход этого игрока
        if self.state.current_turn_owner_id != player_id:
            return actions

        # Определяем игрока и противника
        player, opponent = self._resolve_player_pair(player_id)
        if player is None or opponent is None:
            return actions

        # 1. Действия розыгрыша карт из руки
        for hand_idx, card in enumerate(player.hand):
            # Проверка маны (с учётом spells_free для зелий)
            effective_mana_cost = 0 if (self.classic_params.spells_free and card.card_type == CardType.POTION) else card.mana_cost
            if player.mana < effective_mana_cost:
                continue

            # Для существ
            if card.card_type == CardType.WARRIOR:
                # Проверка места на доске
                if len(player.board) >= 5 and "consume_ally" not in card.mechanics:
                    continue

                # Проверка choose_shield_damage: может быть с целью ИЛИ без
                has_choose_shield_damage = "choose_shield_damage" in card.mechanics
                
                # Если требуется цель (Battlecry)
                if self._card_requires_play_target(card) or has_choose_shield_damage:
                    # Генерируем действия для каждой возможной цели
                    possible_targets = self._get_possible_targets(
                        player, opponent, card.mechanics
                    )
                    for target_id in possible_targets:
                        actions.append(
                            PlayCardAction(
                                hand_index=hand_idx,
                                target_id=target_id,
                                position=len(player.board),
                            )
                        )
                    
                    # Для choose_shield_damage добавляем вариант БЕЗ цели (щит)
                    if has_choose_shield_damage:
                        actions.append(
                            PlayCardAction(
                                hand_index=hand_idx,
                                target_id=None,
                                position=len(player.board),
                            )
                        )
                else:
                    # Без цели
                    actions.append(
                        PlayCardAction(
                            hand_index=hand_idx,
                            target_id=None,
                            position=len(player.board),
                        )
                    )

            # Для зелий
            elif card.card_type == CardType.POTION:
                if requires_target(card.mechanics):
                    # Генерируем действия для каждой возможной цели
                    possible_targets = self._get_possible_targets(
                        player, opponent, card.mechanics
                    )
                    for target_id in possible_targets:
                        actions.append(
                            PlayCardAction(
                                hand_index=hand_idx,
                                target_id=target_id,
                                position=None,
                            )
                        )
                else:
                    # AOE зелье без цели
                    actions.append(
                        PlayCardAction(
                            hand_index=hand_idx,
                            target_id=None,
                            position=None,
                        )
                    )

        # 2. Действия атаки существами
        for unit in player.board:
            # Проверка готовности
            if not unit.is_ready:
                continue

            effective_attack = self._apply_aura_bonuses(unit, player)
            if effective_attack <= 0:
                continue

            # Проверка bypass_taunt
            has_bypass = "bypass_taunt" in unit.mechanics
            
            # Проверка Taunt (только если нет bypass_taunt)
            if not has_bypass and has_taunt(opponent.board):
                # Можем атаковать только Taunt существ
                taunt_units = get_taunt_targets(opponent.board)
                for target in taunt_units:
                    actions.append(
                        AttackAction(
                            attacker_id=str(unit.instance_id),
                            target_id=str(target.instance_id),
                            target_is_hero=False,
                        )
                    )
            else:
                # Можем атаковать любого (bypass_taunt или нет taunt на доске)
                # Атака существ
                for target in opponent.board:
                    actions.append(
                        AttackAction(
                            attacker_id=str(unit.instance_id),
                            target_id=str(target.instance_id),
                            target_is_hero=False,
                        )
                    )

                # Атака героя
                actions.append(
                    AttackAction(
                        attacker_id=str(unit.instance_id),
                        target_id=None,
                        target_is_hero=True,
                    )
                )

        # 3. Всегда можно завершить ход
        actions.append(EndTurnAction())

        # 4. Добор карт за ману — доступен только когда действие действительно
        # исполнимо: рука не заполнена, есть мана и есть карта в deck/graveyard.
        # Раньше здесь выдавался ManaDrawAction при пустых deck+graveyard, хотя
        # step() неизбежно возвращал no_cards_to_draw; это делало V5 policy
        # выбирать формально legal, но фактически неисполняемое действие.
        if len(player.hand) < HAND_CAP and (player.deck or player.graveyard):
            mana_draw_cost = MANA_DRAW_BASE * (player.mana_draw_count_this_turn + 1)
            if player.mana >= mana_draw_cost:
                actions.append(ManaDrawAction())

        return actions

    def _get_possible_targets(
        self,
        player: PlayerState,
        opponent: PlayerState,
        mechanics: List[str],
    ) -> List[str]:
        """
        Получить возможные цели для эффектов карты.
        
        Args:
            player: Владелец карты
            opponent: Противник
            mechanics: Механики карты
            
        Returns:
            Список ID возможных целей
        """
        targets: List[str] = []

        # Проверяем специальные механики
        is_consume_ally = any("consume_ally" in m for m in mechanics)
        is_damage = any("damage" in m for m in mechanics)
        is_heal = any("heal" in m for m in mechanics)
        is_heal_target = any("heal_target" in m for m in mechanics)  # Новая механика
        is_buff = any("buff" in m for m in mechanics)
        is_delete = any("delete_target" in m for m in mechanics)
        is_freeze = any("freeze" in m or "battlecry_freeze" in m for m in mechanics)
        is_choose_shield_damage = any("choose_shield_damage" in m for m in mechanics)
        # target_ally_max_hp_plus: universal-вариант нацеливается и на героя,
        # обычный — только на союзных юнитов (НЕ героя).
        is_max_hp_plus_universal = any(
            m.startswith("target_ally_max_hp_plus_universal") for m in mechanics
        )
        is_max_hp_plus = any(
            m.startswith("target_ally_max_hp_plus_")
            and not m.startswith("target_ally_max_hp_plus_universal")
            for m in mechanics
        )

        # КРИТИЧНО: consume_ally (Канеки) - только союзные юниты на доске
        if is_consume_ally:
            for unit in player.board:
                targets.append(str(unit.instance_id))
            return targets  # Возвращаем только союзников

        # Геральт (choose_shield_damage) - может выбрать цель для урона ИЛИ не выбирать (щит)
        # Добавляем вражеские цели + героя
        if is_choose_shield_damage:
            for unit in opponent.board:
                targets.append(str(unit.instance_id))
            targets.append(str(opponent.hero.instance_id))
            return targets

        # КРИТИЧНО: Freeze БЕЗ damage - только существа (героев морозить нельзя)
        if is_freeze and not is_damage:
            for unit in opponent.board:
                targets.append(str(unit.instance_id))
            return targets

        if is_delete:
            for unit in opponent.board:
                targets.append(str(unit.instance_id))
            return targets

        # Урон или заморозка с уроном - вражеские цели + герой
        if is_damage or is_freeze:
            for unit in opponent.board:
                targets.append(str(unit.instance_id))
            # Урон и заморозка могут быть применены к герою
            targets.append(str(opponent.hero.instance_id))
            return targets

        # Лечение с выбором цели (battlecry_heal_target_X) - союзные юниты + герой
        if is_heal_target:
            for unit in player.board:
                # Не включаем саму карту, если она уже на доске
                targets.append(str(unit.instance_id))
            # Всегда включаем героя
            targets.append(str(player.hero.instance_id))
            return targets

        # target_ally_max_hp_plus_universal_N — союзные юниты + герой
        if is_max_hp_plus_universal:
            for unit in player.board:
                targets.append(str(unit.instance_id))
            targets.append(str(player.hero.instance_id))
            return targets

        # target_ally_max_hp_plus_N — только союзные юниты (БЕЗ героя)
        if is_max_hp_plus:
            for unit in player.board:
                targets.append(str(unit.instance_id))
            return targets

        # Лечение - союзные цели (только поврежденные)
        if is_heal:
            for unit in player.board:
                if unit.hp < unit.max_hp:
                    targets.append(str(unit.instance_id))
            if player.hero.hp < player.hero.max_hp:
                targets.append(str(player.hero.instance_id))
            return targets
        
        # Баффы - союзные юниты
        if is_buff:
            for unit in player.board:
                targets.append(str(unit.instance_id))
            return targets

        return targets

    def get_state_copy(self) -> GameState:
        """Получить копию текущего состояния (для симуляций)."""
        import copy
        return copy.deepcopy(self.state)

    def reset_to_state(self, state: GameState) -> None:
        """Сбросить состояние к указанному (для симуляций)."""
        self.state = state
    
    def _describe_play_card(
        self, 
        player: PlayerState, 
        opponent: PlayerState,
        card_info: Optional[Tuple[str, CardType, List[str]]],
        target_id: Optional[str]
    ) -> Tuple[str, str]:
        """
        Формирует типизированное описание розыгрыша карты.
        
        Args:
            player: Игрок, разыгравший карту
            opponent: Противник
            card_info: Кортеж (имя_карты, тип_карты, механики) или None
            target_id: ID цели (для описания эффектов)
            
        Returns:
            Tuple[type, text]: ('player'/'opponent', описание)
        """
        # Определяем тип лога
        log_type = "player" if player.user_id == self.state.p1.user_id else "opponent"
        
        if not card_info:
            return (log_type, "разыграл карту")
        
        card_name, card_type, mechanics = card_info
        
        # Базовое действие
        if card_type == CardType.WARRIOR:
            base_text = f"{card_name} выставлен"
        elif card_type == CardType.POTION:
            base_text = f"{card_name} использован"
        else:
            base_text = f"{card_name} разыгран"
        
        # Добавляем описание эффекта
        effect_text = self._describe_card_effect(card_name, mechanics, target_id, player, opponent)
        
        # Проверка Charge: специальное сообщение
        if "charge" in mechanics:
            charge_msg = f"{card_name} врывается в бой с Рывком!"
            if effect_text:
                return (log_type, f"{charge_msg} ({effect_text})")
            return (log_type, charge_msg)
            
        if effect_text:
            return (log_type, f"{card_name}: {effect_text}")
        
        return (log_type, base_text)
    
    def _describe_card_effect(
        self,
        card_name: str,
        mechanics: List[str],
        target_id: Optional[str],
        owner: PlayerState,
        opponent: PlayerState
    ) -> str:
        """
        Генерирует описание эффекта карты на основе механик.
        
        Args:
            card_name: Имя карты
            mechanics: Список механик карты
            target_id: ID цели
            owner: Владелец карты
            opponent: Противник
            
        Returns:
            Описание эффекта или пустая строка
        """
        # Приоритет: battlecry > spell > special > passive
        
        # Battlecry Damage
        for mechanic in mechanics:
            match = re.match(r"battlecry_damage_(\d+)(?:_random)?$", mechanic)
            if match:
                damage = match.group(1)
                if mechanic.endswith("_random"):
                    return f"наносит {damage} урона случайному врагу"
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"наносит {damage} урона по {target_name}"
        
        # Battlecry Heal
        for mechanic in mechanics:
            match = re.match(r"battlecry_heal_(\d+)", mechanic)
            if match:
                heal = match.group(1)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"исцелил {target_name} (+{heal} здоровья)"
        
        # Battlecry Heal Target (Targeted Heal)
        for mechanic in mechanics:
            match = re.match(r"battlecry_heal_target_(\d+)", mechanic)
            if match:
                heal = match.group(1)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"исцеляет {target_name} на {heal} здоровья"
        
        # Battlecry Heal Hero
        for mechanic in mechanics:
            match = re.match(r"battlecry_heal_hero_(\d+)", mechanic)
            if match:
                heal = match.group(1)
                return f"исцелил вашего героя (+{heal} здоровья)"
        
        # Battlecry Draw
        if "battlecry_draw_card" in mechanics:
            return "взял карту из колоды"
        
        # Battlecry Buff
        for mechanic in mechanics:
            match = re.match(r"battlecry_buff_(\d+)_(\d+)", mechanic)
            if match:
                atk, hp = match.group(1), match.group(2)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"усилил {target_name} (+{atk}/+{hp})"
        
        # Battlecry AOE Damage
        for mechanic in mechanics:
            match = re.match(r"battlecry_aoe_damage_(\d+)", mechanic)
            if match:
                damage = match.group(1)
                return f"наносит {damage} урона всем вражеским существам"
        
        # Battlecry Freeze
        if "battlecry_freeze" in mechanics:
            target_name = self._get_target_name(target_id, owner, opponent)
            if target_name != "цель":
                return f"заморозил {target_name} на 1 ход"
        
        # AOE Freeze
        if "desk_freeze" in mechanics:
            return "заморозил всю вражескую доску на 1 ход"

        if "aoe_freeze" in mechanics:
            return "заморозил до 3 вражеских существ на 1 ход"
        
        # Spell Damage
        for mechanic in mechanics:
            match = re.match(r"spell_damage_(\d+)", mechanic)
            if match:
                damage = match.group(1)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"наносит {damage} урона по {target_name}"
        
        # Spell Heal
        for mechanic in mechanics:
            match = re.match(r"spell_heal_(\d+)", mechanic)
            if match:
                heal = match.group(1)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"исцелил {target_name} (+{heal} здоровья)"
        
        # Spell AOE Damage
        for mechanic in mechanics:
            match = re.match(r"spell_aoe_damage_(\d+)", mechanic)
            if match:
                damage = match.group(1)
                return f"наносит {damage} урона всем вражеским существам"
        
        # AOE Damage (зелья и эффекты)
        for mechanic in mechanics:
            match = re.match(r"aoe_damage_(\d+)", mechanic)
            if match:
                damage = match.group(1)
                return f"наносит {damage} урона всем вражеским существам"
        
        # Buff All
        for mechanic in mechanics:
            match = re.match(r"buff_all_(\d+)_(\d+)", mechanic)
            if match:
                atk, hp = match.group(1), match.group(2)
                return f"усилил всех союзных существ (+{atk}/+{hp})"

        # AOE Silence (Солдатик)
        if "aoe_silence_all" in mechanics:
            return "лишает механик всех вражеских существ"
        if "aoe_silence" in mechanics:
            return "лишает механик до 3 вражеских существ"

        # Team-wide shield (Соул Гудман)
        if "team_wide_shield_all" in mechanics:
            return "даёт одноразовый щит всем союзным существам"
        if "team_wide_shield" in mechanics:
            return "даёт одноразовый щит до 3 союзных существ"

        # Max HP buff (Криста Ленц)
        for mechanic in mechanics:
            match = re.match(r"target_ally_max_hp_plus(?:_universal)?_(\d+)", mechanic)
            if match:
                amount = match.group(1)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"увеличил макс. здоровье {target_name} на {amount}"
        
        # Delete Target
        if "delete_target" in mechanics:
            target_name = self._get_target_name(target_id, owner, opponent)
            return f"мгновенно уничтожил {target_name}"
        
        # Mana Gain
        for mechanic in mechanics:
            match = re.match(r"mana_gain_(\d+)", mechanic)
            if match:
                mana = match.group(1)
                return f"восстановил {mana} маны"
        
        # Mana Drain
        for mechanic in mechanics:
            match = re.match(r"mana_drain_(\d+)", mechanic)
            if match:
                mana = match.group(1)
                return f"отнял {mana} маны у противника"
        
        # Cleave
        for mechanic in mechanics:
            match = re.match(r"cleave_(\d+)(?:_\d+)?$", mechanic)
            if match:
                damage = match.group(1)
                return f"наносит {damage} урона соседям цели"
        
        # Cast Random Spell
        if "cast_random_spell" in mechanics:
            return ""
        
        # Choose Shield Damage (Геральт)
        if "choose_shield_damage" in mechanics:
            if target_id:
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"наносит 3 урона по {target_name}"
            else:
                return "получил щит"
        
        # Summon
        for mechanic in mechanics:
            if mechanic.startswith("summon_") and mechanic != "summon":
                return "призвал существо на поле"
        
        # Consume Ally
        if "consume_ally" in mechanics:
            target_name = self._get_target_name(target_id, owner, opponent)
            return f"поглотил {target_name} (получил его характеристики)"
        
        # Instant Kill
        if "unit_killer" in mechanics:
            return "мгновенно убивает атакованную цель-юнита"

        if "instant_kill" in mechanics:
            return "один раз мгновенно убивает выбранную цель при атаке"
        
        # Bypass Taunt
        if "bypass_taunt" in mechanics:
            return "игнорирует провокацию"
        
        # Deathrattle
        if "deathrattle" in mechanics:
            return "при смерти активирует эффект"
        
        # Passive механики (показываем только для существ)
        passives = []
        if "taunt" in mechanics:
            passives.append("Провокация")
        if "shield" in mechanics:
            passives.append("Щит")
        if "permanent_shield" in mechanics:
            passives.append("Вечный щит")
        if "charge" in mechanics:
            passives.append("Рывок")
        if "lifesteal" in mechanics:
            passives.append("Вампиризм")
        
        # Reflect
        for mechanic in mechanics:
            if mechanic.startswith("reflect_"):
                try:
                    amount = mechanic[8:]
                    passives.append(f"Отражение {amount}")
                    break
                except:
                    pass
        
        # Armor
        for mechanic in mechanics:
            if mechanic.startswith("armor_"):
                try:
                    amount = mechanic[6:]
                    passives.append(f"Броня {amount}")
                    break
                except:
                    pass
        
        # Regen
        for mechanic in mechanics:
            if mechanic.startswith("regen_"):
                try:
                    amount = mechanic[6:]
                    passives.append(f"Регенерация {amount}")
                    break
                except:
                    pass
        
        # Aura
        for mechanic in mechanics:
            if mechanic.startswith("aura_atk_"):
                try:
                    amount = mechanic[9:]
                    passives.append(f"Аура +{amount} атаки")
                    break
                except:
                    pass
        
        # Start Mana (для героев)
        for mechanic in mechanics:
            if mechanic.startswith("start_mana_"):
                try:
                    amount = mechanic[11:]
                    passives.append(f"Стартовый бонус: +{amount} маны")
                    break
                except:
                    pass

        # Rebirth (Бан) — пассивная механика возрождения
        for mechanic in mechanics:
            match = re.match(r"rebirth_(\d+)", mechanic)
            if match:
                passives.append(f"Возрождение {match.group(1)}")
                break

        # Crime and Punishment (Достоевский) — пассивная кара героя
        for mechanic in mechanics:
            match = re.match(r"crime_and_punishment_(\d+)", mechanic)
            if match:
                passives.append(
                    f"Преступление и наказание: {match.group(1)} урона герою врага за смерть союзника"
                )
                break

        if passives:
            return f"({', '.join(passives)})"
        
        return ""
    
    def _get_target_name(
        self, 
        target_id: Optional[str],
        owner: PlayerState,
        opponent: PlayerState
    ) -> str:
        """
        Получить имя цели по ID с правильным падежом.
        
        Args:
            target_id: ID цели
            owner: Владелец карты
            opponent: Противник
            
        Returns:
            Имя цели с правильным падежом
        """
        if not target_id:
            return "цель"
        
        # Проверяем героев
        if str(owner.hero.instance_id) == target_id:
            return "вашему герою"
        if str(opponent.hero.instance_id) == target_id:
            return "Герою противника"
        
        # Проверяем существ владельца
        for unit in owner.board:
            if str(unit.instance_id) == target_id:
                return unit.name
        
        # Проверяем существ противника
        for unit in opponent.board:
            if str(unit.instance_id) == target_id:
                return unit.name
        
        return "цель"
    
    def _describe_attack(
        self, player: PlayerState, opponent: PlayerState, action: AttackAction
    ) -> Tuple[str, str]:
        """
        Формирует типизированное описание атаки.
        
        Args:
            player: Игрок, атакующий
            opponent: Противник
            action: Действие атаки
            
        Returns:
            Tuple[type, text]: ('player'/'opponent', описание)
        """
        # Определяем тип лога
        log_type = "player" if player.user_id == self.state.p1.user_id else "opponent"
        
        # Находим атакующее существо
        attacker = self._find_unit_by_id(player.board, action.attacker_id)
        attacker_name = attacker.name if attacker else "существо"
        
        # Вычисляем эффективную атаку с учетом аур
        effective_attack = self._apply_aura_bonuses(attacker, player) if attacker else 0
        attack_result = getattr(self, "_last_attack_result", None)
        damage_dealt = (
            attack_result.get("damage_dealt", effective_attack)
            if isinstance(attack_result, dict)
            else effective_attack
        )
        
        if action.target_is_hero:
            # Атака героя
            text = f"{attacker_name} наносит {damage_dealt} урона по Герою"
        else:
            # Атака существа
            target_name = (
                attack_result.get("target_name")
                if isinstance(attack_result, dict)
                else None
            )
            if not target_name:
                target = self._find_unit_by_id(opponent.board, action.target_id)
                target_name = target.name if target else "существо"
            text = f"{attacker_name} атакует {target_name} и наносит {damage_dealt} урона"
        
        # Проверка Lifesteal
        if attacker and "lifesteal" in attacker.mechanics and damage_dealt > 0:
            text += "...и восстанавливает здоровье герою"
        
        return (log_type, text)
    
    def _describe_end_turn(self, player: PlayerState) -> Tuple[str, str]:
        """
        Формирует типизированное описание завершения хода.
        
        Args:
            player: Игрок, завершивший ход
            
        Returns:
            Tuple[type, text]: ('player'/'opponent', описание)
        """
        log_type = "player" if player.user_id == self.state.p1.user_id else "opponent"
        return (log_type, "завершил(а) ход")

    def _describe_mana_draw(self, player: PlayerState) -> Tuple[str, str]:
        """Формирует типизированное описание «Добор карт» для action_history."""
        log_type = "player" if player.user_id == self.state.p1.user_id else "opponent"
        return (log_type, "добрал(а) карту за ману")
