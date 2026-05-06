"""
Ядро боевого движка.
Здесь реализован основной игровой цикл.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from core.actions import AttackAction, BaseAction, EndTurnAction, PlayCardAction
from core.effects import (
    apply_damage,
    apply_lifesteal,
    get_taunt_targets,
    has_taunt,
    process_effects,
    requires_target,
)
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState


logger = logging.getLogger(__name__)


def scale_card_by_level(card: CardInstance, level: int) -> CardInstance:
    """
    Масштабировать характеристики карты в зависимости от уровня (1-10).
    
    Правила масштабирования:
    - Warriors: +1 Atk на четных уровнях, +1 HP на нечетных
    - Potions: Линейное масштабирование урона (+1 за уровень), снижение mana_cost для delete_target (мин. 4)
    - Heroes: +2 HP за уровень, +1 к reflect_X/armor_X/start_mana_X каждые 3 уровня (макс. 10 маны)
    
    Args:
        card: Исходная карта
        level: Уровень карты (1-10)
        
    Returns:
        Масштабированная карта (модифицируется in-place и возвращается)
    """
    if level < 1 or level > 10:
        logger.warning("[SCALE] Некорректный уровень %d для карты %s, используется 1", level, card.name)
        level = 1
    
    # Устанавливаем уровень
    card.level = level
    
    if level == 1:
        # Базовый уровень - без изменений
        return card
    
    # === WARRIORS: +1 Atk на четных, +1 HP на нечетных + Масштабирование механик ===
    if card.card_type == CardType.WARRIOR:
        # Четные уровни: +1 Atk за каждый четный уровень (2, 4, 6, 8, 10)
        even_levels = level // 2
        card.attack += even_levels
        
        # Нечетные уровни: +1 HP за каждый нечетный уровень (3, 5, 7, 9)
        odd_levels = (level - 1) // 2
        card.hp += odd_levels
        card.max_hp += odd_levels

        # Масштабирование механик (AOE damage, Targeted Heal/Damage)
        scaled_mechanics = []
        for mechanic in card.mechanics:
            # 1. AOE Damage (Нерф AOE): +1 каждые 2 уровня
            # Паттерн: Любая механика, содержащая aoe_damage_X (deathrattle_aoe_damage_3 и т.д.)
            aoe_match = re.match(r"(.*aoe_damage_)(\d+)", mechanic)
            if aoe_match:
                prefix, base_val = aoe_match.groups()
                new_val = int(base_val) + ((level - 1) // 2)
                scaled_mechanics.append(f"{prefix}{new_val}")
                continue

            # 2. Targeted Heal / Damage (Single Target): Tiered рост каждые 2 уровня
            target_match = re.match(r"(.*(?:heal_target|damage)_)(\d+)", mechanic)
            if target_match:
                prefix, base_val = target_match.groups()
                new_val = int(base_val) + ((level - 1) // 2)
                scaled_mechanics.append(f"{prefix}{new_val}")
                continue

            # 3. Passive mechanics (regen, armor, aura, reflect): Tiered рост каждые 3 уровня
            passive_scaled = False
            for passive_prefix in ["regen_", "armor_", "aura_atk_", "reflect_"]:
                passive_match = re.match(rf"({passive_prefix})(\d+)", mechanic)
                if passive_match:
                    prefix, base_val = passive_match.groups()
                    new_val = int(base_val) + ((level - 1) // 3)
                    scaled_mechanics.append(f"{prefix}{new_val}")
                    passive_scaled = True
                    break
            if passive_scaled:
                continue

            # Остальные механики без изменений (taunt, shield, charge, lifesteal, etc.)
            scaled_mechanics.append(mechanic)
        
        card.mechanics = scaled_mechanics
        
        logger.debug(
            "[SCALE] Warrior %s (lvl %d): Atk +%d, HP +%d, mechanics scaled",
            card.name, level, even_levels, odd_levels
        )
    
    # === POTIONS: Масштабирование урона и снижение стоимости ===
    elif card.card_type == CardType.POTION:
        # Масштабируем урон в механиках (damage_X -> damage_(X + level - 1))
        scaled_mechanics = []
        for mechanic in card.mechanics:
            # Парсим damage_X
            match = re.match(r"(spell_|battlecry_)?damage_(\d+)", mechanic)
            if match:
                prefix = match.group(1) or ""
                base_damage = int(match.group(2))
                new_damage = base_damage + ((level - 1) // 3)
                scaled_mechanic = f"{prefix}damage_{new_damage}"
                scaled_mechanics.append(scaled_mechanic)
                logger.debug(
                    "[SCALE] Potion %s (lvl %d): %s -> %s",
                    card.name, level, mechanic, scaled_mechanic
                )
            else:
                scaled_mechanics.append(mechanic)
        
        card.mechanics = scaled_mechanics
        
        # Снижаем стоимость delete_target (минимум 4 маны)
        if "delete_target" in card.mechanics:
            card.mana_cost = max(4, card.mana_cost - (level - 1))
            logger.debug(
                "[SCALE] Potion %s (lvl %d): delete_target mana_cost снижена до %d",
                card.name, level, card.mana_cost
            )
    
    # === HEROES: +2 HP за уровень, +1 к механикам каждые 3 уровня ===
    elif card.card_type == CardType.HERO:
        # +2 HP за уровень
        hp_bonus = (level - 1) * 2
        card.hp += hp_bonus
        card.max_hp += hp_bonus
        
        # +1 к reflect_X, armor_X, start_mana_X каждые 3 уровня
        bonus_tiers = (level - 1) // 3  # 0 на уровнях 1-3, 1 на 4-6, 2 на 7-9, 3 на 10
        
        if bonus_tiers > 0:
            scaled_mechanics = []
            for mechanic in card.mechanics:
                # reflect_X
                match = re.match(r"reflect_(\d+)", mechanic)
                if match:
                    base_value = int(match.group(1))
                    new_value = base_value + bonus_tiers
                    scaled_mechanics.append(f"reflect_{new_value}")
                    logger.debug(
                        "[SCALE] Hero %s (lvl %d): reflect_%d -> reflect_%d",
                        card.name, level, base_value, new_value
                    )
                    continue
                
                # armor_X
                match = re.match(r"armor_(\d+)", mechanic)
                if match:
                    base_value = int(match.group(1))
                    new_value = base_value + bonus_tiers
                    scaled_mechanics.append(f"armor_{new_value}")
                    logger.debug(
                        "[SCALE] Hero %s (lvl %d): armor_%d -> armor_%d",
                        card.name, level, base_value, new_value
                    )
                    continue
                
                # start_mana_X (ограничение min(10, ...))
                match = re.match(r"start_mana_(\d+)", mechanic)
                if match:
                    base_value = int(match.group(1))
                    new_value = min(10, base_value + bonus_tiers)
                    scaled_mechanics.append(f"start_mana_{new_value}")
                    logger.debug(
                        "[SCALE] Hero %s (lvl %d): start_mana_%d -> start_mana_%d",
                        card.name, level, base_value, new_value
                    )
                    continue
                
                # Остальные механики без изменений
                scaled_mechanics.append(mechanic)
            
            card.mechanics = scaled_mechanics
        
        logger.debug(
            "[SCALE] Hero %s (lvl %d): HP +%d, mechanics bonus tier %d",
            card.name, level, hp_bonus, bonus_tiers
        )
    
    return card


class ArenaEnvironment:
    """
    Безголовый игровой движок для пошаговых боев.
    Хранит состояние и обрабатывает действия игроков.
    """

    def __init__(self, state: GameState, mana_per_turn: int = 1) -> None:
        """
        Инициализировать среду с начальным состоянием.
        
        Args:
            state: Начальное игровое состояние
            mana_per_turn: Прирост маны за ход (default 1, blitz=2)
        """
        self.state = state
        self.mana_per_turn = mana_per_turn
        # Применяем стартовые эффекты героев (например, start_mana)
        self.apply_start_game_effects()

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

        # Определяем игрока и противника
        if self.state.p1.user_id == player_id:
            player = self.state.p1
            opponent = self.state.p2
        else:
            player = self.state.p2
            opponent = self.state.p1

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

        else:
            return False, "unknown_action"

        # Проверка состояния после действия
        self._cleanup_dead_units(player)
        self._cleanup_dead_units(opponent)
        self._check_game_over()

        # Запись в историю
        self.state.history.append(action.to_dict())
        
        # Обновляем action_history (последние 100 действий для расширенного лога)
        if action_description:
            self.state.action_history.append(action_description)
            # Если это был EndTurn, добавляем разделитель
            if isinstance(action, EndTurnAction):
                self.state.action_history.append(turn_separator)
            self.state.action_history = self.state.action_history[-100:]

        return True, ""

    def _handle_play_card(
        self, player: PlayerState, opponent: PlayerState, action: PlayCardAction
    ) -> Tuple[bool, str]:
        """Обработать розыгрыш карты."""
        # Проверка индекса
        if action.hand_index < 0 or action.hand_index >= len(player.hand):
            return False, "invalid_hand_index"

        card = player.hand[action.hand_index]

        # Проверка маны
        if player.mana < card.mana_cost:
            return False, "insufficient_mana"

        # КРИТИЧНО: Проверка target_id для карт с requires_target
        if requires_target(card.mechanics) and not action.target_id:
            logger.warning(
                "[CORE] play_card: карта %s требует цель, но target_id отсутствует",
                card.name,
            )
            return False, "target_required"

        # Списываем ману
        player.mana -= card.mana_cost

        # Обработка по типу карты
        if card.card_type == CardType.WARRIOR:
            # Проверка места на доске
            if len(player.board) >= 7:  # Максимум 7 существ на доске
                return False, "board_full"
            
            # Обработка consume_ally ПЕРЕД выставлением на доску
            if "consume_ally" in card.mechanics:
                if not action.target_id:
                    return False, "consume_requires_target"
                
                # Ищем союзного юнита для поглощения
                consumed_unit = None
                for unit in player.board:
                    if str(unit.instance_id) == action.target_id:
                        consumed_unit = unit
                        break
                
                if not consumed_unit:
                    return False, "consume_target_not_found"
                
                # Поглощаем союзника: добавляем его статы к карте
                card.attack += consumed_unit.attack
                card.hp += consumed_unit.hp
                card.max_hp += consumed_unit.max_hp
                
                # Удаляем поглощенного юнита и отправляем в сброс
                player.board.remove(consumed_unit)
                player.graveyard.append(consumed_unit)
                
                logger.debug(
                    "[CORE] %s поглотил %s: теперь %d/%d (отправлен в сброс)",
                    card.name,
                    consumed_unit.name,
                    card.attack,
                    card.hp,
                )

            # Вставляем на указанную позицию или в конец
            if action.position is not None and 0 <= action.position <= len(player.board):
                player.board.insert(action.position, card)
            else:
                player.board.append(card)

            # SUMMONING SICKNESS: Существо спит в первый ход (если нет Charge)
            if "charge" in card.mechanics:
                card.is_ready = True
                logger.debug("[CORE] %s получил Charge - готов к атаке", card.name)
            else:
                card.is_ready = False  # Существо спит в первый ход
                logger.debug("[CORE] %s выставлен на доску - спит до следующего хода", card.name)

            # Обрабатываем Battlecry
            process_effects(self.state, card, player, opponent, action.target_id)

        elif card.card_type == CardType.POTION:
            # КРИТИЧНО: Для зелий с target_id, если цель - герой, передаем объект героя
            target_obj = None
            if action.target_id:
                # Ищем цель среди существ противника
                target_obj = self._find_unit_by_id(opponent.board, action.target_id)
                # Если не нашли среди существ, проверяем героя
                if not target_obj and str(opponent.hero.instance_id) == action.target_id:
                    target_obj = opponent.hero
            
            # Применяем эффекты зелья
            process_effects(self.state, card, player, opponent, action.target_id)
            # Зелье не остается на доске, отправляем в сброс
            player.graveyard.append(card)

        # Убираем карту из руки
        player.hand.pop(action.hand_index)

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

        # Проверка атаки
        if attacker.attack <= 0:
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

        # Вычисляем эффективную атаку с учетом аур
        effective_attack = self._apply_aura_bonuses(attacker, player.board)
        
        # Обработка атаки
        if action.target_is_hero:
            # Атака героя
            damage_dealt = apply_damage(opponent.hero, effective_attack, attacker)
            
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
            target_effective_attack = self._apply_aura_bonuses(target, opponent.board)

            # Обмен ударами (с передачей атакующего для reflect)
            damage_dealt = apply_damage(target, effective_attack, attacker)
            apply_damage(attacker, target_effective_attack, target)
            
            # Применяем Lifesteal
            if damage_dealt > 0:
                apply_lifesteal(attacker, damage_dealt, player.hero)
            
            # Проверка instant_kill
            if "instant_kill" in attacker.mechanics and target.hp > 0:
                target.hp = 0
                logger.debug("[CORE] INSTANT KILL: %s убил %s мгновенно!", attacker.name, target.name)
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

        # Восстанавливаем ману противнику
        opponent.max_mana = min(10, opponent.max_mana + self.mana_per_turn)
        opponent.mana = opponent.max_mana

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

            # Проверка регенерации regen_X
            for mechanic in unit.mechanics:
                if mechanic.startswith("regen_"):
                    try:
                        regen_amount = int(mechanic[6:])  # Убираем "regen_"
                        old_hp = unit.hp
                        unit.hp = min(unit.max_hp, unit.hp + regen_amount)
                        logger.debug(
                            "[CORE] %s регенерирует %d HP (%d -> %d)",
                            unit.name,
                            regen_amount,
                            old_hp,
                            unit.hp,
                        )
                        break  # Обрабатываем только первую регенерацию
                    except (ValueError, IndexError):
                        logger.warning("[CORE] Некорректный формат regen: %s", mechanic)

        # ЧЕСТНЫЙ ЦИКЛ КОЛОДЫ: Противник тянет карту (лимит руки: 4 карты)
        # Если колода пуста, перемешиваем сброс обратно
        if not opponent.deck:
            if opponent.graveyard:
                logger.info(
                    "[RESHUFFLE] Колода закончилась! Сброс замешан обратно. (%s: %d карт)",
                    opponent.user_id,
                    len(opponent.graveyard)
                )
                # Сбрасываем состояние карт до базового
                for card in opponent.graveyard:
                    card.hp = card.max_hp
                    card.is_ready = False
                    card.is_frozen = False
                
                # Перемешиваем и возвращаем в колоду
                random.shuffle(opponent.graveyard)
                opponent.deck = opponent.graveyard[:]
                opponent.graveyard.clear()
            else:
                # Fatigue: и колода, и сброс пусты - карта не берется
                logger.info(
                    "[FATIGUE] %s не может взять карту - колода и сброс пусты",
                    opponent.user_id
                )
                return
        
        # Берем карту из колоды (честный pop)
        if len(opponent.hand) < 4:
            drawn_card = opponent.deck.pop(0)
            opponent.hand.append(drawn_card)
        else:
            # Карта сгорает при переполнении руки
            burned_card = opponent.deck.pop(0)
            logger.info(
                "[BURN] Hand limit reached for %s, card %s destroyed",
                opponent.user_id,
                burned_card.name,
            )

    def _cleanup_dead_units(self, player: PlayerState) -> None:
        """Удалить мертвых существ с доски игрока, активируя Deathrattle."""
        # Определяем противника для Deathrattle эффектов
        if player.user_id == self.state.p1.user_id:
            opponent = self.state.p2
            log_type = "player"
        else:
            opponent = self.state.p1
            log_type = "opponent"
        
        # Обрабатываем Deathrattle ПЕРЕД удалением
        dead_units = [unit for unit in player.board if unit.hp <= 0]
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
                        # Наносим урон ВСЕМ вражеским юнитам
                        for enemy_unit in opponent.board:
                            apply_damage(enemy_unit, damage)
                        # Наносим урон вражескому герою
                        apply_damage(opponent.hero, damage)

                        # Добавляем в лог
                        self.state.action_history.append((log_type, f"{unit.name} взрывается после смерти и наносит {damage} урона всем врагам!"))
                        self.state.action_history = self.state.action_history[-100:]
                    break  # Обрабатываем только первый deathrattle
            
            # Отправляем мертвое существо в сброс
            player.graveyard.append(unit)
        
        # Удаляем мертвых
        player.board = [unit for unit in player.board if unit.hp > 0]

    def _check_game_over(self) -> None:
        """Проверить условие окончания игры."""
        if self.state.p1.hero.hp <= 0:
            self.state.status = GameStatus.P2_WIN
        elif self.state.p2.hero.hp <= 0:
            self.state.status = GameStatus.P1_WIN
    
    def apply_start_game_effects(self) -> None:
        """
        Применить стартовые эффекты героев при инициализации игры.
        Обрабатывает механики типа start_mana_X (Капитулюга).
        """
        # Обрабатываем героя P1
        for mechanic in self.state.p1.hero.mechanics:
            if mechanic.startswith("start_mana_"):
                try:
                    mana_bonus = int(mechanic[11:])  # Убираем "start_mana_"
                    self.state.p1.mana = min(10, self.state.p1.mana + mana_bonus)
                    logger.debug(
                        "[CORE] Герой P1 получает стартовую ману: +%d (итого: %d)",
                        mana_bonus,
                        self.state.p1.mana,
                    )
                except ValueError:
                    logger.warning(
                        "[CORE] Некорректный формат start_mana у героя P1: %s",
                        mechanic,
                    )
        
        # Обрабатываем героя P2
        for mechanic in self.state.p2.hero.mechanics:
            if mechanic.startswith("start_mana_"):
                try:
                    mana_bonus = int(mechanic[11:])  # Убираем "start_mana_"
                    self.state.p2.mana = min(10, self.state.p2.mana + mana_bonus)
                    logger.debug(
                        "[CORE] Герой P2 получает стартовую ману: +%d (итого: %d)",
                        mana_bonus,
                        self.state.p2.mana,
                    )
                except ValueError:
                    logger.warning(
                        "[CORE] Некорректный формат start_mana у героя P2: %s",
                        mechanic,
                    )

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
    
    def _get_aura_bonus(self, board: List[CardInstance], unit: CardInstance) -> int:
        """
        Получить бонус атаки от аур на доске.
        
        Args:
            board: Доска игрока
            unit: Юнит, для которого считается бонус
            
        Returns:
            Суммарный бонус атаки от всех аур
        """
        bonus = 0
        for aura_unit in board:
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
    
    def _apply_aura_bonuses(self, attacker: CardInstance, board: List[CardInstance]) -> int:
        """
        Получить эффективную атаку юнита с учетом аур.
        
        Args:
            attacker: Атакующий юнит
            board: Доска игрока
            
        Returns:
            Эффективная атака с бонусами
        """
        base_attack = attacker.attack
        aura_bonus = self._get_aura_bonus(board, attacker)
        return base_attack + aura_bonus

    def get_preview_delta(self, action: BaseAction) -> Dict[str, int]:
        """
        Симулировать действие и вернуть изменения HP объектов.
        
        Args:
            action: Действие для симуляции
            
        Returns:
            Словарь {instance_id: delta_hp}, где delta_hp - изменение HP (отрицательное для урона)
        """
        import copy
        
        # Создаем временную копию состояния
        temp_state = copy.deepcopy(self.state)
        temp_env = ArenaEnvironment(temp_state)
        
        # Сохраняем HP до действия
        hp_before: Dict[str, int] = {}
        
        # Собираем HP всех объектов
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
        
        # Собираем HP после действия
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
        if self.state.p1.user_id == player_id:
            player = self.state.p1
            opponent = self.state.p2
        else:
            player = self.state.p2
            opponent = self.state.p1

        # 1. Действия розыгрыша карт из руки
        for hand_idx, card in enumerate(player.hand):
            # Проверка маны
            if player.mana < card.mana_cost:
                continue

            # Для существ
            if card.card_type == CardType.WARRIOR:
                # Проверка места на доске
                if len(player.board) >= 7:
                    continue

                # Проверка choose_shield_damage: может быть с целью ИЛИ без
                has_choose_shield_damage = "choose_shield_damage" in card.mechanics
                
                # Если требуется цель (Battlecry)
                if requires_target(card.mechanics) or has_choose_shield_damage:
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
            if not unit.is_ready or unit.attack <= 0:
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

        # Урон, удаление или заморозка с уроном - вражеские цели + герой
        if is_damage or is_delete or is_freeze:
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
            match = re.match(r"battlecry_damage_(\d+)", mechanic)
            if match:
                damage = match.group(1)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"наносит {damage} урона по {target_name}"
        
        # Battlecry Heal
        for mechanic in mechanics:
            match = re.match(r"battlecry_heal_(\d+)", mechanic)
            if match:
                heal = match.group(1)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"исцелил {target_name} (+{heal} ХП)"
        
        # Battlecry Heal Target (Targeted Heal)
        for mechanic in mechanics:
            match = re.match(r"battlecry_heal_target_(\d+)", mechanic)
            if match:
                heal = match.group(1)
                target_name = self._get_target_name(target_id, owner, opponent)
                return f"исцеляет {target_name} на {heal} HP"
        
        # Battlecry Heal Hero
        for mechanic in mechanics:
            match = re.match(r"battlecry_heal_hero_(\d+)", mechanic)
            if match:
                heal = match.group(1)
                return f"исцелил вашего героя (+{heal} ХП)"
        
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
        if "aoe_freeze" in mechanics:
            return "заморозил всех вражеских существ на 1 ход"
        
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
                return f"исцелил {target_name} (+{heal} ХП)"
        
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
            match = re.match(r"cleave_(\d+)_(\d+)", mechanic)
            if match:
                damage, times = match.group(1), match.group(2)
                return f"наносит {damage} урона {times} случайным целям"
        
        # Cast Random Spell
        if "cast_random_spell" in mechanics:
            return "применил случайное заклинание"
        
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
        if "instant_kill" in mechanics:
            return "мгновенно убивает цель при атаке"
        
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
                    passives.append(f"Старт: +{amount} маны")
                    break
                except:
                    pass
        
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
        effective_attack = self._apply_aura_bonuses(attacker, player.board) if attacker else 0
        
        if action.target_is_hero:
            # Атака героя
            text = f"{attacker_name} наносит {effective_attack} урона по Герою"
        else:
            # Атака существа
            target = self._find_unit_by_id(opponent.board, action.target_id)
            target_name = target.name if target else "существо"
            text = f"{attacker_name} атакует {target_name}"
        
        # Проверка Lifesteal
        if attacker and "lifesteal" in attacker.mechanics and effective_attack > 0:
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
