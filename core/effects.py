"""
Эффекты и механики карт.
Реализация всех игровых механик (taunt, щит, баттлкрай и т.д.).
"""
from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from core.state import GameState, CardInstance, PlayerState


logger = logging.getLogger(__name__)


# Для механики summon (импорт будет выполнен при первом использовании)
_card_from_db = None


# ============================================================================
# EFFECT HANDLERS REGISTRY
# ============================================================================

EffectHandler = Callable[["GameState", "CardInstance", "PlayerState", "PlayerState", Optional[str]], None]

EFFECT_HANDLERS: Dict[str, EffectHandler] = {}


def register_effect(name: str) -> Callable[[EffectHandler], EffectHandler]:
    """Декоратор для регистрации обработчика эффекта."""
    def decorator(func: EffectHandler) -> EffectHandler:
        EFFECT_HANDLERS[name] = func
        return func
    return decorator


def record_card_feedback_event(
    state: "GameState",
    card: "CardInstance",
    *,
    mechanic: str,
    effect_code: str,
) -> None:
    events = getattr(state, "pending_card_feedback_events", None)
    if events is None:
        events = []
        state.pending_card_feedback_events = events
    events.append({
        "card_id": getattr(card, "card_id", None),
        "instance_id": str(getattr(card, "instance_id", "")),
        "mechanic": mechanic,
        "effect_code": effect_code,
    })


def is_random_battlecry_damage_card(card: "CardInstance") -> bool:
    """Cards opt into random battlecry damage through battlecry_damage_X_random."""
    return any(
        re.match(r"battlecry_damage_\d+_random$", mechanic)
        for mechanic in getattr(card, "mechanics", [])
    )


def _apply_random_battlecry_damage(
    card: "CardInstance",
    opponent: "PlayerState",
    damage_amount: int,
) -> None:
    targets = list(opponent.board) + [opponent.hero]
    if not targets:
        return

    target = random.choice(targets)
    apply_damage(target, damage_amount)
    logger.debug(
        "[EFFECTS] %s наносит %d случайного урона по %s",
        card.name,
        damage_amount,
        target.name,
    )


# ============================================================================
# PASSIVE MECHANICS (применяются постоянно)
# ============================================================================

@register_effect("taunt")
def effect_taunt(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """
    Taunt: противник должен атаковать это существо в первую очередь.
    Не требует активации, работает пассивно (проверяется в валидации атак).
    """
    pass  # Логика реализуется в движке при валидации целей атаки


@register_effect("shield")
def effect_shield(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """
    Shield: игнорирует первый полученный урон.
    Не требует активации, работает пассивно (проверяется при получении урона).
    """
    pass  # Логика реализуется в движке при обработке урона


# ============================================================================
# BATTLECRY MECHANICS (срабатывают при розыгрыше)
# ============================================================================

@register_effect("battlecry_damage_1")
def effect_battlecry_damage_1(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Battlecry: нанести 1 урон выбранной цели."""
    if not target_id:
        return
    
    # Ищем цель на доске противника
    for unit in opponent.board:
        if str(unit.instance_id) == target_id:
            apply_damage(unit, 1)
            return
    
    # Проверяем героя противника
    if str(opponent.hero.instance_id) == target_id:
        apply_damage(opponent.hero, 1)


@register_effect("battlecry_damage_2")
def effect_battlecry_damage_2(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Battlecry: нанести 2 урона выбранной цели."""
    if not target_id:
        return
    
    for unit in opponent.board:
        if str(unit.instance_id) == target_id:
            apply_damage(unit, 2)
            return
    
    if str(opponent.hero.instance_id) == target_id:
        apply_damage(opponent.hero, 2)


@register_effect("battlecry_damage_3")
def effect_battlecry_damage_3(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Battlecry: нанести 3 урона выбранной цели."""
    if not target_id:
        return
    
    for unit in opponent.board:
        if str(unit.instance_id) == target_id:
            apply_damage(unit, 3)
            return
    
    if str(opponent.hero.instance_id) == target_id:
        apply_damage(opponent.hero, 3)


@register_effect("battlecry_heal_2")
def effect_battlecry_heal_2(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Battlecry: восстановить 2 HP выбранной цели."""
    if not target_id:
        return
    
    # Ищем цель на своей доске
    for unit in owner.board:
        if str(unit.instance_id) == target_id:
            apply_heal(unit, 2)
            return
    
    # Проверяем своего героя
    if str(owner.hero.instance_id) == target_id:
        apply_heal(owner.hero, 2)


# ============================================================================
# TARGETED HEAL (выбор союзника для лечения)
# ============================================================================

def _register_battlecry_heal_target_effects():
    """Регистрирует динамические battlecry_heal_target эффекты."""
    for heal_amount in range(1, 11):  # battlecry_heal_target_1 до battlecry_heal_target_10
        def make_handler(heal: int):
            def handler(
                state: GameState,
                card: CardInstance,
                owner: PlayerState,
                opponent: PlayerState,
                target_id: Optional[str] = None,
            ) -> None:
                f"""Battlecry: восстановить {heal} HP выбранному союзнику или герою."""
                if not target_id:
                    return
                
                # Ищем цель на своей доске
                for unit in owner.board:
                    if str(unit.instance_id) == target_id:
                        apply_heal(unit, heal)
                        logger.debug(
                            "[EFFECTS] battlecry_heal_target_%d: %s восстановил %d HP",
                            heal,
                            unit.name,
                            heal,
                        )
                        return
                
                # Проверяем своего героя
                if str(owner.hero.instance_id) == target_id:
                    apply_heal(owner.hero, heal)
                    logger.debug(
                        "[EFFECTS] battlecry_heal_target_%d: герой восстановил %d HP",
                        heal,
                        heal,
                    )
            return handler
        
        EFFECT_HANDLERS[f"battlecry_heal_target_{heal_amount}"] = make_handler(heal_amount)


# Регистрируем battlecry_heal_target эффекты
_register_battlecry_heal_target_effects()


@register_effect("battlecry_draw_card")
def effect_battlecry_draw_card(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Battlecry: взять карту из колоды."""
    if owner.deck and len(owner.hand) < 4:  # Лимит руки - 4 карты
        drawn_card = owner.deck.pop(0)
        owner.hand.append(drawn_card)


@register_effect("battlecry_aoe_damage_1")
def effect_battlecry_aoe_damage_1(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Battlecry: нанести 1 урон всем вражеским существам."""
    for unit in opponent.board:
        apply_damage(unit, 1)


@register_effect("battlecry_aoe_damage_2")
def effect_battlecry_aoe_damage_2(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Battlecry: нанести 2 урона всем вражеским существам."""
    for unit in opponent.board:
        apply_damage(unit, 2)


# ============================================================================
# ДИНАМИЧЕСКИЕ BATTLECRY (парсятся из строки механики)
# ============================================================================

def _try_parse_battlecry_heal_hero(mechanic: str) -> Optional[int]:
    """Парсит battlecry_heal_hero_X и возвращает X."""
    match = re.match(r"battlecry_heal_hero_(\d+)", mechanic)
    if match:
        return int(match.group(1))
    return None


def _try_parse_battlecry_damage(mechanic: str) -> Optional[int]:
    """Парсит battlecry_damage_X и возвращает X (для динамических значений)."""
    match = re.match(r"battlecry_damage_(\d+)", mechanic)
    if match:
        return int(match.group(1))
    return None


# ============================================================================
# SPELL EFFECTS (прямые эффекты зелий)
# ============================================================================

@register_effect("spell_damage_3")
def effect_spell_damage_3(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Зелье: нанести 3 урона цели."""
    if not target_id:
        return
    
    for unit in opponent.board:
        if str(unit.instance_id) == target_id:
            apply_damage(unit, 3)
            return
    
    if str(opponent.hero.instance_id) == target_id:
        apply_damage(opponent.hero, 3)


@register_effect("spell_damage_5")
def effect_spell_damage_5(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Зелье: нанести 5 урона цели."""
    if not target_id:
        return
    
    for unit in opponent.board:
        if str(unit.instance_id) == target_id:
            apply_damage(unit, 5)
            return
    
    if str(opponent.hero.instance_id) == target_id:
        apply_damage(opponent.hero, 5)


@register_effect("spell_heal_5")
def effect_spell_heal_5(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Зелье: восстановить 5 HP цели."""
    if not target_id:
        return
    
    for unit in owner.board:
        if str(unit.instance_id) == target_id:
            apply_heal(unit, 5)
            return
    
    if str(owner.hero.instance_id) == target_id:
        apply_heal(owner.hero, 5)


@register_effect("spell_aoe_damage_2")
def effect_spell_aoe_damage_2(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Зелье: нанести 2 урона всем вражеским существам."""
    for unit in opponent.board:
        apply_damage(unit, 2)


# ============================================================================
# AOE EFFECTS (массовые эффекты)
# ============================================================================

def _register_aoe_damage_effects():
    """Регистрирует динамические AOE урон эффекты."""
    for damage in range(1, 11):  # aoe_damage_1 до aoe_damage_10
        def make_handler(dmg: int):
            def handler(
                state: GameState,
                card: CardInstance,
                owner: PlayerState,
                opponent: PlayerState,
                target_id: Optional[str] = None,
            ) -> None:
                f"""AOE: нанести {dmg} урона всем вражеским существам."""
                logger.debug("[EFFECTS] aoe_damage_%d: %s бьёт всех врагов (opponent=%s, targets=%d)", dmg, card.name, opponent.user_id, len(opponent.board))
                for unit in opponent.board:
                    apply_damage(unit, dmg)
            return handler
        
        EFFECT_HANDLERS[f"aoe_damage_{damage}"] = make_handler(damage)


# Регистрируем AOE урон эффекты
_register_aoe_damage_effects()


@register_effect("aoe_freeze")
def effect_aoe_freeze(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """AOE: заморозить до 3 вражеских существ."""
    for unit in opponent.board[:3]:
        if consume_shield(unit, "aoe_freeze"):
            continue
        if not unit.is_frozen:
            unit.is_frozen = True
            logger.debug("[EFFECTS] Юнит %s заморожен", unit.name)


@register_effect("desk_freeze")
def effect_desk_freeze(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Заморозить всех вражеских существ на доске."""
    for unit in opponent.board:
        if consume_shield(unit, "desk_freeze"):
            continue
        if not unit.is_frozen:
            unit.is_frozen = True
            logger.debug("[EFFECTS] Юнит %s заморожен desk_freeze", unit.name)


@register_effect("freeze")
def effect_freeze(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Заморозить выбранную цель (зелья). Героев замораживать нельзя."""
    if not target_id:
        return
    
    # Ищем цель на доске противника
    for unit in opponent.board:
        if str(unit.instance_id) == target_id:
            if consume_shield(unit, "freeze"):
                return
            if not unit.is_frozen:
                unit.is_frozen = True
                logger.debug("[EFFECTS] Юнит %s заморожен", unit.name)
            return


@register_effect("battlecry_freeze")
def effect_battlecry_freeze(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Battlecry: заморозить выбранную цель."""
    # Используем ту же логику, что и freeze
    effect_freeze(state, card, owner, opponent, target_id)


# ============================================================================
# MANA CONTROL (управление маной)
# ============================================================================

def _register_mana_effects():
    """Регистрирует динамические mana эффекты."""
    # mana_gain_X: восстановить X маны владельцу
    for gain in range(1, 11):
        def make_gain_handler(amount: int):
            def handler(
                state: GameState,
                card: CardInstance,
                owner: PlayerState,
                opponent: PlayerState,
                target_id: Optional[str] = None,
            ) -> None:
                f"""Восстановить {amount} маны владельцу."""
                owner.mana = min(owner.max_mana, owner.mana + amount)
                logger.debug("[EFFECTS] Игрок %s восстановил %d маны", owner.user_id, amount)
            return handler
        
        EFFECT_HANDLERS[f"mana_gain_{gain}"] = make_gain_handler(gain)
    
    # mana_drain_X: отнять X маны у противника и передать владельцу
    for drain in range(1, 11):
        def make_drain_handler(amount: int):
            def handler(
                state: GameState,
                card: CardInstance,
                owner: PlayerState,
                opponent: PlayerState,
                target_id: Optional[str] = None,
            ) -> None:
                f"""Отнять {amount} маны у противника и передать владельцу."""
                current_drained = min(opponent.mana, amount)
                opponent.mana -= current_drained

                pending_drains = getattr(state, "pending_mana_drain_by_player", None)
                if pending_drains is None:
                    pending_drains = {}
                    state.pending_mana_drain_by_player = pending_drains

                existing_pending = pending_drains.get(opponent.user_id, 0)
                future_pool = max(0, opponent.max_mana - existing_pending)
                future_drained = min(amount - current_drained, future_pool)
                if future_drained > 0:
                    pending_drains[opponent.user_id] = existing_pending + future_drained

                drained = current_drained + future_drained
                owner.mana = min(owner.max_mana, owner.mana + drained)
                logger.debug(
                    "[EFFECTS] mana_drain: Игрок %s потерял %d маны сейчас и %d следующим ходом, игрок %s получил %d маны (стало=%d/%d)",
                    opponent.user_id,
                    current_drained,
                    future_drained,
                    owner.user_id,
                    drained,
                    owner.mana,
                    owner.max_mana,
                )
            return handler
        
        EFFECT_HANDLERS[f"mana_drain_{drain}"] = make_drain_handler(drain)


# Регистрируем mana эффекты
_register_mana_effects()


# ============================================================================
# BUFF MECHANICS (усиление существ)
# ============================================================================

def _register_buff_effects():
    """Регистрирует динамические buff эффекты."""
    # buff_all_X_Y: дать всем союзным существам +X атаки и +Y HP
    for atk_bonus in range(1, 11):
        for hp_bonus in range(1, 11):
            def make_buff_all_handler(atk: int, hp: int):
                def handler(
                    state: GameState,
                    card: CardInstance,
                    owner: PlayerState,
                    opponent: PlayerState,
                    target_id: Optional[str] = None,
                ) -> None:
                    f"""Buff All: дать всем союзным существам +{atk}/+{hp}."""
                    for unit in owner.board:
                        # Не баффаем саму карту
                        if unit.instance_id == card.instance_id:
                            continue
                        
                        unit.attack += atk
                        unit.max_hp += hp
                        unit.hp += hp
                        logger.debug(
                            "[EFFECTS] %s получил бафф +%d/+%d (теперь %d/%d)",
                            unit.name,
                            atk,
                            hp,
                            unit.attack,
                            unit.hp,
                        )
                return handler
            
            EFFECT_HANDLERS[f"buff_all_{atk_bonus}_{hp_bonus}"] = make_buff_all_handler(atk_bonus, hp_bonus)
    
    # battlecry_buff_X_Y: дать выбранной цели +X атаки и +Y HP
    for atk_bonus in range(1, 11):
        for hp_bonus in range(1, 11):
            def make_battlecry_buff_handler(atk: int, hp: int):
                def handler(
                    state: GameState,
                    card: CardInstance,
                    owner: PlayerState,
                    opponent: PlayerState,
                    target_id: Optional[str] = None,
                ) -> None:
                    f"""Battlecry: дать цели +{atk}/+{hp}."""
                    if not target_id:
                        return
                    
                    # Ищем цель на доске владельца
                    for unit in owner.board:
                        if str(unit.instance_id) == target_id:
                            unit.attack += atk
                            unit.max_hp += hp
                            unit.hp += hp
                            logger.debug(
                                "[EFFECTS] %s получил battlecry бафф +%d/+%d",
                                unit.name,
                                atk,
                                hp,
                            )
                            return
                return handler
            
            EFFECT_HANDLERS[f"battlecry_buff_{atk_bonus}_{hp_bonus}"] = make_battlecry_buff_handler(atk_bonus, hp_bonus)


# Регистрируем buff эффекты
_register_buff_effects()


# ============================================================================
# SUMMON MECHANICS (призыв существ)
# ============================================================================

def _lazy_import_card_from_db():
    """Ленивый импорт card_from_db для избежания циклических зависимостей."""
    global _card_from_db
    if _card_from_db is None:
        from core.converter import card_from_db
        _card_from_db = card_from_db
    return _card_from_db


@register_effect("summon")
def effect_summon_generic(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """
    Призыв существа. Парсит ID из механик card и создает новое существо.
    Формат: ищем в mechanics строки вида "summon_ID" где ID - числовой идентификатор карты.
    """
    # Ищем механику summon_X
    summon_id = None
    for mechanic in card.mechanics:
        if mechanic.startswith("summon_") and mechanic != "summon":
            try:
                summon_id = int(mechanic[7:])  # Убираем "summon_"
                break
            except ValueError:
                logger.warning("[EFFECTS] Некорректный формат summon: %s", mechanic)
    
    if summon_id is None:
        logger.warning("[EFFECTS] Механика summon без корректного ID")
        return
    
    # Проверка места на доске
    if len(owner.board) >= 7:
        logger.debug("[EFFECTS] Не удалось призвать существо - доска полная")
        return
    
    # Получаем card_from_db через ленивый импорт
    card_from_db = _lazy_import_card_from_db()
    
    # TODO: В реальной игре нужно получить card_data из БД
    # Здесь используем заглушку - создаем базовое существо
    # В production версии это должно быть:
    # from infrastructure.database import get_card_by_id
    # card_data = get_card_by_id(summon_id)
    
    # Временная заглушка для тестирования
    logger.info("[EFFECTS] Summon механика вызвана для card_id=%d (требуется интеграция с БД)", summon_id)


# ============================================================================
# CLEAVE MECHANICS (множественный случайный урон)
# ============================================================================

def _register_cleave_effects():
    """Регистрирует динамические cleave эффекты."""
    # cleave_X_Y: нанести X урона Y случайным целям
    for damage in range(1, 11):
        for times in range(1, 6):  # До 5 случайных ударов
            def make_cleave_handler(dmg: int, hits: int):
                def handler(
                    state: GameState,
                    card: CardInstance,
                    owner: PlayerState,
                    opponent: PlayerState,
                    target_id: Optional[str] = None,
                ) -> None:
                    f"""Cleave: нанести {dmg} урона {hits} случайным целям."""
                    for _ in range(hits):
                        # Обновляем список живых целей перед каждым ударом
                        alive_targets = [u for u in opponent.board if u.hp > 0]
                        if not alive_targets:
                            break  # Нет живых целей
                        
                        # Выбираем случайную цель
                        target = random.choice(alive_targets)
                        apply_damage(target, dmg)
                        logger.debug(
                            "[EFFECTS] Cleave удар %d урона по %s (осталось %d HP)",
                            dmg,
                            target.name,
                            target.hp,
                        )
                return handler
            
            EFFECT_HANDLERS[f"cleave_{damage}_{times}"] = make_cleave_handler(damage, times)


# Регистрируем cleave эффекты
_register_cleave_effects()


# ============================================================================
# DEATHRATTLE MECHANICS (эффекты при смерти)
# ============================================================================

def _register_deathrattle_aoe_damage_effects():
    """Регистрирует динамические deathrattle_aoe_damage эффекты."""
    for damage in range(1, 11):  # deathrattle_aoe_damage_1 до deathrattle_aoe_damage_10
        def make_handler(dmg: int):
            def handler(
                state: GameState,
                card: CardInstance,
                owner: PlayerState,
                opponent: PlayerState,
                target_id: Optional[str] = None,
            ) -> None:
                f"""Deathrattle: нанести {dmg} урона всем вражеским существам И герою."""
                # Урон всем живым вражеским юнитам
                for unit in opponent.board:
                    if unit.hp > 0:
                        apply_damage(unit, dmg)
                # Урон вражескому герою
                if opponent.hero.hp > 0:
                    apply_damage(opponent.hero, dmg)
                logger.debug(
                    "[EFFECTS] Deathrattle: %s нанес %d урона всем вражеским существам и герою",
                    card.name,
                    dmg,
                )
            return handler
        
        EFFECT_HANDLERS[f"deathrattle_aoe_damage_{damage}"] = make_handler(damage)


# Регистрируем deathrattle эффекты
_register_deathrattle_aoe_damage_effects()


# ============================================================================
# DELETE TARGET (мгновенное удаление)
# ============================================================================

@register_effect("delete_target")
def effect_delete_target(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """Мгновенно удалить цель с доски без нанесения урона."""
    if not target_id:
        return
    
    # Ищем цель на доске противника
    for unit in opponent.board[:]:  # Копия списка для безопасного удаления
        if str(unit.instance_id) == target_id:
            if consume_shield(unit, "delete_target"):
                return
            opponent.board.remove(unit)
            opponent.graveyard.append(unit)
            logger.debug("[EFFECTS] %s мгновенно удален с доски", unit.name)
            return


# ============================================================================
# RANDOM SPELL (случайное зелье)
# ============================================================================

@register_effect("cast_random_spell")
def effect_cast_random_spell(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """
    Мидория: разыграть случайное заклинание с масштабированием по уровню.
    
    Варианты (1 из 4):
    1. Texas Smash: Урон случайному врагу (4 + level - 1)
    2. Recovery: Хил своему герою (5 + level - 1)
    3. Blackwhip: Заморозка врагов (1 враг, или 2 если level >= 5)
    4. Full Cowl: Бафф себе (+2 + (level-1)//2 к Атаке и HP)
    """
    level = card.level
    
    # Выбираем случайный вариант (1-4)
    spell_choice = random.randint(1, 4)
    
    # Определяем тип лога
    log_type = "player" if owner.user_id == state.p1.user_id else "opponent"
    
    if spell_choice == 1:
        record_card_feedback_event(
            state,
            card,
            mechanic="cast_random_spell",
            effect_code="midoriya_texas_smash",
        )
        # Texas Smash: Урон случайному врагу
        dmg = 4 + (level - 1)
        
        # Собираем все возможные цели (юниты + герой)
        possible_targets = opponent.board[:] + [opponent.hero]
        
        if possible_targets:
            target = random.choice(possible_targets)
            apply_damage(target, dmg)
            state.action_history.append((
                log_type,
                f"⚡ {card.name} применяет «Техасский удар»: {dmg} урона по {target.name}!"
            ))
            logger.debug("[EFFECTS] Мидория (lvl %d): Texas Smash %d урона по %s", level, dmg, target.name)
        else:
            logger.debug("[EFFECTS] Мидория: Texas Smash - нет целей")
    
    elif spell_choice == 2:
        record_card_feedback_event(
            state,
            card,
            mechanic="cast_random_spell",
            effect_code="midoriya_recovery",
        )
        # Recovery: Хил своему герою
        heal = 5 + (level - 1)
        apply_heal(owner.hero, heal)
        state.action_history.append((
            log_type,
            f"⚡ {card.name} применяет «Восстановление»: +{heal} здоровья герою!"
        ))
        logger.debug("[EFFECTS] Мидория (lvl %d): Recovery %d HP", level, heal)
    
    elif spell_choice == 3:
        record_card_feedback_event(
            state,
            card,
            mechanic="cast_random_spell",
            effect_code="midoriya_blackwhip",
        )
        # Blackwhip: Заморозка врагов
        freeze_count = 2 if level >= 5 else 1
        
        # Собираем всех незамороженных врагов
        unfrozen_enemies = [u for u in opponent.board if not u.is_frozen]
        
        if unfrozen_enemies:
            # Замораживаем до freeze_count случайных врагов
            targets_to_freeze = random.sample(unfrozen_enemies, min(freeze_count, len(unfrozen_enemies)))
            
            for target in targets_to_freeze:
                if consume_shield(target, "blackwhip"):
                    continue
                target.is_frozen = True
            
            target_names = ", ".join([t.name for t in targets_to_freeze])
            state.action_history.append((
                log_type,
                f"⚡ {card.name} применяет «Чёрный кнут»: заморожены {target_names}!"
            ))
            logger.debug("[EFFECTS] Мидория (lvl %d): Blackwhip заморозил %d врагов", level, len(targets_to_freeze))
        else:
            logger.debug("[EFFECTS] Мидория: Blackwhip - нет незамороженных целей")
    
    else:  # spell_choice == 4
        record_card_feedback_event(
            state,
            card,
            mechanic="cast_random_spell",
            effect_code="midoriya_full_cowl",
        )
        # Full Cowl: Бафф себе
        buff = 2 + ((level - 1) // 2)
        card.attack += buff
        card.hp += buff
        card.max_hp += buff
        state.action_history.append((
            log_type,
            f"⚡ {card.name} применяет «Полный покров»: получает +{buff} атаки и +{buff} здоровья!"
        ))
        logger.debug("[EFFECTS] Мидория (lvl %d): Full Cowl +%d/+%d (теперь %d/%d)", 
                    level, buff, buff, card.attack, card.hp)


# ============================================================================
# CHOOSE MECHANICS (выбор действия)
# ============================================================================

@register_effect("choose_shield_damage")
def effect_choose_shield_damage(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """
    Геральт: выбор между щитом и уроном.
    Если target_id передан - наносим 3 урона, иначе даем себе щит.
    """
    if target_id:
        # Выбор: нанести урон
        for unit in opponent.board:
            if str(unit.instance_id) == target_id:
                apply_damage(unit, 3)
                logger.debug("[EFFECTS] Геральт наносит 3 урона %s", unit.name)
                return
        
        # Проверяем героя
        if str(opponent.hero.instance_id) == target_id:
            apply_damage(opponent.hero, 3)
            logger.debug("[EFFECTS] Геральт наносит 3 урона герою")
    else:
        # Выбор: получить щит
        if "shield" not in card.mechanics:
            card.mechanics.append("shield")
            logger.debug("[EFFECTS] Геральт получает щит")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_damage_modifiers(target: CardInstance, damage: int) -> int:
    """
    Применить модификаторы урона к цели.
    
    Args:
        target: Цель атаки
        damage: Исходный урон
        
    Returns:
        Модифицированный урон после применения щита, брони и т.д.
    """
    # Проверка permanent_shield - блокирует весь урон навсегда
    if "permanent_shield" in target.mechanics:
        logger.debug("[EFFECTS] Permanent Shield поглотил %d урона", damage)
        return 0
    
    # Проверка обычного щита - блокирует весь урон один раз
    if consume_shield(target, f"{damage} урона"):
        return 0
    
    # Проверка брони - уменьшает урон
    armor_value = 0
    armor_mechanic_to_remove = None
    
    for mechanic in target.mechanics:
        # Паттерн armor_X или armor_X_Y
        if mechanic.startswith("armor_"):
            armor_mechanic_to_remove = mechanic
            # Парсим значение брони
            armor_part = mechanic[6:]  # Убираем "armor_"
            
            if "_" in armor_part:
                # Диапазон armor_X_Y
                try:
                    min_val, max_val = armor_part.split("_")
                    armor_value = random.randint(int(min_val), int(max_val))
                except (ValueError, IndexError):
                    logger.warning("[EFFECTS] Некорректный формат брони: %s", mechanic)
            else:
                # Фиксированное значение armor_X
                try:
                    armor_value = int(armor_part)
                except ValueError:
                    logger.warning("[EFFECTS] Некорректное значение брони: %s", mechanic)
            
            break  # Обрабатываем только первую найденную броню
    
    # Применяем броню
    if armor_value > 0:
        damage = max(0, damage - armor_value)
        logger.debug("[EFFECTS] Броня поглотила %d урона, осталось %d", armor_value, damage)
    
    return damage


def consume_shield(target: CardInstance, source: str = "") -> bool:
    """Снять одноразовый щит, если он блокирует входящий эффект."""
    if "shield" not in target.mechanics:
        return False

    mechanics = list(target.mechanics)
    mechanics.remove("shield")
    target.mechanics = mechanics
    if source:
        logger.debug("[EFFECTS] Щит %s поглотил эффект: %s", target.name, source)
    else:
        logger.debug("[EFFECTS] Щит %s поглотил эффект", target.name)
    return True


def apply_damage(target: CardInstance, damage: int, attacker: Optional[CardInstance] = None) -> int:
    """
    Нанести урон существу/герою.
    
    Args:
        target: Цель атаки
        damage: Урон
        attacker: Атакующий (для reflect урона и lifesteal)
        
    Returns:
        Фактически нанесенный урон (после модификаторов)
    """
    # Применяем модификаторы урона
    modified_damage = apply_damage_modifiers(target, damage)
    old_hp = target.hp

    target.hp -= modified_damage
    if target.hp < 0:
        target.hp = 0
    actual_damage = max(0, old_hp - target.hp)
    
    # Проверка reflect_X - отражаем урон атакующему
    if attacker and actual_damage > 0:
        for mechanic in target.mechanics:
            if mechanic.startswith("reflect_"):
                try:
                    reflect_amount = int(mechanic[8:])  # Убираем "reflect_"
                    # Наносим reflect урон атакующему (без рекурсии)
                    reflect_damage = apply_damage_modifiers(attacker, reflect_amount)
                    attacker.hp -= reflect_damage
                    if attacker.hp < 0:
                        attacker.hp = 0
                    logger.debug(
                        "[EFFECTS] %s отразил %d урона на %s",
                        target.name,
                        reflect_amount,
                        attacker.name,
                    )
                    break  # Обрабатываем только первый reflect
                except (ValueError, IndexError):
                    logger.warning("[EFFECTS] Некорректный формат reflect: %s", mechanic)
    
    return actual_damage


def apply_heal(target: CardInstance, heal: int) -> None:
    """
    Восстановить HP существу/герою.
    КРИТИЧНО: HP не может превышать max_hp (перехил запрещен).
    """
    old_hp = target.hp
    # Ограничиваем HP максимальным значением
    target.hp = min(target.max_hp, target.hp + heal)
    
    if old_hp < target.hp:
        logger.debug(
            "[CORE] Лечение: %s восстановлено %d HP (%d -> %d, max: %d)",
            target.name,
            target.hp - old_hp,
            old_hp,
            target.hp,
            target.max_hp,
        )


def apply_lifesteal(attacker: CardInstance, damage_dealt: int, owner_hero: CardInstance) -> None:
    """
    Применить эффект Lifesteal: восстановить HP герою владельца атакующего.
    
    Args:
        attacker: Атакующее существо
        damage_dealt: Урон, который был нанесен
        owner_hero: Герой владельца атакующего
    """
    if "lifesteal" in attacker.mechanics and damage_dealt > 0:
        apply_heal(owner_hero, damage_dealt)
        logger.debug(
            "[EFFECTS] %s восстановил %d HP герою через Lifesteal",
            attacker.name,
            damage_dealt,
        )


def process_effects(
    state: GameState,
    card: CardInstance,
    owner: PlayerState,
    opponent: PlayerState,
    target_id: Optional[str] = None,
) -> None:
    """
    Обработать все эффекты карты при её активации.
    Поддерживает динамический парсинг механик через RegEx.
    
    Args:
        state: Текущее состояние игры
        card: Карта, эффекты которой обрабатываются
        owner: Владелец карты
        opponent: Противник
        target_id: ID цели (если требуется)
    """
    for mechanic in card.mechanics:
        # КРИТИЧНО: пропускаем deathrattle_* механики - они обрабатываются только в _cleanup_dead_units
        if mechanic.startswith("deathrattle_"):
            continue

        # КРИТИЧНО: cleave_* для WARRIOR - пассивная механика атаки (обрабатывается в _handle_attack)
        # Для POTION cleave_* остаётся battlecry-эффектом (random hits)
        if mechanic.startswith("cleave_") and card.card_type.value == "warrior":
            continue

        random_battlecry_damage = re.match(r"battlecry_damage_(\d+)_random$", mechanic)
        if random_battlecry_damage and is_random_battlecry_damage_card(card):
            _apply_random_battlecry_damage(
                card,
                opponent,
                int(random_battlecry_damage.group(1)),
            )
            continue
        
        # Пробуем зарегистрированные обработчики
        handler = EFFECT_HANDLERS.get(mechanic)
        if handler:
            logger.debug(
                "[CORE] process_effects: карта=%s механика=%s цель=%s владелец=%s",
                card.name,
                mechanic,
                target_id,
                owner.user_id,
            )
            handler(state, card, owner, opponent, target_id)
            continue
        
        # Динамический парсинг через RegEx
        
        # battlecry_heal_hero_X
        match = re.match(r"battlecry_heal_hero_(\d+)", mechanic)
        if match:
            heal_amount = int(match.group(1))
            logger.debug(
                "[CORE] process_effects: карта=%s battlecry_heal_hero_%d владелец=%s",
                card.name,
                heal_amount,
                owner.user_id,
            )
            apply_heal(owner.hero, heal_amount)
            continue
        
        # battlecry_damage_X или damage_X_Y (динамический урон)
        # КРИТИЧНО: игнорируем deathrattle_* механики (они обрабатываются только при смерти)
        if mechanic.startswith("deathrattle_"):
            continue
        
        match = re.match(r"(battlecry_|spell_)?damage_(\d+)(?:_(\d+))?", mechanic)
        if match and target_id:
            damage_amount = int(match.group(2))
            logger.debug(
                "[CORE] process_effects: карта=%s damage_%d цель=%s",
                card.name,
                damage_amount,
                target_id,
            )
            # Ищем цель на доске противника
            target_found = False
            for unit in opponent.board:
                if str(unit.instance_id) == target_id:
                    apply_damage(unit, damage_amount)
                    target_found = True
                    break
            
            # Проверяем героя противника
            if not target_found and str(opponent.hero.instance_id) == target_id:
                apply_damage(opponent.hero, damage_amount)
            continue
        
        # heal_target_X (динамическое лечение с выбором цели)
        match = re.match(r"(battlecry_|spell_)?heal_target_(\d+)", mechanic)
        if match and target_id:
            heal_amount = int(match.group(2))
            logger.debug(
                "[CORE] process_effects: карта=%s heal_target_%d цель=%s",
                card.name,
                heal_amount,
                target_id,
            )
            # Ищем цель на доске владельца
            target_found = False
            for unit in owner.board:
                if str(unit.instance_id) == target_id:
                    apply_heal(unit, heal_amount)
                    target_found = True
                    break
            
            # Проверяем героя владельца
            if not target_found and str(owner.hero.instance_id) == target_id:
                apply_heal(owner.hero, heal_amount)
            continue
        
        # heal_X (динамическое лечение)
        match = re.match(r"(battlecry_|spell_)?heal_(\d+)", mechanic)
        if match and target_id:
            heal_amount = int(match.group(2))
            logger.debug(
                "[CORE] process_effects: карта=%s heal_%d цель=%s",
                card.name,
                heal_amount,
                target_id,
            )
            # Ищем цель на доске владельца
            target_found = False
            for unit in owner.board:
                if str(unit.instance_id) == target_id:
                    apply_heal(unit, heal_amount)
                    target_found = True
                    break
            
            # Проверяем героя владельца
            if not target_found and str(owner.hero.instance_id) == target_id:
                apply_heal(owner.hero, heal_amount)
            continue
        
        # freeze_X (динамическая заморозка)
        match = re.match(r"freeze_(\d+)", mechanic)
        if match and target_id:
            logger.debug(
                "[CORE] process_effects: карта=%s freeze цель=%s",
                card.name,
                target_id,
            )
            # Ищем цель на доске противника
            for unit in opponent.board:
                if str(unit.instance_id) == target_id:
                    if consume_shield(unit, mechanic):
                        break
                    unit.is_frozen = True
                    break
            
            # Проверяем героя противника
            if str(opponent.hero.instance_id) == target_id:
                logger.debug("[CORE] freeze_X не применяется к героям: %s", opponent.hero.name)
            continue
        
        # reflect_X (динамическое отражение - пассивная механика, обрабатывается в apply_damage)
        match = re.match(r"reflect_(\d+)", mechanic)
        if match:
            # Пассивная механика, не требует активации
            continue


def has_taunt(board: List[CardInstance]) -> bool:
    """Проверить, есть ли на доске существа с Taunt."""
    return any("taunt" in unit.mechanics for unit in board)


def get_taunt_targets(board: List[CardInstance]) -> List[CardInstance]:
    """Получить всех существ с Taunt на доске."""
    return [unit for unit in board if "taunt" in unit.mechanics]


def requires_target(mechanics: List[str]) -> bool:
    """
    Проверить, требует ли карта цель для активации.
    КРИТИЧНО: choose_shield_damage НЕ требует обязательной цели (может выбрать щит).
    КРИТИЧНО: cast_random_spell НЕ требует цели (Мидория выбирает сам).
    
    Использует RegEx для динамических механик типа damage_X, heal_X, freeze_X.
    """
    # Проверяем механики, которые НЕ требуют цели
    if "cast_random_spell" in mechanics:
        return False
    
    # Проверяем статические механики
    static_targeting = [
        "delete_target",
        "consume_ally",
        "freeze",
        "battlecry_freeze",
    ]
    
    if any(m in mechanics for m in static_targeting):
        return True
    
    # Проверяем динамические механики через RegEx
    for mechanic in mechanics:
        # КРИТИЧНО: игнорируем deathrattle_* механики (они не требуют цели при розыгрыше)
        if mechanic.startswith("deathrattle_"):
            continue
        
        # damage_X или damage_X_Y (включая battlecry_damage_X)
        if re.match(r"(battlecry_)?damage_\d+(_\d+)?", mechanic):
            return True
        # heal_X (включая battlecry_heal_X, spell_heal_X, battlecry_heal_target_X)
        if re.match(r"(battlecry_|spell_)?heal(_target)?_\d+", mechanic):
            return True
        # spell_damage_X
        if re.match(r"spell_damage_\d+", mechanic):
            return True
        # battlecry_buff_X_Y
        if re.match(r"battlecry_buff_\d+_\d+", mechanic):
            return True
        # freeze_X (динамическая заморозка с параметром)
        if re.match(r"freeze_\d+", mechanic):
            return True
    
    return False
