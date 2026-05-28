"""
Конвертер данных из БД в игровые структуры.
Связь между infrastructure/database.py и core/state.py.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from uuid import uuid4

from core.state import CardInstance, CardType


def _get_tiered_value(base_value: int, level: int) -> int:
    """
    Рассчитать значение механики с учетом уровня (по уровням).
    
    Args:
        base_value: Базовое значение механики
        level: Уровень карты (1-10)
        
    Returns:
        Итоговое значение с бонусом от уровня
    """
    # Lvl 1-3: +0, Lvl 4-6: +1, Lvl 7-9: +2, Lvl 10: +3
    tier_bonus = (level - 1) // 3
    return base_value + tier_bonus


def _normalize_mechanic(mechanic: str, level: int) -> str:
    """
    Нормализовать механику с диапазоном в одно значение на основе уровня.
    Применяет tiered scaling к числовым значениям механик.
    
    Args:
        mechanic: Механика (например, aura_atk_1_3, damage_5, start_mana_1_5)
        level: Уровень карты (1-10)
        
    Returns:
        Нормализованная механика с учетом уровня
    """
    # Обрабатываем параметрические механики с диапазоном или числовым значением
    parametric_prefixes = [
        "aura_atk_", "reflect_", "armor_", "regen_", 
        "battlecry_damage_", "battlecry_heal_", "battlecry_buff_",
        "damage_", "heal_", "buff_all_",
        "start_mana_"
    ]
    
    # Специальные механики без скалирования
    no_scaling_prefixes = ["summon_"]
    
    # Проверяем механики без скалирования (например, summon_ID)
    for prefix in no_scaling_prefixes:
        if mechanic.startswith(prefix):
            return mechanic  # Возвращаем как есть
    
    for prefix in parametric_prefixes:
        if mechanic.startswith(prefix):
            # Убираем префикс
            value_part = mechanic[len(prefix):]
            
            # Проверяем наличие диапазона (X_Y)
            if "_" in value_part:
                parts = value_part.split("_")
                if len(parts) == 2:
                    try:
                        min_val = int(parts[0])
                        max_val = int(parts[1])
                        
                        # start_mana: engine hero scaling добавляет bonus_tiers
                        if prefix == "start_mana_":
                            return f"{prefix}{min_val}"
                        
                        # buff_all_X_Y, battlecry_buff_X_Y: engine warrior scaling handles buff
                        elif prefix in ["buff_all_", "battlecry_buff_"]:
                            return f"{prefix}{min_val}_{max_val}"
                        
                        else:
                            # Для остальных механик с диапазоном: берём min_val
                            return f"{prefix}{min_val}"
                    except ValueError:
                        # Если не удалось распарсить, возвращаем как есть
                        pass
            else:
                # Одиночное числовое значение — возвращаем как есть (scaling в engine)
                try:
                    int(value_part)
                    return mechanic
                except ValueError:
                    pass
            
            # Если не удалось обработать, возвращаем как есть
            return mechanic
    
    return mechanic


def card_from_db(
    card_data: Dict[str, Any],
    level: int = 1,
) -> CardInstance:
    """
    Конвертировать данные карты из БД в CardInstance.
    Применяет scaling статов на основе уровня карты (1-10) через scale_card_by_level.
    
    Args:
        card_data: Данные карты из БД (dict с полями id, name, rarity, etc.)
        level: Уровень карты (1-10, из таблицы user_cards)
        
    Returns:
        CardInstance для использования в движке
    """
    # Извлекаем базовые данные
    card_id = card_data.get("id", 0)
    name = card_data.get("name", "Unknown")
    description = card_data.get("description", "")
    mechanics_desc = card_data.get("mechanics_desc", "") or ""
    rarity = card_data.get("rarity", "common")
    card_type_str = card_data.get("card_type", "warrior")
    simplified_levelup = bool(card_data.get("simplified_levelup", False))
    
    # Конвертируем тип карты
    if card_type_str == "hero":
        card_type = CardType.HERO
    elif card_type_str == "potion":
        card_type = CardType.POTION
    else:
        card_type = CardType.WARRIOR
    
    # Извлекаем базовые статы (БЕЗ масштабирования - это делает scale_card_by_level)
    base_attack = card_data.get("current_attack", card_data.get("base_attack", 0))
    base_hp = card_data.get("current_hp", card_data.get("base_hp", 0))
    mana_cost = card_data.get("mana_cost", 0)
    
    # Извлекаем механики
    mechanics_raw = card_data.get("mechanics", [])
    if isinstance(mechanics_raw, list):
        mechanics = mechanics_raw
    elif isinstance(mechanics_raw, str):
        try:
            parsed = json.loads(mechanics_raw)
            mechanics = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            mechanics = [mechanics_raw] if mechanics_raw else []
    else:
        mechanics = []
    
    # Нормализуем механики с учетом уровня (для диапазонов типа X_Y)
    normalized_mechanics = [_normalize_mechanic(m, level) for m in mechanics]
    
    # Создаем базовую CardInstance (уровень 1)
    card = CardInstance(
        instance_id=uuid4(),
        card_id=card_id,
        name=name,
        card_type=card_type,
        rarity=rarity,
        mana_cost=mana_cost,
        attack=base_attack,
        hp=base_hp,
        max_hp=base_hp,
        mechanics=normalized_mechanics,
        is_ready=False,
        description=description,
        mechanics_desc=mechanics_desc,
        level=1,  # Базовый уровень
        simplified_levelup=simplified_levelup,
    )
    
    # Применяем единое масштабирование, совпадающее с формулой коллекции.
    from core.card_scaling import scale_card_by_level
    return scale_card_by_level(card, level)


def deck_from_card_ids(
    card_ids: List[int],
    cards_data: Dict[int, Dict[str, Any]],
    user_levels: Dict[int, int] | None = None,
    slot_levels: List[int] | None = None,
) -> List[CardInstance]:
    """
    Создать колоду из списка ID карт.

    Args:
        card_ids: Список ID карт
        cards_data: Словарь {card_id: card_data} с данными всех карт
        user_levels: Словарь {card_id: level} с уровнями карт пользователя (игроки)
        slot_levels: Список уровней по позиции карты в колоде (боты, per-card offset).
                      Если передан, приоритет над user_levels для карт на этом индексе.

    Returns:
        Список CardInstance для колоды
    """
    deck: List[CardInstance] = []

    for idx, card_id in enumerate(card_ids):
        if card_id not in cards_data:
            continue

        card_data = cards_data[card_id]

        # Slot-based уровень имеет приоритет (для ботов с per-card offset)
        if slot_levels is not None and idx < len(slot_levels):
            level = int(slot_levels[idx])
        elif user_levels:
            level = user_levels.get(card_id, 1)
        else:
            level = 1

        card_instance = card_from_db(card_data, level)
        deck.append(card_instance)

    return deck
