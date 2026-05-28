"""
Единая формула расчета статов карт (коллекция + арена).

Формула роста: base * (1 + rarity_growth) ** (level - 1)
- Warriors/Potions: exponential growth based on rarity
- Heroes: +2 HP per level, no attack growth
- Mechanics scaling: handled separately per card_type by scale_card_by_level
"""
from __future__ import annotations

import logging
import math
import re
from typing import List

from core.state import CardInstance, CardType

logger = logging.getLogger(__name__)

RARITY_STATS: dict[str, float] = {
    "common": 0.10,
    "rare": 0.10,
    "superrare": 0.10,
    "epic": 0.11,
    "legendary": 0.12,
    "mythic": 0.12,
    "limited": 0.13,
    "divine": 0.15,
    "unique": 0.15,
}


def get_rarity_growth(rarity: str) -> float:
    return RARITY_STATS.get(str(rarity).lower(), 0.10)


def scale_base_stats(base_attack: int, base_hp: int, rarity_growth: float, level: int) -> tuple[int, int]:
    """
    Рассчитать scaled attack/hp по экспоненциальной формуле.
    base * (1 + growth) ** (level - 1), округление до целого.
    """
    safe_level = max(1, int(level or 1))
    if safe_level == 1:
        return base_attack, base_hp
    multiplier = (1 + rarity_growth) ** (safe_level - 1)
    scaled_attack = int(math.ceil(base_attack * multiplier))
    scaled_hp = int(math.ceil(base_hp * multiplier))
    return scaled_attack, scaled_hp


def scale_card_by_level(card: CardInstance, level: int) -> CardInstance:
    """
    Единая функция масштабирования карты для арены.
    Использует:
      - Экспоненциальный рост статов для WARRIOR: base * (1 + growth) ^ (level-1)
      - Экспоненциальный рост статов для POTION (только HP/attack, если есть)
      - HERO: +2 HP за уровень (без экспоненты)
      - Механики warrior/potion/hero масштабируются по правилам из engine

    Заменяет старую scale_card_by_level из engine.py.
    """
    if card.simplified_levelup:
        if level < 1:
            logger.warning("[SCALE] Некорректный simplified-уровень %d для карты %s, используется 1", level, card.name)
            level = 1
        elif level > 2:
            level = 2

        card.level = level
        if level == 2:
            card.mana_cost = max(0, card.mana_cost - 1)
        return card

    if level < 1 or level > 10:
        logger.warning("[SCALE] Некорректный уровень %d для карты %s, используется 1", level, card.name)
        level = 1

    card.level = level

    if level == 1:
        return card

    rarity_growth = get_rarity_growth(card.rarity)

    if card.card_type == CardType.WARRIOR:
        # Экспоненциальный рост статов
        scaled_atk, scaled_hp = scale_base_stats(card.attack, card.max_hp, rarity_growth, level)
        card.attack = scaled_atk
        card.hp = scaled_hp
        card.max_hp = scaled_hp

        # Scaling механик как раньше
        card.mechanics = _scale_warrior_mechanics(card.mechanics, level)

    elif card.card_type == CardType.POTION:
        scaled_mechanics = []
        for mechanic in card.mechanics:
            match = re.match(r"((?:spell_|battlecry_)?(?:aoe_)?damage_)(\d+)$", mechanic)
            if match:
                prefix = match.group(1)
                base_damage = int(match.group(2))
                new_damage = base_damage + ((level - 1) // 3)
                scaled_mechanics.append(f"{prefix}{new_damage}")
            else:
                scaled_mechanics.append(mechanic)

        card.mechanics = scaled_mechanics

    elif card.card_type == CardType.HERO:
        hp_bonus = (level - 1) * 2
        card.hp += hp_bonus
        card.max_hp += hp_bonus

        bonus_tiers = (level - 1) // 3
        if bonus_tiers > 0:
            scaled_mechanics = []
            for mechanic in card.mechanics:
                match = re.match(r"reflect_(\d+)", mechanic)
                if match:
                    scaled_mechanics.append(f"reflect_{int(match.group(1)) + bonus_tiers}")
                    continue
                match = re.match(r"armor_(\d+)", mechanic)
                if match:
                    scaled_mechanics.append(f"armor_{int(match.group(1)) + bonus_tiers}")
                    continue
                match = re.match(r"aura_atk_(\d+)", mechanic)
                if match:
                    scaled_mechanics.append(f"aura_atk_{int(match.group(1)) + bonus_tiers}")
                    continue
                match = re.match(r"start_mana_(\d+)", mechanic)
                if match:
                    scaled_mechanics.append(f"start_mana_{min(10, int(match.group(1)) + bonus_tiers)}")
                    continue
                match = re.match(r"regen_(\d+)", mechanic)
                if match:
                    scaled_mechanics.append(f"regen_{int(match.group(1)) + bonus_tiers}")
                    continue
                scaled_mechanics.append(mechanic)
            card.mechanics = scaled_mechanics

    return card


def _scale_warrior_mechanics(mechanics: List[str], level: int) -> List[str]:
    scaled = []
    for mechanic in mechanics:
        aoe_match = re.match(r"(.*aoe_damage_)(\d+)", mechanic)
        if aoe_match:
            prefix, base_val = aoe_match.groups()
            scaled.append(f"{prefix}{int(base_val) + ((level - 1) // 2)}")
            continue

        target_match = re.match(r"(.*(?:heal_target|damage)_)(\d+)", mechanic)
        if target_match:
            prefix, base_val = target_match.groups()
            scaled.append(f"{prefix}{int(base_val) + ((level - 1) // 2)}")
            continue

        buff_match = re.match(r"((?:battlecry_)?buff_all_|battlecry_buff_)(\d+)_(\d+)$", mechanic)
        if buff_match:
            prefix, base_atk, base_hp = buff_match.groups()
            bonus = (level - 1) // 2
            scaled.append(f"{prefix}{int(base_atk) + bonus}_{int(base_hp) + bonus}")
            continue

        for passive_prefix in ["regen_", "armor_", "aura_atk_", "reflect_"]:
            passive_match = re.match(rf"({passive_prefix})(\d+)", mechanic)
            if passive_match:
                prefix, base_val = passive_match.groups()
                scaled.append(f"{prefix}{int(base_val) + ((level - 1) // 3)}")
                break
        else:
            scaled.append(mechanic)

    return scaled


def calculate_card_display_stats(base_attack: int, base_hp: int, rarity: str, level: int) -> tuple[int, int]:
    """Расчет статов для отображения (коллекция/превью)."""
    growth = get_rarity_growth(rarity)
    return scale_base_stats(base_attack, base_hp, growth, level)
