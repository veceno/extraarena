"""
Система открытия кейсов для ChibiArena.

Реализует:
- Процесс открытия с 4 нажатиями
- Апгрейд тира кейса
- Генерацию наград по таблицам вероятностей
- Обработку дубликатов и конвертацию в частицы
- Специальную логику для стартовой редкости (Юни)
"""

import random
from typing import Dict, List, Any, Optional
from datetime import datetime

from infrastructure.case_config import (
    TIER_RARITY_PROBABILITIES,
    MAX_RARITY_BY_TIER,
    START_RARITY_REPLACEMENT,
    TIER_REWARDS_COUNT,
    BASE_PARTICLES_BY_RARITY,
    TIER_PARTICLES_MULTIPLIER,
    T5_COMMON_JACKPOT_PARTICLES,
    TIER_UPGRADE_CHANCES,
    LIMITED_EVENT_ACTIVE,
    LIMITED_EVENT_PROBABILITY,
)

MAX_TAP_NUMBER = max(TIER_UPGRADE_CHANCES.keys(), default=4)


def roll_tier_upgrade(current_tier: int, tap_number: int) -> int:
    """
    Проверить и выполнить апгрейд тира кейса на определенном тапе.
    
    Args:
        current_tier: Текущий тир кейса (1-5)
        tap_number: Номер тапа (1-4)
    
    Returns:
        Новый тир кейса (может остаться прежним, если апгрейд не произошел)
    """
    if current_tier >= 5:
        return current_tier  # Максимальный тир достигнут
    
    if tap_number not in TIER_UPGRADE_CHANCES:
        return current_tier
    
    upgrade_chance = TIER_UPGRADE_CHANCES[tap_number]
    if random.random() < upgrade_chance:
        return min(current_tier + 1, 5)
    
    return current_tier


def select_rarity(tier: int) -> str:
    """
    Выбрать редкость карты для заданного тира кейса.
    
    Args:
        tier: Тир кейса (1-5)
    
    Returns:
        Название редкости
    """
    probabilities = TIER_RARITY_PROBABILITIES.get(tier, {})
    
    # Если тир 5 и активено событие, добавляем шанс на лимитированную
    if tier == 5 and LIMITED_EVENT_ACTIVE:
        probabilities = probabilities.copy()
        probabilities["limited"] = LIMITED_EVENT_PROBABILITY
        # Нормализуем остальные вероятности
        total_without_limited = sum(p for r, p in probabilities.items() if r != "limited")
        if total_without_limited > 0:
            scale = (1.0 - LIMITED_EVENT_PROBABILITY) / total_without_limited
            for rarity in probabilities:
                if rarity != "limited":
                    probabilities[rarity] *= scale
    
    # Нормализуем вероятности (на случай, если сумма не равна 1.0)
    total = sum(probabilities.values())
    if total == 0:
        return "common"  # Fallback
    
    # Выбираем редкость по вероятностям
    rand = random.random() * total
    cumulative = 0.0
    for rarity, prob in probabilities.items():
        cumulative += prob
        if rand <= cumulative:
            return rarity
    
    # Fallback
    return "common"


def check_start_rarity_replacement(selected_rarity: str) -> str:
    """
    Проверить, нужно ли заменить редкость на Стартовую (Юни).
    
    Args:
        selected_rarity: Выбранная редкость
    
    Returns:
        Финальная редкость (может быть заменена на "start")
    """
    if selected_rarity in START_RARITY_REPLACEMENT:
        replacement_chance = START_RARITY_REPLACEMENT[selected_rarity]
        if random.random() < replacement_chance:
            return "start"
    
    return selected_rarity


def get_available_rarities_for_tier(tier: int) -> List[str]:
    """Получить список доступных редкостей для тира."""
    max_rarity = MAX_RARITY_BY_TIER.get(tier, "common")
    rarity_order = ["common", "rare", "start", "superrare", "epic", "legendary", "mythic", "divine", "limited"]
    
    max_index = rarity_order.index(max_rarity) if max_rarity in rarity_order else 0
    return rarity_order[:max_index + 1]


async def select_card_by_rarity(db, rarity: str, tier: int) -> Optional[Dict[str, Any]]:
    """
    Выбрать конкретную карту по редкости из доступных в БД.
    
    Args:
        db: Экземпляр Database
        rarity: Редкость карты
        tier: Тир кейса (для ограничения доступных редкостей)
    
    Returns:
        Словарь с данными карты или None
    """
    # Получаем доступные редкости для тира
    available_rarities = get_available_rarities_for_tier(tier)
    
    # Если выбранная редкость недоступна для тира, понижаем до максимально доступной
    if rarity not in available_rarities:
        rarity = available_rarities[-1] if available_rarities else "common"
    
    # Специальная обработка для стартовой редкости (Юни)
    if rarity == "start":
        uni_card = await db.get_uni_card()
        if uni_card:
            return uni_card
        # Если карта Юни не найдена, понижаем до редкой
        rarity = "rare"
    
    # Получаем все карты с нужной редкостью
    cards = await db.get_cards_by_rarity(rarity)
    
    if not cards:
        # Если карт с нужной редкостью нет, понижаем редкость
        rarity_order = ["common", "rare", "superrare", "epic", "legendary", "mythic", "divine"]
        if rarity in rarity_order:
            idx = rarity_order.index(rarity)
            for lower_rarity in reversed(rarity_order[:idx]):
                cards = await db.get_cards_by_rarity(lower_rarity)
                if cards:
                    break
    
    if not cards:
        return None
    
    # Выбираем случайную карту
    return random.choice(cards)


def calculate_particles_for_duplicate(rarity: str, tier: int, is_t5_common: bool = False) -> int:
    """
    Рассчитать количество частиц за дубликат карты.
    
    Args:
        rarity: Редкость карты
        tier: Тир кейса
        is_t5_common: Флаг, что это обычная карта из T5 (джекпот)
    
    Returns:
        Количество частиц
    """
    # Джекпот для обычной карты из T5
    if is_t5_common and tier == 5 and rarity == "common":
        return T5_COMMON_JACKPOT_PARTICLES
    
    # Базовое количество частиц
    base_particles = BASE_PARTICLES_BY_RARITY.get(rarity, 0)
    
    # Множитель от тира
    multiplier = TIER_PARTICLES_MULTIPLIER.get(tier, 1.0)
    
    return int(base_particles * multiplier)


def normalize_tap_results(tap_results: Optional[List[int]], final_tier: int) -> List[int]:
    """
    Привести последовательность тиров после тапов к безопасному виду.
    
    - Игнорируем некорректные значения
    - Не даем значениям выходить за диапазон 1-5 и за пределы финального тира
    - Обеспечиваем неубывающую последовательность
    - Докладываем последовательность до количества тапов (по умолчанию 4)
    """
    normalized: List[int] = []
    if final_tier < 1:
        final_tier = 1
    final_tier = min(final_tier, 5)
    
    previous_value = 1
    for raw_value in (tap_results or []):
        try:
            tap_tier = int(raw_value)
        except (TypeError, ValueError):
            continue
        
        tap_tier = max(1, min(tap_tier, 5))
        tap_tier = min(tap_tier, final_tier)
        if normalized:
            previous_value = normalized[-1]
        if tap_tier < previous_value:
            tap_tier = previous_value
        
        normalized.append(tap_tier)
        if len(normalized) >= MAX_TAP_NUMBER:
            break
    
    if not normalized:
        normalized = [final_tier] * MAX_TAP_NUMBER
    elif len(normalized) < MAX_TAP_NUMBER:
        normalized.extend([final_tier] * (MAX_TAP_NUMBER - len(normalized)))
    
    return normalized


async def generate_case_rewards(
    db,
    tier: int,
    user_id: int,
    user_card_ids: set[int]
) -> Dict[str, Any]:
    """
    Сгенерировать награды для кейса на основе финального тира.
    
    Args:
        db: Экземпляр Database
        tier: Финальный тир кейса (после всех 4 тапов)
        user_id: ID пользователя
        user_card_ids: Множество ID карт, которые уже есть у пользователя
    
    Returns:
        Словарь с наградами:
        {
            "coins": int,
            "cards": List[Dict],
            "particles": List[Dict],
            "gems": int (опционально),
            "limited_shards": int (опционально),
            "jackpot": bool (если был джекпот частиц)
        }
    """
    rewards = {
        "coins": 0,
        "cards": [],
        "particles": [],
        "gems": 0,
        "limited_shards": 0,
        "jackpot": False,
    }
    
    tier_config = TIER_REWARDS_COUNT.get(tier, TIER_REWARDS_COUNT[1])
    
    # Генерируем монеты
    coins_range = tier_config["coins"]
    rewards["coins"] = random.randint(coins_range[0], coins_range[1])
    
    # Генерируем карты
    cards_range = tier_config["cards"]
    num_cards = random.randint(cards_range[0], cards_range[1])
    
    for _ in range(num_cards):
        # Выбираем редкость
        rarity = select_rarity(tier)
        
        # Проверяем замену на стартовую
        rarity = check_start_rarity_replacement(rarity)
        
        # Выбираем конкретную карту по редкости
        card = await select_card_by_rarity(db, rarity, tier)
        
        if not card:
            # Если карта не найдена, пропускаем
            continue
        
        card_id = card["id"]
        is_duplicate = card_id in user_card_ids
        
        if is_duplicate:
            # Генерируем частицы
            is_t5_common_jackpot = (tier == 5 and rarity == "common")
            particles = calculate_particles_for_duplicate(rarity, tier, is_t5_common_jackpot)
            
            rewards["particles"].append({
                "card_id": card_id,
                "card_name": card.get("name", ""),
                "rarity": rarity,
                "particles": particles,
            })
            
            if is_t5_common_jackpot:
                rewards["jackpot"] = True
        else:
            # Новая карта
            rewards["cards"].append({
                "card_id": card_id,
                "card_name": card.get("name", ""),
                "rarity": rarity,
                "is_new": True,
            })
            user_card_ids.add(card_id)
    
    # Бонусные награды для T5
    if tier == 5:
        if "gems_chance" in tier_config:
            if random.random() < tier_config["gems_chance"]:
                gems_range = tier_config.get("gems_amount", (10, 50))
                rewards["gems"] = random.randint(gems_range[0], gems_range[1])
        
        if "limited_shards_chance" in tier_config:
            if random.random() < tier_config["limited_shards_chance"]:
                shards_range = tier_config.get("limited_shards_amount", (1, 5))
                rewards["limited_shards"] = random.randint(shards_range[0], shards_range[1])
    
    return rewards


async def process_case_opening(
    db,
    user_id: int,
    user_case_id: int,
    tap_results: List[int]  # Результаты 4 тапов (тиры после каждого тапа)
) -> Dict[str, Any]:
    """
    Обработать открытие кейса.
    
    Args:
        db: Экземпляр Database
        user_id: ID пользователя
        user_case_id: ID кейса пользователя
        tap_results: Список из 4 элементов - тиры после каждого тапа
    
    Returns:
        Словарь с результатами открытия
    """
    # Получаем кейс пользователя
    user_case = await db.get_user_case(user_case_id, user_id)
    if not user_case:
        return {"success": False, "error": "case_not_found"}
    
    default_case_id = await db.get_default_case_id()
    is_legacy_key_case = user_case.get("case_id") == default_case_id
    
    final_tier = int(user_case.get("tier") or 1)
    final_tier = max(1, min(final_tier, 5))
    tap_results = normalize_tap_results(tap_results, final_tier)
    
    # Получаем карты пользователя для проверки дубликатов
    user_cards = await db.get_user_cards(user_id)
    user_card_ids = {card["id"] for card in user_cards}
    
    # Генерируем награды
    rewards = await generate_case_rewards(db, final_tier, user_id, user_card_ids)
    
    # Выдаем награды пользователю
    result = {
        "success": True,
        "final_tier": final_tier,
        "rewards": rewards,
        "tap_results": tap_results,
    }
    
    # Выдаем монеты
    if rewards["coins"] > 0:
        await db.add_coins(user_id, rewards["coins"])
    
    # Выдаем карты
    for card_reward in rewards["cards"]:
        await db.add_card_to_user(user_id, card_reward["card_id"])
    
    # Выдаем частицы
    for particle_reward in rewards["particles"]:
        await db.add_particles_to_card(
            user_id,
            particle_reward["card_id"],
            particle_reward["particles"]
        )
    
    # Выдаем гемы (если есть)
    if rewards["gems"] > 0:
        await db.add_gems(user_id, rewards["gems"])
    
    # Удаляем кейс
    await db.remove_user_case(user_case_id, user_id)
    
    if is_legacy_key_case:
        await db.decrement_user_keys(user_id, 1)
    
    return result

