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
from datetime import datetime, timezone

from infrastructure.case_config import (
    TIER_RARITY_PROBABILITIES,
    MAX_RARITY_BY_TIER,
    START_RARITY_REPLACEMENT,
    TIER_REWARDS_COUNT,
    BASE_PARTICLES_BY_RARITY,
    TIER_UPGRADE_CHANCES,
    LIMITED_EVENT_ACTIVE,
    LIMITED_EVENT_PROBABILITY,
    resolve_case_config,
)
from infrastructure.card_economy import calculate_duplicate_particles

MAX_TAP_NUMBER = max(TIER_UPGRADE_CHANCES.keys(), default=4)
ULTRA_REROLL_ATTEMPTS = 1

# Правки 2026-06-25: бонусы шансов за Ultra вырезаны (см. CASE_SYSTEM.md).
# Ultra-игроки больше не получают +шансы к редкостям, +шанс тап-апгрейда или
# +шанс T5-гемов. Остаётся только ручной Ultra-реролл (ULTRA_REROLL_ATTEMPTS,
# get_case_reroll_attempts, score_case_rewards).
# Параметр extra_pass сохранён в сигнатурах для совместимости с кодом,
# проверяющим уровень ExtraPass для реролла.

RARITY_SCORE = {
    "common": 1,
    "rare": 2,
    "start": 3,
    "superrare": 4,
    "epic": 7,
    "legendary": 12,
    "mythic": 20,
    "divine": 36,
    "limited": 48,
}


def normalize_extra_pass_status(extra_pass: Optional[str], expires_at: Any = None) -> str:
    """Вернуть активный для кейсов уровень ExtraPass с учетом legacy-срока."""
    pass_status = (extra_pass or "inactive").lower()
    if pass_status not in {"active", "ultra"}:
        return "inactive"

    if expires_at:
        try:
            expires = expires_at
            if isinstance(expires, str):
                expires = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            now = datetime.now(expires.tzinfo) if expires.tzinfo else datetime.now()
            if expires < now:
                return "inactive"
        except (TypeError, ValueError):
            return "inactive"

    return pass_status


async def get_user_case_pass_status(db, user_id: int) -> str:
    """Получить уровень ExtraPass, который влияет на кейсы."""
    profile = await db.get_user_profile(user_id)
    if not profile:
        return "inactive"
    return normalize_extra_pass_status(
        profile.get("extra_pass"),
        profile.get("extra_pass_expires_at"),
    )


def roll_tier_upgrade(current_tier: int, tap_number: int, extra_pass: str = "inactive", case_config: dict = None) -> int:
    """
    Проверить и выполнить апгрейд тира кейса на определенном тапе.

    Args:
        current_tier: Текущий тир кейса (1-5)
        tap_number: Номер тапа (1-4)

    Returns:
        Новый тир кейса (может остаться прежним, если апгрейд не произошел)
    """
    cc = resolve_case_config(case_config)
    if current_tier >= 5:
        return current_tier  # Максимальный тир достигнут

    tier_upgrade_chances = cc["tier_upgrade_chances"]
    if tap_number not in tier_upgrade_chances:
        return current_tier

    upgrade_chance = tier_upgrade_chances[tap_number]
    if random.random() < upgrade_chance:
        return min(current_tier + 1, 5)

    return current_tier


def simulate_case_tap_results(initial_tier: int = 1, extra_pass: str = "inactive", case_config: dict = None) -> List[int]:
    """Сгенерировать серверную последовательность тиров для полного открытия кейса."""
    extra_pass = normalize_extra_pass_status(extra_pass)
    try:
        current_tier = int(initial_tier)
    except (TypeError, ValueError):
        current_tier = 1
    current_tier = max(1, min(current_tier, 5))

    tap_results: List[int] = []
    for tap_number in range(1, MAX_TAP_NUMBER + 1):
        # 3-arg call preserves existing 3-param roll_tier_upgrade test fakes;
        # do not collapse to always-4-arg.
        if case_config is not None:
            current_tier = roll_tier_upgrade(current_tier, tap_number, extra_pass, case_config)
        else:
            current_tier = roll_tier_upgrade(current_tier, tap_number, extra_pass)
        tap_results.append(current_tier)
    return tap_results


def select_rarity(tier: int, extra_pass: str = "inactive", case_config: dict = None) -> str:
    """
    Выбрать редкость карты для заданного тира кейса.

    Args:
        tier: Тир кейса (1-5)

    Returns:
        Название редкости
    """
    cc = resolve_case_config(case_config)
    tier_rarity_probabilities = cc["tier_rarity_probabilities"]
    probabilities = tier_rarity_probabilities.get(tier, {})

    limited_event_active = cc["limited_event_active"]
    limited_event_probability = cc["limited_event_probability"]

    # Если тир 5 и активено событие, добавляем шанс на лимитированную
    if tier == 5 and limited_event_active:
        probabilities = probabilities.copy()
        probabilities["limited"] = limited_event_probability
        # Нормализуем остальные вероятности
        total_without_limited = sum(p for r, p in probabilities.items() if r != "limited")
        if total_without_limited > 0:
            scale = (1.0 - limited_event_probability) / total_without_limited
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


def check_start_rarity_replacement(selected_rarity: str, case_config: dict = None) -> str:
    """
    Проверить, нужно ли заменить редкость на Стартовую (Юни).

    Args:
        selected_rarity: Выбранная редкость

    Returns:
        Финальная редкость (может быть заменена на "start")
    """
    cc = resolve_case_config(case_config)
    start_rarity_replacement = cc["start_rarity_replacement"]
    if selected_rarity in start_rarity_replacement:
        replacement_chance = start_rarity_replacement[selected_rarity]
        if random.random() < replacement_chance:
            return "start"

    return selected_rarity


def get_available_rarities_for_tier(tier: int, case_config: dict = None) -> List[str]:
    """Получить список доступных редкостей для тира.

    Для T5 при активном limited-событии добавляем 'limited' в конец — иначе
    select_card_by_rarity понизит свёрнутую 'limited' редкость до 'divine', и
    лимитированные карты никогда не будут выпадать (баг: limited-event не доставлял
    limited-карты, а дубликаты divine получали limited-частицы). С событием limited
    остаётся в available_rarities → get_cards_by_rarity('limited') выбирает реальную
    limited-карту.
    """
    cc = resolve_case_config(case_config)
    max_rarity = cc["max_rarity_by_tier"].get(tier, "common")
    rarity_order = ["common", "rare", "start", "superrare", "epic", "legendary", "mythic", "divine", "limited"]

    max_index = rarity_order.index(max_rarity) if max_rarity in rarity_order else 0
    available = list(rarity_order[:max_index + 1])
    # limited доступна только на T5 и только при активном событии (см. select_rarity).
    limited_active = bool(cc.get("limited_event_active"))
    if case_config is None:
        limited_active = limited_active or bool(LIMITED_EVENT_ACTIVE)
    if tier == 5 and limited_active and "limited" not in available:
        available.append("limited")
    return available


async def select_card_by_rarity(db, rarity: str, tier: int, case_config: dict = None) -> Optional[Dict[str, Any]]:
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
    available_rarities = get_available_rarities_for_tier(tier, case_config)

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


def calculate_particles_for_duplicate(rarity: str, tier: int, is_t5_common: bool = False, case_config: dict = None) -> int:
    """
    Рассчитать количество частиц за дубликат карты.

    Args:
        rarity: Редкость карты
        tier: Тир кейса
        is_t5_common: Флаг, что это обычная карта из T5 (джекпот)

    Returns:
        Количество частиц
    """
    if case_config is None:
        return calculate_duplicate_particles(
            rarity,
            tier,
            is_t5_common=is_t5_common,
        )

    cc = resolve_case_config(case_config)
    # Джекпот для обычной карты из T5
    if is_t5_common and tier == 5 and rarity == "common":
        return cc["t5_common_jackpot_particles"]

    # Базовое количество частиц
    base_particles = cc["base_particles_by_rarity"].get(rarity, 0)

    # Множитель от тира
    multiplier = cc["tier_particles_multiplier"].get(tier, 1.0)

    amount = int(base_particles * multiplier)
    # Гарантируем >=1 частицу для дубликата ненулевой базы: защита от int()-усечения
    # в ноль при будущей правке множителей через live-config (регрессия 960a5e8e).
    # Явный ноль от администратора (base==0) сохраняется.
    if base_particles > 0 and amount < 1:
        amount = 1
    return amount


def get_case_reroll_attempts(extra_pass: str) -> int:
    return ULTRA_REROLL_ATTEMPTS if extra_pass == "ultra" else 0


def score_case_rewards(rewards: Dict[str, Any]) -> float:
    """Оценка наград для автоматического Ultra-реролла."""
    score = float(rewards.get("coins", 0)) / 100
    score += float(rewards.get("gems", 0)) * 8
    if rewards.get("jackpot"):
        score += 80

    for card_reward in rewards.get("cards", []):
        rarity = card_reward.get("rarity", "common")
        score += RARITY_SCORE.get(rarity, 1) * 100

    for particle_reward in rewards.get("particles", []):
        rarity = particle_reward.get("rarity", "common")
        score += RARITY_SCORE.get(rarity, 1) * 30
        score += float(particle_reward.get("particles", 0)) * 0.6

    return score


async def _generate_single_case_rewards(
    db,
    tier: int,
    user_id: int,
    user_card_ids: set[int],
    extra_pass: str = "inactive",
    case_config: dict = None,
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
            "jackpot": bool (если был джекпот частиц)
        }
    """
    cc = resolve_case_config(case_config)
    rewards = {
        "coins": 0,
        "cards": [],
        "particles": [],
        "gems": 0,
        "jackpot": False,
    }

    tier_rewards_count = cc["tier_rewards_count"]
    tier_config = tier_rewards_count.get(tier, tier_rewards_count[1])

    # Генерируем монеты
    coins_range = tier_config["coins"]
    rewards["coins"] = random.randint(coins_range[0], coins_range[1])

    # Генерируем карты
    cards_range = tier_config["cards"]
    num_cards = random.randint(cards_range[0], cards_range[1])

    for _ in range(num_cards):
        # Выбираем редкость
        rarity = select_rarity(tier, extra_pass, case_config)

        # Проверяем замену на стартовую
        rarity = check_start_rarity_replacement(rarity, case_config)

        # Выбираем конкретную карту по редкости
        card = await select_card_by_rarity(db, rarity, tier, case_config)

        if not card:
            # Если карта не найдена, пропускаем
            continue

        # Fallback внутри select_card_by_rarity мог вернуть карту другой
        # редкости; reward metadata и компенсация должны описывать факт.
        rarity = str(card.get("rarity") or rarity).lower()
        card_id = card["id"]
        is_duplicate = card_id in user_card_ids

        if is_duplicate:
            # Генерируем частицы
            is_t5_common_jackpot = (tier == 5 and rarity == "common")
            particles = calculate_particles_for_duplicate(rarity, tier, is_t5_common_jackpot, case_config)

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
            gems_chance = tier_config["gems_chance"]
            if random.random() < gems_chance:
                gems_range = tier_config.get("gems_amount", (10, 50))
                rewards["gems"] = random.randint(gems_range[0], gems_range[1])

    return rewards


async def generate_case_rewards(
    db,
    tier: int,
    user_id: int,
    user_card_ids: set[int],
    extra_pass: str = "inactive",
    case_config: dict = None,
) -> Dict[str, Any]:
    """
    Сгенерировать награды для кейса на основе финального тира.

    Ultra-бонусы:
    - повышенные веса редких карт;
    - повышенные шансы T5-бонусов.
    """
    extra_pass = normalize_extra_pass_status(extra_pass)
    rewards = await _generate_single_case_rewards(
        db,
        tier,
        user_id,
        set(user_card_ids),
        extra_pass,
        case_config,
    )

    if extra_pass == "ultra":
        rewards["extra_pass_bonus"] = {
            "tier": "ultra",
            "reroll_available": True,
            "reroll_attempts": get_case_reroll_attempts(extra_pass),
        }

    return rewards


async def process_case_opening(
    db,
    user_id: int,
    user_case_id: int,
    case_config: dict = None,
) -> Dict[str, Any]:
    """
    Подготовить pending opening для user_cases flow (без применения наград).

    Для Т-кейсов с предустановленным тиром анимация тапов не используется.
    Финальный тир берётся из `user_cases.tier` (зафиксирован при создании).
    Награды генерируются и возвращаются в payload, но НЕ применяются здесь —
    атомарное применение делает _apply_case_opening_rewards в
    `_claim_user_case_opening` на сервере (после возможного Ultra-реролла).

    Args:
        db: Экземпляр Database
        user_id: ID пользователя
        user_case_id: ID кейса пользователя

    Returns:
        Словарь с результатами открытия (rewards не применены):
        {
            "success": True,
            "final_tier": int,
            "rewards": {...},
            "tap_results": [final_tier] * 4,
            "extra_pass_bonus": {...},
        }
        либо {"success": False, "error": "case_not_found"} если кейса нет.
    """
    user_case = await db.get_user_case(user_case_id, user_id)
    if not user_case:
        return {"success": False, "error": "case_not_found"}

    final_tier = int(user_case.get("tier") or 1)
    final_tier = max(1, min(final_tier, 5))

    user_cards = await db.get_user_cards(user_id)
    user_card_ids = {card["id"] for card in user_cards}

    extra_pass = await get_user_case_pass_status(db, user_id)
    rewards = await generate_case_rewards(db, final_tier, user_id, user_card_ids, extra_pass, case_config)

    return {
        "success": True,
        "final_tier": final_tier,
        "rewards": rewards,
        "tap_results": [final_tier] * 4,
        "extra_pass_bonus": rewards.get("extra_pass_bonus"),
    }


async def _apply_case_opening_rewards(
    db,
    *,
    user_id: int,
    user_case_id: int,
    rewards: Dict[str, Any],
    final_tier: int,
    decrement_legacy_key: bool,
) -> Dict[str, Any]:
    converted_duplicates: list[dict[str, Any]] = []
    converted_card_ids: list[int] = []
    original_particle_rewards = list(rewards.get("particles", []))
    applied_particle_rewards: list[dict[str, Any]] = []
    duplicate_fallback_coins = 0

    def apply_duplicate_result(
        result: dict[str, Any],
        *,
        source_card_id: int,
        source_card_name: str,
        rarity: str,
    ) -> None:
        nonlocal duplicate_fallback_coins
        reward_type = str(result.get("reward_type") or "particles")
        if reward_type == "coins":
            duplicate_fallback_coins += int(result.get("coins_added") or result.get("amount") or 0)
            return
        applied_particle_rewards.append({
            "card_id": int(result.get("card_id") or source_card_id),
            "card_name": str(result.get("card_name") or source_card_name),
            "rarity": str(result.get("rarity") or rarity),
            "particles": int(result.get("particles_added") or result.get("amount") or 0),
            "redirected": bool(result.get("redirected", False)),
            "source_card_id": source_card_id,
            "source_card_name": source_card_name,
        })

    pool = getattr(db, "_pool", None)
    if pool:
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    if rewards["coins"] > 0:
                        await conn.execute(
                            "UPDATE users SET coins = GREATEST(0, COALESCE(coins, 0) + $1), updated_at = NOW() WHERE user_id = $2",
                            int(rewards["coins"]), user_id,
                        )
                    catchup_snapshot = None
                    snapshot_reader = getattr(
                        db,
                        "_get_card_catchup_snapshot_on_conn",
                        None,
                    )
                    if rewards["cards"] and callable(snapshot_reader):
                        catchup_snapshot = await snapshot_reader(conn, user_id)
                    for card_reward in rewards["cards"]:
                        card_id = int(card_reward["card_id"])
                        grant_card = getattr(db, "grant_card_or_duplicate", None)
                        if callable(grant_card):
                            grant_result = await grant_card(
                                user_id,
                                card_id,
                                tier=final_tier,
                                source="case",
                                source_metadata={
                                    "user_case_id": user_case_id,
                                    "final_tier": final_tier,
                                },
                                catchup_snapshot=catchup_snapshot,
                                conn=conn,
                            )
                        else:
                            card_exists = await conn.fetchval(
                                "SELECT 1 FROM cards WHERE id = $1",
                                card_id,
                            )
                            if not card_exists:
                                raise ValueError(f"card_not_found:{card_id}")
                            inserted = await conn.fetchrow(
                                """
                                INSERT INTO user_cards (user_id, card_id, level, particles)
                                VALUES ($1, $2, 1, 0)
                                ON CONFLICT (user_id, card_id) DO NOTHING
                                RETURNING card_id
                                """,
                                user_id, card_id,
                            )
                            grant_result = {
                                "success": True,
                                "is_new": bool(inserted),
                            }
                        if not grant_result.get("success"):
                            raise ValueError(
                                str(grant_result.get("error") or f"card_grant_failed:{card_id}")
                            )
                        if grant_result.get("is_new", False):
                            card_reward["is_new"] = True
                            card_reward["level"] = int(grant_result.get("level") or 1)
                            if grant_result.get("catchup"):
                                card_reward["catchup"] = dict(grant_result["catchup"])
                        else:
                            rarity = str(card_reward.get("rarity") or "common").lower()
                            if not callable(grant_card):
                                particles = calculate_duplicate_particles(rarity, final_tier)
                                updated = await conn.fetchrow(
                                    """
                                    UPDATE user_cards
                                    SET particles = COALESCE(particles, 0) + $1
                                    WHERE user_id = $2 AND card_id = $3
                                    RETURNING particles
                                    """,
                                    particles,
                                    user_id,
                                    card_id,
                                )
                                grant_result = {
                                    "success": bool(updated),
                                    "reward_type": "particles",
                                    "card_id": card_id,
                                    "card_name": card_reward.get("card_name", ""),
                                    "rarity": rarity,
                                    "particles_added": particles,
                                }
                            converted_card_ids.append(card_id)
                            apply_duplicate_result(
                                grant_result,
                                source_card_id=card_id,
                                source_card_name=str(card_reward.get("card_name") or ""),
                                rarity=rarity,
                            )
                            if str(grant_result.get("reward_type") or "particles") == "particles":
                                converted_duplicates.append(applied_particle_rewards[-1])
                    for particle_reward in original_particle_rewards:
                        particle_card_id = int(particle_reward["card_id"])
                        particle_rarity = str(particle_reward.get("rarity") or "common").lower()
                        particle_amount = int(particle_reward["particles"])
                        duplicate_grant = getattr(db, "grant_duplicate_particles", None)
                        if callable(duplicate_grant):
                            grant_result = await duplicate_grant(
                                user_id,
                                particle_card_id,
                                particle_rarity,
                                particle_amount,
                                conn=conn,
                            )
                        else:
                            await conn.execute(
                                """
                                UPDATE user_cards
                                SET particles = COALESCE(particles, 0) + $1
                                WHERE user_id = $2 AND card_id = $3
                                """,
                                particle_amount,
                                user_id,
                                particle_card_id,
                            )
                            grant_result = {
                                "success": True,
                                "reward_type": "particles",
                                "card_id": particle_card_id,
                                "card_name": particle_reward.get("card_name", ""),
                                "rarity": particle_rarity,
                                "particles_added": particle_amount,
                            }
                        if not grant_result.get("success"):
                            raise ValueError(str(grant_result.get("error") or "duplicate_grant_failed"))
                        apply_duplicate_result(
                            grant_result,
                            source_card_id=particle_card_id,
                            source_card_name=str(particle_reward.get("card_name") or ""),
                            rarity=particle_rarity,
                        )
                    if rewards["gems"] > 0:
                        await conn.execute(
                            "UPDATE users SET gems = GREATEST(0, COALESCE(gems, 0) + $1), updated_at = NOW() WHERE user_id = $2",
                            int(rewards["gems"]), user_id,
                        )
                    deleted = await conn.fetchval(
                        """
                        DELETE FROM user_cases
                        WHERE id = $1 AND user_id = $2 AND status = 'pending'
                        RETURNING 1
                        """,
                        user_case_id, user_id,
                    )
                    if not deleted:
                        raise ValueError("case_already_opened")
                    if decrement_legacy_key:
                        await conn.execute(
                            "UPDATE users SET keys = GREATEST(0, COALESCE(keys, 0) - 1), updated_at = NOW() WHERE user_id = $1",
                            user_id,
                        )
                    # The user_case deletion is the idempotency gate. Commit quest
                    # progress in the same transaction so a successful case claim
                    # can never lose its open_case_1 event.
                    if await db._daily_quests_enabled_on_conn(conn):
                        await db._apply_daily_quest_ops_on_conn(
                            conn,
                            user_id,
                            [("open_case_1", 1, False)],
                        )
            rewards["particles"] = applied_particle_rewards
            rewards["coins"] = int(rewards.get("coins") or 0) + duplicate_fallback_coins
            return {
                "success": True,
                "converted_duplicates": converted_duplicates,
                "converted_card_ids": converted_card_ids,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    if rewards["coins"] > 0:
        await db.update_user_coins(user_id, rewards["coins"])
    for card_reward in rewards["cards"]:
        grant_duplicate = getattr(db, "grant_card_or_duplicate", None)
        if callable(grant_duplicate):
            grant_result = await grant_duplicate(
                user_id,
                card_reward["card_id"],
                tier=final_tier,
                source="case",
                source_metadata={
                    "user_case_id": user_case_id,
                    "final_tier": final_tier,
                },
            )
            if not grant_result.get("success"):
                return grant_result
            if grant_result.get("is_new", False):
                card_reward["is_new"] = True
                card_reward["level"] = int(grant_result.get("level") or 1)
                if grant_result.get("catchup"):
                    card_reward["catchup"] = dict(grant_result["catchup"])
            else:
                source_card_id = int(card_reward["card_id"])
                converted_card_ids.append(source_card_id)
                apply_duplicate_result(
                    grant_result,
                    source_card_id=source_card_id,
                    source_card_name=str(card_reward.get("card_name") or ""),
                    rarity=str(card_reward.get("rarity") or "common"),
                )
                if str(grant_result.get("reward_type") or "particles") == "particles":
                    converted_duplicates.append(applied_particle_rewards[-1])
        else:
            await db.add_card_to_user(user_id, card_reward["card_id"])
    for particle_reward in original_particle_rewards:
        duplicate_grant = getattr(db, "grant_duplicate_particles", None)
        if callable(duplicate_grant):
            grant_result = await duplicate_grant(
                user_id,
                int(particle_reward["card_id"]),
                str(particle_reward.get("rarity") or "common"),
                int(particle_reward["particles"]),
            )
            if not grant_result.get("success"):
                return grant_result
            apply_duplicate_result(
                grant_result,
                source_card_id=int(particle_reward["card_id"]),
                source_card_name=str(particle_reward.get("card_name") or ""),
                rarity=str(particle_reward.get("rarity") or "common"),
            )
        else:
            await db.add_particles_to_card(
                user_id,
                particle_reward["card_id"],
                particle_reward["particles"],
            )
            applied_particle_rewards.append(dict(particle_reward))
    if rewards["gems"] > 0:
        await db.add_gems(user_id, rewards["gems"])
    await db.remove_user_case(user_case_id, user_id)
    if decrement_legacy_key:
        await db.decrement_user_keys(user_id, 1)
    await db.increment_daily_quest(user_id, "open_case_1", 1)
    rewards["particles"] = applied_particle_rewards
    rewards["coins"] = int(rewards.get("coins") or 0) + duplicate_fallback_coins
    return {
        "success": True,
        "converted_duplicates": converted_duplicates,
        "converted_card_ids": converted_card_ids,
    }
