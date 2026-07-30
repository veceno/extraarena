"""Canonical economic rules for card progression and duplicate rewards.

This module deliberately contains no combat-stat calculations.  It is the
single backend source for prices and duplicate conversion used by cases,
reward tracks, shop sets, and the card upgrade API.
"""

from __future__ import annotations


UPGRADE_PARTICLES_BY_LEVEL = {
    1: 5,
    2: 10,
    3: 20,
    4: 40,
    5: 80,
    6: 160,
    7: 320,
    8: 640,
    9: 1280,
}

UPGRADE_COINS_BY_LEVEL = {
    1: 50,
    2: 150,
    3: 400,
    4: 900,
    5: 2000,
    6: 4500,
    7: 8000,
    8: 13000,
    9: 25000,
}

CATCHUP_POLICY_VERSION = 1
CATCHUP_REFERENCE_CARD_COUNT = 9
CATCHUP_LEVEL_LAG = 2
CATCHUP_MAX_TARGET_LEVEL = 7

# Simplified cards have a single 1→2 transition.  Keep its price explicit so
# balancing the regular 9→10 transition cannot silently change it.
SIMPLIFIED_UPGRADE_PARTICLES = 640
SIMPLIFIED_UPGRADE_COINS = 13000

# Base duplicate particles by rarity.  Limited cards are deliberately above
# divine: a duplicate must never silently turn into a zero-value reward.
BASE_PARTICLES_BY_RARITY = {
    "common": 2,
    "rare": 3,
    "start": 4,
    "superrare": 5,
    "epic": 10,
    "legendary": 20,
    "mythic": 40,
    "divine": 100,
    "limited": 160,
}

TIER_PARTICLES_MULTIPLIER = {
    1: 1.30,
    2: 0.65,
    3: 0.81,
    4: 1.14,
    5: 1.95,
}

T5_COMMON_JACKPOT_PARTICLES = 125


def calculate_upgrade_particles(rarity: str, level: int) -> int:
    """Return the level-transition price; rarity is kept for API compatibility."""
    base = UPGRADE_PARTICLES_BY_LEVEL.get(int(level), UPGRADE_PARTICLES_BY_LEVEL[1])
    return int(base)


def calculate_upgrade_coins(rarity: str, level: int) -> int:
    """Return the level-transition price; rarity is kept for API compatibility."""
    base = UPGRADE_COINS_BY_LEVEL.get(int(level), UPGRADE_COINS_BY_LEVEL[1])
    return int(base)


def calculate_card_upgrade_cost(
    rarity: str,
    level: int,
    *,
    simplified_levelup: bool = False,
) -> dict[str, int]:
    """Return the canonical economic price for one card-level transition."""
    if simplified_levelup:
        return {
            "particles": SIMPLIFIED_UPGRADE_PARTICLES,
            "coins": SIMPLIFIED_UPGRADE_COINS,
        }
    return {
        "particles": calculate_upgrade_particles(rarity, level),
        "coins": calculate_upgrade_coins(rarity, level),
    }


def calculate_new_card_catchup(
    reference_level: int,
    *,
    eligible_count: int = CATCHUP_REFERENCE_CARD_COUNT,
    simplified_levelup: bool = False,
) -> dict[str, int | bool]:
    """Return a first-acquisition particle reserve without changing card level."""
    normalized_count = max(0, int(eligible_count or 0))
    normalized_reference = max(1, min(10, int(reference_level or 1)))
    eligible = (
        not simplified_levelup
        and normalized_count >= CATCHUP_REFERENCE_CARD_COUNT
    )
    target_level = (
        max(
            1,
            min(
                CATCHUP_MAX_TARGET_LEVEL,
                normalized_reference - CATCHUP_LEVEL_LAG,
            ),
        )
        if eligible
        else 1
    )
    particles = sum(
        UPGRADE_PARTICLES_BY_LEVEL[level]
        for level in range(1, target_level)
    )
    return {
        "eligible": eligible,
        "eligible_count": normalized_count,
        "reference_level": normalized_reference,
        "target_level": target_level,
        "particles": particles,
        "policy_version": CATCHUP_POLICY_VERSION,
    }


def calculate_duplicate_particles(
    rarity: str,
    tier: int = 1,
    *,
    is_t5_common: bool = False,
) -> int:
    normalized_rarity = str(rarity or "common").lower()
    try:
        normalized_tier = int(tier)
    except (TypeError, ValueError):
        normalized_tier = 1
    if is_t5_common and normalized_tier == 5 and normalized_rarity == "common":
        return T5_COMMON_JACKPOT_PARTICLES
    base = BASE_PARTICLES_BY_RARITY.get(normalized_rarity, 0)
    if base <= 0:
        return 0
    multiplier = TIER_PARTICLES_MULTIPLIER.get(normalized_tier, 1.0)
    return max(1, int(base * multiplier))
