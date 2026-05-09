from __future__ import annotations

SHOP_PRICES: dict[str, int] = {
    "keys_1": 5,
    "keys_3": 13,
    "keys_10": 40,
    "keys_25": 90,
    "keys_50": 180,
    "keys_100": 320,
    "coins_300": 35,
    "coins_500": 50,
    "coins_1400": 145,
    "coins_2000": 180,
    "coins_5000": 400,
    "coins_20000": 1400,
    "case": 10,
    "case_tier_1": 5,
    "case_tier_2": 15,
    "case_tier_3": 40,
    "case_tier_4": 100,
    "case_tier_5": 250,
}

GEM_PACKAGES: dict[str, dict] = {
    "starter_once": {"gems": 50, "price": 49, "one_time": True, "discount_pct": 0},
    "gems_100":     {"gems": 100, "price": 99, "discount_pct": 0},
    "gems_250":     {"gems": 250, "price": 229, "discount_pct": 7},
    "gems_600":     {"gems": 600, "price": 499, "discount_pct": 16},
    "gems_1300":    {"gems": 1300, "price": 999, "discount_pct": 22},
    "gems_2500":    {"gems": 2500, "price": 1499, "discount_pct": 33},
}

PARTICLES_COSTS: dict[str, dict[str, int]] = {
    "common": {"particles": 50, "coins": 200},
    "rare":   {"particles": 30, "coins": 400},
    "epic":   {"particles": 15, "coins": 1000},
}
