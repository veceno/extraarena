from __future__ import annotations

import hashlib
import json
from typing import Any

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

CASE_PACKS: dict[str, dict[str, int]] = {
    "case_pack_1":  {"keys": 1,  "gems": 25},
    "case_pack_3":  {"keys": 3,  "gems": 75},
    "case_pack_5":  {"keys": 5,  "gems": 125},
    "case_pack_10": {"keys": 10, "gems": 250},
}

PARTICLES_COSTS: dict[str, dict[str, int]] = {
    "common": {"particles": 50, "coins": 200},
    "rare":   {"particles": 30, "coins": 400},
    "epic":   {"particles": 15, "coins": 1000},
}


RARITY_SHOP_ORDER: dict[str, int] = {
    "common": 1,
    "rare": 2,
    "superrare": 3,
    "epic": 4,
    "legendary": 5,
    "mythic": 6,
    "divine": 7,
    "limited": 8,
    "unique": 9,
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _format_amount(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _coin_amount(item_type: str) -> int:
    return _as_int(item_type.replace("coins_", "", 1))


def build_shop_catalog(ruble_products: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Build the in-game shop catalog from server-owned pricing sources."""
    ruble_products = ruble_products or []

    case_packs = []
    for index, (pack_id, pack) in enumerate(sorted(CASE_PACKS.items(), key=lambda item: item[1]["keys"])):
        keys = _as_int(pack.get("keys"))
        case_packs.append({
            "id": pack_id,
            "item_type": pack_id,
            "title": f"{keys} кейс" if keys == 1 else f"{keys} кейса",
            "subtitle": {
                1: "быстрый шанс карты",
                3: "серия открытий",
                5: "популярно",
                10: "запас на вечер",
            }.get(keys, "кейсы для открытия"),
            "keys": keys,
            "gems_price": _as_int(pack.get("gems")),
            "badge": "popular" if pack_id == "case_pack_5" else None,
            "sort_order": (index + 1) * 10,
        })

    coin_offers = []
    for index, (item_type, gems_price) in enumerate(
        sorted(
            ((item_type, price) for item_type, price in SHOP_PRICES.items() if item_type.startswith("coins_")),
            key=lambda item: _coin_amount(item[0]),
        )
    ):
        coins = _coin_amount(item_type)
        coin_offers.append({
            "id": item_type,
            "item_type": item_type,
            "title": f"{_format_amount(coins)} монет",
            "coins": coins,
            "gems_price": _as_int(gems_price),
            "badge": "featured" if item_type == "coins_2000" else None,
            "sort_order": (index + 1) * 10,
        })

    products_by_package: dict[str, dict[str, Any]] = {}
    pass_products = []
    for product in ruble_products:
        product_dict = dict(product)
        item_type = str(product_dict.get("item_type") or "")
        package_type = product_dict.get("package_type")
        if item_type == "gems_package" and package_type:
            products_by_package[str(package_type)] = product_dict
        elif item_type in {"extrapass", "extrapass_ultra", "extrapass_gift"}:
            pass_products.append({
                "code": str(product_dict.get("code") or item_type),
                "item_type": item_type,
                "package_type": product_dict.get("package_type"),
                "title": str(product_dict.get("name") or item_type),
                "description": product_dict.get("description") or "",
                "rub_price": _as_float(product_dict.get("price")),
                "badge": product_dict.get("badge"),
                "sort_order": _as_int(product_dict.get("sort_order"), 100),
                "metadata": _as_dict(product_dict.get("metadata")),
            })

    gem_products = []
    for package_type, package in GEM_PACKAGES.items():
        product = products_by_package.get(package_type, {})
        code = str(product.get("code") or ("gems_starter_once" if package_type == "starter_once" else package_type))
        gem_products.append({
            "code": code,
            "payment_code": code,
            "item_type": "gems_package",
            "package_type": package_type,
            "title": str(product.get("name") or f"{package['gems']} гемов"),
            "description": product.get("description") or "",
            "gems": _as_int(package.get("gems")),
            "rub_price": _as_float(product.get("price"), _as_float(package.get("price"))),
            "discount_pct": _as_int(package.get("discount_pct")),
            "one_time": bool(package.get("one_time")),
            "badge": product.get("badge") or ("one-time" if package.get("one_time") else None),
            "sort_order": _as_int(product.get("sort_order"), 100),
            "metadata": _as_dict(product.get("metadata")),
        })

    return {
        "case_packs": case_packs,
        "coin_offers": coin_offers,
        "gem_products": sorted(gem_products, key=lambda item: (item["sort_order"], item["gems"])),
        "pass_products": sorted(pass_products, key=lambda item: (item["sort_order"], item["rub_price"])),
    }


def order_particles_for_shop(cards: list[dict[str, Any]], rotation_date: str) -> list[dict[str, Any]]:
    """Return particle cards with the featured card first."""
    if len(cards) <= 1:
        return list(cards)

    indexed_cards = [(index, dict(card)) for index, card in enumerate(cards)]
    highest_rarity = max(
        RARITY_SHOP_ORDER.get(str(card.get("rarity") or "common"), 0)
        for _, card in indexed_cards
    )
    rarity_candidates = [
        (index, card)
        for index, card in indexed_cards
        if RARITY_SHOP_ORDER.get(str(card.get("rarity") or "common"), 0) == highest_rarity
    ]
    highest_particles = max(_as_int(card.get("particles")) for _, card in rarity_candidates)
    candidates = [
        (index, card)
        for index, card in rarity_candidates
        if _as_int(card.get("particles")) == highest_particles
    ]

    if len(candidates) == 1:
        selected_index, featured = candidates[0]
    else:
        candidate_ids = sorted(str(card.get("id") or "") for _, card in candidates)
        seed = f"{rotation_date}:{','.join(candidate_ids)}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        selected_candidate = int(digest[:8], 16) % len(candidates)
        selected_index, featured = sorted(candidates, key=lambda item: str(item[1].get("id") or ""))[selected_candidate]

    return [featured] + [card for index, card in indexed_cards if index != selected_index]
