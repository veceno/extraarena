from pathlib import Path

from infrastructure.shop_config import (
    CASE_PACKS,
    GEM_PACKAGES,
    PARTICLE_SHOP_RARITIES,
    SHOP_PRICES,
    build_shop_catalog,
    calculate_particle_shop_offer,
    order_particles_for_shop,
)


INDEX = Path("webapp/index.html")
SERVER = Path("web/server.py")


def test_particle_shop_offer_depends_on_next_level_cost_not_rarity():
    assert calculate_particle_shop_offer(5) == {"particles": 10, "coins": 40}
    assert calculate_particle_shop_offer(80) == {"particles": 20, "coins": 80}
    assert calculate_particle_shop_offer(640) == {"particles": 160, "coins": 640}
    assert calculate_particle_shop_offer(1280) == {"particles": 320, "coins": 1280}
    assert PARTICLE_SHOP_RARITIES == (
        "common",
        "rare",
        "start",
        "superrare",
        "epic",
        "legendary",
        "mythic",
        "divine",
        "limited",
    )


def test_build_shop_catalog_uses_server_prices_for_case_and_coin_offers():
    catalog = build_shop_catalog([])

    case_prices = {item["id"]: item["gems_price"] for item in catalog["case_packs"]}
    assert case_prices == {pack_id: pack["gems"] for pack_id, pack in CASE_PACKS.items()}

    coin_prices = {item["item_type"]: item["gems_price"] for item in catalog["coin_offers"]}
    expected_coin_prices = {
        item_type: price
        for item_type, price in SHOP_PRICES.items()
        if item_type.startswith("coins_")
    }
    assert coin_prices == expected_coin_prices


def test_build_shop_catalog_uses_clear_case_pack_copy_and_plural_forms():
    catalog = build_shop_catalog([])

    case_copy = {
        item["id"]: (item["title"], item["subtitle"])
        for item in catalog["case_packs"]
    }

    assert case_copy == {
        "case_pack_1": ("1 кейс", "Открыть один кейс"),
        "case_pack_3": ("3 кейса", "Три открытия подряд"),
        "case_pack_5": ("5 кейсов", "Выгодный набор"),
        "case_pack_10": ("10 кейсов", "Большой запас кейсов"),
    }


def test_build_shop_catalog_merges_ruble_products_with_gem_package_config():
    catalog = build_shop_catalog([
        {
            "code": "gems_250",
            "item_type": "gems_package",
            "package_type": "gems_250",
            "name": "250 гемов",
            "price": 1,
            "badge": "discount",
            "sort_order": 60,
        },
        {
            "code": "extrapass_ultra",
            "item_type": "extrapass_ultra",
            "package_type": None,
            "name": "ExtraPass Ultra",
            "price": 349,
            "badge": "popular",
            "sort_order": 20,
            "description": "Ultra pass",
        },
    ])

    gems_250 = next(item for item in catalog["gem_products"] if item["package_type"] == "gems_250")
    assert gems_250["gems"] == GEM_PACKAGES["gems_250"]["gems"]
    assert gems_250["rub_price"] == 1
    assert gems_250["payment_code"] == "gems_250"
    assert gems_250["discount_pct"] == GEM_PACKAGES["gems_250"]["discount_pct"]

    assert catalog["pass_products"] == [
        {
            "code": "extrapass_ultra",
            "item_type": "extrapass_ultra",
            "package_type": None,
            "title": "ExtraPass Ultra",
            "description": "Ultra pass",
            "rub_price": 349,
            "badge": "popular",
            "sort_order": 20,
            "metadata": {},
        }
    ]


def test_order_particles_featured_prefers_highest_rarity_then_particles_amount():
    cards = [
        {"id": 1, "rarity": "rare", "particles": 200},
        {"id": 2, "rarity": "epic", "particles": 15},
        {"id": 3, "rarity": "common", "particles": 999},
    ]

    ordered = order_particles_for_shop(cards, "2026-05-28")

    assert [card["id"] for card in ordered] == [2, 1, 3]


def test_order_particles_featured_uses_particles_amount_when_rarity_matches():
    cards = [
        {"id": 10, "rarity": "rare", "particles": 30},
        {"id": 11, "rarity": "rare", "particles": 50},
        {"id": 12, "rarity": "rare", "particles": 40},
    ]

    ordered = order_particles_for_shop(cards, "2026-05-28")

    assert [card["id"] for card in ordered] == [11, 10, 12]


def test_order_particles_featured_tie_break_is_stable_for_rotation():
    cards = [
        {"id": 21, "rarity": "epic", "particles": 15},
        {"id": 22, "rarity": "epic", "particles": 15},
        {"id": 23, "rarity": "epic", "particles": 15},
    ]

    first = order_particles_for_shop(cards, "2026-05-28")
    second = order_particles_for_shop(list(reversed(cards)), "2026-05-28")

    assert first[0]["id"] == second[0]["id"]
    assert {card["id"] for card in first} == {21, 22, 23}


def test_order_particles_featured_softly_penalizes_unowned_cards():
    cards = [
        {"id": 1, "rarity": "rare", "particles": 30, "owned": True},
        {"id": 2, "rarity": "epic", "particles": 15, "owned": False},
        {"id": 3, "rarity": "common", "particles": 50, "owned": True},
    ]

    ordered = order_particles_for_shop(cards, "2026-05-28")

    assert [card["id"] for card in ordered] == [1, 2, 3]


def test_particles_daily_rotation_uses_owned_upgradeable_cards():
    server = SERVER.read_text(encoding="utf-8")
    particles_block = server.split("async def particles_daily_handler", 1)[1].split(
        "async def particles_buy_handler",
        1,
    )[0]

    assert "get_random_owned_upgradeable_cards_by_rarity" in particles_block
    assert "get_random_cards_by_rarities" not in particles_block
    assert "uc.card_id IS NOT NULL AS owned" in particles_block
    assert '"owned": bool(row.get("owned"))' in particles_block


def test_shop_catalog_endpoint_and_mobile_bootstrap_are_wired():
    server = SERVER.read_text(encoding="utf-8")

    assert "shop_catalog_handler" in server
    assert 'app.router.add_get("/api/shop/catalog", shop_catalog_handler)' in server
    assert "shop_catalog_status, shop_catalog_payload" in server
    assert '"shop_catalog": shop_catalog_status' in server
    assert 'relative_path.endswith(".css")' in server
    assert "no-store, no-cache, must-revalidate" in server


def test_shop_frontend_uses_catalog_endpoint_and_redesign_classes():
    source = INDEX.read_text(encoding="utf-8")

    assert "/api/shop/catalog" in source
    assert "shopCatalog" in source
    assert "/styles.css?v=shop-pack-ui-20260619" in source
    assert "ea-shop-root" in source
    assert "ea-shop-featured-particle" in source
    assert "window.openExternalLink(historyUrl)" in source


def test_shop_frontend_polishes_particles_copy_layout_and_debug_section():
    source = INDEX.read_text(encoding="utf-8")
    styles = Path("webapp/styles.css").read_text(encoding="utf-8")
    shop_block = source.split("const ShopScreen", 1)[1].split(
        "// ═══════════════════════════════════════════\n// ARENA MAIN",
        1,
    )[0]

    assert "Видно, кому именно покупается" not in shop_block
    assert "Размер зависит от следующего уровня, а не редкости" in shop_block
    assert "Debug Section" not in shop_block
    assert "1 ключ (дебаг)" not in shop_block
    assert ".ea-shop-particle-card:not(.ea-shop-featured-particle) .ea-shop-particle-art" in styles
    assert "caseArtForPack" in shop_block
