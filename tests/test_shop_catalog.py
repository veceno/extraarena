from pathlib import Path

from infrastructure.shop_config import (
    CASE_PACKS,
    GEM_PACKAGES,
    SHOP_PRICES,
    build_shop_catalog,
    order_particles_for_shop,
)


INDEX = Path("webapp/index.html")
SERVER = Path("web/server.py")


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
    assert "/styles.css?v=shop-redesign-20260528" in source
    assert "ea-shop-root" in source
    assert "ea-shop-featured-particle" in source
    assert "window.openExternalLink(historyUrl)" in source
