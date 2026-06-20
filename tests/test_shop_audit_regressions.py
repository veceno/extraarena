import json
import sys
import time
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.config import get_settings
from web import server as web_server


INDEX = Path("webapp/index.html")
STRONG_TEST_JWT_SECRET = "test-shop-audit-jwt-secret-that-is-long-enough-2026"


@pytest.fixture(autouse=True)
def _strong_jwt_secret(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeExtraIDDB:
    def __init__(self, session_id: str):
        self.session_id = session_id

    async def verify_session(self, session_uuid, token: str):
        if str(session_uuid) != self.session_id:
            return None
        return {"session_id": session_uuid, "user_id": 1001}

    async def check_rate_limit(self, key, max_requests, window_seconds):
        return True


class ShopAuditFakeDB:
    def __init__(self, *, admin_ids=(), fail_profile=False, allow_max_level_particles=False, starter_pack_used=False):
        self.admin_ids = {int(user_id) for user_id in admin_ids}
        self.fail_profile = fail_profile
        self.allow_max_level_particles = allow_max_level_particles
        self.starter_pack_used = bool(starter_pack_used)
        self.fetchrow_calls = []
        self.execute_calls = []
        self.settings_updates = []
        self.case_grants = []
        self.economy_events = []
        self.ruble_products = []
        self.ruble_product_creates = []
        self.ruble_product_updates = []
        self.shop_sets = {}
        self.claimed_shop_set_ids = set()
        self.cosmetics = {}
        self.last_ruble_products_surface = None

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {"shop": True},
            "shop": {
                "allow_max_level_particle_purchase": self.allow_max_level_particles,
            },
            "disabled_card_ids": [],
        }

    async def process_weekly_squad_cbrp(self):
        return {"processed": False}

    async def refresh_due_rating_snapshots(self, scope="players"):
        return {"refreshed": []}

    async def get_game_setting(self, key, default=None):
        if key in {
            "allow_max_level_particle_purchase",
            "shop_allow_max_level_particle_purchase",
            "allow_max_level_particles",
        }:
            return self.allow_max_level_particles
        return default

    async def is_admin(self, user_id):
        return int(user_id) in self.admin_ids

    async def get_user_profile(self, user_id):
        if self.fail_profile:
            raise RuntimeError("secret_shop_audit_table leaked")
        return {"user_id": int(user_id), "gems": 1000, "coins": 10000}

    async def get_user_settings(self, user_id):
        return {
            "starter_pack_used": self.starter_pack_used,
            "particles_rotation_date": datetime.now(timezone.utc).date(),
            "particles_rotation_cards": json.dumps([55]),
            "particles_purchased_today": "[]",
        }

    def get_card_max_level(self, card_obj):
        return 10

    def get_upgrade_cost(self, rarity, level, simplified_levelup=False):
        return {"particles": 0 if int(level) >= 10 else 100}

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        normalized = " ".join(str(query).split())

        if "FROM cards c LEFT JOIN user_cards" in normalized or "SELECT c.rarity" in normalized:
            return {
                "rarity": "common",
                "level": 10,
                "current_particles": 500,
                "simplified_levelup": False,
            }
        if "UPDATE users SET coins" in normalized:
            return {"coins": 9800}
        if "SELECT COALESCE(particles" in normalized:
            return {"particles": 550}
        if "UPDATE users" in normalized and "RETURNING gems" in normalized:
            return {"gems": 999}
        if "inserted_case" in normalized:
            return {"remaining_gems": 985, "user_case_id": 777}
        if "FROM cosmetic_items" in normalized and "slug = $1" in normalized:
            row = self.cosmetics.get(str(args[0] if args else ""))
            if row and row.get("is_active", True):
                return dict(row)
            return None
        return None

    async def fetch(self, query, *args):
        return []

    async def fetchval(self, query, *args):
        return 1

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"

    async def get_admin_case_id(self, tier):
        return 200 + int(tier)

    async def add_user_case(self, user_id, case_id, tier):
        self.case_grants.append((int(user_id), int(case_id), int(tier)))
        return {"success": True, "user_case_id": 9000 + int(tier)}

    async def increment_user_keys(self, user_id, amount):
        return {"success": True}

    async def sync_user_key_cases(self, user_id):
        return {"success": True}

    async def get_ruble_products(self, active_only=True, surface=None):
        self.last_ruble_products_surface = surface
        rows = [dict(row) for row in self.ruble_products]
        if active_only:
            rows = [row for row in rows if row.get("is_active", True)]
        if surface == "shop":
            rows = [row for row in rows if row.get("show_in_shop", True)]
        elif surface == "game":
            rows = [row for row in rows if row.get("show_in_game", True)]
        return rows

    async def get_ruble_product(self, code):
        for row in self.ruble_products:
            if str(row.get("code")) == str(code):
                return dict(row)
        return None

    async def get_shop_sets(self, active_only=True):
        rows = [dict(row) for row in self.shop_sets.values()]
        if active_only:
            rows = [row for row in rows if row.get("is_active", True)]
        return rows

    async def create_ruble_product(self, **kwargs):
        self.ruble_product_creates.append(dict(kwargs))
        return {"success": True, "product_id": len(self.ruble_product_creates)}

    async def update_ruble_product(self, code_or_id, **kwargs):
        self.ruble_product_updates.append((code_or_id, dict(kwargs)))
        try:
            identity_id = int(code_or_id)
        except (TypeError, ValueError):
            identity_id = None
        for row in self.ruble_products:
            if str(row.get("code")) == str(code_or_id) or (identity_id is not None and int(row.get("id") or 0) == identity_id):
                row.update(kwargs)
                return {"success": True}
        return {"success": False, "error": "product_not_found"}

    async def get_shop_set(self, set_id):
        row = self.shop_sets.get(int(set_id))
        return dict(row) if row else None

    async def get_claimed_shop_set_ids(self, user_id):
        return {int(set_id) for set_id in self.claimed_shop_set_ids}

    async def track_economy_event(self, **kwargs):
        self.economy_events.append(kwargs)
        return {"success": True}

    async def update_user_settings(self, user_id, **kwargs):
        self.settings_updates.append((int(user_id), kwargs))
        return {"success": True}


def _auth_token(session_id: str, user_id: int = 1001) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "user_id": user_id,
            "session_id": session_id,
            "iat": now,
            "exp": now + 600,
        },
        get_settings().jwt_secret,
        algorithm="HS256",
    )


async def _client(db, **app_kwargs):
    session_id = str(uuid.uuid4())
    app = web_server.create_web_app(
        db,
        bot_token="bot-token",
        extraid_db=FakeExtraIDDB(session_id),
        webapp_url="https://game.example",
        **app_kwargs,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, session_id


@pytest.mark.asyncio
async def test_public_case_tier_items_cannot_be_bought_through_shop_buy():
    db = ShopAuditFakeDB()
    client, session_id = await _client(db)
    try:
        token = _auth_token(session_id, user_id=1001)
        response = await client.post(
            "/api/shop/buy",
            headers={"Authorization": f"Bearer {token}"},
            json={"item_type": "case_tier_2", "item_name": "Public tier case"},
        )
        body = await response.json()

        assert response.status in {400, 403}
        assert body.get("success") is not True
        assert not any("UPDATE users" in query for query, _args in db.fetchrow_calls)
        assert db.case_grants == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_admin_case_tier_items_remain_admin_only_and_grant_cases_for_admins():
    db = ShopAuditFakeDB(admin_ids={web_server.ADMIN_ID})
    client, session_id = await _client(db)
    try:
        user_token = _auth_token(session_id, user_id=1001)
        denied = await client.post(
            "/api/shop/buy",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"item_type": "admin_case_tier_2", "item_name": "Admin tier case"},
        )
        assert denied.status == 403
        assert db.case_grants == []

        admin_token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        allowed = await client.post(
            "/api/shop/buy",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"item_type": "admin_case_tier_2", "item_name": "Admin tier case"},
        )
        body = await allowed.json()

        assert allowed.status == 200
        assert body["success"] is True
        assert body["granted_case_tier"] == 2
        assert db.case_grants == [(web_server.ADMIN_ID, 202, 2)]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_shop_buy_500_response_does_not_expose_internal_exception_text():
    db = ShopAuditFakeDB(fail_profile=True)
    client, session_id = await _client(db)
    try:
        token = _auth_token(session_id, user_id=1001)
        response = await client.post(
            "/api/shop/buy",
            headers={"Authorization": f"Bearer {token}"},
            json={"item_type": "coins_300", "item_name": "Coins"},
        )
        body = await response.json()

        assert response.status == 500
        serialized = json.dumps(body)
        assert body["error"] == "internal_server_error"
        assert "message" in body
        assert "secret_shop_audit_table" not in serialized
        assert "RuntimeError" not in serialized
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_particle_buy_rejects_max_level_cards_by_default_before_debit():
    db = ShopAuditFakeDB()
    client, session_id = await _client(db)
    try:
        token = _auth_token(session_id, user_id=1001)
        response = await client.post(
            "/api/shop/particles/buy",
            headers={"Authorization": f"Bearer {token}"},
            json={"card_id": 55},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "max_level_reached"
        assert not any("UPDATE users SET coins" in query for query, _args in db.fetchrow_calls)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_particle_buy_allows_max_level_cards_when_shop_flag_is_enabled():
    db = ShopAuditFakeDB()
    client, session_id = await _client(db, shop_allow_max_level_particles=True)
    try:
        token = _auth_token(session_id, user_id=1001)
        response = await client.post(
            "/api/shop/particles/buy",
            headers={"Authorization": f"Bearer {token}"},
            json={"card_id": 55},
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert body["updated_particles"] == 550
        assert any("UPDATE users SET coins" in query for query, _args in db.fetchrow_calls)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_ruble_products_unknown_surface_keeps_shop_visibility_filter():
    db = ShopAuditFakeDB()
    db.ruble_products = [
        {
            "code": "visible",
            "item_type": "extrapass",
            "name": "Visible",
            "price": 179,
            "currency": "rubles",
            "is_active": True,
            "show_in_shop": True,
            "show_in_game": True,
        },
        {
            "code": "hidden",
            "item_type": "extrapass",
            "name": "Hidden",
            "price": 179,
            "currency": "rubles",
            "is_active": True,
            "show_in_shop": False,
            "show_in_game": False,
        },
    ]
    client, _session_id = await _client(db)
    try:
        response = await client.get("/api/shop/ruble-products?surface=anything")
        body = await response.json()

        assert response.status == 200
        assert db.last_ruble_products_surface == "shop"
        assert [product["code"] for product in body["products"]] == ["visible"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_game_shop_sets_exclude_sets_owned_by_active_ruble_products():
    db = ShopAuditFakeDB()
    db.shop_sets = {
        7: {"id": 7, "name": "Shop-only bundle", "is_active": True, "rewards": [{"type": "gems", "amount": 10}]},
        8: {"id": 8, "name": "Legacy bundle", "is_active": True, "rewards": [{"type": "coins", "amount": 50}]},
    }
    db.ruble_products = [
        {
            "code": "shop_only_bundle",
            "item_type": "shop_set",
            "shop_set_id": 7,
            "name": "Shop-only bundle",
            "price": 99,
            "currency": "rubles",
            "is_active": True,
            "show_in_shop": True,
            "show_in_game": False,
        }
    ]
    client, _session_id = await _client(db)
    try:
        response = await client.get("/api/shop/sets?surface=game")
        body = await response.json()

        assert response.status == 200
        assert [shop_set["id"] for shop_set in body["sets"]] == [8]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_shop_sets_hide_sets_with_inactive_cosmetic_rewards():
    db = ShopAuditFakeDB()
    db.cosmetics = {
        "avatar_live": {
            "slug": "avatar_live",
            "item_type": "avatar",
            "class": "rare",
            "name": "Live Avatar",
            "asset_path": "/DesignAssets/PlayerCosmetics/Avatars/Admin/avatar_live.png",
            "media_type": "image",
            "is_active": True,
        },
        "avatar_archived": {
            "slug": "avatar_archived",
            "item_type": "avatar",
            "class": "rare",
            "name": "Archived Avatar",
            "asset_path": "/DesignAssets/PlayerCosmetics/Avatars/Admin/avatar_archived.png",
            "media_type": "image",
            "is_active": False,
        },
    }
    db.shop_sets = {
        9: {"id": 9, "name": "Live cosmetics", "is_active": True, "rewards": [{"type": "cosmetic", "cosmetic_slug": "avatar_live"}]},
        10: {"id": 10, "name": "Archived cosmetics", "is_active": True, "rewards": [{"type": "cosmetic", "cosmetic_slug": "avatar_archived"}]},
    }
    client, _session_id = await _client(db)
    try:
        response = await client.get("/api/shop/sets?surface=shop")
        body = await response.json()

        assert response.status == 200
        assert [shop_set["id"] for shop_set in body["sets"]] == [9]
        assert body["sets"][0]["rewards"][0]["cosmetic_type"] == "avatar"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mobile_shop_bootstrap_hides_claimed_shop_sets_and_products():
    db = ShopAuditFakeDB()
    db.claimed_shop_set_ids = {7, 9}
    db.shop_sets = {
        7: {"id": 7, "name": "Claimed legacy", "is_active": True, "rewards": [{"type": "gems", "amount": 10}]},
        8: {"id": 8, "name": "Visible legacy", "is_active": True, "rewards": [{"type": "coins", "amount": 50}]},
        9: {"id": 9, "name": "Claimed paid", "is_active": True, "rewards": [{"type": "keys", "amount": 1}]},
        10: {"id": 10, "name": "Visible paid", "is_active": True, "rewards": [{"type": "gems", "amount": 25}]},
    }
    db.ruble_products = [
        {
            "code": "claimed_paid",
            "item_type": "shop_set",
            "shop_set_id": 9,
            "name": "Claimed paid",
            "price": 10,
            "currency": "rubles",
            "is_active": True,
            "show_in_game": True,
        },
        {
            "code": "visible_paid",
            "item_type": "shop_set",
            "shop_set_id": 10,
            "name": "Visible paid",
            "price": 10,
            "currency": "rubles",
            "is_active": True,
            "show_in_game": True,
        },
    ]
    client, session_id = await _client(db)
    try:
        token = _auth_token(session_id, user_id=1001)
        response = await client.get(f"/api/mobile/shop-bootstrap?_auth={token}")
        body = await response.json()

        assert response.status == 200
        assert [shop_set["id"] for shop_set in body["shop_sets"]["sets"]] == [8]
        assert [product["code"] for product in body["ruble_products"]["products"]] == ["visible_paid"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_shop_catalog_hides_starter_once_gems_after_first_purchase():
    db = ShopAuditFakeDB(starter_pack_used=True)
    db.ruble_products = [
        {
            "code": "gems_starter_once",
            "item_type": "gems_package",
            "package_type": "starter_once",
            "name": "50 гемов (стартовый)",
            "price": 49,
            "show_in_shop": True,
            "is_active": True,
        },
        {
            "code": "gems_100",
            "item_type": "gems_package",
            "package_type": "gems_100",
            "name": "100 гемов",
            "price": 99,
            "show_in_shop": True,
            "is_active": True,
        },
    ]
    client, session_id = await _client(db)
    try:
        response = await client.get(f"/api/shop/catalog?_auth={_auth_token(session_id)}")
        body = await response.json()

        assert response.status == 200
        package_types = {item["package_type"] for item in body["gem_products"]}
        assert "starter_once" not in package_types
        assert "gems_100" in package_types
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_admin_ruble_product_rejects_zero_price_for_paid_products():
    db = ShopAuditFakeDB(admin_ids={web_server.ADMIN_ID})
    client, session_id = await _client(db)
    try:
        response = await client.post(
            "/api/admin/ruble-products/create",
            headers={"Authorization": f"Bearer {_auth_token(session_id, user_id=web_server.ADMIN_ID)}"},
            json={
                "code": "free_pass",
                "item_type": "extrapass",
                "name": "Free Pass",
                "price": 0,
                "currency": "rubles",
            },
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "invalid_price"
        assert db.ruble_product_creates == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_admin_ruble_product_rejects_unsafe_product_codes():
    db = ShopAuditFakeDB(admin_ids={web_server.ADMIN_ID})
    client, session_id = await _client(db)
    try:
        response = await client.post(
            "/api/admin/ruble-products/create",
            headers={"Authorization": f"Bearer {_auth_token(session_id, user_id=web_server.ADMIN_ID)}"},
            json={
                "code": 'bad"]code',
                "item_type": "extrapass",
                "name": "Unsafe Code",
                "price": 179,
                "currency": "rubles",
            },
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "invalid_code"
        assert db.ruble_product_creates == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_admin_gift_shop_set_requires_shop_set_and_zero_price():
    db = ShopAuditFakeDB(admin_ids={web_server.ADMIN_ID})
    db.shop_sets[7] = {"id": 7, "is_active": True, "rewards": [{"type": "gems", "amount": 10}]}
    client, session_id = await _client(db)
    try:
        missing_set = await client.post(
            "/api/admin/ruble-products/create",
            headers={"Authorization": f"Bearer {_auth_token(session_id, user_id=web_server.ADMIN_ID)}"},
            json={
                "code": "gift_missing_set",
                "item_type": "gift_shop_set",
                "name": "Gift Missing Set",
                "price": 0,
                "currency": "rubles",
            },
        )
        missing_body = await missing_set.json()
        paid_gift = await client.post(
            "/api/admin/ruble-products/create",
            headers={"Authorization": f"Bearer {_auth_token(session_id, user_id=web_server.ADMIN_ID)}"},
            json={
                "code": "paid_gift",
                "item_type": "gift_shop_set",
                "name": "Paid Gift",
                "shop_set_id": 7,
                "price": 1,
                "currency": "rubles",
            },
        )
        paid_body = await paid_gift.json()
        wrong_currency = await client.post(
            "/api/admin/ruble-products/create",
            headers={"Authorization": f"Bearer {_auth_token(session_id, user_id=web_server.ADMIN_ID)}"},
            json={
                "code": "gift_wrong_currency",
                "item_type": "gift_shop_set",
                "name": "Wrong Currency Gift",
                "shop_set_id": 7,
                "price": 0,
                "currency": "gems",
            },
        )
        wrong_currency_body = await wrong_currency.json()

        assert missing_set.status == 400
        assert missing_body["error"] == "shop_set_id_required"
        assert paid_gift.status == 400
        assert paid_body["error"] == "invalid_price"
        assert wrong_currency.status == 400
        assert wrong_currency_body["error"] == "invalid_currency"
        assert db.ruble_product_creates == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_admin_ruble_product_update_by_id_preserves_gift_price_currency_invariants():
    db = ShopAuditFakeDB(admin_ids={web_server.ADMIN_ID})
    db.shop_sets[7] = {"id": 7, "is_active": True, "rewards": [{"type": "gems", "amount": 10}]}
    db.ruble_products.append({
        "id": 42,
        "code": "paid_set",
        "item_type": "shop_set",
        "shop_set_id": 7,
        "name": "Paid Set",
        "price": 199.0,
        "currency": "rubles",
        "is_active": True,
    })
    client, session_id = await _client(db)
    try:
        response = await client.post(
            "/api/admin/ruble-products/update",
            headers={"Authorization": f"Bearer {_auth_token(session_id, user_id=web_server.ADMIN_ID)}"},
            json={
                "id": 42,
                "item_type": "gift_shop_set",
                "shop_set_id": 7,
            },
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "invalid_price"
        assert db.ruble_product_updates == []

        currency_response = await client.post(
            "/api/admin/ruble-products/update",
            headers={"Authorization": f"Bearer {_auth_token(session_id, user_id=web_server.ADMIN_ID)}"},
            json={
                "id": 42,
                "item_type": "gift_shop_set",
                "shop_set_id": 7,
                "price": 0,
                "currency": "gems",
            },
        )
        currency_body = await currency_response.json()

        assert currency_response.status == 400
        assert currency_body["error"] == "invalid_currency"
        assert db.ruble_product_updates == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_shop_set_image_proxy_does_not_expose_exception_text(monkeypatch):
    class ExplodingBot:
        def __init__(self, token):
            self.session = types.SimpleNamespace(close=lambda: None)

        async def get_file(self, file_id):
            raise RuntimeError("secret_telegram_file_path")

    fake_aiogram = types.SimpleNamespace(Bot=ExplodingBot)
    monkeypatch.setitem(sys.modules, "aiogram", fake_aiogram)

    db = ShopAuditFakeDB()
    client, _session_id = await _client(db)
    try:
        response = await client.get("/api/shop/sets/image?file_id=bad-file")
        body = await response.json()
        serialized = json.dumps(body)

        assert response.status == 500
        assert body["error"] == "image_fetch_failed"
        assert "secret_telegram_file_path" not in serialized
        assert "RuntimeError" not in serialized
    finally:
        await client.close()


def test_particle_buy_checks_shop_flag_before_debiting_max_level_cards():
    server = Path("web/server.py").read_text(encoding="utf-8")
    buy_block = server.split("async def particles_buy_handler", 1)[1].split(
        "async def shop_sets_image_handler",
        1,
    )[0]

    assert "max_level_reached" in buy_block
    assert (
        "allow_max_level_particle_purchase" in buy_block
        or "shop_allow_max_level_particle_purchase" in buy_block
        or "allow_max_level_particles" in buy_block
    )
    assert buy_block.index("max_level_reached") < buy_block.index("UPDATE users SET coins")


def test_real_money_success_modal_does_not_interpolate_payment_text_as_html():
    source = INDEX.read_text(encoding="utf-8")
    modal_block = source.split("function showRealMoneyPurchaseSuccess", 1)[1].split(
        "async function refreshProfileStateAfterPaymentSuccess",
        1,
    )[0]

    assert "escapeHtml" in modal_block
    for unsafe_fragment in (
        "+ payload.title +",
        "+ payload.subtitle +",
        "+ payload.providerLabel +",
        "+ reward.icon +",
        "+ reward.label +",
        "+ reward.value +",
    ):
        assert unsafe_fragment not in modal_block


def test_canceled_payment_status_clears_pending_payment_session():
    source = INDEX.read_text(encoding="utf-8")
    status_poll = source.split("var startAppPaymentStatusCheck = function(paymentId, provider)", 1)[1].split(
        "var startAppCheckoutSessionCheck = function(jti, provider)",
        1,
    )[0]
    checkout_poll = source.split("var startAppCheckoutSessionCheck = function(jti, provider)", 1)[1].split(
        "var doStars",
        1,
    )[0]
    visibility_poll = source.split("const pendingCheckoutJti = sessionStorage.getItem('pending_checkout_jti');", 1)[1].split(
        "await triggerPaymentSuccessFromRecent(authData);",
        1,
    )[0]

    canceled_status_block = status_poll.split("if (d.status === 'canceled')", 1)[1].split("return;", 1)[0]
    canceled_checkout_block = checkout_poll.split("if (d.payment_status === 'canceled')", 1)[1].split("return;", 1)[0]

    assert "clearPendingPaymentSession();" in canceled_status_block
    assert "clearPendingPaymentSession();" in canceled_checkout_block
    assert "session.payment_status === 'canceled'" in visibility_poll
    assert visibility_poll.count("clearPendingPaymentStorage(") >= 2
