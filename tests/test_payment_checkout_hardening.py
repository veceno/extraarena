import asyncio
import hashlib
import json
import time
import uuid
from threading import Lock

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.robokassa_payments import RobokassaPaymentService, RobokassaSettings
from infrastructure.config import DatabaseSettings, get_settings
from infrastructure.database import Database
from web import server as web_server


STRONG_TEST_JWT_SECRET = "test-payment-checkout-jwt-secret-that-is-long-enough-2026"


@pytest.fixture(autouse=True)
def _strong_jwt_secret(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeExtraIDDB:
    def __init__(self, session_id: str, user_id: int = 1001):
        self.session_id = session_id
        self.user_id = user_id

    async def verify_session(self, session_uuid, token: str):
        if str(session_uuid) != self.session_id:
            return None
        return {"session_id": session_uuid, "user_id": self.user_id}

    async def check_rate_limit(self, key, max_requests, window_seconds):
        return True


@pytest.mark.asyncio
async def test_database_get_checkout_session_decodes_jsonb_metadata_string():
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))
    db._pool = object()

    async def fake_fetchrow(query, checkout_jti):
        return {
            "checkout_jti": checkout_jti,
            "user_id": 1001,
            "item_type": "extrapass_ultra",
            "amount": 349.0,
            "metadata": json.dumps({"item_name": "ExtraPass Ultra", "amount_rub": 349.0}),
            "payment_id": None,
            "confirmation_url": None,
            "status": "created",
            "expires_at": time.time() + 3600,
        }

    db.fetchrow = fake_fetchrow

    session = await db.get_checkout_session("checkout-jsonb")

    assert session["metadata"] == {"item_name": "ExtraPass Ultra", "amount_rub": 349.0}


class CheckoutFakeDB:
    def __init__(self):
        self.created_payments = []
        self.payment_records = {}
        self.updated_statuses = []
        self.checkout_sessions = {}
        self.one_time_reservations = {}
        self.settings = {}
        self.shop_sets = {}
        self.gift_claims = {}
        self.user_gems = {}
        self.created_mail = []
        self.economy_events = []
        self.cards = {}
        self.cosmetics = {}
        self.owned_cards = []
        self.product = {
            "code": "extrapass",
            "item_type": "extrapass",
            "package_type": None,
            "name": "ExtraPass",
            "price": 179.0,
            "currency": "rubles",
            "is_active": False,
        }

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},
            "disabled_card_ids": [],
        }

    async def is_admin(self, user_id):
        return False

    async def get_match_mode_overrides(self):
        return []

    async def is_match_mode_enabled(self, mode_id):
        return True

    async def get_disabled_card_ids(self):
        return []

    async def get_ruble_product(self, code):
        if str(code) == self.product["code"]:
            return dict(self.product)
        return None

    async def get_ruble_products(self, active_only=True, surface=None):
        rows = [dict(self.product)]
        if active_only:
            rows = [row for row in rows if row.get("is_active", True)]
        return rows

    async def fetchrow(self, query, *args):
        if "FROM cards" in query:
            card = self.cards.get(int(args[0]))
            return dict(card) if card else None
        if "FROM cosmetic_items" in query:
            cosmetic = self.cosmetics.get(str(args[0]))
            return dict(cosmetic) if cosmetic else None
        return None

    async def get_user_settings(self, user_id):
        return dict(self.settings)

    async def get_shop_set(self, set_id):
        row = self.shop_sets.get(int(set_id))
        return dict(row) if row else None

    async def get_user_cards(self, user_id):
        return [dict(card) for card in self.owned_cards]

    async def claim_gift_shop_set(self, user_id, product_code):
        product = await self.get_ruble_product(product_code)
        if not product:
            return {"success": False, "error": "product_not_found"}
        if product.get("item_type") != "gift_shop_set":
            return {"success": False, "error": "not_gift_shop_set"}
        if not product.get("show_in_game", True) and not product.get("show_in_shop", True):
            return {"success": False, "error": "product_not_found"}
        if float(product.get("price") or 0) != 0:
            return {"success": False, "error": "invalid_gift_price"}
        shop_set_id = int(product.get("shop_set_id") or 0)
        if shop_set_id <= 0 or shop_set_id not in self.shop_sets:
            return {"success": False, "error": "shop_set_not_found"}
        key = (int(user_id), str(product_code))
        if key in self.gift_claims:
            return {"success": False, "error": "already_claimed"}
        granted = [{"type": "gems", "amount": 25}]
        self.gift_claims[key] = {"shop_set_id": shop_set_id, "granted": granted}
        return {"success": True, "product_code": str(product_code), "shop_set_id": shop_set_id, "granted": granted}

    async def expire_announcements(self):
        return 0

    async def process_weekly_squad_cbrp(self):
        return {"processed": 0}

    async def refresh_due_rating_snapshots(self, scope="players"):
        return {"refreshed": 0}

    async def create_payment(self, **kwargs):
        self.created_payments.append(kwargs)
        self.payment_records[kwargs["payment_id"]] = {
            **kwargs,
            "status": kwargs.get("status", "pending"),
            "rewards_processed": False,
        }
        return {"success": True}

    async def get_payment_by_id(self, payment_id):
        record = self.payment_records.get(payment_id)
        return dict(record) if record else None

    async def update_payment_status(self, *, payment_id, status):
        self.updated_statuses.append((payment_id, status))
        if payment_id in self.payment_records:
            self.payment_records[payment_id]["status"] = status

    async def execute(self, query, *args):
        if "UPDATE users SET gems = gems + $1 WHERE user_id = $2" in query:
            amount, user_id = int(args[0]), int(args[1])
            self.user_gems[user_id] = self.user_gems.get(user_id, 0) + amount
        return "UPDATE 1"

    async def fetchval(self, query, *args):
        if "UPDATE payments" in query and "SET rewards_processed = TRUE" in query:
            payment_id = str(args[0])
            if payment_id in self.payment_records and not self.payment_records[payment_id].get("rewards_processed"):
                self.payment_records[payment_id]["rewards_processed"] = True
                return 1
            return None
        return 1

    async def create_mail(self, **kwargs):
        self.created_mail.append(kwargs)
        return {"success": True, "mail_id": len(self.created_mail)}

    async def track_economy_event(self, **kwargs):
        self.economy_events.append(kwargs)
        return {"success": True}

    async def create_checkout_session(self, **kwargs):
        jti = kwargs["checkout_jti"]
        self.checkout_sessions[jti] = {
            "checkout_jti": jti,
            "user_id": kwargs["user_id"],
            "item_type": kwargs["item_type"],
            "amount": kwargs["amount"],
            "metadata": dict(kwargs.get("metadata") or {}),
            "payment_id": None,
            "confirmation_url": None,
            "status": "created",
            "expires_at": kwargs["expires_at"],
            "updated_at": time.time(),
        }
        return {"success": True}

    async def get_checkout_session(self, checkout_jti):
        row = self.checkout_sessions.get(checkout_jti)
        return dict(row) if row else None

    async def attach_checkout_payment(self, checkout_jti, payment_id, confirmation_url):
        row = self.checkout_sessions.get(checkout_jti)
        if not row:
            return {"success": False, "error": "session_not_found"}
        if row.get("payment_id"):
            return {
                "success": True,
                "payment_id": row["payment_id"],
                "confirmation_url": row["confirmation_url"],
            }
        row["payment_id"] = payment_id
        row["confirmation_url"] = confirmation_url
        row["status"] = "payment_created"
        row["updated_at"] = time.time()
        return {"success": True, "payment_id": payment_id, "confirmation_url": confirmation_url}

    async def claim_checkout_session_for_payment(self, checkout_jti):
        row = self.checkout_sessions.get(checkout_jti)
        if not row:
            return {"success": False, "error": "session_not_found"}
        if row.get("payment_id") and row.get("confirmation_url"):
            return {
                "success": False,
                "error": "payment_already_created",
                "payment_id": row["payment_id"],
                "confirmation_url": row["confirmation_url"],
            }
        if row.get("status") == "payment_creating" and time.time() - float(row.get("updated_at") or 0) <= 60:
            return {"success": False, "error": "payment_creation_in_progress"}
        row["status"] = "payment_creating"
        row["updated_at"] = time.time()
        return {"success": True, "session": dict(row)}

    async def reserve_one_time_payment(self, **kwargs):
        key = (int(kwargs["user_id"]), str(kwargs["product_key"]))
        if self.settings.get("starter_pack_used"):
            return {"success": False, "error": "already_used"}
        existing = self.one_time_reservations.get(key)
        if existing and existing.get("status") in {"reserved", "payment_created"}:
            return {
                "success": False,
                "error": "starter_checkout_in_progress",
                "reservation": dict(existing),
            }
        record = {
            "user_id": key[0],
            "product_key": key[1],
            "reservation_id": kwargs["reservation_id"],
            "checkout_jti": kwargs.get("checkout_jti"),
            "payment_id": None,
            "provider": kwargs.get("provider"),
            "status": "reserved",
            "expires_at": kwargs.get("expires_at"),
        }
        self.one_time_reservations[key] = record
        return {"success": True, "reservation": dict(record)}

    async def attach_one_time_payment(self, **kwargs):
        key = (int(kwargs["user_id"]), str(kwargs["product_key"]))
        row = self.one_time_reservations.get(key)
        if not row:
            return {"success": False, "error": "reservation_not_found"}
        row["payment_id"] = kwargs["payment_id"]
        row["provider"] = kwargs.get("provider")
        row["status"] = "payment_created"
        return {"success": True, "reservation": dict(row)}

    async def release_one_time_payment_reservation(self, **kwargs):
        key = (int(kwargs["user_id"]), str(kwargs["product_key"]))
        row = self.one_time_reservations.get(key)
        if row and row.get("reservation_id") == kwargs.get("reservation_id"):
            row["status"] = "released"
        return {"success": True}


class FakePaymentService:
    def __init__(self):
        self.created = []
        self.statuses = {}
        self._lock = Lock()
        self._counter = 0

    def create_payment(self, **kwargs):
        time.sleep(0.02)
        with self._lock:
            self._counter += 1
            index = self._counter
            self.created.append(kwargs)
        return {
            "success": True,
            "payment_id": f"pay-{index}",
            "confirmation_url": f"https://pay.example/confirm/{index}",
            "status": "pending",
        }

    def get_payment_status(self, payment_id):
        return dict(self.statuses.get(payment_id, {"success": False, "error": "not_found"}))


class FakeStarsSession:
    async def close(self):
        return None


class FakeStarsBot:
    invoices = []
    messages = []

    def __init__(self, token, **kwargs):
        self.token = token
        self.kwargs = kwargs
        self.session = FakeStarsSession()

    async def send_message(self, chat_id, text, parse_mode=None):
        self.messages.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode})

    async def send_invoice(self, **kwargs):
        self.invoices.append(kwargs)
        return type("FakeStarsMessage", (), {"message_id": len(self.invoices)})()


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


async def _client(db=None, payment_service=None, robokassa_payment_service=None):
    session_id = str(uuid.uuid4())
    app = web_server.create_web_app(
        db or CheckoutFakeDB(),
        bot_token="bot-token",
        extraid_db=FakeExtraIDDB(session_id),
        payment_service=payment_service or FakePaymentService(),
        robokassa_payment_service=robokassa_payment_service,
        webapp_url="https://game.example",
        extra_shop_url="https://laveqox.ru",
        payment_primary_provider="robokassa" if robokassa_payment_service else "yookassa",
        payment_fallback_provider="yookassa",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, session_id


@pytest.mark.asyncio
async def test_payment_create_rejects_inactive_db_product_without_legacy_fallback():
    db = CheckoutFakeDB()
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={
                "product_code": "extrapass",
                "item_type": "extrapass",
            },
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "product_inactive"
        assert payment_service.created == []
        assert db.created_payments == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_create_rejects_db_starter_once_after_purchase_before_provider_call():
    db = CheckoutFakeDB()
    db.settings = {"starter_pack_used": True}
    db.product = {
        "code": "gems_starter_once",
        "item_type": "gems_package",
        "package_type": "starter_once",
        "name": "Starter gems",
        "price": 49.0,
        "currency": "rubles",
        "is_active": True,
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={"product_code": "gems_starter_once"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "already_used"
        assert payment_service.created == []
        assert db.created_payments == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_create_rejects_non_ruble_db_product_before_provider_call():
    db = CheckoutFakeDB()
    db.product = {
        "code": "coins_product",
        "item_type": "extrapass",
        "package_type": None,
        "name": "Coin-priced pass",
        "price": 179.0,
        "currency": "coins",
        "is_active": True,
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={"product_code": "coins_product"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "product_currency_not_rubles"
        assert payment_service.created == []
        assert db.created_payments == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_create_omits_unsafe_db_product_image_url_from_metadata():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
        "image_url": 'https://cdn.example/pass.jpg" onerror="alert(1)',
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={"product_code": "extrapass"},
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert "image_url" not in payment_service.created[0]["metadata"]
        assert "image_url" not in db.created_payments[0]["metadata"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_create_rejects_db_gem_package_with_unknown_package_type():
    db = CheckoutFakeDB()
    db.product = {
        "code": "bad_gems",
        "item_type": "gems_package",
        "package_type": "missing_package",
        "name": "Bad gems",
        "price": 99.0,
        "currency": "rubles",
        "is_active": True,
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={"product_code": "bad_gems"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "invalid_package_type"
        assert payment_service.created == []
        assert db.created_payments == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_create_rejects_misconfigured_shop_set_product():
    db = CheckoutFakeDB()
    db.product = {
        "code": "bad_set",
        "item_type": "shop_set",
        "package_type": None,
        "shop_set_id": 7,
        "name": "Bad Set",
        "price": 199.0,
        "currency": "rubles",
        "is_active": True,
    }
    db.shop_sets[7] = {
        "id": 7,
        "name": "Bad Set",
        "price": 199.0,
        "currency": "rubles",
        "is_active": True,
        "rewards": [],
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={"product_code": "bad_set"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "empty_rewards"
        assert payment_service.created == []
        assert db.created_payments == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_create_rejects_gift_shop_set_before_provider_call():
    db = CheckoutFakeDB()
    db.product = {
        "code": "welcome_gift",
        "item_type": "gift_shop_set",
        "package_type": None,
        "shop_set_id": 9,
        "name": "Welcome Gift",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
    }
    db.shop_sets[9] = {
        "id": 9,
        "name": "Welcome Gift Set",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
        "rewards": [{"type": "gems", "amount": 25}],
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={"product_code": "welcome_gift"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "gift_checkout_forbidden"
        assert payment_service.created == []
        assert db.created_payments == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_start_rejects_gift_shop_set_without_session():
    db = CheckoutFakeDB()
    db.product = {
        "code": "welcome_gift",
        "item_type": "gift_shop_set",
        "package_type": None,
        "shop_set_id": 9,
        "name": "Welcome Gift",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
    }
    db.shop_sets[9] = {
        "id": 9,
        "name": "Welcome Gift Set",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
        "rewards": [{"type": "gems", "amount": 25}],
    }
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "welcome_gift"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "gift_checkout_forbidden"
        assert db.checkout_sessions == {}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_authenticated_gift_shop_set_claim_grants_once_per_product():
    db = CheckoutFakeDB()
    db.product = {
        "code": "welcome_gift",
        "item_type": "gift_shop_set",
        "package_type": None,
        "shop_set_id": 9,
        "name": "Welcome Gift",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
    }
    db.shop_sets[9] = {
        "id": 9,
        "name": "Welcome Gift Set",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
        "rewards": [{"type": "gems", "amount": 25}],
    }
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id)
        first = await client.post(
            f"/api/shop/gifts/claim?_auth={token}",
            json={"product_code": "welcome_gift"},
        )
        assert first.status == 200
        first_body = await first.json()
        second = await client.post(
            f"/api/shop/gifts/claim?_auth={token}",
            json={"product_code": "welcome_gift"},
        )
        assert second.status == 409
        second_body = await second.json()

        assert first_body["success"] is True
        assert first_body["granted"] == [{"type": "gems", "amount": 25}]
        assert second_body["error"] == "already_claimed"
        assert len(db.gift_claims) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_ruble_products_enrich_shop_set_rewards_for_pack_cards():
    db = CheckoutFakeDB()
    db.product = {
        "code": "welcome_gift",
        "item_type": "gift_shop_set",
        "package_type": None,
        "shop_set_id": 9,
        "name": "Welcome Gift",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
        "show_in_shop": True,
    }
    db.shop_sets[9] = {
        "id": 9,
        "name": "Welcome Gift Set",
        "description": "Gift pack",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
        "rewards": [
            {"type": "card", "card_id": 77},
            {"type": "cosmetic", "cosmetic_slug": "avatar_gold", "auto_equip": True},
            {"type": "gems", "amount": 25},
        ],
    }
    db.cards[77] = {
        "id": 77,
        "name": "Arena Captain",
        "description": "Leader card",
        "rarity": "epic",
        "mechanics": {"description": "Charges first.", "charge": 1},
        "mechanics_desc": "",
        "card_type": "warrior",
        "mana_cost": 4,
        "base_attack": 6,
        "base_hp": 7,
    }
    db.cosmetics["avatar_gold"] = {
        "slug": "avatar_gold",
        "item_type": "avatar",
        "class": "gold",
        "name": "Gold Avatar",
        "asset_path": "/static/avatar_gold.png",
        "media_type": "image",
    }
    client, _session_id = await _client(db=db)
    try:
        response = await client.get("/api/shop/ruble-products?surface=shop")
        body = await response.json()

        assert response.status == 200
        product = body["products"][0]
        assert product["is_gift"] is True
        assert product["shop_set"]["name"] == "Welcome Gift Set"
        assert product["rewards"][0]["card_name"] == "Arena Captain"
        assert product["rewards"][0]["mechanics"] == "Charges first."
        assert product["rewards"][0]["card_mana"] == 4
        assert product["rewards"][0]["card_attack"] == 6
        assert product["rewards"][0]["card_hp"] == 7
        assert product["rewards"][0]["card_image_url"] == "/api/cards/image?card_id=77"
        assert product["rewards"][1]["cosmetic_slug"] == "avatar_gold"
        assert product["rewards"][1]["cosmetic_type"] == "avatar"
        assert product["rewards"][1]["name"] == "Gold Avatar"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_ruble_products_hide_owned_card_only_shop_set():
    db = CheckoutFakeDB()
    db.product = {
        "code": "owned_card_pack",
        "item_type": "shop_set",
        "package_type": None,
        "shop_set_id": 11,
        "name": "Owned Card Pack",
        "price": 199.0,
        "currency": "rubles",
        "is_active": True,
        "show_in_game": True,
    }
    db.shop_sets[11] = {
        "id": 11,
        "name": "Owned Card Set",
        "description": "Only card",
        "price": 199.0,
        "currency": "rubles",
        "is_active": True,
        "rewards": [{"type": "card", "card_id": 77}],
    }
    db.cards[77] = {
        "id": 77,
        "name": "Arena Captain",
        "description": "Leader card",
        "rarity": "epic",
        "mechanics": {},
        "mechanics_desc": "",
        "card_type": "warrior",
        "mana_cost": 4,
        "base_attack": 6,
        "base_hp": 7,
    }
    db.owned_cards = [{"id": 77}]
    client, session_id = await _client(db=db)
    try:
        response = await client.get(f"/api/shop/ruble-products?surface=game&_auth={_auth_token(session_id)}")
        body = await response.json()

        assert response.status == 200
        assert body["products"] == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_ruble_products_replace_owned_card_cosmetic_set_with_gems():
    db = CheckoutFakeDB()
    db.product = {
        "code": "owned_card_style_pack",
        "item_type": "shop_set",
        "package_type": None,
        "shop_set_id": 12,
        "name": "Owned Card Style Pack",
        "price": 299.0,
        "currency": "rubles",
        "is_active": True,
        "show_in_game": True,
    }
    db.shop_sets[12] = {
        "id": 12,
        "name": "Owned Card Style Set",
        "description": "Card with style",
        "price": 299.0,
        "currency": "rubles",
        "is_active": True,
        "rewards": [
            {"type": "card", "card_id": 77},
            {"type": "cosmetic", "cosmetic_slug": "bg_gold"},
        ],
    }
    db.cards[77] = {
        "id": 77,
        "name": "Arena Captain",
        "description": "Leader card",
        "rarity": "epic",
        "mechanics": {},
        "mechanics_desc": "",
        "card_type": "warrior",
        "mana_cost": 4,
        "base_attack": 6,
        "base_hp": 7,
    }
    db.cosmetics["bg_gold"] = {
        "slug": "bg_gold",
        "item_type": "profile_background",
        "class": "gold",
        "name": "Gold Background",
        "asset_path": "/static/bg_gold.png",
        "media_type": "image",
    }
    db.owned_cards = [{"id": 77}]
    client, session_id = await _client(db=db)
    try:
        response = await client.get(f"/api/shop/ruble-products?surface=game&_auth={_auth_token(session_id)}")
        body = await response.json()

        assert response.status == 200
        product = body["products"][0]
        assert not any(reward["type"] == "card" for reward in product["rewards"])
        assert {"type": "gems", "amount": 50, "fallback_for": "owned_card", "card_ids": [77]} in product["rewards"]
        assert product["owned_card_fallback"]["amount"] == 50
        assert product["owned_card_fallback"]["cards"][0]["card_id"] == 77
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_hidden_gift_shop_set_cannot_be_claimed_by_code():
    db = CheckoutFakeDB()
    db.product = {
        "code": "hidden_gift",
        "item_type": "gift_shop_set",
        "package_type": None,
        "shop_set_id": 9,
        "name": "Hidden Gift",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
        "show_in_game": False,
        "show_in_shop": False,
    }
    db.shop_sets[9] = {
        "id": 9,
        "name": "Hidden Gift Set",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
        "rewards": [{"type": "gems", "amount": 25}],
    }
    client, session_id = await _client(db=db)
    try:
        response = await client.post(
            f"/api/shop/gifts/claim?_auth={_auth_token(session_id)}",
            json={"product_code": "hidden_gift"},
        )
        body = await response.json()

        assert response.status == 404
        assert body["error"] == "product_not_found"
        assert db.gift_claims == {}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_ruble_products_skip_products_linked_to_inactive_sets():
    db = CheckoutFakeDB()
    db.product = {
        "code": "inactive_set_product",
        "item_type": "gift_shop_set",
        "package_type": None,
        "shop_set_id": 9,
        "name": "Inactive Set Product",
        "price": 0.0,
        "currency": "rubles",
        "is_active": True,
        "show_in_shop": True,
    }
    db.shop_sets[9] = {
        "id": 9,
        "name": "Inactive Gift Set",
        "price": 0.0,
        "currency": "rubles",
        "is_active": False,
        "rewards": [{"type": "gems", "amount": 25}],
    }
    client, _session_id = await _client(db=db)
    try:
        response = await client.get("/api/shop/ruble-products?surface=shop")
        body = await response.json()

        assert response.status == 200
        assert body["products"] == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_parallel_direct_starter_payment_allows_one_provider_call():
    db = CheckoutFakeDB()
    db.product = {
        "code": "gems_starter_once",
        "item_type": "gems_package",
        "package_type": "starter_once",
        "name": "Starter gems",
        "price": 49.0,
        "currency": "rubles",
        "is_active": True,
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)

        responses = await asyncio.gather(
            client.post(f"/api/payments/create?_auth={token}", json={"product_code": "gems_starter_once"}),
            client.post(f"/api/payments/create?_auth={token}", json={"product_code": "gems_starter_once"}),
        )
        bodies = [await response.json() for response in responses]
        statuses = sorted(response.status for response in responses)

        assert statuses == [200, 409]
        assert len(payment_service.created) == 1
        assert len(db.created_payments) == 1
        assert {body.get("error") for body in bodies if body.get("error")} == {"starter_checkout_in_progress"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_parallel_stars_starter_invoice_reserves_and_attaches_once(monkeypatch):
    import aiogram

    FakeStarsBot.invoices = []
    FakeStarsBot.messages = []
    monkeypatch.setattr(aiogram, "Bot", FakeStarsBot)

    db = CheckoutFakeDB()
    db.product = {
        "code": "gems_starter_once",
        "item_type": "gems_package",
        "package_type": "starter_once",
        "name": "Starter gems",
        "price": 49.0,
        "currency": "rubles",
        "is_active": True,
    }
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id)

        responses = await asyncio.gather(
            client.post(f"/api/payments/stars/create?_auth={token}", json={"product_code": "gems_starter_once"}),
            client.post(f"/api/payments/stars/create?_auth={token}", json={"product_code": "gems_starter_once"}),
        )
        bodies = [await response.json() for response in responses]
        statuses = sorted(response.status for response in responses)

        assert statuses == [200, 409]
        assert len(FakeStarsBot.invoices) == 1
        assert len(db.created_payments) == 1
        payment_id = db.created_payments[0]["payment_id"]
        reservation = db.one_time_reservations[(1001, "gems_package:starter_once")]
        assert reservation["status"] == "payment_created"
        assert reservation["payment_id"] == payment_id
        assert db.created_payments[0]["metadata"]["one_time_reservation_id"] == reservation["reservation_id"]
        assert {body.get("error") for body in bodies if body.get("error")} == {"starter_checkout_in_progress"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_parallel_starter_checkout_start_creates_one_active_session():
    db = CheckoutFakeDB()
    db.product = {
        "code": "gems_starter_once",
        "item_type": "gems_package",
        "package_type": "starter_once",
        "name": "Starter gems",
        "price": 49.0,
        "currency": "rubles",
        "is_active": True,
    }
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id)

        responses = await asyncio.gather(
            client.post(f"/api/payments/checkout/start?_auth={token}", json={"product_code": "gems_starter_once"}),
            client.post(f"/api/payments/checkout/start?_auth={token}", json={"product_code": "gems_starter_once"}),
        )
        bodies = [await response.json() for response in responses]
        statuses = sorted(response.status for response in responses)

        assert statuses == [200, 409]
        assert len(db.checkout_sessions) == 1
        assert {body.get("error") for body in bodies if body.get("error")} == {"starter_checkout_in_progress"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_checkout_start_rejects_one_time_product_without_reserving_target():
    db = CheckoutFakeDB()
    db.product = {
        "code": "gems_starter_once",
        "item_type": "gems_package",
        "package_type": "starter_once",
        "name": "Starter gems",
        "price": 49.0,
        "currency": "rubles",
        "is_active": True,
    }
    client, _session_id = await _client(db=db)
    try:
        response = await client.post(
            "/api/payments/checkout/public/start",
            json={"telegram_id": 2002, "product_code": "gems_starter_once"},
        )
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "one_time_public_checkout_forbidden"
        assert db.one_time_reservations == {}
        assert db.checkout_sessions == {}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_authenticated_checkout_start_ignores_body_telegram_id_for_one_time_reservation():
    db = CheckoutFakeDB()
    db.product = {
        "code": "gems_starter_once",
        "item_type": "gems_package",
        "package_type": "starter_once",
        "name": "Starter gems",
        "price": 49.0,
        "currency": "rubles",
        "is_active": True,
    }
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id, user_id=1001)
        response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={
                "telegram_id": 2002,
                "recipient_id": 2002,
                "product_code": "gems_starter_once",
            },
        )

        assert response.status == 200
        assert (1001, "gems_package:starter_once") in db.one_time_reservations
        assert (2002, "gems_package:starter_once") not in db.one_time_reservations
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_parallel_checkout_create_for_same_jti_makes_one_provider_call():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()
        assert start_response.status == 200

        checkout_token = start_body["checkout_jti"]
        assert "token" not in start_body
        assert "?checkout=" not in start_body["checkout_url"]
        assert "#checkout_jti=" in start_body["checkout_url"]
        responses = await asyncio.gather(
            client.post("/api/payments/checkout/create", json={"checkout_jti": checkout_token}),
            client.post("/api/payments/checkout/create", json={"checkout_jti": checkout_token}),
        )
        bodies = [await response.json() for response in responses]

        assert [response.status for response in responses] == [200, 200]
        assert len(payment_service.created) == 1
        assert {body["payment_id"] for body in bodies} == {"pay-1"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_create_recovers_stale_payment_creating_session():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()
        assert start_response.status == 200

        checkout_token = start_body["checkout_jti"]
        db.checkout_sessions[checkout_token]["status"] = "payment_creating"
        db.checkout_sessions[checkout_token]["updated_at"] = time.time() - 61

        create_response = await client.post(
            "/api/payments/checkout/create",
            json={"checkout_jti": checkout_token},
        )
        body = await create_response.json()

        assert create_response.status == 200
        assert body["success"] is True
        assert body["payment_id"] == "pay-1"
        assert len(payment_service.created) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_create_with_jti_uses_server_side_session_not_client_body():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()
        assert start_response.status == 200

        create_response = await client.post(
            "/api/payments/checkout/create",
            json={
                "checkout_jti": start_body["checkout_jti"],
                "user_id": 9999,
                "amount_rub": 1,
                "item_type": "attacker_controlled",
                "metadata": {"item_name": "Wrong"},
            },
        )
        body = await create_response.json()

        assert create_response.status == 200
        assert body["success"] is True
        assert len(payment_service.created) == 1
        created = payment_service.created[0]
        assert created["amount"] == 179.0
        assert created["description"] == "ExtraPass"
        assert created["metadata"]["user_id"] == 1001
        assert created["metadata"]["item_type"] == "extrapass"
        assert created["metadata"]["item_name"] == "ExtraPass"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_summary_returns_safe_order_display_fields():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()

        response = await client.get(
            f"/api/payments/checkout/summary?jti={start_body['checkout_jti']}",
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert body["checkout_jti"] == start_body["checkout_jti"]
        assert body["item_type"] == "extrapass"
        assert body["item_name"] == "ExtraPass"
        assert body["amount_rub"] == 179.0
        assert body["currency"] == "RUB"
        assert "user_id" not in body
        assert "metadata" not in body
        assert "payment_id" not in body
        assert "provider" not in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_create_uses_robokassa_primary_provider_in_test_mode():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }
    yookassa_service = FakePaymentService()
    robokassa_service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="test-pass-1",
            password2="test-pass-2",
            hash_algo="sha256",
            test_mode=True,
            result_url="https://laveqox.ru/api/payments/robokassa/result",
            success_url="https://laveqox.ru/extraShop/payment-success",
            fail_url="https://laveqox.ru/extraShop/payment-fail",
        )
    )
    client, session_id = await _client(
        db=db,
        payment_service=yookassa_service,
        robokassa_payment_service=robokassa_service,
    )
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()
        assert start_response.status == 200

        checkout_token = start_body["checkout_jti"]
        assert "token" not in start_body
        assert "?checkout=" not in start_body["checkout_url"]
        assert "#checkout_jti=" in start_body["checkout_url"]
        create_response = await client.post(
            "/api/payments/checkout/create",
            json={"checkout_jti": checkout_token},
        )
        body = await create_response.json()

        assert create_response.status == 200
        assert body["success"] is True
        assert body["provider"] == "robokassa"
        assert body["payment_id"].startswith("robokassa_")
        assert body["confirmation_url"].startswith("https://auth.robokassa.ru/Merchant/Index.aspx?")
        assert "IsTest=1" in body["confirmation_url"]
        assert yookassa_service.created == []
        assert len(db.created_payments) == 1
        payment_record = db.created_payments[0]
        assert payment_record["payment_id"] == body["payment_id"]
        assert payment_record["metadata"]["provider"] == "robokassa"
        assert payment_record["metadata"]["robokassa_test_mode"] is True
        assert payment_record["metadata"]["robokassa_form"]["IsTest"] == "1"
        assert payment_record["metadata"]["robokassa_form"]["Receipt"]
        assert payment_record["metadata"]["robokassa_form"]["SuccessUrl2"] == "https://laveqox.ru/extraShop/payment-success"
        assert payment_record["metadata"]["robokassa_form"]["FailUrl2"] == "https://laveqox.ru/extraShop/payment-fail"
        assert "robokassa_payment_page_url" not in payment_record["metadata"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_create_falls_back_to_yookassa_when_robokassa_fails():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }

    class FailingRobokassaPaymentService:
        def __init__(self):
            self.created = []

        def create_payment(self, **kwargs):
            self.created.append(kwargs)
            return {"success": False, "error": "robokassa_unavailable"}

    yookassa_service = FakePaymentService()
    robokassa_service = FailingRobokassaPaymentService()
    client, session_id = await _client(
        db=db,
        payment_service=yookassa_service,
        robokassa_payment_service=robokassa_service,
    )
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()

        response = await client.post(
            "/api/payments/checkout/create",
            json={"checkout_jti": start_body["checkout_jti"]},
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert body["provider"] == "yookassa"
        assert body["payment_id"] == "pay-1"
        assert body["confirmation_url"] == "https://pay.example/confirm/1"
        assert len(robokassa_service.created) == 1
        assert len(yookassa_service.created) == 1
        assert len(db.created_payments) == 1
        assert db.created_payments[0]["metadata"]["provider"] == "yookassa"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_create_returns_existing_robokassa_confirmation_url():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }
    robokassa_service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="test-pass-1",
            password2="test-pass-2",
            hash_algo="sha256",
            test_mode=True,
            result_url="https://laveqox.ru/api/payments/robokassa/result",
        )
    )
    client, session_id = await _client(db=db, robokassa_payment_service=robokassa_service)
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()
        checkout_token = start_body["checkout_jti"]
        assert "token" not in start_body
        assert "?checkout=" not in start_body["checkout_url"]
        assert "#checkout_jti=" in start_body["checkout_url"]
        jti = next(iter(db.checkout_sessions))
        legacy_result = robokassa_service.create_payment(
            amount=179,
            currency="RUB",
            description="ExtraPass",
            return_url="https://game.example",
            metadata={"item_name": "ExtraPass", "item_type": "extrapass"},
            inv_id=123456,
        )
        legacy_internal_url = "https://laveqox.ru/api/payments/robokassa/pay/robokassa_123456"
        db.checkout_sessions[jti]["payment_id"] = legacy_result["payment_id"]
        db.checkout_sessions[jti]["confirmation_url"] = legacy_internal_url
        db.payment_records[legacy_result["payment_id"]] = {
            "payment_id": legacy_result["payment_id"],
            "amount": 179.0,
            "currency": "RUB",
            "status": "pending",
            "metadata": {
                "provider": "robokassa",
                "robokassa_form": legacy_result["form"],
                "robokassa_form_action": legacy_result["form_action"],
            },
        }

        response = await client.post("/api/payments/checkout/create", json={"checkout_jti": checkout_token})
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert body["provider"] == "robokassa"
        assert body["payment_id"] == "robokassa_123456"
        assert body["confirmation_url"] == legacy_internal_url
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_robokassa_payment_page_serves_post_form_without_passwords():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }
    robokassa_service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="test-pass-1",
            password2="test-pass-2",
            hash_algo="sha256",
            test_mode=True,
            result_url="https://laveqox.ru/api/payments/robokassa/result",
        )
    )
    client, session_id = await _client(db=db, robokassa_payment_service=robokassa_service)
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()
        checkout_token = start_body["checkout_jti"]
        assert "token" not in start_body
        assert "?checkout=" not in start_body["checkout_url"]
        assert "#checkout_jti=" in start_body["checkout_url"]
        create_response = await client.post("/api/payments/checkout/create", json={"checkout_jti": checkout_token})
        create_body = await create_response.json()

        page_path = f"/api/payments/robokassa/pay/{create_body['payment_id']}"
        page_response = await client.get(page_path)
        page = await page_response.text()

        assert page_response.status == 200
        assert 'method="POST"' in page
        assert "https://auth.robokassa.ru/Merchant/Index.aspx" in page
        assert 'name="IsTest" value="1"' in page
        assert 'name="Receipt"' in page
        assert "test-pass-1" not in page
        assert "test-pass-2" not in page
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_robokassa_success_return_processes_payment_when_result_webhook_was_missed():
    db = CheckoutFakeDB()
    payment_id = "robokassa_777"
    db.payment_records[payment_id] = {
        "payment_id": payment_id,
        "user_id": 1001,
        "amount": 1.0,
        "currency": "RUB",
        "status": "pending",
        "description": "Test payment",
        "metadata": {
            "provider": "robokassa",
            "item_type": "test_payment",
            "item_name": "Test payment",
            "gems_amount": 10,
        },
        "rewards_processed": False,
    }
    robokassa_service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="test-pass-1",
            password2="test-pass-2",
            hash_algo="sha256",
            test_mode=True,
        )
    )
    payload = {
        "OutSum": "1.00",
        "InvId": "777",
        "Shp_payment_id": payment_id,
    }
    payload["SignatureValue"] = hashlib.sha256(
        f"1.00:777:test-pass-1:Shp_payment_id={payment_id}".encode()
    ).hexdigest()
    client, _session_id = await _client(db=db, robokassa_payment_service=robokassa_service)
    try:
        response = await client.get("/extraShop/payment-success", params=payload)
        page = await response.text()

        assert response.status == 200
        assert "Оплата прошла" in page
        assert db.payment_records[payment_id]["status"] == "succeeded"
        assert db.payment_records[payment_id]["rewards_processed"] is True
        assert db.user_gems[1001] == 10
        assert db.created_mail
        assert "Источник: Robokassa" in db.created_mail[0]["content"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_session_status_reconciles_yookassa_payment_and_grants_rewards():
    db = CheckoutFakeDB()
    db.product = {
        "code": "extrapass",
        "item_type": "extrapass",
        "package_type": None,
        "name": "ExtraPass",
        "price": 179.0,
        "currency": "rubles",
        "is_active": True,
    }
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        start_response = await client.post(
            f"/api/payments/checkout/start?_auth={token}",
            json={"product_code": "extrapass"},
        )
        start_body = await start_response.json()
        assert start_response.status == 200

        checkout_token = start_body["checkout_jti"]
        assert "token" not in start_body
        assert "?checkout=" not in start_body["checkout_url"]
        assert "#checkout_jti=" in start_body["checkout_url"]
        create_response = await client.post(
            "/api/payments/checkout/create",
            json={"checkout_jti": checkout_token},
        )
        create_body = await create_response.json()
        assert create_response.status == 200

        payment_id = create_body["payment_id"]
        db.payment_records[payment_id]["metadata"].update({
            "item_type": "test_payment",
            "gems_amount": 10,
        })
        payment_service.statuses[payment_id] = {
            "success": True,
            "status": "succeeded",
            "paid": True,
            "amount": 179.0,
            "currency": "RUB",
        }

        response = await client.get(
            f"/api/payments/checkout/session-status?_auth={token}&jti={start_body['checkout_jti']}",
        )
        body = await response.json()

        assert response.status == 200
        assert body["payment_id"] == payment_id
        assert body["payment_status"] == "succeeded"
        assert body["rewards_processed"] is True
        assert db.payment_records[payment_id]["status"] == "succeeded"
        assert db.payment_records[payment_id]["rewards_processed"] is True
        assert db.user_gems[1001] == 10
        assert db.created_mail
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_checkout_session_status_processes_succeeded_robokassa_payment_rewards():
    db = CheckoutFakeDB()
    jti = "robokassa-checkout-jti"
    payment_id = "robokassa_4242"
    db.checkout_sessions[jti] = {
        "checkout_jti": jti,
        "user_id": 1001,
        "item_type": "test_payment",
        "amount": 1.0,
        "metadata": {"item_type": "test_payment"},
        "payment_id": payment_id,
        "confirmation_url": "https://pay.example/robokassa",
        "status": "payment_created",
        "expires_at": time.time() + 600,
    }
    db.payment_records[payment_id] = {
        "payment_id": payment_id,
        "user_id": 1001,
        "amount": 1.0,
        "currency": "RUB",
        "status": "succeeded",
        "description": "Test payment",
        "metadata": {
            "provider": "robokassa",
            "item_type": "test_payment",
            "item_name": "Test payment",
            "gems_amount": 10,
        },
        "rewards_processed": False,
    }
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id)
        response = await client.get(
            f"/api/payments/checkout/session-status?_auth={token}&jti={jti}",
        )
        body = await response.json()

        assert response.status == 200
        assert body["provider"] == "robokassa"
        assert body["payment_status"] == "succeeded"
        assert body["rewards_processed"] is True
        assert db.payment_records[payment_id]["rewards_processed"] is True
        assert db.user_gems[1001] == 10
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_robokassa_result_rejects_invalid_signature_before_status_update():
    db = CheckoutFakeDB()
    robokassa_service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="test-pass-1",
            password2="test-pass-2",
            hash_algo="sha256",
            test_mode=True,
        )
    )
    client, _session_id = await _client(db=db, robokassa_payment_service=robokassa_service)
    try:
        response = await client.post(
            "/api/payments/robokassa/result",
            data={
                "OutSum": "179.00",
                "InvId": "12345",
                "Shp_payment_id": "robokassa_12345",
                "SignatureValue": "bad-signature",
            },
        )
        text = await response.text()

        assert response.status == 400
        assert "invalid_signature" in text
        assert db.updated_statuses == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_robokassa_result_rejects_amount_mismatch_without_status_update():
    db = CheckoutFakeDB()
    db.payment_records["robokassa_12345"] = {
        "payment_id": "robokassa_12345",
        "user_id": 1001,
        "amount": 179.0,
        "currency": "RUB",
        "description": "ExtraPass",
        "metadata": {"provider": "robokassa", "item_type": "extrapass"},
        "status": "pending",
        "rewards_processed": False,
    }
    robokassa_service = RobokassaPaymentService(
        RobokassaSettings(
            merchant_login="extraarena",
            password1="test-pass-1",
            password2="test-pass-2",
            hash_algo="sha256",
            test_mode=True,
        )
    )
    signature = hashlib.sha256(
        "1.00:12345:test-pass-2:Shp_payment_id=robokassa_12345".encode()
    ).hexdigest()
    client, _session_id = await _client(db=db, robokassa_payment_service=robokassa_service)
    try:
        response = await client.post(
            "/api/payments/robokassa/result",
            data={
                "OutSum": "1.00",
                "InvId": "12345",
                "Shp_payment_id": "robokassa_12345",
                "SignatureValue": signature,
            },
        )
        body = await response.json()

        assert response.status == 202
        assert body["reason"] == "amount_mismatch"
        assert db.updated_statuses == []
    finally:
        await client.close()
