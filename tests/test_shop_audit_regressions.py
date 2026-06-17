import json
import time
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
    def __init__(self, *, admin_ids=(), fail_profile=False, allow_max_level_particles=False):
        self.admin_ids = {int(user_id) for user_id in admin_ids}
        self.fail_profile = fail_profile
        self.allow_max_level_particles = allow_max_level_particles
        self.fetchrow_calls = []
        self.execute_calls = []
        self.settings_updates = []
        self.case_grants = []
        self.economy_events = []

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
        return None

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
