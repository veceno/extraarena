import json
import time
import uuid

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.config import DECK_SIZE, get_settings
from infrastructure.shop_config import GEM_PACKAGES
from web import server as web_server


class FakeExtraIDDB:
    def __init__(self, session_id: str, user_id: int = 1001):
        self.session_id = session_id
        self.user_id = user_id

    async def verify_session(self, session_uuid, token: str):
        if str(session_uuid) != self.session_id:
            return None
        return {"session_id": session_uuid, "user_id": self.user_id}


class SecurityFakeDB:
    def __init__(self):
        self.created_payments = []
        self.updated_statuses = []
        self.created_from_webhook = []
        self.payment_records = {}
        self.profile_trophies = 500
        self.runtime_config_updates = []

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},
            "disabled_card_ids": [],
        }

    async def set_runtime_config(self, **kwargs):
        self.runtime_config_updates.append(kwargs)
        return await self.get_runtime_config()

    async def is_admin(self, user_id):
        return False

    async def get_match_mode_overrides(self):
        return []

    async def is_match_mode_enabled(self, mode_id):
        return True

    async def get_disabled_card_ids(self):
        return []

    async def get_user_profile(self, user_id):
        return {"user_id": user_id, "trophies": self.profile_trophies}

    async def get_user_deck_presets(self, user_id):
        card_ids = list(range(1, DECK_SIZE + 1))
        preset = {
            "preset_number": 1,
            "card_ids": card_ids,
            "used_by_bot": False,
        }
        for idx, card_id in enumerate(card_ids, start=1):
            preset[f"card_slot_{idx}"] = card_id
        return [preset]

    async def get_player_deck_max_level(self, user_id, selected_deck_id=None):
        return 1

    async def fetchval(self, query, *args):
        if "primary_deck" in query:
            return 1
        return None

    async def get_ruble_products(self, active_only=True, surface=None):
        return []

    async def get_user_settings(self, user_id):
        return {}

    async def create_payment(self, **kwargs):
        self.created_payments.append(kwargs)
        return {"success": True}

    async def get_payment_by_id(self, payment_id):
        return self.payment_records.get(payment_id)

    async def update_payment_status(self, *, payment_id, status):
        self.updated_statuses.append((payment_id, status))


class FailingCreatePaymentDB(SecurityFakeDB):
    async def create_payment(self, **kwargs):
        self.created_payments.append(kwargs)
        return {"success": False, "error": "db unavailable"}


class FailingRuntimeConfigDB(SecurityFakeDB):
    async def get_runtime_config(self):
        raise RuntimeError("secret_table_name leaked from database path")


class FakePaymentService:
    def __init__(self):
        self.created = []

    def create_payment(self, **kwargs):
        self.created.append(kwargs)
        return {
            "success": True,
            "payment_id": "pay-secure",
            "confirmation_url": "https://pay.example/confirm",
            "status": "pending",
        }

    def parse_webhook(self, body):
        obj = body.get("object") or {}
        return {
            "event": body.get("event"),
            "payment_id": obj.get("id"),
            "status": obj.get("status"),
            "paid": obj.get("paid", False),
            "amount": float((obj.get("amount") or {}).get("value") or 0),
            "currency": (obj.get("amount") or {}).get("currency", "RUB"),
            "metadata": obj.get("metadata") or {},
        }

    def get_payment_status(self, payment_id):
        return {
            "success": True,
            "payment_id": payment_id,
            "status": "succeeded",
            "paid": True,
            "amount": "99.00",
            "currency": "RUB",
        }


class MismatchedPaymentService(FakePaymentService):
    def get_payment_status(self, payment_id):
        return {
            "success": True,
            "payment_id": payment_id,
            "status": "succeeded",
            "paid": True,
            "amount": "1.00",
            "currency": "RUB",
        }


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


async def _client(db=None, payment_service=None):
    session_id = str(uuid.uuid4())
    app = web_server.create_web_app(
        db or SecurityFakeDB(),
        bot_token="bot-token",
        extraid_db=FakeExtraIDDB(session_id),
        payment_service=payment_service or FakePaymentService(),
        webapp_url="https://game.example",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, session_id


@pytest.mark.asyncio
async def test_admin_api_rejects_dev_user_id_fallback(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get(f"/api/admin/runtime-config?user_id={web_server.ADMIN_ID}")
        body = await response.json()

        assert response.status == 401
        assert body["error"] == "authentication_required"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_api_accepts_authorization_header_without_cors_wildcard(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=SecurityFakeDB())
    try:
        token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        response = await client.get(
            "/api/admin/runtime-config",
            headers={"Authorization": f"Bearer {token}"},
        )
        body = await response.json()

        assert response.status == 200
        assert body["status"] == "ok"
        assert "Access-Control-Allow-Origin" not in response.headers
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_html_requires_admin_auth_and_sets_security_headers(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=SecurityFakeDB())
    try:
        anonymous = await client.get("/extraShop/admin")
        anonymous_body = await anonymous.json()
        assert anonymous.status == 401
        assert anonymous_body["error"] == "authentication_required"

        token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        authorized = await client.get(f"/extraShop/admin?_auth={token}")
        assert authorized.status == 200
        assert authorized.headers["Referrer-Policy"] == "no-referrer"
        assert authorized.headers["X-Content-Type-Options"] == "nosniff"
        assert "default-src 'self'" in authorized.headers["Content-Security-Policy"]
        assert web_server.ADMIN_SESSION_COOKIE_NAME in authorized.cookies

        cookie = authorized.cookies[web_server.ADMIN_SESSION_COOKIE_NAME].value
        reloaded = await client.get(
            "/extraShop/admin",
            cookies={web_server.ADMIN_SESSION_COOKIE_NAME: cookie},
        )
        assert reloaded.status == 200
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_500_responses_do_not_expose_internal_exception_text(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=FailingRuntimeConfigDB())
    try:
        token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        response = await client.get(
            "/api/admin/runtime-config",
            headers={"Authorization": f"Bearer {token}"},
        )
        body = await response.json()

        assert response.status == 500
        assert body == {
            "error": "internal_server_error",
            "message": "Internal server error",
        }
        assert "secret_table_name" not in json.dumps(body)
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_match_find_uses_db_trophies_not_client_payload():
    db = SecurityFakeDB()
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            "/api/match/find",
            json={
                "_auth": token,
                "trophies": 1,
                "selected_deck_id": 1,
                "game_mode": "classic",
            },
        )
        body = await response.json()

        assert response.status == 200
        assert body["status"] == "waiting"
        assert client.server.app["matchmaker"]._queue[0].trophies == db.profile_trophies
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_create_ignores_client_amount_and_reward_metadata():
    db = SecurityFakeDB()
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={
                "amount": 1,
                "metadata": {
                    "item_type": "gems_package",
                    "package_type": "gems_100",
                    "package_gems": 999999,
                },
            },
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert payment_service.created[0]["amount"] == float(GEM_PACKAGES["gems_100"]["price"])
        assert payment_service.created[0]["currency"] == "RUB"
        assert payment_service.created[0]["metadata"]["package_gems"] == GEM_PACKAGES["gems_100"]["gems"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_create_fails_when_payment_record_cannot_be_persisted():
    db = FailingCreatePaymentDB()
    payment_service = FakePaymentService()
    client, session_id = await _client(db=db, payment_service=payment_service)
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/create?_auth={token}",
            json={
                "metadata": {
                    "item_type": "gems_package",
                    "package_type": "gems_100",
                },
            },
        )
        body = await response.json()

        assert response.status == 500
        assert body["success"] is False
        assert body["error"] == "payment_record_not_saved"
        assert payment_service.created
        assert db.created_payments
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_yookassa_webhook_does_not_create_unknown_payments_from_metadata():
    db = SecurityFakeDB()
    client, _session_id = await _client(db=db, payment_service=FakePaymentService())
    try:
        response = await client.post(
            "/api/payments/webhook",
            json={
                "event": "payment.succeeded",
                "object": {
                    "id": "attacker-payment",
                    "status": "succeeded",
                    "paid": True,
                    "amount": {"value": "1.00", "currency": "RUB"},
                    "metadata": {
                        "user_id": 1001,
                        "item_type": "gems",
                        "gems_amount": 999999,
                    },
                },
            },
        )
        body = await response.json()

        assert response.status == 202
        assert body["reason"] == "unknown_payment"
        assert db.created_payments == []
        assert db.updated_statuses == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_payment_status_does_not_grant_rewards_on_provider_amount_mismatch():
    db = SecurityFakeDB()
    db.payment_records["pay-mismatch"] = {
        "payment_id": "pay-mismatch",
        "user_id": 1001,
        "amount": 99.0,
        "currency": "RUB",
        "description": "100 гемов",
        "metadata": {"item_type": "gems", "gems_amount": 100},
        "status": "pending",
        "rewards_processed": False,
    }
    client, session_id = await _client(db=db, payment_service=MismatchedPaymentService())
    try:
        token = _auth_token(session_id)
        response = await client.get(f"/api/payments/status?payment_id=pay-mismatch&_auth={token}")
        body = await response.json()

        assert response.status == 200
        assert body["verification_failed"] is True
        assert body["reason"] == "amount_mismatch"
        assert body["rewards_processed"] is False
        assert db.updated_statuses == []
    finally:
        await client.close()
