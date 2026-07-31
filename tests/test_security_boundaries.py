import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from infrastructure import card_assets
from infrastructure.config import DECK_SIZE, get_settings
from infrastructure.shop_config import GEM_PACKAGES
from web import server as web_server


STRONG_TEST_JWT_SECRET = "test-security-jwt-secret-that-is-long-enough-2026"


def test_telegram_channel_auth_checks_the_verified_user_id(monkeypatch):
    request = SimpleNamespace(app={"bot_token": "bot-token"})
    verified = {"auth_date": str(int(time.time()))}
    monkeypatch.setattr(web_server, "_request_auth_token", lambda _request: "signed-init-data")
    monkeypatch.setattr(web_server, "_verify_init_data", lambda _token, _secret: verified)
    monkeypatch.setattr(web_server, "_validate_auth_date", lambda _data: True)
    monkeypatch.setattr(web_server, "_extract_user_id_from_init_data", lambda _data: 1001)

    assert web_server._telegram_init_data_for_request(
        request,
        expected_user_id=1001,
    ) == verified
    assert web_server._telegram_init_data_for_request(
        request,
        expected_user_id=1002,
    ) is None


@pytest.fixture(autouse=True)
def _strong_jwt_secret(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-admin-session-secret-that-is-long-enough-2026")
    monkeypatch.setenv("MCP_TOKEN_SECRET", "test-security-mcp-secret-that-is-long-enough-2026")
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://test.local")
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


class SecurityFakeDB:
    def __init__(self):
        self.created_payments = []
        self.updated_statuses = []
        self.created_from_webhook = []
        self.payment_records = {}
        self.profile_trophies = 500
        self.runtime_config_updates = []
        self.particle_grants = []
        self.keys_by_user = {}
        self.settings_updates = []
        self.settings = {
            "notif_cases": True,
            "notif_daily_rewards": True,
            "notif_game_invites": True,
            "notif_friend_requests": True,
            "notif_events": True,
            "notif_news": True,
            "notif_generator": True,
            "notif_shop": False,
            "notif_reminders": True,
            "notif_squad_member_role": True,
            "notif_squad_new_member": True,
            "notif_squad_disbanded": True,
            "notif_squad_boost": True,
            "notif_extra_arena_modifiers": True,
            "notification_delivery_mode": "app_then_telegram",
            "ads_enabled": True,
            "sound_music": True,
            "sound_sfx": True,
            "social_block_friend_requests": False,
            "social_block_friendly_invites_from_friends": False,
            "social_block_friendly_invites_from_non_friends": True,
            "social_disable_talkies": False,
            "nickname_glow_disabled": False,
            "hide_player_id_public": False,
            "wins_since_last_case": 0,
        }

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
        return int(user_id) == web_server.ADMIN_ID

    async def execute(self, query, *args):
        if "UPDATE users SET keys = COALESCE(keys,0)+1" in query:
            user_id = int(args[0])
            self.keys_by_user[user_id] = self.keys_by_user.get(user_id, 0) + 1
            return "UPDATE 1"
        return "UPDATE 0"

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

    async def get_player_deck_avg_level(self, user_id, selected_deck_id=None):
        return 1.0

    async def fetchrow(self, query, *args):
        return None

    async def fetchval(self, query, *args):
        if "SELECT COALESCE(keys,0) FROM users" in query:
            return self.keys_by_user.get(int(args[0]), 0)
        if "primary_deck" in query:
            return 1
        return None

    async def get_ruble_products(self, active_only=True, surface=None):
        return []

    async def get_user_settings(self, user_id):
        return dict(self.settings)

    async def update_user_settings(self, user_id, **kwargs):
        self.settings_updates.append((user_id, kwargs))
        self.settings.update(kwargs)

    async def add_particles_to_card(self, user_id, card_id, particles):
        self.particle_grants.append((int(user_id), int(card_id), int(particles)))
        return {"success": True, "particles": int(particles)}

    async def get_onboarding_state(self, user_id):
        return {"status": "completed", "completed": True}

    async def process_weekly_squad_cbrp(self):
        return {"processed": False}

    async def refresh_due_rating_snapshots(self, scope="players"):
        return {"refreshed": []}

    async def expire_announcements(self):
        return 0

    async def create_payment(self, **kwargs):
        self.created_payments.append(kwargs)
        return {"success": True}

    async def get_payment_by_id(self, payment_id):
        return self.payment_records.get(payment_id)

    async def update_payment_status(self, *, payment_id, status):
        self.updated_statuses.append((payment_id, status))

    async def admin_approve_idea(self, post_id):
        return {"success": True, "post_id": post_id}

    async def update_idea_status(self, post_id, status):
        return {"success": True, "post_id": post_id, "status": status}

    async def delete_idea(self, post_id):
        if post_id == 999:
            return {"success": False, "error": "post_not_found"}
        return {"success": True, "post_id": post_id}

    async def get_bugs_for_admin(self, limit=50, offset=0):
        return [{"id": 1, "title": "Bug"}]


class CommunityPaginationDB(SecurityFakeDB):
    def __init__(self):
        super().__init__()
        self.community_calls = {}

    async def get_community_posts(self, **kwargs):
        self.community_calls["posts"] = kwargs
        return []

    async def get_news_posts(self, **kwargs):
        self.community_calls["news"] = kwargs
        return []

    async def get_ideas(self, **kwargs):
        self.community_calls["ideas"] = kwargs
        return []

    async def get_announcements(self, **kwargs):
        self.community_calls["announcements"] = kwargs
        return []

    async def get_bugs_for_admin(self, **kwargs):
        self.community_calls["bugs"] = kwargs
        return []


class FailingCreatePaymentDB(SecurityFakeDB):
    async def create_payment(self, **kwargs):
        self.created_payments.append(kwargs)
        return {"success": False, "error": "db unavailable"}


class FailingRuntimeConfigDB(SecurityFakeDB):
    async def get_runtime_config(self):
        raise RuntimeError("secret_table_name leaked from database path")


class FailingRuStorePaymentDB(SecurityFakeDB):
    async def get_payment_by_id(self, payment_id):
        raise RuntimeError("secret rustore payment table leaked")


class FailingCommunityAdminDB(SecurityFakeDB):
    async def admin_approve_idea(self, post_id):
        raise RuntimeError("secret_community_table leaked")


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


class FakeRuStoreVerifier:
    def __init__(self):
        self.calls = []

    def verify_invoice(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "success": True,
            "status": "succeeded",
            "paid": True,
            "invoice_id": kwargs["invoice_id"],
            "purchase_id": "purchase-ok",
            "invoice_status": "CONFIRMED",
            "sandbox": True,
        }


class RuStoreRetryDB(SecurityFakeDB):
    def __init__(self):
        super().__init__()
        self.rustore_payment_service = FakeRuStoreVerifier()
        self.gems_added = 0
        self.marked_processed = False
        self.payment_records["rustore_retry"] = {
            "payment_id": "rustore_retry",
            "user_id": 1001,
            "amount": 99.0,
            "currency": "RUB",
            "description": "100 gems",
            "metadata": {
                "provider": "rustore",
                "rustore_invoice_id": "inv-1",
                "rustore_product_id": "gems_100",
                "item_type": "gems",
                "gems_amount": 100,
            },
            "status": "succeeded",
            "rewards_processed": False,
        }

    async def claim_payment_for_processing(self, payment_id):
        record = self.payment_records.get(payment_id)
        if not record or record.get("rewards_processed"):
            return None
        return dict(record)

    async def release_payment_processing_claim(self, payment_id):
        return None

    async def execute(self, query, *args):
        if "UPDATE users SET gems" in query and "+ $1" in query:
            self.gems_added += int(args[0])
            return "UPDATE 1"
        if "SET metadata = metadata || $2::jsonb" in query:
            patch = json.loads(args[1])
            self.payment_records[args[0]]["metadata"].update(patch)
            return "UPDATE 1"
        return await super().execute(query, *args)

    async def fetchval(self, query, *args):
        if "SET rewards_processed = TRUE" in query:
            self.payment_records[args[0]]["rewards_processed"] = True
            self.marked_processed = True
            return 1
        return await super().fetchval(query, *args)

    async def update_payment_status(self, *, payment_id, status):
        await super().update_payment_status(payment_id=payment_id, status=status)
        self.payment_records[payment_id]["status"] = status

    async def create_mail(self, **kwargs):
        return {"success": True}

    async def track_economy_event(self, **kwargs):
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


async def _client(db=None, payment_service=None, *, user_id=1001):
    session_id = str(uuid.uuid4())
    app = web_server.create_web_app(
        db or SecurityFakeDB(),
        bot_token="bot-token",
        extraid_db=FakeExtraIDDB(session_id, user_id=user_id),
        payment_service=payment_service or FakePaymentService(),
        rustore_payment_service=getattr(db, "rustore_payment_service", None) if db is not None else None,
        webapp_url="https://game.example",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, session_id


@pytest.mark.asyncio
async def test_card_particle_grant_rejects_regular_user():
    db = SecurityFakeDB()
    client, session_id = await _client(db=db)
    token = _auth_token(session_id)
    try:
        response = await client.post(
            "/api/cards/add-particles",
            headers={"Authorization": f"Bearer {token}"},
            json={"card_id": 77, "particles": 999999},
        )
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "admin_access_required"
        assert db.particle_grants == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_card_particle_grant_allows_admin():
    db = SecurityFakeDB()
    client, session_id = await _client(db=db, user_id=web_server.ADMIN_ID)
    token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
    try:
        response = await client.post(
            "/api/cards/add-particles",
            headers={"Authorization": f"Bearer {token}"},
            json={"card_id": 77, "particles": 15},
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert db.particle_grants == [(web_server.ADMIN_ID, 77, 15)]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_settings_api_rejects_internal_game_state_fields(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = SecurityFakeDB()
    client, session_id = await _client(db=db)
    token = _auth_token(session_id)
    try:
        response = await client.post(
            "/api/settings",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "sound_music": False,
                "nickname_glow_disabled": True,
                "wins_since_last_case": 99,
                "starter_pack_used": True,
                "particles_rotation_cards": [1, 2, 3],
                "particles_purchased_today": [],
            },
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "invalid_settings_fields"
        assert not db.settings_updates
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_wave1b_env(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("PAYMENT_WEBHOOK_DIAGNOSTICS_ENABLED", raising=False)


def test_production_rejects_missing_or_default_jwt_secret(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-admin-session-secret-that-is-long-enough-2026")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="ENVIRONMENT|JWT_SECRET"):
        get_settings()

    monkeypatch.setenv("JWT_SECRET", "dev_secret_change_in_production!")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="ENVIRONMENT|JWT_SECRET"):
        get_settings()


def test_non_local_bind_requires_explicit_environment(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("WEBAPP_HOST", "0.0.0.0")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-admin-session-secret-that-is-long-enough-2026")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="ENVIRONMENT"):
        get_settings()


def test_production_requires_separate_admin_session_secret(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.delenv("ADMIN_SESSION_SECRET", raising=False)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="ADMIN_SESSION_SECRET"):
        get_settings()

    monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_TEST_JWT_SECRET)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="ADMIN_SESSION_SECRET"):
        get_settings()


def test_public_bind_rejects_default_development_secrets(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "0.0.0.0")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_SESSION_SECRET", raising=False)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="ENVIRONMENT|JWT_SECRET"):
        get_settings()


def test_memory_match_state_rejects_multi_worker_beta_config(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-admin-session-secret-that-is-long-enough-2026")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    monkeypatch.setenv("MATCH_STATE_BACKEND", "memory")
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="MATCH_STATE_BACKEND=memory|WEB_CONCURRENCY"):
        get_settings()


@pytest.mark.parametrize("worker_env", ["GUNICORN_WORKERS", "UVICORN_WORKERS"])
def test_memory_match_state_rejects_alternate_multi_worker_envs(monkeypatch, worker_env):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-admin-session-secret-that-is-long-enough-2026")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    monkeypatch.setenv("MATCH_STATE_BACKEND", "memory")
    monkeypatch.setenv(worker_env, "2")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="MATCH_STATE_BACKEND=memory|WEB_CONCURRENCY"):
        get_settings()


def test_web_app_creation_does_not_bypass_public_bind_settings_guard(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "0.0.0.0")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-admin-session-secret-that-is-long-enough-2026")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="ENVIRONMENT"):
        web_server.create_web_app(
            SecurityFakeDB(),
            bot_token="bot-token",
            extraid_db=FakeExtraIDDB(str(uuid.uuid4())),
            payment_service=FakePaymentService(),
            webapp_url="https://game.example",
        )


def test_telegram_insecure_tls_flag_is_local_development_only(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-admin-session-secret-that-is-long-enough-2026")
    monkeypatch.setenv("TELEGRAM_API_INSECURE_SSL", "true")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="TELEGRAM_API_INSECURE_SSL"):
        get_settings()


def test_development_allows_default_jwt_secret(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_SESSION_SECRET", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.jwt_secret == "dev_secret_change_in_production!"


@pytest.mark.asyncio
async def test_production_redirects_trusted_proxy_http_to_configured_https_origin(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.extraarena.space")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get(
            "/health?probe=extraid",
            headers={
                "CF-Visitor": '{"scheme":"http"}',
                "Host": "attacker.example",
                "X-Forwarded-Host": "also-attacker.example",
            },
            allow_redirects=False,
        )

        assert response.status == 308
        assert response.headers["Location"] == (
            "https://app.extraarena.space/health?probe=extraid"
        )
        assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate"
        assert response.headers["Strict-Transport-Security"] == "max-age=31536000"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_production_post_redirect_is_308_and_does_not_call_handler(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.extraarena.space")
    get_settings.cache_clear()
    request = SimpleNamespace(
        secure=False,
        remote="127.0.0.1",
        headers={"X-Forwarded-Proto": "http"},
        raw_path="/api/extraid/login?return=%2Farena",
        rel_url="/api/extraid/login?return=/arena",
    )
    handler_calls = 0

    async def handler(_request):
        nonlocal handler_calls
        handler_calls += 1
        return web.Response(text="should not run")

    response = await web_server.enforce_https_middleware(request, handler)

    assert response.status == 308
    assert response.headers["Location"] == (
        "https://app.extraarena.space/api/extraid/login?return=%2Farena"
    )
    assert handler_calls == 0


@pytest.mark.asyncio
async def test_production_ignores_spoofed_forwarded_scheme_from_untrusted_client(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.extraarena.space")
    get_settings.cache_clear()
    request = SimpleNamespace(
        remote="203.0.113.9",
        headers={
            "CF-Visitor": '{"scheme":"http"}',
            "X-Forwarded-Proto": "http",
        },
        rel_url="/health",
    )
    handler_calls = 0

    async def handler(_request):
        nonlocal handler_calls
        handler_calls += 1
        return web.Response(text="ok")

    response = await web_server.enforce_https_middleware(request, handler)

    assert response.status == 200
    assert handler_calls == 1


@pytest.mark.parametrize(
    "invalid_origin",
    [
        "",
        "http://app.extraarena.space",
        "https://user@app.extraarena.space",
        "https://app.extraarena.space/path",
        "https://app.extraarena.space?query=1",
        "https://app.extraarena.space#fragment",
        "https://app.extraarena.space\\@attacker.example",
        "https://app.extraarena.space:bad-port",
    ],
)
def test_https_redirect_origin_is_strict_and_does_not_fallback(monkeypatch, invalid_origin):
    monkeypatch.setenv("PUBLIC_BASE_URL", invalid_origin)
    monkeypatch.setenv("WEBAPP_URL", "https://fallback.example")

    assert web_server._configured_https_origin() is None


@pytest.mark.asyncio
async def test_production_http_fails_closed_without_valid_redirect_origin(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://app.extraarena.space")
    get_settings.cache_clear()
    request = SimpleNamespace(
        secure=False,
        remote="127.0.0.1",
        headers={"CF-Visitor": '{"scheme":"http"}'},
        raw_path="/health",
        rel_url="/health",
    )

    response = await web_server.enforce_https_middleware(
        request,
        lambda _request: pytest.fail("handler must not run"),
    )

    assert response.status == 503
    assert "Location" not in response.headers
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate"


@pytest.mark.parametrize(
    ("headers", "secure", "expected"),
    [
        ({"X-Forwarded-Proto": "http,https"}, False, None),
        ({"X-Forwarded-Proto": "https, http"}, False, None),
        ({"X-Forwarded-Proto": "httpx"}, False, None),
        ({"CF-Visitor": "[]", "X-Forwarded-Proto": "https, http"}, False, None),
        (
            {
                "CF-Visitor": '{"scheme":"https"}',
                "X-Forwarded-Proto": "http",
            },
            False,
            "https",
        ),
        (
            {
                "CF-Visitor": '{"scheme":"http"}',
                "X-Forwarded-Proto": "https",
            },
            False,
            "http",
        ),
        ({"CF-Visitor": '{"scheme":"http"}'}, True, "https"),
        ({}, False, None),
    ],
)
def test_trusted_forwarded_scheme_is_unambiguous(headers, secure, expected):
    request = SimpleNamespace(
        secure=secure,
        remote="127.0.0.1",
        headers=headers,
    )

    assert web_server._trusted_forwarded_scheme(request) == expected


@pytest.mark.asyncio
async def test_production_https_response_includes_hsts(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get(
            "/health",
            headers={"CF-Visitor": '{"scheme":"https"}'},
        )

        assert response.status == 200
        assert response.headers["Strict-Transport-Security"] == "max-age=31536000"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_production_hsts_is_present_on_unknown_route(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get(
            "/route-that-does-not-exist",
            headers={"CF-Visitor": '{"scheme":"https"}'},
        )

        assert response.status == 404
        assert response.headers["Strict-Transport-Security"] == "max-age=31536000"
    finally:
        await client.close()
        get_settings.cache_clear()


def test_admin_session_cookie_does_not_trust_direct_forwarded_proto(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    request = SimpleNamespace(
        secure=False,
        remote="203.0.113.9",
        headers={"X-Forwarded-Proto": "https"},
    )
    response = web.Response()

    web_server._set_admin_session_cookie(request, response, web_server.ADMIN_ID)

    assert response.cookies[web_server.ADMIN_SESSION_COOKIE_NAME]["secure"] is False


def test_admin_session_cookie_uses_admin_session_secret(monkeypatch):
    admin_secret = "test-admin-session-cookie-secret-that-is-long-enough-2026"
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", admin_secret)
    get_settings.cache_clear()

    token = web_server._make_admin_session_token(web_server.ADMIN_ID)

    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(token, STRONG_TEST_JWT_SECRET, algorithms=["HS256"])
    payload = jwt.decode(token, admin_secret, algorithms=["HS256"])
    assert payload["typ"] == "admin_session"
    assert payload["user_id"] == web_server.ADMIN_ID


@pytest.mark.asyncio
async def test_readiness_requires_payments_when_configured(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    get_settings.cache_clear()
    session_id = str(uuid.uuid4())
    app = web_server.create_web_app(
        SecurityFakeDB(),
        bot_token="bot-token",
        extraid_db=FakeExtraIDDB(session_id),
        payment_service=FakePaymentService(),
        payment_primary_provider="robokassa",
        payments_required=True,
        webapp_url="https://game.example",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/ready")
        body = await response.json()

        assert response.status == 503
        assert body["status"] == "degraded"
        assert body["components"]["payments"]["required"] is True
        assert body["components"]["payments"]["status"] == "not_configured"
        assert body["components"]["payments"]["primary_provider"] == "robokassa"
    finally:
        await client.close()
        get_settings.cache_clear()


def test_telegram_init_data_hash_uses_constant_time_compare():
    source = (Path(__file__).resolve().parents[1] / "web" / "server.py").read_text(encoding="utf-8")
    init_data_block = source.split("def _verify_init_data", 1)[1].split("def _extract_user_id_from_init_data", 1)[0]

    assert "hmac.compare_digest" in init_data_block
    assert "calculated_hash != received_hash" not in init_data_block


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
async def test_public_debug_add_key_route_is_not_available_to_regular_users(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = SecurityFakeDB()
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id, user_id=1001)
        response = await client.post(
            "/api/debug/add-key",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )

        assert response.status in {403, 404, 405}
        assert db.keys_by_user.get(1001, 0) == 0
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_debug_add_key_requires_admin_and_increments_only_for_admin(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = SecurityFakeDB()
    client, session_id = await _client(db=db)
    try:
        user_token = _auth_token(session_id, user_id=1001)
        denied = await client.post(
            "/api/admin/debug/add-key",
            headers={"Authorization": f"Bearer {user_token}"},
            json={},
        )
        assert denied.status == 403
        assert db.keys_by_user.get(1001, 0) == 0

        admin_token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        allowed = await client.post(
            "/api/admin/debug/add-key",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={},
        )
        body = await allowed.json()

        assert allowed.status == 200
        assert body["success"] is True
        assert body["keys"] == 1
        assert db.keys_by_user[web_server.ADMIN_ID] == 1
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
async def test_production_api_cors_echoes_allowed_origin_without_wildcard(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get(
            "/api/payments/config",
            headers={"Origin": "https://game.example"},
        )

        assert response.status == 200
        assert response.headers["Access-Control-Allow-Origin"] == "https://game.example"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_production_api_cors_rejects_unlisted_origin(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get(
            "/api/payments/config",
            headers={"Origin": "https://evil.example"},
        )

        assert response.status == 200
        assert "Access-Control-Allow-Origin" not in response.headers
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_card_image_response_does_not_set_wildcard_cors(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get(
            "/api/cards/image?card_id=8",
            headers={"Origin": "https://evil.example"},
        )

        assert response.status == 200
        assert "Access-Control-Allow-Origin" not in response.headers
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_card_image_preview_variant_serves_webp_preview(monkeypatch, tmp_path):
    cards_dir = tmp_path / "Cards"
    previews_dir = tmp_path / "CardsPreview" / "w384"
    cards_dir.mkdir()
    previews_dir.mkdir(parents=True)
    (cards_dir / "8.png").write_bytes(b"original-image")
    (previews_dir / "8.webp").write_bytes(b"preview-image")
    monkeypatch.setattr(card_assets, "CARD_ASSETS_DIR", cards_dir)
    monkeypatch.setattr(card_assets, "CARD_PREVIEW_ASSETS_DIR", previews_dir)
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get("/api/cards/image?card_id=8&variant=preview")

        assert response.status == 200
        assert response.headers["Content-Type"].startswith("image/webp")
        assert response.headers["Cache-Control"] == "public, max-age=3600, must-revalidate"
        assert await response.read() == b"preview-image"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_card_image_preview_variant_falls_back_to_original(monkeypatch, tmp_path):
    cards_dir = tmp_path / "Cards"
    previews_dir = tmp_path / "CardsPreview" / "w384"
    cards_dir.mkdir()
    previews_dir.mkdir(parents=True)
    (cards_dir / "8.png").write_bytes(b"original-image")
    monkeypatch.setattr(card_assets, "CARD_ASSETS_DIR", cards_dir)
    monkeypatch.setattr(card_assets, "CARD_PREVIEW_ASSETS_DIR", previews_dir)
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get("/api/cards/image?card_id=8&variant=preview")

        assert response.status == 200
        assert response.headers["Content-Type"].startswith("image/png")
        assert response.headers["Cache-Control"] == "public, max-age=86400"
        assert await response.read() == b"original-image"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_maintenance_allows_card_images_but_keeps_collection_closed():
    db = SecurityFakeDB()

    async def maintenance_config():
        return {
            "maintenance_mode": {"enabled": True},
            "feature_availability": {},
            "disabled_card_ids": [],
        }

    db.get_runtime_config = maintenance_config
    client, _session_id = await _client(db=db)
    try:
        image_response = await client.get("/api/cards/image?card_id=8&variant=preview")
        collection_response = await client.get("/api/cards/collection")

        assert image_response.status == 200
        assert image_response.headers["Content-Type"].startswith("image/webp")
        assert collection_response.status == 503
        assert (await collection_response.json())["error"] == "maintenance_mode"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_production_rejects_all_query_auth_but_accepts_authorization_header(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    get_settings.cache_clear()

    client, session_id = await _client()
    try:
        token = _auth_token(session_id, user_id=1001)

        query_response = await client.get(f"/api/profile?_auth={token}")
        query_body = await query_response.json()

        assert query_response.status == 401
        assert query_body["error"] == "query_auth_not_allowed"

        telegram_query_response = await client.get(
            "/api/profile?_auth=auth_date%3D1%26hash%3Dsecret"
        )
        telegram_query_body = await telegram_query_response.json()
        assert telegram_query_response.status == 401
        assert telegram_query_body["error"] == "query_auth_not_allowed"

        header_response = await client.get(
            "/api/profile",
            headers={"Authorization": f"Bearer {token}"},
        )
        header_body = await header_response.json()

        assert header_response.status == 200
        assert header_body["user_id"] == 1001
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
        query_auth = await client.get(f"/extraShop/admin?_auth={token}")
        query_auth_body = await query_auth.json()
        assert query_auth.status == 401
        assert query_auth_body["error"] == "authentication_required"

        session_response = await client.post(
            "/api/admin/session",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
        session_body = await session_response.json()
        assert session_response.status == 200
        assert session_body == {"status": "ok", "admin_url": "/extraShop/admin"}
        assert web_server.ADMIN_SESSION_COOKIE_NAME in session_response.cookies

        cookie = session_response.cookies[web_server.ADMIN_SESSION_COOKIE_NAME].value
        authorized = await client.get(
            "/extraShop/admin",
            cookies={web_server.ADMIN_SESSION_COOKIE_NAME: cookie},
        )
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
async def test_diagnostic_payment_webhooks_disabled_in_production_for_public_users(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        test_response = await client.get("/api/payments/webhook/test")
        debug_response = await client.post("/api/payments/webhook/debug", json={"event": "x"})

        assert test_response.status == 404
        assert debug_response.status == 404
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_diagnostic_payment_webhooks_require_explicit_dev_flag(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("PAYMENT_WEBHOOK_DIAGNOSTICS_ENABLED", "true")
    get_settings.cache_clear()
    client, _session_id = await _client(db=SecurityFakeDB())
    try:
        response = await client.get("/api/payments/webhook/test")
        body = await response.json()

        assert response.status == 200
        assert body["ok"] is True
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_community_admin_routes_use_central_boundary_and_reject_query_jwt(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=SecurityFakeDB())
    try:
        token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        approve_response = await client.post(
            f"/api/community/ideas/admin/approve?_auth={token}",
            json={"post_id": 12},
        )
        approve_body = await approve_response.json()
        status_response = await client.post(
            f"/api/community/ideas/admin/status?_auth={token}",
            json={"post_id": 12, "status": "approved"},
        )
        status_body = await status_response.json()
        bugs_response = await client.get(f"/api/community/bugs?_auth={token}")
        bugs_body = await bugs_response.json()

        assert approve_response.status == 401
        assert approve_body["error"] == "authentication_required"
        assert status_response.status == 401
        assert status_body["error"] == "authentication_required"
        assert bugs_response.status == 401
        assert bugs_body["error"] == "authentication_required"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_community_admin_status_and_bugs_reject_non_admin_header_auth(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=SecurityFakeDB())
    try:
        token = _auth_token(session_id, user_id=1001)
        status_response = await client.post(
            "/api/community/ideas/admin/status",
            headers={"Authorization": f"Bearer {token}"},
            json={"post_id": 12, "status": "approved"},
        )
        status_body = await status_response.json()
        bugs_response = await client.get(
            "/api/community/bugs",
            headers={"Authorization": f"Bearer {token}"},
        )
        bugs_body = await bugs_response.json()

        assert status_response.status == 403
        assert status_body["error"] == "admin_access_required"
        assert bugs_response.status == 403
        assert bugs_body["error"] == "admin_access_required"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_community_ideas_admin_delete_rejects_query_jwt(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=SecurityFakeDB())
    try:
        token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        response = await client.post(
            f"/api/community/ideas/admin/delete?_auth={token}",
            json={"post_id": 12},
        )
        body = await response.json()

        assert response.status == 401
        assert body["error"] == "query_auth_not_allowed"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_community_ideas_admin_delete_rejects_non_admin_header_auth(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=SecurityFakeDB())
    try:
        token = _auth_token(session_id, user_id=1001)
        response = await client.post(
            "/api/community/ideas/admin/delete",
            headers={"Authorization": f"Bearer {token}"},
            json={"post_id": 12},
        )
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "admin_access_required"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_community_ideas_admin_delete_allows_admin_header_auth(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=SecurityFakeDB())
    try:
        token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        response = await client.post(
            "/api/community/ideas/admin/delete",
            headers={"Authorization": f"Bearer {token}"},
            json={"post_id": 12},
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_community_admin_routes_allow_admin_authorization_header(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=SecurityFakeDB())
    try:
        token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        response = await client.post(
            "/api/community/ideas/admin/approve",
            headers={"Authorization": f"Bearer {token}"},
            json={"post_id": 12},
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert "Access-Control-Allow-Origin" not in response.headers
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_community_admin_errors_do_not_expose_raw_exception(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    client, session_id = await _client(db=FailingCommunityAdminDB())
    try:
        token = _auth_token(session_id, user_id=web_server.ADMIN_ID)
        response = await client.post(
            "/api/community/ideas/admin/approve",
            headers={"Authorization": f"Bearer {token}"},
            json={"post_id": 12},
        )
        body = await response.json()

        assert response.status == 500
        assert body == {
            "error": "internal_server_error",
            "message": "Internal server error",
        }
        assert "secret_community_table" not in json.dumps(body)
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.parametrize(
    ("route", "call_name", "expected_limit", "user_id"),
    [
        ("/api/community/news", "news", 50, 1001),
        ("/api/community/ideas", "ideas", 50, 1001),
        ("/api/community/announcements", "announcements", 50, 1001),
        ("/api/community/bugs", "bugs", 100, web_server.ADMIN_ID),
    ],
)
@pytest.mark.asyncio
async def test_community_pagination_clamps_huge_limits_before_db_call(
    monkeypatch, route, call_name, expected_limit, user_id
):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = CommunityPaginationDB()
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id, user_id=user_id)
        response = await client.get(
            f"{route}?limit=999999",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status == 200
        assert db.community_calls[call_name]["limit"] == expected_limit
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.parametrize(
    ("route", "call_name", "raw_offset", "expected_offset", "user_id"),
    [
        ("/api/community/news", "news", "-25", 0, 1001),
        ("/api/community/ideas", "ideas", "999999", 10000, 1001),
        ("/api/community/announcements", "announcements", "-1", 0, 1001),
        ("/api/community/bugs", "bugs", "1000000", 10000, web_server.ADMIN_ID),
    ],
)
@pytest.mark.asyncio
async def test_community_pagination_clamps_offsets_before_db_call(
    monkeypatch, route, call_name, raw_offset, expected_offset, user_id
):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = CommunityPaginationDB()
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id, user_id=user_id)
        response = await client.get(
            f"{route}?offset={raw_offset}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status == 200
        assert db.community_calls[call_name]["offset"] == expected_offset
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.parametrize(
    ("route", "user_id"),
    [
        ("/api/community/posts", 1001),
        ("/api/community/news", 1001),
        ("/api/community/ideas", 1001),
        ("/api/community/announcements", 1001),
        ("/api/community/bugs", web_server.ADMIN_ID),
    ],
)
@pytest.mark.asyncio
async def test_community_pagination_rejects_non_integer_limit(monkeypatch, route, user_id):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = CommunityPaginationDB()
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id, user_id=user_id)
        response = await client.get(
            f"{route}?limit=not-a-number",
            headers={"Authorization": f"Bearer {token}"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "invalid_limit"
        assert db.community_calls == {}
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.parametrize(
    ("route", "user_id"),
    [
        ("/api/community/news", 1001),
        ("/api/community/ideas", 1001),
        ("/api/community/announcements", 1001),
        ("/api/community/bugs", web_server.ADMIN_ID),
    ],
)
@pytest.mark.asyncio
async def test_community_pagination_rejects_non_integer_offset(monkeypatch, route, user_id):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = CommunityPaginationDB()
    client, session_id = await _client(db=db)
    try:
        token = _auth_token(session_id, user_id=user_id)
        response = await client.get(
            f"{route}?offset=not-a-number",
            headers={"Authorization": f"Bearer {token}"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "invalid_offset"
        assert db.community_calls == {}
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


@pytest.mark.asyncio
async def test_rustore_status_retries_succeeded_unprocessed_rewards():
    db = RuStoreRetryDB()
    client, session_id = await _client(db=db, payment_service=FakePaymentService())
    try:
        token = _auth_token(session_id)
        response = await client.get(f"/api/payments/status?payment_id=rustore_retry&_auth={token}")
        body = await response.json()

        assert response.status == 200
        assert body["provider"] == "rustore"
        assert body["status"] == "succeeded"
        assert body["rewards_processed"] is True
        assert db.gems_added == 100
        assert db.marked_processed is True
        assert db.rustore_payment_service.calls
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_rustore_payment_500_response_does_not_expose_exception_text():
    db = FailingRuStorePaymentDB()
    client, session_id = await _client(db=db, payment_service=FakePaymentService())
    try:
        token = _auth_token(session_id)
        response = await client.post(
            f"/api/payments/rustore/attach?_auth={token}",
            json={"payment_id": "rustore-secret", "invoice_id": "inv-secret"},
        )
        body = await response.json()

        assert response.status == 500
        assert body == {
            "error": "internal_server_error",
            "message": "Internal server error",
        }
        assert "secret rustore payment table" not in json.dumps(body)
    finally:
        await client.close()
