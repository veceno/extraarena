import hashlib
import time
import uuid
from types import SimpleNamespace

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.config import DECK_SIZE, get_settings
from infrastructure.extraid_database import SYNTHETIC_USER_ID_MIN
from web import server as web_server


STRONG_TEST_JWT_SECRET = "test-rlhf-jwt-secret-that-is-long-enough-2026"
SYNTHETIC_USER_ONE = SYNTHETIC_USER_ID_MIN + 101
SYNTHETIC_USER_TWO = SYNTHETIC_USER_ID_MIN + 202


@pytest.fixture(autouse=True)
def _secure_settings(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv(
        "ADMIN_SESSION_SECRET",
        "test-rlhf-admin-secret-that-is-long-enough-2026",
    )
    monkeypatch.setenv(
        "MCP_TOKEN_SECRET",
        "test-rlhf-mcp-secret-that-is-long-enough-2026",
    )
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://test.local")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://test.local")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class RlhfGameDB:
    def __init__(self):
        self.users = {SYNTHETIC_USER_ONE, SYNTHETIC_USER_TWO}
        self.mail = []

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},
            "disabled_card_ids": [],
        }

    async def is_admin(self, _user_id):
        return False

    async def fetchrow(self, query, *args):
        if "SELECT user_id FROM users" in query:
            user_id = int(args[0])
            return {"user_id": user_id} if user_id in self.users else None
        if "extra_pass, extra_pass_expires_at" in query:
            return {"extra_pass": False, "extra_pass_expires_at": None}
        return None

    async def fetchval(self, query, *args):
        if "primary_deck" in query:
            return 1
        return None

    async def get_user_deck_presets(self, _user_id):
        return [
            {
                "preset_number": 1,
                "preset_name": "Основная",
                "card_ids": list(range(1, DECK_SIZE + 1)),
                "is_playable": True,
                "has_hero": True,
                "updated_at": None,
            },
        ]

    async def create_mail(self, **kwargs):
        self.mail.append(kwargs)
        return {"success": True}


class RlhfExtraIDDB:
    def __init__(self):
        self.codes = {}
        self.sessions = {}
        self.rate_counts = {}
        self.consume_calls = 0

    async def check_rate_limit(self, key, max_requests, window_seconds):
        del window_seconds
        count = self.rate_counts.get(key, 0) + 1
        self.rate_counts[key] = count
        return count <= int(max_requests)

    async def fetchrow(self, _query, *_args):
        return None

    async def get_extra_account_by_user_id(self, user_id):
        if int(user_id) == SYNTHETIC_USER_ONE:
            return {"id": "00000000-0000-0000-0000-000000000101", "user_id": int(user_id)}
        return None

    async def account_has_identity_provider(self, _extra_account_id, provider):
        return str(provider) == "synthetic_user"

    async def cleanup_old_bot_codes(self, user_id, purpose=None):
        for code, row in list(self.codes.items()):
            if int(row["user_id"]) != int(user_id):
                continue
            if purpose is not None and row["purpose"] != purpose:
                continue
            if row.get("used_at") is None:
                self.codes.pop(code)

    async def create_bot_auth_code(
        self,
        code,
        user_id,
        purpose="telegram_transfer",
    ):
        self.codes[str(code)] = {
            "code": str(code),
            "user_id": int(user_id),
            "purpose": str(purpose),
            "used_at": None,
        }
        return {"created": True, **self.codes[str(code)]}

    async def consume_bot_auth_code(
        self,
        code,
        purpose="telegram_transfer",
        user_id=None,
    ):
        self.consume_calls += 1
        row = self.codes.get(str(code))
        if (
            not row
            or row.get("used_at") is not None
            or row["purpose"] != str(purpose)
            or (user_id is not None and int(row["user_id"]) != int(user_id))
        ):
            return None
        row["used_at"] = time.time()
        return dict(row)

    async def create_auth_session(
        self,
        user_id,
        auth_method,
        token_hash,
        expires_at,
        device_label=None,
        session_id=None,
    ):
        del device_label
        self.sessions[str(session_id)] = {
            "session_id": session_id,
            "user_id": int(user_id),
            "auth_method": str(auth_method),
            "token_hash": str(token_hash),
            "expires_at": expires_at,
        }
        return dict(self.sessions[str(session_id)])

    async def consume_bot_auth_code_and_create_session(
        self,
        code,
        *,
        purpose,
        user_id,
        session_id,
        auth_method,
        token_hash,
        expires_at,
        device_label=None,
    ):
        consumed = await self.consume_bot_auth_code(
            code,
            purpose=purpose,
            user_id=user_id,
        )
        if not consumed:
            return None
        await self.create_auth_session(
            user_id=user_id,
            auth_method=auth_method,
            token_hash=token_hash,
            expires_at=expires_at,
            device_label=device_label,
            session_id=session_id,
        )
        self.codes[str(code)]["session_id"] = session_id
        return consumed

    async def verify_session(self, session_id, token):
        row = self.sessions.get(str(session_id))
        if not row:
            return None
        if row["token_hash"] != hashlib.sha256(token.encode()).hexdigest():
            return None
        return dict(row)

    async def mark_bot_code_used(
        self,
        code,
        session_id,
        purpose="telegram_transfer",
        user_id=None,
    ):
        del purpose, user_id
        if str(code) in self.codes:
            self.codes[str(code)]["session_id"] = session_id


async def _client():
    game_db = RlhfGameDB()
    extraid_db = RlhfExtraIDDB()
    app = web_server.create_web_app(
        game_db,
        bot_token="bot-token",
        extraid_db=extraid_db,
        webapp_url="https://game.example",
    )
    # These tests exercise request handlers, not the perpetual maintenance jobs.
    app.on_startup.clear()
    app.on_cleanup.clear()
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, game_db, extraid_db


@pytest.mark.asyncio
async def test_request_code_has_uniform_anti_enumeration_response(monkeypatch):
    monkeypatch.setattr(web_server, "_make_bot_auth_code", lambda: "123456")
    client, game_db, extraid_db = await _client()
    try:
        existing = await client.post(
            "/api/rlhf/request-code",
            json={"identifier": str(SYNTHETIC_USER_ONE)},
        )
        missing = await client.post(
            "/api/rlhf/request-code",
            json={"identifier": "999999999999999999"},
        )

        assert existing.status == missing.status == 200
        assert await existing.json() == await missing.json() == {"status": "code_sent"}
        assert existing.headers["Cache-Control"] == missing.headers["Cache-Control"]
        assert "no-store" in existing.headers["Cache-Control"]
        assert game_db.mail
        assert extraid_db.codes["123456"]["purpose"] == web_server.RLHF_OTP_PURPOSE
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_oversized_rlhf_identifier_is_bounded_without_integer_parsing():
    client, _game_db, extraid_db = await _client()
    oversized = "9" * (web_server.RLHF_IDENTIFIER_MAX_LENGTH + 1)
    try:
        requested = await client.post(
            "/api/rlhf/request-code",
            json={"identifier": oversized},
        )
        verified = await client.post(
            "/api/rlhf/verify",
            json={"identifier": oversized, "code": "123456"},
        )

        assert requested.status == 200
        assert await requested.json() == {"status": "code_sent"}
        assert verified.status == 400
        assert await verified.json() == {"error": "invalid_input"}
        assert extraid_db.consume_calls == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_verify_wrong_identifier_does_not_consume_code():
    client, _game_db, extraid_db = await _client()
    await extraid_db.create_bot_auth_code(
        "123456",
        SYNTHETIC_USER_ONE,
        purpose=web_server.RLHF_OTP_PURPOSE,
    )
    try:
        wrong = await client.post(
            "/api/rlhf/verify",
            json={"identifier": str(SYNTHETIC_USER_TWO), "code": "123456"},
        )
        assert wrong.status == 400
        assert extraid_db.codes["123456"]["used_at"] is None

        correct = await client.post(
            "/api/rlhf/verify",
            json={"identifier": str(SYNTHETIC_USER_ONE), "code": "123456"},
        )
        assert correct.status == 200
        assert extraid_db.codes["123456"]["used_at"] is not None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_rlhf_verify_rejects_cross_purpose_code_without_consuming_it():
    client, _game_db, extraid_db = await _client()
    await extraid_db.create_bot_auth_code(
        "123456",
        SYNTHETIC_USER_ONE,
        purpose="telegram_transfer",
    )
    try:
        response = await client.post(
            "/api/rlhf/verify",
            json={"identifier": str(SYNTHETIC_USER_ONE), "code": "123456"},
        )
        assert response.status == 400
        assert await response.json() == {"error": "invalid_code"}
        assert extraid_db.codes["123456"]["used_at"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_rlhf_verify_is_limited_to_five_attempts_per_window():
    client, _game_db, extraid_db = await _client()
    try:
        statuses = []
        for attempt in range(6):
            response = await client.post(
                "/api/rlhf/verify",
                json={
                    "identifier": str(SYNTHETIC_USER_ONE),
                    "code": f"{attempt:06d}",
                },
            )
            statuses.append(response.status)

        assert statuses == [400, 400, 400, 400, 400, 429]
        assert extraid_db.consume_calls == 5
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_rlhf_scoped_token_is_rejected_by_game_api_and_accepted_by_decks():
    client, _game_db, extraid_db = await _client()
    await extraid_db.create_bot_auth_code(
        "123456",
        SYNTHETIC_USER_ONE,
        purpose=web_server.RLHF_OTP_PURPOSE,
    )
    try:
        verified = await client.post(
            "/api/rlhf/verify",
            json={"identifier": str(SYNTHETIC_USER_ONE), "code": "123456"},
        )
        body = await verified.json()
        assert verified.status == 200
        token = body["token"]

        claims = jwt.decode(
            token,
            STRONG_TEST_JWT_SECRET,
            algorithms=["HS256"],
            audience=web_server.RLHF_TOKEN_AUDIENCE,
        )
        assert claims["typ"] == web_server.RLHF_TOKEN_TYPE
        assert claims["scope"] == web_server.RLHF_TOKEN_SCOPE
        assert claims["exp"] - claims["iat"] == web_server.RLHF_TOKEN_TTL_SECONDS

        game_response = await client.get(
            "/api/profile",
            headers={"Authorization": f"Bearer {token}"},
        )
        decks_response = await client.get(
            "/api/rlhf/decks",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert game_response.status == 401
        assert decks_response.status == 200
        assert (await decks_response.json())["user_id"] == SYNTHETIC_USER_ONE

        normal_session_id = uuid.uuid4()
        normal_token, normal_expiry = web_server._make_jwt_session(
            SYNTHETIC_USER_ONE,
            normal_session_id,
            get_settings(),
        )
        await extraid_db.create_auth_session(
            user_id=SYNTHETIC_USER_ONE,
            auth_method="extraid",
            token_hash=hashlib.sha256(normal_token.encode()).hexdigest(),
            expires_at=normal_expiry,
            session_id=normal_session_id,
        )
        normal_token_response = await client.get(
            "/api/rlhf/decks",
            headers={"Authorization": f"Bearer {normal_token}"},
        )
        assert normal_token_response.status == 401
    finally:
        await client.close()


def test_trusted_client_ip_ignores_spoofed_forwarded_headers_from_direct_clients(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_CIDRS", "172.17.0.0/16")
    direct_request = SimpleNamespace(
        remote="203.0.113.5",
        headers={
            "CF-Connecting-IP": "198.51.100.10",
            "X-Forwarded-For": "198.51.100.11",
        },
    )
    proxied_request = SimpleNamespace(
        remote="127.0.0.1",
        headers={"CF-Connecting-IP": "198.51.100.10"},
    )
    docker_proxy_request = SimpleNamespace(
        remote="172.17.0.1",
        headers={"CF-Connecting-IP": "198.51.100.12"},
    )
    proxy_chain_request = SimpleNamespace(
        remote="127.0.0.1",
        headers={
            "X-Forwarded-For": "192.0.2.66, 198.51.100.20, 172.17.0.2"
        },
    )

    assert web_server._trusted_client_ip(direct_request) == "203.0.113.5"
    assert web_server._trusted_client_ip(proxied_request) == "198.51.100.10"
    assert web_server._trusted_client_ip(docker_proxy_request) == "198.51.100.12"
    assert web_server._trusted_client_ip(proxy_chain_request) == "198.51.100.20"


@pytest.mark.asyncio
async def test_main_html_responses_set_practical_security_headers():
    client, _game_db, _extraid_db = await _client()
    try:
        for path in ("/", "/arena", "/index.html"):
            response = await client.get(path, headers={"Accept-Encoding": "identity"})
            assert response.status == 200
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            csp = response.headers["Content-Security-Policy"]
            assert "default-src 'self'" in csp
            assert "https://telegram.org" in csp
            assert "https://unpkg.com" in csp
            assert "frame-ancestors 'self'" in csp
    finally:
        await client.close()
