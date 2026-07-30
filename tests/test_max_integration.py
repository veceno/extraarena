import hashlib
import hmac
import json
from pathlib import Path
import time
import uuid
from types import SimpleNamespace
from urllib.parse import urlencode

import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import main
from bot.max_client import normalize_max_update
from web import max_integration


BOT_TOKEN = "max-test-bot-token"


def _signed_init_data(
    *,
    user_id: int = 67890,
    auth_date: int | None = None,
    bot_token: str = BOT_TOKEN,
) -> str:
    values = {
        "auth_date": str(auth_date or int(time.time())),
        "query_id": str(uuid.uuid4()),
        "user": json.dumps(
            {
                "id": user_id,
                "first_name": "Max",
                "last_name": "Player",
                "username": "max_player",
                "language_code": "ru",
                "photo_url": "https://example.com/max-player.jpg",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    launch_params = "\n".join(
        f"{key}={value}" for key, value in sorted(values.items())
    )
    secret = hmac.new(
        b"WebAppData",
        bot_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    values["hash"] = hmac.new(
        secret,
        launch_params.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return urlencode(values)


def _settings(**overrides):
    values = {
        "max_bot_token": BOT_TOKEN,
        "max_bot_webhook_secret": "webhook-secret",
        "jwt_secret": "max-session-jwt-secret-long-enough-for-hs256-tests",
        "jwt_expiry_days": 30,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_max_init_data_signature_and_user_are_verified():
    verified = max_integration.verify_max_init_data(_signed_init_data(), BOT_TOKEN)

    assert verified is not None
    assert max_integration.extract_max_user(verified) == {
        "id": 67890,
        "first_name": "Max",
        "last_name": "Player",
        "username": "max_player",
        "language_code": "ru",
        "photo_url": "https://example.com/max-player.jpg",
    }


def test_max_init_data_rejects_tampering_duplicates_and_stale_launches():
    valid = _signed_init_data()
    assert max_integration.verify_max_init_data(
        valid.replace("Max%22", "Evil%22"),
        BOT_TOKEN,
    ) is None
    assert max_integration.verify_max_init_data(
        valid + "&auth_date=1",
        BOT_TOKEN,
    ) is None
    assert max_integration.verify_max_init_data(
        _signed_init_data(auth_date=int(time.time()) - 7200),
        BOT_TOKEN,
    ) is None


def test_max_webhook_update_normalization_supports_bot_and_message_shapes():
    started = normalize_max_update(
        {
            "update_type": "bot_started",
            "user": {"user_id": 123, "first_name": "Игрок"},
        }
    )
    message = normalize_max_update(
        {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 456, "first_name": "Макс"},
                "body": {"text": "/start payload"},
            },
        }
    )

    assert started["user_id"] == "123"
    assert started["display_name"] == "Игрок"
    assert message["user_id"] == "456"
    assert message["text"] == "/start payload"


def test_shared_webapp_bootstraps_max_before_render_and_keeps_identity_bound():
    index = Path("webapp/index.html").read_text(encoding="utf-8")
    arena = Path("webapp/arena.js").read_text(encoding="utf-8")
    server = Path("web/server.py").read_text(encoding="utf-8")

    assert '<script src="https://st.max.ru/js/max-web-app.js"></script>' in index
    assert '<script src="platform-bridge.js"></script>' in index
    assert "await window.ExtraArenaPlatform.ensureAuthSession();" in index
    assert "if (isMaxGameClient())" in index
    assert "add('auth', getMaxAuthToken(), 'max');" in index
    assert "const isPlatformBound = isPlatformFlow" in index
    assert "hasLocalExtraSession && !isPlatformFlow" in index
    assert "cannot_delete_max_bound" in Path(
        "web/extraid_handlers.py"
    ).read_text(encoding="utf-8")
    assert "register_max_routes(app)" in server
    assert "await platform.ensureAuthSession();" in arena
    assert "window.ExtraArenaPlatform?.isMax?.()" in arena


@pytest.mark.asyncio
async def test_max_auth_exchange_issues_internal_session(monkeypatch):
    class GameDB:
        def __init__(self):
            self.calls = []

        async def resolve_or_create_platform_user(self, **kwargs):
            self.calls.append(kwargs)
            return 8_000_000_000_000_001, True

    class ExtraIDDB:
        def __init__(self):
            self.executed = []
            self.session = None

        async def execute(self, query, *args):
            self.executed.append((query, args))

        async def create_auth_session(self, *args, **kwargs):
            self.session = (args, kwargs)

        async def get_any_extra_account_by_user_id(self, user_id):
            assert user_id == 8_000_000_000_000_001
            return None

    game_db = GameDB()
    extra_db = ExtraIDDB()
    app = web.Application()
    app["db"] = game_db
    app["extraid_db"] = extra_db
    app.router.add_post("/api/auth/max", max_integration.max_auth_exchange_handler)
    monkeypatch.setattr(max_integration, "get_settings", lambda: _settings())

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/auth/max",
            json={"init_data": _signed_init_data(user_id=987654321)},
        )
        body = await response.json()
    finally:
        await client.close()

    assert response.status == 200
    assert body["ok"] is True
    assert body["provider"] == "max"
    assert body["user_id"] == 8_000_000_000_000_001
    decoded = jwt.decode(
        body["token"],
        "max-session-jwt-secret-long-enough-for-hs256-tests",
        algorithms=["HS256"],
    )
    assert decoded["user_id"] == 8_000_000_000_000_001
    assert game_db.calls[0]["provider"] == "max"
    assert game_db.calls[0]["subject"] == "987654321"
    assert extra_db.session[0][1] == "max"


@pytest.mark.asyncio
async def test_max_webhook_requires_secret_and_sends_open_app(monkeypatch):
    class Bot:
        def __init__(self):
            self.sent = []

        async def send_message(self, user_id, text, *, open_app=False, **kwargs):
            self.sent.append((user_id, text, open_app))
            return {"ok": True, "status": 200, "data": {}}

    bot = Bot()
    app = web.Application()
    app["max_bot_client"] = bot
    app.router.add_post("/api/max/webhook", max_integration.max_bot_webhook_handler)
    monkeypatch.setattr(max_integration, "get_settings", lambda: _settings())

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        rejected = await client.post(
            "/api/max/webhook",
            json={"update_type": "bot_started", "user": {"user_id": 123}},
        )
        accepted = await client.post(
            "/api/max/webhook",
            headers={"X-Max-Bot-Api-Secret": "webhook-secret"},
            json={"update_type": "bot_started", "user": {"user_id": 123}},
        )
    finally:
        await client.close()

    assert rejected.status == 401
    assert accepted.status == 200
    assert bot.sent == [
        (
            "123",
            "Добро пожаловать в ExtraArena!\n\n"
            "Открой игру кнопкой ниже. MAX-аккаунт будет подтверждён автоматически, "
            "а ExtraID можно создать и навсегда привязать внутри игры.",
            True,
        )
    ]


@pytest.mark.asyncio
async def test_notification_outbox_routes_max_identity_to_max_bot():
    class DB:
        def __init__(self):
            self.sent = []
            self.failed = []

        async def get_notification_delivery_mode(self, user_id):
            return "telegram_only"

        async def get_platform_identity_for_user(self, user_id, provider):
            assert provider == "max"
            return {"subject": "1122334455", "user_id": user_id}

        async def mark_notification_sent(self, notification_id):
            self.sent.append(notification_id)

        async def mark_notification_failed(self, notification_id):
            self.failed.append(notification_id)

    class MaxBot:
        def __init__(self):
            self.sent = []

        async def send_message(self, user_id, text, **kwargs):
            self.sent.append((user_id, text, kwargs))
            return {"ok": True, "status": 200, "data": {"body": {"mid": "m1"}}}

    class TelegramBot:
        async def send_message(self, **kwargs):
            raise AssertionError("MAX user must not be sent to Telegram")

    db = DB()
    max_bot = MaxBot()
    await main._deliver_notification(
        TelegramBot(),
        db,
        "https://example.com/game",
        {
            "id": 77,
            "user_id": 8_000_000_000_000_777,
            "category": "generator",
            "event_type": "generator_new_key",
            "payload": {"keys": 1},
            "attempts": 1,
        },
        max_bot=max_bot,
    )

    assert db.sent == [77]
    assert db.failed == []
    assert max_bot.sent[0][0] == "1122334455"
    assert max_bot.sent[0][2] == {"open_app": True, "text_format": "html"}
