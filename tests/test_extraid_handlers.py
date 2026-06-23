import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrastructure.config import get_settings
from web import extraid_handlers


class FakeRequest:
    def __init__(self, payload, query=None, app=None, headers=None):
        self._payload = payload
        self.rel_url = SimpleNamespace(query=query or {})
        self.app = app or {"bot_token": "bot-token"}
        self.headers = headers or {}
        self.remote = "127.0.0.1"

    async def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _local_settings(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.delenv("TELEGRAM_API_INSECURE_SSL", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeGameDB:
    def __init__(self):
        self.ensured_users = []
        self.executed = []
        self.changed_nicknames = []
        self.registered_push_devices = []
        self.unregistered_push_devices = []
        self.deleted_users = []

    async def ensure_user(self, **kwargs):
        self.ensured_users.append(kwargs)

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "UPDATE 1"

    async def change_nickname(self, user_id, nickname, cost_gems=0):
        self.changed_nicknames.append((user_id, nickname, cost_gems))
        return {"success": True}

    async def fetchval(self, query, *args):
        if "FROM users" in query:
            return 1
        return None

    async def delete_user(self, user_id):
        self.deleted_users.append(user_id)
        return True

    async def register_push_device(self, user_id, **kwargs):
        record = {"id": 501, "user_id": user_id, **kwargs}
        self.registered_push_devices.append(record)
        return record

    async def unregister_push_device(self, user_id, *, token):
        self.unregistered_push_devices.append((user_id, token))
        return True


class FakeExtraIDDB:
    def __init__(self):
        self.account_id = uuid.uuid4()
        self.executed = []
        self.bonus_marked = None
        self.revoked_user_sessions = []
        self.deleted_extra_accounts = []

    async def get_synthetic_user_id(self):
        return 987654321

    async def get_extra_account_by_email(self, email):
        return None

    async def get_extra_account_by_user_id(self, user_id):
        return None

    async def get_any_extra_account_by_user_id(self, user_id):
        return None

    async def get_any_extra_account_by_email(self, email):
        return None

    async def has_user_claimed_reg_bonus(self, user_id):
        return False

    async def fetchrow(self, query, *args):
        return None

    async def fetchval(self, query, *args):
        return None

    async def create_extra_account(self, user_id, display_id, email, password_hash, nickname=None):
        return {
            "id": self.account_id,
            "user_id": user_id,
            "display_id": display_id,
            "email": email,
            "password_hash": password_hash,
            "nickname": nickname,
            "reg_bonus_claimed": False,
        }

    async def mark_reg_bonus_claimed(self, extra_account_id):
        self.bonus_marked = extra_account_id
        return True

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "UPDATE 1"

    async def create_bot_auth_code(self, code, user_id):
        self.created_bot_code = (code, user_id)
        return {"code": code}

    async def verify_bot_auth_code(self, code):
        return None

    async def mark_bot_code_used(self, code, session_id):
        self.used_bot_code = (code, session_id)

    async def cleanup_old_bot_codes(self, user_id):
        self.cleaned_bot_codes_for = user_id

    async def revoke_all_user_sessions(self, user_id):
        self.revoked_user_sessions.append(user_id)

    async def soft_delete_extra_account(self, extra_account_id):
        self.deleted_extra_accounts.append(extra_account_id)


def async_lambda(value):
    """Return an async function that ignores its args and returns `value`."""
    async def _wrapper(*args, **kwargs):
        return value
    return _wrapper


def test_rate_limit_store_prunes_expired_keys(monkeypatch):
    extraid_handlers._rate_limit_store.clear()
    extraid_handlers._rate_limit_store["stale:key"] = [100.0]
    monkeypatch.setattr(extraid_handlers.time, "time", lambda: 1000.0)

    assert extraid_handlers._check_rate_limit("fresh:key", 3, 60) is True
    assert "stale:key" not in extraid_handlers._rate_limit_store


@pytest.mark.asyncio
async def test_extraid_rate_limit_requires_shared_backend_outside_development(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "test-extraid-jwt-secret-that-is-long-enough-2026")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-extraid-admin-secret-that-is-long-enough-2026")
    get_settings.cache_clear()
    request = FakeRequest(
        {},
        app={"extraid_db": object()},
    )

    with pytest.raises(RuntimeError, match="shared rate limit"):
        await extraid_handlers._check_rate_limit_for_request(request, "login:key", 10, 60)


@pytest.mark.asyncio
async def test_extraid_rate_limit_uses_database_backend_when_available(monkeypatch):
    class SharedRateLimitDB:
        def __init__(self):
            self.calls = []

        async def check_rate_limit(self, key, max_requests, window_seconds):
            self.calls.append((key, max_requests, window_seconds))
            return False

    backend = SharedRateLimitDB()
    request = FakeRequest(
        {},
        app={"extraid_db": backend},
    )

    allowed = await extraid_handlers._check_rate_limit_for_request(request, "login:key", 10, 60)

    assert allowed is False
    assert backend.calls == [("login:key", 10, 60)]


def test_bot_auth_code_uses_secure_random_source():
    source = Path("web/extraid_handlers.py").read_text(encoding="utf-8")

    assert "secrets.choice" in source
    assert "random.choices" not in source


def test_email_validator_rejects_broken_domain_shapes():
    assert extraid_handlers._valid_email("player@example.com") is True
    assert extraid_handlers._valid_email("a@.") is False
    assert extraid_handlers._valid_email("@@..") is False


@pytest.mark.asyncio
async def test_anonymous_auth_creates_jwt_session(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {"nickname": "GuestHero", "device_label": "Pixel"},
        app={"db": game_db, "extraid_db": extra_db},
    )
    request.remote = "127.0.0.1"

    response = await extraid_handlers.anonymous_auth_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert body["anonymous"] is True
    assert body["user_id"] == 987654321
    assert body["token"]
    assert game_db.ensured_users[0]["first_name"] == "GuestHero"
    assert game_db.changed_nicknames == []
    assert any("auth_source = 'android_anonymous'" in query for query, _ in game_db.executed)
    assert any("INSERT INTO auth_sessions" in query for query, _ in extra_db.executed)


@pytest.mark.asyncio
async def test_anonymous_auth_rejects_invalid_nickname_symbols(monkeypatch):
    extraid_handlers._rate_limit_store.clear()
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {"nickname": "bad nick!", "device_label": "Pixel"},
        app={"db": game_db, "extraid_db": extra_db},
    )

    response = await extraid_handlers.anonymous_auth_handler(request)
    body = json.loads(response.text)

    assert response.status == 400
    assert body["error"] == "invalid_nickname"
    assert game_db.ensured_users == []


@pytest.mark.asyncio
async def test_extraid_register_uses_strict_email_validation(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {"email": "a@.", "password": "strongpass", "nickname": "Player_1"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 400
    assert body["error"] == "invalid_email"
    assert game_db.ensured_users == []


@pytest.mark.asyncio
async def test_telegram_register_links_without_email_login(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {"email": "Player@Example.com", "password": "strongpass", "nickname": "Player_1"},
        query={"_auth": "telegram-init-data"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: {"user": "ok"})
    monkeypatch.setattr(extraid_handlers, "_validate_auth_date_fn", lambda data: True)
    monkeypatch.setattr(extraid_handlers, "_extract_user_id_from_init_data_fn", lambda data: 12345)
    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "1234-ABC")
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", lambda *args: None)
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert body["linked_telegram"] is True
    assert body["reg_bonus"] is True
    assert "token" not in body
    assert extra_db.bonus_marked == extra_db.account_id
    assert any("keys = COALESCE(keys, 0) + 3" in query for query, _ in game_db.executed)


@pytest.mark.asyncio
async def test_mobile_register_with_jwt_links_current_app_user(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {
            "email": "Mobile@Example.com",
            "password": "strongpass",
            "nickname": "MobileHero",
            "client": "android_app",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
        headers={"Authorization": "Bearer mobile-jwt"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        assert token == "mobile-jwt"
        return 4242, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: None)
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "4242-APP")
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert body["linked_telegram"] is False
    assert body["reg_bonus"] is False
    assert "token" not in body
    assert game_db.ensured_users[0]["user_id"] == 4242
    assert extra_db.bonus_marked is None
    assert any("auth_source = $2" in query and args[1] == "extraid_mobile" for query, args in game_db.executed)


@pytest.mark.asyncio
async def test_mobile_register_accepts_bearer_jwt_without_query_auth(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {
            "email": "Bearer@Example.com",
            "password": "strongpass",
            "nickname": "BearerHero",
            "client": "android_app",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
        headers={"Authorization": "Bearer mobile-jwt"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        assert token == "mobile-jwt"
        return 5252, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: None)
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "5252-APP")
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert game_db.ensured_users[0]["user_id"] == 5252
    assert any("auth_source = $2" in query and args[1] == "extraid_mobile" for query, args in game_db.executed)


@pytest.mark.asyncio
async def test_mobile_register_rejects_query_jwt(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    token = "a.b.c"
    request = FakeRequest(
        {
            "email": "QueryJwt@Example.com",
            "password": "strongpass",
            "nickname": "QueryHero",
            "client": "android_app",
        },
        query={"_auth": token},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    async def verify_jwt(*args):
        raise AssertionError("query JWT must not be verified")

    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: None)
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 401
    assert body["error"] == "invalid_auth"
    assert game_db.ensured_users == []


class LoginExtraIDDB:
    def __init__(self):
        self.account_id = uuid.uuid4()
        self.inserted_session_id = None

    async def get_extra_account_by_email(self, email):
        return {
            "id": self.account_id,
            "user_id": 777,
            "display_id": "7777-AAA",
            "password_hash": "hash",
            "reg_bonus_claimed": True,
        }

    async def get_any_extra_account_by_user_id(self, user_id):
        return None

    async def has_user_claimed_reg_bonus(self, user_id):
        return True

    async def execute(self, query, *args):
        if "INSERT INTO auth_sessions" in query:
            self.inserted_session_id = args[0]
        return "INSERT 0 1"


@pytest.mark.asyncio
async def test_email_login_inserts_uuid_session_id(monkeypatch):
    game_db = FakeGameDB()
    extra_db = LoginExtraIDDB()
    request = FakeRequest(
        {"email": "player@example.com", "password": "strongpass"},
        app={"db": game_db, "extraid_db": extra_db},
    )

    monkeypatch.setattr(extraid_handlers.bcrypt, "checkpw", lambda password, password_hash: True)

    response = await extraid_handlers.extraid_login_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert isinstance(extra_db.inserted_session_id, uuid.UUID)


@pytest.mark.asyncio
async def test_email_login_applies_rate_limit_before_password_check(monkeypatch):
    calls = []

    def deny_rate_limit(key, max_requests, window_seconds):
        calls.append((key, max_requests, window_seconds))
        return False

    game_db = FakeGameDB()
    extra_db = LoginExtraIDDB()
    request = FakeRequest(
        {"email": "Player@Example.com", "password": "strongpass"},
        app={"db": game_db, "extraid_db": extra_db},
    )
    request.remote = "203.0.113.10"

    monkeypatch.setattr(extraid_handlers, "_check_rate_limit", deny_rate_limit)

    response = await extraid_handlers.extraid_login_handler(request)
    body = json.loads(response.text)

    assert response.status == 429
    assert body["error"] == "rate_limited"
    assert calls
    assert calls[0][0].startswith("extraid_login:203.0.113.10:")
    assert calls[0][1:] == (10, 60)


class LoginUnclaimedBonusAlreadyMarkedDB(LoginExtraIDDB):
    async def get_extra_account_by_email(self, email):
        extra = await super().get_extra_account_by_email(email)
        extra["reg_bonus_claimed"] = False
        return extra

    async def has_user_claimed_reg_bonus(self, user_id):
        return False

    async def mark_reg_bonus_claimed(self, extra_account_id):
        return False


@pytest.mark.asyncio
async def test_email_login_does_not_double_grant_registration_bonus(monkeypatch):
    game_db = FakeGameDB()
    extra_db = LoginUnclaimedBonusAlreadyMarkedDB()
    request = FakeRequest(
        {"email": "player@example.com", "password": "strongpass"},
        app={"db": game_db, "extraid_db": extra_db},
    )

    monkeypatch.setattr(extraid_handlers.bcrypt, "checkpw", lambda password, password_hash: True)

    response = await extraid_handlers.extraid_login_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    # reg_bonus is False because mark_reg_bonus_claimed returned False (race-loser),
    # but keys were still credited first (P1#2: credit before mark to avoid losing bonus).
    assert body["reg_bonus"] is False
    assert any("keys = COALESCE(keys, 0) + 3" in query for query, _ in game_db.executed)


class FakeBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, user_id, text):
        self.messages.append((user_id, text))


@pytest.mark.asyncio
async def test_telegram_transfer_request_code_sends_dm(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    bot = FakeBot()
    request = FakeRequest(
        {"telegram_id": "12345"},
        app={"db": game_db, "extraid_db": extra_db, "telegram_bot": bot, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_make_bot_auth_code", lambda: "123456")

    response = await extraid_handlers.telegram_transfer_request_code_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert extra_db.created_bot_code == ("123456", 12345)
    assert bot.messages == [(12345, "Код для переноса ExtraArena в приложение: 123456\n\nОн действует 5 минут. Никому его не показывай.")]


class TelegramTransferExtraIDDB(FakeExtraIDDB):
    def __init__(self):
        super().__init__()
        self.inserted_session_id = None
        self.used_bot_code = None
        self.created_accounts = []

    async def verify_bot_auth_code(self, code):
        if code == "123456":
            return {"code": code, "user_id": 12345}
        return None

    async def consume_bot_auth_code(self, code):
        return await self.verify_bot_auth_code(code)

    async def create_extra_account(self, user_id, display_id, email, password_hash, nickname=None):
        account = await super().create_extra_account(user_id, display_id, email, password_hash, nickname)
        self.created_accounts.append(account)
        return account

    async def execute(self, query, *args):
        self.executed.append((query, args))
        if "INSERT INTO auth_sessions" in query:
            self.inserted_session_id = args[0]
        return "INSERT 0 1"


@pytest.mark.asyncio
async def test_telegram_transfer_complete_creates_extraid_session(monkeypatch):
    game_db = FakeGameDB()
    extra_db = TelegramTransferExtraIDDB()
    request = FakeRequest(
        {
            "telegram_id": "12345",
            "code": "123456",
            "email": "Player@Example.com",
            "password": "strongpass",
            "device_label": "Pixel",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "5555-TG")
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.telegram_transfer_complete_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert body["user_id"] == 12345
    assert body["display_id"] == "5555-TG"
    assert body["token"]
    assert body["reg_bonus"] is True
    assert isinstance(extra_db.inserted_session_id, uuid.UUID)
    assert extra_db.used_bot_code == ("123456", extra_db.inserted_session_id)
    assert game_db.ensured_users == []
    assert any("auth_source = 'telegram_transfer'" in query for query, _ in game_db.executed)
    assert any("keys = COALESCE(keys, 0) + 3" in query for query, _ in game_db.executed)


class TelegramTransferEmailTakenDB(TelegramTransferExtraIDDB):
    def __init__(self):
        super().__init__()
        self.consumed_codes = []

    async def get_any_extra_account_by_email(self, email):
        # Active (non-deleted) account with this email — should block registration.
        return {"id": uuid.uuid4(), "email": email, "deleted_at": None}

    async def consume_bot_auth_code(self, code):
        self.consumed_codes.append(code)
        return await super().consume_bot_auth_code(code)


@pytest.mark.asyncio
async def test_telegram_transfer_returns_email_taken_for_active_duplicate_email(monkeypatch):
    game_db = FakeGameDB()
    extra_db = TelegramTransferEmailTakenDB()
    request = FakeRequest(
        {
            "telegram_id": "12345",
            "code": "123456",
            "email": "Taken@Example.com",
            "password": "strongpass",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    response = await extraid_handlers.telegram_transfer_complete_handler(request)
    body = json.loads(response.text)

    assert response.status == 409
    assert body["error"] == "email_taken"
    assert extra_db.consumed_codes == []
    assert not any("INSERT INTO auth_sessions" in query for query, _ in extra_db.executed)


class TelegramTransferSoftDeletedEmailDB(TelegramTransferExtraIDDB):
    """Soft-deleted email must be reusable (partial unique index on active rows)."""
    async def get_any_extra_account_by_email(self, email):
        return None  # soft-deleted rows are filtered out -> email is free


class TelegramTransferAlreadyConsumedDB(TelegramTransferExtraIDDB):
    async def verify_bot_auth_code(self, code):
        return {"code": code, "user_id": 12345}

    async def consume_bot_auth_code(self, code):
        return None


@pytest.mark.asyncio
async def test_telegram_transfer_complete_does_not_create_account_after_duplicate_consume(monkeypatch):
    game_db = FakeGameDB()
    extra_db = TelegramTransferAlreadyConsumedDB()
    request = FakeRequest(
        {
            "telegram_id": "12345",
            "code": "123456",
            "email": "Player@Example.com",
            "password": "strongpass",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "7777-X")
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.telegram_transfer_complete_handler(request)
    body = json.loads(response.text)

    # P1#3: consume now happens after session insert. On duplicate consume, the created
    # account is rolled back (soft-deleted + users.extra_account_id cleared) and 400 returned.
    assert response.status == 400
    assert body["error"] == "invalid_code"
    # Account was created then rolled back via soft_delete
    assert extra_db.account_id in extra_db.deleted_extra_accounts
    assert any("extra_account_id = NULL" in query and "auth_source = 'telegram'" in query for query, _ in game_db.executed)


class LinkExtraIDDB(FakeExtraIDDB):
    SYNTHETIC_JWT_USER = 9_100_000_004_242

    def __init__(self):
        super().__init__()
        self.account_id = uuid.uuid4()

    async def get_extra_account_by_user_id(self, user_id):
        if user_id == self.SYNTHETIC_JWT_USER:
            return {
                "id": self.account_id,
                "user_id": self.SYNTHETIC_JWT_USER,
                "display_id": "4242-APP",
                "email": "player@example.com",
            }
        return None

    async def get_any_extra_account_by_user_id(self, user_id):
        # Only the synthetic account exists; 12345 (Telegram) has no prior ExtraID.
        if user_id == self.SYNTHETIC_JWT_USER:
            return await self.get_extra_account_by_user_id(user_id)
        return None


@pytest.mark.asyncio
async def test_extraid_link_rejects_stale_telegram_init_data(monkeypatch):
    game_db = FakeGameDB()
    extra_db = LinkExtraIDDB()
    request = FakeRequest(
        {"tg_init_data": "stale-init-data"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
        headers={"Authorization": "Bearer mobile-jwt"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        assert token == "mobile-jwt"
        return LinkExtraIDDB.SYNTHETIC_JWT_USER, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: {"auth_date": "1", "user": "{}"})
    monkeypatch.setattr(extraid_handlers, "_validate_auth_date_fn", lambda data: False)
    monkeypatch.setattr(extraid_handlers, "_extract_user_id_from_init_data_fn", lambda data: 12345)

    response = await extraid_handlers.extraid_link_handler(request)
    body = json.loads(response.text)

    assert response.status == 400
    assert body["error"] == "invalid_tg_init_data"
    assert extra_db.executed == []
    assert game_db.executed == []


@pytest.mark.asyncio
async def test_extraid_link_rejects_query_jwt(monkeypatch):
    game_db = FakeGameDB()
    extra_db = LinkExtraIDDB()
    request = FakeRequest(
        {"tg_init_data": "valid-init-data"},
        query={"_auth": "a.b.c"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    async def verify_jwt(*args):
        raise AssertionError("query JWT must not be verified")

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)

    response = await extraid_handlers.extraid_link_handler(request)
    body = json.loads(response.text)

    assert response.status == 401
    assert body["error"] == "auth_required"
    assert extra_db.executed == []


@pytest.mark.asyncio
async def test_extraid_link_clears_old_owner_and_revokes_old_sessions(monkeypatch):
    game_db = FakeGameDB()
    extra_db = LinkExtraIDDB()
    request = FakeRequest(
        {"tg_init_data": "valid-init-data"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
        headers={"Authorization": "Bearer mobile-jwt"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        assert token == "mobile-jwt"
        return LinkExtraIDDB.SYNTHETIC_JWT_USER, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: {"auth_date": "1", "user": "{}"})
    monkeypatch.setattr(extraid_handlers, "_validate_auth_date_fn", lambda data: True)
    monkeypatch.setattr(extraid_handlers, "_extract_user_id_from_init_data_fn", lambda data: 12345)

    response = await extraid_handlers.extraid_link_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert any("UPDATE extra_accounts SET user_id" in query and args[0] == 12345 for query, args in extra_db.executed)
    assert any("extra_account_id = NULL" in query and args == (LinkExtraIDDB.SYNTHETIC_JWT_USER,) for query, args in game_db.executed)
    assert extra_db.revoked_user_sessions == [LinkExtraIDDB.SYNTHETIC_JWT_USER]


@pytest.mark.asyncio
async def test_push_register_binds_fcm_token_to_jwt_user(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {
            "platform": "android",
            "token": "fcm-token",
            "app_version": "0.1.0",
            "device_label": "Pixel 8",
            "os_name": "Android",
            "os_version": "14",
            "timezone": "Europe/Moscow",
            "utc_offset_minutes": 180,
        },
        app={"db": game_db, "extraid_db": extra_db},
        headers={"Authorization": "Bearer jwt-token"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        assert token == "jwt-token"
        return 4242, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)

    response = await extraid_handlers.push_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body == {"ok": True, "device_id": 501}
    assert game_db.registered_push_devices == [{
        "id": 501,
        "user_id": 4242,
        "token": "fcm-token",
        "platform": "android",
        "app_version": "0.1.0",
        "device_label": "Pixel 8",
        "os_name": "Android",
        "os_version": "14",
        "timezone": "Europe/Moscow",
        "utc_offset_minutes": 180,
    }]


@pytest.mark.asyncio
async def test_push_unregister_disables_user_token(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {"auth": "jwt-token", "token": "fcm-token"},
        app={"db": game_db, "extraid_db": extra_db},
    )

    async def verify_jwt(token, _extra_db, _settings):
        assert token == "jwt-token"
        return 4242, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)

    response = await extraid_handlers.push_unregister_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body == {"ok": True, "removed": True}
    assert game_db.unregistered_push_devices == [(4242, "fcm-token")]


# ═══════════════════════════════════════════════════════════════════════════
# New tests: duplication block, email reuse, reg-bonus per-user, delete block,
# compensating cleanup
# ═══════════════════════════════════════════════════════════════════════════


class TelegramBoundDeleteDB(FakeExtraIDDB):
    """ExtraID bound to a real Telegram ID (< SYNTHETIC_USER_ID_MIN)."""
    TG_USER = 12345

    async def get_extra_account_by_user_id(self, user_id):
        if user_id == self.TG_USER:
            return {
                "id": self.account_id,
                "user_id": self.TG_USER,
                "display_id": "1234-TG",
                "email": "tg@example.com",
                "password_hash": "hash",
                "reg_bonus_claimed": True,
            }
        return None


@pytest.mark.asyncio
async def test_delete_refused_for_telegram_bound_account(monkeypatch):
    """Telegram-bound ExtraID cannot be deleted (closes duplication exploit)."""
    game_db = FakeGameDB()
    extra_db = TelegramBoundDeleteDB()
    request = FakeRequest(
        {"password": "strongpass", "confirm": "DELETE"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "require_user_id_fn", async_lambda(TelegramBoundDeleteDB.TG_USER))
    monkeypatch.setattr(extraid_handlers.bcrypt, "checkpw", lambda password, password_hash: True)

    response = await extraid_handlers.extraid_delete_account_handler(request)
    body = json.loads(response.text)

    assert response.status == 403
    assert body["error"] == "cannot_delete_telegram_bound"
    # Account NOT soft-deleted, sessions NOT revoked
    assert extra_db.deleted_extra_accounts == []
    assert extra_db.revoked_user_sessions == []


class SyntheticDeleteDB(FakeExtraIDDB):
    """ExtraID bound to a synthetic user_id (email-only/anonymous)."""
    SYNTH_USER = 9_100_000_000_001

    async def get_extra_account_by_user_id(self, user_id):
        if user_id == self.SYNTH_USER:
            return {
                "id": self.account_id,
                "user_id": self.SYNTH_USER,
                "display_id": "9100-APP",
                "email": "synth@example.com",
                "password_hash": "hash",
                "reg_bonus_claimed": False,
            }
        return None


@pytest.mark.asyncio
async def test_delete_allowed_for_synthetic_account(monkeypatch):
    """Synthetic (email-only) ExtraID can be deleted; synthetic users row removed."""
    game_db = FakeGameDB()
    extra_db = SyntheticDeleteDB()
    request = FakeRequest(
        {"password": "strongpass", "confirm": "DELETE"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "require_user_id_fn", async_lambda(SyntheticDeleteDB.SYNTH_USER))
    monkeypatch.setattr(extraid_handlers.bcrypt, "checkpw", lambda password, password_hash: True)

    response = await extraid_handlers.extraid_delete_account_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert extra_db.account_id in extra_db.deleted_extra_accounts
    assert SyntheticDeleteDB.SYNTH_USER in extra_db.revoked_user_sessions
    # Synthetic users row should be deleted (not just updated)
    assert SyntheticDeleteDB.SYNTH_USER in game_db.deleted_users


@pytest.mark.asyncio
async def test_delete_rejects_empty_password(monkeypatch):
    """Empty password returns 401 invalid_password, not 500."""
    game_db = FakeGameDB()
    extra_db = SyntheticDeleteDB()
    request = FakeRequest(
        {"password": "", "confirm": "DELETE"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "require_user_id_fn", async_lambda(SyntheticDeleteDB.SYNTH_USER))

    response = await extraid_handlers.extraid_delete_account_handler(request)
    body = json.loads(response.text)

    assert response.status == 401
    assert body["error"] == "invalid_password"


class DuplicationBlockDB(FakeExtraIDDB):
    """Simulates a soft-deleted ExtraID for Telegram user 12345."""
    TG_USER = 12345

    async def get_any_extra_account_by_user_id(self, user_id):
        if user_id == self.TG_USER:
            return {
                "id": uuid.uuid4(),
                "user_id": user_id,
                "display_id": "OLD-TG",
                "email": "old@example.com",
                "deleted_at": "2026-01-01T00:00:00Z",
            }
        return None

    async def verify_bot_auth_code(self, code):
        if code == "123456":
            return {"code": code, "user_id": self.TG_USER}
        return None

    async def consume_bot_auth_code(self, code):
        return await self.verify_bot_auth_code(code)


@pytest.mark.asyncio
async def test_register_blocks_duplicate_after_delete(monkeypatch):
    """After delete, same Telegram ID cannot create a new ExtraID."""
    game_db = FakeGameDB()
    extra_db = DuplicationBlockDB()
    request = FakeRequest(
        {"email": "new@example.com", "password": "strongpass", "nickname": "NewHero"},
        query={"_auth": "telegram-init-data"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: {"user": "ok"})
    monkeypatch.setattr(extraid_handlers, "_validate_auth_date_fn", lambda data: True)
    monkeypatch.setattr(extraid_handlers, "_extract_user_id_from_init_data_fn", lambda data: 12345)
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", lambda *args: None)

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 409
    assert body["error"] == "extraid_already_exists"
    assert game_db.ensured_users == []


@pytest.mark.asyncio
async def test_transfer_complete_blocks_duplicate_after_delete(monkeypatch):
    """After delete, same Telegram ID cannot transfer-create a new ExtraID."""
    game_db = FakeGameDB()
    extra_db = DuplicationBlockDB()
    request = FakeRequest(
        {"telegram_id": "12345", "code": "123456", "email": "new@example.com", "password": "strongpass"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    response = await extraid_handlers.telegram_transfer_complete_handler(request)
    body = json.loads(response.text)

    assert response.status == 409
    assert body["error"] == "extraid_already_exists"


@pytest.mark.asyncio
async def test_transfer_request_code_blocks_duplicate_after_delete(monkeypatch):
    """After delete, same Telegram ID cannot even request a transfer code."""
    game_db = FakeGameDB()
    extra_db = DuplicationBlockDB()
    request = FakeRequest(
        {"telegram_id": "12345"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    response = await extraid_handlers.telegram_transfer_request_code_handler(request)
    body = json.loads(response.text)

    assert response.status == 409
    assert body["error"] == "extraid_already_exists"


class RegBonusPerUserDB(FakeExtraIDDB):
    """Tracks reg-bonus claims per user_id to prevent farming."""
    def __init__(self):
        super().__init__()
        self.claimed_users = set()

    async def has_user_claimed_reg_bonus(self, user_id):
        return user_id in self.claimed_users

    async def mark_reg_bonus_claimed(self, extra_account_id):
        self.bonus_marked = extra_account_id
        return True


@pytest.mark.asyncio
async def test_reg_bonus_not_granted_if_already_claimed_by_user(monkeypatch):
    """Reg-bonus is per-user_id, not per-account — no farming via delete/re-create."""
    game_db = FakeGameDB()
    extra_db = RegBonusPerUserDB()
    extra_db.claimed_users.add(12345)  # user already claimed bonus before
    request = FakeRequest(
        {"email": "new@example.com", "password": "strongpass", "nickname": "NewHero"},
        query={"_auth": "telegram-init-data"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: {"user": "ok"})
    monkeypatch.setattr(extraid_handlers, "_validate_auth_date_fn", lambda data: True)
    monkeypatch.setattr(extraid_handlers, "_extract_user_id_from_init_data_fn", lambda data: 12345)
    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "1234-ABC")
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", lambda *args: None)
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["reg_bonus"] is False
    assert not any("keys = COALESCE(keys, 0) + 3" in query for query, _ in game_db.executed)


class CompensatingCleanupDB(FakeExtraIDDB):
    """Simulates create_extra_account failure to test compensating cleanup."""
    SYNTH_USER = 9_100_000_000_999

    async def get_synthetic_user_id(self):
        return self.SYNTH_USER

    async def create_extra_account(self, user_id, display_id, email, password_hash, nickname=None):
        raise RuntimeError("db_failure_during_create")


@pytest.mark.asyncio
async def test_register_cleans_up_synthetic_user_on_create_failure(monkeypatch):
    """If create_extra_account fails, the orphaned synthetic users row is deleted."""
    game_db = FakeGameDB()
    extra_db = CompensatingCleanupDB()
    request = FakeRequest(
        {"email": "fail@example.com", "password": "strongpass", "nickname": "FailHero"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "9999-FAIL")
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", lambda *args: None)
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    # Should be a 500 server_error due to the RuntimeError re-raise
    assert response.status == 500
    # The synthetic user created by ensure_user should have been cleaned up via delete_user
    assert len(game_db.ensured_users) == 1
    assert game_db.deleted_users == [game_db.ensured_users[0]["user_id"]]


class LinkAlreadyBoundDB(FakeExtraIDDB):
    """ExtraID already bound to a Telegram ID — link re-bind should be refused."""
    TG_USER = 12345
    SYNTH_USER = 9_100_000_000_4242

    async def get_extra_account_by_user_id(self, user_id):
        if user_id == self.SYNTH_USER:
            return {
                "id": self.account_id,
                "user_id": self.TG_USER,  # already re-pointed to Telegram
                "display_id": "4242-APP",
                "email": "player@example.com",
            }
        return None

    async def get_any_extra_account_by_user_id(self, user_id):
        if user_id == self.TG_USER:
            return {
                "id": self.account_id,
                "user_id": self.TG_USER,
                "display_id": "4242-APP",
            }
        return None


@pytest.mark.asyncio
async def test_link_refuses_already_telegram_bound(monkeypatch):
    """Link handler refuses to re-bind an ExtraID already linked to Telegram."""
    game_db = FakeGameDB()
    extra_db = LinkAlreadyBoundDB()
    request = FakeRequest(
        {"tg_init_data": "valid-init-data"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
        headers={"Authorization": "Bearer mobile-jwt"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        return LinkAlreadyBoundDB.SYNTH_USER, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: {"auth_date": "1", "user": "{}"})
    monkeypatch.setattr(extraid_handlers, "_validate_auth_date_fn", lambda data: True)
    monkeypatch.setattr(extraid_handlers, "_extract_user_id_from_init_data_fn", lambda data: 99999)

    response = await extraid_handlers.extraid_link_handler(request)
    body = json.loads(response.text)

    assert response.status == 409
    assert body["error"] == "already_linked_to_telegram"
    assert extra_db.executed == []


# ═══════════════════════════════════════════════════════════════════════════
# P1#2: reg-bonus credit-before-mark (bonus not lost on mark failure)
# P1#3: transfer consume-after-session (code not spent on failure)
# P2#4: link compensating rollback
# ═══════════════════════════════════════════════════════════════════════════


class FailingKeysGameDB(FakeGameDB):
    """Game DB whose execute() raises on the keys-crediting UPDATE."""
    def __init__(self):
        super().__init__()
        self._raise_on_keys = True

    async def execute(self, query, *args):
        if self._raise_on_keys and "keys = COALESCE(keys, 0) + 3" in query:
            raise RuntimeError("keys_credit_failed")
        return await super().execute(query, *args)


@pytest.mark.asyncio
async def test_reg_bonus_not_marked_when_keys_credit_fails_register(monkeypatch):
    """P1#2: if crediting keys fails, mark_reg_bonus_claimed is NOT called (retry possible)."""
    game_db = FailingKeysGameDB()
    extra_db = RegBonusPerUserDB()
    request = FakeRequest(
        {"email": "new@example.com", "password": "strongpass", "nickname": "NewHero"},
        query={"_auth": "telegram-init-data"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: {"user": "ok"})
    monkeypatch.setattr(extraid_handlers, "_validate_auth_date_fn", lambda data: True)
    monkeypatch.setattr(extraid_handlers, "_extract_user_id_from_init_data_fn", lambda data: 12345)
    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "1234-ABC")
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", lambda *args: None)
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.extraid_register_handler(request)

    # The RuntimeError propagates -> 500, but mark_reg_bonus_claimed was never called.
    assert response.status == 500
    assert extra_db.bonus_marked is None


@pytest.mark.asyncio
async def test_reg_bonus_not_marked_when_keys_credit_fails_login(monkeypatch):
    """P1#2: login path — keys credit failure does not mark bonus claimed."""
    game_db = FailingKeysGameDB()

    class LoginDB(LoginExtraIDDB):
        def __init__(self):
            super().__init__()
            self.bonus_marked = None

        async def get_extra_account_by_email(self, email):
            return {
                "id": self.account_id,
                "user_id": 777,
                "display_id": "7777-AAA",
                "password_hash": "hash",
                "reg_bonus_claimed": False,
            }

        async def has_user_claimed_reg_bonus(self, user_id):
            return False

        async def mark_reg_bonus_claimed(self, extra_account_id):
            self.bonus_marked = extra_account_id
            return True

    extra_db = LoginDB()
    request = FakeRequest(
        {"email": "player@example.com", "password": "strongpass"},
        app={"db": game_db, "extraid_db": extra_db},
    )

    monkeypatch.setattr(extraid_handlers.bcrypt, "checkpw", lambda password, password_hash: True)

    response = await extraid_handlers.extraid_login_handler(request)

    assert response.status == 500
    assert extra_db.bonus_marked is None


class TransferFailingSessionDB(TelegramTransferExtraIDDB):
    """Session INSERT fails; consume must not have happened (code still usable)."""
    def __init__(self):
        super().__init__()
        self.consume_calls = 0

    async def execute(self, query, *args):
        if "INSERT INTO auth_sessions" in query:
            raise RuntimeError("session_insert_failed")
        return await super().execute(query, *args)

    async def consume_bot_auth_code(self, code):
        self.consume_calls += 1
        return {"code": code, "user_id": 12345}


@pytest.mark.asyncio
async def test_transfer_code_not_consumed_when_session_insert_fails(monkeypatch):
    """P1#3: if session insert fails, bot code is NOT consumed (retry possible)."""
    game_db = FakeGameDB()
    extra_db = TransferFailingSessionDB()
    request = FakeRequest(
        {
            "telegram_id": "12345",
            "code": "123456",
            "email": "Player@Example.com",
            "password": "strongpass",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "5555-TG")
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.telegram_transfer_complete_handler(request)

    assert response.status == 500
    # consume_bot_auth_code was never called because session insert failed first
    assert extra_db.consume_calls == 0


class LinkFailingExtraAccountsDB(LinkExtraIDDB):
    """UPDATE extra_accounts fails; users update must be rolled back."""
    async def execute(self, query, *args):
        if "UPDATE extra_accounts SET user_id" in query:
            raise RuntimeError("extra_accounts_update_failed")
        return await super().execute(query, *args)


@pytest.mark.asyncio
async def test_link_rolls_back_users_update_on_extra_accounts_failure(monkeypatch):
    """P2#4: if UPDATE extra_accounts fails, the users update is rolled back."""
    game_db = FakeGameDB()
    extra_db = LinkFailingExtraAccountsDB()
    request = FakeRequest(
        {"tg_init_data": "valid-init-data"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
        headers={"Authorization": "Bearer mobile-jwt"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        return LinkExtraIDDB.SYNTHETIC_JWT_USER, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: {"auth_date": "1", "user": "{}"})
    monkeypatch.setattr(extraid_handlers, "_validate_auth_date_fn", lambda data: True)
    monkeypatch.setattr(extraid_handlers, "_extract_user_id_from_init_data_fn", lambda data: 12345)

    response = await extraid_handlers.extraid_link_handler(request)
    body = json.loads(response.text)

    assert response.status == 500
    # users update for tg_user_id happened, then was rolled back
    assert any("extra_account_id = $1" in q and "auth_source = 'telegram'" in q for q, _ in game_db.executed)
    assert any("extra_account_id = NULL" in q and args == (12345,) for q, args in game_db.executed)
