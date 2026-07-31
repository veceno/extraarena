import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrastructure import database as database_module
from infrastructure.database import Database
from infrastructure.config import get_settings
from web import extraid_handlers


class FakeRequest:
    def __init__(self, payload, query=None, app=None, headers=None, method="POST"):
        self._payload = payload
        self.rel_url = SimpleNamespace(query=query or {})
        self.app = app or {"bot_token": "bot-token"}
        self.headers = headers or {}
        self.remote = "127.0.0.1"
        self.method = method

    async def json(self):
        return self._payload


def test_public_email_action_origin_is_explicit_and_https_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "JWT_SECRET",
        "test-extraid-public-origin-secret-that-is-long-enough-2026",
    )
    monkeypatch.setenv(
        "ADMIN_SESSION_SECRET",
        "test-extraid-admin-origin-secret-that-is-long-enough-2026",
    )
    monkeypatch.setenv(
        "MCP_TOKEN_SECRET",
        "test-extraid-mcp-origin-secret-that-is-long-enough-2026",
    )
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://game.example")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("WEBAPP_URL", raising=False)
    get_settings.cache_clear()
    assert extraid_handlers._public_base_url() == ""

    monkeypatch.setenv("PUBLIC_BASE_URL", "http://game.example")
    get_settings.cache_clear()
    assert extraid_handlers._public_base_url() == ""

    monkeypatch.setenv("PUBLIC_BASE_URL", "https://game.example/path?token=bad")
    get_settings.cache_clear()
    assert extraid_handlers._public_base_url() == ""

    monkeypatch.setenv("PUBLIC_BASE_URL", "https://game.example/")
    get_settings.cache_clear()
    assert extraid_handlers._public_base_url() == "https://game.example"

    monkeypatch.setenv("BREVO_API_KEY", "test-key")
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "not-an-email")
    assert extraid_handlers._email_delivery_configured() is False
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "accounts@game.example")
    assert extraid_handlers._email_delivery_configured() is True


@pytest.fixture(autouse=True)
def _local_settings(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.delenv("TELEGRAM_API_INSECURE_SSL", raising=False)
    extraid_handlers._rate_limit_store.clear()
    get_settings.cache_clear()
    yield
    extraid_handlers._rate_limit_store.clear()
    get_settings.cache_clear()


class FakeGameDB:
    def __init__(self):
        self.ensured_users = []
        self.executed = []
        self.changed_nicknames = []
        self.registered_push_devices = []
        self.unregistered_push_devices = []
        self.deleted_users = []
        self.claimed_bonuses = set()

    async def ensure_user(self, **kwargs):
        self.ensured_users.append(kwargs)
        return True

    async def execute(self, query, *args):
        self.executed.append((query, args))
        if "DELETE FROM extra_accounts" in query and args:
            self.deleted_extra_accounts.append(args[0])
        return "UPDATE 1"

    async def change_nickname(self, user_id, nickname, cost_gems=0):
        self.changed_nicknames.append((user_id, nickname, cost_gems))
        return {"success": True}

    async def fetchval(self, query, *args):
        if "FROM users" in query:
            return 1
        return None

    async def fetchrow(self, query, *args):
        self.executed.append((query, args))
        return {"user_id": args[-1] if args else 1}

    async def delete_user(self, user_id):
        self.deleted_users.append(user_id)
        return True

    async def claim_extraid_registration_bonus(self, user_id, *, keys=3):
        if user_id in self.claimed_bonuses:
            return False
        self.claimed_bonuses.add(user_id)
        self.executed.append(("claim_extraid_registration_bonus", (user_id, keys)))
        return True

    async def ensure_extra_account_link(
        self,
        *,
        user_id,
        extra_account_id,
        auth_source=None,
    ):
        self.executed.append(
            ("ensure_extra_account_link", (user_id, extra_account_id, auth_source))
        )
        return "linked"

    async def transfer_extra_account_link(self, *, extra_account_id, old_user_id, new_user_id):
        self.executed.append(
            (
                "transfer_extra_account_link",
                (extra_account_id, old_user_id, new_user_id),
            )
        )
        return "linked"

    async def rollback_extra_account_link(
        self,
        *,
        extra_account_id,
        source_user_id,
        telegram_user_id,
    ):
        self.executed.append(
            (
                "rollback_extra_account_link",
                (extra_account_id, source_user_id, telegram_user_id),
            )
        )
        return "rolled_back"

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
        self.identity_providers = {}

    async def get_synthetic_user_id(self):
        return 9_100_000_000_123

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

    async def create_extra_account(
        self,
        user_id,
        display_id,
        email,
        password_hash,
        nickname=None,
        identity_provider=None,
        identity_subject=None,
        registration_origin="standalone",
    ):
        account = {
            "id": self.account_id,
            "user_id": user_id,
            "display_id": display_id,
            "email": email,
            "password_hash": password_hash,
            "nickname": nickname,
            "reg_bonus_claimed": False,
            "is_email_verified": False,
            "email_verification_required": True,
            "registration_origin": registration_origin,
        }
        self.identity_providers[self.account_id] = {
            identity_provider
            or (
                "telegram"
                if user_id < extraid_handlers.SYNTHETIC_USER_ID_MIN
                else "synthetic_user"
            )
        }
        return account

    async def enqueue_account_email_action(self, email, purpose):
        self.enqueued_email_action = (email, purpose)

    async def create_account_action_token(self, **kwargs):
        self.created_account_action_token = kwargs
        return True

    async def revoke_account_action_token(self, token_id):
        self.revoked_account_action_token = token_id
        return True

    async def claim_account_email_outbox(self, limit=20):
        return []

    async def complete_account_email_outbox(self, outbox_id):
        self.completed_email_outbox = outbox_id

    async def retry_account_email_outbox(self, outbox_id, *, error):
        self.retried_email_outbox = (outbox_id, error)
        return True

    async def mark_reg_bonus_claimed(self, extra_account_id):
        self.bonus_marked = extra_account_id
        return True

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "UPDATE 1"

    async def create_bot_auth_code(self, code, user_id, purpose="telegram_transfer"):
        self.created_bot_code = (code, user_id, purpose)
        return {"created": True}

    async def verify_bot_auth_code(self, code, purpose="telegram_transfer", user_id=None):
        return None

    async def consume_bot_auth_code(self, code, purpose="telegram_transfer", user_id=None):
        return await self.verify_bot_auth_code(code, purpose=purpose, user_id=user_id)

    async def mark_bot_code_used(self, code, session_id, purpose="telegram_transfer", user_id=None):
        self.used_bot_code = (code, session_id)

    async def cleanup_old_bot_codes(self, user_id, purpose=None):
        self.cleaned_bot_codes_for = (user_id, purpose)

    async def invalidate_bot_auth_code(self, code, *, purpose="telegram_transfer", user_id=None):
        self.invalidated_bot_code = (code, purpose, user_id)

    async def revoke_all_user_sessions(self, user_id):
        self.revoked_user_sessions.append(user_id)

    async def soft_delete_extra_account(self, extra_account_id):
        self.deleted_extra_accounts.append(extra_account_id)

    async def begin_account_deletion(self, extra_account_id, *, user_id):
        self.deleted_extra_accounts.append(extra_account_id)
        self.revoked_user_sessions.append(user_id)
        return True

    async def complete_account_deletion(self, extra_account_id):
        self.completed_deletion = extra_account_id

    async def account_has_identity_provider(self, extra_account_id, provider):
        providers = self.identity_providers.get(extra_account_id)
        if providers is not None:
            return provider in providers
        return False

    async def link_extra_account_to_user(self, extra_account_id, old_user_id, new_user_id):
        self.revoked_user_sessions.append(old_user_id)
        self.identity_providers.setdefault(extra_account_id, set()).add("telegram")
        self.executed.append(
            ("UPDATE extra_accounts SET user_id = $1 WHERE id = $2", (new_user_id, extra_account_id))
        )
        return "linked"

    async def rollback_extra_account_link(self, extra_account_id, expected_user_id, restore_user_id):
        self.rolled_back_link = (extra_account_id, expected_user_id, restore_user_id)
        return True

    async def complete_extra_account_link(self, extra_account_id, expected_user_id):
        self.completed_link = (extra_account_id, expected_user_id)
        return True

    async def mark_extra_account_link_reconcile_required(
        self,
        extra_account_id,
        *,
        old_user_id,
        new_user_id,
    ):
        self.reconcile_required = (extra_account_id, old_user_id, new_user_id)

    async def mark_primary_link_reconcile_required(
        self,
        extra_account_id,
        *,
        user_id,
    ):
        self.primary_reconcile_required = (extra_account_id, user_id)

    async def complete_primary_link_reconciliation(
        self,
        extra_account_id,
        *,
        user_id,
    ):
        self.completed_primary_reconciliation = (extra_account_id, user_id)
        return True


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
    monkeypatch.setenv("MCP_TOKEN_SECRET", "test-extraid-mcp-secret-that-is-long-enough-2026")
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://game.example")
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
    assert body["user_id"] == 9_100_000_000_123
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
        {
            "email": "Player@Example.com",
            "password": "strongpass",
            "nickname": "Player_1",
            "tg_init_data": "telegram-init-data",
        },
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
    assert body["reg_bonus"] is False
    assert body["email_verification_required"] is True
    assert "token" not in body
    assert extra_db.bonus_marked is None


@pytest.mark.asyncio
async def test_max_register_creates_immutable_platform_binding(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {
            "email": "MaxPlayer@Example.com",
            "password": "strongpass",
            "nickname": "MaxPlayer",
            "client": "max",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
        headers={"Authorization": "Bearer max-session-jwt"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        assert token == "max-session-jwt"
        return 8_000_000_000_000_123, uuid.uuid4()

    async def max_identity(user_id, provider):
        assert user_id == 8_000_000_000_000_123
        assert provider == "max"
        return {"provider": "max", "subject": "987654321", "user_id": user_id}

    game_db.get_platform_identity_for_user = max_identity
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "MAX-1234")
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert body["linked_telegram"] is False
    assert body["linked_max"] is True
    assert extra_db.identity_providers[extra_db.account_id] == {"max"}
    assert any(
        query == "ensure_extra_account_link" and args[2] == "max"
        for query, args in game_db.executed
    )


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
    assert game_db.ensured_users == []
    assert extra_db.bonus_marked is None
    assert any(
        query == "ensure_extra_account_link" and args[2] == "extraid_mobile"
        for query, args in game_db.executed
    )


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
    assert game_db.ensured_users == []
    assert any(
        query == "ensure_extra_account_link" and args[2] == "extraid_mobile"
        for query, args in game_db.executed
    )


@pytest.mark.asyncio
async def test_register_keeps_pending_account_and_queues_ambiguous_email_failure(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {
            "email": "pending@example.com",
            "password": "strongpass",
            "nickname": "PendingHero",
            "client": "android_app",
        },
        app={
            "db": game_db,
            "extraid_db": extra_db,
            "bot_token": "bot-token",
            "extraid_email_sender": lambda **_kwargs: False,
        },
        headers={"Authorization": "Bearer mobile-jwt"},
    )

    async def verify_jwt(_token, _extra_db, _settings):
        return 5252, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_init_data_fn", lambda auth, token: None)
    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)
    monkeypatch.setattr(extraid_handlers, "_display_id_generator_fn", lambda check: "5252-PND")
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda nick: True)
    monkeypatch.setattr(extraid_handlers.bcrypt, "gensalt", lambda rounds=12: b"salt")
    monkeypatch.setattr(extraid_handlers.bcrypt, "hashpw", lambda password, salt: b"hash")

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert body["email_sent"] is False
    assert extra_db.enqueued_email_action == (
        "pending@example.com",
        "verify_email",
    )
    assert not any(
        "DELETE FROM extra_accounts" in query
        for query, _args in extra_db.executed
    )


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

    assert response.status == 400
    assert body["error"] == "auth_in_url_not_allowed"
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

    async def account_has_identity_provider(self, extra_account_id, provider):
        return False

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
    assert calls[0][0].startswith("extraid_login:client:")
    assert calls[0][1:] == (30, 600)


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
    assert body["reg_bonus"] is False
    assert not any(query == "claim_extraid_registration_bonus" for query, _ in game_db.executed)


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
    assert extra_db.created_bot_code == ("123456", 12345, "telegram_transfer")
    assert bot.messages == [(12345, "🔐 Код для переноса ExtraArena в приложение: 123456\n\nОн действует 5 минут. Никому его не показывай.")]


class TelegramTransferExtraIDDB(FakeExtraIDDB):
    def __init__(self):
        super().__init__()
        self.inserted_session_id = None
        self.used_bot_code = None
        self.created_accounts = []

    async def verify_bot_auth_code(self, code, purpose="telegram_transfer", user_id=None):
        if code == "123456":
            return {"code": code, "user_id": 12345}
        return None

    async def consume_bot_auth_code(self, code, purpose="telegram_transfer", user_id=None):
        return await self.verify_bot_auth_code(code, purpose=purpose, user_id=user_id)

    async def create_extra_account(self, user_id, display_id, email, password_hash, nickname=None, **kwargs):
        account = await super().create_extra_account(user_id, display_id, email, password_hash, nickname, **kwargs)
        self.created_accounts.append(account)
        return account

    async def execute(self, query, *args):
        self.executed.append((query, args))
        if "INSERT INTO auth_sessions" in query:
            self.inserted_session_id = args[0]
        return "INSERT 0 1"


@pytest.mark.asyncio
async def test_telegram_transfer_complete_requires_email_verification_before_session(monkeypatch):
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
    assert "token" not in body
    assert body["reg_bonus"] is False
    assert body["verification_pending"] is True
    assert extra_db.inserted_session_id is None
    assert game_db.ensured_users == []
    assert any("auth_source = 'telegram_transfer'" in query for query, _ in game_db.executed)
    assert not any(query == "claim_extraid_registration_bonus" for query, _ in game_db.executed)


@pytest.mark.asyncio
async def test_telegram_transfer_queues_ambiguous_verification_delivery_failure(
    monkeypatch,
):
    game_db = FakeGameDB()
    extra_db = TelegramTransferExtraIDDB()
    request = FakeRequest(
        {
            "telegram_id": "12345",
            "code": "123456",
            "email": "PendingTransfer@Example.com",
            "password": "strongpass",
        },
        app={
            "db": game_db,
            "extraid_db": extra_db,
            "bot_token": "bot-token",
            "extraid_email_sender": lambda **_kwargs: False,
        },
    )

    monkeypatch.setattr(
        extraid_handlers,
        "_display_id_generator_fn",
        lambda check: "5555-OUTBOX",
    )
    monkeypatch.setattr(
        extraid_handlers.bcrypt,
        "gensalt",
        lambda rounds=12: b"salt",
    )
    monkeypatch.setattr(
        extraid_handlers.bcrypt,
        "hashpw",
        lambda password, salt: b"hash",
    )

    response = await extraid_handlers.telegram_transfer_complete_handler(
        request
    )
    body = json.loads(response.text)

    assert response.status == 200
    assert body["ok"] is True
    assert body["email_sent"] is False
    assert body["verification_pending"] is True
    assert extra_db.enqueued_email_action == (
        "pendingtransfer@example.com",
        "verify_email",
    )


class TelegramTransferEmailTakenDB(TelegramTransferExtraIDDB):
    def __init__(self):
        super().__init__()
        self.consumed_codes = []

    async def get_any_extra_account_by_email(self, email):
        # Active (non-deleted) account with this email — should block registration.
        return {"id": uuid.uuid4(), "email": email, "deleted_at": None}

    async def consume_bot_auth_code(self, code, purpose="telegram_transfer", user_id=None):
        self.consumed_codes.append(code)
        return await super().consume_bot_auth_code(code, purpose=purpose, user_id=user_id)


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
    async def verify_bot_auth_code(self, code, purpose="telegram_transfer", user_id=None):
        return {"code": code, "user_id": 12345}

    async def consume_bot_auth_code(self, code, purpose="telegram_transfer", user_id=None):
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

    # The final atomic consume loses the race, so the just-created account is
    # hard-rolled back and no login session is issued.
    assert response.status == 400
    assert body["error"] == "invalid_code"
    # Account was created then hard-rolled back.
    assert any(
        "DELETE FROM extra_accounts" in query and args == (extra_db.account_id,)
        for query, args in extra_db.executed
    )
    assert any("extra_account_id = NULL" in query and "auth_source = 'telegram'" in query for query, _ in game_db.executed)


@pytest.mark.asyncio
async def test_new_account_rollback_clears_primary_pointer_before_credentials():
    events = []

    class Primary:
        async def execute(self, query, *args):
            events.append(("primary", query, args))
            return "UPDATE 1"

    class Credentials:
        async def execute(self, query, *args):
            events.append(("credentials", query, args))
            return "DELETE 1"

    account_id = uuid.uuid4()
    await extraid_handlers._rollback_created_extra_account(
        Primary(),
        Credentials(),
        extra_account_id=account_id,
        user_id=4242,
        created_synthetic=False,
    )

    assert [event[0] for event in events] == ["primary", "credentials"]
    assert "extra_account_id = $2" in events[0][1]
    assert events[0][2] == (4242, account_id)


@pytest.mark.asyncio
async def test_telegram_transfer_rollback_is_cas_and_primary_first():
    events = []

    class Primary:
        async def execute(self, query, *args):
            events.append(("primary", query, args))
            return "UPDATE 1"

    class Credentials:
        async def execute(self, query, *args):
            events.append(("credentials", query, args))
            return "DELETE 1"

    account_id = uuid.uuid4()
    await extraid_handlers._rollback_telegram_transfer_account(
        Primary(),
        Credentials(),
        extra_account_id=account_id,
        telegram_id=12345,
    )

    assert [event[0] for event in events] == ["primary", "credentials"]
    assert "WHERE user_id = $1 AND extra_account_id = $2" in events[0][1]
    assert events[0][2] == (12345, account_id)


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
async def test_extraid_link_is_support_only_before_auth_processing(monkeypatch):
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

    assert response.status == 410
    assert body["error"] == "support_required"
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

    assert response.status == 410
    assert body["error"] == "support_required"
    assert extra_db.executed == []


@pytest.mark.asyncio
async def test_extraid_link_support_only_performs_no_mutations(monkeypatch):
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

    assert response.status == 410
    assert body["error"] == "support_required"
    assert extra_db.executed == []
    assert game_db.executed == []
    assert extra_db.revoked_user_sessions == []


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

    def __init__(self):
        super().__init__()
        self.identity_providers[self.account_id] = {"telegram"}

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

    def __init__(self):
        super().__init__()
        self.identity_providers[self.account_id] = {"synthetic_user"}

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


class MaxBoundDeleteDB(FakeExtraIDDB):
    MAX_USER = 8_000_000_000_000_456

    def __init__(self):
        super().__init__()
        self.identity_providers[self.account_id] = {"max"}

    async def get_extra_account_by_user_id(self, user_id):
        if user_id == self.MAX_USER:
            return {
                "id": self.account_id,
                "user_id": self.MAX_USER,
                "display_id": "MAX-BOUND",
                "email": "max@example.com",
                "password_hash": "hash",
                "reg_bonus_claimed": True,
            }
        return None


@pytest.mark.asyncio
async def test_delete_refused_for_max_bound_account(monkeypatch):
    game_db = FakeGameDB()
    extra_db = MaxBoundDeleteDB()
    request = FakeRequest(
        {"password": "strongpass", "confirm": "DELETE"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    monkeypatch.setattr(
        extraid_handlers,
        "require_user_id_fn",
        async_lambda(MaxBoundDeleteDB.MAX_USER),
    )

    response = await extraid_handlers.extraid_delete_account_handler(request)
    body = json.loads(response.text)

    assert response.status == 403
    assert body["error"] == "cannot_delete_max_bound"
    assert extra_db.deleted_extra_accounts == []
    assert extra_db.revoked_user_sessions == []


@pytest.mark.asyncio
async def test_logout_refused_for_max_launch_session(monkeypatch):
    session_id = uuid.uuid4()

    class MaxSessionDB:
        def __init__(self):
            self.revoked = []

        async def fetchrow(self, query, *args):
            assert "auth_method" in query
            assert args == (session_id,)
            return {"auth_method": "max"}

        async def revoke_session(self, target):
            self.revoked.append(target)

    extra_db = MaxSessionDB()
    request = FakeRequest(
        {},
        app={"extraid_db": extra_db},
        headers={"Authorization": "Bearer max-session-jwt"},
    )

    async def verify_jwt(token, _extra_db, _settings):
        assert token == "max-session-jwt"
        return MaxBoundDeleteDB.MAX_USER, str(session_id)

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)

    response = await extraid_handlers.auth_logout_handler(request)
    body = json.loads(response.text)

    assert response.status == 403
    assert body["error"] == "cannot_logout_max_bound"
    assert extra_db.revoked == []


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

    async def verify_bot_auth_code(
        self,
        code,
        purpose="telegram_transfer",
        user_id=None,
    ):
        if code == "123456":
            return {"code": code, "user_id": self.TG_USER}
        return None

    async def consume_bot_auth_code(
        self,
        code,
        purpose="telegram_transfer",
        user_id=None,
    ):
        return await self.verify_bot_auth_code(
            code,
            purpose=purpose,
            user_id=user_id,
        )


@pytest.mark.asyncio
async def test_register_blocks_duplicate_after_delete(monkeypatch):
    """After delete, same Telegram ID cannot create a new ExtraID."""
    game_db = FakeGameDB()
    extra_db = DuplicationBlockDB()
    request = FakeRequest(
        {
            "email": "new@example.com",
            "password": "strongpass",
            "nickname": "NewHero",
            "tg_init_data": "telegram-init-data",
        },
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
async def test_transfer_request_code_hides_duplicate_after_delete(monkeypatch):
    """Duplicate and unknown Telegram IDs share the same anti-enumeration response."""
    game_db = FakeGameDB()
    extra_db = DuplicationBlockDB()
    request = FakeRequest(
        {"telegram_id": "12345"},
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )

    response = await extraid_handlers.telegram_transfer_request_code_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body == {"ok": True, "status": "code_sent", "ttl_seconds": 300}


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
        {
            "email": "new@example.com",
            "password": "strongpass",
            "nickname": "NewHero",
            "tg_init_data": "telegram-init-data",
        },
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

    async def create_extra_account(
        self,
        user_id,
        display_id,
        email,
        password_hash,
        nickname=None,
        **kwargs,
    ):
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

    assert response.status == 410
    assert body["error"] == "support_required"
    assert extra_db.executed == []


# ═══════════════════════════════════════════════════════════════════════════
# P1#2: reg-bonus credit-before-mark (bonus not lost on mark failure)
# P1#3: transfer consume-after-session (code not spent on failure)
# P2#4: link compensating rollback
# ═══════════════════════════════════════════════════════════════════════════


class FailingKeysGameDB(FakeGameDB):
    """Game DB whose authoritative bonus-ledger transaction fails."""
    def __init__(self):
        super().__init__()
        self._raise_on_keys = True

    async def execute(self, query, *args):
        if self._raise_on_keys and "keys = COALESCE(keys, 0) + 3" in query:
            raise RuntimeError("keys_credit_failed")
        return await super().execute(query, *args)

    async def claim_extraid_registration_bonus(self, user_id, *, keys=3):
        raise RuntimeError("keys_credit_failed")


@pytest.mark.asyncio
async def test_reg_bonus_not_marked_when_keys_credit_fails_register(monkeypatch):
    """Registration never grants the bonus before email verification."""
    game_db = FailingKeysGameDB()
    extra_db = RegBonusPerUserDB()
    request = FakeRequest(
        {
            "email": "new@example.com",
            "password": "strongpass",
            "nickname": "NewHero",
            "tg_init_data": "telegram-init-data",
        },
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

    assert response.status == 200
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

        async def account_has_identity_provider(self, extra_account_id, provider):
            return provider == "telegram"

    extra_db = LoginDB()
    request = FakeRequest(
        {"email": "player@example.com", "password": "strongpass"},
        app={"db": game_db, "extraid_db": extra_db},
    )

    monkeypatch.setattr(extraid_handlers.bcrypt, "checkpw", lambda password, password_hash: True)

    response = await extraid_handlers.extraid_login_handler(request)

    assert response.status == 200
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
async def test_link_failure_backend_is_not_reached_for_support_only_route(monkeypatch):
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

    assert response.status == 410
    assert body["error"] == "support_required"
    assert game_db.executed == []
    assert extra_db.executed == []


@pytest.mark.asyncio
async def test_production_register_requires_game_session_before_side_effects(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "JWT_SECRET",
        "test-extraid-register-secret-that-is-long-enough-2026",
    )
    monkeypatch.setenv(
        "ADMIN_SESSION_SECRET",
        "test-extraid-register-admin-that-is-long-enough-2026",
    )
    monkeypatch.setenv(
        "MCP_TOKEN_SECRET",
        "test-extraid-register-mcp-that-is-long-enough-2026",
    )
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://game.example")
    get_settings.cache_clear()
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {
            "email": "new@example.com",
            "password": "strongpass",
            "nickname": "NewHero",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda value: True)
    monkeypatch.setattr(
        extraid_handlers,
        "_check_rate_limit_for_request",
        async_lambda(True),
    )

    async def forbidden_email(*args, **kwargs):
        raise AssertionError("email must not be sent")

    monkeypatch.setattr(extraid_handlers, "_send_extraid_email", forbidden_email)

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 401
    assert body["error"] == "game_session_required"
    assert game_db.ensured_users == []
    assert game_db.executed == []
    assert extra_db.executed == []


@pytest.mark.asyncio
async def test_production_register_requires_email_delivery_before_side_effects(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "JWT_SECRET",
        "test-extraid-register-secret-that-is-long-enough-2026",
    )
    monkeypatch.setenv(
        "ADMIN_SESSION_SECRET",
        "test-extraid-register-admin-that-is-long-enough-2026",
    )
    monkeypatch.setenv(
        "MCP_TOKEN_SECRET",
        "test-extraid-register-mcp-that-is-long-enough-2026",
    )
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://game.example")
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("EMAIL_FROM_ADDRESS", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    get_settings.cache_clear()
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {
            "email": "new@example.com",
            "password": "strongpass",
            "nickname": "NewHero",
            "client": "android_app",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
        headers={"Authorization": "Bearer valid-mobile-jwt"},
    )
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda value: True)
    monkeypatch.setattr(
        extraid_handlers,
        "_check_rate_limit_for_request",
        async_lambda(True),
    )

    async def verify_jwt(_token, _extra_db, _settings):
        return 6262, uuid.uuid4()

    monkeypatch.setattr(extraid_handlers, "_verify_jwt_token_async_fn", verify_jwt)

    response = await extraid_handlers.extraid_register_handler(request)
    body = json.loads(response.text)

    assert response.status == 503
    assert body["error"] == "email_delivery_unavailable"
    assert game_db.ensured_users == []
    assert game_db.executed == []
    assert extra_db.executed == []
    assert extra_db.identity_providers == {}


@pytest.mark.asyncio
async def test_register_rejects_password_over_bcrypt_utf8_limit(monkeypatch):
    game_db = FakeGameDB()
    extra_db = FakeExtraIDDB()
    request = FakeRequest(
        {
            "email": "new@example.com",
            "password": "😀" * 19,
            "nickname": "NewHero",
        },
        app={"db": game_db, "extraid_db": extra_db, "bot_token": "bot-token"},
    )
    monkeypatch.setattr(extraid_handlers, "_nickname_valid_fn", lambda value: True)

    response = await extraid_handlers.extraid_register_handler(request)

    assert response.status == 400
    assert json.loads(response.text)["error"] == "password_too_long"
    assert game_db.ensured_users == []


@pytest.mark.asyncio
async def test_login_uses_one_dummy_bcrypt_check_for_unknown_email(monkeypatch):
    class MissingLoginDB:
        async def get_extra_account_by_email(self, email):
            return None

    checked = []

    def checkpw(password, password_hash):
        checked.append((password, password_hash))
        return False

    request = FakeRequest(
        {"email": "missing@example.com", "password": "strongpass"},
        app={"db": FakeGameDB(), "extraid_db": MissingLoginDB()},
    )
    monkeypatch.setattr(extraid_handlers.bcrypt, "checkpw", checkpw)

    response = await extraid_handlers.extraid_login_handler(request)

    assert response.status == 401
    assert json.loads(response.text)["error"] == "invalid_credentials"
    assert checked == [(b"strongpass", extraid_handlers._DUMMY_PASSWORD_HASH)]


def test_rate_limit_subjects_are_keyed_and_proxy_headers_are_trusted_selectively(
    monkeypatch,
):
    email = "player@example.com"
    subject = extraid_handlers._rate_limit_subject("email", email)

    assert email not in subject
    assert subject != extraid_handlers._rate_limit_subject("telegram", email)

    direct = FakeRequest({}, headers={"X-Forwarded-For": "198.51.100.7"})
    direct.remote = "203.0.113.9"
    assert extraid_handlers._trusted_client_ip(direct) == "203.0.113.9"

    proxied = FakeRequest({}, headers={"X-Forwarded-For": "198.51.100.7"})
    proxied.remote = "127.0.0.1"
    assert extraid_handlers._trusted_client_ip(proxied) == "198.51.100.7"

    monkeypatch.setenv("TRUSTED_PROXY_CIDRS", "10.0.0.0/8")
    chained = FakeRequest(
        {},
        headers={"X-Forwarded-For": "192.0.2.66, 198.51.100.7, 10.20.30.40"},
    )
    chained.remote = "127.0.0.1"
    assert extraid_handlers._trusted_client_ip(chained) == "198.51.100.7"


@pytest.mark.asyncio
async def test_ensure_extra_account_link_maps_unique_owner_collision(monkeypatch):
    expected_collision = RuntimeError("legacy cross-owner pointer")
    unexpected_failure = RuntimeError("database unavailable")
    db = Database.__new__(Database)

    async def collide(*args, **kwargs):
        raise expected_collision

    db.fetchrow = collide
    monkeypatch.setattr(
        database_module,
        "_is_unique_violation",
        lambda exc: exc is expected_collision,
    )

    result = await db.ensure_extra_account_link(
        user_id=17,
        extra_account_id=uuid.uuid4(),
        auth_source="extraid_mobile",
    )
    assert result == "owner_conflict"

    async def fail(*args, **kwargs):
        raise unexpected_failure

    db.fetchrow = fail
    with pytest.raises(RuntimeError, match="database unavailable"):
        await db.ensure_extra_account_link(
            user_id=17,
            extra_account_id=uuid.uuid4(),
            auth_source="extraid_mobile",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "purpose"),
    (
        (extraid_handlers.extraid_email_resend_handler, "verify_email"),
        (extraid_handlers.extraid_password_reset_request_handler, "password_reset"),
    ),
)
async def test_generic_email_request_queues_without_waiting_for_sender(
    monkeypatch,
    handler,
    purpose,
):
    class QueueDB(FakeExtraIDDB):
        def __init__(self):
            super().__init__()
            self.queued = []

        async def enqueue_account_email_action(self, email, queued_purpose):
            self.queued.append((email, queued_purpose))

    sender_calls = []

    async def sender(**kwargs):
        sender_calls.append(kwargs)
        return True

    extra_db = QueueDB()
    request = FakeRequest(
        {"email": "Player@Example.com"},
        app={"extraid_db": extra_db, "extraid_email_sender": sender},
    )

    response = await handler(request)

    assert response.status == 200
    assert json.loads(response.text) == {
        "ok": True,
        "status": "if_account_exists_email_sent",
    }
    assert extra_db.queued == [("player@example.com", purpose)]
    assert sender_calls == []


@pytest.mark.asyncio
async def test_email_outbox_worker_issues_hash_only_and_sends_branded_code_email(
    monkeypatch,
):
    class WorkerDB(FakeExtraIDDB):
        def __init__(self):
            super().__init__()
            self.outbox_id = uuid.uuid4()
            self.action_tokens = []
            self.completed = []

        async def claim_account_email_outbox(self, limit=20):
            return [
                {
                    "outbox_id": self.outbox_id,
                    "extra_account_id": self.account_id,
                    "purpose": "verify_email",
                    "email_snapshot": "player@example.com",
                    "attempts": 1,
                    "id": self.account_id,
                    "user_id": 12345,
                    "email": "player@example.com",
                    "is_email_verified": False,
                    "email_verification_required": True,
                    "deleted_at": None,
                    "reg_bonus_claimed": False,
                }
            ]

        async def create_account_action_token(self, **kwargs):
            self.action_tokens.append(kwargs)
            return True

        async def revoke_account_action_token(self, token_id):
            raise AssertionError("successful delivery must not revoke tokens")

        async def complete_account_email_outbox(self, outbox_id):
            self.completed.append(outbox_id)

    sent = []

    async def sender(**kwargs):
        sent.append(kwargs)
        return True

    extra_db = WorkerDB()
    verification_token_id = uuid.uuid4()
    monkeypatch.setattr(
        extraid_handlers,
        "_make_email_verification_code",
        lambda _email: (verification_token_id, "123456", "a" * 64),
    )
    app = {
        "extraid_db": extra_db,
        "extraid_email_sender": sender,
    }

    processed = await extraid_handlers._run_extraid_email_outbox_once(app)

    assert processed == 1
    assert extra_db.completed == [extra_db.outbox_id]
    assert len(extra_db.action_tokens) == 2
    assert {row["purpose"] for row in extra_db.action_tokens} == {
        "verify_email",
        "cancel_registration",
    }
    assert all(len(row["token_hash"]) == 64 for row in extra_db.action_tokens)
    assert all("token" not in row for row in extra_db.action_tokens)
    assert "Код подтверждения: 123456" in sent[0]["text"]
    assert "#extraid_verify_token=" not in sent[0]["text"]
    assert "#extraid_cancel_token=" in sent[0]["text"]
    assert "?extraid_" not in sent[0]["text"]
    assert "<!doctype html>" in sent[0]["html"]
    assert "ExtraID Security" in sent[0]["html"]
    assert "123456" in sent[0]["html"]
    assert "background:#f5921e" in sent[0]["html"]


@pytest.mark.asyncio
async def test_email_verification_accepts_six_digit_code_and_links_account():
    game_db = FakeGameDB()

    class VerificationCodeDB(FakeExtraIDDB):
        def __init__(self):
            super().__init__()
            self.consumed_code = None

        async def consume_email_verification_code(self, *, email, code):
            self.consumed_code = (email, code)
            return {
                "extra_account_id": self.account_id,
                "user_id": 9_100_000_000_123,
                "reg_bonus_claimed": False,
            }

    extra_db = VerificationCodeDB()
    request = FakeRequest(
        {"email": "Player@Example.com", "code": "123456"},
        app={"db": game_db, "extraid_db": extra_db},
    )

    response = await extraid_handlers.extraid_email_verify_handler(request)
    body = json.loads(response.text)

    assert response.status == 200
    assert body == {"ok": True, "email_verified": True, "reg_bonus": False}
    assert extra_db.consumed_code == (
        "player@example.com",
        "123456",
    )
    assert (
        "ensure_extra_account_link",
        (9_100_000_000_123, extra_db.account_id, None),
    ) in game_db.executed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"email": "player@example.com", "code": "12345"},
        {"email": "player@example.com", "code": "12345a"},
        {"email": "not-an-email", "code": "123456"},
        {"email": "", "code": "123456"},
    ],
)
async def test_email_verification_rejects_malformed_code_without_database_lookup(
    payload,
):
    class NoLookupDB(FakeExtraIDDB):
        async def consume_email_verification_code(self, **_kwargs):
            raise AssertionError("malformed verification input must not reach DB")

    request = FakeRequest(
        payload,
        app={"db": FakeGameDB(), "extraid_db": NoLookupDB()},
    )

    response = await extraid_handlers.extraid_email_verify_handler(request)

    assert response.status == 400
    assert json.loads(response.text)["error"] == "invalid_or_expired_code"


@pytest.mark.asyncio
async def test_email_verification_rejects_code_and_email_in_query():
    request = FakeRequest(
        {},
        query={"code": "123456", "email": "player@example.com"},
        app={"db": FakeGameDB(), "extraid_db": FakeExtraIDDB()},
    )

    response = await extraid_handlers.extraid_email_verify_handler(request)

    assert response.status == 400
    assert json.loads(response.text)["error"] == "code_in_url_not_allowed"


def test_email_verification_code_is_six_digits_and_keyed():
    _token_id, code, code_hash = extraid_handlers._make_email_verification_code(
        "player@example.com"
    )

    assert len(code) == 6
    assert code.isdigit()
    assert len(code_hash) == 64
    assert code not in code_hash
    assert code_hash != extraid_handlers._account_email_code_hash(
        code,
        "other@example.com",
        _token_id,
    )


@pytest.mark.asyncio
async def test_reconciliation_rolls_back_legacy_link_and_finishes_pending_work():
    registration_id = uuid.uuid4()
    deletion_id = uuid.uuid4()
    link_id = uuid.uuid4()
    old_user_id = 9_100_000_000_999
    new_user_id = 12345

    class ReconcileGameDB(FakeGameDB):
        async def fetchval(self, query, *args):
            user_id = int(args[0])
            if user_id == new_user_id:
                return link_id
            if user_id == old_user_id:
                return None
            return None

    class ReconcileExtraDB(FakeExtraIDDB):
        def __init__(self):
            super().__init__()
            self.finalized_registrations = []
            self.completed_deletions = []

        async def expire_pending_registrations(self, limit):
            return [
                {
                    "id": registration_id,
                    "user_id": 9_100_000_001_001,
                    "registration_origin": "standalone",
                }
            ]

        async def get_pending_registration_cleanups(self, limit):
            return []

        async def finalize_pending_registration_cleanup(self, account_id):
            self.finalized_registrations.append(account_id)
            return True

        async def get_pending_account_deletions(self, limit):
            return [{"id": deletion_id, "user_id": 9_100_000_001_002}]

        async def complete_account_deletion(self, account_id):
            self.completed_deletions.append(account_id)

        async def get_pending_identity_reconciliations(self, limit):
            return [
                {
                    "id": link_id,
                    "user_id": new_user_id,
                    "link_previous_user_id": old_user_id,
                    "link_state": "pending_primary",
                }
            ]

    game_db = ReconcileGameDB()
    extra_db = ReconcileExtraDB()
    stats = await extraid_handlers._run_extraid_reconciliation_once(
        {"db": game_db, "extraid_db": extra_db}
    )

    assert stats["registration_cleanups"] == 1
    assert stats["account_deletions"] == 1
    assert stats["links_rolled_back"] == 1
    assert stats["links_completed"] == 0
    assert extra_db.rolled_back_link == (link_id, new_user_id, old_user_id)
    assert (
        "rollback_extra_account_link",
        (link_id, old_user_id, new_user_id),
    ) in game_db.executed


def test_identity_migrations_fail_closed_before_unique_index_changes():
    extraid_source = Path("infrastructure/extraid_database.py").read_text(
        encoding="utf-8"
    )
    primary_source = Path("infrastructure/database.py").read_text(encoding="utf-8")

    assert extraid_source.index("duplicate_active_extraid_user") < extraid_source.index(
        "idx_extra_accounts_active_user_id_unique"
    )
    assert extraid_source.index("duplicate_active_extraid_email") < extraid_source.index(
        "_drop_legacy_email_unique_constraints()"
    )
    assert primary_source.index("duplicate_primary_extra_account_link") < primary_source.index(
        "idx_users_extra_account_id_unique"
    )
    assert "ON CONFLICT (extra_account_id" not in primary_source


@pytest.mark.asyncio
async def test_email_outbox_retry_budget_is_24_attempts(monkeypatch):
    from infrastructure.extraid_database import (
        EMAIL_OUTBOX_MAX_ATTEMPTS,
        ExtraIDDatabase,
    )

    database = ExtraIDDatabase("postgresql://unused")
    calls = []

    async def execute(query, *args):
        calls.append((query, args))
        return "UPDATE 1"

    monkeypatch.setattr(database, "execute", execute)
    kept = await database.retry_account_email_outbox(
        uuid.uuid4(),
        error="temporary_provider_failure",
    )

    assert kept is True
    assert EMAIL_OUTBOX_MAX_ATTEMPTS == 24
    assert calls[0][1][-1] == 24
    assert "LEAST(" in calls[0][0] and "3600" in calls[0][0]
