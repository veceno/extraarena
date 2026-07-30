import json
from pathlib import Path

import pytest

from ai.bot_factory import BotGenerator
from infrastructure import config as config_module
from infrastructure.config import get_settings
from infrastructure.database import Database
from web import server as web_server


STRONG_TEST_JWT_SECRET = "test-runtime-jwt-secret-that-is-long-enough-2026"
STRONG_TEST_ADMIN_SECRET = "test-runtime-admin-secret-that-is-long-enough-2026"
STRONG_TEST_MCP_SECRET = "test-runtime-mcp-secret-that-is-long-enough-2026"


class RuntimeDBHarness(Database):
    def __init__(self, settings=None):
        self.settings = dict(settings or {})

    async def fetch(self, _query, keys):
        return [{"key": key, "value": self.settings[key]} for key in keys if key in self.settings]

    async def fetchval(self, _query, key):
        return self.settings.get(key)

    async def execute(self, _query, key, value, _description=""):
        self.settings[key] = json.loads(value) if isinstance(value, str) else value


@pytest.mark.asyncio
async def test_runtime_config_defaults_and_normalization():
    db = RuntimeDBHarness({
        "feature_availability": {"shop": False, "unknown": False},
        "maintenance_mode": {"enabled": 1},
        "disabled_card_ids": ["2", "bad", 2, 5],
    })

    config = await db.get_runtime_config()

    assert config["maintenance_mode"] == {"enabled": True}
    assert config["feature_availability"]["shop"] is False
    assert config["feature_availability"]["collection"] is True
    assert config["feature_availability"]["rating"] is True
    assert config["disabled_card_ids"] == [2, 5]


@pytest.mark.asyncio
async def test_runtime_config_set_preserves_defaults():
    db = RuntimeDBHarness()

    config = await db.set_runtime_config(
        maintenance_mode={"enabled": True},
        feature_availability={"training": False},
        disabled_card_ids=["7", "7", None, 8],
    )

    assert config["maintenance_mode"]["enabled"] is True
    assert config["feature_availability"]["training"] is False
    assert config["feature_availability"]["classic"] is True
    assert config["feature_availability"]["rating"] is True
    assert config["disabled_card_ids"] == [7, 8]


def test_production_rejects_wildcard_cors_origins(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_TEST_ADMIN_SECRET)
    monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_TEST_MCP_SECRET)
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://admin.example")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="CORS_ALLOWED_ORIGINS"):
        get_settings()

    get_settings.cache_clear()


def test_production_rejects_payment_webhook_diagnostics_flag(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_TEST_ADMIN_SECRET)
    monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_TEST_MCP_SECRET)
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://admin.example")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    monkeypatch.setenv("PAYMENT_WEBHOOK_DIAGNOSTICS_ENABLED", "true")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="PAYMENT_WEBHOOK_DIAGNOSTICS_ENABLED"):
        get_settings()

    get_settings.cache_clear()


def test_payment_test_modes_are_not_default_enabled_in_production(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_TEST_ADMIN_SECRET)
    monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_TEST_MCP_SECRET)
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://admin.example")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    monkeypatch.setenv("YOOKASSA_SHOP_ID", "shop-id")
    monkeypatch.setenv("YOOKASSA_SECRET_KEY", "secret")
    monkeypatch.setenv("ROBOKASSA_MERCHANT_LOGIN", "merchant")
    monkeypatch.setenv("ROBOKASSA_PASSWORD1", "password-1")
    monkeypatch.setenv("ROBOKASSA_PASSWORD2", "password-2")
    monkeypatch.delenv("YOOKASSA_TEST_MODE", raising=False)
    monkeypatch.delenv("ROBOKASSA_TEST_MODE", raising=False)
    monkeypatch.delenv("STARS_TEST_MODE", raising=False)
    monkeypatch.delenv("PAYMENT_PROVIDER_ORDER", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.yookassa is not None
    assert settings.yookassa.test_mode is False
    assert settings.robokassa is not None
    assert settings.robokassa.test_mode is False
    assert settings.stars_test_mode is False
    assert settings.payment_provider_order.split(",")[0] == "robokassa"

    get_settings.cache_clear()


def test_production_rejects_explicit_payment_test_modes_without_override(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_TEST_ADMIN_SECRET)
    monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_TEST_MCP_SECRET)
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://admin.example")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    monkeypatch.setenv("YOOKASSA_SHOP_ID", "shop-id")
    monkeypatch.setenv("YOOKASSA_SECRET_KEY", "secret")
    monkeypatch.setenv("ROBOKASSA_MERCHANT_LOGIN", "merchant")
    monkeypatch.setenv("ROBOKASSA_PASSWORD1", "password-1")
    monkeypatch.setenv("ROBOKASSA_PASSWORD2", "password-2")
    monkeypatch.setenv("YOOKASSA_TEST_MODE", "true")
    monkeypatch.setenv("ROBOKASSA_TEST_MODE", "true")
    monkeypatch.delenv("ALLOW_PAYMENT_TEST_MODE", raising=False)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="Payment test/sandbox modes"):
        get_settings()

    get_settings.cache_clear()


def test_env_example_does_not_enable_real_money_test_modes_by_copy_paste():
    env_example = Path(".env.example").read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "ROBOKASSA_TEST_MODE=true" not in active_lines
    assert "YOOKASSA_TEST_MODE=true" not in active_lines
    assert "ROBOKASSA_TEST_MODE=false" in active_lines
    assert "YOOKASSA_TEST_MODE=false" in active_lines
    assert "PAYMENT_PROVIDER_ORDER=robokassa,yookassa,rustore,stars" in active_lines


def test_development_keeps_ruble_payment_test_modes_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("YOOKASSA_SHOP_ID", "shop-id")
    monkeypatch.setenv("YOOKASSA_SECRET_KEY", "secret")
    monkeypatch.setenv("ROBOKASSA_MERCHANT_LOGIN", "merchant")
    monkeypatch.setenv("ROBOKASSA_PASSWORD1", "password-1")
    monkeypatch.setenv("ROBOKASSA_PASSWORD2", "password-2")
    monkeypatch.delenv("YOOKASSA_TEST_MODE", raising=False)
    monkeypatch.delenv("ROBOKASSA_TEST_MODE", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.yookassa is not None
    assert settings.yookassa.test_mode is True
    assert settings.robokassa is not None
    assert settings.robokassa.test_mode is True

    get_settings.cache_clear()


@pytest.mark.parametrize("worker_env_var", ["WEB_CONCURRENCY", "GUNICORN_WORKERS", "UVICORN_WORKERS"])
def test_memory_match_state_rejects_multiple_web_workers(monkeypatch, tmp_path, worker_env_var):
    monkeypatch.setattr(config_module, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("MATCH_STATE_BACKEND", "memory")
    monkeypatch.setenv(worker_env_var, "2")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="MATCH_STATE_BACKEND=memory"):
        get_settings()

    get_settings.cache_clear()


def test_production_socketio_cors_uses_allowlist_not_wildcard(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_TEST_ADMIN_SECRET)
    monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_TEST_MCP_SECRET)
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://admin.example")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    get_settings.cache_clear()

    settings = get_settings()
    web_server._configure_socketio_cors(settings)

    assert web_server.sio.eio.cors_allowed_origins == ["https://game.example"]

    get_settings.cache_clear()


def test_database_url_populates_main_database_settings(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("DATABASE_URL", "postgresql://beta_user:p%40ssword@db.example.com:15432/extraarena_beta")
    monkeypatch.setenv("DB_HOST", "wrong-host.example")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.database.host == "db.example.com"
    assert settings.database.port == 15432
    assert settings.database.user == "beta_user"
    assert settings.database.password == "p@ssword"
    assert settings.database.database == "extraarena_beta"

    get_settings.cache_clear()


def test_extraid_database_url_populates_extraid_database_settings(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("EXTRAID_DATABASE_URL", "postgresql://extra_user:extra_pass@extraid-db.example.com/extraid_beta")
    monkeypatch.setenv("EXTRAID_DB_HOST", "wrong-extraid-host.example")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.extraid_database.host == "extraid-db.example.com"
    assert settings.extraid_database.port == 5434
    assert settings.extraid_database.user == "extra_user"
    assert settings.extraid_database.password == "extra_pass"
    assert settings.extraid_database.database == "extraid_beta"

    get_settings.cache_clear()


def test_socketio_server_does_not_initialize_with_wildcard_cors():
    source = Path("web/server.py").read_text(encoding="utf-8")
    socketio_block = source.split("sio = socketio.AsyncServer(", 1)[1].split(")", 1)[0]

    assert "cors_allowed_origins='*'" not in socketio_block
    assert 'cors_allowed_origins="*"' not in socketio_block


def test_development_allows_permissive_local_cors(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("WEBAPP_ALLOWED_ORIGINS", raising=False)
    get_settings.cache_clear()

    settings = get_settings()
    web_server._configure_socketio_cors(settings)

    assert settings.cors_allowed_origins == ("*",)
    assert web_server.sio.eio.cors_allowed_origins == "*"

    get_settings.cache_clear()


class BotDBHarness:
    def __init__(self, disabled):
        self.disabled = disabled

    async def get_disabled_card_ids(self):
        return list(self.disabled)

    async def get_cards_list(self):
        return [
            {"id": 1, "name": "Hero A", "card_type": "hero"},
            {"id": 2, "name": "Hero B", "card_type": "hero"},
            {"id": 3, "name": "Unit A", "card_type": "unit"},
            {"id": 4, "name": "Unit B", "card_type": "unit"},
            {"id": 5, "name": "Unit C", "card_type": "unit"},
            {"id": 6, "name": "Unit D", "card_type": "unit"},
            {"id": 7, "name": "Unit E", "card_type": "unit"},
            {"id": 8, "name": "Unit F", "card_type": "unit"},
            {"id": 9, "name": "Unit G", "card_type": "unit"},
            {"id": 10, "name": "Unit H", "card_type": "unit"},
            {"id": 11, "name": "Unit I", "card_type": "unit"},
        ]


@pytest.mark.asyncio
async def test_bot_generator_replaces_blacklisted_cards_same_type():
    generator = BotGenerator(BotDBHarness(disabled={1, 3}))

    deck = await generator._sanitize_deck([1, 3, 4])

    assert 1 not in deck
    assert 3 not in deck
    assert 2 in deck
    assert len(deck) == 9


@pytest.mark.asyncio
async def test_bot_generator_fallback_deck_excludes_blacklist_and_keeps_hero():
    generator = BotGenerator(BotDBHarness(disabled={1, 3}))

    deck = await generator._build_bot_deck(100)

    assert 1 not in deck
    assert 3 not in deck
    assert 2 in deck
