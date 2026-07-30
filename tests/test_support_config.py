import pytest

from infrastructure import config as config_module
from infrastructure.config import get_settings


STRONG_JWT_SECRET = "test-support-jwt-secret-that-is-long-enough-2026"
STRONG_ADMIN_SECRET = "test-support-admin-secret-that-is-long-enough-2026"
STRONG_MCP_SECRET = "test-support-mcp-secret-that-is-long-enough-2026"
STRONG_SUPPORT_TOKEN = "support-bot-token-that-is-long-enough-2026"
STRONG_SUPPORT_WEBHOOK_SECRET = "support-max-webhook-secret-that-is-long-enough-2026"


def _base_env(monkeypatch, tmp_path, *, environment: str = "development") -> None:
    monkeypatch.setattr(config_module, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    if environment != "development":
        monkeypatch.setenv("JWT_SECRET", STRONG_JWT_SECRET)
        monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_ADMIN_SECRET)
        monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_MCP_SECRET)
        monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://admin.example")
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    for name in (
        "SUPPORT_ENABLED",
        "SUPPORT_TELEGRAM_BOT_TOKEN",
        "SUPPORT_MAX_BOT_TOKEN",
        "SUPPORT_MAX_WEBHOOK_SECRET",
        "SUPPORT_TELEGRAM_ADMIN_ID",
        "SUPPORT_MAX_ADMIN_ID",
        "SUPPORT_DATABASE_URL",
        "SUPPORT_DB_HOST",
        "SUPPORT_DB_PORT",
        "SUPPORT_DB_USER",
        "SUPPORT_DB_PASSWORD",
        "SUPPORT_DB_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()


def test_development_support_defaults_off_and_tolerates_missing_tokens(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)

    settings = get_settings()

    assert settings.support_enabled is False
    assert settings.support_telegram_bot_token == ""
    assert settings.support_max_bot_token == ""
    assert settings.support_max_webhook_secret == ""
    assert settings.support_telegram_admin_id == ""
    assert settings.support_max_admin_id == ""
    assert settings.support_database.database == "extraarena_support"

    get_settings.cache_clear()


def test_support_database_url_overrides_support_db_parts(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "SUPPORT_DATABASE_URL",
        "postgresql://support_user:p%40ss@support-db.example.com:15435/support_beta",
    )
    monkeypatch.setenv("SUPPORT_DB_HOST", "wrong-host.example")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.support_database.host == "support-db.example.com"
    assert settings.support_database.port == 15435
    assert settings.support_database.user == "support_user"
    assert settings.support_database.password == "p@ss"
    assert settings.support_database.database == "support_beta"

    get_settings.cache_clear()


def test_production_without_support_tokens_does_not_require_support_admin(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path, environment="production")

    settings = get_settings()

    assert settings.support_enabled is False
    assert settings.support_max_admin_id == ""

    get_settings.cache_clear()


def test_production_support_enabled_requires_bot_token_and_admin_id(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path, environment="production")
    monkeypatch.setenv("SUPPORT_ENABLED", "true")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="SUPPORT_TELEGRAM_BOT_TOKEN|SUPPORT_MAX_BOT_TOKEN"):
        get_settings()

    monkeypatch.setenv("SUPPORT_TELEGRAM_BOT_TOKEN", STRONG_SUPPORT_TOKEN)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="SUPPORT_TELEGRAM_ADMIN_ID"):
        get_settings()

    monkeypatch.setenv("SUPPORT_TELEGRAM_ADMIN_ID", "6803854304")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.support_enabled is True
    assert settings.support_telegram_admin_id == "6803854304"

    get_settings.cache_clear()


def test_production_max_support_requires_strong_webhook_secret(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path, environment="production")
    monkeypatch.setenv("SUPPORT_MAX_BOT_TOKEN", STRONG_SUPPORT_TOKEN)
    monkeypatch.setenv("SUPPORT_TELEGRAM_ADMIN_ID", "6803854304")
    monkeypatch.setenv("SUPPORT_MAX_ADMIN_ID", "987654321")
    monkeypatch.setenv("SUPPORT_MAX_WEBHOOK_SECRET", "weak")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="SUPPORT_MAX_WEBHOOK_SECRET"):
        get_settings()

    monkeypatch.setenv("SUPPORT_MAX_WEBHOOK_SECRET", STRONG_SUPPORT_WEBHOOK_SECRET)
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.support_enabled is True
    assert settings.support_max_admin_id == "987654321"

    get_settings.cache_clear()
