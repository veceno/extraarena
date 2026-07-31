from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv

from infrastructure.robokassa_payments import RobokassaSettings

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
DEFAULT_WEBAPP_URL = "https://clumsily-deft-guan.cloudpub.ru/"
DEFAULT_EXTRA_SHOP_URL = "https://clumsily-deft-guan.cloudpub.ru/"
DEFAULT_STARS_RATE_RUB = 1.5
DEFAULT_STARS_MARKUP = 1.2
DEFAULT_STARS_TEST_MODE = False
DEFAULT_ANDROID_LATEST_VERSION_CODE = 45
DEFAULT_ANDROID_LATEST_VERSION_NAME = "0.4.5"
DEFAULT_ANDROID_UPDATE_CHANNEL_URL = "https://t.me/extraarenamobile"
DEFAULT_ANDROID_APK_URL = "https://apk.laveqox.ru"
DEFAULT_ANDROID_RELEASE_STORAGE_DIR = str(BASE_DIR / "releases" / "android")
DEFAULT_ANDROID_RELEASE_PACKAGE_NAME = "ru.extraarena.app"
DEFAULT_ANDROID_RELEASE_MAX_BYTES = 1024 * 1024 * 1024
DEFAULT_ANDROID_RELEASE_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_ANDROID_UPLOAD_TOKEN_TTL_SECONDS = 60 * 60
DEFAULT_RUSTORE_CONSOLE_APP_ID = "2063712624"
DEFAULT_RUSTORE_APP_URL = "https://www.rustore.ru/catalog/app/ru.extraarena.app"
DEFAULT_PAYMENT_PROVIDER_ORDER = "robokassa,yookassa,rustore,stars"
DEFAULT_PAYMENT_PRIMARY_PROVIDER = "yookassa"
DEFAULT_PAYMENT_FALLBACK_PROVIDER = "yookassa"
DEFAULT_JWT_SECRET = "dev_secret_change_in_production!"
DEFAULT_ADMIN_SESSION_SECRET = "dev_admin_session_secret_change_in_production!"
DEFAULT_MCP_TOKEN_SECRET = "dev_mcp_token_secret_change_in_production!"
DEFAULT_MCP_ENDPOINT_PATH = "/admin/mcp"
DEFAULT_MCP_SESSION_PATH = "/api/admin/mcp/session"
DEFAULT_MCP_TOKEN_TTL_SECONDS = 15 * 60
DEFAULT_SUPPORT_DB_NAME = "extraarena_support"
MIN_PRODUCTION_JWT_SECRET_LENGTH = 32
MIN_PRODUCTION_ADMIN_SESSION_SECRET_LENGTH = 32
MIN_PRODUCTION_MCP_TOKEN_SECRET_LENGTH = 32
MIN_PRODUCTION_SUPPORT_BOT_TOKEN_LENGTH = 24
MIN_PRODUCTION_SUPPORT_WEBHOOK_SECRET_LENGTH = 32

MM_TROPHY_LIMIT_CLASSIC = 300
MM_BOT_TIMEOUT = 15
DECK_SIZE = 9
MAX_FREE_DECK_PRESETS = 3
MAX_TOTAL_DECK_PRESETS = 5
BOT_EXTRA_PASS_ROLL_PROBABILITIES = {
    "ultra": 0.03,
    "active": 0.15,
    "inactive": 0.82,
}

BOT_MODEL_PROFILES = {
    "extra-lr-v4-micro": {
        "model_path": "ai/models/extra-lr-v4-micro.onnx",
        "format": "train_v2_classic_v1",
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "neuron_count": "TrainV2 v4 Micro",
        "placement_mode": "append_only",
        "verify_mask": False,
        "metronome_enabled": True,
        "metronome_model_path": "ai/models/extra_lr_metronome_v1.onnx",
    },
    "extra-lr-v5-lite": {
        "model_path": "ai/models/extra-lr-v5-lite.onnx",
        "format": "v5",
        "obs_dim": 7128,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "neuron_count": "TrainV3 v5 Lite",
        "placement_mode": "append_only",
        "verify_mask": False,
        "mana_draw_head": True,
        "enemy_hand_known": True,
        "enemy_deck_known": True,
        "enemy_deck_order_known": True,
        "assembler_enabled": False,
        "cardoptimum_enabled": False,
        "metronome_enabled": True,
        "metronome_model_path": "ai/models/extra_lr_metronome_v1.onnx",
    },
    "extra-lr-v5": {
        "model_path": "ai/models/extra-lr-v5.onnx",
        "format": "v5",
        "obs_dim": 7128,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "neuron_count": "TrainV3 v5",
        "placement_mode": "append_only",
        "verify_mask": False,
        "mana_draw_head": True,
        "enemy_hand_known": True,
        "enemy_deck_known": True,
        "enemy_deck_order_known": True,
        "assembler_enabled": False,
        "cardoptimum_enabled": False,
        "metronome_enabled": True,
        "metronome_model_path": "ai/models/extra_lr_metronome_v1.onnx",
    },
    "extra-lr-v5-ultra": {
        # Ultra intentionally shares the post-C policy with V5; its strength
        # comes from the production assist stack below.
        "model_path": "ai/models/extra-lr-v5.onnx",
        "format": "v5",
        "obs_dim": 7128,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "neuron_count": "TrainV3 v5 Ultra",
        "placement_mode": "append_only",
        "verify_mask": False,
        "mana_draw_head": True,
        "enemy_hand_known": True,
        "enemy_deck_known": True,
        "enemy_deck_order_known": True,
        "assembler_enabled": True,
        "assembler_model_path": "ai/models/extra_lr_assembler_v1.onnx",
        "cardoptimum_enabled": True,
        "cardoptimum_model_path": "ai/models/extra_lr_cardoptimum_v1.onnx",
        "metronome_enabled": True,
        "metronome_model_path": "ai/models/extra_lr_metronome_v1.onnx",
    },
}

BOT_STRENGTH_TIERS = (
    {
        "key": "tier_lite_0000",
        "min_trophies": 0,
        "max_trophies": 99,
        "difficulty_label": "lite",
        "brain_profile": "extra-lr-v4-micro",
        "selection": "softmax",
        "temperature": 5.5,
        "level_policy": {"delta_min": -1, "delta_max": -1, "cap": 1, "boost_fraction": 0.0},
        "deck_policy": "starter_random",
    },
    {
        "key": "tier_easy_0100",
        "min_trophies": 100,
        "max_trophies": 299,
        "difficulty_label": "easy",
        "brain_profile": "extra-lr-v4-micro",
        "selection": "softmax",
        "temperature": 4.0,
        "level_policy": {"delta_min": -1, "delta_max": -1, "cap": 1, "boost_fraction": 0.0},
        "deck_policy": "weak_donor",
    },
    {
        "key": "tier_easy_plus_0300",
        "min_trophies": 300,
        "max_trophies": 599,
        "difficulty_label": "easy+",
        "brain_profile": "extra-lr-v5-lite",
        "selection": "softmax",
        "temperature": 3.5,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 2, "boost_fraction": 0.0},
        "deck_policy": "donor",
    },
    {
        "key": "tier_easy_plus_0600",
        "min_trophies": 600,
        "max_trophies": 999,
        "difficulty_label": "easy+",
        "brain_profile": "extra-lr-v5-lite",
        "selection": "softmax",
        "temperature": 3.0,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 3, "boost_fraction": 0.0},
        "deck_policy": "similar_donor",
    },
    {
        "key": "tier_medium_minus_1000",
        "min_trophies": 1000,
        "max_trophies": 1199,
        "difficulty_label": "medium-",
        "brain_profile": "extra-lr-v5-lite",
        "selection": "softmax",
        "temperature": 2.4,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 4, "boost_fraction": 0.0},
        "deck_policy": "similar_donor",
    },
    {
        "key": "tier_medium_1200",
        "min_trophies": 1200,
        "max_trophies": 1999,
        "difficulty_label": "medium",
        "brain_profile": "extra-lr-v5",
        "selection": "softmax",
        "temperature": 1.8,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 5, "boost_fraction": 0.0},
        "deck_policy": "decent_donor",
    },
    {
        "key": "tier_medium_plus_2000",
        "min_trophies": 2000,
        "max_trophies": 2999,
        "difficulty_label": "medium+",
        "brain_profile": "extra-lr-v5",
        "selection": "softmax",
        "temperature": 1.35,
        "level_policy": {"delta_min": -1, "delta_max": 0, "cap": 6, "boost_fraction": 0.0},
        "deck_policy": "donor_basic_synergy",
    },
    {
        "key": "tier_hard_minus_3000",
        "min_trophies": 3000,
        "max_trophies": 4499,
        "difficulty_label": "hard-",
        "brain_profile": "extra-lr-v5",
        "selection": "softmax",
        "temperature": 1.0,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 7, "boost_fraction": 0.0},
        "deck_policy": "strong_donor",
    },
    {
        "key": "tier_hard_4500",
        "min_trophies": 4500,
        "max_trophies": 5999,
        "difficulty_label": "hard",
        "brain_profile": "extra-lr-v5-ultra",
        "selection": "softmax",
        "temperature": 1.6,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 8, "boost_fraction": 0.0},
        "deck_policy": "curated_donor",
    },
    {
        "key": "tier_hard_plus_6000",
        "min_trophies": 6000,
        "max_trophies": 7499,
        "difficulty_label": "hard+",
        "brain_profile": "extra-lr-v5-ultra",
        "selection": "softmax",
        "temperature": 1.0,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 9, "boost_fraction": 0.15},
        "deck_policy": "strong_meta",
    },
    {
        "key": "tier_max_minus_7500",
        "min_trophies": 7500,
        "max_trophies": 8999,
        "difficulty_label": "max-",
        "brain_profile": "extra-lr-v5-ultra",
        "selection": "softmax",
        "temperature": 2.0,
        "level_policy": {"delta_min": -1, "delta_max": 0, "cap": 10, "boost_fraction": 0.25},
        "deck_policy": "meta",
    },
    {
        "key": "tier_max_9000",
        "min_trophies": 9000,
        "max_trophies": 1000000000,
        "difficulty_label": "max",
        "brain_profile": "extra-lr-v5-ultra",
        "selection": "softmax",
        "temperature": 0.45,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 10, "boost_fraction": 0.4},
        "deck_policy": "meta_boss",
    },
)


def _build_bot_profile(model_key: str, *, temperature: float, selection: str) -> dict:
    profile = dict(BOT_MODEL_PROFILES[model_key])
    profile["temperature_range"] = (temperature, temperature)
    profile["selection"] = selection
    return profile


BOT_DIFFICULTY_PROFILES = {
    tier["key"]: _build_bot_profile(
        tier["brain_profile"],
        temperature=tier["temperature"],
        selection=tier["selection"],
    )
    for tier in BOT_STRENGTH_TIERS
}

BOT_DIFFICULTY_ALIASES = {
    # Stable model aliases used by the training UI and service clients.
    "v4-micro": "tier_lite_0000",
    "v5-lite": "tier_easy_plus_0300",
    "v5": "tier_medium_1200",
    "v5-ultra": "tier_hard_4500",
    "extra-lr-v4-micro": "tier_lite_0000",
    "extra-lr-v5-lite": "tier_easy_plus_0300",
    "extra-lr-v5": "tier_medium_1200",
    "extra-lr-v5-ultra": "tier_hard_4500",
    # Backward-compatible public labels.
    "lite": "tier_lite_0000",
    "easy": "tier_easy_0100",
    "medium": "tier_medium_1200",
    "hard": "tier_hard_4500",
    "max": "tier_max_9000",
}

# Backward-compatible public labels point at canonical strength tiers.
BOT_DIFFICULTY_PROFILES.update(
    {
        alias: dict(BOT_DIFFICULTY_PROFILES[tier_key])
        for alias, tier_key in BOT_DIFFICULTY_ALIASES.items()
    }
)


def get_bot_strength_tier(player_trophies: int) -> dict:
    trophies = max(0, int(player_trophies or 0))
    for tier in BOT_STRENGTH_TIERS:
        if tier["min_trophies"] <= trophies <= tier["max_trophies"]:
            return tier
    return BOT_STRENGTH_TIERS[-1]

LEAGUE_CONFIG = {
    1:  {"name": "Novice",       "emoji": "\U0001f331", "color": "#2ECC71", "min_trophies": 0},
    2:  {"name": "Bronze",       "emoji": "\U0001f949", "color": "#E67E22", "min_trophies": 300},
    3:  {"name": "Silver",       "emoji": "\U0001f948", "color": "#95A5A6", "min_trophies": 600},
    4:  {"name": "Gold",         "emoji": "\U0001f947", "color": "#F1C40F", "min_trophies": 1200},
    5:  {"name": "Crystal",      "emoji": "\U0001f4ce", "color": "#3498DB", "min_trophies": 2000},
    6:  {"name": "Master",       "emoji": "\u2b50",     "color": "#F39C12", "min_trophies": 3000},
    7:  {"name": "Champion",     "emoji": "\U0001f3c6", "color": "#E74C3C", "min_trophies": 4500},
    8:  {"name": "Grandmaster",  "emoji": "\U0001f4ab", "color": "#9B59B6", "min_trophies": 6000},
    9:  {"name": "Legendary",    "emoji": "\U0001f451", "color": "#FF6B6B", "min_trophies": 7500},
    10: {"name": "Extra",        "emoji": "\U0001f3df\ufe0f", "color": "#FFD700", "min_trophies": 9000},
}

LEAGUE_NEXT_TROPHIES = [300, 600, 1200, 2000, 3000, 4500, 6000, 7500, 9000, 10000]


def get_league_by_trophies_fn(trophies: int) -> int:
    for lid in range(10, 0, -1):
        if trophies >= LEAGUE_CONFIG[lid]["min_trophies"]:
            return lid
    return 1


TROPHY_TIERS = {
    "novice":     {"range": range(0, 301),    "win": (25, 35), "loss": (3, 7),   "coin_range": (10, 15)},
    "student":    {"range": range(301, 701),  "win": (20, 30), "loss": (5, 10),  "coin_range": (15, 25)},
    "advanced":   {"range": range(701, 2501), "win": (20, 25), "loss": (15, 25), "coin_range": (25, 40)},
    "advanced_2": {"range": range(2501, 5001),"win": (15, 20), "loss": (20, 25), "coin_range": (40, 65)},
    "pro":        {"range": range(5001, 8001),"win": (10, 18), "loss": (25, 30), "coin_range": (70, 100)},
    "master":     {"range": range(8001, 100000),"win": (10, 15),"loss": (27, 30), "coin_range": (120, 180)},
}

DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5434
DEFAULT_DB_USER = "postgres"
DEFAULT_DB_PASSWORD = ""
DEFAULT_DB_NAME = "extraarena"


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class YooKassaSettings:
    shop_id: str
    secret_key: str
    test_mode: bool = False


@dataclass(frozen=True)
class RuStorePaymentSettings:
    public_token: str = ""
    console_app_id: str = ""
    sandbox: bool = False
    key_id: str = ""
    private_key: str = ""
    private_key_file: str = ""


@dataclass(frozen=True)
class Settings:
    bot_token: str
    webapp_url: str
    extra_shop_url: str
    max_bot_token: str = ""
    max_bot_webhook_secret: str = ""
    max_bot_username: str = ""
    environment: str = "development"
    database: DatabaseSettings | None = None
    extraid_database: DatabaseSettings | None = None
    support_database: DatabaseSettings | None = None
    web_host: str = "127.0.0.1"
    web_port: int = 8081
    yookassa: YooKassaSettings | None = None
    robokassa: RobokassaSettings | None = None
    rustore: RuStorePaymentSettings | None = None
    rustore_app_url: str = DEFAULT_RUSTORE_APP_URL
    payment_provider_order: str = DEFAULT_PAYMENT_PROVIDER_ORDER
    payment_primary_provider: str = DEFAULT_PAYMENT_PRIMARY_PROVIDER
    payment_fallback_provider: str = DEFAULT_PAYMENT_FALLBACK_PROVIDER
    stars_rate_rub: float = DEFAULT_STARS_RATE_RUB
    stars_markup: float = DEFAULT_STARS_MARKUP
    stars_test_mode: bool = DEFAULT_STARS_TEST_MODE
    jwt_secret: str = DEFAULT_JWT_SECRET
    admin_session_secret: str = DEFAULT_ADMIN_SESSION_SECRET
    jwt_expiry_days: int = 30
    mcp_enabled: bool = True
    mcp_token_secret: str = DEFAULT_MCP_TOKEN_SECRET
    mcp_endpoint_path: str = DEFAULT_MCP_ENDPOINT_PATH
    mcp_session_path: str = DEFAULT_MCP_SESSION_PATH
    mcp_token_ttl_seconds: int = DEFAULT_MCP_TOKEN_TTL_SECONDS
    mcp_allowed_origins: tuple[str, ...] = ("*",)
    support_enabled: bool = False
    support_telegram_bot_token: str = ""
    support_max_bot_token: str = ""
    support_max_webhook_secret: str = ""
    support_telegram_admin_id: str = ""
    support_max_admin_id: str = ""
    telegram_api_insecure_ssl: bool = False
    cors_allowed_origins: tuple[str, ...] = ("*",)
    payment_webhook_diagnostics_enabled: bool = False
    payments_required: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "noreply@extraarena.gg"
    ip_geo_api_key: str = ""
    android_latest_version_code: int = DEFAULT_ANDROID_LATEST_VERSION_CODE
    android_latest_version_name: str = DEFAULT_ANDROID_LATEST_VERSION_NAME
    android_min_supported_version_code: int = DEFAULT_ANDROID_LATEST_VERSION_CODE
    android_update_channel_url: str = DEFAULT_ANDROID_UPDATE_CHANNEL_URL
    android_apk_url: str = DEFAULT_ANDROID_APK_URL
    android_releases_enabled: bool = True
    android_release_storage_dir: str = DEFAULT_ANDROID_RELEASE_STORAGE_DIR
    android_release_public_base_url: str = ""
    android_release_package_name: str = DEFAULT_ANDROID_RELEASE_PACKAGE_NAME
    android_direct_signing_cert_sha256: str = ""
    android_rustore_signing_cert_sha256: str = ""
    android_apksigner_command: str = "apksigner"
    android_aapt_command: str = "aapt2"
    android_release_max_bytes: int = DEFAULT_ANDROID_RELEASE_MAX_BYTES
    android_release_chunk_bytes: int = DEFAULT_ANDROID_RELEASE_CHUNK_BYTES
    android_upload_token_ttl_seconds: int = DEFAULT_ANDROID_UPLOAD_TOKEN_TTL_SECONDS
    match_state_backend: str = "memory"
    web_concurrency: int = 1
    auto_migrate_on_start: bool = True
    shop_allow_max_level_particles: bool = False


def _load_env() -> None:
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower().strip("[]")
    return (
        normalized in {"localhost", "127.0.0.1", "::1"}
        or normalized.startswith("127.")
    )


def _env_list(name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, "").split(",") if part.strip())


def _origin_from_url(url: str) -> str:
    parsed = urlsplit(url or "")
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _https_origin_from_url(url: str) -> str:
    text = str(url or "").strip()
    if not text or "\\" in text or any(ord(character) < 32 or ord(character) == 127 for character in text):
        return ""
    try:
        parsed = urlsplit(text)
        _ = parsed.port
    except ValueError:
        return ""
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    return f"https://{parsed.netloc}"


def _database_settings_from_env(
    *,
    url_env: str,
    prefix: str,
    default_database: str,
) -> DatabaseSettings:
    database_url = os.getenv(url_env, "").strip()
    if database_url:
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or not parsed.path.strip("/"):
            raise RuntimeError(f"{url_env} must be a postgresql:// URL with host and database name.")
        try:
            port = parsed.port or DEFAULT_DB_PORT
        except ValueError as exc:
            raise RuntimeError(f"{url_env} has an invalid port.") from exc
        return DatabaseSettings(
            host=parsed.hostname,
            port=port,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=unquote(parsed.path.strip("/")),
        )

    return DatabaseSettings(
        host=os.getenv(f"{prefix}_HOST", DEFAULT_DB_HOST),
        port=int(os.getenv(f"{prefix}_PORT", str(DEFAULT_DB_PORT))),
        user=os.getenv(f"{prefix}_USER", DEFAULT_DB_USER),
        password=os.getenv(f"{prefix}_PASSWORD", DEFAULT_DB_PASSWORD),
        database=os.getenv(f"{prefix}_NAME", default_database),
    )


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def _normalize_endpoint_path(name: str, default: str) -> str:
    path = os.getenv(name, default).strip() or default
    if not path.startswith("/"):
        path = f"/{path}"
    if len(path) > 1:
        path = path.rstrip("/")
    if any(char.isspace() for char in path) or "?" in path or "#" in path:
        raise RuntimeError(f"{name} must be a URL path without whitespace, query, or fragment.")
    return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env()

    environment_explicit = "ENVIRONMENT" in os.environ
    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    default_payment_test_mode = environment == "development"

    db_settings = _database_settings_from_env(
        url_env="DATABASE_URL",
        prefix="DB",
        default_database=DEFAULT_DB_NAME,
    )

    extraid_db_settings = _database_settings_from_env(
        url_env="EXTRAID_DATABASE_URL",
        prefix="EXTRAID_DB",
        default_database="extraid",
    )

    support_db_settings = _database_settings_from_env(
        url_env="SUPPORT_DATABASE_URL",
        prefix="SUPPORT_DB",
        default_database=DEFAULT_SUPPORT_DB_NAME,
    )

    yookassa_shop_id = os.getenv("YOOKASSA_SHOP_ID", "")
    yookassa_secret_key = os.getenv("YOOKASSA_SECRET_KEY", "")
    yookassa_test_mode = _env_bool("YOOKASSA_TEST_MODE", default_payment_test_mode)
    robokassa_merchant_login = os.getenv("ROBOKASSA_MERCHANT_LOGIN", "").strip()
    robokassa_password1 = os.getenv("ROBOKASSA_PASSWORD1", "").strip()
    robokassa_password2 = os.getenv("ROBOKASSA_PASSWORD2", "").strip()
    robokassa_password3 = os.getenv("ROBOKASSA_PASSWORD3", "").strip()
    robokassa_hash_algo = os.getenv("ROBOKASSA_HASH_ALGO", "SHA256").strip()
    robokassa_test_mode = _env_bool("ROBOKASSA_TEST_MODE", default_payment_test_mode)
    if robokassa_test_mode:
        robokassa_merchant_login = os.getenv("ROBOKASSA_TEST_MERCHANT_LOGIN", robokassa_merchant_login).strip()
        robokassa_password1 = os.getenv("ROBOKASSA_TEST_PASSWORD1", robokassa_password1).strip()
        robokassa_password2 = os.getenv("ROBOKASSA_TEST_PASSWORD2", robokassa_password2).strip()
        robokassa_password3 = os.getenv("ROBOKASSA_TEST_PASSWORD3", robokassa_password3).strip()
        robokassa_hash_algo = os.getenv("ROBOKASSA_TEST_HASH_ALGO", robokassa_hash_algo).strip()
    robokassa_result_url = os.getenv("ROBOKASSA_RESULT_URL", "").strip()
    robokassa_success_url = os.getenv("ROBOKASSA_SUCCESS_URL", "").strip()
    robokassa_fail_url = os.getenv("ROBOKASSA_FAIL_URL", "").strip()
    robokassa_payment_object = os.getenv("ROBOKASSA_PAYMENT_OBJECT", "service").strip() or "service"
    rustore_public_token = os.getenv("RUSTORE_PUBLIC_TOKEN", "").strip()
    rustore_console_app_id = os.getenv("RUSTORE_CONSOLE_APP_ID", DEFAULT_RUSTORE_CONSOLE_APP_ID).strip()
    rustore_sandbox = os.getenv("RUSTORE_SANDBOX", "false").lower() in {"1", "true", "yes"}
    rustore_key_id = os.getenv("RUSTORE_KEY_ID", "").strip()
    rustore_private_key = os.getenv("RUSTORE_PRIVATE_KEY", "").strip()
    rustore_private_key_file = os.getenv("RUSTORE_PRIVATE_KEY_FILE", "").strip()

    yookassa_settings = None
    if yookassa_shop_id and yookassa_secret_key:
        yookassa_settings = YooKassaSettings(
            shop_id=yookassa_shop_id,
            secret_key=yookassa_secret_key,
            test_mode=yookassa_test_mode,
        )

    robokassa_settings = None
    if robokassa_merchant_login and robokassa_password1 and robokassa_password2:
        robokassa_settings = RobokassaSettings(
            merchant_login=robokassa_merchant_login,
            password1=robokassa_password1,
            password2=robokassa_password2,
            password3=robokassa_password3,
            hash_algo=robokassa_hash_algo,
            test_mode=robokassa_test_mode,
            result_url=robokassa_result_url,
            success_url=robokassa_success_url,
            fail_url=robokassa_fail_url,
            payment_object=robokassa_payment_object,
        )

    rustore_settings = None
    if rustore_public_token or (rustore_key_id and (rustore_private_key or rustore_private_key_file)):
        rustore_settings = RuStorePaymentSettings(
            public_token=rustore_public_token,
            console_app_id=rustore_console_app_id,
            sandbox=rustore_sandbox,
            key_id=rustore_key_id,
            private_key=rustore_private_key,
            private_key_file=rustore_private_key_file,
        )

    stars_rate_rub = float(os.getenv("STARS_RATE_RUB", DEFAULT_STARS_RATE_RUB))
    stars_markup = float(os.getenv("STARS_MARKUP", DEFAULT_STARS_MARKUP))
    stars_test_mode = os.getenv("STARS_TEST_MODE", str(DEFAULT_STARS_TEST_MODE)).lower() in {"1", "true", "yes"}

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN env var is required (not set in .env or environment)")
    webapp_url = os.getenv("WEBAPP_URL", DEFAULT_WEBAPP_URL)
    extra_shop_url = os.getenv("EXTRA_SHOP_URL", DEFAULT_EXTRA_SHOP_URL)
    android_release_public_base_url_raw = os.getenv("ANDROID_RELEASE_PUBLIC_BASE_URL", "").strip()
    if android_release_public_base_url_raw:
        android_release_public_base_url = _https_origin_from_url(android_release_public_base_url_raw)
        if not android_release_public_base_url:
            raise RuntimeError("ANDROID_RELEASE_PUBLIC_BASE_URL must be an absolute HTTPS URL without credentials.")
    else:
        android_release_public_base_url = (
            _https_origin_from_url(os.getenv("PUBLIC_BASE_URL", ""))
            or _https_origin_from_url(webapp_url)
        )
    max_bot_token = os.getenv("MAX_BOT_TOKEN", "").strip()
    max_bot_webhook_secret = os.getenv("MAX_BOT_WEBHOOK_SECRET", "").strip()
    max_bot_username = os.getenv("MAX_BOT_USERNAME", "").strip().lstrip("@")
    web_host = os.getenv("WEBAPP_HOST", "127.0.0.1").strip()
    jwt_secret = os.getenv("JWT_SECRET", DEFAULT_JWT_SECRET).strip()
    admin_session_secret = os.getenv("ADMIN_SESSION_SECRET", DEFAULT_ADMIN_SESSION_SECRET).strip()
    mcp_enabled = _env_bool("MCP_ENABLED", True)
    mcp_token_secret = os.getenv("MCP_TOKEN_SECRET", DEFAULT_MCP_TOKEN_SECRET).strip()
    mcp_endpoint_path = _normalize_endpoint_path("MCP_ENDPOINT_PATH", DEFAULT_MCP_ENDPOINT_PATH)
    mcp_session_path = _normalize_endpoint_path("MCP_SESSION_PATH", DEFAULT_MCP_SESSION_PATH)
    mcp_token_ttl_seconds = _env_positive_int("MCP_TOKEN_TTL_SECONDS", DEFAULT_MCP_TOKEN_TTL_SECONDS)
    support_telegram_bot_token = os.getenv("SUPPORT_TELEGRAM_BOT_TOKEN", "").strip()
    support_max_bot_token = os.getenv("SUPPORT_MAX_BOT_TOKEN", "").strip()
    support_max_webhook_secret = os.getenv("SUPPORT_MAX_WEBHOOK_SECRET", "").strip()
    support_telegram_admin_id = os.getenv("SUPPORT_TELEGRAM_ADMIN_ID", "").strip()
    support_max_admin_id = os.getenv("SUPPORT_MAX_ADMIN_ID", "").strip()
    support_enabled = (
        _env_bool("SUPPORT_ENABLED", False)
        or bool(support_telegram_bot_token)
        or bool(support_max_bot_token)
    )
    match_state_backend = os.getenv("MATCH_STATE_BACKEND", "memory").strip().lower() or "memory"
    web_concurrency = max(
        int(os.getenv("WEB_CONCURRENCY", "0") or 0),
        int(os.getenv("GUNICORN_WORKERS", "0") or 0),
        int(os.getenv("UVICORN_WORKERS", "0") or 0),
        1,
    )
    auto_migrate_on_start = _env_bool("AUTO_MIGRATE_ON_START", environment == "development")
    payments_required = _env_bool("PAYMENTS_REQUIRED", environment != "development")
    telegram_api_insecure_ssl = _env_bool("TELEGRAM_API_INSECURE_SSL", False)
    configured_cors_origins = _env_list("CORS_ALLOWED_ORIGINS") or _env_list("WEBAPP_ALLOWED_ORIGINS")
    if not configured_cors_origins:
        configured_cors_origins = ("*",) if environment == "development" else tuple(
            origin for origin in {
                _origin_from_url(webapp_url),
                _origin_from_url(extra_shop_url),
            } if origin
        )
    configured_mcp_allowed_origins = _env_list("MCP_ALLOWED_ORIGINS")
    if not configured_mcp_allowed_origins:
        configured_mcp_allowed_origins = ("*",) if environment == "development" else ()
    payment_webhook_diagnostics_enabled = _env_bool("PAYMENT_WEBHOOK_DIAGNOSTICS_ENABLED", False)
    public_bind = not _is_loopback_host(web_host)
    if public_bind and not environment_explicit:
        raise RuntimeError("ENVIRONMENT must be set explicitly for non-local WEBAPP_HOST binds.")
    if public_bind and environment == "development":
        raise RuntimeError("ENVIRONMENT must not be development for non-local WEBAPP_HOST binds.")
    if (
        environment != "development"
        and (
            not jwt_secret
            or jwt_secret == DEFAULT_JWT_SECRET
            or len(jwt_secret) < MIN_PRODUCTION_JWT_SECRET_LENGTH
        )
    ):
        raise RuntimeError(
            "JWT_SECRET must be set to a strong secret outside development "
            f"({MIN_PRODUCTION_JWT_SECRET_LENGTH}+ characters, not the default)."
        )
    if environment != "development" and (
        not admin_session_secret
        or admin_session_secret == DEFAULT_ADMIN_SESSION_SECRET
        or admin_session_secret == jwt_secret
        or len(admin_session_secret) < MIN_PRODUCTION_ADMIN_SESSION_SECRET_LENGTH
    ):
        raise RuntimeError(
            "ADMIN_SESSION_SECRET must be set to a separate strong secret outside development "
            f"({MIN_PRODUCTION_ADMIN_SESSION_SECRET_LENGTH}+ characters, not the default or JWT_SECRET)."
        )
    if mcp_enabled and environment != "development":
        if (
            not mcp_token_secret
            or mcp_token_secret == DEFAULT_MCP_TOKEN_SECRET
            or mcp_token_secret in {jwt_secret, admin_session_secret}
            or len(mcp_token_secret) < MIN_PRODUCTION_MCP_TOKEN_SECRET_LENGTH
        ):
            raise RuntimeError(
                "MCP_TOKEN_SECRET must be set to a separate strong secret when MCP is enabled outside development "
                f"({MIN_PRODUCTION_MCP_TOKEN_SECRET_LENGTH}+ characters, not the default, JWT_SECRET, or ADMIN_SESSION_SECRET)."
            )
        if not configured_mcp_allowed_origins or any(
            origin in {"", "*"} for origin in configured_mcp_allowed_origins
        ):
            raise RuntimeError("MCP_ALLOWED_ORIGINS must be an explicit allowlist when MCP is enabled outside development.")
    if support_enabled and environment != "development":
        if not (support_telegram_bot_token or support_max_bot_token):
            raise RuntimeError(
                "SUPPORT_TELEGRAM_BOT_TOKEN or SUPPORT_MAX_BOT_TOKEN is required when support is enabled outside development."
            )
        if support_telegram_bot_token and len(support_telegram_bot_token) < MIN_PRODUCTION_SUPPORT_BOT_TOKEN_LENGTH:
            raise RuntimeError(
                "SUPPORT_TELEGRAM_BOT_TOKEN must be a strong production token when support is enabled outside development."
            )
        if support_max_bot_token and len(support_max_bot_token) < MIN_PRODUCTION_SUPPORT_BOT_TOKEN_LENGTH:
            raise RuntimeError(
                "SUPPORT_MAX_BOT_TOKEN must be a strong production token when support is enabled outside development."
            )
        if support_telegram_bot_token and not support_telegram_admin_id:
            raise RuntimeError("SUPPORT_TELEGRAM_ADMIN_ID is required when SUPPORT_TELEGRAM_BOT_TOKEN is configured outside development.")
        if support_max_bot_token and not support_max_admin_id:
            raise RuntimeError("SUPPORT_MAX_ADMIN_ID is required when SUPPORT_MAX_BOT_TOKEN is configured outside development.")
        if support_max_bot_token and len(support_max_webhook_secret) < MIN_PRODUCTION_SUPPORT_WEBHOOK_SECRET_LENGTH:
            raise RuntimeError(
                "SUPPORT_MAX_WEBHOOK_SECRET must be a strong secret when SUPPORT_MAX_BOT_TOKEN is configured outside development."
            )
    if max_bot_token and environment != "development":
        if len(max_bot_token) < MIN_PRODUCTION_SUPPORT_BOT_TOKEN_LENGTH:
            raise RuntimeError(
                "MAX_BOT_TOKEN must be a strong production token outside development."
            )
        if len(max_bot_webhook_secret) < MIN_PRODUCTION_SUPPORT_WEBHOOK_SECRET_LENGTH:
            raise RuntimeError(
                "MAX_BOT_WEBHOOK_SECRET must be a strong secret when MAX_BOT_TOKEN is configured outside development."
            )
    if mcp_enabled:
        if mcp_endpoint_path == mcp_session_path:
            raise RuntimeError("MCP_ENDPOINT_PATH and MCP_SESSION_PATH must be different.")
        if not mcp_session_path.startswith("/api/admin/"):
            raise RuntimeError("MCP_SESSION_PATH must remain under /api/admin/ so central admin auth protects token bootstrap.")
    if public_bind and (
        not jwt_secret
        or jwt_secret == DEFAULT_JWT_SECRET
        or len(jwt_secret) < MIN_PRODUCTION_JWT_SECRET_LENGTH
    ):
        raise RuntimeError("JWT_SECRET must be strong and non-default for non-local WEBAPP_HOST binds.")
    if public_bind and (
        not admin_session_secret
        or admin_session_secret == DEFAULT_ADMIN_SESSION_SECRET
        or admin_session_secret == jwt_secret
        or len(admin_session_secret) < MIN_PRODUCTION_ADMIN_SESSION_SECRET_LENGTH
    ):
        raise RuntimeError("ADMIN_SESSION_SECRET must be separate, strong, and non-default for non-local WEBAPP_HOST binds.")
    if telegram_api_insecure_ssl and (environment != "development" or public_bind):
        raise RuntimeError("TELEGRAM_API_INSECURE_SSL is allowed only for local development.")
    if match_state_backend == "memory" and web_concurrency > 1:
        raise RuntimeError(
            "MATCH_STATE_BACKEND=memory requires a single web worker; set WEB_CONCURRENCY=1 "
            "or configure a shared match state backend before multi-worker deployment."
        )
    if environment != "development":
        if not configured_cors_origins or any(origin in {"", "*"} for origin in configured_cors_origins):
            raise RuntimeError("CORS_ALLOWED_ORIGINS must be an explicit allowlist outside development.")
        if payment_webhook_diagnostics_enabled:
            raise RuntimeError("PAYMENT_WEBHOOK_DIAGNOSTICS_ENABLED is allowed only in development.")
    if environment == "production" and not _env_bool("ALLOW_PAYMENT_TEST_MODE", False):
        enabled_test_modes = []
        if yookassa_settings and yookassa_settings.test_mode:
            enabled_test_modes.append("YOOKASSA_TEST_MODE")
        if robokassa_settings and robokassa_settings.test_mode:
            enabled_test_modes.append("ROBOKASSA_TEST_MODE")
        if rustore_settings and rustore_settings.sandbox:
            enabled_test_modes.append("RUSTORE_SANDBOX")
        if stars_test_mode:
            enabled_test_modes.append("STARS_TEST_MODE")
        if enabled_test_modes:
            raise RuntimeError(
                "Payment test/sandbox modes are not allowed in production without "
                f"ALLOW_PAYMENT_TEST_MODE=true: {', '.join(enabled_test_modes)}."
            )

    payment_primary_provider = os.getenv(
        "PAYMENT_PRIMARY_PROVIDER",
        "robokassa" if robokassa_settings else DEFAULT_PAYMENT_PRIMARY_PROVIDER,
    ).strip().lower()
    payment_fallback_provider = os.getenv(
        "PAYMENT_FALLBACK_PROVIDER",
        DEFAULT_PAYMENT_FALLBACK_PROVIDER,
    ).strip().lower()

    return Settings(
        bot_token=bot_token,
        webapp_url=webapp_url,
        extra_shop_url=extra_shop_url,
        max_bot_token=max_bot_token,
        max_bot_webhook_secret=max_bot_webhook_secret,
        max_bot_username=max_bot_username,
        environment=environment,
        database=db_settings,
        extraid_database=extraid_db_settings,
        support_database=support_db_settings,
        web_host=web_host,
        web_port=int(os.getenv("WEBAPP_PORT", "8081")),
        yookassa=yookassa_settings,
        robokassa=robokassa_settings,
        rustore=rustore_settings,
        rustore_app_url=os.getenv("RUSTORE_APP_URL", DEFAULT_RUSTORE_APP_URL),
        payment_provider_order=os.getenv("PAYMENT_PROVIDER_ORDER", DEFAULT_PAYMENT_PROVIDER_ORDER),
        payment_primary_provider=payment_primary_provider,
        payment_fallback_provider=payment_fallback_provider,
        stars_rate_rub=stars_rate_rub,
        stars_markup=stars_markup,
        stars_test_mode=stars_test_mode,
        jwt_secret=jwt_secret,
        admin_session_secret=admin_session_secret,
        jwt_expiry_days=int(os.getenv("JWT_EXPIRY_DAYS", "30")),
        mcp_enabled=mcp_enabled,
        mcp_token_secret=mcp_token_secret,
        mcp_endpoint_path=mcp_endpoint_path,
        mcp_session_path=mcp_session_path,
        mcp_token_ttl_seconds=mcp_token_ttl_seconds,
        mcp_allowed_origins=configured_mcp_allowed_origins,
        support_enabled=support_enabled,
        support_telegram_bot_token=support_telegram_bot_token,
        support_max_bot_token=support_max_bot_token,
        support_max_webhook_secret=support_max_webhook_secret,
        support_telegram_admin_id=support_telegram_admin_id,
        support_max_admin_id=support_max_admin_id,
        telegram_api_insecure_ssl=telegram_api_insecure_ssl,
        cors_allowed_origins=configured_cors_origins,
        payment_webhook_diagnostics_enabled=payment_webhook_diagnostics_enabled,
        payments_required=payments_required,
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        smtp_from=os.getenv("SMTP_FROM", "noreply@extraarena.gg"),
        ip_geo_api_key=os.getenv("IP_GEO_API_KEY", ""),
        android_latest_version_code=int(os.getenv("ANDROID_LATEST_VERSION_CODE", str(DEFAULT_ANDROID_LATEST_VERSION_CODE))),
        android_latest_version_name=os.getenv("ANDROID_LATEST_VERSION_NAME", DEFAULT_ANDROID_LATEST_VERSION_NAME),
        android_min_supported_version_code=int(os.getenv(
            "ANDROID_MIN_SUPPORTED_VERSION_CODE",
            os.getenv("ANDROID_LATEST_VERSION_CODE", str(DEFAULT_ANDROID_LATEST_VERSION_CODE)),
        )),
        android_update_channel_url=os.getenv("EXTRAARENA_UPDATE_CHANNEL_URL", DEFAULT_ANDROID_UPDATE_CHANNEL_URL),
        android_apk_url=os.getenv("EXTRAARENA_APK_URL", DEFAULT_ANDROID_APK_URL),
        android_releases_enabled=_env_bool("ANDROID_RELEASES_ENABLED", True),
        android_release_storage_dir=os.getenv(
            "ANDROID_RELEASE_STORAGE_DIR",
            DEFAULT_ANDROID_RELEASE_STORAGE_DIR,
        ),
        android_release_public_base_url=android_release_public_base_url,
        android_release_package_name=os.getenv(
            "ANDROID_RELEASE_PACKAGE_NAME",
            DEFAULT_ANDROID_RELEASE_PACKAGE_NAME,
        ).strip(),
        android_direct_signing_cert_sha256=os.getenv("ANDROID_DIRECT_SIGNING_CERT_SHA256", "").strip(),
        android_rustore_signing_cert_sha256=os.getenv("ANDROID_RUSTORE_SIGNING_CERT_SHA256", "").strip(),
        android_apksigner_command=os.getenv("ANDROID_APKSIGNER_COMMAND", "apksigner").strip(),
        android_aapt_command=os.getenv("ANDROID_AAPT_COMMAND", "aapt2").strip(),
        android_release_max_bytes=int(os.getenv(
            "ANDROID_RELEASE_MAX_BYTES",
            str(DEFAULT_ANDROID_RELEASE_MAX_BYTES),
        )),
        android_release_chunk_bytes=int(os.getenv(
            "ANDROID_RELEASE_CHUNK_BYTES",
            str(DEFAULT_ANDROID_RELEASE_CHUNK_BYTES),
        )),
        android_upload_token_ttl_seconds=int(os.getenv(
            "ANDROID_UPLOAD_TOKEN_TTL_SECONDS",
            str(DEFAULT_ANDROID_UPLOAD_TOKEN_TTL_SECONDS),
        )),
        match_state_backend=match_state_backend,
        web_concurrency=web_concurrency,
        auto_migrate_on_start=auto_migrate_on_start,
        shop_allow_max_level_particles=_env_bool("SHOP_ALLOW_MAX_LEVEL_PARTICLES", False),
    )
