from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

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
DEFAULT_RUSTORE_CONSOLE_APP_ID = "2063712624"
DEFAULT_RUSTORE_APP_URL = "https://www.rustore.ru/catalog/app/ru.extraarena.app"
DEFAULT_PAYMENT_PROVIDER_ORDER = "yookassa,rustore,stars"

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
    "extra-lr-v1": {
        "model_path": "ai/models/OnlyVersusRandomBiggest.onnx",
        "obs_dim": 621,
        "neuron_count": "7.6M Lite",
    },
    "extra-lr-v2": {
        "model_path": "ai/models/extra-lr-v3-medium.onnx",
        "obs_dim": 997,
        "neuron_count": "9.5M Berserk",
    },
    "extra-lr-v3-max": {
        "model_path": "ai/models/extra-lr-v3-max.onnx",
        "obs_dim": 997,
        "neuron_count": "Legacy Max",
    },
    "extra-lr-v4-lite": {
        "model_path": "ai/models/extra-lr-v4-lite.onnx",
        "format": "train_v2_classic_v1",
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "neuron_count": "TrainV2 v4 Lite",
        "placement_mode": "append_only",
        "verify_mask": False,
    },
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
    },
    "extra-lr-v4-opti": {
        "model_path": "ai/models/extra-lr-v4-opti.onnx",
        "format": "train_v2_classic_v1",
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "neuron_count": "TrainV2 v4 Opti",
        "placement_mode": "append_only",
        "verify_mask": False,
    },
    "extra-lr-v4-max": {
        "model_path": "ai/models/extra-lr-v4-max.onnx",
        "format": "train_v2_classic_v1",
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "neuron_count": "TrainV2 v4",
        "placement_mode": "append_only",
        "verify_mask": False,
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
        "temperature": 5.0,
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
        "temperature": 3.2,
        "level_policy": {"delta_min": -1, "delta_max": -1, "cap": 2, "boost_fraction": 0.0},
        "deck_policy": "weak_donor",
    },
    {
        "key": "tier_easy_plus_0300",
        "min_trophies": 300,
        "max_trophies": 599,
        "difficulty_label": "easy+",
        "brain_profile": "extra-lr-v4-lite",
        "selection": "softmax",
        "temperature": 2.8,
        "level_policy": {"delta_min": 0, "delta_max": 0, "cap": 3, "boost_fraction": 0.0},
        "deck_policy": "donor",
    },
    {
        "key": "tier_medium_minus_0600",
        "min_trophies": 600,
        "max_trophies": 1199,
        "difficulty_label": "medium-",
        "brain_profile": "extra-lr-v4-opti",
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
        "brain_profile": "extra-lr-v4-opti",
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
        "brain_profile": "extra-lr-v4-opti",
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
        "brain_profile": "extra-lr-v4-opti",
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
        "brain_profile": "extra-lr-v4-max",
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
        "brain_profile": "extra-lr-v4-max",
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
        "brain_profile": "extra-lr-v4-max",
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
        "brain_profile": "extra-lr-v4-max",
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
DEFAULT_DB_USER = "laveqox"
DEFAULT_DB_PASSWORD = "123112"
DEFAULT_DB_NAME = "laveqox"


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
    test_mode: bool = True


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
    environment: str = "development"
    database: DatabaseSettings | None = None
    extraid_database: DatabaseSettings | None = None
    web_host: str = "0.0.0.0"
    web_port: int = 8081
    yookassa: YooKassaSettings | None = None
    rustore: RuStorePaymentSettings | None = None
    rustore_app_url: str = DEFAULT_RUSTORE_APP_URL
    payment_provider_order: str = DEFAULT_PAYMENT_PROVIDER_ORDER
    stars_rate_rub: float = DEFAULT_STARS_RATE_RUB
    stars_markup: float = DEFAULT_STARS_MARKUP
    stars_test_mode: bool = True
    jwt_secret: str = "dev_secret_change_in_production!"
    jwt_expiry_days: int = 30
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


def _load_env() -> None:
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env()

    db_settings = DatabaseSettings(
        host=os.getenv("DB_HOST", DEFAULT_DB_HOST),
        port=int(os.getenv("DB_PORT", str(DEFAULT_DB_PORT))),
        user=os.getenv("DB_USER", DEFAULT_DB_USER),
        password=os.getenv("DB_PASSWORD", DEFAULT_DB_PASSWORD),
        database=os.getenv("DB_NAME", DEFAULT_DB_NAME),
    )

    extraid_db_settings = DatabaseSettings(
        host=os.getenv("EXTRAID_DB_HOST", DEFAULT_DB_HOST),
        port=int(os.getenv("EXTRAID_DB_PORT", str(DEFAULT_DB_PORT))),
        user=os.getenv("EXTRAID_DB_USER", DEFAULT_DB_USER),
        password=os.getenv("EXTRAID_DB_PASSWORD", DEFAULT_DB_PASSWORD),
        database=os.getenv("EXTRAID_DB_NAME", "extraid"),
    )

    yookassa_shop_id = os.getenv("YOOKASSA_SHOP_ID", "")
    yookassa_secret_key = os.getenv("YOOKASSA_SECRET_KEY", "")
    yookassa_test_mode = os.getenv("YOOKASSA_TEST_MODE", "true").lower() == "true"
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
    return Settings(
        bot_token=bot_token,
        webapp_url=os.getenv("WEBAPP_URL", DEFAULT_WEBAPP_URL),
        extra_shop_url=os.getenv("EXTRA_SHOP_URL", DEFAULT_EXTRA_SHOP_URL),
        environment=os.getenv("ENVIRONMENT", "development"),
        database=db_settings,
        extraid_database=extraid_db_settings,
        web_host=os.getenv("WEBAPP_HOST", "0.0.0.0"),
        web_port=int(os.getenv("WEBAPP_PORT", "8081")),
        yookassa=yookassa_settings,
        rustore=rustore_settings,
        rustore_app_url=os.getenv("RUSTORE_APP_URL", DEFAULT_RUSTORE_APP_URL),
        payment_provider_order=os.getenv("PAYMENT_PROVIDER_ORDER", DEFAULT_PAYMENT_PROVIDER_ORDER),
        stars_rate_rub=stars_rate_rub,
        stars_markup=stars_markup,
        stars_test_mode=stars_test_mode,
        jwt_secret=os.getenv("JWT_SECRET", "dev_secret_change_in_production!"),
        jwt_expiry_days=int(os.getenv("JWT_EXPIRY_DAYS", "30")),
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
    )
