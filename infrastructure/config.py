from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
DEFAULT_WEBAPP_URL = "https://digitally-upbeat-chow.cloudpub.ru/"
DEFAULT_EXTRA_SHOP_URL = "https://digitally-upbeat-chow.cloudpub.ru/"
DEFAULT_STARS_RATE_RUB = 1.5
DEFAULT_STARS_MARKUP = 1.2
DEFAULT_STARS_TEST_MODE = False

MM_TROPHY_LIMIT_CLASSIC = 300
MM_BOT_TIMEOUT = 15
DECK_SIZE = 9
MAX_FREE_DECK_PRESETS = 3
MAX_TOTAL_DECK_PRESETS = 5

BOT_DIFFICULTY_PROFILES = {
    "lite": {"model_path": "ai/models/OnlyVersusRandomBiggest.onnx", "obs_dim": 621, "temperature_range": (1.2, 1.8)},
    "easy": {"model_path": "ai/models/OnlyVersusRandomBiggest.onnx", "obs_dim": 621, "temperature_range": (1.2, 1.8)},
    "medium": {"model_path": "ai/models/extra-lr-v3-medium.onnx", "obs_dim": 997, "temperature_range": (0.6, 0.6)},
    "hard": {"model_path": "ai/models/extra-lr-v3-max.onnx", "obs_dim": 997, "temperature_range": (0.1, 0.1)},
    "max": {"model_path": "ai/models/extra-lr-v3-max.onnx", "obs_dim": 997, "temperature_range": (0.1, 0.1)},
}

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

    yookassa_settings = None
    if yookassa_shop_id and yookassa_secret_key:
        yookassa_settings = YooKassaSettings(
            shop_id=yookassa_shop_id,
            secret_key=yookassa_secret_key,
            test_mode=yookassa_test_mode,
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
    )
