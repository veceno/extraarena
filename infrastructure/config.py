from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # Поднимаемся на уровень выше (корень проекта)
ENV_FILE = BASE_DIR / ".env"
DEFAULT_BOT_TOKEN = "7486089212:AAFCc70OihXGVbiseIgrgLRIy4QBBRAT-KI"
DEFAULT_WEBAPP_URL = "https://digitally-upbeat-chow.cloudpub.ru/"
DEFAULT_STARS_RATE_RUB = 1.5  # руб. за 1 Telegram Star
DEFAULT_STARS_MARKUP = 1.2    # коэффициент наценки (20%)
DEFAULT_STARS_TEST_MODE = False

# Настройки матчмейкинга
MM_TROPHY_LIMIT_CLASSIC = 300  # лимит трофеев для мгновенного PvE
MM_BOT_TIMEOUT = 15  # таймаут поиска реального игрока (секунды)
DECK_SIZE = 9  # размер колоды (1 герой + 8 обычных карт)

# Профили сложности ботов с ONNX-моделями
BOT_DIFFICULTY_PROFILES = {
    "lite": {
        "model_path": "ai/models/OnlyVersusRandomBiggest.onnx",
        "obs_dim": 621,
        "temperature_range": (1.2, 1.8),  # Высокая температура = случайность
    },
    "easy": {
        "model_path": "ai/models/OnlyVersusRandomBiggest.onnx",
        "obs_dim": 621,
        "temperature_range": (1.2, 1.8),
    },
    "medium": {
        "model_path": "ai/models/extra-lr-v3-medium.onnx",
        "obs_dim": 997,
        "temperature_range": (0.6, 0.6),  # Средняя температура
    },
    "hard": {
        "model_path": "ai/models/extra-lr-v3-max.onnx",
        "obs_dim": 997,
        "temperature_range": (0.1, 0.1),  # Низкая температура = жадность
    },
    "max": {
        "model_path": "ai/models/extra-lr-v3-max.onnx",
        "obs_dim": 997,
        "temperature_range": (0.1, 0.1),
    },
}

# Тиры прогрессии для динамического расчёта трофеев и монет
TROPHY_TIERS = {
    "novice": {
        "range": range(0, 301),
        "win": (25, 35),        # Награда за победу (трофеи)
        "loss": (3, 7),         # Штраф за поражение (трофеи)
        "coin_range": (10, 15), # Награда монетами
    },
    "student": {
        "range": range(301, 701),
        "win": (20, 30),
        "loss": (5, 10),
        "coin_range": (15, 25),
    },
    "advanced": {
        "range": range(701, 2501),
        "win": (20, 25),
        "loss": (15, 25),
        "coin_range": (25, 40),
    },
    "advanced_2": {
        "range": range(2501, 5001),
        "win": (15, 20),
        "loss": (20, 25),
        "coin_range": (40, 65),
    },
    "pro": {
        "range": range(5001, 8001),
        "win": (10, 18),
        "loss": (25, 30),
        "coin_range": (70, 100),
    },
    "master": {
        "range": range(8001, 100000),
        "win": (10, 15),
        "loss": (27, 30),
        "coin_range": (120, 180),
    },
}

# Дефолтные настройки PostgreSQL для локальной БД
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
    environment: str = "development"
    database: DatabaseSettings | None = None
    web_host: str = "0.0.0.0"
    web_port: int = 8081
    yookassa: YooKassaSettings | None = None
    stars_rate_rub: float = DEFAULT_STARS_RATE_RUB
    stars_markup: float = DEFAULT_STARS_MARKUP
    stars_test_mode: bool = True


def _load_env() -> None:
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env()
    
    # Используем значения из .env, если заданы, иначе дефолтные для локальной БД
    db_settings = DatabaseSettings(
        host=os.getenv("DB_HOST", DEFAULT_DB_HOST),
        port=int(os.getenv("DB_PORT", str(DEFAULT_DB_PORT))),
        user=os.getenv("DB_USER", DEFAULT_DB_USER),
        password=os.getenv("DB_PASSWORD", DEFAULT_DB_PASSWORD),
        database=os.getenv("DB_NAME", DEFAULT_DB_NAME),
    )
    
    # Настройки YooKassa
    yookassa_shop_id = os.getenv("YOOKASSA_SHOP_ID", "")
    yookassa_secret_key = os.getenv("YOOKASSA_SECRET_KEY", "")
    yookassa_test_mode = os.getenv("YOOKASSA_TEST_MODE", "true").lower() == "true"
    
    yookassa_settings = None
    if yookassa_shop_id and yookassa_secret_key:
        yookassa_settings = YooKassaSettings(
            shop_id=yookassa_shop_id,
            secret_key=yookassa_secret_key,
            test_mode=yookassa_test_mode
        )
    
    stars_rate_rub = float(os.getenv("STARS_RATE_RUB", DEFAULT_STARS_RATE_RUB))
    stars_markup = float(os.getenv("STARS_MARKUP", DEFAULT_STARS_MARKUP))
    stars_test_mode = os.getenv("STARS_TEST_MODE", str(DEFAULT_STARS_TEST_MODE)).lower() in {"1", "true", "yes"}

    return Settings(
        bot_token=os.getenv("BOT_TOKEN", DEFAULT_BOT_TOKEN),
        webapp_url=os.getenv("WEBAPP_URL", DEFAULT_WEBAPP_URL),
        environment=os.getenv("ENVIRONMENT", "development"),
        database=db_settings,
        web_host=os.getenv("WEBAPP_HOST", "0.0.0.0"),
        web_port=int(os.getenv("WEBAPP_PORT", "8081")),
        yookassa=yookassa_settings,
        stars_rate_rub=stars_rate_rub,
        stars_markup=stars_markup,
        stars_test_mode=stars_test_mode,
    )


