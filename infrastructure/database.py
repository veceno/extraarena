from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Optional, Dict, TYPE_CHECKING

# asyncpg требуется только для реальной работы с БД.
# В юнит-тестах модуль может отсутствовать, поэтому подменяем импорт,
# чтобы логика карт и статы могли импортироваться без установки asyncpg.
try:  # pragma: no cover - ветка с отсутствием asyncpg проверяется непрямо
    import asyncpg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - альтернативный путь при локальных тестах
    asyncpg = None  # type: ignore

try:
    import bcrypt  # type: ignore
except ModuleNotFoundError:
    bcrypt = None  # type: ignore

from infrastructure.config import DECK_SIZE, MAX_FREE_DECK_PRESETS, MAX_TOTAL_DECK_PRESETS, DatabaseSettings, get_league_by_trophies_fn, LEAGUE_CONFIG
from infrastructure.notifications import (
    NOTIFICATION_DEFAULTS,
    NOTIFICATION_SETTING_BY_CATEGORY,
    classify_generator_event,
    choose_reminder_payload,
    format_notification_message,
)


# Версию схемы повышаем при изменении структуры таблиц
SCHEMA_VERSION = 39
BOT_USER_ID_MIN = 900_000_000
BOT_USER_ID_MAX = 8_999_999_999_999
NOTIFICATION_DELIVERY_MODES = {"app_then_telegram", "app_only", "telegram_only"}
DEFAULT_NOTIFICATION_DELIVERY_MODE = "app_then_telegram"

RUNTIME_FEATURE_DEFAULTS: dict[str, bool] = {
    "shop": True,
    "collection": True,
    "squads": True,
    "training": True,
    "friendly": True,
    "classic": True,
    "extra_arena": True,
}

RUNTIME_SETTINGS_DEFAULTS: dict[str, Any] = {
    "maintenance_mode": {"enabled": False},
    "feature_availability": dict(RUNTIME_FEATURE_DEFAULTS),
    "disabled_card_ids": [],
}

DEFAULT_EXTRA_PASS_SEASON: dict[str, Any] = {
    "slug": "arena-rift",
    "name": "Разлом Арены",
    "subtitle": "45 этапов | звезды за бои",
    "description": "Забирай награды по этапам. ExtraPass открывает вторую дорожку, Ultra добавляет финал.",
    "season_number": 1,
    "status": "active",
    "auto_switch": True,
    "preset_key": "default",
    "max_stars": 45,
    "free_track_type": "bp_free",
    "pass_track_type": "bp_premium",
    "ultra_track_type": "bp_ultra",
    "pass_end_position": 40,
    "ultra_start_position": 41,
    "theme": {},
}

SQUAD_SETTINGS_DEFAULTS: dict[str, Any] = {
    "squad_creation_policy": "beta_free",
    "squad_weekly_cbrp_enabled": True,
    "squad_seasonal_cbrp_enabled": False,
    "squad_clan_boost_token_multiplier": 1.2,
    "squad_creator_passive_tax_pct": 0.15,
    "squad_weekly_delta_divisor": 25,
    "squad_weekly_personal_tokens_divisor": 100,
    "squad_weekly_treasury_tokens_divisor": 150,
    "squad_seasonal_cbrp_divisor": 50,
    "squad_seasonal_personal_tokens_divisor": 200,
    "squad_seasonal_treasury_tokens_divisor": 300,
    "squad_upgrades": {
        "boost": {
            "title": "BOOST",
            "levels": [
                {"level": 1, "cost": 1200, "boost": True},
            ],
        },
        "member_slots": {
            "title": "Слоты участников",
            "levels": [
                {"level": 1, "cost": 500, "slots_added": 5},
                {"level": 2, "cost": 1000, "slots_added": 5},
                {"level": 3, "cost": 2000, "slots_added": 5},
            ],
        },
        "cbrp_boost": {
            "title": "Буст CBRP",
            "levels": [
                {"level": 1, "cost": 800, "cbrp_multiplier": 1.05},
                {"level": 2, "cost": 1600, "cbrp_multiplier": 1.10},
                {"level": 3, "cost": 3200, "cbrp_multiplier": 1.15},
            ],
        },
        "customization": {
            "title": "Кастомизация",
            "levels": [
                {"level": 1, "cost": 400, "unlock": "avatar_banner"},
            ],
        },
    },
    "squad_personal_rewards": [
        {"id": "title_squadmate", "name": "Титул «Сквадмейт»", "rarity": "common", "cost": 100, "kind": "cosmetic", "cosmetic_slug": "title_squadmate", "auto_equip": True},
        {"id": "title_squad_business", "name": "Титул «Сквад это бизнес»", "rarity": "rare", "cost": 250, "kind": "cosmetic", "cosmetic_slug": "title_squad_business", "auto_equip": True},
        {"id": "title_cbrp_hunter", "name": "Титул «CBRP Hunter»", "rarity": "epic", "cost": 600, "kind": "cosmetic", "cosmetic_slug": "title_cbrp_hunter", "auto_equip": True},
    ],
    "squad_rewards": {
        "battle_win": {"cbrp": 15, "personal_tokens": 3, "treasury_tokens": 2},
        "battle_loss": {"cbrp": 5, "personal_tokens": 1, "treasury_tokens": 1},
        "case_open": {
            "tiers": {
                "1": {"cbrp": 5, "personal_tokens": 1, "treasury_tokens": 0},
                "2": {"cbrp": 10, "personal_tokens": 2, "treasury_tokens": 1},
                "3": {"cbrp": 18, "personal_tokens": 3, "treasury_tokens": 2},
                "4": {"cbrp": 30, "personal_tokens": 5, "treasury_tokens": 3},
                "5": {"cbrp": 45, "personal_tokens": 8, "treasury_tokens": 5},
            }
        },
        "card_upgrade": {
            "levels": [
                {"min": 2, "max": 3, "cbrp": 5, "personal_tokens": 1, "treasury_tokens": 0},
                {"min": 4, "max": 6, "cbrp": 12, "personal_tokens": 2, "treasury_tokens": 1},
                {"min": 7, "max": 9, "cbrp": 25, "personal_tokens": 4, "treasury_tokens": 2},
                {"min": 10, "max": 10, "cbrp": 50, "personal_tokens": 8, "treasury_tokens": 5},
            ]
        },
        "new_card": {"cbrp": 8, "personal_tokens": 1, "treasury_tokens": 1},
        "new_epic_plus_card_bonus": {"cbrp": 10, "personal_tokens": 1, "treasury_tokens": 1},
    },
}

# Таблица ростов статов по редкости (урон/хп растут одинаково)
RARITY_STATS: dict[str, float] = {
    "common": 0.10,
    "rare": 0.10,
    "superrare": 0.10,
    "epic": 0.11,
    "legendary": 0.12,
    "mythic": 0.12,
    "limited": 0.13,
    "divine": 0.15,
    "unique": 0.15,
}


def _ensure_asyncpg() -> None:
    """
    Проверяем доступность asyncpg перед любыми действиями с базой.
    Так тесты, которые не используют БД, получают понятное сообщение,
    а продовый код по-прежнему требует установленную зависимость.
    """
    if asyncpg is None:
        raise ImportError(
            "asyncpg не установлен. Добавьте зависимость (pip install asyncpg) "
            "или подставьте тестовый двойник Database."
        )


_CARD_MECHANICS_DESC: dict[int, str] = {
    3:  "Аура: усиливает атаку всех союзных существ на доске",
    4:  "Отражает 2 урона обратно атакующему при каждом получении урона",
    5:  "Броня: уменьшает входящий урон на случайное значение от 1 до 3",
    6:  "Восстанавливает 1 HP в конце каждого своего хода",
    7:  "Даёт дополнительную ману и увеличивает максимум маны в начале боя",
    8:  "Наносит прямой урон выбранной вражеской цели (юниту или герою)",
    10: "Наносит урон всем вражеским существам на доске",
    11: "Замораживает выбранного врага: цель пропускает следующую атаку",
    12: "Крадёт до 2 маны у противника и отдаёт её владельцу карты",
    13: "Мгновенно уничтожает выбранное вражеское существо (игнорирует щиты и броню)",
    14: "При выходе на поле лечит своего героя на 2 HP",
    15: "При выходе на поле наносит 1 урон случайному врагу",
    16: "Игнорирует Провокацию: может атаковать любую цель",
    17: "Одноразовый щит: полностью блокирует первый полученный урон и исчезает",
    18: "Броня: уменьшает каждый входящий урон ровно на 1",
    19: "При выходе на поле замораживает выбранного врага",
    20: "Уничтожает выбранного союзника и прибавляет его атаку и здоровье к своим",
    21: "При выходе выбор: нанести 3 урона вражеской цели ИЛИ получить одноразовый щит",
    22: "При выходе замораживает всех вражеских существ на доске",
    23: "При атаке наносит дополнительный урон соседним существам противника",
    24: "Вечный щит: блокирует весь входящий урон и никогда не исчезает",
    25: "При атаке по существу: если цель выжила — мгновенно убивает её (HP → 0)",
    26: "При выходе разыгрывает случайное заклинание с масштабированием",
    30: "Провокация: противник обязан атаковать это существо в первую очередь",
    32: "Рывок: может атаковать в тот же ход, когда был выставлен на доску",
    33: "При нанесении урона герой владельца восстанавливает HP на величину урона",
    34: "После смерти наносит 3 урона всем вражеским существам и вражескому герою",
    35: "При выходе на поле лечит выбранного союзника на 5 HP",
    36: "При выходе на поле лечит выбранного союзника на 3 HP",
    39: "Провокация: противник обязан атаковать это существо в первую очередь",
}


def _normalize_mechanics(raw: Any) -> list[str]:
    """Безопасно приводим поле mechanics к списку строк."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def is_simplified_levelup(card_obj: Any) -> bool:
    if isinstance(card_obj, dict):
        return bool(card_obj.get("simplified_levelup", False))
    return bool(getattr(card_obj, "simplified_levelup", False))


def get_card_max_level(card_obj: Any) -> int:
    return 2 if is_simplified_levelup(card_obj) else 10


def get_effective_card_level(card_obj: Any, level: int) -> int:
    safe_level = max(1, int(level or 1))
    return min(safe_level, get_card_max_level(card_obj))


def get_upgrade_cost_level(card_obj: Any, level: int) -> int | None:
    effective_level = get_effective_card_level(card_obj, level)
    max_level = get_card_max_level(card_obj)
    if effective_level >= max_level:
        return None
    return 9 if is_simplified_levelup(card_obj) else effective_level


def _scaled_card_stats(card_obj: Any, level: int) -> dict[str, Any]:
    from core.converter import card_from_db

    if isinstance(card_obj, dict):
        payload = dict(card_obj)
    else:
        payload = {
            "id": getattr(card_obj, "id", 0),
            "name": getattr(card_obj, "name", "Unknown"),
            "description": getattr(card_obj, "description", ""),
            "rarity": getattr(card_obj, "rarity", "common"),
            "power": getattr(card_obj, "power", 0),
            "mana_cost": getattr(card_obj, "mana_cost", 0),
            "base_attack": getattr(card_obj, "base_attack", 0),
            "base_hp": getattr(card_obj, "base_hp", 0),
            "mechanics": getattr(card_obj, "mechanics", []),
            "card_type": getattr(card_obj, "card_type", "warrior"),
            "mechanics_desc": getattr(card_obj, "mechanics_desc", ""),
            "simplified_levelup": getattr(card_obj, "simplified_levelup", False),
        }

    effective_level = get_effective_card_level(payload, level)
    card = card_from_db(payload, level=effective_level)
    rarity_raw = payload.get("rarity", "") or ""
    rarity = str(rarity_raw).lower()
    growth = 0.0 if payload.get("card_type") == "hero" else RARITY_STATS.get(rarity, 0.10)
    max_level = get_card_max_level(payload)
    return {
        "attack": card.attack,
        "hp": card.hp,
        "mana": card.mana_cost,
        "mechanics": list(card.mechanics),
        "growth": growth,
        "level": card.level,
        "max_level": max_level,
        "is_max_level": card.level >= max_level,
    }


def _json_safe(value: Any) -> Any:
    """Convert DB scalar/container values into JSON-serializable values."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _normalize_poll_options(raw: Any) -> list[dict[str, Any]]:
    """Normalize poll options to non-empty choices with stable numeric ids."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []

    options: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("label") or "").strip()
            raw_id = item.get("id", index)
        else:
            text = str(item or "").strip()
            raw_id = index
        if not text:
            continue
        try:
            option_id = int(raw_id)
        except (TypeError, ValueError):
            option_id = index
        options.append({"id": option_id, "text": text})
    return options


def _parse_poll_expires_at(value: Any) -> datetime:
    """Parse poll expiry values accepted by the API into an aware datetime."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            parsed = datetime.now(timezone.utc) + timedelta(days=7)
        else:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def calculate_card_stats(card_obj: Any, level: int) -> dict[str, Any]:
    """
    Рассчитываем текущее значение атаки/хп/маны для карты.
    Использует тот же scaling-путь, что и арена.
    """
    return _scaled_card_stats(card_obj, level)


@dataclass
class Card:
    """Модель карты в Python с новыми статами."""

    id: int
    name: str
    description: str | None
    rarity: str
    power: int
    mana_cost: int
    base_attack: int
    base_hp: int
    mechanics: list[str]
    card_type: str = 'warrior'  # Подтип карты (warrior, mage, archer и т.д.)
    # level_multiplier берем из конфигурации редкости, чтобы рост статов был единообразным
    level_multiplier: float = 0.10
    image_file_id: str | None = None
    created_by: int | None = None
    created_at: datetime | None = None
    mechanics_desc: str | None = None
    simplified_levelup: bool = False

    @classmethod
    def from_row(cls, row: Any) -> "Card":
        """Формируем Card из asyncpg.Record или словаря, сразу подтягивая множитель роста по редкости."""
        data = dict(row)
        rarity_raw = (data.get("rarity") or "").lower()
        growth = RARITY_STATS.get(rarity_raw, 0.10)
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            rarity=data.get("rarity"),
            power=data.get("power", 0),
            mana_cost=data.get("mana_cost", 3),
            base_attack=data.get("base_attack", 100),
            base_hp=data.get("base_hp", 100),
            mechanics=_normalize_mechanics(data.get("mechanics")),
            card_type=data.get("card_type", "warrior"),
            level_multiplier=growth,
            image_file_id=data.get("image_file_id"),
            created_by=data.get("created_by"),
            created_at=data.get("created_at"),
            mechanics_desc=data.get("mechanics_desc"),
            simplified_levelup=bool(data.get("simplified_levelup", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Отдаем сериализованный словарь (удобно для JSON)."""
        return asdict(self)

    def get_current_stats(self, level: int) -> dict[str, Any]:
        """
        Высчитываем актуальные атак/хп/ману для боевого движка.
        Использует тот же scaling-путь, что и арена.
        """
        return _scaled_card_stats(self, level)




class Database:
    """Класс для работы с PostgreSQL через asyncpg."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self._pool: Optional[asyncpg.Pool] = None
        self.schema_changed_last_run: bool = False
        self.schema_last_updated: Optional[datetime] = None

    async def connect(self) -> None:
        """Создать пул подключений к БД."""
        # Проверяем, что asyncpg доступен, иначе даем явную ошибку о зависимости.
        _ensure_asyncpg()
        self._pool = await asyncpg.create_pool(
            host=self.settings.host,
            port=self.settings.port,
            user=self.settings.user,
            password=self.settings.password,
            database=self.settings.database,
            min_size=1,
            max_size=10,
        )

    async def close(self) -> None:
        """Закрыть пул подключений."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute(self, query: str, *args) -> str:
        """Выполнить SQL-запрос без возврата результата."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        """Выполнить SELECT-запрос и вернуть список записей."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Выполнить SELECT-запрос и вернуть одну запись."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Optional[Any]:
        """Выполнить SELECT-запрос и вернуть одно значение."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute_sql(self, query: str, *args) -> str:
        """
        Алиас для execute с более говорящим названием, полезен в сид-скриптах/миграциях.
        Добавлен отдельно, чтобы явно обозначать намерение выполнить произвольный SQL.
        """
        return await self.execute(query, *args)

    async def init_schema(self) -> bool:
        """Инициализировать схему БД (создать/обновить таблицы при необходимости)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        schema_changed = False

        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                id SMALLINT PRIMARY KEY DEFAULT 1,
                version INT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        row = await self.fetchrow("SELECT version, updated_at FROM schema_version WHERE id = 1")
        current_version = row["version"] if row else None
        self.schema_last_updated = row["updated_at"] if row else None

        users_changed = await self._ensure_users_table()
        profiles_changed = await self._ensure_profiles_table()
        news_changed = await self._ensure_news_table()
        settings_changed = await self._ensure_user_settings_table()
        promocodes_changed = await self._ensure_promocodes_table()
        user_mail_changed = await self._ensure_user_mail_table()
        cards_changed = await self._ensure_cards_table()
        user_cards_changed = await self._ensure_user_cards_table()
        deck_presets_changed = await self._ensure_deck_presets_table()
        battle_results_changed = await self._ensure_battle_results_table()
        cooldowns_changed = await self._ensure_cooldowns_table()
        notifications_changed = await self._ensure_notifications_table()
        notification_outbox_changed = await self._ensure_notification_outbox_table()
        notification_schedules_changed = await self._ensure_notification_schedules_table()
        push_devices_changed = await self._ensure_push_devices_table()
        global_chat_changed = await self._ensure_global_chat_table()
        community_posts_changed = await self._ensure_community_posts_table()
        post_likes_changed = await self._ensure_post_likes_table()
        reward_tracks_changed = await self._ensure_reward_tracks_table()
        claimed_rewards_changed = await self._ensure_claimed_rewards_table()
        seasons_changed = await self._ensure_seasons_table()
        shop_sets_changed = await self._ensure_shop_sets_table()
        ruble_products_changed = await self._ensure_ruble_products_table()
        payments_changed = await self._ensure_payments_table()
        checkout_sessions_changed = await self._ensure_payment_checkout_sessions_table()
        friend_invites_changed = await self._ensure_friend_invites_table()
        friend_requests_changed = await self._ensure_friend_requests_table()
        generator_changed = await self._ensure_generator_state_table()
        economy_events_changed = await self._ensure_economy_events_table()
        user_sessions_changed = await self._ensure_user_sessions_table()
        onboarding_events_changed = await self._ensure_onboarding_events_table()
        battle_summary_changed = await self._ensure_battle_summary_table()
        battle_actions_changed = await self._ensure_battle_actions_table()
        admin_actions_changed = await self._ensure_admin_account_actions_table()
        cosmetics_changed = await self._ensure_cosmetic_tables()
        user_cases_changed = await self._ensure_user_cases_table()
        match_mode_overrides_changed = await self._ensure_match_mode_overrides_table()
        user_roles_changed = await self._ensure_user_roles_table()
        clans_changed = await self._ensure_clans_table()
        clan_members_changed = await self._ensure_clan_members_table()
        clan_requests_changed = await self._ensure_clan_join_requests_table()
        clan_activity_changed = await self._ensure_clan_activity_table()
        clan_upgrades_changed = await self._ensure_clan_upgrades_table()
        game_settings_changed = await self._ensure_game_settings_table()
        squad_cbrp_events_changed = await self._ensure_squad_cbrp_events_table()
        squad_trophy_snapshots_changed = await self._ensure_squad_trophy_snapshots_table()
        squad_shop_purchases_changed = await self._ensure_squad_shop_purchases_table()
        await self._repair_squad_shop_cosmetic_purchases()
        community_votes_changed = await self._ensure_community_votes_table()
        community_polls_changed = await self._ensure_community_polls_table()
        community_submissions_changed = await self._ensure_community_submissions_table()
        schema_changed = (
            (current_version != SCHEMA_VERSION)
            or users_changed
            or profiles_changed
            or news_changed
            or settings_changed
            or promocodes_changed
            or user_mail_changed
            or cards_changed
            or user_cards_changed
            or deck_presets_changed
            or battle_results_changed
            or cooldowns_changed
            or notifications_changed
            or notification_outbox_changed
            or notification_schedules_changed
            or push_devices_changed
            or global_chat_changed
            or community_posts_changed
            or post_likes_changed
            or reward_tracks_changed
            or claimed_rewards_changed
            or seasons_changed
            or shop_sets_changed
            or ruble_products_changed
            or payments_changed
            or checkout_sessions_changed
            or friend_invites_changed
            or friend_requests_changed
            or generator_changed
            or economy_events_changed
            or user_sessions_changed
            or onboarding_events_changed
            or battle_summary_changed
            or battle_actions_changed
            or admin_actions_changed
            or cosmetics_changed
            or user_cases_changed
            or match_mode_overrides_changed
            or user_roles_changed
            or clans_changed
            or clan_members_changed
            or clan_requests_changed
            or clan_activity_changed
            or clan_upgrades_changed
            or game_settings_changed
            or squad_cbrp_events_changed
            or squad_trophy_snapshots_changed
            or squad_shop_purchases_changed
            or community_votes_changed
            or community_polls_changed
            or community_submissions_changed
        )

        # Обновляем референсные данные для новой боевой системы
        await self._seed_game_defaults()

        if schema_changed:
            await self.execute(
                """
                INSERT INTO schema_version (id, version, updated_at)
                VALUES (1, $1, NOW())
                ON CONFLICT (id) DO UPDATE
                SET version = EXCLUDED.version,
                    updated_at = NOW()
                """,
                SCHEMA_VERSION,
            )
            row = await self.fetchrow("SELECT updated_at FROM schema_version WHERE id = 1")
            self.schema_last_updated = row["updated_at"] if row else datetime.now(timezone.utc)

        self.schema_changed_last_run = schema_changed
        return schema_changed

    async def ensure_user(
        self,
        *,
        user_id: int,
        username: Optional[str],
        first_name: Optional[str],
        last_name: Optional[str],
    ) -> bool:
        """Гарантировать наличие записи пользователя и профиля. Возвращает True, если создана новая запись."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        created = False

        exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)

        if exists:
            await self.execute(
                """
                UPDATE users
                SET username = $2,
                    first_name = $3,
                    last_name = $4,
                    updated_at = NOW()
                WHERE user_id = $1
                """,
                user_id,
                username,
                first_name,
                last_name,
            )
        else:
            await self.execute(
                """
                INSERT INTO users (user_id, username, first_name, last_name, league)
                VALUES ($1, $2, $3, $4, 1)
                """,
                user_id,
                username,
                first_name,
                last_name,
            )
            created = True
            # Создаем 2 пресета по умолчанию для нового пользователя
            await self._ensure_user_has_default_presets(user_id)
            
            # Выдаем все карты стартовой редкости новому пользователю
            try:
                start_cards_result = await self.grant_start_cards(user_id)
                if not start_cards_result.get("success"):
                    raise RuntimeError(start_cards_result.get("error", "unknown_error"))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Не удалось выдать стартовые карты пользователю {user_id}: {e}")
            
            # Выдаем приветственные бонусы: 50 гемов и 200 монет
            try:
                await self.execute(
                    "UPDATE users SET gems = gems + 50, coins = coins + 200 WHERE user_id = $1",
                    user_id
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Не удалось выдать приветственные бонусы пользователю {user_id}: {e}")

            # Выдаем стартовые косметические предметы
            try:
                await self.grant_starter_cosmetics(user_id)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Не удалось выдать стартовую косметику пользователю {user_id}: {e}"
                )

        await self.execute(
            """
            INSERT INTO profiles (user_id, img, title)
            VALUES ($1, '', 'Игрок')
            ON CONFLICT (user_id) DO NOTHING
            """,
            user_id,
        )

        # Создаем настройки по умолчанию
        await self.execute(
            """
            INSERT INTO user_settings (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
            """,
            user_id,
        )

        return created

    async def get_next_bot_id(self) -> int:
        """
        Рассчитываем следующий безопасный идентификатор для ботов, начиная с 900000000.
        Держим диапазон отдельно от Telegram и app-only synthetic игроков.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        next_id = await self.fetchval(
            """
            SELECT COALESCE(MAX(user_id), $1::bigint - 1) + 1
            FROM users
            WHERE user_id >= $1::bigint
              AND user_id <= $2::bigint
              AND COALESCE(is_bot, FALSE) = TRUE
            """,
            BOT_USER_ID_MIN,
            BOT_USER_ID_MAX,
        )
        next_id = int(next_id or BOT_USER_ID_MIN)
        if next_id > BOT_USER_ID_MAX:
            raise RuntimeError("bot_user_id_range_exhausted")
        return next_id

    async def get_user_profile(self, user_id: int) -> Optional[asyncpg.Record]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        record = await self.fetchrow(
            """
            SELECT
                u.user_id,
                u.username,
                u.first_name,
                u.extra_pass,
                u.trophies,
                u.max_trophies,
                u.league,
                COALESCE(u.keys, 0) as keys,
                u.gems,
                u.coins,
                u.squad_id,
                u.status,
                u.reg_date,
                COALESCE(u.stars, 0) as stars,
                COALESCE(u.energy, 5) as energy,
                u.energy_cd,
                COALESCE(u.season, 0) as season,
                u.extra_pass_expires_at,
                u.extra_account_id,
                COALESCE(p.img, '') as img,
                COALESCE(p.title, 'Игрок') as title,
                p.custom_nickname,
                COALESCE(p.nickname_changed, FALSE) as nickname_changed,
                equipped_avatar.asset_path as equipped_avatar_url
            FROM users u
            LEFT JOIN profiles p ON p.user_id = u.user_id
            LEFT JOIN user_equipped_cosmetics uec ON uec.user_id = u.user_id AND uec.item_type = 'avatar'
            LEFT JOIN cosmetic_items equipped_avatar ON equipped_avatar.id = uec.cosmetic_id AND equipped_avatar.item_type = 'avatar'
            WHERE u.user_id = $1
            """,
            user_id,
        )
        
        # Если профиля нет, создаем его
        if record and not record.get("title"):
            await self.execute(
                """
                INSERT INTO profiles (user_id, img, title)
                VALUES ($1, '', 'Игрок')
                ON CONFLICT (user_id) DO UPDATE
                SET title = COALESCE(profiles.title, 'Игрок')
                """,
                user_id,
            )
            # Обновляем record
            record = await self.fetchrow(
                """
                SELECT
                    u.user_id,
                    u.username,
                    u.first_name,
                    u.extra_pass,
                    u.trophies,
                    u.max_trophies,
                    u.league,
                    COALESCE(u.keys, 0) as keys,
                    u.gems,
                    u.coins,
                    u.squad_id,
                    u.status,
                    u.reg_date,
                    COALESCE(u.stars, 0) as stars,
                    COALESCE(u.energy, 5) as energy,
                    u.energy_cd,
                    COALESCE(u.season, 0) as season,
                    u.extra_pass_expires_at,
                    COALESCE(p.img, '') as img,
                    COALESCE(p.title, 'Игрок') as title,
                    p.custom_nickname,
                    COALESCE(p.nickname_changed, FALSE) as nickname_changed
                FROM users u
                LEFT JOIN profiles p ON p.user_id = u.user_id
                WHERE u.user_id = $1
                """,
                user_id,
            )
        
        return record

    async def get_random_display_names(self, count: int, exclude_user_id: int) -> list[str]:
        """
        Возвращает набор случайных отображаемых имен (custom_nickname/имя/username).
        Исключаем переданного пользователя и ботов, чтобы не брать их никнеймы для новых ботов.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT COALESCE(p.custom_nickname, u.first_name, u.username, 'Бот') AS display_name
            FROM profiles p
            JOIN users u ON u.user_id = p.user_id
            WHERE p.user_id <> $1
              AND COALESCE(u.is_bot, FALSE) = FALSE
            ORDER BY RANDOM()
            LIMIT $2
            """,
            exclude_user_id,
            count,
        )
        return [row["display_name"] for row in rows]

    async def get_random_users_with_avatars(self, count: int, exclude_user_id: int) -> list[dict[str, Any]]:
        """
        Возвращает случайных пользователей с их именами и аватарками (img).
        Исключаем переданного пользователя и ботов, чтобы не брать их данные для новых ботов.
        
        Returns:
            list[dict]: Список словарей с ключами "display_name" и "img"
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT 
                COALESCE(p.custom_nickname, u.first_name, u.username, 'Бот') AS display_name,
                p.img
            FROM profiles p
            JOIN users u ON u.user_id = p.user_id
            WHERE p.user_id <> $1
              AND COALESCE(u.is_bot, FALSE) = FALSE
            ORDER BY RANDOM()
            LIMIT $2
            """,
            exclude_user_id,
            count,
        )
        return [{"display_name": row["display_name"], "img": row.get("img")} for row in rows]

    async def get_random_donor_profile(self, exclude_user_id: int) -> dict[str, Any] | None:
        """
        Возвращает профиль случайного реального игрока-донора для создания бота.
        Включает имя, аватар, колоду и средний уровень карт.
        
        Returns:
            dict с ключами: display_name, img, deck_ids, avg_level, trophies
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        # Находим донора с готовой колодой (минимум 5 карт в пресете)
        donor = await self.fetchrow(
            """
            SELECT 
                u.user_id,
                COALESCE(p.custom_nickname, u.first_name, u.username, 'Донор') AS display_name,
                p.img,
                u.trophies,
                dp.card_slot_1, dp.card_slot_2, dp.card_slot_3, dp.card_slot_4, dp.card_slot_5,
                dp.card_slot_6, dp.card_slot_7, dp.card_slot_8, dp.card_slot_9
            FROM users u
            JOIN profiles p ON p.user_id = u.user_id
            LEFT JOIN deck_presets dp ON dp.user_id = u.user_id AND dp.preset_number = 1
            WHERE u.user_id <> $1
              AND COALESCE(u.is_bot, FALSE) = FALSE
              AND u.trophies >= 0
              AND (
                  dp.card_slot_1 IS NOT NULL OR dp.card_slot_2 IS NOT NULL OR 
                  dp.card_slot_3 IS NOT NULL OR dp.card_slot_4 IS NOT NULL OR 
                  dp.card_slot_5 IS NOT NULL
              )
            ORDER BY RANDOM()
            LIMIT 1
            """,
            exclude_user_id,
        )
        
        if not donor:
            return None
        
        # Собираем колоду из непустых слотов
        deck_ids = []
        for i in range(1, 10):
            card_id = donor.get(f"card_slot_{i}")
            if card_id:
                deck_ids.append(int(card_id))
        
        # Получаем средний уровень карт донора
        avg_level_result = await self.fetchval(
            """
            SELECT COALESCE(AVG(level), 1)
            FROM user_cards
            WHERE user_id = $1 AND card_id = ANY($2::bigint[])
            """,
            donor["user_id"],
            deck_ids,
        )
        avg_level = int(avg_level_result or 1)
        
        return {
            "display_name": donor["display_name"],
            "img": donor.get("img"),
            "deck_ids": deck_ids,
            "avg_level": avg_level,
            "trophies": donor.get("trophies", 0),
        }

    async def get_ai_deck_preset(self, trophies: int) -> list[int]:
        """
        Подбирает подходящий набор карт для бота на основе трофеев игрока.
        Берем ближайший по верхней границе пресет; если не найден, берем самый верхний.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        row = await self.fetchrow(
            """
            SELECT card_ids
            FROM ai_presets
            WHERE trophy_range_max >= $1
            ORDER BY trophy_range_max ASC
            LIMIT 1
            """,
            trophies,
        )

        if not row:
            row = await self.fetchrow(
                """
                SELECT card_ids
                FROM ai_presets
                ORDER BY trophy_range_max DESC
                LIMIT 1
                """
            )

        if not row or not row.get("card_ids"):
            return []

        # asyncpg возвращает list-like объект, приводим к обычному списку int
        return [int(card_id) for card_id in row["card_ids"]]

    async def create_or_update_bot_profile(
        self,
        *,
        bot_id: int,
        display_name: str,
        trophies: int,
        level: int,
        deck_ids: list[int] | None = None,
        avatar_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Создает или обновляет бот-профиль в таблицах users/profiles и кладет его колоду в deck_presets.
        - users: сохраняем трофеи и флаг is_bot
        - profiles: фиксируем никнейм и аватарку для отображения в матчах/истории
        - deck_presets: опционально сохраняем выбранные карты для повторного использования
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        safe_trophies = max(0, int(trophies))
        safe_level = max(1, int(level))

        # Обновляем или создаем запись пользователя с флагом бота
        await self.execute(
            """
            INSERT INTO users (user_id, username, first_name, last_name, trophies, max_trophies, league, is_bot, status)
            VALUES ($1, NULL, $2, NULL, $3, $3, $4, TRUE, 'active')
            ON CONFLICT (user_id) DO UPDATE
            SET trophies = EXCLUDED.trophies,
                max_trophies = GREATEST(users.max_trophies, EXCLUDED.max_trophies),
                league = EXCLUDED.league,
                is_bot = TRUE,
                first_name = EXCLUDED.first_name,
                updated_at = NOW(),
                status = 'active'
            """,
            bot_id,
            display_name,
            safe_trophies,
            get_league_by_trophies_fn(safe_trophies),
        )

        # Поддерживаем профиль для отображения имени и аватарки
        await self.execute(
            """
            INSERT INTO profiles (user_id, custom_nickname, nickname_changed, title, img)
            VALUES ($1, $2, TRUE, 'Бот', $3)
            ON CONFLICT (user_id) DO UPDATE
            SET custom_nickname = EXCLUDED.custom_nickname,
                nickname_changed = TRUE,
                title = EXCLUDED.title,
                img = EXCLUDED.img
            """,
            bot_id,
            display_name,
            avatar_url,
        )

        # Сохраняем колоду бота в первый пресет (preset_number = 1), если она передана
        if deck_ids:
            # КРИТИЧНО: Гарантируем ровно 9 слотов (защита от IndexError при slots[8])
            slots = (list(deck_ids) + [None] * 9)[:9]
            await self.execute(
                """
                INSERT INTO deck_presets (
                    user_id, preset_name, preset_number,
                    card_slot_1, card_slot_2, card_slot_3, card_slot_4, card_slot_5,
                    card_slot_6, card_slot_7, card_slot_8, card_slot_9, used_by_bot
                )
                VALUES ($1, $2, 1, $3, $4, $5, $6, $7, $8, $9, $10, $11, TRUE)
                ON CONFLICT (user_id, preset_number) DO UPDATE
                SET preset_name = EXCLUDED.preset_name,
                    card_slot_1 = EXCLUDED.card_slot_1,
                    card_slot_2 = EXCLUDED.card_slot_2,
                    card_slot_3 = EXCLUDED.card_slot_3,
                    card_slot_4 = EXCLUDED.card_slot_4,
                    card_slot_5 = EXCLUDED.card_slot_5,
                    card_slot_6 = EXCLUDED.card_slot_6,
                    card_slot_7 = EXCLUDED.card_slot_7,
                    card_slot_8 = EXCLUDED.card_slot_8,
                    card_slot_9 = EXCLUDED.card_slot_9,
                    used_by_bot = EXCLUDED.used_by_bot,
                    updated_at = NOW()
                """,
                bot_id,
                "Бот-колода",
                slots[0],
                slots[1],
                slots[2],
                slots[3],
                slots[4],
                slots[5],
                slots[6],
                slots[7],
                slots[8],
            )

        return {
            "bot_id": bot_id,
            "trophies": safe_trophies,
            "level": safe_level,
            "deck_ids": list(deck_ids) if deck_ids else [],
        }

    async def change_nickname(self, user_id: int, new_nickname: str, cost_gems: int = 0) -> dict[str, Any]:
        """Изменить никнейм пользователя. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        
        # Проверяем текущий статус смены никнейма
        profile_record = await self.fetchrow(
            "SELECT nickname_changed, custom_nickname FROM profiles WHERE user_id = $1",
            user_id,
        )
        
        nickname_changed = profile_record["nickname_changed"] if profile_record else False
        is_first_change = not nickname_changed
        
        # Если это не первая смена, проверяем наличие гемов
        if not is_first_change and cost_gems > 0:
            user_gems = await self.fetchval("SELECT gems FROM users WHERE user_id = $1", user_id)
            if not user_gems or user_gems < cost_gems:
                return {"success": False, "error": "insufficient_gems", "required": cost_gems, "current": user_gems or 0}
            
            # Списываем гемы
            await self.execute(
                "UPDATE users SET gems = gems - $1 WHERE user_id = $2",
                cost_gems, user_id
            )
        
        # Сохраняем никнейм (вставка с обновлением при конфликте)
        await self.execute(
            """
            INSERT INTO profiles (user_id, custom_nickname, nickname_changed)
            VALUES ($2, $1, TRUE)
            ON CONFLICT (user_id)
            DO UPDATE SET custom_nickname = EXCLUDED.custom_nickname,
                          nickname_changed = TRUE
            """,
            new_nickname, user_id
        )
        
        return {"success": True, "is_first_change": is_first_change}

    async def delete_user(self, user_id: int) -> bool:
        """
        Полностью удалить пользователя и все его данные из БД.
        
        Удаляет:
        - Карты пользователя (user_cards)
        - Колоды (deck_presets)
        - Настройки (user_settings)
        - Почту (user_mail)
        - Сообщения в чате (global_chat)
        - Посты в коммьюнити (community_posts)
        - Лайки (post_likes)
        - Профиль (profiles)
        - Пользователя (users)
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            # Проверяем существование пользователя
            exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)
            if not exists:
                return False

            # Удаляем карты пользователя
            await self.execute("DELETE FROM user_cards WHERE user_id = $1", user_id)
            
            # Удаляем колоды пользователя
            await self.execute("DELETE FROM deck_presets WHERE user_id = $1", user_id)
            
            # Удаляем настройки пользователя
            await self.execute("DELETE FROM user_settings WHERE user_id = $1", user_id)
            
            # Удаляем почту пользователя
            await self.execute("DELETE FROM user_mail WHERE user_id = $1", user_id)

            # Удаляем сообщения в чате
            await self.execute("DELETE FROM global_chat WHERE user_id = $1", user_id)

            # Удаляем лайки пользователя
            await self.execute("DELETE FROM post_likes WHERE user_id = $1", user_id)

            # Удаляем посты коммьюнити (и лайки к ним)
            posts = await self.fetch("SELECT id FROM community_posts WHERE author_id = $1", user_id)
            for post in posts:
                await self.execute("DELETE FROM post_likes WHERE post_id = $1", post["id"])
            await self.execute("DELETE FROM community_posts WHERE author_id = $1", user_id)
            
            # Удаляем профиль пользователя
            await self.execute("DELETE FROM profiles WHERE user_id = $1", user_id)
            
            # Удаляем пользователя (это должно быть последним, так как могут быть CASCADE ограничения)
            await self.execute("DELETE FROM users WHERE user_id = $1", user_id)
            
            return True
        except Exception as e:
            # Логируем ошибку для отладки
            import logging
            logging.getLogger(__name__).error(f"Ошибка при удалении пользователя {user_id}: {e}", exc_info=True)
            return False

    async def get_statistics(self) -> dict[str, int]:
        """Получить агрегированную статистику по игрокам."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        players = await self.fetchval("SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE")
        extra_pass_active = await self.fetchval(
            "SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE AND extra_pass = 'active'"
        )
        total_trophies = await self.fetchval(
            "SELECT COALESCE(SUM(trophies), 0) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE"
        )
        max_trophies_global = await self.fetchval(
            "SELECT COALESCE(MAX(max_trophies), 0) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE"
        )

        return {
            "players": int(players or 0),
            "extra_pass_active": int(extra_pass_active or 0),
            "total_trophies": int(total_trophies or 0),
            "max_trophies_global": int(max_trophies_global or 0),
        }

    async def create_news_entry(
        self,
        *,
        author_id: int,
        text: str,
        category: str,
        button_text: Optional[str],
        button_url: Optional[str],
        photo_file_id: Optional[str],
    ) -> None:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        await self.execute(
            """
            INSERT INTO news (author_id, text, category, button_text, button_url, photo_file_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            author_id,
            text,
            category,
            button_text,
            button_url,
            photo_file_id,
        )

    async def get_recent_news(self, limit: int = 5) -> list[asyncpg.Record]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        return await self.fetch(
            """
            SELECT id,
                   author_id,
                   text,
                   category,
                   button_text,
                   button_url,
                   photo_file_id,
                   created_at
            FROM news
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )

    async def _ensure_users_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.users')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    extra_pass TEXT NOT NULL DEFAULT 'inactive',
                    trophies INTEGER NOT NULL DEFAULT 0,
                    max_trophies INTEGER NOT NULL DEFAULT 0,
                    league INTEGER NOT NULL DEFAULT 1,
                    keys INTEGER NOT NULL DEFAULT 0,
                    gems INTEGER NOT NULL DEFAULT 0,
                    coins INTEGER NOT NULL DEFAULT 0,
                    squad_id BIGINT NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    reg_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("users")

        changed |= await self._add_column_if_missing(
            "users", columns, "extra_pass TEXT NOT NULL DEFAULT 'inactive'"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "trophies INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "max_trophies INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "league INTEGER NOT NULL DEFAULT 1"
        )
        # Миграция: заполнить league по текущим трофеям
        await self.execute(
            """
            UPDATE users SET league = CASE
                WHEN trophies >= 9000 THEN 10
                WHEN trophies >= 7500 THEN 9
                WHEN trophies >= 6000 THEN 8
                WHEN trophies >= 4500 THEN 7
                WHEN trophies >= 3000 THEN 6
                WHEN trophies >= 2000 THEN 5
                WHEN trophies >= 1200 THEN 4
                WHEN trophies >= 600  THEN 3
                WHEN trophies >= 300  THEN 2
                ELSE 1
            END
            """
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "keys INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "stars INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "energy INTEGER NOT NULL DEFAULT 5"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "energy_cd TIMESTAMPTZ"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "gems INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "coins INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "squad_id BIGINT NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "status TEXT NOT NULL DEFAULT 'active'"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "is_bot BOOLEAN NOT NULL DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "primary_deck INTEGER DEFAULT NULL"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "season INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "reg_date TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "extra_pass_expires_at TIMESTAMPTZ"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "is_banned BOOLEAN NOT NULL DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "ban_reason TEXT"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "banned_until TIMESTAMPTZ"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "warnings_count INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "extra_account_id UUID"
        )
        changed |= await self._add_column_if_missing(
            "users", columns, "auth_source TEXT NOT NULL DEFAULT 'telegram'"
        )

        if not await self._constraint_exists("users", "users_status_check"):
            await self.execute(
                """
                ALTER TABLE users
                ADD CONSTRAINT users_status_check
                CHECK (status IN ('active', 'warn', 'banned'))
                """
            )
            changed = True

        await self.execute(
            """
            CREATE OR REPLACE FUNCTION set_updated_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )

        await self.execute("DROP TRIGGER IF EXISTS users_set_updated_at ON users")
        await self.execute(
            """
            CREATE TRIGGER users_set_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW
            EXECUTE PROCEDURE set_updated_at();
            """
        )

        return changed

    async def _ensure_profiles_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.profiles')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE profiles (
                    user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                    img TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT 'Игрок'
                )
                """
            )
            changed = True

        columns = await self._get_columns("profiles")
        changed |= await self._add_column_if_missing(
            "profiles",
            columns,
            "img TEXT NOT NULL DEFAULT ''",
        )
        changed |= await self._add_column_if_missing(
            "profiles",
            columns,
            "custom_nickname TEXT",
        )
        changed |= await self._add_column_if_missing(
            "profiles",
            columns,
            "nickname_changed BOOLEAN NOT NULL DEFAULT FALSE",
        )
        changed |= await self._add_column_if_missing(
            "profiles",
            columns,
            "title TEXT NOT NULL DEFAULT 'Игрок'",
        )


        # Обновляем старые записи с "Новичок" на "Игрок"
        await self.execute(
            "UPDATE profiles SET title = 'Игрок' WHERE title = 'Новичок'"
        )

        return changed

    async def _ensure_user_settings_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.user_settings')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE user_settings (
                    user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                    notif_cases BOOLEAN NOT NULL DEFAULT true,
                    notif_daily_rewards BOOLEAN NOT NULL DEFAULT true,
                    notif_game_invites BOOLEAN NOT NULL DEFAULT true,
                    notif_friend_requests BOOLEAN NOT NULL DEFAULT true,
                    notif_events BOOLEAN NOT NULL DEFAULT true,
                    notif_news BOOLEAN NOT NULL DEFAULT true,
                    notif_dice BOOLEAN NOT NULL DEFAULT false,
                    notif_generator BOOLEAN NOT NULL DEFAULT true,
                    notif_shop BOOLEAN NOT NULL DEFAULT false,
                    notif_reminders BOOLEAN NOT NULL DEFAULT true,
                    notif_squad_member_role BOOLEAN NOT NULL DEFAULT true,
                    notif_squad_new_member BOOLEAN NOT NULL DEFAULT true,
                    notif_squad_disbanded BOOLEAN NOT NULL DEFAULT true,
                    notif_squad_boost BOOLEAN NOT NULL DEFAULT true,
                    notif_extra_arena_modifiers BOOLEAN NOT NULL DEFAULT true,
                    ads_enabled BOOLEAN NOT NULL DEFAULT true,
                    sound_music BOOLEAN NOT NULL DEFAULT true,
                    sound_sfx BOOLEAN NOT NULL DEFAULT true,
                    social_block_friend_requests BOOLEAN NOT NULL DEFAULT false,
                    notification_delivery_mode TEXT NOT NULL DEFAULT 'app_then_telegram',
                    welcome_shown BOOLEAN NOT NULL DEFAULT false,
                    starter_pack_used BOOLEAN NOT NULL DEFAULT false,
                    particles_rotation_cards JSONB,
                    particles_rotation_date DATE,
                    particles_purchased_today JSONB DEFAULT '[]',
                    wins_since_last_case INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True
        else:
            # Создаем настройки для всех существующих пользователей
            await self.execute(
                """
                INSERT INTO user_settings (user_id)
                SELECT user_id FROM users
                WHERE user_id NOT IN (SELECT user_id FROM user_settings)
                """
            )
            
            # Добавляем колонку notif_dice, если её нет
            columns = await self._get_columns("user_settings")
            changed |= await self._add_column_if_missing(
                "user_settings", columns, 
                "notif_dice BOOLEAN NOT NULL DEFAULT false"
            )
            # Добавляем колонку notif_generator, если её нет
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notif_generator BOOLEAN NOT NULL DEFAULT true"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notif_shop BOOLEAN NOT NULL DEFAULT false"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notif_reminders BOOLEAN NOT NULL DEFAULT true"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notif_squad_member_role BOOLEAN NOT NULL DEFAULT true"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notif_squad_new_member BOOLEAN NOT NULL DEFAULT true"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notif_squad_disbanded BOOLEAN NOT NULL DEFAULT true"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notif_squad_boost BOOLEAN NOT NULL DEFAULT true"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notif_extra_arena_modifiers BOOLEAN NOT NULL DEFAULT true"
            )
            # Добавляем колонку welcome_shown, если её нет
            changed |= await self._add_column_if_missing(
                "user_settings", columns, 
                "welcome_shown BOOLEAN NOT NULL DEFAULT false"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "starter_pack_used BOOLEAN NOT NULL DEFAULT false"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "particles_rotation_cards JSONB"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "particles_rotation_date DATE"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "particles_purchased_today JSONB DEFAULT '[]'"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "wins_since_last_case INTEGER NOT NULL DEFAULT 0"
            )
            changed |= await self._add_column_if_missing(
                "user_settings", columns,
                "notification_delivery_mode TEXT NOT NULL DEFAULT 'app_then_telegram'"
            )

        return changed

    async def get_user_settings(self, user_id: int) -> Optional[asyncpg.Record]:
        """Получить настройки пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        return await self.fetchrow(
            "SELECT * FROM user_settings WHERE user_id = $1", user_id
        )

    async def update_user_settings(self, user_id: int, **kwargs) -> None:
        """Обновить настройки пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        valid_keys = {
            "notif_cases", "notif_daily_rewards", "notif_game_invites",
            "notif_friend_requests", "notif_events", "notif_news", "notif_dice",
            "notif_generator", "notif_shop", "notif_reminders",
            "notif_squad_member_role", "notif_squad_new_member",
            "notif_squad_disbanded", "notif_squad_boost",
            "notif_extra_arena_modifiers",
            "notification_delivery_mode",
            "ads_enabled", "sound_music", "sound_sfx", "social_block_friend_requests",
            "welcome_shown",
            "starter_pack_used", "particles_rotation_cards", "particles_rotation_date",
            "particles_purchased_today", "wins_since_last_case",
        }

        updates = []
        values = []
        for key, value in kwargs.items():
            if key in valid_keys:
                if key == "notification_delivery_mode":
                    value = str(value or "").strip()
                    if value not in NOTIFICATION_DELIVERY_MODES:
                        continue
                updates.append(f"{key} = ${len(values) + 1}")
                values.append(value)

        if not updates:
            return

        values.append(user_id)
        query = f"""
            UPDATE user_settings
            SET {', '.join(updates)}, updated_at = NOW()
            WHERE user_id = ${len(values)}
        """
        await self.execute(query, *values)

    async def update_user_trophies(self, user_id: int, delta: int) -> dict[str, Any]:
        """
        Обновить трофеи пользователя с защитой от отрицательных значений.
        
        Args:
            user_id: ID пользователя
            delta: Изменение трофеев (может быть положительным или отрицательным)
        
        Returns:
            dict с новыми значениями trophies, max_trophies и league
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        
        # КРИТИЧНО: Используем GREATEST для защиты от отрицательных трофеев
        # max_trophies обновляется только если новое значение больше текущего
        # league автоматически пересчитывается по новым трофеям
        result = await self.fetchrow(
            """
            UPDATE users 
            SET 
                trophies = GREATEST(0, trophies + $1),
                max_trophies = GREATEST(max_trophies, GREATEST(0, trophies + $1)),
                league = CASE 
                    WHEN GREATEST(0, trophies + $1) >= 9000 THEN 10
                    WHEN GREATEST(0, trophies + $1) >= 7500 THEN 9
                    WHEN GREATEST(0, trophies + $1) >= 6000 THEN 8
                    WHEN GREATEST(0, trophies + $1) >= 4500 THEN 7
                    WHEN GREATEST(0, trophies + $1) >= 3000 THEN 6
                    WHEN GREATEST(0, trophies + $1) >= 2000 THEN 5
                    WHEN GREATEST(0, trophies + $1) >= 1200 THEN 4
                    WHEN GREATEST(0, trophies + $1) >= 600  THEN 3
                    WHEN GREATEST(0, trophies + $1) >= 300  THEN 2
                    ELSE 1
                END,
                updated_at = NOW()
            WHERE user_id = $2
            RETURNING trophies, max_trophies, league
            """,
            delta, user_id
        )
        
        if result:
            return {
                "trophies": result["trophies"],
                "max_trophies": result["max_trophies"],
                "league": result["league"]
            }
        
        return {"trophies": 0, "max_trophies": 0, "league": 1}

    async def get_user_info(self, user_id: int) -> Optional[dict[str, Any]]:
        """
        Получить базовую информацию о пользователе.
        Используется в server.py для проверки extra_pass и трофеев.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        
        record = await self.fetchrow(
            """
            SELECT 
                user_id, 
                trophies, 
                max_trophies, 
                league,
                coins, 
                gems, 
                extra_pass,
                primary_deck,
                energy,
                energy_cd
            FROM users 
            WHERE user_id = $1
            """,
            user_id
        )
        
        if record:
            return dict(record)
        return None

    async def get_unread_mail_count(self, user_id: int) -> int:
        """
        Получить количество непрочитанных писем пользователя.
        Используется в server.py для API /api/mail/unread-count.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        
        try:
            record = await self.fetchrow(
                """
                SELECT COUNT(*) as unread_count
                FROM user_mail
                WHERE user_id = $1 AND is_read = FALSE
                """,
                user_id
            )
            
            if record:
                return int(record["unread_count"] or 0)
            return 0
        except Exception:
            # Если таблица mail не существует или ошибка БД, возвращаем 0
            return 0

    async def update_user_coins(self, user_id: int, delta: int) -> dict[str, Any]:
        """
        Обновить монеты пользователя с защитой от отрицательных значений.
        
        Args:
            user_id: ID пользователя
            delta: Изменение монет (может быть положительным или отрицательным)
        
        Returns:
            dict с новым значением coins
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        
        result = await self.fetchrow(
            """
            UPDATE users 
            SET 
                coins = GREATEST(0, coins + $1),
                updated_at = NOW()
            WHERE user_id = $2
            RETURNING coins
            """,
            delta, user_id
        )
        
        if result:
            return {"coins": result["coins"]}
        
        return {"coins": 0}

    async def add_gems(self, user_id: int, amount: int) -> dict[str, Any]:
        """Добавить гемы пользователю."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        result = await self.fetchrow(
            """
            UPDATE users
            SET gems = GREATEST(0, COALESCE(gems, 0) + $1),
                updated_at = NOW()
            WHERE user_id = $2
            RETURNING gems
            """,
            amount, user_id
        )
        return {"gems": result["gems"] if result else 0}

    async def set_primary_deck(self, user_id: int, preset_number: int | None) -> dict[str, Any]:
        """Установить основную колоду игрока с проверкой валидности."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")

        if preset_number is not None:
            # H1: verify preset exists, belongs to user, and has at least one card
            preset = await self.fetchrow(
                """
                SELECT card_slot_1, card_slot_2, card_slot_3, card_slot_4, card_slot_5,
                       card_slot_6, card_slot_7, card_slot_8, card_slot_9
                FROM deck_presets
                WHERE user_id = $1 AND preset_number = $2
                """,
                user_id, preset_number,
            )
            if not preset:
                return {"success": False, "error": "preset_not_found", "message": "Пресет не найден"}
            has_cards = any(preset.get(f"card_slot_{i}") is not None for i in range(1, 10))
            if not has_cards:
                return {"success": False, "error": "empty_preset", "message": "Нельзя сделать пустой пресет основным"}

        await self.execute(
            "UPDATE users SET primary_deck = $1, updated_at = NOW() WHERE user_id = $2",
            preset_number, user_id
        )

        # Patch 6: invalidate cache on mutation
        deck_cache = getattr(self, "deck_presets_cache", None)
        if isinstance(deck_cache, dict):
            deck_cache.pop(user_id, None)

        return {"success": True, "primary_deck": preset_number}

    async def _ensure_news_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.news')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE news (
                    id BIGSERIAL PRIMARY KEY,
                    author_id BIGINT NOT NULL,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'update',
                    button_text TEXT,
                    button_url TEXT,
                    photo_file_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("news")
        changed |= await self._add_column_if_missing(
            "news", columns, "author_id BIGINT NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "news", columns, "text TEXT NOT NULL DEFAULT ''"
        )
        changed |= await self._add_column_if_missing(
            "news", columns, "category TEXT NOT NULL DEFAULT 'update'"
        )
        changed |= await self._add_column_if_missing("news", columns, "button_text TEXT")
        changed |= await self._add_column_if_missing("news", columns, "button_url TEXT")
        changed |= await self._add_column_if_missing("news", columns, "photo_file_id TEXT")
        changed |= await self._add_column_if_missing(
            "news", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        if not await self._constraint_exists("news", "news_category_check"):
            await self.execute(
                """
                ALTER TABLE news
                ADD CONSTRAINT news_category_check
                CHECK (category IN ('update', 'event', 'important', 'interesting'))
                """
            )
            changed = True

        return changed

    async def _ensure_battle_results_table(self) -> bool:
        """Создать таблицу результатов боя."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.battle_results')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE battle_results (
                    id BIGSERIAL PRIMARY KEY,
                    match_id TEXT NOT NULL,
                    winner_id BIGINT,
                    loser_id BIGINT,
                    winner_score INTEGER DEFAULT 0,
                    loser_score INTEGER DEFAULT 0,
                    match_duration INTEGER DEFAULT 0,
                    match_type TEXT DEFAULT 'pvp',
                    p1_id BIGINT,
                    p2_id BIGINT,
                    p1_trophy_change INTEGER DEFAULT 0,
                    p2_trophy_change INTEGER DEFAULT 0,
                    turns_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("battle_results")
        changed |= await self._add_column_if_missing("battle_results", columns, "match_id TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("battle_results", columns, "winner_id BIGINT")
        changed |= await self._add_column_if_missing("battle_results", columns, "loser_id BIGINT")
        changed |= await self._add_column_if_missing("battle_results", columns, "winner_score INTEGER DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_results", columns, "loser_score INTEGER DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_results", columns, "match_duration INTEGER DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_results", columns, "match_type TEXT DEFAULT 'pvp'")
        changed |= await self._add_column_if_missing("battle_results", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("battle_results", columns, "p1_id BIGINT")
        changed |= await self._add_column_if_missing("battle_results", columns, "p2_id BIGINT")
        changed |= await self._add_column_if_missing("battle_results", columns, "p1_trophy_change INTEGER DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_results", columns, "p2_trophy_change INTEGER DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_results", columns, "turns_count INTEGER DEFAULT 0")

        # Создаем индексы для быстрого поиска
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'battle_results' AND indexname = 'battle_results_match_id_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX battle_results_match_id_idx ON battle_results(match_id)")
            changed = True

        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'battle_results' AND indexname = 'battle_results_created_at_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX battle_results_created_at_idx ON battle_results(created_at DESC)")
            changed = True

        unique_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'battle_results' AND indexname = 'battle_results_match_id_unique_idx'
            """
        )
        if not unique_exists:
            # Безопасная миграция: удаляем строки с пустым match_id и дедуплицируем дубли,
            # чтобы CREATE UNIQUE INDEX не упал на проде.
            dupes_deleted = await self.execute(
                """
                DELETE FROM battle_results
                WHERE match_id = ''
                   OR match_id IS NULL
                """
            )
            if int(dupes_deleted.split()[-1]) > 0:
                import logging
                logging.getLogger(__name__).warning(
                    "Cleaned %s rows with empty/null match_id before creating unique index", dupes_deleted
                )
            await self.execute(
                """
                DELETE FROM battle_results a
                USING battle_results b
                WHERE a.match_id = b.match_id
                  AND a.id < b.id
                """
            )
            await self.execute(
                "CREATE UNIQUE INDEX battle_results_match_id_unique_idx ON battle_results(match_id)"
            )
            changed = True

        return changed

    async def save_battle_result(self, **kwargs) -> None:
        """Сохранить результат боя. Идемпотентно: ON CONFLICT DO NOTHING."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        await self.execute(
            """
            INSERT INTO battle_results (match_id, winner_id, loser_id, winner_score, loser_score, match_duration, match_type, p1_id, p2_id, p1_trophy_change, p2_trophy_change, turns_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (match_id) DO NOTHING
            """,
            kwargs.get('match_id'),
            kwargs.get('winner_id'),
            kwargs.get('loser_id'),
            kwargs.get('winner_score', 0),
            kwargs.get('loser_score', 0),
            kwargs.get('match_duration', 0),
            kwargs.get('match_type', 'pvp'),
            kwargs.get('p1_id'),
            kwargs.get('p2_id'),
            kwargs.get('p1_trophy_change', 0),
            kwargs.get('p2_trophy_change', 0),
            kwargs.get('turns_count', 0)
        )

    @staticmethod
    def _resolve_legacy_opponent(user_id: int, row: dict) -> tuple[Any | None, int | None, int | None]:
        """
        Resolve (p1, p2, opponent_id) from a legacy battle_results row.
        Returns (p1, p2, opponent_id).  opponent_id is None if user is not a participant.
        Works for rows with p1/p2 filled AND for rows with null p1/p2 (via winner/loser).
        """
        p1 = row.get("p1_id")
        p2 = row.get("p2_id")
        w = row.get("winner_id")
        l = row.get("loser_id")
        if p1 is not None and p2 is not None:
            if p1 == user_id:
                return (p1, p2, p2)
            if p2 == user_id:
                return (p1, p2, p1)
        if w == user_id:
            return (None, None, l)
        if l == user_id:
            return (None, None, w)
        return (None, None, None)

    async def get_battle_history(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        """
        Унифицированная история боёв: battle_summary (canonical) + legacy battle_results.
        Сначала дедуплицирует legacy по match_id (newest per match), затем исключает
        match_id из summary, затем объединяет через UNION и делает глобальный
        ORDER BY created_at DESC LIMIT.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        # ---- Step 1: fetch ALL candidate rows (no LIMIT yet) ----
        summary_rows = await self.fetch(
            """
            SELECT match_id, p1_user_id, p2_user_id, winner_user_id, loser_user_id,
                   p1_trophy_change, p2_trophy_change, game_mode, match_type,
                   duration_seconds, turns_count, created_at
            FROM battle_summary
            WHERE p1_user_id = $1 OR p2_user_id = $1
            """,
            user_id,
        )

        summary_match_ids = {r["match_id"] for r in summary_rows}

        legacy_raw = await self.fetch(
            """
            SELECT match_id, p1_id, p2_id, winner_id, loser_id,
                   p1_trophy_change, p2_trophy_change, match_type,
                   match_duration, turns_count, created_at
            FROM battle_results
            WHERE match_id != ALL($2::text[])
              AND (
                  p1_id = $1 OR p2_id = $1
                  OR (p1_id IS NULL AND p2_id IS NULL AND (winner_id = $1 OR loser_id = $1))
              )
            """,
            user_id, summary_match_ids or [""],
        )

        # ---- Step 2: deduplicate legacy by match_id (keep newest) ----
        legacy_best: dict[str, Any] = {}
        for r in legacy_raw:
            mid = r["match_id"]
            if mid not in legacy_best:
                legacy_best[mid] = r
            else:
                cur_ts = legacy_best[mid]["created_at"]
                new_ts = r["created_at"]
                if new_ts > cur_ts:
                    legacy_best[mid] = r

        # ---- Step 3: resolve participants and collect opponent ids ----
        all_opponent_ids: set[int] = set()
        for r in summary_rows:
            opp = r["p1_user_id"] if r["p2_user_id"] == user_id else r["p2_user_id"]
            if opp:
                all_opponent_ids.add(opp)

        valid_legacy: list[Any] = []
        for r in legacy_best.values():
            _, _, opp_id = self._resolve_legacy_opponent(user_id, r)
            if opp_id is not None:
                valid_legacy.append(r)
                all_opponent_ids.add(opp_id)

        # ---- Step 4: load opponent info ----
        opponent_info: dict[int, dict[str, Any]] = {}
        if all_opponent_ids:
            opp_rows = await self.fetch(
                """
                SELECT u.user_id, u.first_name, u.username, u.is_bot, u.trophies,
                       p.custom_nickname, p.img,
                       equipped_avatar.asset_path AS equipped_avatar_url
                FROM users u
                LEFT JOIN profiles p ON p.user_id = u.user_id
                LEFT JOIN user_equipped_cosmetics uec ON uec.user_id = u.user_id AND uec.item_type = 'avatar'
                LEFT JOIN cosmetic_items equipped_avatar ON equipped_avatar.id = uec.cosmetic_id AND equipped_avatar.item_type = 'avatar'
                WHERE u.user_id = ANY($1)
                """,
                list(all_opponent_ids),
            )
            for row in opp_rows:
                is_bot = bool(row.get("is_bot", False))
                opponent_info[row["user_id"]] = {
                    "name": (row.get("custom_nickname") or row.get("first_name") or row.get("username") or "Игрок"),
                    "is_bot": is_bot,
                    "avatar_url": row.get("equipped_avatar_url") or row.get("img"),
                    "trophies": row.get("trophies") or 0,
                }

        # ---- Step 5: format all rows ----
        rows: list[dict[str, Any]] = []

        for r in summary_rows:
            opp_id = r["p1_user_id"] if r["p2_user_id"] == user_id else r["p2_user_id"]
            rows.append(self._format_battle_row(
                match_id=r["match_id"], player_id=user_id,
                p1_id=r["p1_user_id"], p2_id=r["p2_user_id"],
                winner_id=r.get("winner_user_id"), loser_id=r.get("loser_user_id"),
                p1_trophy_change=r.get("p1_trophy_change", 0),
                p2_trophy_change=r.get("p2_trophy_change", 0),
                game_mode=r.get("game_mode"), match_type=r.get("match_type"),
                duration_seconds=r.get("duration_seconds", 0),
                turns_count=r.get("turns_count", 0),
                created_at=r["created_at"],
                opponent_id=opp_id, opponent_info=opponent_info,
            ))

        for r in valid_legacy:
            p1, p2, opp_id = self._resolve_legacy_opponent(user_id, r)
            if opp_id is None:
                continue
            rows.append(self._format_battle_row(
                match_id=r["match_id"], player_id=user_id,
                p1_id=p1, p2_id=p2,
                winner_id=r.get("winner_id"), loser_id=r.get("loser_id"),
                p1_trophy_change=r.get("p1_trophy_change", 0),
                p2_trophy_change=r.get("p2_trophy_change", 0),
                game_mode=None, match_type=r.get("match_type"),
                duration_seconds=r.get("match_duration", 0),
                turns_count=r.get("turns_count", 0),
                created_at=r["created_at"],
                opponent_id=opp_id, opponent_info=opponent_info,
            ))

        # ---- Step 6: global sort + limit ----
        rows.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return rows[:limit]

    @staticmethod
    def _format_battle_row(
        *, match_id: str, player_id: int, p1_id, p2_id,
        winner_id, loser_id, p1_trophy_change: int, p2_trophy_change: int,
        game_mode, match_type, duration_seconds: int, turns_count: int,
        created_at, opponent_id, opponent_info: dict,
    ) -> dict[str, Any]:
        opp = opponent_info.get(opponent_id, {}) if opponent_id else {}
        if winner_id is None:
            result = "draw"
        elif winner_id == player_id:
            result = "win"
        else:
            result = "lose"
        if p1_id is not None and p2_id is not None:
            trophies_change = p1_trophy_change if p1_id == player_id else p2_trophy_change
        else:
            trophies_change = 0
        mode = game_mode or match_type or "classic"
        created_at_str = created_at.isoformat() if isinstance(created_at, datetime) else str(created_at)
        return {
            "battle_id": match_id,
            "opponent_id": opponent_id,
            "opponent_name": opp.get("name", "Игрок"),
            "opponent_avatar_url": opp.get("avatar_url"),
            "opponent_trophies": opp.get("trophies", 0),
            "opponent_is_bot": opp.get("is_bot", False),
            "result": result,
            "trophies_change": trophies_change,
            "mode": mode,
            "duration_seconds": duration_seconds or 0,
            "turns_count": turns_count or 0,
            "created_at": created_at_str,
        }

    async def get_public_player_card(self, user_id: int) -> dict | None:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        record = await self.fetchrow(
            """
            SELECT
                u.user_id,
                u.trophies,
                u.max_trophies,
                u.league,
                u.first_name,
                COALESCE(p.img, '') as img,
                COALESCE(p.title, 'Игрок') as title,
                COALESCE(p.custom_nickname, u.first_name, 'Игрок') as display_name,
                (
                    (SELECT COUNT(*) FROM battle_summary
                     WHERE p1_user_id = u.user_id OR p2_user_id = u.user_id)
                    +
                    (SELECT COUNT(*) FROM battle_results
                     WHERE match_id NOT IN (SELECT match_id FROM battle_summary)
                       AND (p1_id = u.user_id OR p2_id = u.user_id
                            OR (p1_id IS NULL AND p2_id IS NULL AND (winner_id = u.user_id OR loser_id = u.user_id))))
                ) as battle_count,
                (
                    (SELECT COUNT(*) FROM battle_summary
                     WHERE winner_user_id = u.user_id)
                    +
                    (SELECT COUNT(*) FROM battle_results
                     WHERE match_id NOT IN (SELECT match_id FROM battle_summary)
                       AND (
                           (p1_id = u.user_id AND winner_id = u.user_id)
                           OR (p2_id = u.user_id AND winner_id = u.user_id)
                           OR (p1_id IS NULL AND p2_id IS NULL AND winner_id = u.user_id)
                       ))
                ) as win_count
            FROM users u
            LEFT JOIN profiles p ON p.user_id = u.user_id
            WHERE u.user_id = $1
            """,
            user_id,
        )
        if not record:
            return None

        league = LEAGUE_CONFIG.get(record["league"], LEAGUE_CONFIG[1])

        return {
            "user_id": record["user_id"],
            "display_name": record["display_name"],
            "trophies": record["trophies"],
            "max_trophies": record["max_trophies"],
            "league": record["league"],
            "league_name": league["name"],
            "league_emoji": league["emoji"],
            "league_color": league["color"],
            "title": record["title"],
            "battle_count": record["battle_count"],
            "win_count": record["win_count"],
            "first_name": record["first_name"],
            "img": record["img"],
        }

    async def _ensure_cooldowns_table(self) -> bool:
        """Создать универсальную таблицу кулдаунов."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.cooldowns')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE cooldowns (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    cooldown_type TEXT NOT NULL,
                    cooldown_until TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(user_id, cooldown_type)
                )
                """
            )
            changed = True

        columns = await self._get_columns("cooldowns")
        changed |= await self._add_column_if_missing("cooldowns", columns, "user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("cooldowns", columns, "cooldown_type TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("cooldowns", columns, "cooldown_until TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("cooldowns", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        # Создаем индекс для быстрого поиска
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'cooldowns' AND indexname = 'cooldowns_user_type_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX cooldowns_user_type_idx ON cooldowns(user_id, cooldown_type)")
            changed = True

        # Создаем уникальный индекс
        unique_constraint_exists = await self._constraint_exists("cooldowns", "cooldowns_user_id_cooldown_type_key")
        if not unique_constraint_exists:
            await self.execute(
                """
                ALTER TABLE cooldowns
                ADD CONSTRAINT cooldowns_user_id_cooldown_type_key
                UNIQUE (user_id, cooldown_type)
                """
            )
            changed = True

        return changed

    async def _ensure_notifications_table(self) -> bool:
        """Создать таблицу уведомлений."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.notifications')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE notifications (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    notification_type TEXT NOT NULL,
                    sent BOOLEAN NOT NULL DEFAULT FALSE,
                    sent_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(user_id, notification_type)
                )
                """
            )
            changed = True

        columns = await self._get_columns("notifications")
        changed |= await self._add_column_if_missing("notifications", columns, "user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("notifications", columns, "notification_type TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("notifications", columns, "sent BOOLEAN NOT NULL DEFAULT FALSE")
        changed |= await self._add_column_if_missing("notifications", columns, "sent_at TIMESTAMPTZ")
        changed |= await self._add_column_if_missing("notifications", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        # Создаем индекс для быстрого поиска
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'notifications' AND indexname = 'notifications_user_type_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX notifications_user_type_idx ON notifications(user_id, notification_type)")
            changed = True

        # Создаем уникальный индекс
        unique_constraint_exists = await self._constraint_exists("notifications", "notifications_user_id_notification_type_key")
        if not unique_constraint_exists:
            await self.execute(
                """
                ALTER TABLE notifications
                ADD CONSTRAINT notifications_user_id_notification_type_key
                UNIQUE (user_id, notification_type)
                """
            )
            changed = True

        return changed

    async def check_dice_ready_notifications(self) -> list[dict[str, Any]]:
        """Проверить, кому нужно отправить уведомление о готовности кубика."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        # Используем универсальную систему уведомлений
        now = datetime.now(timezone.utc)

        # Находим пользователей, у которых кулдаун истек и уведомление не отправлено
        rows = await self.fetch(
            """
            SELECT DISTINCT c.user_id
            FROM cooldowns c
            INNER JOIN users u ON u.user_id = c.user_id
            LEFT JOIN notifications n ON n.user_id = c.user_id AND n.notification_type = 'dice'
            LEFT JOIN user_settings us ON us.user_id = c.user_id
            WHERE c.cooldown_type = 'dice'
              AND c.cooldown_until <= $1
              AND u.status = 'active'
              AND (n.sent IS NULL OR n.sent = FALSE)
              AND (us.notif_dice = TRUE OR us.notif_dice IS NULL)
            """,
            now
        )

        return [{"user_id": row["user_id"]} for row in rows]

    async def mark_dice_notification_sent(self, user_id: int) -> None:
        """Отметить, что уведомление о готовности кубика было отправлено."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        await self.execute(
            """
            INSERT INTO notifications (user_id, notification_type, sent, sent_at)
            VALUES ($1, 'dice', TRUE, NOW())
            ON CONFLICT (user_id, notification_type) DO UPDATE
            SET sent = TRUE, sent_at = NOW()
            """,
            user_id
        )

    async def _ensure_notification_outbox_table(self) -> bool:
        """Создать очередь личных Telegram-уведомлений."""
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.notification_outbox')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE notification_outbox (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    category TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    not_before_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    sent_at TIMESTAMPTZ
                )
                """
            )
            changed = True

        columns = await self._get_columns("notification_outbox")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "category TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "event_type TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "payload JSONB NOT NULL DEFAULT '{}'::jsonb")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "dedupe_key TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "status TEXT NOT NULL DEFAULT 'pending'")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "attempts INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "not_before_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("notification_outbox", columns, "sent_at TIMESTAMPTZ")
        await self.execute("CREATE UNIQUE INDEX IF NOT EXISTS notification_outbox_dedupe_key_idx ON notification_outbox(dedupe_key)")
        await self.execute("CREATE INDEX IF NOT EXISTS notification_outbox_pending_due_idx ON notification_outbox(status, not_before_at, created_at)")
        return changed

    async def _ensure_notification_schedules_table(self) -> bool:
        """Создать состояние персонального расписания уведомлений."""
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.notification_schedules')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE notification_schedules (
                    user_id BIGINT NOT NULL,
                    schedule_type TEXT NOT NULL,
                    next_send_at TIMESTAMPTZ NOT NULL,
                    last_sent_at TIMESTAMPTZ,
                    last_payload JSONB,
                    PRIMARY KEY (user_id, schedule_type)
                )
                """
            )
            changed = True

        columns = await self._get_columns("notification_schedules")
        changed |= await self._add_column_if_missing("notification_schedules", columns, "user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("notification_schedules", columns, "schedule_type TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("notification_schedules", columns, "next_send_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("notification_schedules", columns, "last_sent_at TIMESTAMPTZ")
        changed |= await self._add_column_if_missing("notification_schedules", columns, "last_payload JSONB")
        await self.execute("CREATE INDEX IF NOT EXISTS notification_schedules_due_idx ON notification_schedules(schedule_type, next_send_at)")
        return changed

    async def _ensure_push_devices_table(self) -> bool:
        """Store Android FCM tokens bound to app auth sessions."""
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.push_devices')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE push_devices (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    platform TEXT NOT NULL DEFAULT 'android',
                    token TEXT NOT NULL UNIQUE,
                    app_version TEXT,
                    device_label TEXT,
                    os_name TEXT,
                    os_version TEXT,
                    timezone TEXT,
                    utc_offset_minutes INTEGER,
                    enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    last_error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("push_devices")
        changed |= await self._add_column_if_missing("push_devices", columns, "user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("push_devices", columns, "platform TEXT NOT NULL DEFAULT 'android'")
        changed |= await self._add_column_if_missing("push_devices", columns, "token TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("push_devices", columns, "app_version TEXT")
        changed |= await self._add_column_if_missing("push_devices", columns, "device_label TEXT")
        changed |= await self._add_column_if_missing("push_devices", columns, "os_name TEXT")
        changed |= await self._add_column_if_missing("push_devices", columns, "os_version TEXT")
        changed |= await self._add_column_if_missing("push_devices", columns, "timezone TEXT")
        changed |= await self._add_column_if_missing("push_devices", columns, "utc_offset_minutes INTEGER")
        changed |= await self._add_column_if_missing("push_devices", columns, "enabled BOOLEAN NOT NULL DEFAULT TRUE")
        changed |= await self._add_column_if_missing("push_devices", columns, "last_error TEXT")
        changed |= await self._add_column_if_missing("push_devices", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("push_devices", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("push_devices", columns, "last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        await self.execute("CREATE UNIQUE INDEX IF NOT EXISTS push_devices_token_idx ON push_devices(token)")
        await self.execute("CREATE INDEX IF NOT EXISTS push_devices_user_idx ON push_devices(user_id, enabled)")
        return changed

    async def register_push_device(
        self,
        user_id: int,
        *,
        token: str,
        platform: str = "android",
        app_version: str | None = None,
        device_label: str | None = None,
        os_name: str | None = None,
        os_version: str | None = None,
        timezone: str | None = None,
        utc_offset_minutes: int | None = None,
    ) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        row = await self.fetchrow(
            """
            INSERT INTO push_devices (
                user_id, platform, token, app_version, device_label, os_name, os_version,
                timezone, utc_offset_minutes,
                enabled, last_error, created_at, updated_at, last_seen_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, TRUE, NULL, NOW(), NOW(), NOW())
            ON CONFLICT (token) DO UPDATE
            SET user_id = EXCLUDED.user_id,
                platform = EXCLUDED.platform,
                app_version = EXCLUDED.app_version,
                device_label = EXCLUDED.device_label,
                os_name = EXCLUDED.os_name,
                os_version = EXCLUDED.os_version,
                timezone = EXCLUDED.timezone,
                utc_offset_minutes = EXCLUDED.utc_offset_minutes,
                enabled = TRUE,
                last_error = NULL,
                updated_at = NOW(),
                last_seen_at = NOW()
            RETURNING id, user_id, platform, token, app_version, device_label, os_name, os_version,
                      timezone, utc_offset_minutes, enabled
            """,
            user_id,
            (platform or "android").lower(),
            token,
            app_version,
            device_label,
            os_name,
            os_version,
            timezone,
            utc_offset_minutes,
        )
        return dict(row) if row else {}

    async def unregister_push_device(self, user_id: int, *, token: str) -> bool:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        result = await self.execute(
            """
            UPDATE push_devices
            SET enabled = FALSE, updated_at = NOW()
            WHERE user_id = $1 AND token = $2
            """,
            user_id,
            token,
        )
        return result.endswith(" 1") if isinstance(result, str) else bool(result)

    async def get_push_devices(self, user_id: int, *, platform: str | None = "android") -> list[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        if platform:
            rows = await self.fetch(
                """
                SELECT id, user_id, platform, token, app_version, device_label, os_name, os_version,
                       timezone, utc_offset_minutes, enabled
                FROM push_devices
                WHERE user_id = $1 AND platform = $2 AND enabled = TRUE AND token <> ''
                ORDER BY last_seen_at DESC
                """,
                user_id,
                platform,
            )
        else:
            rows = await self.fetch(
                """
                SELECT id, user_id, platform, token, app_version, device_label, os_name, os_version,
                       timezone, utc_offset_minutes, enabled
                FROM push_devices
                WHERE user_id = $1 AND enabled = TRUE AND token <> ''
                ORDER BY last_seen_at DESC
                """,
                user_id,
            )
        return [dict(row) for row in rows]

    async def count_push_devices(self, *, platform: str | None = "android") -> int:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        if platform:
            value = await self.fetchval(
                """
                SELECT COUNT(*)
                FROM push_devices
                WHERE platform = $1 AND enabled = TRUE AND token <> ''
                """,
                platform,
            )
        else:
            value = await self.fetchval(
                """
                SELECT COUNT(*)
                FROM push_devices
                WHERE enabled = TRUE AND token <> ''
                """
            )
        return int(value or 0)

    async def get_push_devices_for_broadcast(
        self,
        *,
        platform: str | None = "android",
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        safe_limit = max(1, min(int(limit or 10000), 50000))
        if platform:
            rows = await self.fetch(
                """
                SELECT id, user_id, platform, token, app_version, device_label, os_name, os_version,
                       timezone, utc_offset_minutes, enabled
                FROM push_devices
                WHERE platform = $1 AND enabled = TRUE AND token <> ''
                ORDER BY last_seen_at DESC
                LIMIT $2
                """,
                platform,
                safe_limit,
            )
        else:
            rows = await self.fetch(
                """
                SELECT id, user_id, platform, token, app_version, device_label, os_name, os_version,
                       timezone, utc_offset_minutes, enabled
                FROM push_devices
                WHERE enabled = TRUE AND token <> ''
                ORDER BY last_seen_at DESC
                LIMIT $1
                """,
                safe_limit,
            )
        return [dict(row) for row in rows]

    async def mark_push_device_error(self, token: str, error: str, *, permanent: bool = False) -> None:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        await self.execute(
            """
            UPDATE push_devices
            SET last_error = $2,
                enabled = CASE WHEN $3 THEN FALSE ELSE enabled END,
                updated_at = NOW()
            WHERE token = $1
            """,
            token,
            (error or "")[:500],
            permanent,
        )

    async def is_notification_enabled(self, user_id: int, category: str) -> bool:
        setting = NOTIFICATION_SETTING_BY_CATEGORY.get(category)
        if not setting:
            return True
        if setting not in {
            "notif_generator", "notif_shop", "notif_reminders",
            "notif_squad_member_role", "notif_squad_new_member",
            "notif_squad_disbanded", "notif_squad_boost",
            "notif_extra_arena_modifiers",
        }:
            return True
        row = await self.fetchrow(f"SELECT {setting} FROM user_settings WHERE user_id = $1", user_id)
        if row is None:
            return bool(NOTIFICATION_DEFAULTS.get(setting, True))
        value = row[setting]
        if value is None:
            return bool(NOTIFICATION_DEFAULTS.get(setting, True))
        return bool(value)

    async def get_notification_delivery_mode(self, user_id: int) -> str:
        row = await self.fetchrow(
            "SELECT notification_delivery_mode FROM user_settings WHERE user_id = $1",
            user_id,
        )
        if row is None:
            return DEFAULT_NOTIFICATION_DELIVERY_MODE
        mode = str(row.get("notification_delivery_mode") or "").strip()
        if mode not in NOTIFICATION_DELIVERY_MODES:
            return DEFAULT_NOTIFICATION_DELIVERY_MODE
        return mode

    async def enqueue_notification(
        self,
        user_id: int,
        *,
        category: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> bool:
        """Поставить личное уведомление в очередь с учетом настроек пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        if not await self.is_notification_enabled(user_id, category):
            return False
        payload = payload or {}
        if dedupe_key is None:
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            dedupe_key = f"{user_id}:{category}:{event_type}:{raw}"
        row = await self.fetchrow(
            """
            INSERT INTO notification_outbox (user_id, category, event_type, payload, dedupe_key)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (dedupe_key) DO NOTHING
            RETURNING id
            """,
            user_id,
            category,
            event_type,
            json.dumps(payload, ensure_ascii=False),
            dedupe_key,
        )
        if row is None:
            return False

        await self._create_mail_from_notification(
            user_id=user_id,
            category=category,
            event_type=event_type,
            payload=payload,
        )
        return True

    async def _create_mail_from_notification(
        self,
        *,
        user_id: int,
        category: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Mirror durable service notifications into in-game mail."""
        subject_by_category = {
            "generator": "Генератор ключей",
            "shop": "Магазин",
            "reminders": "Напоминание арены",
            "squad_member_role": "Сквад",
            "squad_new_member": "Сквад",
            "squad_disbanded": "Сквад",
            "squad_boost": "Сквад",
        }
        icon_by_category = {
            "generator": "🔑",
            "shop": "🛒",
            "reminders": "⚔️",
            "squad_member_role": "👥",
            "squad_new_member": "👥",
            "squad_disbanded": "👥",
            "squad_boost": "⚡",
        }
        mail_category_by_notification = {
            "generator": "system",
            "shop": "news",
            "reminders": "event",
            "squad_member_role": "squad",
            "squad_new_member": "squad",
            "squad_disbanded": "squad",
            "squad_boost": "squad",
        }

        try:
            result = await self.create_mail(
                user_id=user_id,
                sender="Система",
                subject=subject_by_category.get(category, "Событие ExtraArena"),
                text=format_notification_message(event_type, payload),
                category=mail_category_by_notification.get(category, "system"),
                icon=icon_by_category.get(category, "📬"),
                attachments={"event_type": event_type, **(payload or {})},
            )
            if not result.get("success"):
                logging.getLogger(__name__).warning(
                    "notification mail failed: user_id=%s category=%s event_type=%s error=%s",
                    user_id,
                    category,
                    event_type,
                    result.get("error", "unknown"),
                )
        except Exception:
            logging.getLogger(__name__).warning(
                "notification mail failed: user_id=%s category=%s event_type=%s",
                user_id,
                category,
                event_type,
                exc_info=True,
            )

    async def fetch_pending_notifications(self, limit: int = 50) -> list[dict[str, Any]]:
        """Забрать пачку pending-уведомлений для отправки."""
        rows = await self.fetch(
            """
            WITH picked AS (
                SELECT id
                FROM notification_outbox
                WHERE status = 'pending'
                  AND attempts < 5
                  AND not_before_at <= NOW()
                ORDER BY not_before_at ASC, created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE notification_outbox n
            SET status = 'sending', attempts = attempts + 1
            FROM picked
            WHERE n.id = picked.id
            RETURNING n.id, n.user_id, n.category, n.event_type, n.payload, n.attempts
            """,
            limit,
        )
        return [dict(r) for r in rows]

    async def mark_notification_sent(self, notification_id: int) -> None:
        await self.execute(
            "UPDATE notification_outbox SET status = 'sent', sent_at = NOW() WHERE id = $1",
            notification_id,
        )

    async def mark_notification_failed(self, notification_id: int, *, permanent: bool = False) -> None:
        await self.execute(
            """
            UPDATE notification_outbox
            SET status = CASE WHEN $2 OR attempts >= 5 THEN 'failed' ELSE 'pending' END
            WHERE id = $1
            """,
            notification_id,
            permanent,
        )

    async def postpone_notification(self, notification_id: int, not_before_at: datetime) -> None:
        await self.execute(
            """
            UPDATE notification_outbox
            SET status = 'pending',
                attempts = GREATEST(attempts - 1, 0),
                not_before_at = $2
            WHERE id = $1
            """,
            notification_id,
            not_before_at,
        )

    async def mark_notification_blocked(self, notification_id: int) -> None:
        await self.execute(
            "UPDATE notification_outbox SET status = 'blocked', sent_at = NOW() WHERE id = $1",
            notification_id,
        )

    async def enqueue_due_scheduled_notifications(self, *, limit: int = 100) -> int:
        """Создать due-уведомления магазина и дневных напоминаний."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        now = datetime.now(timezone.utc)
        created = 0
        try:
            from infrastructure.match_modes import get_current_extra_arena_mode
            rotation = get_current_extra_arena_mode(now)
        except Exception:
            rotation = None
        if rotation is not None:
            modifier_rows = await self.fetch(
                """
                SELECT DISTINCT u.user_id, ns.next_send_at
                FROM users u
                JOIN user_settings us ON us.user_id = u.user_id
                JOIN push_devices pd ON pd.user_id = u.user_id
                LEFT JOIN notification_schedules ns
                  ON ns.user_id = u.user_id AND ns.schedule_type = 'extra_arena_modifier'
                WHERE u.status = 'active'
                  AND us.notif_extra_arena_modifiers = TRUE
                  AND pd.platform = 'android'
                  AND pd.enabled = TRUE
                  AND pd.token <> ''
                  AND (ns.next_send_at IS NULL OR ns.next_send_at <= $1)
                ORDER BY u.user_id
                LIMIT $2
                """,
                now,
                limit,
            )
            next_rotation = datetime.fromtimestamp(rotation.next_rotation_at, tz=timezone.utc)
            for row in modifier_rows:
                user_id = int(row["user_id"])
                if row["next_send_at"] is None:
                    await self._upsert_notification_schedule(
                        user_id,
                        "extra_arena_modifier",
                        next_rotation,
                        {"mode_id": rotation.mode_id, "label": rotation.label},
                    )
                    continue
                cycle_key = f"{rotation.mode_id}:{rotation.next_rotation_at}"
                if await self.enqueue_notification(
                    user_id,
                    category="extra_arena_modifier",
                    event_type="extra_arena_modifier_changed",
                    payload={
                        "section": "arena",
                        "mode_id": rotation.mode_id,
                        "label": rotation.label,
                        "mobile_only": True,
                    },
                    dedupe_key=f"extra_arena_modifier:{user_id}:{cycle_key}",
                ):
                    created += 1
                await self._upsert_notification_schedule(
                    user_id,
                    "extra_arena_modifier",
                    next_rotation,
                    {"mode_id": rotation.mode_id, "label": rotation.label},
                )

        shop_rows = await self.fetch(
            """
            SELECT u.user_id
            FROM users u
            JOIN user_settings us ON us.user_id = u.user_id
            LEFT JOIN notification_schedules ns
              ON ns.user_id = u.user_id AND ns.schedule_type = 'shop_particles'
            WHERE u.status = 'active'
              AND us.notif_shop = TRUE
              AND (ns.next_send_at IS NULL OR ns.next_send_at <= $1)
            ORDER BY COALESCE(ns.next_send_at, u.created_at)
            LIMIT $2
            """,
            now,
            limit,
        )
        for row in shop_rows:
            user_id = int(row["user_id"])
            today = now.date().isoformat()
            if await self.enqueue_notification(
                user_id,
                category="shop",
                event_type="shop_particles",
                payload={"section": "shop", "rotation_date": today},
                dedupe_key=f"shop_particles:{user_id}:{today}",
            ):
                created += 1
            await self._upsert_notification_schedule(
                user_id,
                "shop_particles",
                now,
                {"rotation_date": today},
            )

        reminder_rows = await self.fetch(
            """
            SELECT u.user_id, u.trophies, u.squad_id, u.extra_pass,
                   COALESCE(us.wins_since_last_case, 0) AS wins_since_last_case
            FROM users u
            JOIN user_settings us ON us.user_id = u.user_id
            LEFT JOIN notification_schedules ns
              ON ns.user_id = u.user_id AND ns.schedule_type = 'daily_reminder'
            WHERE u.status = 'active'
              AND us.notif_reminders = TRUE
              AND (ns.next_send_at IS NULL OR ns.next_send_at <= $1)
            ORDER BY COALESCE(ns.next_send_at, u.created_at)
            LIMIT $2
            """,
            now,
            limit,
        )
        for row in reminder_rows:
            user_id = int(row["user_id"])
            payload = choose_reminder_payload(dict(row))
            if await self.enqueue_notification(
                user_id,
                category="reminders",
                event_type="daily_reminder",
                payload=payload,
                dedupe_key=f"daily_reminder:{user_id}:{now.date().isoformat()}",
            ):
                created += 1
            await self._upsert_notification_schedule(user_id, "daily_reminder", now, payload)
        return created

    async def _upsert_notification_schedule(
        self,
        user_id: int,
        schedule_type: str,
        sent_at: datetime,
        payload: dict[str, Any],
    ) -> None:
        next_send_at = sent_at + timedelta(hours=24, seconds=random.randint(0, 3600))
        await self.execute(
            """
            INSERT INTO notification_schedules (user_id, schedule_type, next_send_at, last_sent_at, last_payload)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (user_id, schedule_type) DO UPDATE
            SET next_send_at = EXCLUDED.next_send_at,
                last_sent_at = EXCLUDED.last_sent_at,
                last_payload = EXCLUDED.last_payload
            """,
            user_id,
            schedule_type,
            next_send_at,
            sent_at,
            json.dumps(payload, ensure_ascii=False),
        )

    async def _ensure_promocodes_table(self) -> bool:
        """Создать таблицу промокодов и таблицу использованных промокодов."""
        changed = False

        # Таблица промокодов
        table_exists = await self.fetchval("SELECT to_regclass('public.promocodes')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE promocodes (
                    id BIGSERIAL PRIMARY KEY,
                    code TEXT NOT NULL UNIQUE,
                    type TEXT NOT NULL CHECK (type IN ('permanent', 'personal', 'welcome')),
                    reward_gems INT DEFAULT 0,
                    reward_coins INT DEFAULT 0,
                    reward_keys INT DEFAULT 0,
                    reward_extrapass BOOLEAN DEFAULT FALSE,
                    created_by BIGINT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ
                )
                """
            )
            changed = True

        # Таблица использованных промокодов
        used_table_exists = await self.fetchval("SELECT to_regclass('public.promocode_usage')")
        if not used_table_exists:
            await self.execute(
                """
                CREATE TABLE promocode_usage (
                    id BIGSERIAL PRIMARY KEY,
                    promocode_id BIGINT NOT NULL REFERENCES promocodes(id) ON DELETE CASCADE,
                    user_id BIGINT NOT NULL,
                    used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(promocode_id, user_id)
                )
                """
            )
            changed = True

        columns = await self._get_columns("promocodes")
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "code TEXT NOT NULL DEFAULT ''"
        )
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "type TEXT NOT NULL DEFAULT 'permanent'"
        )
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "reward_gems INT DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "reward_coins INT DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "reward_keys INT DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "reward_extrapass BOOLEAN DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "created_by BIGINT"
        )
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "promocodes", columns, "expires_at TIMESTAMPTZ"
        )

        # Добавляем уникальный индекс для code, если его нет
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'promocodes' AND indexname = 'promocodes_code_unique'
            """
        )
        if not index_exists:
            await self.execute("CREATE UNIQUE INDEX promocodes_code_unique ON promocodes(code)")
            changed = True

        return changed

    async def _ensure_user_mail_table(self) -> bool:
        """Создать таблицу почты пользователей."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.user_mail')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE user_mail (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    sender TEXT NOT NULL DEFAULT 'Система',
                    subject TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    is_read BOOLEAN NOT NULL DEFAULT FALSE,
                    category TEXT,
                    icon TEXT,
                    attachments JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("user_mail")
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "user_id BIGINT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "sender TEXT NOT NULL DEFAULT 'Система'"
        )
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "subject TEXT NOT NULL DEFAULT ''"
        )
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "text TEXT NOT NULL DEFAULT ''"
        )
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "is_read BOOLEAN NOT NULL DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "category TEXT"
        )
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "icon TEXT"
        )
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "attachments JSONB"
        )
        changed |= await self._add_column_if_missing(
            "user_mail", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        if "content" in columns:
            await self.execute("ALTER TABLE user_mail ALTER COLUMN content SET DEFAULT ''")

        # Создаем индекс для быстрого поиска по user_id
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'user_mail' AND indexname = 'user_mail_user_id_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX user_mail_user_id_idx ON user_mail(user_id, created_at DESC)")
            changed = True

        return changed

    async def create_mail(
        self,
        *,
        user_id: int,
        sender: str = "Система",
        subject: str,
        text: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        icon: Optional[str] = None,
        attachments: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Создать письмо пользователю. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        import json as _json

        try:
            mail_text = text if text is not None else (content or "")
            attachments_json = _json.dumps(attachments, ensure_ascii=False) if attachments else None
            columns = await self._get_columns("user_mail")
            if "content" in columns:
                await self.execute(
                    """
                    INSERT INTO user_mail (user_id, sender, subject, text, content, category, icon, attachments)
                    VALUES ($1, $2, $3, $4, $4, $5, $6, $7::jsonb)
                    """,
                    user_id,
                    sender,
                    subject,
                    mail_text,
                    category,
                    icon,
                    attachments_json,
                )
            else:
                await self.execute(
                    """
                    INSERT INTO user_mail (user_id, sender, subject, text, category, icon, attachments)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    """,
                    user_id,
                    sender,
                    subject,
                    mail_text,
                    category,
                    icon,
                    attachments_json,
                )
            return {"success": True}
        except Exception as e:
            logging.getLogger(__name__).warning(
                "create_mail failed: user_id=%s subject=%s error=%s",
                user_id,
                subject,
                e,
                exc_info=True,
            )
            return {"success": False, "error": str(e)}

    async def get_user_mail(
        self,
        user_id: int,
        limit: int = 50,
        unread_only: bool = False,
        category: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Получить письма пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        query = """
            SELECT id, user_id, sender, subject, text, is_read, category, icon, attachments, created_at
            FROM user_mail
            WHERE user_id = $1
        """
        params = [user_id]
        param_idx = 2

        if unread_only:
            query += " AND is_read = FALSE"

        if category:
            query += f" AND category = ${param_idx}"
            params.append(category)
            param_idx += 1

        query += f" ORDER BY created_at DESC LIMIT ${param_idx}"
        params.append(limit)

        rows = await self.fetch(query, *params)

        import json as _json

        result = []
        for row in rows:
            mail = dict(row)
            attachments = mail.get("attachments")
            if isinstance(attachments, str):
                try:
                    mail["attachments"] = _json.loads(attachments)
                except (TypeError, ValueError):
                    mail["attachments"] = {}
            elif attachments is None:
                mail["attachments"] = {}
            mail.setdefault("content", mail.get("text") or "")
            mail.setdefault("body", mail.get("text") or "")
            mail.setdefault("mail_id", mail.get("id"))
            result.append(mail)
        return result

    async def mark_mail_as_read(self, mail_id: int, user_id: int) -> dict[str, Any]:
        """Отметить письмо как прочитанное."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            await self.execute(
                "UPDATE user_mail SET is_read = TRUE WHERE id = $1 AND user_id = $2",
                mail_id,
                user_id,
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # --- Глобальный чат ---

    async def _ensure_global_chat_table(self) -> bool:
        """Создать таблицу глобального чата."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.global_chat')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE global_chat (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("global_chat")
        changed |= await self._add_column_if_missing(
            "global_chat", columns, "user_id BIGINT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "global_chat", columns, "message TEXT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "global_chat", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        # Создаем индекс для быстрого поиска по времени
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'global_chat' AND indexname = 'global_chat_created_at_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX global_chat_created_at_idx ON global_chat(created_at DESC)")
            changed = True

        return changed

    async def create_chat_message(
        self,
        *,
        user_id: int,
        message: str,
    ) -> dict[str, Any]:
        """Создать сообщение в глобальном чате. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        if not message or not message.strip():
            return {"success": False, "error": "empty_message"}

        if len(message) > 500:
            return {"success": False, "error": "message_too_long"}

        try:
            await self.execute(
                """
                INSERT INTO global_chat (user_id, message)
                VALUES ($1, $2)
                """,
                user_id,
                message.strip(),
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_chat_messages(self, limit: int = 100) -> list[dict[str, Any]]:
        """Получить сообщения из глобального чата."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT gc.id, gc.user_id, gc.message, gc.created_at,
                   u.username, u.first_name,
                   COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') as display_name,
                   NULL as user_photo_url
            FROM global_chat gc
            LEFT JOIN users u ON u.user_id = gc.user_id
            LEFT JOIN profiles p ON p.user_id = gc.user_id
            ORDER BY gc.created_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(row) for row in rows]

    # --- Заглушки для Кейсов (система удалена) ---

    async def _ensure_user_cases_table(self) -> bool:
        changed = False
        if not await self.fetchval("SELECT to_regclass('public.user_cases')"):
            await self.execute("""
                CREATE TABLE user_cases (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    case_id INTEGER NOT NULL,
                    tier INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await self.execute("CREATE INDEX IF NOT EXISTS idx_user_cases_user_id ON user_cases(user_id)")
            changed = True

        columns = await self._get_columns("user_cases")
        changed |= await self._add_column_if_missing("user_cases", columns, "id SERIAL PRIMARY KEY")
        changed |= await self._add_column_if_missing("user_cases", columns, "user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE")
        changed |= await self._add_column_if_missing("user_cases", columns, "case_id INTEGER NOT NULL DEFAULT 1")
        changed |= await self._add_column_if_missing("user_cases", columns, "tier INTEGER NOT NULL DEFAULT 1")
        changed |= await self._add_column_if_missing("user_cases", columns, "status TEXT NOT NULL DEFAULT 'pending'")
        changed |= await self._add_column_if_missing("user_cases", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        return changed

    async def get_cases_list(self) -> list[dict[str, Any]]:
        """Заглушка для списка кейсов."""
        return []

    async def get_user_cases(self, user_id: int) -> list[dict[str, Any]]:
        rows = await self.fetch(
            "SELECT * FROM user_cases WHERE user_id = $1 AND status = 'pending'", user_id
        )
        return [dict(r) for r in rows]

    async def add_user_case(self, user_id: int, case_id: int, tier: int) -> dict:
        row = await self.fetchrow(
            """
            INSERT INTO user_cases (user_id, case_id, tier, status)
            VALUES ($1, $2, $3, 'pending')
            RETURNING *
            """,
            user_id, case_id, tier
        )
        if row:
            return dict(row)
        return {"error": "failed"}

    # --- Коммьюнити (Посты и лайки) ---

    async def _ensure_community_posts_table(self) -> bool:
        """Создать таблицу постов коммьюнити."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.community_posts')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE community_posts (
                    id BIGSERIAL PRIMARY KEY,
                    author_id BIGINT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    photo_file_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("community_posts")
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "author_id BIGINT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "title TEXT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "content TEXT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "photo_file_id TEXT"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "post_type TEXT NOT NULL DEFAULT 'news'"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "tags TEXT[] DEFAULT '{}'"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "cover_image_url TEXT"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "content_html TEXT"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "status TEXT NOT NULL DEFAULT 'active'"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "moderation_status TEXT NOT NULL DEFAULT 'approved'"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "moderation_reason TEXT"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "clan_id BIGINT"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "expires_at TIMESTAMPTZ"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "is_pinned BOOLEAN NOT NULL DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "pin_price INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "gems_paid INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "upvotes INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "downvotes INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "community_posts", columns, "admin_approved BOOLEAN NOT NULL DEFAULT FALSE"
        )

        return changed

    async def _ensure_post_likes_table(self) -> bool:
        """Создать таблицу лайков постов."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.post_likes')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE post_likes (
                    id BIGSERIAL PRIMARY KEY,
                    post_id BIGINT NOT NULL,
                    user_id BIGINT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(post_id, user_id)
                )
                """
            )
            changed = True

        columns = await self._get_columns("post_likes")
        changed |= await self._add_column_if_missing(
            "post_likes", columns, "post_id BIGINT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "post_likes", columns, "user_id BIGINT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "post_likes", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        # Создаем уникальный индекс для post_id + user_id
        unique_constraint_exists = await self._constraint_exists("post_likes", "post_likes_post_id_user_id_key")
        if not unique_constraint_exists:
            await self.execute(
                """
                ALTER TABLE post_likes
                ADD CONSTRAINT post_likes_post_id_user_id_key
                UNIQUE (post_id, user_id)
                """
            )
            changed = True

        # Создаем индекс для быстрого поиска по post_id
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'post_likes' AND indexname = 'post_likes_post_id_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX post_likes_post_id_idx ON post_likes(post_id)")
            changed = True

        return changed

    # ── Community: new tables ──────────────────────────────────────────────

    async def _ensure_user_roles_table(self) -> bool:
        """Роли пользователей (admin/user)."""
        changed = False
        from bot.constants import ADMIN_ID

        table_exists = await self.fetchval("SELECT to_regclass('public.user_roles')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE user_roles (
                    user_id BIGINT PRIMARY KEY,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            # seed admin
            await self.execute(
                "INSERT INTO user_roles (user_id, role) VALUES ($1, 'admin') ON CONFLICT DO NOTHING",
                ADMIN_ID,
            )
            changed = True
        else:
            # ensure admin seed exists
            await self.execute(
                "INSERT INTO user_roles (user_id, role) VALUES ($1, 'admin') ON CONFLICT DO NOTHING",
                ADMIN_ID,
            )
        return changed

    async def _ensure_clans_table(self) -> bool:
        """Кланы."""
        changed = False
        from infrastructure.community_config import MOCK_CLANS

        table_exists = await self.fetchval("SELECT to_regclass('public.clans')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE clans (
                    id BIGSERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_id BIGINT NOT NULL,
                    trophies INTEGER NOT NULL DEFAULT 0,
                    rank INTEGER NOT NULL DEFAULT 0,
                    members_count INTEGER NOT NULL DEFAULT 1,
                    max_members INTEGER NOT NULL DEFAULT 15,
                    has_boost BOOLEAN NOT NULL DEFAULT FALSE,
                    public_id INTEGER UNIQUE,
                    boost_public_id INTEGER UNIQUE,
                    avatar_url TEXT,
                    tag TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    type TEXT NOT NULL DEFAULT 'open',
                    min_trophies INTEGER NOT NULL DEFAULT 0,
                    treasury_tokens INTEGER NOT NULL DEFAULT 0,
                    cbrp INTEGER NOT NULL DEFAULT 0,
                    banner_url TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            for c in MOCK_CLANS:
                await self.execute(
                    """
                    INSERT INTO clans (id, name, owner_id, trophies, rank, members_count, max_members, has_boost)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT DO NOTHING
                    """,
                    c["id"], c["name"], c["owner_id"], c["trophies"],
                    c["rank"], c["members_count"], c["max_members"], c["has_boost"],
                )
            changed = True

        columns = await self._get_columns("clans")
        changed |= await self._add_column_if_missing("clans", columns, "tag TEXT")
        changed |= await self._add_column_if_missing("clans", columns, "description TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("clans", columns, "type TEXT NOT NULL DEFAULT 'open'")
        changed |= await self._add_column_if_missing("clans", columns, "min_trophies INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("clans", columns, "treasury_tokens INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("clans", columns, "cbrp INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("clans", columns, "banner_url TEXT")
        changed |= await self._add_column_if_missing("clans", columns, "public_id INTEGER UNIQUE")
        changed |= await self._add_column_if_missing("clans", columns, "boost_public_id INTEGER UNIQUE")

        idx_exists = await self.fetchval(
            "SELECT 1 FROM pg_indexes WHERE tablename='clans' AND indexname='clans_tag_unique_idx'"
        )
        if not idx_exists:
            await self.execute("CREATE UNIQUE INDEX clans_tag_unique_idx ON clans (tag) WHERE tag IS NOT NULL")
            changed = True

        await self.execute("CREATE UNIQUE INDEX IF NOT EXISTS clans_public_id_unique_idx ON clans (public_id) WHERE public_id IS NOT NULL")
        await self.execute("CREATE UNIQUE INDEX IF NOT EXISTS clans_boost_public_id_unique_idx ON clans (boost_public_id) WHERE boost_public_id IS NOT NULL")
        await self._ensure_clan_public_ids()

        await self.execute(
            """
            SELECT setval(
                pg_get_serial_sequence('clans', 'id'),
                COALESCE((SELECT MAX(id) FROM clans), 1),
                (SELECT COUNT(*) FROM clans) > 0
            )
            """
        )

        return changed

    async def _ensure_clan_members_table(self) -> bool:
        """Участники кланов."""
        changed = False
        from infrastructure.community_config import MOCK_CLANS

        table_exists = await self.fetchval("SELECT to_regclass('public.clan_members')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE clan_members (
                    clan_id BIGINT NOT NULL,
                    user_id BIGINT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'member',
                    personal_tokens INTEGER NOT NULL DEFAULT 0,
                    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (clan_id, user_id)
                )
                """
            )
            for c in MOCK_CLANS:
                await self.execute(
                    """
                    INSERT INTO clan_members (clan_id, user_id, role)
                    VALUES ($1, $2, 'owner')
                    ON CONFLICT DO NOTHING
                    """,
                    c["id"], c["owner_id"],
                )
            changed = True

        columns = await self._get_columns("clan_members")
        changed |= await self._add_column_if_missing("clan_members", columns, "personal_tokens INTEGER NOT NULL DEFAULT 0")
        await self.execute("UPDATE clan_members SET role = 'creator' WHERE role = 'owner'")

        return changed

    async def _generate_unique_clan_public_id(self, *, digits: int, column: str) -> int:
        if column not in {"public_id", "boost_public_id"}:
            raise ValueError("invalid_public_id_column")
        low = 10 ** (digits - 1)
        high = (10 ** digits) - 1
        for _ in range(200):
            value = random.randint(low, high)
            exists = await self.fetchval(f"SELECT 1 FROM clans WHERE {column} = $1", value)
            if not exists:
                return value
        raise RuntimeError(f"Could not generate unique {digits}-digit clan id")

    async def _ensure_clan_public_ids(self) -> None:
        rows = await self.fetch("SELECT id FROM clans WHERE public_id IS NULL ORDER BY id")
        for row in rows:
            public_id = await self._generate_unique_clan_public_id(digits=5, column="public_id")
            await self.execute(
                "UPDATE clans SET public_id = $1 WHERE id = $2 AND public_id IS NULL",
                public_id,
                row["id"],
            )

    async def _ensure_clan_join_requests_table(self) -> bool:
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.clan_join_requests')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE clan_join_requests (
                    id BIGSERIAL PRIMARY KEY,
                    clan_id BIGINT NOT NULL REFERENCES clans(id) ON DELETE CASCADE,
                    user_id BIGINT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    decided_by BIGINT,
                    decided_at TIMESTAMPTZ,
                    UNIQUE(clan_id, user_id)
                )
                """
            )
            changed = True
        return changed

    async def _ensure_clan_activity_table(self) -> bool:
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.clan_activity')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE clan_activity (
                    id BIGSERIAL PRIMARY KEY,
                    clan_id BIGINT NOT NULL REFERENCES clans(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    user_id BIGINT,
                    target_user_id BIGINT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await self.execute("CREATE INDEX clan_activity_clan_time_idx ON clan_activity (clan_id, created_at DESC)")
            changed = True
        return changed

    async def _ensure_clan_upgrades_table(self) -> bool:
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.clan_upgrades')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE clan_upgrades (
                    clan_id BIGINT NOT NULL REFERENCES clans(id) ON DELETE CASCADE,
                    upgrade_type TEXT NOT NULL,
                    level INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (clan_id, upgrade_type)
                )
                """
            )
            changed = True
        return changed

    async def _ensure_game_settings_table(self) -> bool:
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.game_settings')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE game_settings (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL DEFAULT 'null',
                    description TEXT NOT NULL DEFAULT '',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        for key, value in SQUAD_SETTINGS_DEFAULTS.items():
            await self.execute(
                """
                INSERT INTO game_settings (key, value, description)
                VALUES ($1, $2::jsonb, $3)
                ON CONFLICT (key) DO NOTHING
                """,
                key,
                json.dumps(value, ensure_ascii=False),
                "Squad runtime setting",
            )
        for key, value in RUNTIME_SETTINGS_DEFAULTS.items():
            await self.execute(
                """
                INSERT INTO game_settings (key, value, description)
                VALUES ($1, $2::jsonb, $3)
                ON CONFLICT (key) DO NOTHING
                """,
                key,
                json.dumps(value, ensure_ascii=False),
                "Game runtime setting",
            )
        await self._merge_squad_runtime_defaults()
        await self._merge_runtime_defaults()
        return changed

    async def _merge_runtime_defaults(self) -> None:
        """Add newly introduced runtime config entries without clobbering admin edits."""
        availability = await self.get_game_setting("feature_availability", {})
        if not isinstance(availability, dict):
            availability = {}
        changed = False
        for key, value in RUNTIME_FEATURE_DEFAULTS.items():
            if key not in availability:
                availability[key] = value
                changed = True
        if changed:
            await self.set_game_setting(
                "feature_availability",
                availability,
                "Game runtime setting",
            )

        maintenance = await self.get_game_setting("maintenance_mode", {})
        if not isinstance(maintenance, dict) or "enabled" not in maintenance:
            await self.set_game_setting(
                "maintenance_mode",
                {"enabled": bool((maintenance or {}).get("enabled", False)) if isinstance(maintenance, dict) else False},
                "Game runtime setting",
            )

        disabled_cards = await self.get_game_setting("disabled_card_ids", [])
        if not isinstance(disabled_cards, list):
            await self.set_game_setting("disabled_card_ids", [], "Game runtime setting")

    async def _merge_squad_runtime_defaults(self) -> None:
        """Add newly introduced squad config entries without clobbering admin edits."""
        upgrades = await self.get_game_setting("squad_upgrades", {})
        if not isinstance(upgrades, dict):
            upgrades = {}
        default_upgrades = SQUAD_SETTINGS_DEFAULTS["squad_upgrades"]
        changed = False
        for key, value in default_upgrades.items():
            if key not in upgrades:
                upgrades[key] = value
                changed = True
        if changed:
            await self.set_game_setting("squad_upgrades", upgrades, "Squad runtime setting")

        rewards = await self.get_game_setting("squad_personal_rewards", [])
        if not isinstance(rewards, list):
            rewards = []
        by_id = {str(item.get("id")): dict(item) for item in rewards if isinstance(item, dict)}
        reward_changed = False
        for default_reward in SQUAD_SETTINGS_DEFAULTS["squad_personal_rewards"]:
            rid = str(default_reward["id"])
            if rid not in by_id:
                rewards.append(default_reward)
                reward_changed = True
                continue
            merged = dict(default_reward)
            merged.update(by_id[rid])
            for key, value in default_reward.items():
                if key not in by_id[rid]:
                    merged[key] = value
                    reward_changed = True
            if default_reward.get("kind") == "cosmetic" and by_id[rid].get("kind") == "title":
                merged["kind"] = "cosmetic"
                reward_changed = True
            if merged != by_id[rid]:
                for idx, item in enumerate(rewards):
                    if isinstance(item, dict) and str(item.get("id")) == rid:
                        rewards[idx] = merged
                        break
        if reward_changed:
            await self.set_game_setting("squad_personal_rewards", rewards, "Squad runtime setting")

    async def _ensure_squad_cbrp_events_table(self) -> bool:
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.squad_cbrp_events')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE squad_cbrp_events (
                    id BIGSERIAL PRIMARY KEY,
                    clan_id BIGINT NOT NULL REFERENCES clans(id) ON DELETE CASCADE,
                    user_id BIGINT NOT NULL,
                    event_type TEXT NOT NULL,
                    source_id TEXT,
                    cbrp INTEGER NOT NULL DEFAULT 0,
                    personal_tokens INTEGER NOT NULL DEFAULT 0,
                    treasury_tokens INTEGER NOT NULL DEFAULT 0,
                    owner_tax_tokens INTEGER NOT NULL DEFAULT 0,
                    period_key TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        await self.execute(
            "CREATE INDEX IF NOT EXISTS squad_cbrp_events_clan_time_idx ON squad_cbrp_events (clan_id, created_at DESC)"
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS squad_cbrp_events_user_time_idx ON squad_cbrp_events (user_id, created_at DESC)"
        )
        await self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS squad_cbrp_events_source_unique_idx
            ON squad_cbrp_events (event_type, source_id, user_id)
            WHERE source_id IS NOT NULL
            """
        )
        return changed

    async def _ensure_squad_trophy_snapshots_table(self) -> bool:
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.squad_trophy_snapshots')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE squad_trophy_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    period_key TEXT NOT NULL,
                    clan_id BIGINT NOT NULL REFERENCES clans(id) ON DELETE CASCADE,
                    user_id BIGINT NOT NULL,
                    start_trophies INTEGER NOT NULL DEFAULT 0,
                    end_trophies INTEGER,
                    delta_trophies INTEGER,
                    cbrp_awarded INTEGER NOT NULL DEFAULT 0,
                    personal_tokens_awarded INTEGER NOT NULL DEFAULT 0,
                    treasury_tokens_awarded INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    closed_at TIMESTAMPTZ,
                    awarded_at TIMESTAMPTZ,
                    UNIQUE (period_key, clan_id, user_id)
                )
                """
            )
            changed = True

        await self.execute(
            """
            CREATE INDEX IF NOT EXISTS squad_trophy_snapshots_open_idx
            ON squad_trophy_snapshots (period_key, awarded_at)
            """
        )
        await self.execute(
            """
            CREATE INDEX IF NOT EXISTS squad_trophy_snapshots_user_idx
            ON squad_trophy_snapshots (user_id, created_at DESC)
            """
        )
        return changed

    async def _ensure_squad_shop_purchases_table(self) -> bool:
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.squad_shop_purchases')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE squad_shop_purchases (
                    id BIGSERIAL PRIMARY KEY,
                    clan_id BIGINT NOT NULL REFERENCES clans(id) ON DELETE CASCADE,
                    user_id BIGINT NOT NULL,
                    reward_id TEXT NOT NULL,
                    cost INTEGER NOT NULL DEFAULT 0,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (user_id, reward_id)
                )
                """
            )
            changed = True

        await self.execute(
            """
            CREATE INDEX IF NOT EXISTS squad_shop_purchases_clan_time_idx
            ON squad_shop_purchases (clan_id, created_at DESC)
            """
        )
        return changed

    async def _ensure_community_votes_table(self) -> bool:
        """Голоса для идей и объявлений (upvote/downvote/like/dislike)."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.community_votes')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE community_votes (
                    post_id BIGINT NOT NULL,
                    user_id BIGINT NOT NULL,
                    vote_type TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (post_id, user_id)
                )
                """
            )
            await self.execute("CREATE INDEX community_votes_post_id_idx ON community_votes(post_id)")
            changed = True
        return changed

    async def _ensure_community_polls_table(self) -> bool:
        """Голосования (опросы) в новостях."""
        changed = False

        polls_exists = await self.fetchval("SELECT to_regclass('public.community_polls')")
        if not polls_exists:
            await self.execute(
                """
                CREATE TABLE community_polls (
                    id BIGSERIAL PRIMARY KEY,
                    post_id BIGINT NOT NULL UNIQUE,
                    question TEXT NOT NULL,
                    options JSONB NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        poll_votes_exists = await self.fetchval("SELECT to_regclass('public.community_poll_votes')")
        if not poll_votes_exists:
            await self.execute(
                """
                CREATE TABLE community_poll_votes (
                    poll_id BIGINT NOT NULL,
                    user_id BIGINT NOT NULL,
                    option_id INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (poll_id, user_id)
                )
                """
            )
            changed = True
        return changed

    async def _ensure_community_submissions_table(self) -> bool:
        """Трекер отправок для rate-limiting модерации."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.community_submissions')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE community_submissions (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    category TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await self.execute(
                "CREATE INDEX community_submissions_user_rate_idx ON community_submissions(user_id, created_at)"
            )
            changed = True
        return changed

    # ── Community CRUD ─────────────────────────────────────────────────────

    async def is_admin(self, user_id: int) -> bool:
        """Проверить, является ли пользователь администратором."""
        if not self._pool:
            return False
        from bot.constants import ADMIN_ID
        if user_id == ADMIN_ID:
            return True
        role = await self.fetchval(
            "SELECT role FROM user_roles WHERE user_id = $1", user_id
        )
        return role == "admin"

    async def get_user_role(self, user_id: int) -> str:
        """Получить роль пользователя."""
        if not self._pool:
            return "user"
        from bot.constants import ADMIN_ID
        if user_id == ADMIN_ID:
            return "admin"
        role = await self.fetchval(
            "SELECT role FROM user_roles WHERE user_id = $1", user_id
        )
        return role or "user"

    async def get_user_clan(self, user_id: int) -> Optional[dict]:
        """Получить информацию о клане пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            """
            SELECT c.id, c.name, c.owner_id, c.trophies, c.rank,
                   c.members_count, c.max_members, c.has_boost, c.avatar_url,
                   c.public_id, c.boost_public_id,
                   c.tag, c.description, c.type, c.min_trophies,
                   c.treasury_tokens, c.cbrp, c.banner_url,
                   CASE WHEN cm.role = 'owner' THEN 'creator' ELSE cm.role END AS member_role,
                   cm.personal_tokens
            FROM clan_members cm
            JOIN clans c ON c.id = cm.clan_id
            WHERE cm.user_id = $1
            ORDER BY cm.joined_at
            LIMIT 1
            """,
            user_id,
        )
        if not row:
            return None
        return dict(row)

    async def get_game_setting(self, key: str, default: Any = None) -> Any:
        value = await self.fetchval("SELECT value FROM game_settings WHERE key = $1", key)
        if value is None:
            return default
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    async def set_game_setting(self, key: str, value: Any, description: str = "") -> None:
        await self.execute(
            """
            INSERT INTO game_settings (key, value, description, updated_at)
            VALUES ($1, $2::jsonb, $3, NOW())
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value,
                description = COALESCE(NULLIF(EXCLUDED.description, ''), game_settings.description),
                updated_at = NOW()
            """,
            key,
            json.dumps(value, ensure_ascii=False),
            description,
        )

    async def get_runtime_config(self) -> dict[str, Any]:
        """Return DB-backed runtime switches with stable defaults."""
        rows = await self.fetch(
            "SELECT key, value FROM game_settings WHERE key = ANY($1::text[])",
            list(RUNTIME_SETTINGS_DEFAULTS.keys()),
        )
        config = {
            "maintenance_mode": dict(RUNTIME_SETTINGS_DEFAULTS["maintenance_mode"]),
            "feature_availability": dict(RUNTIME_FEATURE_DEFAULTS),
            "disabled_card_ids": [],
        }
        for row in rows:
            key = row["key"]
            value = row["value"]
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
            if key == "feature_availability" and isinstance(value, dict):
                merged = dict(RUNTIME_FEATURE_DEFAULTS)
                merged.update({str(k): bool(v) for k, v in value.items()})
                config[key] = merged
            elif key == "maintenance_mode" and isinstance(value, dict):
                config[key] = {"enabled": bool(value.get("enabled", False))}
            elif key == "disabled_card_ids" and isinstance(value, list):
                normalized: list[int] = []
                for raw in value:
                    try:
                        card_id = int(raw)
                    except (TypeError, ValueError):
                        continue
                    if card_id not in normalized:
                        normalized.append(card_id)
                config[key] = normalized
        return config

    async def set_runtime_config(
        self,
        *,
        maintenance_mode: dict[str, Any] | None = None,
        feature_availability: dict[str, Any] | None = None,
        disabled_card_ids: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Persist runtime switches and return normalized config."""
        current = await self.get_runtime_config()

        if maintenance_mode is not None:
            current["maintenance_mode"] = {
                "enabled": bool(maintenance_mode.get("enabled", False)),
            }

        if feature_availability is not None:
            merged = dict(RUNTIME_FEATURE_DEFAULTS)
            existing = current.get("feature_availability") or {}
            if isinstance(existing, dict):
                merged.update({str(k): bool(v) for k, v in existing.items()})
            for key in RUNTIME_FEATURE_DEFAULTS:
                if key in feature_availability:
                    merged[key] = bool(feature_availability[key])
            current["feature_availability"] = merged

        if disabled_card_ids is not None:
            normalized: list[int] = []
            for raw in disabled_card_ids:
                try:
                    card_id = int(raw)
                except (TypeError, ValueError):
                    continue
                if card_id > 0 and card_id not in normalized:
                    normalized.append(card_id)
            current["disabled_card_ids"] = normalized

        await self.set_game_setting(
            "maintenance_mode",
            current["maintenance_mode"],
            "Game runtime setting",
        )
        await self.set_game_setting(
            "feature_availability",
            current["feature_availability"],
            "Game runtime setting",
        )
        await self.set_game_setting(
            "disabled_card_ids",
            current["disabled_card_ids"],
            "Game runtime setting",
        )
        return current

    async def is_feature_enabled(self, feature_key: str) -> bool:
        config = await self.get_runtime_config()
        availability = config.get("feature_availability") or {}
        return bool(availability.get(feature_key, True))

    async def get_disabled_card_ids(self) -> list[int]:
        config = await self.get_runtime_config()
        return list(config.get("disabled_card_ids") or [])

    async def get_squad_runtime_config(self) -> dict[str, Any]:
        rows = await self.fetch(
            "SELECT key, value FROM game_settings WHERE key = ANY($1::text[])",
            list(SQUAD_SETTINGS_DEFAULTS.keys()),
        )
        config = dict(SQUAD_SETTINGS_DEFAULTS)
        for row in rows:
            value = row["value"]
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
            config[row["key"]] = value
        return config

    def _squad_week_key(self, moment: Optional[datetime] = None) -> str:
        moment = moment or datetime.now(timezone.utc)
        moscow = moment.astimezone(timezone(timedelta(hours=3)))
        iso = moscow.date().isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    async def ensure_current_squad_trophy_snapshot(
        self,
        user_id: int,
        clan_id: Optional[int] = None,
        trophies: Optional[int] = None,
        *,
        moment: Optional[datetime] = None,
    ) -> None:
        period_key = self._squad_week_key(moment)
        if clan_id is None:
            clan_id = await self.fetchval(
                "SELECT clan_id FROM clan_members WHERE user_id = $1 LIMIT 1",
                user_id,
            )
        if not clan_id:
            return
        if trophies is None:
            trophies = await self.fetchval("SELECT trophies FROM users WHERE user_id = $1", user_id)
        await self.execute(
            """
            INSERT INTO squad_trophy_snapshots (period_key, clan_id, user_id, start_trophies)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (period_key, clan_id, user_id) DO NOTHING
            """,
            period_key,
            int(clan_id),
            user_id,
            max(0, int(trophies or 0)),
        )

    async def process_weekly_squad_cbrp(self, *, moment: Optional[datetime] = None) -> dict[str, Any]:
        config = await self.get_squad_runtime_config()
        if not bool(config.get("squad_weekly_cbrp_enabled", True)):
            return {"processed": False, "reason": "disabled", "awarded": 0}

        current_key = self._squad_week_key(moment)
        await self.execute(
            """
            INSERT INTO squad_trophy_snapshots (period_key, clan_id, user_id, start_trophies)
            SELECT $1, cm.clan_id, cm.user_id, COALESCE(u.trophies, 0)
            FROM clan_members cm
            JOIN users u ON u.user_id = cm.user_id
            ON CONFLICT (period_key, clan_id, user_id) DO NOTHING
            """,
            current_key,
        )

        open_rows = await self.fetch(
            """
            SELECT s.id, s.period_key, s.clan_id, s.user_id, s.start_trophies,
                   COALESCE(u.trophies, 0) AS current_trophies
            FROM squad_trophy_snapshots s
            JOIN clan_members cm ON cm.clan_id = s.clan_id AND cm.user_id = s.user_id
            JOIN users u ON u.user_id = s.user_id
            WHERE s.period_key <> $1
              AND s.awarded_at IS NULL
            ORDER BY s.period_key, s.id
            LIMIT 500
            """,
            current_key,
        )

        weekly_divisor = max(1, int(config.get("squad_weekly_delta_divisor") or 25))
        personal_divisor = max(1, int(config.get("squad_weekly_personal_tokens_divisor") or 100))
        treasury_divisor = max(1, int(config.get("squad_weekly_treasury_tokens_divisor") or 150))
        awarded_count = 0
        closed_count = 0
        total_cbrp = 0

        for row in open_rows:
            end_trophies = max(0, int(row["current_trophies"] or 0))
            start_trophies = max(0, int(row["start_trophies"] or 0))
            delta = max(0, end_trophies - start_trophies)
            cbrp_value = delta // weekly_divisor
            personal_value = delta // personal_divisor
            treasury_value = delta // treasury_divisor
            source_id = f"weekly_trophy_delta:{row['period_key']}:{row['clan_id']}:{row['user_id']}"

            if cbrp_value > 0 or personal_value > 0 or treasury_value > 0:
                result = await self.award_squad_cbrp(
                    int(row["user_id"]),
                    "weekly_trophy_delta",
                    source_id=source_id,
                    period_key=str(row["period_key"]),
                    metadata={
                        "start_trophies": start_trophies,
                        "end_trophies": end_trophies,
                        "delta_trophies": delta,
                        "weekly_delta_divisor": weekly_divisor,
                    },
                    cbrp=cbrp_value,
                    personal_tokens=personal_value,
                    treasury_tokens=treasury_value,
                )
                if result.get("awarded"):
                    awarded_count += 1
                    total_cbrp += int(result.get("cbrp") or 0)

            await self.execute(
                """
                UPDATE squad_trophy_snapshots
                SET end_trophies = $1,
                    delta_trophies = $2,
                    cbrp_awarded = $3,
                    personal_tokens_awarded = $4,
                    treasury_tokens_awarded = $5,
                    closed_at = COALESCE(closed_at, NOW()),
                    awarded_at = NOW()
                WHERE id = $6
                """,
                end_trophies,
                delta,
                cbrp_value,
                personal_value,
                treasury_value,
                int(row["id"]),
            )
            closed_count += 1

        return {
            "processed": True,
            "period_key": current_key,
            "closed": closed_count,
            "awarded": awarded_count,
            "cbrp": total_cbrp,
        }

    async def award_squad_seasonal_cbrp(
        self,
        user_id: int,
        *,
        season_id: int | str,
        delta_trophies: int,
    ) -> dict[str, Any]:
        """Hook для будущего season reset: начислить CBRP за сезонную дельту."""
        config = await self.get_squad_runtime_config()
        if not bool(config.get("squad_seasonal_cbrp_enabled", False)):
            return {"awarded": False, "reason": "disabled"}

        delta = max(0, int(delta_trophies or 0))
        cbrp_divisor = max(1, int(config.get("squad_seasonal_cbrp_divisor") or 50))
        personal_divisor = max(1, int(config.get("squad_seasonal_personal_tokens_divisor") or 200))
        treasury_divisor = max(1, int(config.get("squad_seasonal_treasury_tokens_divisor") or 300))
        return await self.award_squad_cbrp(
            user_id,
            "seasonal_reset",
            source_id=f"seasonal_reset:{season_id}:{user_id}",
            period_key=str(season_id),
            metadata={
                "season_id": season_id,
                "delta_trophies": delta,
                "seasonal_cbrp_divisor": cbrp_divisor,
            },
            cbrp=delta // cbrp_divisor,
            personal_tokens=delta // personal_divisor,
            treasury_tokens=delta // treasury_divisor,
        )

    def _resolve_squad_reward(
        self,
        config: dict[str, Any],
        event_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        rewards = config.get("squad_rewards") or {}
        metadata = metadata or {}
        reward: Any = rewards.get(event_type) or {}

        if event_type == "case_open":
            tier = str(int(metadata.get("tier") or 1))
            reward = (reward.get("tiers") or {}).get(tier, {})
        elif event_type == "card_upgrade":
            level = int(metadata.get("new_level") or 0)
            reward = {}
            for item in (rewards.get("card_upgrade") or {}).get("levels", []):
                if int(item.get("min", 0)) <= level <= int(item.get("max", 0)):
                    reward = item
                    break

        return {
            "cbrp": max(0, int(reward.get("cbrp") or 0)),
            "personal_tokens": max(0, int(reward.get("personal_tokens") or 0)),
            "treasury_tokens": max(0, int(reward.get("treasury_tokens") or 0)),
        }

    async def award_squad_cbrp(
        self,
        user_id: int,
        event_type: str,
        *,
        source_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        period_key: Optional[str] = None,
        cbrp: Optional[int] = None,
        personal_tokens: Optional[int] = None,
        treasury_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Начислить CBRP и токены текущему скваду игрока. Идемпотентно по source_id."""
        if not self._pool:
            return {"awarded": False, "reason": "db_not_connected"}

        metadata = metadata or {}
        member = await self.fetchrow(
            """
            SELECT cm.clan_id, cm.role, c.owner_id, c.has_boost
            FROM clan_members cm
            JOIN clans c ON c.id = cm.clan_id
            WHERE cm.user_id = $1
            LIMIT 1
            """,
            user_id,
        )
        if not member:
            return {"awarded": False, "reason": "no_squad"}

        config = await self.get_squad_runtime_config()
        resolved = self._resolve_squad_reward(config, event_type, metadata)
        cbrp_value = resolved["cbrp"] if cbrp is None else max(0, int(cbrp))
        personal_value = resolved["personal_tokens"] if personal_tokens is None else max(0, int(personal_tokens))
        treasury_value = resolved["treasury_tokens"] if treasury_tokens is None else max(0, int(treasury_tokens))
        clan_id = int(member["clan_id"])

        upgrades = await self.get_clan_upgrades(clan_id)
        cbrp_boost_level = int(upgrades.get("cbrp_boost", 0) or 0)
        if cbrp_value > 0 and cbrp_boost_level > 0:
            import math
            cbrp_multiplier = 1.0
            boost_cfg = ((config.get("squad_upgrades") or {}).get("cbrp_boost") or {})
            for item in boost_cfg.get("levels", []):
                if int(item.get("level", 0)) == cbrp_boost_level:
                    cbrp_multiplier = float(item.get("cbrp_multiplier") or 1.0)
                    break
            cbrp_value = int(math.ceil(cbrp_value * cbrp_multiplier))

        if bool(member["has_boost"]):
            import math
            multiplier = float(config.get("squad_clan_boost_token_multiplier") or 1.0)
            personal_value = int(math.ceil(personal_value * multiplier))
            treasury_value = int(math.ceil(treasury_value * multiplier))

        role = "creator" if member["role"] == "owner" else str(member["role"])
        owner_id = int(member["owner_id"])
        owner_tax = 0
        if user_id != owner_id and personal_value > 0:
            import math
            tax_pct = max(0.0, min(0.9, float(config.get("squad_creator_passive_tax_pct") or 0)))
            owner_tax = int(math.floor(personal_value * tax_pct))
            personal_value = max(0, personal_value - owner_tax)

        if cbrp_value <= 0 and personal_value <= 0 and treasury_value <= 0 and owner_tax <= 0:
            return {"awarded": False, "reason": "zero_reward"}

        metadata_json = json.dumps(metadata, ensure_ascii=False)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO squad_cbrp_events (
                        clan_id, user_id, event_type, source_id, cbrp,
                        personal_tokens, treasury_tokens, owner_tax_tokens,
                        period_key, metadata
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                    """,
                    clan_id,
                    user_id,
                    event_type,
                    source_id,
                    cbrp_value,
                    personal_value,
                    treasury_value,
                    owner_tax,
                    period_key,
                    metadata_json,
                )
                if not row:
                    return {"awarded": False, "reason": "duplicate"}

                await conn.execute(
                    """
                    UPDATE clans
                    SET cbrp = cbrp + $1,
                        treasury_tokens = treasury_tokens + $2
                    WHERE id = $3
                    """,
                    cbrp_value,
                    treasury_value,
                    clan_id,
                )
                if personal_value > 0:
                    await conn.execute(
                        """
                        UPDATE clan_members
                        SET personal_tokens = personal_tokens + $1
                        WHERE clan_id = $2 AND user_id = $3
                        """,
                        personal_value,
                        clan_id,
                        user_id,
                    )
                if owner_tax > 0:
                    await conn.execute(
                        """
                        UPDATE clan_members
                        SET personal_tokens = personal_tokens + $1
                        WHERE clan_id = $2 AND user_id = $3
                        """,
                        owner_tax,
                        clan_id,
                        owner_id,
                    )

        return {
            "awarded": True,
            "clan_id": clan_id,
            "event_type": event_type,
            "cbrp": cbrp_value,
            "personal_tokens": personal_value,
            "treasury_tokens": treasury_value,
            "owner_tax_tokens": owner_tax,
            "role": role,
        }

    async def get_squad_cbrp_events(
        self,
        *,
        clan_id: Optional[int] = None,
        user_id: Optional[int] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if clan_id is not None:
            params.append(clan_id)
            where.append(f"e.clan_id = ${len(params)}")
        if user_id is not None:
            params.append(user_id)
            where.append(f"e.user_id = ${len(params)}")
        params.append(limit)
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        rows = await self.fetch(
            f"""
            SELECT e.*, COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок') AS nick
            FROM squad_cbrp_events e
            LEFT JOIN users u ON u.user_id = e.user_id
            LEFT JOIN profiles p ON p.user_id = e.user_id
            {where_sql}
            ORDER BY e.created_at DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [dict(r) for r in rows]

    async def get_mobile_squad_personal_cbrp_widget(self, user_id: int) -> dict[str, Any]:
        clan = await self.get_user_clan(user_id)
        if not clan:
            return {"in_squad": False}
        clan_id = int(clan["id"])
        totals = await self.fetchrow(
            """
            SELECT COALESCE(SUM(cbrp), 0)::INT AS personal_cbrp,
                   COALESCE(SUM(cbrp) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours'), 0)::INT AS delta_24h,
                   COALESCE(SUM(personal_tokens), 0)::INT AS personal_tokens_earned
            FROM squad_cbrp_events
            WHERE clan_id = $1 AND user_id = $2
            """,
            clan_id,
            user_id,
        )
        events = await self.get_squad_cbrp_events(clan_id=clan_id, user_id=user_id, limit=8)
        return {
            "in_squad": True,
            "squad": {
                "id": clan_id,
                "name": clan.get("name"),
                "tag": clan.get("tag"),
                "cbrp": int(clan.get("cbrp") or 0),
            },
            "personal_cbrp": int((totals or {}).get("personal_cbrp") or 0),
            "delta_24h": int((totals or {}).get("delta_24h") or 0),
            "personal_tokens": int(clan.get("personal_tokens") or 0),
            "personal_tokens_earned": int((totals or {}).get("personal_tokens_earned") or 0),
            "events": events,
        }

    async def get_mobile_squad_owner_overview_widget(self, user_id: int) -> dict[str, Any]:
        clan = await self.get_user_clan(user_id)
        if not clan:
            return {"in_squad": False}
        role = str(clan.get("member_role") or "")
        if role not in {"creator", "owner"}:
            return {"in_squad": True, "error": "owner_required", "role": role}
        clan_id = int(clan["id"])
        activity = await self.get_clan_activity(clan_id, limit=8)
        return {
            "in_squad": True,
            "role": role,
            "squad": {
                "id": clan_id,
                "name": clan.get("name"),
                "tag": clan.get("tag"),
                "cbrp": int(clan.get("cbrp") or 0),
                "members_count": int(clan.get("members_count") or 0),
                "max_members": int(clan.get("max_members") or 0),
                "treasury_tokens": int(clan.get("treasury_tokens") or 0),
                "has_boost": bool(clan.get("has_boost")),
            },
            "activity": activity,
        }

    async def get_mobile_squad_owner_cbrp_widget(self, user_id: int) -> dict[str, Any]:
        clan = await self.get_user_clan(user_id)
        if not clan:
            return {"in_squad": False}
        role = str(clan.get("member_role") or "")
        if role not in {"creator", "owner"}:
            return {"in_squad": True, "error": "owner_required", "role": role}
        clan_id = int(clan["id"])
        totals = await self.fetchrow(
            """
            SELECT COALESCE(SUM(cbrp), 0)::INT AS cbrp_total,
                   COALESCE(SUM(cbrp) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours'), 0)::INT AS delta_24h
            FROM squad_cbrp_events
            WHERE clan_id = $1
            """,
            clan_id,
        )
        top_members = await self.fetch(
            """
            SELECT cm.user_id,
                   COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок #' || cm.user_id::TEXT) AS nick,
                   COALESCE(cm.personal_tokens, 0)::INT AS personal_tokens,
                   COALESCE(SUM(e.cbrp), 0)::INT AS cbrp,
                   COALESCE(SUM(e.cbrp) FILTER (WHERE e.created_at >= NOW() - INTERVAL '24 hours'), 0)::INT AS delta_24h
            FROM clan_members cm
            LEFT JOIN users u ON u.user_id = cm.user_id
            LEFT JOIN profiles p ON p.user_id = cm.user_id
            LEFT JOIN squad_cbrp_events e ON e.clan_id = cm.clan_id AND e.user_id = cm.user_id
            WHERE cm.clan_id = $1
            GROUP BY cm.user_id, cm.personal_tokens, p.custom_nickname, u.username, u.first_name
            ORDER BY cbrp DESC, delta_24h DESC, personal_tokens DESC
            LIMIT 6
            """,
            clan_id,
        )
        events = await self.get_squad_cbrp_events(clan_id=clan_id, limit=8)
        return {
            "in_squad": True,
            "role": role,
            "squad": {
                "id": clan_id,
                "name": clan.get("name"),
                "tag": clan.get("tag"),
                "cbrp": int(clan.get("cbrp") or 0),
                "treasury_tokens": int(clan.get("treasury_tokens") or 0),
            },
            "cbrp_total": int((totals or {}).get("cbrp_total") or clan.get("cbrp") or 0),
            "delta_24h": int((totals or {}).get("delta_24h") or 0),
            "top_members": [dict(row) for row in top_members],
            "events": events,
        }

    # ── Clan CRUD ──────────────────────────────────────────────────────

    async def create_clan(
        self, owner_id: int, name: str, tag: str,
        description: str = "", clan_type: str = "open", min_trophies: int = 0,
    ) -> dict:
        from infrastructure.clan_config import CLAN_BASE_SLOTS
        existing = await self.fetchval(
            "SELECT clan_id FROM clan_members WHERE user_id = $1", owner_id
        )
        if existing:
            raise ValueError("already_in_squad")
        public_id = await self._generate_unique_clan_public_id(digits=5, column="public_id")
        row = await self.fetchrow(
            """
            INSERT INTO clans (name, tag, owner_id, description, type, min_trophies, max_members, public_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
            """,
            name, tag, owner_id, description, clan_type, min_trophies, CLAN_BASE_SLOTS, public_id,
        )
        clan = dict(row)
        await self.execute(
            "INSERT INTO clan_members (clan_id, user_id, role) VALUES ($1, $2, 'creator')",
            clan["id"], owner_id,
        )
        await self.execute("UPDATE users SET squad_id = $1 WHERE user_id = $2", clan["id"], owner_id)
        await self.ensure_current_squad_trophy_snapshot(owner_id, clan_id=clan["id"])
        await self._log_clan_activity(clan["id"], "created", f"{name} создан", user_id=owner_id)
        return clan

    async def get_clan(self, clan_id: int) -> Optional[dict]:
        row = await self.fetchrow("SELECT * FROM clans WHERE id = $1", clan_id)
        if not row:
            return None
        return dict(row)

    async def resolve_clan_identifier(self, identifier: int | str) -> Optional[dict]:
        """Найти сквад по внутреннему id, 5-значному public_id или активному BOOST id."""
        ident = str(identifier).strip()
        if not ident.isdigit():
            return None
        value = int(ident)
        row = await self.fetchrow(
            """
            SELECT *
            FROM clans
            WHERE id = $1
               OR public_id = $1
               OR (has_boost = TRUE AND boost_public_id = $1)
            LIMIT 1
            """,
            value,
        )
        return dict(row) if row else None

    async def get_clan_by_tag(self, tag: str) -> Optional[dict]:
        row = await self.fetchrow("SELECT * FROM clans WHERE tag = $1", tag)
        return dict(row) if row else None

    async def is_tag_unique(self, tag: str) -> bool:
        count = await self.fetchval("SELECT COUNT(*) FROM clans WHERE tag = $1", tag)
        return count == 0

    async def update_clan_settings(self, clan_id: int, **fields) -> bool:
        allowed = {"name", "tag", "description", "type", "min_trophies", "avatar_url", "banner_url", "has_boost", "boost_public_id"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        sets = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates))
        vals = [clan_id] + list(updates.values())
        await self.execute(f"UPDATE clans SET {sets} WHERE id = $1", *vals)
        return True

    async def delete_clan(self, clan_id: int) -> bool:
        await self.execute("UPDATE users SET squad_id = 0 WHERE squad_id = $1", clan_id)
        await self.execute("DELETE FROM clan_members WHERE clan_id = $1", clan_id)
        await self.execute("DELETE FROM clan_join_requests WHERE clan_id = $1", clan_id)
        result = await self.execute("DELETE FROM clans WHERE id = $1", clan_id)
        return "DELETE 1" in str(result)

    async def transfer_ownership(self, clan_id: int, from_id: int, to_id: int) -> bool:
        to_role = await self.fetchval(
            "SELECT role FROM clan_members WHERE clan_id = $1 AND user_id = $2", clan_id, to_id
        )
        if not to_role:
            return False
        if int(from_id or 0) == int(to_id or 0):
            return True
        await self.execute("UPDATE clans SET owner_id = $1 WHERE id = $2", to_id, clan_id)
        await self.execute(
            "UPDATE clan_members SET role = 'creator' WHERE clan_id = $1 AND user_id = $2", clan_id, to_id
        )
        await self.execute(
            "UPDATE clan_members SET role = 'member' WHERE clan_id = $1 AND user_id = $2", clan_id, from_id
        )
        await self._log_clan_activity(clan_id, "transfer", "Владение передано", user_id=from_id, target_user_id=to_id)
        return True

    async def delete_clan_by_owner(self, clan_id: int, actor_id: int) -> bool:
        role = await self.get_member_role(clan_id, actor_id)
        if role != "creator":
            raise ValueError("only_creator_can_delete")
        await self._log_clan_activity(clan_id, "delete", "Сквад распущен", user_id=actor_id)
        return await self.delete_clan(clan_id)

    # ── Membership ─────────────────────────────────────────────────────

    async def clan_join(self, clan_id: int, user_id: int) -> dict:
        existing = await self.fetchval(
            "SELECT clan_id FROM clan_members WHERE user_id = $1", user_id
        )
        if existing:
            raise ValueError("already_in_clan")

        row = await self.fetchrow(
            """
            UPDATE clans SET members_count = members_count + 1
            WHERE id = $1 AND members_count < max_members
            RETURNING id, members_count, max_members
            """,
            clan_id,
        )
        if not row:
            raise ValueError("clan_full")

        await self.execute(
            "INSERT INTO clan_members (clan_id, user_id, role) VALUES ($1, $2, 'member')",
            clan_id, user_id,
        )
        await self.execute("UPDATE users SET squad_id = $1 WHERE user_id = $2", clan_id, user_id)
        await self.ensure_current_squad_trophy_snapshot(user_id, clan_id=clan_id)
        nick = await self.fetchval(
            "SELECT COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок') FROM users u LEFT JOIN profiles p ON p.user_id = u.user_id WHERE u.user_id = $1",
            user_id,
        )
        await self._log_clan_activity(clan_id, "join", f"{nick} вступил в сквад", user_id=user_id)
        return dict(row)

    async def random_join_squad(self, user_id: int) -> dict:
        existing = await self.fetchval(
            "SELECT clan_id FROM clan_members WHERE user_id = $1", user_id
        )
        if existing:
            raise ValueError("already_in_clan")
        trophies = await self.fetchval("SELECT trophies FROM users WHERE user_id = $1", user_id)
        trophies = int(trophies or 0)
        row = await self.fetchrow(
            """
            SELECT id
            FROM clans
            WHERE type = 'open'
              AND members_count < max_members
              AND min_trophies <= $1
            ORDER BY random()
            LIMIT 1
            """,
            trophies,
        )
        if not row:
            raise ValueError("no_suitable_squad")
        await self.clan_join(int(row["id"]), user_id)
        clan = await self.get_user_clan(user_id)
        return clan or {"id": int(row["id"])}

    async def clan_leave(self, clan_id: int, user_id: int) -> bool:
        role = await self.fetchval(
            "SELECT role FROM clan_members WHERE clan_id = $1 AND user_id = $2", clan_id, user_id
        )
        if not role:
            return False
        if role == "creator":
            count = await self.fetchval("SELECT members_count FROM clans WHERE id = $1", clan_id)
            if count and count > 1:
                raise ValueError("creator_must_transfer_or_delete")

        await self.execute(
            "DELETE FROM clan_members WHERE clan_id = $1 AND user_id = $2", clan_id, user_id
        )
        await self.execute("UPDATE users SET squad_id = 0 WHERE user_id = $1", user_id)
        remaining = await self.fetchval(
            "UPDATE clans SET members_count = GREATEST(members_count - 1, 0) WHERE id = $1 RETURNING members_count",
            clan_id,
        )
        nick = await self.fetchval(
            "SELECT COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок') FROM users u LEFT JOIN profiles p ON p.user_id = u.user_id WHERE u.user_id = $1",
            user_id,
        )
        if remaining is not None and remaining <= 0:
            await self.delete_clan(clan_id)
        else:
            await self._log_clan_activity(clan_id, "leave", f"{nick} покинул сквад", user_id=user_id)
        return True

    async def get_clan_members(self, clan_id: int, search: Optional[str] = None) -> list[dict]:
        query = """
            SELECT cm.user_id,
                   CASE WHEN cm.role = 'owner' THEN 'creator' ELSE cm.role END AS role,
                   cm.personal_tokens, cm.joined_at,
                   COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок #' || cm.user_id::TEXT) AS nick,
                   COALESCE(equipped_avatar.asset_path, NULLIF(p.img, '')) AS avatar_url,
                   equipped_background.asset_path AS profile_background_url,
                   COALESCE(equipped_title.name, NULLIF(p.title, ''), 'Игрок') AS title,
                   COALESCE(equipped_title.class, 'starter') AS title_class,
                   COALESCE(u.trophies, 0) AS trophies,
                   COALESCE(u.is_bot, FALSE) AS is_bot,
                   CASE WHEN u.updated_at > NOW() - INTERVAL '5 minutes' THEN true ELSE false END AS online
            FROM clan_members cm
            LEFT JOIN users u ON u.user_id = cm.user_id
            LEFT JOIN profiles p ON p.user_id = cm.user_id
            LEFT JOIN user_equipped_cosmetics uec_avatar ON uec_avatar.user_id = cm.user_id AND uec_avatar.item_type = 'avatar'
            LEFT JOIN cosmetic_items equipped_avatar ON equipped_avatar.id = uec_avatar.cosmetic_id AND equipped_avatar.item_type = 'avatar'
            LEFT JOIN user_equipped_cosmetics uec_background ON uec_background.user_id = cm.user_id AND uec_background.item_type = 'profile_background'
            LEFT JOIN cosmetic_items equipped_background ON equipped_background.id = uec_background.cosmetic_id AND equipped_background.item_type = 'profile_background'
            LEFT JOIN user_equipped_cosmetics uec_title ON uec_title.user_id = cm.user_id AND uec_title.item_type = 'title'
            LEFT JOIN cosmetic_items equipped_title ON equipped_title.id = uec_title.cosmetic_id AND equipped_title.item_type = 'title'
            WHERE cm.clan_id = $1
        """
        params: list = [clan_id]
        if search:
            query += " AND (COALESCE(p.custom_nickname, u.username, u.first_name) ILIKE $2)"
            params.append(f"%{search}%")
        query += """
            ORDER BY
                CASE cm.role WHEN 'creator' THEN 0 WHEN 'officer' THEN 1 ELSE 2 END,
                u.trophies DESC
        """
        rows = await self.fetch(query, *params)
        return [dict(r) for r in rows]

    async def get_member_role(self, clan_id: int, user_id: int) -> Optional[str]:
        role = await self.fetchval(
            "SELECT role FROM clan_members WHERE clan_id = $1 AND user_id = $2", clan_id, user_id
        )
        return "creator" if role == "owner" else role

    async def kick_member(self, clan_id: int, actor_id: int, target_id: int) -> bool:
        actor_role = await self.get_member_role(clan_id, actor_id)
        target_role = await self.get_member_role(clan_id, target_id)
        if not actor_role or actor_role not in ("creator", "officer"):
            raise ValueError("no_permission")
        if not target_role or target_role == "creator":
            raise ValueError("cannot_kick")
        if actor_role == "officer" and target_role == "officer":
            raise ValueError("cannot_kick_officer")

        await self.execute("DELETE FROM clan_members WHERE clan_id = $1 AND user_id = $2", clan_id, target_id)
        await self.execute("UPDATE users SET squad_id = 0 WHERE user_id = $1", target_id)
        await self.execute(
            "UPDATE clans SET members_count = GREATEST(members_count - 1, 0) WHERE id = $1", clan_id
        )
        nick = await self.fetchval(
            "SELECT COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок') FROM users u LEFT JOIN profiles p ON p.user_id = u.user_id WHERE u.user_id = $1",
            target_id,
        )
        await self._log_clan_activity(clan_id, "kick", f"{nick} исключён из сквада", user_id=actor_id, target_user_id=target_id)
        return True

    async def promote_member(self, clan_id: int, actor_id: int, target_id: int) -> bool:
        from infrastructure.clan_config import CLAN_MAX_OFFICERS
        actor_role = await self.get_member_role(clan_id, actor_id)
        if actor_role != "creator":
            raise ValueError("only_creator_can_promote")
        target_role = await self.get_member_role(clan_id, target_id)
        if target_role != "member":
            raise ValueError("already_officer_or_creator")
        officer_count = await self.fetchval(
            "SELECT COUNT(*) FROM clan_members WHERE clan_id = $1 AND role = 'officer'", clan_id
        )
        if officer_count >= CLAN_MAX_OFFICERS:
            raise ValueError("max_officers_reached")
        await self.execute(
            "UPDATE clan_members SET role = 'officer' WHERE clan_id = $1 AND user_id = $2", clan_id, target_id
        )
        nick = await self.fetchval(
            "SELECT COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок') FROM users u LEFT JOIN profiles p ON p.user_id = u.user_id WHERE u.user_id = $1",
            target_id,
        )
        await self._log_clan_activity(clan_id, "promote", f"{nick} повышен до Офицера", user_id=actor_id, target_user_id=target_id)
        return True

    async def demote_member(self, clan_id: int, actor_id: int, target_id: int) -> bool:
        actor_role = await self.get_member_role(clan_id, actor_id)
        if actor_role != "creator":
            raise ValueError("only_creator_can_demote")
        target_role = await self.get_member_role(clan_id, target_id)
        if target_role != "officer":
            raise ValueError("not_an_officer")
        await self.execute(
            "UPDATE clan_members SET role = 'member' WHERE clan_id = $1 AND user_id = $2", clan_id, target_id
        )
        nick = await self.fetchval(
            "SELECT COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок') FROM users u LEFT JOIN profiles p ON p.user_id = u.user_id WHERE u.user_id = $1",
            target_id,
        )
        await self._log_clan_activity(clan_id, "demote", f"{nick} понижен до Участника", user_id=actor_id, target_user_id=target_id)
        return True

    # ── Join Requests ──────────────────────────────────────────────────

    async def create_join_request(self, clan_id: int, user_id: int) -> dict:
        existing = await self.fetchval(
            "SELECT clan_id FROM clan_members WHERE user_id = $1", user_id
        )
        if existing:
            raise ValueError("already_in_clan")
        pending = await self.fetchval(
            "SELECT id FROM clan_join_requests WHERE clan_id = $1 AND user_id = $2 AND status = 'pending'",
            clan_id, user_id,
        )
        if pending:
            raise ValueError("request_already_exists")
        row = await self.fetchrow(
            """
            INSERT INTO clan_join_requests (clan_id, user_id) VALUES ($1, $2)
            ON CONFLICT (clan_id, user_id) DO UPDATE SET status = 'pending', created_at = NOW(), decided_by = NULL, decided_at = NULL
            RETURNING *
            """,
            clan_id, user_id,
        )
        return dict(row)

    async def get_join_requests(self, clan_id: int, status: str = "pending") -> list[dict]:
        rows = await self.fetch(
            """
            SELECT jr.*, COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок #' || jr.user_id::TEXT) AS nick,
                   COALESCE(u.trophies, 0) AS trophies,
                   COALESCE(u.is_bot, FALSE) AS is_bot
            FROM clan_join_requests jr
            LEFT JOIN users u ON u.user_id = jr.user_id
            LEFT JOIN profiles p ON p.user_id = jr.user_id
            WHERE jr.clan_id = $1 AND jr.status = $2
            ORDER BY jr.created_at DESC
            """,
            clan_id, status,
        )
        return [dict(r) for r in rows]

    async def accept_join_request(self, request_id: int, decided_by: int) -> dict:
        row = await self.fetchrow(
            "SELECT * FROM clan_join_requests WHERE id = $1 AND status = 'pending'", request_id
        )
        if not row:
            raise ValueError("request_not_found")
        req = dict(row)
        await self.clan_join(req["clan_id"], req["user_id"])
        updated = await self.fetchrow(
            """
            UPDATE clan_join_requests
            SET status = 'accepted', decided_by = $1, decided_at = NOW()
            WHERE id = $2
            RETURNING *
            """,
            decided_by, request_id,
        )
        return dict(updated)

    async def reject_join_request(self, request_id: int, decided_by: int) -> dict:
        row = await self.fetchrow(
            "SELECT * FROM clan_join_requests WHERE id = $1 AND status = 'pending'", request_id
        )
        if not row:
            raise ValueError("request_not_found")
        updated = await self.fetchrow(
            """
            UPDATE clan_join_requests
            SET status = 'rejected', decided_by = $1, decided_at = NOW()
            WHERE id = $2
            RETURNING *
            """,
            decided_by, request_id,
        )
        return dict(updated)

    async def get_user_pending_request(self, user_id: int, clan_id: int) -> Optional[dict]:
        row = await self.fetchrow(
            "SELECT * FROM clan_join_requests WHERE user_id = $1 AND clan_id = $2 AND status = 'pending'",
            user_id, clan_id,
        )
        return dict(row) if row else None

    # ── Search ─────────────────────────────────────────────────────────

    async def search_clans(
        self, query: Optional[str] = None, filter_type: Optional[str] = None,
        sort: str = "cbrp", limit: int = 20, offset: int = 0,
    ) -> list[dict]:
        where_clauses = []
        params: list = []
        idx = 1

        if query:
            query_text = query.strip()
            if query_text.isdigit():
                public_value = int(query_text)
                where_clauses.append(
                    f"(c.public_id = ${idx} OR (c.has_boost = TRUE AND c.boost_public_id = ${idx}))"
                )
                params.append(public_value)
                idx += 1
            else:
                where_clauses.append(f"(c.name ILIKE ${idx} OR c.tag ILIKE ${idx})")
                params.append(f"%{query_text}%")
                idx += 1

        if filter_type == "open":
            where_clauses.append("c.type = 'open' AND c.members_count < c.max_members")
        elif filter_type == "boost":
            where_clauses.append("c.has_boost = true")

        where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        order = "c.cbrp DESC" if sort == "cbrp" else "c.members_count DESC"

        rows = await self.fetch(
            f"""
            SELECT c.* FROM clans c
            {where}
            ORDER BY {order}
            LIMIT ${idx} OFFSET ${idx + 1}
            """,
            *params, limit, offset,
        )
        return [dict(r) for r in rows]

    # ── Admin squads / clans ──────────────────────────────────────────

    async def get_admin_squads_analytics(self, days: int = 30) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        days = max(1, min(int(days or 30), 365))

        summary = await self.fetchrow(
            """
            SELECT
                COUNT(*)::INT AS total_squads,
                COUNT(*) FILTER (WHERE has_boost = TRUE)::INT AS boosted_squads,
                (
                    SELECT COUNT(*)::INT
                    FROM clan_members cm
                    JOIN users u ON u.user_id = cm.user_id
                    WHERE COALESCE(u.is_bot, FALSE) = FALSE
                ) AS total_members,
                (
                    SELECT COUNT(*)::INT
                    FROM clan_members cm
                    JOIN users u ON u.user_id = cm.user_id
                    WHERE COALESCE(u.is_bot, FALSE) = TRUE
                ) AS bot_members,
                COALESCE(SUM(max_members), 0)::INT AS total_slots,
                COALESCE(SUM(cbrp), 0)::INT AS total_cbrp,
                COALESCE(SUM(treasury_tokens), 0)::INT AS total_treasury,
                COUNT(*) FILTER (WHERE created_at >= NOW() - ($1::INT * INTERVAL '1 day'))::INT AS new_squads
            FROM clans
            """,
            days,
        )
        requests_summary = await self.fetch(
            """
            SELECT status, COUNT(*)::INT AS count
            FROM clan_join_requests
            WHERE created_at >= NOW() - ($1::INT * INTERVAL '1 day')
            GROUP BY status
            ORDER BY count DESC
            """,
            days,
        )
        growth = await self.fetch(
            """
            SELECT DATE(created_at) AS day, COUNT(*)::INT AS count
            FROM clans
            WHERE created_at >= NOW() - ($1::INT * INTERVAL '1 day')
            GROUP BY DATE(created_at)
            ORDER BY day
            """,
            days,
        )
        top_cbrp = await self.fetch(
            """
            SELECT id, public_id, boost_public_id, name, tag, members_count, max_members,
                   has_boost, cbrp, treasury_tokens,
                   (
                       SELECT COUNT(*)::INT
                       FROM clan_members cm
                       JOIN users u ON u.user_id = cm.user_id
                       WHERE cm.clan_id = c.id AND COALESCE(u.is_bot, FALSE) = FALSE
                   ) AS human_members,
                   (
                       SELECT COUNT(*)::INT
                       FROM clan_members cm
                       JOIN users u ON u.user_id = cm.user_id
                       WHERE cm.clan_id = c.id AND COALESCE(u.is_bot, FALSE) = TRUE
                   ) AS bot_members
            FROM clans c
            ORDER BY cbrp DESC, members_count DESC
            LIMIT 12
            """
        )
        top_treasury = await self.fetch(
            """
            SELECT id, public_id, name, tag, members_count, treasury_tokens, cbrp,
                   (
                       SELECT COUNT(*)::INT
                       FROM clan_members cm
                       JOIN users u ON u.user_id = cm.user_id
                       WHERE cm.clan_id = c.id AND COALESCE(u.is_bot, FALSE) = FALSE
                   ) AS human_members,
                   (
                       SELECT COUNT(*)::INT
                       FROM clan_members cm
                       JOIN users u ON u.user_id = cm.user_id
                       WHERE cm.clan_id = c.id AND COALESCE(u.is_bot, FALSE) = TRUE
                   ) AS bot_members
            FROM clans c
            ORDER BY treasury_tokens DESC, cbrp DESC
            LIMIT 12
            """
        )
        member_roles = await self.fetch(
            """
            SELECT CASE WHEN role = 'owner' THEN 'creator' ELSE role END AS role,
                   COUNT(*)::INT AS count,
                   COALESCE(SUM(personal_tokens), 0)::INT AS personal_tokens
            FROM clan_members cm
            JOIN users u ON u.user_id = cm.user_id
            WHERE COALESCE(u.is_bot, FALSE) = FALSE
            GROUP BY CASE WHEN role = 'owner' THEN 'creator' ELSE role END
            ORDER BY count DESC
            """
        )
        cbrp_events = await self.fetch(
            """
            SELECT event_type,
                   COUNT(*)::INT AS count,
                   COALESCE(SUM(cbrp), 0)::INT AS cbrp,
                   COALESCE(SUM(personal_tokens), 0)::INT AS personal_tokens,
                   COALESCE(SUM(treasury_tokens), 0)::INT AS treasury_tokens
            FROM squad_cbrp_events
            WHERE created_at >= NOW() - ($1::INT * INTERVAL '1 day')
            GROUP BY event_type
            ORDER BY cbrp DESC, count DESC
            """,
            days,
        )
        activity = await self.fetch(
            """
            SELECT type, COUNT(*)::INT AS count
            FROM clan_activity
            WHERE created_at >= NOW() - ($1::INT * INTERVAL '1 day')
            GROUP BY type
            ORDER BY count DESC
            """,
            days,
        )
        upgrades = await self.fetch(
            """
            SELECT upgrade_type, level, COUNT(*)::INT AS squads
            FROM clan_upgrades
            GROUP BY upgrade_type, level
            ORDER BY upgrade_type, level
            """
        )
        purchases = await self.fetch(
            """
            SELECT reward_id,
                   COUNT(*)::INT AS count,
                   COALESCE(SUM(cost), 0)::INT AS tokens_spent
            FROM squad_shop_purchases
            WHERE created_at >= NOW() - ($1::INT * INTERVAL '1 day')
            GROUP BY reward_id
            ORDER BY count DESC, tokens_spent DESC
            LIMIT 20
            """,
            days,
        )
        snapshots = await self.fetch(
            """
            SELECT period_key,
                   COUNT(*)::INT AS rows,
                   COALESCE(SUM(delta_trophies), 0)::INT AS delta_trophies,
                   COALESCE(SUM(cbrp_awarded), 0)::INT AS cbrp_awarded,
                   COALESCE(SUM(personal_tokens_awarded), 0)::INT AS personal_tokens_awarded,
                   COALESCE(SUM(treasury_tokens_awarded), 0)::INT AS treasury_tokens_awarded
            FROM squad_trophy_snapshots
            GROUP BY period_key
            ORDER BY period_key DESC
            LIMIT 12
            """
        )
        config = await self.get_squad_runtime_config()

        return _json_safe({
            "summary": dict(summary or {}),
            "requests": [dict(r) for r in requests_summary],
            "growth": [dict(r) for r in growth],
            "top_cbrp": [dict(r) for r in top_cbrp],
            "top_treasury": [dict(r) for r in top_treasury],
            "member_roles": [dict(r) for r in member_roles],
            "cbrp_events": [dict(r) for r in cbrp_events],
            "activity": [dict(r) for r in activity],
            "upgrades": [dict(r) for r in upgrades],
            "purchases": [dict(r) for r in purchases],
            "snapshots": [dict(r) for r in snapshots],
            "config": config,
        })

    async def search_admin_squads(
        self,
        query: Optional[str] = None,
        filter_type: str = "all",
        sort: str = "cbrp",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        limit = max(1, min(int(limit or 50), 200))
        offset = max(0, int(offset or 0))
        where: list[str] = []
        params: list[Any] = []

        if query:
            q = str(query).strip()
            if q.isdigit():
                params.append(int(q))
                where.append(
                    f"(c.id = ${len(params)} OR c.public_id = ${len(params)} OR c.boost_public_id = ${len(params)})"
                )
            else:
                params.append(f"%{q}%")
                where.append(f"(c.name ILIKE ${len(params)} OR c.tag ILIKE ${len(params)} OR c.description ILIKE ${len(params)})")

        if filter_type == "open":
            where.append("c.type = 'open'")
        elif filter_type == "closed":
            where.append("c.type = 'closed'")
        elif filter_type == "boost":
            where.append("c.has_boost = TRUE")
        elif filter_type == "full":
            where.append("c.members_count >= c.max_members")

        where_sql = "WHERE " + " AND ".join(where) if where else ""
        order_sql = {
            "members": "c.members_count DESC, c.cbrp DESC",
            "treasury": "c.treasury_tokens DESC, c.cbrp DESC",
            "created": "c.created_at DESC",
            "rank": "c.rank ASC, c.cbrp DESC",
        }.get(sort, "c.cbrp DESC, c.members_count DESC")

        total = await self.fetchval(f"SELECT COUNT(*) FROM clans c {where_sql}", *params)
        rows = await self.fetch(
            f"""
            SELECT c.*,
                   COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок #' || c.owner_id::TEXT) AS owner_name,
                   (SELECT COUNT(*) FROM clan_join_requests jr WHERE jr.clan_id = c.id AND jr.status = 'pending')::INT AS pending_requests,
                   (SELECT COUNT(*) FROM clan_activity ca WHERE ca.clan_id = c.id AND ca.created_at >= NOW() - INTERVAL '7 days')::INT AS activity_7d,
                   (
                       SELECT COUNT(*)::INT
                       FROM clan_members cm
                       JOIN users mu ON mu.user_id = cm.user_id
                       WHERE cm.clan_id = c.id AND COALESCE(mu.is_bot, FALSE) = FALSE
                   ) AS human_members,
                   (
                       SELECT COUNT(*)::INT
                       FROM clan_members cm
                       JOIN users mu ON mu.user_id = cm.user_id
                       WHERE cm.clan_id = c.id AND COALESCE(mu.is_bot, FALSE) = TRUE
                   ) AS bot_members
            FROM clans c
            LEFT JOIN users u ON u.user_id = c.owner_id
            LEFT JOIN profiles p ON p.user_id = c.owner_id
            {where_sql}
            ORDER BY {order_sql}
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """,
            *params,
            limit,
            offset,
        )
        return _json_safe({"total": int(total or 0), "squads": [dict(r) for r in rows]})

    async def get_admin_squad_detail(self, clan_id: int) -> dict[str, Any]:
        clan = await self.fetchrow(
            """
            SELECT c.*,
                   COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок #' || c.owner_id::TEXT) AS owner_name
            FROM clans c
            LEFT JOIN users u ON u.user_id = c.owner_id
            LEFT JOIN profiles p ON p.user_id = c.owner_id
            WHERE c.id = $1
            """,
            int(clan_id),
        )
        if not clan:
            return {"error": "clan_not_found"}
        members = await self.get_clan_members(int(clan_id))
        requests = await self.fetch(
            """
            SELECT jr.*, COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок #' || jr.user_id::TEXT) AS nick,
                   COALESCE(u.trophies, 0) AS trophies,
                   COALESCE(u.is_bot, FALSE) AS is_bot
            FROM clan_join_requests jr
            LEFT JOIN users u ON u.user_id = jr.user_id
            LEFT JOIN profiles p ON p.user_id = jr.user_id
            WHERE jr.clan_id = $1
            ORDER BY jr.created_at DESC
            LIMIT 80
            """,
            int(clan_id),
        )
        activity = await self.get_clan_activity(int(clan_id), limit=80)
        events = await self.get_squad_cbrp_events(clan_id=int(clan_id), limit=80)
        upgrades = await self.get_clan_upgrades(int(clan_id))
        purchases = await self.fetch(
            """
            SELECT sp.*, COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок #' || sp.user_id::TEXT) AS nick
            FROM squad_shop_purchases sp
            LEFT JOIN users u ON u.user_id = sp.user_id
            LEFT JOIN profiles p ON p.user_id = sp.user_id
            WHERE sp.clan_id = $1
            ORDER BY sp.created_at DESC
            LIMIT 80
            """,
            int(clan_id),
        )
        snapshots = await self.fetch(
            """
            SELECT st.*, COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок #' || st.user_id::TEXT) AS nick
            FROM squad_trophy_snapshots st
            LEFT JOIN users u ON u.user_id = st.user_id
            LEFT JOIN profiles p ON p.user_id = st.user_id
            WHERE st.clan_id = $1
            ORDER BY st.created_at DESC
            LIMIT 80
            """,
            int(clan_id),
        )
        return _json_safe({
            "clan": dict(clan),
            "members": members,
            "requests": [dict(r) for r in requests],
            "activity": activity,
            "events": events,
            "upgrades": upgrades,
            "purchases": [dict(r) for r in purchases],
            "snapshots": [dict(r) for r in snapshots],
        })

    async def admin_update_squad(
        self,
        admin_user_id: int,
        clan_id: int,
        fields: dict[str, Any],
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        allowed = {
            "name": "name",
            "tag": "tag",
            "description": "description",
            "type": "type",
            "min_trophies": "min_trophies",
            "max_members": "max_members",
            "has_boost": "has_boost",
            "boost_public_id": "boost_public_id",
            "avatar_url": "avatar_url",
            "banner_url": "banner_url",
            "owner_id": "owner_id",
            "rank": "rank",
            "trophies": "trophies",
            "cbrp": "cbrp",
            "treasury_tokens": "treasury_tokens",
        }
        exists = await self.fetchval("SELECT 1 FROM clans WHERE id = $1", int(clan_id))
        if not exists:
            return {"error": "clan_not_found"}
        updates: list[str] = []
        values: list[Any] = []
        normalized: dict[str, Any] = {}
        for key, column in allowed.items():
            if key not in fields:
                continue
            value = fields.get(key)
            if key in {"min_trophies", "max_members", "boost_public_id", "owner_id", "rank", "trophies", "cbrp", "treasury_tokens"}:
                value = None if value in ("", None) and key == "boost_public_id" else max(0, int(value or 0))
                if key == "max_members":
                    current_members = await self.fetchval(
                        "SELECT COUNT(*) FROM clan_members WHERE clan_id = $1",
                        int(clan_id),
                    ) or 0
                    value = max(int(value or 0), int(current_members or 0))
                elif key == "owner_id" and value:
                    user_exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", int(value))
                    if not user_exists:
                        return {"error": "owner_not_found"}
            elif key == "has_boost":
                value = bool(value)
            elif key == "type":
                value = str(value or "open").strip().lower()
                if value not in {"open", "closed"}:
                    return {"error": "invalid_type"}
            elif value is not None:
                value = str(value).strip()
            updates.append(f"{column} = ${len(values) + 1}")
            values.append(value)
            normalized[key] = value
        if not updates:
            return {"error": "no_valid_fields"}
        values.append(int(clan_id))
        await self.execute(
            f"UPDATE clans SET {', '.join(updates)} WHERE id = ${len(values)}",
            *values,
        )
        if "owner_id" in normalized and normalized["owner_id"]:
            await self.execute(
                """
                UPDATE clan_members
                SET role = 'member'
                WHERE clan_id = $1
                  AND user_id <> $2
                  AND role IN ('creator', 'owner')
                """,
                int(clan_id),
                int(normalized["owner_id"]),
            )
            await self.execute(
                """
                INSERT INTO clan_members (clan_id, user_id, role)
                VALUES ($1, $2, 'creator')
                ON CONFLICT (clan_id, user_id) DO UPDATE SET role = 'creator'
                """,
                int(clan_id),
                int(normalized["owner_id"]),
            )
            await self.execute(
                "UPDATE users SET squad_id = $1 WHERE user_id = $2",
                int(clan_id),
                int(normalized["owner_id"]),
            )
            await self.execute(
                "UPDATE clans SET members_count = (SELECT COUNT(*) FROM clan_members WHERE clan_id = $1) WHERE id = $1",
                int(clan_id),
            )
        await self._log_clan_activity(
            int(clan_id),
            "admin_update",
            "Админ изменил настройки сквада" + (f": {reason}" if reason else ""),
            user_id=admin_user_id,
        )
        return {"status": "ok", "fields": normalized}

    async def admin_adjust_squad_balance(
        self,
        admin_user_id: int,
        clan_id: int,
        resource: str,
        amount: int,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        if resource not in {"cbrp", "treasury_tokens"}:
            return {"error": "invalid_resource"}
        exists = await self.fetchval("SELECT 1 FROM clans WHERE id = $1", int(clan_id))
        if not exists:
            return {"error": "clan_not_found"}
        amount = int(amount or 0)
        await self.execute(
            f"UPDATE clans SET {resource} = GREATEST({resource} + $1, 0) WHERE id = $2",
            amount,
            int(clan_id),
        )
        await self._log_clan_activity(
            int(clan_id),
            "admin_balance",
            f"Админ изменил {resource} на {amount:+d}" + (f": {reason}" if reason else ""),
            user_id=admin_user_id,
        )
        return {"status": "ok", "resource": resource, "amount": amount}

    async def admin_squad_member_action(
        self,
        admin_user_id: int,
        clan_id: int,
        action: str,
        target_user_id: int,
        *,
        personal_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        clan_id = int(clan_id)
        target_user_id = int(target_user_id)
        action = str(action or "").strip()
        exists = await self.fetchval("SELECT 1 FROM clans WHERE id = $1", clan_id)
        if not exists:
            return {"error": "clan_not_found"}
        user_exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", target_user_id)
        if not user_exists:
            return {"error": "user_not_found"}
        if action == "add":
            old_clan = await self.fetchval("SELECT clan_id FROM clan_members WHERE user_id = $1", target_user_id)
            if old_clan and int(old_clan) != clan_id:
                await self.execute("DELETE FROM clan_members WHERE user_id = $1", target_user_id)
                await self.execute(
                    "UPDATE clans SET members_count = GREATEST(members_count - 1, 0) WHERE id = $1",
                    int(old_clan),
                )
            await self.execute(
                """
                INSERT INTO clan_members (clan_id, user_id, role)
                VALUES ($1, $2, 'member')
                ON CONFLICT (clan_id, user_id) DO NOTHING
                """,
                clan_id,
                target_user_id,
            )
            await self.execute("UPDATE users SET squad_id = $1 WHERE user_id = $2", clan_id, target_user_id)
            await self.execute(
                "UPDATE clans SET members_count = (SELECT COUNT(*) FROM clan_members WHERE clan_id = $1) WHERE id = $1",
                clan_id,
            )
        elif action == "kick":
            is_member = await self.fetchval(
                "SELECT 1 FROM clan_members WHERE clan_id = $1 AND user_id = $2",
                clan_id,
                target_user_id,
            )
            if not is_member:
                return {"error": "member_not_found"}
            await self.execute("DELETE FROM clan_members WHERE clan_id = $1 AND user_id = $2", clan_id, target_user_id)
            await self.execute("UPDATE users SET squad_id = 0 WHERE user_id = $1 AND squad_id = $2", target_user_id, clan_id)
            await self.execute(
                "UPDATE clans SET members_count = (SELECT COUNT(*) FROM clan_members WHERE clan_id = $1) WHERE id = $1",
                clan_id,
            )
        elif action in {"promote", "demote"}:
            target_role = await self.get_member_role(clan_id, target_user_id)
            if not target_role:
                return {"error": "member_not_found"}
            if target_role == "creator":
                return {"error": "cannot_change_owner_role"}
            role = "officer" if action == "promote" else "member"
            await self.execute(
                "UPDATE clan_members SET role = $1 WHERE clan_id = $2 AND user_id = $3",
                role,
                clan_id,
                target_user_id,
            )
        elif action == "transfer":
            old_owner = await self.fetchval("SELECT owner_id FROM clans WHERE id = $1", clan_id)
            transferred = await self.transfer_ownership(clan_id, old_owner, target_user_id)
            if not transferred:
                return {"error": "member_not_found"}
        elif action == "set_tokens":
            is_member = await self.fetchval(
                "SELECT 1 FROM clan_members WHERE clan_id = $1 AND user_id = $2",
                clan_id,
                target_user_id,
            )
            if not is_member:
                return {"error": "member_not_found"}
            await self.execute(
                "UPDATE clan_members SET personal_tokens = $1 WHERE clan_id = $2 AND user_id = $3",
                max(0, int(personal_tokens or 0)),
                clan_id,
                target_user_id,
            )
        else:
            return {"error": "invalid_action"}
        await self._log_clan_activity(
            clan_id,
            "admin_member",
            f"Админ выполнил действие {action} для {target_user_id}",
            user_id=admin_user_id,
            target_user_id=target_user_id,
        )
        return {"status": "ok", "action": action, "target_user_id": target_user_id}

    # ── Activity ───────────────────────────────────────────────────────

    async def _log_clan_activity(
        self, clan_id: int, activity_type: str, text: str,
        user_id: Optional[int] = None, target_user_id: Optional[int] = None,
    ) -> None:
        await self.execute(
            """
            INSERT INTO clan_activity (clan_id, type, text, user_id, target_user_id)
            VALUES ($1, $2, $3, $4, $5)
            """,
            clan_id, activity_type, text, user_id, target_user_id,
        )

    async def get_clan_activity(self, clan_id: int, limit: int = 20) -> list[dict]:
        rows = await self.fetch(
            """
            SELECT ca.id, ca.type, ca.text, ca.user_id, ca.target_user_id, ca.created_at
            FROM clan_activity ca
            WHERE ca.clan_id = $1
            ORDER BY ca.created_at DESC
            LIMIT $2
            """,
            clan_id, limit,
        )
        return [dict(r) for r in rows]

    # ── Shop / Upgrades ────────────────────────────────────────────────

    async def get_clan_upgrades(self, clan_id: int) -> dict:
        rows = await self.fetch(
            "SELECT upgrade_type, level FROM clan_upgrades WHERE clan_id = $1", clan_id
        )
        return {r["upgrade_type"]: r["level"] for r in rows}

    async def get_squad_shop_state(self, clan_id: int, user_id: int) -> dict[str, Any]:
        config = await self.get_squad_runtime_config()
        upgrades = await self.get_clan_upgrades(clan_id)
        has_boost = await self.fetchval("SELECT COALESCE(has_boost, FALSE) FROM clans WHERE id = $1", clan_id)
        if "boost" in (config.get("squad_upgrades") or {}):
            upgrades["boost"] = 1 if has_boost else 0
        purchased_rows = await self.fetch(
            "SELECT reward_id, cost, metadata, created_at FROM squad_shop_purchases WHERE user_id = $1 ORDER BY created_at DESC",
            user_id,
        )
        return {
            "upgrades": upgrades,
            "upgrade_catalog": config.get("squad_upgrades") or {},
            "personal_rewards": config.get("squad_personal_rewards") or [],
            "purchases": [dict(r) for r in purchased_rows],
        }

    async def buy_clan_upgrade(self, clan_id: int, actor_id: int, upgrade_type: str) -> dict[str, Any]:
        config = await self.get_squad_runtime_config()
        catalog = config.get("squad_upgrades") or {}
        upgrade_cfg = catalog.get(upgrade_type)
        if not upgrade_cfg:
            raise ValueError("unknown_upgrade")

        upgrades = await self.get_clan_upgrades(clan_id)
        current_level = 0 if upgrade_type == "boost" else int(upgrades.get(upgrade_type, 0) or 0)
        next_level = current_level + 1
        level_cfg = None
        for item in upgrade_cfg.get("levels", []):
            if int(item.get("level", 0)) == next_level:
                level_cfg = item
                break
        if not level_cfg:
            raise ValueError("max_level_reached")

        cost = max(0, int(level_cfg.get("cost") or 0))
        boost_public_id: int | None = None
        if upgrade_type == "boost":
            existing_boost_id = await self.fetchval(
                "SELECT boost_public_id FROM clans WHERE id = $1", clan_id
            )
            boost_public_id = int(existing_boost_id) if existing_boost_id else await self._generate_unique_clan_public_id(
                digits=3,
                column="boost_public_id",
            )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT c.treasury_tokens, c.max_members, c.has_boost, c.boost_public_id,
                           CASE WHEN cm.role = 'owner' THEN 'creator' ELSE cm.role END AS role
                    FROM clans c
                    JOIN clan_members cm ON cm.clan_id = c.id AND cm.user_id = $2
                    WHERE c.id = $1
                    FOR UPDATE OF c
                    """,
                    clan_id,
                    actor_id,
                )
                if not row:
                    raise ValueError("not_in_squad")
                if row["role"] not in ("creator", "officer"):
                    raise ValueError("no_permission")
                if upgrade_type == "boost" and bool(row["has_boost"]):
                    raise ValueError("boost_already_active")
                if int(row["treasury_tokens"] or 0) < cost:
                    raise ValueError("insufficient_treasury")

                slots_added = max(0, int(level_cfg.get("slots_added") or 0))
                if upgrade_type == "boost":
                    boost_public_id = int(row["boost_public_id"] or boost_public_id)
                    await conn.execute(
                        """
                        UPDATE clans
                        SET treasury_tokens = treasury_tokens - $1,
                            has_boost = TRUE,
                            boost_public_id = COALESCE(boost_public_id, $2)
                        WHERE id = $3
                        """,
                        cost,
                        boost_public_id,
                        clan_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE clans
                        SET treasury_tokens = treasury_tokens - $1,
                            max_members = max_members + $2
                        WHERE id = $3
                        """,
                        cost,
                        slots_added,
                        clan_id,
                    )
                await conn.execute(
                    """
                    INSERT INTO clan_upgrades (clan_id, upgrade_type, level)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (clan_id, upgrade_type) DO UPDATE SET level = $3
                    """,
                    clan_id,
                    upgrade_type,
                    next_level,
                )

        title = upgrade_cfg.get("title") or upgrade_type
        await self._log_clan_activity(clan_id, "upgrade", f"{title} улучшено до ур. {next_level}", user_id=actor_id)
        updated = await self.get_clan(clan_id)
        return {
            "upgrade_type": upgrade_type,
            "level": next_level,
            "cost": cost,
            "config": level_cfg,
            "clan": updated,
        }

    async def buy_slot_upgrade(self, clan_id: int) -> dict:
        clan = await self.get_clan(clan_id)
        if not clan:
            raise ValueError("clan_not_found")
        return await self.buy_clan_upgrade(clan_id, int(clan["owner_id"]), "member_slots")

    async def buy_squad_personal_reward(self, user_id: int, reward_id: str) -> dict[str, Any]:
        config = await self.get_squad_runtime_config()
        rewards = config.get("squad_personal_rewards") or []
        reward = next((r for r in rewards if str(r.get("id")) == reward_id), None)
        if not reward:
            raise ValueError("reward_not_found")
        cost = max(0, int(reward.get("cost") or 0))
        cosmetic_slug = str(reward.get("cosmetic_slug") or "").strip()
        cosmetic_item = None
        if reward.get("kind") == "cosmetic" or cosmetic_slug:
            if not cosmetic_slug:
                raise ValueError("cosmetic_not_found")
            cosmetic_item = await self.fetchrow(
                """
                SELECT id, slug, item_type, class, name, asset_path, media_type
                FROM cosmetic_items
                WHERE slug = $1 AND is_active = TRUE
                """,
                cosmetic_slug,
            )
            if not cosmetic_item:
                raise ValueError("cosmetic_not_found")

        member = await self.fetchrow(
            """
            SELECT cm.clan_id, cm.personal_tokens
            FROM clan_members cm
            WHERE cm.user_id = $1
            LIMIT 1
            """,
            user_id,
        )
        if not member:
            raise ValueError("not_in_squad")

        metadata_json = json.dumps(reward, ensure_ascii=False)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                duplicate = await conn.fetchval(
                    "SELECT 1 FROM squad_shop_purchases WHERE user_id = $1 AND reward_id = $2",
                    user_id,
                    reward_id,
                )
                if duplicate:
                    raise ValueError("already_purchased")
                row = await conn.fetchrow(
                    """
                    UPDATE clan_members
                    SET personal_tokens = personal_tokens - $1
                    WHERE user_id = $2 AND clan_id = $3 AND personal_tokens >= $1
                    RETURNING personal_tokens
                    """,
                    cost,
                    user_id,
                    int(member["clan_id"]),
                )
                if not row:
                    raise ValueError("insufficient_personal_tokens")
                purchase = await conn.fetchrow(
                    """
                    INSERT INTO squad_shop_purchases (clan_id, user_id, reward_id, cost, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    RETURNING *
                    """,
                    int(member["clan_id"]),
                    user_id,
                    reward_id,
                    cost,
                    metadata_json,
                )

        granted_cosmetic = None
        if cosmetic_item:
            granted_cosmetic = await self.grant_cosmetic_by_slug(
                user_id,
                cosmetic_slug,
                source=f"squad_shop:{reward_id}",
                auto_equip=bool(reward.get("auto_equip", True)),
            )
        elif reward.get("kind") == "title" and reward.get("title"):
            await self.execute(
                """
                INSERT INTO profiles (user_id, title)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE SET title = EXCLUDED.title
                """,
                user_id,
                str(reward["title"])[:64],
            )

        await self._log_clan_activity(
            int(member["clan_id"]),
            "personal_reward",
            f"Куплена личная награда: {reward.get('name', reward_id)}",
            user_id=user_id,
        )
        return {"purchase": dict(purchase), "reward": reward, "cosmetic": granted_cosmetic}

    async def _repair_squad_shop_cosmetic_purchases(self) -> None:
        """Backfill cosmetics for squad-shop purchases made before rewards used cosmetic_slug."""
        if not self._pool:
            return
        try:
            config = await self.get_squad_runtime_config()
            rewards = {
                str(item.get("id")): item
                for item in (config.get("squad_personal_rewards") or [])
                if isinstance(item, dict)
            }
            rows = await self.fetch(
                """
                SELECT user_id, reward_id, created_at
                FROM squad_shop_purchases
                ORDER BY user_id, created_at ASC
                """
            )
            for row in rows:
                reward = rewards.get(str(row["reward_id"]))
                if not reward:
                    continue
                cosmetic_slug = str(reward.get("cosmetic_slug") or "").strip()
                if not cosmetic_slug:
                    continue
                already_owned = await self.fetchval(
                    """
                    SELECT 1
                    FROM user_cosmetics uc
                    JOIN cosmetic_items ci ON ci.id = uc.cosmetic_id
                    WHERE uc.user_id = $1 AND ci.slug = $2
                    """,
                    int(row["user_id"]),
                    cosmetic_slug,
                )
                await self.grant_cosmetic_by_slug(
                    int(row["user_id"]),
                    cosmetic_slug,
                    source=f"squad_shop:{row['reward_id']}",
                    auto_equip=(not already_owned) and bool(reward.get("auto_equip", True)),
                )
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to repair squad shop cosmetic purchases",
                exc_info=True,
            )

    async def record_submission(self, user_id: int, category: str) -> None:
        """Записать отправку на модерацию (rate limit)."""
        await self.execute(
            "INSERT INTO community_submissions (user_id, category) VALUES ($1, $2)",
            user_id, category,
        )

    async def count_recent_submissions(
        self,
        user_id: int,
        *,
        hours: int | None = None,
        minutes: int | None = None,
    ) -> int:
        """Количество отправок пользователя за последние N часов/минут."""
        interval_value = minutes if minutes is not None else (hours if hours is not None else 3)
        interval_unit = "minutes" if minutes is not None else "hours"
        count = await self.fetchval(
            """
            SELECT COUNT(*) FROM community_submissions
            WHERE user_id = $1 AND created_at > NOW() - ($2 || ' ' || $3)::INTERVAL
            """,
            user_id, str(interval_value), interval_unit,
        )
        return count or 0

    # News CRUD ──────────────────────────────────────────────────────────────

    async def get_news_posts(
        self,
        limit: int = 30,
        offset: int = 0,
        user_id: Optional[int] = None,
    ) -> list[dict]:
        """Получить список новостей и опросов."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT
                cp.id, cp.author_id, cp.title, cp.content, cp.content_html,
                cp.cover_image_url, cp.tags, cp.post_type, cp.is_pinned,
                cp.created_at,
                COALESCE(u.username, u.first_name, 'Администрация') AS author_name,
                (SELECT COUNT(*) FROM post_likes pl WHERE pl.post_id = cp.id) AS likes_count,
                CASE WHEN $3::BIGINT IS NOT NULL THEN
                    EXISTS(SELECT 1 FROM post_likes pl WHERE pl.post_id = cp.id AND pl.user_id = $3)
                ELSE FALSE END AS is_liked,
                p.id AS poll_id, p.question, p.options, p.expires_at
            FROM community_posts cp
            LEFT JOIN users u ON u.user_id = cp.author_id
            LEFT JOIN community_polls p ON p.post_id = cp.id
            WHERE cp.post_type IN ('news', 'poll')
              AND cp.moderation_status = 'approved'
              AND cp.status != 'hidden'
            ORDER BY cp.is_pinned DESC, cp.created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset, user_id,
        )
        result = []
        for r in rows:
            d = dict(r)
            if d.get("created_at"):
                d["created_at"] = d["created_at"].isoformat()
            if d.get("expires_at"):
                d["expires_at"] = d["expires_at"].isoformat()
            d["has_poll"] = bool(d.get("poll_id"))
            if d["has_poll"]:
                poll_results = await self.get_poll_results(poll_id=int(d["poll_id"]), user_id=user_id)
                if poll_results.get("success"):
                    d.update(poll_results)
                    d["post_type"] = "poll"
                else:
                    d["has_poll"] = False
            elif d.get("post_type") == "poll":
                d["post_type"] = "news"
            result.append(d)
        return result

    async def create_news_post(
        self,
        *,
        author_id: int,
        title: str,
        content: str,
        content_html: str = "",
        tags: list[str] | None = None,
        cover_image_url: Optional[str] = None,
        post_type: str = "news",
    ) -> dict:
        """Создать новостной пост (только для админа)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            row = await self.fetchrow(
                """
                INSERT INTO community_posts
                    (author_id, title, content, content_html, tags, cover_image_url,
                     post_type, moderation_status, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'approved', 'active')
                RETURNING id, created_at
                """,
                author_id, title, content, content_html,
                tags or [], cover_image_url, post_type,
            )
            return {"success": True, "id": row["id"], "created_at": row["created_at"].isoformat()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_poll(
        self,
        *,
        post_id: int,
        question: str,
        options: list[dict],
        expires_at: str,
    ) -> dict:
        """Создать опрос для поста."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            normalized_options = _normalize_poll_options(options)
            if not question.strip() or len(normalized_options) < 2:
                return {"success": False, "error": "invalid_poll"}
            row = await self.fetchrow(
                """
                INSERT INTO community_polls (post_id, question, options, expires_at)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                post_id, question.strip(), json.dumps(normalized_options), _parse_poll_expires_at(expires_at),
            )
            return {"success": True, "poll_id": row["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def vote_poll(self, *, poll_id: int, user_id: int, option_id: int) -> dict:
        """Проголосовать в опросе."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        poll = await self.fetchrow(
            "SELECT expires_at, options FROM community_polls WHERE id = $1", poll_id
        )
        if not poll:
            return {"success": False, "error": "poll_not_found"}
        option_ids = {opt["id"] for opt in _normalize_poll_options(poll["options"])}
        if option_id not in option_ids:
            return {"success": False, "error": "invalid_option"}
        from datetime import timezone as _tz
        import datetime as _dt
        if poll["expires_at"].replace(tzinfo=_tz.utc) < _dt.datetime.now(_tz.utc):
            return {"success": False, "error": "poll_expired"}
        existing = await self.fetchval(
            "SELECT 1 FROM community_poll_votes WHERE poll_id = $1 AND user_id = $2",
            poll_id, user_id,
        )
        if existing:
            return {"success": False, "error": "already_voted"}
        await self.execute(
            "INSERT INTO community_poll_votes (poll_id, user_id, option_id) VALUES ($1, $2, $3)",
            poll_id, user_id, option_id,
        )
        return {"success": True}

    async def get_poll_results(self, *, poll_id: int, user_id: Optional[int] = None) -> dict:
        """Получить результаты опроса."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        poll = await self.fetchrow(
            "SELECT id, post_id, question, options, expires_at FROM community_polls WHERE id = $1",
            poll_id,
        )
        if not poll:
            return {"success": False, "error": "poll_not_found"}

        options = _normalize_poll_options(poll["options"])

        vote_rows = await self.fetch(
            "SELECT option_id, COUNT(*) AS cnt FROM community_poll_votes WHERE poll_id = $1 GROUP BY option_id",
            poll_id,
        )
        vote_counts = {r["option_id"]: r["cnt"] for r in vote_rows}
        total = sum(vote_counts.values()) or 1

        user_vote = None
        if user_id:
            user_vote = await self.fetchval(
                "SELECT option_id FROM community_poll_votes WHERE poll_id = $1 AND user_id = $2",
                poll_id, user_id,
            )

        options_with_results = []
        for opt in options:
            oid = opt.get("id")
            cnt = vote_counts.get(oid, 0)
            options_with_results.append({
                "id": oid,
                "text": opt.get("text", ""),
                "votes": cnt,
                "percent": round(cnt / total * 100, 1),
            })

        from datetime import timezone as _tz
        import datetime as _dt
        expired = poll["expires_at"].replace(tzinfo=_tz.utc) < _dt.datetime.now(_tz.utc)
        return {
            "success": True,
            "poll_id": poll_id,
            "question": poll["question"],
            "options": options_with_results,
            "user_vote": user_vote,
            "expires_at": poll["expires_at"].isoformat(),
            "expired": expired,
            "total_votes": sum(vote_counts.values()),
        }

    # Ideas / Bugs CRUD ──────────────────────────────────────────────────────

    async def get_ideas(
        self,
        limit: int = 30,
        offset: int = 0,
        sort_by: str = "votes",
        user_id: Optional[int] = None,
    ) -> list[dict]:
        """Получить список идей (без багов)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        order = "cp.upvotes - cp.downvotes DESC, cp.created_at DESC" if sort_by == "votes" else "cp.created_at DESC"
        rows = await self.fetch(
            f"""
            SELECT
                cp.id, cp.title, cp.content AS description, cp.post_type,
                cp.upvotes, cp.downvotes, cp.status, cp.admin_approved,
                cp.created_at,
                COALESCE(pr.custom_nickname, u.username, u.first_name, 'Игрок') AS author_name,
                CASE WHEN $3::BIGINT IS NOT NULL THEN
                    (SELECT vote_type FROM community_votes cv WHERE cv.post_id = cp.id AND cv.user_id = $3)
                ELSE NULL END AS user_vote
            FROM community_posts cp
            LEFT JOIN users u ON u.user_id = cp.author_id
            LEFT JOIN profiles pr ON pr.user_id = cp.author_id
            WHERE cp.post_type = 'idea'
              AND cp.moderation_status = 'approved'
              AND cp.status NOT IN ('hidden', 'expired')
            ORDER BY {order}
            LIMIT $1 OFFSET $2
            """,
            limit, offset, user_id,
        )
        result = []
        for r in rows:
            d = dict(r)
            if d.get("created_at"):
                d["created_at"] = d["created_at"].isoformat()
            result.append(d)
        return result

    async def create_idea(
        self,
        *,
        author_id: int,
        title: str,
        description: str,
        post_type: str = "idea",
        moderation_status: str = "approved",
        moderation_reason: Optional[str] = None,
    ) -> dict:
        """Создать идею или баг."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            row = await self.fetchrow(
                """
                INSERT INTO community_posts
                    (author_id, title, content, post_type,
                     moderation_status, moderation_reason, status)
                VALUES ($1, $2, $3, $4, $5, $6, 'active')
                RETURNING id
                """,
                author_id, title, description, post_type,
                moderation_status, moderation_reason,
            )
            return {"success": True, "id": row["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def vote_idea(self, *, post_id: int, user_id: int, vote_type: str) -> dict:
        """Проголосовать за/против идеи. Повторное голосование убирает голос."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        existing = await self.fetchrow(
            "SELECT vote_type FROM community_votes WHERE post_id = $1 AND user_id = $2",
            post_id, user_id,
        )
        async with self._pool.acquire() as conn:
            if existing:
                if existing["vote_type"] == vote_type:
                    # убираем голос
                    await conn.execute(
                        "DELETE FROM community_votes WHERE post_id = $1 AND user_id = $2",
                        post_id, user_id,
                    )
                    if vote_type == "up":
                        await conn.execute(
                            "UPDATE community_posts SET upvotes = GREATEST(0, upvotes - 1) WHERE id = $1", post_id
                        )
                    else:
                        await conn.execute(
                            "UPDATE community_posts SET downvotes = GREATEST(0, downvotes - 1) WHERE id = $1", post_id
                        )
                    new_vote = None
                else:
                    # меняем голос
                    await conn.execute(
                        "UPDATE community_votes SET vote_type = $3, created_at = NOW() WHERE post_id = $1 AND user_id = $2",
                        post_id, user_id, vote_type,
                    )
                    if vote_type == "up":
                        await conn.execute(
                            "UPDATE community_posts SET upvotes = upvotes + 1, downvotes = GREATEST(0, downvotes - 1) WHERE id = $1",
                            post_id,
                        )
                    else:
                        await conn.execute(
                            "UPDATE community_posts SET downvotes = downvotes + 1, upvotes = GREATEST(0, upvotes - 1) WHERE id = $1",
                            post_id,
                        )
                    new_vote = vote_type
            else:
                await conn.execute(
                    "INSERT INTO community_votes (post_id, user_id, vote_type) VALUES ($1, $2, $3)",
                    post_id, user_id, vote_type,
                )
                if vote_type == "up":
                    await conn.execute("UPDATE community_posts SET upvotes = upvotes + 1 WHERE id = $1", post_id)
                else:
                    await conn.execute("UPDATE community_posts SET downvotes = downvotes + 1 WHERE id = $1", post_id)
                new_vote = vote_type

        row = await self.fetchrow(
            "SELECT upvotes, downvotes FROM community_posts WHERE id = $1", post_id
        )
        return {
            "success": True,
            "user_vote": new_vote if not existing or existing["vote_type"] != vote_type else None,
            "upvotes": row["upvotes"] if row else 0,
            "downvotes": row["downvotes"] if row else 0,
        }

    async def admin_approve_idea(self, post_id: int) -> dict:
        """Пометить идею одобренной администрацией."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        await self.execute(
            "UPDATE community_posts SET admin_approved = TRUE WHERE id = $1 AND post_type = 'idea'",
            post_id,
        )
        return {"success": True}

    async def update_idea_status(self, post_id: int, status: str) -> dict:
        """Изменить статус идеи/бага (admin)."""
        valid = {"reviewing", "accepted", "in_progress", "done", "rejected", "fixed", "hidden"}
        if status not in valid:
            return {"success": False, "error": "invalid_status"}
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        await self.execute(
            "UPDATE community_posts SET status = $2 WHERE id = $1 AND post_type IN ('idea', 'bug')",
            post_id, status,
        )
        return {"success": True}

    async def get_bugs_for_admin(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Получить баг-репорты (только для администрации)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT
                cp.id, cp.title, cp.content AS description,
                cp.status, cp.moderation_status, cp.moderation_reason,
                cp.created_at,
                COALESCE(u.username, u.first_name, 'Игрок') AS author_name,
                cp.author_id
            FROM community_posts cp
            LEFT JOIN users u ON u.user_id = cp.author_id
            WHERE cp.post_type = 'bug'
            ORDER BY cp.created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset,
        )
        result = []
        for r in rows:
            d = dict(r)
            if d.get("created_at"):
                d["created_at"] = d["created_at"].isoformat()
            result.append(d)
        return result

    # Announcements CRUD ─────────────────────────────────────────────────────

    async def get_announcements(
        self,
        limit: int = 20,
        offset: int = 0,
        user_id: Optional[int] = None,
    ) -> list[dict]:
        """Список активных объявлений."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT
                cp.id, cp.title, cp.content, cp.cover_image_url, cp.is_pinned,
                cp.pin_price, cp.gems_paid, cp.expires_at, cp.created_at,
                cp.clan_id,
                COALESCE(cl.name, 'Сквад') AS clan_name,
                cl.tag AS clan_tag,
                cl.public_id AS clan_public_id,
                cl.boost_public_id AS clan_boost_public_id,
                cl.type AS clan_type,
                COALESCE(cl.min_trophies, 0) AS clan_min_trophies,
                COALESCE(cl.cbrp, 0) AS clan_cbrp,
                cl.description AS clan_description,
                COALESCE(cl.trophies, 0) AS clan_trophies,
                COALESCE(cl.rank, 0) AS clan_rank,
                COALESCE(cl.members_count, 0) AS clan_members,
                COALESCE(cl.max_members, 50) AS clan_max_members,
                COALESCE(cl.has_boost, FALSE) AS clan_has_boost,
                cl.avatar_url AS clan_avatar_url,
                (SELECT COUNT(*) FROM community_votes cv WHERE cv.post_id = cp.id AND cv.vote_type = 'like') AS likes,
                (SELECT COUNT(*) FROM community_votes cv WHERE cv.post_id = cp.id AND cv.vote_type = 'dislike') AS dislikes,
                CASE WHEN $3::BIGINT IS NOT NULL THEN
                    (SELECT vote_type FROM community_votes cv WHERE cv.post_id = cp.id AND cv.user_id = $3)
                ELSE NULL END AS user_reaction
            FROM community_posts cp
            LEFT JOIN clans cl ON cl.id = cp.clan_id
            WHERE cp.post_type = 'announcement'
              AND cp.moderation_status = 'approved'
              AND cp.status NOT IN ('hidden', 'expired')
            ORDER BY cp.is_pinned DESC, cp.created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset, user_id,
        )
        result = []
        for r in rows:
            d = dict(r)
            if d.get("created_at"):
                d["created_at"] = d["created_at"].isoformat()
            if d.get("expires_at"):
                d["expires_at"] = d["expires_at"].isoformat()
            result.append(d)
        return result

    async def get_active_pin(self) -> Optional[dict]:
        """Получить текущее закреплённое объявление."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            """
            SELECT id, pin_price FROM community_posts
            WHERE post_type = 'announcement' AND is_pinned = TRUE AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        return dict(row) if row else None

    async def create_announcement(
        self,
        *,
        author_id: int,
        clan_id: int,
        content: str,
        image_url: Optional[str],
        duration_key: str,
        is_pinned: bool,
        gems_to_pay: int,
        pin_price: int,
    ) -> dict:
        """Создать объявление с транзакционным списанием гемов и обработкой пина."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        from infrastructure.community_config import ANNOUNCE_DURATION_COSTS, ANNOUNCE_PIN_OVERBID_STEP
        import datetime as _dt
        from datetime import timezone as _tz

        duration_days = {"1d": 1, "3d": 3, "7d": 7, "forever": None}.get(duration_key, 1)

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # check balance
                    row = await conn.fetchrow(
                        "SELECT gems FROM users WHERE user_id = $1 FOR UPDATE", author_id
                    )
                    if not row or (row["gems"] or 0) < gems_to_pay:
                        return {
                            "success": False,
                            "error": "insufficient_gems",
                            "required": gems_to_pay,
                            "current": row["gems"] if row else 0,
                        }

                    # handle pin overbid
                    if is_pinned:
                        old_pin = await conn.fetchrow(
                            """
                            SELECT id, pin_price FROM community_posts
                            WHERE post_type = 'announcement' AND is_pinned = TRUE AND status = 'active'
                            ORDER BY created_at DESC LIMIT 1
                            """
                        )
                        if old_pin:
                            min_required = old_pin["pin_price"] + ANNOUNCE_PIN_OVERBID_STEP
                            if pin_price < min_required:
                                return {
                                    "success": False,
                                    "error": "pin_price_too_low",
                                    "min_required": min_required,
                                    "current_pin_price": old_pin["pin_price"],
                                }
                            await conn.execute(
                                "UPDATE community_posts SET is_pinned = FALSE WHERE id = $1",
                                old_pin["id"],
                            )

                    # deduct gems
                    await conn.execute(
                        "UPDATE users SET gems = GREATEST(0, gems - $1), updated_at = NOW() WHERE user_id = $2",
                        gems_to_pay, author_id,
                    )

                    # expires_at
                    expires_at = None
                    if duration_days:
                        expires_at = _dt.datetime.now(_tz.utc) + _dt.timedelta(days=duration_days)

                    # insert
                    inserted = await conn.fetchrow(
                        """
                        INSERT INTO community_posts
                            (author_id, title, content, cover_image_url, post_type,
                             clan_id, moderation_status, status, is_pinned, pin_price,
                             gems_paid, expires_at)
                        VALUES ($1, '', $2, $3, 'announcement', $4, 'approved', 'active',
                                $5, $6, $7, $8)
                        RETURNING id, created_at
                        """,
                        author_id, content, image_url, clan_id,
                        is_pinned, pin_price if is_pinned else 0,
                        gems_to_pay, expires_at,
                    )

                    gems_after = await conn.fetchval(
                        "SELECT gems FROM users WHERE user_id = $1", author_id
                    )

            return {
                "success": True,
                "id": inserted["id"],
                "created_at": inserted["created_at"].isoformat(),
                "gems_remaining": gems_after,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def react_announcement(self, *, post_id: int, user_id: int, vote_type: str) -> dict:
        """Лайк/дизлайк объявления (toggle)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        if vote_type not in ("like", "dislike"):
            return {"success": False, "error": "invalid_vote_type"}

        existing = await self.fetchrow(
            "SELECT vote_type FROM community_votes WHERE post_id = $1 AND user_id = $2",
            post_id, user_id,
        )
        if existing:
            if existing["vote_type"] == vote_type:
                await self.execute(
                    "DELETE FROM community_votes WHERE post_id = $1 AND user_id = $2",
                    post_id, user_id,
                )
                new_reaction = None
            else:
                await self.execute(
                    "UPDATE community_votes SET vote_type = $3 WHERE post_id = $1 AND user_id = $2",
                    post_id, user_id, vote_type,
                )
                new_reaction = vote_type
        else:
            await self.execute(
                "INSERT INTO community_votes (post_id, user_id, vote_type) VALUES ($1, $2, $3)",
                post_id, user_id, vote_type,
            )
            new_reaction = vote_type

        row = await self.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM community_votes WHERE post_id = $1 AND vote_type = 'like') AS likes,
                (SELECT COUNT(*) FROM community_votes WHERE post_id = $1 AND vote_type = 'dislike') AS dislikes
            """,
            post_id,
        )
        return {
            "success": True,
            "user_reaction": new_reaction,
            "likes": row["likes"] if row else 0,
            "dislikes": row["dislikes"] if row else 0,
        }

    async def expire_announcements(self) -> int:
        """Пометить истёкшие объявления. Возвращает кол-во обновлённых."""
        if not self._pool:
            return 0
        result = await self.execute(
            """
            UPDATE community_posts
            SET status = 'expired'
            WHERE post_type = 'announcement'
              AND status = 'active'
              AND expires_at IS NOT NULL
              AND expires_at < NOW()
            """
        )
        # result is a string like "UPDATE 3"
        try:
            return int(result.split()[-1])
        except Exception:
            return 0

    async def _ensure_reward_tracks_table(self) -> bool:
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.reward_tracks')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE reward_tracks (
                    id SERIAL PRIMARY KEY,
                    track_type TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    reward_type TEXT NOT NULL,
                    reward_amount INTEGER NOT NULL DEFAULT 0,
                    reward_meta JSONB DEFAULT NULL,
                    extra_pass_required BOOLEAN NOT NULL DEFAULT FALSE,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(track_type, position, reward_type)
                )
                """
            )
            changed = True

        columns = await self._get_columns("reward_tracks")

        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "track_type TEXT NOT NULL DEFAULT 'glory'"
        )
        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "position INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "reward_type TEXT NOT NULL DEFAULT 'coins'"
        )
        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "reward_amount INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "reward_meta JSONB DEFAULT NULL"
        )
        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "extra_pass_required BOOLEAN NOT NULL DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "is_active BOOLEAN NOT NULL DEFAULT TRUE"
        )
        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "reward_tracks", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        if not await self._constraint_exists("reward_tracks", "reward_tracks_track_position_reward_key"):
            await self.execute(
                """
                ALTER TABLE reward_tracks
                ADD CONSTRAINT reward_tracks_track_position_reward_key
                UNIQUE (track_type, position, reward_type)
                """
            )
            changed = True

        return changed

    async def _ensure_claimed_rewards_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.claimed_rewards')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE claimed_rewards (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(user_id),
                    track_type TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(user_id, track_type, position)
                )
                """
            )
            changed = True

        columns = await self._get_columns("claimed_rewards")

        changed |= await self._add_column_if_missing(
            "claimed_rewards", columns, "user_id BIGINT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "claimed_rewards", columns, "track_type TEXT NOT NULL DEFAULT 'glory'"
        )
        changed |= await self._add_column_if_missing(
            "claimed_rewards", columns, "position INTEGER NOT NULL DEFAULT 0"
        )
        changed |= await self._add_column_if_missing(
            "claimed_rewards", columns, "claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        if not await self._constraint_exists("claimed_rewards", "claimed_rewards_user_track_position_key"):
            await self.execute(
                """
                ALTER TABLE claimed_rewards
                ADD CONSTRAINT claimed_rewards_user_track_position_key
                UNIQUE (user_id, track_type, position)
                """
            )
            changed = True

        return changed

    async def _ensure_seasons_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.seasons')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE seasons (
                    id SERIAL PRIMARY KEY,
                    slug TEXT NOT NULL DEFAULT 'arena-rift',
                    name TEXT NOT NULL DEFAULT '',
                    subtitle TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    season_number INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'draft',
                    auto_switch BOOLEAN NOT NULL DEFAULT TRUE,
                    preset_key TEXT,
                    start_date TIMESTAMPTZ,
                    end_date TIMESTAMPTZ,
                    is_active BOOLEAN NOT NULL DEFAULT FALSE,
                    max_stars INTEGER NOT NULL DEFAULT 45,
                    free_track_type TEXT NOT NULL DEFAULT 'bp_free',
                    pass_track_type TEXT NOT NULL DEFAULT 'bp_premium',
                    ultra_track_type TEXT NOT NULL DEFAULT 'bp_ultra',
                    pass_end_position INTEGER NOT NULL DEFAULT 40,
                    ultra_start_position INTEGER NOT NULL DEFAULT 41,
                    theme JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("seasons")

        changed |= await self._add_column_if_missing(
            "seasons", columns, "slug TEXT NOT NULL DEFAULT 'arena-rift'"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "name TEXT NOT NULL DEFAULT ''"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "subtitle TEXT NOT NULL DEFAULT ''"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "description TEXT NOT NULL DEFAULT ''"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "season_number INTEGER NOT NULL DEFAULT 1"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "status TEXT NOT NULL DEFAULT 'draft'"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "auto_switch BOOLEAN NOT NULL DEFAULT TRUE"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "preset_key TEXT"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "start_date TIMESTAMPTZ"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "end_date TIMESTAMPTZ"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "is_active BOOLEAN NOT NULL DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "max_stars INTEGER NOT NULL DEFAULT 45"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "free_track_type TEXT NOT NULL DEFAULT 'bp_free'"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "pass_track_type TEXT NOT NULL DEFAULT 'bp_premium'"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "ultra_track_type TEXT NOT NULL DEFAULT 'bp_ultra'"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "pass_end_position INTEGER NOT NULL DEFAULT 40"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "ultra_start_position INTEGER NOT NULL DEFAULT 41"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "theme JSONB NOT NULL DEFAULT '{}'::jsonb"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "seasons", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        await self.execute("UPDATE seasons SET status = 'active' WHERE is_active = TRUE")
        await self.execute(
            "UPDATE seasons SET season_number = id WHERE season_number <= 0 OR (season_number = 1 AND id <> 1)"
        )

        return changed

    async def _ensure_shop_sets_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.shop_sets')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE shop_sets (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    image_file_id TEXT,
                    price DECIMAL(10,2) NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'rubles' CHECK (currency IN ('rubles','gems','coins')),
                    rewards JSONB NOT NULL DEFAULT '[]',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_by BIGINT REFERENCES users(user_id),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("shop_sets")
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "name TEXT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "description TEXT"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "image_file_id TEXT"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "price DECIMAL(10,2) NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "currency TEXT NOT NULL DEFAULT 'rubles'"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "rewards JSONB NOT NULL DEFAULT '[]'"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "is_active BOOLEAN NOT NULL DEFAULT TRUE"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "created_by BIGINT"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "shop_sets", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        return changed

    async def _ensure_ruble_products_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.ruble_products')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE ruble_products (
                    id SERIAL PRIMARY KEY,
                    code TEXT NOT NULL UNIQUE,
                    item_type TEXT NOT NULL,
                    package_type TEXT,
                    shop_set_id INTEGER REFERENCES shop_sets(id),
                    name TEXT NOT NULL,
                    description TEXT,
                    price DECIMAL(10,2) NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'rubles' CHECK (currency IN ('rubles','gems','coins')),
                    image_url TEXT,
                    badge TEXT,
                    sort_order INTEGER NOT NULL DEFAULT 100,
                    show_in_game BOOLEAN NOT NULL DEFAULT TRUE,
                    show_in_shop BOOLEAN NOT NULL DEFAULT TRUE,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    is_system BOOLEAN NOT NULL DEFAULT FALSE,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("ruble_products")
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "code TEXT NOT NULL UNIQUE"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "item_type TEXT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "package_type TEXT"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "shop_set_id INTEGER REFERENCES shop_sets(id)"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "name TEXT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "description TEXT"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "price DECIMAL(10,2) NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "currency TEXT NOT NULL DEFAULT 'rubles'"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "image_url TEXT"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "badge TEXT"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "sort_order INTEGER NOT NULL DEFAULT 100"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "show_in_game BOOLEAN NOT NULL DEFAULT TRUE"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "show_in_shop BOOLEAN NOT NULL DEFAULT TRUE"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "is_active BOOLEAN NOT NULL DEFAULT TRUE"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "is_system BOOLEAN NOT NULL DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "metadata JSONB NOT NULL DEFAULT '{}'"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "ruble_products", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        await self._seed_ruble_products()
        return changed

    async def _seed_ruble_products(self):
        system_products = [
            {"code": "extrapass",        "item_type": "extrapass",        "package_type": None,      "name": "ExtraPass",                "price": 179,  "badge": None,            "sort_order": 10,  "show_in_game": True,  "show_in_shop": True,  "description": "Без рекламы, премиальная дорожка Battle Pass, +2 пресета колод, x1.2 монет в боях."},
            {"code": "extrapass_ultra",  "item_type": "extrapass_ultra",  "package_type": None,      "name": "ExtraPass Ultra",          "price": 349,  "badge": "popular",       "sort_order": 20,  "show_in_game": True,  "show_in_shop": True,  "description": "Всё из ExtraPass + золотой ник, реролл кейсов, расширенная статистика."},
            {"code": "starter_boost",    "item_type": "starter_boost",    "package_type": None,      "name": "Starter Boost",            "price": 499,  "badge": None,            "sort_order": 30,  "show_in_game": True,  "show_in_shop": True,  "description": "ExtraPass на 30 дней, гемы, монеты и кейсы."},
            {"code": "gems_starter_once","item_type": "gems_package",     "package_type": "starter_once", "name": "50 гемов (стартовый)",  "price": 49,   "badge": "one-time",      "sort_order": 40,  "show_in_game": False, "show_in_shop": True,  "description": "Стартовый пакет гемов, доступен 1 раз."},
            {"code": "gems_100",         "item_type": "gems_package",     "package_type": "gems_100",     "name": "100 гемов",           "price": 99,   "badge": None,            "sort_order": 50,  "show_in_game": False, "show_in_shop": True,  "description": None},
            {"code": "gems_250",         "item_type": "gems_package",     "package_type": "gems_250",     "name": "250 гемов",           "price": 229,  "badge": "discount",      "sort_order": 60,  "show_in_game": False, "show_in_shop": True,  "description": None},
            {"code": "gems_600",         "item_type": "gems_package",     "package_type": "gems_600",     "name": "600 гемов",           "price": 499,  "badge": "discount",      "sort_order": 70,  "show_in_game": False, "show_in_shop": True,  "description": None},
            {"code": "gems_1300",        "item_type": "gems_package",     "package_type": "gems_1300",    "name": "1300 гемов",          "price": 999,  "badge": "discount",      "sort_order": 80,  "show_in_game": False, "show_in_shop": True,  "description": None},
            {"code": "gems_2500",        "item_type": "gems_package",     "package_type": "gems_2500",    "name": "2500 гемов",          "price": 1499, "badge": "best-value",    "sort_order": 90,  "show_in_game": False, "show_in_shop": True,  "description": None},
        ]
        for p in system_products:
            row = await self.fetchrow("SELECT id FROM ruble_products WHERE code = $1", p["code"])
            if not row:
                await self.execute(
                    """
                    INSERT INTO ruble_products (code, item_type, package_type, name, description, price, currency, badge, sort_order, show_in_game, show_in_shop, is_active, is_system)
                    VALUES ($1, $2, $3, $4, $5, $6, 'rubles', $7, $8, $9, $10, TRUE, TRUE)
                    """,
                    p["code"], p["item_type"], p["package_type"], p["name"], p["description"],
                    p["price"], p["badge"], p["sort_order"], p["show_in_game"], p["show_in_shop"],
                )

    # ── CRUD: ruble_products ──

    async def get_ruble_products(
        self,
        active_only: bool = False,
        surface: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        conditions = []
        params = []
        pid = 1
        if active_only:
            conditions.append(f"is_active = TRUE")
        if surface == "shop":
            conditions.append(f"show_in_shop = TRUE")
        elif surface == "game":
            conditions.append(f"show_in_game = TRUE")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM ruble_products {where} ORDER BY sort_order, id"
        rows = await self.fetch(query)
        return [_json_safe(dict(r)) for r in rows]

    async def get_ruble_product(self, code: str) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            "SELECT * FROM ruble_products WHERE code = $1", code
        )
        return _json_safe(dict(row)) if row else None

    async def create_ruble_product(self, **kwargs) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json
        code = kwargs.get("code", "").strip()
        item_type = kwargs.get("item_type", "").strip()
        name = kwargs.get("name", "").strip()
        if not code or not item_type or not name:
            return {"success": False, "error": "code, item_type and name are required"}
        try:
            record = await self.fetchrow(
                """
                INSERT INTO ruble_products (code, item_type, package_type, shop_set_id, name, description, price, currency, image_url, badge, sort_order, show_in_game, show_in_shop, is_active, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15::jsonb)
                RETURNING id
                """,
                code,
                item_type,
                kwargs.get("package_type"),
                kwargs.get("shop_set_id"),
                name,
                kwargs.get("description"),
                float(kwargs.get("price", 0)),
                kwargs.get("currency", "rubles"),
                kwargs.get("image_url"),
                kwargs.get("badge"),
                int(kwargs.get("sort_order", 100)),
                bool(kwargs.get("show_in_game", True)),
                bool(kwargs.get("show_in_shop", True)),
                bool(kwargs.get("is_active", True)),
                _json.dumps(kwargs.get("metadata", {})),
            )
            return {"success": True, "product_id": record["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def update_ruble_product(self, code_or_id: str | int, **kwargs) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json
        valid_keys = {
            "code", "item_type", "package_type", "shop_set_id",
            "name", "description", "price", "currency", "image_url", "badge",
            "sort_order", "show_in_game", "show_in_shop", "is_active", "is_system",
            "metadata",
        }
        updates = []
        values = []
        for key, value in kwargs.items():
            if key in valid_keys:
                if key == "metadata" and not isinstance(value, str):
                    value = _json.dumps(value)
                    updates.append(f"{key} = ${len(values) + 1}::jsonb")
                else:
                    updates.append(f"{key} = ${len(values) + 1}")
                values.append(value)
        if not updates:
            return {"success": False, "error": "no_valid_fields"}
        updates.append("updated_at = NOW()")
        if isinstance(code_or_id, int) or code_or_id.isdigit():
            values.append(int(code_or_id))
            where = f"id = ${len(values)}"
        else:
            values.append(str(code_or_id))
            where = f"code = ${len(values)}"
        query = f"UPDATE ruble_products SET {', '.join(updates)} WHERE {where}"
        try:
            await self.execute(query, *values)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_ruble_product(self, code_or_id: str | int) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            if isinstance(code_or_id, int) or str(code_or_id).isdigit():
                await self.execute(
                    "UPDATE ruble_products SET is_active = FALSE, updated_at = NOW() WHERE id = $1",
                    int(code_or_id),
                )
            else:
                await self.execute(
                    "UPDATE ruble_products SET is_active = FALSE, updated_at = NOW() WHERE code = $1",
                    str(code_or_id),
                )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_shop_sets(self, active_only: bool = True) -> list[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        query = "SELECT * FROM shop_sets"
        if active_only:
            query += " WHERE is_active = TRUE"
        query += " ORDER BY created_at DESC"
        rows = await self.fetch(query)
        return [_json_safe(dict(r)) for r in rows]

    async def get_shop_set(self, set_id: int) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            "SELECT * FROM shop_sets WHERE id = $1", set_id
        )
        return _json_safe(dict(row)) if row else None

    async def create_shop_set(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        image_file_id: Optional[str] = None,
        price: float,
        currency: str = "rubles",
        created_by: int,
        rewards: Optional[list] = None,
    ) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json
        rewards_json = _json.dumps(rewards or [])
        try:
            record = await self.fetchrow(
                """
                INSERT INTO shop_sets (name, description, image_file_id, price, currency, created_by, rewards)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                RETURNING id
                """,
                name, description, image_file_id, price, currency, created_by, rewards_json,
            )
            return {"success": True, "set_id": record["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def update_shop_set(self, set_id: int, **kwargs) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json
        valid_keys = {
            "name", "description", "image_file_id",
            "price", "currency", "rewards", "is_active",
        }
        updates = []
        values = []
        for key, value in kwargs.items():
            if key in valid_keys:
                if key == "rewards":
                    if not isinstance(value, str):
                        value = _json.dumps(value)
                    updates.append(f"{key} = ${len(values) + 1}::jsonb")
                else:
                    updates.append(f"{key} = ${len(values) + 1}")
                values.append(value)
        if not updates:
            return {"success": False, "error": "no_valid_fields"}
        updates.append(f"updated_at = NOW()")
        values.append(set_id)
        query = f"UPDATE shop_sets SET {', '.join(updates)} WHERE id = ${len(values)}"
        try:
            await self.execute(query, *values)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_shop_set(self, set_id: int) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            await self.execute(
                "UPDATE shop_sets SET is_active = FALSE, updated_at = NOW() WHERE id = $1",
                set_id,
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def grant_shop_set_rewards(
        self, user_id: int, set_id: int
    ) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT id, rewards FROM shop_sets WHERE id = $1 AND is_active = TRUE",
                    set_id,
                )
                if not row:
                    return {"success": False, "error": "set_not_found"}
                rewards, error = self._normalize_shop_set_rewards(row["rewards"])
                if error:
                    return {"success": False, "error": error}
                granted = await self._apply_shop_set_rewards_on_conn(conn, user_id, set_id, rewards)
                return {"success": True, "granted": granted}

    async def purchase_shop_set(self, user_id: int, set_id: int) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT id, price, currency, rewards FROM shop_sets WHERE id = $1 AND is_active = TRUE",
                    set_id,
                )
                if not row:
                    return {"success": False, "error": "set_not_found"}
                rewards, error = self._normalize_shop_set_rewards(row["rewards"])
                if error:
                    return {"success": False, "error": error}

                currency = str(row["currency"] or "rubles")
                price = float(row["price"] or 0)
                if currency not in {"gems", "coins"}:
                    return {"success": False, "error": "invalid_currency"}
                if price <= 0:
                    return {"success": False, "error": "invalid_price"}

                balance_row = await conn.fetchrow(
                    f"""
                    UPDATE users
                    SET {currency} = GREATEST(0, COALESCE({currency}, 0) - $1),
                        updated_at = NOW()
                    WHERE user_id = $2
                      AND COALESCE({currency}, 0) >= $1
                    RETURNING gems, coins
                    """,
                    price, user_id,
                )
                if not balance_row:
                    return {
                        "success": False,
                        "error": f"insufficient_{currency}",
                        "price": price,
                        "currency": currency,
                    }

                await conn.execute(
                    """
                    INSERT INTO economy_events (user_id, event_type, resource, amount, source, metadata)
                    VALUES ($1, 'spend', $2, $3, 'shop', $4::jsonb)
                    """,
                    user_id,
                    currency,
                    float(price),
                    json.dumps({"item_type": f"shop_set_{set_id}", "set_id": set_id}, ensure_ascii=False),
                )
                granted = await self._apply_shop_set_rewards_on_conn(conn, user_id, set_id, rewards)
                return {
                    "success": True,
                    "granted": granted,
                    "currency": currency,
                    "price": price,
                    "gems": balance_row["gems"],
                    "coins": balance_row["coins"],
                }

    def _normalize_shop_set_rewards(self, rewards_data: Any) -> tuple[list[dict[str, Any]], Optional[str]]:
        import json as _json

        if isinstance(rewards_data, str):
            rewards_data = _json.loads(rewards_data)
        if not isinstance(rewards_data, list):
            return [], "invalid_rewards_format"
        if not rewards_data:
            return [], "empty_rewards"

        normalized: list[dict[str, Any]] = []
        for reward in rewards_data:
            if not isinstance(reward, dict):
                return [], "invalid_reward"
            r_type = str(reward.get("type") or "")
            amount = int(reward.get("amount", 0) or 0)
            card_id = reward.get("card_id")
            if r_type in {"gems", "coins", "keys", "case"}:
                if amount <= 0 and r_type != "case":
                    return [], "invalid_reward_amount"
                normalized.append({"type": r_type, "amount": max(1, amount) if r_type == "case" else amount})
            elif r_type == "card":
                if not card_id:
                    return [], "invalid_reward_card"
                normalized.append({"type": r_type, "card_id": int(card_id)})
            elif r_type == "particles":
                if amount <= 0 or not card_id:
                    return [], "invalid_reward_particles"
                normalized.append({"type": r_type, "amount": amount, "card_id": int(card_id)})
            else:
                return [], "unknown_reward_type"
        return normalized, None

    async def _apply_shop_set_rewards_on_conn(
        self,
        conn,
        user_id: int,
        set_id: int,
        rewards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        from datetime import datetime, timezone

        granted: list[dict[str, Any]] = []
        for reward in rewards:
            r_type = reward["type"]
            amount = int(reward.get("amount") or 0)
            card_id = reward.get("card_id")
            if r_type == "gems":
                await conn.execute("UPDATE users SET gems = COALESCE(gems, 0) + $1 WHERE user_id = $2", amount, user_id)
                granted.append({"type": "gems", "amount": amount})
            elif r_type == "coins":
                await conn.execute("UPDATE users SET coins = COALESCE(coins, 0) + $1 WHERE user_id = $2", amount, user_id)
                granted.append({"type": "coins", "amount": amount})
            elif r_type in {"keys", "case"}:
                await conn.execute("UPDATE users SET keys = COALESCE(keys, 0) + $1 WHERE user_id = $2", amount, user_id)
                granted.append({"type": "keys", "amount": amount})
            elif r_type == "card":
                await conn.execute(
                    """
                    INSERT INTO user_cards (user_id, card_id, level, particles, obtained_at)
                    VALUES ($1, $2, 1, 0, $3)
                    ON CONFLICT (user_id, card_id) DO UPDATE
                    SET level = user_cards.level + 1,
                        obtained_at = $3
                    """,
                    user_id, int(card_id), datetime.now(timezone.utc),
                )
                granted.append({"type": "card", "card_id": int(card_id)})
            elif r_type == "particles":
                await conn.execute(
                    """
                    INSERT INTO user_cards (user_id, card_id, level, particles, obtained_at)
                    VALUES ($1, $2, 1, $3, NOW())
                    ON CONFLICT (user_id, card_id) DO UPDATE
                    SET particles = COALESCE(user_cards.particles, 0) + $3
                    """,
                    user_id, int(card_id), amount,
                )
                granted.append({"type": "particles", "amount": amount, "card_id": int(card_id)})

            if r_type in {"gems", "coins", "keys", "case"}:
                resource = "keys" if r_type == "case" else r_type
                await conn.execute(
                    """
                    INSERT INTO economy_events (user_id, event_type, resource, amount, source, metadata)
                    VALUES ($1, 'earn', $2, $3, 'shop_set', $4::jsonb)
                    """,
                    user_id,
                    resource,
                    float(amount),
                    json.dumps({"set_id": set_id}, ensure_ascii=False),
                )
        return granted

    async def _ensure_payments_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.payments')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE payments (
                    id SERIAL PRIMARY KEY,
                    payment_id TEXT NOT NULL UNIQUE,
                    user_id BIGINT NOT NULL,
                    amount DECIMAL(12,2) NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'RUB',
                    description TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    rewards_processed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("payments")
        changed |= await self._add_column_if_missing(
            "payments", columns, "payment_id TEXT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "user_id BIGINT NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "amount DECIMAL(12,2) NOT NULL"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "currency TEXT NOT NULL DEFAULT 'RUB'"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "description TEXT"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "metadata JSONB NOT NULL DEFAULT '{}'"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "status TEXT NOT NULL DEFAULT 'pending'"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "rewards_processed BOOLEAN NOT NULL DEFAULT FALSE"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "payments", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

        if not await self._constraint_exists("payments", "payments_payment_id_key"):
            try:
                await self.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS payments_payment_id_idx ON payments(payment_id)"
                )
            except Exception:
                pass

        return changed

    async def _ensure_payment_checkout_sessions_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.payment_checkout_sessions')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE payment_checkout_sessions (
                    id SERIAL PRIMARY KEY,
                    checkout_jti TEXT NOT NULL UNIQUE,
                    user_id BIGINT NOT NULL,
                    item_type TEXT NOT NULL,
                    amount DECIMAL(12,2) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    payment_id TEXT,
                    confirmation_url TEXT,
                    status TEXT NOT NULL DEFAULT 'created',
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("payment_checkout_sessions")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "checkout_jti TEXT NOT NULL")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "user_id BIGINT NOT NULL")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "item_type TEXT NOT NULL")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "amount DECIMAL(12,2) NOT NULL")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "metadata JSONB NOT NULL DEFAULT '{}'")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "payment_id TEXT")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "confirmation_url TEXT")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "status TEXT NOT NULL DEFAULT 'created'")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "expires_at TIMESTAMPTZ NOT NULL")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("payment_checkout_sessions", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        if not await self._constraint_exists("payment_checkout_sessions", "payment_checkout_sessions_checkout_jti_key"):
            try:
                await self.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_checkout_sessions_jti ON payment_checkout_sessions(checkout_jti)"
                )
            except Exception:
                pass

        return changed

    async def create_checkout_session(
        self,
        *,
        checkout_jti: str,
        user_id: int,
        item_type: str,
        amount: float,
        metadata: dict[str, Any],
        expires_at,
    ) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json
        try:
            await self.execute(
                """
                INSERT INTO payment_checkout_sessions (checkout_jti, user_id, item_type, amount, metadata, expires_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                ON CONFLICT (checkout_jti) DO NOTHING
                """,
                checkout_jti, user_id, item_type, amount, _json.dumps(metadata or {}), expires_at,
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_checkout_session(self, checkout_jti: str) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            "SELECT * FROM payment_checkout_sessions WHERE checkout_jti = $1",
            checkout_jti,
        )
        return dict(row) if row else None

    async def attach_checkout_payment(
        self,
        checkout_jti: str,
        payment_id: str,
        confirmation_url: str,
    ) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            result = await self.fetchrow(
                """
                UPDATE payment_checkout_sessions
                SET payment_id = $2,
                    confirmation_url = $3,
                    status = 'payment_created',
                    updated_at = NOW()
                WHERE checkout_jti = $1
                  AND payment_id IS NULL
                RETURNING payment_id, confirmation_url
                """,
                checkout_jti, payment_id, confirmation_url,
            )
            if result:
                return {"success": True, "payment_id": result["payment_id"], "confirmation_url": result["confirmation_url"]}
            existing = await self.get_checkout_session(checkout_jti)
            if existing and existing.get("payment_id"):
                return {"success": True, "payment_id": existing["payment_id"], "confirmation_url": existing.get("confirmation_url")}
            return {"success": False, "error": "session_not_found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _ensure_friend_invites_table(self) -> bool:
        """Создать таблицу friend_invites для дружеских матчей."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.friend_invites')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE friend_invites (
                    id SERIAL PRIMARY KEY,
                    from_user_id BIGINT NOT NULL REFERENCES users(user_id),
                    to_user_id BIGINT NOT NULL REFERENCES users(user_id),
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '5 minutes',
                    battle_id TEXT DEFAULT NULL,
                    from_selected_deck_id INTEGER,
                    to_selected_deck_id INTEGER
                )
                """
            )
            changed = True

        columns = await self._get_columns("friend_invites")
        changed |= await self._add_column_if_missing("friend_invites", columns, "from_user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("friend_invites", columns, "to_user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("friend_invites", columns, "status TEXT NOT NULL DEFAULT 'pending'")
        changed |= await self._add_column_if_missing("friend_invites", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("friend_invites", columns, "expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '5 minutes'")
        changed |= await self._add_column_if_missing("friend_invites", columns, "battle_id TEXT DEFAULT NULL")
        changed |= await self._add_column_if_missing("friend_invites", columns, "from_selected_deck_id INTEGER")
        changed |= await self._add_column_if_missing("friend_invites", columns, "to_selected_deck_id INTEGER")

        index_exists = await self.fetchval(
            "SELECT 1 FROM pg_indexes WHERE tablename = 'friend_invites' AND indexname = 'friend_invites_to_user_idx'"
        )
        if not index_exists:
            await self.execute("CREATE INDEX friend_invites_to_user_idx ON friend_invites(to_user_id, status)")
            changed = True

        return changed

    async def _ensure_generator_state_table(self) -> bool:
        """Создать таблицу состояния генератора ключей."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.generator_state')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE generator_state (
                    user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                    level INTEGER NOT NULL DEFAULT 1,
                    accumulated_keys INTEGER NOT NULL DEFAULT 0,
                    last_tick_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    notified BOOLEAN NOT NULL DEFAULT FALSE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("generator_state")
        changed |= await self._add_column_if_missing("generator_state", columns, "user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("generator_state", columns, "level INTEGER NOT NULL DEFAULT 1")
        changed |= await self._add_column_if_missing("generator_state", columns, "accumulated_keys INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("generator_state", columns, "last_tick_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("generator_state", columns, "notified BOOLEAN NOT NULL DEFAULT FALSE")
        changed |= await self._add_column_if_missing("generator_state", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        return changed

    async def _ensure_economy_events_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.economy_events')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE economy_events (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    event_type TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    amount NUMERIC(12,2) NOT NULL DEFAULT 0,
                    source TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("economy_events")
        changed |= await self._add_column_if_missing("economy_events", columns, "user_id BIGINT NOT NULL")
        changed |= await self._add_column_if_missing("economy_events", columns, "event_type TEXT NOT NULL")
        changed |= await self._add_column_if_missing("economy_events", columns, "resource TEXT NOT NULL")
        changed |= await self._add_column_if_missing("economy_events", columns, "amount NUMERIC(12,2) NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("economy_events", columns, "source TEXT")
        changed |= await self._add_column_if_missing("economy_events", columns, "metadata JSONB NOT NULL DEFAULT '{}'")
        changed |= await self._add_column_if_missing("economy_events", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        if not await self._constraint_exists("economy_events", "economy_events_user_id_fkey"):
            try:
                await self.execute(
                    "ALTER TABLE economy_events ADD CONSTRAINT economy_events_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE"
                )
            except Exception:
                pass

        return changed

    async def _ensure_user_sessions_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.user_sessions')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE user_sessions (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    session_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL DEFAULT 'webapp',
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ended_at TIMESTAMPTZ,
                    duration_seconds INTEGER,
                    screens_visited JSONB NOT NULL DEFAULT '[]',
                    battles_played INTEGER NOT NULL DEFAULT 0,
                    cases_opened INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            changed = True

        columns = await self._get_columns("user_sessions")
        changed |= await self._add_column_if_missing("user_sessions", columns, "user_id BIGINT NOT NULL")
        changed |= await self._add_column_if_missing("user_sessions", columns, "session_id TEXT NOT NULL")
        changed |= await self._add_column_if_missing("user_sessions", columns, "source TEXT NOT NULL DEFAULT 'webapp'")
        changed |= await self._add_column_if_missing("user_sessions", columns, "started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("user_sessions", columns, "ended_at TIMESTAMPTZ")
        changed |= await self._add_column_if_missing("user_sessions", columns, "duration_seconds INTEGER")
        changed |= await self._add_column_if_missing("user_sessions", columns, "screens_visited JSONB NOT NULL DEFAULT '[]'")
        changed |= await self._add_column_if_missing("user_sessions", columns, "battles_played INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("user_sessions", columns, "cases_opened INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("user_sessions", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        return changed

    async def _ensure_onboarding_events_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.onboarding_events')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE onboarding_events (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    step TEXT NOT NULL,
                    completed BOOLEAN NOT NULL DEFAULT FALSE,
                    time_spent_seconds NUMERIC(8,1),
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("onboarding_events")
        changed |= await self._add_column_if_missing("onboarding_events", columns, "user_id BIGINT NOT NULL")
        changed |= await self._add_column_if_missing("onboarding_events", columns, "step TEXT NOT NULL")
        changed |= await self._add_column_if_missing("onboarding_events", columns, "completed BOOLEAN NOT NULL DEFAULT FALSE")
        changed |= await self._add_column_if_missing("onboarding_events", columns, "time_spent_seconds NUMERIC(8,1)")
        changed |= await self._add_column_if_missing("onboarding_events", columns, "metadata JSONB NOT NULL DEFAULT '{}'")
        changed |= await self._add_column_if_missing("onboarding_events", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        return changed

    async def _ensure_battle_summary_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.battle_summary')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE battle_summary (
                    id BIGSERIAL PRIMARY KEY,
                    match_id TEXT NOT NULL UNIQUE,
                    p1_user_id BIGINT NOT NULL,
                    p2_user_id BIGINT NOT NULL,
                    winner_user_id BIGINT,
                    loser_user_id BIGINT,
                    p1_hero_id INTEGER,
                    p2_hero_id INTEGER,
                    p1_deck JSONB NOT NULL DEFAULT '[]',
                    p2_deck JSONB NOT NULL DEFAULT '[]',
                    surrender BOOLEAN NOT NULL DEFAULT FALSE,
                    afk BOOLEAN NOT NULL DEFAULT FALSE,
                    match_type TEXT NOT NULL DEFAULT 'pvp',
                    game_mode TEXT,
                    duration_seconds INTEGER NOT NULL DEFAULT 0,
                    turns_count INTEGER NOT NULL DEFAULT 0,
                    p1_trophy_change INTEGER NOT NULL DEFAULT 0,
                    p2_trophy_change INTEGER NOT NULL DEFAULT 0,
                    p1_coins_earned INTEGER NOT NULL DEFAULT 0,
                    p2_coins_earned INTEGER NOT NULL DEFAULT 0,
                    p1_cards_played INTEGER NOT NULL DEFAULT 0,
                    p2_cards_played INTEGER NOT NULL DEFAULT 0,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("battle_summary")
        changed |= await self._add_column_if_missing("battle_summary", columns, "match_id TEXT NOT NULL")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p1_user_id BIGINT NOT NULL")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p2_user_id BIGINT NOT NULL")
        changed |= await self._add_column_if_missing("battle_summary", columns, "winner_user_id BIGINT")
        changed |= await self._add_column_if_missing("battle_summary", columns, "loser_user_id BIGINT")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p1_hero_id INTEGER")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p2_hero_id INTEGER")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p1_deck JSONB NOT NULL DEFAULT '[]'")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p2_deck JSONB NOT NULL DEFAULT '[]'")
        changed |= await self._add_column_if_missing("battle_summary", columns, "surrender BOOLEAN NOT NULL DEFAULT FALSE")
        changed |= await self._add_column_if_missing("battle_summary", columns, "afk BOOLEAN NOT NULL DEFAULT FALSE")
        changed |= await self._add_column_if_missing("battle_summary", columns, "match_type TEXT NOT NULL DEFAULT 'pvp'")
        changed |= await self._add_column_if_missing("battle_summary", columns, "game_mode TEXT")
        changed |= await self._add_column_if_missing("battle_summary", columns, "duration_seconds INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_summary", columns, "turns_count INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p1_trophy_change INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p2_trophy_change INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p1_coins_earned INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p2_coins_earned INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p1_cards_played INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_summary", columns, "p2_cards_played INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_summary", columns, "metadata JSONB NOT NULL DEFAULT '{}'")
        changed |= await self._add_column_if_missing("battle_summary", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_summary_created_at ON battle_summary(created_at)")
        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_summary_p1_user_id ON battle_summary(p1_user_id)")
        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_summary_p2_user_id ON battle_summary(p2_user_id)")
        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_summary_winner_user_id ON battle_summary(winner_user_id)")
        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_summary_game_mode ON battle_summary(game_mode)")

        return changed

    async def _ensure_battle_actions_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.battle_actions')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE battle_actions (
                    id BIGSERIAL PRIMARY KEY,
                    battle_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL DEFAULT 0,
                    acting_player INTEGER,
                    acting_user_id BIGINT,
                    is_bot BOOLEAN NOT NULL DEFAULT FALSE,
                    state_json JSONB NOT NULL DEFAULT '{}',
                    action_json JSONB NOT NULL DEFAULT '{}',
                    quality_score DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("battle_actions")
        changed |= await self._add_column_if_missing("battle_actions", columns, "battle_id TEXT NOT NULL")
        changed |= await self._add_column_if_missing("battle_actions", columns, "turn_number INTEGER NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("battle_actions", columns, "acting_player INTEGER")
        changed |= await self._add_column_if_missing("battle_actions", columns, "acting_user_id BIGINT")
        changed |= await self._add_column_if_missing("battle_actions", columns, "is_bot BOOLEAN NOT NULL DEFAULT FALSE")
        changed |= await self._add_column_if_missing("battle_actions", columns, "state_json JSONB NOT NULL DEFAULT '{}'")
        changed |= await self._add_column_if_missing("battle_actions", columns, "action_json JSONB NOT NULL DEFAULT '{}'")
        changed |= await self._add_column_if_missing("battle_actions", columns, "quality_score DOUBLE PRECISION")
        changed |= await self._add_column_if_missing("battle_actions", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_actions_battle_id ON battle_actions(battle_id)")
        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_actions_acting_user_id ON battle_actions(acting_user_id)")
        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_actions_created_at ON battle_actions(created_at)")
        await self.execute("CREATE INDEX IF NOT EXISTS idx_battle_actions_action_type ON battle_actions((action_json->>'type'))")

        return changed

    async def create_friend_invite(
        self,
        from_user_id: int,
        to_user_id: int,
        from_selected_deck_id: Optional[int] = None,
    ) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            row = await self.fetchrow(
                """
                INSERT INTO friend_invites (from_user_id, to_user_id, from_selected_deck_id)
                VALUES ($1, $2, $3)
                RETURNING id, created_at, expires_at
                """,
                from_user_id, to_user_id, from_selected_deck_id,
            )
            return {"success": True, "id": row["id"], "created_at": row["created_at"].isoformat(), "expires_at": row["expires_at"].isoformat()}
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("create_friend_invite error: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_pending_invite(self, to_user_id: int) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            """
            SELECT i.*, p.custom_nickname, p.img AS avatar_url,
                   COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') AS display_name
            FROM friend_invites i
            JOIN users u ON u.user_id = i.from_user_id
            LEFT JOIN profiles p ON p.user_id = i.from_user_id
            WHERE i.to_user_id = $1
              AND i.status = 'pending'
              AND i.expires_at > NOW()
            ORDER BY i.created_at DESC
            LIMIT 1
            """,
            to_user_id,
        )
        if not row:
            return None
        return {
            "id": row["id"],
            "from_user_id": row["from_user_id"],
            "from_username": row["display_name"],
            "from_avatar_url": row["avatar_url"],
            "status": row["status"],
            "created_at": row["created_at"].isoformat(),
            "expires_at": row["expires_at"].isoformat(),
            "battle_id": row["battle_id"],
        }

    async def get_friend_invite_by_id(self, invite_id: int) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            "SELECT * FROM friend_invites WHERE id = $1", invite_id
        )
        return dict(row) if row else None

    async def has_active_pending_invite(self, from_user_id: int, to_user_id: int) -> bool:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        val = await self.fetchval(
            """
            SELECT 1 FROM friend_invites
            WHERE from_user_id = $1 AND to_user_id = $2
              AND status = 'pending' AND expires_at > NOW()
            LIMIT 1
            """,
            from_user_id, to_user_id,
        )
        return bool(val)

    async def update_invite_status(
        self,
        invite_id: int,
        status: str,
        battle_id: Optional[str] = None,
        to_selected_deck_id: Optional[int] = None,
    ) -> None:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        if battle_id:
            await self.execute(
                """
                UPDATE friend_invites
                SET status = $1, battle_id = $2, to_selected_deck_id = COALESCE($3, to_selected_deck_id)
                WHERE id = $4
                """,
                status, battle_id, to_selected_deck_id, invite_id,
            )
        else:
            await self.execute(
                """
                UPDATE friend_invites
                SET status = $1, to_selected_deck_id = COALESCE($2, to_selected_deck_id)
                WHERE id = $3
                """,
                status, to_selected_deck_id, invite_id,
            )

    async def get_friend_invite_for_user(self, invite_id: int, user_id: int) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            """
            SELECT * FROM friend_invites
            WHERE id = $1 AND (from_user_id = $2 OR to_user_id = $2)
            """,
            invite_id, user_id,
        )
        return dict(row) if row else None

    async def expire_old_invites(self) -> int:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        result = await self.execute(
            """
            UPDATE friend_invites SET status = 'expired'
            WHERE expires_at < NOW() AND status = 'pending'
            """
        )
        return result

    # ========== Таблица friend_requests (постоянные заявки в друзья + список друзей) ==========

    async def _ensure_friend_requests_table(self) -> bool:
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.friend_requests')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE friend_requests (
                    id SERIAL PRIMARY KEY,
                    requester_id BIGINT NOT NULL REFERENCES users(user_id),
                    addressee_id BIGINT NOT NULL REFERENCES users(user_id),
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT fr_no_self CHECK (requester_id <> addressee_id),
                    CONSTRAINT fr_unique_pair UNIQUE (requester_id, addressee_id)
                )
                """
            )
            changed = True

        columns = await self._get_columns("friend_requests")
        for col, coldef in [
            ("requester_id", "BIGINT NOT NULL DEFAULT 0"),
            ("addressee_id", "BIGINT NOT NULL DEFAULT 0"),
            ("status", "TEXT NOT NULL DEFAULT 'pending'"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT NOW()"),
            ("updated_at", "TIMESTAMPTZ NOT NULL DEFAULT NOW()"),
        ]:
            changed |= await self._add_column_if_missing("friend_requests", columns, f"{col} {coldef}")

        # Индексы
        if not await self.fetchval(
            "SELECT 1 FROM pg_indexes WHERE tablename = 'friend_requests' AND indexname = 'idx_fr_addressee_status'"
        ):
            await self.execute("CREATE INDEX idx_fr_addressee_status ON friend_requests(addressee_id, status)")
            changed = True

        if not await self.fetchval(
            "SELECT 1 FROM pg_indexes WHERE tablename = 'friend_requests' AND indexname = 'idx_fr_requester_status'"
        ):
            await self.execute("CREATE INDEX idx_fr_requester_status ON friend_requests(requester_id, status)")
            changed = True

        return changed

    async def has_pending_friend_request_pair(self, user_a: int, user_b: int) -> bool:
        """Есть ли уже pending-запрос между двумя пользователями в любом направлении."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        val = await self.fetchval(
            """
            SELECT 1 FROM friend_requests
            WHERE status = 'pending'
              AND ((requester_id = $1 AND addressee_id = $2)
                   OR (requester_id = $2 AND addressee_id = $1))
            LIMIT 1
            """,
            user_a, user_b,
        )
        return bool(val)

    async def create_friend_request(self, requester_id: int, addressee_id: int) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            row = await self.fetchrow(
                """
                INSERT INTO friend_requests (requester_id, addressee_id)
                VALUES ($1, $2)
                RETURNING id, created_at
                """,
                requester_id, addressee_id,
            )
            return {"success": True, "id": row["id"], "created_at": row["created_at"].isoformat()}
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("create_friend_request error: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_friend_request_by_id(self, request_id: int) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow("SELECT * FROM friend_requests WHERE id = $1", request_id)
        return dict(row) if row else None

    async def update_friend_request_status(self, request_id: int, status: str) -> bool:
        """Обновить статус заявки. Возвращает True если обновлено."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        result = await self.execute(
            "UPDATE friend_requests SET status = $1, updated_at = NOW() WHERE id = $2",
            status, request_id,
        )
        return bool(result)

    async def get_incoming_friend_requests(self, user_id: int) -> list[dict[str, Any]]:
        """Входящие заявки в друзья (адресат = user_id)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT fr.id, fr.requester_id, fr.status, fr.created_at,
                   COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') AS display_name,
                   COALESCE(equipped_avatar.asset_path, NULLIF(p.img, '')) AS avatar_url
            FROM friend_requests fr
            JOIN users u ON u.user_id = fr.requester_id
            LEFT JOIN profiles p ON p.user_id = fr.requester_id
            LEFT JOIN user_equipped_cosmetics uec_avatar ON uec_avatar.user_id = fr.requester_id AND uec_avatar.item_type = 'avatar'
            LEFT JOIN cosmetic_items equipped_avatar ON equipped_avatar.id = uec_avatar.cosmetic_id AND equipped_avatar.item_type = 'avatar'
            WHERE fr.addressee_id = $1 AND fr.status = 'pending'
            ORDER BY fr.created_at DESC
            """,
            user_id,
        )
        return [
            {
                "id": r["id"],
                "requester_id": r["requester_id"],
                "display_name": r["display_name"],
                "avatar_url": r["avatar_url"],
                "status": r["status"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    async def get_outgoing_friend_requests(self, user_id: int) -> list[dict[str, Any]]:
        """Исходящие заявки в друзья (requester = user_id)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT fr.id, fr.addressee_id, fr.status, fr.created_at,
                   COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') AS display_name,
                   COALESCE(equipped_avatar.asset_path, NULLIF(p.img, '')) AS avatar_url
            FROM friend_requests fr
            JOIN users u ON u.user_id = fr.addressee_id
            LEFT JOIN profiles p ON p.user_id = fr.addressee_id
            LEFT JOIN user_equipped_cosmetics uec_avatar ON uec_avatar.user_id = fr.addressee_id AND uec_avatar.item_type = 'avatar'
            LEFT JOIN cosmetic_items equipped_avatar ON equipped_avatar.id = uec_avatar.cosmetic_id AND equipped_avatar.item_type = 'avatar'
            WHERE fr.requester_id = $1 AND fr.status = 'pending'
            ORDER BY fr.created_at DESC
            """,
            user_id,
        )
        return [
            {
                "id": r["id"],
                "addressee_id": r["addressee_id"],
                "display_name": r["display_name"],
                "avatar_url": r["avatar_url"],
                "status": r["status"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    async def get_friend_list(self, user_id: int) -> list[dict[str, Any]]:
        """Список подтверждённых друзей (accepted rows в любом направлении)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT friend_id, display_name, avatar_url FROM (
                SELECT fr.addressee_id AS friend_id,
                       COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') AS display_name,
                       COALESCE(equipped_avatar.asset_path, NULLIF(p.img, '')) AS avatar_url
                FROM friend_requests fr
                JOIN users u ON u.user_id = fr.addressee_id
                LEFT JOIN profiles p ON p.user_id = fr.addressee_id
                LEFT JOIN user_equipped_cosmetics uec_avatar ON uec_avatar.user_id = fr.addressee_id AND uec_avatar.item_type = 'avatar'
                LEFT JOIN cosmetic_items equipped_avatar ON equipped_avatar.id = uec_avatar.cosmetic_id AND equipped_avatar.item_type = 'avatar'
                WHERE fr.requester_id = $1 AND fr.status = 'accepted'
                UNION
                SELECT fr.requester_id AS friend_id,
                       COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') AS display_name,
                       COALESCE(equipped_avatar.asset_path, NULLIF(p.img, '')) AS avatar_url
                FROM friend_requests fr
                JOIN users u ON u.user_id = fr.requester_id
                LEFT JOIN profiles p ON p.user_id = fr.requester_id
                LEFT JOIN user_equipped_cosmetics uec_avatar ON uec_avatar.user_id = fr.requester_id AND uec_avatar.item_type = 'avatar'
                LEFT JOIN cosmetic_items equipped_avatar ON equipped_avatar.id = uec_avatar.cosmetic_id AND equipped_avatar.item_type = 'avatar'
                WHERE fr.addressee_id = $1 AND fr.status = 'accepted'
            ) sub
            ORDER BY display_name
            """,
            user_id,
        )
        return [
            {
                "user_id": r["friend_id"],
                "display_name": r["display_name"],
                "avatar_url": r["avatar_url"],
            }
            for r in rows
        ]

    async def get_friend_ids_set(self, user_id: int) -> set[int]:
        """Быстрый lookup: множество ID друзей пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT friend_id FROM (
                SELECT addressee_id AS friend_id
                FROM friend_requests
                WHERE requester_id = $1 AND status = 'accepted'
                UNION
                SELECT requester_id AS friend_id
                FROM friend_requests
                WHERE addressee_id = $1 AND status = 'accepted'
            ) sub
            """,
            user_id,
        )
        return {r["friend_id"] for r in rows}

    async def remove_friendship(self, user_id: int, friend_id: int) -> bool:
        """Удалить дружбу (статус accepted) между двумя пользователями.
        Возвращает True если удалено."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        result = await self.execute(
            """
            DELETE FROM friend_requests
            WHERE status = 'accepted'
              AND ((requester_id = $1 AND addressee_id = $2)
                   OR (requester_id = $2 AND addressee_id = $1))
            """,
            user_id, friend_id,
        )
        return bool(result)

    async def are_friends(self, user_a: int, user_b: int) -> bool:
        """Проверка: друзья ли два пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        val = await self.fetchval(
            """
            SELECT 1 FROM friend_requests
            WHERE status = 'accepted'
              AND ((requester_id = $1 AND addressee_id = $2)
                   OR (requester_id = $2 AND addressee_id = $1))
            LIMIT 1
            """,
            user_a, user_b,
        )
        return bool(val)

    async def get_recent_opponents(self, user_id: int, limit: int = 5) -> list[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            WITH all_opponents AS (
                -- battle_summary: p1_user_id / p2_user_id are always filled
                SELECT p2_user_id AS opponent_id, created_at
                FROM battle_summary
                WHERE p1_user_id = $1 AND p2_user_id IS NOT NULL
                UNION ALL
                SELECT p1_user_id AS opponent_id, created_at
                FROM battle_summary
                WHERE p2_user_id = $1 AND p1_user_id IS NOT NULL
                UNION ALL
                -- legacy with p1/p2 filled
                SELECT p2_id AS opponent_id, created_at
                FROM battle_results
                WHERE p1_id = $1 AND p2_id IS NOT NULL
                  AND match_id NOT IN (SELECT match_id FROM battle_summary)
                UNION ALL
                SELECT p1_id AS opponent_id, created_at
                FROM battle_results
                WHERE p2_id = $1 AND p1_id IS NOT NULL
                  AND match_id NOT IN (SELECT match_id FROM battle_summary)
                UNION ALL
                -- legacy with null p1/p2: resolve via winner/loser
                SELECT loser_id AS opponent_id, created_at
                FROM battle_results
                WHERE winner_id = $1 AND loser_id IS NOT NULL
                  AND p1_id IS NULL AND p2_id IS NULL
                  AND match_id NOT IN (SELECT match_id FROM battle_summary)
                UNION ALL
                SELECT winner_id AS opponent_id, created_at
                FROM battle_results
                WHERE loser_id = $1 AND winner_id IS NOT NULL
                  AND p1_id IS NULL AND p2_id IS NULL
                  AND match_id NOT IN (SELECT match_id FROM battle_summary)
            ),
            latest AS (
                SELECT DISTINCT ON (opponent_id) opponent_id, created_at
                FROM all_opponents
                WHERE opponent_id IS NOT NULL
                ORDER BY opponent_id, created_at DESC
            )
            SELECT l.opponent_id, p.custom_nickname,
                   COALESCE(equipped_avatar.asset_path, NULLIF(p.img, '')) AS avatar_url,
                   COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') AS display_name,
                   (SELECT 1 FROM friend_requests fr
                    WHERE fr.status = 'accepted'
                      AND ((fr.requester_id = $1 AND fr.addressee_id = l.opponent_id)
                           OR (fr.requester_id = l.opponent_id AND fr.addressee_id = $1))
                    LIMIT 1) IS NOT NULL AS is_friend,
                   l.created_at AS last_battle_at
            FROM latest l
            JOIN users u ON u.user_id = l.opponent_id
            LEFT JOIN profiles p ON p.user_id = l.opponent_id
            LEFT JOIN user_equipped_cosmetics uec_avatar ON uec_avatar.user_id = l.opponent_id AND uec_avatar.item_type = 'avatar'
            LEFT JOIN cosmetic_items equipped_avatar ON equipped_avatar.id = uec_avatar.cosmetic_id AND equipped_avatar.item_type = 'avatar'
            ORDER BY l.created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )
        return [
            {
                "user_id": r["opponent_id"],
                "display_name": r["display_name"],
                "avatar_url": r["avatar_url"],
                "is_friend": bool(r["is_friend"]),
            }
            for r in rows
        ]

    async def create_payment(
        self,
        *,
        user_id: int,
        payment_id: str,
        amount: float,
        currency: str,
        description: str,
        metadata: Optional[dict] = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json
        try:
            record = await self.fetchrow(
                """
                INSERT INTO payments (payment_id, user_id, amount, currency, description, metadata, status)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                ON CONFLICT (payment_id) DO UPDATE
                SET amount = EXCLUDED.amount,
                    currency = EXCLUDED.currency,
                    description = EXCLUDED.description,
                    metadata = EXCLUDED.metadata,
                    status = EXCLUDED.status,
                    updated_at = NOW()
                RETURNING id
                """,
                payment_id,
                user_id,
                amount,
                currency,
                description,
                _json.dumps(metadata or {}),
                status,
            )
            return {"success": True, "id": record["id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_payment_by_id(self, payment_id: str) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            "SELECT * FROM payments WHERE payment_id = $1", payment_id
        )
        return dict(row) if row else None

    async def update_payment_status(self, payment_id: str, status: str) -> None:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        await self.execute(
            "UPDATE payments SET status = $1, updated_at = NOW() WHERE payment_id = $2",
            status, payment_id,
        )

    async def claim_payment_for_processing(self, payment_id: str) -> Optional[dict[str, Any]]:
        """Atomically reserve a payment reward grant.

        The payment processor has multiple entry points (webhook, status polling,
        Telegram Stars). This single UPDATE is the idempotency gate that ensures
        only one worker can grant rewards for a payment.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            """
            UPDATE payments
            SET rewards_processed = TRUE,
                updated_at = NOW()
            WHERE payment_id = $1
              AND rewards_processed = FALSE
            RETURNING *
            """,
            payment_id,
        )
        return dict(row) if row else None

    async def release_payment_processing_claim(self, payment_id: str) -> None:
        """Allow retry when reward processing failed before any reward was granted."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        await self.execute(
            """
            UPDATE payments
            SET rewards_processed = FALSE,
                updated_at = NOW()
            WHERE payment_id = $1
            """,
            payment_id,
        )

    async def get_user_payment_history(
        self, user_id: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT payment_id, user_id, amount, currency, description, metadata,
                   status, rewards_processed, created_at, updated_at
            FROM payments
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )
        return [dict(row) for row in rows]

    async def mark_payment_modal_shown(self, payment_id: str, user_id: int) -> bool:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        updated = await self.fetchval(
            """
            UPDATE payments
            SET metadata = metadata || '{"modal_shown": true}'::jsonb
            WHERE payment_id = $1
              AND user_id = $2
              AND (metadata->>'modal_shown') IS DISTINCT FROM 'true'
            RETURNING 1
            """,
            payment_id,
            user_id,
        )
        return bool(updated)

    async def has_any_purchase(self, user_id: int) -> bool:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            "SELECT 1 FROM payments WHERE user_id = $1 AND status = 'succeeded' LIMIT 1",
            user_id,
        )
        return bool(row)

    async def create_community_post(
        self,
        *,
        author_id: int,
        title: str,
        content: str,
        photo_file_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Создать пост в коммьюнити. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            await self.execute(
                """
                INSERT INTO community_posts (author_id, title, content, photo_file_id)
                VALUES ($1, $2, $3, $4)
                """,
                author_id,
                title,
                content,
                photo_file_id,
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_community_post(self, post_id: int, user_id: int) -> dict[str, Any]:
        """Удалить пост коммьюнити. Только админ может удалять посты."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        # Проверяем, что пользователь - админ
        if user_id != 6803854304:
            return {"success": False, "error": "admin_only"}

        try:
            # Проверяем существование поста
            post = await self.fetchrow(
                "SELECT id FROM community_posts WHERE id = $1",
                post_id
            )
            if not post:
                return {"success": False, "error": "post_not_found"}

            # Удаляем лайки поста
            await self.execute(
                "DELETE FROM post_likes WHERE post_id = $1",
                post_id
            )

            # Удаляем пост
            await self.execute(
                "DELETE FROM community_posts WHERE id = $1",
                post_id
            )

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_community_posts(self, limit: int = 50, user_id: int | None = None) -> list[dict[str, Any]]:
        """Получить список постов коммьюнити. Если указан user_id, добавляет информацию о лайках пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT cp.id, cp.author_id, cp.title, cp.content, cp.photo_file_id, cp.created_at,
                   u.username, u.first_name,
                   COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') as author_name,
                   (SELECT COUNT(*) FROM post_likes WHERE post_id = cp.id) as likes_count,
                   NULL as author_photo_url
            FROM community_posts cp
            LEFT JOIN users u ON u.user_id = cp.author_id
            LEFT JOIN profiles p ON p.user_id = cp.author_id
            ORDER BY cp.created_at DESC
            LIMIT $1
            """,
            limit,
        )
        
        posts = [dict(row) for row in rows]
        
        # Если указан user_id, добавляем информацию о том, лайкнул ли пользователь каждый пост
        if user_id:
            for post in posts:
                post["is_liked"] = await self.is_post_liked_by_user(post["id"], user_id)
        else:
            for post in posts:
                post["is_liked"] = False
        
        return posts

    async def toggle_post_like(self, post_id: int, user_id: int) -> dict[str, Any]:
        """Переключить лайк на посте. Возвращает dict с результатом и текущим состоянием."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            # Проверяем, есть ли уже лайк
            existing = await self.fetchrow(
                "SELECT id FROM post_likes WHERE post_id = $1 AND user_id = $2",
                post_id, user_id
            )

            if existing:
                # Удаляем лайк
                await self.execute(
                    "DELETE FROM post_likes WHERE post_id = $1 AND user_id = $2",
                    post_id, user_id
                )
                liked = False
            else:
                # Добавляем лайк
                await self.execute(
                    "INSERT INTO post_likes (post_id, user_id) VALUES ($1, $2)",
                    post_id, user_id
                )
                liked = True

            # Получаем количество лайков
            likes_count = await self.fetchval(
                "SELECT COUNT(*) FROM post_likes WHERE post_id = $1",
                post_id
            )

            return {
                "success": True,
                "liked": liked,
                "likes_count": int(likes_count or 0)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def is_post_liked_by_user(self, post_id: int, user_id: int) -> bool:
        """Проверить, лайкнул ли пользователь пост."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        exists = await self.fetchval(
            "SELECT 1 FROM post_likes WHERE post_id = $1 AND user_id = $2",
            post_id, user_id
        )
        return exists is not None

    async def create_promocode(
        self,
        code: str,
        type: str,
        reward_gems: int = 0,
        reward_coins: int = 0,
        reward_keys: int = 0,
        reward_extrapass: bool = False,
        created_by: int | None = None,
        expires_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Создать промокод. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            await self.execute(
                """
                INSERT INTO promocodes (code, type, reward_gems, reward_coins, reward_keys, reward_extrapass, created_by, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                code.upper(), type, reward_gems, reward_coins, reward_keys, reward_extrapass, created_by, expires_at
            )
            return {"success": True}
        except asyncpg.UniqueViolationError:
            return {"success": False, "error": "code_exists"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def use_promocode(self, user_id: int, code: str) -> dict[str, Any]:
        """Использовать промокод. Возвращает dict с результатом и наградами."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        code_upper = code.upper()
        
        # Получаем промокод
        promocode = await self.fetchrow(
            """
            SELECT id, type, reward_gems, reward_coins, reward_keys, reward_extrapass, expires_at
            FROM promocodes
            WHERE code = $1
            """,
            code_upper
        )

        if not promocode:
            return {"success": False, "error": "not_found"}

        # Проверяем срок действия
        if promocode["expires_at"]:
            now = datetime.now(timezone.utc)
            if promocode["expires_at"] < now:
                return {"success": False, "error": "expired"}

        # Проверяем, использовал ли пользователь этот промокод
        usage = await self.fetchrow(
            """
            SELECT id FROM promocode_usage
            WHERE promocode_id = $1 AND user_id = $2
            """,
            promocode["id"], user_id
        )

        if usage:
            return {"success": False, "error": "already_used"}

        # Для welcome промокодов проверяем, играл ли пользователь бои
        if promocode["type"] == "welcome":
            try:
                battles_count = await self.fetchval(
                    """
                    SELECT COUNT(*) FROM battles
                    WHERE (player1_id = $1 OR player2_id = $1)
                    """,
                    user_id
                )
                if battles_count and battles_count > 0:
                    return {"success": False, "error": "not_eligible"}
            except Exception:
                # Если таблицы battles нет, считаем что пользователь новый
                pass

        # Используем промокод
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Записываем использование
                await conn.execute(
                    """
                    INSERT INTO promocode_usage (promocode_id, user_id)
                    VALUES ($1, $2)
                    """,
                    promocode["id"], user_id
                )

                # Выдаем награды
                if promocode["reward_gems"] > 0:
                    await conn.execute(
                        "UPDATE users SET gems = gems + $1 WHERE user_id = $2",
                        promocode["reward_gems"], user_id
                    )
                if promocode["reward_coins"] > 0:
                    await conn.execute(
                        "UPDATE users SET coins = coins + $1 WHERE user_id = $2",
                        promocode["reward_coins"], user_id
                    )
                if promocode["reward_keys"] > 0:
                    await conn.execute(
                        "UPDATE users SET keys = keys + $1 WHERE user_id = $2",
                        promocode["reward_keys"], user_id
                    )
                if promocode["reward_extrapass"]:
                    await conn.execute(
                        "UPDATE users SET extra_pass = 'active' WHERE user_id = $1",
                        user_id,
                    )

                # Если промокод персональный, удаляем его
                if promocode["type"] == "personal":
                    await conn.execute(
                        "DELETE FROM promocodes WHERE id = $1",
                        promocode["id"]
                    )

        return {
            "success": True,
            "rewards": {
                "gems": promocode["reward_gems"],
                "coins": promocode["reward_coins"],
                "keys": promocode["reward_keys"],
                "extrapass": promocode["reward_extrapass"]
            }
        }

    # ── Generator (Генератор ключей) ──────────────────────────────────────────

    async def _ensure_generator_state(self, user_id: int) -> None:
        """Гарантировать наличие записи генератора для пользователя."""
        await self.execute(
            """
            INSERT INTO generator_state (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
            """,
            user_id,
        )

    async def _compute_generator_accumulated(
        self, row: Any
    ) -> tuple[int, int, int, int]:
        """
        Вычислить накопленные ключи из raw DB row без записи в БД.
        Возвращает (accumulated, new_keys, cap, interval_seconds).
        Работает как с asyncpg.Record, так и с dict.
        """
        from infrastructure.generator_config import GENERATOR_LEVELS

        level = row["level"]
        extra_pass_raw = (row.get("extra_pass") or "inactive")
        pass_tier = extra_pass_raw if extra_pass_raw in ("ultra", "active") else "f2p"
        level_cfg = GENERATOR_LEVELS.get(level, GENERATOR_LEVELS[1])
        tier_cfg = level_cfg.get(pass_tier, level_cfg["f2p"])
        interval_seconds = tier_cfg["interval_hours"] * 3600
        cap = tier_cfg["cap"]

        now = datetime.now(timezone.utc)
        last_tick = row["last_tick_at"]
        if isinstance(last_tick, str):
            last_tick = datetime.fromisoformat(last_tick.replace("Z", "+00:00"))

        stored = row.get("accumulated_keys", 0)
        elapsed = (now - last_tick).total_seconds()
        new_keys = int(elapsed / interval_seconds) if elapsed > 0 else 0
        accumulated = min(stored + new_keys, cap)

        return accumulated, new_keys, cap, interval_seconds

    async def get_generator_status(self, user_id: int) -> dict[str, Any]:
        """Получить полный статус генератора с расчётом текущих накоплений."""
        from infrastructure.generator_config import (
            GENERATOR_LEVELS, GENERATOR_UPGRADE_COST, GENERATOR_MAX_LEVEL,
        )

        await self._ensure_generator_state(user_id)

        row = await self.fetchrow(
            """
            SELECT g.level, g.accumulated_keys, g.last_tick_at, g.notified,
                   u.extra_pass, u.coins, u.gems, u.keys
            FROM generator_state g
            JOIN users u ON u.user_id = g.user_id
            WHERE g.user_id = $1
            """,
            user_id,
        )
        if not row:
            return {"error": "user_not_found"}

        accumulated, new_keys, cap, interval_seconds = await self._compute_generator_accumulated(row)
        interval_hours = interval_seconds // 3600
        level = row["level"]
        extra_pass = row["extra_pass"] or "inactive"
        tier = "ultra" if extra_pass == "ultra" else "active" if extra_pass == "active" else "f2p"

        now = datetime.now(timezone.utc)
        last_tick = row["last_tick_at"]
        if isinstance(last_tick, str):
            last_tick = datetime.fromisoformat(last_tick.replace("Z", "+00:00"))

        ticks_used = new_keys * interval_seconds
        next_key_at = last_tick.timestamp() + ticks_used + interval_seconds if accumulated < cap else None
        next_key_seconds = max(0, next_key_at - now.timestamp()) if next_key_at else None

        upgrade_cost = None
        next_level = level + 1
        if next_level in GENERATOR_UPGRADE_COST and next_level <= GENERATOR_MAX_LEVEL:
            upgrade_cost = dict(GENERATOR_UPGRADE_COST[next_level])

        if new_keys > 0:
            await self.execute(
                """
                UPDATE generator_state
                SET accumulated_keys = $2, last_tick_at = last_tick_at + ($3::int * INTERVAL '1 second'),
                    updated_at = NOW()
                WHERE user_id = $1
                """,
                user_id, accumulated, ticks_used,
            )

        return {
            "level": level,
            "max_level": GENERATOR_MAX_LEVEL,
            "accumulated_keys": accumulated,
            "cap": cap,
            "interval_hours": interval_hours,
            "interval_seconds": interval_seconds,
            "tier": tier,
            "levels": [
                {
                    "level": lvl,
                    "interval_hours": cfg[tier]["interval_hours"],
                    "cap": cfg[tier]["cap"],
                    "upgrade_cost": dict(GENERATOR_UPGRADE_COST.get(lvl, {})),
                }
                for lvl, cfg in sorted(GENERATOR_LEVELS.items())
            ],
            "can_claim": accumulated > 0,
            "notified": row["notified"],
            "next_key_seconds": round(next_key_seconds, 1) if next_key_seconds is not None else None,
            "upgrade_cost": upgrade_cost,
            "user_coins": row["coins"] or 0,
            "user_gems": row["gems"] or 0,
            "user_keys": row["keys"] or 0,
        }

    async def claim_generator_keys(self, user_id: int) -> dict[str, Any]:
        """Забрать накопленные ключи из генератора в users.keys. Пересчитывает накопление под локом."""
        await self._ensure_generator_state(user_id)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT g.level, g.accumulated_keys, g.last_tick_at, g.notified,
                           u.extra_pass
                    FROM generator_state g
                    JOIN users u ON u.user_id = g.user_id
                    WHERE g.user_id = $1
                    FOR UPDATE
                    """,
                    user_id,
                )
                if not row:
                    return {"success": False, "error": "no_generator_state"}

                accumulated, new_keys, cap, interval_seconds = await self._compute_generator_accumulated(row)
                if accumulated <= 0:
                    return {"success": False, "error": "no_keys_accumulated"}

                ticks_used = new_keys * interval_seconds

                await conn.execute(
                    """
                    UPDATE generator_state
                    SET accumulated_keys = 0,
                        last_tick_at = CASE WHEN $2 > 0 THEN last_tick_at + ($2::int * INTERVAL '1 second') ELSE NOW() END,
                        notified = FALSE, updated_at = NOW()
                    WHERE user_id = $1
                    """,
                    user_id, ticks_used,
                )
                await conn.execute(
                    "UPDATE users SET keys = COALESCE(keys, 0) + $2, updated_at = NOW() WHERE user_id = $1",
                    user_id, accumulated,
                )
                await conn.execute(
                    "UPDATE notifications SET sent = FALSE, sent_at = NULL WHERE user_id = $1 AND notification_type = 'generator'",
                    user_id,
                )
                total_keys = await conn.fetchval(
                    "SELECT keys FROM users WHERE user_id = $1", user_id
                )

        return {
            "success": True,
            "keys_claimed": accumulated,
            "claimed": accumulated,
            "level": row["level"],
            "total_keys": total_keys or accumulated,
        }

    async def upgrade_generator(self, user_id: int, currency: str | None = None) -> dict[str, Any]:
        """Повысить уровень генератора за gems. Legacy currency payload игнорируется."""
        from infrastructure.generator_config import (
            GENERATOR_UPGRADE_COST, GENERATOR_MAX_LEVEL,
        )

        await self._ensure_generator_state(user_id)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT g.level, u.coins, u.gems
                    FROM generator_state g
                    JOIN users u ON u.user_id = g.user_id
                    WHERE g.user_id = $1
                    FOR UPDATE
                    """,
                    user_id,
                )
                if not row:
                    return {"success": False, "error": "user_not_found"}

                current_level = row["level"]
                next_level = current_level + 1

                if next_level > GENERATOR_MAX_LEVEL:
                    return {"success": False, "error": "max_level_reached"}

                costs = GENERATOR_UPGRADE_COST.get(next_level)
                if not costs:
                    return {"success": False, "error": "no_upgrade_available"}

                gem_price = int(costs.get("gems", 0))
                current_gems = row["gems"] or 0
                if current_gems < gem_price:
                    return {
                        "success": False,
                        "error": "not_enough_gems",
                        "required": gem_price,
                        "have": current_gems,
                    }

                await conn.execute(
                    "UPDATE users SET gems = GREATEST(0, COALESCE(gems, 0) - $2), updated_at = NOW() WHERE user_id = $1",
                    user_id, gem_price,
                )

                await conn.execute(
                    """
                    UPDATE generator_state
                    SET level = $2, accumulated_keys = 0, last_tick_at = NOW(),
                        notified = FALSE, updated_at = NOW()
                    WHERE user_id = $1
                    """,
                    user_id, next_level,
                )
                await conn.execute(
                    "UPDATE notifications SET sent = FALSE, sent_at = NULL WHERE user_id = $1 AND notification_type = 'generator'",
                    user_id,
                )

                coins_after = await conn.fetchval("SELECT coins FROM users WHERE user_id = $1", user_id)
                gems_after = await conn.fetchval("SELECT gems FROM users WHERE user_id = $1", user_id)

        return {
            "success": True,
            "old_level": current_level,
            "new_level": next_level,
            "currency_spent": "gems",
            "amount_spent": gem_price,
            "cost": {"gems": gem_price},
            "coins_remaining": coins_after or 0,
            "gems_remaining": gems_after or 0,
        }

    async def check_generator_notifications(self) -> list[dict[str, Any]]:
        """Поставить в очередь события генератора ключей.
        Рассчитывает готовность из last_tick_at + конфиг, НЕ из stored accumulated_keys."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")

        rows = await self.fetch(
            """
            SELECT g.user_id, g.level, g.accumulated_keys, g.last_tick_at, g.notified,
                   u.extra_pass, u.status
            FROM generator_state g
            INNER JOIN users u ON u.user_id = g.user_id
            LEFT JOIN user_settings us ON us.user_id = g.user_id
            WHERE u.status = 'active'
              AND (us.notif_generator = TRUE OR us.notif_generator IS NULL)
            """
        )

        result = []
        for r in rows:
            accumulated, new_keys, cap, interval_seconds = await self._compute_generator_accumulated(r)
            stored = int(r["accumulated_keys"] or 0)
            event_type = classify_generator_event(
                stored_keys=stored,
                new_keys=new_keys,
                cap=cap,
            )
            if not event_type:
                continue
            last_tick = r["last_tick_at"]
            if isinstance(last_tick, str):
                last_tick = datetime.fromisoformat(last_tick.replace("Z", "+00:00"))
            tick_bucket = int((datetime.now(timezone.utc) - last_tick).total_seconds() // interval_seconds)
            payload = {
                "keys": accumulated,
                "cap": cap,
                "level": int(r["level"] or 1),
                "section": "generator",
            }
            enqueued = await self.enqueue_notification(
                int(r["user_id"]),
                category="generator",
                event_type=event_type,
                payload=payload,
                dedupe_key=f"generator:{r['user_id']}:{event_type}:{tick_bucket}",
            )
            if enqueued:
                result.append({"user_id": r["user_id"], "event_type": event_type, **payload})

        return result

    async def mark_generator_notification_sent(self, user_id: int) -> None:
        """Отметить, что уведомление о генераторе отправлено."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        await self.execute(
            """
            INSERT INTO notifications (user_id, notification_type, sent, sent_at)
            VALUES ($1, 'generator', TRUE, NOW())
            ON CONFLICT (user_id, notification_type) DO UPDATE
            SET sent = TRUE, sent_at = NOW()
            """,
            user_id,
        )
        await self.execute(
            "UPDATE generator_state SET notified = TRUE, updated_at = NOW() WHERE user_id = $1",
            user_id,
        )


    async def get_promocodes_list(self, created_by: int | None = None) -> list[dict[str, Any]]:
        """Получить список промокодов."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        if created_by:
            rows = await self.fetch(
                """
                SELECT id, code, type, reward_gems, reward_coins, reward_keys, reward_extrapass, created_at, expires_at,
                       (SELECT COUNT(*) FROM promocode_usage WHERE promocode_id = promocodes.id) as usage_count
                FROM promocodes
                WHERE created_by = $1
                ORDER BY created_at DESC
                """,
                created_by
            )
        else:
            rows = await self.fetch(
                """
                SELECT id, code, type, reward_gems, reward_coins, reward_keys, reward_extrapass, created_at, expires_at,
                       (SELECT COUNT(*) FROM promocode_usage WHERE promocode_id = promocodes.id) as usage_count
                FROM promocodes
                ORDER BY created_at DESC
                """
            )

        return [dict(row) for row in rows]

    async def delete_promocode(self, promocode_id: int) -> dict[str, Any]:
        """Удалить промокод из админ-панели."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            status = await self.execute("DELETE FROM promocodes WHERE id = $1", int(promocode_id))
            deleted = status.endswith(" 1")
            return {"success": deleted, "error": None if deleted else "not_found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _ensure_cards_table(self) -> bool:
        """Создать таблицу карт."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.cards')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE cards (
                    id BIGSERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    rarity TEXT NOT NULL CHECK (rarity IN ('common', 'rare', 'start', 'superrare', 'epic', 'legendary', 'mythic', 'divine', 'limited', 'unique')),
                    power INT NOT NULL DEFAULT 0,
                    mana_cost INT NOT NULL DEFAULT 3,
                    base_attack INT NOT NULL DEFAULT 100,
                    base_hp INT NOT NULL DEFAULT 100,
                    mechanics JSONB NOT NULL DEFAULT '[]'::jsonb,
                    card_type TEXT NOT NULL DEFAULT 'warrior',
                    image_file_id TEXT,
                    created_by BIGINT,
                    simplified_levelup BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        columns = await self._get_columns("cards")
        changed |= await self._add_column_if_missing("cards", columns, "name TEXT NOT NULL DEFAULT ''")
        changed |= await self._add_column_if_missing("cards", columns, "description TEXT")
        changed |= await self._add_column_if_missing("cards", columns, "rarity TEXT NOT NULL DEFAULT 'common'")
        changed |= await self._add_column_if_missing("cards", columns, "power INT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("cards", columns, "mana_cost INT NOT NULL DEFAULT 3")
        changed |= await self._add_column_if_missing("cards", columns, "base_attack INT NOT NULL DEFAULT 100")
        changed |= await self._add_column_if_missing("cards", columns, "base_hp INT NOT NULL DEFAULT 100")
        changed |= await self._add_column_if_missing("cards", columns, "mechanics JSONB NOT NULL DEFAULT '[]'::jsonb")
        changed |= await self._add_column_if_missing("cards", columns, "card_type TEXT NOT NULL DEFAULT 'warrior'")
        changed |= await self._add_column_if_missing("cards", columns, "image_file_id TEXT")
        changed |= await self._add_column_if_missing("cards", columns, "created_by BIGINT")
        changed |= await self._add_column_if_missing("cards", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("cards", columns, "mechanics_desc TEXT")
        changed |= await self._add_column_if_missing("cards", columns, "simplified_levelup BOOLEAN NOT NULL DEFAULT FALSE")

        rarity_constraint_def = await self.fetchval(
            """
            SELECT pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conrelid = 'public.cards'::regclass
              AND conname = 'cards_rarity_check'
            """
        )
        if not rarity_constraint_def or "'start'::text" not in str(rarity_constraint_def):
            if rarity_constraint_def:
                await self.execute("ALTER TABLE cards DROP CONSTRAINT cards_rarity_check")
            await self.execute(
                """
                ALTER TABLE cards
                ADD CONSTRAINT cards_rarity_check
                CHECK (rarity IN ('common', 'rare', 'start', 'superrare', 'epic', 'legendary', 'mythic', 'divine', 'limited', 'unique'))
                """
            )
            changed = True

        # Populate mechanics_desc for cards that have none yet
        for card_id, desc in _CARD_MECHANICS_DESC.items():
            await self.execute(
                "UPDATE cards SET mechanics_desc = $1 WHERE id = $2 AND (mechanics_desc IS NULL OR mechanics_desc = '')",
                desc, card_id
            )

        await self.execute("UPDATE cards SET simplified_levelup = TRUE WHERE id = ANY($1::bigint[])", [11, 12, 13])
        await self.execute("UPDATE cards SET simplified_levelup = FALSE WHERE id <> ALL($1::bigint[])", [11, 12, 13])

        return changed


    async def _ensure_user_cards_table(self) -> bool:
        """Создать таблицу связи карт с игроками."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.user_cards')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE user_cards (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    card_id BIGINT NOT NULL,
                    obtained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(user_id, card_id)
                )
                """
            )
            changed = True

        columns = await self._get_columns("user_cards")
        changed |= await self._add_column_if_missing("user_cards", columns, "user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("user_cards", columns, "card_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("user_cards", columns, "obtained_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("user_cards", columns, "level INTEGER NOT NULL DEFAULT 1")
        changed |= await self._add_column_if_missing("user_cards", columns, "particles INTEGER NOT NULL DEFAULT 0")

        await self.execute("UPDATE user_cards SET level = LEAST(level, 2) WHERE card_id = ANY($1::bigint[]) AND level > 2", [11, 12, 13])

        # Создаем индекс для быстрого поиска по user_id
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes 
            WHERE tablename = 'user_cards' AND indexname = 'user_cards_user_id_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX user_cards_user_id_idx ON user_cards(user_id)")
            changed = True

        # Создаем индекс для быстрого поиска по card_id
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes 
            WHERE tablename = 'user_cards' AND indexname = 'user_cards_card_id_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX user_cards_card_id_idx ON user_cards(card_id)")
            changed = True

        # Создаем уникальный индекс для user_id + card_id
        unique_constraint_exists = await self._constraint_exists("user_cards", "user_cards_user_id_card_id_key")
        if not unique_constraint_exists:
            await self.execute(
                """
                ALTER TABLE user_cards
                ADD CONSTRAINT user_cards_user_id_card_id_key
                UNIQUE (user_id, card_id)
                """
            )
            changed = True

        return changed


    async def create_card(
        self,
        name: str,
        description: str,
        rarity: str,
        power: int,
        image_file_id: str | None = None,
        created_by: int | None = None,
        mana_cost: int = 3,
        base_attack: int = 100,
        base_hp: int = 100,
        mechanics: list[str] | None = None,
        card_type: str = 'warrior',
    ) -> dict[str, Any]:
        """Создать карту. Возвращает dict с результатом и card_id."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            card_id = await self.fetchval(
                """
                INSERT INTO cards (name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                name,
                description,
                rarity,
                power,
                mana_cost,
                base_attack,
                base_hp,
                json.dumps(mechanics or []),
                card_type,
                image_file_id,
                created_by,
            )
            return {"success": True, "card_id": card_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_card_info(self, card_id: int, level: int = 1) -> dict[str, Any] | None:
        """
        Получить карточку по id со всеми новыми полями (мана/атака/хп/механики).
        level нужен для боевого движка, чтобы сразу вернуть текущие статы.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        row = await self.fetchrow(
            """
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by, created_at, mechanics_desc, simplified_levelup
            FROM cards
            WHERE id = $1
            """,
            card_id,
        )
        if not row:
            return None

        card = Card.from_row(row)
        stats = card.get_current_stats(level=level)
        card_dict = card.to_dict()
        card_dict.update(
            {
                "current_attack": stats["attack"],
                "current_hp": stats["hp"],
                "mana_cost": stats["mana"],
                "mechanics": stats["mechanics"],
                "rarity_growth": stats["growth"],
                "requested_level": stats["level"],
                "max_level": stats["max_level"],
                "is_max_level": stats["is_max_level"],
            }
        )
        card_dict.update(self._upgrade_cost_fields(card_dict["rarity"], stats["level"], card_dict.get("simplified_levelup", False)))
        return card_dict


    async def get_cards_list(self) -> list[dict[str, Any]]:
        """Получить список всех карт."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by, created_at, mechanics_desc, simplified_levelup
            FROM cards
            ORDER BY created_at DESC
            """
        )
        cards: list[dict[str, Any]] = []
        for row in rows:
            card = Card.from_row(row)
            card_dict = card.to_dict()
            # Сразу считаем стартовые статы для UI/боевого движка
            stats = card.get_current_stats(level=1)
            card_dict.update(
                {
                    "current_attack": stats["attack"],
                    "current_hp": stats["hp"],
                    "mana_cost": stats["mana"],
                    "mechanics": stats["mechanics"],
                    "rarity_growth": stats["growth"],  # важно для фронта/балансера
                    "level": stats["level"],
                    "max_level": stats["max_level"],
                    "is_max_level": stats["is_max_level"],
                }
            )
            card_dict.update(self._upgrade_cost_fields(card_dict["rarity"], stats["level"], card_dict.get("simplified_levelup", False)))
            cards.append(card_dict)
        return cards

    async def get_cards_by_rarity(self, rarity: str) -> list[dict[str, Any]]:
        """Получить все карты указанной редкости."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        rows = await self.fetch(
            """
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by, created_at, mechanics_desc, simplified_levelup
            FROM cards
            WHERE rarity = $1
            ORDER BY created_at DESC
            """,
            rarity
        )
        return [dict(row) for row in rows]

    async def get_uni_card(self) -> dict[str, Any] | None:
        """Получить стартовую карту Юни (rarity='start' или card_type='hero')."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        row = await self.fetchrow(
            """
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by, created_at, mechanics_desc, simplified_levelup
            FROM cards
            WHERE rarity = 'start' OR card_type = 'hero'
            LIMIT 1
            """
        )
        return dict(row) if row else None

    async def get_user_cards(self, user_id: int) -> list[dict[str, Any]]:
        """Получить все карты пользователя из его коллекции."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT c.id,
                   c.name,
                   c.description,
                   c.rarity,
                   c.power,
                   c.mana_cost,
                   c.base_attack,
                   c.base_hp,
                   c.mechanics,
                   c.card_type,
                   c.image_file_id,
                   c.created_at,
                   c.mechanics_desc,
                   c.simplified_levelup,
                   uc.obtained_at,
                   COALESCE(uc.level, 1) as level,
                   COALESCE(uc.particles, 0) as particles
            FROM user_cards uc
            INNER JOIN cards c ON c.id = uc.card_id
            WHERE uc.user_id = $1
            ORDER BY COALESCE(uc.obtained_at, c.created_at) DESC, c.created_at DESC
            """,
            user_id
        )
        cards: list[dict[str, Any]] = []
        for row in rows:
            card_model = Card.from_row(row)
            card_dict = card_model.to_dict()
            row_dict = dict(row)
            card_dict["obtained_at"] = row_dict.get("obtained_at")
            card_dict["level"] = row_dict.get("level", 1)
            card_dict["particles"] = row_dict.get("particles", 0)
            # Используем новую модель карты, чтобы рост статов брался из level_multiplier
            stats = card_model.get_current_stats(level=card_dict["level"])
            card_dict.update(
                {
                    "current_attack": stats["attack"],
                    "current_hp": stats["hp"],
                    "mana_cost": stats["mana"],
                    "mechanics": stats["mechanics"],
                    "rarity_growth": stats["growth"],
                    "level": stats["level"],
                    "max_level": stats["max_level"],
                    "is_max_level": stats["is_max_level"],
                }
            )
            card_dict.update(self._upgrade_cost_fields(card_dict["rarity"], stats["level"], card_dict.get("simplified_levelup", False)))
            cards.append(card_dict)
        return cards

    async def get_collection_with_status(self, user_id: int) -> list[dict[str, Any]]:
        """Получить все карты каталога с признаком владения для экрана коллекции."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT c.id, c.name, c.description, c.rarity, c.power, c.mana_cost,
                   c.base_attack, c.base_hp, c.mechanics, c.card_type,
                   c.image_file_id, c.created_by, c.created_at, c.mechanics_desc, c.simplified_levelup,
                   (uc.card_id IS NOT NULL) AS owned,
                   COALESCE(uc.level, 1) AS level,
                   COALESCE(uc.particles, 0) AS particles,
                   uc.obtained_at
            FROM cards c
            LEFT JOIN user_cards uc ON c.id = uc.card_id AND uc.user_id = $1
            ORDER BY c.id
            """,
            user_id
        )
        cards: list[dict[str, Any]] = []
        for row in rows:
            card_model = Card.from_row(row)
            row_dict = dict(row)
            owned = bool(row_dict.get("owned", False))
            level = row_dict.get("level", 1)
            card_dict = card_model.to_dict()
            card_dict["locked"] = not owned
            card_dict["level"] = level
            card_dict["particles"] = row_dict.get("particles", 0)
            card_dict["obtained_at"] = row_dict.get("obtained_at")
            stats = card_model.get_current_stats(level=level)
            card_dict.update(
                {
                    "current_attack": stats["attack"],
                    "current_hp": stats["hp"],
                    "mana_cost": stats["mana"],
                    "mechanics": stats["mechanics"],
                    "rarity_growth": stats["growth"],
                    "level": stats["level"],
                    "max_level": stats["max_level"],
                    "is_max_level": stats["is_max_level"],
                }
            )
            card_dict.update(self._upgrade_cost_fields(card_dict["rarity"], stats["level"], card_dict.get("simplified_levelup", False)))
            cards.append(card_dict)
        return cards

    async def get_player_deck_max_level(
        self, user_id: int, selected_deck_id: int | None = None
    ) -> int:
        """
        Максимальный (не средний) уровень карт выбранной колоды игрока.
        Используется как источник истины для bot difficulty scaling.

        Приоритет пресета: selected_deck_id → users.primary_deck → первый валидный.
        Карты только из deck slots (не включает hero отдельно).
        Fallback: 1.
        """
        try:
            if not self._pool:
                return 1

            presets = await self.get_user_deck_presets(user_id)
            if not presets:
                return 1

            preset = None
            if selected_deck_id is not None:
                preset = next(
                    (p for p in presets if p.get("preset_number") == selected_deck_id),
                    None,
                )

            if not preset:
                try:
                    primary = await self.fetchval(
                        "SELECT primary_deck FROM users WHERE user_id = $1", user_id
                    )
                except Exception:
                    primary = None
                if primary is not None:
                    preset = next(
                        (p for p in presets if p.get("preset_number") == primary),
                        None,
                    )

            if not preset:
                preset = presets[0]

            card_ids = preset.get("card_ids", [])
            if not card_ids:
                return 1

            user_cards = await self.get_user_cards(user_id)
            level_by_id: dict[int, int] = {c["id"]: c.get("level", 1) for c in user_cards}

            max_lvl = 1
            for cid in card_ids:
                lvl = level_by_id.get(cid, 1)
                if lvl > max_lvl:
                    max_lvl = lvl

            return max(1, max_lvl)
        except Exception:
            return 1

    async def get_player_deck_avg_level(
        self, user_id: int, selected_deck_id: int | None = None
    ) -> int:
        """
        Средний уровень карт выбранной колоды игрока для информационного UI.

        Приоритет пресета совпадает с get_player_deck_max_level:
        selected_deck_id → users.primary_deck → первый валидный.
        Fallback: 1.
        """
        try:
            if not self._pool:
                return 1

            presets = await self.get_user_deck_presets(user_id)
            if not presets:
                return 1

            preset = None
            if selected_deck_id is not None:
                preset = next(
                    (p for p in presets if p.get("preset_number") == selected_deck_id),
                    None,
                )

            if not preset:
                try:
                    primary = await self.fetchval(
                        "SELECT primary_deck FROM users WHERE user_id = $1", user_id
                    )
                except Exception:
                    primary = None
                if primary is not None:
                    preset = next(
                        (p for p in presets if p.get("preset_number") == primary),
                        None,
                    )

            if not preset:
                preset = presets[0]

            card_ids = preset.get("card_ids", [])
            if not card_ids:
                return 1

            user_cards = await self.get_user_cards(user_id)
            level_by_id: dict[int, int] = {c["id"]: c.get("level", 1) for c in user_cards}
            levels = [max(1, int(level_by_id.get(cid, 1) or 1)) for cid in card_ids]
            if not levels:
                return 1

            return max(1, round(sum(levels) / len(levels)))
        except Exception:
            return 1

    async def _seed_game_defaults(self) -> None:
        """Приводим ключевые карты к новым статам и инициализируем reward_tracks."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        import logging

        try:
            await self.execute(
                """
                UPDATE cards
                SET mana_cost = 3,
                    base_attack = 300,
                    base_hp = 400
                WHERE LOWER(name) = 'uni'
                """
            )

            await self.execute(
                """
                UPDATE cards
                SET mana_cost = 5,
                    base_attack = 250,
                    base_hp = 1400,
                    mechanics = '["taunt"]'::jsonb
                WHERE LOWER(name) = 'midoriya'
                    """
                )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Не удалось применить дефолтные данные боевой системы: {e}"
            )

        await self._seed_reward_tracks()
        await self._seed_default_season()

    async def _seed_default_season(self) -> None:
        """Создать управляемый активный сезон ExtraPass, если в БД его еще нет."""
        if not self._pool:
            return

        import json as _json

        try:
            await self.execute(
                """
                INSERT INTO seasons (
                    slug, name, subtitle, description, season_number, status,
                    auto_switch, preset_key, start_date, end_date, is_active,
                    max_stars, free_track_type, pass_track_type, ultra_track_type,
                    pass_end_position, ultra_start_position, theme
                )
                SELECT
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, NOW(), NOW() + INTERVAL '30 days', TRUE,
                    $9, $10, $11, $12, $13, $14, $15::jsonb
                WHERE NOT EXISTS (
                    SELECT 1 FROM seasons WHERE is_active = TRUE
                )
                """,
                DEFAULT_EXTRA_PASS_SEASON["slug"],
                DEFAULT_EXTRA_PASS_SEASON["name"],
                DEFAULT_EXTRA_PASS_SEASON["subtitle"],
                DEFAULT_EXTRA_PASS_SEASON["description"],
                DEFAULT_EXTRA_PASS_SEASON["season_number"],
                DEFAULT_EXTRA_PASS_SEASON["status"],
                DEFAULT_EXTRA_PASS_SEASON["auto_switch"],
                DEFAULT_EXTRA_PASS_SEASON["preset_key"],
                DEFAULT_EXTRA_PASS_SEASON["max_stars"],
                DEFAULT_EXTRA_PASS_SEASON["free_track_type"],
                DEFAULT_EXTRA_PASS_SEASON["pass_track_type"],
                DEFAULT_EXTRA_PASS_SEASON["ultra_track_type"],
                DEFAULT_EXTRA_PASS_SEASON["pass_end_position"],
                DEFAULT_EXTRA_PASS_SEASON["ultra_start_position"],
                _json.dumps(DEFAULT_EXTRA_PASS_SEASON["theme"]),
            )
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to seed default season: %s", e)

    async def _seed_reward_tracks(self) -> None:
        """Заполнить таблицу reward_tracks начальными значениями."""
        if not self._pool:
            return

        import json as _json
        import logging

        rows = [
            # Glory Path
            ("glory", 150,  "coins", 500,  None,                          False),
            ("glory", 300,  "keys",  1,    None,                          False),
            ("glory", 500,  "coins", 800,  None,                          False),
            ("glory", 700,  "gems",  50,   None,                          False),
            ("glory", 1000, "gems",  150,  '{"original":"case_tier_2","case_tier":2}', False),
            ("glory", 1500, "keys",  2,    None,                          False),
            ("glory", 2000, "coins", 1200, None,                          False),
            ("glory", 2500, "gems",  100,  None,                          False),
            ("glory", 3000, "gems",  300,  '{"original":"case_tier_3","case_tier":3}', False),
            ("glory", 3000, "card",  1,    '{"rarity":["common","rare"]}', False),
            ("glory", 4000, "keys",  2,    None,                          False),
            ("glory", 5000, "coins", 1500, None,                          False),
            ("glory", 6000, "gems",  150,  None,                          False),
            ("glory", 7000, "gems",  300,  '{"original":"case_tier_3","case_tier":3}', False),
            ("glory", 8000, "keys",  3,    None,                          False),
            ("glory", 9000, "coins", 2000, None,                          False),
            ("glory", 9500, "gems",  200,  None,                          False),
            ("glory", 10000,"gems",  500,  '{"original":"case_tier_4","case_tier":4}', False),

            # BP Free
            ("bp_free", 1,  "coins", 200,  None,                         False),
            ("bp_free", 3,  "keys",  1,    None,                         False),
            ("bp_free", 5,  "coins", 300,  None,                         False),
            ("bp_free", 8,  "gems",  50,   None,                         False),
            ("bp_free", 10, "gems",  150,  '{"original":"case_tier_2","case_tier":2}', False),
            ("bp_free", 13, "coins", 400,  None,                         False),
            ("bp_free", 15, "keys",  1,    None,                         False),
            ("bp_free", 18, "coins", 500,  None,                         False),
            ("bp_free", 20, "gems",  300,  '{"original":"case_tier_3","case_tier":3}', False),
            ("bp_free", 20, "gems",  100,  None,                         False),
            ("bp_free", 23, "keys",  2,    None,                         False),
            ("bp_free", 25, "coins", 600,  None,                         False),
            ("bp_free", 28, "gems",  100,  None,                         False),
            ("bp_free", 30, "gems",  300,  '{"original":"case_tier_3","case_tier":3}', False),
            ("bp_free", 30, "card",  1,    '{"rarity":["common","rare"]}', False),
            ("bp_free", 33, "keys",  2,    None,                         False),
            ("bp_free", 35, "coins", 700,  None,                         False),
            ("bp_free", 38, "gems",  150,  None,                         False),
            ("bp_free", 40, "gems",  500,  '{"original":"case_tier_4","case_tier":4}', False),
            ("bp_free", 43, "keys",  3,    None,                         False),
            ("bp_free", 45, "coins", 500,  None,                         False),
            ("bp_free", 45, "gems",  200,  None,                         False),

            # BP Premium
            ("bp_premium", 1,  "coins", 400,  None,                                      True),
            ("bp_premium", 3,  "keys",  2,    None,                                      True),
            ("bp_premium", 5,  "coins", 600,  None,                                      True),
            ("bp_premium", 8,  "gems",  100,  None,                                      True),
            ("bp_premium", 10, "gems",  300,  '{"original":"case_tier_3","case_tier":3}', True),
            ("bp_premium", 10, "gems",  100,  None,                                      True),
            ("bp_premium", 13, "coins", 700,  None,                                      True),
            ("bp_premium", 15, "keys",  2,    None,                                      True),
            ("bp_premium", 18, "coins", 800,  None,                                      True),
            ("bp_premium", 20, "gems",  500,  '{"original":"case_tier_4","case_tier":4}', True),
            ("bp_premium", 20, "gems",  150,  None,                                      True),
            ("bp_premium", 23, "keys",  3,    None,                                      True),
            ("bp_premium", 25, "coins", 1000, None,                                      True),
            ("bp_premium", 28, "gems",  150,  None,                                      True),
            ("bp_premium", 30, "gems",  500,  '{"original":"case_tier_4","case_tier":4}', True),
            ("bp_premium", 30, "card",  1,    '{"rarity":["common","rare"]}',             True),
            ("bp_premium", 33, "keys",  3,    None,                                      True),
            ("bp_premium", 35, "coins", 1000, None,                                      True),
            ("bp_premium", 35, "gems",  100,  None,                                      True),
            ("bp_premium", 38, "gems",  200,  None,                                      True),
            ("bp_premium", 40, "gems",  800,  '{"original":"case_tier_5","case_tier":5}', True),
            ("bp_premium", 43, "keys",  4,    None,                                      True),
            ("bp_premium", 45, "coins", 1000, None,                                      True),
            ("bp_premium", 45, "gems",  300,  None,                                      True),

            # BP Ultra
            ("bp_ultra", 41, "gems", 120, None,                                      True),
            ("bp_ultra", 42, "gems", 500, '{"original":"case_tier_5","case_tier":5}', True),
            ("bp_ultra", 43, "keys", 4,   None,                                      True),
            ("bp_ultra", 44, "gems", 500, None,                                      True),
            ("bp_ultra", 45, "card", 1,   '{"rarity":["common","rare"]}',            True),
            ("bp_ultra", 45, "gems", 300, None,                                      True),
        ]

        seeded_positions = {(track_type, position) for track_type, position, *_ in rows}

        def append_missing(
            track_type: str,
            position: int,
            reward_type: str,
            reward_amount: int,
            reward_meta: str | None,
            extra_pass_required: bool,
        ) -> None:
            if (track_type, position) in seeded_positions:
                return
            rows.append((track_type, position, reward_type, reward_amount, reward_meta, extra_pass_required))
            seeded_positions.add((track_type, position))

        for position in range(1, 46):
            if position % 9 == 0:
                append_missing("bp_free", position, "keys", 1, None, False)
            elif position % 5 == 0:
                append_missing("bp_free", position, "gems", 35 + position * 3, None, False)
            elif position % 4 == 0:
                append_missing("bp_free", position, "card", 1, '{"rarity":["common"]}', False)
            else:
                append_missing("bp_free", position, "coins", 180 + position * 25, None, False)

        for position in range(1, 41):
            if position % 10 == 0:
                append_missing("bp_premium", position, "gems", 180 + position * 8, None, True)
            elif position % 6 == 0:
                append_missing("bp_premium", position, "keys", 2, None, True)
            elif position % 4 == 0:
                append_missing("bp_premium", position, "card", 1, '{"rarity":["rare","epic"]}', True)
            else:
                append_missing("bp_premium", position, "coins", 350 + position * 35, None, True)

        for track_type, position, reward_type, reward_amount, reward_meta, ep_required in rows:
            try:
                await self.execute(
                    """
                    INSERT INTO reward_tracks (track_type, position, reward_type, reward_amount, reward_meta, extra_pass_required)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                    ON CONFLICT (track_type, position, reward_type) DO NOTHING
                    """,
                    track_type, position, reward_type, reward_amount,
                    _json.dumps(_json.loads(reward_meta)) if reward_meta else None,
                    ep_required,
                )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed to seed reward_track %s pos=%d: %s", track_type, position, e
                )

    async def get_reward_tracks(self, track_type: str) -> list[dict[str, Any]]:
        """Получить все активные тиры трека."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT id, track_type, position, reward_type, reward_amount, reward_meta,
                   extra_pass_required, is_active
            FROM reward_tracks
            WHERE track_type = $1 AND is_active = TRUE
            ORDER BY position
            """,
            track_type,
        )
        return [dict(row) for row in rows]

    async def get_all_reward_tracks(self) -> list[dict[str, Any]]:
        """Получить все тиры всех треков (для админки)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT id, track_type, position, reward_type, reward_amount, reward_meta,
                   extra_pass_required, is_active, created_at, updated_at
            FROM reward_tracks
            ORDER BY track_type, position
            """
        )
        return [dict(row) for row in rows]

    async def get_reward_track_by_id(self, reward_id: int) -> dict[str, Any] | None:
        """Получить один тир по id."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            "SELECT id, track_type, position, reward_type, reward_amount, reward_meta, extra_pass_required, is_active FROM reward_tracks WHERE id = $1",
            reward_id,
        )
        return dict(row) if row else None

    async def get_reward_track_entries(self, track_type: str, position: int) -> list[dict[str, Any]]:
        """Получить все награды на конкретной позиции трека."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT id, track_type, position, reward_type, reward_amount, reward_meta,
                   extra_pass_required, is_active
            FROM reward_tracks
            WHERE track_type = $1 AND position = $2 AND is_active = TRUE
            """,
            track_type, position,
        )
        return [dict(row) for row in rows]

    async def create_reward_track(
        self,
        track_type: str,
        position: int,
        reward_type: str,
        reward_amount: int,
        reward_meta: dict[str, Any] | None = None,
        extra_pass_required: bool = False,
    ) -> dict[str, Any]:
        """Создать новый тир."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json
        try:
            row = await self.fetchrow(
                """
                INSERT INTO reward_tracks (track_type, position, reward_type, reward_amount, reward_meta, extra_pass_required)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                ON CONFLICT (track_type, position, reward_type) DO UPDATE
                SET reward_amount = EXCLUDED.reward_amount,
                    reward_meta = EXCLUDED.reward_meta,
                    extra_pass_required = EXCLUDED.extra_pass_required,
                    is_active = TRUE,
                    updated_at = NOW()
                RETURNING id, track_type, position, reward_type, reward_amount, reward_meta, extra_pass_required, is_active
                """,
                track_type, position, reward_type, reward_amount,
                _json.dumps(reward_meta) if reward_meta else None,
                extra_pass_required,
            )
            return dict(row) if row else {"error": "insert_failed"}
        except Exception as e:
            return {"error": str(e)}

    async def update_reward_track(self, reward_id: int, **fields) -> dict[str, Any]:
        """Обновить поля тира. fields может содержать track_type/position и параметры награды."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json

        existing = await self.get_reward_track_by_id(reward_id)
        if not existing:
            return {"error": "not_found"}

        updates = []
        params = []
        idx = 1

        for key in ("track_type", "position", "reward_type", "reward_amount", "extra_pass_required", "is_active"):
            if key in fields and fields[key] is not None:
                updates.append(f"{key} = ${idx}")
                params.append(fields[key])
                idx += 1

        if "reward_meta" in fields:
            updates.append(f"reward_meta = ${idx}::jsonb")
            params.append(_json.dumps(fields["reward_meta"]) if fields["reward_meta"] else None)
            idx += 1

        if not updates:
            return existing

        updates.append(f"updated_at = NOW()")
        params.append(reward_id)

        row = await self.fetchrow(
            f"UPDATE reward_tracks SET {', '.join(updates)} WHERE id = ${idx} RETURNING id, track_type, position, reward_type, reward_amount, reward_meta, extra_pass_required, is_active",
            *params,
        )
        return dict(row) if row else {"error": "update_failed"}

    async def delete_reward_track(self, reward_id: int) -> bool:
        """Мягкое удаление: is_active = FALSE."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        await self.execute(
            "UPDATE reward_tracks SET is_active = FALSE, updated_at = NOW() WHERE id = $1",
            reward_id,
        )
        return True

    async def activate_due_season(self) -> dict[str, Any] | None:
        """Активировать запланированный сезон, если его старт уже наступил."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        candidate = await self.fetchrow(
            """
            SELECT id
            FROM seasons
            WHERE auto_switch = TRUE
              AND start_date IS NOT NULL
              AND start_date <= NOW()
              AND (end_date IS NULL OR end_date > NOW())
              AND status IN ('scheduled', 'active')
            ORDER BY start_date DESC, season_number DESC, id DESC
            LIMIT 1
            """
        )
        if not candidate:
            return None

        candidate_id = int(candidate["id"])
        await self.execute(
            """
            UPDATE seasons
            SET is_active = CASE WHEN id = $1 THEN TRUE ELSE FALSE END,
                status = CASE WHEN id = $1 THEN 'active' ELSE status END,
                updated_at = NOW()
            WHERE is_active = TRUE OR id = $1
            """,
            candidate_id,
        )
        await self.execute(
            """
            UPDATE seasons
            SET status = 'archived',
                is_active = FALSE,
                updated_at = NOW()
            WHERE id <> $1
              AND end_date IS NOT NULL
              AND end_date <= NOW()
              AND status IN ('active', 'scheduled')
            """,
            candidate_id,
        )
        return await self.get_season_by_id(candidate_id)

    async def get_active_season(self) -> dict[str, Any] | None:
        """Получить активный сезон ExtraPass."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        await self.activate_due_season()
        row = await self.fetchrow(
            """
            SELECT id, slug, name, subtitle, description, season_number, status,
                   auto_switch, preset_key, start_date, end_date, is_active,
                   max_stars, free_track_type, pass_track_type, ultra_track_type,
                   pass_end_position, ultra_start_position, theme, created_at, updated_at
            FROM seasons
            WHERE is_active = TRUE
            ORDER BY id DESC
            LIMIT 1
            """
        )
        return dict(row) if row else None

    async def get_season_by_id(self, season_id: int) -> dict[str, Any] | None:
        """Получить сезон по id."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            """
            SELECT id, slug, name, subtitle, description, season_number, status,
                   auto_switch, preset_key, start_date, end_date, is_active,
                   max_stars, free_track_type, pass_track_type, ultra_track_type,
                   pass_end_position, ultra_start_position, theme, created_at, updated_at
            FROM seasons
            WHERE id = $1
            """,
            season_id,
        )
        return dict(row) if row else None

    async def get_seasons(self) -> list[dict[str, Any]]:
        """Получить все сезоны ExtraPass для админки."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        await self.activate_due_season()
        rows = await self.fetch(
            """
            SELECT id, slug, name, subtitle, description, season_number, status,
                   auto_switch, preset_key, start_date, end_date, is_active,
                   max_stars, free_track_type, pass_track_type, ultra_track_type,
                   pass_end_position, ultra_start_position, theme, created_at, updated_at
            FROM seasons
            ORDER BY start_date NULLS LAST, season_number, id
            """
        )
        return [dict(row) for row in rows]

    async def upsert_active_season(self, **fields: Any) -> dict[str, Any]:
        """Обновить активный сезон. Если активного сезона нет, создает дефолтный."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json

        existing = await self.get_active_season()
        if not existing:
            await self._seed_default_season()
            existing = await self.get_active_season()
        if not existing:
            raise RuntimeError("active_season_unavailable")

        updates: list[str] = []
        params: list[Any] = []
        idx = 1

        for key in (
            "slug", "name", "subtitle", "description", "status", "preset_key",
            "free_track_type", "pass_track_type", "ultra_track_type",
        ):
            if key in fields and fields[key] is not None:
                updates.append(f"{key} = ${idx}")
                params.append(str(fields[key]))
                idx += 1

        for key in ("season_number", "max_stars", "pass_end_position", "ultra_start_position"):
            if key in fields and fields[key] is not None:
                updates.append(f"{key} = ${idx}")
                params.append(int(fields[key]))
                idx += 1

        for key in ("start_date", "end_date"):
            if key in fields:
                updates.append(f"{key} = ${idx}::timestamptz")
                params.append(fields[key])
                idx += 1

        if "is_active" in fields and fields["is_active"] is not None:
            updates.append(f"is_active = ${idx}")
            params.append(bool(fields["is_active"]))
            idx += 1

        if "auto_switch" in fields and fields["auto_switch"] is not None:
            updates.append(f"auto_switch = ${idx}")
            params.append(bool(fields["auto_switch"]))
            idx += 1

        if "theme" in fields:
            updates.append(f"theme = ${idx}::jsonb")
            params.append(_json.dumps(fields["theme"] or {}))
            idx += 1

        if not updates:
            return existing

        updates.append("updated_at = NOW()")
        params.append(existing["id"])
        row = await self.fetchrow(
            f"""
            UPDATE seasons
            SET {', '.join(updates)}
            WHERE id = ${idx}
            RETURNING id, slug, name, subtitle, description, season_number, status,
                      auto_switch, preset_key, start_date, end_date, is_active,
                      max_stars, free_track_type, pass_track_type, ultra_track_type,
                      pass_end_position, ultra_start_position, theme, created_at, updated_at
            """,
            *params,
        )
        return dict(row) if row else existing

    async def update_season(self, season_id: int, **fields: Any) -> dict[str, Any]:
        """Обновить управляемый сезон ExtraPass."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json

        existing = await self.get_season_by_id(season_id)
        if not existing:
            return {"error": "not_found"}

        updates: list[str] = []
        params: list[Any] = []
        idx = 1

        for key in (
            "slug", "name", "subtitle", "description", "status", "preset_key",
            "free_track_type", "pass_track_type", "ultra_track_type",
        ):
            if key in fields and fields[key] is not None:
                updates.append(f"{key} = ${idx}")
                params.append(str(fields[key]))
                idx += 1

        for key in ("season_number", "max_stars", "pass_end_position", "ultra_start_position"):
            if key in fields and fields[key] is not None:
                updates.append(f"{key} = ${idx}")
                params.append(int(fields[key]))
                idx += 1

        for key in ("start_date", "end_date"):
            if key in fields:
                updates.append(f"{key} = ${idx}::timestamptz")
                params.append(fields[key] or None)
                idx += 1

        if "auto_switch" in fields and fields["auto_switch"] is not None:
            updates.append(f"auto_switch = ${idx}")
            params.append(bool(fields["auto_switch"]))
            idx += 1

        if "is_active" in fields and fields["is_active"] is not None:
            updates.append(f"is_active = ${idx}")
            params.append(bool(fields["is_active"]))
            idx += 1

        if "theme" in fields:
            updates.append(f"theme = ${idx}::jsonb")
            params.append(_json.dumps(fields["theme"] or {}))
            idx += 1

        if not updates:
            return existing

        if fields.get("is_active") is True or fields.get("status") == "active":
            await self.execute(
                """
                UPDATE seasons
                SET is_active = FALSE,
                    status = CASE WHEN status = 'active' THEN 'archived' ELSE status END,
                    updated_at = NOW()
                WHERE id <> $1 AND is_active = TRUE
                """,
                season_id,
            )

        updates.append("updated_at = NOW()")
        params.append(season_id)
        row = await self.fetchrow(
            f"""
            UPDATE seasons
            SET {', '.join(updates)}
            WHERE id = ${idx}
            RETURNING id, slug, name, subtitle, description, season_number, status,
                      auto_switch, preset_key, start_date, end_date, is_active,
                      max_stars, free_track_type, pass_track_type, ultra_track_type,
                      pass_end_position, ultra_start_position, theme, created_at, updated_at
            """,
            *params,
        )
        return dict(row) if row else {"error": "update_failed"}

    async def create_season_draft(self, preset_key: str = "blank") -> dict[str, Any]:
        """Создать черновик нового сезона с уникальными ExtraPass track_type."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")

        max_number = await self.fetchval("SELECT COALESCE(MAX(season_number), MAX(id), 0) FROM seasons")
        season_number = int(max_number or 0) + 1
        last_end = await self.fetchval(
            """
            SELECT end_date
            FROM seasons
            ORDER BY season_number DESC, id DESC
            LIMIT 1
            """
        )
        start_date = last_end or datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=30)
        track_prefix = f"bp_s{season_number:02d}"

        row = await self.fetchrow(
            """
            INSERT INTO seasons (
                slug, name, subtitle, description, season_number, status,
                auto_switch, preset_key, start_date, end_date, is_active,
                max_stars, free_track_type, pass_track_type, ultra_track_type,
                pass_end_position, ultra_start_position, theme
            )
            VALUES (
                $1, $2, $3, $4, $5, 'draft',
                TRUE, $6, $7, $8, FALSE,
                45, $9, $10, $11,
                40, 41, '{}'::jsonb
            )
            RETURNING id, slug, name, subtitle, description, season_number, status,
                      auto_switch, preset_key, start_date, end_date, is_active,
                      max_stars, free_track_type, pass_track_type, ultra_track_type,
                      pass_end_position, ultra_start_position, theme, created_at, updated_at
            """,
            f"season-{season_number}",
            f"Сезон {season_number}",
            "45 этапов | автосмена по расписанию",
            "",
            season_number,
            preset_key,
            start_date,
            end_date,
            f"{track_prefix}_free",
            f"{track_prefix}_premium",
            f"{track_prefix}_ultra",
        )
        season = dict(row)

        if preset_key in {"copy_current", "balanced_45"}:
            if preset_key == "copy_current":
                source = await self.get_active_season() or DEFAULT_EXTRA_PASS_SEASON
                source_types = {
                    "free": source.get("free_track_type") or DEFAULT_EXTRA_PASS_SEASON["free_track_type"],
                    "premium": source.get("pass_track_type") or DEFAULT_EXTRA_PASS_SEASON["pass_track_type"],
                    "ultra": source.get("ultra_track_type") or DEFAULT_EXTRA_PASS_SEASON["ultra_track_type"],
                }
            else:
                source_types = {
                    "free": DEFAULT_EXTRA_PASS_SEASON["free_track_type"],
                    "premium": DEFAULT_EXTRA_PASS_SEASON["pass_track_type"],
                    "ultra": DEFAULT_EXTRA_PASS_SEASON["ultra_track_type"],
                }
            target_types = {
                "free": season["free_track_type"],
                "premium": season["pass_track_type"],
                "ultra": season["ultra_track_type"],
            }
            for lane, source_type in source_types.items():
                for entry in await self.get_reward_tracks(str(source_type)):
                    await self.create_reward_track(
                        track_type=str(target_types[lane]),
                        position=int(entry["position"]),
                        reward_type=str(entry["reward_type"]),
                        reward_amount=int(entry["reward_amount"]),
                        reward_meta=entry.get("reward_meta"),
                        extra_pass_required=lane != "free",
                    )

        return season

    async def clear_reward_tracks(self, track_types: list[str]) -> int:
        """Скрыть все активные награды выбранных track_type."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        if not track_types:
            return 0
        status = await self.execute(
            """
            UPDATE reward_tracks
            SET is_active = FALSE,
                updated_at = NOW()
            WHERE track_type = ANY($1::text[])
            """,
            list(track_types),
        )
        try:
            return int(str(status).split()[-1])
        except Exception:
            return 0

    async def get_reward_track_counts_by_type(self, track_types: list[str] | None = None) -> dict[str, int]:
        """Посчитать активные награды по track_type."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        if track_types:
            rows = await self.fetch(
                """
                SELECT track_type, COUNT(*)::int AS count
                FROM reward_tracks
                WHERE is_active = TRUE AND track_type = ANY($1::text[])
                GROUP BY track_type
                """,
                list(track_types),
            )
        else:
            rows = await self.fetch(
                """
                SELECT track_type, COUNT(*)::int AS count
                FROM reward_tracks
                WHERE is_active = TRUE
                GROUP BY track_type
                """
            )
        return {str(row["track_type"]): int(row["count"]) for row in rows}

    async def get_claimed_rewards(self, user_id: int, track_type: str) -> set[int]:
        """Множество позиций, которые игрок уже забрал."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            "SELECT position FROM claimed_rewards WHERE user_id = $1 AND track_type = $2",
            user_id, track_type,
        )
        return {row["position"] for row in rows}

    async def claim_reward(self, user_id: int, track_type: str, position: int) -> bool:
        """Записать получение награды. False если уже была получена."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            await self.execute(
                """
                INSERT INTO claimed_rewards (user_id, track_type, position)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, track_type, position) DO NOTHING
                """,
                user_id, track_type, position,
            )
            return True
        except Exception:
            return False

    async def update_user_stars(self, user_id: int, delta: int) -> int:
        """Обновить звёзды пользователя. Возвращает новое значение."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        result = await self.fetchrow(
            """
            UPDATE users
            SET stars = GREATEST(0, COALESCE(stars, 0) + $1),
                updated_at = NOW()
            WHERE user_id = $2
            RETURNING stars
            """,
            delta, user_id,
        )
        return result["stars"] if result else 0

    async def increment_user_keys(self, user_id: int, amount: int = 1) -> int:
        """Увеличить ключи пользователя. Возвращает новое значение."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        result = await self.fetchrow(
            """
            UPDATE users
            SET keys = COALESCE(keys, 0) + $1,
                updated_at = NOW()
            WHERE user_id = $2
            RETURNING keys
            """,
            amount, user_id,
        )
        return result["keys"] if result else 0

    async def increment_win_counter_and_maybe_grant_key(
        self,
        user_id: int,
        wins_for_case: int,
    ) -> dict[str, Any]:
        """Atomically update wins_since_last_case and grant one key when threshold is reached."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        wins_for_case = max(1, int(wins_for_case or 1))
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                settings = await conn.fetchrow(
                    """
                    SELECT wins_since_last_case
                    FROM user_settings
                    WHERE user_id = $1
                    FOR UPDATE
                    """,
                    user_id,
                )
                if not settings:
                    await conn.execute(
                        "INSERT INTO user_settings (user_id, wins_since_last_case) VALUES ($1, 0) ON CONFLICT (user_id) DO NOTHING",
                        user_id,
                    )
                    settings = await conn.fetchrow(
                        "SELECT wins_since_last_case FROM user_settings WHERE user_id = $1 FOR UPDATE",
                        user_id,
                    )
                current_wins = int(settings["wins_since_last_case"] if settings else 0) + 1
                granted_key = current_wins >= wins_for_case
                if granted_key:
                    await conn.execute(
                        "UPDATE user_settings SET wins_since_last_case = 0, updated_at = NOW() WHERE user_id = $1",
                        user_id,
                    )
                    key_row = await conn.fetchrow(
                        """
                        UPDATE users
                        SET keys = COALESCE(keys, 0) + 1,
                            updated_at = NOW()
                        WHERE user_id = $1
                        RETURNING keys
                        """,
                        user_id,
                    )
                    return {
                        "wins_since_last_case": 0,
                        "current_wins": current_wins,
                        "wins_for_case": wins_for_case,
                        "granted_key": True,
                        "keys": key_row["keys"] if key_row else 0,
                    }
                await conn.execute(
                    "UPDATE user_settings SET wins_since_last_case = $1, updated_at = NOW() WHERE user_id = $2",
                    current_wins,
                    user_id,
                )
                return {
                    "wins_since_last_case": current_wins,
                    "current_wins": current_wins,
                    "wins_for_case": wins_for_case,
                    "granted_key": False,
                    "keys": None,
                }

    async def decrement_user_keys(self, user_id: int, amount: int = 1) -> int:
        """Уменьшить ключи пользователя. Возвращает новое значение."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        result = await self.fetchrow(
            """
            UPDATE users
            SET keys = GREATEST(0, COALESCE(keys, 0) - $1),
                updated_at = NOW()
            WHERE user_id = $2
            RETURNING keys
            """,
            amount, user_id,
        )
        return result["keys"] if result else 0

    async def get_random_cards_by_rarities(self, rarities: list[str], limit: int = 1) -> list[dict[str, Any]]:
        """Получить случайные карты из указанных редкостей без limited и start."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        rows = await self.fetch(
            """
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, simplified_levelup
            FROM cards
            WHERE rarity = ANY($1)
              AND card_type = 'warrior'
            ORDER BY RANDOM()
            LIMIT $2
            """,
            rarities, limit,
        )
        return [dict(row) for row in rows]

    async def add_card_to_user(self, user_id: int, card_id: int) -> dict[str, Any]:
        """Выдать карту игроку. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        # Проверяем существование карты
        card_exists = await self.fetchval("SELECT 1 FROM cards WHERE id = $1", card_id)
        if not card_exists:
            return {"success": False, "error": "card_not_found"}

        try:
            await self.execute(
                """
                INSERT INTO user_cards (user_id, card_id, level, particles)
                VALUES ($1, $2, 1, 0)
                ON CONFLICT (user_id, card_id) DO NOTHING
                """,
                user_id, card_id
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def grant_start_cards(self, user_id: int) -> dict[str, Any]:
        """Выдать игроку все карты редкости start без создания дубликатов."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            inserted = await self.fetch(
                """
                INSERT INTO user_cards (user_id, card_id, level, particles)
                SELECT $1, id, 1, 0
                FROM cards
                WHERE rarity = 'start'
                ON CONFLICT (user_id, card_id) DO NOTHING
                RETURNING card_id
                """,
                user_id,
            )
            return {"success": True, "added": len(inserted)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def remove_card_from_user(self, user_id: int, card_id: int) -> dict[str, Any]:
        """Убрать карту у игрока. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            result = await self.execute(
                "DELETE FROM user_cards WHERE user_id = $1 AND card_id = $2",
                user_id, card_id
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def has_user_card(self, user_id: int, card_id: int) -> bool:
        """Проверить, есть ли у пользователя карта."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        exists = await self.fetchval(
            "SELECT 1 FROM user_cards WHERE user_id = $1 AND card_id = $2",
            user_id, card_id
        )
        return exists is not None

    def get_card_max_level(self, card_obj: Any) -> int:
        return get_card_max_level(card_obj)

    def get_upgrade_cost(self, rarity: str, level: int, simplified_levelup: bool = False) -> dict[str, int]:
        cost_level = get_upgrade_cost_level({"simplified_levelup": simplified_levelup}, level)
        if cost_level is None:
            return {"particles": 0, "coins": 0}
        return {
            "particles": self.calculate_upgrade_particles(rarity, cost_level),
            "coins": self.calculate_upgrade_coins(rarity, cost_level),
        }

    def _upgrade_cost_fields(self, rarity: str, level: int, simplified_levelup: bool = False) -> dict[str, int]:
        cost = self.get_upgrade_cost(rarity, level, simplified_levelup)
        return {
            "upgrade_particles_required": cost["particles"],
            "upgrade_coins_required": cost["coins"],
        }

    def calculate_upgrade_particles(self, rarity: str, level: int) -> int:
        """Рассчитать необходимое количество частиц для улучшения карты.
        
        Args:
            rarity: Редкость карты (common, rare, start, superrare, epic, legendary, mythic, divine, limited)
            level: Текущий уровень карты (1-9, переход на level+1)
            
        Returns:
            Количество частиц, необходимое для улучшения до следующего уровня
        """
        import math
        
        # Базовые значения частиц для каждого перехода уровня (для Обычной карты)
        base_particles_by_level = {
            1: 5,    # 1 → 2
            2: 10,   # 2 → 3
            3: 20,   # 3 → 4
            4: 40,   # 4 → 5
            5: 80,   # 5 → 6
            6: 160,  # 6 → 7
            7: 320,  # 7 → 8
            8: 640,  # 8 → 9
            9: 2500  # 9 → 10 (ценовой обрыв)
        }
        
        # Множители частиц по редкостям
        rarity_multipliers = {
            "common": 1.0,
            "rare": 1.3,
            "start": 1.4,
            "superrare": 1.6,
            "epic": 2.0,
            "legendary": 2.5,
            "mythic": 4.0,
            "divine": 3.0,
            "limited": 3.5
        }
        
        # Получаем базовое значение для текущего уровня
        base_particles = base_particles_by_level.get(level, 5)
        
        # Получаем множитель редкости
        rarity_mult = rarity_multipliers.get(rarity, 1.0)
        
        # Вычисляем финальное количество частиц (округление вверх)
        required_particles = math.ceil(base_particles * rarity_mult)
        
        return int(required_particles)

    def calculate_upgrade_coins(self, rarity: str, level: int) -> int:
        """Рассчитать необходимое количество монет для улучшения карты.
        
        Args:
            rarity: Редкость карты (common, rare, start, superrare, epic, legendary, mythic, divine, limited)
            level: Текущий уровень карты (1-9, переход на level+1)
            
        Returns:
            Количество монет, необходимое для улучшения до следующего уровня
        """
        import math
        
        # Базовые значения монет для каждого перехода уровня (для Обычной карты)
        base_coins_by_level = {
            1: 50,      # 1 → 2
            2: 150,     # 2 → 3
            3: 400,     # 3 → 4
            4: 900,     # 4 → 5
            5: 2000,    # 5 → 6
            6: 4500,    # 6 → 7
            7: 8000,    # 7 → 8
            8: 13000,   # 8 → 9
            9: 40000    # 9 → 10 (ценовой обрыв)
        }
        
        # Множители монет по редкостям
        # Смещение экономики: больше спроса на монеты у легендарок
        rarity_multipliers = {
            "common": 1.0,
            "rare": 1.2,
            "start": 1.3,
            "superrare": 1.5,
            "epic": 2.0,
            "legendary": 3.5,  # было 3.0
            "mythic": 4.0,
            "divine": 5.0,
            "limited": 6.0
        }
        
        # Получаем базовое значение для текущего уровня
        base_coins = base_coins_by_level.get(level, 50)
        
        # Получаем множитель редкости
        rarity_mult = rarity_multipliers.get(rarity, 1.0)
        
        # Вычисляем финальное количество монет (округление вверх)
        required_coins = math.ceil(base_coins * rarity_mult)
        
        return int(required_coins)

    async def add_particles_to_card(self, user_id: int, card_id: int, particles: int) -> dict[str, Any]:
        """Добавить частицы к карте. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        if particles <= 0:
            return {"success": False, "error": "invalid_particles"}

        try:
            await self.execute(
                """
                UPDATE user_cards
                SET particles = COALESCE(particles, 0) + $1
                WHERE user_id = $2 AND card_id = $3
                """,
                particles, user_id, card_id
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def upgrade_card(self, user_id: int, card_id: int) -> dict[str, Any]:
        """Улучшить карту. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            # Получаем информацию о карте пользователя и базовой карте
            user_card = await self.fetchrow(
                """
                SELECT uc.level,
                       uc.particles,
                       c.rarity,
                       c.power,
                       c.base_attack,
                       c.base_hp,
                       c.mana_cost,
                       c.mechanics,
                       c.card_type,
                       c.simplified_levelup
                FROM user_cards uc
                INNER JOIN cards c ON c.id = uc.card_id
                WHERE uc.user_id = $1 AND uc.card_id = $2
                """,
                user_id, card_id
            )
            
            if not user_card:
                return {"success": False, "error": "card_not_found"}
            
            simplified_levelup = bool(user_card.get("simplified_levelup", False))
            max_level = self.get_card_max_level({"simplified_levelup": simplified_levelup})
            current_level = min(max(1, int(user_card["level"] or 1)), max_level)
            
            # Проверяем максимальный уровень
            if current_level >= max_level:
                return {"success": False, "error": "max_level_reached"}
            
            current_particles = user_card["particles"] or 0
            rarity = user_card["rarity"]
            base_power = user_card["power"]
            base_attack = user_card.get("base_attack") or 0
            base_hp = user_card.get("base_hp") or 0
            mana_cost = user_card.get("mana_cost") or 0
            mechanics = _normalize_mechanics(user_card.get("mechanics"))
            card_type = user_card.get("card_type") or "warrior"
            
            # Рассчитываем необходимое количество частиц и монет
            cost = self.get_upgrade_cost(rarity, current_level, simplified_levelup)
            required_particles = cost["particles"]
            required_coins = cost["coins"]
            
            # Получаем текущее количество монет пользователя
            user_coins = await self.fetchval(
                "SELECT coins FROM users WHERE user_id = $1",
                user_id
            ) or 0
            
            # Проверяем частицы
            if current_particles < required_particles:
                return {
                    "success": False,
                    "error": "insufficient_particles",
                    "required": required_particles,
                    "current": current_particles
                }
            
            # Проверяем монеты
            if user_coins < required_coins:
                return {
                    "success": False,
                    "error": "insufficient_coins",
                    "required": required_coins,
                    "current": user_coins
                }
            
            # Улучшаем карту: увеличиваем уровень, сбрасываем частицы, увеличиваем мощность
            new_level = min(current_level + 1, max_level)
            
            # Формула мощности: Power(n) = Power_base × 1.10^(n-1)
            import math
            power_multiplier = math.pow(1.10, new_level - 1)
            new_power = int(base_power * power_multiplier)
            card_base_payload = {
                "rarity": rarity,
                "base_attack": base_attack,
                "base_hp": base_hp,
                "mana_cost": mana_cost,
                "mechanics": mechanics,
                "card_type": card_type,
                "simplified_levelup": simplified_levelup,
            }
            old_stats = calculate_card_stats(card_base_payload, level=current_level)
            new_stats = calculate_card_stats(card_base_payload, level=new_level)
            
            # Выполняем обновление в транзакции
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Обновляем карту: повышаем уровень и списываем частицы
                    await conn.execute(
                        """
                        UPDATE user_cards
                        SET level = $1, particles = particles - $2
                        WHERE user_id = $3 AND card_id = $4
                        """,
                        new_level, required_particles, user_id, card_id
                    )
                    
                    # Списываем монеты
                    await conn.execute(
                        """
                        UPDATE users
                        SET coins = coins - $1
                        WHERE user_id = $2
                        """,
                        required_coins, user_id
                    )
            
            try:
                await self.award_squad_cbrp(
                    user_id,
                    "card_upgrade",
                    source_id=f"card_upgrade:{user_id}:{card_id}:{new_level}",
                    metadata={"card_id": card_id, "new_level": new_level, "rarity": rarity},
                )
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to award squad CBRP for card upgrade: user_id=%s card_id=%s level=%s",
                    user_id,
                    card_id,
                    new_level,
                    exc_info=True,
                )

            return {
                "success": True,
                "new_level": new_level,
                "new_power": new_power,
                "old_power": int(base_power * math.pow(1.10, current_level - 1)),
                "power_increase": new_power - int(base_power * math.pow(1.10, current_level - 1)),
                "particles_spent": required_particles,
                "coins_spent": required_coins,
                # Новая мета-боевка: возвращаем актуальные статы
                "new_attack": new_stats["attack"],
                "new_hp": new_stats["hp"],
                "old_attack": old_stats["attack"],
                "old_hp": old_stats["hp"],
                "mana_cost": new_stats["mana"],
                "mechanics": new_stats["mechanics"],
                "max_level": max_level,
                "is_max_level": new_level >= max_level,
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка улучшения карты: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_all_user_cards(self, user_id: int) -> dict[str, Any]:
        """Удалить все карты у конкретного пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            await self.execute(
                "DELETE FROM user_cards WHERE user_id = $1",
                user_id
            )
            return {"success": True, "message": "Все карты удалены из коллекции пользователя"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def add_all_cards_to_user(self, user_id: int) -> dict[str, Any]:
        """Добавить все существующие карты в коллекцию пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            # Получаем количество карт до добавления
            count_before = await self.fetchval(
                "SELECT COUNT(*) FROM user_cards WHERE user_id = $1",
                user_id
            ) or 0
            
            # Получаем все ID карт из БД
            all_cards = await self.fetch("SELECT id FROM cards")
            
            if not all_cards:
                return {"success": True, "message": "Нет карт для добавления", "added": 0}
            
            # Добавляем все карты в коллекцию пользователя (используя ON CONFLICT для избежания дубликатов)
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for card in all_cards:
                        card_id = card["id"]
                        await conn.execute(
                            """
                            INSERT INTO user_cards (user_id, card_id)
                            VALUES ($1, $2)
                            ON CONFLICT (user_id, card_id) DO NOTHING
                            """,
                            user_id, card_id
                        )
            
            # Получаем количество карт после добавления
            count_after = await self.fetchval(
                "SELECT COUNT(*) FROM user_cards WHERE user_id = $1",
                user_id
            ) or 0
            
            added_count = count_after - count_before
            
            return {"success": True, "message": f"Добавлено карт в коллекцию: {added_count}", "added": int(added_count)}
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка добавления всех карт пользователю {user_id}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_all_cards(self) -> dict[str, Any]:
        """Удалить все карты и связи с пользователями (только для админа)."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            # Сначала удаляем связи с пользователями
            await self.execute("DELETE FROM user_cards")
            # Затем удаляем сами карты
            await self.execute("DELETE FROM cards")
            return {"success": True, "message": "Все карты удалены"}
        except Exception as e:
            return {"success": False, "error": str(e)}



    async def _ensure_deck_presets_table(self) -> bool:
        """Создать таблицу пресетов колод."""
        changed = False

        table_exists = await self.fetchval("SELECT to_regclass('public.deck_presets')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE deck_presets (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    preset_name TEXT NOT NULL DEFAULT 'Колода',
                    preset_number INT NOT NULL,
                    card_slot_1 BIGINT,
                    card_slot_2 BIGINT,
                    card_slot_3 BIGINT,
                    card_slot_4 BIGINT,
                    card_slot_5 BIGINT,
                    card_slot_6 BIGINT,
                    card_slot_7 BIGINT,
                    card_slot_8 BIGINT,
                    card_slot_9 BIGINT,
                    used_by_bot BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(user_id, preset_number)
                )
                """
            )
            changed = True

        columns = await self._get_columns("deck_presets")
        changed |= await self._add_column_if_missing("deck_presets", columns, "user_id BIGINT NOT NULL DEFAULT 0")
        changed |= await self._add_column_if_missing("deck_presets", columns, "preset_name TEXT NOT NULL DEFAULT 'Колода'")
        changed |= await self._add_column_if_missing("deck_presets", columns, "preset_number INT NOT NULL DEFAULT 1")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_1 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_2 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_3 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_4 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_5 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_6 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_7 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_8 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "card_slot_9 BIGINT")
        changed |= await self._add_column_if_missing("deck_presets", columns, "used_by_bot BOOLEAN NOT NULL DEFAULT FALSE")
        changed |= await self._add_column_if_missing("deck_presets", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
        changed |= await self._add_column_if_missing("deck_presets", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")


        # Создаем индекс для быстрого поиска по user_id
        index_exists = await self.fetchval(
            """
            SELECT 1 FROM pg_indexes 
            WHERE tablename = 'deck_presets' AND indexname = 'deck_presets_user_id_idx'
            """
        )
        if not index_exists:
            await self.execute("CREATE INDEX deck_presets_user_id_idx ON deck_presets(user_id)")
            changed = True

        # Создаем уникальный индекс для user_id + preset_number
        unique_constraint_exists = await self._constraint_exists("deck_presets", "deck_presets_user_id_preset_number_key")
        if not unique_constraint_exists:
            await self.execute(
                """
                ALTER TABLE deck_presets
                ADD CONSTRAINT deck_presets_user_id_preset_number_key
                UNIQUE (user_id, preset_number)
                """
            )
            changed = True

        return changed

    async def _ensure_user_has_default_presets(self, user_id: int) -> None:
        """Убедиться, что у пользователя есть пресет по умолчанию."""
        existing = await self.fetchval(
            "SELECT 1 FROM deck_presets WHERE user_id = $1 LIMIT 1",
            user_id
        )
        
        if not existing:
            # Создаем пресет "Моя колода" по умолчанию
            await self.execute(
                """
                INSERT INTO deck_presets (user_id, preset_name, preset_number)
                VALUES ($1, 'Моя колода', 1)
                ON CONFLICT (user_id, preset_number) DO NOTHING
                """,
                user_id
            )

    async def _ensure_admin_account_actions_table(self) -> bool:
        """Ensure admin_account_actions table exists."""
        changed = False
        table_exists = await self.fetchval(
            "SELECT to_regclass('public.admin_account_actions')"
        )
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE admin_account_actions (
                    id BIGSERIAL PRIMARY KEY,
                    admin_user_id BIGINT NOT NULL,
                    target_user_id BIGINT NOT NULL,
                    action_type TEXT NOT NULL,
                    reason TEXT,
                    payload JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            changed = True

        existing = await self._get_columns("admin_account_actions")

        # Note: We already created all columns above, but ensure indices exist
        idx_prefix = "admin_account_actions"
        for col, suffix in [
            ("target_user_id", "target"),
            ("admin_user_id", "admin"),
            ("action_type", "type"),
            ("created_at", "created"),
        ]:
            idx_name = f"idx_{idx_prefix}_{suffix}"
            exists_idx = await self.fetchval(
                "SELECT 1 FROM pg_indexes WHERE indexname = $1", idx_name
            )
            if not exists_idx:
                await self.execute(
                    f"CREATE INDEX {idx_name} ON admin_account_actions ({col})"
                )
                changed = True

        return changed

    async def _ensure_cosmetic_tables(self) -> bool:
        """Создать таблицы косметики, seed стартовых аватарок, мигрировать пользователей."""
        changed = False

        # ── cosmetic_items: каталог всех предметов ──
        if not await self.fetchval("SELECT to_regclass('public.cosmetic_items')"):
            await self.execute("""
                CREATE TABLE cosmetic_items (
                    id          SERIAL PRIMARY KEY,
                    slug        TEXT UNIQUE NOT NULL,
                    item_type   TEXT NOT NULL,
                    class       TEXT NOT NULL DEFAULT 'starter',
                    name        TEXT NOT NULL,
                    asset_path  TEXT,
                    media_type  TEXT NOT NULL DEFAULT 'image',
                    has_sound   BOOLEAN NOT NULL DEFAULT FALSE,
                    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
                    sort_order  INT NOT NULL DEFAULT 0,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            changed = True

        # ── user_cosmetics: что пользователь получил ──
        if not await self.fetchval("SELECT to_regclass('public.user_cosmetics')"):
            await self.execute("""
                CREATE TABLE user_cosmetics (
                    id          SERIAL PRIMARY KEY,
                    user_id     BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    cosmetic_id INT NOT NULL REFERENCES cosmetic_items(id),
                    acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    source      TEXT NOT NULL DEFAULT 'grant',
                    UNIQUE(user_id, cosmetic_id)
                )
            """)
            await self.execute(
                "CREATE INDEX user_cosmetics_user_id_idx ON user_cosmetics(user_id)"
            )
            changed = True

        # ── user_equipped_cosmetics: что сейчас надето ──
        if not await self.fetchval("SELECT to_regclass('public.user_equipped_cosmetics')"):
            await self.execute("""
                CREATE TABLE user_equipped_cosmetics (
                    user_id     BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    item_type   TEXT NOT NULL,
                    cosmetic_id INT NOT NULL REFERENCES cosmetic_items(id),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, item_type)
                )
            """)
            changed = True

        # ── seed стартовых аватарок ──
        await self.execute("""
            INSERT INTO cosmetic_items (slug, item_type, class, name, asset_path, media_type, sort_order) VALUES
                ('avatar_1', 'avatar', 'starter', 'Страж арены',    '/DesignAssets/PlayerCosmetics/Avatars/1.png', 'image', 1),
                ('avatar_2', 'avatar', 'starter', 'Охотник теней',  '/DesignAssets/PlayerCosmetics/Avatars/2.png', 'image', 2),
                ('avatar_3', 'avatar', 'starter', 'Железный кулак', '/DesignAssets/PlayerCosmetics/Avatars/3.png', 'image', 3),
                ('avatar_4', 'avatar', 'starter', 'Пустошь',        '/DesignAssets/PlayerCosmetics/Avatars/4.png', 'image', 4),
                ('avatar_5', 'avatar', 'starter', 'Воин духа',      '/DesignAssets/PlayerCosmetics/Avatars/5.png', 'image', 5),
                ('avatar_6', 'avatar', 'starter', 'Крылатый',       '/DesignAssets/PlayerCosmetics/Avatars/6.png', 'image', 6)
            ON CONFLICT (slug) DO NOTHING
        """)

        # ── seed фонов профиля ──
        await self.execute("""
            INSERT INTO cosmetic_items (slug, item_type, class, name, asset_path, media_type, sort_order) VALUES
                ('bg_7',  'profile_background', 'starter', 'Тёмная арена',    '/DesignAssets/PlayerCosmetics/Background/7.png',  'image', 1),
                ('bg_8',  'profile_background', 'starter', 'Пепельный свод',  '/DesignAssets/PlayerCosmetics/Background/8.png',  'image', 2),
                ('bg_9',  'profile_background', 'rare',    'Грозовой фронт',  '/DesignAssets/PlayerCosmetics/Background/9.png',  'image', 3),
                ('bg_10', 'profile_background', 'rare',    'Алая заря',       '/DesignAssets/PlayerCosmetics/Background/10.png', 'image', 4),
                ('bg_11', 'profile_background', 'epic',    'Кристальные своды','/DesignAssets/PlayerCosmetics/Background/11.png','image', 5),
                ('bg_12', 'profile_background', 'epic',    'Пустота',         '/DesignAssets/PlayerCosmetics/Background/12.png', 'image', 6)
            ON CONFLICT (slug) DO NOTHING
        """)

        # ── seed стартового титула ──
        await self.execute("""
            INSERT INTO cosmetic_items (slug, item_type, class, name, asset_path, media_type, sort_order) VALUES
                ('title_1', 'title', 'starter', 'Игрок ExtraArena', NULL, 'image', 1)
            ON CONFLICT (slug) DO NOTHING
        """)
        await self.execute("""
            INSERT INTO cosmetic_items (slug, item_type, class, name, asset_path, media_type, sort_order) VALUES
                ('title_squadmate',       'title', 'common', 'Сквадмейт', NULL, 'image', 20),
                ('title_squad_business',  'title', 'rare',   'Сквад это бизнес', NULL, 'image', 21),
                ('title_cbrp_hunter',     'title', 'epic',   'CBRP Hunter', NULL, 'image', 22)
            ON CONFLICT (slug) DO UPDATE
            SET item_type = EXCLUDED.item_type,
                class = EXCLUDED.class,
                name = EXCLUDED.name,
                media_type = EXCLUDED.media_type,
                sort_order = EXCLUDED.sort_order,
                is_active = TRUE
        """)

        # ── миграция существующих пользователей: выдать все starter-предметы ──
        await self.execute("""
            INSERT INTO user_cosmetics (user_id, cosmetic_id, source)
            SELECT u.user_id, ci.id, 'grant'
            FROM users u
            CROSS JOIN cosmetic_items ci
            WHERE ci.class = 'starter'
              AND COALESCE(u.is_bot, FALSE) = FALSE
            ON CONFLICT (user_id, cosmetic_id) DO NOTHING
        """)

        # ── миграция: надеть avatar_1 всем у кого нет ни одного equipped avatar ──
        await self.execute("""
            INSERT INTO user_equipped_cosmetics (user_id, item_type, cosmetic_id)
            SELECT u.user_id, 'avatar', ci.id
            FROM users u
            CROSS JOIN cosmetic_items ci
            WHERE ci.slug = 'avatar_1'
              AND COALESCE(u.is_bot, FALSE) = FALSE
              AND NOT EXISTS (
                  SELECT 1 FROM user_equipped_cosmetics uec
                  WHERE uec.user_id = u.user_id AND uec.item_type = 'avatar'
              )
            ON CONFLICT (user_id, item_type) DO NOTHING
        """)

        return changed

    async def grant_starter_cosmetics(self, user_id: int) -> None:
        """Выдать все starter-предметы пользователю и надеть avatar_1 если пусто."""
        # Выдать все starter-предметы
        await self.execute("""
            INSERT INTO user_cosmetics (user_id, cosmetic_id, source)
            SELECT $1, id, 'grant'
            FROM cosmetic_items
            WHERE class = 'starter' AND is_active = TRUE
            ON CONFLICT (user_id, cosmetic_id) DO NOTHING
        """, user_id)

        # Надеть avatar_1 если нет equipped avatar
        await self.execute("""
            INSERT INTO user_equipped_cosmetics (user_id, item_type, cosmetic_id)
            SELECT $1, 'avatar', id
            FROM cosmetic_items
            WHERE slug = 'avatar_1'
            ON CONFLICT (user_id, item_type) DO NOTHING
        """, user_id)

    async def get_user_cosmetics(self, user_id: int) -> dict:
        """Вернуть все предметы пользователя + текущий equipped по каждому типу."""
        owned_rows = await self.fetch("""
            SELECT ci.id, ci.slug, ci.item_type, ci.class, ci.name,
                   ci.asset_path, ci.media_type, ci.has_sound, ci.sort_order
            FROM user_cosmetics uc
            JOIN cosmetic_items ci ON ci.id = uc.cosmetic_id
            WHERE uc.user_id = $1 AND ci.is_active = TRUE
            ORDER BY ci.item_type, ci.sort_order
        """, user_id)

        equipped_rows = await self.fetch("""
            SELECT uec.item_type, ci.id, ci.slug, ci.class, ci.name,
                   ci.asset_path, ci.media_type, ci.has_sound
            FROM user_equipped_cosmetics uec
            JOIN cosmetic_items ci ON ci.id = uec.cosmetic_id
            WHERE uec.user_id = $1
        """, user_id)

        items = [dict(r) for r in owned_rows]
        equipped = {r["item_type"]: dict(r) for r in equipped_rows}

        return {"items": items, "equipped": equipped}

    async def equip_cosmetic(self, user_id: int, cosmetic_id: int) -> dict:
        """Надеть предмет. Проверить, что пользователь им владеет."""
        # Проверка владения
        owned = await self.fetchval("""
            SELECT 1 FROM user_cosmetics
            WHERE user_id = $1 AND cosmetic_id = $2
        """, user_id, cosmetic_id)
        if not owned:
            return {"success": False, "error": "not_owned"}

        # Получить тип предмета
        item = await self.fetchrow("""
            SELECT id, item_type, slug, class, name, asset_path, media_type
            FROM cosmetic_items WHERE id = $1 AND is_active = TRUE
        """, cosmetic_id)
        if not item:
            return {"success": False, "error": "item_not_found"}

        # Надеть
        await self.execute("""
            INSERT INTO user_equipped_cosmetics (user_id, item_type, cosmetic_id, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id, item_type) DO UPDATE
            SET cosmetic_id = EXCLUDED.cosmetic_id, updated_at = NOW()
        """, user_id, item["item_type"], cosmetic_id)

        return {"success": True, "equipped": dict(item)}

    async def grant_cosmetic_by_slug(
        self,
        user_id: int,
        slug: str,
        *,
        source: str = "grant",
        auto_equip: bool = False,
    ) -> dict[str, Any]:
        """Выдать косметический предмет из каталога и опционально надеть его."""
        item = await self.fetchrow(
            """
            SELECT id, slug, item_type, class, name, asset_path, media_type
            FROM cosmetic_items
            WHERE slug = $1 AND is_active = TRUE
            """,
            slug,
        )
        if not item:
            raise ValueError("cosmetic_not_found")
        owned_before = await self.fetchval(
            """
            SELECT 1 FROM user_cosmetics
            WHERE user_id = $1 AND cosmetic_id = $2
            """,
            user_id,
            item["id"],
        )
        await self.execute(
            """
            INSERT INTO user_cosmetics (user_id, cosmetic_id, source)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, cosmetic_id) DO NOTHING
            """,
            user_id,
            item["id"],
            source,
        )
        equipped = None
        if auto_equip:
            equip_result = await self.equip_cosmetic(user_id, int(item["id"]))
            equipped = equip_result.get("equipped") if equip_result.get("success") else None
        return {"item": dict(item), "equipped": equipped, "acquired": not bool(owned_before)}

    async def get_equipped_title(self, user_id: int) -> Optional[dict]:
        """Вернуть надетый титул {name, class} или None."""
        row = await self.fetchrow("""
            SELECT ci.name, ci.class
            FROM user_equipped_cosmetics uec
            JOIN cosmetic_items ci ON ci.id = uec.cosmetic_id
            WHERE uec.user_id = $1 AND uec.item_type = 'title' AND ci.is_active = TRUE
        """, user_id)
        return dict(row) if row else None

    # ── BotFactory v2 reuse ───────────────────────────────────────────

    async def find_reusable_bots(
        self, player_trophies: int, exclude_user_ids: list[int] | None = None,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Find existing bots with trophies close to player for reuse."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        exclude = set(exclude_user_ids or [])
        window = max(50, round(player_trophies * 0.15))
        min_trophies = max(0, player_trophies - window)
        max_trophies = player_trophies + window

        rows = await self.fetch(
            """
            SELECT u.user_id, u.trophies, u.league, u.extra_pass,
                   COALESCE(p.custom_nickname, u.first_name, 'Бот') AS display_name,
                   p.img
            FROM users u
            LEFT JOIN profiles p ON p.user_id = u.user_id
            WHERE u.is_bot = TRUE
              AND u.trophies >= $1 AND u.trophies <= $2
              AND u.status = 'active'
            ORDER BY ABS(u.trophies - $3) ASC, RANDOM()
            LIMIT $4
            """,
            min_trophies, max_trophies, player_trophies, max_results,
        )

        result: list[dict[str, Any]] = []
        for row in rows:
            bid = int(row["user_id"])
            if bid in exclude:
                continue
            result.append({
                "user_id": bid,
                "trophies": int(row["trophies"]),
                "league": int(row.get("league", 1)),
                "extra_pass": row.get("extra_pass", "inactive"),
                "display_name": row["display_name"],
                "img": row.get("img"),
            })
        return result

    async def get_recent_bot_opponents(self, user_id: int, limit: int = 5) -> list[int]:
        """Get bot opponent IDs from battle_summary + legacy battle_results (includes null p1/p2 rows)."""
        if not self._pool:
            return []

        rows = await self.fetch(
            """
            SELECT opponent_id, created_at FROM (
                -- battle_summary (canonical)
                SELECT
                    CASE WHEN bs.p1_user_id = $1 THEN bs.p2_user_id ELSE bs.p1_user_id END AS opponent_id,
                    bs.created_at
                FROM battle_summary bs
                WHERE (bs.p1_user_id = $1 OR bs.p2_user_id = $1)
                  AND (
                      EXISTS (SELECT 1 FROM users u WHERE u.user_id = bs.p1_user_id AND u.is_bot = TRUE)
                      OR
                      EXISTS (SELECT 1 FROM users u WHERE u.user_id = bs.p2_user_id AND u.is_bot = TRUE)
                  )
                UNION ALL
                -- legacy battle_results with p1/p2
                SELECT
                    CASE WHEN br.p1_id = $1 THEN br.p2_id ELSE br.p1_id END AS opponent_id,
                    br.created_at
                FROM battle_results br
                WHERE (br.p1_id = $1 OR br.p2_id = $1)
                  AND br.match_id NOT IN (SELECT match_id FROM battle_summary)
                  AND (
                      EXISTS (SELECT 1 FROM users u WHERE u.user_id = br.p1_id AND u.is_bot = TRUE)
                      OR
                      EXISTS (SELECT 1 FROM users u WHERE u.user_id = br.p2_id AND u.is_bot = TRUE)
                  )
                UNION ALL
                -- legacy battle_results with null p1/p2 but filled winner/loser
                SELECT
                    CASE WHEN br.winner_id = $1 THEN br.loser_id ELSE br.winner_id END AS opponent_id,
                    br.created_at
                FROM battle_results br
                WHERE br.p1_id IS NULL AND br.p2_id IS NULL
                  AND (br.winner_id = $1 OR br.loser_id = $1)
                  AND br.match_id NOT IN (SELECT match_id FROM battle_summary)
                  AND (
                      EXISTS (SELECT 1 FROM users u WHERE u.user_id = br.winner_id AND u.is_bot = TRUE)
                      OR
                      EXISTS (SELECT 1 FROM users u WHERE u.user_id = br.loser_id AND u.is_bot = TRUE)
                  )
            ) sub
            WHERE opponent_id IS NOT NULL
            ORDER BY created_at DESC
            """,
            user_id,
        )
        result: list[int] = []
        seen: set[int] = set()
        for row in rows:
            opponent_id = row.get("opponent_id")
            if not opponent_id:
                continue
            opponent_id_int = int(opponent_id)
            if opponent_id_int in seen:
                continue
            seen.add(opponent_id_int)
            result.append(opponent_id_int)
            if len(result) >= limit:
                break
        return result

    async def get_bot_full_profile(self, bot_id: int) -> dict[str, Any] | None:
        """Get bot profile with deck and equipped cosmetics."""
        if not self._pool:
            return None

        row = await self.fetchrow(
            """
            SELECT u.user_id, u.trophies, u.league, u.extra_pass,
                   COALESCE(p.custom_nickname, u.first_name, 'Бот') AS display_name,
                   p.img, p.title AS legacy_title
            FROM users u
            LEFT JOIN profiles p ON p.user_id = u.user_id
            WHERE u.user_id = $1 AND u.is_bot = TRUE
            """,
            bot_id,
        )
        if not row:
            return None

        deck_ids = await self._get_bot_deck_ids(bot_id)
        equipped_cosmetics = await self._get_bot_equipped_cosmetics(bot_id)

        return {
            "user_id": int(row["user_id"]),
            "trophies": int(row["trophies"]),
            "league": int(row.get("league", 1)),
            "extra_pass": row.get("extra_pass", "inactive"),
            "display_name": row["display_name"] or f"Бот {bot_id}",
            "img": row.get("img"),
            "deck_ids": deck_ids,
            "equipped_cosmetics": equipped_cosmetics,
        }

    async def _get_bot_deck_ids(self, bot_id: int) -> list[int]:
        """Extract card IDs from bot's deck_presets."""
        row = await self.fetchrow(
            """
            SELECT card_slot_1, card_slot_2, card_slot_3, card_slot_4, card_slot_5,
                   card_slot_6, card_slot_7, card_slot_8, card_slot_9
            FROM deck_presets
            WHERE user_id = $1 AND used_by_bot = TRUE
            ORDER BY preset_number ASC LIMIT 1
            """,
            bot_id,
        )
        if not row:
            return []
        return [int(row[f"card_slot_{i}"]) for i in range(1, 10) if row.get(f"card_slot_{i}")]

    async def _get_bot_equipped_cosmetics(self, bot_id: int) -> dict[str, dict]:
        """Get bot's equipped cosmetics: avatar, title, background."""
        rows = await self.fetch(
            """
            SELECT uec.item_type, ci.id, ci.slug, ci.class, ci.name,
                   ci.asset_path, ci.media_type
            FROM user_equipped_cosmetics uec
            JOIN cosmetic_items ci ON ci.id = uec.cosmetic_id
            WHERE uec.user_id = $1
            """,
            bot_id,
        )
        return {r["item_type"]: dict(r) for r in rows}

    async def get_cosmetic_catalog_by_class(self) -> dict[str, list[dict[str, Any]]]:
        """Return cosmetic_items grouped by class."""
        if not self._pool:
            return {}

        rows = await self.fetch(
            """
            SELECT id, slug, item_type, class, name, asset_path, media_type
            FROM cosmetic_items
            WHERE is_active = TRUE
            ORDER BY item_type, sort_order
            """
        )
        catalog: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            cls = r["class"]
            if cls not in catalog:
                catalog[cls] = []
            catalog[cls].append(dict(r))
        return catalog

    async def grant_and_equip_bot_cosmetics(
        self, bot_id: int, avatar_cos_id: int | None,
        title_cos_id: int | None, bg_cos_id: int | None,
    ) -> None:
        """Grant cosmetics to bot and equip them."""
        if not self._pool:
            return

        for item_type, cos_id in [("avatar", avatar_cos_id), ("title", title_cos_id), ("profile_background", bg_cos_id)]:
            if cos_id is None:
                continue
            await self.execute(
                """
                INSERT INTO user_cosmetics (user_id, cosmetic_id, source)
                VALUES ($1, $2, 'bot_grant')
                ON CONFLICT (user_id, cosmetic_id) DO NOTHING
                """,
                bot_id, cos_id,
            )
            await self.execute(
                """
                INSERT INTO user_equipped_cosmetics (user_id, item_type, cosmetic_id, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (user_id, item_type) DO UPDATE
                SET cosmetic_id = EXCLUDED.cosmetic_id, updated_at = NOW()
                """,
                bot_id, item_type, cos_id,
            )

    async def get_bot_deck_from_donor(
        self, player_trophies: int, exclude_user_ids: list[int] | None = None,
    ) -> list[int] | None:
        """Copy a valid deck from a real donor close to player in trophies."""
        if not self._pool:
            return None

        exclude = set(exclude_user_ids or [])
        window = max(100, round(player_trophies * 0.15))
        min_trophies = max(0, player_trophies - window)
        max_trophies = player_trophies + window

        row = await self.fetchrow(
            """
            SELECT dp.card_slot_1, dp.card_slot_2, dp.card_slot_3, dp.card_slot_4,
                   dp.card_slot_5, dp.card_slot_6, dp.card_slot_7, dp.card_slot_8, dp.card_slot_9
            FROM deck_presets dp
            JOIN users u ON u.user_id = dp.user_id
            WHERE dp.preset_number IN (1, 2, 3)
              AND u.trophies >= $1 AND u.trophies <= $2
              AND COALESCE(u.is_bot, FALSE) = FALSE
              AND u.status = 'active'
              AND NOT (u.user_id = ANY($3::bigint[]))
            ORDER BY RANDOM()
            LIMIT 1
            """,
            min_trophies, max_trophies, list(exclude),
        )
        if not row:
            return None

        deck_ids: list[int] = []
        for i in range(1, 10):
            val = row.get(f"card_slot_{i}")
            if val is not None:
                deck_ids.append(int(val))

        if len(deck_ids) < 3:
            return None

        return deck_ids

    async def get_user_deck_presets(self, user_id: int) -> list[dict[str, Any]]:
        """Получить все пресеты колод пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT dp.id,
                   dp.preset_name,
                   dp.preset_number, 
                   dp.card_slot_1,
                   dp.card_slot_2,
                   dp.card_slot_3,
                   dp.card_slot_4,
                   dp.card_slot_5,
                   dp.card_slot_6,
                   dp.card_slot_7,
                   dp.card_slot_8,
                   dp.card_slot_9,
                   dp.used_by_bot,
                   dp.updated_at
            FROM deck_presets dp
            WHERE dp.user_id = $1
            ORDER BY dp.preset_number ASC
            """,
            user_id
        )
        presets: list[dict[str, Any]] = []
        for row in rows:
            preset = dict(row)
            # Собираем массив ID карт из отдельных слотов
            card_ids = []
            for i in range(1, DECK_SIZE + 1):
                val = preset.get(f"card_slot_{i}")
                if val is not None:
                    card_ids.append(int(val))
            preset["card_ids"] = card_ids
            
            # used_by_bot нужен, чтобы движок и матчмейкер могли выбрать правильную колоду бота.
            preset["used_by_bot"] = bool(preset.get("used_by_bot", False))
            presets.append(preset)

        # Обновляем локальный кеш пресетов, чтобы боевой движок мог синхронно
        # прочитать актуальную колоду через _load_deck_from_db_cache, не дергая БД повторно.
        try:
            deck_cache = getattr(self, "deck_presets_cache", None)
            if not isinstance(deck_cache, dict):
                deck_cache = {}
            deck_cache[user_id] = presets
            self.deck_presets_cache = deck_cache
        except Exception:
            # Кеш - вспомогательный; при любой ошибке просто пропускаем запись.
            pass
        return presets

    async def save_deck_preset(
        self,
        user_id: int,
        preset_number: int,
        preset_name: str,
        card_slots: list[int | None],
        used_by_bot: bool = False,
    ) -> dict[str, Any]:
        """Сохранить пресет колоды (DECK_SIZE слотов) с полной валидацией."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        if len(card_slots) != DECK_SIZE:
            return {"success": False, "error": "invalid_slots_count"}

        try:
            preset_name = (preset_name or "").strip()[:64]
        except Exception:
            preset_name = "Колода"

        # Patch 2: reject out-of-range preset_number
        if preset_number < 1 or preset_number > MAX_TOTAL_DECK_PRESETS:
            return {"success": False, "error": "invalid_preset_number", "message": f"Номер пресета должен быть от 1 до {MAX_TOTAL_DECK_PRESETS}"}

        # Patch 1 + H2: hero slot must be filled
        if card_slots[0] is None:
            return {"success": False, "error": "missing_hero", "message": "Выберите героя"}

        non_null_ids = [cid for cid in card_slots if cid is not None]

        # C2: no duplicate card ids
        if len(non_null_ids) != len(set(non_null_ids)):
            return {"success": False, "error": "duplicate_cards", "message": "Колода содержит одинаковые карты"}

        # Patch 5: explicit card existence check (before type/slot validation)
        if non_null_ids:
            rows = await self.fetch(
                "SELECT id, card_type FROM cards WHERE id = ANY($1::bigint[])",
                non_null_ids,
            )
            found_ids = {int(row["id"]) for row in rows}
            card_types = {int(row["id"]): (row["card_type"] or "warrior") for row in rows}
            missing_card_ids = [cid for cid in non_null_ids if cid not in found_ids]
            if missing_card_ids:
                return {"success": False, "error": "invalid_card_ids", "message": f"Карты не найдены: {missing_card_ids}", "card_ids": missing_card_ids}
        else:
            card_types = {}

        # C3: slot 0 must be hero; slots 1..8 must NOT be hero
        hero_id = card_slots[0]
        if card_types.get(hero_id) != "hero":
            return {"success": False, "error": "slot_0_must_be_hero", "message": "Первый слот должен содержать героя"}
        for idx in range(1, DECK_SIZE):
            cid = card_slots[idx]
            if cid is not None and card_types.get(cid) == "hero":
                return {"success": False, "error": "hero_in_warrior_slot", "message": f"Герой не может быть в слоте {idx + 1}"}

        # C1: verify card ownership (unless bot)
        if not used_by_bot:
            owned = await self.fetch(
                "SELECT card_id FROM user_cards WHERE user_id = $1 AND card_id = ANY($2::bigint[])",
                user_id, non_null_ids,
            )
            owned_set = {int(row["card_id"]) for row in owned}
            missing = [cid for cid in non_null_ids if cid not in owned_set]
            if missing:
                import logging
                logging.getLogger(__name__).warning(
                    "save_deck_preset: user %s tried to save unowned cards %s", user_id, missing
                )
                return {
                    "success": False,
                    "error": "unowned_cards",
                    "message": "Колода содержит карты, которых нет у вас",
                    "card_ids": missing,
                }

        try:
            existing = await self.fetchval(
                "SELECT id FROM deck_presets WHERE user_id = $1 AND preset_number = $2",
                user_id, preset_number
            )

            if not existing:
                # Patch 2: creating a new preset — enforce same limits as create_deck_preset
                current_count = await self.fetchval(
                    "SELECT COUNT(*) FROM deck_presets WHERE user_id = $1",
                    user_id,
                )
                if current_count >= MAX_TOTAL_DECK_PRESETS:
                    return {"success": False, "error": "max_presets_reached", "message": "Достигнут максимум пресетов"}

                extra_pass = await self.fetchval(
                    "SELECT extra_pass FROM users WHERE user_id = $1", user_id
                )
                has_pass = (extra_pass or "") in ("active", "ultra")
                if not has_pass and current_count >= MAX_FREE_DECK_PRESETS:
                    return {"success": False, "error": "extra_pass_required", "message": "Требуется ExtraPass для большего числа колод"}
                if not has_pass and preset_number > MAX_FREE_DECK_PRESETS:
                    return {"success": False, "error": "extra_pass_required", "message": "Требуется ExtraPass для этого номера пресета"}

            if existing:
                await self.execute(
                    """
                    UPDATE deck_presets
                    SET preset_name = $1,
                        card_slot_1 = $2,
                        card_slot_2 = $3,
                        card_slot_3 = $4,
                        card_slot_4 = $5,
                        card_slot_5 = $6,
                        card_slot_6 = $7,
                        card_slot_7 = $8,
                        card_slot_8 = $9,
                        card_slot_9 = $10,
                        used_by_bot = $11,
                        updated_at = NOW()
                    WHERE user_id = $12 AND preset_number = $13
                    """,
                    preset_name,
                    card_slots[0], card_slots[1], card_slots[2], card_slots[3],
                    card_slots[4], card_slots[5], card_slots[6], card_slots[7],
                    card_slots[8],
                    used_by_bot,
                    user_id, preset_number
                )
            else:
                await self.execute(
                    """
                    INSERT INTO deck_presets (user_id, preset_name, preset_number, 
                                             card_slot_1, card_slot_2, card_slot_3, card_slot_4,
                                             card_slot_5, card_slot_6, card_slot_7, card_slot_8, card_slot_9,
                                             used_by_bot, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW(), NOW())
                    """,
                    user_id, preset_name, preset_number,
                    card_slots[0], card_slots[1], card_slots[2], card_slots[3],
                    card_slots[4], card_slots[5], card_slots[6], card_slots[7], card_slots[8],
                    used_by_bot
                )

            # Patch 6: invalidate cache on mutation
            deck_cache = getattr(self, "deck_presets_cache", None)
            if isinstance(deck_cache, dict):
                deck_cache.pop(user_id, None)

            return {"success": True}
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка сохранения пресета: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_deck_preset(
        self,
        user_id: int,
        preset_name: str,
    ) -> dict[str, Any]:
        """Создать новый пресет колоды с проверкой лимитов. Возвращает номер созданного пресета."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            preset_name = (preset_name or "").strip()[:64] or "Новая колода"
        except Exception:
            preset_name = "Новая колода"

        # C6 + Patch 4: enforce preset count limits (ultra also counts as pass)
        rows = await self.fetch(
            "SELECT preset_number FROM deck_presets WHERE user_id = $1 ORDER BY preset_number",
            user_id,
        )
        existing_numbers = {int(row["preset_number"]) for row in rows}
        current_count = len(existing_numbers)

        extra_pass = await self.fetchval(
            "SELECT extra_pass FROM users WHERE user_id = $1", user_id
        )
        has_pass = (extra_pass or "") in ("active", "ultra")
        effective_limit = MAX_TOTAL_DECK_PRESETS if has_pass else MAX_FREE_DECK_PRESETS

        if current_count >= effective_limit:
            if has_pass:
                return {"success": False, "error": "max_presets_reached", "message": "Достигнут максимум пресетов"}
            return {"success": False, "error": "extra_pass_required", "message": "Требуется ExtraPass для большего числа колод"}

        # Patch 3: choose lowest free slot in 1..effective_limit
        new_preset_number = None
        for n in range(1, effective_limit + 1):
            if n not in existing_numbers:
                new_preset_number = n
                break
        if new_preset_number is None:
            return {"success": False, "error": "max_presets_reached", "message": "Достигнут максимум пресетов"}

        try:
            await self.execute(
                """
                INSERT INTO deck_presets (user_id, preset_name, preset_number)
                VALUES ($1, $2, $3)
                """,
                user_id, preset_name, new_preset_number
            )

            # Patch 6: invalidate cache on mutation
            deck_cache = getattr(self, "deck_presets_cache", None)
            if isinstance(deck_cache, dict):
                deck_cache.pop(user_id, None)

            return {"success": True, "preset_number": new_preset_number}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_deck_preset(
        self,
        user_id: int,
        preset_number: int,
    ) -> dict[str, Any]:
        """Удалить пресет колоды."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        # Не позволяем удалять, если останется меньше 1 пресета
        preset_count = await self.fetchval(
            "SELECT COUNT(*) FROM deck_presets WHERE user_id = $1",
            user_id
        )

        if preset_count <= 1:
            return {"success": False, "error": "min_presets_required", "message": "Нельзя удалить пресет. Минимум 1 пресет должно остаться."}

        try:
            deleted = await self.fetchrow(
                "DELETE FROM deck_presets WHERE user_id = $1 AND preset_number = $2 RETURNING id",
                user_id, preset_number
            )
            if not deleted:
                return {"success": False, "error": "preset_not_found", "message": "Пресет не найден"}

            # Fix 2: if deleting the current primary, clear it
            primary = await self.fetchval(
                "SELECT primary_deck FROM users WHERE user_id = $1", user_id
            )
            if primary == preset_number:
                await self.execute(
                    "UPDATE users SET primary_deck = NULL, updated_at = NOW() WHERE user_id = $1",
                    user_id,
                )

            # Patch 6: invalidate cache on mutation
            deck_cache = getattr(self, "deck_presets_cache", None)
            if isinstance(deck_cache, dict):
                deck_cache.pop(user_id, None)

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def rename_deck_preset(
        self,
        user_id: int,
        preset_number: int,
        new_name: str,
    ) -> dict[str, Any]:
        """Переименовать пресет колоды."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        new_name = (new_name or "").strip()[:64]
        if not new_name:
            return {"success": False, "error": "empty_name"}

        try:
            updated = await self.fetchrow(
                """
                UPDATE deck_presets
                SET preset_name = $1, updated_at = NOW()
                WHERE user_id = $2 AND preset_number = $3
                RETURNING id
                """,
                new_name, user_id, preset_number
            )
            if not updated:
                return {"success": False, "error": "preset_not_found", "message": "Пресет не найден"}

            # Patch 6: invalidate cache on mutation
            deck_cache = getattr(self, "deck_presets_cache", None)
            if isinstance(deck_cache, dict):
                deck_cache.pop(user_id, None)

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_columns(self, table: str) -> set[str]:
        rows = await self.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1
            """,
            table,
        )
        return {row["column_name"] for row in rows}

    async def _add_column_if_missing(
        self, table: str, existing_columns: set[str], column_definition: str
    ) -> bool:
        column_name = column_definition.split()[0]
        if column_name in existing_columns:
            return False
        await self.execute(f"ALTER TABLE {table} ADD COLUMN {column_definition}")
        return True

    async def _constraint_exists(self, table: str, constraint: str) -> bool:
        result = await self.fetchval(
            """
            SELECT 1
            FROM information_schema.table_constraints
            WHERE table_schema = 'public'
              AND table_name = $1
              AND constraint_name = $2
            """,
            table,
            constraint,
        )
        return result is not None







    # ── Analytics: write methods ──

    async def track_economy_event(
        self,
        user_id: int,
        event_type: str,
        resource: str,
        amount: Any,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._pool:
            return
        try:
            safe_amount = float(amount) if amount is not None else 0.0
            safe_meta = _json_safe(metadata) if metadata else {}
            await self.execute(
                """
                INSERT INTO economy_events (user_id, event_type, resource, amount, source, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                user_id,
                event_type,
                resource,
                safe_amount,
                source,
                json.dumps(safe_meta, ensure_ascii=False),
            )
        except Exception:
            import logging
            logging.getLogger(__name__).error(
                "track_economy_event failed: user_id=%s event_type=%s resource=%s",
                user_id, event_type, resource, exc_info=True,
            )

    async def start_user_session(
        self,
        user_id: int,
        session_id: str,
        source: str = "webapp",
    ) -> None:
        if not self._pool:
            return
        try:
            await self.execute(
                """
                INSERT INTO user_sessions (user_id, session_id, source)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                """,
                user_id, session_id, source,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).error(
                "start_user_session failed: user_id=%s session_id=%s",
                user_id, session_id, exc_info=True,
            )

    async def finish_user_session(
        self,
        session_id: str,
        screens_visited: Optional[list] = None,
        battles_played: int = 0,
        cases_opened: int = 0,
    ) -> None:
        if not self._pool:
            return
        try:
            await self.execute(
                """
                UPDATE user_sessions
                SET ended_at = NOW(),
                    duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INT,
                    screens_visited = $1::jsonb,
                    battles_played = $2,
                    cases_opened = $3,
                    updated_at = NOW()
                WHERE session_id = $4
                  AND ended_at IS NULL
                """,
                json.dumps(screens_visited or [], ensure_ascii=False),
                battles_played,
                cases_opened,
                session_id,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).error(
                "finish_user_session failed: session_id=%s", session_id, exc_info=True,
            )

    async def update_user_session(
        self,
        session_id: str,
        screens_visited: Optional[list] = None,
        battles_played: int = 0,
        cases_opened: int = 0,
    ) -> None:
        if not self._pool:
            return
        try:
            await self.execute(
                """
                UPDATE user_sessions
                SET screens_visited = $1::jsonb,
                    battles_played = GREATEST(battles_played, $2),
                    cases_opened = GREATEST(cases_opened, $3),
                    duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INT,
                    updated_at = NOW()
                WHERE session_id = $4
                  AND ended_at IS NULL
                """,
                json.dumps(screens_visited or [], ensure_ascii=False),
                battles_played,
                cases_opened,
                session_id,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).error(
                "update_user_session failed: session_id=%s", session_id, exc_info=True,
            )

    async def track_onboarding_event(
        self,
        user_id: int,
        step: str,
        completed: bool,
        time_spent_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._pool:
            return
        try:
            await self.execute(
                """
                INSERT INTO onboarding_events (user_id, step, completed, time_spent_seconds, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                user_id,
                step,
                completed,
                time_spent_seconds,
                json.dumps(metadata or {}, ensure_ascii=False),
            )
        except Exception:
            import logging
            logging.getLogger(__name__).error(
                "track_onboarding_event failed: user_id=%s step=%s",
                user_id, step, exc_info=True,
            )

    async def record_battle_summary(
        self,
        *,
        match_id: str,
        p1_user_id: int,
        p2_user_id: int,
        winner_user_id: Optional[int] = None,
        loser_user_id: Optional[int] = None,
        p1_hero_id: Optional[int] = None,
        p2_hero_id: Optional[int] = None,
        p1_deck: Optional[list[int]] = None,
        p2_deck: Optional[list[int]] = None,
        surrender: bool = False,
        afk: bool = False,
        match_type: str = "pvp",
        game_mode: Optional[str] = None,
        duration_seconds: int = 0,
        turns_count: int = 0,
        p1_trophy_change: int = 0,
        p2_trophy_change: int = 0,
        p1_coins_earned: int = 0,
        p2_coins_earned: int = 0,
        p1_cards_played: int = 0,
        p2_cards_played: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._pool:
            return
        try:
            await self.execute(
                """
                INSERT INTO battle_summary (
                    match_id, p1_user_id, p2_user_id,
                    winner_user_id, loser_user_id,
                    p1_hero_id, p2_hero_id, p1_deck, p2_deck,
                    surrender, afk, match_type, game_mode,
                    duration_seconds, turns_count,
                    p1_trophy_change, p2_trophy_change,
                    p1_coins_earned, p2_coins_earned,
                    p1_cards_played, p2_cards_played,
                    metadata
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22
                )
                ON CONFLICT (match_id) DO NOTHING
                """,
                match_id, p1_user_id, p2_user_id,
                winner_user_id, loser_user_id,
                p1_hero_id, p2_hero_id,
                json.dumps(p1_deck or [], ensure_ascii=False),
                json.dumps(p2_deck or [], ensure_ascii=False),
                surrender, afk, match_type, game_mode,
                duration_seconds, turns_count,
                p1_trophy_change, p2_trophy_change,
                p1_coins_earned, p2_coins_earned,
                p1_cards_played, p2_cards_played,
                json.dumps(metadata or {}, ensure_ascii=False),
            )
        except Exception:
            import logging
            logging.getLogger(__name__).error(
                "record_battle_summary failed: match_id=%s", match_id, exc_info=True,
            )

    async def apply_battle_end_rewards_transaction(
        self,
        *,
        match_id: str,
        p1_user_id: int,
        p2_user_id: int,
        winner_user_id: Optional[int] = None,
        loser_user_id: Optional[int] = None,
        p1_hero_id: Optional[int] = None,
        p2_hero_id: Optional[int] = None,
        p1_deck: Optional[list[int]] = None,
        p2_deck: Optional[list[int]] = None,
        surrender: bool = False,
        afk: bool = False,
        match_type: str = "pvp",
        game_mode: Optional[str] = None,
        duration_seconds: int = 0,
        turns_count: int = 0,
        p1_trophy_change: int = 0,
        p2_trophy_change: int = 0,
        p1_coins_earned: int = 0,
        p2_coins_earned: int = 0,
        p1_cards_played: int = 0,
        p2_cards_played: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        battle_result: Optional[Dict[str, Any]] = None,
        rewards: Optional[Dict[int | str, Dict[str, Any]]] = None,
        economy_events: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Apply the final battle summary and all balance rewards atomically.

        battle_summary.match_id is the idempotency gate: duplicate match_id means
        no user balances, counters, mail, or analytics are touched.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена.")

        result: Dict[str, Any] = {
            "applied": False,
            "reason": "duplicate_summary",
            "trophy_changes": {},
            "trophy_totals": {},
            "coins_changes": {},
            "coins_totals": {},
            "stars_changes": {},
            "stars_totals": {},
            "keys_changes": {},
            "keys_totals": {},
            "win_counters": {},
            "league_up": {},
        }

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                summary_row = await conn.fetchrow(
                    """
                    INSERT INTO battle_summary (
                        match_id, p1_user_id, p2_user_id,
                        winner_user_id, loser_user_id,
                        p1_hero_id, p2_hero_id, p1_deck, p2_deck,
                        surrender, afk, match_type, game_mode,
                        duration_seconds, turns_count,
                        p1_trophy_change, p2_trophy_change,
                        p1_coins_earned, p2_coins_earned,
                        p1_cards_played, p2_cards_played,
                        metadata
                    ) VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22
                    )
                    ON CONFLICT (match_id) DO NOTHING
                    RETURNING id
                    """,
                    match_id, p1_user_id, p2_user_id,
                    winner_user_id, loser_user_id,
                    p1_hero_id, p2_hero_id,
                    json.dumps(p1_deck or [], ensure_ascii=False),
                    json.dumps(p2_deck or [], ensure_ascii=False),
                    surrender, afk, match_type, game_mode,
                    duration_seconds, turns_count,
                    p1_trophy_change, p2_trophy_change,
                    p1_coins_earned, p2_coins_earned,
                    p1_cards_played, p2_cards_played,
                    json.dumps(metadata or {}, ensure_ascii=False),
                )
                if not summary_row:
                    return {"applied": False, "reason": "duplicate_summary"}

                result["applied"] = True
                result["reason"] = "applied"
                result["summary_id"] = summary_row["id"]

                battle_result = battle_result or {}
                await conn.execute(
                    """
                    INSERT INTO battle_results (
                        match_id, winner_id, loser_id, winner_score, loser_score,
                        match_duration, match_type, p1_id, p2_id,
                        p1_trophy_change, p2_trophy_change, turns_count
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (match_id) DO NOTHING
                    """,
                    match_id,
                    winner_user_id,
                    loser_user_id,
                    int(battle_result.get("winner_score", 0) or 0),
                    int(battle_result.get("loser_score", 0) or 0),
                    int(battle_result.get("match_duration", duration_seconds) or 0),
                    battle_result.get("match_type", match_type),
                    p1_user_id,
                    p2_user_id,
                    p1_trophy_change,
                    p2_trophy_change,
                    turns_count,
                )

                user_rewards = rewards or {}
                for raw_user_id, plan in user_rewards.items():
                    user_id = int(raw_user_id)
                    plan = plan or {}

                    trophy_delta = int(plan.get("trophies", 0) or 0)
                    if trophy_delta:
                        trophy_row = await conn.fetchrow(
                            """
                            UPDATE users
                            SET
                                trophies = GREATEST(0, COALESCE(trophies, 0) + $1),
                                max_trophies = GREATEST(COALESCE(max_trophies, 0), GREATEST(0, COALESCE(trophies, 0) + $1)),
                                league = CASE
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 9000 THEN 10
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 7500 THEN 9
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 6000 THEN 8
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 4500 THEN 7
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 3000 THEN 6
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 2000 THEN 5
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 1200 THEN 4
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 600  THEN 3
                                    WHEN GREATEST(0, COALESCE(trophies, 0) + $1) >= 300  THEN 2
                                    ELSE 1
                                END,
                                updated_at = NOW()
                            WHERE user_id = $2
                            RETURNING trophies, max_trophies, league
                            """,
                            trophy_delta, user_id,
                        )
                        if trophy_row:
                            result["trophy_changes"][user_id] = trophy_delta
                            result["trophy_totals"][user_id] = trophy_row["trophies"]
                            old_league = int(plan.get("old_league", trophy_row["league"]) or trophy_row["league"])
                            new_league = int(trophy_row["league"] or old_league)
                            if new_league != old_league:
                                await self._insert_battle_league_mail(
                                    conn,
                                    user_id=user_id,
                                    old_league=old_league,
                                    new_league=new_league,
                                )
                                if new_league > old_league and new_league in LEAGUE_CONFIG:
                                    league_data = LEAGUE_CONFIG[new_league]
                                    result["league_up"][user_id] = {
                                        "league_id": new_league,
                                        "name": league_data["name"],
                                        "emoji": league_data["emoji"],
                                        "color": league_data["color"],
                                    }

                    coins_delta = int(plan.get("coins", 0) or 0)
                    if coins_delta:
                        coins_row = await conn.fetchrow(
                            """
                            UPDATE users
                            SET coins = GREATEST(0, COALESCE(coins, 0) + $1),
                                updated_at = NOW()
                            WHERE user_id = $2
                            RETURNING coins
                            """,
                            coins_delta, user_id,
                        )
                        if coins_row:
                            result["coins_changes"][user_id] = coins_delta
                            result["coins_totals"][user_id] = coins_row["coins"]

                    stars_delta = int(plan.get("stars", 0) or 0)
                    if stars_delta:
                        stars_row = await conn.fetchrow(
                            """
                            UPDATE users
                            SET stars = GREATEST(0, COALESCE(stars, 0) + $1),
                                updated_at = NOW()
                            WHERE user_id = $2
                            RETURNING stars
                            """,
                            stars_delta, user_id,
                        )
                        if stars_row:
                            result["stars_changes"][user_id] = stars_delta
                            result["stars_totals"][user_id] = stars_row["stars"]

                    wins_for_case = plan.get("wins_for_case")
                    if wins_for_case:
                        wins_for_case = max(1, int(wins_for_case))
                        settings = await conn.fetchrow(
                            """
                            SELECT wins_since_last_case
                            FROM user_settings
                            WHERE user_id = $1
                            FOR UPDATE
                            """,
                            user_id,
                        )
                        if not settings:
                            await conn.execute(
                                """
                                INSERT INTO user_settings (user_id, wins_since_last_case)
                                VALUES ($1, 0)
                                ON CONFLICT (user_id) DO NOTHING
                                """,
                                user_id,
                            )
                            settings = await conn.fetchrow(
                                """
                                SELECT wins_since_last_case
                                FROM user_settings
                                WHERE user_id = $1
                                FOR UPDATE
                                """,
                                user_id,
                            )
                        current_wins = int(settings["wins_since_last_case"] if settings else 0) + 1
                        granted_key = current_wins >= wins_for_case
                        if granted_key:
                            await conn.execute(
                                "UPDATE user_settings SET wins_since_last_case = 0, updated_at = NOW() WHERE user_id = $1",
                                user_id,
                            )
                            key_row = await conn.fetchrow(
                                """
                                UPDATE users
                                SET keys = COALESCE(keys, 0) + 1,
                                    updated_at = NOW()
                                WHERE user_id = $1
                                RETURNING keys
                                """,
                                user_id,
                            )
                            result["keys_changes"][user_id] = 1
                            result["keys_totals"][user_id] = key_row["keys"] if key_row else 0
                        else:
                            await conn.execute(
                                "UPDATE user_settings SET wins_since_last_case = $1, updated_at = NOW() WHERE user_id = $2",
                                current_wins,
                                user_id,
                            )
                        result["win_counters"][user_id] = {
                            "current_wins": current_wins,
                            "wins_for_case": wins_for_case,
                            "wins_since_last_case": 0 if granted_key else current_wins,
                            "granted_key": granted_key,
                        }

                for event in economy_events or []:
                    safe_meta = _json_safe(event.get("metadata") or {})
                    await conn.execute(
                        """
                        INSERT INTO economy_events (user_id, event_type, resource, amount, source, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                        """,
                        int(event["user_id"]),
                        str(event["event_type"]),
                        str(event["resource"]),
                        float(event.get("amount") or 0),
                        event.get("source"),
                        json.dumps(safe_meta, ensure_ascii=False),
                    )

                return result

    async def _insert_battle_league_mail(
        self,
        conn: Any,
        *,
        user_id: int,
        old_league: int,
        new_league: int,
    ) -> None:
        if new_league > old_league:
            league_data = LEAGUE_CONFIG.get(new_league)
            if not league_data:
                return
            subject = "🏆 Повышение лиги!"
            text = "Ты достиг лиги {} {}! Продолжай в том же духе.".format(
                league_data["emoji"],
                league_data["name"],
            )
            category = "rewards"
            icon = "🏆"
        else:
            old_league_data = LEAGUE_CONFIG.get(old_league)
            if not old_league_data:
                return
            next_min = old_league_data["min_trophies"]
            subject = "📉 Понижение лиги"
            text = "Ты покинул лигу {} {}. Набери {} трофеев чтобы вернуться.".format(
                old_league_data["emoji"],
                old_league_data["name"],
                next_min,
            )
            category = "system"
            icon = "📉"

        has_content = await conn.fetchval(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'user_mail'
              AND column_name = 'content'
            """
        )
        if has_content:
            await conn.execute(
                """
                INSERT INTO user_mail (user_id, sender, subject, text, content, category, icon)
                VALUES ($1, $2, $3, $4, $4, $5, $6)
                """,
                user_id, "Система", subject, text, category, icon,
            )
        else:
            await conn.execute(
                """
                INSERT INTO user_mail (user_id, sender, subject, text, category, icon)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id, "Система", subject, text, category, icon,
            )

    async def record_battle_actions(
        self,
        battle_id: str,
        actions: list[Dict[str, Any]],
    ) -> int:
        if not self._pool or not actions:
            return 0
        inserted = 0
        for item in actions:
            try:
                await self.execute(
                    """
                    INSERT INTO battle_actions (
                        battle_id, turn_number, acting_player, acting_user_id,
                        is_bot, state_json, action_json, quality_score
                    )
                    VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8)
                    """,
                    battle_id,
                    int(item.get("turn_number") or 0),
                    item.get("acting_player"),
                    item.get("acting_user_id"),
                    bool(item.get("is_bot", False)),
                    json.dumps(item.get("state_json") or {}, ensure_ascii=False),
                    json.dumps(item.get("action_json") or {}, ensure_ascii=False),
                    item.get("quality_score"),
                )
                inserted += 1
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "record_battle_actions single row failed: battle_id=%s turn=%s",
                    battle_id, item.get("turn_number"), exc_info=True,
                )
        return inserted

    # ── Analytics: admin queries ──

    async def get_admin_analytics_overview(self, days: int = 7) -> Dict[str, Any]:
        if not self._pool:
            return {}

        days = max(int(days or 7), 1)

        total_users = await self.fetchval("SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE")
        new_users_today = await self.fetchval(
            "SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE AND created_at >= CURRENT_DATE"
        )
        new_users_7d = await self.fetchval(
            "SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE AND created_at >= NOW() - make_interval(days => $1::int)", days
        )
        active_users_24h = await self.fetchval(
            """
            SELECT COUNT(DISTINCT s.user_id)
            FROM user_sessions s
            JOIN users u ON u.user_id = s.user_id
            WHERE COALESCE(u.is_bot, FALSE) = FALSE
              AND s.started_at >= NOW() - INTERVAL '24 hours'
            """
        )

        total_revenue = await self.fetchval(
            """
            SELECT COALESCE(SUM(p.amount), 0)
            FROM payments p
            JOIN users u ON u.user_id = p.user_id
            WHERE p.status = 'succeeded'
              AND COALESCE(u.is_bot, FALSE) = FALSE
            """
        )
        revenue_today = await self.fetchval(
            """
            SELECT COALESCE(SUM(p.amount), 0)
            FROM payments p
            JOIN users u ON u.user_id = p.user_id
            WHERE p.status = 'succeeded'
              AND COALESCE(u.is_bot, FALSE) = FALSE
              AND p.created_at >= CURRENT_DATE
            """
        )
        revenue_7d = await self.fetchval(
            """
            SELECT COALESCE(SUM(p.amount), 0)
            FROM payments p
            JOIN users u ON u.user_id = p.user_id
            WHERE p.status = 'succeeded'
              AND COALESCE(u.is_bot, FALSE) = FALSE
              AND p.created_at >= NOW() - make_interval(days => $1::int)
            """,
            days,
        )
        purchases_today = await self.fetchval(
            """
            SELECT COUNT(*)
            FROM payments p
            JOIN users u ON u.user_id = p.user_id
            WHERE p.status = 'succeeded'
              AND COALESCE(u.is_bot, FALSE) = FALSE
              AND p.created_at >= CURRENT_DATE
            """
        )
        paying_users_total = await self.fetchval(
            """
            SELECT COUNT(DISTINCT p.user_id)
            FROM payments p
            JOIN users u ON u.user_id = p.user_id
            WHERE p.status = 'succeeded'
              AND COALESCE(u.is_bot, FALSE) = FALSE
            """
        )

        battles_today = await self.fetchval(
            """
            SELECT COUNT(*)
            FROM battle_results br
            LEFT JOIN users u1 ON u1.user_id = br.p1_id
            LEFT JOIN users u2 ON u2.user_id = br.p2_id
            WHERE br.created_at >= CURRENT_DATE
              AND COALESCE(u1.is_bot, FALSE) = FALSE
              AND COALESCE(u2.is_bot, FALSE) = FALSE
            """
        )
        battles_7d = await self.fetchval(
            """
            SELECT COUNT(*)
            FROM battle_results br
            LEFT JOIN users u1 ON u1.user_id = br.p1_id
            LEFT JOIN users u2 ON u2.user_id = br.p2_id
            WHERE br.created_at >= NOW() - make_interval(days => $1::int)
              AND COALESCE(u1.is_bot, FALSE) = FALSE
              AND COALESCE(u2.is_bot, FALSE) = FALSE
            """, days
        )
        avg_duration = await self.fetchval(
            """
            SELECT AVG(br.match_duration)
            FROM battle_results br
            LEFT JOIN users u1 ON u1.user_id = br.p1_id
            LEFT JOIN users u2 ON u2.user_id = br.p2_id
            WHERE br.match_duration > 0
              AND br.created_at >= NOW() - make_interval(days => $1::int)
              AND COALESCE(u1.is_bot, FALSE) = FALSE
              AND COALESCE(u2.is_bot, FALSE) = FALSE
            """,
            days,
        )

        top_products = []
        try:
            tp_rows = await self.fetch(
                """
                SELECT description AS item_name, COUNT(*) AS cnt, SUM(amount) AS total_revenue
                FROM payments p
                JOIN users u ON u.user_id = p.user_id
                WHERE p.status = 'succeeded'
                  AND COALESCE(u.is_bot, FALSE) = FALSE
                  AND description IS NOT NULL
                GROUP BY description
                ORDER BY cnt DESC
                LIMIT 5
                """
            )
            top_products = [
                {"name": r["item_name"] or "—", "count": r["cnt"], "revenue": float(r["total_revenue"] or 0)}
                for r in tp_rows
            ]
        except Exception:
            pass

        return {
            "total_users": int(total_users or 0),
            "new_users_today": int(new_users_today or 0),
            "new_users_7d": int(new_users_7d or 0),
            "active_users_24h": int(active_users_24h or 0),
            "total_revenue": float(total_revenue or 0),
            "revenue_today": float(revenue_today or 0),
            "revenue_7d": float(revenue_7d or 0),
            "purchases_today": int(purchases_today or 0),
            "paying_users_total": int(paying_users_total or 0),
            "battles_today": int(battles_today or 0),
            "battles_7d": int(battles_7d or 0),
            "avg_battle_duration": float(round(avg_duration or 0, 1)),
            "top_products": top_products,
        }

    async def get_admin_revenue_analytics(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            return {"revenue_by_day": [], "total": 0}

        days = max(int(days or 30), 1)
        rows = await self.fetch(
            """
            SELECT DATE(p.created_at) AS day,
                   COUNT(*) AS purchases,
                   COALESCE(SUM(p.amount), 0) AS revenue
            FROM payments p
            JOIN users u ON u.user_id = p.user_id
            WHERE p.status = 'succeeded'
              AND COALESCE(u.is_bot, FALSE) = FALSE
              AND p.created_at >= NOW() - make_interval(days => $1::int)
            GROUP BY day ORDER BY day ASC
            """,
            days,
        )
        revenue_by_day = [
            {"day": str(r["day"]), "purchases": r["purchases"], "revenue": float(r["revenue"])}
            for r in rows
        ]
        total = sum(r["revenue"] for r in revenue_by_day)
        return {"revenue_by_day": revenue_by_day, "total": float(total)}

    async def get_admin_players_analytics(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            return {"new_users_by_day": [], "active_users_by_day": [], "total_new": 0}

        days = max(int(days or 30), 1)
        rows = await self.fetch(
            """
            SELECT DATE(created_at) AS day, COUNT(*) AS cnt
            FROM users
            WHERE COALESCE(is_bot, FALSE) = FALSE
              AND created_at >= NOW() - make_interval(days => $1::int)
            GROUP BY day ORDER BY day ASC
            """,
            days,
        )
        new_users_by_day = [
            {"day": str(r["day"]), "count": r["cnt"]}
            for r in rows
        ]
        total_new = sum(r["count"] for r in new_users_by_day)

        active_rows = await self.fetch(
            """
            SELECT DATE(s.started_at) AS day, COUNT(DISTINCT s.user_id) AS cnt
            FROM user_sessions s
            JOIN users u ON u.user_id = s.user_id
            WHERE COALESCE(u.is_bot, FALSE) = FALSE
              AND s.started_at >= NOW() - make_interval(days => $1::int)
            GROUP BY day ORDER BY day ASC
            """,
            days,
        )
        active_by_day = [
            {"day": str(r["day"]), "count": r["cnt"]}
            for r in active_rows
        ]
        return {
            "new_users_by_day": new_users_by_day,
            "active_users_by_day": active_by_day,
            "total_new": total_new,
        }

    async def get_admin_battle_analytics(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            return {"battles_by_day": [], "total": 0}

        days = max(int(days or 30), 1)
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = await self.fetch(
            """
            SELECT DATE(br.created_at) AS day, COUNT(*) AS cnt,
                   AVG(br.match_duration) AS avg_duration,
                   AVG(br.turns_count) AS avg_turns
            FROM battle_results br
            LEFT JOIN users u1 ON u1.user_id = br.p1_id
            LEFT JOIN users u2 ON u2.user_id = br.p2_id
            WHERE br.created_at >= NOW() - make_interval(days => $1::int)
              AND COALESCE(u1.is_bot, FALSE) = FALSE
              AND COALESCE(u2.is_bot, FALSE) = FALSE
            GROUP BY day ORDER BY day ASC
            """,
            days,
        )
        battles_by_day = [
            {
                "day": str(r["day"]),
                "count": r["cnt"],
                "avg_duration": float(round(r["avg_duration"] or 0, 1)),
                "avg_turns": float(round(r["avg_turns"] or 0, 1)),
            }
            for r in rows
        ]
        total = sum(r["count"] for r in battles_by_day)

        summary = {}
        mode_pickrate = []
        bot_mode_outcomes = []
        bot_card_outcomes = {"winning_cards": [], "losing_cards": []}
        try:
            summary_row = await self.fetchrow(
                """
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE winner_user_id IS NOT NULL) AS finished,
                       AVG(duration_seconds) AS avg_duration,
                       AVG(turns_count) AS avg_turns,
                       COUNT(*) FILTER (WHERE surrender) AS surrenders,
                       COUNT(*) FILTER (WHERE afk) AS afk
                FROM battle_summary bs
                LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                WHERE bs.created_at >= $1
                  AND COALESCE(u1.is_bot, FALSE) = FALSE
                  AND COALESCE(u2.is_bot, FALSE) = FALSE
                  AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                  AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                """,
                since,
            )
            if summary_row:
                finished = int(summary_row["finished"] or 0)
                summary = {
                    "total": int(summary_row["total"] or 0),
                    "finished": finished,
                    "draw_rate": round(1 - (finished / max(int(summary_row["total"] or 0), 1)), 3),
                    "avg_duration": round(float(summary_row["avg_duration"] or 0), 1),
                    "avg_turns": round(float(summary_row["avg_turns"] or 0), 1),
                    "surrenders": int(summary_row["surrenders"] or 0),
                    "afk": int(summary_row["afk"] or 0),
                }
        except Exception:
            pass

        try:
            mode_rows = await self.fetch(
                """
                SELECT COALESCE(game_mode, match_type, 'classic') AS mode_id,
                       COUNT(*) AS games,
                       COUNT(*) FILTER (WHERE winner_user_id IS NOT NULL) AS decided
                FROM battle_summary bs
                LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                WHERE bs.created_at >= $1
                  AND COALESCE(u1.is_bot, FALSE) = FALSE
                  AND COALESCE(u2.is_bot, FALSE) = FALSE
                  AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                  AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                GROUP BY mode_id
                ORDER BY games DESC
                """,
                since,
            )
            total_summary_games = sum(int(r["games"] or 0) for r in mode_rows) or 1
            mode_pickrate = [
                {
                    "mode_id": r["mode_id"],
                    "games": int(r["games"] or 0),
                    "pickrate": round(int(r["games"] or 0) / total_summary_games, 4),
                    "decided": int(r["decided"] or 0),
                }
                for r in mode_rows
            ]
        except Exception:
            pass

        try:
            bot_mode_rows = await self.fetch(
                """
                WITH bot_games AS (
                    SELECT COALESCE(game_mode, match_type, 'classic') AS mode_id,
                           metadata->>'p1_is_bot' = 'true' AS p1_is_bot,
                           metadata->>'p2_is_bot' = 'true' AS p2_is_bot,
                           winner_user_id = p1_user_id AS p1_won,
                           winner_user_id = p2_user_id AS p2_won
                    FROM battle_summary
                    WHERE created_at >= $1
                      AND (metadata->>'p1_is_bot' = 'true' OR metadata->>'p2_is_bot' = 'true')
                )
                SELECT mode_id,
                       COUNT(*) AS bot_games,
                       COUNT(*) FILTER (WHERE (p1_is_bot AND p1_won) OR (p2_is_bot AND p2_won)) AS bot_wins,
                       COUNT(*) FILTER (WHERE (p1_is_bot AND NOT p1_won) OR (p2_is_bot AND NOT p2_won)) AS bot_losses
                FROM bot_games
                GROUP BY mode_id
                ORDER BY bot_games DESC
                """,
                since,
            )
            bot_mode_outcomes = [
                {
                    "mode_id": r["mode_id"],
                    "games": int(r["bot_games"] or 0),
                    "wins": int(r["bot_wins"] or 0),
                    "losses": int(r["bot_losses"] or 0),
                    "bot_wins": int(r["bot_wins"] or 0),
                    "bot_losses": int(r["bot_losses"] or 0),
                    "winrate": round(int(r["bot_wins"] or 0) / max(int(r["bot_games"] or 0), 1), 4),
                    "bot_winrate": round(int(r["bot_wins"] or 0) / max(int(r["bot_games"] or 0), 1), 4),
                }
                for r in bot_mode_rows
            ]
        except Exception:
            pass

        try:
            bot_card_rows = await self.fetch(
                """
                WITH bot_decks AS (
                    SELECT jsonb_array_elements_text(p1_deck)::INT AS card_id,
                           winner_user_id = p1_user_id AS won
                    FROM battle_summary
                    WHERE created_at >= $1 AND metadata->>'p1_is_bot' = 'true'
                      AND p1_deck IS NOT NULL AND jsonb_array_length(p1_deck) > 0
                    UNION ALL
                    SELECT jsonb_array_elements_text(p2_deck)::INT AS card_id,
                           winner_user_id = p2_user_id AS won
                    FROM battle_summary
                    WHERE created_at >= $1 AND metadata->>'p2_is_bot' = 'true'
                      AND p2_deck IS NOT NULL AND jsonb_array_length(p2_deck) > 0
                ),
                agg AS (
                    SELECT card_id, COUNT(*) AS games,
                           COUNT(*) FILTER (WHERE won) AS wins,
                           COUNT(*) FILTER (WHERE NOT won) AS losses
                    FROM bot_decks
                    GROUP BY card_id
                )
                SELECT a.card_id, COALESCE(c.name, 'Card #' || a.card_id::TEXT) AS name,
                       a.games, a.wins, a.losses,
                       CASE WHEN a.games > 0 THEN a.wins::FLOAT / a.games ELSE 0 END AS winrate
                FROM agg a
                LEFT JOIN cards c ON c.id = a.card_id
                ORDER BY a.games DESC
                LIMIT 100
                """,
                since,
            )
            all_bot_cards = [
                {
                    "card_id": r["card_id"],
                    "name": r["name"],
                    "games": int(r["games"] or 0),
                    "wins": int(r["wins"] or 0),
                    "losses": int(r["losses"] or 0),
                    "winrate": round(float(r["winrate"] or 0), 4),
                }
                for r in bot_card_rows
            ]
            bot_card_outcomes = {
                "winning_cards": sorted(all_bot_cards, key=lambda x: (x["winrate"], x["games"]), reverse=True)[:20],
                "losing_cards": sorted(all_bot_cards, key=lambda x: (x["winrate"], -x["games"]))[:20],
            }
        except Exception:
            pass

        return {
            "battles_by_day": battles_by_day,
            "total": total,
            "summary": summary,
            "mode_pickrate": mode_pickrate,
            "bot_mode_outcomes": bot_mode_outcomes,
            "bot_card_outcomes": bot_card_outcomes,
        }


    async def get_admin_cards_analytics(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            return {"popular_cards": [], "winner_cards": [], "never_used_cards": []}

        days = max(int(days or 30), 1)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        popular: list[dict] = []
        winner_deck: list[dict] = []
        never_used: list[dict] = []

        try:
            cte_rows = await self.fetch(
                """
                WITH deck_cards AS (
                    SELECT jsonb_array_elements_text(p1_deck)::INT AS card_id, winner_user_id = p1_user_id AS won
                    FROM battle_summary bs
                    LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                    LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                    WHERE bs.created_at >= $1
                      AND COALESCE(u1.is_bot, FALSE) = FALSE
                      AND COALESCE(u2.is_bot, FALSE) = FALSE
                      AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                      AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                      AND p1_deck IS NOT NULL AND jsonb_array_length(p1_deck) > 0
                    UNION ALL
                    SELECT jsonb_array_elements_text(p2_deck)::INT AS card_id, winner_user_id = p2_user_id AS won
                    FROM battle_summary bs
                    LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                    LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                    WHERE bs.created_at >= $1
                      AND COALESCE(u1.is_bot, FALSE) = FALSE
                      AND COALESCE(u2.is_bot, FALSE) = FALSE
                      AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                      AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                      AND p2_deck IS NOT NULL AND jsonb_array_length(p2_deck) > 0
                ),
                agg AS (
                    SELECT card_id, COUNT(*) AS appearances, COUNT(*) FILTER (WHERE won) AS wins
                    FROM deck_cards
                    GROUP BY card_id
                )
                SELECT a.card_id, a.appearances, a.wins,
                       COALESCE(c.name, 'Card #' || a.card_id::TEXT) AS name,
                       COALESCE(c.rarity, 'unknown') AS rarity,
                       CASE WHEN a.appearances > 0 THEN a.wins::FLOAT / a.appearances ELSE 0 END AS winrate
                FROM agg a
                LEFT JOIN cards c ON c.id = a.card_id
                ORDER BY a.appearances DESC
                LIMIT 20
                """,
                since,
            )
            popular = [
                {"card_id": r["card_id"], "name": r["name"], "appearances": r["appearances"],
                 "wins": r["wins"], "winrate": round(float(r["winrate"]), 3)}
                for r in cte_rows
            ]
        except Exception:
            pass

        try:
            win_rows = await self.fetch(
                """
                WITH deck_cards AS (
                    SELECT jsonb_array_elements_text(p1_deck)::INT AS card_id
                    FROM battle_summary bs
                    LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                    LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                    WHERE bs.created_at >= $1
                      AND COALESCE(u1.is_bot, FALSE) = FALSE
                      AND COALESCE(u2.is_bot, FALSE) = FALSE
                      AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                      AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                      AND winner_user_id = p1_user_id AND p1_deck IS NOT NULL AND jsonb_array_length(p1_deck) > 0
                    UNION ALL
                    SELECT jsonb_array_elements_text(p2_deck)::INT AS card_id
                    FROM battle_summary bs
                    LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                    LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                    WHERE bs.created_at >= $1
                      AND COALESCE(u1.is_bot, FALSE) = FALSE
                      AND COALESCE(u2.is_bot, FALSE) = FALSE
                      AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                      AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                      AND winner_user_id = p2_user_id AND p2_deck IS NOT NULL AND jsonb_array_length(p2_deck) > 0
                )
                SELECT card_id, COUNT(*) AS appearances
                FROM deck_cards
                GROUP BY card_id
                ORDER BY appearances DESC
                LIMIT 20
                """,
                since,
            )
            winner_deck = [
                {"card_id": r["card_id"], "appearances": r["appearances"]}
                for r in win_rows
            ]
        except Exception:
            pass

        try:
            all_card_ids = await self.fetch("SELECT id FROM cards")
            all_ids = {r["id"] for r in all_card_ids}
            used_rows = await self.fetch(
                """
                SELECT DISTINCT jsonb_array_elements_text(p1_deck)::INT AS card_id
                FROM battle_summary bs
                LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                WHERE p1_deck IS NOT NULL AND jsonb_array_length(p1_deck) > 0 AND bs.created_at >= $1
                  AND COALESCE(u1.is_bot, FALSE) = FALSE
                  AND COALESCE(u2.is_bot, FALSE) = FALSE
                  AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                  AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                UNION
                SELECT DISTINCT jsonb_array_elements_text(p2_deck)::INT AS card_id
                FROM battle_summary bs
                LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                WHERE p2_deck IS NOT NULL AND jsonb_array_length(p2_deck) > 0 AND bs.created_at >= $1
                  AND COALESCE(u1.is_bot, FALSE) = FALSE
                  AND COALESCE(u2.is_bot, FALSE) = FALSE
                  AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                  AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                """,
                since,
            )
            used_ids = {r["card_id"] for r in used_rows}
            never_ids = all_ids - used_ids
            if never_ids:
                never_rows = await self.fetch(
                    "SELECT id, name, rarity FROM cards WHERE id = ANY($1::int[]) ORDER BY id LIMIT 50",
                    list(never_ids),
                )
                never_used = [
                    {"card_id": r["id"], "name": r["name"] or f"Card #{r['id']}", "rarity": r["rarity"] or "unknown"}
                    for r in never_rows
                ]
        except Exception:
            pass

        bot_cards = {"winning_cards": [], "losing_cards": []}
        try:
            rows = await self.fetch(
                """
                WITH bot_decks AS (
                    SELECT jsonb_array_elements_text(p1_deck)::INT AS card_id,
                           winner_user_id = p1_user_id AS won
                    FROM battle_summary
                    WHERE created_at >= $1 AND metadata->>'p1_is_bot' = 'true'
                      AND p1_deck IS NOT NULL AND jsonb_array_length(p1_deck) > 0
                    UNION ALL
                    SELECT jsonb_array_elements_text(p2_deck)::INT AS card_id,
                           winner_user_id = p2_user_id AS won
                    FROM battle_summary
                    WHERE created_at >= $1 AND metadata->>'p2_is_bot' = 'true'
                      AND p2_deck IS NOT NULL AND jsonb_array_length(p2_deck) > 0
                ),
                agg AS (
                    SELECT card_id, COUNT(*) AS games,
                           COUNT(*) FILTER (WHERE won) AS wins,
                           COUNT(*) FILTER (WHERE NOT won) AS losses
                    FROM bot_decks
                    GROUP BY card_id
                )
                SELECT a.card_id, COALESCE(c.name, 'Card #' || a.card_id::TEXT) AS name,
                       a.games, a.wins, a.losses,
                       CASE WHEN a.games > 0 THEN a.wins::FLOAT / a.games ELSE 0 END AS winrate
                FROM agg a
                LEFT JOIN cards c ON c.id = a.card_id
                ORDER BY a.games DESC
                LIMIT 100
                """,
                since,
            )
            cards = [
                {"card_id": r["card_id"], "name": r["name"], "games": r["games"],
                 "wins": r["wins"], "losses": r["losses"], "winrate": round(float(r["winrate"]), 3)}
                for r in rows
            ]
            bot_cards = {
                "winning_cards": sorted(cards, key=lambda x: (x["winrate"], x["games"]), reverse=True)[:20],
                "losing_cards": sorted(cards, key=lambda x: (x["winrate"], -x["games"]))[:20],
            }
        except Exception:
            pass

        return {"popular_cards": popular, "winner_cards": winner_deck, "never_used_cards": never_used, "bot_cards": bot_cards}

    async def get_admin_heroes_analytics(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            return {"heroes": []}

        days = max(int(days or 30), 1)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            rows = await self.fetch(
                """
                WITH hero_games AS (
                    SELECT p1_hero_id AS hero_id, winner_user_id = p1_user_id AS won
                    FROM battle_summary bs
                    LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                    LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                    WHERE bs.created_at >= $1
                      AND COALESCE(u1.is_bot, FALSE) = FALSE
                      AND COALESCE(u2.is_bot, FALSE) = FALSE
                      AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                      AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                      AND p1_hero_id IS NOT NULL
                    UNION ALL
                    SELECT p2_hero_id AS hero_id, winner_user_id = p2_user_id AS won
                    FROM battle_summary bs
                    LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                    LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                    WHERE bs.created_at >= $1
                      AND COALESCE(u1.is_bot, FALSE) = FALSE
                      AND COALESCE(u2.is_bot, FALSE) = FALSE
                      AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                      AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                      AND p2_hero_id IS NOT NULL
                )
                SELECT h.hero_id, COUNT(*) AS games, COUNT(*) FILTER (WHERE won) AS wins,
                       CASE WHEN COUNT(*) > 0 THEN COUNT(*) FILTER (WHERE won)::FLOAT / COUNT(*) ELSE 0 END AS winrate
                FROM hero_games h
                GROUP BY h.hero_id
                ORDER BY games DESC
                """,
                since,
            )
            hero_ids = [r["hero_id"] for r in rows]
            names = {}
            if hero_ids:
                name_rows = await self.fetch(
                    "SELECT id, name FROM cards WHERE id = ANY($1::int[])",
                    hero_ids,
                )
                names = {r["id"]: r["name"] or f"Hero #{r['id']}" for r in name_rows}
            heroes = [
                {"hero_id": r["hero_id"], "name": names.get(r["hero_id"], f"Hero #{r['hero_id']}"),
                 "games": r["games"], "wins": r["wins"], "winrate": round(float(r["winrate"]), 3)}
                for r in rows
            ]
        except Exception:
            heroes = []

        return {"heroes": heroes}

    async def get_admin_retention_analytics(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            return {"sessions_by_day": [], "avg_session_duration": 0, "avg_screens_per_session": 0,
                    "returning_users_by_day": [], "top_screens": []}

        days = max(int(days or 30), 1)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        sessions_by_day: list[dict] = []
        avg_duration = 0.0
        avg_screens = 0.0
        returning_by_day: list[dict] = []
        top_screens: list[dict] = []

        try:
            s_rows = await self.fetch(
                """
                SELECT DATE(s.started_at) AS day, COUNT(*) AS sessions, COUNT(DISTINCT s.user_id) AS users
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.started_at >= $1
                  AND COALESCE(u.is_bot, FALSE) = FALSE
                GROUP BY day ORDER BY day ASC
                """,
                since,
            )
            sessions_by_day = [{"day": str(r["day"]), "sessions": r["sessions"], "users": r["users"]} for r in s_rows]
        except Exception:
            pass

        try:
            avg_duration = await self.fetchval(
                """
                SELECT AVG(s.duration_seconds)
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.duration_seconds IS NOT NULL
                  AND s.started_at >= $1
                  AND COALESCE(u.is_bot, FALSE) = FALSE
                """,
                since,
            ) or 0.0
        except Exception:
            pass

        try:
            avg_screens_raw = await self.fetchval(
                """
                SELECT AVG(jsonb_array_length(s.screens_visited))
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.screens_visited IS NOT NULL
                  AND s.started_at >= $1
                  AND jsonb_array_length(s.screens_visited) > 0
                  AND COALESCE(u.is_bot, FALSE) = FALSE
                """,
                since,
            ) or 0.0
            avg_screens = round(float(avg_screens_raw), 1)
        except Exception:
            pass

        try:
            ret_rows = await self.fetch(
                """
                SELECT DATE(s.started_at) AS day, COUNT(DISTINCT s.user_id) AS users
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.started_at >= $1
                  AND COALESCE(u.is_bot, FALSE) = FALSE
                GROUP BY day HAVING COUNT(*) > 1 ORDER BY day ASC
                """,
                since,
            )
            returning_by_day = [{"day": str(r["day"]), "users": r["users"]} for r in ret_rows]
        except Exception:
            pass

        try:
            screen_items = await self.fetch(
                """
                SELECT s.screens_visited
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.screens_visited IS NOT NULL
                  AND s.started_at >= $1
                  AND jsonb_array_length(s.screens_visited) > 0
                  AND COALESCE(u.is_bot, FALSE) = FALSE
                """,
                since,
            )
            screen_counts: dict[str, int] = {}
            for row in screen_items:
                items = row["screens_visited"]
                if isinstance(items, str):
                    try:
                        items = json.loads(items)
                    except Exception:
                        items = []
                if isinstance(items, list):
                    for item in items:
                        s = item if isinstance(item, str) else (item.get("screen") if isinstance(item, dict) else None)
                        if s:
                            screen_counts[s] = screen_counts.get(s, 0) + 1
            top_screens = sorted(
                [{"screen": k, "visits": v} for k, v in screen_counts.items()],
                key=lambda x: x["visits"], reverse=True
            )[:20]
        except Exception:
            pass

        return {
            "sessions_by_day": sessions_by_day,
            "avg_session_duration": round(float(avg_duration), 1),
            "avg_screens_per_session": float(avg_screens),
            "returning_users_by_day": returning_by_day,
            "top_screens": top_screens,
        }

    async def get_admin_onboarding_analytics(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            return {"steps": [], "raw": []}

        days = max(int(days or 30), 1)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        raw: list[dict] = []
        steps: list[dict] = []

        try:
            raw_rows = await self.fetch(
                """
                SELECT oe.step, oe.completed, COUNT(*) AS cnt
                FROM onboarding_events oe
                JOIN users u ON u.user_id = oe.user_id
                WHERE oe.created_at >= $1
                  AND COALESCE(u.is_bot, FALSE) = FALSE
                GROUP BY oe.step, oe.completed
                ORDER BY oe.step, oe.completed
                """,
                since,
            )
            raw = [{"step": r["step"], "completed": r["completed"], "count": r["cnt"]} for r in raw_rows]

            by_step: dict[str, dict] = {}
            for r in raw_rows:
                entry = by_step.setdefault(r["step"], {"step": r["step"], "started": 0, "completed": 0, "total": 0})
                entry["total"] += r["cnt"]
                if r["completed"]:
                    entry["completed"] = r["cnt"]
                else:
                    entry["started"] = r["cnt"]
            steps = list(by_step.values())
        except Exception:
            pass

        return {"steps": steps, "raw": raw}

    async def get_admin_battle_actions_analytics(self, days: int = 7) -> Dict[str, Any]:
        if not self._pool:
            return {"total_actions": 0, "actions_by_type": [], "bot_player_split": {},
                    "avg_actions_per_battle": 0, "quality_labeled": 0}

        days = max(int(days or 7), 1)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        total_actions = 0
        actions_by_type: list[dict] = []
        bot_split: dict = {"bot": 0, "player": 0}
        avg_per_battle = 0.0
        quality_labeled = 0

        try:
            total_actions = await self.fetchval(
                "SELECT COUNT(*) FROM battle_actions WHERE created_at >= $1 AND COALESCE(is_bot, FALSE) = FALSE",
                since,
            ) or 0
        except Exception:
            pass

        try:
            type_rows = await self.fetch(
                "SELECT COALESCE(action_json->>'type', 'unknown') AS atype, COUNT(*) AS cnt "
                "FROM battle_actions WHERE created_at >= $1 AND COALESCE(is_bot, FALSE) = FALSE "
                "GROUP BY atype ORDER BY cnt DESC",
                since,
            )
            actions_by_type = [{"type": r["atype"], "count": r["cnt"]} for r in type_rows]
        except Exception:
            pass

        try:
            bot_count = await self.fetchval(
                "SELECT COUNT(*) FROM battle_actions WHERE created_at >= $1 AND is_bot = TRUE", since
            ) or 0
            player_count = await self.fetchval(
                "SELECT COUNT(*) FROM battle_actions WHERE created_at >= $1 AND COALESCE(is_bot, FALSE) = FALSE",
                since,
            ) or 0
            bot_split = {"bot": int(bot_count or 0), "player": int(player_count or 0)}
        except Exception:
            pass

        try:
            avg_raw = await self.fetchval(
                "SELECT CASE WHEN COUNT(DISTINCT battle_id) > 0 THEN COUNT(*)::FLOAT / COUNT(DISTINCT battle_id) ELSE 0 END "
                "FROM battle_actions WHERE created_at >= $1 AND COALESCE(is_bot, FALSE) = FALSE",
                since,
            ) or 0.0
            avg_per_battle = round(float(avg_raw), 1)
        except Exception:
            pass

        try:
            quality_labeled = await self.fetchval(
                "SELECT COUNT(*) FROM battle_actions WHERE created_at >= $1 AND COALESCE(is_bot, FALSE) = FALSE AND quality_score IS NOT NULL",
                since,
            ) or 0
        except Exception:
            pass

        return {
            "total_actions": int(total_actions or 0),
            "actions_by_type": actions_by_type,
            "bot_player_split": bot_split,
            "avg_actions_per_battle": float(avg_per_battle),
            "quality_labeled": int(quality_labeled or 0),
        }


    # ── Players Admin: audit ──

    async def record_admin_account_action(
        self,
        admin_user_id: int,
        target_user_id: int,
        action_type: str,
        reason: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            safe_payload = _json_safe(payload) if payload else {}
            row = await self.fetchrow(
                """
                INSERT INTO admin_account_actions
                    (admin_user_id, target_user_id, action_type, reason, payload)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                RETURNING id, admin_user_id, target_user_id, action_type, reason,
                          payload, created_at
                """,
                admin_user_id, target_user_id, action_type, reason,
                json.dumps(safe_payload, ensure_ascii=False),
            )
            return dict(row) if row else {}
        except Exception:
            import logging
            logging.getLogger(__name__).error(
                "record_admin_account_action failed: admin=%s target=%s action=%s",
                admin_user_id, target_user_id, action_type, exc_info=True,
            )
            return {"error": "audit_log_failed"}

    # ── Players Admin: analytics ──

    async def get_admin_players_overview(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            now = datetime.utcnow()

            total = await self.fetchval("SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE") or 0
            active_24h = await self.fetchval(
                """
                SELECT COUNT(DISTINCT s.user_id)
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.started_at >= $1 AND COALESCE(u.is_bot, FALSE) = FALSE
                """,
                now - timedelta(hours=24),
            ) or 0
            active_7d = await self.fetchval(
                """
                SELECT COUNT(DISTINCT s.user_id)
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.started_at >= $1 AND COALESCE(u.is_bot, FALSE) = FALSE
                """,
                now - timedelta(days=7),
            ) or 0
            active_30d = await self.fetchval(
                """
                SELECT COUNT(DISTINCT s.user_id)
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.started_at >= $1 AND COALESCE(u.is_bot, FALSE) = FALSE
                """,
                now - timedelta(days=30),
            ) or 0
            new_today = await self.fetchval(
                "SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE AND reg_date >= $1",
                now.replace(hour=0, minute=0, second=0, microsecond=0),
            ) or 0
            new_7d = await self.fetchval(
                "SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE AND reg_date >= $1",
                now - timedelta(days=7),
            ) or 0
            banned_total = await self.fetchval(
                "SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE AND is_banned = TRUE"
            ) or 0
            warned_total = await self.fetchval(
                "SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE AND warnings_count > 0"
            ) or 0
            paying_users_total = await self.fetchval(
                """
                SELECT COUNT(DISTINCT p.user_id)
                FROM payments p
                JOIN users u ON u.user_id = p.user_id
                WHERE p.status = 'succeeded' AND COALESCE(u.is_bot, FALSE) = FALSE
                """
            ) or 0

            # Dormant users: no session in the last N days
            dormant_7d = await self.fetchval(
                """
                SELECT COUNT(*) FROM users u
                WHERE COALESCE(u.is_bot, FALSE) = FALSE
                  AND NOT EXISTS (
                    SELECT 1 FROM user_sessions s
                    WHERE s.user_id = u.user_id AND s.started_at >= $1
                )
                """,
                now - timedelta(days=7),
            ) or 0
            dormant_30d = await self.fetchval(
                """
                SELECT COUNT(*) FROM users u
                WHERE COALESCE(u.is_bot, FALSE) = FALSE
                  AND NOT EXISTS (
                    SELECT 1 FROM user_sessions s
                    WHERE s.user_id = u.user_id AND s.started_at >= $1
                )
                """,
                now - timedelta(days=30),
            ) or 0

            return {
                "total_users": int(total),
                "active_24h": int(active_24h),
                "active_7d": int(active_7d),
                "active_30d": int(active_30d),
                "new_today": int(new_today),
                "new_7d": int(new_7d),
                "banned_total": int(banned_total),
                "warned_total": int(warned_total),
                "paying_users_total": int(paying_users_total),
                "dormant_7d": int(dormant_7d),
                "dormant_30d": int(dormant_30d),
            }
        except Exception:
            import logging
            logging.getLogger(__name__).error("get_admin_players_overview failed", exc_info=True)
            return {"error": "analytics_overview_failed"}

    async def get_admin_players_leagues(self) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            rows = await self.fetch(
                """
                SELECT league, COUNT(*) as count, ROUND(AVG(trophies), 1) as avg_trophies
                FROM users
                WHERE COALESCE(is_bot, FALSE) = FALSE
                GROUP BY league
                ORDER BY league ASC
                """
            )
            return {
                "rows": [
                    {
                        "league": int(r["league"]),
                        "count": int(r["count"]),
                        "avg_trophies": float(r["avg_trophies"] or 0),
                    }
                    for r in rows
                ]
            }
        except Exception:
            import logging
            logging.getLogger(__name__).error("get_admin_players_leagues failed", exc_info=True)
            return {"error": "leagues_failed"}

    async def get_admin_players_activity(self, days: int = 30) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            now = datetime.utcnow()
            active_by_day = []
            new_by_day = []
            sessions_by_day = []
            battles_by_day = []

            for i in range(days - 1, -1, -1):
                d_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
                d_end = d_start + timedelta(days=1)
                day_label = d_start.strftime("%Y-%m-%d")

                active = await self.fetchval(
                    """
                    SELECT COUNT(DISTINCT s.user_id)
                    FROM user_sessions s
                    JOIN users u ON u.user_id = s.user_id
                    WHERE s.started_at >= $1 AND s.started_at < $2
                      AND COALESCE(u.is_bot, FALSE) = FALSE
                    """,
                    d_start, d_end,
                ) or 0
                active_by_day.append({"day": day_label, "count": int(active)})

                newc = await self.fetchval(
                    "SELECT COUNT(*) FROM users WHERE COALESCE(is_bot, FALSE) = FALSE AND reg_date >= $1 AND reg_date < $2",
                    d_start, d_end,
                ) or 0
                new_by_day.append({"day": day_label, "count": int(newc)})

                sess = await self.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM user_sessions s
                    JOIN users u ON u.user_id = s.user_id
                    WHERE s.started_at >= $1 AND s.started_at < $2
                      AND COALESCE(u.is_bot, FALSE) = FALSE
                    """,
                    d_start, d_end,
                ) or 0
                sessions_by_day.append({"day": day_label, "count": int(sess)})

                batt = await self.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM battle_summary bs
                    LEFT JOIN users u1 ON u1.user_id = bs.p1_user_id
                    LEFT JOIN users u2 ON u2.user_id = bs.p2_user_id
                    WHERE bs.created_at >= $1 AND bs.created_at < $2
                      AND COALESCE(u1.is_bot, FALSE) = FALSE
                      AND COALESCE(u2.is_bot, FALSE) = FALSE
                      AND COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'
                      AND COALESCE(bs.metadata->>'p2_is_bot', 'false') <> 'true'
                    """,
                    d_start, d_end,
                ) or 0
                battles_by_day.append({"day": day_label, "count": int(batt)})

            return {
                "active_by_day": active_by_day,
                "new_by_day": new_by_day,
                "sessions_by_day": sessions_by_day,
                "battles_by_day": battles_by_day,
            }
        except Exception:
            import logging
            logging.getLogger(__name__).error("get_admin_players_activity failed", exc_info=True)
            return {"error": "activity_failed"}

    async def search_admin_players(
        self,
        query: str = "",
        status: str = "all",
        league: Optional[int] = None,
        activity: str = "all",
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            where_clauses = []
            params: list[Any] = []
            param_idx = 0

            if status == "bots":
                where_clauses.append("(COALESCE(u.is_bot, FALSE) = TRUE)")
            else:
                where_clauses.append("(COALESCE(u.is_bot, FALSE) = FALSE)")

            if query:
                if query.isdigit():
                    param_idx += 1
                    where_clauses.append(f"(u.user_id = ${param_idx})")
                    params.append(int(query))
                else:
                    param_idx += 1
                    where_clauses.append(
                        f"(LOWER(u.username) LIKE LOWER(${param_idx}) OR LOWER(u.first_name) LIKE LOWER(${param_idx}))"
                    )
                    params.append(f"%{query}%")

            if status == "active":
                where_clauses.append("(u.is_banned = FALSE AND u.status = 'active')")
            elif status == "banned":
                where_clauses.append("(u.is_banned = TRUE)")
            elif status == "warned":
                where_clauses.append("(u.warnings_count > 0)")

            if league is not None:
                param_idx += 1
                where_clauses.append(f"(u.league = ${param_idx})")
                params.append(league)

            now = datetime.utcnow()
            if activity == "active_24h":
                where_clauses.append(
                    "(EXISTS (SELECT 1 FROM user_sessions s WHERE s.user_id = u.user_id AND s.started_at >= $__now24h__))"
                )
            elif activity == "active_7d":
                where_clauses.append(
                    "(EXISTS (SELECT 1 FROM user_sessions s WHERE s.user_id = u.user_id AND s.started_at >= $__now7d__))"
                )
            elif activity == "dormant_7d":
                where_clauses.append(
                    "(NOT EXISTS (SELECT 1 FROM user_sessions s WHERE s.user_id = u.user_id AND s.started_at >= $__now7d__))"
                )
            elif activity == "dormant_30d":
                where_clauses.append(
                    "(NOT EXISTS (SELECT 1 FROM user_sessions s WHERE s.user_id = u.user_id AND s.started_at >= $__now30d__))"
                )
            elif activity == "paying":
                where_clauses.append(
                    "(EXISTS (SELECT 1 FROM payments p WHERE p.user_id = u.user_id AND p.status = 'succeeded'))"
                )

            # Resolve all named markers
            now_24h = now - timedelta(hours=24)
            now_7d = now - timedelta(days=7)
            now_30d = now - timedelta(days=30)

            def _resolve_markers(text: str, all_params: list[Any]) -> str:
                while "$__sessions_since__" in text:
                    all_params.append(now_30d)
                    text = text.replace("$__sessions_since__", f"${len(all_params)}", 1)
                while "$__now24h__" in text:
                    all_params.append(now_24h)
                    text = text.replace("$__now24h__", f"${len(all_params)}", 1)
                while "$__now7d__" in text:
                    all_params.append(now_7d)
                    text = text.replace("$__now7d__", f"${len(all_params)}", 1)
                while "$__now30d__" in text:
                    all_params.append(now_30d)
                    text = text.replace("$__now30d__", f"${len(all_params)}", 1)
                return text

            where_sql = ""
            if where_clauses:
                resolved = []
                resolved_params: list[Any] = []
                for clause in where_clauses:
                    clause = _resolve_markers(clause, resolved_params)
                    resolved.append(clause)
                where_sql = "WHERE " + " AND ".join(resolved)
                params.extend(resolved_params)

            count_sql = f"SELECT COUNT(*) FROM users u {where_sql}"
            count_params = list(params)
            count_sql = _resolve_markers(count_sql, count_params)
            total = await self.fetchval(count_sql, *count_params)
            total = int(total or 0)

            limit_clause = f"LIMIT {int(limit)} OFFSET {int(offset)}"
            data_sql = f"""
                SELECT
                    u.user_id,
                    u.username,
                    u.first_name,
                    COALESCE(u.last_name, '') as last_name,
                    u.extra_pass,
                    u.trophies,
                    u.league,
                    u.gems,
                    u.coins,
                    u.keys,
                    u.stars,
                    COALESCE(u.is_bot, FALSE) AS is_bot,
                    u.is_banned,
                    u.warnings_count,
                    u.status,
                    u.reg_date,
                    u.updated_at as last_seen_raw,
                    (SELECT COUNT(*) FROM user_sessions s WHERE s.user_id = u.user_id AND s.started_at >= $__sessions_since__) as sessions_30d,
                    (SELECT COUNT(*) FROM payments p WHERE p.user_id = u.user_id AND p.status = 'succeeded') as purchases_total,
                    (SELECT COALESCE(SUM(p.amount), 0) FROM payments p WHERE p.user_id = u.user_id AND p.status = 'succeeded') as revenue_total
                FROM users u
                {where_sql}
                ORDER BY u.trophies DESC
                {limit_clause}
            """
            # Note: data_sql is rebuilt here, we resolve markers on its final form
            data_params = list(params)
            data_sql = _resolve_markers(data_sql, data_params)
            rows = await self.fetch(data_sql, *data_params)

            players = []
            for r in rows:
                rd = dict(r)
                if rd.get("reg_date") and isinstance(rd["reg_date"], datetime):
                    rd["reg_date"] = rd["reg_date"].isoformat()
                last_seen = rd.pop("last_seen_raw", None)
                if last_seen and isinstance(last_seen, datetime):
                    rd["last_seen"] = last_seen.isoformat()
                else:
                    rd["last_seen"] = last_seen.isoformat() if last_seen else None
                rd["last_name"] = rd.get("last_name") or ""
                rd["sessions_30d"] = int(rd.get("sessions_30d") or 0)
                rd["purchases_total"] = int(rd.get("purchases_total") or 0)
                rd["revenue_total"] = float(rd.get("revenue_total") or 0)
                rd["is_bot"] = bool(rd.get("is_bot", False))
                players.append(rd)

            return {"players": players, "total": total}
        except Exception:
            import logging
            logging.getLogger(__name__).error("search_admin_players failed", exc_info=True)
            return {"error": "search_failed", "players": [], "total": 0}

    async def get_admin_player_detail(self, user_id: int) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            profile = await self.fetchrow(
                """
                SELECT user_id, username, first_name, last_name,
                       extra_pass, extra_pass_expires_at,
                       trophies, max_trophies, league,
                       gems, coins, keys, stars, energy,
                       COALESCE(is_bot, FALSE) AS is_bot,
                       is_banned, ban_reason, banned_until,
                       warnings_count, status,
                       reg_date, created_at, updated_at
                FROM users WHERE user_id = $1
                """,
                user_id,
            )
            if not profile:
                return {"error": "user_not_found"}

            prof = dict(profile)
            for field in ("reg_date", "created_at", "updated_at", "extra_pass_expires_at", "banned_until"):
                if prof.get(field) and isinstance(prof[field], datetime):
                    prof[field] = prof[field].isoformat()

            sessions = await self.fetch(
                """
                SELECT session_id, source, started_at, ended_at, duration_seconds,
                       battles_played, cases_opened
                FROM user_sessions
                WHERE user_id = $1
                ORDER BY started_at DESC
                LIMIT 20
                """,
                user_id,
            )
            sessions_list = []
            for s in sessions:
                sd = dict(s)
                for f in ("started_at", "ended_at"):
                    if sd.get(f) and isinstance(sd[f], datetime):
                        sd[f] = sd[f].isoformat()
                sessions_list.append(sd)

            payments = await self.fetch(
                """
                SELECT id, payment_id, amount, currency, description,
                       status, created_at
                FROM payments
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT 20
                """,
                user_id,
            )
            payments_list = []
            for p in payments:
                pd = dict(p)
                if pd.get("created_at") and isinstance(pd["created_at"], datetime):
                    pd["created_at"] = pd["created_at"].isoformat()
                payments_list.append(pd)

            econ_events = await self.fetch(
                """
                SELECT id, event_type, resource, amount, source, metadata, created_at
                FROM economy_events
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT 50
                """,
                user_id,
            )
            econ_list = []
            for e in econ_events:
                ed = dict(e)
                if ed.get("created_at") and isinstance(ed["created_at"], datetime):
                    ed["created_at"] = ed["created_at"].isoformat()
                if ed.get("metadata") and isinstance(ed.get("metadata"), str):
                    try:
                        ed["metadata"] = json.loads(ed["metadata"])
                    except Exception:
                        pass
                econ_list.append(ed)

            battles = await self.fetch(
                """
                SELECT id, match_id, winner_user_id, loser_user_id,
                       p1_trophy_change, p2_trophy_change,
                       duration_seconds, turns_count, created_at
                FROM battle_summary
                WHERE p1_user_id = $1 OR p2_user_id = $1
                ORDER BY created_at DESC
                LIMIT 20
                """,
                user_id,
            )
            battles_list = []
            for b in battles:
                bd = dict(b)
                if bd.get("created_at") and isinstance(bd["created_at"], datetime):
                    bd["created_at"] = bd["created_at"].isoformat()
                battles_list.append(bd)

            admin_actions = await self.fetch(
                """
                SELECT id, admin_user_id, target_user_id, action_type,
                       reason, payload, created_at
                FROM admin_account_actions
                WHERE target_user_id = $1
                ORDER BY created_at DESC
                LIMIT 50
                """,
                user_id,
            )
            admin_actions_list = []
            for a in admin_actions:
                ad = dict(a)
                if ad.get("created_at") and isinstance(ad["created_at"], datetime):
                    ad["created_at"] = ad["created_at"].isoformat()
                if ad.get("payload") and isinstance(ad.get("payload"), str):
                    try:
                        ad["payload"] = json.loads(ad["payload"])
                    except Exception:
                        pass
                admin_actions_list.append(ad)

            return _json_safe({
                "profile": prof,
                "balances": {
                    "gems": int(prof.get("gems") or 0),
                    "coins": int(prof.get("coins") or 0),
                    "keys": int(prof.get("keys") or 0),
                    "stars": int(prof.get("stars") or 0),
                    "energy": int(prof.get("energy") or 0),
                },
                "sessions": sessions_list,
                "payments": payments_list,
                "economy_events": econ_list,
                "battles": battles_list,
                "admin_actions": admin_actions_list,
            })
        except Exception:
            import logging
            logging.getLogger(__name__).error("get_admin_player_detail failed", exc_info=True)
            return {"error": "detail_failed"}

    # ── Players Admin: actions ──

    async def _create_admin_mail(
        self,
        target_user_id: int,
        subject: str,
        text: str,
        *,
        icon: str = "🛡️",
        category: str = "system",
        attachments: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            result = await self.create_mail(
                user_id=target_user_id,
                sender="Администрация ExtraArena",
                subject=subject,
                text=text,
                category=category,
                icon=icon,
                attachments=attachments or {},
            )
            if not result.get("success"):
                logging.getLogger(__name__).warning(
                    "admin mail failed: user_id=%s subject=%s error=%s",
                    target_user_id,
                    subject,
                    result.get("error", "unknown"),
                )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "admin mail failed: user_id=%s subject=%s",
                target_user_id, subject, exc_info=True,
            )

    async def admin_update_user_account(
        self,
        admin_user_id: int,
        target_user_id: int,
        fields: Dict[str, Any],
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", target_user_id)
            if not exists:
                return {"error": "user_not_found"}

            allowed: dict[str, str] = {
                "username": "username",
                "first_name": "first_name",
                "last_name": "last_name",
                "trophies": "trophies",
                "max_trophies": "max_trophies",
                "league": "league",
                "status": "status",
                "energy": "energy",
            }
            updates: list[str] = []
            values: list[Any] = []
            normalized: dict[str, Any] = {}
            for key, column in allowed.items():
                if key not in fields:
                    continue
                value = fields.get(key)
                if key in {"trophies", "max_trophies", "league", "energy"}:
                    value = max(0, int(value or 0))
                elif key == "status":
                    value = str(value or "active").strip().lower()
                    if value not in {"active", "warn", "banned"}:
                        return {"error": "invalid_status"}
                    updates.append(f"is_banned = ${len(values) + 1}")
                    values.append(value == "banned")
                elif value is not None:
                    value = str(value).strip()
                updates.append(f"{column} = ${len(values) + 1}")
                values.append(value)
                normalized[key] = value

            if not updates:
                return {"error": "no_valid_fields"}

            values.append(target_user_id)
            await self.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE user_id = ${len(values)}",
                *values,
            )
            await self.record_admin_account_action(
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                action_type="update_account",
                reason=reason,
                payload={"fields": normalized},
            )
            await self._create_admin_mail(
                target_user_id,
                "Данные аккаунта обновлены",
                "Администрация обновила данные вашего аккаунта."
                + (f"\nПричина: {reason}" if reason else ""),
                attachments={"fields": list(normalized.keys())},
            )
            return {"status": "ok", "action": "update_account", "fields": normalized}
        except Exception:
            import logging
            logging.getLogger(__name__).error("admin_update_user_account failed", exc_info=True)
            return {"error": "update_account_failed"}

    async def admin_ban_user(
        self,
        admin_user_id: int,
        target_user_id: int,
        reason: Optional[str] = None,
        until: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", target_user_id)
            if not exists:
                return {"error": "user_not_found"}

            await self.execute(
                "UPDATE users SET is_banned = TRUE, ban_reason = $2, banned_until = $3, status = 'banned' WHERE user_id = $1",
                target_user_id, reason, until,
            )

            await self.record_admin_account_action(
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                action_type="ban",
                reason=reason,
                payload={"until": until.isoformat() if until else None},
            )
            until_text = f" до {until.isoformat()}" if until else ""
            await self._create_admin_mail(
                target_user_id,
                "Аккаунт заблокирован",
                f"Ваш аккаунт был заблокирован{until_text}."
                + (f"\nПричина: {reason}" if reason else ""),
                icon="⛔",
            )
            return {"status": "ok", "action": "ban"}
        except Exception:
            import logging
            logging.getLogger(__name__).error("admin_ban_user failed", exc_info=True)
            return {"error": "ban_failed"}

    async def admin_unban_user(
        self,
        admin_user_id: int,
        target_user_id: int,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", target_user_id)
            if not exists:
                return {"error": "user_not_found"}

            await self.execute(
                "UPDATE users SET is_banned = FALSE, ban_reason = NULL, banned_until = NULL, status = 'active' WHERE user_id = $1",
                target_user_id,
            )

            await self.record_admin_account_action(
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                action_type="unban",
                reason=reason,
            )
            await self._create_admin_mail(
                target_user_id,
                "Блокировка снята",
                "Ваш аккаунт снова активен."
                + (f"\nКомментарий: {reason}" if reason else ""),
                icon="✅",
            )
            return {"status": "ok", "action": "unban"}
        except Exception:
            import logging
            logging.getLogger(__name__).error("admin_unban_user failed", exc_info=True)
            return {"error": "unban_failed"}

    async def admin_warn_user(
        self,
        admin_user_id: int,
        target_user_id: int,
        reason: str,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", target_user_id)
            if not exists:
                return {"error": "user_not_found"}

            await self.execute(
                "UPDATE users SET warnings_count = warnings_count + 1 WHERE user_id = $1",
                target_user_id,
            )

            await self.record_admin_account_action(
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                action_type="warn",
                reason=reason,
            )
            await self._create_admin_mail(
                target_user_id,
                "Вы получили предупреждение",
                "Администрация вынесла предупреждение вашему аккаунту."
                + (f"\nПричина: {reason}" if reason else ""),
                icon="⚠️",
            )
            return {"status": "ok", "action": "warn"}
        except Exception:
            import logging
            logging.getLogger(__name__).error("admin_warn_user failed", exc_info=True)
            return {"error": "warn_failed"}

    async def admin_note_user(
        self,
        admin_user_id: int,
        target_user_id: int,
        note: str,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", target_user_id)
            if not exists:
                return {"error": "user_not_found"}

            await self.record_admin_account_action(
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                action_type="note",
                reason=note,
            )
            return {"status": "ok", "action": "note"}
        except Exception:
            import logging
            logging.getLogger(__name__).error("admin_note_user failed", exc_info=True)
            return {"error": "note_failed"}

    async def admin_adjust_resource(
        self,
        admin_user_id: int,
        target_user_id: int,
        resource: str,
        amount: float,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            resource_columns = {
                "gems": "gems",
                "coins": "coins",
                "keys": "keys",
                "stars": "stars",
            }
            column = resource_columns.get(resource)
            if column is None:
                return {"error": f"invalid_resource, allowed: {', '.join(sorted(resource_columns))}"}

            exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", target_user_id)
            if not exists:
                return {"error": "user_not_found"}

            amount = int(float(amount))
            if amount == 0:
                return {"error": "amount_cannot_be_zero"}

            if amount > 0:
                action = "grant"
                event_type = "earn"
                sign = "+"
            else:
                action = "deduct"
                event_type = "spend"
                sign = "-"

            # Prevent balance going below 0
            if amount < 0:
                current = await self.fetchval(
                    f"SELECT {column} FROM users WHERE user_id = $1",
                    target_user_id,
                ) or 0
                if current + amount < 0:
                    return {"error": f"insufficient_{resource}", "current": int(current), "requested": abs(int(amount))}

            await self.execute(
                f"UPDATE users SET {column} = GREATEST(0, {column} + $1) WHERE user_id = $2",
                amount, target_user_id,
            )

            await self.track_economy_event(
                user_id=target_user_id,
                event_type=event_type,
                resource=resource,
                amount=abs(amount),
                source="admin_action",
                metadata={
                    "admin_user_id": admin_user_id,
                    "reason": reason or "",
                },
            )

            await self.record_admin_account_action(
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                action_type=action,
                reason=reason,
                payload={"resource": resource, "amount": amount},
            )
            if amount > 0:
                await self._create_admin_mail(
                    target_user_id,
                    "Вам выдана награда",
                    f"Администрация начислила вам {amount} {resource}."
                    + (f"\nПричина: {reason}" if reason else ""),
                    icon="🎁",
                    category="rewards",
                    attachments={"resource": resource, "amount": amount},
                )
            return {"status": "ok", "action": action, "resource": resource, "amount": amount}
        except Exception:
            import logging
            logging.getLogger(__name__).error("admin_adjust_resource failed", exc_info=True)
            return {"error": "adjust_resource_failed"}

    async def admin_set_extra_pass(
        self,
        admin_user_id: int,
        target_user_id: int,
        mode: str,
        days: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        try:
            if mode not in ("inactive", "active", "ultra"):
                return {"error": "invalid_mode, allowed: inactive, active, ultra"}

            exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", target_user_id)
            if not exists:
                return {"error": "user_not_found"}

            if mode == "inactive":
                await self.execute(
                    "UPDATE users SET extra_pass = 'inactive', extra_pass_expires_at = NULL WHERE user_id = $1",
                    target_user_id,
                )
                payload = {"mode": "inactive"}
            else:
                if not days or days < 1:
                    return {"error": "days_required_for_active_mode"}
                expires = datetime.utcnow() + timedelta(days=int(days))
                await self.execute(
                    "UPDATE users SET extra_pass = $2, extra_pass_expires_at = $3 WHERE user_id = $1",
                    target_user_id, mode, expires,
                )
                payload = {"mode": mode, "days": days, "expires_at": expires.isoformat()}

            await self.record_admin_account_action(
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                action_type="set_extra_pass",
                reason=reason,
                payload=payload,
            )
            if mode == "inactive":
                subject = "ExtraPass отключен"
                text = "Администрация отключила ExtraPass на вашем аккаунте."
            else:
                subject = "Вам выдан ExtraPass"
                text = f"Администрация выдала вам ExtraPass ({mode}) на {days} дн."
            await self._create_admin_mail(
                target_user_id,
                subject,
                text + (f"\nПричина: {reason}" if reason else ""),
                icon="⭐",
                category="rewards" if mode != "inactive" else "system",
                attachments=payload,
            )
            return {"status": "ok", "action": "set_extra_pass", "mode": mode}
        except Exception:
            import logging
            logging.getLogger(__name__).error("admin_set_extra_pass failed", exc_info=True)
            return {"error": "extra_pass_failed"}

    async def export_train_v2_battle_dataset(
        self,
        days: int = 30,
        limit: int = 5000,
        include_players: bool = False,
    ) -> list[Dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        days = max(int(days or 30), 1)
        limit = min(max(int(limit or 5000), 1), 50000)
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = await self.fetch(
            """
            SELECT ba.id, ba.battle_id, ba.turn_number, ba.acting_player,
                   ba.acting_user_id, ba.is_bot, ba.state_json, ba.action_json,
                   ba.quality_score, ba.created_at,
                   bs.game_mode, bs.match_type, bs.winner_user_id,
                   bs.p1_user_id, bs.p2_user_id, bs.metadata AS battle_metadata
            FROM battle_actions ba
            LEFT JOIN battle_summary bs ON bs.match_id = ba.battle_id
            WHERE ba.created_at >= $1
            ORDER BY ba.created_at DESC
            LIMIT $2
            """,
            since,
            limit,
        )
        dataset = []
        for r in rows:
            acting_user_id = r["acting_user_id"] if include_players else None
            winner_user_id = r["winner_user_id"] if include_players else None
            sample_won = (r["winner_user_id"] == r["acting_user_id"]) if r["winner_user_id"] and r["acting_user_id"] else None
            dataset.append(_json_safe({
                "format": "train_v2_admin_battle_action_jsonl_v1",
                "id": r["id"],
                "battle_id": r["battle_id"],
                "turn_number": r["turn_number"],
                "acting_player": r["acting_player"],
                "acting_user_id": acting_user_id,
                "is_bot": r["is_bot"],
                "game_mode": r["game_mode"] or r["match_type"] or "classic",
                "winner_user_id": winner_user_id,
                "won": sample_won,
                "state_json": r["state_json"] or {},
                "action_json": r["action_json"] or {},
                "quality_score": r["quality_score"],
                "battle_metadata": r["battle_metadata"] or {},
                "created_at": r["created_at"],
            }))
        return dataset


    async def get_welcome_status(self, user_id: int) -> dict[str, Any]:
        """Получить статус приветствия для пользователя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        
        # Проверяем, существует ли пользователь
        user_exists = await self.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)
        if not user_exists:
            return {"should_show": True}
        
        # Получаем статус welcome_shown из настроек
        welcome_shown = await self.fetchval(
            "SELECT welcome_shown FROM user_settings WHERE user_id = $1",
            user_id
        )
        
        # Если настройки не найдены, считаем, что приветствие нужно показать
        if welcome_shown is None:
            return {"should_show": True}
        
        return {"should_show": not welcome_shown}

    async def mark_welcome_shown(self, user_id: int) -> None:
        """Отметить, что приветствие было показано пользователю."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        
        # Убеждаемся, что настройки существуют
        await self.execute(
            """
            INSERT INTO user_settings (user_id, welcome_shown)
            VALUES ($1, true)
            ON CONFLICT (user_id) DO UPDATE
            SET welcome_shown = true, updated_at = NOW()
            """,
            user_id
        )

    # --- Match mode overrides ---

    async def _ensure_match_mode_overrides_table(self) -> bool:
        """Создать таблицу override'ов доступности режимов."""
        changed = False
        table_exists = await self.fetchval("SELECT to_regclass('public.match_mode_overrides')")
        if not table_exists:
            await self.execute(
                """
                CREATE TABLE match_mode_overrides (
                    mode_id TEXT PRIMARY KEY,
                    enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata JSONB NOT NULL DEFAULT '{}'
                )
                """
            )
            changed = True
        columns = await self._get_columns("match_mode_overrides")
        changed |= await self._add_column_if_missing(
            "match_mode_overrides", columns, "mode_id TEXT PRIMARY KEY"
        )
        changed |= await self._add_column_if_missing(
            "match_mode_overrides", columns, "enabled BOOLEAN NOT NULL DEFAULT TRUE"
        )
        changed |= await self._add_column_if_missing(
            "match_mode_overrides", columns, "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )
        changed |= await self._add_column_if_missing(
            "match_mode_overrides", columns, "metadata JSONB NOT NULL DEFAULT '{}'"
        )
        return changed

    async def get_match_mode_overrides(self) -> list[dict[str, Any]]:
        """Вернуть все overrides режимов."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        rows = await self.fetch("SELECT mode_id, enabled, updated_at, metadata FROM match_mode_overrides")
        return [
            {
                "mode_id": r["mode_id"],
                "enabled": r["enabled"],
                "updated_at": r["updated_at"],
                "metadata": r["metadata"] or {},
            }
            for r in rows
        ]

    async def is_match_mode_enabled(self, mode_id: str) -> bool:
        """Проверить доступность режима с учётом DB override."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        row = await self.fetchrow(
            "SELECT enabled FROM match_mode_overrides WHERE mode_id = $1", mode_id
        )
        if row is not None:
            return bool(row["enabled"])
        return True  # default: enabled if no override row

    async def set_match_mode_enabled(self, mode_id: str, enabled: bool) -> None:
        """Установить override доступности режима."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        await self.execute(
            """
            INSERT INTO match_mode_overrides (mode_id, enabled, updated_at, metadata)
            VALUES ($1, $2, NOW(), '{}')
            ON CONFLICT (mode_id) DO UPDATE
            SET enabled = EXCLUDED.enabled,
                updated_at = NOW()
            """,
            mode_id,
            enabled,
        )
