from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Dict, TYPE_CHECKING

# asyncpg требуется только для реальной работы с БД.
# В юнит-тестах модуль может отсутствовать, поэтому подменяем импорт,
# чтобы логика карт и статы могли импортироваться без установки asyncpg.
try:  # pragma: no cover - ветка с отсутствием asyncpg проверяется непрямо
    import asyncpg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - альтернативный путь при локальных тестах
    asyncpg = None  # type: ignore

from infrastructure.config import DECK_SIZE, MAX_FREE_DECK_PRESETS, MAX_TOTAL_DECK_PRESETS, DatabaseSettings, get_league_by_trophies_fn, LEAGUE_CONFIG


# Версию схемы повышаем при изменении структуры таблиц
SCHEMA_VERSION = 24

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
    return []


def calculate_card_stats(card_obj: Any, level: int) -> dict[str, Any]:
    """
    Рассчитываем текущее значение атаки/хп/маны для карты.
    Формула роста: Base * (1 + growth) ** (level - 1).
    
    Для героев (card_type = 'hero') используются базовые значения без формул роста.
    Герой - это статичная башня с фиксированным HP и атакой = 0.
    """
    safe_level = max(1, int(level or 1))

    def _get(field: str, default: int = 0) -> int:
        """Достаем поле из dataclass/словаря/объекта, чтобы не дублировать код."""
        if isinstance(card_obj, dict):
            return int(card_obj.get(field, default) or default)
        return int(getattr(card_obj, field, default) or default)

    # Проверяем, является ли карта героем
    card_type = (
        card_obj.get("card_type") if isinstance(card_obj, dict) else getattr(card_obj, "card_type", "warrior")
    ) or "warrior"
    
    base_attack = _get("base_attack", 0)
    base_hp = _get("base_hp", 0)
    mana_cost = _get("mana_cost", 0)
    mechanics = _normalize_mechanics(
        card_obj.get("mechanics") if isinstance(card_obj, dict) else getattr(card_obj, "mechanics", None)
    )

    # Для карт-героев используем базовые значения без формул роста
    if card_type == 'hero':
        return {
            "attack": 0,  # Герой-башня не может атаковать
            "hp": base_hp or 30,
            "mana": mana_cost,
            "mechanics": mechanics,
            "growth": 0.0,  # Герои не растут по уровням
        }

    # Для обычных карт применяем формулу роста
    rarity_raw = (
        card_obj.get("rarity") if isinstance(card_obj, dict) else getattr(card_obj, "rarity", "")
    ) or ""
    rarity = str(rarity_raw).lower()
    growth = RARITY_STATS.get(rarity, 0.10)

    attack = int(round(base_attack * ((1 + growth) ** (safe_level - 1))))
    hp = int(round(base_hp * ((1 + growth) ** (safe_level - 1))))

    return {
        "attack": attack,
        "hp": hp,
        "mana": mana_cost,
        "mechanics": mechanics,
        "growth": growth,
    }


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
        )

    def to_dict(self) -> dict[str, Any]:
        """Отдаем сериализованный словарь (удобно для JSON)."""
        return asdict(self)

    def get_current_stats(self, level: int) -> dict[str, Any]:
        """
        Высчитываем актуальные атак/хп/ману для боевого движка.
        Формула роста: Base * (1 + level_multiplier) ** (level - 1).
        
        Для героев (card_type = 'hero') используются базовые значения без формул роста.
        Герой - это статичная башня с фиксированным HP и атакой = 0.
        """
        safe_level = max(1, int(level or 1))
        
        # Для карт-героев используем базовые значения без формул роста
        if self.card_type == 'hero':
            return {
                "attack": 0,  # Герой-башня не может атаковать
                "hp": int(self.base_hp or 30),
                "mana": int(self.mana_cost or 0),
                "mechanics": _normalize_mechanics(self.mechanics),
                "growth": 0.0,  # Герои не растут по уровням
                "level": safe_level,
            }
        
        # Для обычных карт применяем формулу роста
        attack = int(round(self.base_attack * ((1 + self.level_multiplier) ** (safe_level - 1))))
        hp = int(round(self.base_hp * ((1 + self.level_multiplier) ** (safe_level - 1))))

        return {
            "attack": attack,
            "hp": hp,
            "mana": int(self.mana_cost or 0),
            "mechanics": _normalize_mechanics(self.mechanics),
            "growth": self.level_multiplier,
            "level": safe_level,
        }




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
        global_chat_changed = await self._ensure_global_chat_table()
        community_posts_changed = await self._ensure_community_posts_table()
        post_likes_changed = await self._ensure_post_likes_table()
        reward_tracks_changed = await self._ensure_reward_tracks_table()
        claimed_rewards_changed = await self._ensure_claimed_rewards_table()
        seasons_changed = await self._ensure_seasons_table()
        shop_sets_changed = await self._ensure_shop_sets_table()
        payments_changed = await self._ensure_payments_table()
        friend_invites_changed = await self._ensure_friend_invites_table()
        friend_requests_changed = await self._ensure_friend_requests_table()
        generator_changed = await self._ensure_generator_state_table()

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
            or global_chat_changed
            or community_posts_changed
            or post_likes_changed
            or reward_tracks_changed
            or claimed_rewards_changed
            or seasons_changed
            or shop_sets_changed
            or payments_changed
            or friend_invites_changed
            or friend_requests_changed
            or generator_changed
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
            
            # Выдаем первую карту (ID 9) новому пользователю
            try:
                await self.add_card_to_user(user_id, 9)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Не удалось выдать первую карту пользователю {user_id}: {e}")
            
            # Выдаем приветственные бонусы: 50 гемов и 200 монет
            try:
                await self.execute(
                    "UPDATE users SET gems = gems + 50, coins = coins + 200 WHERE user_id = $1",
                    user_id
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Не удалось выдать приветственные бонусы пользователю {user_id}: {e}")

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
        Держим диапазон отдельно от игроков, чтобы избежать коллизий и коллизий с Telegram ID.
        """
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        next_id = await self.fetchval(
            """
            SELECT COALESCE(MAX(user_id), 899999999) + 1
            FROM users
            WHERE user_id >= 900000000
            """
        )
        return int(next_id or 900000000)

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
        
        # Обновляем никнейм
        await self.execute(
            """
            UPDATE profiles
            SET custom_nickname = $1,
                nickname_changed = TRUE
            WHERE user_id = $2
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

        players = await self.fetchval("SELECT COUNT(*) FROM users")
        extra_pass_active = await self.fetchval(
            "SELECT COUNT(*) FROM users WHERE extra_pass = 'active'"
        )
        total_trophies = await self.fetchval("SELECT COALESCE(SUM(trophies), 0) FROM users")
        max_trophies_global = await self.fetchval(
            "SELECT COALESCE(MAX(max_trophies), 0) FROM users"
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
                    ads_enabled BOOLEAN NOT NULL DEFAULT true,
                    sound_music BOOLEAN NOT NULL DEFAULT true,
                    sound_sfx BOOLEAN NOT NULL DEFAULT true,
                    social_block_friend_requests BOOLEAN NOT NULL DEFAULT false,
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
            "notif_generator",
            "ads_enabled", "sound_music", "sound_sfx", "social_block_friend_requests",
            "welcome_shown",
            "starter_pack_used", "particles_rotation_cards", "particles_rotation_date",
            "particles_purchased_today", "wins_since_last_case",
        }

        updates = []
        values = []
        for key, value in kwargs.items():
            if key in valid_keys:
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
                return record.get("unread_count", 0)
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
            SET gems = COALESCE(gems, 0) + $1,
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

        return changed

    async def save_battle_result(self, **kwargs) -> None:
        """Сохранить результат боя."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        await self.execute(
            """
            INSERT INTO battle_results (match_id, winner_id, loser_id, winner_score, loser_score, match_duration, match_type, p1_id, p2_id, p1_trophy_change, p2_trophy_change, turns_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
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

    async def get_battle_history(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        """Получить историю боёв игрока."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT b.id, b.match_id, b.winner_id, b.loser_id,
                   b.p1_id, b.p2_id,
                   b.p1_trophy_change, b.p2_trophy_change,
                   b.match_duration, b.match_type, b.turns_count, b.created_at,
                   CASE WHEN b.p1_id = $1 THEN b.p2_id ELSE b.p1_id END AS opponent_id
            FROM battle_results b
            WHERE b.p1_id = $1 OR b.p2_id = $1
            ORDER BY b.created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )

        opponent_ids = [r["opponent_id"] for r in rows if r["opponent_id"]]
        opponent_names: dict[int, str] = {}
        if opponent_ids:
            opp_rows = await self.fetch(
                "SELECT user_id, COALESCE(first_name, username, 'Игрок') AS name FROM users WHERE user_id = ANY($1)",
                opponent_ids,
            )
            opponent_names = {r["user_id"]: r["name"] for r in opp_rows}

        return [
            {
                "battle_id": r["id"],
                "opponent_id": r["opponent_id"],
                "opponent_name": opponent_names.get(r["opponent_id"], "Игрок"),
                "result": (
                    "win" if r["winner_id"] and r["p1_id"] == user_id and r["winner_id"] == user_id
                    else "win" if r["winner_id"] and r["p2_id"] == user_id and r["winner_id"] == user_id
                    else "draw" if r["winner_id"] is None
                    else "lose"
                ),
                "trophies_change": (
                    r["p1_trophy_change"] if r["p1_id"] == user_id
                    else r["p2_trophy_change"]
                ),
                "mode": r["match_type"] or "classic",
                "duration_seconds": r["match_duration"] or 0,
                "turns_count": r["turns_count"] or 0,
                "created_at": r["created_at"].isoformat() if isinstance(r["created_at"], datetime) else str(r["created_at"]),
            }
            for r in rows
        ]

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
                (SELECT COUNT(*) FROM battle_results
                 WHERE p1_id = u.user_id OR p2_id = u.user_id) as battle_count,
                (SELECT COUNT(*) FROM battle_results
                 WHERE (p1_id = u.user_id AND winner_id = u.user_id)
                    OR (p2_id = u.user_id AND winner_id = u.user_id)) as win_count
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
            "user_mail", columns, "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        )

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
        text: str,
    ) -> dict[str, Any]:
        """Создать письмо пользователю. Возвращает dict с результатом."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        try:
            await self.execute(
                """
                INSERT INTO user_mail (user_id, sender, subject, text)
                VALUES ($1, $2, $3, $4)
                """,
                user_id,
                sender,
                subject,
                text,
            )
            return {"success": True}
        except Exception as e:
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
            SELECT id, user_id, sender, subject, text, is_read, category, created_at
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

        result = []
        for row in rows:
            mail = dict(row)
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

    async def get_cases_list(self) -> list[dict[str, Any]]:
        """Заглушка для списка кейсов."""
        return []

    async def get_user_cases(self, user_id: int) -> list[dict[str, Any]]:
        """Заглушка для кейсов пользователя."""
        return []

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
                    name TEXT NOT NULL DEFAULT '',
                    start_date TIMESTAMPTZ,
                    end_date TIMESTAMPTZ,
                    is_active BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )
            changed = True

        columns = await self._get_columns("seasons")

        changed |= await self._add_column_if_missing(
            "seasons", columns, "name TEXT NOT NULL DEFAULT ''"
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

    async def get_shop_sets(self, active_only: bool = True) -> list[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        query = "SELECT * FROM shop_sets"
        if active_only:
            query += " WHERE is_active = TRUE"
        query += " ORDER BY created_at DESC"
        rows = await self.fetch(query)
        return [dict(r) for r in rows]

    async def get_shop_set(self, set_id: int) -> Optional[dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        row = await self.fetchrow(
            "SELECT * FROM shop_sets WHERE id = $1", set_id
        )
        return dict(row) if row else None

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
                if key == "rewards" and not isinstance(value, str):
                    value = _json.dumps(value)
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
        import json as _json

        row = await self.fetchrow(
            "SELECT id, rewards FROM shop_sets WHERE id = $1 AND is_active = TRUE",
            set_id,
        )
        if not row:
            return {"success": False, "error": "set_not_found"}

        rewards_data = row["rewards"]
        if isinstance(rewards_data, str):
            rewards_data = _json.loads(rewards_data)
        if not isinstance(rewards_data, list):
            return {"success": False, "error": "invalid_rewards_format"}

        granted: list[dict[str, Any]] = []

        for reward in rewards_data:
            r_type = reward.get("type")
            amount = int(reward.get("amount", 0))
            card_id = reward.get("card_id")

            if r_type == "gems" and amount > 0:
                await self.execute(
                    "UPDATE users SET gems = gems + $1 WHERE user_id = $2",
                    amount, user_id,
                )
                granted.append({"type": "gems", "amount": amount})

            elif r_type == "coins" and amount > 0:
                await self.execute(
                    "UPDATE users SET coins = coins + $1 WHERE user_id = $2",
                    amount, user_id,
                )
                granted.append({"type": "coins", "amount": amount})

            elif r_type == "keys" and amount > 0:
                await self.increment_user_keys(user_id, amount)
                granted.append({"type": "keys", "amount": amount})

            elif r_type == "case":
                count = max(1, amount)
                await self.increment_user_keys(user_id, count)
                granted.append({"type": "keys", "amount": count})

            elif r_type == "card" and card_id:
                from datetime import datetime, timezone
                await self.execute(
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

            elif r_type == "particles" and amount > 0 and card_id:
                await self.execute(
                    """
                    INSERT INTO user_cards (user_id, card_id, level, particles, obtained_at)
                    VALUES ($1, $2, 1, $3, NOW())
                    ON CONFLICT (user_id, card_id) DO UPDATE
                    SET particles = COALESCE(user_cards.particles, 0) + $3
                    """,
                    user_id, int(card_id), amount,
                )
                granted.append({"type": "particles", "amount": amount, "card_id": int(card_id)})

        return {"success": True, "granted": granted}

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
                    battle_id TEXT DEFAULT NULL
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

    async def create_friend_invite(self, from_user_id: int, to_user_id: int) -> dict[str, Any]:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        try:
            row = await self.fetchrow(
                """
                INSERT INTO friend_invites (from_user_id, to_user_id)
                VALUES ($1, $2)
                RETURNING id, created_at, expires_at
                """,
                from_user_id, to_user_id,
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
        self, invite_id: int, status: str, battle_id: Optional[str] = None
    ) -> None:
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        if battle_id:
            await self.execute(
                "UPDATE friend_invites SET status = $1, battle_id = $2 WHERE id = $3",
                status, battle_id, invite_id,
            )
        else:
            await self.execute(
                "UPDATE friend_invites SET status = $1 WHERE id = $2",
                status, invite_id,
            )

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
                   p.img AS avatar_url
            FROM friend_requests fr
            JOIN users u ON u.user_id = fr.requester_id
            LEFT JOIN profiles p ON p.user_id = fr.requester_id
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
                   p.img AS avatar_url
            FROM friend_requests fr
            JOIN users u ON u.user_id = fr.addressee_id
            LEFT JOIN profiles p ON p.user_id = fr.addressee_id
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
                       p.img AS avatar_url
                FROM friend_requests fr
                JOIN users u ON u.user_id = fr.addressee_id
                LEFT JOIN profiles p ON p.user_id = fr.addressee_id
                WHERE fr.requester_id = $1 AND fr.status = 'accepted'
                UNION
                SELECT fr.requester_id AS friend_id,
                       COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') AS display_name,
                       p.img AS avatar_url
                FROM friend_requests fr
                JOIN users u ON u.user_id = fr.requester_id
                LEFT JOIN profiles p ON p.user_id = fr.requester_id
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
            WITH opponents AS (
                SELECT winner_id AS opponent_id
                FROM battle_results
                WHERE loser_id = $1 AND winner_id IS NOT NULL
                UNION ALL
                SELECT loser_id AS opponent_id
                FROM battle_results
                WHERE winner_id = $1 AND loser_id IS NOT NULL
            ),
            ranked AS (
                SELECT DISTINCT ON (opponent_id) opponent_id
                FROM opponents
                ORDER BY opponent_id
                LIMIT $2
            )
            SELECT r.opponent_id, p.custom_nickname, p.img AS avatar_url,
                   COALESCE(p.custom_nickname, u.first_name, u.username, 'Игрок') AS display_name,
                   (SELECT 1 FROM friend_requests fr
                    WHERE fr.status = 'accepted'
                      AND ((fr.requester_id = $1 AND fr.addressee_id = r.opponent_id)
                           OR (fr.requester_id = r.opponent_id AND fr.addressee_id = $1))
                    LIMIT 1) IS NOT NULL AS is_friend
            FROM ranked r
            JOIN users u ON u.user_id = r.opponent_id
            LEFT JOIN profiles p ON p.user_id = r.opponent_id
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
            GENERATOR_UPGRADE_COST, GENERATOR_MAX_LEVEL,
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
            "claimed": accumulated,
            "total_keys": total_keys or accumulated,
        }

    async def upgrade_generator(self, user_id: int, currency: str) -> dict[str, Any]:
        """Повысить уровень генератора за coins или gems. Транзакционно с FOR UPDATE."""
        from infrastructure.generator_config import (
            GENERATOR_LEVELS, GENERATOR_UPGRADE_COST, GENERATOR_MAX_LEVEL,
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

                if currency not in costs:
                    return {"success": False, "error": "invalid_currency"}

                price = costs[currency]

                if currency == "coins":
                    if (row["coins"] or 0) < price:
                        return {"success": False, "error": "not_enough_coins", "required": price, "have": row["coins"] or 0}
                    await conn.execute(
                        "UPDATE users SET coins = GREATEST(0, coins - $2), updated_at = NOW() WHERE user_id = $1",
                        user_id, price,
                    )
                elif currency == "gems":
                    if (row["gems"] or 0) < price:
                        return {"success": False, "error": "not_enough_gems", "required": price, "have": row["gems"] or 0}
                    await conn.execute(
                        "UPDATE users SET gems = GREATEST(0, gems - $2), updated_at = NOW() WHERE user_id = $1",
                        user_id, price,
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
            "new_level": next_level,
            "currency_spent": currency,
            "amount_spent": price,
            "coins_remaining": coins_after or 0,
            "gems_remaining": gems_after or 0,
        }

    async def check_generator_notifications(self) -> list[dict[str, Any]]:
        """Найти пользователей, которым нужно отправить уведомление о готовых ключах.
        Рассчитывает готовность из last_tick_at + конфиг, НЕ из stored accumulated_keys."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")

        rows = await self.fetch(
            """
            SELECT g.user_id, g.level, g.accumulated_keys, g.last_tick_at, g.notified,
                   u.extra_pass, u.status
            FROM generator_state g
            INNER JOIN users u ON u.user_id = g.user_id
            LEFT JOIN notifications n ON n.user_id = g.user_id AND n.notification_type = 'generator'
            LEFT JOIN user_settings us ON us.user_id = g.user_id
            WHERE u.status = 'active'
              AND (n.sent IS NULL OR n.sent = FALSE)
              AND (us.notif_generator = TRUE OR us.notif_generator IS NULL)
            """
        )

        result = []
        for r in rows:
            accumulated, new_keys, cap, interval_seconds = await self._compute_generator_accumulated(r)
            if accumulated > 0:
                result.append({"user_id": r["user_id"]})

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
                    rarity TEXT NOT NULL CHECK (rarity IN ('common', 'rare', 'superrare', 'epic', 'legendary', 'mythic', 'divine', 'limited', 'unique')),
                    power INT NOT NULL DEFAULT 0,
                    mana_cost INT NOT NULL DEFAULT 3,
                    base_attack INT NOT NULL DEFAULT 100,
                    base_hp INT NOT NULL DEFAULT 100,
                    mechanics JSONB NOT NULL DEFAULT '[]'::jsonb,
                    card_type TEXT NOT NULL DEFAULT 'warrior',
                    image_file_id TEXT,
                    created_by BIGINT,
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
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by, created_at
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
            }
        )
        return card_dict


    async def get_cards_list(self) -> list[dict[str, Any]]:
        """Получить список всех карт."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")

        rows = await self.fetch(
            """
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by, created_at
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
                }
            )
            cards.append(card_dict)
        return cards

    async def get_cards_by_rarity(self, rarity: str) -> list[dict[str, Any]]:
        """Получить все карты указанной редкости."""
        if not self._pool:
            raise RuntimeError("База данных не подключена. Вызовите connect() сначала.")
        rows = await self.fetch(
            """
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by, created_at
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
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id, created_by, created_at
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
                }
            )
            cards.append(card_dict)
        return cards


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
            ("bp_ultra", 42, "gems", 500, '{"original":"case_tier_5","case_tier":5}', True),
            ("bp_ultra", 43, "keys", 4,   None,                                      True),
            ("bp_ultra", 44, "gems", 500, None,                                      True),
            ("bp_ultra", 45, "card", 1,   '{"rarity":["common","rare"]}',            True),
            ("bp_ultra", 45, "gems", 300, None,                                      True),
        ]

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
        """Обновить поля тира. fields может содержать: reward_type, reward_amount, reward_meta, extra_pass_required, is_active."""
        if not self._pool:
            raise RuntimeError("База данных не подключена.")
        import json as _json

        existing = await self.get_reward_track_by_id(reward_id)
        if not existing:
            return {"error": "not_found"}

        updates = []
        params = []
        idx = 1

        for key in ("reward_type", "reward_amount", "extra_pass_required", "is_active"):
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
            SELECT id, name, description, rarity, power, mana_cost, base_attack, base_hp, mechanics, card_type, image_file_id
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
                       c.mechanics
                FROM user_cards uc
                INNER JOIN cards c ON c.id = uc.card_id
                WHERE uc.user_id = $1 AND uc.card_id = $2
                """,
                user_id, card_id
            )
            
            if not user_card:
                return {"success": False, "error": "card_not_found"}
            
            current_level = user_card["level"] or 1
            
            # Проверяем максимальный уровень
            if current_level >= 10:
                return {"success": False, "error": "max_level_reached"}
            
            current_particles = user_card["particles"] or 0
            rarity = user_card["rarity"]
            base_power = user_card["power"]
            base_attack = user_card.get("base_attack") or 0
            base_hp = user_card.get("base_hp") or 0
            mana_cost = user_card.get("mana_cost") or 0
            mechanics = _normalize_mechanics(user_card.get("mechanics"))
            
            # Рассчитываем необходимое количество частиц и монет
            required_particles = self.calculate_upgrade_particles(rarity, current_level)
            required_coins = self.calculate_upgrade_coins(rarity, current_level)
            
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
            new_level = current_level + 1
            
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
                "mana_cost": mana_cost,
                "mechanics": new_stats["mechanics"],
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

