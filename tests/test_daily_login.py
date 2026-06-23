import pytest
from datetime import datetime, timedelta, timezone

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database
from infrastructure.notifications import (
    NOTIFICATION_DEFAULTS,
    NOTIFICATION_SETTING_BY_CATEGORY,
    format_android_notification_title,
    format_notification_message,
    format_telegram_notification_message,
    notification_section,
)


class _AsyncContext:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DailyLoginFakeConnection:
    """Эмулирует asyncpg.Connection для claim_daily_login_reward / get_daily_login_status."""

    def __init__(self, *, available_at=None, claimed=False, streak=0, streak_day=0,
                 reward_type="coins", reward_amount=50, multiplier=1, coins=100, gems=10, stars=3, keys=2):
        self.row = {
            "daily_login_streak": streak,
            "daily_login_streak_day": streak_day,
            "daily_login_available_at": available_at,
            "daily_login_reward_type": reward_type,
            "daily_login_reward_amount": reward_amount,
            "daily_login_multiplier": multiplier,
            "daily_login_claimed": claimed,
        }
        self.balance = {"coins": coins, "gems": gems, "stars": stars, "keys": keys}
        self.executed = []
        self.claim_row_id = 1  # Имитируем успешную вставку в claimed_rewards.
        self._claim_returned = False

    def transaction(self):
        return _AsyncContext()

    async def fetchrow(self, query, *args):
        self.executed.append((query, args))
        if "FOR UPDATE" in query and "daily_login_available_at" in query:
            # claim_daily_login_reward: SELECT ... FOR UPDATE
            return dict(self.row)
        if "SELECT daily_login_available_at" in query and "FOR UPDATE" in query and "daily_login_streak_day" not in query:
            # _advance_daily_login_cycle lock
            return {"daily_login_available_at": self.row["daily_login_available_at"],
                    "daily_login_claimed": self.row["daily_login_claimed"],
                    "daily_login_streak": self.row["daily_login_streak"],
                    "daily_login_streak_day": self.row["daily_login_streak_day"],
                    "daily_login_reward_type": self.row["daily_login_reward_type"],
                    "daily_login_reward_amount": self.row["daily_login_reward_amount"],
                    "daily_login_multiplier": self.row["daily_login_multiplier"]}
        if "INSERT INTO claimed_rewards" in query:
            if self._claim_returned:
                return None  # второй вызов = уже клеймлено
            self._claim_returned = True
            return {"id": self.claim_row_id}
        if "SELECT daily_login_streak, daily_login_streak_day, daily_login_available_at" in query:
            # get_daily_login_status read
            return {
                **dict(self.row),
                "daily_login_notified": False,
                "daily_login_last_claim_at": None,
            }
        return dict(self.row)

    async def execute(self, query, *args):
        self.executed.append((query, args))
        if "UPDATE users SET daily_login_streak" in query and "daily_login_claimed = TRUE" in query:
            # Финальный апдейт после клейма: обновляем серию и следующий цикл.
            self.row["daily_login_streak"] = args[1]
            self.row["daily_login_streak_day"] = args[2]
            self.row["daily_login_available_at"] = args[3]
            self.row["daily_login_reward_type"] = args[4]
            self.row["daily_login_reward_amount"] = args[5]
            self.row["daily_login_multiplier"] = args[6]
            self.row["daily_login_claimed"] = True
        elif "UPDATE users SET daily_login_streak = 0" in query:
            self.row["daily_login_streak"] = 0
            self.row["daily_login_streak_day"] = 0
        elif "SET coins = GREATEST" in query and "COALESCE(coins" in query:
            self.balance["coins"] = max(0, self.balance["coins"] + int(args[0]))
        elif "SET gems = GREATEST" in query and "COALESCE(gems" in query:
            self.balance["gems"] = max(0, self.balance["gems"] + int(args[0]))
        elif "SET stars = GREATEST" in query and "COALESCE(stars" in query:
            self.balance["stars"] = max(0, self.balance["stars"] + int(args[0]))
        elif "SET keys = COALESCE(keys" in query:
            self.balance["keys"] = self.balance["keys"] + int(args[0])
        elif "INSERT INTO economy_events" in query:
            pass  # аудит


class _DailyLoginFakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _AsyncContext(self.conn)


def _db_with_conn(conn):
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    db._pool = _DailyLoginFakePool(conn)
    return db


def test_claim_daily_login_reward_credits_currency_and_sets_next_cycle():
    now = datetime.now(timezone.utc)
    available_at = now - timedelta(minutes=1)  # доступна
    conn = _DailyLoginFakeConnection(
        available_at=available_at, claimed=False, streak=1, streak_day=1,
        reward_type="coins", reward_amount=50, multiplier=1, coins=100,
    )
    db = _db_with_conn(conn)
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(db.claim_daily_login_reward(42))
    assert result["success"] is True
    granted = result["granted"]
    assert granted["reward_type"] == "coins"
    assert granted["reward_amount"] == 50
    # Серия продолжается: streak_day был 1 → новый 2
    assert result["next"]["streak_day"] == 2
    # Баланс монет увеличен
    assert conn.balance["coins"] == 150


def test_claim_daily_login_reward_idempotent_second_call_returns_already_claimed():
    now = datetime.now(timezone.utc)
    available_at = now - timedelta(minutes=1)
    conn = _DailyLoginFakeConnection(
        available_at=available_at, claimed=False, streak=1, streak_day=1,
        reward_type="gems", reward_amount=5, multiplier=1, gems=10,
    )
    db = _db_with_conn(conn)
    import asyncio
    loop = asyncio.get_event_loop()
    first = loop.run_until_complete(db.claim_daily_login_reward(7))
    assert first["success"] is True
    # Второй клейм: claim_row возвращает None (уже вставлено)
    second = loop.run_until_complete(db.claim_daily_login_reward(7))
    assert second["success"] is False
    assert second["error"] == "already_claimed"


def test_claim_not_claimable_when_timer_not_expired():
    now = datetime.now(timezone.utc)
    available_at = now + timedelta(minutes=30)  # ещё не доступна
    conn = _DailyLoginFakeConnection(
        available_at=available_at, claimed=False, streak=0, streak_day=1,
        reward_type="coins", reward_amount=50, coins=0,
    )
    db = _db_with_conn(conn)
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(db.claim_daily_login_reward(1))
    assert result["success"] is False
    assert result["error"] == "not_claimable"


def test_special_reward_multiplier_3_on_third_day():
    now = datetime.now(timezone.utc)
    available_at = now - timedelta(minutes=1)
    # streak_day=3 → multiplier=3 → amount=round(base*3). Но в claim мы используем
    # multiplier из БД (установленный при инициализации цикла).
    conn = _DailyLoginFakeConnection(
        available_at=available_at, claimed=False, streak=2, streak_day=3,
        reward_type="stars", reward_amount=9, multiplier=3, stars=3,
    )
    db = _db_with_conn(conn)
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(db.claim_daily_login_reward(5))
    assert result["success"] is True
    assert result["granted"]["reward_amount"] == 9
    assert result["granted"]["multiplier"] == 3
    # Следующий день = 4 → multiplier=1
    assert result["next"]["multiplier"] == 1
    assert conn.balance["stars"] == 12


def test_choose_daily_login_reward_distribution():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    types_seen = set()
    import random
    random.seed(1234)
    for _ in range(200):
        rt, base, final = db._choose_daily_login_reward(1)
        types_seen.add(rt)
        assert rt in {"coins", "gems", "stars", "keys"}
        assert base in {50, 5, 3, 1}
        assert final == base  # день 1 → multiplier 1
    assert types_seen == {"coins", "gems", "stars", "keys"}


def test_choose_daily_login_reward_special_day_multiplies():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    for day in (3, 6, 9, 12):
        rt, base, final = db._choose_daily_login_reward(day)
        assert final == round(base * 3), f"day {day}: {final} != {round(base*3)}"
    for day in (1, 2, 4, 5, 7, 8):
        rt, base, final = db._choose_daily_login_reward(day)
        assert final == base, f"day {day}: {final} != {base}"


def test_notification_daily_login_reward_message_android():
    msg = format_notification_message("daily_login_reward", {})
    assert "Забери свою награду за вход" in msg
    assert "уже доступна" in msg


def test_notification_daily_login_reward_message_telegram():
    msg = format_telegram_notification_message("daily_login_reward", {})
    assert "Забери свою награду за вход" in msg


def test_notification_daily_login_reward_title_android():
    title = format_android_notification_title("daily_rewards", "daily_login_reward", {})
    assert "Забери награду за вход" in title


def test_notification_daily_login_reward_custom_payload_title_wins():
    title = format_android_notification_title("daily_rewards", "daily_login_reward", {"title": "Моя награда"})
    assert title == "Моя награда"


def test_notification_daily_rewards_category_mapped_to_setting():
    assert NOTIFICATION_SETTING_BY_CATEGORY["daily_rewards"] == "notif_daily_rewards"
    assert NOTIFICATION_DEFAULTS["notif_daily_rewards"] is True
    assert notification_section("daily_rewards", {}) == "arena"


def test_notification_reminders_default_is_false():
    """Напоминания арены по умолчанию ВЫКЛ — заменены системой ежедневных наград."""
    assert NOTIFICATION_DEFAULTS["notif_reminders"] is False


def test_daily_login_rewards_presets_contain_four_types():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    types = {p["type"] for p in db.DAILY_LOGIN_REWARD_PRESETS}
    assert types == {"coins", "gems", "stars", "keys"}
    amounts = {p["amount"] for p in db.DAILY_LOGIN_REWARD_PRESETS}
    assert amounts == {50, 5, 3, 1}


def test_claim_keys_reward_calls_sync_user_key_cases():
    now = datetime.now(timezone.utc)
    available_at = now - timedelta(minutes=1)
    conn = _DailyLoginFakeConnection(
        available_at=available_at, claimed=False, streak=0, streak_day=1,
        reward_type="keys", reward_amount=1, multiplier=1, keys=5,
    )
    db = _db_with_conn(conn)
    called = {"sync": False}

    async def fake_sync(user_id):
        called["sync"] = True

    db.sync_user_key_cases = fake_sync
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(db.claim_daily_login_reward(9))
    assert result["success"] is True
    assert called["sync"] is True
    assert conn.balance["keys"] == 6