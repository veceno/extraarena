import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database
from infrastructure.generator_config import (
    GENERATOR_LEVELS,
    GENERATOR_MAX_LEVEL,
    GENERATOR_UPGRADE_COST,
)


class _AsyncContext:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, *, level=1, gems=100):
        self.level = level
        self.coins = 9999
        self.gems = gems
        self.executed = []

    def transaction(self):
        return _AsyncContext()

    async def fetchrow(self, query, user_id):
        return {"level": self.level, "coins": self.coins, "gems": self.gems}

    async def execute(self, query, *args):
        self.executed.append((query, args))
        if "UPDATE users SET gems" in query:
            self.gems = max(0, self.gems - args[1])
        elif "UPDATE generator_state" in query and "SET level" in query:
            self.level = args[1]

    async def fetchval(self, query, user_id):
        if "SELECT coins" in query:
            return self.coins
        if "SELECT gems" in query:
            return self.gems
        return None


class _FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _AsyncContext(self.conn)


def _db_with_connection(conn):
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    db._pool = _FakePool(conn)

    async def _noop(user_id):
        return None

    db._ensure_generator_state = _noop
    return db


def test_generator_config_has_six_gem_only_levels():
    assert GENERATOR_MAX_LEVEL == 6
    assert GENERATOR_LEVELS[1]["f2p"] == {"interval_hours": 8, "cap": 2}
    assert GENERATOR_LEVELS[1]["active"] == {"interval_hours": 6, "cap": 3}
    assert GENERATOR_LEVELS[1]["ultra"] == {"interval_hours": 4, "cap": 4}
    assert GENERATOR_UPGRADE_COST == {
        2: {"gems": 100},
        3: {"gems": 250},
        4: {"gems": 500},
        5: {"gems": 900},
        6: {"gems": 1500},
    }


@pytest.mark.asyncio
async def test_upgrade_generator_spends_only_gems():
    conn = _FakeConnection(level=1, gems=100)
    result = await _db_with_connection(conn).upgrade_generator(42)

    assert result["success"] is True
    assert result["old_level"] == 1
    assert result["new_level"] == 2
    assert result["cost"] == {"gems": 100}
    assert result["currency_spent"] == "gems"
    assert result["amount_spent"] == 100
    assert result["coins_remaining"] == 9999
    assert result["gems_remaining"] == 0


@pytest.mark.asyncio
async def test_upgrade_generator_ignores_legacy_currency_payload():
    conn = _FakeConnection(level=1, gems=100)
    result = await _db_with_connection(conn).upgrade_generator(42, currency="coins")

    assert result["success"] is True
    assert result["new_level"] == 2
    assert result["currency_spent"] == "gems"


@pytest.mark.asyncio
async def test_upgrade_generator_requires_enough_gems():
    result = await _db_with_connection(_FakeConnection(level=1, gems=99)).upgrade_generator(42)

    assert result == {
        "success": False,
        "error": "not_enough_gems",
        "required": 100,
        "have": 99,
    }


@pytest.mark.asyncio
async def test_upgrade_generator_stops_at_max_level():
    result = await _db_with_connection(_FakeConnection(level=6, gems=9999)).upgrade_generator(42)

    assert result == {"success": False, "error": "max_level_reached"}
