import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def _generator_api_block() -> str:
    source = Path("web/server.py").read_text(encoding="utf-8")
    start = source.index("async def generator_status_handler")
    end = source.index("async def season_current_handler", start)
    return source[start:end]


class _FakeClaimConnection:
    def __init__(self, *, last_tick_at, accumulated_keys=1, user_keys=5):
        self.row = {
            "level": 1,
            "accumulated_keys": accumulated_keys,
            "last_tick_at": last_tick_at,
            "notified": False,
            "extra_pass": "inactive",
            "extra_pass_expires_at": None,
        }
        self.user_keys = user_keys
        self.executed = []

    def transaction(self):
        return _AsyncContext()

    async def fetchrow(self, query, user_id):
        assert "FOR UPDATE" in query
        return self.row

    async def execute(self, query, *args):
        self.executed.append((query, args))
        if "UPDATE users SET keys" in query:
            self.user_keys += int(args[1])

    async def fetchval(self, query, user_id):
        if "SELECT keys" in query:
            return self.user_keys
        return None


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
async def test_generator_accumulation_uses_f2p_config_for_expired_extra_pass():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    now = datetime.now(timezone.utc)

    accumulated, new_keys, cap, interval_seconds = await db._compute_generator_accumulated({
        "level": 1,
        "accumulated_keys": 0,
        "last_tick_at": now - timedelta(hours=4, minutes=5),
        "extra_pass": "ultra",
        "extra_pass_expires_at": now - timedelta(minutes=1),
    })

    assert accumulated == 0
    assert new_keys == 0
    assert cap == GENERATOR_LEVELS[1]["f2p"]["cap"]
    assert interval_seconds == GENERATOR_LEVELS[1]["f2p"]["interval_hours"] * 3600


@pytest.mark.asyncio
async def test_generator_accumulation_keeps_ultra_config_for_unexpired_extra_pass():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    now = datetime.now(timezone.utc)

    accumulated, new_keys, cap, interval_seconds = await db._compute_generator_accumulated({
        "level": 1,
        "accumulated_keys": 0,
        "last_tick_at": now - timedelta(hours=4, minutes=5),
        "extra_pass": "ultra",
        "extra_pass_expires_at": now + timedelta(hours=1),
    })

    assert accumulated == 1
    assert new_keys == 1
    assert cap == GENERATOR_LEVELS[1]["ultra"]["cap"]
    assert interval_seconds == GENERATOR_LEVELS[1]["ultra"]["interval_hours"] * 3600


@pytest.mark.asyncio
async def test_generator_status_is_read_only_even_when_keys_are_ready():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    now = datetime.now(timezone.utc)
    executed = []

    async def _noop(user_id):
        return None

    async def _fetchrow(query, user_id):
        return {
            "level": 1,
            "accumulated_keys": 0,
            "last_tick_at": now - timedelta(hours=9),
            "notified": False,
            "extra_pass": "inactive",
            "extra_pass_expires_at": None,
            "coins": 10,
            "gems": 20,
            "keys": 30,
        }

    async def _execute(query, *args):
        executed.append((query, args))

    db._ensure_generator_state = _noop
    db.fetchrow = _fetchrow
    db.execute = _execute

    status = await db.get_generator_status(42)

    assert status["accumulated_keys"] == 1
    assert status["can_claim"] is True
    assert executed == []


@pytest.mark.asyncio
async def test_generator_status_after_claim_does_not_report_stale_keys():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    now = datetime.now(timezone.utc)

    async def _noop(user_id):
        return None

    async def _fetchrow(query, user_id):
        return {
            "level": 1,
            "accumulated_keys": 0,
            "last_tick_at": now - timedelta(hours=1),
            "notified": False,
            "extra_pass": "inactive",
            "extra_pass_expires_at": None,
            "coins": 10,
            "gems": 20,
            "keys": 31,
        }

    async def _execute(query, *args):
        raise AssertionError("generator status must not write after claim")

    db._ensure_generator_state = _noop
    db.fetchrow = _fetchrow
    db.execute = _execute

    status = await db.get_generator_status(42)

    assert status["accumulated_keys"] == 0
    assert status["can_claim"] is False
    assert status["next_key_seconds"] > 0


@pytest.mark.asyncio
async def test_claim_generator_preserves_partial_tick_progress_when_only_stored_keys_exist():
    partial_tick_at = datetime.now(timezone.utc) - timedelta(hours=2)
    conn = _FakeClaimConnection(last_tick_at=partial_tick_at, accumulated_keys=1, user_keys=5)

    result = await _db_with_connection(conn).claim_generator_keys(42)

    generator_updates = [
        query
        for query, _args in conn.executed
        if "UPDATE generator_state" in query and "SET accumulated_keys = 0" in query
    ]
    assert result["success"] is True
    assert result["keys_claimed"] == 1
    assert generator_updates
    assert "ELSE NOW()" not in generator_updates[0]


def test_generator_api_500_responses_do_not_expose_raw_exception_text():
    generator_api = _generator_api_block()

    for handler_name in (
        "generator_status_handler",
        "generator_claim_handler",
        "generator_upgrade_handler",
    ):
        handler = generator_api.split(f"async def {handler_name}", 1)[1].split(
            "\n    async def ",
            1,
        )[0]
        assert '{"error": str(e)}' not in handler
        assert '"internal_server_error"' in handler
        assert "web.json_response" in handler


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
