import pytest
from datetime import datetime, timedelta, timezone, date, time

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database


def _make_db():
    return Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))


def test_schema_version_bumped_to_49():
    from infrastructure.database import SCHEMA_VERSION
    assert SCHEMA_VERSION == 49


def test_daily_quests_constant_has_five_fixed_quests():
    db = _make_db()
    ids = [q["id"] for q in db.DAILY_QUESTS]
    assert ids == ["login_once", "open_case_1", "win_battle_1", "win_battle_5", "win_streak_5"]
    for q in db.DAILY_QUESTS:
        assert {"id", "title", "description", "target", "reward_type", "reward_amount"} <= set(q.keys())
        assert isinstance(q["target"], int) and q["target"] >= 1
        assert q["reward_type"] in ("coins", "case")
        assert isinstance(q["reward_amount"], int) and q["reward_amount"] >= 1
    # case-reward quests carry case_tier=1
    case_quests = [q for q in db.DAILY_QUESTS if q["reward_type"] == "case"]
    assert case_quests, "expected at least one case-reward quest"
    assert all(q.get("case_tier") == 1 for q in case_quests)