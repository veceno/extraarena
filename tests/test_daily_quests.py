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

class _AsyncCtx:
    def __init__(self, value=None): self.value = value
    async def __aenter__(self): return self.value
    async def __aexit__(self, *a): return False


class _QuestsFakeConn:
    """Эмулирует asyncpg.Connection для daily-quests методов."""
    def __init__(self, rows=None, balance=None):
        # rows: dict quest_id -> {progress, claimed}
        self.rows = dict(rows or {})
        self.balance = dict(balance or {"coins": 100})
        self.executed = []
        self.user_cases_inserts = []
        self.economy_events = []
        self._claim_row_returned = False

    def transaction(self): return _AsyncCtx()

    async def fetchrow(self, query, *args):
        self.executed.append((query, args))
        q = " ".join(query.split())
        if "FOR UPDATE" in q:
            qid = args[2]
            r = self.rows.get(qid, {"progress": (1 if qid == "login_once" else 0), "claimed": False})
            return {"quest_id": qid, "progress": r["progress"], "claimed": r["claimed"]}
        return None

    async def fetch(self, query, *args):
        self.executed.append((query, args))
        return [{"quest_id": k, "progress": v["progress"], "claimed": v["claimed"]} for k, v in self.rows.items()]

    async def execute(self, query, *args):
        self.executed.append((query, args))
        q = " ".join(query.split())
        if "INSERT INTO user_cases" in q:
            self.user_cases_inserts.append(args)
            return "INSERT 0 1"
        if "INSERT INTO economy_events" in q:
            self.economy_events.append(args)
            return "INSERT 0 1"
        if "UPDATE daily_quests_progress SET claimed" in q:
            qid = args[1]
            self.rows.setdefault(qid, {"progress": 0, "claimed": False})["claimed"] = True
            return "UPDATE 1"
        if "UPDATE users SET coins" in q and "GREATEST" in q:
            self.balance["coins"] = max(0, (self.balance.get("coins") or 0) + int(args[0]))
            return "UPDATE 1"
        # INSERT ... ON CONFLICT / DO NOTHING / DO UPDATE — best-effort simulate for login_once
        if "INSERT INTO daily_quests_progress" in q:
            # args: user_id, quest_id, reset_date, [progress]
            qid = args[1]
            if qid not in self.rows:
                self.rows[qid] = {"progress": (1 if qid == "login_once" else 0), "claimed": False}
            elif qid == "login_once" and "DO UPDATE" in q:
                self.rows[qid]["progress"] = max(self.rows[qid]["progress"], 1)
            return "INSERT 0 1"
        return ""


class _QuestsFakePool:
    def __init__(self, conn): self.conn = conn
    def acquire(self): return _AsyncCtx(self.conn)


def _db_with_conn(conn):
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    db._pool = _QuestsFakePool(conn)
    return db


def test_status_lazy_inits_five_rows_and_login_once_progresses_to_one():
    conn = _QuestsFakeConn()
    db = _db_with_conn(conn)
    import asyncio
    status = asyncio.new_event_loop().run_until_complete(db.get_daily_quests_status(42))
    assert status["enabled"] is True
    assert "reset_at" in status and isinstance(status["reset_seconds"], int)
    quests = {q["id"]: q for q in status["quests"]}
    assert set(quests.keys()) == {"login_once", "open_case_1", "win_battle_1", "win_battle_5", "win_streak_5"}
    # login_once is auto-completed by opening the status
    assert quests["login_once"]["progress"] == 1
    assert quests["login_once"]["target"] == 1
    assert quests["login_once"]["claimable"] is True
    assert quests["login_once"]["claimed"] is False
    # others start at 0, not claimable
    assert quests["win_battle_5"]["progress"] == 0
    assert quests["win_battle_5"]["claimable"] is False


def test_status_caps_display_progress_at_target():
    # win_streak_5 progress stored above target -> display capped at 5
    conn = _QuestsFakeConn(rows={"win_streak_5": {"progress": 7, "claimed": False},
                                  "login_once": {"progress": 1, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    status = asyncio.new_event_loop().run_until_complete(db.get_daily_quests_status(7))
    q = {x["id"]: x for x in status["quests"]}["win_streak_5"]
    assert q["progress"] == 5  # min(progress, target)
    assert q["target"] == 5
    assert q["claimable"] is True  # progress >= target and not claimed
