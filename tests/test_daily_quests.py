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
            qid = args[1]
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
        # INSERT ... ON CONFLICT / DO NOTHING / DO UPDATE — best-effort simulate for daily-quests upserts
        if "INSERT INTO daily_quests_progress" in q:
            # args: user_id, quest_id, reset_date, [delta/progress, [target]]
            qid = args[1]
            if qid not in self.rows:
                if "LEAST" in q and "VALUES" in q:
                    init = min(int(args[3]), int(args[4]))
                elif qid == "login_once":
                    init = 1
                else:
                    init = 0
                self.rows[qid] = {"progress": init, "claimed": False}
            if "DO UPDATE" in q:
                if "progress = 0" in q:  # reset_on_loss
                    self.rows[qid]["progress"] = 0
                elif "LEAST" in q and "progress + $4" in q:
                    tgt = int(args[4])
                    self.rows[qid]["progress"] = min(self.rows[qid]["progress"] + int(args[3]), tgt)
                elif qid == "login_once":
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


def test_claim_unknown_quest_returns_error():
    db = _db_with_conn(_QuestsFakeConn())
    import asyncio
    r = asyncio.new_event_loop().run_until_complete(db.claim_daily_quest_reward(1, "nope"))
    assert r["success"] is False and r["error"] == "unknown_quest"


def test_claim_coins_grants_and_marks_claimed():
    conn = _QuestsFakeConn(rows={"login_once": {"progress": 1, "claimed": False}}, balance={"coins": 100})
    db = _db_with_conn(conn)
    import asyncio
    r = asyncio.new_event_loop().run_until_complete(db.claim_daily_quest_reward(42, "login_once"))
    assert r["success"] is True
    assert r["granted"] == {"reward_type": "coins", "reward_amount": 50}
    assert conn.balance["coins"] == 150
    assert conn.rows["login_once"]["claimed"] is True
    # exactly one economy_events row, source via metadata json
    assert len(conn.economy_events) == 1


def test_claim_case_grants_n_t1_user_cases_rows():
    conn = _QuestsFakeConn(rows={"win_battle_5": {"progress": 5, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    r = asyncio.new_event_loop().run_until_complete(db.claim_daily_quest_reward(42, "win_battle_5"))
    assert r["success"] is True
    assert r["granted"] == {"reward_type": "case", "reward_amount": 3}
    # 3 T1 user_cases inserts
    assert len(conn.user_cases_inserts) == 3
    for args in conn.user_cases_inserts:
        # INSERT INTO user_cases (user_id, case_id, tier, status) VALUES ($1,1,1,'pending')
        assert args[1] == 1 and args[2] == 1
    assert conn.rows["win_battle_5"]["claimed"] is True


def test_claim_already_claimed_returns_409_error():
    conn = _QuestsFakeConn(rows={"login_once": {"progress": 1, "claimed": True}})
    db = _db_with_conn(conn)
    import asyncio
    r = asyncio.new_event_loop().run_until_complete(db.claim_daily_quest_reward(42, "login_once"))
    assert r["success"] is False and r["error"] == "already_claimed"
    assert conn.user_cases_inserts == [] and conn.balance["coins"] == 100  # no grant


def test_claim_below_target_returns_not_claimable():
    conn = _QuestsFakeConn(rows={"win_battle_5": {"progress": 2, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    r = asyncio.new_event_loop().run_until_complete(db.claim_daily_quest_reward(42, "win_battle_5"))
    assert r["success"] is False and r["error"] == "not_claimable"


def test_increment_caps_at_target():
    conn = _QuestsFakeConn(rows={"win_battle_5": {"progress": 4, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(db.increment_daily_quest(7, "win_battle_5", 1))
    assert conn.rows["win_battle_5"]["progress"] == 5  # 4+1=5 == target


def test_increment_overflow_caps_at_target():
    conn = _QuestsFakeConn(rows={"win_streak_5": {"progress": 5, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(db.increment_daily_quest(7, "win_streak_5", 1))
    assert conn.rows["win_streak_5"]["progress"] == 5  # LEAST(5+1, 5)=5


def test_increment_seeds_initial_progress():
    conn = _QuestsFakeConn()  # no row yet
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(db.increment_daily_quest(7, "win_battle_1", 1))
    assert conn.rows["win_battle_1"]["progress"] == 1


def test_increment_reset_on_loss_zeroes_streak():
    conn = _QuestsFakeConn(rows={"win_streak_5": {"progress": 3, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(db.increment_daily_quest(7, "win_streak_5", 0, reset_on_loss=True))
    assert conn.rows["win_streak_5"]["progress"] == 0


def test_increment_swallows_db_errors():
    class _BoomConn(_QuestsFakeConn):
        async def execute(self, query, *args):
            raise RuntimeError("db down")
    db = _db_with_conn(_BoomConn())
    import asyncio
    # must NOT raise
    asyncio.new_event_loop().run_until_complete(db.increment_daily_quest(7, "win_battle_1", 1))
