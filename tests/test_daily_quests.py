import pytest
import json
from datetime import datetime, timedelta, timezone, date, time

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database


def _make_db():
    return Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))


def test_schema_version_includes_daily_quests_and_returnclock():
    from infrastructure.database import SCHEMA_VERSION
    assert SCHEMA_VERSION == 55


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
    def __init__(self, rows=None, balance=None, feature_availability=None):
        # rows: dict quest_id -> {progress, claimed}
        self.rows = dict(rows or {})
        self.balance = dict(balance or {"coins": 100})
        # feature_availability: dict (e.g. {"daily_quests": False}) surfaced via the
        # game_settings SELECT so is_feature_enabled('daily_quests') reflects it.
        # None => no game_settings row => RUNTIME_FEATURE_DEFAULTS (daily_quests=True).
        self.feature_availability = feature_availability
        self.executed = []
        self.user_cases_inserts = []
        # economy_events: list of (query, args) so tests can assert source (literal in SQL) + metadata.
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
        q = " ".join(query.split())
        # Runtime-config query (is_feature_enabled → get_runtime_config). Return the
        # feature_availability row when a test overrides it; otherwise [] so defaults
        # apply (daily_quests=True). Must not return the quest rows for this query.
        if "FROM game_settings" in q:
            if self.feature_availability is not None:
                return [{"key": "feature_availability", "value": dict(self.feature_availability)}]
            return []
        return [{"quest_id": k, "progress": v["progress"], "claimed": v["claimed"]} for k, v in self.rows.items()]

    async def fetchval(self, query, *args):
        self.executed.append((query, args))
        if "FROM game_settings" in query:
            return dict(self.feature_availability) if self.feature_availability is not None else None
        return None

    async def execute(self, query, *args):
        self.executed.append((query, args))
        q = " ".join(query.split())
        if "INSERT INTO user_cases" in q:
            self.user_cases_inserts.append(args)
            return "INSERT 0 1"
        if "INSERT INTO economy_events" in q:
            self.economy_events.append((query, args))
            return "INSERT 0 1"
        if "UPDATE daily_quests_progress SET claimed" in q:
            qid = args[1]
            self.rows.setdefault(qid, {"progress": 0, "claimed": False})["claimed"] = True
            return "UPDATE 1"
        if "DELETE FROM daily_quests_progress" in q:
            # cleanup_old_daily_quests_progress: delete rows with reset_date < cutoff.
            # The main _QuestsFakeConn is not date-keyed, so this returns a count
            # without mutating; date-aware deletion is exercised via _DateAwareConn.
            return "DELETE 0"
        if "UPDATE users SET coins" in q and "GREATEST" in q:
            self.balance["coins"] = max(0, (self.balance.get("coins") or 0) + int(args[0]))
            return "UPDATE 1"
        # INSERT ... ON CONFLICT — faithful asyncpg semantics: the ON CONFLICT DO UPDATE
        # branch fires ONLY when the row already existed (a real conflict). For a NEW row,
        # the INSERT VALUES(...) succeeds and the DO UPDATE does NOT fire. The previous fake
        # applied both (init + delta) for new rows, over-counting progress (gave 5 where real
        # asyncpg gives 3) and masking the seed-initial-progress path.
        if "INSERT INTO daily_quests_progress" in q:
            if "FROM unnest" in q:
                # Bulk lazy-init (get_daily_quests_status): args = (user_id, today, qids[], inits[]).
                # On a real conflict the DO UPDATE fires per-row; the fake applies login_once bump
                # only when progress < 1 (matching the SQL CASE). New rows seed init.
                qids = list(args[2]); inits = list(args[3])
                for i, qid in enumerate(qids):
                    init = int(inits[i])
                    if qid not in self.rows:
                        self.rows[qid] = {"progress": init, "claimed": False}
                    elif qid == "login_once" and self.rows[qid]["progress"] < 1:
                        self.rows[qid]["progress"] = 1
                return "INSERT 0 5"
            # args: user_id, quest_id, reset_date, [delta/progress, [target]]
            qid = args[1]
            existed = qid in self.rows
            is_do_update = "DO UPDATE" in q
            is_do_nothing = "DO NOTHING" in q
            is_reset = is_do_update and "progress = 0" in q
            is_login_status = is_do_update and "progress = 1" in q and "progress < 1" in q
            is_inc = is_do_update and "progress + $4" in q and "LEAST" in q
            if not existed:
                if is_inc:
                    init = min(int(args[3]), int(args[4]))      # VALUES(..., LEST($4,$5))
                elif is_login_status:
                    init = 1                                     # VALUES(..., 1)
                else:                                            # DO NOTHING or reset_on_loss insert
                    init = 0
                self.rows[qid] = {"progress": init, "claimed": False}
            elif is_do_update:                                   # conflict fired → apply UPDATE
                if is_reset:
                    # P1 guard: only reset when current progress is below target
                    # (args[3] = target). A completed/claimed streak (>= target) is frozen.
                    tgt = int(args[3])
                    if self.rows[qid]["progress"] < tgt:
                        self.rows[qid]["progress"] = 0
                elif is_inc:
                    tgt = int(args[4])
                    self.rows[qid]["progress"] = min(self.rows[qid]["progress"] + int(args[3]), tgt)
                elif is_login_status:
                    if self.rows[qid]["progress"] < 1:
                        self.rows[qid]["progress"] = 1
                # DO NOTHING → no change
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
    init_sql = next(q for q, _args in conn.executed if "FROM unnest" in q)
    assert "WHERE daily_quests_progress.quest_id = 'login_once'" in init_sql


def test_status_uses_one_clock_snapshot_for_date_and_reset():
    conn = _QuestsFakeConn()
    db = _db_with_conn(conn)
    msk = timezone(timedelta(hours=3))
    calls = []

    def clock():
        calls.append(True)
        return datetime(2026, 7, 15, 23, 59, 58, tzinfo=msk)

    db._daily_quests_now = clock
    import asyncio
    status = asyncio.new_event_loop().run_until_complete(db.get_daily_quests_status(42))

    assert len(calls) == 1
    assert status["reset_seconds"] == 2
    assert all(args[1] == date(2026, 7, 15) for query, args in conn.executed if "FROM unnest" in query)


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
    # exactly one economy_events row, with source='daily_quest' (literal in SQL) + full metadata json.
    assert len(conn.economy_events) == 1
    ev_q, ev_a = conn.economy_events[0]
    assert "'daily_quest'" in ev_q and "'earn'" in ev_q          # source + event_type literals in SQL
    assert ev_a[0] == 42                                          # user_id
    assert ev_a[1] == "coins"                                     # resource
    assert ev_a[2] == 50                                          # amount
    import json as _json
    meta = _json.loads(ev_a[3])                                   # metadata json
    assert meta["quest_id"] == "login_once"
    assert meta["reward_type"] == "coins"
    assert meta["reward_amount"] == 50
    assert "reset_date" in meta


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
    quest_sql = next(q for q, _args in conn.executed if "INSERT INTO daily_quests_progress" in q)
    assert "WHERE daily_quests_progress.progress < $5" in quest_sql


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


def test_increment_new_row_caps_delta_at_target():
    # Production path: the battle-end hook calls increment directly for a user who has never
    # opened the Quests sheet (no prior row). Real asyncpg: INSERT VALUES(...,LEST(delta,target))
    # creates the row; the ON CONFLICT DO UPDATE branch does NOT fire (no conflict). So progress
    # = LEST(delta,target). The previous fake double-applied (init + delta) and gave 5 here —
    # masking this exact path. Asserts the fake now matches asyncpg.
    conn = _QuestsFakeConn()  # no row yet
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(db.increment_daily_quest(7, "win_battle_5", 3))
    assert conn.rows["win_battle_5"]["progress"] == 3  # LEST(3,5)=3, NOT min(3+3,5)=5


@pytest.mark.parametrize("quest_id,expected_cases", [
    ("win_battle_1", 1),
    ("win_battle_5", 3),
    ("win_streak_5", 5),
])
def test_claim_case_grants_for_each_case_quest(quest_id, expected_cases):
    # Spec line 140: every case-reward quest grants N T1 user_cases rows. The old test only
    # exercised win_battle_5 (3); win_battle_1 (1) and win_streak_5 (5) were never claim-tested.
    db_defs = _make_db()
    target = next(q["target"] for q in db_defs.DAILY_QUESTS if q["id"] == quest_id)
    conn = _QuestsFakeConn(rows={quest_id: {"progress": target, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    r = asyncio.new_event_loop().run_until_complete(db.claim_daily_quest_reward(42, quest_id))
    assert r["success"] is True
    assert r["granted"] == {"reward_type": "case", "reward_amount": expected_cases}
    assert len(conn.user_cases_inserts) == expected_cases
    for args in conn.user_cases_inserts:
        # INSERT INTO user_cases (user_id, case_id, tier, status) VALUES ($1,1,1,'pending') — T1.
        assert args[1] == 1 and args[2] == 1
    assert conn.rows[quest_id]["claimed"] is True


def test_status_reset_date_rollover_creates_fresh_rows():
    # Spec line 138: status fetch on a new day creates fresh rows (old rows untouched).
    # Uses a reset_date-aware fake keyed by (quest_id, reset_date) — the main _QuestsFakeConn
    # keys by quest_id only and cannot represent two days at once. _daily_quests_today() is
    # overridden on the instance (no production-code change) to drive the day boundary.
    import asyncio

    class _DateAwareConn:
        def __init__(self): self.rows = {}  # (quest_id, reset_date) -> {progress, claimed}
        def transaction(self): return _AsyncCtx()
        async def execute(self, query, *args):
            q = " ".join(query.split())
            if "INSERT INTO daily_quests_progress" not in q:
                return ""
            if "FROM unnest" in q:
                # Bulk lazy-init: args = (user_id, today, qids[], inits[]).
                qids = list(args[2]); inits = list(args[3]); rdate = args[1]
                for i, qid in enumerate(qids):
                    init = int(inits[i])
                    key = (qid, rdate)
                    if key not in self.rows:
                        self.rows[key] = {"progress": init, "claimed": False}
                    elif qid == "login_once" and self.rows[key]["progress"] < 1:
                        self.rows[key]["progress"] = 1
                return "INSERT 0 5"
            qid, rdate = args[1], args[2]
            key = (qid, rdate)
            if key in self.rows:
                return "INSERT 0 1"  # conflict: DO NOTHING / DO UPDATE best-effort (not exercised here)
            # login_once status upsert seeds progress=1; everything else seeds 0.
            self.rows[key] = {"progress": 1 if ("DO UPDATE" in q and "progress = 1" in q) else 0,
                              "claimed": False}
            return "INSERT 0 1"
        async def fetch(self, query, *args):
            q = " ".join(query.split())
            if "FROM game_settings" in q:
                return []  # is_feature_enabled → defaults (daily_quests=True)
            rdate = args[1]  # SELECT ... WHERE user_id=$1 AND reset_date=$2
            return [{"quest_id": k[0], "progress": v["progress"], "claimed": v["claimed"]}
                    for k, v in self.rows.items() if k[1] == rdate]
        async def fetchval(self, query, *args):
            return None

    class _DatePool:
        def __init__(self, c): self.c = c
        def acquire(self): return _AsyncCtx(self.c)

    conn = _DateAwareConn()
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    db._pool = _DatePool(conn)
    day1 = date(2026, 7, 15)
    day2 = date(2026, 7, 16)

    # Day 1: open status (5 fresh rows), then complete + claim win_battle_1.
    msk = timezone(timedelta(hours=3))
    db._daily_quests_now = lambda: datetime(2026, 7, 15, 12, 0, tzinfo=msk)
    asyncio.new_event_loop().run_until_complete(db.get_daily_quests_status(42))
    conn.rows[("win_battle_1", day1)]["progress"] = 1
    conn.rows[("win_battle_1", day1)]["claimed"] = True

    # Day 2: status fetch must create fresh rows; day1 rows untouched.
    db._daily_quests_now = lambda: datetime(2026, 7, 16, 12, 0, tzinfo=msk)
    status = asyncio.new_event_loop().run_until_complete(db.get_daily_quests_status(42))
    qm = {q["id"]: q for q in status["quests"]}
    assert qm["win_battle_1"]["progress"] == 0          # fresh row for day2
    assert qm["win_battle_1"]["claimed"] is False
    # Old day1 rows preserved.
    assert ("win_battle_1", day1) in conn.rows
    assert conn.rows[("win_battle_1", day1)]["progress"] == 1
    assert conn.rows[("win_battle_1", day1)]["claimed"] is True
    # Day2 rows coexist (not the same row reused).
    assert ("win_battle_1", day2) in conn.rows


# --- P1: win_streak_5 freeze at target on loss ---------------------------------

def test_loss_after_target_before_claim_does_not_reset():
    """A 5/5 streak that is still claimable must survive a later loss (the P1 bug:
    an unclaimed reward was erased by the next loss)."""
    conn = _QuestsFakeConn(rows={"win_streak_5": {"progress": 5, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(
        db.increment_daily_quest(7, "win_streak_5", 0, reset_on_loss=True))
    assert conn.rows["win_streak_5"]["progress"] == 5           # frozen, not reset
    assert conn.rows["win_streak_5"]["claimed"] is False
    status = asyncio.new_event_loop().run_until_complete(db.get_daily_quests_status(7))
    q = {x["id"]: x for x in status["quests"]}["win_streak_5"]
    assert q["claimable"] is True                               # reward still claimable


def test_loss_after_claim_does_not_create_claimed_true_progress_zero():
    """A claimed 5/5 streak must not be driven to claimed=true/progress=0 by a loss."""
    conn = _QuestsFakeConn(rows={"win_streak_5": {"progress": 5, "claimed": True}})
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(
        db.increment_daily_quest(7, "win_streak_5", 0, reset_on_loss=True))
    assert conn.rows["win_streak_5"]["progress"] == 5           # not the contradictory 0
    assert conn.rows["win_streak_5"]["claimed"] is True


def test_loss_below_target_still_resets():
    """An in-progress streak (below target) still resets to 0 on loss — the spec intent."""
    conn = _QuestsFakeConn(rows={"win_streak_5": {"progress": 3, "claimed": False}})
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(
        db.increment_daily_quest(7, "win_streak_5", 0, reset_on_loss=True))
    assert conn.rows["win_streak_5"]["progress"] == 0


# --- P2: feature flag ---------------------------------------------------------

def test_status_disabled_returns_enabled_false_and_no_rows():
    """Flag off → enabled=false, empty quests, no lazy progress rows created."""
    conn = _QuestsFakeConn(feature_availability={"daily_quests": False})
    db = _db_with_conn(conn)
    import asyncio
    status = asyncio.new_event_loop().run_until_complete(db.get_daily_quests_status(42))
    assert status["enabled"] is False
    assert status["quests"] == []
    # No progress rows were lazily created for a disabled feature.
    assert conn.rows == {}


def test_claim_disabled_returns_feature_disabled_and_no_economy_mutation():
    """Flag off → claim rejected with feature_disabled; coins/cases unchanged."""
    conn = _QuestsFakeConn(
        rows={"login_once": {"progress": 1, "claimed": False}},
        balance={"coins": 100},
        feature_availability={"daily_quests": False})
    db = _db_with_conn(conn)
    import asyncio
    r = asyncio.new_event_loop().run_until_complete(db.claim_daily_quest_reward(42, "login_once"))
    assert r == {"success": False, "error": "feature_disabled"}
    assert conn.balance["coins"] == 100
    assert conn.user_cases_inserts == []
    assert conn.economy_events == []
    assert conn.rows["login_once"]["claimed"] is False


def test_increment_disabled_is_noop():
    """Flag off → increment does not create or modify any row."""
    conn = _QuestsFakeConn(feature_availability={"daily_quests": False})
    db = _db_with_conn(conn)
    import asyncio
    asyncio.new_event_loop().run_until_complete(db.increment_daily_quest(7, "win_battle_1", 1))
    assert conn.rows == {}
    # The only execute calls should be the feature-flag SELECT (no quest INSERT).
    assert not any("INSERT INTO daily_quests_progress" in q for (q, _a) in conn.executed)


def test_consume_key_and_open_case_progress_share_one_transaction():
    class _KeyConn(_QuestsFakeConn):
        def __init__(self, keys=2, **kwargs):
            super().__init__(**kwargs)
            self.keys = keys

        async def fetchrow(self, query, *args):
            if "UPDATE users" in query and "RETURNING keys" in query:
                self.executed.append((query, args))
                if self.keys <= 0:
                    return None
                self.keys -= 1
                return {"keys": self.keys}
            return await super().fetchrow(query, *args)

    conn = _KeyConn(keys=2)
    db = _db_with_conn(conn)
    import asyncio
    remaining = asyncio.new_event_loop().run_until_complete(
        db.consume_key_for_case_opening(42)
    )

    assert remaining == 1
    assert conn.rows["open_case_1"]["progress"] == 1
    assert any("UPDATE users" in query and "RETURNING keys" in query for query, _args in conn.executed)
    assert any("INSERT INTO daily_quests_progress" in query for query, _args in conn.executed)


def test_consume_key_with_no_keys_does_not_advance_quest():
    class _NoKeyConn(_QuestsFakeConn):
        async def fetchrow(self, query, *args):
            if "UPDATE users" in query and "RETURNING keys" in query:
                return None
            return await super().fetchrow(query, *args)

    conn = _NoKeyConn()
    db = _db_with_conn(conn)
    import asyncio
    remaining = asyncio.new_event_loop().run_until_complete(
        db.consume_key_for_case_opening(42)
    )

    assert remaining is None
    assert "open_case_1" not in conn.rows


def test_consume_key_rolls_back_when_quest_write_fails():
    class _Tx:
        def __init__(self, conn):
            self.conn = conn
            self.exit_exc_type = None
        async def __aenter__(self):
            self.before = self.conn.keys
            return self
        async def __aexit__(self, exc_type, _exc, _tb):
            self.exit_exc_type = exc_type
            if exc_type:
                self.conn.keys = self.before
            return False

    class _Conn:
        def __init__(self):
            self.keys = 1
            self.tx = _Tx(self)
        def transaction(self):
            return self.tx
        async def fetchrow(self, query, *args):
            assert "RETURNING keys" in query
            self.keys -= 1
            return {"keys": self.keys}
        async def fetchval(self, query, *args):
            return None
        async def execute(self, query, *args):
            if "INSERT INTO daily_quests_progress" in query:
                raise RuntimeError("quest write failed")
            return "OK"

    conn = _Conn()
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    db._pool = _QuestsFakePool(conn)
    import asyncio

    with pytest.raises(RuntimeError, match="quest write failed"):
        asyncio.new_event_loop().run_until_complete(
            db.consume_key_for_case_opening(42)
        )

    assert conn.tx.exit_exc_type is RuntimeError
    assert conn.keys == 1


# --- P3: retention cleanup ----------------------------------------------------

def test_cleanup_old_daily_quests_progress_returns_int_and_never_raises():
    """cleanup returns an int count (parsed from DELETE N) and is a no-op without a pool."""
    import asyncio
    # No pool → 0, no raise.
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    assert asyncio.new_event_loop().run_until_complete(
        db.cleanup_old_daily_quests_progress(30)) == 0
    # With a fake conn returning "DELETE 7" → 7.
    class _DelConn(_QuestsFakeConn):
        async def execute(self, query, *args):
            if "DELETE FROM daily_quests_progress" in query:
                return "DELETE 7"
            return await super().execute(query, *args)
    db2 = _db_with_conn(_DelConn())
    n = asyncio.new_event_loop().run_until_complete(db2.cleanup_old_daily_quests_progress(30))
    assert n == 7


def test_daily_quests_cleanup_task_is_wired_in_main():
    """P3: the retention cleanup background task + DB method must be wired (source guard
    so a refactor that drops either is caught)."""
    import re
    main_src = open("main.py", encoding="utf-8").read()
    assert "async def _daily_quests_cleanup_task" in main_src
    assert "_daily_quests_cleanup_task(db)" in main_src  # registered as a background task
    db_src = open("infrastructure/database.py", encoding="utf-8").read()
    assert "async def cleanup_old_daily_quests_progress" in db_src
    assert "DELETE FROM daily_quests_progress AS q" in db_src
    assert "idx_daily_quests_progress_reset_date_id" in db_src


def test_cleanup_old_daily_quests_progress_deletes_by_reset_date():
    """Rows with reset_date older than the cutoff are deleted; newer rows kept.
    Uses a date-aware fake that actually deletes."""
    import asyncio
    from datetime import date

    class _DelDateConn:
        def __init__(self): self.rows = {}  # (quest_id, reset_date) -> {...}
        def transaction(self): return _AsyncCtx()
        async def execute(self, query, *args):
            q = " ".join(query.split())
            if "DELETE FROM daily_quests_progress" in q:
                cutoff = args[0]
                removed = 0
                for k in list(self.rows):
                    if k[1] < cutoff:
                        del self.rows[k]
                        removed += 1
                return f"DELETE {removed}"
            return ""
        async def fetch(self, query, *args):
            return []

    class _DDPool:
        def __init__(self, c): self.c = c
        def acquire(self): return _AsyncCtx(self.c)

    conn = _DelDateConn()
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    db._pool = _DDPool(conn)
    today = date(2026, 7, 16)
    db._daily_quests_today = lambda: today
    old = date(2026, 6, 10)   # 36 days ago → older than 30
    recent = date(2026, 7, 15)  # yesterday → kept
    conn.rows[("win_battle_1", old)] = {"progress": 1, "claimed": True}
    conn.rows[("login_once", old)] = {"progress": 1, "claimed": False}
    conn.rows[("win_streak_5", recent)] = {"progress": 3, "claimed": False}
    removed = asyncio.new_event_loop().run_until_complete(db.cleanup_old_daily_quests_progress(30))
    assert removed == 2
    assert ("win_battle_1", old) not in conn.rows
    assert ("login_once", old) not in conn.rows
    assert ("win_streak_5", recent) in conn.rows
