import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.config import get_settings
from infrastructure.database import Database
from web import server as web_server


class _ClaimDB(Database):
    def __init__(self, rows):
        self._pool = object()
        self.rows = list(rows)
        self.calls = []

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return self.rows.pop(0)


class _ActivateSeasonDB(Database):
    def __init__(self, current_active_id, candidate_id):
        self._pool = object()
        self.current_active_id = current_active_id
        self.candidate_id = candidate_id
        self.executes = []
        self.reset_calls = []

    async def fetchval(self, query, *args):
        return self.current_active_id

    async def fetchrow(self, query, *args):
        if "SELECT id" in query and "FROM seasons" in query:
            return {"id": self.candidate_id}
        return None

    async def execute(self, query, *args):
        self.executes.append(query)
        return "OK"

    async def execute_season_reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return {"status": "completed", "season_id": kwargs.get("season_id")}

    async def get_season_by_id(self, season_id):
        return {"id": season_id}


class _SeasonResetPreviewDB(Database):
    def __init__(self):
        self._pool = object()

    async def fetchrow(self, query, *args):
        if "FROM season_resets" in query:
            return None
        if "COUNT(*)::integer AS players" in query:
            rows = [
                {"user_id": 1, "trophies": 0, "stars": 5, "is_bot": False},
                {"user_id": 2, "trophies": 300, "stars": 6, "is_bot": False},
                {"user_id": 3, "trophies": 650, "stars": 7, "is_bot": False},
                {"user_id": 4, "trophies": 799, "stars": 8, "is_bot": False},
                {"user_id": 5, "trophies": 1250, "stars": 9, "is_bot": False},
                {"user_id": 6, "trophies": 1499, "stars": 10, "is_bot": False},
            ]
            computed = []
            for row in rows:
                old_trophies = row["trophies"]
                boundary = 1200 if old_trophies >= 1200 else 600 if old_trophies >= 600 else 300 if old_trophies >= 300 else 0
                bonus_units = max(0, old_trophies - boundary) // 100
                computed.append({
                    "old_trophies": old_trophies,
                    "boundary": boundary,
                    "old_stars": row["stars"],
                    "bonus_units": bonus_units,
                })
            return {
                "players": len(computed),
                "trophies_reduced": sum(max(0, row["old_trophies"] - row["boundary"]) for row in computed),
                "keys_granted": sum(row["bonus_units"] for row in computed),
                "coins_granted": sum(row["bonus_units"] * 200 for row in computed),
                "stars_reset": sum(row["old_stars"] for row in computed),
            }
        return None

    async def fetch(self, query, *args):
        if "FROM users" not in query:
            return []
        return [
            {"user_id": 1, "trophies": 0, "stars": 5, "is_bot": False},
            {"user_id": 2, "trophies": 300, "stars": 6, "is_bot": False},
            {"user_id": 3, "trophies": 650, "stars": 7, "is_bot": False},
            {"user_id": 4, "trophies": 799, "stars": 8, "is_bot": False},
            {"user_id": 5, "trophies": 1250, "stars": 9, "is_bot": False},
            {"user_id": 6, "trophies": 1499, "stars": 10, "is_bot": False},
            {"user_id": 7, "trophies": 1499, "stars": 11, "is_bot": True},
        ]


class _NoAcquireSeasonResetDB(Database):
    def __init__(self):
        self._pool = object()


class _ManualSeasonUpdateDB(Database):
    def __init__(self):
        self._pool = object()
        self.reset_calls = []
        self.executes = []
        self.current_active_id = 1
        self.seasons = {
            1: {"id": 1, "is_active": True, "status": "active"},
            2: {"id": 2, "is_active": False, "status": "scheduled"},
        }

    async def get_season_by_id(self, season_id):
        row = self.seasons.get(int(season_id))
        return dict(row) if row else None

    async def fetchval(self, query, *args):
        return self.current_active_id

    async def execute(self, query, *args):
        self.executes.append((query, args))
        return "OK"

    async def fetchrow(self, query, *args):
        if "UPDATE seasons" not in query:
            return None
        season_id = int(args[-1])
        row = {
            "id": season_id,
            "slug": f"season-{season_id}",
            "name": f"Season {season_id}",
            "subtitle": "",
            "description": "",
            "season_number": season_id,
            "status": "active",
            "auto_switch": True,
            "preset_key": "blank",
            "start_date": None,
            "end_date": None,
            "is_active": True,
            "max_stars": 45,
            "stage_cost_min": 3,
            "stage_cost_growth": 0.07,
            "stage_cost_exponent": 1.5,
            "stage_cost_cap": 25,
            "free_track_type": f"s{season_id}_free",
            "pass_track_type": f"s{season_id}_pass",
            "ultra_track_type": f"s{season_id}_ultra",
            "pass_end_position": 40,
            "ultra_start_position": 41,
            "theme": {},
            "created_at": None,
            "updated_at": None,
        }
        self.seasons[season_id] = dict(row)
        return row

    async def execute_season_reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return {"status": "completed", "season_id": kwargs.get("season_id")}


class _InactiveSeasonResetConn:
    def __init__(self):
        self.fetchrow_calls = []
        self.fetch_calls = []
        self.execute_calls = []

    async def fetchval(self, query, *args):
        return None

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if "FROM seasons" in query:
            return None
        raise AssertionError("reset should stop before reading season_resets when active guard fails")

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        raise AssertionError("reset should stop before locking users when active guard fails")

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        raise AssertionError("reset should stop before mutations when active guard fails")


class _PartialSeasonResetConn:
    def __init__(self):
        self.fetch_calls = []
        self.execute_calls = []

    async def fetchval(self, query, *args):
        if "COUNT(*)" in query and "season_reset_results" in query:
            return 1
        return None

    async def fetchrow(self, query, *args):
        if "FROM season_resets" in query:
            return {"id": 99, "status": "running"}
        return {"id": args[0] if args else 1}

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        raise AssertionError("partial reset should stop before locking users")

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        raise AssertionError("partial reset should stop before mutations")


class _SeasonResetMutationConn:
    def __init__(self):
        self.fetch_calls = []
        self.fetchrow_calls = []
        self.execute_calls = []
        self.reset_id = 55

    async def fetchval(self, query, *args):
        if "pg_advisory_xact_lock" in query:
            return None
        if "COUNT(*)" in query and "season_reset_results" in query:
            return 0
        return None

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if "FROM seasons" in query:
            return {
                "id": int(args[0]),
                "free_track_type": "s2_free",
                "pass_track_type": "s2_pass",
                "ultra_track_type": "s2_ultra",
            }
        if "FROM season_resets" in query:
            return None
        if "INSERT INTO season_resets" in query:
            return {"id": self.reset_id}
        if "UPDATE season_resets" in query:
            return {
                "id": self.reset_id,
                "season_id": int(args[0] and 2),
                "previous_season_id": 1,
                "trigger": "admin",
                "admin_user_id": 101,
                "reason": "test",
                "status": "completed",
                "processed_players": 1,
                "total_trophies_reduced": 299,
                "total_keys_granted": 2,
                "total_coins_granted": 400,
                "total_stars_reset": 12,
                "created_at": None,
                "completed_at": None,
                "updated_at": None,
            }
        return None

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        if "FROM users" in query:
            return [
                {
                    "user_id": 201,
                    "trophies": 1499,
                    "stars": 12,
                    "keys": 5,
                    "coins": 200,
                    "is_bot": False,
                    "extra_pass": "ultra",
                    "extra_pass_expires_at": "2026-07-01T00:00:00+00:00",
                }
            ]
        return []

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "OK"


class _RollbackTransaction:
    def __init__(self):
        self.rolled_back = False
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.rolled_back = exc_type is not None
        self.committed = exc_type is None
        return False


class _ActivationRollbackConn:
    def __init__(self):
        self.transaction_state = _RollbackTransaction()
        self.execute_calls = []
        self.fetchrow_calls = []

    def transaction(self):
        return self.transaction_state

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if "UPDATE seasons" in query:
            return {
                "id": int(args[-1]),
                "slug": "season-2",
                "name": "Season 2",
                "subtitle": "",
                "description": "",
                "season_number": 2,
                "status": "active",
                "auto_switch": True,
                "preset_key": "blank",
                "start_date": None,
                "end_date": None,
                "is_active": True,
                "max_stars": 45,
                "stage_cost_min": 3,
                "stage_cost_growth": 0.07,
                "stage_cost_exponent": 1.5,
                "stage_cost_cap": 25,
                "free_track_type": "s2_free",
                "pass_track_type": "s2_pass",
                "ultra_track_type": "s2_ultra",
                "pass_end_position": 40,
                "ultra_start_position": 41,
                "theme": {},
                "created_at": None,
                "updated_at": None,
            }
        return None

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "OK"


class _ActivationRollbackAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ActivationRollbackPool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _ActivationRollbackAcquire(self.conn)


class _ActivationRollbackDB(Database):
    def __init__(self):
        self.conn = _ActivationRollbackConn()
        self._pool = _ActivationRollbackPool(self.conn)

    async def get_season_by_id(self, season_id):
        return {"id": int(season_id), "is_active": False, "status": "scheduled"}

    async def fetchval(self, query, *args):
        return 1

    async def _execute_season_reset_on_conn(self, conn, **kwargs):
        return {"error": "season_reset_failed", "season_id": kwargs.get("season_id")}


class _RewardClaimDB:
    def __init__(self, *, extra_pass="active", row_extra_pass_required=True):
        self.claim_attempts = []
        self.extra_pass = extra_pass
        self.row_extra_pass_required = row_extra_pass_required

    async def get_active_season(self):
        return {
            "id": 1,
            "free_track_type": "s1_free",
            "pass_track_type": "s1_pass",
            "ultra_track_type": "s1_ultra",
            "max_stars": 10,
            "pass_end_position": 3,
            "ultra_start_position": 4,
        }

    async def get_reward_track_entries(self, track_type, position):
        return [
            {
                "id": 1,
                "track_type": track_type,
                "position": position,
                "reward_type": "coins",
                "reward_amount": 100,
                "reward_meta": None,
                "extra_pass_required": self.row_extra_pass_required,
                "is_active": True,
            }
        ]

    async def get_user_profile(self, user_id):
        return {
            "user_id": user_id,
            "stars": 999,
            "trophies": 0,
            "extra_pass": self.extra_pass,
            "extra_pass_expires_at": None,
        }

    async def claim_reward_entries_transaction(self, **kwargs):
        self.claim_attempts.append(kwargs)
        return {"success": True, "granted": []}

    async def process_weekly_squad_cbrp(self):
        return {"processed": False}

    async def expire_announcements(self):
        return 0

    async def refresh_due_rating_snapshots(self, scope="players"):
        return {"refreshed": []}


class _CaseRewardClaimDB:
    def __init__(self):
        self.claimed = set()
        self.cases = []

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},
            "disabled_card_ids": [],
        }

    async def get_match_mode_overrides(self):
        return []

    async def is_match_mode_enabled(self, mode_id):
        return True

    async def get_disabled_card_ids(self):
        return []

    async def is_admin(self, user_id):
        return False

    async def get_active_season(self):
        return {
            "id": 1,
            "free_track_type": "s1_free",
            "pass_track_type": "s1_pass",
            "ultra_track_type": "s1_ultra",
            "max_stars": 10,
            "pass_end_position": 3,
            "ultra_start_position": 4,
        }

    async def get_reward_track_entries(self, track_type, position):
        return [
            {
                "id": 1,
                "track_type": track_type,
                "position": position,
                "reward_type": "case",
                "reward_amount": 3,
                "reward_meta": None,
                "extra_pass_required": False,
                "is_active": True,
            }
        ]

    async def get_user_profile(self, user_id):
        return {
            "user_id": user_id,
            "stars": 999,
            "trophies": 0,
            "extra_pass": "inactive",
            "extra_pass_expires_at": None,
        }

    async def claim_reward(self, user_id, track_type, position):
        key = (user_id, track_type, position)
        if key in self.claimed:
            return False
        self.claimed.add(key)
        return True

    async def get_admin_case_id(self, tier):
        return tier

    async def add_user_case(self, user_id, case_id, tier):
        row = {
            "id": len(self.cases) + 1,
            "user_case_id": len(self.cases) + 1,
            "user_id": user_id,
            "case_id": case_id,
            "tier": tier,
            "status": "pending",
        }
        self.cases.append(row)
        return dict(row)

    async def get_user_cases(self, user_id):
        return [dict(case) for case in self.cases if case["user_id"] == user_id and case["status"] == "pending"]

    async def process_weekly_squad_cbrp(self):
        return {"processed": False}

    async def expire_announcements(self):
        return 0

    async def refresh_due_rating_snapshots(self, scope="players"):
        return {"refreshed": []}


class _CardRewardClaimDB(_CaseRewardClaimDB):
    def __init__(self, reward_type, reward_meta, *, random_card=None, card_info=None, owned_cards=None):
        super().__init__()
        self.reward_type = reward_type
        self.reward_meta = reward_meta
        self.random_card = random_card or {"id": 99, "name": "Random Card"}
        self.card_info = card_info or {"id": 46, "name": "Exact Card", "card_type": "warrior"}
        self.owned_cards = list(owned_cards or [])
        self.added_cards = []
        self.coins_added = 0

    async def get_reward_track_entries(self, track_type, position):
        return [
            {
                "id": 1,
                "track_type": track_type,
                "position": position,
                "reward_type": self.reward_type,
                "reward_amount": 1,
                "reward_meta": self.reward_meta,
                "extra_pass_required": False,
                "is_active": True,
            }
        ]

    async def get_random_cards_by_rarities(self, rarities, limit=1):
        return [dict(self.random_card)]

    async def get_card_info(self, card_id, level=1):
        if self.card_info and int(card_id) == int(self.card_info["id"]):
            return dict(self.card_info)
        return None

    async def add_card_to_user(self, user_id, card_id):
        self.added_cards.append(int(card_id))
        return {"success": True}

    async def get_user_cards(self, user_id):
        return list(self.owned_cards)

    async def update_user_coins(self, user_id, amount):
        self.coins_added += int(amount)


class _InvalidSpecificCardRewardClaimDB(_CardRewardClaimDB):
    def __init__(self):
        super().__init__("specific_card", {})

    async def claim_reward(self, user_id, track_type, position):
        raise AssertionError("invalid reward config must not mark the tier claimed")


class _PromoTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _PromoConn:
    def __init__(self, db):
        self.db = db

    def transaction(self):
        return _PromoTransaction()

    async def execute(self, query, *args):
        self.db.conn_executes.append((query, args))
        if "UPDATE users" in query and "extra_pass" in query:
            self.db.pass_update = {"mode": args[0], "expires_at": args[1], "user_id": args[2]}
        return "OK"

    async def fetchrow(self, query, *args):
        if "SELECT extra_pass, extra_pass_expires_at FROM users" in query:
            return dict(self.db.user_pass_row)
        return None


class _PromoAcquire:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return _PromoConn(self.db)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _PromoPool:
    def __init__(self, db):
        self.db = db

    def acquire(self):
        return _PromoAcquire(self.db)


class _PromoDB(Database):
    def __init__(self):
        self.promocode = {
            "id": 1,
            "type": "permanent",
            "reward_gems": 0,
            "reward_coins": 0,
            "reward_keys": 0,
            "reward_extrapass": True,
            "expires_at": None,
        }
        self.user_pass_row = {"extra_pass": "ultra", "extra_pass_expires_at": None}
        self.conn_executes = []
        self.pass_update = None
        self._pool = _PromoPool(self)

    async def fetchrow(self, query, *args):
        if "FROM promocodes" in query:
            return dict(self.promocode)
        if "FROM promocode_usage" in query:
            return None
        return None


@pytest.mark.asyncio
async def test_claim_reward_returns_false_for_duplicate_claim():
    db = _ClaimDB([{"id": 1}, None])

    assert await db.claim_reward(1001, "bp_free", 1) is True
    assert await db.claim_reward(1001, "bp_free", 1) is False
    assert "ON CONFLICT" in db.calls[0][0]
    assert "RETURNING id" in db.calls[0][0]


@pytest.mark.asyncio
async def test_rewards_claim_rejects_out_of_scope_season_position(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db = _RewardClaimDB()
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/rewards/claim?user_id=1001",
            json={"track_type": "s1_pass", "position": 5},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "position_out_of_track_scope"
        assert db.claim_attempts == []
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_rewards_claim_derives_premium_access_from_track_def_even_if_row_flag_false(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db = _RewardClaimDB(extra_pass="inactive", row_extra_pass_required=False)
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/rewards/claim?user_id=1001",
            json={"track_type": "s1_pass", "position": 2},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "extra_pass_required"
        assert "ExtraPass" in body["message"]
        assert db.claim_attempts == []
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_rewards_claim_derives_ultra_access_from_track_def_even_if_row_flag_false(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db = _RewardClaimDB(extra_pass="active", row_extra_pass_required=False)
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/rewards/claim?user_id=1001",
            json={"track_type": "s1_ultra", "position": 4},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "extra_pass_required"
        assert "Ultra" in body["message"]
        assert db.claim_attempts == []
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_reward_track_case_claim_creates_visible_pending_user_case(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("ENVIRONMENT", "development")
    get_settings.cache_clear()
    db = _CaseRewardClaimDB()
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        claim_response = await client.post(
            "/api/rewards/claim?user_id=1001",
            json={"track_type": "s1_free", "position": 3},
        )
        claim_body = await claim_response.json()
        cases_response = await client.get("/api/cases/user?user_id=1001")
        cases_body = await cases_response.json()

        assert claim_response.status == 200
        assert claim_body["granted"][0]["reward_type"] == "case"
        assert claim_body["granted"][0]["user_case_id"] == 1
        assert cases_response.status == 200
        assert cases_body["cases"] == [
            {
                "id": 1,
                "user_case_id": 1,
                "user_id": 1001,
                "case_id": 3,
                "tier": 3,
                "status": "pending",
            }
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_specific_card_reward_claim_grants_configured_card(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("ENVIRONMENT", "development")
    get_settings.cache_clear()
    db = _CardRewardClaimDB("specific_card", {"card_id": 46}, random_card={"id": 99, "name": "Wrong Random"})
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/rewards/claim?user_id=1001",
            json={"track_type": "s1_free", "position": 3},
        )
        body = await response.json()

        assert response.status == 200
        assert body["granted"] == [
            {
                "reward_type": "specific_card",
                "reward_amount": 1,
                "card_id": 46,
                "card_name": "Exact Card",
            }
        ]
        assert db.added_cards == [46]
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_random_card_reward_ignores_specific_card_id_meta(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("ENVIRONMENT", "development")
    get_settings.cache_clear()
    db = _CardRewardClaimDB(
        "card",
        {"card_id": 46, "rarity": ["rare"]},
        random_card={"id": 99, "name": "Random Rare"},
    )
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/rewards/claim?user_id=1001",
            json={"track_type": "s1_free", "position": 3},
        )
        body = await response.json()

        assert response.status == 200
        assert body["granted"][0]["reward_type"] == "card"
        assert body["granted"][0]["card_id"] == 99
        assert db.added_cards == [99]
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_specific_card_reward_missing_card_id_returns_readable_error(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("ENVIRONMENT", "development")
    get_settings.cache_clear()
    db = _InvalidSpecificCardRewardClaimDB()
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/rewards/claim?user_id=1001",
            json={"track_type": "s1_free", "position": 3},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "specific_card_id_required"
        assert "card_id" in body["message"]
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_specific_card_reward_duplicate_falls_back_to_coins(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("ENVIRONMENT", "development")
    get_settings.cache_clear()
    db = _CardRewardClaimDB("specific_card", {"card_id": 46}, owned_cards=[{"id": 46}])
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/rewards/claim?user_id=1001",
            json={"track_type": "s1_free", "position": 3},
        )
        body = await response.json()

        assert response.status == 200
        assert body["granted"] == [
            {
                "reward_type": "coins",
                "reward_amount": 100,
                "fallback_for": "specific_card",
                "card_id": 46,
            }
        ]
        assert db.added_cards == []
        assert db.coins_added == 100
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_reward_extrapass_promocode_preserves_indefinite_ultra_expiry():
    db = _PromoDB()

    result = await db.use_promocode(1001, "PASS")

    assert result["success"] is True
    assert db.pass_update == {"mode": "ultra", "expires_at": None, "user_id": 1001}


@pytest.mark.asyncio
async def test_reward_extrapass_promocode_grants_season_entitlement_without_expiry():
    db = _PromoDB()
    db.user_pass_row = {"extra_pass": "inactive", "extra_pass_expires_at": None}

    result = await db.use_promocode(1001, "PASS")

    assert result["success"] is True
    assert db.pass_update == {"mode": "active", "expires_at": None, "user_id": 1001}


@pytest.mark.asyncio
async def test_season_reset_preview_math_excludes_bots_and_summarizes():
    db = _SeasonResetPreviewDB()

    preview = await db.preview_season_reset(season_id=2)

    assert preview["already_completed"] is False
    assert preview["summary"] == {
        "players": 6,
        "trophies_reduced": 598,
        "keys_granted": 3,
        "coins_granted": 600,
        "stars_reset": 45,
    }
    by_user = {row["user_id"]: row for row in preview["players"]}
    assert by_user[1]["granted_keys"] == 0
    assert by_user[1]["new_trophies"] == 0
    assert by_user[2]["new_trophies"] == 300
    assert by_user[3]["new_trophies"] == 600
    assert by_user[3]["granted_keys"] == 0
    assert by_user[4]["new_trophies"] == 600
    assert by_user[4]["granted_keys"] == 1
    assert by_user[4]["granted_coins"] == 200
    assert by_user[5]["new_trophies"] == 1200
    assert by_user[5]["granted_keys"] == 0
    assert by_user[6]["new_trophies"] == 1200
    assert by_user[6]["granted_keys"] == 2
    assert 7 not in by_user
    assert preview["players_limit"] == 200
    assert preview["players_truncated"] is False


@pytest.mark.asyncio
async def test_activate_due_season_runs_reset_only_when_active_season_changes():
    changed = _ActivateSeasonDB(current_active_id=1, candidate_id=2)
    unchanged = _ActivateSeasonDB(current_active_id=2, candidate_id=2)

    await changed.activate_due_season()
    await unchanged.activate_due_season()

    assert changed.reset_calls == [
        {
            "season_id": 2,
            "previous_season_id": 1,
            "trigger": "auto",
            "admin_user_id": None,
            "reason": "automatic_season_activation",
            "require_active": True,
        }
    ]
    assert unchanged.reset_calls == []
    assert not any("UPDATE users SET stars = 0" in query for query in changed.executes)
    assert not any("UPDATE users SET stars = 0" in query for query in unchanged.executes)


@pytest.mark.asyncio
async def test_manual_season_activation_records_admin_user_id_for_reset():
    db = _ManualSeasonUpdateDB()

    await db.update_season(2, status="active", is_active=True, admin_user_id=4242)

    assert db.reset_calls == [
        {
            "season_id": 2,
            "previous_season_id": 1,
            "trigger": "admin",
            "admin_user_id": 4242,
            "reason": "manual_season_activation",
            "require_active": True,
        }
    ]


@pytest.mark.asyncio
async def test_execute_season_reset_requires_transactional_pool():
    db = _NoAcquireSeasonResetDB()

    with pytest.raises(RuntimeError, match="season_reset_requires_transactional_pool"):
        await db.execute_season_reset(season_id=1)


@pytest.mark.asyncio
async def test_execute_season_reset_active_guard_stops_before_mutations():
    db = Database.__new__(Database)
    conn = _InactiveSeasonResetConn()

    result = await db._execute_season_reset_on_conn(conn, season_id=2, require_active=True)

    assert result == {"error": "season_reset_requires_active_season", "season_id": 2}
    assert conn.fetch_calls == []
    assert conn.execute_calls == []


@pytest.mark.asyncio
async def test_execute_season_reset_partial_running_row_does_not_regrant():
    db = Database.__new__(Database)
    conn = _PartialSeasonResetConn()

    result = await db._execute_season_reset_on_conn(conn, season_id=2)

    assert result == {"error": "season_reset_in_progress", "season_id": 2}
    assert conn.fetch_calls == []
    assert conn.execute_calls == []


@pytest.mark.asyncio
async def test_execute_season_reset_revokes_extra_pass_and_cleans_season_claims():
    db = Database.__new__(Database)
    conn = _SeasonResetMutationConn()

    result = await db._execute_season_reset_on_conn(
        conn,
        season_id=2,
        previous_season_id=1,
        trigger="admin",
        admin_user_id=101,
        reason="test",
        require_active=True,
    )

    assert result["status"] == "completed"
    update_user_call = next(call for call in conn.execute_calls if "UPDATE users" in call[0])
    assert "extra_pass = 'inactive'" in update_user_call[0]
    assert "extra_pass_expires_at = NULL" in update_user_call[0]
    assert any("DELETE FROM claimed_rewards" in query for query, _ in conn.execute_calls)
    mail_call = next(call for call in conn.execute_calls if "INSERT INTO user_mail" in call[0])
    mail_attachments = mail_call[1][3]
    assert '"keys": 2' in mail_attachments
    assert '"coins": 400' in mail_attachments
    assert '"granted_keys": 2' in mail_attachments
    assert '"granted_coins": 400' in mail_attachments
    assert '"old_extra_pass": "ultra"' in mail_attachments


@pytest.mark.asyncio
async def test_execute_season_reset_deactivates_squad_boosts():
    db = Database.__new__(Database)
    conn = _SeasonResetMutationConn()

    result = await db._execute_season_reset_on_conn(
        conn,
        season_id=2,
        previous_season_id=1,
        trigger="admin",
        admin_user_id=101,
        reason="test",
        require_active=True,
    )

    assert result["status"] == "completed"
    boost_reset_call = next(call for call in conn.execute_calls if "UPDATE clans" in call[0] and "has_boost = FALSE" in call[0])
    assert "max_members - COALESCE(boost_member_slots_applied, 0)" in boost_reset_call[0]
    assert "boost_member_slots_applied = 0" in boost_reset_call[0]
    assert "Boost сквада завершен: сезонный сброс" in boost_reset_call[0]
    assert any(
        "DELETE FROM clan_upgrades" in query and "upgrade_type = 'boost'" in query
        for query, _args in conn.execute_calls
    )


@pytest.mark.asyncio
async def test_manual_season_activation_rolls_back_when_reset_fails():
    db = _ActivationRollbackDB()

    result = await db.update_season(2, status="active", is_active=True, admin_user_id=4242)

    assert result == {"error": "season_reset_failed", "season_id": 2}
    assert db.conn.transaction_state.rolled_back is True
    assert any("UPDATE seasons" in query for query, _ in db.conn.fetchrow_calls)
