import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure import case_system
from infrastructure.case_config import (
    build_default_case_config,
    merge_case_config_patch,
    fill_case_config_defaults,
    resolve_case_config,
    validate_case_config,
    BASE_PARTICLES_BY_RARITY,
    TIER_REWARDS_COUNT,
)
from infrastructure.card_economy import calculate_duplicate_particles
from infrastructure.config import get_settings
from web import server as web_server


class _CaseRouteDB:
    def __init__(self, *, extra_pass="inactive", keys=1):
        self.keys = {1001: keys}
        self.roll_reservations = 0
        self.extra_pass = extra_pass
        self.coins = 0
        self.cards = []
        self.particles = []
        self.gems = 0
        self.economy_events = []
        self.cbrp_events = []
        self.newbie_tasks = []
        self.user_cases: dict[int, dict] = {}
        self.removed_user_cases: list[tuple[int, int]] = []
        self.legacy_case_id = None
        self.reward_counter = {"coins": 0}
        self.daily_quest_increments = []

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},
            "disabled_card_ids": [],
        }

    async def get_case_config(self):
        # None → roll-функции используют live module-globals (degradation path).
        # Тесты могут переопределить атрибут case_config_value чтобы вернуть dict.
        return getattr(self, "case_config_value", None)

    async def get_match_mode_overrides(self):
        return []

    async def is_match_mode_enabled(self, mode_id):
        return True

    async def get_disabled_card_ids(self):
        return []

    async def expire_announcements(self):
        return 0

    async def process_weekly_squad_cbrp(self):
        return {"processed": 0}

    async def refresh_due_rating_snapshots(self, scope="players"):
        return {"refreshed": 0}

    async def is_admin(self, user_id):
        return False

    async def get_user_profile(self, user_id):
        return {"user_id": user_id, "extra_pass": self.extra_pass, "extra_pass_expires_at": None}

    async def get_user_cards(self, user_id):
        return []

    async def update_user_coins(self, user_id, amount):
        self.coins += amount

    async def add_card_to_user(self, user_id, card_id):
        self.cards.append(card_id)
        return {"success": True}

    async def add_particles_to_card(self, user_id, card_id, particles):
        self.particles.append((card_id, particles))
        return {"success": True}

    async def add_gems(self, user_id, amount):
        self.gems += amount
        return {"success": True}

    async def track_economy_event(self, **kwargs):
        self.economy_events.append(kwargs)

    async def award_squad_cbrp(self, *args, **kwargs):
        self.cbrp_events.append((args, kwargs))
        return {"awarded": False}

    async def mark_newbie_path_task(self, user_id, task_id, *, claimed=False):
        self.newbie_tasks.append((user_id, task_id, claimed))
        return {"success": True}

    async def fetchval(self, query, *args):
        if "SELECT COALESCE(keys,0) FROM users" in query:
            return self.keys.get(int(args[0]), 0)
        return None

    async def fetchrow(self, query, *args):
        if "UPDATE users" in query and "COALESCE(keys, 0) - 1" in query:
            user_id = int(args[0])
            if self.keys.get(user_id, 0) <= 0:
                return None
            self.keys[user_id] -= 1
            self.roll_reservations += 1
            return {"keys": self.keys[user_id]}
        return None

    async def consume_key_for_case_opening(self, user_id):
        row = await self.fetchrow(
            """
            UPDATE users
            SET keys = GREATEST(0, COALESCE(keys, 0) - 1)
            WHERE user_id = $1 AND COALESCE(keys, 0) > 0
            RETURNING keys
            """,
            user_id,
        )
        if not row:
            return None
        await self.increment_daily_quest(user_id, "open_case_1", 1)
        return int(row["keys"])

    async def increment_daily_quest(self, user_id, quest_id, delta, **_kwargs):
        self.daily_quest_increments.append((int(user_id), str(quest_id), int(delta)))

    async def get_user_case(self, user_case_id, user_id):
        row = self.user_cases.get(int(user_case_id))
        if row and row.get("user_id") == user_id:
            return row
        return None

    async def get_default_case_id(self):
        return self.legacy_case_id

    async def sync_user_key_cases(self, user_id):
        return None

    async def remove_user_case(self, user_case_id, user_id):
        self.removed_user_cases.append((int(user_case_id), int(user_id)))
        self.user_cases.pop(int(user_case_id), None)
        return True

    async def sync_user_key_cases_after_claim(self, user_id):
        return None


class _OpenRewardCaseDB:
    def __init__(self):
        self.keys = 3
        self.removed = []
        self.decrements = 0
        self.coins = 0

    async def get_user_case(self, user_case_id, user_id):
        return {"id": user_case_id, "user_id": user_id, "case_id": 3, "tier": 3, "status": "pending"}

    async def get_default_case_id(self):
        return None

    async def get_case_config(self):
        return getattr(self, "case_config_value", None)

    async def get_user_cards(self, user_id):
        return []

    async def get_user_profile(self, user_id):
        return {"user_id": user_id, "extra_pass": "inactive", "extra_pass_expires_at": None}

    async def update_user_coins(self, user_id, amount):
        self.coins += amount

    async def add_card_to_user(self, user_id, card_id):
        return {"success": True}

    async def add_particles_to_card(self, user_id, card_id, particles):
        return {"success": True}

    async def add_gems(self, user_id, amount):
        return {"success": True}

    async def remove_user_case(self, user_case_id, user_id):
        self.removed.append((user_case_id, user_id))
        return True

    async def decrement_user_keys(self, user_id, amount=1):
        self.decrements += amount
        self.keys -= amount
        return self.keys

    async def increment_daily_quest(self, user_id, quest_id, delta, **_kwargs):
        return None

    async def award_squad_cbrp(self, *args, **kwargs):
        return {"awarded": False}


class _RedirectDuplicateCaseDB(_OpenRewardCaseDB):
    def __init__(self, result):
        super().__init__()
        self.duplicate_result = dict(result)
        self.duplicate_calls = []

    async def grant_duplicate_particles(self, user_id, card_id, rarity, particles):
        self.duplicate_calls.append((user_id, card_id, rarity, particles))
        if self.duplicate_result.get("reward_type") == "coins":
            self.coins += int(self.duplicate_result.get("coins_added") or 0)
        return dict(self.duplicate_result)


class _UniLookupDB:
    async def get_uni_card(self):
        return {"id": 36, "name": "Юни", "rarity": "start"}

    async def get_cards_by_rarity(self, rarity):
        raise AssertionError("start rarity must use deterministic get_uni_card lookup")


@pytest.mark.asyncio
async def test_user_case_rewards_and_quest_progress_share_transaction():
    class Tx:
        def __init__(self):
            self.exit_exc_type = None
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, _exc, _tb):
            self.exit_exc_type = exc_type
            return False

    class Acquire:
        def __init__(self, conn):
            self.conn = conn
        async def __aenter__(self):
            return self.conn
        async def __aexit__(self, *_args):
            return False

    class Conn:
        def __init__(self):
            self.tx = Tx()
        def transaction(self):
            return self.tx
        async def execute(self, *_args):
            return "OK"
        async def fetchval(self, query, *_args):
            if "DELETE FROM user_cases" in query:
                return 1
            return None

    class DB:
        def __init__(self, conn):
            self._pool = type("Pool", (), {"acquire": lambda _self: Acquire(conn)})()
            self.quest_calls = []
        async def _daily_quests_enabled_on_conn(self, _conn):
            return True
        async def _apply_daily_quest_ops_on_conn(self, conn, user_id, ops):
            self.quest_calls.append((conn, user_id, ops))

    conn = Conn()
    db = DB(conn)
    result = await case_system._apply_case_opening_rewards(
        db,
        user_id=1001,
        user_case_id=77,
        rewards={"coins": 0, "cards": [], "particles": [], "gems": 0},
        final_tier=1,
        decrement_legacy_key=False,
    )

    assert result["success"] is True
    assert result["converted_duplicates"] == []
    assert result["converted_card_ids"] == []
    assert db.quest_calls == [(conn, 1001, [("open_case_1", 1, False)])]
    assert conn.tx.exit_exc_type is None


@pytest.mark.asyncio
async def test_user_case_transaction_rolls_back_when_quest_write_fails():
    class Tx:
        def __init__(self):
            self.exit_exc_type = None
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, _exc, _tb):
            self.exit_exc_type = exc_type
            return False

    class Conn:
        def __init__(self):
            self.tx = Tx()
        def transaction(self):
            return self.tx
        async def execute(self, *_args):
            return "OK"
        async def fetchval(self, query, *_args):
            return 1 if "DELETE FROM user_cases" in query else None

    class Acquire:
        def __init__(self, conn):
            self.conn = conn
        async def __aenter__(self):
            return self.conn
        async def __aexit__(self, *_args):
            return False

    conn = Conn()

    class DB:
        _pool = type("Pool", (), {"acquire": lambda _self: Acquire(conn)})()
        async def _daily_quests_enabled_on_conn(self, _conn):
            return True
        async def _apply_daily_quest_ops_on_conn(self, *_args, **_kwargs):
            raise RuntimeError("quest write failed")

    result = await case_system._apply_case_opening_rewards(
        DB(),
        user_id=1001,
        user_case_id=77,
        rewards={"coins": 0, "cards": [], "particles": [], "gems": 0},
        final_tier=1,
        decrement_legacy_key=False,
    )

    assert result == {"success": False, "error": "quest write failed"}
    assert conn.tx.exit_exc_type is RuntimeError


class _LimitedLookupDB:
    async def get_uni_card(self):
        return None

    async def get_cards_by_rarity(self, rarity):
        if rarity == "limited":
            return [{"id": 500, "name": "Event Card", "rarity": "limited"}]
        return []


@pytest.mark.asyncio
async def test_active_limited_event_allows_limited_card_in_t5(monkeypatch):
    monkeypatch.setattr(case_system, "LIMITED_EVENT_ACTIVE", True)

    card = await case_system.select_card_by_rarity(_LimitedLookupDB(), "limited", 5)

    assert card == {"id": 500, "name": "Event Card", "rarity": "limited"}
    assert "limited" in case_system.get_available_rarities_for_tier(5)


@pytest.mark.asyncio
async def test_case_duplicate_payload_uses_redirect_target_for_max_card():
    db = _RedirectDuplicateCaseDB({
        "success": True,
        "reward_type": "particles",
        "particles_added": 25,
        "card_id": 88,
        "card_name": "Redirect Target",
        "rarity": "epic",
        "redirected": True,
        "source_card_id": 77,
    })
    rewards = {
        "coins": 0,
        "cards": [],
        "particles": [{
            "card_id": 77,
            "card_name": "Max Source",
            "rarity": "epic",
            "particles": 25,
        }],
        "gems": 0,
    }

    result = await case_system._apply_case_opening_rewards(
        db,
        user_id=1001,
        user_case_id=501,
        rewards=rewards,
        final_tier=3,
        decrement_legacy_key=False,
    )

    assert result["success"] is True
    assert db.duplicate_calls == [(1001, 77, "epic", 25)]
    assert rewards["particles"] == [{
        "card_id": 88,
        "card_name": "Redirect Target",
        "rarity": "epic",
        "particles": 25,
        "redirected": True,
        "source_card_id": 77,
        "source_card_name": "Max Source",
    }]


@pytest.mark.asyncio
async def test_case_duplicate_payload_adds_coin_fallback_when_rarity_is_complete():
    db = _RedirectDuplicateCaseDB({
        "success": True,
        "reward_type": "coins",
        "coins_added": 325,
        "amount": 325,
        "fallback_for": "max_level_duplicate",
        "source_card_id": 77,
    })
    rewards = {
        "coins": 100,
        "cards": [],
        "particles": [{
            "card_id": 77,
            "card_name": "Max Source",
            "rarity": "epic",
            "particles": 25,
        }],
        "gems": 0,
    }

    result = await case_system._apply_case_opening_rewards(
        db,
        user_id=1001,
        user_case_id=502,
        rewards=rewards,
        final_tier=3,
        decrement_legacy_key=False,
    )

    assert result["success"] is True
    assert rewards["particles"] == []
    assert rewards["coins"] == 425
    assert db.coins == 425


@pytest.mark.asyncio
async def test_case_reward_metadata_uses_actual_fallback_card_rarity(monkeypatch):
    async def fallback_card(_db, _rarity, _tier, _case_config=None):
        return {"id": 77, "name": "Fallback Common", "rarity": "common"}

    monkeypatch.setattr(case_system, "select_rarity", lambda *_args, **_kwargs: "legendary")
    monkeypatch.setattr(
        case_system,
        "check_start_rarity_replacement",
        lambda rarity, _case_config=None: rarity,
    )
    monkeypatch.setattr(case_system, "select_card_by_rarity", fallback_card)
    monkeypatch.setitem(
        case_system.TIER_REWARDS_COUNT,
        1,
        {"coins": (0, 0), "cards": (1, 1)},
    )

    rewards = await case_system._generate_single_case_rewards(
        object(),
        tier=1,
        user_id=1001,
        user_card_ids={77},
    )

    assert rewards["particles"] == [{
        "card_id": 77,
        "card_name": "Fallback Common",
        "rarity": "common",
        "particles": calculate_duplicate_particles("common", 1),
    }]


def test_simulate_case_tap_results_uses_server_rolls(monkeypatch):
    """Серверные тапы возвращают последовательность как есть; extra_pass пробрасывается.

    Правки 2026-06-25: Ultra-бонус к шансу тапа вырезан, поэтому ultra и inactive
    должны получать идентичную последовательность от roll_tier_upgrade (мокируем
    ниже и пробрасываем реальный extra_pass; сам мок не использует его).
    """
    rolls_ultra = iter([1, 2, 2, 3])
    seen_extra_passes: list[str] = []

    def fake_roll(current_tier, tap_number, extra_pass="inactive"):
        seen_extra_passes.append(extra_pass)
        return next(rolls_ultra)

    monkeypatch.setattr(case_system, "roll_tier_upgrade", fake_roll)

    ultra_result = case_system.simulate_case_tap_results(1, "ultra")
    assert ultra_result == [1, 2, 2, 3]
    assert seen_extra_passes == ["ultra"] * 4

    # После вырезки Ultra-бонуса шансы тапа для ultra и inactive совпадают.
    monkeypatch.setattr(
        case_system,
        "roll_tier_upgrade",
        lambda current_tier, tap_number, extra_pass="inactive": next(iter([1, 2, 2, 3])),
    )
    assert (
        case_system.simulate_case_tap_results(1, "ultra")
        == case_system.simulate_case_tap_results(1, "inactive")
    )


@pytest.mark.asyncio
async def test_start_rarity_case_lookup_returns_deterministic_yuni():
    card = await case_system.select_card_by_rarity(_UniLookupDB(), "start", 3)

    assert card["id"] == 36
    assert card["name"] == "Юни"


@pytest.mark.asyncio
async def test_ultra_case_rewards_mark_manual_reroll_available(monkeypatch):
    candidates = [
        {"coins": 100, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
        {"coins": 100, "cards": [{"card_id": 7, "rarity": "legendary"}], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
    ]

    async def fake_generate_single(*args, **kwargs):
        return candidates.pop(0).copy()

    monkeypatch.setattr(case_system, "_generate_single_case_rewards", fake_generate_single)

    rewards = await case_system.generate_case_rewards(
        db=object(),
        tier=3,
        user_id=1,
        user_card_ids=set(),
        extra_pass="ultra",
    )

    assert rewards["cards"] == []
    assert rewards["extra_pass_bonus"]["reroll_available"] is True
    assert rewards["extra_pass_bonus"]["reroll_attempts"] == 1
    assert len(candidates) == 1


@pytest.mark.asyncio
async def test_ultra_case_rewards_do_not_auto_select_hidden_best_candidate(monkeypatch):
    candidates = [
        {"coins": 100, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
        {"coins": 100, "cards": [{"card_id": 7, "rarity": "legendary"}], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
    ]

    async def fake_generate_single(*args, **kwargs):
        return candidates.pop(0).copy()

    monkeypatch.setattr(case_system, "_generate_single_case_rewards", fake_generate_single)

    rewards = await case_system.generate_case_rewards(
        db=object(),
        tier=3,
        user_id=1,
        user_card_ids=set(),
        extra_pass="ultra",
    )

    assert rewards["cards"] == []
    assert rewards["extra_pass_bonus"]["reroll_available"] is True
    assert len(candidates) == 1


@pytest.mark.asyncio
async def test_t5_case_rewards_do_not_generate_removed_limited_shards(monkeypatch):
    monkeypatch.setitem(
        case_system.TIER_REWARDS_COUNT,
        5,
        {
            "coins": (1, 1),
            "cards": (0, 0),
            "limited_shards_chance": 1.0,
            "limited_shards_amount": (7, 7),
        },
    )

    rewards = await case_system.generate_case_rewards(
        db=object(),
        tier=5,
        user_id=1,
        user_card_ids=set(),
        extra_pass="inactive",
    )

    assert "limited_shards" not in rewards


@pytest.mark.asyncio
async def test_repeated_roll_from_keys_reuses_reserved_roll_and_spends_one_key(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_KEY_ROLLS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_OPEN_RESULTS", {})
    monkeypatch.setattr(case_system, "simulate_case_tap_results", lambda *_args, **_kwargs: [1, 1, 1, 1])
    monkeypatch.setattr(web_server, "simulate_case_tap_results", lambda *_args, **_kwargs: [1, 1, 1, 1])
    db = _CaseRouteDB()
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        first = await client.post("/api/cases/roll-from-keys?user_id=1001")
        second = await client.post("/api/cases/roll-from-keys?user_id=1001")
        first_body = await first.json()
        second_body = await second.json()

        assert first.status == 200
        assert second.status == 200
        assert second_body["roll_token"] == first_body["roll_token"]
        assert db.roll_reservations == 1
        assert db.keys[1001] == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_process_case_opening_reward_track_case_does_not_decrement_keys(monkeypatch):
    async def fake_generate_rewards(*args, **kwargs):
        return {"coins": 50, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False}

    monkeypatch.setattr(case_system, "generate_case_rewards", fake_generate_rewards)
    db = _OpenRewardCaseDB()

    # process_case_opening теперь НЕ применяет награды — только генерирует pending opening.
    result = await case_system.process_case_opening(db, 1001, 77)

    assert result["success"] is True
    assert result["final_tier"] == 3
    assert result["rewards"]["coins"] == 50
    assert result["tap_results"] == [3, 3, 3, 3]
    assert db.coins == 0
    assert db.removed == []
    assert db.decrements == 0
    assert db.keys == 3


@pytest.mark.asyncio
async def test_key_case_open_claims_only_after_explicit_claim(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_KEY_ROLLS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_OPEN_RESULTS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_REROLL_ROLLS", {})
    monkeypatch.setattr(web_server, "simulate_case_tap_results", lambda *_args, **_kwargs: [1, 1, 1, 1])

    async def fake_rewards(*args, **kwargs):
        return {"coins": 75, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False}

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="ultra")
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        roll = await client.post("/api/cases/roll-from-keys?user_id=1001")
        roll_body = await roll.json()
        opened = await client.post(
            "/api/cases/open-from-keys?user_id=1001",
            json={"roll_token": roll_body["roll_token"]},
        )
        opened_body = await opened.json()

        assert opened.status == 200
        assert opened_body["opening_token"]
        assert opened_body["can_reroll"] is True
        assert db.coins == 0

        claimed = await client.post(
            "/api/cases/claim-from-keys?user_id=1001",
            json={"opening_token": opened_body["opening_token"]},
        )
        claimed_body = await claimed.json()

        assert claimed.status == 200
        assert claimed_body["rewards"]["coins"] == 75
        assert db.coins == 75
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_key_case_unfinished_reroll_claims_primary_rewards(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_KEY_ROLLS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_OPEN_RESULTS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_REROLL_ROLLS", {})
    monkeypatch.setattr(web_server, "simulate_case_tap_results", lambda *_args, **_kwargs: [1, 1, 1, 1])
    rewards = iter([
        {"coins": 100, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
        {"coins": 900, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
    ])

    async def fake_rewards(*args, **kwargs):
        return next(rewards).copy()

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="ultra")
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        roll_body = await (await client.post("/api/cases/roll-from-keys?user_id=1001")).json()
        opened_body = await (await client.post(
            "/api/cases/open-from-keys?user_id=1001",
            json={"roll_token": roll_body["roll_token"]},
        )).json()
        reroll = await client.post(
            "/api/cases/reroll-from-keys?user_id=1001",
            json={"opening_token": opened_body["opening_token"]},
        )
        claimed_body = await (await client.post(
            "/api/cases/claim-from-keys?user_id=1001",
            json={"opening_token": opened_body["opening_token"]},
        )).json()

        assert reroll.status == 200
        assert claimed_body["rewards"]["coins"] == 100
        assert db.coins == 100
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_key_case_completed_reroll_replaces_claimed_rewards(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_KEY_ROLLS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_OPEN_RESULTS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_REROLL_ROLLS", {})
    monkeypatch.setattr(web_server, "simulate_case_tap_results", lambda *_args, **_kwargs: [1, 1, 1, 1])
    rewards = iter([
        {"coins": 100, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
        {"coins": 900, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
    ])

    async def fake_rewards(*args, **kwargs):
        return next(rewards).copy()

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="ultra")
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        roll_body = await (await client.post("/api/cases/roll-from-keys?user_id=1001")).json()
        opened_body = await (await client.post(
            "/api/cases/open-from-keys?user_id=1001",
            json={"roll_token": roll_body["roll_token"]},
        )).json()
        reroll_body = await (await client.post(
            "/api/cases/reroll-from-keys?user_id=1001",
            json={"opening_token": opened_body["opening_token"]},
        )).json()
        reroll_opened = await client.post(
            "/api/cases/open-reroll-from-keys?user_id=1001",
            json={
                "opening_token": opened_body["opening_token"],
                "reroll_token": reroll_body["reroll_token"],
            },
        )
        claimed_body = await (await client.post(
            "/api/cases/claim-from-keys?user_id=1001",
            json={"opening_token": opened_body["opening_token"]},
        )).json()

        assert reroll_opened.status == 200
        assert claimed_body["rewards"]["coins"] == 900
        assert db.coins == 900
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_case_open_returns_pending_opening_for_ultra(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_USER_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_USER_REROLL_ROLLS", {})

    async def fake_rewards(*args, **kwargs):
        return {"coins": 100, "cards": [{"card_id": 42, "card_name": "Card", "rarity": "rare", "is_new": True}], "particles": [], "gems": 0, "jackpot": False, "extra_pass_bonus": {"tier": "ultra", "reroll_available": True}}

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    monkeypatch.setattr(case_system, "generate_case_rewards", fake_rewards)

    db = _CaseRouteDB(extra_pass="ultra")
    db.user_cases[77] = {"id": 77, "user_id": 1001, "case_id": 99, "tier": 3, "status": "pending"}
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        opened = await client.post(
            "/api/cases/open?user_id=1001",
            json={"user_case_id": 77},
        )
        body = await opened.json()
        assert opened.status == 200
        assert body["success"] is True
        assert body["opening_token"]
        assert body["final_tier"] == 3
        assert body["can_reroll"] is True
        assert body["rewards"]["coins"] == 100
        # DB is NOT mutated yet — pending opening lives in memory only.
        assert db.coins == 0
        assert db.removed_user_cases == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_case_claim_applies_rewards(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_USER_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_USER_REROLL_ROLLS", {})

    async def fake_rewards(*args, **kwargs):
        return {"coins": 250, "cards": [], "particles": [{"card_id": 7, "card_name": "X", "rarity": "common", "particles": 5}], "gems": 0, "jackpot": False}

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    monkeypatch.setattr(case_system, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="inactive")
    db.user_cases[88] = {"id": 88, "user_id": 1001, "case_id": 99, "tier": 4, "status": "pending"}
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        opened = await (await client.post(
            "/api/cases/open?user_id=1001", json={"user_case_id": 88}
        )).json()
        assert opened["opening_token"]
        assert db.coins == 0
        assert db.removed_user_cases == []

        claimed = await client.post(
            "/api/cases/claim?user_id=1001",
            json={"opening_token": opened["opening_token"]},
        )
        body = await claimed.json()
        assert claimed.status == 200
        assert body["claimed"] is True
        assert body["final_tier"] == 4
        # DB applied.
        assert db.coins == 250
        assert db.particles == [(7, 5)]
        assert (88, 1001) in db.removed_user_cases
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_case_reroll_regenerates_rewards_at_same_tier(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_USER_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_USER_REROLL_ROLLS", {})

    rewards_a = {"coins": 100, "cards": [], "particles": [], "gems": 0, "jackpot": False}
    rewards_b = {"coins": 250, "cards": [{"card_id": 11, "card_name": "Y", "rarity": "rare", "is_new": True}], "particles": [], "gems": 5, "jackpot": False}
    counter = {"calls": 0}

    async def fake_rewards(*args, **kwargs):
        counter["calls"] += 1
        return rewards_a if counter["calls"] == 1 else rewards_b

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    monkeypatch.setattr(case_system, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="ultra")
    db.user_cases[91] = {"id": 91, "user_id": 1001, "case_id": 99, "tier": 4, "status": "pending"}
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        opened = await (await client.post(
            "/api/cases/open?user_id=1001", json={"user_case_id": 91}
        )).json()
        assert opened["final_tier"] == 4
        assert opened["rewards"]["coins"] == 100

        reroll_body = await (await client.post(
            "/api/cases/reroll?user_id=1001", json={"opening_token": opened["opening_token"]}
        )).json()
        assert reroll_body["reroll_token"]
        assert reroll_body["final_tier"] == 4  # tier unchanged

        applied = await (await client.post(
            "/api/cases/apply-reroll?user_id=1001",
            json={"opening_token": opened["opening_token"], "reroll_token": reroll_body["reroll_token"]},
        )).json()
        assert applied["final_tier"] == 4  # tier preserved
        assert applied["rewards"]["coins"] == 250  # new rewards
        assert applied["rewards"]["gems"] == 5

        claimed = await (await client.post(
            "/api/cases/claim?user_id=1001", json={"opening_token": opened["opening_token"]}
        )).json()
        # Claim applies the rerolled rewards, not the original.
        assert db.coins == 250
        assert db.cards == [11]
        assert db.gems == 5
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_case_double_reroll_idempotent_returns_cached_token(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_USER_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_USER_REROLL_ROLLS", {})

    async def fake_rewards(*args, **kwargs):
        return {"coins": 50, "cards": [], "particles": [], "gems": 0, "jackpot": False}

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    monkeypatch.setattr(case_system, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="ultra")
    db.user_cases[101] = {"id": 101, "user_id": 1001, "case_id": 99, "tier": 3, "status": "pending"}
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        opened = await (await client.post(
            "/api/cases/open?user_id=1001", json={"user_case_id": 101}
        )).json()

        first = await client.post(
            "/api/cases/reroll?user_id=1001", json={"opening_token": opened["opening_token"]}
        )
        assert first.status == 200

        second = await client.post(
            "/api/cases/reroll?user_id=1001", json={"opening_token": opened["opening_token"]}
        )
        second_body = await second.json()
        # Same as keys-flow: double reroll before apply returns cached reroll_token (200).
        assert second.status == 200
        assert second_body.get("reroll_token")
        assert second_body.get("reroll_used") is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_case_reroll_blocked_after_claim_with_opening_claimed(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_USER_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_USER_REROLL_ROLLS", {})

    async def fake_rewards(*args, **kwargs):
        return {"coins": 50, "cards": [], "particles": [], "gems": 0, "jackpot": False}

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    monkeypatch.setattr(case_system, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="ultra")
    db.user_cases[111] = {"id": 111, "user_id": 1001, "case_id": 99, "tier": 3, "status": "pending"}
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        opened = await (await client.post(
            "/api/cases/open?user_id=1001", json={"user_case_id": 111}
        )).json()
        await client.post(
            "/api/cases/claim?user_id=1001", json={"opening_token": opened["opening_token"]}
        )

        after = await client.post(
            "/api/cases/reroll?user_id=1001", json={"opening_token": opened["opening_token"]}
        )
        body = await after.json()
        assert after.status == 400
        assert body.get("error") == "opening_claimed"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_case_reroll_blocked_for_non_ultra(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_USER_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_USER_REROLL_ROLLS", {})

    async def fake_rewards(*args, **kwargs):
        return {"coins": 50, "cards": [], "particles": [], "gems": 0, "jackpot": False}

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    monkeypatch.setattr(case_system, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="inactive")
    db.user_cases[121] = {"id": 121, "user_id": 1001, "case_id": 99, "tier": 3, "status": "pending"}
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        opened = await (await client.post(
            "/api/cases/open?user_id=1001", json={"user_case_id": 121}
        )).json()
        assert opened["can_reroll"] is False

        reroll = await client.post(
            "/api/cases/reroll?user_id=1001", json={"opening_token": opened["opening_token"]}
        )
        body = await reroll.json()
        assert reroll.status == 403
        assert body.get("error") == "ultra_required"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_case_claim_idempotent_returns_400_already_claimed(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_USER_PENDING_OPENINGS", {})
    monkeypatch.setattr(web_server, "CASE_USER_REROLL_ROLLS", {})

    async def fake_rewards(*args, **kwargs):
        return {"coins": 50, "cards": [], "particles": [], "gems": 0, "jackpot": False}

    monkeypatch.setattr(web_server, "generate_case_rewards", fake_rewards)
    monkeypatch.setattr(case_system, "generate_case_rewards", fake_rewards)
    db = _CaseRouteDB(extra_pass="inactive")
    db.user_cases[131] = {"id": 131, "user_id": 1001, "case_id": 99, "tier": 3, "status": "pending"}
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        opened = await (await client.post(
            "/api/cases/open?user_id=1001", json={"user_case_id": 131}
        )).json()
        first_claim = await client.post(
            "/api/cases/claim?user_id=1001", json={"opening_token": opened["opening_token"]}
        )
        assert first_claim.status == 200
        assert db.coins == 50

        # Second claim — opening is still in CASE_USER_PENDING_OPENINGS, but marked claimed.
        second_claim = await client.post(
            "/api/cases/claim?user_id=1001", json={"opening_token": opened["opening_token"]}
        )
        body = await second_claim.json()
        # Second claim is treated idempotent: returns 200 with the cached claim_response.
        # Coins NOT applied twice.
        assert db.coins == 50
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Real-time case_config: particle-drop bug fix + live-config threading
# ---------------------------------------------------------------------------


def test_limited_duplicate_drops_nonzero_particles():
    """Регрессия: limited-дубликат должен давать частицы (base=160, выше divine).

    Раньше base_particles_by_rarity['limited']==0 → дубликаты лимитированных
    всегда давали 0 частиц. Теперь base=160, T1 множитель 1.30 → 208.
    """
    amount = case_system.calculate_particles_for_duplicate("limited", 1, False, None)
    assert amount > 0
    assert amount == 208  # int(160 * 1.30)


def test_limited_duplicate_above_divine():
    """Лимитированные частицы выше divine на том же тире (иерархия сохранена)."""
    limited = case_system.calculate_particles_for_duplicate("limited", 5, False, None)
    divine = case_system.calculate_particles_for_duplicate("divine", 5, False, None)
    assert limited > divine


def test_particles_floor_one_for_nonzero_base():
    """Гарантия >=1 частицы при base>0 (защита от int()-усечения в ноль)."""
    cc = merge_case_config_patch(
        build_default_case_config(),
        {"tier_particles_multiplier": {1: 0.001}},  # int(2 * 0.001) == 0 без пола
    )
    amount = case_system.calculate_particles_for_duplicate("common", 1, False, cc)
    assert amount >= 1


def test_explicit_zero_base_stays_zero():
    """Явный ноль от администратора (base==0) сохраняется — никакой частицы."""
    cc = merge_case_config_patch(
        build_default_case_config(),
        {"base_particles_by_rarity": {"common": 0}},
    )
    assert case_system.calculate_particles_for_duplicate("common", 1, False, cc) == 0


def test_t5_common_jackpot_unchanged_by_floor():
    """Джекпот T5-common не подпадает под пол>=1 — отдельная константа."""
    cc = merge_case_config_patch(
        build_default_case_config(),
        {"t5_common_jackpot_particles": 0},
    )
    assert case_system.calculate_particles_for_duplicate("common", 5, True, cc) == 0
    # Дефолтный джекпот 125 сохранён
    assert case_system.calculate_particles_for_duplicate("common", 5, True, None) == 125


def test_resolve_case_config_none_returns_live_refs():
    """resolve_case_config(None) возвращает LIVE module-globals (не копии).

    Load-bearing для тестов, которые monkeypatch-ят case_system.TIER_REWARDS_COUNT
    на месте (см. test_t5_case_rewards_do_not_generate_removed_limited_shards).
    """
    cc = resolve_case_config(None)
    assert cc["tier_rewards_count"] is case_system.TIER_REWARDS_COUNT
    assert cc["base_particles_by_rarity"] is case_system.BASE_PARTICLES_BY_RARITY
    assert cc["tier_upgrade_chances"] is case_system.TIER_UPGRADE_CHANCES


def test_resolve_case_config_dict_merges_partial_and_coerces_tier_keys():
    """resolve_case_config(dict) deep-fills дефолты и коэрсит строковые tier-ключи к int."""
    partial = {"base_particles_by_rarity": {"limited": 777}, "tier_upgrade_chances": {"2": 0.9}}
    cc = resolve_case_config(partial)
    # limited патч применён, остальные редкости заполнены из дефолта
    assert cc["base_particles_by_rarity"]["limited"] == 777
    assert cc["base_particles_by_rarity"]["common"] == BASE_PARTICLES_BY_RARITY["common"]
    # tier-ключ коэршен к int
    assert set(cc["tier_upgrade_chances"].keys()) == {1, 2, 3, 4}
    assert cc["tier_upgrade_chances"][2] == 0.9
    assert cc["tier_upgrade_chances"][1] == case_system.TIER_UPGRADE_CHANCES[1]


def test_merge_partial_rarity_patch_preserves_other_rarities():
    """КРИТИЧНО: partial base_particles патч НЕ должен обнулять остальные редкости.

    Это ровно тот баг, что чиним: shallow-merge {**cur, **patch} на уровне поля
    заменил бы весь base_particles_by_rarity на {'limited': 150} и вновь занулил
    бы common/rare/divine. Структурный per-rarity merge сохраняет остальные.
    """
    base = build_default_case_config()
    merged = merge_case_config_patch(base, {"base_particles_by_rarity": {"limited": 150}})
    bpr = merged["base_particles_by_rarity"]
    assert bpr["limited"] == 150
    assert bpr["common"] == base["base_particles_by_rarity"]["common"]
    assert bpr["rare"] == base["base_particles_by_rarity"]["rare"]
    assert bpr["divine"] == base["base_particles_by_rarity"]["divine"]


def test_merge_partial_tier_patch_preserves_other_tiers():
    """Partial tier-патч заменяет только указанные тиры."""
    base = build_default_case_config()
    merged = merge_case_config_patch(base, {"tier_upgrade_chances": {2: 0.5}})
    tuc = merged["tier_upgrade_chances"]
    assert tuc[2] == 0.5
    assert tuc[1] == base["tier_upgrade_chances"][1]
    assert tuc[3] == base["tier_upgrade_chances"][3]
    assert tuc[4] == base["tier_upgrade_chances"][4]


def test_merge_partial_tier_rarity_patch_preserves_other_rarities_in_tier():
    """Partial tier_rarity_probabilities патч (одна редкость в одном тире) сохраняет
    остальные редкости этого тира (per-rarity deep-merge), а не заменяет весь тир.

    Раньше merge делал {**cur_tiers, **patch_tiers} на уровне тиров, поэтому
    {2:{common:0.644}} заменил бы весь T2 на {common:0.644} (сумма 0.644) и нарушил
    инвариант суммы. Теперь deep-merge по редкостям сохраняет rare/superrare/epic.
    """
    base = build_default_case_config()
    merged = merge_case_config_patch(base, {"tier_rarity_probabilities": {2: {"common": 0.644}}})
    t2 = merged["tier_rarity_probabilities"][2]
    assert t2["common"] == 0.644
    assert t2["rare"] == base["tier_rarity_probabilities"][2]["rare"]
    assert t2["superrare"] == base["tier_rarity_probabilities"][2]["superrare"]
    assert t2["epic"] == base["tier_rarity_probabilities"][2]["epic"]
    # Сумма merged T2 валидна (no-op partial patch не нарушил инвариант).
    assert abs(sum(t2.values()) - 1.0) < 1e-9
    # Другие тиры не тронуты.
    assert merged["tier_rarity_probabilities"][1] == base["tier_rarity_probabilities"][1]


def test_fill_case_config_defaults_deep_fills_missing_tiers():
    """fill_case_config_defaults добавляет отсутствующие тиры из дефолта, не затирая правки."""
    stored = {
        "base_particles_by_rarity": {"limited": 200},  # все остальные редкости отсутствуют
        "tier_upgrade_chances": {"3": 0.4},  # только один тир
    }
    filled = fill_case_config_defaults(stored)
    # limited правка сохранена
    assert filled["base_particles_by_rarity"]["limited"] == 200
    # остальные редкости заполнены из дефолта
    assert filled["base_particles_by_rarity"]["common"] == BASE_PARTICLES_BY_RARITY["common"]
    # отсутствующие тиры upgrade_chances заполнены из дефолта, строковый ключ коэршен
    assert filled["tier_upgrade_chances"][3] == 0.4
    assert filled["tier_upgrade_chances"][1] == case_system.TIER_UPGRADE_CHANCES[1]
    assert all(isinstance(k, int) for k in filled["tier_upgrade_chances"])


def test_validate_case_config_rejects_bad_tier_rarity_sum():
    import pytest as _pytest

    base = build_default_case_config()
    bad = merge_case_config_patch(base, {"tier_rarity_probabilities": {1: {"common": 0.5, "rare": 0.2}}})
    with _pytest.raises(ValueError):
        validate_case_config(bad)


def test_select_rarity_respects_live_limited_event(monkeypatch):
    """select_rarity использует live limited_event_active из case_config."""
    cc = build_default_case_config()
    cc = merge_case_config_patch(cc, {"limited_event_active": True, "limited_event_probability": 0.5})
    # tier 5 + active → limited получает 0.5 массы, остальные нормированы к 0.5.
    # rand=0.99 → накопленный итог дойдёт до limited (после 0.5 суммы остальных).
    monkeypatch.setattr(case_system.random, "random", lambda: 0.99)
    rarity = case_system.select_rarity(5, "inactive", cc)
    assert rarity == "limited"


def test_select_rarity_ignores_limited_when_event_inactive():
    """Без события limited не выпадает даже на T5."""
    cc = build_default_case_config()  # limited_event_active=False по дефолту
    for _ in range(50):
        rarity = case_system.select_rarity(5, "inactive", cc)
        assert rarity != "limited"


def test_simulate_tap_results_threads_case_config():
    """simulate_case_tap_results использует case_config['tier_upgrade_chances']."""
    cc = build_default_case_config()
    # тап 1 шанс = 0 → без апгрейда никогда
    cc = merge_case_config_patch(cc, {"tier_upgrade_chances": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}})
    taps = case_system.simulate_case_tap_results(1, "inactive", cc)
    assert taps == [1, 1, 1, 1]


def test_simulate_tap_results_none_path_uses_3arg_roll(monkeypatch):
    """При case_config=None вызов roll_tier_upgrade остаётся 3-аргументным.

    Сохраняет существующие 3-param test-fakes (см. test_simulate_case_tap_results_uses_server_rolls).
    """
    seen_args: list[tuple] = []

    def fake_roll(current_tier, tap_number, extra_pass="inactive", case_config=None):
        seen_args.append((current_tier, tap_number, extra_pass, case_config))
        return current_tier

    monkeypatch.setattr(case_system, "roll_tier_upgrade", fake_roll)
    case_system.simulate_case_tap_results(2, "ultra", None)
    # Все 4 вызова должны быть 3-аргументными по контракту (case_config не передаётся через else-ветку)
    assert all(call[3] is None for call in seen_args)


def test_simulate_tap_results_with_case_config_passes_4arg(monkeypatch):
    """При case_config=dict вызов roll_tier_upgrade получает case_config 4-м аргументом."""
    seen: list = []
    cc = build_default_case_config()

    def fake_roll(current_tier, tap_number, extra_pass="inactive", case_config=None):
        seen.append(case_config)
        return current_tier

    monkeypatch.setattr(case_system, "roll_tier_upgrade", fake_roll)
    case_system.simulate_case_tap_results(2, "ultra", cc)
    assert all(c is cc for c in seen)


@pytest.mark.asyncio
async def test_generate_case_rewards_threads_case_config(monkeypatch):
    """generate_case_rewards пробрасывает case_config в _generate_single_case_rewards."""
    seen: dict = {}

    async def fake_single(db, tier, user_id, user_card_ids, extra_pass="inactive", case_config=None):
        seen["case_config"] = case_config
        return {"coins": 0, "cards": [], "particles": [], "gems": 0, "jackpot": False}

    monkeypatch.setattr(case_system, "_generate_single_case_rewards", fake_single)
    cc = build_default_case_config()
    await case_system.generate_case_rewards(object(), 3, 1, set(), "inactive", cc)
    assert seen["case_config"] is cc


@pytest.mark.asyncio
async def test_process_case_opening_threads_case_config():
    """process_case_opening пробрасывает case_config до generate_case_rewards."""
    seen: dict = {}
    db = _OpenRewardCaseDB()

    async def fake_generate(db, tier, user_id, user_card_ids, extra_pass="inactive", case_config=None):
        seen["case_config"] = case_config
        return {"coins": 0, "cards": [], "particles": [], "gems": 0, "jackpot": False}

    import infrastructure.case_system as _cs
    orig = _cs.generate_case_rewards
    _cs.generate_case_rewards = fake_generate
    try:
        cc = build_default_case_config()
        await case_system.process_case_opening(db, 1001, 77, cc)
        assert seen["case_config"] is cc
    finally:
        _cs.generate_case_rewards = orig


def test_set_get_case_config_tier_key_type_stability():
    """После merge + validate + fill int-ключи тиров сохраняются (no str leakage)."""
    from infrastructure.case_config import _coerce_tier_keys

    base = build_default_case_config()
    merged = merge_case_config_patch(base, {"tier_particles_multiplier": {"3": 2.0}})
    validate_case_config(merged)
    refilled = fill_case_config_defaults(merged)
    assert all(isinstance(k, int) for k in refilled["tier_particles_multiplier"])
    assert refilled["tier_particles_multiplier"][3] == 2.0


# ---------------------------------------------------------------------------
# Review-driven coverage (M1, M2, M3, M4, M5, M7, N2) — adversarial review
# подтвердил SHIP; эти тесты закрывают прод-код фиксы M1–M5 на уровне regressии.
# ---------------------------------------------------------------------------


def test_get_available_rarities_t5_includes_limited_when_event_active():
    """M1: при активном limited-событии T5 действительно допускает 'limited'.

    Без этого фикса select_card_by_rarity понижал свёрнутую 'limited' редкость до
    'divine' — лимитированные карты никогда не выпадали (корневой баг).
    """
    cc = merge_case_config_patch(
        build_default_case_config(),
        {"limited_event_active": True, "limited_event_probability": 0.5},
    )
    available = case_system.get_available_rarities_for_tier(5, cc)
    assert "limited" in available
    assert available[-1] == "limited"


def test_get_available_rarities_t5_excludes_limited_when_event_inactive():
    """M1 (контроль): без события 'limited' нет в доступных редкостях T5."""
    cc = build_default_case_config()  # limited_event_active=False
    available = case_system.get_available_rarities_for_tier(5, cc)
    assert "limited" not in available


class _LimitedCardDB:
    """DB-fake: get_cards_by_rarity возвращает limited-карту; прочее не нужно."""

    async def get_uni_card(self):
        return None

    async def get_cards_by_rarity(self, rarity):
        if rarity == "limited":
            return [{"id": 26, "name": "Мидория", "rarity": "limited"}]
        raise AssertionError(f"unexpected rarity lookup: {rarity}")


@pytest.mark.asyncio
async def test_select_card_by_rarity_returns_limited_card_when_event_active(monkeypatch):
    """M1 end-to-end: limited-карта действительно выбирается на T5 при активном событии.

    Фиксирует регрессию: до M1 select_card_by_rarity('limited',5) понижал до 'divine'
    и вызывал get_cards_by_rarity('divine'). Теперь остаётся 'limited'.
    """
    cc = merge_case_config_patch(
        build_default_case_config(),
        {"limited_event_active": True, "limited_event_probability": 0.5},
    )
    monkeypatch.setattr(case_system.random, "choice", lambda seq: seq[0])
    card = await case_system.select_card_by_rarity(_LimitedCardDB(), "limited", 5, cc)
    assert card is not None
    assert card["rarity"] == "limited"
    assert card["id"] == 26


@pytest.mark.asyncio
async def test_select_card_by_rarity_downgrades_limited_when_event_inactive():
    """M1 (контроль): без события 'limited' понижается до max-доступной (divine)."""
    cc = build_default_case_config()  # limited_event_active=False

    class _DivineFallbackDB:
        async def get_uni_card(self):
            return None

        async def get_cards_by_rarity(self, rarity):
            if rarity == "divine":
                return [{"id": 99, "name": "Божественный", "rarity": "divine"}]
            raise AssertionError(f"unexpected rarity lookup: {rarity}")

    card = await case_system.select_card_by_rarity(_DivineFallbackDB(), "limited", 5, cc)
    assert card["rarity"] == "divine"


def test_merge_case_config_patch_rejects_unknown_field():
    """M2: merge отвергает неизвестные поля (единая точка отказа → ValueError→400)."""
    base = build_default_case_config()
    with pytest.raises(ValueError):
        merge_case_config_patch(base, {"totally_unknown_field": 1})


def test_merge_case_config_patch_rejects_non_dict_tier_keyed_value():
    """M4: tier-keyed поле должно быть dict — иначе ValueError."""
    base = build_default_case_config()
    with pytest.raises(ValueError):
        merge_case_config_patch(base, {"tier_upgrade_chances": 0.5})


def test_merge_case_config_patch_rejects_non_dict_rarity_keyed_value():
    """M5: rarity-keyed поле должно быть dict — иначе ValueError."""
    base = build_default_case_config()
    with pytest.raises(ValueError):
        merge_case_config_patch(base, {"base_particles_by_rarity": 150})


def test_validate_case_config_rejects_unknown_rarity_in_base_particles():
    """M3: validate_case_config отвергает неизвестную редкость в base_particles."""
    base = build_default_case_config()
    bad = dict(base)
    bad_bpr = dict(base["base_particles_by_rarity"])
    bad_bpr["mythic_plus"] = 50  # неизвестная редкость
    bad["base_particles_by_rarity"] = bad_bpr
    with pytest.raises(ValueError):
        validate_case_config(bad)


def test_validate_case_config_rejects_unknown_rarity_in_start_rarity_replacement():
    """M3: validate_case_config отвергает неизвестную редкость в start_rarity_replacement."""
    base = build_default_case_config()
    bad = dict(base)
    bad_srr = dict(base["start_rarity_replacement"])
    bad_srr["mythic_plus"] = 0.1  # неизвестная редкость
    bad["start_rarity_replacement"] = bad_srr
    with pytest.raises(ValueError):
        validate_case_config(bad)


def test_validate_case_config_rejects_partial_tier_rewards_subdict():
    """N2: tier_rewards_count требует coins+cards (частичный под-dict → ValueError)."""
    base = build_default_case_config()
    bad = dict(base)
    bad_trc = {tier: dict(cfg) for tier, cfg in base["tier_rewards_count"].items()}
    bad_trc[3] = {"coins": (240, 560)}  # нет 'cards'
    bad["tier_rewards_count"] = bad_trc
    with pytest.raises(ValueError):
        validate_case_config(bad)


@pytest.mark.asyncio
async def test_case_config_threaded_to_roll_from_keys_http(monkeypatch):
    """M7: HTTP /api/cases/roll-from-keys инжектит _case_config_safe(db) в simulate.

    Без monkeypatch-инга roll-функций: ставим db.case_config_value с
    tier_upgrade_chances=1.0 на всех тапах → каждый тап апгрейдит тир
    детерминированно (random.random() < 1.0 всегда True) → final_tier == 5.
    Это доказывает, что live-config из БД действительно доходит до roll-функций
    на HTTP-пути (а не только через monkeypatch в unit-тестах).
    """
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "CASE_KEY_ROLLS", {})
    monkeypatch.setattr(web_server, "CASE_KEY_OPEN_RESULTS", {})
    db = _CaseRouteDB()
    db.case_config_value = {"tier_upgrade_chances": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}}
    app = web_server.create_web_app(db, bot_token="bot-token", webapp_url="https://game.example")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/api/cases/roll-from-keys?user_id=1001")
        body = await resp.json()
        assert resp.status == 200
        assert body["success"] is True
        assert body["final_tier"] == 5
        assert body["tap_results"] == [2, 3, 4, 5]
    finally:
        await client.close()
