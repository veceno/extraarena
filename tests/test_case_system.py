import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure import case_system
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

    async def award_squad_cbrp(self, *args, **kwargs):
        return {"awarded": False}


class _UniLookupDB:
    async def get_uni_card(self):
        return {"id": 36, "name": "Юни", "rarity": "start"}

    async def get_cards_by_rarity(self, rarity):
        raise AssertionError("start rarity must use deterministic get_uni_card lookup")


def test_simulate_case_tap_results_uses_server_rolls(monkeypatch):
    rolls = iter([1, 2, 2, 3])

    def fake_roll(current_tier, tap_number, extra_pass="inactive"):
        assert extra_pass == "ultra"
        return next(rolls)

    monkeypatch.setattr(case_system, "roll_tier_upgrade", fake_roll)

    assert case_system.simulate_case_tap_results(1, "ultra") == [1, 2, 2, 3]


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

    result = await case_system.process_case_opening(db, 1001, 77, [3, 3, 3, 3])

    assert result["success"] is True
    assert db.coins == 50
    assert db.removed == [(77, 1001)]
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
