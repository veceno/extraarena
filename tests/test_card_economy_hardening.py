import asyncio
from pathlib import Path

import pytest

from infrastructure.card_economy import (
    calculate_card_upgrade_cost,
    calculate_duplicate_particles,
    calculate_new_card_catchup,
    calculate_upgrade_coins,
    calculate_upgrade_particles,
)
from infrastructure.config import DatabaseSettings
from infrastructure.database import Database


WORKTREE_ROOT = Path(__file__).resolve().parents[1]


class _UpgradeTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        await self.conn.lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.conn.lock.release()
        return False


class _UpgradeConnection:
    def __init__(
        self,
        *,
        coins: int = 50,
        level: int = 1,
        particles: int = 5,
        simplified_levelup: bool = False,
    ):
        self.lock = asyncio.Lock()
        self.coins = coins
        self.level = level
        self.particles = particles
        self.simplified_levelup = simplified_levelup

    def transaction(self):
        return _UpgradeTransaction(self)

    async def fetchrow(self, query, *args):
        normalized = " ".join(str(query).split())
        if "FROM users" in normalized and "FOR UPDATE" in normalized:
            return {"coins": self.coins}
        if "FROM user_cards uc" in normalized and "FOR UPDATE OF uc" in normalized:
            return {
                "level": self.level,
                "particles": self.particles,
                "rarity": "common",
                "power": 100,
                "base_attack": 10,
                "base_hp": 100,
                "mana_cost": 3,
                "mechanics": {},
                "card_type": "warrior",
                "simplified_levelup": self.simplified_levelup,
            }
        if normalized.startswith("UPDATE user_cards"):
            new_level, particle_cost, _user_id, _card_id, expected_level = args
            if self.level != expected_level or self.particles < particle_cost:
                return None
            self.level = int(new_level)
            self.particles -= int(particle_cost)
            return {"level": self.level, "particles": self.particles}
        if normalized.startswith("UPDATE users"):
            coin_cost = int(args[0])
            if self.coins < coin_cost:
                return None
            self.coins -= coin_cost
            return {"coins": self.coins}
        raise AssertionError(f"Unexpected query: {normalized}")


class _UpgradePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return self

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DuplicateConnection:
    def __init__(self, *, source_level, redirect_card_id=None, rarity="epic"):
        self.source_level = source_level
        self.redirect_card_id = redirect_card_id
        self.rarity = rarity
        self.card_particles = {77: 900}
        if redirect_card_id is not None:
            self.card_particles[redirect_card_id] = 30
        self.coins = 1000

    async def fetchrow(self, query, *args):
        normalized = " ".join(str(query).split())
        if "FROM users" in normalized and "FOR UPDATE" in normalized:
            return {"coins": self.coins}
        if "uc.card_id <> $2" in normalized:
            if self.redirect_card_id is None:
                return None
            return {
                "card_id": self.redirect_card_id,
                "name": "Redirect Target",
                "simplified_levelup": False,
            }
        if "FROM user_cards uc" in normalized and "FOR UPDATE OF uc" in normalized:
            return {
                "card_id": 77,
                "level": self.source_level,
                "name": "Source Card",
                "simplified_levelup": False,
            }
        if normalized.startswith("UPDATE user_cards"):
            amount, _user_id, card_id = args
            if int(card_id) not in self.card_particles:
                return None
            self.card_particles[int(card_id)] += int(amount)
            return {"particles": self.card_particles[int(card_id)]}
        if normalized.startswith("UPDATE users"):
            amount, _user_id = args
            self.coins += int(amount)
            return {"coins": self.coins}
        raise AssertionError(f"Unexpected query: {normalized}")


class _CatchupConnection:
    def __init__(
        self,
        *,
        reference_level: int = 9,
        eligible_count: int = 9,
        simplified_levelup: bool = False,
    ):
        self.reference_level = reference_level
        self.eligible_count = eligible_count
        self.simplified_levelup = simplified_levelup
        self.inserted_particles = None
        self.events = []

    async def fetchrow(self, query, *args):
        normalized = " ".join(str(query).split())
        if "SELECT user_id FROM users" in normalized and "FOR UPDATE" in normalized:
            return {"user_id": int(args[0])}
        if "SELECT id, name, rarity, simplified_levelup FROM cards" in normalized:
            return {
                "id": int(args[0]),
                "name": "New Card",
                "rarity": "divine",
                "simplified_levelup": self.simplified_levelup,
            }
        if "PERCENTILE_DISC" in normalized:
            return {
                "eligible_count": self.eligible_count,
                "reference_level": self.reference_level,
            }
        if normalized.startswith("INSERT INTO user_cards"):
            _user_id, card_id, particles = args
            self.inserted_particles = int(particles)
            return {"card_id": int(card_id), "level": 1, "particles": int(particles)}
        raise AssertionError(f"Unexpected query: {normalized}")

    async def execute(self, query, *args):
        normalized = " ".join(str(query).split())
        if "INSERT INTO economy_events" in normalized:
            self.events.append(args)
            return "INSERT 0 1"
        raise AssertionError(f"Unexpected query: {normalized}")


def _database_with_pool(pool):
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))
    db._pool = pool

    async def no_squad_award(*_args, **_kwargs):
        return {"awarded": False}

    db.award_squad_cbrp = no_squad_award
    return db


def test_canonical_card_economy_uses_level_only_upgrade_prices():
    rarities = (
        "common",
        "rare",
        "start",
        "superrare",
        "epic",
        "legendary",
        "mythic",
        "divine",
        "limited",
    )

    assert {calculate_upgrade_particles(rarity, 1) for rarity in rarities} == {5}
    assert {calculate_upgrade_particles(rarity, 9) for rarity in rarities} == {1280}
    assert {calculate_upgrade_coins(rarity, 1) for rarity in rarities} == {50}
    assert {calculate_upgrade_coins(rarity, 9) for rarity in rarities} == {25000}
    assert sum(calculate_upgrade_particles("common", level) for level in range(1, 10)) == 2555
    assert sum(calculate_upgrade_coins("common", level) for level in range(1, 10)) == 54000
    assert calculate_card_upgrade_cost(
        "limited",
        1,
        simplified_levelup=True,
    ) == {"particles": 640, "coins": 13000}
    assert calculate_duplicate_particles("limited", 1) == 208
    assert calculate_duplicate_particles("common", 5, is_t5_common=True) == 125


@pytest.mark.parametrize(
    ("reference_level", "target_level", "particles"),
    (
        (1, 1, 0),
        (3, 1, 0),
        (4, 2, 5),
        (5, 3, 15),
        (6, 4, 35),
        (7, 5, 75),
        (8, 6, 155),
        (9, 7, 315),
        (10, 7, 315),
    ),
)
def test_new_card_catchup_uses_median_minus_two_with_level_seven_cap(
    reference_level,
    target_level,
    particles,
):
    result = calculate_new_card_catchup(reference_level)

    assert result["eligible"] is True
    assert result["target_level"] == target_level
    assert result["particles"] == particles


def test_new_card_catchup_requires_full_reference_set_and_regular_leveling():
    assert calculate_new_card_catchup(10, eligible_count=8)["particles"] == 0
    assert calculate_new_card_catchup(
        10,
        simplified_levelup=True,
    )["particles"] == 0


@pytest.mark.asyncio
async def test_first_card_acquisition_stays_level_one_and_gets_catchup_reserve():
    conn = _CatchupConnection(reference_level=9)
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))

    result = await db.grant_card_or_duplicate(
        1001,
        77,
        source="case",
        source_metadata={"final_tier": 5},
        conn=conn,
    )

    assert result["success"] is True
    assert result["is_new"] is True
    assert result["level"] == 1
    assert result["particles"] == 315
    assert result["catchup"] == {
        "particles": 315,
        "reference_level": 9,
        "target_level": 7,
        "eligible_count": 9,
        "policy_version": 1,
    }
    assert conn.inserted_particles == 315
    assert len(conn.events) == 1


@pytest.mark.asyncio
async def test_simplified_first_card_never_gets_regular_catchup_reserve():
    conn = _CatchupConnection(reference_level=10, simplified_levelup=True)
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))

    result = await db.grant_card_or_duplicate(1001, 11, source="case", conn=conn)

    assert result["success"] is True
    assert result["level"] == 1
    assert result["particles"] == 0
    assert result["catchup"]["target_level"] == 1
    assert conn.events == []


@pytest.mark.asyncio
async def test_operational_or_starter_grant_can_explicitly_disable_catchup():
    conn = _CatchupConnection(reference_level=10)
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))

    result = await db.grant_card_or_duplicate(
        1001,
        36,
        source="starter",
        apply_catchup=False,
        conn=conn,
    )

    assert result["success"] is True
    assert result["particles"] == 0
    assert result["catchup"]["target_level"] == 1
    assert conn.events == []


def test_web_client_uses_server_upgrade_cost_fields():
    legacy_source = (WORKTREE_ROOT / "webapp" / "main.js").read_text(encoding="utf-8")
    app_source = (WORKTREE_ROOT / "webapp" / "index.html").read_text(encoding="utf-8")

    assert "function calculateUpgradeParticles" not in legacy_source
    assert "function calculateUpgradeCoins" not in legacy_source
    assert "const _UP_PARTICLES" not in app_source
    assert "const _UP_COINS" not in app_source
    for source in (legacy_source, app_source):
        assert "upgrade_particles_required" in source
        assert "upgrade_coins_required" in source
        assert "Бонус освоения" in source
        assert "catchup" in source


@pytest.mark.asyncio
async def test_concurrent_upgrade_requests_cannot_double_spend():
    conn = _UpgradeConnection()
    db = _database_with_pool(_UpgradePool(conn))

    results = await asyncio.gather(
        db.upgrade_card(1001, 77),
        db.upgrade_card(1001, 77),
    )

    assert sum(result["success"] is True for result in results) == 1
    assert conn.level == 2
    assert conn.particles == 0
    assert conn.coins == 0
    assert all(value >= 0 for value in (conn.particles, conn.coins))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "particles", "coins", "simplified_levelup", "expected_level"),
    (
        (9, 1280, 25000, False, 10),
        (1, 640, 13000, True, 2),
    ),
)
async def test_upgrade_card_charges_rebalanced_transition_price(
    level,
    particles,
    coins,
    simplified_levelup,
    expected_level,
):
    conn = _UpgradeConnection(
        coins=coins,
        level=level,
        particles=particles,
        simplified_levelup=simplified_levelup,
    )
    db = _database_with_pool(_UpgradePool(conn))

    result = await db.upgrade_card(1001, 77)

    assert result["success"] is True
    assert conn.level == expected_level
    assert conn.particles == 0
    assert conn.coins == 0


@pytest.mark.asyncio
async def test_max_level_duplicate_redirects_to_non_maxed_same_rarity_card():
    conn = _DuplicateConnection(source_level=10, redirect_card_id=88)
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))

    result = await db.grant_duplicate_particles(
        1001,
        77,
        "epic",
        25,
        conn=conn,
    )

    assert result["success"] is True
    assert result["reward_type"] == "particles"
    assert result["redirected"] is True
    assert result["source_card_id"] == 77
    assert result["card_id"] == 88
    assert conn.card_particles == {77: 900, 88: 55}
    assert conn.coins == 1000


@pytest.mark.asyncio
async def test_max_level_duplicate_becomes_coins_when_rarity_is_complete():
    conn = _DuplicateConnection(source_level=10, rarity="epic")
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))

    result = await db.grant_duplicate_particles(
        1001,
        77,
        "epic",
        25,
        conn=conn,
    )

    assert result["success"] is True
    assert result["reward_type"] == "coins"
    assert result["fallback_for"] == "max_level_duplicate"
    assert result["coins_added"] == 325
    assert conn.card_particles == {77: 900}
    assert conn.coins == 1325


@pytest.mark.asyncio
async def test_non_max_duplicate_stays_on_original_card():
    conn = _DuplicateConnection(source_level=9, redirect_card_id=88)
    db = Database(DatabaseSettings(host="localhost", port=5434, user="test", password="", database="test"))

    result = await db.grant_duplicate_particles(
        1001,
        77,
        "epic",
        25,
        conn=conn,
    )

    assert result["success"] is True
    assert result["reward_type"] == "particles"
    assert result["redirected"] is False
    assert result["card_id"] == 77
    assert conn.card_particles == {77: 925, 88: 30}
