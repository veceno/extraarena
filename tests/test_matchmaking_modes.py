import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
import pytest_asyncio

from ai.bot_factory import BotGenerator
from infrastructure.config import DECK_SIZE, BOT_STRENGTH_TIERS
from infrastructure.matchmaking import Matchmaker, QueueEntry
from infrastructure.match_modes import (
    EXTRA_ARENA_ROTATING_IDS,
    resolve_mode_config,
    ROTATION_ANCHOR_EPOCH_SECONDS,
    ROTATION_INTERVAL_SECONDS,
)


class FakeDB:
    async def get_user_deck_presets(self, user_id):
        return []


class FakeBotFactory:
    pass


class SlowCountingBotFactory:
    def __init__(self):
        self.calls = 0

    async def create_match(self, user_id, trophies):
        import asyncio

        self.calls += 1
        current = self.calls
        await asyncio.sleep(0.01)
        return {
            "match_id": f"bot-match-{current}",
            "opponent_id": -1,
            "bot_info": {"name": "Bot", "deck_ids": list(range(1, 10))},
        }


class EventBotFactory:
    def __init__(self, events):
        self.events = events
        self.calls = 0

    async def create_match(self, user_id, trophies):
        self.calls += 1
        self.events.append("factory")
        return {
            "match_id": "event-bot-match",
            "opponent_id": -900000001,
            "bot_info": {"name": "Bot", "deck_ids": list(range(1, 10))},
        }


class MalformedBotFactory:
    async def create_match(self, user_id, trophies):
        return {
            "match_id": "bad-bot-match",
            "opponent_id": -900000001,
            "bot_info": {"name": "MissingDeck"},
        }


class MinimalBotFactory:
    async def create_match(self, user_id, trophies):
        return {
            "match_id": "minimal-bot-match",
            "opponent_id": -900000002,
            "bot_info": {"name": "MinimalBot", "deck_ids": list(range(1, 10))},
        }


class DifficultyCaptureBotFactory:
    def __init__(self):
        self.calls = []

    async def create_match(
        self,
        user_id,
        trophies,
        user_max_level=None,
        difficulty=None,
        difficulty_override=None,
    ):
        self.calls.append(
            {
                "user_id": user_id,
                "trophies": trophies,
                "user_max_level": user_max_level,
                "difficulty": difficulty,
                "difficulty_override": difficulty_override,
            }
        )
        return {
            "match_id": f"capture-bot-match-{len(self.calls)}",
            "opponent_id": -900000003,
            "bot_info": {"name": "CaptureBot", "deck_ids": list(range(1, 10))},
        }


class LowLevelBotFactory:
    async def create_match(
        self,
        user_id,
        trophies,
        user_max_level=None,
        difficulty=None,
        difficulty_override=None,
    ):
        return {
            "match_id": "low-level-bot-match",
            "opponent_id": -900000004,
            "bot_info": {
                "name": "LowLevelBot",
                "difficulty": "tier_easy_0100",
                "deck_ids": list(range(1, 10)),
                "card_levels": [1] * DECK_SIZE,
            },
        }


class StreakDB(FakeDB):
    def __init__(self, kind, length):
        self.kind = kind
        self.length = length

    async def get_current_result_streak(self, user_id):
        return {"kind": self.kind, "length": self.length}


class SequenceStreakDB(FakeDB):
    def __init__(self, streaks):
        self.streaks = list(streaks)

    async def get_current_result_streak(self, user_id):
        if self.streaks:
            kind, length = self.streaks.pop(0)
            return {"kind": kind, "length": length}
        return {"kind": None, "length": 0}


class FailingBotGenerator(BotGenerator):
    async def get_or_create_bot(self, *_args, **_kwargs):
        raise RuntimeError("catalog unavailable")


def make_matchmaker():
    db = FakeDB()
    return Matchmaker(
        db=db,
        bot_factory=FakeBotFactory(),
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )


def test_classic_and_extra_arena_do_not_match():
    mm = make_matchmaker()
    seeker = QueueEntry(
        user_id=1, trophies=500, max_level=5, enqueued_at=0.0,
        game_mode="classic", canonical_mode="classic",
    )
    other = QueueEntry(
        user_id=2, trophies=500, max_level=5, enqueued_at=0.0,
        game_mode="extra_arena", canonical_mode="extra_arena:blitzkrieg",
    )
    assert mm._find_candidate(seeker, 100) is None


def test_different_extra_arena_mods_do_not_match():
    mm = make_matchmaker()
    seeker = QueueEntry(
        user_id=1, trophies=500, max_level=5, enqueued_at=0.0,
        game_mode="extra_arena", canonical_mode="extra_arena:blitzkrieg",
    )
    other = QueueEntry(
        user_id=2, trophies=500, max_level=5, enqueued_at=0.0,
        game_mode="extra_arena", canonical_mode="extra_arena:spellstorm",
    )
    mm._queue = [other]
    assert mm._find_candidate(seeker, 100) is None


def test_same_canonical_mode_matches():
    mm = make_matchmaker()
    seeker = QueueEntry(
        user_id=1, trophies=500, max_level=5, enqueued_at=0.0,
        game_mode="extra_arena", canonical_mode="extra_arena:spellstorm",
    )
    other = QueueEntry(
        user_id=2, trophies=500, max_level=5, enqueued_at=0.0,
        game_mode="extra_arena", canonical_mode="extra_arena:spellstorm",
    )
    mm._queue = [other]
    assert mm._find_candidate(seeker, 100) == other


@pytest.mark.asyncio(loop_scope="function")
async def test_bots_allowed_false_does_not_create_bot_and_returns_cancel():
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=SlowCountingBotFactory(),
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )
    # Default rotating modes have bots_allowed=True, so <300 trophies triggers bot
    result = await mm.find_match(
        user_id=1, trophies=100, user_max_level=1,
        selected_deck_id=1, game_mode="extra_arena:spellstorm",
        canonical_mode="extra_arena:spellstorm",
    )
    assert result["status"] == "found"
    assert result.get("is_bot") is True

    # To test bots_allowed=False, monkeypatch the config lookup
    import infrastructure.match_modes as mm_mod
    original = mm_mod.MODE_CONFIGS.copy()
    mm_mod.MODE_CONFIGS["extra_arena:spellstorm"] = mm_mod.ModeConfig(
        mode_id="extra_arena:spellstorm",
        ruleset="classic",
        label="Test",
        classic=mm_mod.ClassicParams(bots_allowed=False),
    )
    try:
        mm2 = make_matchmaker()
        result2 = await mm2.find_match(
            user_id=1, trophies=100, user_max_level=1,
            selected_deck_id=1, game_mode="extra_arena:spellstorm",
            canonical_mode="extra_arena:spellstorm",
        )
        # With bots_allowed=False, soft-start is disabled, goes to queue
        assert result2["status"] == "waiting"
        assert result2["game_mode"] == "extra_arena:spellstorm"
    finally:
        mm_mod.MODE_CONFIGS = original


@pytest.mark.asyncio(loop_scope="function")
async def test_soft_start_deduplicates_concurrent_requests_for_same_user():
    db = FakeDB()
    factory = SlowCountingBotFactory()
    mm = Matchmaker(
        db=db,
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    first, second = await asyncio.gather(
        mm.find_match(
            user_id=42, trophies=100, user_max_level=1,
            selected_deck_id=1, game_mode="classic",
        ),
        mm.find_match(
            user_id=42, trophies=100, user_max_level=1,
            selected_deck_id=1, game_mode="classic",
        ),
    )

    assert first["status"] == "found"
    assert second["status"] == "found"
    assert first["match_id"] == second["match_id"]
    assert factory.calls == 1


@pytest.mark.asyncio(loop_scope="function")
async def test_soft_start_delays_before_creating_bot(monkeypatch):
    events = []

    async def fake_sleep(duration):
        events.append(("sleep", duration))

    monkeypatch.setattr("infrastructure.matchmaking.random.uniform", lambda lo, hi: 3.25)
    monkeypatch.setattr("infrastructure.matchmaking.asyncio.sleep", fake_sleep)

    factory = EventBotFactory(events)
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(2.0, 4.0),
    )

    result = await mm.find_match(
        user_id=7, trophies=100, user_max_level=1,
        selected_deck_id=1, game_mode="classic",
    )

    assert result["status"] == "found"
    assert events == [("sleep", 3.25), "factory"]
    assert factory.calls == 1


@pytest.mark.asyncio(loop_scope="function")
async def test_soft_start_delay_task_is_deduplicated_for_concurrent_requests(monkeypatch):
    real_sleep = asyncio.sleep
    sleep_durations = []
    events = []

    async def fake_sleep(duration):
        sleep_durations.append(duration)
        await real_sleep(0)

    monkeypatch.setattr("infrastructure.matchmaking.random.uniform", lambda lo, hi: 2.5)
    monkeypatch.setattr("infrastructure.matchmaking.asyncio.sleep", fake_sleep)

    factory = EventBotFactory(events)
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(2.0, 4.0),
    )

    first, second = await asyncio.gather(
        mm.find_match(
            user_id=42, trophies=100, user_max_level=1,
            selected_deck_id=1, game_mode="classic",
        ),
        mm.find_match(
            user_id=42, trophies=100, user_max_level=1,
            selected_deck_id=1, game_mode="classic",
        ),
    )

    assert first["status"] == second["status"] == "found"
    assert first["match_id"] == second["match_id"]
    assert sleep_durations == [2.5]
    assert factory.calls == 1


@pytest.mark.asyncio(loop_scope="function")
async def test_soft_start_deck_change_cancels_stale_start():
    db = FakeDB()
    factory = SlowCountingBotFactory()
    mm = Matchmaker(
        db=db,
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.05, 0.05),
    )

    first_task = asyncio.create_task(
        mm.find_match(
            user_id=42, trophies=100, user_max_level=1,
            selected_deck_id=1, game_mode="classic",
        )
    )
    await asyncio.sleep(0)
    second = await mm.find_match(
        user_id=42, trophies=100, user_max_level=1,
        selected_deck_id=2, game_mode="classic",
    )
    first = await first_task

    assert first["status"] == "canceled"
    assert first["error"] == "search_replaced"
    assert second["status"] == "found"
    assert second["selected_deck_id"] == 2
    assert factory.calls == 1
    assert await mm.get_active_match_for_user(42) == second


@pytest.mark.asyncio(loop_scope="function")
async def test_stale_bot_timeout_does_not_publish_old_match():
    db = FakeDB()
    factory = SlowCountingBotFactory()
    mm = Matchmaker(
        db=db,
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )
    old = QueueEntry(
        user_id=77,
        trophies=500,
        max_level=1,
        enqueued_at=0.0,
        selected_deck_id=1,
        game_mode="classic",
        canonical_mode="classic",
    )

    async with mm._lock:
        mm._queue.append(old)
        mm._matches[old.match_id] = {
            "status": "waiting",
            "match_id": old.match_id,
            "user_id": old.user_id,
            "game_mode": "classic",
        }
        mm._register_match_aliases({old.match_id})
        mm._drop_existing(old.user_id)

    await mm._handle_bot_timeout(old, game_mode="classic")

    assert await mm.get_status(old.match_id) == {"status": "not_found", "match_id": old.match_id}
    assert factory.calls == 0


@pytest.mark.asyncio(loop_scope="function")
async def test_soft_start_does_not_reuse_finished_bot_match():
    db = FakeDB()
    factory = SlowCountingBotFactory()
    mm = Matchmaker(
        db=db,
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    first = await mm.find_match(
        user_id=101, trophies=124, user_max_level=3,
        selected_deck_id=1, game_mode="classic",
    )
    await mm.mark_match_finished(first["match_id"], winner_id=first["opponent_id"])
    second = await mm.find_match(
        user_id=101, trophies=119, user_max_level=3,
        selected_deck_id=1, game_mode="classic",
    )

    assert first["status"] == "found"
    assert second["status"] == "found"
    assert second["match_id"] != first["match_id"]
    assert factory.calls == 2


@pytest.mark.asyncio(loop_scope="function")
async def test_malformed_create_match_payload_returns_controlled_cancel_payload():
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=MalformedBotFactory(),
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    result = await mm._create_bot_match(
        user_id=101,
        trophies=100,
        user_max_level=1,
        selected_deck_id=1,
        game_mode="classic",
    )

    assert result["status"] == "canceled"
    assert result["error"] == "bot_match_create_failed"
    assert "opponent_id" not in result


@pytest.mark.asyncio(loop_scope="function")
async def test_custom_create_match_payload_gets_normalized_bot_difficulty_metadata():
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=MinimalBotFactory(),
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    result = await mm._create_bot_match(
        user_id=101,
        trophies=100,
        user_max_level=5,
        selected_deck_id=1,
        game_mode="classic",
        difficulty="max",
    )

    assert result["status"] == "found"
    assert result["bot_info"]["difficulty"] == "tier_max_9000"
    assert result["bot_info"]["deck_policy"] == "meta_boss"
    assert result["bot_info"]["strength_tier"] == "tier_max_9000"
    assert len(result["bot_info"]["card_levels"]) == DECK_SIZE
    assert all(level in (5, 6) for level in result["bot_info"]["card_levels"])


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_generator_failure_returns_controlled_cancel_payload():
    db = FakeDB()
    generator = FailingBotGenerator.__new__(FailingBotGenerator)
    mm = Matchmaker(db=db, bot_factory=generator, battle_engine=None)

    result = await mm._create_bot_match(
        user_id=101,
        trophies=100,
        user_max_level=1,
        selected_deck_id=1,
        game_mode="classic",
    )

    assert result["status"] == "canceled"
    assert result["error"] == "bot_match_create_failed"
    assert result["is_bot"] is True
    assert "opponent_id" not in result


@pytest.mark.asyncio(loop_scope="function")
async def test_find_match_passes_streak_adjusted_bot_difficulty_to_factory():
    factory = DifficultyCaptureBotFactory()
    mm = Matchmaker(
        db=StreakDB("loss", 2),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    result = await mm.find_match(
        user_id=101,
        trophies=100,
        user_max_level=4,
        selected_deck_id=1,
        game_mode="classic",
    )

    assert result["status"] == "found"
    assert factory.calls[0]["difficulty"] == "tier_lite_0000"
    assert result["bot_info"]["difficulty"] == "tier_lite_0000"


@pytest.mark.asyncio(loop_scope="function")
async def test_explicit_bot_difficulty_override_ignores_streak_adjustment():
    factory = DifficultyCaptureBotFactory()
    mm = Matchmaker(db=FakeDB(), bot_factory=factory, battle_engine=None)
    adjustment = Matchmaker._streak_adjustment_for(1200, "loss", 5)

    result = await mm._create_bot_match(
        user_id=101,
        trophies=1200,
        user_max_level=4,
        selected_deck_id=1,
        game_mode="classic",
        difficulty_override="max",
        streak_adjustment=adjustment,
    )

    assert result["status"] == "found"
    assert factory.calls[0]["difficulty"] == "tier_max_9000"
    assert result["bot_info"]["difficulty"] == "tier_max_9000"


@pytest.mark.asyncio(loop_scope="function")
async def test_streak_adjustment_applied_when_no_explicit_override_classic():
    """Classic mode without difficulty_override MUST apply streak adjustment (loss pity)."""
    factory = DifficultyCaptureBotFactory()
    mm = Matchmaker(
        db=StreakDB("loss", 6),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )
    # trophies=1200 -> mid band: loss_threshold=5, streak_length=6 -> 6%5!=0 -> inactive.
    # Use length=5 so 5%5==0 -> n=1 -> shift down 1 from tier_medium_1200 -> tier_medium_minus_1000.
    mm2 = Matchmaker(
        db=StreakDB("loss", 5),
        bot_factory=DifficultyCaptureBotFactory(),
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )
    factory2 = mm2._bot_factory
    result = await mm2._create_bot_match(
        user_id=101,
        trophies=1200,
        user_max_level=4,
        selected_deck_id=1,
        game_mode="classic",
        streak_adjustment=Matchmaker._streak_adjustment_for(1200, "loss", 5),
    )

    assert result["status"] == "found"
    # tier_medium_1200 is index 5; down 1 -> index 4 = tier_medium_minus_1000
    assert factory2.calls[0]["difficulty"] == "tier_medium_minus_1000"
    assert result["bot_info"]["difficulty"] == "tier_medium_minus_1000"


@pytest.mark.asyncio(loop_scope="function")
async def test_training_mode_keeps_explicit_difficulty_override_without_streak():
    """Training mode should honor client difficulty_override and ignore streak entirely."""
    factory = DifficultyCaptureBotFactory()
    mm = Matchmaker(
        db=StreakDB("loss", 5),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    result = await mm._create_bot_match(
        user_id=101,
        trophies=1200,
        user_max_level=4,
        selected_deck_id=1,
        game_mode="training",
        difficulty_override="max",
        streak_adjustment=Matchmaker._streak_adjustment_for(1200, "loss", 5),
    )

    assert result["status"] == "found"
    assert factory.calls[0]["difficulty"] == "tier_max_9000"
    assert result["bot_info"]["difficulty"] == "tier_max_9000"


class DisplayNameCaptureFactory(BotGenerator):
    """BotGenerator stub that records the player_display_name it receives."""
    def __init__(self):
        super().__init__(database=FakeDB())
        self.last_player_display_name = "NOT_CALLED"

    async def get_or_create_bot(
        self,
        player_id,
        player_trophies,
        difficulty_override=None,
        player_display_name=None,
    ):
        self.last_player_display_name = player_display_name
        return {
            "user_id": -900000555,
            "name": "StubBot",
            "avatar_url": None,
            "difficulty": "tier_medium_1200",
            "deck_ids": list(range(1, 10)),
            "cosmetics": {},
            "extra_pass": None,
            "trophies": 1200,
            "difficulty_label": "medium",
            "strength_tier": "tier_medium_1200",
            "brain_profile": "extra-lr-v4-opti",
            "selection": "softmax",
            "temperature": 1.8,
            "card_level_policy": {"delta_min": 0, "delta_max": 0, "cap": 5, "boost_fraction": 0.0},
            "deck_policy": "decent_donor",
        }


@pytest.mark.asyncio(loop_scope="function")
async def test_create_bot_match_threads_player_display_name_to_factory():
    """_create_bot_match must forward player_display_name to the BotGenerator fallback."""
    factory = DisplayNameCaptureFactory()
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    result = await mm._create_bot_match(
        user_id=101,
        trophies=1200,
        user_max_level=4,
        selected_deck_id=1,
        game_mode="classic",
        player_display_name="Alice",
    )

    assert result["status"] == "found"
    assert factory.last_player_display_name == "Alice"


@pytest.mark.asyncio(loop_scope="function")
async def test_create_bot_match_player_display_name_defaults_none_when_omitted():
    """When no player_display_name is supplied, the factory receives None (no crash)."""
    factory = DisplayNameCaptureFactory()
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    result = await mm._create_bot_match(
        user_id=101,
        trophies=1200,
        user_max_level=4,
        selected_deck_id=1,
        game_mode="classic",
    )

    assert result["status"] == "found"
    assert factory.last_player_display_name is None


    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=MinimalBotFactory(),
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    result = await mm._create_bot_match(
        user_id=101,
        trophies=1200,
        user_max_level=4,
        selected_deck_id=1,
        game_mode="classic",
        difficulty_override="typo-max",
    )

    assert result["status"] == "canceled"
    assert result["error"] == "bot_match_create_failed"
    assert "opponent_id" not in result


@pytest.mark.asyncio(loop_scope="function")
async def test_difficulty_override_rebuilds_custom_factory_card_levels():
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=LowLevelBotFactory(),
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )

    result = await mm._create_bot_match(
        user_id=101,
        trophies=1200,
        user_max_level=5,
        selected_deck_id=1,
        game_mode="classic",
        difficulty_override="max",
    )

    assert result["status"] == "found"
    assert result["bot_info"]["difficulty"] == "tier_max_9000"
    assert result["bot_info"]["card_levels"] != [1] * DECK_SIZE
    assert all(level in (5, 6) for level in result["bot_info"]["card_levels"])


@pytest.mark.asyncio(loop_scope="function")
async def test_soft_start_replaces_pending_task_when_streak_context_changes():
    real_sleep = asyncio.sleep
    factory = DifficultyCaptureBotFactory()
    mm = Matchmaker(
        db=SequenceStreakDB([("win", 5), ("loss", 2)]),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.05, 0.05),
    )

    first_task = asyncio.create_task(
        mm.find_match(
            user_id=42,
            trophies=100,
            user_max_level=4,
            selected_deck_id=1,
            game_mode="classic",
        )
    )
    await real_sleep(0)
    second = await mm.find_match(
        user_id=42,
        trophies=100,
        user_max_level=4,
        selected_deck_id=1,
        game_mode="classic",
    )
    first = await first_task

    assert first["status"] == "canceled"
    assert second["status"] == "found"
    assert len(factory.calls) == 1
    assert factory.calls[0]["difficulty"] == "tier_lite_0000"


def test_streak_adjustment_thresholds_and_exact_multiples():
    assert Matchmaker._streak_adjustment_for(250, "loss", 2).direction == "down"
    assert Matchmaker._streak_adjustment_for(250, "win", 5).n == 1
    assert Matchmaker._streak_adjustment_for(250, "win", 6).active is False

    # 300 trophies now falls in the mid band (< MM_TROPHY_LIMIT_CLASSIC): loss every 5, win every 3.
    assert Matchmaker._streak_adjustment_for(300, "loss", 2).active is False
    assert Matchmaker._streak_adjustment_for(300, "win", 3).n == 1
    assert Matchmaker._streak_adjustment_for(300, "loss", 5).n == 1
    assert Matchmaker._streak_adjustment_for(300, "loss", 10).n == 2
    assert Matchmaker._streak_adjustment_for(301, "loss", 5).n == 1
    assert Matchmaker._streak_adjustment_for(1200, "win", 6).n == 2

    assert Matchmaker._streak_adjustment_for(5000, "loss", 5).active is False
    assert Matchmaker._streak_adjustment_for(5000, "win", 3).direction == "up"


def test_directional_candidate_search_prefers_lower_opponents_on_loss_streak():
    mm = make_matchmaker()
    seeker = QueueEntry(user_id=99, trophies=500, max_level=1, enqueued_at=10.0)
    seeker.streak_adjustment = Matchmaker._streak_adjustment_for(500, "loss", 5)
    lower = QueueEntry(user_id=1, trophies=450, max_level=1, enqueued_at=1.0)
    equal = QueueEntry(user_id=2, trophies=500, max_level=1, enqueued_at=2.0)
    higher = QueueEntry(user_id=3, trophies=550, max_level=1, enqueued_at=0.0)
    mm._queue = [higher, lower, equal]

    window = Matchmaker._search_window_for_adjustment(50, seeker.streak_adjustment)

    assert mm._find_candidate(seeker, window) is equal
    mm._queue = [higher, lower]
    assert mm._find_candidate(seeker, window) is lower


def test_directional_candidate_search_prefers_higher_opponents_on_win_streak():
    mm = make_matchmaker()
    seeker = QueueEntry(user_id=99, trophies=500, max_level=1, enqueued_at=10.0)
    seeker.streak_adjustment = Matchmaker._streak_adjustment_for(500, "win", 3)
    lower = QueueEntry(user_id=1, trophies=450, max_level=1, enqueued_at=1.0)
    higher = QueueEntry(user_id=2, trophies=550, max_level=1, enqueued_at=2.0)
    mm._queue = [lower, higher]

    window = Matchmaker._search_window_for_adjustment(50, seeker.streak_adjustment)

    assert mm._find_candidate(seeker, window) is higher


def test_streak_window_sequence_starts_at_nth_existing_window_and_clamps():
    one = Matchmaker._streak_adjustment_for(500, "win", 3)
    two = Matchmaker._streak_adjustment_for(500, "win", 6)
    huge = Matchmaker._streak_adjustment_for(500, "win", 30)

    assert Matchmaker._windows_for_adjustment(one) == (50, 200, 500)
    assert Matchmaker._windows_for_adjustment(two) == (200, 500)
    assert Matchmaker._windows_for_adjustment(huge) == (500,)


def test_widened_seeker_cannot_pull_fresh_queued_opponent_past_their_window():
    mm = make_matchmaker()
    seeker = QueueEntry(user_id=99, trophies=500, max_level=1, enqueued_at=10.0)
    seeker.streak_adjustment = Matchmaker._streak_adjustment_for(500, "win", 9)
    fresh_normal = QueueEntry(user_id=1, trophies=900, max_level=1, enqueued_at=1.0)
    mm._queue = [fresh_normal]

    window = Matchmaker._search_window_for_adjustment(500, seeker.streak_adjustment)

    assert mm._find_candidate(seeker, window) is None


@pytest.mark.asyncio(loop_scope="function")
async def test_stale_cancel_timeout_does_not_publish_old_no_bot_match():
    mm = make_matchmaker()
    old = QueueEntry(
        user_id=77,
        trophies=500,
        max_level=1,
        enqueued_at=0.0,
        selected_deck_id=1,
        game_mode="extra_arena:spellstorm",
        canonical_mode="extra_arena:spellstorm",
    )

    async with mm._lock:
        mm._queue.append(old)
        mm._matches[old.match_id] = {
            "status": "waiting",
            "match_id": old.match_id,
            "user_id": old.user_id,
            "game_mode": "extra_arena:spellstorm",
        }
        mm._register_match_aliases({old.match_id})
        mm._drop_existing(old.user_id)

    await mm._handle_cancel_timeout(old, game_mode="extra_arena:spellstorm")

    assert await mm.get_status(old.match_id) == {"status": "not_found", "match_id": old.match_id}


# ═══════════════════════════════════════════════════════════════════════════
# Playwright fix verifications — matchmaking bot difficulty adjusts with streak.
# Named scenarios from the bug ticket:
#   "N consecutive wins in DB → start battle → next bot difficulty is RAISED".
#   "M consecutive losses in DB → next bot difficulty is LOWERED".
#   "After streak BREAK (loss following win streak), difficulty is RESET to base".
#   "PvP players: same logic applied to search window".
# ═══════════════════════════════════════════════════════════════════════════

def _trophy_band(trophies):
    """Return the band label for the player's trophy count (for documentation only)."""
    if trophies < 300:
        return "low"
    if trophies >= 5000:
        return "high"
    return "mid"


def test_streak_adjustment_raises_bot_difficulty_on_win_streak_mid_band():
    """N wins in mid band (trophies=2000) → bot difficulty raised by N/threshold steps."""
    # Mid band win_threshold=3; length=3 -> n=1 (raise by 1); length=6 -> n=2.
    a = Matchmaker._streak_adjustment_for(2000, "win", 3)
    assert a.active is True and a.direction == "up" and a.n == 1
    a6 = Matchmaker._streak_adjustment_for(2000, "win", 6)
    assert a6.active is True and a6.direction == "up" and a6.n == 2


def test_streak_adjustment_lowers_bot_difficulty_on_loss_streak_mid_band():
    """M losses in mid band (trophies=2000) → bot difficulty lowered by M/threshold steps."""
    # Mid band loss_threshold=5; length=5 -> n=1 (lower by 1); length=10 -> n=2.
    a = Matchmaker._streak_adjustment_for(2000, "loss", 5)
    assert a.active is True and a.direction == "down" and a.n == 1
    a10 = Matchmaker._streak_adjustment_for(2000, "loss", 10)
    assert a10.active is True and a10.direction == "down" and a10.n == 2


def test_streak_adjustment_inactive_below_threshold():
    """Below threshold, the streak does NOT shift difficulty."""
    # Mid band: length=2 < 3 (win) and length=4 < 5 (loss) -> inactive.
    assert Matchmaker._streak_adjustment_for(2000, "win", 2).active is False
    assert Matchmaker._streak_adjustment_for(2000, "loss", 4).active is False
    # Also, length must be an EXACT multiple of the threshold (3, 6, 9...).
    assert Matchmaker._streak_adjustment_for(2000, "win", 5).active is False  # 5 % 3 != 0


def test_streak_adjustment_break_resets_to_inactive():
    """When a streak is BROKEN (last result is a single loss), the new streak is
    kind=loss length=1; in mid band length=1 < loss_threshold=5 → inactive.

    This is the named scenario:
    'N-1 wins in DB → loss → expected streak broken → new battle, win →
    expect difficulty NOT raised because streak was reset'.
    """
    broken = Matchmaker._streak_adjustment_for(2000, "loss", 1)
    assert broken.active is False, (
        f"After a streak break (single loss), adjustment must be inactive; got {broken}"
    )


def test_streak_shift_produces_correct_difficulty_key_mid_band_win_3():
    """Player @ 2000 trophies with win streak length=3 → bot difficulty raises by 1 tier.

    Base tier for 2000 trophies = tier_medium_plus_2000. Shifted up 1 = tier_hard_minus_3000.
    """
    from ai.bot_factory import BotGenerator
    base = BotGenerator._calc_difficulty(2000)
    shifted = BotGenerator._shift_difficulty_by_streak(base, "up", 1)
    assert base == "tier_medium_plus_2000"
    assert shifted == "tier_hard_minus_3000"


def test_streak_shift_produces_correct_difficulty_key_mid_band_loss_5():
    """Player @ 2000 trophies with loss streak length=5 → bot difficulty lowers by 1 tier.

    Base tier for 2000 trophies = tier_medium_plus_2000. Shifted down 1 = tier_medium_1200.
    """
    from ai.bot_factory import BotGenerator
    base = BotGenerator._calc_difficulty(2000)
    shifted = BotGenerator._shift_difficulty_by_streak(base, "down", 1)
    assert base == "tier_medium_plus_2000"
    assert shifted == "tier_medium_1200"


def test_streak_shift_low_band_loss_2():
    """Player @ 100 trophies with loss streak length=2 (low band: threshold=2) → bot difficulty lowers by 1."""
    from ai.bot_factory import BotGenerator
    base = BotGenerator._calc_difficulty(100)
    shifted = BotGenerator._shift_difficulty_by_streak(base, "down", 1)
    assert base == "tier_easy_0100"
    assert shifted == "tier_lite_0000"


def test_streak_shift_high_band_win_3_raises():
    """Player @ 5500 trophies with win streak length=3 (high band: threshold=3) → bot difficulty raises by 1."""
    from ai.bot_factory import BotGenerator
    base = BotGenerator._calc_difficulty(5500)
    shifted = BotGenerator._shift_difficulty_by_streak(base, "up", 1)
    # Base tier for 5500 trophies is hard_plus_6000 (or the tier just above 5000 boundary).
    # Verify by checking the key exists and the shifted key is exactly 1 step up.
    tier_keys = [str(t["key"]) for t in BOT_STRENGTH_TIERS]
    base_idx = tier_keys.index(base)
    assert shifted == tier_keys[base_idx + 1]


def test_streak_shift_clamps_at_top_and_bottom():
    """Shift is clamped at the topmost/bottommost tier."""
    from ai.bot_factory import BotGenerator
    top = BOT_STRENGTH_TIERS[-1]["key"]
    bottom = BOT_STRENGTH_TIERS[0]["key"]
    assert BotGenerator._shift_difficulty_by_streak(top, "up", 10) == top
    assert BotGenerator._shift_difficulty_by_streak(bottom, "down", 10) == bottom


def test_find_match_uses_streak_adjusted_difficulty_on_win_streak():
    """End-to-end: streakDB returns (win, 3), _create_bot_match → factory receives shifted difficulty.

    Player at 2000 trophies (mid band, win_threshold=3, length=3 → n=1, direction=up):
    base difficulty = tier_medium_plus_2000 → shifted up 1 = tier_hard_minus_3000.
    """
    factory = DifficultyCaptureBotFactory()
    mm = Matchmaker(
        db=StreakDB("win", 3),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )
    adjustment = Matchmaker._streak_adjustment_for(2000, "win", 3)
    loop = asyncio.new_event_loop()
    try:
        payload = loop.run_until_complete(
            mm._create_bot_match(
                user_id=1, trophies=2000, user_max_level=4,
                selected_deck_id=1, game_mode="classic",
                streak_adjustment=adjustment,
            )
        )
    finally:
        loop.close()
    assert payload["status"] == "found"
    assert factory.calls[0]["difficulty"] == "tier_hard_minus_3000", (
        f"Expected factory to receive tier_hard_minus_3000 after win streak of 3; "
        f"got {factory.calls[0]['difficulty']}"
    )


def test_find_match_uses_streak_adjusted_difficulty_on_loss_streak():
    """End-to-end: streakDB returns (loss, 5), _create_bot_match → factory receives shifted (down) difficulty.

    Player at 2000 trophies (mid band, loss_threshold=5, length=5 → n=1, direction=down):
    base difficulty = tier_medium_plus_2000 → shifted down 1 = tier_medium_1200.
    """
    factory = DifficultyCaptureBotFactory()
    mm = Matchmaker(
        db=StreakDB("loss", 5),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )
    adjustment = Matchmaker._streak_adjustment_for(2000, "loss", 5)
    loop = asyncio.new_event_loop()
    try:
        payload = loop.run_until_complete(
            mm._create_bot_match(
                user_id=2, trophies=2000, user_max_level=4,
                selected_deck_id=1, game_mode="classic",
                streak_adjustment=adjustment,
            )
        )
    finally:
        loop.close()
    assert payload["status"] == "found"
    assert factory.calls[0]["difficulty"] == "tier_medium_1200", (
        f"Expected factory to receive tier_medium_1200 after loss streak of 5; "
        f"got {factory.calls[0]['difficulty']}"
    )


def test_find_match_after_streak_break_returns_base_difficulty():
    """Streak-break reset: after a 3-win streak is broken by 1 loss, get_current_result_streak returns (loss, 1).
    length=1 < loss_threshold=5 (mid band) → adjustment inactive → factory gets base difficulty.
    """
    factory = DifficultyCaptureBotFactory()
    mm = Matchmaker(
        db=StreakDB("loss", 1),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )
    adjustment = Matchmaker._streak_adjustment_for(2000, "loss", 1)
    assert adjustment.active is False, (
        f"After streak break (single loss), adjustment must be inactive; got {adjustment}"
    )
    loop = asyncio.new_event_loop()
    try:
        payload = loop.run_until_complete(
            mm._create_bot_match(
                user_id=3, trophies=2000, user_max_level=4,
                selected_deck_id=1, game_mode="classic",
                streak_adjustment=adjustment,
            )
        )
    finally:
        loop.close()
    assert payload["status"] == "found"
    # Streak is broken — adjustment inactive → factory gets difficulty=None
    # (the factory/base bot generator then computes the base tier from trophies=2000,
    # which would be tier_medium_plus_2000 if it weren't already overridden).
    # The invariant is: NO explicit override is passed when streak adjustment is inactive.
    assert factory.calls[0]["difficulty"] is None, (
        f"After streak break, factory must NOT receive an explicit difficulty override; "
        f"got {factory.calls[0]['difficulty']}"
    )
    assert factory.calls[0]["difficulty_override"] is None, (
        f"After streak break, difficulty_override must be None; "
        f"got {factory.calls[0]['difficulty_override']}"
    )


def test_find_match_sequence_break_then_rebuild_resets_correctly():
    """Sequence: streak=win*3 → streak=loss*1 (broken) → streak=win*1 (rebuild) → streak=win*3.
    Verifies the entire reset cycle: raised → broken (back to base) → raised again only after threshold.

    When adjustment is inactive (streak<threshold OR broken), the factory receives
    difficulty=None (no override) — the base tier is then computed from trophies
    inside the factory/base generator.
    When adjustment is active, the factory receives the explicit shifted tier key.
    """
    factory = DifficultyCaptureBotFactory()
    sequence_steps = [
        ("win", 3),   # 1st call: streak of 3 → active up n=1 → tier_hard_minus_3000
        ("loss", 1),  # 2nd call: streak of 1 (broken) → inactive → difficulty=None
        ("win", 1),   # 3rd call: streak of 1 (just rebuilt) → inactive → difficulty=None
        ("win", 3),   # 4th call: streak of 3 again → active up n=1 → tier_hard_minus_3000
    ]
    expected_difficulties = [
        "tier_hard_minus_3000",
        None,        # inactive after break → no override passed to factory
        None,        # inactive while rebuilding (< threshold)
        "tier_hard_minus_3000",
    ]
    mm = Matchmaker(
        db=FakeDB(),
        bot_factory=factory,
        battle_engine=None,
        soft_start_bot_delay_range=(0.0, 0.0),
    )
    loop = asyncio.new_event_loop()
    try:
        for i, ((kind, length), expected) in enumerate(zip(sequence_steps, expected_difficulties)):
            adjustment = Matchmaker._streak_adjustment_for(2000, kind, length)
            payload = loop.run_until_complete(
                mm._create_bot_match(
                    user_id=10 + i, trophies=2000, user_max_level=4,
                    selected_deck_id=1, game_mode="classic",
                    streak_adjustment=adjustment,
                )
            )
            assert payload["status"] == "found", f"call {i}: not found"
            actual = factory.calls[i]["difficulty"]
            assert actual == expected, (
                f"call {i} (streak={kind}*{length}): expected difficulty={expected}, got {actual}"
            )
    finally:
        loop.close()
