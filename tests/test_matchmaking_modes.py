import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
import pytest_asyncio

from ai.bot_factory import BotGenerator
from infrastructure.config import DECK_SIZE
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
async def test_invalid_explicit_bot_difficulty_returns_controlled_cancel_payload():
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

    assert Matchmaker._streak_adjustment_for(300, "loss", 2).n == 1
    assert Matchmaker._streak_adjustment_for(300, "win", 5).n == 1
    assert Matchmaker._streak_adjustment_for(300, "loss", 5).active is False
    assert Matchmaker._streak_adjustment_for(300, "loss", 6).n == 3
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
