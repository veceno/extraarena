import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
import pytest_asyncio

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
            "bot_info": {"name": "Bot"},
        }


def make_matchmaker():
    db = FakeDB()
    return Matchmaker(db=db, bot_factory=FakeBotFactory(), battle_engine=None)


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
    mm = make_matchmaker()
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
    mm = Matchmaker(db=db, bot_factory=factory, battle_engine=None)

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
async def test_soft_start_does_not_reuse_finished_bot_match():
    db = FakeDB()
    factory = SlowCountingBotFactory()
    mm = Matchmaker(db=db, bot_factory=factory, battle_engine=None)

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
