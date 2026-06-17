"""
Tests for get_battle_history unified query (summary + legacy fallback)
and _resolve_legacy_opponent helper.
"""

from datetime import datetime, timezone, timedelta
from infrastructure.database import Database


def _fake_db(fetch_returns=None):
    db = Database.__new__(Database)
    db._pool = True
    if fetch_returns is not None:
        db.fetch = fetch_returns
    return db


def _row(**kw):
    class F:
        def __init__(self, d):
            self._d = d
            for k, v in d.items():
                setattr(self, k, v)
        def get(self, key, default=None):
            return self._d.get(key, default)
        def __getitem__(self, key):
            return self._d[key]
        def __contains__(self, key):
            return key in self._d
    return F(kw)


NOW = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
USER = 100
OPP = 200


# ═══════════════════════════════════════════
# _resolve_legacy_opponent tests
# ═══════════════════════════════════════════

class TestResolveLegacyOpponent:
    def test_p1_p2_filled_user_is_p1(self):
        p1, p2, opp = Database._resolve_legacy_opponent(USER, {
            "p1_id": USER, "p2_id": OPP, "winner_id": USER, "loser_id": OPP,
        })
        assert (p1, p2, opp) == (USER, OPP, OPP)

    def test_p1_p2_filled_user_is_p2(self):
        p1, p2, opp = Database._resolve_legacy_opponent(USER, {
            "p1_id": OPP, "p2_id": USER, "winner_id": OPP, "loser_id": USER,
        })
        assert (p1, p2, opp) == (OPP, USER, OPP)

    def test_null_p1_p2_winner_is_user(self):
        p1, p2, opp = Database._resolve_legacy_opponent(USER, {
            "p1_id": None, "p2_id": None, "winner_id": USER, "loser_id": OPP,
        })
        assert (p1, p2, opp) == (None, None, OPP)

    def test_null_p1_p2_loser_is_user(self):
        p1, p2, opp = Database._resolve_legacy_opponent(USER, {
            "p1_id": None, "p2_id": None, "winner_id": OPP, "loser_id": USER,
        })
        assert (p1, p2, opp) == (None, None, OPP)

    def test_user_not_participant(self):
        p1, p2, opp = Database._resolve_legacy_opponent(USER, {
            "p1_id": OPP, "p2_id": 300, "winner_id": OPP, "loser_id": 300,
        })
        assert opp is None

    def test_draw_p1_p2_filled(self):
        p1, p2, opp = Database._resolve_legacy_opponent(USER, {
            "p1_id": USER, "p2_id": OPP, "winner_id": None, "loser_id": None,
        })
        assert (p1, p2, opp) == (USER, OPP, OPP)

    def test_draw_null_p1_p2_no_winner(self):
        p1, p2, opp = Database._resolve_legacy_opponent(USER, {
            "p1_id": None, "p2_id": None, "winner_id": None, "loser_id": None,
        })
        assert opp is None


# ═══════════════════════════════════════════
# _format_battle_row tests
# ═══════════════════════════════════════════

class TestFormatBattleRow:
    def test_user_p1_win(self):
        row = Database._format_battle_row(
            match_id="m1", player_id=USER, p1_id=USER, p2_id=OPP,
            winner_id=USER, loser_id=OPP, p1_trophy_change=25, p2_trophy_change=-10,
            game_mode="classic", match_type=None, duration_seconds=120, turns_count=5,
            created_at=NOW, opponent_id=OPP,
            opponent_info={OPP: {"name": "Bob", "is_bot": False, "avatar_url": "/a.png"}},
        )
        assert row["result"] == "win"
        assert row["trophies_change"] == 25
        assert row["opponent_name"] == "Bob"

    def test_user_p2_lose(self):
        row = Database._format_battle_row(
            match_id="m2", player_id=USER, p1_id=OPP, p2_id=USER,
            winner_id=OPP, loser_id=USER, p1_trophy_change=30, p2_trophy_change=-15,
            game_mode="extra_arena:blitz", match_type=None, duration_seconds=80, turns_count=7,
            created_at=NOW, opponent_id=OPP, opponent_info={},
        )
        assert row["result"] == "lose"
        assert row["trophies_change"] == -15
        assert row["mode"] == "extra_arena:blitz"

    def test_draw(self):
        row = Database._format_battle_row(
            match_id="m3", player_id=USER, p1_id=USER, p2_id=OPP,
            winner_id=None, loser_id=None, p1_trophy_change=0, p2_trophy_change=0,
            game_mode="classic", match_type=None, duration_seconds=50, turns_count=3,
            created_at=NOW, opponent_id=OPP, opponent_info={},
        )
        assert row["result"] == "draw"

    def test_mode_fallback_match_type(self):
        row = Database._format_battle_row(
            match_id="m4", player_id=USER, p1_id=USER, p2_id=OPP,
            winner_id=USER, loser_id=OPP, p1_trophy_change=20, p2_trophy_change=-10,
            game_mode=None, match_type="friendly", duration_seconds=30, turns_count=2,
            created_at=NOW, opponent_id=OPP, opponent_info={},
        )
        assert row["mode"] == "friendly"

    def test_mode_fallback_classic(self):
        row = Database._format_battle_row(
            match_id="m5", player_id=USER, p1_id=USER, p2_id=OPP,
            winner_id=USER, loser_id=OPP, p1_trophy_change=20, p2_trophy_change=-10,
            game_mode=None, match_type=None, duration_seconds=30, turns_count=2,
            created_at=NOW, opponent_id=OPP, opponent_info={},
        )
        assert row["mode"] == "classic"

    def test_legacy_null_p1_p2_win(self):
        row = Database._format_battle_row(
            match_id="m6", player_id=USER, p1_id=None, p2_id=None,
            winner_id=USER, loser_id=OPP, p1_trophy_change=0, p2_trophy_change=0,
            game_mode=None, match_type="pvp", duration_seconds=55, turns_count=4,
            created_at=NOW, opponent_id=OPP, opponent_info={},
        )
        assert row["result"] == "win"
        assert row["trophies_change"] == 0

    def test_legacy_null_p1_p2_lose(self):
        row = Database._format_battle_row(
            match_id="m7", player_id=USER, p1_id=None, p2_id=None,
            winner_id=OPP, loser_id=USER, p1_trophy_change=0, p2_trophy_change=0,
            game_mode=None, match_type="pvp", duration_seconds=55, turns_count=4,
            created_at=NOW, opponent_id=OPP, opponent_info={},
        )
        assert row["result"] == "lose"

    def test_opponent_name_custom_nickname(self):
        row = Database._format_battle_row(
            match_id="m8", player_id=USER, p1_id=USER, p2_id=OPP,
            winner_id=USER, loser_id=OPP, p1_trophy_change=20, p2_trophy_change=-10,
            game_mode=None, match_type=None, duration_seconds=10, turns_count=1,
            created_at=NOW, opponent_id=OPP,
            opponent_info={OPP: {"name": "ShadowNinja", "is_bot": False, "avatar_url": None}},
        )
        assert row["opponent_name"] == "ShadowNinja"

    def test_opponent_name_fallback(self):
        row = Database._format_battle_row(
            match_id="m9", player_id=USER, p1_id=USER, p2_id=OPP,
            winner_id=USER, loser_id=OPP, p1_trophy_change=20, p2_trophy_change=-10,
            game_mode=None, match_type=None, duration_seconds=10, turns_count=1,
            created_at=NOW, opponent_id=OPP, opponent_info={},
        )
        assert row["opponent_name"] == "Игрок"

    def test_bot_opponent_flags(self):
        row = Database._format_battle_row(
            match_id="m10", player_id=USER, p1_id=USER, p2_id=OPP,
            winner_id=USER, loser_id=OPP, p1_trophy_change=20, p2_trophy_change=-10,
            game_mode=None, match_type=None, duration_seconds=10, turns_count=1,
            created_at=NOW, opponent_id=OPP,
            opponent_info={OPP: {"name": "Bot_Alpha", "is_bot": True, "avatar_url": "/bot.png"}},
        )
        assert row["opponent_name"] == "Bot_Alpha"
        assert row["opponent_is_bot"] is True


# ═══════════════════════════════════════════
# get_battle_history integration tests (3 API calls: summary, legacy, opponents)
# ═══════════════════════════════════════════

class TestGetBattleHistoryIntegration:
    def test_summary_only(self):
        """User p1, win — summary only, empty legacy."""
        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # summary
                return [_row(match_id="s1", p1_user_id=USER, p2_user_id=OPP,
                             winner_user_id=USER, loser_user_id=OPP,
                             p1_trophy_change=30, p2_trophy_change=-12,
                             game_mode="classic", match_type=None,
                             duration_seconds=180, turns_count=6, created_at=NOW)]
            if call_count == 2:  # legacy
                return []
            if call_count == 3:  # opponents
                return [_row(user_id=OPP, first_name="Alice", username="alice",
                             is_bot=False, custom_nickname="CoolAlice",
                             img="/a.jpg", equipped_avatar_url=None)]
            return []

        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER))
        assert len(history) == 1
        b = history[0]
        assert b["battle_id"] == "s1"
        assert b["result"] == "win"
        assert b["opponent_name"] == "CoolAlice"

    def test_summary_user_p2_lose(self):
        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1:
                return [_row(match_id="s2", p1_user_id=OPP, p2_user_id=USER,
                             winner_user_id=OPP, loser_user_id=USER,
                             p1_trophy_change=25, p2_trophy_change=-15,
                             game_mode="extra_arena:blitz", match_type=None,
                             duration_seconds=90, turns_count=4, created_at=NOW)]
            if call_count == 2: return []
            if call_count == 3:
                return [_row(user_id=OPP, first_name=None, username="bob123",
                             is_bot=False, custom_nickname=None, img=None, equipped_avatar_url=None)]
            return []
        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER))
        assert len(history) == 1
        assert history[0]["result"] == "lose"
        assert history[0]["opponent_name"] == "bob123"

    def test_draw(self):
        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1:
                return [_row(match_id="s3", p1_user_id=USER, p2_user_id=OPP,
                             winner_user_id=None, loser_user_id=None,
                             p1_trophy_change=0, p2_trophy_change=0,
                             game_mode="classic", match_type=None,
                             duration_seconds=200, turns_count=20, created_at=NOW)]
            if call_count == 2: return []
            if call_count == 3:
                return [_row(user_id=OPP, first_name="Charlie", username="ch",
                             is_bot=False, custom_nickname=None, img=None, equipped_avatar_url=None)]
            return []
        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER))
        assert len(history) == 1
        assert history[0]["result"] == "draw"

    def test_legacy_p1_p2_filled(self):
        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1: return []
            if call_count == 2:
                return [_row(match_id="lr1", p1_id=USER, p2_id=OPP,
                             winner_id=USER, loser_id=OPP,
                             p1_trophy_change=20, p2_trophy_change=-8,
                             match_type="classic", match_duration=150,
                             turns_count=5, created_at=NOW)]
            if call_count == 3:
                return [_row(user_id=OPP, first_name="Dan", username="dan",
                             is_bot=False, custom_nickname=None, img=None, equipped_avatar_url=None)]
            return []
        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER))
        assert len(history) == 1
        assert history[0]["result"] == "win"
        assert history[0]["trophies_change"] == 20

    def test_legacy_null_p1_p2_winner_is_user(self):
        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1: return []
            if call_count == 2:
                return [_row(match_id="lr2", p1_id=None, p2_id=None,
                             winner_id=USER, loser_id=OPP,
                             p1_trophy_change=0, p2_trophy_change=0,
                             match_type="pvp", match_duration=60,
                             turns_count=3, created_at=NOW)]
            if call_count == 3:
                return [_row(user_id=OPP, first_name="Eve", username="eve",
                             is_bot=False, custom_nickname=None, img=None, equipped_avatar_url=None)]
            return []
        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER))
        assert len(history) == 1
        assert history[0]["result"] == "win"

    def test_legacy_null_p1_p2_loser_is_user(self):
        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1: return []
            if call_count == 2:
                return [_row(match_id="lr3", p1_id=None, p2_id=None,
                             winner_id=OPP, loser_id=USER,
                             p1_trophy_change=0, p2_trophy_change=0,
                             match_type="pvp", match_duration=70,
                             turns_count=4, created_at=NOW)]
            if call_count == 3:
                return [_row(user_id=OPP, first_name="Frank", username="frank",
                             is_bot=True, custom_nickname="BotFrank",
                             img="/bot.png", equipped_avatar_url=None)]
            return []
        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER))
        assert len(history) == 1
        assert history[0]["result"] == "lose"
        assert history[0]["opponent_is_bot"] is True

    def test_row_user_not_participant_skipped(self):
        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1: return []
            if call_count == 2:
                return [
                    _row(match_id="lr4", p1_id=None, p2_id=None,
                         winner_id=OPP, loser_id=999,
                         p1_trophy_change=0, p2_trophy_change=0,
                         match_type="pvp", match_duration=50,
                         turns_count=3, created_at=NOW),
                    _row(match_id="lr5", p1_id=USER, p2_id=OPP,
                         winner_id=USER, loser_id=OPP,
                         p1_trophy_change=10, p2_trophy_change=-5,
                         match_type="classic", match_duration=100,
                         turns_count=4, created_at=NOW),
                ]
            if call_count == 3:
                return [_row(user_id=OPP, first_name="Gina", username="gina",
                             is_bot=False, custom_nickname=None, img=None, equipped_avatar_url=None)]
            return []
        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER))
        assert len(history) == 1
        assert history[0]["battle_id"] == "lr5"

    def test_same_match_id_prefers_summary(self):
        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1:
                return [_row(match_id="dup1", p1_user_id=USER, p2_user_id=OPP,
                             winner_user_id=USER, loser_user_id=OPP,
                             p1_trophy_change=30, p2_trophy_change=-12,
                             game_mode="extra_arena:blitz", match_type=None,
                             duration_seconds=120, turns_count=5, created_at=NOW)]
            if call_count == 2: return []  # excluded by match_id NOT IN
            if call_count == 3:
                return [_row(user_id=OPP, first_name="Helen", username="helen",
                             is_bot=False, custom_nickname=None, img=None, equipped_avatar_url=None)]
            return []
        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER))
        assert len(history) == 1
        assert history[0]["mode"] == "extra_arena:blitz"

    def test_more_legacy_than_remaining_global_sort(self):
        """
        Regression: 1 summary + 12 legacy = 13 total, limit=5.
        Legacy match_ids don't match created_at ordering.
        Must pass: global ORDER BY created_at DESC, then LIMIT.
        """
        t = lambda h: datetime(2026, 5, 17, h, 0, 0, tzinfo=timezone.utc)

        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1:
                return [_row(match_id="s1", p1_user_id=USER, p2_user_id=OPP,
                             winner_user_id=USER, loser_user_id=OPP,
                             p1_trophy_change=10, p2_trophy_change=-5,
                             game_mode="classic", match_type=None,
                             duration_seconds=100, turns_count=4, created_at=t(10))]
            if call_count == 2:
                # 12 legacy rows with varying created_at, sorted mix of old and new
                legacy = []
                for i in range(12):
                    legacy.append(_row(
                        match_id=f"lr{i}", p1_id=USER, p2_id=OPP,
                        winner_id=USER, loser_id=OPP,
                        p1_trophy_change=5, p2_trophy_change=-3,
                        match_type="pvp", match_duration=50,
                        turns_count=3, created_at=t(23 - i),  # lr0=23:00, lr1=22:00, ...
                    ))
                return legacy
            if call_count == 3:
                return [_row(user_id=OPP, first_name="Zoe", username="z",
                             is_bot=False, custom_nickname=None, img=None, equipped_avatar_url=None)]
            return []

        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER, limit=5))

        assert len(history) == 5
        # Top 5 by created_at DESC: lr0 (23:00), lr1 (22:00), lr2 (21:00), lr3 (20:00), lr4 (19:00)
        expected = ["lr0", "lr1", "lr2", "lr3", "lr4"]
        actual = [b["battle_id"] for b in history]
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_legacy_dedup_keeps_newest_per_match_id(self):
        """
        Legacy has 2 rows for the same match_id with different dates.
        Only the newer one should count (dedup then global sort).
        """
        t_old = datetime(2026, 5, 16, 10, 0, 0, tzinfo=timezone.utc)
        t_new = datetime(2026, 5, 17, 15, 0, 0, tzinfo=timezone.utc)

        call_count = 0
        async def mock_fetch(query, *args):
            nonlocal call_count; call_count += 1
            if call_count == 1: return []
            if call_count == 2:
                return [
                    _row(match_id="dup", p1_id=USER, p2_id=OPP,
                         winner_id=USER, loser_id=OPP,
                         p1_trophy_change=5, p2_trophy_change=-3,
                         match_type="pvp", match_duration=50,
                         turns_count=3, created_at=t_old),
                    _row(match_id="dup", p1_id=USER, p2_id=OPP,
                         winner_id=USER, loser_id=OPP,
                         p1_trophy_change=8, p2_trophy_change=-4,
                         match_type="pvp", match_duration=60,
                         turns_count=4, created_at=t_new),
                ]
            if call_count == 3:
                return [_row(user_id=OPP, first_name="Yuri", username="y",
                             is_bot=False, custom_nickname=None, img=None, equipped_avatar_url=None)]
            return []
        db = _fake_db(fetch_returns=mock_fetch)
        history = _run(db.get_battle_history(USER, limit=10))
        assert len(history) == 1
        assert history[0]["battle_id"] == "dup"
        assert history[0]["trophies_change"] == 8  # from newer row


def _run(coro):
    import asyncio
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro)


class TestCurrentResultStreak:
    def test_summary_win_streak_skips_training_and_friendly(self):
        async def mock_fetch(query, *args):
            if "FROM battle_results" in query:
                return []
            if "FROM battle_summary" in query:
                return [
                    _row(id=5, match_id="training-win", p1_user_id=USER, p2_user_id=OPP,
                         winner_user_id=USER, loser_user_id=OPP,
                         game_mode="training", match_type="pve", created_at=NOW + timedelta(seconds=2)),
                    _row(id=4, match_id="friendly-loss", p1_user_id=USER, p2_user_id=OPP,
                         winner_user_id=OPP, loser_user_id=USER,
                         game_mode="classic", match_type="friendly", created_at=NOW + timedelta(seconds=1)),
                    _row(id=3, match_id="s1", p1_user_id=USER, p2_user_id=OPP,
                         winner_user_id=USER, loser_user_id=OPP,
                         game_mode="classic", match_type="pvp", created_at=NOW),
                    _row(id=2, match_id="s2", p1_user_id=OPP, p2_user_id=USER,
                         winner_user_id=USER, loser_user_id=OPP,
                         game_mode="extra_arena:spellstorm", match_type="pvp", created_at=NOW - timedelta(seconds=1)),
                    _row(id=1, match_id="s3", p1_user_id=USER, p2_user_id=OPP,
                         winner_user_id=OPP, loser_user_id=USER,
                         game_mode="classic", match_type="pvp", created_at=NOW - timedelta(seconds=2)),
                ]
            return []

        db = _fake_db(fetch_returns=mock_fetch)
        assert _run(db.get_current_result_streak(USER)) == {"kind": "win", "length": 2}

    def test_legacy_loss_streak_dedupes_summary_and_breaks_on_draw(self):
        async def mock_fetch(query, *args):
            if "FROM battle_results" in query:
                return [
                    _row(id=50, match_id="dup", p1_id=USER, p2_id=OPP,
                         winner_id=USER, loser_id=OPP,
                         match_type="pvp", created_at=NOW + timedelta(seconds=5)),
                    _row(id=40, match_id="l1", p1_id=OPP, p2_id=USER,
                         winner_id=OPP, loser_id=USER,
                         match_type="pvp", created_at=NOW - timedelta(seconds=1)),
                    _row(id=30, match_id="draw", p1_id=USER, p2_id=OPP,
                         winner_id=None, loser_id=None,
                         match_type="pvp", created_at=NOW - timedelta(seconds=2)),
                    _row(id=20, match_id="old", p1_id=OPP, p2_id=USER,
                         winner_id=OPP, loser_id=USER,
                         match_type="pvp", created_at=NOW - timedelta(seconds=3)),
                ]
            if "FROM battle_summary" in query:
                return [
                    _row(id=60, match_id="dup", p1_user_id=USER, p2_user_id=OPP,
                         winner_user_id=OPP, loser_user_id=USER,
                         game_mode="classic", match_type="pvp", created_at=NOW),
                ]
            return []

        db = _fake_db(fetch_returns=mock_fetch)
        assert _run(db.get_current_result_streak(USER)) == {"kind": "loss", "length": 2}

    def test_same_timestamp_prefers_summary_then_higher_id_deterministically(self):
        async def mock_fetch(query, *args):
            if "FROM battle_results" in query:
                return [
                    _row(id=100, match_id="legacy-new", p1_id=USER, p2_id=OPP,
                         winner_id=OPP, loser_id=USER,
                         match_type="pvp", created_at=NOW),
                ]
            if "FROM battle_summary" in query:
                return [
                    _row(id=10, match_id="summary-old-id", p1_user_id=USER, p2_user_id=OPP,
                         winner_user_id=OPP, loser_user_id=USER,
                         game_mode="classic", match_type="pvp", created_at=NOW),
                    _row(id=11, match_id="summary-new-id", p1_user_id=USER, p2_user_id=OPP,
                         winner_user_id=USER, loser_user_id=OPP,
                         game_mode="classic", match_type="pvp", created_at=NOW),
                ]
            return []

        db = _fake_db(fetch_returns=mock_fetch)
        assert _run(db.get_current_result_streak(USER)) == {"kind": "win", "length": 1}
