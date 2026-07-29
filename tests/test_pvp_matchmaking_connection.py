import asyncio
import ast
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from core.classic_setup import create_classic_game_state
from core.actions import EndTurnAction
from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, ReplacementStatus
import battle_engine as battle_engine_module
from battle_engine import BattleEngine
from infrastructure.database import Database
from infrastructure.matchmaking import Matchmaker
from infrastructure.matchmaking import QueueEntry
from infrastructure.match_modes import ClassicParams, ModeConfig
from web import server as web_server


def _hero(card_id: int = 1) -> CardInstance:
    return CardInstance(
        instance_id=uuid.uuid4(),
        card_id=card_id,
        name="Hero",
        card_type=CardType.HERO,
        rarity="common",
        mana_cost=0,
        attack=0,
        hp=30,
        max_hp=30,
        mechanics=[],
    )


class FakeDB:
    async def get_user_deck_presets(self, user_id):
        return []


class FakeBotFactory:
    pass


class ReadyBotFactory:
    async def create_match(self, user_id, trophies):
        return {
            "match_id": "ready-bot-match",
            "opponent_id": -900000001,
            "bot_info": {"name": "ReadyBot", "deck_ids": list(range(1, 10))},
        }


class MinimalWebDB:
    pass


class MatchFindDB:
    def __init__(self):
        self.get_player_deck_max_level_calls = 0

    async def get_onboarding_state(self, user_id):
        return {"completed": True, "status": "completed"}

    async def get_user_info(self, user_id):
        return {
            "user_id": user_id,
            "trophies": 500,
            "max_trophies": 500,
            "league": 1,
            "username": f"user{user_id}",
            "first_name": "User",
            "last_name": "",
            "avatar_url": "",
            "clan": "",
            "title": "",
            "title_class": "",
            "extra_pass": "inactive",
            "background_url": "",
            "nickname_glow_disabled": False,
            "hide_player_id_public": False,
        }

    async def get_deck_presets(self, user_id):
        return [{"preset_number": 1}]

    async def get_player_deck_max_level(self, user_id, selected_deck_id=None):
        self.get_player_deck_max_level_calls += 1
        return 1

    async def is_admin(self, user_id):
        return False

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": dict(web_server.RUNTIME_FEATURE_DEFAULTS),
            "disabled_card_ids": [],
        }

    async def get_match_mode_overrides(self):
        return []


class EndTurnEngine:
    def __init__(self, *, expired: bool = False):
        self.p1_state = SimpleNamespace(user_id=101, is_bot=False, replacement_status=ReplacementStatus.ACTIVE)
        self.p2_state = SimpleNamespace(user_id=202, is_bot=False, replacement_status=ReplacementStatus.ACTIVE)
        self.current_player_id = 101
        self.turn = 1
        self.is_ended = False
        self.rewards_granted = False
        self.battle_end_processed = False
        self.expired = expired
        self.end_turn_calls = 0
        self.activity = []
        self.waiting_for_players = False

    def is_turn_expired(self):
        return self.expired

    def is_waiting_for_players(self):
        return self.waiting_for_players

    def mark_player_activity(self, user_id):
        self.activity.append(user_id)

    def end_turn(self, user_id):
        self.end_turn_calls += 1
        self.current_player_id = 202
        return {"success": True, "current_player_id": self.current_player_id}

    def get_current_player_id(self):
        return self.current_player_id

    def get_player_replacement_status(self, user_id):
        if int(user_id) == int(self.p1_state.user_id):
            return self.p1_state.replacement_status
        if int(user_id) == int(self.p2_state.user_id):
            return self.p2_state.replacement_status
        return ReplacementStatus.ACTIVE

    def get_full_state(self, viewer_id=None):
        return {
            "match_id": "m-turn",
            "viewer_id": viewer_id,
            "current_player_id": self.current_player_id,
            "match_status": "waiting_for_players" if self.waiting_for_players else "active",
        }


class TerminalEndTurnEngine(EndTurnEngine):
    def __init__(self, *, expired: bool = False):
        super().__init__(expired=expired)
        self.winner_id = 101

    def end_turn(self, user_id):
        self.end_turn_calls += 1
        self.is_ended = True
        return {
            "success": True,
            "game_over": True,
            "winner": self.winner_id,
            "winner_id": self.winner_id,
        }

    def check_game_over(self):
        return {"game_over": self.is_ended, "winner_id": self.winner_id if self.is_ended else None}


class FailingEndTurnEngine(EndTurnEngine):
    def end_turn(self, user_id):
        raise RuntimeError("secret battle shard connection string")


class SurrenderEngine:
    def __init__(self):
        self.p1_state = SimpleNamespace(
            user_id=101,
            is_bot=False,
            replacement_status=ReplacementStatus.SURRENDERED,
            surrender_processed=False,
        )
        self.p2_state = SimpleNamespace(user_id=202, is_bot=False, replacement_status=ReplacementStatus.ACTIVE)
        self.game_mode = "classic"
        self._trophy_changes = {}
        self._trophy_totals = {}

    def get_player_state(self, user_id):
        return self.p1_state if int(user_id) == 101 else self.p2_state


class PveSurrenderEngine(EndTurnEngine):
    def __init__(self):
        super().__init__()
        self.p2_state = SimpleNamespace(
            user_id=-900000001,
            is_bot=True,
            replacement_status=ReplacementStatus.ACTIVE,
        )
        self.is_bot_match = True
        self.game_mode = "classic"
        self._trophy_changes = {}
        self._trophy_totals = {}

    def get_player_state(self, user_id):
        return self.p1_state if int(user_id) == 101 else self.p2_state

    def mark_surrender(self, user_id):
        self.get_player_state(user_id).replacement_status = ReplacementStatus.SURRENDERED

    def check_game_over(self):
        return {"game_over": False, "winner_id": None}


class SurrenderDB:
    def __init__(self):
        self.trophy_updates = 0

    async def get_user_info(self, user_id):
        return {"trophies": 100, "league": 1}

    async def update_user_trophies(self, user_id, delta):
        self.trophy_updates += 1
        return {"trophies": 100 + int(delta), "max_trophies": 100, "league": 1}


class IdempotentSurrenderDB(SurrenderDB):
    def __init__(self):
        super().__init__()
        self.claimed = set()
        self.apply_calls = 0
        self._lock = asyncio.Lock()

    async def apply_surrender_penalty_once(self, *, match_id, user_id, reason, penalty_delta):
        self.apply_calls += 1
        key = (str(match_id), int(user_id), str(reason))
        async with self._lock:
            if key in self.claimed:
                return {
                    "applied": False,
                    "already_processed": True,
                    "trophy_penalty": 0,
                    "new_trophies": 100 + (self.trophy_updates * int(penalty_delta)),
                }
            await asyncio.sleep(0)
            self.claimed.add(key)
            self.trophy_updates += 1
            return {
                "applied": True,
                "already_processed": False,
                "trophy_penalty": int(penalty_delta),
                "new_trophies": 100 + int(penalty_delta),
            }


async def _battle_client(monkeypatch, engine, match_id="m-turn"):
    async def fake_check_and_run_bot(_match_id, _active_matches):
        return None

    monkeypatch.setattr(web_server, "check_and_run_bot", fake_check_and_run_bot)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()
    app = web_server.create_web_app(MinimalWebDB(), bot_token="bot-token")
    web_server.ACTIVE_MATCHES[match_id] = engine
    app["active_matches"][match_id] = engine
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _match_find_client(monkeypatch, db, engine, match_id="m-active-matchfind"):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()
    app = web_server.create_web_app(db, bot_token="bot-token")
    web_server.ACTIVE_MATCHES[match_id] = engine
    app["active_matches"][match_id] = engine
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_classic_game_state_defaults_to_p1_start():
    state = create_classic_game_state(101, 202, [_hero(1)], [_hero(2)])

    assert state.current_turn_owner_id == 101
    assert state.p1.mana == state.p1.max_mana == 1
    assert state.p2.mana == state.p2.max_mana == 0


def test_classic_game_state_can_start_p2_without_swapping_sides():
    state = create_classic_game_state(
        101,
        202,
        [_hero(1)],
        [_hero(2)],
        starting_player_id=202,
    )

    assert state.p1.user_id == 101
    assert state.p2.user_id == 202
    assert state.current_turn_owner_id == 202
    assert state.p1.mana == state.p1.max_mana == 0
    assert state.p2.mana == state.p2.max_mana == 1


@pytest.mark.asyncio(loop_scope="function")
async def test_pvp_payload_has_final_match_players_decks_and_starting_player(monkeypatch):
    monkeypatch.setattr("infrastructure.matchmaking.random.choice", lambda ids: ids[1])
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    first = await mm.find_match(1, 500, 1, selected_deck_id=3, game_mode="classic")
    second = await mm.find_match(2, 500, 1, selected_deck_id=4, game_mode="classic")
    first_status = await mm.get_status(first["match_id"])
    final_status = await mm.get_status(second["match_id"])

    assert second["status"] == "found"
    assert second["match_id"] == first_status["match_id"] == final_status["match_id"]
    assert final_status["player_ids"] == [2, 1]
    assert final_status["player_decks"] == {"2": 4, "1": 3}
    assert second["player_decks"] == {"2": 4, "1": 3}
    assert first_status["player_decks"] == {"2": 4, "1": 3}
    assert final_status["starting_player_id"] == 1
    assert first_status["starting_player_id"] == 1


@pytest.mark.asyncio(loop_scope="function")
async def test_matchmaker_status_returns_copy_not_internal_reference():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    first = await mm.find_match(101, 500, 1, selected_deck_id=1, game_mode="classic")
    status = await mm.get_status(first["match_id"])
    status["status"] = "mutated-by-client"
    status["preview_only"] = True

    fresh_status = await mm.get_status(first["match_id"])
    assert fresh_status["status"] == "waiting"
    assert "preview_only" not in fresh_status


def test_matchmaker_candidate_prefers_nearest_trophies_then_wait_time():
    mm = Matchmaker(FakeDB(), FakeBotFactory())
    seeker = QueueEntry(user_id=99, trophies=150, max_level=1, enqueued_at=10.0)
    far = QueueEntry(user_id=1, trophies=100, max_level=1, enqueued_at=1.0)
    near_newer = QueueEntry(user_id=2, trophies=149, max_level=1, enqueued_at=5.0)
    near_older = QueueEntry(user_id=3, trophies=151, max_level=1, enqueued_at=3.0)
    mm._queue = [far, near_newer, near_older]

    assert mm._find_candidate(seeker, 50) is near_older


@pytest.mark.asyncio(loop_scope="function")
async def test_cancel_final_match_id_cancels_all_aliases():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    first = await mm.find_match(11, 500, 1, selected_deck_id=31, game_mode="classic")
    second = await mm.find_match(22, 500, 1, selected_deck_id=42, game_mode="classic")
    final_match_id = second["match_id"]

    canceled = await mm.cancel_match(final_match_id, message="nope", error="unit")
    first_status = await mm.get_status(first["match_id"])
    second_status = await mm.get_status(final_match_id)

    assert canceled["status"] == "canceled"
    assert first_status["status"] == "canceled"
    assert second_status["status"] == "canceled"


@pytest.mark.asyncio(loop_scope="function")
async def test_cancel_final_match_id_preserves_participants_for_status_auth():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    await mm.find_match(11, 500, 1, selected_deck_id=31, game_mode="classic")
    second = await mm.find_match(22, 500, 1, selected_deck_id=42, game_mode="classic")

    canceled = await mm.cancel_match(second["match_id"], message="nope", error="unit")
    final_status = await mm.get_status(second["match_id"])

    assert canceled["status"] == "canceled"
    assert final_status["status"] == "canceled"
    assert final_status["player_ids"] == [22, 11]
    assert set(map(int, final_status["player_ids"])) == {11, 22}
    assert final_status["opponent_id"] == 11
    assert final_status["player_decks"] == {"22": 42, "11": 31}


@pytest.mark.asyncio(loop_scope="function")
async def test_canceled_final_match_status_allows_both_participants(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()

    mm = Matchmaker(FakeDB(), FakeBotFactory())
    await mm.find_match(11, 500, 1, selected_deck_id=31, game_mode="classic")
    second = await mm.find_match(22, 500, 1, selected_deck_id=42, game_mode="classic")
    await mm.cancel_match(second["match_id"], message="nope", error="unit")

    app = web_server.create_web_app(MatchFindDB(), bot_token="bot-token")
    app["matchmaker"] = mm
    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        first_response = await client.get(f"/api/match/status?id={second['match_id']}&user_id=11")
        second_response = await client.get(f"/api/match/status?id={second['match_id']}&user_id=22")
        first_body = await first_response.json()
        second_body = await second_response.json()

        assert first_response.status == 200
        assert second_response.status == 200
        assert first_body["status"] == "canceled"
        assert second_body["status"] == "canceled"
    finally:
        await client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_matchmaker_prunes_terminal_match_aliases_by_ttl():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    first = await mm.find_match(11, 500, 1, selected_deck_id=31, game_mode="classic")
    second = await mm.find_match(22, 500, 1, selected_deck_id=42, game_mode="classic")
    await mm.cancel_match(second["match_id"], message="done", error="unit")

    removed = await mm.prune_expired_matches(now=time.monotonic() + 3600, ttl_seconds=1)

    assert removed >= 1
    assert (await mm.get_status(first["match_id"]))["status"] == "not_found"
    assert (await mm.get_status(second["match_id"]))["status"] == "not_found"


@pytest.mark.asyncio(loop_scope="function")
async def test_matchmaker_prunes_stale_found_match_aliases_by_ttl():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    first = await mm.find_match(11, 500, 1, selected_deck_id=31, game_mode="classic")
    second = await mm.find_match(22, 500, 1, selected_deck_id=42, game_mode="classic")
    aliases = set(mm._match_aliases[second["match_id"]])

    removed = await mm.prune_expired_matches(now=time.monotonic() + 3600, ttl_seconds=1)

    assert second["status"] == "found"
    assert removed >= 1
    assert (await mm.get_status(first["match_id"]))["status"] == "not_found"
    assert (await mm.get_status(second["match_id"]))["status"] == "not_found"
    assert await mm.get_active_match_for_user(11) == {"status": "not_found", "user_id": 11}
    assert await mm.get_active_match_for_user(22) == {"status": "not_found", "user_id": 22}
    assert not aliases & set(mm._matches)
    assert not aliases & set(mm._match_aliases)
    assert not aliases & set(mm._match_updated_at)


@pytest.mark.asyncio(loop_scope="function")
async def test_matchmaker_found_status_polling_refreshes_ttl():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    await mm.find_match(11, 500, 1, selected_deck_id=31, game_mode="classic")
    second = await mm.find_match(22, 500, 1, selected_deck_id=42, game_mode="classic")
    aliases = set(mm._match_aliases[second["match_id"]])
    for alias in aliases:
        mm._match_updated_at[alias] = 10.0

    status = await mm.get_status(second["match_id"])

    assert status["status"] == "found"
    assert all(mm._match_updated_at[alias] > 10.0 for alias in aliases)


@pytest.mark.asyncio(loop_scope="function")
async def test_matchmaker_prunes_stale_pve_found_match_by_ttl():
    mm = Matchmaker(FakeDB(), ReadyBotFactory())

    result = await mm._create_bot_match(
        user_id=10,
        trophies=100,
        user_max_level=1,
        selected_deck_id=1,
        game_mode="classic",
    )

    removed = await mm.prune_expired_matches(now=time.monotonic() + 3600, ttl_seconds=1)

    assert result["status"] == "found"
    assert removed == 1
    assert (await mm.get_status(result["match_id"]))["status"] == "not_found"


@pytest.mark.asyncio(loop_scope="function")
async def test_server_matchmaker_cache_prune_uses_finished_match_ttl():
    class PruneMatchmaker:
        def __init__(self):
            self.ttl_seconds = None

        async def prune_expired_matches(self, *, ttl_seconds):
            self.ttl_seconds = ttl_seconds
            return 2

    matchmaker = PruneMatchmaker()

    removed = await web_server._prune_matchmaker_cache({"matchmaker": matchmaker})

    assert removed == 2
    assert matchmaker.ttl_seconds == web_server.FINISHED_MATCH_TTL_SECONDS


def test_matchmaker_poll_interval_scales_below_short_window():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    assert mm._poll_interval_for_window(5.0) <= 1.25


@pytest.mark.asyncio(loop_scope="function")
async def test_turn_end_client_action_id_is_cached_without_double_execution(monkeypatch):
    engine = EndTurnEngine()
    client = await _battle_client(monkeypatch, engine)

    try:
        body = {"match_id": "m-turn", "user_id": 101, "client_action_id": "turn-1"}
        first = await client.post("/api/battle/end-turn", json=body)
        first_body = await first.json()
        second = await client.post("/api/battle/end-turn", json=body)
        second_body = await second.json()

        assert first.status == 200
        assert second.status == 200
        assert engine.end_turn_calls == 1
        assert first_body == second_body
        assert first_body["state"]["viewer_id"] == 101
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-turn", None)
        web_server.MATCH_LOCKS.pop("m-turn", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_duplicate_client_action_id_is_rechecked_after_match_lock(monkeypatch):
    engine = EndTurnEngine()
    match_id = "m-concurrent-action"
    client = await _battle_client(monkeypatch, engine, match_id=match_id)
    lock = web_server._get_match_lock(match_id)
    await lock.acquire()

    try:
        body = {"match_id": match_id, "user_id": 101, "client_action_id": "turn-race-1"}
        first_task = asyncio.create_task(client.post("/api/battle/end-turn", json=body))
        second_task = asyncio.create_task(client.post("/api/battle/end-turn", json=body))
        await asyncio.sleep(0.05)

        lock.release()
        first, second = await asyncio.gather(first_task, second_task)
        first_body = await first.json()
        second_body = await second.json()

        assert first.status == 200
        assert second.status == 200
        assert engine.end_turn_calls == 1
        assert first_body == second_body
    finally:
        if lock.locked():
            lock.release()
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_turn_end_terminal_state_finalizes_once(monkeypatch):
    engine = TerminalEndTurnEngine()
    client = await _battle_client(monkeypatch, engine, match_id="m-terminal-end")
    finalized = []

    async def fake_process_battle_end(app, match_id, engine_arg, winner_id):
        finalized.append((match_id, engine_arg, winner_id))
        engine_arg.battle_end_processed = True
        engine_arg.rewards_granted = True
        return True

    monkeypatch.setattr(web_server, "_process_battle_end", fake_process_battle_end)

    try:
        first = await client.post(
            "/api/battle/end-turn",
            json={"match_id": "m-terminal-end", "user_id": 101, "client_action_id": "terminal-1"},
        )
        second = await client.post(
            "/api/battle/end-turn",
            json={"match_id": "m-terminal-end", "user_id": 101, "client_action_id": "terminal-2"},
        )

        assert first.status == 200
        assert second.status == 200
        assert finalized == [("m-terminal-end", engine, 101)]
        assert engine.end_turn_calls == 1
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-terminal-end", None)
        web_server.MATCH_LOCKS.pop("m-terminal-end", None)
        web_server.ACTION_RESULT_CACHE.clear()
        web_server.ENDED_MATCH_IDS.discard("m-terminal-end")


@pytest.mark.asyncio(loop_scope="function")
async def test_unprocessed_terminal_match_retries_finalization_without_mutation(monkeypatch):
    engine = TerminalEndTurnEngine()
    client = await _battle_client(monkeypatch, engine, match_id="m-terminal-retry")
    attempts = []

    async def fake_process_battle_end(app, match_id, engine_arg, winner_id):
        attempts.append((match_id, winner_id))
        if len(attempts) == 1:
            return False
        engine_arg.battle_end_processed = True
        engine_arg.rewards_granted = True
        web_server._mark_match_ended(match_id)
        return True

    monkeypatch.setattr(web_server, "_process_battle_end", fake_process_battle_end)

    try:
        first = await client.post(
            "/api/battle/end-turn",
            json={"match_id": "m-terminal-retry", "user_id": 101, "client_action_id": "retry-1"},
        )
        second = await client.post(
            "/api/battle/end-turn",
            json={"match_id": "m-terminal-retry", "user_id": 101, "client_action_id": "retry-2"},
        )
        second_body = await second.json()

        assert first.status == 200
        assert second.status == 200
        assert second_body["error"] == "game_already_ended"
        assert attempts == [("m-terminal-retry", 101), ("m-terminal-retry", 101)]
        assert engine.end_turn_calls == 1
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-terminal-retry", None)
        web_server.MATCH_LOCKS.pop("m-terminal-retry", None)
        web_server.ACTION_RESULT_CACHE.clear()
        web_server.ENDED_MATCH_IDS.discard("m-terminal-retry")


@pytest.mark.asyncio(loop_scope="function")
async def test_natural_timeout_terminal_state_finalizes(monkeypatch):
    engine = TerminalEndTurnEngine(expired=True)
    finalized = []

    async def fake_process_battle_end(app, match_id, engine_arg, winner_id):
        finalized.append((match_id, engine_arg, winner_id))
        engine_arg.battle_end_processed = True
        engine_arg.rewards_granted = True
        return True

    monkeypatch.setattr(web_server, "_process_battle_end", fake_process_battle_end)
    bot_checks = []

    async def fake_check_and_run_bot(match_id, active_matches):
        bot_checks.append(match_id)

    monkeypatch.setattr(web_server, "check_and_run_bot", fake_check_and_run_bot)
    web_server.ACTIVE_MATCHES["m-terminal-timeout"] = engine

    try:
        handled = await web_server._handle_natural_turn_timeout(
            {"socketio": None, "matchmaker": None},
            "m-terminal-timeout",
            engine,
        )

        assert handled is True
        assert finalized == [("m-terminal-timeout", engine, 101)]
        assert engine.end_turn_calls == 1
        assert bot_checks == []
    finally:
        web_server.ACTIVE_MATCHES.pop("m-terminal-timeout", None)
        web_server.MATCH_LOCKS.pop("m-terminal-timeout", None)
        web_server.ENDED_MATCH_IDS.discard("m-terminal-timeout")


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/battle/end-turn", {"match_id": "m-action-id-required", "user_id": 101}),
        ("/api/battle/play-card", {"match_id": "m-action-id-required", "user_id": 101, "card_id": 1}),
        (
            "/api/battle/attack",
            {"match_id": "m-action-id-required", "user_id": 101, "attacker_id": "a1", "target_id": "hero"},
        ),
        ("/api/matches/m-action-id-required/surrender", {"user_id": 101}),
    ],
)
@pytest.mark.asyncio(loop_scope="function")
async def test_battle_mutations_require_client_action_id(monkeypatch, path, body):
    engine = EndTurnEngine()
    client = await _battle_client(monkeypatch, engine, match_id="m-action-id-required")

    try:
        response = await client.post(path, json=body)
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "client_action_id_required"
        assert engine.end_turn_calls == 0
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-action-id-required", None)
        web_server.MATCH_LOCKS.pop("m-action-id-required", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_action_timeout_guard_runs_while_match_lock_is_held(monkeypatch):
    engine = EndTurnEngine(expired=True)
    client = await _battle_client(monkeypatch, engine, match_id="m-lock")
    web_server.MATCH_LOCKS["m-lock"] = asyncio.Lock()

    async def fake_auto_end(app, match_id, engine_arg, viewer_id, client_action_id, **kwargs):
        assert web_server.MATCH_LOCKS[str(match_id)].locked()
        assert kwargs["lock_already_held"] is True
        assert engine_arg is engine
        return web.json_response(
            {"error": "turn_expired", "state": engine.get_full_state(viewer_id=viewer_id)},
            status=409,
        )

    monkeypatch.setattr(web_server, "_auto_end_expired_turn_response", fake_auto_end)

    try:
        response = await client.post(
            "/api/battle/end-turn",
            json={"match_id": "m-lock", "user_id": 101, "client_action_id": "expired-1"},
        )
        body = await response.json()

        assert response.status == 409
        assert body["error"] == "turn_expired"
        assert engine.end_turn_calls == 0
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-lock", None)
        web_server.MATCH_LOCKS.pop("m-lock", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_expired_turn_end_response_does_not_deadlock(monkeypatch):
    engine = EndTurnEngine(expired=True)
    client = await _battle_client(monkeypatch, engine, match_id="m-expired-real")

    try:
        response = await asyncio.wait_for(
            client.post(
                "/api/battle/end-turn",
                json={
                    "match_id": "m-expired-real",
                    "user_id": 101,
                    "client_action_id": "expired-real-1",
                },
            ),
            timeout=0.5,
        )
        body = await response.json()

        assert response.status == 409
        assert body["error"] == "turn_expired"
        assert engine.end_turn_calls == 1
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-expired-real", None)
        web_server.MATCH_LOCKS.pop("m-expired-real", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_state_request_restores_disconnected_viewer_before_timeout(monkeypatch):
    engine = EndTurnEngine(expired=True)
    client = await _battle_client(monkeypatch, engine, match_id="m-state-reconnect")
    web_server.MATCH_DISCONNECT_STATES[("m-state-reconnect", 101)] = {
        "disconnected_at": time.time(),
        "timed_out_turns": 0,
        "takeover_started": False,
    }
    web_server.MATCH_SESSIONS.pop("m-state-reconnect", None)

    try:
        response = await client.get("/api/battle/state?match_id=m-state-reconnect&user_id=101")
        body = await response.json()

        assert response.status == 200
        assert body["current_player_id"] == 101
        assert engine.end_turn_calls == 0
        assert ("m-state-reconnect", 101) not in web_server.MATCH_DISCONNECT_STATES
        assert engine.activity == [101]
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-state-reconnect", None)
        web_server.MATCH_LOCKS.pop("m-state-reconnect", None)
        web_server.MATCH_DISCONNECT_STATES.pop(("m-state-reconnect", 101), None)
        web_server.MATCH_SESSIONS.pop("m-state-reconnect", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_active_returns_redirect_for_in_memory_pve_match(monkeypatch):
    engine = EndTurnEngine()
    engine.p2_state = SimpleNamespace(user_id=-900000001, is_bot=True, replacement_status=ReplacementStatus.ACTIVE)
    engine.is_bot_match = True
    engine.game_mode = "classic"
    client = await _battle_client(monkeypatch, engine, match_id="m-active-pve")

    try:
        response = await client.get("/api/battle/active?user_id=101")
        body = await response.json()

        assert response.status == 200
        assert body["active"] is True
        assert body["match_id"] == "m-active-pve"
        assert body["game_mode"] == "classic"
        assert body["redirect_url"] == "/arena?id=m-active-pve"
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-active-pve", None)
        web_server.MATCH_LOCKS.pop("m-active-pve", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_active_ignores_surrendered_in_memory_match(monkeypatch):
    engine = EndTurnEngine()
    engine.p1_state.replacement_status = ReplacementStatus.SURRENDERED
    client = await _battle_client(monkeypatch, engine, match_id="m-active-surrendered")

    try:
        response = await client.get("/api/battle/active?user_id=101")
        body = await response.json()

        assert response.status == 200
        assert body == {"active": False}
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-active-surrendered", None)
        web_server.MATCH_LOCKS.pop("m-active-surrendered", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_active_does_not_fallback_to_matchmaker_for_surrendered_player(monkeypatch):
    class FoundMatchmaker:
        async def get_active_match_for_user(self, user_id):
            return {
                "status": "found",
                "match_id": "m-active-surrendered-fallback",
                "user_id": 101,
                "opponent_id": 202,
                "player_ids": [101, 202],
                "game_mode": "classic",
            }

    engine = EndTurnEngine()
    engine.p1_state.replacement_status = ReplacementStatus.SURRENDERED
    client = await _battle_client(monkeypatch, engine, match_id="m-active-surrendered-fallback")
    client.server.app["matchmaker"] = FoundMatchmaker()

    try:
        response = await client.get("/api/battle/active?user_id=101")
        body = await response.json()

        assert response.status == 200
        assert body == {"active": False}
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-active-surrendered-fallback", None)
        web_server.MATCH_LOCKS.pop("m-active-surrendered-fallback", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_match_find_returns_existing_active_match_without_requeue(monkeypatch):
    db = MatchFindDB()
    engine = EndTurnEngine()
    client = await _match_find_client(monkeypatch, db, engine, match_id="m-active-matchfind")

    try:
        response = await client.post(
            "/api/match/find",
            json={"user_id": 101, "selected_deck_id": 1, "mode": "classic"},
        )
        body = await response.json()

        assert response.status == 200
        assert body["status"] == "found"
        assert body["match_id"] == "m-active-matchfind"
        assert body["active_existing_match"] is True
        assert body["redirect_url"] == "/arena?id=m-active-matchfind"
        assert db.get_player_deck_max_level_calls == 0
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-active-matchfind", None)
        web_server.MATCH_LOCKS.pop("m-active-matchfind", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_surrendered_player_cannot_submit_http_actions(monkeypatch):
    engine = EndTurnEngine()
    engine.p1_state.replacement_status = ReplacementStatus.SURRENDERED
    client = await _battle_client(monkeypatch, engine, match_id="m-action-surrendered")

    try:
        response = await client.post(
            "/api/battle/end-turn",
            json={"match_id": "m-action-surrendered", "user_id": 101, "client_action_id": "surrendered-action-1"},
        )
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "player_replaced"
        assert engine.end_turn_calls == 0
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-action-surrendered", None)
        web_server.MATCH_LOCKS.pop("m-action-surrendered", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_player_action_is_blocked_until_both_pvp_clients_are_ready(monkeypatch):
    engine = EndTurnEngine()
    engine.waiting_for_players = True
    client = await _battle_client(monkeypatch, engine, match_id="m-ready-gate")

    try:
        body = {"match_id": "m-ready-gate", "user_id": 101, "client_action_id": "ready-gate-1"}
        blocked = await client.post("/api/battle/end-turn", json=body)
        blocked_body = await blocked.json()

        engine.waiting_for_players = False
        allowed = await client.post("/api/battle/end-turn", json=body)
        allowed_body = await allowed.json()

        assert blocked.status == 409
        assert blocked_body["error"] == "match_not_ready"
        assert blocked_body["state"]["match_status"] == "waiting_for_players"
        assert allowed.status == 200
        assert allowed_body["result"]["success"] is True
        assert engine.end_turn_calls == 1
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-ready-gate", None)
        web_server.MATCH_LOCKS.pop("m-ready-gate", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_action_errors_do_not_expose_exception_details(monkeypatch):
    engine = FailingEndTurnEngine()
    client = await _battle_client(monkeypatch, engine, match_id="m-safe-error")

    try:
        response = await client.post(
            "/api/battle/end-turn",
            json={"match_id": "m-safe-error", "user_id": 101, "client_action_id": "safe-error-1"},
        )
        body = await response.json()
        serialized = json.dumps(body)

        assert response.status == 400
        assert body == {"error": "turn_end_failed", "message": "Turn end failed"}
        assert "details" not in body
        assert "secret battle shard" not in serialized
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-safe-error", None)
        web_server.MATCH_LOCKS.pop("m-safe-error", None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_surrender_penalty_helper_is_idempotent():
    db = SurrenderDB()
    engine = SurrenderEngine()

    first = await web_server._apply_surrender_penalty_once({"db": db, "match_game_modes": {}}, "m-surrender", engine, 101)
    second = await web_server._apply_surrender_penalty_once({"db": db, "match_game_modes": {}}, "m-surrender", engine, 101)

    assert first["success"] is True
    assert second["success"] is True
    assert second["already_processed"] is True
    assert first["trophy_penalty"] == second["trophy_penalty"]
    assert first["new_trophies"] == second["new_trophies"]
    assert db.trophy_updates == 1


@pytest.mark.asyncio(loop_scope="function")
async def test_surrender_penalty_helper_uses_db_idempotency_under_concurrency():
    db = IdempotentSurrenderDB()
    engine = SurrenderEngine()

    first, second = await asyncio.gather(
        web_server._apply_surrender_penalty_once({"db": db, "match_game_modes": {}}, "m-surrender-race", engine, 101),
        web_server._apply_surrender_penalty_once({"db": db, "match_game_modes": {}}, "m-surrender-race", engine, 101),
    )

    assert first["success"] is True
    assert second["success"] is True
    assert sum(0 if result.get("already_processed") else 1 for result in (first, second)) == 1
    assert db.trophy_updates == 1
    assert db.apply_calls == 2


@pytest.mark.asyncio(loop_scope="function")
async def test_late_surrender_after_terminal_match_does_not_apply_penalty(monkeypatch):
    db = SurrenderDB()
    engine = SurrenderEngine()
    engine.is_ended = True
    engine.battle_end_processed = True
    engine.rewards_granted = True
    client = await _battle_client(monkeypatch, engine, match_id="m-late-surrender")
    client.server.app["db"] = db

    try:
        response = await client.post(
            "/api/matches/m-late-surrender/surrender",
            json={"user_id": 101, "client_action_id": "late-surrender-1"},
        )
        body = await response.json()

        assert response.status == 200
        assert body["game_over"] is True
        assert body["already_ended"] is True
        assert db.trophy_updates == 0
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-late-surrender", None)
        web_server.MATCH_LOCKS.pop("m-late-surrender", None)
        web_server.ACTION_RESULT_CACHE.clear()
        web_server.ENDED_MATCH_IDS.discard("m-late-surrender")
        web_server.ENDED_MATCH_TIMES.pop("m-late-surrender", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_surrender_retry_after_engine_cleanup_returns_finished_payload(monkeypatch):
    engine = EndTurnEngine()
    client = await _battle_client(monkeypatch, engine, match_id="m-surrender-cleaned")
    web_server._mark_match_ended("m-surrender-cleaned")
    web_server.ACTIVE_MATCHES.pop("m-surrender-cleaned", None)
    client.server.app["active_matches"].pop("m-surrender-cleaned", None)

    try:
        response = await client.post(
            "/api/matches/m-surrender-cleaned/surrender",
            json={"user_id": 101, "client_action_id": "surrender-cleaned-1"},
        )
        body = await response.json()

        assert response.status == 200
        assert body["error"] == "game_already_ended"
        assert body["already_ended"] is True
    finally:
        await client.close()
        web_server.MATCH_LOCKS.pop("m-surrender-cleaned", None)
        web_server.ACTION_RESULT_CACHE.clear()
        web_server.ENDED_MATCH_IDS.discard("m-surrender-cleaned")
        web_server.ENDED_MATCH_TIMES.pop("m-surrender-cleaned", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_human_surrender_in_pve_match_terminates_and_stops_active_redirect(monkeypatch):
    db = SurrenderDB()
    engine = PveSurrenderEngine()
    client = await _battle_client(monkeypatch, engine, match_id="m-pve-surrender")
    client.server.app["db"] = db

    try:
        response = await client.post(
            "/api/matches/m-pve-surrender/surrender",
            json={"user_id": 101, "client_action_id": "pve-surrender-1"},
        )
        body = await response.json()
        active_response = await client.get("/api/battle/active?user_id=101")
        active_body = await active_response.json()

        assert response.status == 200
        assert body["success"] is True
        assert body["game_over"] is True
        assert body["reason"] == "human_surrendered_pve"
        assert "m-pve-surrender" not in web_server.ACTIVE_MATCHES
        assert "m-pve-surrender" not in client.server.app["active_matches"]
        assert "m-pve-surrender" in web_server.ENDED_MATCH_IDS
        assert active_response.status == 200
        assert active_body == {"active": False}
        assert db.trophy_updates == 1
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop("m-pve-surrender", None)
        web_server.MATCH_LOCKS.pop("m-pve-surrender", None)
        web_server.ACTION_RESULT_CACHE.clear()
        web_server.ENDED_MATCH_IDS.discard("m-pve-surrender")
        web_server.ENDED_MATCH_TIMES.pop("m-pve-surrender", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_matchmaker_finds_found_match_for_startup_reconnect():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    first = await mm.find_match(11, 500, 1, selected_deck_id=31, game_mode="classic")
    second = await mm.find_match(22, 500, 1, selected_deck_id=42, game_mode="classic")

    first_resume = await mm.get_active_match_for_user(11)
    second_resume = await mm.get_active_match_for_user(22)

    assert second["status"] == "found"
    assert first_resume["status"] == "found"
    assert second_resume["status"] == "found"
    assert first_resume["match_id"] == second["match_id"]
    assert second_resume["match_id"] == second["match_id"]
    assert first_resume["player_ids"] == second_resume["player_ids"] == [22, 11]
    assert await mm.get_active_match_for_user(999) == {"status": "not_found", "user_id": 999}


@pytest.mark.asyncio(loop_scope="function")
async def test_pve_payload_randomizes_starter_without_swapping_human_and_bot(monkeypatch):
    monkeypatch.setattr("infrastructure.matchmaking.random.choice", lambda ids: ids[1])
    mm = Matchmaker(FakeDB(), ReadyBotFactory())

    result = await mm._create_bot_match(
        user_id=10,
        trophies=100,
        user_max_level=1,
        selected_deck_id=1,
        game_mode="classic",
    )

    assert result["status"] == "found"
    assert result["player_ids"][0] == 10
    assert result["opponent_id"] == result["player_ids"][1]
    assert result["starting_player_id"] == result["player_ids"][1]


class FakeExtraIDDB:
    def __init__(self, expected_session_id: str):
        self.expected_session_id = expected_session_id

    async def verify_session(self, session_uuid, token: str):
        assert str(session_uuid) == self.expected_session_id
        return {"session_id": session_uuid}


@pytest.mark.asyncio(loop_scope="function")
async def test_socket_auth_accepts_jwt_extraid_token():
    session_id = str(uuid.uuid4())
    now = int(time.time())
    token = jwt.encode(
        {
            "user_id": 777,
            "session_id": session_id,
            "iat": now,
            "exp": now + 60,
        },
        web_server.get_settings().jwt_secret,
        algorithm="HS256",
    )

    user_id = await web_server._require_user_id_from_auth_token_str(
        token,
        {"extraid_db": FakeExtraIDDB(session_id), "bot_token": ""},
    )

    assert user_id == 777


class FakeEngine:
    def __init__(self):
        self.statuses = []
        self.current = 42
        self.restored = []
        self.turn = 1
        self.turn_duration = 25
        self.is_ended = False

    def set_player_replacement_status(self, user_id, status):
        self.statuses.append((user_id, status))

    def get_current_player_id(self):
        return self.current

    def get_turn_time_remaining(self):
        return self.turn_duration

    def get_player_replacement_status(self, user_id):
        return ReplacementStatus.ACTIVE

    def is_bot(self, user_id):
        return False

    def restore_player_control(self, user_id):
        self.restored.append(user_id)
        return True


class SurrenderedReconnectEngine(FakeEngine):
    def get_player_replacement_status(self, user_id):
        return ReplacementStatus.SURRENDERED

    def restore_player_control(self, user_id):
        self.restored.append(user_id)
        return False


def test_disconnect_tracks_without_immediate_afk_replacement():
    engine = FakeEngine()
    web_server.ACTIVE_MATCHES["m1"] = engine
    web_server.MATCH_SESSIONS.pop("m1", None)

    try:
        web_server._mark_player_disconnected("m1", 42, engine)

        assert engine.statuses == []
        assert web_server.MATCH_DISCONNECT_STATES[("m1", 42)]["timed_out_turns"] == 0
    finally:
        web_server.ACTIVE_MATCHES.pop("m1", None)
        web_server.MATCH_SESSIONS.pop("m1", None)
        web_server.MATCH_DISCONNECT_STATES.pop(("m1", 42), None)


@pytest.mark.asyncio(loop_scope="function")
async def test_reconnect_restores_afk_and_cancels_pending_takeover():
    engine = FakeEngine()
    web_server.ACTIVE_MATCHES["m1"] = engine
    web_server.MATCH_SESSIONS.pop("m1", None)

    async def sleeper():
        await asyncio.sleep(10)

    pending = asyncio.create_task(sleeper())
    replacement_bot = asyncio.create_task(sleeper())
    web_server.MATCH_DISCONNECT_TASKS[("m1", 42)] = pending
    web_server.MATCH_DISCONNECT_STATES[("m1", 42)] = {"timed_out_turns": 1}
    web_server.BOT_TASKS["m1"] = replacement_bot
    web_server.BOT_TASK_KEYS["m1"] = ("42", 1)

    try:
        web_server._register_session("m1", 42, "sid-new")
        await asyncio.sleep(0)
        assert pending.cancelled()
        assert replacement_bot.cancelled()
        assert engine.restored == [42]
        assert ("m1", 42) not in web_server.MATCH_DISCONNECT_STATES
    finally:
        web_server.ACTIVE_MATCHES.pop("m1", None)
        web_server.MATCH_SESSIONS.pop("m1", None)
        task = web_server.MATCH_DISCONNECT_TASKS.pop(("m1", 42), None)
        if task:
            task.cancel()
        bot_task = web_server.BOT_TASKS.pop("m1", None)
        if bot_task:
            bot_task.cancel()
        web_server.BOT_TASK_KEYS.pop("m1", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_surrendered_reconnect_does_not_cancel_replacement_bot():
    engine = SurrenderedReconnectEngine()
    web_server.ACTIVE_MATCHES["m-surrender-reconnect"] = engine
    web_server.MATCH_SESSIONS.pop("m-surrender-reconnect", None)

    async def sleeper():
        await asyncio.sleep(10)

    replacement_bot = asyncio.create_task(sleeper())
    web_server.MATCH_DISCONNECT_STATES[("m-surrender-reconnect", 42)] = {"timed_out_turns": 1}
    web_server.BOT_TASKS["m-surrender-reconnect"] = replacement_bot
    web_server.BOT_TASK_KEYS["m-surrender-reconnect"] = ("42", 1)

    try:
        web_server._register_session("m-surrender-reconnect", 42, "sid-new")
        await asyncio.sleep(0)

        assert web_server.BOT_TASKS["m-surrender-reconnect"] is replacement_bot
        assert web_server.BOT_TASK_KEYS["m-surrender-reconnect"] == ("42", 1)
        assert not replacement_bot.cancelled()
        assert engine.restored == [42]
        assert web_server.MATCH_SESSIONS["m-surrender-reconnect"][42] == {"sid-new"}
    finally:
        web_server.ACTIVE_MATCHES.pop("m-surrender-reconnect", None)
        web_server.MATCH_SESSIONS.pop("m-surrender-reconnect", None)
        web_server.MATCH_DISCONNECT_STATES.pop(("m-surrender-reconnect", 42), None)
        bot_task = web_server.BOT_TASKS.pop("m-surrender-reconnect", None)
        if bot_task:
            bot_task.cancel()
        web_server.BOT_TASK_KEYS.pop("m-surrender-reconnect", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_surrendered_state_activity_does_not_cancel_replacement_bot():
    engine = SurrenderedReconnectEngine()
    web_server.ACTIVE_MATCHES["m-surrender-state"] = engine

    async def sleeper():
        await asyncio.sleep(10)

    replacement_bot = asyncio.create_task(sleeper())
    web_server.MATCH_DISCONNECT_STATES[("m-surrender-state", 42)] = {"timed_out_turns": 1}
    web_server.BOT_TASKS["m-surrender-state"] = replacement_bot
    web_server.BOT_TASK_KEYS["m-surrender-state"] = ("42", 1)

    try:
        web_server._mark_user_activity_for_match("m-surrender-state", 42, engine)
        await asyncio.sleep(0)

        assert web_server.BOT_TASKS["m-surrender-state"] is replacement_bot
        assert web_server.BOT_TASK_KEYS["m-surrender-state"] == ("42", 1)
        assert not replacement_bot.cancelled()
        assert engine.restored == [42]
    finally:
        web_server.ACTIVE_MATCHES.pop("m-surrender-state", None)
        web_server.MATCH_DISCONNECT_STATES.pop(("m-surrender-state", 42), None)
        bot_task = web_server.BOT_TASKS.pop("m-surrender-state", None)
        if bot_task:
            bot_task.cancel()
        web_server.BOT_TASK_KEYS.pop("m-surrender-state", None)


def _engine_with_state() -> BattleEngine:
    engine = BattleEngine(game_mode="classic")
    state = create_classic_game_state(101, 202, [_hero(1)], [_hero(2)])
    engine._arena = ArenaEnvironment(state, classic_params=engine.mode_config.classic)
    engine.match_id = "unit"
    engine.current_player_id = state.current_turn_owner_id
    engine.turn = state.turn_number
    return engine


def test_battle_engine_timeout_restore_and_serialized_replacement_status():
    engine = _engine_with_state()

    engine.mark_timeout(101)
    assert engine.get_player_replacement_status(101) == ReplacementStatus.ACTIVE
    engine.mark_timeout(101)
    assert engine.get_player_replacement_status(101) == ReplacementStatus.AFK

    restored = engine.restore_player_control(101)

    assert restored is True
    assert engine.get_player_replacement_status(101) == ReplacementStatus.ACTIVE
    assert engine.p1_consecutive_timeouts == 0
    state = engine.get_full_state(viewer_id=101)
    assert state["player"]["replacement_status"] == "active"


def test_battle_engine_surrendered_viewer_cannot_act_even_on_own_turn():
    engine = _engine_with_state()

    engine.mark_surrender(101)
    state = engine.get_full_state(viewer_id=101)

    assert state["player"]["replacement_status"] == "surrendered"
    assert state["is_my_turn"] is False
    assert state["legal_actions"] == []


def test_battle_engine_waits_for_both_human_clients_before_actions_and_timer():
    engine = _engine_with_state()
    engine.game_mode = "friendly"
    engine.is_bot_match = False
    engine.turn_start_time = time.time() - 999

    initial = engine.get_full_state(viewer_id=101)
    first_ready = engine.mark_client_ready(101)
    after_first = engine.get_full_state(viewer_id=101)
    second_ready = engine.mark_client_ready(202)
    after_second = engine.get_full_state(viewer_id=101)

    assert initial["match_status"] == "waiting_for_players"
    assert initial["is_my_turn"] is False
    assert initial["legal_actions"] == []
    assert initial["turn_time_remaining"] == engine.turn_duration
    assert engine.is_turn_expired() is False
    assert first_ready["all_ready"] is False
    assert after_first["match_status"] == "waiting_for_players"
    assert second_ready["all_ready"] is True
    assert engine.client_ready is True
    assert after_second["match_status"] == "active"
    assert after_second["is_my_turn"] is True
    assert after_second["turn_time_remaining"] > engine.turn_duration - 1


def test_disconnect_takeover_window_scales_with_turn_duration():
    engine = FakeEngine()
    engine.turn_duration = 25
    assert web_server._disconnect_takeover_window(engine) == 5.0
    engine.turn_duration = 10
    assert web_server._disconnect_takeover_window(engine) == 2.0
    engine.turn_duration = 5
    assert web_server._disconnect_takeover_window(engine) == 1.0


@pytest.mark.asyncio(loop_scope="function")
async def test_second_disconnected_timeout_switches_player_to_afk_takeover(monkeypatch):
    class TimeoutEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.ended_turns = []
            self.status = ReplacementStatus.ACTIVE

        def is_turn_expired(self):
            return True

        def is_current_player_bot(self):
            return False

        def get_player_replacement_status(self, user_id):
            return self.status

        def set_player_replacement_status(self, user_id, status):
            super().set_player_replacement_status(user_id, status)
            self.status = status

        def end_turn(self, user_id):
            self.ended_turns.append(user_id)
            self.current = 202
            return {"success": True}

        def get_full_state(self, viewer_id=None):
            return {"match_id": "m-timeout", "viewer_id": viewer_id}

    check_calls = []

    async def fake_check_and_run_bot(match_id, active_matches):
        check_calls.append(match_id)

    engine = TimeoutEngine()
    web_server.ACTIVE_MATCHES["m-timeout"] = engine
    web_server.MATCH_DISCONNECT_STATES[("m-timeout", 42)] = {
        "disconnected_at": time.time(),
        "timed_out_turns": 1,
        "takeover_started": False,
    }
    web_server.MATCH_SESSIONS.pop("m-timeout", None)
    monkeypatch.setattr(web_server, "check_and_run_bot", fake_check_and_run_bot)

    try:
        handled = await web_server._handle_natural_turn_timeout(
            {"socketio": None},
            "m-timeout",
            engine,
        )

        assert handled is True
        assert engine.status == ReplacementStatus.AFK
        assert engine.statuses == [(42, ReplacementStatus.AFK)]
        assert engine.ended_turns == []
        assert web_server.MATCH_DISCONNECT_STATES[("m-timeout", 42)]["timed_out_turns"] == 2
        assert web_server.MATCH_DISCONNECT_STATES[("m-timeout", 42)]["takeover_started"] is True
        assert check_calls == ["m-timeout"]
    finally:
        web_server.ACTIVE_MATCHES.pop("m-timeout", None)
        web_server.MATCH_DISCONNECT_STATES.pop(("m-timeout", 42), None)
        web_server.MATCH_SESSIONS.pop("m-timeout", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_natural_timeout_does_not_run_before_pvp_clients_are_ready():
    engine = EndTurnEngine(expired=True)
    engine.waiting_for_players = True
    engine.current_player_id = 101

    handled = await web_server._handle_natural_turn_timeout(
        {"socketio": None},
        "m-timeout-waiting",
        engine,
    )

    assert handled is False
    assert engine.end_turn_calls == 0


def test_finished_match_action_payload_is_terminal_not_missing():
    web_server.ENDED_MATCH_IDS.add("m-ended")

    try:
        payload = web_server._build_finished_match_action_payload("m-ended")

        assert payload["match_id"] == "m-ended"
        assert payload["game_over"] is True
        assert payload["already_ended"] is True
        assert payload["error"] == "game_already_ended"
        assert payload["success"] is True
    finally:
        web_server.ENDED_MATCH_IDS.discard("m-ended")


def test_turn_end_handler_assigns_client_action_id_before_use():
    source = Path("web/server.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    handler = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "battle_turn_end_handler"
    )
    assigned_at = None
    first_load_at = None
    for node in ast.walk(handler):
        if isinstance(node, ast.Name) and node.id == "client_action_id":
            if isinstance(node.ctx, ast.Store):
                assigned_at = min(assigned_at or node.lineno, node.lineno)
            elif isinstance(node.ctx, ast.Load):
                first_load_at = min(first_load_at or node.lineno, node.lineno)

    assert assigned_at is not None
    assert first_load_at is not None
    assert assigned_at < first_load_at


def test_action_timeout_guard_is_inside_match_lock_for_player_actions():
    source = Path("web/server.py").read_text(encoding="utf-8")
    for name in ("battle_play_card_handler", "battle_attack_handler", "battle_turn_end_handler"):
        block = source.split(f"async def {name}", 1)[1].split("\n    async def ", 1)[0]
        lock_pos = block.index("async with lock:")
        timeout_pos = block.index("_auto_end_expired_turn_response")
        activity_pos = block.index("_mark_user_activity_for_match")

        assert lock_pos < timeout_pos < activity_pos


def test_action_id_cache_is_rechecked_inside_match_lock_for_mutations():
    source = Path("web/server.py").read_text(encoding="utf-8")
    for name in (
        "battle_play_card_handler",
        "battle_attack_handler",
        "battle_surrender_handler",
        "battle_turn_end_handler",
    ):
        block = source.split(f"async def {name}", 1)[1].split("\n    async def ", 1)[0]
        lock_pos = block.index("async with lock:")
        terminal_pos = block.index("_terminal_action_payload_if_needed")
        locked_prefix = block[lock_pos:terminal_pos]

        assert "_action_cache_get(match_id, user_id_int, client_action_id)" in locked_prefix


def test_lazy_battle_init_uses_per_match_lock_and_double_check():
    source = Path("web/server.py").read_text(encoding="utf-8")

    assert "MATCH_INIT_LOCKS" in source
    assert "_get_match_init_lock" in source
    assert "async with init_lock:" in source
    assert "init_lock_held=True" in source
    assert 'request.app["active_matches"].get(match_id)' in source


def test_friendly_battle_rehydrate_uses_per_match_init_lock():
    source = Path("web/server.py").read_text(encoding="utf-8")
    block = source.split("async def _ensure_friendly_match_engine", 1)[1].split(
        "\n    app[\"ensure_friendly_match_engine\"]", 1
    )[0]

    assert "init_lock_held: bool = False" in block
    assert "_get_match_init_lock(match_id)" in block
    assert "async with init_lock:" in block
    assert "init_lock_held=True" in block


def test_card_cache_is_int_keyed_dict_payload_for_battle_engine():
    source = Path("web/server.py").read_text(encoding="utf-8")
    load_cache_block = source.split("async def _load_card_cache", 1)[1].split(
        "\n    def _normalize_deck_with_cache", 1
    )[0]

    assert "dict[int, dict" in load_cache_block
    assert "cache[int(card_obj.id)]" in load_cache_block
    assert "card_obj.to_dict()" in load_cache_block


def test_reward_enabled_human_battle_startup_deck_failures_fail_closed():
    source = Path("web/server.py").read_text(encoding="utf-8")
    block = source.split("async def _prepare_and_cache_engine", 1)[1].split(
        "\n    async def _ensure_active_engine_for_request", 1
    )[0]

    assert 'return _fail_battle_init("deck_load_timeout")' in block
    assert 'return _fail_battle_init("deck_unavailable")' in block
    assert 'return _fail_battle_init("deck_incomplete")' in block
    assert "allow_human_cache_fallback = not mode_config.rewards.enabled" in block
    assert "allow_cache_fallback=allow_human_cache_fallback" in block


def test_turn_timer_uses_monotonic_for_remaining_duration(monkeypatch):
    engine = BattleEngine.__new__(BattleEngine)
    engine.turn_duration = 25
    engine.turn_start_time = 1_000_000.0
    engine.turn_start_monotonic = 10.0
    engine.client_ready = True
    engine.is_bot_match = True
    engine.is_ended = False
    engine._arena = SimpleNamespace(
        state=SimpleNamespace(p1=SimpleNamespace(is_bot=False), p2=SimpleNamespace(is_bot=True))
    )

    monkeypatch.setattr(battle_engine_module.time, "time", lambda: 2_000_000.0)
    monkeypatch.setattr(battle_engine_module.time, "monotonic", lambda: 20.0)

    assert engine.get_turn_time_remaining() == 15


def test_battle_state_events_are_not_emitted_room_wide():
    source = Path("web/server.py").read_text(encoding="utf-8")
    for marker in ("room=match_id", "room = match_id"):
        start = 0
        while True:
            pos = source.find(marker, start)
            if pos == -1:
                break
            emit_call = source[max(0, pos - 500):pos + 100]
            assert '"state_changed"' not in emit_call
            assert '"turn_end"' not in emit_call
            start = pos + len(marker)


def test_match_timer_checker_uses_lightweight_expiration_check():
    source = Path("web/server.py").read_text(encoding="utf-8")
    checker_block = source.split("async def match_timer_checker", 1)[1].split(
        "async def _announcement_expiry_loop",
        1,
    )[0]

    assert "is_turn_expired()" in checker_block
    assert "get_full_state" not in checker_block


def test_execute_action_does_not_build_discarded_full_state_for_emit():
    source = Path("battle_engine.py").read_text(encoding="utf-8")
    execute_block = source.split("def execute_action", 1)[1].split(
        "# =========================================================================\n    # МЕТОДЫ СОВМЕСТИМОСТИ",
        1,
    )[0]

    assert "self.get_full_state()" not in execute_block
    assert '"state_p1"' not in execute_block
    assert '"action": action.to_dict()' in execute_block


def test_pvp_battle_paths_do_not_use_print_or_production_debug_strings():
    server_source = Path("web/server.py").read_text(encoding="utf-8")
    engine_source = Path("battle_engine.py").read_text(encoding="utf-8")
    matchmaking_source = Path("infrastructure/matchmaking.py").read_text(encoding="utf-8")

    assert "print(" not in server_source
    assert "print(" not in engine_source
    assert "DEBUG:" not in matchmaking_source


def test_training_bot_profile_overrides_donor_identity():
    profile = web_server._training_bot_profile_payload()
    bot_info = web_server._decorate_training_bot_info({
        "name": "Donor Player",
        "avatar_url": "/DesignAssets/PlayerCosmetics/Avatars/1.png",
        "trophies": 999,
        "extra_pass": "ultra",
        "cosmetics": {
            "avatar": {"asset_path": "/DesignAssets/PlayerCosmetics/Avatars/2.png"},
            "title": {"name": "Human title", "class": "mythic"},
            "profile_background": {"asset_path": "/DesignAssets/PlayerCosmetics/Background/7.png"},
        },
    })

    assert profile["avatar_url"] == "/DesignAssets/Arena/TrainingModeBotCosmetics/avatar.png"
    assert profile["background_url"] == "/DesignAssets/Arena/TrainingModeBotCosmetics/ProfileBackground.png"
    assert profile["title"] == "extra-lr series"
    assert bot_info["name"] == profile["name"]
    assert bot_info["avatar_url"] == profile["avatar_url"]
    assert bot_info["trophies"] == 0
    assert bot_info["extra_pass"] is None
    assert bot_info["cosmetics"]["avatar"]["asset_path"] == profile["avatar_url"]
    assert bot_info["cosmetics"]["title"]["name"] == "extra-lr series"
    assert bot_info["cosmetics"]["profile_background"]["asset_path"] == profile["background_url"]


def test_training_bot_history_uses_training_cosmetics_without_affecting_other_bot_modes():
    training_row = web_server._decorate_training_history_battle({
        "mode": "training",
        "opponent_is_bot": True,
        "opponent_name": "Bot Factory Donor",
        "opponent_avatar_url": "/DesignAssets/BotFactory/avatar.png",
        "opponent_title": "Factory Title",
        "opponent_title_class": "mythic",
        "opponent_background_url": "/DesignAssets/BotFactory/background.png",
        "opponent_trophies": 9000,
    })
    classic_row = web_server._decorate_training_history_battle({
        "mode": "classic",
        "opponent_is_bot": True,
        "opponent_name": "Classic Bot",
        "opponent_avatar_url": "/DesignAssets/BotFactory/classic.png",
        "opponent_title": "Classic Title",
        "opponent_title_class": "rare",
        "opponent_background_url": "/DesignAssets/BotFactory/classic-bg.png",
        "opponent_trophies": 1234,
    })

    assert training_row["opponent_name"] == web_server.TRAINING_BOT_NAME
    assert training_row["opponent_avatar_url"] == web_server.TRAINING_BOT_AVATAR_URL
    assert training_row["opponent_background_url"] == web_server.TRAINING_BOT_BACKGROUND_URL
    assert training_row["opponent_title"] == "extra-lr series"
    assert training_row["opponent_title_class"] == web_server.TRAINING_BOT_TITLE_CLASS
    assert training_row["opponent_trophies"] == 0
    assert classic_row["opponent_name"] == "Classic Bot"
    assert classic_row["opponent_avatar_url"] == "/DesignAssets/BotFactory/classic.png"
    assert classic_row["opponent_title"] == "Classic Title"


def test_battle_history_stats_ignore_training_and_friendly_results():
    stats = web_server._build_battle_history_stats([
        {"mode": "training", "result": "lose", "trophies_change": 0, "turns_count": 40, "duration_seconds": 400},
        {"mode": "friendly", "result": "win", "trophies_change": 0, "turns_count": 30, "duration_seconds": 300},
        {"mode": "classic", "result": "win", "trophies_change": 24, "turns_count": 10, "duration_seconds": 120},
        {"mode": "extra_arena:sudden_death", "result": "lose", "trophies_change": -8, "turns_count": 14, "duration_seconds": 160},
    ])

    assert stats["total"] == 2
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["draws"] == 0
    assert stats["win_rate"] == 50.0
    assert stats["trophy_delta"] == 16
    assert stats["avg_turns"] == 12.0
    assert stats["avg_duration_seconds"] == 140
    assert stats["favorite_mode"] == "classic"


@pytest.mark.asyncio(loop_scope="function")
async def test_disconnected_takeover_check_runs_before_client_ready():
    engine = FakeEngine()
    engine.client_ready = False
    web_server.MATCH_DISCONNECT_STATES[("m-ready", 42)] = {
        "disconnected_at": time.time(),
        "timed_out_turns": 1,
        "takeover_started": False,
    }
    web_server.MATCH_SESSIONS.pop("m-ready", None)

    try:
        consumed = web_server._schedule_disconnected_takeover_if_needed("m-ready", engine)

        assert consumed is True
        assert ("m-ready", 42) in web_server.MATCH_DISCONNECT_TASKS
    finally:
        task = web_server.MATCH_DISCONNECT_TASKS.pop(("m-ready", 42), None)
        if task:
            task.cancel()
        web_server.MATCH_DISCONNECT_STATES.pop(("m-ready", 42), None)
        web_server.MATCH_SESSIONS.pop("m-ready", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_turn_handoff_schedules_disconnected_takeover(monkeypatch):
    engine = FakeEngine()
    engine.current = 99
    engine.turn = 7
    engine.client_ready = True
    engine._active_matches = web_server.ACTIVE_MATCHES

    def is_bot(user_id):
        return int(user_id) == 99

    async def fake_run_bot_routine(routine_engine, bot_id):
        assert routine_engine is engine
        assert int(bot_id) == 99
        routine_engine.current = 42
        routine_engine.turn += 1

    engine.is_bot = is_bot
    web_server.ACTIVE_MATCHES["m-post-bot"] = engine
    web_server.MATCH_DISCONNECT_STATES[("m-post-bot", 42)] = {
        "disconnected_at": time.time(),
        "timed_out_turns": 1,
        "takeover_started": False,
    }
    web_server.MATCH_SESSIONS.pop("m-post-bot", None)
    web_server.BOT_TASKS.pop("m-post-bot", None)
    web_server.BOT_TASK_KEYS.pop("m-post-bot", None)
    monkeypatch.setattr(web_server, "run_bot_routine", fake_run_bot_routine)

    try:
        web_server._start_guarded_bot_task("m-post-bot", engine, 99)
        await web_server.BOT_TASKS["m-post-bot"]

        assert ("m-post-bot", 42) in web_server.MATCH_DISCONNECT_TASKS
    finally:
        task = web_server.MATCH_DISCONNECT_TASKS.pop(("m-post-bot", 42), None)
        if task:
            task.cancel()
        web_server.ACTIVE_MATCHES.pop("m-post-bot", None)
        web_server.MATCH_DISCONNECT_STATES.pop(("m-post-bot", 42), None)
        web_server.MATCH_SESSIONS.pop("m-post-bot", None)
        web_server.BOT_TASKS.pop("m-post-bot", None)
        web_server.BOT_TASK_KEYS.pop("m-post-bot", None)


class FakeProfileDB:
    async def get_user_profile(self, user_id):
        return {
            "display_name": "Visible Name",
            "custom_nickname": "Custom Nick",
            "nickname": "Nick",
            "name": "Legacy Name",
            "username": "username",
            "equipped_avatar_url": "/avatars/equipped.png",
            "img": "/avatars/telegram.png",
            "trophies": 1234,
            "clan": "Arena Clan",
            "title": "Legacy Title",
        }

    async def get_equipped_title(self, user_id):
        return {"name": "Equipped Title", "class": "epic"}

    async def get_user_cosmetics(self, user_id):
        return {
            "equipped": {
                "profile_background": {"asset_path": "/backgrounds/equipped.png"},
            }
        }

    async def fetchval(self, query, *args):
        assert "extra_pass" in query
        return "ultra"


class FakeProfileDBWithDatetime(FakeProfileDB):
    async def get_user_profile(self, user_id):
        profile = await super().get_user_profile(user_id)
        profile["reg_date"] = datetime(2026, 5, 29, tzinfo=timezone.utc)
        profile["energy_cd"] = datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)
        return profile


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_profile_resolves_equipped_visuals_and_extra_pass():
    profile = await web_server._resolve_battle_profile(FakeProfileDB(), 101)

    assert profile["name"] == "Visible Name"
    assert profile["avatar_url"] == "/avatars/equipped.png"
    assert profile["background_url"] == "/backgrounds/equipped.png"
    assert profile["title"] == "Equipped Title"
    assert profile["title_class"] == "epic"
    assert profile["extra_pass"] == "ultra"
    assert profile["trophies"] == 1234
    assert profile["clan"] == "Arena Clan"


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_profile_is_safe_for_json_responses():
    profile = await web_server._resolve_battle_profile(FakeProfileDBWithDatetime(), 101)

    assert "raw_profile" not in profile
    json.dumps(profile)


class FakeAnonymousProfileDB:
    async def get_user_profile(self, user_id):
        return {
            "user_id": user_id,
            "first_name": "GuestHero",
            "username": None,
            "custom_nickname": None,
            "nickname": None,
            "trophies": 42,
        }

    async def get_equipped_title(self, user_id):
        return None

    async def get_user_cosmetics(self, user_id):
        return {"equipped": {}}

    async def fetchval(self, query, *args):
        return None


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_profile_uses_anonymous_first_name_before_id_fallback():
    profile = await web_server._resolve_battle_profile(FakeAnonymousProfileDB(), 777)

    assert profile["name"] == "GuestHero"


class FakeSIO:
    def __init__(self):
        self.emits = []

    async def emit(self, event, payload, **kwargs):
        self.emits.append((event, payload, kwargs))


class PersonalizedEmitEngine:
    def get_full_state(self, viewer_id=None):
        return {
            "match_id": "m-emit",
            "viewer_id": viewer_id,
            "legal_actions": [{"viewer": viewer_id}],
        }


class TalkieEmitEngine:
    def __init__(self):
        self.disabled_users = set()

    def should_deliver_talkie_to(self, user_id):
        return int(user_id) not in self.disabled_users


class TalkieHandlerEngine(TalkieEmitEngine):
    def __init__(self):
        super().__init__()
        self.register_calls = []
        self.settings_calls = []

    def register_talkie(self, user_id, talkie_id):
        self.register_calls.append((user_id, talkie_id))
        return {
            "success": True,
            "event": {
                "event_id": "talkie-1",
                "match_id": "m-talkie-handler",
                "sender_id": int(user_id),
                "turn": 3,
                "talkie_id": str(talkie_id),
                "sound": "happy",
                "remaining": 0,
            },
            "remaining": 0,
        }

    def set_talkie_enabled(self, user_id, enabled):
        self.settings_calls.append((user_id, enabled))
        if enabled:
            self.disabled_users.discard(int(user_id))
        else:
            self.disabled_users.add(int(user_id))
        return {"success": True, "user_id": int(user_id), "enabled": bool(enabled)}


@pytest.mark.asyncio(loop_scope="function")
async def test_personalized_match_state_emit_uses_viewer_specific_sid_payloads():
    fake_sio = FakeSIO()
    engine = PersonalizedEmitEngine()
    web_server.SID_TO_MATCH["sid-p1"] = {"match_id": "m-emit", "user_id": 101}
    web_server.SID_TO_MATCH["sid-p2"] = {"match_id": "m-emit", "user_id": 202}

    try:
        await web_server._emit_personalized_match_state(
            fake_sio,
            "m-emit",
            "state_changed",
            {"data": {"action": "unit"}},
            engine=engine,
        )

        assert [item[2].get("to") for item in fake_sio.emits] == ["sid-p1", "sid-p2"]
        assert all("room" not in item[2] for item in fake_sio.emits)
        assert [item[1]["state"]["viewer_id"] for item in fake_sio.emits] == [101, 202]
    finally:
        web_server.SID_TO_MATCH.pop("sid-p1", None)
        web_server.SID_TO_MATCH.pop("sid-p2", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_talkie_emit_delivers_to_sender_and_opponent_but_filters_muted_users():
    fake_sio = FakeSIO()
    engine = TalkieEmitEngine()
    engine.disabled_users.add(202)
    event = {
        "event_id": "talkie-1",
        "match_id": "m-talkie",
        "sender_id": 101,
        "turn": 5,
        "talkie_id": "5",
        "sound": "happy",
        "remaining": 0,
    }
    web_server.SID_TO_MATCH["sid-sender"] = {"match_id": "m-talkie", "user_id": 101}
    web_server.SID_TO_MATCH["sid-opponent"] = {"match_id": "m-talkie", "user_id": 202}
    web_server.SID_TO_MATCH["sid-other"] = {"match_id": "m-other", "user_id": 303}

    try:
        await web_server._emit_talkie_event(fake_sio, "m-talkie", engine, event)

        assert fake_sio.emits == [("battle_talkie", event, {"to": "sid-sender"})]
    finally:
        web_server.SID_TO_MATCH.pop("sid-sender", None)
        web_server.SID_TO_MATCH.pop("sid-opponent", None)
        web_server.SID_TO_MATCH.pop("sid-other", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_talkie_uses_sid_session_acknowledges_and_delivers_to_both_players(monkeypatch):
    fake_sio = FakeSIO()
    engine = TalkieHandlerEngine()
    web_server.SID_TO_MATCH["sid-sender"] = {"match_id": "m-talkie-handler", "user_id": 101}
    web_server.SID_TO_MATCH["sid-opponent"] = {"match_id": "m-talkie-handler", "user_id": 202}
    web_server.ACTIVE_MATCHES["m-talkie-handler"] = engine
    monkeypatch.setattr(web_server, "sio", fake_sio)

    try:
        await web_server.battle_talkie(
            "sid-sender",
            {"match_id": "wrong-match", "user_id": 202, "talkie_id": "5"},
        )

        assert engine.register_calls == [(101, "5")]
        assert fake_sio.emits[0] == (
            "battle_talkie_ack",
            {
                "success": True,
                "match_id": "m-talkie-handler",
                "event_id": "talkie-1",
                "remaining": 0,
            },
            {"to": "sid-sender"},
        )
        assert fake_sio.emits[1:] == [
            ("battle_talkie", engine.register_talkie(101, "5")["event"], {"to": "sid-sender"}),
            ("battle_talkie", engine.register_talkie(101, "5")["event"], {"to": "sid-opponent"}),
        ]
    finally:
        web_server.SID_TO_MATCH.pop("sid-sender", None)
        web_server.SID_TO_MATCH.pop("sid-opponent", None)
        web_server.ACTIVE_MATCHES.pop("m-talkie-handler", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_talkie_settings_uses_sid_session_and_acknowledges(monkeypatch):
    fake_sio = FakeSIO()
    engine = TalkieHandlerEngine()
    web_server.SID_TO_MATCH["sid-settings"] = {"match_id": "m-talkie-handler", "user_id": 101}
    web_server.ACTIVE_MATCHES["m-talkie-handler"] = engine
    monkeypatch.setattr(web_server, "sio", fake_sio)

    try:
        await web_server.battle_talkie_settings(
            "sid-settings",
            {"match_id": "wrong-match", "user_id": 202, "enabled": False},
        )

        assert engine.settings_calls == [(101, False)]
        assert fake_sio.emits == [
            (
                "battle_talkie_settings_ack",
                {"success": True, "match_id": "m-talkie-handler", "enabled": False},
                {"to": "sid-settings"},
            )
        ]
    finally:
        web_server.SID_TO_MATCH.pop("sid-settings", None)
        web_server.ACTIVE_MATCHES.pop("m-talkie-handler", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_socket_talkie_errors_do_not_expose_exception_details(monkeypatch):
    fake_sio = FakeSIO()
    engine = TalkieHandlerEngine()

    def fail_register_talkie(user_id, talkie_id):
        raise RuntimeError("secret socket room token")

    engine.register_talkie = fail_register_talkie
    web_server.SID_TO_MATCH["sid-safe-talkie"] = {"match_id": "m-safe-talkie", "user_id": 101}
    web_server.ACTIVE_MATCHES["m-safe-talkie"] = engine
    monkeypatch.setattr(web_server, "sio", fake_sio)

    try:
        await web_server.battle_talkie("sid-safe-talkie", {"talkie_id": "5"})

        assert fake_sio.emits == [
            (
                "battle_talkie_ack",
                {"success": False, "error": "internal_server_error"},
                {"to": "sid-safe-talkie"},
            )
        ]
        assert "secret socket room token" not in json.dumps(fake_sio.emits)
    finally:
        web_server.SID_TO_MATCH.pop("sid-safe-talkie", None)
        web_server.ACTIVE_MATCHES.pop("m-safe-talkie", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_leave_match_unregisters_sid_and_session(monkeypatch):
    left = []

    async def fake_leave_room(sid, room):
        left.append((sid, room))

    monkeypatch.setattr(web_server.sio, "leave_room", fake_leave_room)
    web_server.SID_TO_MATCH["sid-leave"] = {"match_id": "m-leave", "user_id": 101}
    web_server.MATCH_SESSIONS["m-leave"] = {101: {"sid-leave"}}

    await web_server.leave_match("sid-leave", {"match_id": "m-leave"})

    assert left == [("sid-leave", "m-leave")]
    assert "sid-leave" not in web_server.SID_TO_MATCH
    assert "m-leave" not in web_server.MATCH_SESSIONS


def _battle_transaction_payload(match_id="m-econ", winner_user_id=101):
    return {
        "match_id": match_id,
        "p1_user_id": 101,
        "p2_user_id": 202,
        "winner_user_id": winner_user_id,
        "loser_user_id": 202 if winner_user_id == 101 else 101,
        "p1_hero_id": 1,
        "p2_hero_id": 2,
        "p1_deck": [11, 12],
        "p2_deck": [21, 22],
        "surrender": False,
        "afk": False,
        "match_type": "classic",
        "game_mode": "classic",
        "duration_seconds": 30,
        "turns_count": 2,
        "p1_trophy_change": 8,
        "p2_trophy_change": -4,
        "p1_coins_earned": 30,
        "p2_coins_earned": 0,
        "p1_cards_played": 1,
        "p2_cards_played": 1,
        "metadata": {"unit": True},
        "battle_result": {
            "winner_score": 0,
            "loser_score": 0,
            "match_duration": 30,
        },
        "rewards": {
            101: {"trophies": 8, "coins": 30, "stars": 3, "wins_for_case": 5, "old_league": 1},
            202: {"trophies": -4, "stars": 1, "old_league": 1},
        },
        "economy_events": [
            {
                "user_id": 101,
                "event_type": "earn",
                "resource": "trophies",
                "amount": 8,
                "source": "battle",
                "metadata": {"match_id": match_id},
            }
        ],
    }


class FakeTransaction:
    def __init__(self):
        self.exit_exc_type = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_exc_type = exc_type
        return False


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return FakeAcquire(self.conn)


class DuplicateSummaryConn:
    def __init__(self):
        self.transaction_obj = FakeTransaction()
        self.queries = []

    def transaction(self):
        return self.transaction_obj

    async def fetchrow(self, query, *args):
        self.queries.append(("fetchrow", query, args))
        if "INSERT INTO battle_summary" in query:
            return None
        raise AssertionError(f"unexpected fetchrow after duplicate gate: {query}")

    async def execute(self, query, *args):
        self.queries.append(("execute", query, args))
        raise AssertionError(f"duplicate gate must not execute balance writes: {query}")


class FailingRewardConn:
    def __init__(self):
        self.transaction_obj = FakeTransaction()
        self.queries = []

    def transaction(self):
        return self.transaction_obj

    async def fetchrow(self, query, *args):
        self.queries.append(("fetchrow", query, args))
        if "INSERT INTO battle_summary" in query:
            return {"id": 1}
        if "UPDATE users" in query and "trophies" in query:
            return {"trophies": 108, "max_trophies": 108, "league": 1}
        if "UPDATE users" in query and "coins" in query:
            raise RuntimeError("coin write failed")
        if "SELECT wins_since_last_case" in query:
            return {"wins_since_last_case": 0}
        raise AssertionError(f"unexpected fetchrow: {query}")

    async def fetchval(self, query, *args):
        self.queries.append(("fetchval", query, args))
        return True

    async def execute(self, query, *args):
        self.queries.append(("execute", query, args))
        return "OK"


class QuestProgressConn:
    def __init__(self, *, fail_quest_write=False):
        self.transaction_obj = FakeTransaction()
        self.queries = []
        self.fail_quest_write = fail_quest_write

    def transaction(self):
        return self.transaction_obj

    async def fetchrow(self, query, *args):
        self.queries.append(("fetchrow", query, args))
        if "INSERT INTO battle_summary" in query:
            return {"id": 1}
        raise AssertionError(f"unexpected fetchrow: {query}")

    async def fetchval(self, query, *args):
        self.queries.append(("fetchval", query, args))
        if "FROM game_settings" in query:
            return {"daily_quests": True}
        raise AssertionError(f"unexpected fetchval: {query}")

    async def execute(self, query, *args):
        self.queries.append(("execute", query, args))
        if self.fail_quest_write and "INSERT INTO daily_quests_progress" in query:
            raise RuntimeError("quest write failed")
        return "OK"


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_end_transaction_duplicate_summary_skips_balance_writes():
    conn = DuplicateSummaryConn()
    db = Database.__new__(Database)
    db._pool = FakePool(conn)
    payload = _battle_transaction_payload()
    payload["daily_quest_ops"] = {101: [("win_battle_1", 1, False)]}

    result = await db.apply_battle_end_rewards_transaction(**payload)

    assert result == {"applied": False, "reason": "duplicate_summary"}
    assert conn.transaction_obj.exit_exc_type is None
    assert len(conn.queries) == 1
    assert "INSERT INTO battle_summary" in conn.queries[0][1]


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_end_transaction_rolls_back_when_reward_write_fails():
    conn = FailingRewardConn()
    db = Database.__new__(Database)
    db._pool = FakePool(conn)

    with pytest.raises(RuntimeError, match="coin write failed"):
        await db.apply_battle_end_rewards_transaction(**_battle_transaction_payload())

    assert conn.transaction_obj.exit_exc_type is RuntimeError
    assert any("INSERT INTO battle_summary" in query for _, query, _ in conn.queries)
    assert any("UPDATE users" in query and "trophies" in query for _, query, _ in conn.queries)


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_end_transaction_commits_daily_quests_behind_summary_gate():
    conn = QuestProgressConn()
    db = Database.__new__(Database)
    db._pool = FakePool(conn)
    payload = _battle_transaction_payload("quest-progress")
    payload["rewards"] = {}
    payload["economy_events"] = []
    payload["daily_quest_ops"] = {
        101: [
            ("win_battle_1", 1, False),
            ("win_battle_5", 1, False),
            ("win_streak_5", 1, False),
        ],
        202: [("win_streak_5", 0, True)],
    }

    result = await db.apply_battle_end_rewards_transaction(**payload)

    assert result["applied"] is True
    quest_writes = [
        args for kind, query, args in conn.queries
        if kind == "execute" and "INSERT INTO daily_quests_progress" in query
    ]
    assert len(quest_writes) == 4
    assert {args[0] for args in quest_writes} == {101, 202}


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_end_transaction_rolls_back_summary_when_quest_write_fails():
    conn = QuestProgressConn(fail_quest_write=True)
    db = Database.__new__(Database)
    db._pool = FakePool(conn)
    payload = _battle_transaction_payload("quest-progress-failure")
    payload["rewards"] = {}
    payload["economy_events"] = []
    payload["daily_quest_ops"] = {101: [("win_battle_1", 1, False)]}

    with pytest.raises(RuntimeError, match="quest write failed"):
        await db.apply_battle_end_rewards_transaction(**payload)

    assert conn.transaction_obj.exit_exc_type is RuntimeError
    assert any("INSERT INTO battle_summary" in query for _, query, _ in conn.queries)
    assert any("INSERT INTO daily_quests_progress" in query for _, query, _ in conn.queries)


class AntiFraudDB:
    def __init__(self, *, friends=False, squads=None):
        self.friends = friends
        self.squads = squads or {}

    async def are_friends(self, user_a, user_b):
        return self.friends

    async def get_user_profile(self, user_id):
        return {"squad_id": self.squads.get(user_id, 0)}


@pytest.mark.asyncio(loop_scope="function")
async def test_bots_allowed_false_override_requires_not_friends_and_different_squads():
    engine = _engine_with_state()
    engine.mode_config = ModeConfig(
        mode_id="no-bots",
        ruleset="classic",
        label="No Bots",
        classic=ClassicParams(bots_allowed=False),
    )
    engine.game_mode = "no-bots"

    engine.runtime_owner_app = {
        "db": AntiFraudDB(friends=False, squads={101: 1, 202: 2}),
        "match_game_modes": {},
    }
    assert await web_server._replacement_bot_allowed("m-af", engine) is True

    engine.runtime_owner_app = {
        "db": AntiFraudDB(friends=True, squads={101: 1, 202: 2}),
        "match_game_modes": {},
    }
    assert await web_server._replacement_bot_allowed("m-af", engine) is False

    engine.runtime_owner_app = {
        "db": AntiFraudDB(friends=False, squads={101: 7, 202: 7}),
        "match_game_modes": {},
    }
    assert await web_server._replacement_bot_allowed("m-af", engine) is False


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_vs_bot_policy_allows_one_full_turn_then_terminates(monkeypatch):
    emitted = []

    async def fake_emit(event, payload, room=None, **kwargs):
        emitted.append((event, payload, room))

    engine = _engine_with_state()
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    engine.set_player_replacement_status(202, ReplacementStatus.AFK)
    web_server.ACTIVE_MATCHES["m-bvb"] = engine
    monkeypatch.setattr(web_server.sio, "emit", fake_emit)
    web_server.BOT_VS_BOT_MARKERS.pop("m-bvb", None)

    try:
        consumed_first = await web_server._handle_bot_vs_bot_policy("m-bvb", engine, web_server.ACTIVE_MATCHES)
        assert consumed_first is False
        assert web_server.ACTIVE_MATCHES["m-bvb"] is engine

        engine.turn += 1
        consumed_second = await web_server._handle_bot_vs_bot_policy("m-bvb", engine, web_server.ACTIVE_MATCHES)
        assert consumed_second is False
        assert web_server.ACTIVE_MATCHES["m-bvb"] is engine

        engine.turn += 1
        consumed_third = await web_server._handle_bot_vs_bot_policy("m-bvb", engine, web_server.ACTIVE_MATCHES)
        assert consumed_third is True
        assert "m-bvb" not in web_server.ACTIVE_MATCHES
        assert "m-bvb" in web_server.ENDED_MATCH_IDS
        assert emitted[-1][1]["reason"] == "bot_vs_bot_after_takeover"
    finally:
        web_server.ACTIVE_MATCHES.pop("m-bvb", None)
        web_server.BOT_VS_BOT_MARKERS.pop("m-bvb", None)
        web_server.ENDED_MATCH_IDS.discard("m-bvb")


@pytest.mark.asyncio(loop_scope="function")
async def test_pve_bot_vs_bot_policy_does_not_terminate_after_afk_side_turn(monkeypatch):
    emitted = []

    async def fake_emit(event, payload, room=None, **kwargs):
        emitted.append((event, payload, room))

    engine = _engine_with_state()
    engine._arena.state.p2.is_bot = True
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    web_server.ACTIVE_MATCHES["m-pve-bvb"] = engine
    monkeypatch.setattr(web_server.sio, "emit", fake_emit)
    web_server.BOT_VS_BOT_MARKERS.pop("m-pve-bvb", None)

    try:
        consumed_first = await web_server._handle_bot_vs_bot_policy(
            "m-pve-bvb",
            engine,
            web_server.ACTIVE_MATCHES,
        )
        assert consumed_first is False

        engine.turn += 1
        consumed_after_replacement_turn = await web_server._handle_bot_vs_bot_policy(
            "m-pve-bvb",
            engine,
            web_server.ACTIVE_MATCHES,
        )
        assert consumed_after_replacement_turn is False
        assert web_server.ACTIVE_MATCHES["m-pve-bvb"] is engine

        engine.turn += 1
        consumed_after_full_cycle = await web_server._handle_bot_vs_bot_policy(
            "m-pve-bvb",
            engine,
            web_server.ACTIVE_MATCHES,
        )
        assert consumed_after_full_cycle is True
        assert "m-pve-bvb" not in web_server.ACTIVE_MATCHES
        assert emitted[-1][1]["reason"] == "bot_vs_bot_after_takeover"
    finally:
        web_server.ACTIVE_MATCHES.pop("m-pve-bvb", None)
        web_server.BOT_VS_BOT_MARKERS.pop("m-pve-bvb", None)
        web_server.ENDED_MATCH_IDS.discard("m-pve-bvb")


@pytest.mark.asyncio(loop_scope="function")
async def test_afk_replacement_human_uses_berserk_inference(monkeypatch):
    engine = _engine_with_state()
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)

    class FakeBerserkBrain:
        def __init__(self):
            self.calls = []

        def has_profile(self, difficulty):
            return True

        async def get_action_async(self, arena_state, bot_id, legal_actions, difficulty):
            self.calls.append((bot_id, difficulty, len(legal_actions)))
            return 0

    fake_brain = FakeBerserkBrain()
    engine.berserk_brain = fake_brain
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)

    await web_server.run_bot_routine(engine, 101)

    assert fake_brain.calls
    assert fake_brain.calls[0][0] == 101


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_routine_uses_match_brain_after_process_alias_changes(monkeypatch):
    engine = _engine_with_state()
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)

    class TrackingBrain:
        def __init__(self, action_id=0):
            self.action_id = action_id
            self.calls = 0

        def has_profile(self, _difficulty):
            return True

        async def get_action_async(self, *_args, **_kwargs):
            self.calls += 1
            return self.action_id

    match_brain = TrackingBrain()
    replacement_app_brain = TrackingBrain()
    engine.berserk_brain = match_brain
    monkeypatch.setattr(web_server, "BERSERK_BRAIN", replacement_app_brain)
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)

    await web_server.run_bot_routine(engine, 101)

    assert match_brain.calls > 0
    assert replacement_app_brain.calls == 0


@pytest.mark.asyncio(loop_scope="function")
async def test_modified_mode_bot_routine_skips_berserk_and_uses_rule_based(monkeypatch):
    engine = _engine_with_state()
    engine.game_mode = "extra_arena:spellstorm"
    engine.mode_config = ModeConfig(
        mode_id="extra_arena:spellstorm",
        ruleset="classic",
        label="ExtraArena Spellstorm",
    )
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)
    monkeypatch.setattr(engine._arena, "get_legal_actions", lambda _bot_id: [EndTurnAction()])

    class FakeBerserkBrain:
        calls = 0

        def has_profile(self, difficulty):
            return True

        async def get_action_async(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("modified modes must not run Berserk inference")

    fake_brain = FakeBerserkBrain()
    engine.berserk_brain = fake_brain

    executed = []

    def fake_execute_bot_action(action_dict):
        executed.append(action_dict)
        engine._arena.state.current_turn_owner_id = 202
        engine.current_player_id = 202
        engine.turn += 1
        return {"success": True}

    monkeypatch.setattr(engine, "execute_bot_action", fake_execute_bot_action)

    await web_server.run_bot_routine(engine, 101)

    assert fake_brain.calls == 0
    assert executed == [{"type": "end_turn"}]


@pytest.mark.asyncio(loop_scope="function")
async def test_missing_berserk_profile_uses_rule_based_without_inference_exception(monkeypatch):
    engine = _engine_with_state()
    engine.match_id = "m-bot-missing-profile"
    engine.bot_difficulty = "missing-tier"
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)
    monkeypatch.setattr(engine._arena, "get_legal_actions", lambda _bot_id: [EndTurnAction()])

    class MissingProfileBrain:
        calls = 0

        def has_profile(self, difficulty):
            return False

        async def get_action_async(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("ONNX inference should not run for a missing profile")

    fake_brain = MissingProfileBrain()
    engine.berserk_brain = fake_brain

    executed = []

    def fake_execute_bot_action(action_dict):
        executed.append(action_dict)
        engine._arena.state.current_turn_owner_id = 202
        engine.current_player_id = 202
        engine.turn += 1
        return {"success": True}

    monkeypatch.setattr(engine, "execute_bot_action", fake_execute_bot_action)

    await web_server.run_bot_routine(engine, 101)

    assert fake_brain.calls == 0
    assert executed == [{"type": "end_turn"}]


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_action_mutation_runs_inside_match_lock(monkeypatch):
    engine = _engine_with_state()
    engine.match_id = "m-bot-lock"
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    web_server.MATCH_LOCKS.pop("m-bot-lock", None)
    engine.berserk_brain = None
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)
    monkeypatch.setattr(engine._arena, "get_legal_actions", lambda _bot_id: [EndTurnAction()])

    lock_states = []

    def fake_execute_bot_action(action_dict):
        lock_states.append(web_server.MATCH_LOCKS["m-bot-lock"].locked())
        engine._arena.state.current_turn_owner_id = 202
        engine.current_player_id = 202
        engine.turn += 1
        return {"success": True}

    monkeypatch.setattr(engine, "execute_bot_action", fake_execute_bot_action)

    try:
        await web_server.run_bot_routine(engine, 101)
        assert lock_states == [True]
    finally:
        web_server.MATCH_LOCKS.pop("m-bot-lock", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_metronome_delays_selected_action_before_mutation_lock(monkeypatch):
    engine = _engine_with_state()
    engine.match_id = "m-metronome-order"
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    web_server.MATCH_LOCKS.pop(engine.match_id, None)

    class EndTurnBrain:
        def has_profile(self, _difficulty):
            return True

        async def get_action_async(
            self,
            _arena_state,
            _bot_id,
            legal_actions,
            difficulty,
        ):
            del difficulty
            return next(
                index
                for index, action in enumerate(legal_actions)
                if isinstance(action, EndTurnAction)
            )

    metronome_calls = []

    class FakeMetronome:
        def sample_ms(
            self,
            state,
            actor_id,
            *,
            action_type,
            legal_action_count,
            rng,
        ):
            lock = web_server.MATCH_LOCKS.get(engine.match_id)
            metronome_calls.append(
                {
                    "state": state,
                    "actor_id": actor_id,
                    "action_type": action_type,
                    "legal_action_count": legal_action_count,
                    "lock_held": bool(lock and lock.locked()),
                    "rng": rng,
                }
            )
            return 250.0

    slept = []

    async def fake_sleep(seconds):
        slept.append(float(seconds))

    engine.berserk_brain = EndTurnBrain()
    engine.extra_lr_aux_runtime = SimpleNamespace(metronome=FakeMetronome())
    monkeypatch.setattr(web_server.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)

    try:
        await web_server.run_bot_routine(engine, 101)
    finally:
        web_server.MATCH_LOCKS.pop(engine.match_id, None)

    assert len(metronome_calls) == 1
    assert metronome_calls[0]["actor_id"] == 101
    assert metronome_calls[0]["action_type"] == "end_turn"
    assert metronome_calls[0]["legal_action_count"] > 0
    assert metronome_calls[0]["lock_held"] is False
    assert slept[0] == pytest.approx(0.25)


@pytest.mark.asyncio(loop_scope="function")
async def test_metronome_fallback_never_spends_emergency_timer_budget(monkeypatch):
    engine = _engine_with_state()
    engine.match_id = "m-metronome-budget"
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    web_server.MATCH_LOCKS.pop(engine.match_id, None)
    threshold = engine.mode_config.classic.bot_emergency_threshold_seconds
    remaining = [threshold + 0.4]

    class EndTurnBrain:
        def has_profile(self, _difficulty):
            return True

        async def get_action_async(
            self,
            _arena_state,
            _bot_id,
            legal_actions,
            difficulty,
        ):
            del difficulty
            return next(
                index
                for index, action in enumerate(legal_actions)
                if isinstance(action, EndTurnAction)
            )

    class BrokenMetronome:
        def sample_ms(self, *_args, **_kwargs):
            raise RuntimeError("injected beta failure")

    slept = []

    async def fake_sleep(seconds):
        seconds = float(seconds)
        slept.append(seconds)
        remaining[0] -= seconds

    monkeypatch.setattr(engine, "get_turn_time_remaining", lambda: remaining[0])
    engine._metronome_rng = SimpleNamespace(uniform=lambda *_args: 6.0)
    engine.berserk_brain = EndTurnBrain()
    engine.extra_lr_aux_runtime = SimpleNamespace(metronome=BrokenMetronome())
    monkeypatch.setattr(web_server.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 1.0)

    try:
        await web_server.run_bot_routine(engine, 101)
    finally:
        web_server.MATCH_LOCKS.pop(engine.match_id, None)

    assert sum(slept) == pytest.approx(0.4)
    assert remaining[0] == pytest.approx(threshold)


@pytest.mark.asyncio(loop_scope="function")
async def test_metronome_cannot_execute_selected_action_after_deadline(monkeypatch):
    engine = _engine_with_state()
    engine.match_id = "m-metronome-deadline"
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    web_server.MATCH_LOCKS.pop(engine.match_id, None)
    remaining = [
        engine.mode_config.classic.bot_emergency_threshold_seconds + 0.1
    ]
    executed = []

    class EndTurnBrain:
        def has_profile(self, _difficulty):
            return True

        async def get_action_async(
            self,
            _arena_state,
            _bot_id,
            legal_actions,
            difficulty,
        ):
            del difficulty
            return next(
                index
                for index, action in enumerate(legal_actions)
                if isinstance(action, EndTurnAction)
            )

    class SlowMetronome:
        def sample_ms(self, *_args, **_kwargs):
            remaining[0] = -0.1
            return 100.0

    monkeypatch.setattr(engine, "get_turn_time_remaining", lambda: remaining[0])
    monkeypatch.setattr(
        engine,
        "execute_bot_action",
        lambda action: executed.append(action) or {"success": True},
    )
    engine.berserk_brain = EndTurnBrain()
    engine.extra_lr_aux_runtime = SimpleNamespace(metronome=SlowMetronome())
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)

    try:
        await web_server.run_bot_routine(engine, 101)
    finally:
        web_server.MATCH_LOCKS.pop(engine.match_id, None)

    assert executed == []
    assert engine.current_player_id == 202


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_forced_end_turn_runs_inside_match_lock(monkeypatch):
    engine = _engine_with_state()
    engine.match_id = "m-bot-force-lock"
    engine.set_player_replacement_status(101, ReplacementStatus.AFK)
    web_server.MATCH_LOCKS.pop("m-bot-force-lock", None)
    engine.berserk_brain = None
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)
    monkeypatch.setattr(engine._arena, "get_legal_actions", lambda _bot_id: [])

    lock_states = []

    def fake_end_turn(user_id):
        lock = web_server.MATCH_LOCKS.get("m-bot-force-lock")
        lock_states.append(bool(lock and lock.locked()))
        engine._arena.state.current_turn_owner_id = 202
        engine.current_player_id = 202
        engine.turn += 1
        return {"success": True}

    monkeypatch.setattr(engine, "end_turn", fake_end_turn)

    try:
        await web_server.run_bot_routine(engine, 101)
        assert lock_states == [True]
    finally:
        web_server.MATCH_LOCKS.pop("m-bot-force-lock", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_terminal_match_cleanup_removes_runtime_disconnect_and_bot_state(monkeypatch):
    emitted = []

    async def fake_emit(event, payload, room=None, **kwargs):
        emitted.append((event, payload, room))

    async def sleeper():
        await asyncio.sleep(10)

    engine = _engine_with_state()
    disconnect_task = asyncio.create_task(sleeper())
    bot_task = asyncio.create_task(sleeper())
    monkeypatch.setattr(web_server.sio, "emit", fake_emit)
    web_server.ACTIVE_MATCHES["m-clean"] = engine
    web_server.MATCH_DISCONNECT_TASKS[("m-clean", 101)] = disconnect_task
    web_server.MATCH_DISCONNECT_STATES[("m-clean", 101)] = {"timed_out_turns": 1}
    web_server.MATCH_SESSIONS["m-clean"] = {101: {"sid-clean"}}
    web_server.SID_TO_MATCH["sid-clean"] = {"match_id": "m-clean", "user_id": 101}
    web_server.BOT_TASKS["m-clean"] = bot_task
    web_server.BOT_TASK_KEYS["m-clean"] = ("101", 1)
    web_server.BOT_VS_BOT_MARKERS["m-clean"] = 1
    web_server.MATCH_LOCKS["m-clean"] = asyncio.Lock()

    try:
        await web_server._terminate_match_without_rewards(
            "m-clean",
            web_server.ACTIVE_MATCHES,
            reason="unit_cleanup",
            message="done",
        )
        await asyncio.sleep(0)

        assert "m-clean" not in web_server.ACTIVE_MATCHES
        assert ("m-clean", 101) not in web_server.MATCH_DISCONNECT_TASKS
        assert ("m-clean", 101) not in web_server.MATCH_DISCONNECT_STATES
        assert "m-clean" not in web_server.MATCH_SESSIONS
        assert "sid-clean" not in web_server.SID_TO_MATCH
        assert "m-clean" not in web_server.BOT_TASKS
        assert "m-clean" not in web_server.BOT_TASK_KEYS
        assert "m-clean" not in web_server.BOT_VS_BOT_MARKERS
        assert "m-clean" in web_server.MATCH_LOCKS
        assert "m-clean" in web_server.ENDED_MATCH_IDS
        assert disconnect_task.cancelled()
        assert bot_task.cancelled()
        assert emitted[-1][1]["reason"] == "unit_cleanup"
    finally:
        for task in (disconnect_task, bot_task):
            if not task.done():
                task.cancel()
        web_server.ACTIVE_MATCHES.pop("m-clean", None)
        web_server.MATCH_DISCONNECT_TASKS.pop(("m-clean", 101), None)
        web_server.MATCH_DISCONNECT_STATES.pop(("m-clean", 101), None)
        web_server.MATCH_SESSIONS.pop("m-clean", None)
        web_server.SID_TO_MATCH.pop("sid-clean", None)
        web_server.BOT_TASKS.pop("m-clean", None)
        web_server.BOT_TASK_KEYS.pop("m-clean", None)
        web_server.BOT_VS_BOT_MARKERS.pop("m-clean", None)
        web_server.MATCH_LOCKS.pop("m-clean", None)
        web_server.ENDED_MATCH_IDS.discard("m-clean")


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_task_guard_starts_one_task_per_turn(monkeypatch):
    calls = []

    async def fake_run_bot_routine(engine, bot_id):
        calls.append((engine.turn, bot_id))
        await asyncio.sleep(0.01)

    class BotEngine:
        turn = 3
        _active_matches = {}

    monkeypatch.setattr(web_server, "run_bot_routine", fake_run_bot_routine)
    web_server.BOT_TASKS.pop("m2", None)
    web_server.BOT_TASK_KEYS.pop("m2", None)

    web_server._start_guarded_bot_task("m2", BotEngine(), 9)
    web_server._start_guarded_bot_task("m2", BotEngine(), 9)
    await web_server.BOT_TASKS["m2"]

    assert calls == [(3, 9)]


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_task_finally_does_not_delete_replacement_task(monkeypatch):
    replacement_task = None

    class BotEngine:
        turn = 3
        current_player_id = 9
        _active_matches = {}

        def get_current_player_id(self):
            return self.current_player_id

        def is_bot(self, user_id):
            return int(user_id) == 9

    async def fake_run_bot_routine(engine, bot_id):
        nonlocal replacement_task
        replacement_task = asyncio.create_task(asyncio.sleep(10))
        web_server.BOT_TASKS["m-identity"] = replacement_task
        web_server.BOT_TASK_KEYS["m-identity"] = (str(bot_id), int(engine.turn))

    monkeypatch.setattr(web_server, "run_bot_routine", fake_run_bot_routine)
    web_server.BOT_TASKS.pop("m-identity", None)
    web_server.BOT_TASK_KEYS.pop("m-identity", None)

    try:
        web_server._start_guarded_bot_task("m-identity", BotEngine(), 9)
        initial_task = web_server.BOT_TASKS["m-identity"]
        await initial_task

        assert web_server.BOT_TASKS["m-identity"] is replacement_task
        assert web_server.BOT_TASK_KEYS["m-identity"] == ("9", 3)
    finally:
        task = web_server.BOT_TASKS.pop("m-identity", None)
        if task and not task.done():
            task.cancel()
        web_server.BOT_TASK_KEYS.pop("m-identity", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_old_app_cleanup_only_closes_owned_runtime_and_matches(monkeypatch):
    class FakeBrain:
        instances = []

        def __init__(self, *, profiles):
            self.name = f"brain-{len(self.instances) + 1}"
            self.sessions = {profile: object() for profile in profiles}
            self.closed = False
            self.instances.append(self)

        def has_profile(self, difficulty):
            return difficulty in self.sessions

        def close(self):
            self.closed = True

    class FakeAuxRuntime:
        instances = []

        def __init__(self):
            self.name = f"aux-{len(self.instances) + 1}"
            self.availability = {}
            self.closed = False
            self.instances.append(self)

        @classmethod
        def from_model_dir(cls):
            return cls()

        def close(self):
            self.closed = True

    async def sleeper(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(web_server, "BerserkInference", FakeBrain)
    monkeypatch.setattr(web_server, "ExtraLRAuxRuntime", FakeAuxRuntime)
    monkeypatch.setattr(web_server, "run_bot_routine", sleeper)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()

    app_a = web_server.create_web_app(MinimalWebDB(), bot_token="a")
    app_b = web_server.create_web_app(MinimalWebDB(), bot_token="b")
    brain_a = app_a["berserk_brain"]
    brain_b = app_b["berserk_brain"]
    aux_a = app_a["extra_lr_aux"]
    aux_b = app_b["extra_lr_aux"]

    engine_a = BattleEngine(
        match_id="reload-a",
        active_matches=app_a["active_matches"],
    )
    engine_b = BattleEngine(
        match_id="reload-b",
        active_matches=app_b["active_matches"],
    )
    web_server._bind_match_runtime(engine_a, app_a)
    web_server._bind_match_runtime(engine_b, app_b)
    app_a["active_matches"]["reload-a"] = engine_a
    app_b["active_matches"]["reload-b"] = engine_b

    disconnect_a = asyncio.create_task(sleeper())
    disconnect_b = asyncio.create_task(sleeper())
    web_server.MATCH_DISCONNECT_TASKS[("reload-a", 1)] = disconnect_a
    web_server.MATCH_DISCONNECT_TASKS[("reload-b", 2)] = disconnect_b
    web_server.MATCH_DISCONNECT_STATES[("reload-a", 1)] = {"disconnected": True}
    web_server.MATCH_DISCONNECT_STATES[("reload-b", 2)] = {"disconnected": True}
    web_server.MATCH_SESSIONS["reload-a"] = {1: {"sid-reload-a"}}
    web_server.MATCH_SESSIONS["reload-b"] = {2: {"sid-reload-b"}}
    web_server.SID_TO_MATCH["sid-reload-a"] = {
        "match_id": "reload-a",
        "user_id": 1,
    }
    web_server.SID_TO_MATCH["sid-reload-b"] = {
        "match_id": "reload-b",
        "user_id": 2,
    }

    web_server._start_guarded_bot_task("reload-a", engine_a, 1)
    web_server._start_guarded_bot_task("reload-b", engine_b, 2)
    bot_task_a = web_server.BOT_TASKS["reload-a"]
    bot_task_b = web_server.BOT_TASKS["reload-b"]
    assert web_server.BOT_TASK_OWNERS["reload-a"] is app_a
    assert web_server.BOT_TASK_OWNERS["reload-b"] is app_b

    cleanup_a = app_a.on_cleanup[-1]
    cleanup_b = app_b.on_cleanup[-1]
    try:
        await cleanup_a(app_a)

        assert bot_task_a.cancelled()
        assert disconnect_a.cancelled()
        assert "reload-a" not in web_server.ACTIVE_MATCHES
        assert ("reload-a", 1) not in web_server.MATCH_DISCONNECT_STATES
        assert "reload-a" not in web_server.MATCH_SESSIONS
        assert "sid-reload-a" not in web_server.SID_TO_MATCH
        assert engine_a.is_ended is True
        assert engine_a.runtime_owner_app is None
        assert engine_a.berserk_brain is None
        assert engine_a.extra_lr_aux_runtime is None
        assert brain_a.closed is True
        assert aux_a.closed is True

        assert bot_task_b.done() is False
        assert disconnect_b.done() is False
        assert web_server.ACTIVE_MATCHES["reload-b"] is engine_b
        assert web_server.MATCH_DISCONNECT_STATES[("reload-b", 2)]
        assert web_server.MATCH_SESSIONS["reload-b"]
        assert web_server.SID_TO_MATCH["sid-reload-b"]["match_id"] == "reload-b"
        assert engine_b.runtime_owner_app is app_b
        assert engine_b.berserk_brain is brain_b
        assert engine_b.extra_lr_aux_runtime is aux_b
        assert brain_b.closed is False
        assert aux_b.closed is False
        assert web_server.BERSERK_BRAIN is brain_b
        assert web_server.EXTRA_LR_AUX is aux_b
    finally:
        await cleanup_b(app_b)
        for task in (bot_task_a, bot_task_b, disconnect_a, disconnect_b):
            if not task.done():
                task.cancel()
        for match_id in ("reload-a", "reload-b"):
            web_server.ACTIVE_MATCHES.pop(match_id, None)
            web_server.MATCH_SESSIONS.pop(match_id, None)
            web_server.BOT_TASKS.pop(match_id, None)
            web_server.BOT_TASK_KEYS.pop(match_id, None)
            web_server.BOT_TASK_OWNERS.pop(match_id, None)
            web_server.ENDED_MATCH_IDS.discard(match_id)
            web_server.ENDED_MATCH_TIMES.pop(match_id, None)
        for key in (("reload-a", 1), ("reload-b", 2)):
            web_server.MATCH_DISCONNECT_TASKS.pop(key, None)
            web_server.MATCH_DISCONNECT_STATES.pop(key, None)
        web_server.SID_TO_MATCH.pop("sid-reload-a", None)
        web_server.SID_TO_MATCH.pop("sid-reload-b", None)
