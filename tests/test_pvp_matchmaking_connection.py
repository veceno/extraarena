import asyncio
import ast
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from core.classic_setup import create_classic_game_state
from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, ReplacementStatus
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


class MinimalWebDB:
    pass


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

    def get_full_state(self, viewer_id=None):
        return {
            "match_id": "m-turn",
            "viewer_id": viewer_id,
            "current_player_id": self.current_player_id,
            "match_status": "waiting_for_players" if self.waiting_for_players else "active",
        }


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


class SurrenderDB:
    def __init__(self):
        self.trophy_updates = 0

    async def get_user_info(self, user_id):
        return {"trophies": 100, "league": 1}

    async def update_user_trophies(self, user_id, delta):
        self.trophy_updates += 1
        return {"trophies": 100 + int(delta), "max_trophies": 100, "league": 1}


async def _battle_client(monkeypatch, engine, match_id="m-turn"):
    async def fake_check_and_run_bot(_match_id, _active_matches):
        return None

    monkeypatch.setattr(web_server, "check_and_run_bot", fake_check_and_run_bot)
    app = web_server.create_web_app(MinimalWebDB(), bot_token="bot-token")
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
async def test_matchmaker_prunes_terminal_match_aliases_by_ttl():
    mm = Matchmaker(FakeDB(), FakeBotFactory())

    first = await mm.find_match(11, 500, 1, selected_deck_id=31, game_mode="classic")
    second = await mm.find_match(22, 500, 1, selected_deck_id=42, game_mode="classic")
    await mm.cancel_match(second["match_id"], message="done", error="unit")

    removed = await mm.prune_expired_matches(now=time.monotonic() + 3600, ttl_seconds=1)

    assert removed >= 1
    assert (await mm.get_status(first["match_id"]))["status"] == "not_found"
    assert (await mm.get_status(second["match_id"]))["status"] == "not_found"


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
async def test_action_timeout_guard_runs_while_match_lock_is_held(monkeypatch):
    engine = EndTurnEngine(expired=True)
    client = await _battle_client(monkeypatch, engine, match_id="m-lock")
    web_server.MATCH_LOCKS["m-lock"] = asyncio.Lock()

    async def fake_auto_end(app, match_id, engine_arg, viewer_id, client_action_id):
        assert web_server.MATCH_LOCKS[str(match_id)].locked()
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
    mm = Matchmaker(FakeDB(), FakeBotFactory())

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
    web_server.MATCH_DISCONNECT_TASKS[("m1", 42)] = pending
    web_server.MATCH_DISCONNECT_STATES[("m1", 42)] = {"timed_out_turns": 1}

    try:
        web_server._register_session("m1", 42, "sid-new")
        await asyncio.sleep(0)
        assert pending.cancelled()
        assert engine.restored == [42]
        assert ("m1", 42) not in web_server.MATCH_DISCONNECT_STATES
    finally:
        web_server.ACTIVE_MATCHES.pop("m1", None)
        web_server.MATCH_SESSIONS.pop("m1", None)
        task = web_server.MATCH_DISCONNECT_TASKS.pop(("m1", 42), None)
        if task:
            task.cancel()


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


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_end_transaction_duplicate_summary_skips_balance_writes():
    conn = DuplicateSummaryConn()
    db = Database.__new__(Database)
    db._pool = FakePool(conn)

    result = await db.apply_battle_end_rewards_transaction(**_battle_transaction_payload())

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
    old_app = getattr(web_server.sio, "app", None)

    try:
        web_server.sio.app = {"db": AntiFraudDB(friends=False, squads={101: 1, 202: 2}), "match_game_modes": {}}
        assert await web_server._replacement_bot_allowed("m-af", engine) is True

        web_server.sio.app = {"db": AntiFraudDB(friends=True, squads={101: 1, 202: 2}), "match_game_modes": {}}
        assert await web_server._replacement_bot_allowed("m-af", engine) is False

        web_server.sio.app = {"db": AntiFraudDB(friends=False, squads={101: 7, 202: 7}), "match_game_modes": {}}
        assert await web_server._replacement_bot_allowed("m-af", engine) is False
    finally:
        if old_app is None:
            try:
                delattr(web_server.sio, "app")
            except AttributeError:
                pass
        else:
            web_server.sio.app = old_app


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
        assert consumed_second is True
        assert "m-bvb" not in web_server.ACTIVE_MATCHES
        assert "m-bvb" in web_server.ENDED_MATCH_IDS
        assert emitted[-1][1]["reason"] == "bot_vs_bot_after_takeover"
    finally:
        web_server.ACTIVE_MATCHES.pop("m-bvb", None)
        web_server.BOT_VS_BOT_MARKERS.pop("m-bvb", None)
        web_server.ENDED_MATCH_IDS.discard("m-bvb")


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
        assert "m-clean" not in web_server.MATCH_LOCKS
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
