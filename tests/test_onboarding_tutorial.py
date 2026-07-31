import asyncio
import hashlib
import hmac
import json
import time
from urllib.parse import quote, urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database
from onboarding_tutorial import ONBOARDING_MIDORIA_ASSET, TUTORIAL_FINAL_STEP, TutorialBattleEngine
from web import server as web_server


USER_ID = 12_345


class _AsyncContext:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeTelegramResponse:
    def __init__(self, payload, *, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, *args, **kwargs):
        return self.payload


class _FakeTelegramSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        payload = self.payloads.pop(0) if self.payloads else {"ok": False, "description": "no fake response"}
        if isinstance(payload, Exception):
            raise payload
        return _FakeTelegramResponse(payload)


class _NewbiePathClaimConn:
    def __init__(self, progress, *, row_exists=True):
        self.progress = progress
        self.row_exists = row_exists
        self.fetches = []
        self.executed = []
        self.coins_granted = 0

    def transaction(self):
        return _AsyncContext()

    async def fetchrow(self, query, *args):
        self.fetches.append((query, args))
        if "SELECT newbie_path_progress" in query:
            if not self.row_exists:
                return None
            return {"newbie_path_progress": self.progress}
        raise AssertionError(f"unexpected fetchrow: {query}")

    async def execute(self, query, *args):
        self.executed.append((query, args))
        if "SET newbie_path_progress" in query:
            self.progress = args[1]
        if "UPDATE users SET coins" in query:
            self.coins_granted += int(args[1])
        return "OK"


class _NewbiePathClaimPool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _AsyncContext(self.conn)


class _NewbiePathClaimDB(Database):
    def __init__(self, conn):
        super().__init__(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
        self._pool = _NewbiePathClaimPool(conn)

    async def get_onboarding_state(self, user_id):
        progress = self._pool.conn.progress
        if isinstance(progress, str):
            progress = json.loads(progress)
        return {"newbie_path_progress": progress}


class _TutorialRouteDB:
    def __init__(self, initial_state=None):
        self.state = {
            "status": "welcome",
            "current_step": "welcome",
            "tutorial_step": 0,
            "tutorial_match_id": None,
            "menu_step": "reward",
            "newbie_path_progress": {},
            "completed": False,
            **(initial_state or {}),
        }
        self.ensure_user_calls = []
        self.events = []
        self.coins_granted = 0
        self.welcome_shown_calls = 0

    async def ensure_user(self, *, user_id, username, first_name, last_name):
        self.ensure_user_calls.append(int(user_id))
        return True

    async def fetchval(self, query, *args):
        if "SELECT 1 FROM users" in query:
            return None
        return None

    async def get_onboarding_state(self, user_id):
        state = dict(self.state)
        state["user_id"] = int(user_id)
        state["completed"] = state.get("status") == "completed"
        return state

    async def set_onboarding_state(self, user_id, **updates):
        self.state.update({key: value for key, value in updates.items() if value is not None})
        self.state["user_id"] = int(user_id)
        return await self.get_onboarding_state(user_id)

    async def mark_newbie_path_task(self, user_id, task_id, *, claimed=False):
        progress = dict(self.state.get("newbie_path_progress") or {})
        tasks = dict(progress.get("tasks") or {})
        entry = dict(tasks.get(task_id) or {})
        entry["completed"] = True
        if claimed:
            entry["claimed"] = True
        entry["completed_at"] = entry.get("completed_at") or "2026-06-10T00:00:00+00:00"
        tasks[str(task_id)] = entry
        progress["tasks"] = tasks
        self.state["newbie_path_progress"] = progress
        return await self.get_onboarding_state(user_id)

    async def claim_newbie_path_task(self, user_id, task_id, *, reward_coins=0, require_completed=True):
        progress = dict(self.state.get("newbie_path_progress") or {})
        tasks = dict(progress.get("tasks") or {})
        entry = dict(tasks.get(task_id) or {})
        if require_completed and not entry.get("completed"):
            return {"success": False, "error": "task_not_completed"}
        already_claimed = bool(entry.get("claimed"))
        entry["completed"] = True
        entry["claimed"] = True
        entry["completed_at"] = entry.get("completed_at") or "2026-06-10T00:00:00+00:00"
        tasks[str(task_id)] = entry
        progress["tasks"] = tasks
        self.state["newbie_path_progress"] = progress
        granted_amount = int(reward_coins or 0) if not already_claimed else 0
        self.coins_granted += granted_amount
        return {
            "success": True,
            "already_claimed": already_claimed,
            "granted_amount": granted_amount,
            "state": await self.get_onboarding_state(user_id),
        }

    async def track_onboarding_event(self, user_id, event_name, completed=False, metadata=None):
        self.events.append((int(user_id), event_name, completed, metadata or {}))

    async def mark_welcome_shown(self, user_id):
        self.welcome_shown_calls += 1

    async def get_user_info(self, user_id):
        return {"extra_pass": "inactive"}


async def _onboarding_route_client(monkeypatch, db):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def _signed_init_data(user_id=USER_ID, bot_token="bot-token"):
    params = {
        "auth_date": str(int(time.time())),
        "query_id": "test-query",
        "user": json.dumps({"id": int(user_id), "first_name": "Test"}, separators=(",", ":")),
    }
    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(params.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    params["hash"] = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    return urlencode(params)


def _signed_auth_query(user_id=USER_ID, bot_token="bot-token"):
    return quote(_signed_init_data(user_id=user_id, bot_token=bot_token), safe="")


def _completed_original_newbie_progress():
    return {
        "tasks": {
            "open_starter_case": {"completed": True, "claimed": True},
            "view_new_card": {"completed": True, "claimed": True},
            "save_first_deck": {"completed": True, "claimed": True},
            "play_regular_battle": {"completed": True, "claimed": True},
        }
    }


def _tutorial(engine: TutorialBattleEngine) -> dict:
    return engine.tutorial_payload()


def _attack_opponent_hero(engine: TutorialBattleEngine) -> dict:
    attacker_id = _tutorial(engine)["attacker_instance_id"]
    assert attacker_id
    return engine.apply_tutorial_action(
        {
            "type": "attack",
            "attacker_id": attacker_id,
            "target_is_hero": True,
        }
    )


def test_tutorial_battle_engine_advances_through_scripted_victory():
    engine = TutorialBattleEngine(user_id=USER_ID)

    assert _tutorial(engine)["step_index"] == 0
    assert _tutorial(engine)["step_id"] == "goal"

    # step0 goal (continue) → step1 play_attacker
    assert engine.apply_tutorial_action({"type": "continue"}) == {
        "success": True,
        "tutorial_step": 1,
    }
    assert _tutorial(engine)["hand_attacker_instance_id"]

    # step1 play_attacker (play Слайм) → step2 sleep
    play_attacker = engine.apply_tutorial_action(
        {"type": "play_card", "card_id": 37, "hand_index": 0}
    )
    assert play_attacker["success"] is True
    assert play_attacker["tutorial_step"] == 2
    assert _tutorial(engine)["attacker_instance_id"]

    # step2 sleep (end_turn) → p2 plays Стив → step3 threat
    end_turn = engine.apply_tutorial_action({"type": "end_turn"})
    assert end_turn["success"] is True
    assert end_turn["tutorial_step"] == 3
    assert [card.card_id for card in engine._arena.state.p2.board] == [40]

    # step3 threat (continue) → p2 ends turn → step4 choose_target
    threat = engine.apply_tutorial_action({"type": "continue"})
    assert threat["success"] is True
    assert threat["tutorial_step"] == 4

    # step4 choose_target (attack hero 8→4) → step5 tempo
    first_attack = _attack_opponent_hero(engine)
    assert first_attack["success"] is True
    assert first_attack["tutorial_step"] == 5
    assert engine._arena.state.p2.hero.hp == 4

    # step5 tempo (continue) → step6 danger → step7 taunt_intro
    assert engine.apply_tutorial_action({"type": "continue"})["tutorial_step"] == 6
    assert engine.apply_tutorial_action({"type": "continue"})["tutorial_step"] == 7

    # step7 taunt_intro (play Альфонс) → p1 ends → step8 taunt_demo
    play_alphonse = engine.apply_tutorial_action(
        {"type": "play_card", "card_id": 39, "hand_index": 0}
    )
    assert play_alphonse["success"] is True
    assert play_alphonse["tutorial_step"] == 8
    assert _tutorial(engine)["step_id"] == "taunt_demo"
    assert _tutorial(engine)["alphonse_instance_id"]
    assert [card.card_id for card in engine._arena.state.p1.board] == [37, 39]

    # step8 taunt_demo (auto_continue) → Стив attacks Альфонс (dies) → p2 ends → step9 lethal
    taunt_demo = engine.apply_tutorial_action({"type": "auto_continue"})
    assert taunt_demo["success"] is True
    assert taunt_demo["tutorial_step"] == 9
    assert taunt_demo["after_message"] == "Враг атакует Альфонса. Вот так работает Провокация — она закрывает героя и ломает план врага."
    assert _tutorial(engine)["step_id"] == "lethal"
    # Real engine death processing removes Альфонс — no hp-0 corpse stays behind.
    assert not any(card.card_id == 39 for card in engine._arena.state.p1.board)
    assert [card.card_id for card in engine._arena.state.p1.board] == [37]

    # step9 lethal (attack hero 4→0) → step10 victory (real P1_WIN)
    lethal_attack = _attack_opponent_hero(engine)
    assert lethal_attack["success"] is True
    assert lethal_attack["tutorial_step"] == TUTORIAL_FINAL_STEP
    assert lethal_attack["game_over"] is True
    assert lethal_attack["winner_id"] == USER_ID
    assert engine.is_ended is True
    assert _tutorial(engine)["step_id"] == "victory"


def test_tutorial_battle_engine_rejects_wrong_card_on_alphonse_step():
    engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=7)

    result = engine.apply_tutorial_action(
        {"type": "play_card", "card_id": 37, "hand_index": 0}
    )

    assert result == {
        "success": False,
        "error": "tutorial_wrong_action",
        "feedback": "Сейчас нужен Альфонс. Он примет удар на себя.",
        "tutorial_step": 7,
    }


def test_tutorial_payload_uses_onboarding_midoria_asset():
    engine = TutorialBattleEngine(user_id=USER_ID)

    assert _tutorial(engine)["midoria_asset"] == ONBOARDING_MIDORIA_ASSET


def test_tutorial_card_ids_survive_step_rebuild_before_attack():
    engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=4)
    attacker_id_from_rendered_state = _tutorial(engine)["attacker_instance_id"]

    engine.set_tutorial_step(4)
    result = engine.apply_tutorial_action(
        {
            "type": "attack",
            "attacker_id": attacker_id_from_rendered_state,
            "target_is_hero": True,
        }
    )

    assert result["success"] is True
    assert result["tutorial_step"] == 5
    assert engine._arena.state.p2.hero.hp == 4


def test_tutorial_battle_uses_tutorial_only_names_and_non_expiring_timer():
    engine = TutorialBattleEngine(user_id=USER_ID)
    state = engine.get_full_state(viewer_id=USER_ID)

    assert state["player"]["name"] == "Ты"
    assert state["opponent"]["name"] == "Кто-то злой"
    assert state["turn_duration"] == 99
    assert state["turn_time_remaining"] == 99

    engine.turn_start_time = time.time() - 10_000
    engine.turn_start_monotonic = time.monotonic() - 10_000
    assert engine.get_turn_time_remaining() == 99
    assert engine.is_turn_expired() is False

    engine.mark_timeout(USER_ID)
    engine.mark_timeout(USER_ID)
    assert engine.p1_consecutive_timeouts == 0
    assert engine.p1_state.replacement_status.value == "active"


@pytest.mark.asyncio
async def test_tutorial_runtime_is_excluded_from_timeout_and_common_bot_routines(monkeypatch):
    engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=8)
    match_id = str(engine.match_id)
    engine.turn_start_time = time.time() - 10_000
    engine.turn_start_monotonic = time.monotonic() - 10_000
    before = engine.get_full_state(viewer_id=USER_ID)

    assert await web_server._handle_natural_turn_timeout({}, match_id, engine) is False

    web_server.BOT_TASKS.pop(match_id, None)
    web_server._start_guarded_bot_task(match_id, engine, engine.bot_id)
    assert match_id not in web_server.BOT_TASKS

    started = []
    monkeypatch.setattr(
        web_server,
        "_start_guarded_bot_task",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )
    await web_server.check_and_run_bot(match_id, {match_id: engine})

    after = engine.get_full_state(viewer_id=USER_ID)
    assert started == []
    assert after["tutorial"]["step_index"] == before["tutorial"]["step_index"] == 8
    assert after["current_player_id"] == before["current_player_id"] == engine.bot_id
    assert engine.p1_state.replacement_status.value == "active"


def test_tutorial_payload_exposes_screen_progress_and_previous_message():
    sleep_payload = TutorialBattleEngine(user_id=USER_ID, tutorial_step=2).tutorial_payload()
    taunt_payload = TutorialBattleEngine(user_id=USER_ID, tutorial_step=8).tutorial_payload()
    lethal_payload = TutorialBattleEngine(user_id=USER_ID, tutorial_step=9).tutorial_payload()

    assert sleep_payload["display_step"] == 2
    assert sleep_payload["display_steps_total"] == TUTORIAL_FINAL_STEP
    assert sleep_payload["player_step"] == sleep_payload["display_step"]
    assert sleep_payload["player_steps_total"] == TUTORIAL_FINAL_STEP
    # no `after` on step1 → previous_message falls back to step1's message
    assert sleep_payload["previous_message"] == "Ставим бойца на поле. Он не атакует сразу: сначала занимает позицию."

    assert taunt_payload["display_step"] == 8
    assert taunt_payload["display_steps_total"] == TUTORIAL_FINAL_STEP
    assert taunt_payload["player_step"] == taunt_payload["display_step"]
    assert taunt_payload["player_steps_total"] == TUTORIAL_FINAL_STEP
    assert taunt_payload["auto_advance_delay_ms"] >= 5000
    assert taunt_payload["previous_message"] == "У Альфонса Провокация. Враг обязан сначала атаковать его."

    assert lethal_payload["display_step"] == 9
    assert lethal_payload["display_steps_total"] == TUTORIAL_FINAL_STEP
    assert lethal_payload["player_step"] == lethal_payload["display_step"]
    assert lethal_payload["player_steps_total"] == TUTORIAL_FINAL_STEP
    assert lethal_payload["previous_message"] == "Враг атакует Альфонса. Вот так работает Провокация — она закрывает героя и ломает план врага."


def test_tutorial_alphonse_step_preserves_card_before_scripted_opponent_attack():
    engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=7)

    result = engine.apply_tutorial_action(
        {"type": "play_card", "card_id": 39, "hand_index": 0}
    )
    state = engine.get_full_state(viewer_id=USER_ID)

    assert result["success"] is True
    assert result["tutorial_step"] == 8
    assert state["tutorial"]["step_id"] == "taunt_demo"
    assert state["tutorial"]["is_auto_step"] is True
    assert state["tutorial"]["player_step"] == 8
    assert state["tutorial"]["auto_advance_delay_ms"] >= 5000
    assert [card["card_id"] for card in state["player"]["board"]] == [37, 39]
    assert state["current_player_id"] == engine.bot_id
    assert state["is_my_turn"] is False


def test_tutorial_auto_taunt_demo_shows_visible_opponent_attack_before_lethal():
    engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=8)

    result = engine.apply_tutorial_action({"type": "auto_continue"})
    state = engine.get_full_state(viewer_id=USER_ID)

    assert result["success"] is True
    assert result["tutorial_step"] == 9
    assert result["after_message"] == "Враг атакует Альфонса. Вот так работает Провокация — она закрывает героя и ломает план врага."
    assert state["tutorial"]["step_id"] == "lethal"
    assert state["tutorial"]["player_step"] == 9
    # the lethal step is a player attack, not an auto step
    assert state["tutorial"]["is_auto_step"] is False
    # Real engine death processing removes Альфонс (taunt absorbed the hit);
    # the opponent attack is still visible (Стив's hp drops, Альфонс gone).
    assert [card["card_id"] for card in state["player"]["board"]] == [37]
    assert state["is_my_turn"] is True


@pytest.mark.asyncio(loop_scope="function")
async def test_tutorial_battle_state_recovers_after_deleted_user(monkeypatch):
    db = _TutorialRouteDB()
    match_id = f"tutorial-{USER_ID}"
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.get(f"/api/battle/state?match_id={match_id}&user_id={USER_ID}")
        body = await response.json()

        assert response.status == 200
        assert body["is_onboarding_tutorial"] is True
        assert body["tutorial"]["step_index"] == 0
        assert db.ensure_user_calls == [USER_ID]
        assert db.state["status"] == "tutorial_battle"
        assert db.state["tutorial_match_id"] == match_id
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_idle_tutorial_runtime_eviction_preserves_progress_and_resumes(monkeypatch):
    match_id = f"tutorial-{USER_ID}"
    sid = "tutorial-idle-sid"
    disconnect_key = (match_id, USER_ID)

    class ExistingTutorialUserDB(_TutorialRouteDB):
        async def fetchval(self, query, *args):
            if "SELECT 1 FROM users" in query:
                return 1
            return await super().fetchval(query, *args)

    db = ExistingTutorialUserDB({
        "status": "tutorial_battle",
        "current_step": "tutorial_battle",
        "tutorial_step": 4,
        "tutorial_match_id": match_id,
    })
    web_server.ENDED_MATCH_IDS.discard(match_id)
    web_server.ENDED_MATCH_TIMES.pop(match_id, None)
    client = await _onboarding_route_client(monkeypatch, db)
    bot_task = None
    disconnect_task = None

    try:
        initial_response = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id={USER_ID}"
        )
        initial_state = await initial_response.json()
        assert initial_response.status == 200
        engine = web_server.ACTIVE_MATCHES[match_id]

        bot_task = asyncio.create_task(asyncio.Event().wait())
        disconnect_task = asyncio.create_task(asyncio.Event().wait())
        web_server.BOT_TASKS[match_id] = bot_task
        web_server.BOT_TASK_KEYS[match_id] = (str(engine.bot_id), int(engine.turn))
        web_server.BOT_TASK_OWNERS[match_id] = client.app
        web_server.MATCH_DISCONNECT_TASKS[disconnect_key] = disconnect_task
        web_server.MATCH_DISCONNECT_STATES[disconnect_key] = {
            "disconnected_at": time.time(),
            "timed_out_turns": 0,
            "takeover_started": False,
        }
        web_server.MATCH_SESSIONS[match_id] = {USER_ID: {sid}}
        web_server.SID_TO_MATCH[sid] = {"match_id": match_id, "user_id": USER_ID}
        web_server.MATCH_INIT_LOCKS[match_id] = asyncio.Lock()

        durable_state_before = dict(db.state)
        events_before = list(db.events)
        timeout_counts_before = (
            engine.p1_consecutive_timeouts,
            engine.p2_consecutive_timeouts,
        )
        replacement_before = engine.get_player_replacement_status(USER_ID)
        clock = time.monotonic()
        web_server._touch_onboarding_tutorial_runtime(
            engine,
            now_monotonic=clock - 11,
            ttl_seconds=10,
        )

        evicted = await web_server._evict_idle_onboarding_tutorial_runtimes(
            client.app,
            now_monotonic=clock,
            ttl_seconds=10,
        )

        assert evicted == 1
        assert match_id not in web_server.ACTIVE_MATCHES
        assert match_id not in client.app["active_matches"]
        assert match_id not in client.app["match_game_modes"]
        assert match_id not in web_server.MATCH_SESSIONS
        assert sid not in web_server.SID_TO_MATCH
        assert disconnect_key not in web_server.MATCH_DISCONNECT_TASKS
        assert disconnect_key not in web_server.MATCH_DISCONNECT_STATES
        assert match_id not in web_server.BOT_TASKS
        assert match_id not in web_server.BOT_TASK_KEYS
        assert match_id not in web_server.BOT_TASK_OWNERS
        assert match_id not in web_server.MATCH_LOCKS
        assert match_id not in web_server.MATCH_INIT_LOCKS
        assert bot_task.cancelled()
        assert disconnect_task.cancelled()
        assert engine.runtime_evicted is True
        assert engine.is_ended is False
        assert not getattr(engine, "runtime_closed", False)
        assert match_id not in web_server.ENDED_MATCH_IDS
        assert dict(db.state) == durable_state_before
        assert db.events == events_before
        assert (
            engine.p1_consecutive_timeouts,
            engine.p2_consecutive_timeouts,
        ) == timeout_counts_before
        assert engine.get_player_replacement_status(USER_ID) == replacement_before

        resumed_response = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id={USER_ID}"
        )
        resumed_state = await resumed_response.json()
        resumed_engine = web_server.ACTIVE_MATCHES[match_id]

        assert resumed_response.status == 200
        assert resumed_engine is not engine
        assert resumed_engine.match_id == engine.match_id == match_id
        assert resumed_engine.tutorial_step == engine.tutorial_step == 4
        assert dict(db.state) == durable_state_before
        assert db.events == events_before
        for key in ("current_player_id", "player", "opponent", "legal_actions", "tutorial"):
            assert resumed_state[key] == initial_state[key]
    finally:
        for task in (bot_task, disconnect_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (bot_task, disconnect_task) if task is not None),
            return_exceptions=True,
        )
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.MATCH_INIT_LOCKS.pop(match_id, None)
        web_server.MATCH_SESSIONS.pop(match_id, None)
        web_server.SID_TO_MATCH.pop(sid, None)
        web_server.MATCH_DISCONNECT_TASKS.pop(disconnect_key, None)
        web_server.MATCH_DISCONNECT_STATES.pop(disconnect_key, None)
        web_server.BOT_TASKS.pop(match_id, None)
        web_server.BOT_TASK_KEYS.pop(match_id, None)
        web_server.BOT_TASK_OWNERS.pop(match_id, None)
        web_server.ENDED_MATCH_IDS.discard(match_id)
        web_server.ENDED_MATCH_TIMES.pop(match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_tutorial_state_interaction_refreshes_idle_deadline(monkeypatch):
    match_id = f"tutorial-{USER_ID}"
    db = _TutorialRouteDB({
        "status": "tutorial_battle",
        "current_step": "tutorial_battle",
        "tutorial_step": 2,
        "tutorial_match_id": match_id,
    })
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        first = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id={USER_ID}"
        )
        assert first.status == 200
        engine = web_server.ACTIVE_MATCHES[match_id]

        ttl_seconds = 5.0
        clock = time.monotonic()
        web_server._touch_onboarding_tutorial_runtime(
            engine,
            now_monotonic=clock - ttl_seconds + 0.1,
            ttl_seconds=ttl_seconds,
        )
        old_deadline = engine.onboarding_runtime_idle_deadline_monotonic

        interaction = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id={USER_ID}"
        )
        assert interaction.status == 200
        refreshed_at = engine.onboarding_runtime_last_activity_monotonic
        assert refreshed_at >= clock
        assert engine.onboarding_runtime_idle_deadline_monotonic > old_deadline

        assert await web_server._evict_idle_onboarding_tutorial_runtimes(
            client.app,
            now_monotonic=old_deadline + 1,
            ttl_seconds=ttl_seconds,
        ) == 0
        assert web_server.ACTIVE_MATCHES[match_id] is engine

        assert await web_server._evict_idle_onboarding_tutorial_runtimes(
            client.app,
            now_monotonic=refreshed_at + ttl_seconds,
            ttl_seconds=ttl_seconds,
        ) == 1
        assert match_id not in web_server.ACTIVE_MATCHES
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ENDED_MATCH_IDS.discard(match_id)
        web_server.ENDED_MATCH_TIMES.pop(match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_tutorial_battle_action_recovers_engine_from_onboarding_state(monkeypatch):
    match_id = f"tutorial-{USER_ID}"
    db = _TutorialRouteDB({
        "status": "tutorial_battle",
        "current_step": "tutorial_battle",
        "tutorial_step": 1,
        "tutorial_match_id": match_id,
    })
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.post(
            "/api/battle/play-card",
            json={
                "match_id": match_id,
                "user_id": USER_ID,
                "hand_index": 0,
                "client_action_id": "tutorial-play-card-1",
            },
        )
        body = await response.json()

        assert response.status == 200
        assert body["result"]["success"] is True
        assert body["result"]["tutorial_step"] == 2
        assert body["state"]["is_onboarding_tutorial"] is True
        assert db.state["tutorial_step"] == 2
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_tutorial_final_attack_waits_for_victory_cta(monkeypatch):
    match_id = f"tutorial-{USER_ID}"
    db = _TutorialRouteDB({
        "status": "tutorial_battle",
        "current_step": "tutorial_battle",
        "tutorial_step": TUTORIAL_FINAL_STEP - 1,
        "tutorial_match_id": match_id,
    })
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        state_response = await client.get(f"/api/battle/state?match_id={match_id}&user_id={USER_ID}")
        state_body = await state_response.json()
        attacker_id = state_body["tutorial"]["attacker_instance_id"]

        response = await client.post(
            "/api/battle/attack",
            json={
                "match_id": match_id,
                "user_id": USER_ID,
                "attacker_id": attacker_id,
                "target_is_hero": True,
                "client_action_id": "tutorial-lethal-attack-1",
            },
        )
        body = await response.json()

        assert response.status == 200
        assert body["result"]["success"] is True
        assert body["result"]["tutorial_step"] == TUTORIAL_FINAL_STEP
        assert body["result"]["game_over"] is True
        assert body["state"]["is_onboarding_tutorial"] is True
        assert body["state"]["tutorial"]["step_index"] == TUTORIAL_FINAL_STEP
        assert db.state["status"] == "tutorial_battle"
        assert db.state["tutorial_step"] == TUTORIAL_FINAL_STEP
        assert all(event[1] != "menu_tour_started" for event in db.events)
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_tutorial_final_complete_endpoint_moves_to_menu_tour(monkeypatch):
    match_id = f"tutorial-{USER_ID}"
    db = _TutorialRouteDB({
        "status": "tutorial_battle",
        "current_step": "tutorial_battle",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "tutorial_match_id": match_id,
    })
    client = await _onboarding_route_client(monkeypatch, db)
    bot_task = None

    try:
        state_response = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id={USER_ID}"
        )
        assert state_response.status == 200
        engine = web_server.ACTIVE_MATCHES[match_id]
        bot_task = asyncio.create_task(asyncio.Event().wait())
        web_server.BOT_TASKS[match_id] = bot_task
        web_server.BOT_TASK_KEYS[match_id] = (str(engine.bot_id), int(engine.turn))
        web_server.BOT_TASK_OWNERS[match_id] = client.app

        response = await client.post(
            "/api/onboarding/tutorial/action",
            json={
                "match_id": match_id,
                "user_id": USER_ID,
                "type": "complete",
                "client_action_id": "tutorial-complete-menu-1",
            },
        )
        body = await response.json()

        assert response.status == 200
        assert body["result"]["success"] is True
        assert body["result"]["tutorial_step"] == TUTORIAL_FINAL_STEP
        assert body["onboarding"]["status"] == "menu_tour"
        assert body["redirect_url"] == "/?onboarding_menu=1"
        assert "state" not in body
        assert db.state["status"] == "menu_tour"
        assert db.state["menu_step"] == "reward"
        assert match_id not in web_server.ACTIVE_MATCHES
        assert match_id not in client.app["active_matches"]
        assert match_id not in client.app["match_game_modes"]
        assert match_id not in web_server.BOT_TASKS
        assert match_id not in web_server.BOT_TASK_KEYS
        assert match_id not in web_server.BOT_TASK_OWNERS
        assert bot_task.cancelled()
        handoff_events_after_first_complete = [
            event for event in db.events
            if event[1] in {"tutorial_battle_completed", "menu_tour_started"}
        ]

        repeat = await client.post(
            "/api/onboarding/tutorial/action",
            json={
                "match_id": match_id,
                "user_id": USER_ID,
                "type": "complete",
                "client_action_id": "tutorial-complete-menu-2",
            },
        )
        repeat_body = await repeat.json()

        assert repeat.status == 200
        assert repeat_body["onboarding"]["status"] == "menu_tour"
        assert repeat_body["redirect_url"] == "/?onboarding_menu=1"
        assert "state" not in repeat_body
        assert db.state["status"] == "menu_tour"
        assert match_id not in web_server.ACTIVE_MATCHES
        assert match_id not in client.app["active_matches"]
        assert [
            event for event in db.events
            if event[1] in {"tutorial_battle_completed", "menu_tour_started"}
        ] == handoff_events_after_first_complete
    finally:
        if bot_task is not None and not bot_task.done():
            bot_task.cancel()
            await asyncio.gather(bot_task, return_exceptions=True)
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.BOT_TASKS.pop(match_id, None)
        web_server.BOT_TASK_KEYS.pop(match_id, None)
        web_server.BOT_TASK_OWNERS.pop(match_id, None)
        web_server.ENDED_MATCH_IDS.discard(match_id)
        web_server.ENDED_MATCH_TIMES.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_tutorial_complete_preserves_completed_onboarding(monkeypatch):
    match_id = f"tutorial-{USER_ID}"
    db = _TutorialRouteDB({
        "status": "completed",
        "current_step": "completed",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "tutorial_match_id": match_id,
        "completed_at": "2026-06-09T09:00:00+00:00",
    })
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.post(
            "/api/onboarding/tutorial/action",
            json={
                "match_id": match_id,
                "user_id": USER_ID,
                "type": "complete",
                "client_action_id": "tutorial-complete-after-done",
            },
        )
        body = await response.json()

        assert response.status == 200
        assert body["result"]["success"] is True
        assert body["onboarding"]["status"] == "completed"
        assert body["redirect_url"] == "/?onboarding_menu=1"
        assert "state" not in body
        assert db.state["status"] == "completed"
        assert all(
            event[1] not in {"tutorial_battle_completed", "menu_tour_started"}
            for event in db.events
        )
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_onboarding_transition_events_have_one_canonical_server_row(monkeypatch):
    match_id = f"tutorial-{USER_ID}"
    db = _TutorialRouteDB()
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        mirrored_welcome = await client.post(
            f"/api/analytics/onboarding?user_id={USER_ID}",
            json={
                "step": "welcome_completed",
                "completed": True,
                "metadata": {"source": "onboarding_gate"},
            },
        )
        assert mirrored_welcome.status == 200
        assert (await mirrored_welcome.json())["canonical"] is True

        welcome = await client.post(
            f"/api/onboarding/welcome/complete?user_id={USER_ID}"
        )
        assert welcome.status == 200
        repeat_welcome = await client.post(
            f"/api/welcome/mark-shown?user_id={USER_ID}"
        )
        assert repeat_welcome.status == 200

        welcome_events = [event for event in db.events if event[1] == "welcome_completed"]
        assert welcome_events == [
            (USER_ID, "welcome_completed", True, {"source": "onboarding_gate"})
        ]

        db.state.update({
            "status": "menu_tour",
            "current_step": "menu_tour",
            "tutorial_step": TUTORIAL_FINAL_STEP,
            "tutorial_match_id": match_id,
            "menu_step": "done",
        })
        mirrored_complete = await client.post(
            f"/api/analytics/onboarding?user_id={USER_ID}",
            json={
                "step": "mandatory_onboarding_completed",
                "completed": True,
                "metadata": {"source": "menu_tour"},
            },
        )
        assert mirrored_complete.status == 200
        assert (await mirrored_complete.json())["canonical"] is True

        complete = await client.post(f"/api/onboarding/complete?user_id={USER_ID}")
        repeat_complete = await client.post(f"/api/onboarding/complete?user_id={USER_ID}")
        assert complete.status == repeat_complete.status == 200

        complete_events = [
            event for event in db.events
            if event[1] == "mandatory_onboarding_completed"
        ]
        assert complete_events == [
            (USER_ID, "mandatory_onboarding_completed", True, {"source": "menu_tour"})
        ]
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_duplicate_tutorial_start_reuses_runtime_and_event(monkeypatch):
    match_id = f"tutorial-{USER_ID}"

    class ConcurrentStartDB(_TutorialRouteDB):
        def __init__(self):
            super().__init__({
                "status": "tutorial_battle",
                "current_step": "tutorial_battle",
                "tutorial_step": 0,
                "tutorial_match_id": match_id,
            })
            self.first_start_event = asyncio.Event()
            self.release_first_start = asyncio.Event()

        async def track_onboarding_event(
            self,
            user_id,
            event_name,
            completed=False,
            metadata=None,
        ):
            if (
                event_name == "tutorial_battle_started"
                and not self.first_start_event.is_set()
            ):
                self.first_start_event.set()
                await self.release_first_start.wait()
            await super().track_onboarding_event(
                user_id,
                event_name,
                completed,
                metadata,
            )

    db = ConcurrentStartDB()
    client = await _onboarding_route_client(monkeypatch, db)
    try:
        first = asyncio.create_task(
            client.post(f"/api/onboarding/tutorial/start?user_id={USER_ID}")
        )
        await asyncio.wait_for(db.first_start_event.wait(), timeout=1)
        duplicate = asyncio.create_task(
            client.post(f"/api/onboarding/tutorial/start?user_id={USER_ID}")
        )
        await asyncio.sleep(0)
        db.release_first_start.set()
        first_response, duplicate_response = await asyncio.gather(
            first,
            duplicate,
        )
        first_body = await first_response.json()
        duplicate_body = await duplicate_response.json()

        assert first_response.status == duplicate_response.status == 200
        assert first_body["match_id"] == duplicate_body["match_id"] == match_id
        assert first_body["state"]["is_onboarding_tutorial"] is True
        assert duplicate_body["state"]["is_onboarding_tutorial"] is True
        assert web_server.ACTIVE_MATCHES[match_id] is client.app["active_matches"][match_id]
        assert [
            event
            for event in db.events
            if event[1] == "tutorial_battle_started"
        ] == [(USER_ID, "tutorial_battle_started", True, {})]
    finally:
        db.release_first_start.set()
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_tutorial_start_waiting_on_complete_does_not_recreate_runtime(monkeypatch):
    match_id = f"tutorial-{USER_ID}"

    class StartVsCompleteDB(_TutorialRouteDB):
        def __init__(self):
            super().__init__({
                "status": "tutorial_battle",
                "current_step": "tutorial_battle",
                "tutorial_step": TUTORIAL_FINAL_STEP,
                "tutorial_match_id": match_id,
            })
            self.menu_transition_reached = asyncio.Event()
            self.release_completion = asyncio.Event()

        async def track_onboarding_event(
            self,
            user_id,
            event_name,
            completed=False,
            metadata=None,
        ):
            if event_name == "menu_tour_started":
                self.menu_transition_reached.set()
                await self.release_completion.wait()
            await super().track_onboarding_event(
                user_id,
                event_name,
                completed,
                metadata,
            )

    db = StartVsCompleteDB()
    client = await _onboarding_route_client(monkeypatch, db)
    try:
        state_response = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id={USER_ID}"
        )
        assert state_response.status == 200
        assert match_id in web_server.ACTIVE_MATCHES

        completion = asyncio.create_task(
            client.post(
                "/api/onboarding/tutorial/action",
                json={
                    "match_id": match_id,
                    "user_id": USER_ID,
                    "type": "complete",
                    "client_action_id": "complete-racing-start",
                },
            )
        )
        await asyncio.wait_for(db.menu_transition_reached.wait(), timeout=1)
        late_start = asyncio.create_task(
            client.post(f"/api/onboarding/tutorial/start?user_id={USER_ID}")
        )
        await asyncio.sleep(0)
        assert late_start.done() is False
        db.release_completion.set()
        complete_response, start_response = await asyncio.gather(
            completion,
            late_start,
        )
        complete_body = await complete_response.json()
        start_body = await start_response.json()

        assert complete_response.status == start_response.status == 200
        assert complete_body["onboarding"]["status"] == "menu_tour"
        assert start_body["onboarding"]["status"] == "menu_tour"
        assert start_body["redirect_url"] == "/?onboarding_menu=1"
        assert "match_id" not in start_body
        assert "state" not in start_body
        assert match_id not in web_server.ACTIVE_MATCHES
        assert match_id not in client.app["active_matches"]
        assert all(
            event[1] != "tutorial_battle_started"
            for event in db.events
        )
    finally:
        db.release_completion.set()
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_state_racing_completion_does_not_return_detached_runtime(monkeypatch):
    match_id = f"tutorial-{USER_ID}"

    class StateVsCompleteDB(_TutorialRouteDB):
        def __init__(self):
            super().__init__({
                "status": "tutorial_battle",
                "current_step": "tutorial_battle",
                "tutorial_step": TUTORIAL_FINAL_STEP,
                "tutorial_match_id": match_id,
            })
            self.pause_next_state_read = False
            self.state_snapshot_captured = asyncio.Event()
            self.release_state_snapshot = asyncio.Event()

        async def get_onboarding_state(self, user_id):
            # Capture the exact stale-snapshot interleaving that used to let a
            # battle-state request return (or recreate) a detached tutorial
            # runtime after completion had already advanced to menu_tour.
            state = await super().get_onboarding_state(user_id)
            if self.pause_next_state_read:
                self.pause_next_state_read = False
                self.state_snapshot_captured.set()
                await self.release_state_snapshot.wait()
            return state

    db = StateVsCompleteDB()
    client = await _onboarding_route_client(monkeypatch, db)
    state_request = None
    try:
        initial_state = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id={USER_ID}"
        )
        assert initial_state.status == 200
        assert match_id in web_server.ACTIVE_MATCHES

        db.pause_next_state_read = True
        state_request = asyncio.create_task(
            client.get(f"/api/battle/state?match_id={match_id}&user_id={USER_ID}")
        )
        await asyncio.wait_for(db.state_snapshot_captured.wait(), timeout=1)

        completion = await client.post(
            "/api/onboarding/tutorial/action",
            json={
                "match_id": match_id,
                "user_id": USER_ID,
                "type": "complete",
                "client_action_id": "complete-racing-battle-state",
            },
        )
        completion_body = await completion.json()
        assert completion.status == 200
        assert completion_body["onboarding"]["status"] == "menu_tour"
        assert match_id not in web_server.ACTIVE_MATCHES

        db.release_state_snapshot.set()
        response = await asyncio.wait_for(state_request, timeout=1)
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "onboarding_required"
        assert body["onboarding"]["status"] == "menu_tour"
        assert match_id not in web_server.ACTIVE_MATCHES
        assert match_id not in client.app["active_matches"]
    finally:
        db.release_state_snapshot.set()
        if state_request is not None and not state_request.done():
            state_request.cancel()
            await asyncio.gather(state_request, return_exceptions=True)
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_battle_state_waits_when_menu_is_persisted_before_runtime_teardown(monkeypatch):
    match_id = f"tutorial-{USER_ID}"

    class StateAfterMenuTransitionDB(_TutorialRouteDB):
        def __init__(self):
            super().__init__({
                "status": "tutorial_battle",
                "current_step": "tutorial_battle",
                "tutorial_step": TUTORIAL_FINAL_STEP,
                "tutorial_match_id": match_id,
            })
            self.menu_transition_reached = asyncio.Event()
            self.release_completion = asyncio.Event()

        async def track_onboarding_event(
            self,
            user_id,
            event_name,
            completed=False,
            metadata=None,
        ):
            if event_name == "menu_tour_started":
                # At this point menu_tour is durable but completion still owns
                # the match lock and has not torn down the runtime yet.
                self.menu_transition_reached.set()
                await self.release_completion.wait()
            await super().track_onboarding_event(
                user_id,
                event_name,
                completed,
                metadata,
            )

    db = StateAfterMenuTransitionDB()
    client = await _onboarding_route_client(monkeypatch, db)
    completion_task = None
    state_request = None
    try:
        initial_state = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id={USER_ID}"
        )
        assert initial_state.status == 200
        assert match_id in web_server.ACTIVE_MATCHES

        completion_task = asyncio.create_task(
            client.post(
                "/api/onboarding/tutorial/action",
                json={
                    "match_id": match_id,
                    "user_id": USER_ID,
                    "type": "complete",
                    "client_action_id": "complete-before-runtime-teardown",
                },
            )
        )
        await asyncio.wait_for(db.menu_transition_reached.wait(), timeout=1)
        assert db.state["status"] == "menu_tour"
        assert match_id in web_server.ACTIVE_MATCHES

        state_request = asyncio.create_task(
            client.get(f"/api/battle/state?match_id={match_id}&user_id={USER_ID}")
        )
        await asyncio.sleep(0)
        assert state_request.done() is False

        db.release_completion.set()
        completion_response, state_response = await asyncio.gather(
            completion_task,
            state_request,
        )
        state_body = await state_response.json()

        assert completion_response.status == 200
        assert state_response.status == 403
        assert state_body["error"] == "onboarding_required"
        assert state_body["onboarding"]["status"] == "menu_tour"
        assert match_id not in web_server.ACTIVE_MATCHES
        assert match_id not in client.app["active_matches"]
    finally:
        db.release_completion.set()
        pending = [
            task
            for task in (completion_task, state_request)
            if task is not None and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()


@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_menu_step_retry_is_idempotent_and_tracks_one_event(monkeypatch):
    match_id = f"tutorial-{USER_ID}"

    class ConcurrentMenuDB(_TutorialRouteDB):
        def __init__(self):
            super().__init__({
                "status": "menu_tour",
                "current_step": "menu_tour",
                "tutorial_step": TUTORIAL_FINAL_STEP,
                "tutorial_match_id": match_id,
                "menu_step": "reward",
            })
            self.first_event_started = asyncio.Event()
            self.release_first_event = asyncio.Event()

        async def track_onboarding_event(
            self,
            user_id,
            event_name,
            completed=False,
            metadata=None,
        ):
            if (
                event_name == "menu_tour_step_completed"
                and not self.first_event_started.is_set()
            ):
                self.first_event_started.set()
                await self.release_first_event.wait()
            await super().track_onboarding_event(
                user_id,
                event_name,
                completed,
                metadata,
            )

    db = ConcurrentMenuDB()
    client = await _onboarding_route_client(monkeypatch, db)
    lock_key = web_server.tutorial_match_id_for_user(USER_ID)
    try:
        first = asyncio.create_task(
            client.post(
                f"/api/onboarding/menu-tour/step?user_id={USER_ID}",
                json={"step_id": "reward"},
            )
        )
        await asyncio.wait_for(db.first_event_started.wait(), timeout=1)
        retry = asyncio.create_task(
            client.post(
                f"/api/onboarding/menu-tour/step?user_id={USER_ID}",
                json={"step_id": "reward"},
            )
        )
        await asyncio.sleep(0)
        db.release_first_event.set()
        first_response, retry_response = await asyncio.gather(first, retry)
        first_body = await first_response.json()
        retry_body = await retry_response.json()

        assert first_response.status == retry_response.status == 200
        assert first_body["success"] is retry_body["success"] is True
        assert first_body["onboarding"]["menu_step"] == "arena"
        assert retry_body["onboarding"]["menu_step"] == "arena"
        assert db.state["menu_step"] == "arena"
        assert [
            event
            for event in db.events
            if event[1] == "menu_tour_step_completed"
        ] == [
            (
                USER_ID,
                "menu_tour_step_completed",
                True,
                {"step": "reward"},
            )
        ]
    finally:
        db.release_first_event.set()
        await client.close()
        web_server.MATCH_LOCKS.pop(lock_key, None)


@pytest.mark.asyncio
async def test_newbie_path_claim_accepts_jsonb_progress_returned_as_string():
    progress = {
        "tasks": {
            "view_new_card": {
                "completed": True,
                "completed_at": "2026-06-09T09:00:00+00:00",
            }
        }
    }
    conn = _NewbiePathClaimConn(json.dumps(progress))
    db = _NewbiePathClaimDB(conn)

    result = await db.claim_newbie_path_task(
        USER_ID,
        "view_new_card",
        reward_coins=50,
        require_completed=True,
    )

    assert result["success"] is True
    assert result["granted_amount"] == 50
    saved_progress = json.loads(conn.progress)
    assert saved_progress["tasks"]["view_new_card"]["completed"] is True
    assert saved_progress["tasks"]["view_new_card"]["claimed"] is True


@pytest.mark.asyncio
async def test_newbie_path_duplicate_claim_is_idempotent_without_second_grant():
    progress = {
        "tasks": {
            "view_new_card": {
                "completed": True,
                "claimed": True,
                "completed_at": "2026-06-09T09:00:00+00:00",
            }
        }
    }
    conn = _NewbiePathClaimConn(json.dumps(progress))
    db = _NewbiePathClaimDB(conn)

    result = await db.claim_newbie_path_task(
        USER_ID,
        "view_new_card",
        reward_coins=50,
        require_completed=True,
    )

    assert result["success"] is True
    assert result["already_claimed"] is True
    assert result["granted_amount"] == 0
    assert conn.coins_granted == 0


@pytest.mark.asyncio
async def test_newbie_path_claim_without_onboarding_row_does_not_grant_reward():
    conn = _NewbiePathClaimConn(None, row_exists=False)
    db = _NewbiePathClaimDB(conn)

    result = await db.claim_newbie_path_task(
        USER_ID,
        "claim_newbie_reward",
        reward_coins=150,
        require_completed=False,
    )

    assert result["success"] is False
    assert result["error"] == "onboarding_not_found"
    assert conn.coins_granted == 0
    assert all("UPDATE users SET coins" not in query for query, _args in conn.executed)


@pytest.mark.asyncio
async def test_newbie_path_mark_task_locks_row_and_preserves_claimed_state():
    progress = {
        "tasks": {
            "view_new_card": {
                "completed": True,
                "claimed": True,
                "completed_at": "2026-06-09T09:00:00+00:00",
            }
        }
    }
    conn = _NewbiePathClaimConn(json.dumps(progress))
    db = _NewbiePathClaimDB(conn)

    result = await db.mark_newbie_path_task(USER_ID, "view_new_card", claimed=False)

    assert result["newbie_path_progress"]["tasks"]["view_new_card"]["claimed"] is True
    saved_progress = json.loads(conn.progress)
    assert saved_progress["tasks"]["view_new_card"]["claimed"] is True
    assert any("FOR UPDATE" in query for query, _args in conn.fetches)


@pytest.mark.asyncio(loop_scope="function")
async def test_newbie_path_route_duplicate_claim_reports_no_grant_without_claim_event(monkeypatch):
    db = _TutorialRouteDB({
        "status": "completed",
        "current_step": "completed",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "newbie_path_progress": {
            "tasks": {
                "view_new_card": {
                    "completed": True,
                    "claimed": True,
                    "completed_at": "2026-06-09T09:00:00+00:00",
                }
            }
        },
    })
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.post(
            f"/api/onboarding/newbie-path?user_id={USER_ID}",
            json={"task_id": "view_new_card"},
        )
        body = await response.json()

        assert response.status == 200
        assert body["already_claimed"] is True
        assert body["granted_amount"] == 0
        assert db.coins_granted == 0
        assert all(event[1] != "newbie_path_task_claimed" for event in db.events)
    finally:
        await client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_newbie_path_telegram_init_data_includes_channel_task(monkeypatch):
    db = _TutorialRouteDB({
        "status": "completed",
        "current_step": "completed",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "newbie_path_progress": {},
    })
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.get(f"/api/onboarding/status?_auth={_signed_auth_query()}")
        body = await response.json()

        assert response.status == 200
        task_ids = [task["id"] for task in body["onboarding"]["newbie_path"]["tasks"]]
        assert "join_telegram_channel" in task_ids
        assert task_ids[-2] == "join_telegram_channel"
        channel_task = next(task for task in body["onboarding"]["newbie_path"]["tasks"] if task["id"] == "join_telegram_channel")
        assert channel_task["action_url"] == "https://t.me/extraarena"
        assert channel_task["reward"]["amount"] == 100
    finally:
        await client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_newbie_path_android_dev_excludes_channel_task_and_can_finish(monkeypatch):
    db = _TutorialRouteDB({
        "status": "completed",
        "current_step": "completed",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "newbie_path_progress": _completed_original_newbie_progress(),
    })
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        status_response = await client.get(
            f"/api/onboarding/status?user_id={USER_ID}&ea_platform=android_app&ea_shell=android&ea_telegram=0"
        )
        status_body = await status_response.json()
        task_ids = [task["id"] for task in status_body["onboarding"]["newbie_path"]["tasks"]]

        assert status_response.status == 200
        assert "join_telegram_channel" not in task_ids
        assert task_ids == [
            "open_starter_case",
            "view_new_card",
            "save_first_deck",
            "play_regular_battle",
            "claim_newbie_reward",
        ]

        claim_response = await client.post(
            f"/api/onboarding/newbie-path?user_id={USER_ID}&ea_platform=android_app&ea_shell=android&ea_telegram=0",
            json={"task_id": "claim_newbie_reward"},
        )
        claim_body = await claim_response.json()

        assert claim_response.status == 200
        assert claim_body["granted_amount"] == 150
        assert db.coins_granted == 150
        assert all(task["claimed"] for task in claim_body["newbie_path"]["tasks"])
    finally:
        await client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_newbie_path_android_dev_rejects_hidden_channel_task(monkeypatch):
    db = _TutorialRouteDB({
        "status": "completed",
        "current_step": "completed",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "newbie_path_progress": {},
    })
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.post(
            f"/api/onboarding/newbie-path?user_id={USER_ID}&ea_platform=android_app&ea_shell=android&ea_telegram=0",
            json={"task_id": "join_telegram_channel"},
        )
        body = await response.json()

        assert response.status == 400
        assert body["error"] == "invalid_task"
        assert db.coins_granted == 0
    finally:
        await client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_newbie_path_telegram_channel_membership_success_grants_reward(monkeypatch):
    db = _TutorialRouteDB({
        "status": "completed",
        "current_step": "completed",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "newbie_path_progress": {},
    })
    fake_session = _FakeTelegramSession([
        {"ok": True, "result": {"status": "member"}},
    ])
    monkeypatch.setattr(web_server, "_create_telegram_api_session", lambda: fake_session)
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.post(
            f"/api/onboarding/newbie-path?_auth={_signed_auth_query()}",
            json={"task_id": "join_telegram_channel"},
        )
        body = await response.json()

        assert response.status == 200
        assert body["granted_amount"] == 100
        assert db.coins_granted == 100
        progress = db.state["newbie_path_progress"]["tasks"]["join_telegram_channel"]
        assert progress["completed"] is True
        assert progress["claimed"] is True
        assert fake_session.requests[0][1]["params"]["chat_id"] == "-1001777559237"
        assert any(event[1] == "newbie_path_task_completed" for event in db.events)
        assert any(event[1] == "newbie_path_task_claimed" for event in db.events)
    finally:
        await client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_newbie_path_telegram_channel_non_member_grants_nothing(monkeypatch):
    db = _TutorialRouteDB({
        "status": "completed",
        "current_step": "completed",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "newbie_path_progress": {},
    })
    monkeypatch.setattr(
        web_server,
        "_create_telegram_api_session",
        lambda: _FakeTelegramSession([{"ok": True, "result": {"status": "left"}}]),
    )
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.post(
            f"/api/onboarding/newbie-path?_auth={_signed_auth_query()}",
            json={"task_id": "join_telegram_channel"},
        )
        body = await response.json()

        assert response.status == 409
        assert body["error"] == "telegram_channel_not_joined"
        assert body["retryable"] is False
        assert db.coins_granted == 0
        assert "join_telegram_channel" not in db.state["newbie_path_progress"].get("tasks", {})
    finally:
        await client.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_newbie_path_telegram_channel_api_failure_grants_nothing(monkeypatch):
    db = _TutorialRouteDB({
        "status": "completed",
        "current_step": "completed",
        "tutorial_step": TUTORIAL_FINAL_STEP,
        "newbie_path_progress": {},
    })
    monkeypatch.setattr(
        web_server,
        "_create_telegram_api_session",
        lambda: _FakeTelegramSession([RuntimeError("telegram down"), RuntimeError("still down")]),
    )
    client = await _onboarding_route_client(monkeypatch, db)

    try:
        response = await client.post(
            f"/api/onboarding/newbie-path?_auth={_signed_auth_query()}",
            json={"task_id": "join_telegram_channel"},
        )
        body = await response.json()

        assert response.status == 409
        assert body["error"] == "telegram_check_failed"
        assert body["retryable"] is True
        assert db.coins_granted == 0
        assert "join_telegram_channel" not in db.state["newbie_path_progress"].get("tasks", {})
    finally:
        await client.close()


def test_tutorial_battle_engine_rejects_malformed_hand_index_without_500():
    engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=1)

    result = engine.apply_tutorial_action(
        {"type": "play_card", "hand_index": "not-a-number"}
    )

    assert result["success"] is False
    assert result["error"] == "tutorial_wrong_action"
    assert result["tutorial_step"] == 1


def test_tutorial_state_exposes_only_current_guided_battle_action():
    expected = {
        0: [],
        1: [("play_card", 0, None, False)],
        2: [("end_turn", None, None, False)],
        3: [],
        4: "choose_target",  # special: hero entry + Стив minion entry (dynamic id)
        5: [],
        6: [],
        7: [("play_card", 0, None, False)],
        8: [],
        9: [("attack", None, None, True)],
        10: [],
    }

    for step, expected_actions in expected.items():
        engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=step)
        state = engine.get_full_state(viewer_id=USER_ID)
        actions = [
            (
                action.get("type"),
                action.get("hand_index"),
                action.get("target_id"),
                bool(action.get("target_is_hero")),
            )
            for action in state["legal_actions"]
        ]
        if expected_actions == "choose_target":
            # hero entry (target_id None, is_hero True) + Стив minion entry (dynamic id)
            assert len(actions) == 2
            assert ("attack", None, None, True) in actions
            steve_id = str(engine._arena.state.p2.board[0].instance_id)
            assert ("attack", None, steve_id, False) in actions
        else:
            assert actions == expected_actions


def test_tutorial_choose_target_advertises_minion_target():
    engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=4)
    actions = engine._tutorial_legal_actions()

    # hero entry + one minion entry per opponent board minion (Стив)
    assert len(actions) == 2
    attacker_id = _tutorial(engine)["attacker_instance_id"]
    steve_id = str(engine._arena.state.p2.board[0].instance_id)

    hero_entry = next(a for a in actions if a["target_is_hero"])
    assert hero_entry == {
        "type": "attack",
        "attacker_id": attacker_id,
        "target_id": None,
        "target_is_hero": True,
    }
    minion_entry = next(a for a in actions if not a["target_is_hero"])
    assert minion_entry == {
        "type": "attack",
        "attacker_id": attacker_id,
        "target_id": steve_id,
        "target_is_hero": False,
    }


def test_tutorial_choose_target_wrong_minion_tap_is_rejected_with_custom_feedback():
    engine = TutorialBattleEngine(user_id=USER_ID, tutorial_step=4)
    slime_id = _tutorial(engine)["attacker_instance_id"]
    steve_id = str(engine._arena.state.p2.board[0].instance_id)
    hero_hp_before = engine._arena.state.p2.hero.hp

    # tapping Стив is mechanically legal but pedagogically wrong → custom feedback, no state change
    result = engine.apply_tutorial_action(
        {"type": "attack", "attacker_id": slime_id, "target_id": steve_id, "target_is_hero": False}
    )
    assert result == {
        "success": False,
        "error": "tutorial_wrong_action",
        "feedback": "Можно, но сейчас выгоднее бить героя: его HP — условие победы.",
        "tutorial_step": 4,
    }
    # no state mutation: hero HP unchanged, still on choose_target
    assert engine._arena.state.p2.hero.hp == hero_hp_before
    assert engine.tutorial_step == 4
    assert _tutorial(engine)["step_id"] == "choose_target"

    # retry by tapping the hero → advances to tempo, real 8→4
    retry = engine.apply_tutorial_action(
        {"type": "attack", "attacker_id": slime_id, "target_is_hero": True}
    )
    assert retry["success"] is True
    assert retry["tutorial_step"] == 5
    assert engine._arena.state.p2.hero.hp == 4
