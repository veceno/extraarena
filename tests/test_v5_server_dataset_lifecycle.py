from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from battle_engine import BattleEngine
from core.actions import EndTurnAction
from core.classic_setup import create_classic_game_state
from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, ReplacementStatus
from web import server as web_server


def _hero(card_id: int) -> CardInstance:
    return CardInstance(
        instance_id=uuid.uuid4(),
        card_id=card_id,
        name=f"Hero {card_id}",
        card_type=CardType.HERO,
        rarity="common",
        mana_cost=0,
        attack=0,
        hp=30,
        max_hp=30,
        mechanics=[],
    )


def _real_engine(
    match_id: str,
    *,
    p1_is_bot: bool = False,
    p2_is_bot: bool = False,
) -> BattleEngine:
    engine = BattleEngine(match_id=match_id, player_ids=[101, 202])
    state = create_classic_game_state(
        101,
        202,
        [_hero(1)],
        [_hero(2)],
        p1_is_bot=p1_is_bot,
        p2_is_bot=p2_is_bot,
    )
    engine._arena = ArenaEnvironment(
        state,
        classic_params=engine.mode_config.classic,
        apply_start_effects=False,
    )
    engine.match_id = match_id
    engine.current_player_id = state.current_turn_owner_id
    engine.turn = state.turn_number
    engine.is_bot_match = p1_is_bot or p2_is_bot
    return engine


class TraceDB:
    """Small in-memory stand-in for the production V5 journal."""

    def __init__(self) -> None:
        self.checkpoints: list[dict[str, Any]] = []
        self.finalizations: list[dict[str, Any]] = []
        self.aborts: list[dict[str, Any]] = []

    async def get_onboarding_state(self, _user_id: int) -> dict[str, Any]:
        return {"completed": True, "status": "completed"}

    async def upsert_v5_battle_trace_checkpoint(
        self,
        *,
        battle_id: str,
        meta: dict[str, Any],
        turns: list[dict[str, Any]],
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        row = {
            "battle_id": str(battle_id),
            "meta": deepcopy(meta),
            "turns": deepcopy(turns),
            "actions": deepcopy(actions),
        }
        self.checkpoints.append(row)
        return {"applied": True, "action_count": len(actions)}

    async def finalize_v5_battle_trace(
        self,
        *,
        battle_id: str,
        status: str,
        winner_user_id: int | None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "battle_id": str(battle_id),
            "status": str(status),
            "winner_user_id": winner_user_id,
            "meta": deepcopy(meta or {}),
        }
        self.finalizations.append(row)
        return {"applied": True, **row}

    async def mark_v5_battle_trace_aborted(
        self,
        *,
        battle_id: str,
        reason: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "battle_id": str(battle_id),
            "reason": str(reason),
            "meta": deepcopy(meta or {}),
        }
        self.aborts.append(row)
        return {"applied": True, **row}


class StreamingExportDB(TraceDB):
    def __init__(self) -> None:
        super().__init__()
        self.bundle_calls: list[tuple[str, bool]] = []

    async def list_v5_export_battle_ids(
        self,
        *,
        days: int,
        limit_battles: int,
    ) -> dict[str, Any]:
        return {
            "format": "extraarena_v5_dataset_export_v1",
            "format_version": 1,
            "storage_schema": "rlhf_v5_storage_v1",
            "created_at": "2026-07-28T00:00:00+00:00",
            "days": int(days),
            "limit_battles": int(limit_battles),
            "battle_ids": ["stream-battle-1", "stream-battle-2"],
        }

    async def get_v5_export_battle_bundle(
        self,
        *,
        battle_id: str,
        include_players: bool,
        record_id_namespace: uuid.UUID,
    ) -> dict[str, Any]:
        self.bundle_calls.append((str(battle_id), bool(include_players)))
        exported_id = (
            str(battle_id)
            if include_players
            else f"record_{uuid.uuid5(record_id_namespace, str(battle_id)).hex}"
        )
        return {
            "battle_id": exported_id,
            "storage_schema": "rlhf_v5_storage_v1",
            "status": "p1_win",
            "finished_at": "2026-07-28T00:00:00+00:00",
            "meta": {
                "battle_id": exported_id,
                "match_id": exported_id,
                "p1_user_id": 1,
                "p2_user_id": 2,
            },
            "turns": [{"turn_number": 1}],
            "actions": [{"seq": 1}],
        }

    async def export_v5_battle_dataset(self, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("HTTP exporter must not materialize the full dataset")


class HTTPActionEngine:
    """Minimal endpoint-facing engine with one configurable end-turn result."""

    def __init__(self, match_id: str, *, accepted: bool) -> None:
        self.match_id = match_id
        self.accepted = bool(accepted)
        self.p1_state = SimpleNamespace(
            user_id=101,
            is_bot=False,
            replacement_status=ReplacementStatus.ACTIVE,
        )
        self.p2_state = SimpleNamespace(
            user_id=202,
            is_bot=False,
            replacement_status=ReplacementStatus.ACTIVE,
        )
        self.current_player_id = 101
        self.turn = 1
        self.is_ended = False
        self.rewards_granted = False
        self.battle_end_processed = False
        self.recorded_contexts: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.decision_clock_arms: list[int] = []

    def is_turn_expired(self) -> bool:
        return False

    def is_waiting_for_players(self) -> bool:
        return False

    def is_bot(self, _user_id: int) -> bool:
        return False

    def get_current_player_id(self) -> int:
        return self.current_player_id

    def get_player_replacement_status(self, _user_id: int) -> ReplacementStatus:
        return ReplacementStatus.ACTIVE

    def arm_human_decision_clock(self, user_id: int) -> None:
        self.decision_clock_arms.append(int(user_id))

    def restore_player_control(self, _user_id: int) -> bool:
        return False

    def record_analytics_action(
        self,
        user_id: int,
        action_json: dict[str, Any],
        **context: Any,
    ) -> dict[str, Any]:
        captured = {
            "user_id": int(user_id),
            "action": deepcopy(action_json),
            **deepcopy(context),
        }
        self.recorded_contexts.append(captured)
        return captured

    def end_turn(self, user_id: int) -> dict[str, Any]:
        context = self.recorded_contexts[-1]
        row = {
            "seq": len(self.actions) + 1,
            "actor_user_id": int(user_id),
            "action_native": deepcopy(context["action"]),
            "accepted": self.accepted,
            "error": None if self.accepted else "not_your_turn",
            "decision_source": context.get("decision_source"),
            "client_action_id": context.get("client_action_id"),
        }
        self.actions.append(row)
        if not self.accepted:
            return {"success": False, "error": "not_your_turn"}
        self.current_player_id = 202
        return {"success": True, "current_player_id": 202}

    def get_full_state(self, viewer_id: int | None = None) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "viewer_id": viewer_id,
            "current_player_id": self.current_player_id,
        }

    def checkpoint_v5_dataset(self, *, reason: str | None = None) -> dict[str, Any]:
        return {
            "meta": {
                "schema_version": "rlhf_v5_storage_v1",
                "battle_id": self.match_id,
                "match_id": self.match_id,
                "status": "ongoing",
                "checkpoint_reason": reason,
            },
            "turns": [],
            "actions": deepcopy(self.actions),
        }


class LifecycleEngine:
    def __init__(self, match_id: str) -> None:
        self.match_id = match_id
        self.is_ended = False
        self.rewards_granted = False
        self.battle_end_processed = False
        self.p1_state = SimpleNamespace(
            user_id=101,
            is_bot=False,
            replacement_status=ReplacementStatus.ACTIVE,
        )
        self.p2_state = SimpleNamespace(
            user_id=202,
            is_bot=False,
            replacement_status=ReplacementStatus.ACTIVE,
        )
        self.runtime_owner_app: Any = None
        self.berserk_brain: Any = None
        self.extra_lr_aux_runtime: Any = None
        self.checkpoint_reasons: list[str | None] = []
        self.finalize_calls: list[dict[str, Any]] = []
        self.abort_calls: list[dict[str, Any]] = []
        self._arena = SimpleNamespace(
            state=SimpleNamespace(
                p1=SimpleNamespace(user_id=101),
                p2=SimpleNamespace(user_id=202),
                status=SimpleNamespace(value="p1_win"),
            )
        )

    def checkpoint_v5_dataset(self, *, reason: str | None = None) -> dict[str, Any]:
        self.checkpoint_reasons.append(reason)
        return {
            "meta": {
                "schema_version": "rlhf_v5_storage_v1",
                "battle_id": self.match_id,
                "match_id": self.match_id,
                "status": "ongoing",
                "checkpoint_reason": reason,
            },
            "turns": [{"turn": 1}],
            "actions": [{"seq": 1, "accepted": True}],
        }

    def finalize_v5_dataset(
        self,
        *,
        winner_user_id: int | None = None,
        status: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        call = {
            "winner_user_id": winner_user_id,
            "status": status,
            "reason": reason,
            "metadata": deepcopy(metadata or {}),
        }
        self.finalize_calls.append(call)
        return {
            "meta": {
                "schema_version": "rlhf_v5_storage_v1",
                "battle_id": self.match_id,
                "match_id": self.match_id,
                "status": status,
                "winner_user_id": winner_user_id,
                "terminal_reason": reason,
            },
            "turns": [{"turn": 1}],
            "actions": [{"seq": 1, "accepted": True}],
        }

    def abort_v5_dataset(
        self,
        reason: str,
        *,
        status: str = "aborted",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        call = {
            "reason": str(reason),
            "status": str(status),
            "metadata": deepcopy(metadata or {}),
        }
        self.abort_calls.append(call)
        return {
            "meta": {
                "schema_version": "rlhf_v5_storage_v1",
                "battle_id": self.match_id,
                "match_id": self.match_id,
                "status": "aborted",
                "abort_reason": str(reason),
            },
            "turns": [{"turn": 1}],
            "actions": [{"seq": 1, "accepted": True}],
        }

    def is_bot(self, _user_id: int) -> bool:
        return False


async def _battle_client(
    monkeypatch: pytest.MonkeyPatch,
    db: TraceDB,
    engine: HTTPActionEngine,
) -> TestClient:
    async def no_bot_move(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(web_server, "check_and_run_bot", no_bot_move)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()
    app = web_server.create_web_app(db, bot_token="bot-token")
    engine.runtime_owner_app = app
    web_server.ACTIVE_MATCHES[engine.match_id] = engine
    app["active_matches"][engine.match_id] = engine
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.parametrize(
    ("accepted", "expected_status"),
    [(True, 200), (False, 409)],
)
@pytest.mark.asyncio(loop_scope="function")
async def test_http_human_action_checkpoints_accepted_and_rejected_rows(
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
    expected_status: int,
) -> None:
    match_id = f"v5-http-{'accepted' if accepted else 'rejected'}"
    db = TraceDB()
    engine = HTTPActionEngine(match_id, accepted=accepted)
    client = await _battle_client(monkeypatch, db, engine)

    try:
        response = await client.post(
            "/api/battle/end-turn",
            json={
                "match_id": match_id,
                "user_id": 101,
                "client_action_id": f"{match_id}-nonce",
            },
        )

        assert response.status == expected_status
        assert len(engine.recorded_contexts) == 1
        captured = engine.recorded_contexts[0]
        assert captured["decision_source"] == "human"
        assert captured["client_action_id"] == f"{match_id}-nonce"
        assert isinstance(captured["request_monotonic"], float)
        assert db.checkpoints
        persisted = db.checkpoints[-1]["actions"]
        assert len(persisted) == 1
        assert persisted[0]["accepted"] is accepted
        assert persisted[0]["client_action_id"] == f"{match_id}-nonce"
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()
        web_server.ENDED_MATCH_IDS.discard(match_id)


@pytest.mark.asyncio(loop_scope="function")
async def test_http_state_delivery_arms_reconnected_human_decision_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    match_id = "v5-http-state-clock"
    db = TraceDB()
    engine = HTTPActionEngine(match_id, accepted=True)
    client = await _battle_client(monkeypatch, db, engine)

    try:
        response = await client.get(
            f"/api/battle/state?match_id={match_id}&user_id=101"
        )

        assert response.status == 200
        assert engine.decision_clock_arms == [101]
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()
        web_server.ENDED_MATCH_IDS.discard(match_id)


@pytest.mark.asyncio(loop_scope="function")
async def test_admin_v5_dataset_export_streams_one_bundle_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    match_id = "v5-http-stream-export"
    db = StreamingExportDB()
    engine = HTTPActionEngine(match_id, accepted=True)
    client = await _battle_client(monkeypatch, db, engine)

    try:
        response = await client.get(
            "/api/admin/analytics/dataset/export"
            "?days=7&limit_battles=2",
            headers={
                "Cookie": (
                    f"{web_server.ADMIN_SESSION_COOKIE_NAME}="
                    f"{web_server._make_admin_session_token(web_server.ADMIN_ID)}"
                )
            },
        )
        payload = await response.text()

        assert response.status == 200
        assert response.content_type == "application/x-ndjson"
        records = [
            json.loads(line)
            for line in payload.splitlines()
            if line.strip()
        ]
        assert records[0]["record_type"] == "header"
        assert records[0]["battle_count"] == 2
        assert records[0]["privacy"] == "side_pseudonyms_p1_1_p2_2"
        assert records[0]["record_id_scheme"] == (
            "random_per_export_record_ids_v1"
        )
        exported_ids = [row["battle_id"] for row in records[1:]]
        assert len(set(exported_ids)) == 2
        assert all(
            re.fullmatch(r"record_[0-9a-f]{32}", value)
            for value in exported_ids
        )
        assert "stream-battle" not in payload
        assert db.bundle_calls == [
            ("stream-battle-1", False),
            ("stream-battle-2", False),
        ]
    finally:
        await client.close()
        web_server.ACTIVE_MATCHES.pop(match_id, None)
        web_server.MATCH_LOCKS.pop(match_id, None)
        web_server.ACTION_RESULT_CACHE.clear()
        web_server.ENDED_MATCH_IDS.discard(match_id)


@pytest.mark.asyncio(loop_scope="function")
async def test_terminal_trace_is_sealed_even_when_rewards_were_already_processed() -> None:
    db = TraceDB()
    engine = LifecycleEngine("v5-terminal-idempotent")
    engine.is_ended = True
    engine.battle_end_processed = True
    engine.rewards_granted = True
    app = {
        "db": db,
        "matchmaker": None,
        "match_game_modes": {},
    }

    try:
        processed = await web_server._process_battle_end(
            app,
            engine.match_id,
            engine,
            winner_id=101,
        )

        assert processed is True
        assert len(engine.finalize_calls) == 1
        assert engine.finalize_calls[0]["status"] == "p1_win"
        assert engine.finalize_calls[0]["winner_user_id"] == 101
        assert db.checkpoints[-1]["battle_id"] == engine.match_id
        assert db.finalizations == [
            {
                "battle_id": engine.match_id,
                "status": "p1_win",
                "winner_user_id": 101,
                "meta": db.finalizations[0]["meta"],
            }
        ]
    finally:
        web_server.ENDED_MATCH_IDS.discard(engine.match_id)
        web_server.ENDED_MATCH_TIMES.pop(engine.match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_control_events_survive_checkpoint_in_canonical_meta() -> None:
    db = TraceDB()
    engine = _real_engine("v5-control-events")
    engine.set_player_replacement_status(
        101,
        ReplacementStatus.AFK,
        reason="disconnect",
    )
    app = {"db": db, "active_matches": {engine.match_id: engine}}

    result = await web_server._persist_v5_dataset_checkpoint(
        app,
        engine,
        reason="control_change",
    )

    assert result["applied"] is True
    events = db.checkpoints[-1]["meta"]["control_events"]
    assert len(events) == 1
    assert events[0]["user_id"] == 101
    assert events[0]["new_status"] == "afk"
    assert events[0]["reason"] == "disconnect"


@pytest.mark.asyncio(loop_scope="function")
async def test_onboarding_tutorial_is_excluded_from_production_v5_journal() -> None:
    db = TraceDB()
    engine = _real_engine("v5-onboarding-tutorial")
    engine.is_onboarding_tutorial = True

    result = await web_server._persist_v5_dataset_checkpoint(
        {"db": db, "active_matches": {engine.match_id: engine}},
        engine,
        reason="tutorial_action",
    )

    assert result == {
        "applied": False,
        "reason": "onboarding_tutorial_excluded",
    }
    assert db.checkpoints == []
    assert db.finalizations == []
    assert db.aborts == []


@pytest.mark.asyncio(loop_scope="function")
async def test_abort_request_cannot_overwrite_terminal_recorder() -> None:
    db = TraceDB()
    engine = _real_engine("v5-terminal-abort-race")
    engine.finalize_v5_dataset(
        winner_user_id=101,
        status="p1_win",
        reason="lethal_action",
    )
    app = {"db": db, "active_matches": {engine.match_id: engine}}

    result = await web_server._persist_v5_dataset_checkpoint(
        app,
        engine,
        reason="server_reload",
        abort_reason="server_reload",
    )

    assert result["seal"]["status"] == "p1_win"
    assert db.aborts == []
    assert db.finalizations[-1]["battle_id"] == engine.match_id
    assert db.finalizations[-1]["winner_user_id"] == 101
    assert engine.get_v5_dataset_snapshot()["meta"]["aborted"] is False


@pytest.mark.asyncio(loop_scope="function")
async def test_rehydrated_engine_checkpoints_to_fresh_dataset_generation() -> None:
    db = TraceDB()
    gameplay_match_id = "v5-rehydrated-friendly"
    initial = _real_engine(gameplay_match_id)
    app = {"db": db, "active_matches": {gameplay_match_id: initial}}

    initial_result = await web_server._persist_v5_dataset_checkpoint(
        app,
        initial,
        reason="initial_generation",
        abort_reason="server_reload",
    )
    assert initial_result["seal"]["applied"] is True
    assert db.checkpoints[-1]["battle_id"] == gameplay_match_id

    rehydrated = _real_engine(gameplay_match_id)
    trace_id = rehydrated.start_new_v5_dataset_generation(
        reason="friendly_rehydrate"
    )
    rehydrated.record_analytics_action(
        101,
        {"type": "end_turn"},
        decision_source="human",
    )
    assert rehydrated.end_turn(101)["success"] is True

    result = await web_server._persist_v5_dataset_checkpoint(
        app,
        rehydrated,
        reason="rehydrated_generation",
    )

    assert result["applied"] is True
    assert trace_id != gameplay_match_id
    assert len(trace_id) <= 128
    assert re.fullmatch(r"[A-Za-z0-9._-]+", trace_id)
    assert db.checkpoints[-1]["battle_id"] == trace_id
    assert db.checkpoints[-1]["meta"]["battle_id"] == trace_id
    assert db.checkpoints[-1]["meta"]["match_id"] == gameplay_match_id
    assert db.checkpoints[-1]["meta"]["dataset_generation"] == 2
    assert db.checkpoints[-1]["actions"][0]["battle_id"] == trace_id
    assert db.checkpoints[-1]["actions"][0]["match_id"] == gameplay_match_id


def test_invalid_rehydrated_trace_id_does_not_advance_generation() -> None:
    engine = _real_engine("v5-generation-atomic")

    with pytest.raises(ValueError, match="safe"):
        engine.start_new_v5_dataset_generation(
            reason="friendly_rehydrate",
            trace_id="../unsafe",
        )

    assert engine.v5_dataset_generation == 1
    assert engine.v5_dataset_generation_reason == "initial"
    assert engine.v5_dataset_trace_id == engine.match_id


@pytest.mark.asyncio(loop_scope="function")
async def test_terminate_without_rewards_persists_abort_before_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = TraceDB()
    engine = LifecycleEngine("v5-terminate-abort")
    active_matches: dict[str, Any] = {engine.match_id: engine}
    app = {"db": db, "active_matches": active_matches, "match_game_modes": {}}
    engine.runtime_owner_app = app

    async def no_emit(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(web_server.sio, "emit", no_emit)
    web_server.ACTIVE_MATCHES[engine.match_id] = engine
    try:
        await web_server._terminate_match_without_rewards(
            engine.match_id,
            active_matches,
            reason="opponent_disconnected",
            message="gone",
        )

        assert engine.abort_calls
        assert engine.abort_calls[-1]["reason"] == "opponent_disconnected"
        assert db.aborts[-1]["battle_id"] == engine.match_id
        assert db.aborts[-1]["reason"] == "opponent_disconnected"
        assert engine.match_id not in active_matches
        assert engine.match_id not in web_server.ACTIVE_MATCHES
    finally:
        web_server.ACTIVE_MATCHES.pop(engine.match_id, None)
        web_server.ENDED_MATCH_IDS.discard(engine.match_id)
        web_server.ENDED_MATCH_TIMES.pop(engine.match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_app_cleanup_aborts_owned_unfinished_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBrain:
        sessions: dict[str, Any] = {}

        def __init__(self, **_kwargs: Any) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeAux:
        availability: dict[str, Any] = {}

        @classmethod
        def from_model_dir(cls) -> "FakeAux":
            return cls()

        def close(self) -> None:
            return None

    monkeypatch.setattr(web_server, "BerserkInference", FakeBrain)
    monkeypatch.setattr(web_server, "ExtraLRAuxRuntime", FakeAux)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()

    db = TraceDB()
    app = web_server.create_web_app(db, bot_token="bot-token")
    engine = LifecycleEngine("v5-reload-abort")
    engine.runtime_owner_app = app
    app["active_matches"][engine.match_id] = engine
    web_server.ACTIVE_MATCHES[engine.match_id] = engine

    try:
        await app.on_cleanup[-1](app)

        assert engine.abort_calls
        assert engine.abort_calls[-1]["reason"] == "server_reload"
        assert db.aborts[-1]["battle_id"] == engine.match_id
        assert db.aborts[-1]["reason"] == "server_reload"
        assert engine.match_id not in app["active_matches"]
        assert engine.match_id not in web_server.ACTIVE_MATCHES
    finally:
        web_server.ACTIVE_MATCHES.pop(engine.match_id, None)
        web_server.ENDED_MATCH_IDS.discard(engine.match_id)
        web_server.ENDED_MATCH_TIMES.pop(engine.match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_app_cleanup_seals_terminal_trace_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBrain:
        sessions: dict[str, Any] = {}

        def __init__(self, **_kwargs: Any) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeAux:
        availability: dict[str, Any] = {}

        @classmethod
        def from_model_dir(cls) -> "FakeAux":
            return cls()

        def close(self) -> None:
            return None

    monkeypatch.setattr(web_server, "BerserkInference", FakeBrain)
    monkeypatch.setattr(web_server, "ExtraLRAuxRuntime", FakeAux)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()

    db = TraceDB()
    app = web_server.create_web_app(db, bot_token="bot-token")
    engine = LifecycleEngine("v5-reload-terminal")
    engine.is_ended = True
    engine.check_game_over = lambda: {
        "game_over": True,
        "winner_id": 101,
    }
    engine.runtime_owner_app = app
    app["active_matches"][engine.match_id] = engine
    web_server.ACTIVE_MATCHES[engine.match_id] = engine

    try:
        await app.on_cleanup[-1](app)

        assert engine.abort_calls == []
        assert engine.finalize_calls
        assert engine.finalize_calls[-1]["status"] == "p1_win"
        assert db.aborts == []
        assert db.finalizations[-1]["battle_id"] == engine.match_id
        assert db.finalizations[-1]["status"] == "p1_win"
    finally:
        web_server.ACTIVE_MATCHES.pop(engine.match_id, None)
        web_server.ENDED_MATCH_IDS.discard(engine.match_id)
        web_server.ENDED_MATCH_TIMES.pop(engine.match_id, None)


def test_pve_timer_and_timestamp_anchor_wait_for_human_client_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _real_engine("v5-pve-ready-anchor", p2_is_bot=True)
    engine.client_ready = False
    engine.client_ready_users.clear()
    engine.turn_start_time = time.time() - 999.0
    engine.turn_start_monotonic = time.monotonic() - 999.0
    engine.match_start_time = engine.turn_start_time
    engine.match_start_monotonic = engine.turn_start_monotonic

    assert engine.is_waiting_for_players() is True
    assert engine.is_turn_expired() is False

    wall_now = 2_000.0
    mono_now = 1_000.0
    monkeypatch.setattr("battle_engine.time.time", lambda: wall_now)
    monkeypatch.setattr("battle_engine.time.monotonic", lambda: mono_now)
    readiness = engine.mark_client_ready(101)

    assert readiness["all_ready"] is True
    assert engine.is_waiting_for_players() is False
    assert engine.turn_start_time == wall_now
    assert engine.turn_start_monotonic == mono_now
    assert engine.match_start_time == wall_now
    assert engine.match_start_monotonic == mono_now
    assert engine.is_turn_expired() is False
    snapshot = engine.get_v5_dataset_snapshot()
    assert snapshot["meta"]["start_metadata"] == {
        "turn_number": 1,
        "starting_player_id": 101,
        "client_ready_anchored": True,
        "anchor_reason": "client_ready",
    }


@pytest.mark.asyncio(loop_scope="function")
async def test_bot_action_persists_metronome_prediction_and_applied_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _real_engine("v5-bot-metronome", p1_is_bot=True)
    db = TraceDB()
    app = {"db": db, "active_matches": {engine.match_id: engine}}
    engine.runtime_owner_app = app
    engine.get_turn_time_remaining = lambda: 25.0

    class EndTurnBrain:
        def has_profile(self, _difficulty: str) -> bool:
            return True

        async def get_action_async(
            self,
            _state: Any,
            _bot_id: int,
            legal_actions: list[Any],
            difficulty: str,
        ) -> int:
            del difficulty
            return next(
                index
                for index, action in enumerate(legal_actions)
                if isinstance(action, EndTurnAction)
            )

    class Metronome:
        def sample_ms(self, *_args: Any, **_kwargs: Any) -> float:
            return 375.0

    async def no_sleep(_seconds: float) -> None:
        return None

    engine.berserk_brain = EndTurnBrain()
    engine.extra_lr_aux_runtime = SimpleNamespace(metronome=Metronome())
    monkeypatch.setattr(web_server.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(web_server.random, "uniform", lambda *_args: 0.0)
    web_server.ACTIVE_MATCHES[engine.match_id] = engine
    try:
        await web_server.run_bot_routine(engine, 101)

        rows = engine.get_v5_dataset_snapshot()["actions"]
        assert rows
        row = rows[0]
        assert row["accepted"] is True
        assert row["decision_source"] == "bot"
        assert row["control_source"] == "bot"
        assert row["metronome_prediction_ms"] == pytest.approx(375.0)
        assert row["metronome_applied_ms"] == pytest.approx(375.0)
        assert row["metronome_fallback_used"] is False
        assert db.checkpoints
        assert db.checkpoints[-1]["actions"][0]["seq"] == row["seq"]
    finally:
        web_server.ACTIVE_MATCHES.pop(engine.match_id, None)
        web_server.MATCH_LOCKS.pop(engine.match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_v5_inference_failure_forces_safe_end_and_degrades_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _real_engine("v5-inference-fail-closed", p1_is_bot=True)
    engine.bot_brain_profile = "extra-lr-v5"
    engine.get_turn_time_remaining = lambda: 25.0
    db = TraceDB()
    app = {"db": db, "active_matches": {engine.match_id: engine}}
    engine.runtime_owner_app = app

    class BrokenV5Brain:
        def has_profile(self, _difficulty: str) -> bool:
            return True

        async def get_action_async(self, *_args: Any, **_kwargs: Any) -> int:
            raise RuntimeError(
                "postgresql://alice:SUPERSECRET@example.invalid/prod"
            )

    async def no_sleep(_seconds: float) -> None:
        return None

    engine.berserk_brain = BrokenV5Brain()
    monkeypatch.setattr(web_server.asyncio, "sleep", no_sleep)
    web_server.ACTIVE_MATCHES[engine.match_id] = engine
    try:
        await web_server.run_bot_routine(engine, 101)

        meta = engine.get_v5_dataset_snapshot()["meta"]
        assert meta["degraded"] is True
        assert meta["policy_warnings"] == [
            "v5_policy_failure:unexpected_failure"
        ]
        assert "SUPERSECRET" not in json.dumps(meta)
        assert db.checkpoints
        persisted_meta = db.checkpoints[-1]["meta"]
        assert persisted_meta["degraded"] is True
        assert persisted_meta["policy_warnings"] == meta["policy_warnings"]
    finally:
        web_server.ACTIVE_MATCHES.pop(engine.match_id, None)
        web_server.MATCH_LOCKS.pop(engine.match_id, None)


@pytest.mark.asyncio(loop_scope="function")
async def test_natural_timeout_is_a_persisted_bot_control_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _real_engine("v5-natural-timeout")
    db = TraceDB()
    app = {"db": db, "active_matches": {engine.match_id: engine}, "socketio": None}
    engine.runtime_owner_app = app
    engine.mark_client_ready(101)
    engine.mark_client_ready(202)
    monkeypatch.setattr(engine, "is_turn_expired", lambda: True)

    async def no_bot_move(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(web_server, "check_and_run_bot", no_bot_move)
    web_server.ACTIVE_MATCHES[engine.match_id] = engine
    try:
        handled = await web_server._handle_natural_turn_timeout(
            app,
            engine.match_id,
            engine,
        )

        assert handled is True
        rows = engine.get_v5_dataset_snapshot()["actions"]
        assert rows
        row = rows[0]
        assert row["accepted"] is True
        assert row["action_native"]["type"] == "end_turn"
        assert row["decision_source"] == "bot"
        assert row["control_source"] == "timeout"
        assert row["human_decision_time_ms"] is None
        assert db.checkpoints
        assert db.checkpoints[-1]["actions"][0]["control_source"] == "timeout"
    finally:
        web_server.ACTIVE_MATCHES.pop(engine.match_id, None)
        web_server.MATCH_LOCKS.pop(engine.match_id, None)
