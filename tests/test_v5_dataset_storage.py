from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone

import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import (
    EXTRAARENA_V5_DATASET_EXPORT_SCHEMA,
    RLHF_V5_STORAGE_SCHEMA,
    Database,
)


def _db() -> Database:
    return Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))


def _state(owner: int, *, turn: int = 1, status: str = "ongoing") -> dict:
    return {
        "turn_number": turn,
        "current_turn_owner_id": owner,
        "status": status,
        "p1": {"user_id": 101, "is_bot": False},
        "p2": {"user_id": 202, "is_bot": True},
        "action_history": [],
        "history": [],
        "v5_history_events": [],
        "pending_card_feedback_events": [],
    }


def _meta(*, status: str = "ongoing") -> dict:
    return {
        "schema_version": RLHF_V5_STORAGE_SCHEMA,
        "battle_id": "battle-1",
        "status": status,
        "p1_user_id": 101,
        "p2_user_id": 202,
        "p1_actor_type": "human",
        "p2_actor_type": "bot",
        "battle_tag": "human-vs-bot",
        "game_mode": "classic",
    }


def _action(seq: int, actor: int, actor_player: int) -> dict:
    pre = _state(actor, turn=seq)
    post = _state(202 if actor == 101 else 101, turn=seq + 1)
    return {
        "seq": seq,
        "battle_id": "battle-1",
        "turn_number": seq,
        "actor_user_id": actor,
        "actor_player": actor_player,
        "decision_source": "human" if actor_player == 1 else "bot",
        "legal_action_index": 0,
        "action_type": "end_turn",
        "action_json": {"type": "end_turn"},
        "action_native": {"type": "end_turn"},
        "legal_actions": [{"type": "end_turn"}],
        "legal_action_count": 1,
        "pre_state": pre,
        "post_state": post,
        "accepted": True,
        "error": None,
    }


def _terminal_row() -> dict:
    return {
        "battle_id": "battle-1",
        "storage_schema": RLHF_V5_STORAGE_SCHEMA,
        "status": "p1_win",
        "winner_user_id": 101,
        "p1_user_id": 101,
        "p2_user_id": 202,
        "p1_actor_type": "human",
        "p2_actor_type": "bot",
        "battle_tag": "human-vs-bot",
        "game_mode": "classic",
        "meta_json": {
            **_meta(status="p1_win"),
            "winner_user_id": 101,
        },
        "turns_json": [_state(101)],
        "action_count": 2,
        "finished_at": datetime(2026, 7, 28, tzinfo=timezone.utc),
        "actions_json": [
            _action(1, 101, 1),
            _action(2, 202, 2),
        ],
    }


def _stored_header(last_seq: int) -> dict:
    return {
        "status": "ongoing",
        "last_seq": int(last_seq),
        "p1_user_id": 101,
        "p2_user_id": 202,
        "p1_actor_type": "human",
        "p2_actor_type": "bot",
    }


def _fallback_bot_state(
    owner: int,
    *,
    turn: int = 1,
    status: str = "ongoing",
) -> dict:
    state = _state(owner, turn=turn, status=status)
    state["p2"]["user_id"] = -202
    if state["current_turn_owner_id"] == 202:
        state["current_turn_owner_id"] = -202
    return state


def _fallback_bot_action(seq: int, actor: int, actor_player: int) -> dict:
    action = _action(seq, actor, actor_player)
    if action["actor_user_id"] == 202:
        action["actor_user_id"] = -202
    for key in ("pre_state", "post_state"):
        action[key]["p2"]["user_id"] = -202
        if action[key]["current_turn_owner_id"] == 202:
            action[key]["current_turn_owner_id"] = -202
    return action


class _Transaction:
    def __init__(self) -> None:
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Connection:
    def __init__(self, fetchrows: list[dict]) -> None:
        self.fetchrows = list(fetchrows)
        self.executed: list[tuple[str, tuple]] = []
        self.executemany_calls: list[tuple[str, list[tuple]]] = []
        self.tx = _Transaction()

    def transaction(self) -> _Transaction:
        return self.tx

    async def execute(self, query: str, *args):
        self.executed.append((query, args))
        return "OK"

    async def executemany(self, query: str, args):
        rows = list(args)
        self.executemany_calls.append((query, rows))

    async def fetchrow(self, query: str, *args):
        assert self.fetchrows, f"unexpected fetchrow: {query}"
        return self.fetchrows.pop(0)


class _Acquire:
    def __init__(self, conn: _Connection) -> None:
        self.conn = conn

    async def __aenter__(self) -> _Connection:
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Pool:
    def __init__(self, conn: _Connection) -> None:
        self.conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self.conn)


@pytest.mark.asyncio
async def test_v5_checkpoint_upsert_is_transactional_and_seq_idempotent():
    conn = _Connection([_stored_header(0)])
    db = _db()
    db._pool = _Pool(conn)
    actions = [_action(1, 101, 1), _action(2, 202, 2)]

    result = await db.upsert_v5_battle_trace_checkpoint(
        battle_id="battle-1",
        meta=_meta(),
        turns=[_state(101)],
        actions=actions,
    )

    assert conn.tx.entered is True
    assert result == {
        "applied": True,
        "reason": "checkpoint_upserted",
        "battle_id": "battle-1",
        "status": "ongoing",
        "last_seq": 2,
        "action_count": 2,
    }
    assert len(conn.executemany_calls) == 1
    action_query, action_rows = conn.executemany_calls[0]
    assert "ON CONFLICT (battle_id, seq) DO UPDATE" in action_query
    assert [(row[0], row[1]) for row in action_rows] == [
        ("battle-1", 1),
        ("battle-1", 2),
    ]
    assert json.loads(action_rows[0][2])["accepted"] is True
    assert any(
        "DELETE FROM battle_v5_trace_actions" in query and args == ("battle-1", 2)
        for query, args in conn.executed
    )


@pytest.mark.asyncio
async def test_v5_checkpoint_rejects_regression_without_touching_action_rows():
    conn = _Connection([_stored_header(2)])
    db = _db()
    db._pool = _Pool(conn)

    result = await db.upsert_v5_battle_trace_checkpoint(
        battle_id="battle-1",
        meta=_meta(),
        turns=[_state(101)],
        actions=[_action(1, 101, 1)],
    )

    assert result["reason"] == "stale_checkpoint"
    assert result["last_seq"] == 2
    assert conn.executemany_calls == []
    assert not any(
        "DELETE FROM battle_v5_trace_actions" in query
        for query, _args in conn.executed
    )


@pytest.mark.asyncio
async def test_v5_checkpoint_writes_only_the_new_action_suffix():
    conn = _Connection([_stored_header(1)])
    db = _db()
    db._pool = _Pool(conn)

    result = await db.upsert_v5_battle_trace_checkpoint(
        battle_id="battle-1",
        meta=_meta(),
        turns=[_state(101)],
        actions=[
            _action(1, 101, 1),
            _action(2, 202, 2),
            _action(3, 101, 1),
        ],
    )

    assert result["last_seq"] == 3
    assert len(conn.executemany_calls) == 1
    _query, rows = conn.executemany_calls[0]
    assert [(row[0], row[1]) for row in rows] == [
        ("battle-1", 2),
        ("battle-1", 3),
    ]


@pytest.mark.asyncio
async def test_v5_checkpoint_rejects_noncanonical_action_before_db_io():
    db = _db()
    db._pool = object()
    malformed = _action(1, 101, 1)
    malformed.pop("accepted")

    with pytest.raises(ValueError, match="missing_fields:accepted"):
        await db.upsert_v5_battle_trace_checkpoint(
            battle_id="battle-1",
            meta=_meta(),
            turns=[_state(101)],
            actions=[malformed],
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("p1_user_id", None, "invalid_v5_participant_id"),
        ("p1_user_id", "101", "invalid_v5_participant_id"),
        ("p1_user_id", True, "invalid_v5_participant_id"),
        ("p1_user_id", 0, "invalid_v5_participant_id"),
        ("p2_user_id", 101, "participant_ids_must_be_distinct"),
    ],
)
@pytest.mark.asyncio
async def test_v5_checkpoint_rejects_ambiguous_participants_before_db_io(
    field: str,
    value: object,
    error: str,
):
    db = _db()
    db._pool = object()
    meta = _meta()
    meta[field] = value

    with pytest.raises(ValueError, match=error):
        await db.upsert_v5_battle_trace_checkpoint(
            battle_id="battle-1",
            meta=meta,
            turns=[_state(101)],
            actions=[_action(1, 101, 1)],
        )


@pytest.mark.asyncio
async def test_v5_checkpoint_accepts_unpersisted_negative_fallback_bot_id():
    meta = _meta()
    meta["p2_user_id"] = -202
    stored = _stored_header(0)
    stored["p2_user_id"] = -202
    conn = _Connection([stored])
    db = _db()
    db._pool = _Pool(conn)

    result = await db.upsert_v5_battle_trace_checkpoint(
        battle_id="battle-1",
        meta=meta,
        turns=[_fallback_bot_state(101)],
        actions=[_fallback_bot_action(1, 101, 1)],
    )

    assert result["applied"] is True
    assert result["action_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("turn_p1", "turn_1_p1_user_id_mismatch"),
        ("pre_state_p2", "action_1_pre_state_p2_user_id_mismatch"),
        ("actor", "action_1_actor_user_id_mismatch"),
    ],
)
@pytest.mark.asyncio
async def test_v5_checkpoint_rejects_nested_participant_mismatches(
    mutation: str,
    error: str,
):
    db = _db()
    db._pool = object()
    turn = _state(101)
    action = _action(1, 101, 1)
    if mutation == "turn_p1":
        turn["p1"]["user_id"] = 303
    elif mutation == "pre_state_p2":
        action["pre_state"]["p2"]["user_id"] = 303
    else:
        action["actor_user_id"] = 202

    with pytest.raises(ValueError, match=error):
        await db.upsert_v5_battle_trace_checkpoint(
            battle_id="battle-1",
            meta=_meta(),
            turns=[turn],
            actions=[action],
        )


@pytest.mark.asyncio
async def test_v5_checkpoint_cannot_change_persisted_seat_identity():
    stored = _stored_header(1)
    stored["p1_user_id"] = 303
    conn = _Connection([stored])
    db = _db()
    db._pool = _Pool(conn)

    with pytest.raises(ValueError, match="participant_ids_changed"):
        await db.upsert_v5_battle_trace_checkpoint(
            battle_id="battle-1",
            meta=_meta(),
            turns=[_state(101)],
            actions=[_action(1, 101, 1)],
        )
    assert conn.executemany_calls == []


@pytest.mark.asyncio
async def test_finalize_v5_trace_checks_completeness_and_seals_terminal_meta():
    conn = _Connection(
        [
            {
                "status": "ongoing",
                "winner_user_id": None,
                "p1_user_id": 101,
                "p2_user_id": 202,
                "p1_actor_type": "human",
                "p2_actor_type": "bot",
                "meta_json": _meta(),
                "turns_json": [_state(101)],
                "action_count": 2,
                "last_seq": 2,
                "stored_action_count": 2,
            }
        ]
    )
    db = _db()
    db._pool = _Pool(conn)

    result = await db.finalize_v5_battle_trace(
        battle_id="battle-1",
        status="p1_win",
        winner_user_id=101,
    )

    assert result["applied"] is True
    assert result["status"] == "p1_win"
    update = next(
        (query, args)
        for query, args in conn.executed
        if "UPDATE battle_v5_traces" in query
    )
    persisted_meta = json.loads(update[1][3])
    assert persisted_meta["status"] == "p1_win"
    assert persisted_meta["winner_user_id"] == 101
    assert persisted_meta["finished_at"]


@pytest.mark.asyncio
async def test_finalize_v5_trace_rejects_terminal_meta_identity_change():
    conn = _Connection(
        [
            {
                "status": "ongoing",
                "winner_user_id": None,
                "p1_user_id": 101,
                "p2_user_id": 202,
                "p1_actor_type": "human",
                "p2_actor_type": "bot",
                "meta_json": _meta(),
                "turns_json": [_state(101)],
                "action_count": 1,
                "last_seq": 1,
                "stored_action_count": 1,
            }
        ]
    )
    db = _db()
    db._pool = _Pool(conn)
    terminal_meta = _meta(status="p1_win")
    terminal_meta["p1_user_id"] = 303

    with pytest.raises(ValueError, match="participant_ids_mismatch"):
        await db.finalize_v5_battle_trace(
            battle_id="battle-1",
            status="p1_win",
            winner_user_id=101,
            meta=terminal_meta,
        )
    assert not any(
        "UPDATE battle_v5_traces" in query
        for query, _args in conn.executed
    )


@pytest.mark.asyncio
async def test_v5_export_limits_battles_not_actions_and_pseudonymizes_by_default():
    db = _db()
    db._pool = object()
    action_one = _action(1, 101, 1)
    action_two = _action(2, 202, 2)
    action_one["timestamp_ms"] = 101
    action_one["action_json"]["card_id"] = 101
    for state in (
        action_one["pre_state"],
        action_one["post_state"],
        action_two["pre_state"],
        action_two["post_state"],
    ):
        state["p1"]["hero"] = {
            "card_id": 101,
            "hp": 202,
            "level": 101,
            "owner_id": 101,
        }
        state["v5_history_events"] = [
            {
                "actor_id": 101,
                "action_id": 202,
                "source_card": {"card_id": 101, "hp": 202},
            }
        ]
    turn_state = _state(101)
    turn_state["p1"]["hero"] = {
        "card_id": 101,
        "hp": 202,
        "level": 101,
        "owner_id": 101,
    }
    turn_state["v5_history_events"] = [
        {
            "actor_id": 101,
            "action_id": 202,
            "source_card": {"card_id": 101, "hp": 202},
        }
    ]
    nemesis_record = {
        "features": {
            "base": {
                "seats": {
                    "p1": {"participant_id": 101},
                    "p2": {"participant_id": 202},
                }
            }
        }
    }
    row = {
        "battle_id": "battle-1",
        "storage_schema": RLHF_V5_STORAGE_SCHEMA,
        "status": "p1_win",
        "winner_user_id": 101,
        "p1_user_id": 101,
        "p2_user_id": 202,
        "p1_actor_type": "human",
        "p2_actor_type": "bot",
        "battle_tag": "human-vs-bot",
        "game_mode": "classic",
        "meta_json": {
            **_meta(status="p1_win"),
            "winner_user_id": 101,
            "nemesis_record": nemesis_record,
        },
        "turns_json": [turn_state],
        "action_count": 2,
        "finished_at": datetime(2026, 7, 28, tzinfo=timezone.utc),
        "actions_json": [action_one, action_two],
    }
    captured: dict = {}

    async def fake_fetch(query: str, *args):
        captured["query"] = query
        captured["args"] = args
        return [row]

    db.fetch = fake_fetch
    exported = await db.export_v5_battle_dataset(
        days=10,
        limit_battles=1,
    )

    assert exported["format"] == EXTRAARENA_V5_DATASET_EXPORT_SCHEMA
    assert exported["storage_schema"] == RLHF_V5_STORAGE_SCHEMA
    assert exported["privacy"] == "side_pseudonyms_p1_1_p2_2"
    assert exported["battle_count"] == 1
    bundle = exported["battles"][0]
    assert len(bundle["actions"]) == 2
    assert bundle["meta"]["p1_user_id"] == 1
    assert bundle["meta"]["p2_user_id"] == 2
    assert bundle["meta"]["winner_user_id"] == 1
    nemesis_seats = bundle["meta"]["nemesis_record"]["features"]["base"]["seats"]
    assert nemesis_seats["p1"]["participant_id"] == 1
    assert nemesis_seats["p2"]["participant_id"] == 2
    assert bundle["turns"][0]["current_turn_owner_id"] == 1
    assert bundle["turns"][0]["p1"]["hero"] == {
        "card_id": 101,
        "hp": 202,
        "level": 101,
        "owner_id": 1,
    }
    assert bundle["turns"][0]["v5_history_events"][0]["actor_id"] == 1
    assert bundle["turns"][0]["v5_history_events"][0]["action_id"] == 202
    assert bundle["turns"][0]["v5_history_events"][0]["source_card"] == {
        "card_id": 101,
        "hp": 202,
    }
    assert bundle["actions"][0]["actor_user_id"] == 1
    assert bundle["actions"][1]["actor_user_id"] == 2
    assert bundle["actions"][0]["timestamp_ms"] == 101
    assert bundle["actions"][0]["action_json"]["card_id"] == 101
    assert captured["query"].count("LIMIT") == 1
    assert "jsonb_agg(action.payload_json ORDER BY action.seq)" in captured["query"]
    assert "trace.p1_actor_type = 'human'" in captured["query"]
    assert "trace.p2_actor_type = 'human'" in captured["query"]

    raw = await db.export_v5_battle_dataset(
        days=10,
        limit_battles=1,
        include_players=True,
    )
    assert raw["privacy"] == "raw_player_ids"
    assert raw["battles"][0]["meta"]["p1_user_id"] == 101
    assert raw["battles"][0]["actions"][1]["actor_user_id"] == 202
    raw_nemesis_seats = raw["battles"][0]["meta"]["nemesis_record"]["features"][
        "base"
    ]["seats"]
    assert raw_nemesis_seats["p1"]["participant_id"] == 101
    assert raw_nemesis_seats["p2"]["participant_id"] == 202


@pytest.mark.asyncio
async def test_v5_export_skips_row_when_participant_mapping_is_not_total():
    db = _db()
    db._pool = object()
    row = _terminal_row()
    row["p1_user_id"] = None

    async def fake_fetch(_query: str, *_args):
        return [row]

    db.fetch = fake_fetch
    exported = await db.export_v5_battle_dataset(
        days=10,
        limit_battles=1,
    )

    assert exported["battle_count"] == 0
    assert exported["skipped_invalid"] == 1
    assert exported["battles"] == []


@pytest.mark.asyncio
async def test_v5_stream_selection_fetches_only_small_eligible_id_list():
    db = _db()
    db._pool = object()
    captured: dict = {}

    async def fake_fetch(query: str, *args):
        captured["query"] = query
        captured["args"] = args
        return [{"battle_id": "battle-2"}, {"battle_id": "battle-1"}]

    db.fetch = fake_fetch
    selected = await db.list_v5_export_battle_ids(
        days=7,
        limit_battles=25,
    )

    assert selected["battle_ids"] == ["battle-2", "battle-1"]
    assert selected["days"] == 7
    assert selected["limit_battles"] == 25
    assert "SELECT trace.battle_id" in captured["query"]
    assert "jsonb_agg" not in captured["query"]
    assert "trace.p1_user_id IS NOT NULL" in captured["query"]
    assert "trace.p1_user_id <> trace.p2_user_id" in captured["query"]
    assert "start_metadata,client_ready_anchored" in captured["query"]
    assert "timestamp_features" in captured["query"]
    assert captured["args"][1] == 25


@pytest.mark.asyncio
async def test_v5_stream_bundle_loads_and_pseudonymizes_one_battle():
    db = _db()
    db._pool = object()
    row = _terminal_row()
    captured: dict = {}

    async def fake_fetchrow(query: str, *args):
        captured["query"] = query
        captured["args"] = args
        return row

    db.fetchrow = fake_fetchrow
    bundle = await db.get_v5_export_battle_bundle(
        battle_id="battle-1",
    )

    assert bundle is not None
    assert bundle["meta"]["p1_user_id"] == 1
    assert bundle["meta"]["p2_user_id"] == 2
    assert bundle["actions"][0]["actor_user_id"] == 1
    assert bundle["actions"][1]["actor_user_id"] == 2
    assert captured["args"] == ("battle-1",)
    assert "jsonb_agg(action.payload_json ORDER BY action.seq)" in captured["query"]
    assert "trace.p1_user_id <> trace.p2_user_id" in captured["query"]
    assert "start_metadata,client_ready_anchored" in captured["query"]


@pytest.mark.asyncio
async def test_v5_stream_bundle_pseudonymizes_unpersisted_negative_fallback_bot():
    db = _db()
    db._pool = object()
    meta = _meta(status="p1_win")
    meta.update({"p2_user_id": -202, "winner_user_id": 101})
    row = {
        "battle_id": "battle-1",
        "storage_schema": RLHF_V5_STORAGE_SCHEMA,
        "status": "p1_win",
        "winner_user_id": 101,
        "p1_user_id": 101,
        "p2_user_id": -202,
        "p1_actor_type": "human",
        "p2_actor_type": "bot",
        "battle_tag": "human-vs-bot",
        "game_mode": "classic",
        "meta_json": meta,
        "turns_json": [_fallback_bot_state(101)],
        "action_count": 2,
        "finished_at": datetime(2026, 7, 28, tzinfo=timezone.utc),
        "actions_json": [
            _fallback_bot_action(1, 101, 1),
            _fallback_bot_action(2, 202, 2),
        ],
    }

    async def fake_fetchrow(_query: str, *_args):
        return row

    db.fetchrow = fake_fetchrow
    bundle = await db.get_v5_export_battle_bundle(battle_id="battle-1")

    assert bundle is not None
    assert bundle["meta"]["p1_user_id"] == 1
    assert bundle["meta"]["p2_user_id"] == 2
    assert bundle["actions"][1]["actor_user_id"] == 2
    assert bundle["actions"][1]["pre_state"]["p2"]["user_id"] == 2


@pytest.mark.asyncio
async def test_stale_marker_and_prune_only_operate_on_whole_trace_headers():
    db = _db()
    db._pool = object()
    calls: list[tuple[str, tuple]] = []

    async def fake_fetch(query: str, *args):
        calls.append((query, args))
        if "UPDATE battle_v5_traces" in query:
            return [{"battle_id": "stale-1"}]
        return [{"battle_id": "old-terminal-1"}]

    db.fetch = fake_fetch
    stale = await db.mark_stale_v5_battle_traces_aborted(
        older_than_seconds=60,
    )
    pruned = await db.prune_v5_battle_traces(
        terminal_older_than_days=90,
        aborted_older_than_days=30,
        limit_battles=25,
    )

    assert stale["battle_ids"] == ["stale-1"]
    assert "WHERE status = 'ongoing'" in calls[0][0]
    assert pruned["battle_ids"] == ["old-terminal-1"]
    prune_query = calls[1][0]
    assert "DELETE FROM battle_v5_traces AS trace" in prune_query
    assert "DELETE FROM battle_v5_trace_actions" not in prune_query
    assert "FOR UPDATE SKIP LOCKED" in prune_query
    assert calls[1][1] == (90, 30, 25)


def test_v5_journal_migration_has_required_keys_and_indexes():
    source = inspect.getsource(Database._ensure_battle_v5_trace_tables)

    assert "battle_v5_traces" in source
    assert "battle_v5_trace_actions" in source
    assert "PRIMARY KEY (battle_id, seq)" in source
    assert "ON DELETE CASCADE" in source
    assert "idx_battle_v5_traces_terminal_finished" in source
    assert "idx_battle_v5_traces_updated_at" in source
    assert "idx_battle_v5_traces_battle_tag" in source
