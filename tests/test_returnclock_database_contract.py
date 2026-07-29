from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
import json

import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import (
    Database,
    RETURNCLOCK_ANALYTICS_VERSION,
    RETURNCLOCK_DATASET_SCHEMA,
)


def _db() -> Database:
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    db._pool = object()
    return db


def test_returnclock_schema_is_part_of_init_schema_change_aggregation():
    source = inspect.getsource(Database.init_schema)

    assert "returnclock_changed = await self._ensure_returnclock_tables()" in source
    assert "or returnclock_changed" in source
    assert RETURNCLOCK_ANALYTICS_VERSION == 2
    assert RETURNCLOCK_DATASET_SCHEMA == "returnclock_dataset_raw_v1"


def test_returnclock_schema_captures_assignment_propensity_and_delivery_outcomes():
    source = inspect.getsource(Database._ensure_returnclock_tables)

    assert "CREATE TABLE returnclock_decisions" in source
    assert "experiment_id TEXT" in source
    assert "treatment_arm TEXT" in source
    assert "assignment_probability DOUBLE PRECISION" in source
    assert "eligible_actions JSONB" in source
    assert "CREATE TABLE returnclock_delivery_events" in source
    assert "event_id TEXT NOT NULL UNIQUE" in source
    assert "provider_message_id TEXT" in source
    assert "client_event_at TIMESTAMPTZ" in source
    assert "ON DELETE CASCADE" in source


def test_user_session_schema_has_returnclock_attribution_and_lifecycle_fields():
    source = inspect.getsource(Database._ensure_user_sessions_table)

    assert "analytics_version INTEGER NOT NULL DEFAULT 1" in source
    assert "timezone TEXT" in source
    assert "utc_offset_minutes INTEGER" in source
    assert "entrypoint TEXT" in source
    assert "returnclock_decision_id TEXT" in source
    assert "returnclock_delivery_id TEXT" in source
    assert "last_heartbeat_at TIMESTAMPTZ" in source
    assert "last_resumed_at TIMESTAMPTZ" in source
    assert "resume_count INTEGER NOT NULL DEFAULT 0" in source
    assert "battle_ids JSONB NOT NULL DEFAULT '[]'::jsonb" in source
    assert "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in source
    assert "user_sessions_user_started_idx" in source
    assert "user_sessions_open_heartbeat_idx" in source


@pytest.mark.asyncio
async def test_session_start_replay_is_immutable_and_reports_created_false():
    db = _db()
    calls: list[tuple[str, tuple]] = []

    async def fake_fetchrow(query: str, *args):
        calls.append((query, args))
        return {
            "id": 1,
            "created": len(calls) == 1,
            "active": True,
            "entrypoint": "notification",
            "returnclock_decision_id": "decision-1",
            "returnclock_delivery_id": "delivery-1",
            "metadata": {
                "returnclock_attribution_verified": True,
            },
        }

    db.fetchrow = fake_fetchrow

    first = await db.start_user_session(
        42,
        "session-1",
        analytics_version=2,
        timezone_name="Europe/Moscow",
        utc_offset_minutes=180,
        entrypoint="notification",
        returnclock_decision_id="decision-1",
        returnclock_delivery_id="delivery-1",
        metadata={"client": "android"},
        return_status=True,
    )
    replay = await db.start_user_session(
        42,
        "session-1",
        analytics_version=9,
        timezone_name="UTC",
        entrypoint=None,
        metadata={"client": "replayed"},
        return_status=True,
    )

    expected_attribution = {
        "active": True,
        "entrypoint": "notification",
        "returnclock_decision_id": "decision-1",
        "returnclock_delivery_id": "delivery-1",
        "returnclock_attribution_verified": True,
    }
    assert first == {
        "ok": True,
        "created": True,
        **expected_attribution,
    }
    assert replay == {
        "ok": True,
        "created": False,
        **expected_attribution,
    }
    assert "ON CONFLICT (session_id) DO NOTHING" in calls[0][0]
    assert "DO UPDATE" not in calls[0][0]
    assert "FALSE AS created" in calls[0][0]
    assert "ended_at IS NULL AS active" in calls[0][0]
    assert "WHERE user_id = $1 AND session_id = $2" in calls[0][0]
    assert calls[0][1][0:4] == (42, "session-1", "webapp", 2)
    assert calls[0][1][7:9] == ("decision-1", "delivery-1")
    # A retry may carry different client data, but INSERT-only replay cannot
    # rewrite the original assignment/lifecycle row.
    assert calls[1][1][3] == 9
    assert calls[1][1][4] == "UTC"


@pytest.mark.asyncio
async def test_session_writes_dedupe_battle_ids_and_never_reopen_closed_rows():
    db = _db()
    calls: list[tuple[str, tuple]] = []

    async def fake_fetchrow(query: str, *args):
        calls.append((query, args))
        return {"id": 1}

    db.fetchrow = fake_fetchrow

    assert await db.update_user_session(
        42,
        "session-1",
        heartbeat=True,
        resumed=True,
        battle_ids=["battle-1", "battle-1", " battle-2 ", "battle-2"],
    )
    update_query, update_args = calls[0]
    assert "sessions.user_id = $1" in update_query
    assert "sessions.session_id = $2" in update_query
    assert "sessions.ended_at IS NULL" in update_query
    assert "(ended_at IS NULL OR" not in update_query
    assert "ended_at = CASE WHEN" not in update_query
    assert "$3::jsonb" in update_query
    assert "sessions.screens_visited" in update_query
    assert "battle_ids" in update_query
    assert update_args[0:3] == (42, "session-1", None)
    list_payloads = [
        json.loads(value)
        for value in update_args
        if isinstance(value, str) and value.startswith("[")
    ]
    assert ["battle-1", "battle-2"] in list_payloads

    ended_at = datetime(2026, 7, 29, 9, 30, tzinfo=timezone.utc)
    assert await db.finish_user_session(
        42,
        "session-1",
        battle_ids=["battle-2", "battle-3", "battle-3"],
        ended_at=ended_at,
    )
    finish_query, finish_args = calls[1]
    assert "sessions.user_id = $1" in finish_query
    assert "sessions.session_id = $2" in finish_query
    assert "sessions.ended_at IS NULL" in finish_query
    assert "battle_ids" in finish_query
    assert "LEAST(NOW(), COALESCE($" in finish_query
    assert finish_args[0:3] == (42, "session-1", None)
    assert ended_at in finish_args
    finish_list_payloads = [
        json.loads(value)
        for value in finish_args
        if isinstance(value, str) and value.startswith("[")
    ]
    assert ["battle-2", "battle-3"] in finish_list_payloads


@pytest.mark.asyncio
async def test_decision_and_delivery_event_writes_are_idempotent_and_user_scoped():
    db = _db()
    calls: list[tuple[str, tuple]] = []

    async def fake_fetchrow(query: str, *args):
        calls.append((query, args))
        if "returnclock_delivery_events" in query:
            return {
                "event_id": args[2],
                "decision_id": args[1],
                "user_id": args[0],
                "event_type": args[3],
            }
        return {
            "decision_id": args[0],
            "user_id": args[1],
            "decision": args[3],
        }

    db.fetchrow = fake_fetchrow

    decision = await db.create_returnclock_decision(
        42,
        decision_id="decision-1",
        decision="schedule",
        policy_version="policy-v1",
        experiment_id="pilot-1",
        treatment_arm="returnclock",
        assignment_probability=0.5,
        eligible_actions=["skip", "send_19_21"],
    )
    assert decision["decision_id"] == "decision-1"
    decision_query = calls[0][0]
    assert "ON CONFLICT (decision_id) DO NOTHING" in decision_query
    # The conflict lookup intentionally fetches by globally unique key first;
    # the payload assertion below must reject a collision from another user.
    assert "WHERE decision_id = $1" in decision_query

    event = await db.record_returnclock_delivery_event(
        42,
        "decision-1",
        event_id="event-1",
        event_type="opened",
        delivery_id="delivery-1",
        channel="android",
    )
    assert event["event_id"] == "event-1"
    event_query = calls[1][0]
    assert "WHERE d.user_id = $1 AND d.decision_id = $2" in event_query
    assert "ON CONFLICT (event_id) DO NOTHING" in event_query
    assert "WHERE event_id = $3" in event_query


@pytest.mark.asyncio
async def test_decision_replay_rejects_divergent_idempotency_payload():
    db = _db()

    async def fake_fetchrow(query: str, *args):
        return {
            "decision_id": args[0],
            "user_id": args[1],
            "decision": "skip",
            "schedule_type": args[2],
            "decision_source": args[4],
            "policy_version": args[5],
            "treatment_arm": args[8],
        }

    db.fetchrow = fake_fetchrow

    with pytest.raises(ValueError, match="idempotency"):
        await db.create_returnclock_decision(
            42,
            decision_id="decision-1",
            decision="schedule",
            policy_version="policy-v1",
        )


@pytest.mark.asyncio
async def test_decision_replay_rejects_divergent_expiry():
    db = _db()
    original_expiry = datetime(
        2026, 7, 30, 12, 0, tzinfo=timezone.utc
    )

    async def fake_fetchrow(query: str, *args):
        return {
            "decision_id": args[0],
            "user_id": args[1],
            "schedule_type": args[2],
            "decision": args[3],
            "decision_source": args[4],
            "policy_version": args[5],
            "model_version": args[6],
            "experiment_id": args[7],
            "treatment_arm": args[8],
            "assignment_probability": args[9],
            "eligible_at": args[10],
            "planned_send_at": args[11],
            "expires_at": original_expiry,
            "eligible_actions": [],
            "prediction": {},
            "context": {},
            "reason_code": args[16],
            "source_session_id": args[17],
        }

    db.fetchrow = fake_fetchrow

    with pytest.raises(
        ValueError,
        match="returnclock_decision_idempotency_conflict:expires_at",
    ):
        await db.create_returnclock_decision(
            42,
            decision_id="decision-expiry",
            decision="schedule",
            policy_version="policy-v1",
            expires_at=original_expiry + timedelta(hours=1),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "stored_value", "requested_value"),
    [
        (
            "planned_send_at",
            datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 30, 13, 0, tzinfo=timezone.utc),
        ),
        ("prediction", {"return_hours": 12.0}, {"return_hours": 6.0}),
        ("context", {"source": "arena"}, {"source": "shop"}),
    ],
)
async def test_decision_replay_rejects_divergent_assignment_envelope_fields(
    field: str,
    stored_value,
    requested_value,
):
    db = _db()

    async def fake_fetchrow(query: str, *args):
        row = {
            "decision_id": args[0],
            "user_id": args[1],
            "schedule_type": args[2],
            "decision": args[3],
            "decision_source": args[4],
            "policy_version": args[5],
            "model_version": args[6],
            "experiment_id": args[7],
            "treatment_arm": args[8],
            "assignment_probability": args[9],
            "eligible_at": args[10],
            "planned_send_at": args[11],
            "expires_at": args[12],
            "eligible_actions": json.loads(args[13]),
            "prediction": json.loads(args[14]),
            "context": json.loads(args[15]),
            "reason_code": args[16],
            "source_session_id": args[17],
        }
        row[field] = stored_value
        return row

    db.fetchrow = fake_fetchrow
    kwargs = {
        "planned_send_at": None,
        "prediction": {},
        "context": {},
    }
    kwargs[field] = requested_value

    with pytest.raises(
        ValueError,
        match=f"returnclock_decision_idempotency_conflict:{field}",
    ):
        await db.create_returnclock_decision(
            42,
            decision_id=f"decision-{field}",
            decision="schedule",
            policy_version="policy-v1",
            **kwargs,
        )


@pytest.mark.asyncio
async def test_decision_status_update_cannot_mutate_assignment_envelope():
    db = _db()
    calls: list[tuple[str, tuple]] = []

    async def fake_fetchrow(query: str, *args):
        calls.append((query, args))
        return {"decision_id": args[1], "user_id": args[0]}

    db.fetchrow = fake_fetchrow
    await db.update_returnclock_decision(
        42,
        "decision-1",
        status="queued",
        outbox_id=77,
    )

    query, args = calls[0]
    assert "planned_send_at =" not in query
    assert "reason_code =" not in query
    assert "prediction =" not in query
    assert "context =" not in query
    assert args == (42, "decision-1", "queued", 77, None, None)


@pytest.mark.asyncio
async def test_delivery_event_replay_rejects_divergent_idempotency_payload():
    db = _db()

    async def fake_fetchrow(query: str, *args):
        return {
            "event_id": args[2],
            "decision_id": args[1],
            "user_id": args[0],
            "event_type": "dismissed",
        }

    db.fetchrow = fake_fetchrow

    with pytest.raises(ValueError, match="idempotency"):
        await db.record_returnclock_delivery_event(
            42,
            "decision-1",
            event_id="event-1",
            event_type="opened",
        )


@pytest.mark.asyncio
async def test_assignment_probability_rejects_invalid_values_before_database_write():
    db = _db()

    with pytest.raises(ValueError, match="assignment_probability"):
        await db.create_returnclock_decision(
            42,
            decision="schedule",
            policy_version="policy-v1",
            assignment_probability=1.1,
        )


@pytest.mark.asyncio
async def test_stale_cancellation_only_targets_pending_discretionary_rows_and_records_event():
    db = _db()
    calls: list[tuple[str, tuple]] = []

    async def fake_fetch(query: str, *args):
        calls.append((query, args))
        return [{"id": 11}, {"id": 12}]

    db.fetch = fake_fetch
    returned_at = datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc)

    cancelled = await db.cancel_stale_returnclock_notifications(
        42,
        returned_at=returned_at,
        session_id="session-return",
    )

    assert cancelled == [11, 12]
    query, args = calls[0]
    assert "n.status = 'pending'" in query
    assert "n.is_discretionary = TRUE" in query
    assert "n.created_at <= $2" in query
    assert "INSERT INTO returnclock_delivery_events" in query
    assert "'cancelled'" in query
    assert args == (
        42,
        returned_at,
        "user_returned",
        ["reminders"],
        "session-return",
    )


@pytest.mark.asyncio
async def test_dataset_fetch_returns_bounded_raw_streams():
    db = _db()
    calls: list[tuple[str, tuple]] = []
    stream_rows = [
        [{"session_id": "session-1"}],
        [{"decision_id": "decision-1"}],
        [{"event_id": "event-1"}],
    ]

    async def fake_fetch(query: str, *args):
        calls.append((query, args))
        return stream_rows[len(calls) - 1]

    db.fetch = fake_fetch
    start_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
    ingested_before = datetime(2026, 8, 1, 0, 10, tzinfo=timezone.utc)

    result = await db.fetch_returnclock_dataset_rows(
        start_at=start_at,
        end_at=end_at,
        ingested_before=ingested_before,
        user_id=42,
        limit=9_999_999,
    )

    assert result["schema"] == RETURNCLOCK_DATASET_SCHEMA
    assert result["sessions"] == [{"session_id": "session-1"}]
    assert result["decisions"] == [{"decision_id": "decision-1"}]
    assert result["delivery_events"] == [{"event_id": "event-1"}]
    assert result["window"]["limit_per_stream"] == 1_000_000
    assert result["window"]["page_size"] == 50_000
    assert result["window"]["pagination"] == "keyset"
    assert result["window"]["snapshot_isolation"] == "repeatable_read"
    assert result["window"]["ingested_before"] == ingested_before.isoformat()
    assert len(calls) == 3
    assert all(
        args
        == (
            start_at,
            end_at,
            42,
            50_000,
            None,
            None,
            ingested_before,
        )
        for _, args in calls
    )
    assert all("JOIN users u" in query for query, _ in calls)
    assert all("COALESCE(u.is_bot, FALSE) = FALSE" in query for query, _ in calls)
    assert "FROM returnclock_delivery_events linked_event" in calls[1][0]
    assert "s.battle_ids" in calls[0][0]
    assert "s.created_at < $7" in calls[0][0]
    assert "d.created_at < $7" in calls[1][0]
    assert "linked_event.created_at < $7" in calls[1][0]
    assert "e.created_at < $7" in calls[2][0]
    assert "s.updated_at < $2" not in calls[0][0]
    assert "d.updated_at < $2" not in calls[1][0]
    assert "(s.started_at, s.id)" in calls[0][0]
    assert "(e.event_at, e.id)" in calls[2][0]
    source = inspect.getsource(Database.fetch_returnclock_dataset_rows)
    assert 'isolation="repeatable_read"' in source
    assert "readonly=True" in source


@pytest.mark.asyncio
async def test_dataset_fetch_separates_event_and_ingestion_boundaries():
    db = _db()
    event_end = datetime(2026, 8, 1, tzinfo=timezone.utc)

    with pytest.raises(
        ValueError,
        match="ingested_before must be >= end_at",
    ):
        await db.fetch_returnclock_dataset_rows(
            end_at=event_end,
            ingested_before=event_end - timedelta(seconds=1),
        )

    source = inspect.getsource(Database.fetch_returnclock_dataset_rows)
    assert "s.updated_at < $2" not in source
    assert "d.updated_at < $2" not in source
    assert "s.created_at < $7" in source
    assert "d.created_at < $7" in source
    assert "e.created_at < $7" in source


@pytest.mark.asyncio
async def test_legacy_daily_reminder_retry_reuses_assignment_envelope(
    monkeypatch: pytest.MonkeyPatch,
):
    db = _db()
    due_at = datetime(2026, 7, 28, 10, 30, tzinfo=timezone.utc)
    decisions: list[dict] = []
    enqueues: list[dict] = []

    monkeypatch.setattr(
        "infrastructure.match_modes.get_current_extra_arena_mode",
        lambda now: None,
    )

    async def fake_fetch(query: str, *args):
        if "schedule_type = 'shop_particles'" in query:
            return []
        if "schedule_type = 'daily_reminder'" in query:
            return [
                {
                    "user_id": 42,
                    "trophies": 100,
                    "squad_id": None,
                    "extra_pass": "inactive",
                    "wins_since_last_case": 0,
                    "user_created_at": due_at - timedelta(days=10),
                    "schedule_due_at": due_at,
                }
            ]
        raise AssertionError("unexpected scheduled-notification query")

    async def create_decision(user_id: int, **kwargs):
        decisions.append({"user_id": user_id, **kwargs})
        return {"decision_id": kwargs["decision_id"]}

    async def enqueue(user_id: int, **kwargs):
        enqueues.append({"user_id": user_id, **kwargs})
        return len(enqueues) == 1

    async def schedule(*args, **kwargs):
        return None

    db.fetch = fake_fetch
    db.create_returnclock_decision = create_decision
    db.enqueue_notification = enqueue
    db._upsert_notification_schedule = schedule

    await db.enqueue_due_scheduled_notifications()
    await db.enqueue_due_scheduled_notifications()

    assert [row["decision_id"] for row in decisions] == [
        "legacy-daily-reminder:42:2026-07-28",
        "legacy-daily-reminder:42:2026-07-28",
    ]
    assert all(row["eligible_at"] == due_at for row in decisions)
    assert all(row["planned_send_at"] == due_at for row in decisions)
    assert decisions[0]["context"] == decisions[1]["context"]
    assert enqueues[0]["payload"] == enqueues[1]["payload"]
    assert enqueues[0]["returnclock_delivery_id"] == enqueues[1][
        "returnclock_delivery_id"
    ]
    assert enqueues[0]["dedupe_key"] == enqueues[1]["dedupe_key"]


@pytest.mark.asyncio
async def test_regular_notifications_get_observational_assignment_envelopes():
    db = _db()
    calls: dict[str, object] = {}

    async def enabled(user_id: int, category: str) -> bool:
        return True

    async def create_decision(user_id: int, **kwargs):
        calls["decision"] = (user_id, kwargs)
        return {"decision_id": kwargs["decision_id"]}

    async def update_decision(user_id: int, decision_id: str, **kwargs):
        calls["update"] = (user_id, decision_id, kwargs)
        return {"decision_id": decision_id}

    async def insert_outbox(query: str, *args):
        calls["insert"] = (query, args)
        return {
            "id": 77,
            "user_id": args[0],
            "category": args[1],
            "event_type": args[2],
            "payload": json.loads(args[3]),
            "dedupe_key": args[4],
            "returnclock_decision_id": args[5],
            "returnclock_delivery_id": args[6],
            "is_discretionary": args[7],
            "created": True,
        }

    db.is_notification_enabled = enabled
    db.create_returnclock_decision = create_decision
    db.update_returnclock_decision = update_decision
    db.fetchrow = insert_outbox

    assert await db.enqueue_notification(
        42,
        category="shop",
        event_type="shop_particles",
        payload={"section": "shop"},
        dedupe_key="shop:42:2026-07-29",
    )

    decision_user, decision = calls["decision"]
    assert decision_user == 42
    assert decision["decision_source"] == "notification_system"
    assert decision["policy_version"] == "notification-observational-v1"
    assert decision["assignment_probability"] == 1.0
    insert_query, insert_args = calls["insert"]
    assert "returnclock_decision_id" in insert_query
    assert str(insert_args[5]).startswith("notification-observational:")
    assert insert_args[6]


@pytest.mark.asyncio
async def test_notification_retry_repairs_decision_link_after_insert_committed():
    db = _db()
    updates = 0
    rows: list[dict] = []

    async def enabled(user_id: int, category: str) -> bool:
        return True

    async def fetchrow(query: str, *args):
        row = {
            "id": 77,
            "user_id": args[0],
            "category": args[1],
            "event_type": args[2],
            "payload": json.loads(args[3]),
            "dedupe_key": args[4],
            "returnclock_decision_id": args[5],
            "returnclock_delivery_id": args[6],
            "is_discretionary": args[7],
            "created": not rows,
        }
        rows.append(row)
        return row

    async def update(user_id: int, decision_id: str, **kwargs):
        nonlocal updates
        updates += 1
        if updates == 1:
            return None
        return {"decision_id": decision_id, **kwargs}

    db.is_notification_enabled = enabled
    db.fetchrow = fetchrow
    db.update_returnclock_decision = update

    kwargs = {
        "category": "reminders",
        "event_type": "daily",
        "payload": {"day": "2026-07-29"},
        "dedupe_key": "daily:42:2026-07-29",
        "returnclock_decision_id": "decision-42",
    }
    with pytest.raises(
        RuntimeError,
        match="notification_returnclock_link_failed",
    ):
        await db.enqueue_notification(42, **kwargs)

    assert await db.enqueue_notification(42, **kwargs) is False
    assert updates == 2
    assert rows[0]["returnclock_delivery_id"] == rows[1][
        "returnclock_delivery_id"
    ]
    assert "UNION ALL" in inspect.getsource(Database.enqueue_notification)


@pytest.mark.asyncio
async def test_returnclock_attribution_separates_owner_from_delivery_binding():
    db = _db()
    responses = iter(
        [
            {
                "decision_valid": True,
                "delivery_binding_valid": False,
            },
            {
                "decision_valid": False,
                "delivery_binding_valid": False,
            },
        ]
    )

    async def fetchrow(query: str, *args):
        assert "decision_valid" in query
        assert "delivery_binding_valid" in query
        return next(responses)

    db.fetchrow = fetchrow
    details = await db.get_returnclock_attribution_validation(
        42,
        "owned-broadcast",
        delivery_id="not-yet-recorded",
    )
    assert details == {
        "decision_valid": True,
        "delivery_binding_valid": False,
    }
    assert await db.validate_returnclock_attribution(
        42,
        "foreign",
        delivery_id="fake",
    ) is False


def test_notification_claim_recovers_abandoned_sending_rows():
    source = inspect.getsource(Database.fetch_pending_notifications)

    assert "WHEN attempts >= 5 THEN 'failed'" in source
    assert "COALESCE(last_attempt_at, created_at)" in source
    assert "INTERVAL '10 minutes'" in source
    assert "last_attempt_at = NOW()" in source
