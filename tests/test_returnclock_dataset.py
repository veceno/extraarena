from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import stat

import pytest

from infrastructure.returnclock_dataset import (
    DATASET_FORMAT,
    ReturnClockDatasetError,
    materialize_returnclock_dataset,
    write_returnclock_jsonl,
)
from rlhf_env.components.dataset_toolbox import DatasetToolbox


UTC = timezone.utc
TEST_PRIVACY_SALT = "returnclock-test-hmac-secret-32-bytes-minimum"


def _session(
    session_id: str,
    *,
    user_id: int = 42,
    started_at: datetime,
    duration_seconds: int = 300,
    analytics_version: int = 2,
    decision_id: str | None = None,
) -> dict:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "source": "android_app",
        "started_at": started_at,
        "ended_at": started_at + timedelta(seconds=duration_seconds),
        "duration_seconds": duration_seconds,
        "screens_visited": [
            {"screen": "menu", "ts": int(started_at.timestamp() * 1000)},
            {"screen": "collection", "ts": int(started_at.timestamp() * 1000) + 1000},
        ],
        "battles_played": 0,
        "cases_opened": 0,
        "analytics_version": analytics_version,
        "timezone": "Europe/Moscow",
        "utc_offset_minutes": 180,
        "entrypoint": "notification" if decision_id else "direct",
        "returnclock_decision_id": decision_id,
        "metadata": (
            {"returnclock_attribution_verified": True}
            if decision_id
            else {}
        ),
    }


def test_returnclock_materializer_builds_observed_and_censored_examples():
    first = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    sessions = [
        _session("s1", started_at=first),
        _session("s2", started_at=first + timedelta(hours=25)),
        _session("s3", started_at=first + timedelta(days=10)),
        _session(
            "legacy",
            started_at=first - timedelta(days=1),
            analytics_version=1,
        ),
    ]

    dataset = materialize_returnclock_dataset(
        sessions=sessions,
        dataset_end=first + timedelta(days=20),
        privacy_salt=TEST_PRIVACY_SALT,
        horizon_hours=7 * 24,
    )

    assert dataset.header["format"] == DATASET_FORMAT
    assert dataset.header["pseudonymization_key_id"]
    assert dataset.header["summary"]["excluded_legacy_sessions"] == 1
    assert len(dataset.examples) == 3

    observed = dataset.examples[0]
    assert observed["label"]["target_observed"] is True
    assert observed["label"]["right_censored"] is False
    assert observed["label"]["time_to_return_minutes"] == 24.0 * 60.0 + 55.0
    assert observed["post_cutoff"]["organic_candidate"] is True
    assert observed["features"]["timezone"] == "Europe/Moscow"
    assert observed["features"]["local_hour"] == 19
    assert "user_id" not in observed
    assert len(observed["user_id_hash"]) == 32
    assert set(observed["features"]) == set(dataset.header["feature_columns"])
    assert "next_session_at" not in observed
    assert "target_observed" not in observed["features"]

    full_horizon_negative = dataset.examples[1]
    assert full_horizon_negative["label"]["target_observed"] is False
    assert full_horizon_negative["label"]["right_censored"] is False
    assert (
        full_horizon_negative["label"]["observation_window_minutes"]
        == 7 * 24 * 60
    )

    final = dataset.examples[2]
    assert final["label"]["target_observed"] is False
    assert final["label"]["right_censored"] is False


def test_returnclock_materializer_keeps_notification_treatment_and_attribution():
    first = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    cutoff = first + timedelta(minutes=5)
    next_start = first + timedelta(hours=24)
    sessions = [
        _session("s1", started_at=first),
        _session("s2", started_at=next_start, decision_id="decision-1"),
    ]
    decisions = [
        {
            "user_id": 42,
            "decision_id": "decision-1",
            "decision": "send",
            "experiment_id": "pilot-1",
            "treatment_arm": "returnclock",
            "assignment_probability": 0.5,
            # Assignment can precede the prediction cutoff while its delivery
            # still happens inside the label interval.
            "created_at": cutoff - timedelta(hours=1),
            # Mutable delivery status may be repaired after dataset_end. The
            # immutable assignment must still keep this interval non-organic.
            "updated_at": first + timedelta(days=20),
        }
    ]
    events = [
        {
            "user_id": 42,
            "decision_id": "decision-1",
            "delivery_id": "delivery-1",
            "event_type": "provider_accepted",
            "channel": "android",
            "occurred_at": cutoff + timedelta(hours=20),
        },
        {
            "user_id": 42,
            "decision_id": "decision-1",
            "delivery_id": "delivery-1",
            "event_type": "opened",
            "channel": "android",
            # The server writes deeplink_opened only after session INSERT, so
            # its event_at is slightly later than the return boundary.
            "occurred_at": next_start + timedelta(milliseconds=50),
            "session_id": "s2",
        },
    ]

    dataset = materialize_returnclock_dataset(
        sessions=sessions,
        decisions=decisions,
        delivery_events=events,
        dataset_end=first + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    example = dataset.examples[0]
    treatment = example["post_cutoff"]
    assert treatment["notification_decision_count"] == 1
    assert treatment["provider_accepted_count"] == 1
    assert treatment["notification_sent_count"] == 0
    assert treatment["notification_opened_count"] == 1
    assert treatment["notification_channels"] == ["android"]
    assert treatment["treatment_arms"] == ["returnclock"]
    assert treatment["assignments"] == [
        {
            "experiment_id": "pilot-1",
            "treatment_arm": "returnclock",
            "assignment_probability": 0.5,
            "decision": "send",
            "decision_source": "unknown",
            "policy_version": "unknown",
            "model_version": None,
        }
    ]
    assert treatment["notification_attributed"] is True
    assert treatment["organic_candidate"] is False


def test_attributed_session_counts_as_open_if_best_effort_event_is_missing():
    first = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    sessions = [
        _session("direct", started_at=first),
        _session(
            "notification-return",
            started_at=first + timedelta(hours=24),
            decision_id="decision-without-event",
        ),
    ]

    dataset = materialize_returnclock_dataset(
        sessions=sessions,
        decisions=[
            {
                "user_id": 42,
                "decision_id": "decision-without-event",
                "decision": "send",
                "created_at": first + timedelta(hours=20),
            }
        ],
        delivery_events=[],
        dataset_end=first + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    treatment = dataset.examples[0]["post_cutoff"]
    assert treatment["notification_attributed"] is True
    assert treatment["notification_opened_count"] == 1
    assert treatment["organic_candidate"] is False


def test_unverified_client_attribution_is_ignored():
    first = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    forged = _session(
        "forged-notification",
        started_at=first + timedelta(hours=24),
    )
    forged["entrypoint"] = "notification"
    forged["returnclock_decision_id"] = "foreign-decision"
    forged["returnclock_delivery_id"] = "fake-delivery"
    forged["metadata"] = {
        "returnclock_attribution_verified": False,
    }

    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("direct", started_at=first),
            forged,
        ],
        dataset_end=first + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    treatment = dataset.examples[0]["post_cutoff"]
    assert treatment["notification_attributed"] is False
    assert treatment["notification_opened_count"] == 0
    assert treatment["organic_candidate"] is True


def test_each_verified_short_attributed_session_counts_as_one_open():
    first = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    short_returns = []
    for index, hours in enumerate((2, 4), start=1):
        row = _session(
            f"notification-short-{index}",
            started_at=first + timedelta(hours=hours),
            duration_seconds=20,
            decision_id=f"decision-{index}",
        )
        row["screens_visited"] = row["screens_visited"][:1]
        short_returns.append(row)

    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("cutoff", started_at=first),
            *short_returns,
            _session(
                "meaningful-return",
                started_at=first + timedelta(hours=8),
            ),
        ],
        dataset_end=first + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    treatment = dataset.examples[0]["post_cutoff"]
    assert treatment["notification_attributed"] is True
    assert treatment["notification_opened_count"] == 2
    assert treatment["organic_candidate"] is False


def test_returnclock_materializer_marks_incomplete_final_horizon_right_censored():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[_session("s1", started_at=start)],
        dataset_end=start + timedelta(hours=12),
        privacy_salt=TEST_PRIVACY_SALT,
        horizon_hours=24,
    )

    example = dataset.examples[0]
    assert example["label"]["target_observed"] is False
    assert example["label"]["right_censored"] is True
    assert example["label"]["observation_window_minutes"] == 11 * 60 + 55


def test_non_returnclock_notification_entrypoint_is_not_labeled_organic():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    next_session = _session(
        "from-shop-push",
        started_at=start + timedelta(hours=24),
    )
    next_session["entrypoint"] = "notification"
    next_session["returnclock_delivery_id"] = "delivery-without-rc-decision"
    next_session["metadata"] = {
        "returnclock_attribution_verified": True,
    }

    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("direct", started_at=start),
            next_session,
        ],
        dataset_end=start + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    treatment = dataset.examples[0]["post_cutoff"]
    assert treatment["notification_attributed"] is True
    assert treatment["organic_candidate"] is False


def test_returnclock_jsonl_writer_emits_header_then_examples(tmp_path):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[_session("s1", started_at=start)],
        dataset_end=start + timedelta(days=8),
        privacy_salt=TEST_PRIVACY_SALT,
    )
    output = write_returnclock_jsonl(dataset, tmp_path / "returnclock.jsonl")

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["record_type"] == "header"
    assert records[0]["summary"]["example_count"] == 1
    assert records[1]["record_type"] == "example"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_returnclock_sessionization_merges_reload_fragments_and_overlapping_tabs():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    sessions = [
        _session("doc-1", started_at=start, duration_seconds=60),
        _session(
            "reload",
            started_at=start + timedelta(minutes=2),
            duration_seconds=60,
        ),
        _session(
            "real-return",
            started_at=start + timedelta(hours=3),
            duration_seconds=180,
        ),
    ]
    sessions[0]["screens_visited"] = sessions[0]["screens_visited"][:1]
    sessions[1]["screens_visited"] = sessions[1]["screens_visited"][:1]

    dataset = materialize_returnclock_dataset(
        sessions=sessions,
        dataset_end=start + timedelta(days=8),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    assert dataset.header["sessionization_gap_minutes"] == 30
    assert len(dataset.examples) == 2
    first = dataset.examples[0]
    assert first["features"]["sessions_28d"] == 1
    assert first["features"]["last_session_duration_seconds"] == 120
    assert first["label"]["target_observed"] is True
    assert first["label"]["time_to_return_minutes"] == 177.0


@pytest.mark.parametrize(
    ("duration_seconds", "screen_count"),
    [
        (120, 1),
        (119, 2),
    ],
)
def test_returnclock_meaningful_session_requires_duration_and_screen_thresholds(
    duration_seconds,
    screen_count,
):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    session = _session(
        "single-criterion",
        started_at=start,
        duration_seconds=duration_seconds,
    )
    session["screens_visited"] = session["screens_visited"][:screen_count]

    dataset = materialize_returnclock_dataset(
        sessions=[session],
        dataset_end=start + timedelta(days=8),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    assert dataset.examples == []
    assert dataset.header["summary"]["excluded_nonmeaningful_sessions"] == 1


@pytest.mark.parametrize("override_field", ["battles_played", "cases_opened"])
def test_returnclock_battle_or_case_makes_short_session_meaningful(override_field):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    session = _session(
        f"short-with-{override_field}",
        started_at=start,
        duration_seconds=30,
    )
    session["screens_visited"] = session["screens_visited"][:1]
    session[override_field] = 1

    dataset = materialize_returnclock_dataset(
        sessions=[session],
        dataset_end=start + timedelta(days=8),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    assert len(dataset.examples) == 1
    assert dataset.header["summary"]["excluded_nonmeaningful_sessions"] == 0


@pytest.mark.parametrize(
    ("attribution_field", "attribution_value"),
    [
        ("entrypoint", "notification"),
        ("returnclock_decision_id", "decision-from-reload"),
        ("returnclock_delivery_id", "delivery-from-reload"),
    ],
)
def test_returnclock_sessionization_preserves_late_notification_attribution(
    attribution_field,
    attribution_value,
):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    return_start = start + timedelta(hours=24)
    return_fragment = _session(
        "return-fragment",
        started_at=return_start,
        duration_seconds=60,
    )
    return_fragment["screens_visited"] = return_fragment["screens_visited"][:1]
    reload_fragment = _session(
        "return-reload",
        started_at=return_start + timedelta(minutes=2),
        duration_seconds=60,
    )
    reload_fragment["screens_visited"] = reload_fragment["screens_visited"][:1]
    reload_fragment[attribution_field] = attribution_value
    reload_fragment["metadata"] = {
        "returnclock_attribution_verified": True,
    }

    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("previous", started_at=start),
            return_fragment,
            reload_fragment,
        ],
        dataset_end=start + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    treatment = dataset.examples[0]["post_cutoff"]
    assert treatment["notification_attributed"] is True
    assert treatment["organic_candidate"] is False


def test_returnclock_sessionization_deduplicates_terminal_battle_ids():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    first = _session(
        "battle-tab-1",
        started_at=start,
        duration_seconds=60,
    )
    second = _session(
        "battle-tab-2",
        started_at=start + timedelta(minutes=2),
        duration_seconds=60,
    )
    first["battle_ids"] = ["battle-x"]
    first["battles_played"] = 1
    second["battle_ids"] = ["battle-x", "battle-y"]
    second["battles_played"] = 2

    dataset = materialize_returnclock_dataset(
        sessions=[first, second],
        dataset_end=start + timedelta(days=8),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    assert len(dataset.examples) == 1
    assert dataset.examples[0]["features"]["last_session_battles"] == 2


def test_returnclock_infers_old_open_session_end_but_excludes_recent_open_session():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    stale = _session("stale", started_at=start)
    stale["ended_at"] = None
    stale["last_heartbeat_at"] = start + timedelta(minutes=5)
    recent = _session("recent", started_at=start + timedelta(hours=2))
    recent["ended_at"] = None
    recent["last_heartbeat_at"] = start + timedelta(hours=2, minutes=5)

    dataset = materialize_returnclock_dataset(
        sessions=[stale, recent],
        dataset_end=start + timedelta(hours=2, minutes=20),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    assert dataset.header["summary"]["inferred_stale_session_ends"] == 1
    assert dataset.header["summary"]["excluded_unfinished_sessions"] == 1
    assert len(dataset.examples) == 1
    assert dataset.examples[0]["features"]["last_session_end_inferred"] is True


def test_recent_open_overlap_invalidates_the_whole_coalesced_cutoff():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    closed = _session("closed", started_at=start)
    recent_tab = _session(
        "recent-tab",
        started_at=start + timedelta(minutes=10),
    )
    recent_tab["ended_at"] = None
    recent_tab["last_heartbeat_at"] = start + timedelta(minutes=20)

    dataset = materialize_returnclock_dataset(
        sessions=[closed, recent_tab],
        dataset_end=start + timedelta(minutes=25),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    assert dataset.examples == []


def test_recent_open_meaningful_session_is_target_but_not_cutoff():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    recent_return = _session(
        "recent-return",
        started_at=start + timedelta(hours=2),
    )
    recent_return["ended_at"] = None
    recent_return["last_heartbeat_at"] = start + timedelta(
        hours=2,
        minutes=5,
    )

    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("finished-cutoff", started_at=start),
            recent_return,
        ],
        dataset_end=start + timedelta(hours=2, minutes=20),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    assert len(dataset.examples) == 1
    assert dataset.examples[0]["label"]["target_observed"] is True
    assert dataset.examples[0]["label"]["time_to_return_minutes"] == 115.0


def test_historical_deeplink_open_counts_as_prior_notification():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("first", started_at=start),
            _session("second", started_at=start + timedelta(hours=12)),
        ],
        delivery_events=[
            {
                "user_id": 42,
                "decision_id": "decision-history",
                "delivery_id": "delivery-history",
                "event_type": "deeplink_opened",
                "event_at": start + timedelta(hours=6),
                "session_id": "history-session",
            }
        ],
        dataset_end=start + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
    )

    second = dataset.examples[1]
    assert second["features"]["notifications_24h"] == 1
    assert second["features"]["notifications_7d"] == 1


def test_returnclock_materializer_rejects_weak_pseudonymization_salt():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)

    with pytest.raises(ReturnClockDatasetError, match="32"):
        materialize_returnclock_dataset(
            sessions=[_session("s1", started_at=start)],
            dataset_end=start + timedelta(days=8),
            privacy_salt="too-short",
        )


def test_returnclock_header_records_nonsecret_pseudonymization_key_id():
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[_session("s1", started_at=start)],
        dataset_end=start + timedelta(days=8),
        privacy_salt=TEST_PRIVACY_SALT,
        privacy_key_id="returnclock-test-key-v1",
    )

    assert dataset.header["pseudonymization_key_id"] == "returnclock-test-key-v1"
    serialized = json.dumps(dataset.header)
    assert TEST_PRIVACY_SALT not in serialized


def test_returnclock_validator_requires_grouped_and_temporal_split_support(tmp_path):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    sessions = [
        _session("u1-s1", user_id=1, started_at=start),
        _session("u1-s2", user_id=1, started_at=start + timedelta(hours=24)),
        _session("u2-s1", user_id=2, started_at=start + timedelta(hours=1)),
        _session("u2-s2", user_id=2, started_at=start + timedelta(hours=25)),
        _session("u3-s1", user_id=3, started_at=start + timedelta(hours=2)),
        _session("u3-s2", user_id=3, started_at=start + timedelta(hours=26)),
    ]
    dataset = materialize_returnclock_dataset(
        sessions=sessions,
        dataset_end=start + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
        privacy_key_id="returnclock-test-key-v1",
    )
    datasets_dir = tmp_path / "datasets"
    output = write_returnclock_jsonl(dataset, datasets_dir / "returnclock.jsonl")

    validation = DatasetToolbox(datasets_dir).validate_artifact(output.name)

    assert validation["ok"] is True
    assert validation["training_ready"] is True
    assert validation["summary"]["distinct_users"] == 3
    assert validation["summary"]["grouped_user_split_possible"] is True
    assert validation["summary"]["temporal_split_possible"] is True


def test_returnclock_validator_does_not_claim_two_user_split_ready(tmp_path):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("u1-s1", user_id=1, started_at=start),
            _session(
                "u2-s1",
                user_id=2,
                started_at=start + timedelta(hours=1),
            ),
        ],
        dataset_end=start + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
        privacy_key_id="returnclock-test-key-v1",
    )
    datasets_dir = tmp_path / "datasets"
    output = write_returnclock_jsonl(
        dataset,
        datasets_dir / "returnclock.jsonl",
    )

    validation = DatasetToolbox(datasets_dir).validate_artifact(output.name)

    assert validation["ok"] is True
    assert validation["training_ready"] is False
    assert validation["summary"]["organic_distinct_users"] == 2
    assert validation["summary"]["grouped_user_split_possible"] is False
    assert validation["summary"]["temporal_split_possible"] is False


def test_returnclock_validator_requires_three_first_cutoff_cohorts(tmp_path):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[
            _session(
                f"u{user_id}-s1",
                user_id=user_id,
                started_at=start,
            )
            for user_id in (1, 2, 3)
        ],
        dataset_end=start + timedelta(days=10),
        privacy_salt=TEST_PRIVACY_SALT,
        privacy_key_id="returnclock-test-key-v1",
    )
    datasets_dir = tmp_path / "datasets"
    output = write_returnclock_jsonl(
        dataset,
        datasets_dir / "returnclock.jsonl",
    )

    validation = DatasetToolbox(datasets_dir).validate_artifact(output.name)

    assert validation["ok"] is True
    assert validation["training_ready"] is False
    assert validation["summary"]["organic_distinct_users"] == 3
    assert (
        validation["summary"]["organic_distinct_first_cutoff_cohorts"]
        == 1
    )
    assert validation["summary"]["grouped_user_split_possible"] is True
    assert validation["summary"]["temporal_split_possible"] is False


def test_returnclock_validator_rejects_missing_pseudonymization_key_id(tmp_path):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    dataset = materialize_returnclock_dataset(
        sessions=[
            _session("u1", user_id=1, started_at=start),
            _session("u2", user_id=2, started_at=start + timedelta(hours=1)),
        ],
        dataset_end=start + timedelta(days=8),
        privacy_salt=TEST_PRIVACY_SALT,
        privacy_key_id="returnclock-test-key-v1",
    )
    dataset.header.pop("pseudonymization_key_id")
    datasets_dir = tmp_path / "datasets"
    output = write_returnclock_jsonl(dataset, datasets_dir / "returnclock.jsonl")

    validation = DatasetToolbox(datasets_dir).validate_artifact(output.name)

    assert validation["ok"] is False
    assert any(
        "pseudonymization_key_id" in issue
        for issue in validation["issues"]
    )


def test_returnclock_records_distinct_ingestion_watermark(tmp_path):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    dataset_end = start + timedelta(days=8)
    ingested_before = dataset_end + timedelta(minutes=10)
    dataset = materialize_returnclock_dataset(
        sessions=[_session("u1", started_at=start)],
        dataset_end=dataset_end,
        ingested_before=ingested_before,
        privacy_salt=TEST_PRIVACY_SALT,
        privacy_key_id="returnclock-test-key-v1",
    )

    assert dataset.header["dataset_end"] == "2026-07-09T16:00:00Z"
    assert (
        dataset.header["ingested_before"]
        == "2026-07-09T16:10:00Z"
    )
    datasets_dir = tmp_path / "datasets"
    output = write_returnclock_jsonl(
        dataset,
        datasets_dir / "returnclock.jsonl",
    )
    validation = DatasetToolbox(datasets_dir).validate_artifact(
        output.name
    )
    assert validation["ok"] is True, validation["issues"]

    with pytest.raises(
        ReturnClockDatasetError,
        match="ingested_before must be >= dataset_end",
    ):
        materialize_returnclock_dataset(
            sessions=[_session("u1", started_at=start)],
            dataset_end=dataset_end,
            ingested_before=dataset_end - timedelta(seconds=1),
            privacy_salt=TEST_PRIVACY_SALT,
            privacy_key_id="returnclock-test-key-v1",
        )


def test_returnclock_splitter_is_grouped_by_user_and_temporal(tmp_path):
    start = datetime(2026, 7, 1, 16, 0, tzinfo=UTC)
    sessions = [
        _session(
            f"user-{user_id}",
            user_id=user_id,
            started_at=start + timedelta(days=user_id),
        )
        for user_id in range(12)
    ]
    dataset = materialize_returnclock_dataset(
        sessions=sessions,
        dataset_end=start + timedelta(days=30),
        privacy_salt=TEST_PRIVACY_SALT,
        privacy_key_id="returnclock-test-key-v1",
    )
    datasets_dir = tmp_path / "datasets"
    source = write_returnclock_jsonl(dataset, datasets_dir / "returnclock.jsonl")
    toolbox = DatasetToolbox(datasets_dir)

    result = toolbox.split_returnclock(
        source=source.name,
        output_dir="returnclock-split",
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    assert result["ok"] is True
    split_dir = datasets_dir / "returnclock-split"
    manifest = json.loads(
        (split_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["pseudonymization_key_id"] == "returnclock-test-key-v1"
    assert manifest["strategy"]["grouped_by"] == "user_id_hash"
    assert manifest["strategy"]["ordered_by"] == "prediction_cutoff_at"

    split_rows: dict[str, list[dict]] = {}
    for split_name in ("train", "validation", "test"):
        split_path = split_dir / f"{split_name}.jsonl"
        assert stat.S_IMODE(split_path.stat().st_mode) == 0o600
        split_rows[split_name] = [
            json.loads(line)
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("record_type") == "example"
        ]
        assert split_rows[split_name]

    split_users = {
        split_name: {row["user_id_hash"] for row in rows}
        for split_name, rows in split_rows.items()
    }
    assert split_users["train"].isdisjoint(split_users["validation"])
    assert split_users["train"].isdisjoint(split_users["test"])
    assert split_users["validation"].isdisjoint(split_users["test"])
    assert set().union(*split_users.values()) == {
        row["user_id_hash"] for row in dataset.examples
    }

    # This fixture has one cutoff per user, so chronological user-group
    # assignment produces strict, inspectable temporal boundaries.
    assert max(
        row["prediction_cutoff_at"] for row in split_rows["train"]
    ) < min(
        row["prediction_cutoff_at"] for row in split_rows["validation"]
    )
    assert max(
        row["prediction_cutoff_at"] for row in split_rows["validation"]
    ) < min(
        row["prediction_cutoff_at"] for row in split_rows["test"]
    )
