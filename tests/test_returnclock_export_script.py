from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from scripts import export_returnclock_dataset


TEST_PRIVACY_SALT = "returnclock-export-hmac-secret-32-bytes-minimum"


@pytest.mark.asyncio
async def test_returnclock_export_uses_bounded_raw_streams_and_hides_user_ids(
    monkeypatch,
    tmp_path,
):
    start = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    calls: dict = {}

    class FakeDatabase:
        def __init__(self, settings):
            calls["settings"] = settings

        async def connect(self):
            calls["connected"] = True

        async def close(self):
            calls["closed"] = True

        async def fetch_returnclock_dataset_rows(self, **kwargs):
            calls["fetch"] = kwargs
            return {
                "schema": "returnclock_dataset_raw_v1",
                "sessions": [
                    {
                        "user_id": 123456,
                        "session_id": "s1",
                        "source": "android_app",
                        "analytics_version": 2,
                        "timezone": "Europe/Moscow",
                        "utc_offset_minutes": 180,
                        "started_at": start,
                        "ended_at": start + timedelta(minutes=5),
                        "duration_seconds": 300,
                        "screens_visited": [
                            {"screen": "menu"},
                            {"screen": "collection"},
                        ],
                        "battles_played": 0,
                        "cases_opened": 0,
                    }
                ],
                "decisions": [],
                "delivery_events": [],
            }

    monkeypatch.setattr(export_returnclock_dataset, "Database", FakeDatabase)
    monkeypatch.setattr(
        export_returnclock_dataset,
        "get_settings",
        lambda: SimpleNamespace(database="db-settings"),
    )
    monkeypatch.setenv("RETURNCLOCK_DATASET_SALT", TEST_PRIVACY_SALT)
    monkeypatch.setenv(
        "RETURNCLOCK_DATASET_SALT_KEY_ID",
        "returnclock-export-test-v1",
    )
    output = tmp_path / "returnclock.jsonl"
    args = SimpleNamespace(
        output=output,
        start=start.isoformat(),
        end=(start + timedelta(days=8)).isoformat(),
        horizon_hours=168,
        min_analytics_version=2,
        limit=123,
        user_id=None,
        salt_env="RETURNCLOCK_DATASET_SALT",
    )

    result = await export_returnclock_dataset._export(args)

    assert result["ok"] is True
    assert result["safety_lag_minutes"] == 0
    assert calls["connected"] is True
    assert calls["closed"] is True
    assert calls["fetch"]["start_at"] == start - timedelta(days=28)
    assert calls["fetch"]["end_at"] == start + timedelta(days=8)
    assert (
        calls["fetch"]["ingested_before"]
        > calls["fetch"]["end_at"]
    )
    assert calls["fetch"]["limit"] == 123
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["summary"]["example_count"] == 1
    assert records[0]["dataset_end"].endswith("Z")
    assert records[0]["ingested_before"].endswith("Z")
    assert (
        records[0]["ingested_before"]
        == result["ingested_before"].replace("+00:00", "Z")
    )
    assert (
        records[0]["pseudonymization_key_id"]
        == "returnclock-export-test-v1"
    )
    assert TEST_PRIVACY_SALT not in output.read_text(encoding="utf-8")
    assert "user_id" not in records[1]
    assert records[1]["user_id_hash"] != "123456"


@pytest.mark.asyncio
async def test_returnclock_export_fails_closed_when_a_raw_stream_hits_limit(
    monkeypatch,
    tmp_path,
):
    class FakeDatabase:
        def __init__(self, settings):
            pass

        async def connect(self):
            pass

        async def close(self):
            pass

        async def fetch_returnclock_dataset_rows(self, **kwargs):
            return {
                "schema": "returnclock_dataset_raw_v1",
                "sessions": [{"user_id": 1}],
                "decisions": [],
                "delivery_events": [],
            }

    monkeypatch.setattr(export_returnclock_dataset, "Database", FakeDatabase)
    monkeypatch.setattr(
        export_returnclock_dataset,
        "get_settings",
        lambda: SimpleNamespace(database="db-settings"),
    )
    monkeypatch.setenv("RETURNCLOCK_DATASET_SALT", TEST_PRIVACY_SALT)
    monkeypatch.setenv(
        "RETURNCLOCK_DATASET_SALT_KEY_ID",
        "returnclock-export-test-v1",
    )
    args = SimpleNamespace(
        output=tmp_path / "must-not-exist.jsonl",
        start="2026-07-01T00:00:00Z",
        end="2026-07-02T00:00:00Z",
        horizon_hours=24,
        min_analytics_version=2,
        limit=1,
        user_id=None,
        salt_env="RETURNCLOCK_DATASET_SALT",
    )

    with pytest.raises(RuntimeError, match="sessions"):
        await export_returnclock_dataset._export(args)

    assert not args.output.exists()


def test_returnclock_export_defaults_to_a_ten_minute_ingestion_lag():
    args = export_returnclock_dataset._build_parser().parse_args(
        ["--output", "returnclock.jsonl"]
    )

    assert args.safety_lag_minutes == 10
    assert args.end is None


@pytest.mark.asyncio
async def test_returnclock_export_rejects_explicit_future_end_before_database_access(
    monkeypatch,
    tmp_path,
):
    database_accessed = False

    class UnexpectedDatabase:
        def __init__(self, settings):
            nonlocal database_accessed
            database_accessed = True

    monkeypatch.setattr(export_returnclock_dataset, "Database", UnexpectedDatabase)
    monkeypatch.setenv("RETURNCLOCK_DATASET_SALT", TEST_PRIVACY_SALT)
    args = SimpleNamespace(
        output=tmp_path / "must-not-exist.jsonl",
        start=None,
        end=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        horizon_hours=168,
        min_analytics_version=2,
        limit=100,
        user_id=None,
        salt_env="RETURNCLOCK_DATASET_SALT",
        safety_lag_minutes=10,
    )

    with pytest.raises(ValueError, match="future"):
        await export_returnclock_dataset._export(args)

    assert database_accessed is False
    assert not args.output.exists()


@pytest.mark.asyncio
async def test_returnclock_export_rejects_weak_salt_before_database_access(
    monkeypatch,
    tmp_path,
):
    database_accessed = False

    class UnexpectedDatabase:
        def __init__(self, settings):
            nonlocal database_accessed
            database_accessed = True

    monkeypatch.setattr(export_returnclock_dataset, "Database", UnexpectedDatabase)
    monkeypatch.setattr(
        export_returnclock_dataset,
        "get_settings",
        lambda: SimpleNamespace(database="must-not-be-used"),
    )
    monkeypatch.setenv("RETURNCLOCK_DATASET_SALT", "weak")
    args = SimpleNamespace(
        output=tmp_path / "must-not-exist.jsonl",
        start="2026-07-01T00:00:00Z",
        end="2026-07-02T00:00:00Z",
        horizon_hours=24,
        min_analytics_version=2,
        limit=100,
        user_id=None,
        salt_env="RETURNCLOCK_DATASET_SALT",
        safety_lag_minutes=10,
    )

    with pytest.raises((RuntimeError, ValueError), match="32"):
        await export_returnclock_dataset._export(args)

    assert database_accessed is False
    assert not args.output.exists()
