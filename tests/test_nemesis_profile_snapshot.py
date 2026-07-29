from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from infrastructure.database import Database


def _run(awaitable):
    return asyncio.run(awaitable)


def _db(row):
    db = Database.__new__(Database)
    db._pool = True
    calls = []

    async def fetchrow(query, *args):
        calls.append((query, args))
        return row

    db.fetchrow = fetchrow
    return db, calls


def test_nemesis_snapshot_is_bounded_deidentified_and_time_causal():
    captured = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 7, 28, 11, 55, tzinfo=timezone.utc)
    db, calls = _db(
        {
            "captured_at": captured,
            "trophies": 1234,
            "total": 4,
            "wins": 2,
            "losses": 1,
            "draws": 1,
            "avg_turns": 8.5,
            "avg_duration_seconds": 91.25,
            "recent": [
                {
                    "result": "win",
                    "opponent_actor_type": "bot",
                    "game_mode": "classic",
                    "completed_at": completed,
                    "duration_seconds": 88,
                    "turns_count": 8,
                    "trophy_change": 20,
                    "started_first": True,
                }
            ],
        }
    )

    snapshot = _run(db.get_nemesis_profile_snapshot(101, history_limit=999))

    assert calls[0][1] == (101, 128)
    query = calls[0][0]
    assert "bs.created_at < cutoff.captured_at" in query
    assert "FROM battle_results br, cutoff" in query
    assert "NOT EXISTS" in query
    assert "LIMIT $2" in query
    assert snapshot["captured_at"] == "2026-07-28T12:00:00Z"
    assert snapshot["profile"] == {"wins": 2, "losses": 1, "trophies": 1234}
    assert snapshot["recent"][0]["completed_at"] == "2026-07-28T11:55:00Z"
    serialized = repr(snapshot)
    assert "match_id" not in serialized
    assert "opponent_id" not in serialized
    assert "name" not in serialized


def test_nemesis_snapshot_fails_closed_for_missing_user():
    db, _calls = _db(None)
    with pytest.raises(LookupError, match="user_not_found"):
        _run(db.get_nemesis_profile_snapshot(404))


def test_nemesis_snapshot_requires_database():
    db = Database.__new__(Database)
    db._pool = None
    with pytest.raises(RuntimeError, match="не подключена"):
        _run(db.get_nemesis_profile_snapshot(101))
