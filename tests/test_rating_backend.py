from datetime import datetime, timezone
from pathlib import Path

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database, _rating_next_refresh_at


class _RecordingRatingConn:
    def __init__(self):
        self.calls = []

    async def fetch(self, query, *args):
        self.calls.append((query, args))
        return []


def test_rating_next_refresh_uses_global_moscow_cycles():
    now = datetime(2026, 5, 29, 20, 58, 10, tzinfo=timezone.utc)

    daily = _rating_next_refresh_at("daily", now)
    preview = _rating_next_refresh_at("preview", now)

    assert daily.isoformat() == "2026-05-29T21:00:00+00:00"
    assert preview.isoformat() == "2026-05-29T21:00:00+00:00"


def test_rating_cache_schema_and_refresh_methods_exist():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")

    assert "rating_snapshot_cache" in source
    assert "_ensure_rating_snapshot_cache_table" in source
    assert "refresh_due_rating_snapshots" in source
    assert "get_community_rating" in source
    assert "pg_try_advisory_xact_lock" in source
    assert "RATING_ALGORITHM_VERSION" in source
    assert "payload.get(\"algorithm_version\") != RATING_ALGORITHM_VERSION" in source


def test_rating_queries_exclude_bots_and_banned_without_auth_source_filter():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    start = source.index("async def _rating_fetch_entries")
    end = source.index("    # ── Community CRUD", start)
    rating_block = source[start:end]

    assert "COALESCE(u.is_bot, FALSE) = FALSE" in rating_block
    assert "COALESCE(u.status, 'active') IN ('active', 'warn')" in rating_block
    assert "auth_source" not in rating_block


def test_rating_categories_use_expected_metrics_and_sources():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")

    assert "GREATEST(COALESCE(u.max_trophies, 0), COALESCE(u.trophies, 0))" in source
    assert "wins * winrate" in source
    assert "wins * SQRT(GREATEST(battles, 1)) * winrate" not in source
    assert "squad_cbrp_events" in source
    assert "COUNT(DISTINCT uco.cosmetic_id)" in source
    start = source.index("elif category_key == \"items\"")
    end = source.index("        else:", start)
    items_block = source[start:end]
    assert "COUNT(DISTINCT uc.card_id)" not in items_block
    assert "user_cases" not in items_block


def test_rating_payload_passes_period_window_to_category_queries():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    conn = _RecordingRatingConn()
    generated_at = datetime(2026, 6, 15, 12, 30, tzinfo=timezone.utc)

    import asyncio
    asyncio.run(db._build_player_rating_payload(period="preview", generated_at=generated_at, conn=conn))

    assert len(conn.calls) == 4
    for _query, args in conn.calls:
        assert generated_at in args
        assert any(getattr(arg, "tzinfo", None) is not None and arg < generated_at for arg in args)


def test_rating_queries_apply_time_windows_to_period_sensitive_sources():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    start = source.index("async def _rating_fetch_entries")
    end = source.index("        else:", start)
    rating_block = source[start:end]

    assert "bs.created_at >= $" in rating_block
    assert "br.created_at >= $" in rating_block
    assert "squad_cbrp_events" in rating_block and "created_at >= $" in rating_block
    assert "uco.acquired_at >= $" in rating_block


def test_preview_rating_masks_top_three_on_backend():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")

    assert "def _rating_mask_preview_entry" in source
    assert "period == \"preview\" and entry.get(\"rank\", 0) <= 3" in source
    mask_start = source.index("def _rating_mask_preview_entry")
    mask_end = source.index("async def _build_player_rating_payload", mask_start)
    mask_block = source[mask_start:mask_end]

    assert "\"masked\": True" in mask_block
    assert "\"display_name\"" not in mask_block
    assert "\"metric\"" not in mask_block


def test_rating_refresh_marks_existing_payload_stale_on_error():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    start = source.index("Rating snapshot refresh failed")
    end = source.index("return {\"success\": True, \"scope\": scope", start)
    error_block = source[start:end]

    assert "ELSE 'stale'" in error_block
    assert "stale_payload" in error_block


def test_preview_rating_is_remasked_when_reading_stale_cache():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    start = source.index("async def get_community_rating")
    end = source.index("    async def refresh_due_rating_snapshots", start)
    read_block = source[start:end]

    assert "self._rating_mask_preview_categories(categories)" in read_block


def test_preview_rating_remask_tolerates_malformed_stale_entries():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))

    categories = db._rating_mask_preview_categories([
        {"key": "score", "entries": [
            {"rank": "1", "display_name": "hidden", "metric": 99},
            {"rank": "oops", "display_name": "kept", "metric": 10},
            {"rank": 4, "display_name": "visible", "metric": 5},
        ]}
    ])

    assert categories[0]["entries"][0] == {"rank": 1, "masked": True}
    assert categories[0]["entries"][1]["display_name"] == "kept"
    assert categories[0]["entries"][2]["display_name"] == "visible"


def test_rating_cache_migration_guarantees_scope_period_conflict_target():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    start = source.index("async def _ensure_rating_snapshot_cache_table")
    end = source.index("    def _rating_empty_period", start)
    schema_block = source[start:end]

    assert "rating_snapshot_cache_scope_period_key" in schema_block
    assert "UNIQUE (scope, period)" in schema_block
    assert "DELETE FROM rating_snapshot_cache" in schema_block
    assert "ROW_NUMBER() OVER (PARTITION BY scope, period" in schema_block


def test_rating_api_route_and_runtime_gate_are_registered():
    server = Path("web/server.py").read_text(encoding="utf-8")
    database = Path("infrastructure/database.py").read_text(encoding="utf-8")

    assert "/api/community/rating" in server
    assert "(\"rating\", (\"/api/community/rating\",))" in server
    assert "_rating_snapshot_refresh_loop" in server
    assert "_rating_unavailable_response" in database
    assert "get_community_rating failed" in database


def test_rating_background_loop_uses_non_aggressive_poll_interval():
    server = Path("web/server.py").read_text(encoding="utf-8")
    start = server.index("async def _rating_snapshot_refresh_loop")
    end = server.index("    async def start_background_tasks", start)
    loop_block = server[start:end]

    assert "await asyncio.sleep(300)" in loop_block


def test_community_handlers_do_not_return_raw_exception_messages():
    server = Path("web/server.py").read_text(encoding="utf-8")
    community_block = server[
        server.index("    # ── Image upload"):
        server.index("    app.router.add_get(\"/api/recent-opponents\"", server.index("    # ── Image upload"))
    ]

    assert "str(e)" not in community_block
    assert "str(exc)" not in community_block
