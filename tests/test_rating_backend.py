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

    # Base CTE always excludes bots/banned from rating candidates (regardless
    # of the rating_human_vs_human flag, which only gates per-battle filters).
    assert "WHERE COALESCE(u.is_bot, FALSE) = FALSE" in rating_block
    assert "COALESCE(u.status, 'active') IN ('active', 'warn')" in rating_block
    assert "auth_source" not in rating_block
    # The per-battle human-vs-human filter is gated behind the runtime flag.
    assert "human_vs_human: bool" in rating_block
    assert "_bs_clause" in rating_block
    assert "_br_clause" in rating_block


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

    async def _feat(_feature_key: str) -> bool:
        return True

    db.is_feature_enabled = _feat
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
    end = source.index("    # ── Community CRUD", start)
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


def _make_rating_db():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))

    async def _noop(*args, **kwargs):
        return None

    db._ensure_rating_snapshot_cache_table = _noop
    db.refresh_due_rating_snapshots = _noop
    return db


def test_community_rating_surfaces_snapshot_status_error_row():
    import asyncio

    db = _make_rating_db()
    rows = [
        {
            "period": "daily",
            "status": "error",
            "payload": None,
            "generated_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "next_refresh_at": datetime(2026, 6, 23, 1, tzinfo=timezone.utc),
            "source_count": 0,
            "error": "refresh_failed",
        },
        {
            "period": "preview",
            "status": "ready",
            "payload": '{"categories": []}',
            "generated_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "next_refresh_at": datetime(2026, 6, 23, 1, tzinfo=timezone.utc),
            "source_count": 0,
            "error": None,
        },
    ]

    async def fake_fetch(query, *args):
        return rows

    db.fetch = fake_fetch

    result = asyncio.run(db.get_community_rating(scope="players"))
    assert result["periods"]["daily"]["snapshot_status"] == "error"
    assert result["periods"]["daily"]["snapshot_error"] == "refresh_failed"
    assert result["periods"]["preview"]["snapshot_status"] == "ready"


def test_community_rating_surfaces_snapshot_status_ready_with_empty_categories():
    import asyncio

    db = _make_rating_db()
    rows = [
        {
            "period": "daily",
            "status": "ready",
            "payload": {"categories": []},
            "generated_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "next_refresh_at": datetime(2026, 6, 23, 1, tzinfo=timezone.utc),
            "source_count": 0,
            "error": None,
        },
        {
            "period": "preview",
            "status": "ready",
            "payload": {"categories": []},
            "generated_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "next_refresh_at": datetime(2026, 6, 23, 1, tzinfo=timezone.utc),
            "source_count": 0,
            "error": None,
        },
    ]

    async def fake_fetch(query, *args):
        return rows

    db.fetch = fake_fetch

    result = asyncio.run(db.get_community_rating(scope="players"))
    assert result["periods"]["daily"]["snapshot_status"] == "ready"
    assert result["periods"]["preview"]["snapshot_status"] == "ready"


def test_community_rating_marks_corrupt_ready_payload_as_error_and_warns(caplog):
    import asyncio
    import logging

    db = _make_rating_db()
    rows = [
        {
            "period": "daily",
            "status": "ready",
            "payload": "not-json{corrupt",
            "generated_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "next_refresh_at": datetime(2026, 6, 23, 1, tzinfo=timezone.utc),
            "source_count": 0,
            "error": None,
        },
        {
            "period": "preview",
            "status": "ready",
            "payload": {"categories": []},
            "generated_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "next_refresh_at": datetime(2026, 6, 23, 1, tzinfo=timezone.utc),
            "source_count": 0,
            "error": None,
        },
    ]

    async def fake_fetch(query, *args):
        return rows

    db.fetch = fake_fetch

    with caplog.at_level(logging.WARNING, logger="infrastructure.database"):
        result = asyncio.run(db.get_community_rating(scope="players"))

    assert result["periods"]["daily"]["snapshot_status"] == "error"
    assert any("payload is not a dict" in record.getMessage() for record in caplog.records)
    assert result["periods"]["preview"]["snapshot_status"] == "ready"


def test_rating_unavailable_response_includes_snapshot_status():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    response = db._rating_unavailable_response(scope="players")
    assert response["periods"]["daily"]["snapshot_status"] == "error"
    assert response["periods"]["preview"]["snapshot_status"] == "error"
    assert response["periods"]["daily"]["status"] == "error"


def _rating_db_with_flag(flag: bool):
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))

    async def _feat(feature_key: str) -> bool:
        return flag

    db.is_feature_enabled = _feat
    return db


def test_rating_fetch_entries_includes_per_battle_bot_filter_when_human_vs_human_true():
    import asyncio

    db = _rating_db_with_flag(True)
    conn = _RecordingRatingConn()
    generated_at = datetime(2026, 6, 15, 12, 30, tzinfo=timezone.utc)

    asyncio.run(db._build_player_rating_payload(period="daily", generated_at=generated_at, conn=conn))

    assert len(conn.calls) == 4
    trophies_query = conn.calls[0][0]
    score_query = conn.calls[1][0]
    # Per-battle is_bot/metadata filters present (human-vs-human enforced).
    assert "COALESCE(u1.is_bot, FALSE) = FALSE" in trophies_query
    assert "COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'" in trophies_query
    assert "COALESCE(u1.is_bot, FALSE) = FALSE" in score_query
    assert "COALESCE(bs.metadata->>'p1_is_bot', 'false') <> 'true'" in score_query
    # Base CTE human_players is_bot filter still present unconditionally.
    assert "WHERE COALESCE(u.is_bot, FALSE) = FALSE" in trophies_query
    assert "WHERE COALESCE(u.is_bot, FALSE) = FALSE" in score_query


def test_rating_fetch_entries_omits_per_battle_bot_filter_when_human_vs_human_false():
    import asyncio

    db = _rating_db_with_flag(False)
    conn = _RecordingRatingConn()
    generated_at = datetime(2026, 6, 15, 12, 30, tzinfo=timezone.utc)

    asyncio.run(db._build_player_rating_payload(period="daily", generated_at=generated_at, conn=conn))

    assert len(conn.calls) == 4
    trophies_query = conn.calls[0][0]
    score_query = conn.calls[1][0]
    # Per-battle is_bot/metadata filters removed (bot-vs-human battles count).
    assert "COALESCE(u1.is_bot, FALSE) = FALSE" not in trophies_query
    assert "COALESCE(u2.is_bot, FALSE) = FALSE" not in trophies_query
    assert "metadata->>'p1_is_bot'" not in trophies_query
    assert "COALESCE(u1.is_bot, FALSE) = FALSE" not in score_query
    assert "COALESCE(u2.is_bot, FALSE) = FALSE" not in score_query
    assert "metadata->>'p1_is_bot'" not in score_query
    # Base CTE human_players is_bot filter still present (bots never top candidates).
    assert "WHERE COALESCE(u.is_bot, FALSE) = FALSE" in trophies_query
    assert "WHERE COALESCE(u.is_bot, FALSE) = FALSE" in score_query


def test_runtime_config_includes_rating_human_vs_human_default_false():
    import asyncio

    db = _make_rating_db()

    async def fake_fetch(query, *args):
        return []

    db.fetch = fake_fetch

    config = asyncio.run(db.get_runtime_config())
    assert "rating_human_vs_human" in config["feature_availability"]
    assert config["feature_availability"]["rating_human_vs_human"] is False


def test_rating_squad_categories_are_wired():
    from infrastructure.database import RATING_SQUAD_CATEGORIES, RATING_SCOPES

    keys = [c["key"] for c in RATING_SQUAD_CATEGORIES]
    assert keys == ["squad_cbrp", "squad_score", "squad_items"]
    titles = {c["key"]: c["title"] for c in RATING_SQUAD_CATEGORIES}
    assert titles["squad_cbrp"] == "Истинные короли"
    assert titles["squad_score"] == "Монополисты на победы"
    assert titles["squad_items"] == "Золотые сокрома"
    assert "squads" in RATING_SCOPES
    assert "players" in RATING_SCOPES


def test_squad_rating_payload_uses_period_window_in_each_query():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))

    class _RecordingSquadConn:
        def __init__(self):
            self.calls = []

        async def fetch(self, query, *args):
            self.calls.append((query, args))
            return []

    import asyncio

    async def _feat(_feature_key: str) -> bool:
        return True

    db.is_feature_enabled = _feat
    conn = _RecordingSquadConn()
    generated_at = datetime(2026, 6, 15, 12, 30, tzinfo=timezone.utc)

    asyncio.run(db._build_squad_rating_payload(period="preview", generated_at=generated_at, conn=conn))

    assert len(conn.calls) == 3
    for query, args in conn.calls:
        assert generated_at in args
        assert any(getattr(arg, "tzinfo", None) is not None and arg < generated_at for arg in args)
    # Каждая SQL-ветка должна фильтровать по дате.
    joined = "\n".join(q for q, _ in conn.calls)
    assert "squad_cbrp_events" in joined and "e.created_at >= $" in joined
    assert "battle_summary" in joined and "bs.created_at >= $" in joined
    assert "user_cosmetics" in joined and "uco.acquired_at >= $" in joined


def test_community_rating_supports_squad_scope_dispatch():
    db = _make_rating_db()
    rows = [
        {
            "period": "daily",
            "status": "ready",
            "payload": {"categories": [
                {"key": "squad_cbrp", "title": "CBRP", "subtitle": "", "metric_label": "CBRP", "entries": [], "status": "empty"},
            ]},
            "generated_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "next_refresh_at": datetime(2026, 6, 23, 1, tzinfo=timezone.utc),
            "source_count": 0,
            "error": None,
        },
        {
            "period": "preview",
            "status": "ready",
            "payload": {"categories": []},
            "generated_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
            "next_refresh_at": datetime(2026, 6, 23, 1, tzinfo=timezone.utc),
            "source_count": 0,
            "error": None,
        },
    ]

    async def fake_fetch(query, *args):
        return rows

    db.fetch = fake_fetch
    import asyncio

    result = asyncio.run(db.get_community_rating(scope="squads"))
    assert result["scope"] == "squads"
    assert result["success"] is True
    assert result["periods"]["daily"]["status"] == "ready"
    # Категории в squad-scope содержат squad_*-ключи
    cats = result["periods"]["daily"]["categories"]
    assert any(c["key"] == "squad_cbrp" for c in cats)


def test_refresh_due_rating_snapshots_routes_by_scope():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    # Блок refresh_due_rating_snapshots: ранний return убран, есть if/elif по scope.
    start = source.index("async def refresh_due_rating_snapshots")
    end = source.index("def _rating_entry_from_row", start)
    block = source[start:end]
    assert "if scope not in RATING_SCOPES" in block
    assert "_build_squad_rating_payload" in block
    assert "_build_player_rating_payload" in block


def test_rating_snapshot_refresh_loop_refreshes_both_scopes():
    server = Path("web/server.py").read_text(encoding="utf-8")
    start = server.index("async def _rating_snapshot_refresh_loop")
    end = server.index("async def start_background_tasks", start)
    loop_block = server[start:end]
    assert 'for scope in ("players", "squads")' in loop_block
    assert "refresh_due_rating_snapshots(scope=scope)" in loop_block


def test_rating_empty_period_uses_scope_specific_categories():
    db = Database(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
    player_empty = db._rating_empty_period("daily", scope="players", status="pending")
    squad_empty = db._rating_empty_period("daily", scope="squads", status="pending")
    player_keys = {c["key"] for c in player_empty["categories"]}
    squad_keys = {c["key"] for c in squad_empty["categories"]}
    assert {"trophies", "score", "cbrp", "items"} <= player_keys
    assert {"squad_cbrp", "squad_score", "squad_items"} <= squad_keys
    assert player_keys.isdisjoint(squad_keys)


def test_rating_player_subtitle_uses_russian_formula():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    assert "Счет = победы × процент побед." in source
    assert "score = wins × winrate." not in source
