from __future__ import annotations

import inspect

from web.server import create_web_app


def _analytics_block() -> str:
    source = inspect.getsource(create_web_app)
    return source.split(
        "# ── Analytics endpoints (public: session + onboarding tracking) ──",
        1,
    )[1].split(
        "# ── Analytics endpoints (admin only) ──",
        1,
    )[0]


def test_returnclock_session_start_is_attributed_and_telemetry_is_best_effort():
    source = _analytics_block()

    assert "analytics_version=analytics_version" in source
    assert "returnclock_decision_id=decision_id" in source
    assert "returnclock_delivery_id=delivery_id" in source
    assert "return_status=True" in source
    assert 'event_type="deeplink_opened"' in source
    assert 'event_id=f"session:{session_id}:deeplink_opened"' in source
    assert "and active" in source
    assert 'hasattr(db, "cancel_stale_returnclock_notifications")' in source
    assert (
        "persisted_attribution_verified\n"
        "            and persisted_decision_id\n"
        "            and hasattr("
        'db, "record_returnclock_delivery_event")'
    ) in source
    assert '"created": created' in source
    assert '"active": active' in source
    assert "get_returnclock_attribution_validation" in source
    assert '"returnclock_delivery_verified": delivery_verified' in source
    assert "ReturnClock stale-notification cancellation failed" in source
    assert "ReturnClock deeplink attribution failed" in source
    assert 'source not in {"web", "webapp", "telegram_webapp", "android_app"}' in source
    assert 'entrypoint = "notification" if entrypoint_raw == "notification" else None' in source
    assert "not 1 <= notification_id <= 9_223_372_036_854_775_807" in source


def test_returnclock_session_updates_forward_resume_without_reopening_closed_rows():
    source = _analytics_block()

    assert "updated = await db.update_user_session(" in source
    assert "user_id,\n            session_id," in source
    assert "resumed=data.get(\"resumed\") is True" in source
    # The public payload may report a real foreground resume. The database
    # lifecycle guard, tested separately, is what makes this safe for closed
    # rows.
    assert "resumed=False" not in source


def test_returnclock_session_endpoints_use_server_deduped_battle_ids():
    source = _analytics_block()

    assert "def _safe_battle_ids(" in source
    assert "if battle_id in seen:" in source
    assert "battle_ids = _safe_battle_ids(data.get(\"battle_ids\"))" in source
    # Both heartbeat updates and terminal writes persist the same identity
    # set; a client counter alone is never authoritative.
    assert source.count("battle_ids=battle_ids") >= 2


def test_returnclock_session_end_accepts_only_bounded_aware_client_timestamp():
    source = _analytics_block()

    assert "datetime.fromisoformat" in source
    assert "parsed.tzinfo is None or parsed.utcoffset() is None" in source
    assert "parsed > now + timedelta(minutes=5)" in source
    assert "parsed < now - timedelta(days=31)" in source
    assert "ended_at=ended_at" in source


def test_returnclock_client_telemetry_is_bounded_and_screen_shaped():
    source = _analytics_block()

    assert "if depth >= 5:" in source
    assert "list(value.items())[:64]" in source
    assert "value[:2048]" in source
    assert "for item in value[:max_items]:" in source
    assert 'screen = str(item.get("screen") or "").strip()[:128]' in source
    assert 'normalized: dict[str, Any] = {"screen": screen}' in source
    assert '"authorization"' in source
