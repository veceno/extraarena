from datetime import datetime, timedelta, timezone

from infrastructure import database
from web import server

def test_ultra_counts_as_extra_pass_access():
    access = server._extra_pass_access("ultra")

    assert access["has_extra_pass"] is True
    assert access["has_ultra"] is True


def test_active_pass_has_premium_without_ultra_access():
    access = server._extra_pass_access("active")

    assert access["has_extra_pass"] is True
    assert access["has_ultra"] is False


def test_expired_pass_is_treated_as_inactive():
    access = server._extra_pass_access(
        "ultra",
        datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    assert access["mode"] == "inactive"
    assert access["has_extra_pass"] is False
    assert access["has_ultra"] is False


def test_future_pass_keeps_effective_tier():
    access = server._extra_pass_access(
        "ultra",
        (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )

    assert access["mode"] == "ultra"
    assert access["has_extra_pass"] is True
    assert access["has_ultra"] is True


def test_database_effective_pass_helper_honors_expiry():
    assert database._extra_pass_mode_active(
        {
            "extra_pass": "active",
            "extra_pass_expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
        }
    )
    assert not database._extra_pass_mode_active(
        {
            "extra_pass": "ultra",
            "extra_pass_expires_at": datetime.now(timezone.utc) - timedelta(hours=1),
        }
    )
