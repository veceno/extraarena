"""Tests for the win-streak bonus mechanic.

Covers:
- ``_compute_win_streak_bonus`` pure helper (web.server).
- ``_build_battle_history_stats`` win-streak fields (current_win_streak /
  max_win_streak) and the new draw-does-not-break-win-streak semantics.
"""

from web import server as web_server


# ═══════════════════════════════════════════
# _compute_win_streak_bonus
# ═══════════════════════════════════════════

def test_compute_win_streak_bonus_zero_for_no_streak():
    assert web_server._compute_win_streak_bonus(0) == 0


def test_compute_win_streak_bonus_equals_prior_streak():
    # пример из ТЗ: победил 3-й раз при серии 2 -> +2
    assert web_server._compute_win_streak_bonus(2) == 2
    # победил 2-й раз при серии 1 -> +1
    assert web_server._compute_win_streak_bonus(1) == 1


def test_compute_win_streak_bonus_capped_at_ten():
    assert web_server._compute_win_streak_bonus(10) == 10
    assert web_server._compute_win_streak_bonus(19) == 10
    assert web_server._compute_win_streak_bonus(99) == 10


def test_compute_win_streak_bonus_negative_treated_as_zero():
    assert web_server._compute_win_streak_bonus(-3) == 0


# ═══════════════════════════════════════════
# _build_battle_history_stats win-streak fields
# ═══════════════════════════════════════════

def _b(mode, result, **kw):
    base = {"mode": mode, "result": result, "trophies_change": 0,
            "turns_count": 5, "duration_seconds": 60}
    base.update(kw)
    return base


def test_build_stats_current_win_streak_and_max():
    stats = web_server._build_battle_history_stats([
        _b("classic", "win"),    # newest
        _b("classic", "win"),
        _b("classic", "lose"),
        _b("classic", "win"),
        _b("classic", "win"),
        _b("classic", "win"),    # oldest
    ])
    assert stats["current_streak_result"] == "win"
    assert stats["current_streak_count"] == 2
    assert stats["current_win_streak"] == 2
    assert stats["max_win_streak"] == 3


def test_build_stats_draw_does_not_break_win_streak():
    stats = web_server._build_battle_history_stats([
        _b("classic", "win"),     # newest
        _b("classic", "draw"),
        _b("classic", "win"),
        _b("classic", "win"),     # oldest
    ])
    # draw пропускается: W, W, (D skip), W -> 3
    assert stats["current_streak_result"] == "win"
    assert stats["current_streak_count"] == 3
    assert stats["current_win_streak"] == 3
    assert stats["max_win_streak"] == 3


def test_build_stats_current_win_streak_zero_on_loss_streak():
    stats = web_server._build_battle_history_stats([
        _b("classic", "lose"),
        _b("classic", "lose"),
    ])
    assert stats["current_streak_result"] == "lose"
    assert stats["current_streak_count"] == 2
    assert stats["current_win_streak"] == 0
    assert stats["max_win_streak"] == 0


def test_build_stats_training_and_friendly_excluded_from_streaks():
    stats = web_server._build_battle_history_stats([
        _b("classic", "win"),
        _b("classic", "win"),
        _b("friendly", "win"),
        _b("training", "win"),
    ])
    assert stats["current_streak_count"] == 2
    assert stats["current_win_streak"] == 2
    assert stats["max_win_streak"] == 2