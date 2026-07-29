"""Hook-level regression tests for Daily Quests.

Battle quest operations are built before the main battle transaction and committed
inside it. The battle_summary match_id gate therefore provides both idempotency and
all-or-nothing durability for rewards and quest progress.
"""

from web import server as web_server


_WINNER_OPS = [
    ("win_battle_1", 1, False),
    ("win_battle_5", 1, False),
    ("win_streak_5", 1, False),
]
_LOSER_RESET_OPS = [("win_streak_5", 0, True)]


def _ops(**kwargs):
    defaults = {
        "eligible_mode": True,
        "winner_id_int": 1,
        "p1_id_int": 1,
        "p2_id_int": 2,
        "p1_is_bot": False,
        "p2_is_bot": False,
    }
    defaults.update(kwargs)
    return web_server._daily_quest_battle_end_ops(**defaults)


def test_human_vs_human_increments_winner_and_resets_loser():
    assert _ops() == {1: _WINNER_OPS, 2: _LOSER_RESET_OPS}


def test_draw_skips_all_quest_progress():
    assert _ops(winner_id_int=None) == {}


def test_invalid_winner_id_skips_all_quest_progress():
    assert _ops(winner_id_int=999) == {}


def test_bot_winner_does_not_count_but_human_loser_resets():
    assert _ops(winner_id_int=2, p2_is_bot=True) == {1: _LOSER_RESET_OPS}


def test_human_winner_against_bot_has_no_bot_loser_reset():
    assert _ops(p2_is_bot=True) == {1: _WINNER_OPS}


def test_training_or_friendly_mode_skips_quest_progress():
    assert _ops(eligible_mode=False) == {}


def test_winner_operations_are_one_transaction_payload():
    result = _ops(p2_is_bot=True)
    assert result == {1: _WINNER_OPS}
    assert len(result[1]) == 3


def test_case_open_progress_is_bound_to_durable_case_transactions():
    server = open("web/server.py", encoding="utf-8").read()
    case_system = open("infrastructure/case_system.py", encoding="utf-8").read()
    # Key flow: key decrement and open_case_1 share Database transaction.
    assert server.count("consume_key_for_case_opening(user_id)") == 2
    # user_case flow: case deletion/rewards and open_case_1 share one transaction.
    assert "_apply_daily_quest_ops_on_conn(" in case_system
    assert '[("open_case_1", 1, False)]' in case_system
