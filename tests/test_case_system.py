import pytest

from infrastructure import case_system


def test_simulate_case_tap_results_uses_server_rolls(monkeypatch):
    rolls = iter([1, 2, 2, 3])

    def fake_roll(current_tier, tap_number, extra_pass="inactive"):
        assert extra_pass == "ultra"
        return next(rolls)

    monkeypatch.setattr(case_system, "roll_tier_upgrade", fake_roll)

    assert case_system.simulate_case_tap_results(1, "ultra") == [1, 2, 2, 3]


@pytest.mark.asyncio
async def test_ultra_case_reroll_selects_best_candidate(monkeypatch):
    candidates = [
        {"coins": 100, "cards": [], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
        {"coins": 100, "cards": [{"card_id": 7, "rarity": "legendary"}], "particles": [], "gems": 0, "limited_shards": 0, "jackpot": False},
    ]

    async def fake_generate_single(*args, **kwargs):
        return candidates.pop(0).copy()

    monkeypatch.setattr(case_system, "_generate_single_case_rewards", fake_generate_single)

    rewards = await case_system.generate_case_rewards(
        db=object(),
        tier=3,
        user_id=1,
        user_card_ids=set(),
        extra_pass="ultra",
    )

    assert rewards["cards"][0]["card_id"] == 7
    assert rewards["extra_pass_bonus"]["reroll_attempts"] == 1
    assert rewards["extra_pass_bonus"]["total_attempts"] == 2
    assert rewards["extra_pass_bonus"]["selected_attempt"] == 2
