"""Тесты battle_runner: 1 бой end-to-end."""
from __future__ import annotations

import asyncio
import json
import random
import tempfile
from pathlib import Path

import pytest

from rlhf_env.components.battle_runner import BattleRunner
from rlhf_env.components.deck_builder import load_catalog, build_random_arena_deck
from rlhf_env.components.policy_factory import build_policy
from rlhf_env.components.session_manager import _build_game_state
from core.engine import ArenaEnvironment


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def _make_battle(tmp_path, catalog, p1="random", p2="end_turn", seed=0, max_turns=20):
    rng = random.Random(seed)
    p1_ids = build_random_arena_deck(catalog, rng=rng)
    p2_ids = build_random_arena_deck(catalog, rng=rng)
    gs = _build_game_state(p1_ids, p2_ids, catalog, starting_player="random", rng=rng)
    engine = ArenaEnvironment(gs)
    p1_pol = build_policy({"name": p1})
    p2_pol = build_policy({"name": p2})
    bp = tmp_path / "battle.json"
    runner = BattleRunner(
        group_id="t", battle_id="b1",
        policy_a=p1_pol, policy_b=p2_pol,
        engine=engine, battle_log_path=bp,
        max_turns=max_turns,
    )
    return runner, bp


def test_run_battle_writes_log(catalog, tmp_path):
    runner, bp = _make_battle(tmp_path, catalog, p1="random", p2="end_turn", seed=0, max_turns=10)
    log = asyncio.run(runner.arun())
    assert bp.exists()
    assert log["log_version"] == "1.0"
    assert log["battle_id"] == "b1"
    assert len(log["actions"]) >= 1
    # статус — либо P1/P2_WIN, либо ONGOING (если не успели за max_turns)
    assert log["result"]["status"] in {"P1_WIN", "P2_WIN", "DRAW", "ONGOING"}


def test_run_battle_log_is_valid_json(catalog, tmp_path):
    runner, bp = _make_battle(tmp_path, catalog, seed=1, max_turns=8)
    asyncio.run(runner.arun())
    data = json.loads(bp.read_text(encoding="utf-8"))
    assert data["battle_id"] == "b1"
    assert "duration_seconds" in data
    assert isinstance(data["actions"], list)


def test_run_battle_records_state_summaries(catalog, tmp_path):
    runner, bp = _make_battle(tmp_path, catalog, seed=2, max_turns=12)
    log = asyncio.run(runner.arun())
    assert log["final_state_summary"]["turn_number"] >= 1
    for action in log["actions"]:
        assert "state_before_summary" in action
        assert "state_after_summary" in action
        assert "timestamp_ms" in action


def test_run_battle_winner_in_user_ids(catalog, tmp_path):
    # random vs end_turn: либо p1 победит, либо бой не успеет
    runner, bp = _make_battle(tmp_path, catalog, p1="random", p2="end_turn", seed=3, max_turns=15)
    log = asyncio.run(runner.arun())
    assert log["result"]["winner_user_id"] in {1000, 2000, None}


def test_run_battle_with_v4_max(catalog, tmp_path):
    """V4-Max vs end_turn: должен побеждать V4-Max (он не сдаётся)."""
    from pathlib import Path as P
    if not (P("ai/models/extra-lr-v4-max.onnx").exists()):
        pytest.skip("V4-Max not present")
    runner, bp = _make_battle(tmp_path, catalog, p1="extra-lr-v4-max", p2="end_turn", seed=4, max_turns=30)
    log = asyncio.run(runner.arun())
    # V4-Max должен победить
    assert log["result"]["winner_user_id"] == 1000
    assert log["result"]["status"] == "P1_WIN"