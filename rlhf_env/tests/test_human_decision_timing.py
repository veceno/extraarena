"""Human-only server-observed action timing for bot-humanisation data."""
from __future__ import annotations

import asyncio
import time

from rlhf_env.tests._v5_helpers import create_match, make_manager, read_jsonl, v5_dir_for


def test_human_timing_flows_to_all_action_surfaces(tmp_path):
    mgr = make_manager(tmp_path)
    match, _engine, runner = create_match(
        mgr, p1_actor_type="human", starting_player="p1", seed=901,
    )

    async def go():
        runner.mark_human_decision_start()
        runner._human_decision_started_monotonic = time.monotonic() - 0.125
        return await runner.execute_human_action(
            {"type": "end_turn", "client_action_id": "human-timing-1"}
        )

    response = asyncio.run(go())
    assert response["result"]["success"] is True

    analytics_row = match.recorder._buffer[0]
    v5_row = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")[0]
    battle_step = runner.battle_log["actions"][0]
    for row in (analytics_row, v5_row, battle_step):
        assert 100 <= row["human_decision_time_ms"] <= 250


def test_non_human_actors_never_get_human_timing(tmp_path):
    mgr = make_manager(tmp_path)
    match, _engine, runner = create_match(
        mgr, p1_actor_type="llm", starting_player="p1", seed=902,
    )

    async def go():
        runner.mark_human_decision_start()
        # Even a forged/internal clock must not escape the actor-type gate.
        runner._human_decision_started_monotonic = time.monotonic() - 1.0
        return await runner.execute_human_action(
            {"type": "end_turn", "client_action_id": "llm-timing-1"}
        )

    response = asyncio.run(go())
    assert response["result"]["success"] is True
    analytics_row = match.recorder._buffer[0]
    v5_row = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")[0]
    battle_step = runner.battle_log["actions"][0]
    assert analytics_row["human_decision_time_ms"] is None
    assert v5_row["human_decision_time_ms"] is None
    assert battle_step["human_decision_time_ms"] is None
