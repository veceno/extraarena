"""Disk-backed ArenaMatchManager recovery across process restarts."""
from __future__ import annotations

from rlhf_env.tests._v5_helpers import make_manager


def _append_result(match, *, status="P1_WIN"):
    match.manifest.append_battle_result(
        battle_id=match.battle_id,
        battle_log_path=f"battles/{match.battle_id}.json",
        winner_user_id=1000 if status == "P1_WIN" else 2000,
        loser_user_id=2000 if status == "P1_WIN" else 1000,
        status=status,
        turns=7,
        duration_seconds=12.5,
    )


def test_restart_restores_history_and_resumes_next_unfinished_battle(tmp_path):
    manager = make_manager(tmp_path)
    first = manager.create_series({
        "p2_model": "random",
        "p1_actor_type": "human",
        "battles_planned": 3,
        "starting_player": "p1",
        "seed": 77123,
    })
    _append_result(first)

    restarted = make_manager(tmp_path)
    groups = restarted.list_groups()
    restored = next(group for group in groups if group["group_id"] == first.group_id)
    assert restored["status"] == "running"
    assert restored["battles_finished"] == 1
    assert restored["current_battle"] == 0
    assert restored["current_match_id"] is None

    resumed = restarted.next_match(first.group_id)
    assert resumed is not None
    assert resumed.group_id == first.group_id
    assert resumed.battle_index == 1
    assert restarted.get_group(first.group_id).current_match_id == resumed.engine.match_id


def test_restart_restores_completed_group_as_read_only_history(tmp_path):
    manager = make_manager(tmp_path)
    match = manager.create_series({
        "p2_model": "random",
        "p1_actor_type": "human",
        "battles_planned": 1,
        "starting_player": "p1",
        "seed": 77124,
    })
    _append_result(match)
    assert match.manifest.manifest["finished_at"] is not None

    restarted = make_manager(tmp_path)
    restored = next(group for group in restarted.list_groups() if group["group_id"] == match.group_id)
    assert restored["status"] == "completed"
    assert restored["battles_finished"] == 1
    assert restarted.next_match(match.group_id) is None


def test_restart_skips_corrupt_manifest_without_hiding_valid_groups(tmp_path):
    manager = make_manager(tmp_path)
    valid = manager.create_series({
        "p2_model": "random", "battles_planned": 1, "seed": 77125,
    })
    corrupt_dir = manager.sessions_dir / "broken-group"
    corrupt_dir.mkdir()
    (corrupt_dir / "manifest.json").write_text("{broken", encoding="utf-8")

    restarted = make_manager(tmp_path)
    ids = {group["group_id"] for group in restarted.list_groups()}
    assert valid.group_id in ids
    assert "broken-group" not in ids
