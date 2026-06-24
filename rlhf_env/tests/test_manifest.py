"""Тесты manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlhf_env.components.manifest import ManifestWriter


def test_manifest_writer_initial_state(tmp_path):
    mw = ManifestWriter(
        group_id="g1",
        spec={"battles_planned": 3, "p1_model": "v4-max"},
        group_dir=tmp_path / "g1",
    )
    assert mw.manifest_path.exists()
    data = json.loads(mw.manifest_path.read_text(encoding="utf-8"))
    assert data["group_id"] == "g1"
    assert data["manifest_version"] == "1.0"
    assert data["results"]["battles_finished"] == 0
    assert data["env"]["rlhf_env_version"] == "0.1.0"


def test_manifest_writer_appends_results(tmp_path):
    mw = ManifestWriter(
        group_id="g1",
        spec={"battles_planned": 2},
        group_dir=tmp_path / "g1",
    )
    mw.append_battle_result(
        battle_id="b1",
        battle_log_path="/p",
        winner_user_id=1000,
        loser_user_id=2000,
        status="P1_WIN",
        turns=12,
        duration_seconds=1.5,
    )
    assert mw.manifest["results"]["battles_finished"] == 1
    assert mw.manifest["results"]["p1_wins"] == 1
    assert mw.manifest["results"]["winrate_p1"] == 1.0
    assert mw.manifest["battle_ids"] == ["b1"]


def test_manifest_writer_winrate_calculation(tmp_path):
    mw = ManifestWriter(
        group_id="g1",
        spec={"battles_planned": 4},
        group_dir=tmp_path / "g1",
    )
    mw.append_battle_result(
        battle_id="b1", battle_log_path="/p",
        winner_user_id=1000, loser_user_id=2000, status="P1_WIN", turns=5, duration_seconds=1.0,
    )
    mw.append_battle_result(
        battle_id="b2", battle_log_path="/p",
        winner_user_id=2000, loser_user_id=1000, status="P2_WIN", turns=6, duration_seconds=1.2,
    )
    mw.append_battle_result(
        battle_id="b3", battle_log_path="/p",
        winner_user_id=1000, loser_user_id=2000, status="P1_WIN", turns=7, duration_seconds=1.5,
    )
    mw.append_battle_result(
        battle_id="b4", battle_log_path="/p",
        winner_user_id=None, loser_user_id=None, status="DRAW", turns=8, duration_seconds=1.7,
    )
    assert mw.manifest["results"]["battles_finished"] == 4
    assert mw.manifest["results"]["p1_wins"] == 2
    assert mw.manifest["results"]["p2_wins"] == 1
    assert mw.manifest["results"]["draws"] == 1
    assert mw.manifest["results"]["winrate_p1"] == 0.5
    assert mw.manifest["results"]["winrate_p2"] == 0.25
    assert mw.manifest["results"]["avg_turns"] == 6.5


def test_manifest_writer_finalize(tmp_path):
    mw = ManifestWriter(
        group_id="g1", spec={"battles_planned": 1}, group_dir=tmp_path / "g1",
    )
    mw.append_battle_result(
        battle_id="b1", battle_log_path="/p",
        winner_user_id=1000, loser_user_id=2000, status="P1_WIN", turns=5, duration_seconds=1.0,
    )
    finalized = mw.finalize()
    assert finalized["finished_at"] is not None
    assert finalized["finished_at"].endswith("Z")
    # summary.json создан
    summary_path = tmp_path / "g1" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["group_id"] == "g1"
    assert summary["battles_finished"] == 1


def test_manifest_writer_resumes_existing(tmp_path):
    """Если manifest уже есть на диске — ManifestWriter должен подхватить его."""
    group_dir = tmp_path / "g1"
    group_dir.mkdir()
    # Сначала создаём и пишем 1 батл
    mw1 = ManifestWriter(group_id="g1", spec={"battles_planned": 3}, group_dir=group_dir)
    mw1.append_battle_result(
        battle_id="b1", battle_log_path="/p",
        winner_user_id=1000, loser_user_id=2000, status="P1_WIN", turns=5, duration_seconds=1.0,
    )
    mw1._flush()
    # Теперь «перезапускаем» — должен загрузить существующее состояние
    mw2 = ManifestWriter(group_id="g1", spec={"battles_planned": 3}, group_dir=group_dir)
    assert mw2.manifest["results"]["battles_finished"] == 1
    assert mw2.manifest["results"]["p1_wins"] == 1
    mw2.append_battle_result(
        battle_id="b2", battle_log_path="/p",
        winner_user_id=2000, loser_user_id=1000, status="P2_WIN", turns=6, duration_seconds=1.5,
    )
    assert mw2.manifest["results"]["battles_finished"] == 2
    assert mw2.manifest["results"]["p1_wins"] == 1
    assert mw2.manifest["results"]["p2_wins"] == 1
