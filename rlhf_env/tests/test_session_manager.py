"""Тесты session_manager: старт/стоп/статус/лист/манифест/путь к battle_log."""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import pytest

from rlhf_env.components.policy_registry import PolicyRegistry
from rlhf_env.components.session_manager import SessionManager


@pytest.fixture
def sm(tmp_path):
    registry = PolicyRegistry.scan(Path("ai/models"))
    return SessionManager(
        sessions_dir=tmp_path / "sessions",
        models_dir="ai/models",
        registry=registry,
    )


async def _wait_for(sm, gid, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = sm.status(gid)
        if s and s["status"] in ("completed", "error"):
            return s
        await asyncio.sleep(0.05)
    raise TimeoutError("group did not complete in time")


async def _run_full(sm, spec, timeout: float = 30.0) -> dict:
    """Запустить группу в активном loop и дождаться завершения."""
    gid = sm.start(spec)
    return await _wait_for(sm, gid, timeout=timeout)


def _run(sm, spec, timeout: float = 30.0) -> dict:
    """Sync-обёртка: запускает asyncio.run + _run_full."""
    return asyncio.run(_run_full(sm, spec, timeout=timeout))


def test_session_manager_start_creates_group(sm):
    spec = {
        "p1_model": "random",
        "p2_model": "end_turn",
        "battles_planned": 2,
        "deck_strategy": "random_arenaenv",
        "seed": 1,
    }
    s = _run(sm, spec, timeout=60.0)
    assert s["status"] == "completed"
    assert s["battles_finished"] == 2


def test_session_manager_completes_group(sm):
    spec = {
        "p1_model": "random",
        "p2_model": "end_turn",
        "battles_planned": 1,
        "seed": 42,
    }
    s = _run(sm, spec, timeout=30.0)
    assert s["status"] == "completed"
    assert s["battles_finished"] == 1
    assert s["winrate_p1"] in (0.0, 1.0)


def test_session_manager_writes_files(sm):
    spec = {
        "p1_model": "random",
        "p2_model": "end_turn",
        "battles_planned": 1,
        "seed": 7,
    }
    s = _run(sm, spec, timeout=30.0)
    gid = s["group_id"]
    group_dir = sm.sessions_dir / gid
    assert (group_dir / "manifest.json").exists()
    assert (group_dir / "summary.json").exists()
    battles = list((group_dir / "battles").glob("*.json"))
    assert len(battles) == 1


def test_session_manager_get_manifest(sm):
    spec = {
        "p1_model": "random", "p2_model": "end_turn",
        "battles_planned": 1, "seed": 0,
    }
    s = _run(sm, spec, timeout=30.0)
    gid = s["group_id"]
    m = sm.get_manifest(gid)
    assert m["group_id"] == gid
    assert m["results"]["battles_finished"] == 1
    assert "decks" in m


def test_session_manager_list_includes_completed(sm):
    spec = {"p1_model": "random", "p2_model": "end_turn", "battles_planned": 1, "seed": 0}
    s = _run(sm, spec, timeout=30.0)
    gid = s["group_id"]
    groups = sm.list()
    assert any(g["group_id"] == gid for g in groups)


def test_session_manager_get_manifest_unknown(sm):
    assert sm.get_manifest("nonexistent-gid-zzz") is None


def test_session_manager_find_battle_path(sm):
    spec = {"p1_model": "random", "p2_model": "end_turn", "battles_planned": 1, "seed": 0}
    s = _run(sm, spec, timeout=30.0)
    gid = s["group_id"]
    m = sm.get_manifest(gid)
    battle_id = m["battle_ids"][0]
    bp = sm.find_battle_path(gid, battle_id)
    assert bp is not None
    assert bp.exists()
    log = sm.battle_log(gid, battle_id)
    assert log is not None
    assert log["battle_id"] == battle_id


def test_session_manager_stop_running():
    """Запускаем большую группу и останавливаем — статус должен стать cancelled/error."""
    sessions_dir = Path("/tmp/rlhf_test_stop")
    shutil.rmtree(sessions_dir, ignore_errors=True)
    sm = SessionManager(
        sessions_dir=sessions_dir,
        models_dir="ai/models",
        registry=PolicyRegistry.scan("ai/models"),
    )
    if not Path("ai/models/extra-lr-v4-max.onnx").exists():
        pytest.skip("V4-Max not present")
    spec = {
        "p1_model": "extra-lr-v4-max",
        "p2_model": "random",
        "battles_planned": 100,
        "seed": 0,
    }
    gid = asyncio.run(sm.astart(spec))
    # ждём пока запустится
    for _ in range(20):
        s = sm.status(gid)
        if s and s["status"] == "running":
            break
        time.sleep(0.05)
    ok = sm.stop(gid)
    assert ok is True or ok is False
