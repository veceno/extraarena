"""Тесты session_manager: интерактивная модель human-vs-model.

В новой архитектуре SessionManager.start() НЕ запускает бой автоматически —
только создаёт группу с предвыделенными battle_ids и каркасом manifest.json.
Бой проигрывается через WS-сессию (server.py). Здесь проверяем:
  - start возвращает group_id и создаёт каталоги;
  - battle_ids генерируются в нужном количестве;
  - статус "loaded" → "running" → "completed" обновляется через
    прямые вызовы append_battle_result/active_battle_id (это делает server.py).
"""
from __future__ import annotations

import json
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


def test_session_manager_start_creates_group(sm):
    """start() должен вернуть group_id и создать каталоги."""
    spec = {
        "p1_model": "human",
        "p2_model": "end_turn",
        "battles_planned": 2,
        "deck_strategy": "random_arenaenv",
        "seed": 1,
        "interactive": True,
        "human_player": 1000,
    }
    gid = sm.start(spec)
    assert isinstance(gid, str) and len(gid) >= 8
    s = sm.status(gid)
    assert s is not None
    assert s["status"] == "loaded"  # ещё никто не открыл WS
    assert s["battles_planned"] == 2
    assert len(s["battle_ids"]) == 2
    assert s["human_player"] == 1000
    # Каталоги созданы
    assert (sm.sessions_dir / gid).is_dir()
    assert (sm.sessions_dir / gid / "battles").is_dir()


def test_session_manager_status_default_human_is_p1(sm):
    spec = {"p1_model": "human", "p2_model": "end_turn", "battles_planned": 1}
    gid = sm.start(spec)
    s = sm.status(gid)
    assert s["human_player"] == 1000  # default P1


def test_session_manager_battles_planned_default(sm):
    spec = {"p1_model": "human", "p2_model": "end_turn"}
    gid = sm.start(spec)
    s = sm.status(gid)
    assert s["battles_planned"] == 1
    assert len(s["battle_ids"]) == 1


def test_session_manager_manifest_written_on_start(sm):
    """manifest.json должен быть создан сразу при start() с правильным spec."""
    spec = {
        "p1_model": "human", "p2_model": "end_turn",
        "battles_planned": 3, "seed": 7,
    }
    gid = sm.start(spec)
    m = sm.get_manifest(gid)
    assert m is not None
    assert m["group_id"] == gid
    assert m["spec"]["p2_model"] == "end_turn"
    assert m["spec"]["battles_planned"] == 3
    # battle_ids предвыделены в state (и попадают в to_dict())
    s = sm.status(gid)
    assert len(s["battle_ids"]) == 3


def test_session_manager_get_manifest_unknown(sm):
    assert sm.get_manifest("nonexistent-gid-zzz") is None


def test_session_manager_list_includes_loaded(sm):
    """list() должен включать только что созданные (loaded) группы."""
    before = len(sm.list())
    sm.start({"p1_model": "human", "p2_model": "end_turn", "battles_planned": 1})
    after = len(sm.list())
    assert after == before + 1


def test_session_manager_find_battle_path_unknown(sm):
    gid = sm.start({"p1_model": "human", "p2_model": "end_turn", "battles_planned": 1})
    # Нет файлов боёв — путь не должен найтись
    s = sm.status(gid)
    assert sm.find_battle_path(gid, s["battle_ids"][0]) is None


def test_session_manager_active_battle_lifecycle(sm):
    """Имитация жизненного цикла боя (как делает server.py):
    loaded → running (кто-то открыл WS) → completed (бой записан)."""
    spec = {"p1_model": "human", "p2_model": "end_turn", "battles_planned": 2}
    gid = sm.start(spec)
    s0 = sm.status(gid)
    assert s0["status"] == "loaded"
    # WS открыт → active_battle_id
    state_obj = sm._groups[gid]
    first_bid = s0["battle_ids"][0]
    state_obj.active_battle_id = first_bid
    state_obj.current_battle = 1
    s1 = sm.status(gid)
    assert s1["status"] == "running"
    assert s1["active_battle_id"] == first_bid
    assert s1["current_battle"] == 1
    # Бой завершён → active_battle_id=None, но current_battle=1
    state_obj.active_battle_id = None
    s2 = sm.status(gid)
    assert s2["status"] == "loaded"  # следующий бой ещё не открыт
    # Серия завершена
    from datetime import datetime, timezone
    state_obj.finished_at = datetime.now(timezone.utc).isoformat()
    s3 = sm.status(gid)
    assert s3["status"] == "completed"


def test_session_manager_battle_ids_unique(sm):
    """Все battle_ids в серии должны быть уникальны."""
    spec = {"p1_model": "human", "p2_model": "end_turn", "battles_planned": 10}
    gid = sm.start(spec)
    s = sm.status(gid)
    ids = s["battle_ids"]
    assert len(ids) == len(set(ids))


def test_session_manager_find_battle_path_after_file_written(sm):
    """После записи battle_log.json find_battle_path должен его найти."""
    spec = {"p1_model": "human", "p2_model": "end_turn", "battles_planned": 1}
    gid = sm.start(spec)
    s = sm.status(gid)
    bid = s["battle_ids"][0]
    bp = sm.sessions_dir / gid / "battles" / f"{bid}.json"
    bp.write_text(json.dumps({"battle_id": bid, "result": {"winner_user_id": 1000}}), encoding="utf-8")
    found = sm.find_battle_path(gid, bid)
    assert found is not None
    assert found.exists()
    log = sm.battle_log(gid, bid)
    assert log["battle_id"] == bid