"""Регрессионные тесты на дефекты, найденные отдельным ultracode-аудитом
(Workflow A) после фиксации BUG 1-6. Фиксируют:

  F1 (HIGH): next_match на естественном завершении серии (next_index >=
    battles_planned) теперь освобождает кодовое имя агента (помечает finished),
    а не оставляет его busy навсегда. status() при этом busy=False + история.
  F4 (MEDIUM): run_auto с двумя сломанными политиками (v5-vs-v5 stub'ы →
    всегда NotImplementedError → fallback end_turn) раньше вис вечно (нет
    урона → is_ended никогда не True). Теперь жёсткий turn-cap → stalemate.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rlhf_env.tests._extensive_mcp_harness import make_server, call, start_rl_series

_REPO_ROOT = Path(__file__).resolve().parents[2]
_V5_ONNX = str(_REPO_ROOT / "ai" / "models" / "fake_v5.onnx")


def test_f1_natural_completion_releases_agent(tmp_path):
    """Серия доиграна до конца через next_battle (НЕ finish_series) → агент
    освобождён, status показывает busy=False + историю."""
    srv, hub, tmp = make_server(tmp_path)
    r = start_rl_series(srv, agent_name="Mentalist", battles_planned=2)
    gid = r["group_id"]
    # бои 0 уже доигран start_series'ом; доигрываем бои 1, затем series_complete.
    nb1 = call(srv, "next_battle", {"group_id": gid})
    assert nb1.get("is_ended") is True  # бой 1 доигран run_auto
    nb2 = call(srv, "next_battle", {"group_id": gid})
    assert nb2.get("status") == "series_complete"

    # F1: имя освобождено (finished), не утекло.
    assert hub.manager.agent_registry.is_busy("Mentalist") is False
    st = call(srv, "get_agent_status", {"agent_name": "Mentalist"})
    assert st["busy"] is False
    assert st["status"] == "completed"
    assert st["battles_finished"] == 2  # история доступна после release
    # имя можно пере-клеймить (не застряло в пуле).
    assert hub.manager.agent_registry.claim("Mentalist") is True


def test_f4_run_auto_turn_cap_on_broken_policies(tmp_path):
    """v5-vs-v5 (оба stub'а raise NotImplementedError → fallback end_turn) —
    run_auto не виснет вечно, доходит до turn-cap и финализирует ничьей."""
    srv, hub, tmp = make_server(tmp_path)
    spec = {
        "p2_model": "v5p2", "p2_model_kind": "v5", "p2_model_path": _V5_ONNX,
        "p1_model": "v5p1", "p1_model_kind": "v5", "p1_model_path": _V5_ONNX,
        "p1_actor_type": "rl", "battles_planned": 1, "seed": 7,
        "starting_player": "p1",
    }
    r = call(srv, "start_series", {"spec": spec})
    # start_series прогоняет run_auto (bounded) — должен завершиться, не висеть.
    assert r.get("is_ended") is True
    # обе политики сломаны → degradation зафиксирована.
    assert r.get("degraded") is True
    assert r.get("battle_tag") == "rl-vs-rl"
    # manifest зафиксировал бой (stalemate / no winner — главное не hang).
    man = call(srv, "get_battle_group_manifest", {"group_id": r["group_id"]})
    res = (man or {}).get("results", {}) or {}
    assert res.get("battles_finished", 0) == 1


def test_wfB1_build_writes_resolved_path_kind_into_spec(tmp_path):
    """Workflow-B Issue #1: AdapterRegistry.build должен писать резолвнутые
    path/kind обратно в spec — иначе фабрики, читающие spec.get('path'),
    получают None («action_onnx/v4 adapter requires 'path'»)."""
    from rlhf_env.components.policy_adapters import default_registry, PolicyAdapter

    reg = default_registry()
    captured = {}

    class _CapturingAdapter:
        kind = "capture_test"
        name = "capture_test"
        model_path = None
        weights_hash = None
        weights_version = None

        def __init__(self, spec, _registry=None):
            captured["path"] = spec.get("path")
            captured["kind"] = spec.get("kind")
            self.name = spec.get("name", "capture_test")

        def select_action(self, engine, player_id):
            return 0

    reg.register("capture_test", lambda spec, _reg: _CapturingAdapter(spec, _reg))
    try:
        out = reg.build({"name": "mymodel", "kind": "capture_test",
                         "path": str(_REPO_ROOT / "ai" / "models" / "fake_v5.onnx")})
        assert isinstance(out, _CapturingAdapter)
        # path/kind дошли до фабрики (было None до фикса).
        assert captured["path"] is not None
        assert captured["kind"] == "capture_test"
    finally:
        # не засоряем default_registry перманентно — удаляем kind.
        reg._factories.pop("capture_test", None)


def test_wfB2_get_match_status_winner_after_surrender(tmp_path):
    """Workflow-B Issue #2: после human-surrender get_match_status должен
    возвращать winner_id=bot (не null) — _finalize теперь ставит state.status."""
    srv, hub, tmp = make_server(tmp_path)
    spec = {
        "p2_model": "random", "p1_actor_type": "human",
        "battles_planned": 1, "seed": 11, "starting_player": "p1",
        "agent_name": "Humar",
    }
    r = call(srv, "start_series", {"spec": spec})
    mid = r["match_id"]
    bot_uid = r["player_ids"][1]
    # сдаёмся (human p1).
    sur = call(srv, "surrender", {"match_id": mid})
    assert (sur or {}).get("result", {}).get("winner_id") == bot_uid
    # polling-инструмент должен видеть того же победителя.
    ms = call(srv, "get_match_status", {"match_id": mid})
    assert ms.get("is_ended") is True
    assert ms.get("winner_id") == bot_uid  # было None до фикса
    assert ms.get("is_my_turn") is False  # ended-игра — ничей ход


def test_agent_leak_released_on_get_match_status(tmp_path):
    """Agent-leak фикс (L1): после естественного завершения 1-боевой серии
    (hero death) БЕЗ next_battle/finish_series, get_match_status должен
    освободить кодовое имя агента (reap_completed) И очистить live.current_match_id.
    До фикса имя оставалось busy навсегда в agents_index.json."""
    srv, hub, tmp = make_server(tmp_path)
    spec = {"p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
            "battles_planned": 1, "seed": 21, "starting_player": "p1",
            "agent_name": "Veceno"}
    r = call(srv, "start_series", {"spec": spec})
    assert r.get("is_ended") is True  # rl-vs-bot auto-play до game_over
    gid = r["group_id"]
    # До polling-вызова агент ещё busy (manifest finalized на диске, но registry
    # не self-heal'ился — is_busy/status не вызывались).
    assert hub.manager.agent_registry._busy.get("Veceno", {}).get("finished") is None
    # Не зовём next_battle/finish_series — симулируем клиента, ушедшего после боя.
    ms = call(srv, "get_match_status", {"match_id": r["match_id"]})
    assert ms.get("is_ended") is True
    # L1: reap_completed в get_match_status очистил live.current_match_id.
    assert hub.manager._groups[gid].current_match_id is None
    # Агент освобождён (L1 release_group + L2 self-heal).
    assert hub.manager.agent_registry.is_busy("Veceno") is False
    st = call(srv, "get_agent_status", {"agent_name": "Veceno"})
    assert st["busy"] is False
    assert st["status"] == "completed"
    # имя можно пере-клеймить (не утекло).
    assert hub.manager.agent_registry.claim("Veceno") is True


def test_agent_registry_self_heal_cross_process(tmp_path):
    """Agent-leak фикс (L2): cross-process recovery. Упавший процесс оставил
    busy-запись в agents_index.json; новый процесс (fresh AgentRegistry) не имеет
    _groups-записи → manager.reap_completed не поможет. Registry сам читает
    манифест группы и освобождает имя при первом is_busy/claim/claim_auto.

    Важно: в процессе A мы НЕ зовём is_busy/claim/status (каждый из них self-heal'ит
    in-process) — имитируем краш ДО любого polling-вызова. Проверяем raw _busy."""
    from rlhf_env.components.agent_registry import AgentRegistry

    srv, hub, tmp = make_server(tmp_path)
    sessions = hub.manager.sessions_dir
    spec = {"p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
            "battles_planned": 1, "seed": 31, "starting_player": "p1",
            "agent_name": "dranik"}
    r = call(srv, "start_series", {"spec": spec})
    assert r.get("is_ended") is True
    # Процесс A «умирает» здесь. agents_index (persisted) имеет busy dranik
    # (finished=None); манифест группы на диске — auto-finalized (finished_at).
    # Не вызываем is_busy/claim/status в процессе A → registry не self-heal'ился.
    reg_a = hub.manager.agent_registry
    assert reg_a._busy.get("dranik", {}).get("finished") is None

    # NEW process: fresh AgentRegistry поверх того же sessions_dir (persisted).
    reg2 = AgentRegistry(sessions / "agents_index.json",
                         sessions_dir=sessions, cards_path=str(_REPO_ROOT / "ai" / "cards.json"))
    # is_busy должен self-heal: читает манифест, видит finished_at → finished=True.
    assert reg2.is_busy("dranik") is False
    # теперь имя можно пере-клеймить в новом процессе.
    assert reg2.claim("dranik") is True
    reg2.release("dranik")


def test_reap_does_not_release_mid_series(tmp_path):
    """reap_completed НЕ должен освобождать агента в середине multi-battle серии:
    после боя 0 (battles_finished=1 < planned=3) агент остаётся busy для боёв 1,2.
    L2 self-heal тоже не срабатывает — manifest не finalized (finished<planned)."""
    srv, hub, tmp = make_server(tmp_path)
    spec = {"p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
            "battles_planned": 3, "seed": 41, "starting_player": "p1",
            "agent_name": "Sinaf"}
    r = call(srv, "start_series", {"spec": spec})
    # start_series доиграл бой 0 (rl auto-play). Серия не завершена (1/3).
    ms = call(srv, "get_match_status", {"match_id": r["match_id"]})
    assert ms.get("is_ended") is True  # бой 0 завершён
    # но серия 1/3 → reap не сработал (battles_finished<planned), live не тронут.
    assert hub.manager._groups[r["group_id"]].current_match_id == r["match_id"]
    # L2 self-heal: manifest не finalized (finished<planned) → агент остаётся busy.
    assert hub.manager.agent_registry.is_busy("Sinaf") is True