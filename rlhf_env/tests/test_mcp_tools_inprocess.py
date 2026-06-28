"""In-process тесты MCP-оркестратора (Block E) — без stdio/HTTP.

Драйвит MCPServer._tool напрямую через HeadlessHub (тот же ArenaMatchManager +
MatchRunner, что и prod-MCP). Покрывает:
  - start_series с p1_actor_type='rl' (model-vs-model) → авто-доигрывание,
    agent_name, battle_tag rl-vs-bot, resolved p1/p2 model info.
  - list_active_series (группировка по агенту + by_model).
  - get_agent_status (aggregate из manifest).
  - get_match_status / get_action_history (lightweight player-tools).
  - finish_series → агент освобождается.
  - register_custom_model + list_models.
  - V5 orchestrator: get_v5_dataset_summary / list_v5_groups / get_v5_trace /
    validate_v5_traces; v5/meta.json содержит agent_name + p1_is_bot.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from rlhf_env.components.policy_registry import PolicyRegistry
from rlhf_env.mcp_server import HeadlessHub, MCPServer
from rlhf_env.tests._v5_helpers import v5_dir_for, read_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = str(_REPO_ROOT / "ai" / "models")
_CARDS_PATH = str(_REPO_ROOT / "ai" / "cards.json")


@pytest.fixture
def server(tmp_path):
    hub = HeadlessHub(
        sessions_dir=str(tmp_path / "sessions"),
        models_dir=_MODELS_DIR,
        cards_path=_CARDS_PATH,
    )
    reg = PolicyRegistry.scan(_MODELS_DIR)
    return MCPServer(hub, reg), hub, tmp_path


def _run(coro):
    return asyncio.run(coro)


def test_start_series_rl_autoplays_with_agent(server):
    srv, hub, tmp = server
    r = _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
        "battles_planned": 1, "agent_name": "Veceno", "starting_player": "p1",
    }}))
    # rl-vs-bot auto-играет до game_over без submit_action.
    assert r["agent_name"] == "Veceno"
    assert r["p1_actor_type"] == "rl"
    assert r["battle_tag"] == "rl-vs-bot"
    assert r["is_ended"] is True
    assert isinstance(r["winner_id"], int)
    assert r["p1_model"]["kind"] == "random"
    assert r["p2_model"]["kind"] == "random"


def test_list_active_series_and_agent_status(server):
    srv, hub, tmp = server
    _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
        "battles_planned": 2, "agent_name": "Mentalist", "starting_player": "p1",
    }}))
    la = _run(srv._tool("list_active_series", {}))
    assert la["count"] >= 1
    ag = next(a for a in la["agents"] if a["agent_name"] == "Mentalist")
    assert ag["battles"].startswith("1/") or ag["battles"].startswith("0/")
    assert any(bm["model"] == "random" for bm in la["by_model"])

    gs = _run(srv._tool("get_agent_status", {"agent_name": "Mentalist"}))
    assert gs["agent_name"] == "Mentalist"
    assert gs["busy"] is True
    assert gs["p1_actor_type"] == "rl"


def test_match_status_and_action_history(server):
    srv, hub, tmp = server
    r = _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
        "battles_planned": 1, "starting_player": "p1",
    }}))
    mid = r["match_id"]
    ms = _run(srv._tool("get_match_status", {"match_id": mid}))
    assert ms["is_ended"] is True
    assert ms["battle_tag"] == "rl-vs-bot"
    assert ms["agent_name"] == r["agent_name"]

    ah = _run(srv._tool("get_action_history", {"match_id": mid}))
    assert ah["count"] > 0
    assert isinstance(ah["actions"], list)


def test_finish_series_releases_agent(server):
    srv, hub, tmp = server
    r = _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
        "battles_planned": 1, "agent_name": "Pvwell", "starting_player": "p1",
    }}))
    gid = r["group_id"]
    # бой уже доигран (rl-vs-bot auto); finish_series закрывает manifest + release.
    fs = _run(srv._tool("finish_series", {"group_id": gid}))
    assert fs.get("finished_at") is not None
    gs = _run(srv._tool("get_agent_status", {"agent_name": "Pvwell"}))
    assert gs["busy"] is False


def test_register_custom_model_and_list(server):
    srv, hub, tmp = server
    rc = _run(srv._tool("register_custom_model", {
        "name": "my-v5-snap", "path": "ai/models/fake_v5.onnx", "kind": "v5",
    }))
    assert rc["registered"] is True
    assert rc["kind"] == "v5"
    lm = _run(srv._tool("list_models", {}))
    names = [m["name"] for m in lm["models"]]
    assert "my-v5-snap" in names


def test_register_custom_model_rejects_path_traversal(server):
    srv, hub, tmp = server
    with pytest.raises(Exception):
        _run(srv._tool("register_custom_model", {
            "name": "evil", "path": "../../../etc/passwd", "kind": "v5",
        }))


def test_list_preset_decks_returns_note(server):
    srv, hub, tmp = server
    r = _run(srv._tool("list_preset_decks", {}))
    assert r["presets"] == []
    assert "note" in r


def test_v5_orchestrator_tools(server):
    srv, hub, tmp = server
    r = _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
        "battles_planned": 2, "agent_name": "Sinaf", "starting_player": "p1",
    }}))
    gid = r["group_id"]
    # доиграть 2-й бой
    nb = _run(srv._tool("next_battle", {"group_id": gid}))
    assert nb.get("is_ended") is True

    sm = _run(srv._tool("get_v5_dataset_summary", {"group_id": gid}))
    assert sm["battles_finished"] == 2
    assert sm["v5_trace_ok_count"] == 2
    assert sm["battle_tag_distribution"].get("rl-vs-bot") == 2
    assert sm["actions_total"] > 0

    lg = _run(srv._tool("list_v5_groups", {"battle_tag": "rl-vs-bot"}))
    assert any(g["group_id"] == gid for g in lg["groups"])

    vt = _run(srv._tool("validate_v5_traces", {"group_id": gid}))
    assert vt["checked"] == 2 and vt["ok"] == 2 and vt["broken"] == []

    man = _run(srv._tool("get_battle_group_manifest", {"group_id": gid}))
    bid0 = man["battle_ids"][0]
    tr = _run(srv._tool("get_v5_trace", {"group_id": gid, "battle_id": bid0, "what": "meta"}))
    assert tr["data"]["p1_is_bot"] is True
    assert tr["data"]["agent_name"] == "Sinaf"

    ta = _run(srv._tool("get_v5_trace", {"group_id": gid, "battle_id": bid0, "what": "actions"}))
    assert ta["rows_count"] > 0


def test_submit_action_rejected_for_rl_p1(server):
    """F4(audit): submit_action на p1-RL (model-vs-model) бое должен Reject'нуть —
    иначе внешний клиент вливает p1-action мимо match.p1_policy auto-play и он
    мис-тегируется decision_source='rl' в v5/actions.jsonl (портит V5 training-data).
    Симметрично surrender. Guard возвращается до execute_human_action → 0 rows."""
    srv, hub, tmp = server
    # create_series только строит матч (НЕ auto-play как start_series) → бой live.
    match = hub.manager.create_series({
        "p2_model": "random", "p1_actor_type": "rl", "p1_model": "random",
        "battles_planned": 1, "agent_name": "dranik", "starting_player": "p1",
    })
    mid = match.engine.match_id
    assert match.engine.p1_actor_type == "rl"
    resp = _run(srv._tool("submit_action", {
        "match_id": mid, "action": {"type": "end_turn"},
    }))
    assert resp.get("error") == "submit_action_unavailable_for_rl_p1"
    # guard сработал до execute_human_action → ни одной строки в v5/actions.jsonl.
    v5 = v5_dir_for(match, tmp / "sessions")
    assert read_jsonl(v5 / "actions.jsonl") == []