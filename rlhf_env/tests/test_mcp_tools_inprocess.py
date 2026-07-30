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
from rlhf_env.mcp_server import (
    HeadlessHub,
    MCPServer,
    _aggregate_quality_reports,
    _semi_synthetic_quality,
)
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


def test_semi_synthetic_quality_requires_50_games_and_wilson_floor(tmp_path):
    results = []
    for i in range(50):
        bid = f"b_{i}"
        apath = tmp_path / "battles" / bid / "v5"
        apath.mkdir(parents=True)
        (apath / "actions.jsonl").write_text(json.dumps({
            "decision_source": "llm", "action_type": "attack", "accepted": True,
            "error": None, "legal_actions": [{"type": "attack"}],
        }) + "\n", encoding="utf-8")
        results.append({
            "battle_id": bid, "battle_tag": "llm-vs-bot",
            "status": "P1_WIN" if i < 8 else "P2_WIN",
            "policy_warnings": [], "degraded": False,
        })
    quality = _semi_synthetic_quality(tmp_path, results)
    assert quality["p1_win_rate"] == pytest.approx(0.16)
    assert quality["p1_win_rate_wilson_lower_95"] > 0.03
    assert quality["semi_synthetic_usable"] is True
    results[0]["degraded"] = True
    assert _semi_synthetic_quality(tmp_path, results)["semi_synthetic_usable"] is False


def test_quality_reports_pool_across_short_series():
    one = {
        "llm_battles": 25, "llm_wins": 4, "llm_decisions": 100,
        "rejected_or_error_decisions": 0, "degraded_battles": 0,
        "mana_draw_decisions": 2, "end_turn_decisions": 20,
        "end_turn_with_attack_legal": 0, "end_turn_with_play_legal": 0,
    }
    pooled = _aggregate_quality_reports([one, one])
    assert pooled["llm_battles"] == 50
    assert pooled["llm_wins"] == 8
    assert pooled["p1_win_rate_wilson_lower_95"] > 0.03
    assert pooled["semi_synthetic_usable"] is True


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


def test_next_battle_rejects_unfinished_llm_match(server):
    srv, _hub, _tmp = server
    started = _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "llm",
        "battles_planned": 2, "starting_player": "p1",
    }}))
    assert started["is_ended"] is False
    blocked = _run(srv._tool("next_battle", {"group_id": started["group_id"]}))
    assert blocked == {
        "error": "current_battle_in_progress",
        "group_id": started["group_id"],
    }
    status = _run(srv._tool("get_match_status", {"match_id": started["match_id"]}))
    assert status["is_ended"] is False


def test_compact_player_state_and_submit_response_are_decision_complete(server):
    srv, _hub, _tmp = server
    started = _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "llm",
        "battles_planned": 1, "starting_player": "p1", "seed": 77,
    }}))
    mid = started["match_id"]
    full = _run(srv._tool("get_state", {"match_id": mid}))
    compact = _run(srv._tool("get_state", {
        "match_id": mid, "compact": True, "history_limit": 4,
    }))
    assert compact["compact"] is True
    assert [
        {k: v for k, v in action.items() if k != "legal_action_index"}
        for action in compact["legal_actions"]
    ] == full["legal_actions"]
    assert [a["legal_action_index"] for a in compact["legal_actions"]] == list(
        range(len(compact["legal_actions"]))
    )
    assert compact["player"]["hand"]
    assert compact["player"]["hero"]["mechanics"] == full["player"]["hero"]["mechanics"]
    assert len(json.dumps(compact)) < len(json.dumps(full))

    end_turn = next(a for a in compact["legal_actions"] if a["type"] == "end_turn")
    resp = _run(srv._tool("submit_action", {
        "match_id": mid, "action": end_turn, "compact_response": True,
    }))
    assert resp["state"]["compact"] is True
    assert "legal_actions" in resp["state"]


def test_submit_action_by_legal_index_avoids_uuid_transcription(server):
    srv, _hub, _tmp = server
    started = _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "llm",
        "battles_planned": 1, "starting_player": "p1", "seed": 79,
    }}))
    mid = started["match_id"]
    compact = _run(srv._tool("get_state", {
        "match_id": mid, "compact": True,
    }))
    chosen = next(a for a in compact["legal_actions"] if a["type"] == "end_turn")
    resp = _run(srv._tool("submit_action", {
        "match_id": mid,
        "legal_action_index": chosen["legal_action_index"],
        "compact_response": True,
    }))
    assert resp["result"]["success"] is True
    assert resp["resolved_legal_action_index"] == chosen["legal_action_index"]
    assert resp["resolved_legal_action"] == {"type": "end_turn"}


def test_invalid_legal_index_is_rejected_before_trace_recording(server):
    srv, hub, tmp = server
    started = _run(srv._tool("start_series", {"spec": {
        "p2_model": "random", "p1_actor_type": "llm",
        "battles_planned": 1, "starting_player": "p1", "seed": 81,
    }}))
    mid = started["match_id"]
    match = hub._match(mid)
    resp = _run(srv._tool("submit_action", {
        "match_id": mid, "legal_action_index": 999999,
    }))
    assert resp["error"] == "legal_action_index_out_of_range"
    assert resp["legal_action_count"] > 0
    v5 = v5_dir_for(match, tmp / "sessions")
    assert read_jsonl(v5 / "actions.jsonl") == []


def test_tools_call_uses_standard_text_content_block(server):
    srv, _hub, _tmp = server
    response = _run(srv.dispatch("tools/call", {
        "name": "list_models", "arguments": {},
    }))
    assert response["isError"] is False
    assert response["content"][0]["type"] == "text"
    parsed = json.loads(response["content"][0]["text"])
    assert parsed == response["structuredContent"]
    assert "models" in parsed


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
