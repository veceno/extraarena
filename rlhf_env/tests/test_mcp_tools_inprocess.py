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
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import pytest

from rlhf_env.components.policy_registry import PolicyRegistry
from rlhf_env.mcp_server import HeadlessHub, MCPServer, _amain
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


def test_dispatch_uses_standard_mcp_text_content(server):
    """Claude Code rejects the legacy non-standard ``type: json`` block."""
    srv, _hub, _tmp = server
    response = _run(srv.dispatch("tools/call", {"name": "list_models", "arguments": {}}))
    item = response["content"][0]
    assert item["type"] == "text"
    assert "data" not in item
    assert json.loads(item["text"]) == response["structuredContent"]
    assert "models" in response["structuredContent"]


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


def test_cross_contour_dataset_tools_are_unique_and_private(server):
    srv, hub, tmp = server
    names = [tool["name"] for tool in srv.tools]
    expected = {
        "get_training_data_status",
        "list_training_exports",
        "inspect_training_export",
        "validate_training_export",
        "export_v5_training_dataset",
        "materialize_v5_training_dataset",
        "export_nemesis_training_dataset",
        "split_nemesis_training_dataset",
        "export_returnclock_training_dataset",
        "split_returnclock_training_dataset",
    }

    assert len(names) == len(set(names))
    assert expected <= set(names)

    schemas = {
        tool["name"]: tool["inputSchema"]
        for tool in srv.tools
        if tool["name"] in expected
    }
    assert all(
        schema.get("additionalProperties") is False
        for schema in schemas.values()
    )
    returnclock_properties = schemas[
        "export_returnclock_training_dataset"
    ]["properties"]
    split_properties = schemas[
        "split_returnclock_training_dataset"
    ]["properties"]
    nemesis_split_properties = schemas[
        "split_nemesis_training_dataset"
    ]["properties"]
    v5_properties = schemas["export_v5_training_dataset"]["properties"]
    assert "salt" not in returnclock_properties
    assert "salt_env" not in returnclock_properties
    assert "user_id" not in returnclock_properties
    assert "salt" not in split_properties
    assert "user_id" not in split_properties
    assert "include_players" not in nemesis_split_properties
    assert "player_group_alias" not in nemesis_split_properties
    assert "include_players" not in v5_properties
    nemesis_schema = schemas["export_nemesis_training_dataset"]
    assert set(nemesis_schema["required"]) == {"output"}
    assert {"input_path", "group_id"} <= set(
        nemesis_schema["properties"]
    )


def test_mcp_tool_call_uses_standard_text_and_structured_content(server):
    srv, hub, tmp = server

    response = _run(srv.dispatch("tools/call", {
        "name": "get_training_data_status",
        "arguments": {},
    }))

    assert response["isError"] is False
    assert set(response) == {"content", "structuredContent", "isError"}
    assert response["content"][0]["type"] == "text"
    assert (
        json.loads(response["content"][0]["text"])
        == response["structuredContent"]
    )
    assert response["structuredContent"]["production_data_enabled"] is False


def test_mcp_dataset_error_uses_same_wire_contract_without_secret(server):
    srv, hub, tmp = server
    secret = "must-never-appear-in-tool-errors"

    response = _run(srv.dispatch("tools/call", {
        "name": "export_v5_training_dataset",
        "arguments": {
            "output": "v5/disabled.jsonl",
            "privacy_salt": secret,
        },
    }))

    assert response["isError"] is True
    assert response["content"][0]["type"] == "text"
    assert (
        json.loads(response["content"][0]["text"])
        == response["structuredContent"]
    )
    assert (
        response["structuredContent"]["error"]
        == "unknown tool arguments: privacy_salt"
    )
    assert secret not in json.dumps(response, ensure_ascii=False)


@pytest.mark.parametrize(
    ("exc", "error_code", "public_error"),
    [
        (
            ValueError(
                "database_dsn=postgresql://alice:"
                "value-error-secret@db/prod"
            ),
            "invalid_request",
            None,
        ),
        (
            RuntimeError("runtime-error-secret-must-not-leak"),
            "internal_tool_error",
            "internal tool error",
        ),
    ],
)
def test_mcp_tool_errors_are_redacted_and_stable(
    server,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exc: Exception,
    error_code: str,
    public_error: str | None,
) -> None:
    srv, _, _ = server

    async def fail_tool(name: str, arguments: dict) -> dict:
        del name, arguments
        raise exc

    monkeypatch.setattr(srv, "_tool", fail_tool)
    response = _run(
        srv.dispatch(
            "tools/call",
            {
                "name": "get_training_data_status",
                "arguments": {},
            },
        )
    )
    serialized = json.dumps(response, ensure_ascii=False)

    assert response["isError"] is True
    assert response["structuredContent"]["error_code"] == error_code
    if public_error is not None:
        assert response["structuredContent"]["error"] == public_error
    assert (
        json.loads(response["content"][0]["text"])
        == response["structuredContent"]
    )
    assert "value-error-secret" not in serialized
    assert "runtime-error-secret" not in serialized
    assert "value-error-secret" not in caplog.text
    assert "runtime-error-secret" not in caplog.text


def test_mcp_dataset_inspection_rejects_path_traversal(server):
    srv, hub, tmp = server

    response = _run(srv.dispatch("tools/call", {
        "name": "inspect_training_export",
        "arguments": {"path": "../../outside.jsonl"},
    }))

    assert response["isError"] is True
    assert "inside datasets_dir" in response["structuredContent"]["error"]
    assert (
        json.loads(response["content"][0]["text"])
        == response["structuredContent"]
    )


@pytest.mark.parametrize(
    ("tool_name", "arguments", "error_fragment"),
    [
        (
            "export_v5_training_dataset",
            {"output": "v5/wrong-type.jsonl", "days": "7"},
            "days must be integer",
        ),
        (
            "export_v5_training_dataset",
            {"output": "v5/wrong-bool.jsonl", "overwrite": "false"},
            "overwrite must be boolean",
        ),
        (
            "export_returnclock_training_dataset",
            {
                "output": "returnclock/below-minimum.jsonl",
                "start": "2026-07-01T00:00:00Z",
                "end": "2026-07-02T00:00:00Z",
                "min_analytics_version": 1,
            },
            "min_analytics_version is below minimum",
        ),
    ],
)
def test_mcp_dataset_tool_schemas_reject_wrong_types_and_minimums(
    server,
    tool_name: str,
    arguments: dict,
    error_fragment: str,
) -> None:
    srv, _, _ = server

    response = _run(
        srv.dispatch(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
    )

    assert response["isError"] is True
    assert error_fragment in response["structuredContent"]["error"]


def test_mcp_stdio_notifications_do_not_receive_jsonrpc_responses(
    server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    srv, _, _ = server
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        },
    ]
    stdin = io.StringIO(
        "".join(json.dumps(request) + "\n" for request in requests)
    )
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    _run(_amain(srv))

    responses = [
        json.loads(line)
        for line in stdout.getvalue().splitlines()
        if line.strip()
    ]
    assert [response["id"] for response in responses] == [1, 2]
    assert all("result" in response for response in responses)


def test_mcp_stdio_internal_error_does_not_expose_exception(
    server,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    srv, _, _ = server
    secret = "stdio-runtime-secret-must-not-leak"

    async def fail_dispatch(method: str, params: dict) -> dict:
        del method, params
        raise RuntimeError(secret)

    monkeypatch.setattr(srv, "dispatch", fail_dispatch)
    stdin = io.StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/list",
                "params": {},
            }
        )
        + "\n"
    )
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    _run(_amain(srv))

    response = json.loads(stdout.getvalue())
    assert response["error"] == {
        "code": -32603,
        "message": "internal server error",
    }
    assert secret not in stdout.getvalue()
    assert secret not in caplog.text


def test_mcp_dataset_toolbox_uses_configured_cards_catalog(
    tmp_path: Path,
) -> None:
    cards_path = tmp_path / "custom-cards.json"
    custom_catalog = Path(_CARDS_PATH).read_bytes() + b"\n"
    cards_path.write_bytes(custom_catalog)
    hub = HeadlessHub(
        sessions_dir=str(tmp_path / "sessions"),
        models_dir=_MODELS_DIR,
        cards_path=str(cards_path),
    )
    srv = MCPServer(hub, PolicyRegistry.scan(_MODELS_DIR))

    response = _run(
        srv.dispatch(
            "tools/call",
            {"name": "get_training_data_status", "arguments": {}},
        )
    )

    assert response["isError"] is False
    assert response["structuredContent"]["current_catalog_hash"] == (
        hashlib.sha256(custom_catalog).hexdigest()
    )


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
    assert vt["accepted_training_rows"] > 0
    assert vt["training_ready"] is True
    assert vt["training_ready_scope"] == "v5_policy_only"
    assert vt["v5_policy_training_ready"] is True
    assert vt["metronome_training_ready"] is False
    assert vt["timestamp_training_ready"] is False

    man = _run(srv._tool("get_battle_group_manifest", {"group_id": gid}))
    bid0 = man["battle_ids"][0]
    tr = _run(srv._tool("get_v5_trace", {"group_id": gid, "battle_id": bid0, "what": "meta"}))
    assert tr["data"]["p1_is_bot"] is True
    assert tr["data"]["agent_name"] == "Sinaf"

    ta = _run(srv._tool("get_v5_trace", {"group_id": gid, "battle_id": bid0, "what": "actions"}))
    assert ta["rows_count"] > 0

    completed = _run(
        srv._tool("next_battle", {"group_id": gid})
    )
    assert completed["status"] == "series_complete"

    nemesis = _run(
        srv._tool(
            "export_nemesis_training_dataset",
            {
                "group_id": gid,
                "output": "nemesis/from-headless.jsonl",
            },
        )
    )
    assert nemesis["ok"] is True
    assert nemesis["battle_count"] == 2
    assert nemesis["input"] == f"headless-group:{gid}"


def test_nemesis_export_requires_exactly_one_source(server) -> None:
    srv, _, _ = server

    for arguments in (
        {"output": "nemesis/missing-source.jsonl"},
        {
            "input_path": "v5/source.jsonl",
            "group_id": "group-1",
            "output": "nemesis/two-sources.jsonl",
        },
    ):
        response = _run(
            srv.dispatch(
                "tools/call",
                {
                    "name": "export_nemesis_training_dataset",
                    "arguments": arguments,
                },
            )
        )
        assert response["isError"] is True
        assert (
            "exactly one of input_path or group_id"
            in response["structuredContent"]["error"]
        )


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
