"""In-process + stdio tests for the ExtraOrchestra MCP server.

No visual input — pure JSON-RPC over stdio + direct dispatch. Validates that
agents can load / create-as-graphs / preview / export scenarios via MCP.

Run:  python3 -m pytest extra_orchestra/tests/test_mcp_server.py -x
      (needs orchestra HTTP server on 127.0.0.1:8095; auto-spawns if absent)
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import aiohttp
import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from extra_orchestra.mcp_server import (  # noqa: E402
    MCPServer, OrchestraClient, create_blank_scenario, build_graph, summarize_frames,
)

BASE = os.environ.get("ORCH_BASE_URL", "http://127.0.0.1:8095")
V2 = "extra_orchestra.scenario.v2"
V1 = "extra_orchestra.scenario.v1"

ALL_TOOLS = {
    "list_scenarios", "get_scenario", "migrate_v1_to_v2",
    "create_blank_scenario", "build_graph", "save_scenario",
    "delete_scenario", "validate_scenario", "preview_frames", "get_frames",
    "export_mp4", "export_gif", "get_record_status", "get_record_file",
    "list_cards", "list_cosmetics",
}


# ---------------------------------------------------------------------------
# fixtures: ensure HTTP server is up (spawn if needed)
# ---------------------------------------------------------------------------

def _port_up(base: str) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(base + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def http_server():
    if _port_up(BASE):
        yield BASE
        return
    proc = subprocess.Popen([sys.executable, "-m", "extra_orchestra.server",
                             "--host", "127.0.0.1", "--port", "8095"],
                            cwd=str(_REPO))
    try:
        for _ in range(40):
            if _port_up(BASE):
                break
            time.sleep(0.25)
        # явный assert — иначе каскад connection-ошибок маскирует упавший fixture
        assert _port_up(BASE), "orchestra HTTP server did not come up on " + BASE
        yield BASE
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_mcp_async(base: str):
    """Build a MCPServer with a fresh aiohttp session, INSIDE a loop."""
    session = aiohttp.ClientSession()
    client = OrchestraClient(base, session)
    server = MCPServer(client, base)
    return server, session


def _call_tool(base: str, name: str, args: dict):
    async def _go():
        server, session = _make_mcp_async(base)
        try:
            return await server._tool(name, args)
        finally:
            await session.close()
    return _run(_go())


def _dispatch(base: str, name: str, args: dict):
    async def _go():
        server, session = _make_mcp_async(base)
        try:
            return await server.dispatch("tools/call", {"name": name, "arguments": args})
        finally:
            await session.close()
    return _run(_go())


def _dispatch_method(base: str, method: str, params: dict):
    """Dispatch an arbitrary JSON-RPC method (not just tools/call)."""
    async def _go():
        server, session = _make_mcp_async(base)
        try:
            return await server.dispatch(method, params)
        finally:
            await session.close()
    return _run(_go())


def _tool_with_client(client, name: str, args: dict):
    """Run a tool call against an injected (stub) client — no HTTP, no aiohttp."""
    server = MCPServer(client=client, base_url=BASE)
    async def _go():
        return await server._tool(name, args)
    return _run(_go())


# ---------------------------------------------------------------------------
# pure helpers (no HTTP)
# ---------------------------------------------------------------------------

def test_create_blank_scenario_full_meta():
    sc = create_blank_scenario({
        "name": "test blank", "seed": 7, "turn_number": 3, "starting_side": "p2",
        "p1_name": "Alice", "p2_name": "Bob", "p1_hero_id": 5, "p2_hero_id": 9,
        "viewer_side": "p2", "match_id": "match-x",
    })
    assert sc["schema"] == V2
    assert sc["graph"]["start"] == "s0"
    assert len(sc["graph"]["nodes"]) == 1
    init = sc["graph"]["nodes"][0]
    assert init["kind"] == "scene" and init["scene"]["type"] == "init"
    assert sc["seed"] == 7
    assert sc["name"] == "test blank"
    assert sc["viewer_side"] == "p2"
    assert sc["match_id"] == "match-x"
    assert init["scene"]["turn_number"] == 3
    assert init["scene"]["starting_side"] == "p2"
    assert init["scene"]["display_ms"] == 2000  # matches editor.js blankV2
    assert init["scene"]["p1"]["nickname"] == "Alice"
    assert init["scene"]["p2"]["nickname"] == "Bob"
    assert init["scene"]["p1"]["hero"]["card_id"] == 5
    assert init["scene"]["p2"]["hero"]["card_id"] == 9


def test_build_graph_auto_chain_and_field_propagation():
    spec = {
        "name": "chain test", "seed": 11,
        "nodes": [
            {"id": "n1", "kind": "turn", "side": "p1", "intro_ms": 400},
            {"id": "n2", "kind": "action", "side": "p1", "type": "play_card",
             "hand_index": 0, "delay_ms": 700},
            {"id": "n3", "kind": "action", "side": "p1", "type": "end_turn"},
            {"id": "n4", "kind": "scene", "display_ms": 800},
        ],
    }
    sc = build_graph(spec)
    assert sc["schema"] == V2
    assert sc["match_id"] == "new-scenario"  # editor.js blankV2 default
    ids = [n["id"] for n in sc["graph"]["nodes"]]
    assert ids == ["s0", "n1", "n2", "n3", "n4"]
    pairs = [(e["from"], e["to"]) for e in sc["graph"]["edges"]]
    assert pairs == [("s0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4")]
    turn_node = sc["graph"]["nodes"][1]
    assert turn_node["turn"]["side"] == "p1"
    assert turn_node["turn"]["intro_ms"] == 400 and isinstance(turn_node["turn"]["intro_ms"], int)
    act = sc["graph"]["nodes"][2]
    assert act["kind"] == "action" and act["side"] == "p1"
    assert act["action"]["type"] == "play_card"
    assert act["action"]["delay_ms"] == 700 and isinstance(act["action"]["delay_ms"], int)
    assert act["action"]["hand_index"] == 0
    end = sc["graph"]["nodes"][3]
    assert end["action"]["type"] == "end_turn" and end["action"]["delay_ms"] == 500
    assert sc["graph"]["nodes"][4]["scene"]["type"] == "hold"
    assert sc["graph"]["nodes"][4]["scene"]["display_ms"] == 800


def test_build_graph_canonical_nested_action_node():
    """Major regression: canonical v2 node {kind:action, side, action:{...}}
    (as produced by editor.js / saved scenarios) must be accepted, not flattened
    into action.type=None."""
    spec = {
        "nodes": [
            {"id": "n1", "kind": "action", "side": "p1",
             "action": {"type": "attack", "delay_ms": 300, "attacker_index": 0,
                        "target_is_hero": True}},
        ],
    }
    sc = build_graph(spec)
    act = sc["graph"]["nodes"][1]
    assert act["action"]["type"] == "attack"
    assert act["action"]["delay_ms"] == 300
    assert act["action"]["attacker_index"] == 0
    assert act["action"]["target_is_hero"] is True
    assert act["side"] == "p1"


def test_build_graph_action_missing_side_is_none():
    """Nit: missing side → None (not silent 'p1'), so validate_graph_structure
    surfaces an authoring-time error rather than a runtime wrong-side failure."""
    sc = build_graph({"nodes": [{"id": "n1", "kind": "action", "type": "end_turn"}]})
    assert sc["graph"]["nodes"][1]["side"] is None


def test_build_graph_id_collision_raises():
    with pytest.raises(ValueError, match="duplicate node id"):
        build_graph({"nodes": [
            {"id": "n1", "kind": "turn", "side": "p1"},
            {"id": "n1", "kind": "action", "side": "p1", "type": "end_turn"},
        ]})


def test_build_graph_auto_id_collision_actually_raises():
    with pytest.raises(ValueError, match="duplicate node id"):
        build_graph({"nodes": [
            {"kind": "turn", "side": "p1"},          # auto → n1
            {"id": "n1", "kind": "action", "side": "p1", "type": "end_turn"},
        ]})


def test_build_graph_auto_id_skips_used():
    # auto-gen must skip ids already used by earlier nodes (no collision)
    sc = build_graph({"nodes": [
        {"id": "n1", "kind": "turn", "side": "p1"},          # explicit n1
        {"kind": "action", "side": "p1", "type": "end_turn"},  # auto → n2 (skips n1)
    ]})
    ids = [n["id"] for n in sc["graph"]["nodes"]]
    assert ids == ["s0", "n1", "n2"]


def test_build_graph_edge_missing_from_to_raises():
    with pytest.raises(ValueError, match="'from' and 'to' are required"):
        build_graph({"nodes": [{"id": "h", "kind": "scene", "display_ms": 500}],
                     "edges": [{"to": "h"}]})


def test_build_graph_base_without_init_raises():
    """build_graph is pure: base_scenario_name without pre-injected init_scene
    is a caller bug → ValueError (the _tool handler does the loading)."""
    with pytest.raises(ValueError, match="base_scenario_name"):
        build_graph({"base_scenario_name": "soldatik-demo", "nodes": []})


def test_build_graph_explicit_edges():
    spec = {
        "name": "explicit edges", "seed": 3,
        "nodes": [{"id": "h", "kind": "scene", "display_ms": 500}],
        "edges": [{"from": "s0", "to": "h"}],
    }
    sc = build_graph(spec)
    assert [(e["from"], e["to"]) for e in sc["graph"]["edges"]] == [("s0", "h")]


def test_build_graph_rejects_bad_kind():
    with pytest.raises(ValueError):
        build_graph({"nodes": [{"id": "x", "kind": "banana"}]})


def test_summarize_frames_p1_viewer_and_hidden_hand():
    run = {
        "run_id": "r1", "frame_count": 2, "total_ms": 2000, "viewer_uid": 1001,
        "side_uids": {"p1": 1001, "p2": 2002}, "error": None,
        "frames": [
            {"node_id": "s0", "action_kind": "init", "turn_id": "__no_turn__",
             "display_ms": 1000,
             "snapshot": {"turn": 1, "current_player_id": 1001,
                          "player": {"user_id": 1001, "name": "P", "mana": 6,
                                     "max_mana": 6, "hand": [{"card_id": 5}],
                                     "board": [], "hero": {"card_id": 1, "hp": 30, "max_hp": 30}},
                          "opponent": {"user_id": 2002, "name": "O", "mana": 6,
                                       "max_mana": 6,
                                       "hand": [{"hidden": True}, {"hidden": True}],
                                       "board": [], "hero": {"card_id": 3, "hp": 30, "max_hp": 30}}}},
            {"node_id": "n1", "action_kind": "play_card", "turn_id": "t1",
             "display_ms": 700,
             "snapshot": {"turn": 1, "current_player_id": 1001,
                          "player": {"user_id": 1001, "name": "P", "mana": 5,
                                     "max_mana": 6, "hand": [], "board": [],
                                     "hero": {"card_id": 1, "hp": 30, "max_hp": 30}},
                          "opponent": {"user_id": 2002, "name": "O", "mana": 6,
                                       "max_mana": 6, "hand": [], "board": [],
                                       "hero": {"card_id": 3, "hp": 30, "max_hp": 30}}}},
        ],
    }
    s = summarize_frames(run)
    assert s["frame_count"] == 2
    assert s["side_uids"] == {"p1": 1001, "p2": 2002}
    assert s["viewer_uid"] == 1001
    assert s["viewer_side"] == "p1"  # derived from side_uids, not the raw uid
    f0 = s["frames"][0]
    assert f0["p1"]["user_id"] == 1001 and f0["p2"]["user_id"] == 2002
    assert f0["p1"]["hand_card_ids"] == [5]            # own hand visible
    assert f0["p1"]["hand_hidden"] is False
    assert f0["p2"]["hand_card_ids"] == []             # opponent hand hidden → no card_ids
    assert f0["p2"]["hand_hidden"] is True
    assert f0["p2"]["hand_count"] == 2
    assert s["frames"][1]["action_kind"] == "play_card"


def test_summarize_frames_p2_viewer():
    """Else-branch: viewer is p2 → player.user_id == p2_uid → p1=opponent, p2=player."""
    run = {
        "run_id": "r2", "frame_count": 1, "total_ms": 1000, "viewer_uid": 2002,
        "side_uids": {"p1": 1001, "p2": 2002}, "error": None,
        "frames": [{"node_id": "s0", "action_kind": "init", "turn_id": "t",
                    "display_ms": 1000,
                    "snapshot": {"turn": 1, "current_player_id": 1001,
                                 "player": {"user_id": 2002, "name": "Viewer", "hand": [],
                                            "board": [], "hero": {"card_id": 9, "hp": 30}},
                                 "opponent": {"user_id": 1001, "name": "Foe", "hand": [],
                                              "board": [], "hero": {"card_id": 1, "hp": 30}}}}],
    }
    s = summarize_frames(run)
    assert s["viewer_side"] == "p2"
    # player is viewer (p2 uid), so else-branch: p1=opponent(1001), p2=player(2002)
    assert s["frames"][0]["p1"]["user_id"] == 1001
    assert s["frames"][0]["p2"]["user_id"] == 2002


# ---------------------------------------------------------------------------
# tools/list + initialize (no HTTP needed)
# ---------------------------------------------------------------------------

def test_tools_list_has_16_tools():
    server = MCPServer(client=None, base_url=BASE)
    names = {t["name"] for t in server.tools}
    assert names == ALL_TOOLS
    assert len(names) == 16
    for t in server.tools:
        assert "description" in t and "inputSchema" in t


def test_dispatch_initialize_no_client():
    fake = MCPServer.__new__(MCPServer)
    fake.tools = []
    r = _run(fake.dispatch("initialize", {}))
    assert r["protocolVersion"] == "2024-11-05"
    assert r["serverInfo"]["name"] == "extra-orchestra"
    assert "tools" in r["capabilities"]


def test_dispatch_unknown_method_is_jsonrpc_error():
    """Major: unknown method → top-level {error:{code:-32601}}, no 'content'."""
    r = _dispatch_method(BASE, "bogus/method", {})
    assert "error" in r and "content" not in r
    assert r["error"]["code"] == -32601
    assert "unknown method" in r["error"]["message"]


# ---------------------------------------------------------------------------
# live tool calls — load / create / preview / validate / cards / cosmetics
# ---------------------------------------------------------------------------

def test_tool_list_scenarios(http_server):
    r = _call_tool(http_server, "list_scenarios", {})
    assert "scenarios" in r
    assert any(s["id"] == "soldatik-demo" for s in r["scenarios"])


def test_tool_get_scenario_migrates_v1(http_server):
    r = _call_tool(http_server, "get_scenario", {"name": "soldatik-demo", "as_v2": True})
    assert r.get("schema") == V2
    assert "graph" in r
    assert r["graph"]["start"] == "s0"


def test_tool_get_scenario_as_v1_passthrough(http_server):
    r = _call_tool(http_server, "get_scenario", {"name": "soldatik-demo", "as_v2": False})
    # stored file is v1 — as_v2=False must skip migration and return it raw
    assert r.get("schema") == V1
    assert "graph" not in r  # v2 migration shape must not appear


def test_tool_migrate_v1_to_v2_direct(http_server):
    v1 = _call_tool(http_server, "get_scenario", {"name": "soldatik-demo", "as_v2": False})
    assert v1.get("schema") == V1
    r = _call_tool(http_server, "migrate_v1_to_v2", {"scenario": v1})
    assert r.get("schema") == V2
    assert r["graph"]["start"] == "s0"
    assert len(r["graph"]["nodes"]) >= 1


def test_tool_get_scenario_missing_name_returns_error(http_server):
    r = _call_tool(http_server, "get_scenario", {"name": "definitely-not-a-scenario-xyz"})
    assert r.get("error"), r


def test_tool_create_blank(http_server):
    r = _call_tool(http_server, "create_blank_scenario", {"meta": {"name": "mcp blank", "seed": 5}})
    assert r["schema"] == V2
    assert r["graph"]["start"] == "s0"


def test_tool_build_graph_validates(http_server):
    spec = {
        "name": "mcp build test", "seed": 42,
        "nodes": [
            {"id": "n1", "kind": "turn", "side": "p1"},
            {"id": "n2", "kind": "action", "side": "p1", "type": "end_turn"},
        ],
    }
    r = _call_tool(http_server, "build_graph", {"spec": spec})
    assert "scenario" in r and "validation" in r
    sc = r["scenario"]
    assert sc["schema"] == V2
    assert sc["graph"]["start"] == "s0"
    assert r["validation"].get("ok") is True, r["validation"]


def test_tool_build_graph_inherits_base(http_server):
    """build_graph with base_scenario_name inherits init + meta from a saved v1 scenario."""
    spec = {
        "base_scenario_name": "soldatik-demo",
        "nodes": [{"id": "n1", "kind": "turn", "side": "p1"},
                  {"id": "n2", "kind": "action", "side": "p1", "type": "end_turn"}],
    }
    r = _call_tool(http_server, "build_graph", {"spec": spec})
    assert "scenario" in r and r["validation"].get("ok") is True, r
    sc = r["scenario"]
    # init inherited from soldatik (turn 15)
    init = sc["graph"]["nodes"][0]["scene"]
    assert init["type"] == "init"
    assert init["turn_number"] == 15
    # name inherited (not the default "Новый сценарий")
    assert sc["name"] == "Солдатик demo"


def test_tool_build_graph_base_with_init_scene_inherits_meta(http_server):
    """Minor: base_scenario_name + init_scene → meta still inherited, init_scene wins for board."""
    custom_init = _call_tool(http_server, "create_blank_scenario",
                             {"meta": {"name": "custom", "turn_number": 2}})["graph"]["nodes"][0]["scene"]
    spec = {
        "base_scenario_name": "soldatik-demo",
        "init_scene": custom_init,   # override board
        "nodes": [{"id": "n1", "kind": "action", "side": "p1", "type": "end_turn"}],
    }
    r = _call_tool(http_server, "build_graph", {"spec": spec})
    sc = r["scenario"]
    init = sc["graph"]["nodes"][0]["scene"]
    assert init["turn_number"] == 2          # custom init wins
    assert r["validation"].get("ok") is True, r["validation"]


def test_tool_validate_blank(http_server):
    sc = create_blank_scenario({"name": "validate blank"})
    r = _call_tool(http_server, "validate_scenario", {"scenario": sc})
    assert r.get("ok") is True, r


def test_tool_preview_frames_soldatik(http_server):
    sc = _call_tool(http_server, "get_scenario", {"name": "soldatik-demo", "as_v2": True})
    r = _call_tool(http_server, "preview_frames", {"scenario": sc})
    assert r.get("frame_count", 0) >= 1, r
    assert "frames" in r
    assert "preview_arena_url" in r
    assert "_auth=" in r["preview_arena_url"]
    assert r["viewer_side"] in ("p1", "p2")
    last = r["frames"][-1]
    assert "p1" in last and "p2" in last
    assert last["p1"]["user_id"] in (1001, 2002)


def test_tool_preview_frames_error_propagates(http_server):
    sc = create_blank_scenario({"name": "broken"})
    # p2 end_turn on p1's init turn → failfast (side_uids[p2] != current owner)
    sc["graph"]["nodes"].append({"id": "n1", "kind": "action", "side": "p2",
                                 "action": {"type": "end_turn", "delay_ms": 500}})
    sc["graph"]["edges"].append({"id": "e1", "from": "s0", "to": "n1"})
    r = _call_tool(http_server, "preview_frames", {"scenario": sc})
    # must surface a top-level error (compute_frames failfast), not succeed silently
    assert r.get("error"), r


def test_tool_get_frames_summary_and_raw(http_server):
    sc = _call_tool(http_server, "get_scenario", {"name": "soldatik-demo", "as_v2": True})
    prev = _call_tool(http_server, "preview_frames", {"scenario": sc})
    run_id = prev["run_id"]
    # summary (default)
    s = _call_tool(http_server, "get_frames", {"run_id": run_id})
    assert s["frame_count"] >= 1
    assert "frames" in s and "p1" in s["frames"][0]
    assert "preview_arena_url" not in s  # get_frames has no arena url
    # raw
    raw = _call_tool(http_server, "get_frames", {"run_id": run_id, "summary": False})
    assert raw["run_id"] == run_id
    assert "side_uids" in raw and "frames" in raw
    # raw frames carry the full snapshot (not the summarized p1/p2 split)
    assert "snapshot" in raw["frames"][0]


def test_tool_get_frames_unknown_run(http_server):
    r = _call_tool(http_server, "get_frames", {"run_id": "no-such-run-xyz"})
    assert r.get("error"), r


def test_tool_list_cards_filter(http_server):
    r = _call_tool(http_server, "list_cards", {"filter": "солдат"})
    assert "cards" in r
    assert r["count"] >= 1
    assert any("солдат" in (c.get("name") or "").lower() for c in r["cards"])


def test_tool_list_cards_filter_by_id(http_server):
    allc = _call_tool(http_server, "list_cards", {"filter": ""})
    assert allc["count"] >= 1
    cid = str(allc["cards"][0]["id"])
    r = _call_tool(http_server, "list_cards", {"filter": cid})
    assert r["count"] == 1
    assert str(r["cards"][0]["id"]) == cid


def test_tool_list_cosmetics(http_server):
    r = _call_tool(http_server, "list_cosmetics", {})
    assert isinstance(r, dict)
    assert "error" not in r
    # cosmetics endpoint exposes avatars/backgrounds
    assert ("avatars" in r) or ("backgrounds" in r) or ("cosmetics" in r)


def test_save_and_delete_scenario_roundtrip(http_server):
    name = "mcp roundtrip delete me"
    sc = create_blank_scenario({"name": name, "seed": 13})
    saved = _call_tool(http_server, "save_scenario", {"scenario": sc})
    assert saved.get("ok") or saved.get("file")
    # confirm it persisted
    got = _call_tool(http_server, "get_scenario", {"name": name, "as_v2": False})
    assert got.get("schema") == V2
    assert got["seed"] == 13
    # delete
    dele = _call_tool(http_server, "delete_scenario", {"name": name})
    assert dele.get("ok") or dele.get("deleted") is True
    # confirm it's gone
    gone = _call_tool(http_server, "get_scenario", {"name": name, "as_v2": False})
    assert gone.get("error")


def test_tool_delete_scenario_missing(http_server):
    r = _call_tool(http_server, "delete_scenario", {"name": "no-such-scenario-xyz"})
    # server returns ok:false or error for missing
    assert r.get("ok") is False or r.get("error") or r.get("deleted") is False


def test_dispatch_tools_call_wraps_content(http_server):
    sc = create_blank_scenario({"name": "dispatch wrap"})
    r = _dispatch(http_server, "validate_scenario", {"scenario": sc})
    assert r["isError"] is False
    assert r["content"][0]["type"] == "text"
    payload = json.loads(r["content"][0]["text"])
    assert payload.get("ok") is True


def test_dispatch_unknown_tool_is_error(http_server):
    r = _dispatch(http_server, "nope", {})
    assert r["isError"] is True


# ---------------------------------------------------------------------------
# export_mp4 via stub client (no Playwright / no HTTP) — covers wait/timeout/error
# ---------------------------------------------------------------------------

class _StubClient:
    """Minimal async stub mimicking OrchestraClient.record/record_status/record_download."""
    def __init__(self, record_resp, status_seq, download_resp=b"\x00\x01\x02mp4data",
                 download_ctype="video/mp4"):
        self.record_resp = record_resp
        self._statuses = deque(status_seq)
        self.download_resp = download_resp
        self.download_ctype = download_ctype
        self.record_calls = []
        self.status_calls = []
        self.download_calls = []

    async def record(self, scenario, fmt="mp4"):
        self.record_calls.append((scenario, fmt))
        return self.record_resp

    async def record_status(self, job_id):
        self.status_calls.append(job_id)
        if self._statuses:
            return self._statuses.popleft()
        return {"status": "failed", "error": "exhausted"}

    async def record_download(self, job_id):
        self.download_calls.append(job_id)
        if self.download_resp is None:
            return None, "download failed (status=404): not ready"
        return self.download_resp, self.download_ctype


def test_export_mp4_wait_false_returns_pending():
    client = _StubClient({"job_id": "j1", "run_id": "r1", "status": "pending"}, [])
    r = _tool_with_client(client, "export_mp4", {"scenario": {"name": "x"}, "wait": False})
    assert r["job_id"] == "j1"
    assert r["status"] == "pending"            # server vocabulary, not 'started'
    assert r["format"] == "mp4"
    assert r["download_url"].endswith("/record/j1/download")
    assert client.status_calls == []           # no polling
    assert client.record_calls == [({"name": "x"}, "mp4")]  # fmt passed through


def test_export_mp4_wait_true_done():
    client = _StubClient({"job_id": "j2", "run_id": "r2", "status": "pending"},
                         [{"status": "pending"}, {"status": "done", "mp4_name": "out.mp4", "file_name": "out.mp4"}])
    r = _tool_with_client(client, "export_mp4", {"scenario": {"name": "x"}, "timeout": 30})
    assert r["status"] == "done"
    assert r["mp4_name"] == "out.mp4"
    assert r["format"] == "mp4"
    assert r["download_url"].endswith("/record/j2/download")


def test_export_mp4_wait_true_failed():
    client = _StubClient({"job_id": "j3", "run_id": "r3", "status": "pending"},
                         [{"status": "failed", "error": "boom"}])
    r = _tool_with_client(client, "export_mp4", {"scenario": {"name": "x"}, "timeout": 30})
    assert r["status"] == "failed"
    assert r["error"] == "boom"


def test_export_mp4_unknown_job_does_not_hang():
    """Minor: lost/unknown job (status None / error) breaks immediately, not stall to timeout."""
    client = _StubClient({"job_id": "j4", "run_id": "r4", "status": "pending"},
                         [{"error": "unknown_job"}])  # server 404 → no status key
    r = _tool_with_client(client, "export_mp4", {"scenario": {"name": "x"}, "timeout": 30})
    assert r["job_id"] == "j4"
    assert r.get("error") == "unknown_job"
    assert r["download_url"].endswith("/record/j4/download")


def test_export_gif_wait_true_done():
    """export_gif → record(scenario, fmt='gif') + status polling → gif file."""
    client = _StubClient({"job_id": "g1", "run_id": "rg", "status": "pending"},
                         [{"status": "pending"}, {"status": "done", "format": "gif", "file_name": "demo-g1.gif"}])
    r = _tool_with_client(client, "export_gif", {"scenario": {"name": "demo"}, "timeout": 30})
    assert r["status"] == "done"
    assert r["format"] == "gif"
    assert r["file_name"] == "demo-g1.gif"
    assert r["download_url"].endswith("/record/g1/download")
    assert client.record_calls == [({"name": "demo"}, "gif")]


def test_export_gif_wait_false_returns_pending():
    client = _StubClient({"job_id": "g2", "run_id": "rg2", "status": "pending"}, [])
    r = _tool_with_client(client, "export_gif", {"scenario": {"name": "demo"}, "wait": False})
    assert r["job_id"] == "g2"
    assert r["status"] == "pending"
    assert r["format"] == "gif"
    assert client.record_calls == [({"name": "demo"}, "gif")]
    assert client.status_calls == []


def test_get_record_status_injects_download_url():
    client = _StubClient({}, [{"status": "done", "mp4_name": "z.mp4"}])
    r = _tool_with_client(client, "get_record_status", {"job_id": "j5"})
    assert r["status"] == "done"
    assert r["download_url"].endswith("/record/j5/download")


# ---------------------------------------------------------------------------
# get_record_file / inline bytes — агент ПОЛУЧАЕТ файл (байты), а не только URL
# ---------------------------------------------------------------------------

def test_get_record_file_mp4_returns_inline_bytes():
    """get_record_file доставляет байты mp4 инлайн: _content = resource-blob."""
    client = _StubClient(
        {}, [{"status": "done", "format": "mp4", "file_name": "demo-abc.mp4"}],
        download_resp=b"FAKEMP4BYTES", download_ctype="video/mp4")
    r = _tool_with_client(client, "get_record_file", {"job_id": "jvid"})
    assert r["status"] == "done"
    assert r["format"] == "mp4"
    assert r["file_name"] == "demo-abc.mp4"
    assert r["size_bytes"] == len(b"FAKEMP4BYTES")
    assert r["mime_type"] == "video/mp4"
    assert r["inline"] is True
    assert r["download_url"].endswith("/record/jvid/download")
    assert client.download_calls == ["jvid"]
    # _content — resource-blob с base64 байтами mp4 (dispatch его вынесет из text)
    assert isinstance(r.get("_content"), list) and len(r["_content"]) == 1
    item = r["_content"][0]
    assert item["type"] == "resource"
    assert item["resource"]["mimeType"] == "video/mp4"
    assert item["resource"]["uri"].endswith("/record/jvid/download")
    import base64 as _b64
    assert _b64.b64decode(item["resource"]["blob"]) == b"FAKEMP4BYTES"


def test_get_record_file_gif_returns_image_content():
    """gif → ImageContent (omni-клиенты рендерят инлайн) + байты доступны."""
    client = _StubClient(
        {}, [{"status": "done", "format": "gif", "file_name": "demo-abc.gif"}],
        download_resp=b"GIF89a...", download_ctype="image/gif")
    r = _tool_with_client(client, "get_record_file", {"job_id": "jgif"})
    assert r["mime_type"] == "image/gif"
    item = r["_content"][0]
    assert item["type"] == "image"
    assert item["mimeType"] == "image/gif"
    import base64 as _b64
    assert _b64.b64decode(item["data"]) == b"GIF89a..."


def test_get_record_file_not_ready_returns_error():
    """status != done → error, байты не запрашиваются (download_calls пуст)."""
    client = _StubClient({}, [{"status": "pending"}])
    r = _tool_with_client(client, "get_record_file", {"job_id": "jpen"})
    assert r.get("error")
    assert "not ready" in r["error"]
    assert client.download_calls == []  # не дёргаем download у неготового job


def test_get_record_file_download_failure_returns_error():
    """download провалился → error с метаданными, без _content."""
    client = _StubClient(
        {}, [{"status": "done", "format": "mp4", "file_name": "x.mp4"}],
        download_resp=None)  # record_download → (None, err)
    r = _tool_with_client(client, "get_record_file", {"job_id": "jdl"})
    assert r.get("error")
    assert "_content" not in r
    assert r["download_url"].endswith("/record/jdl/download")


def test_export_mp4_inline_true_attaches_bytes():
    """export_mp4 wait=true + inline=true → байты mp4 инлайн в ответе."""
    client = _StubClient(
        {"job_id": "ji", "run_id": "ri", "status": "pending"},
        [{"status": "pending"}, {"status": "done", "format": "mp4", "file_name": "out.mp4"}],
        download_resp=b"MP4BODY", download_ctype="video/mp4")
    r = _tool_with_client(client, "export_mp4",
                          {"scenario": {"name": "x"}, "timeout": 30, "inline": True})
    assert r["status"] == "done"
    assert r["inline"] is True
    assert r["size_bytes"] == len(b"MP4BODY")
    assert r["mime_type"] == "video/mp4"
    item = r["_content"][0]
    assert item["type"] == "resource"
    import base64 as _b64
    assert _b64.b64decode(item["resource"]["blob"]) == b"MP4BODY"
    assert client.download_calls == ["ji"]


def test_export_mp4_inline_false_no_bytes():
    """inline=false (default) → только URL+метаданные, download не дёргается."""
    client = _StubClient(
        {"job_id": "jn", "run_id": "rn", "status": "pending"},
        [{"status": "done", "format": "mp4", "file_name": "out.mp4"}])
    r = _tool_with_client(client, "export_mp4", {"scenario": {"name": "x"}, "timeout": 30})
    assert r["status"] == "done"
    assert r["download_url"].endswith("/record/jn/download")
    assert "_content" not in r
    assert client.download_calls == []


def test_export_gif_inline_true_image_content():
    """export_gif inline=true → ImageContent (gif)."""
    client = _StubClient(
        {"job_id": "jgi", "run_id": "rg", "status": "pending"},
        [{"status": "done", "format": "gif", "file_name": "out.gif"}],
        download_resp=b"GIFBODY", download_ctype="image/gif")
    r = _tool_with_client(client, "export_gif",
                          {"scenario": {"name": "x"}, "timeout": 30, "inline": True})
    assert r["status"] == "done" and r["mime_type"] == "image/gif"
    assert r["_content"][0]["type"] == "image"


def test_dispatch_inline_bytes_split_from_text():
    """dispatch выносит _content из text-метаданных в отдельный content-item:
    text НЕ содержит base2 (не раздут), content = [text, resource/image]."""
    client = _StubClient(
        {}, [{"status": "done", "format": "mp4", "file_name": "z.mp4"}],
        download_resp=b"BLOB", download_ctype="video/mp4")
    server = MCPServer(client=client, base_url=BASE)
    async def _go():
        return await server.dispatch("tools/call", {"name": "get_record_file",
                                                    "arguments": {"job_id": "jd"}})
    r = _run(_go())
    assert r["isError"] is False
    # два content-элемента: text-метаданные + resource-blob
    assert len(r["content"]) == 2
    assert r["content"][0]["type"] == "text"
    meta = json.loads(r["content"][0]["text"])
    assert meta["file_name"] == "z.mp4"
    assert "_content" not in meta          # base2 не продублирован в text
    assert "blob" not in r["content"][0]["text"]
    assert r["content"][1]["type"] == "resource"
    assert r["content"][1]["resource"]["mimeType"] == "video/mp4"


# ---------------------------------------------------------------------------
# stdio end-to-end: spawn `python -m extra_orchestra.mcp_server`, talk JSON-RPC
# ---------------------------------------------------------------------------

def _stdio_call(proc, req):
    proc.stdin.write((json.dumps(req) + "\n").encode())
    proc.stdin.flush()
    return json.loads(proc.stdout.readline().decode())


def _spawn_mcp(base: str):
    env = dict(os.environ)
    env["ORCH_AUTO_START"] = "0"  # server already up
    return subprocess.Popen(
        [sys.executable, "-m", "extra_orchestra.mcp_server", "--base-url", base],
        cwd=str(_REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env)


def test_stdio_initialize_and_tools_list(http_server):
    proc = _spawn_mcp(http_server)
    try:
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert r["id"] == 1
        assert r["result"]["serverInfo"]["name"] == "extra-orchestra"
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = {t["name"] for t in r["result"]["tools"]}
        assert names == ALL_TOOLS and len(names) == 16
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                               "params": {"name": "list_scenarios", "arguments": {}}})
        payload = json.loads(r["result"]["content"][0]["text"])
        assert any(s["id"] == "soldatik-demo" for s in payload["scenarios"])
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stdio_unknown_method_top_level_error(http_server):
    """Major: unknown method over stdio → top-level 'error', no 'result'."""
    proc = _spawn_mcp(http_server)
    try:
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 7, "method": "bogus/method", "params": {}})
        assert "error" in r and "result" not in r
        assert r["error"]["code"] == -32601
        assert r["id"] == 7
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stdio_notification_gets_no_response(http_server):
    """Major: notification (no id) → server must NOT write a response line."""
    proc = _spawn_mcp(http_server)
    try:
        # send a notification (no id) then a real request; the next stdout line
        # must be the response to id=99, NOT to the notification.
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized",
                                      "params": {}}) + "\n").encode())
        proc.stdin.flush()
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}})
        assert r["id"] == 99  # not echoed as 0 / not a notification response
        assert "tools" in r["result"]
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stdio_parse_error_id_null(http_server):
    """Minor: malformed JSON → -32700 with id:null (not 0)."""
    proc = _spawn_mcp(http_server)
    try:
        proc.stdin.write(b"{not valid json\n")
        proc.stdin.flush()
        r = json.loads(proc.stdout.readline().decode())
        assert "error" in r and "result" not in r
        assert r["error"]["code"] == -32700
        assert r["id"] is None
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stdio_full_agent_flow(http_server):
    """Agent flow over stdio: load → build → preview → validate → get_frames."""
    proc = _spawn_mcp(http_server)
    try:
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                               "params": {"name": "get_scenario",
                                          "arguments": {"name": "soldatik-demo", "as_v2": True}}})
        sc = json.loads(r["result"]["content"][0]["text"])
        assert sc["schema"] == V2
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                               "params": {"name": "preview_frames",
                                          "arguments": {"scenario": sc}}})
        prev = json.loads(r["result"]["content"][0]["text"])
        assert prev["frame_count"] >= 1
        assert "_auth=" in prev["preview_arena_url"]
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 12, "method": "tools/call",
                               "params": {"name": "validate_scenario",
                                          "arguments": {"scenario": sc}}})
        val = json.loads(r["result"]["content"][0]["text"])
        assert val["ok"] is True
        # re-fetch frames by run_id (get_frames tool)
        r = _stdio_call(proc, {"jsonrpc": "2.0", "id": 13, "method": "tools/call",
                               "params": {"name": "get_frames",
                                          "arguments": {"run_id": prev["run_id"],
                                                        "summary": False}}})
        raw = json.loads(r["result"]["content"][0]["text"])
        assert raw["run_id"] == prev["run_id"] and "snapshot" in raw["frames"][0]
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# audit-fix regression tests (2026-06-28): isError on error-dict, -32602,
# batch/non-object guard, ping, spawn host/port, path-traversal in store.load
# ---------------------------------------------------------------------------

def test_dispatch_iserror_true_on_execution_error(http_server):
    """M1: tool returning {error:...} (execution failure) → isError=true,
    NOT a success with error buried in text."""
    r = _dispatch(http_server, "get_scenario", {"name": "definitely-not-a-scenario-xyz"})
    assert r["isError"] is True
    body = json.loads(r["content"][0]["text"])
    assert body.get("error")


def test_dispatch_iserror_false_on_validation_result(http_server):
    """M1 edge: validate_scenario {ok:false,error:...} — это корректный ответ
    инструмента (валидация провалена), isError=false; агент читает ok=false."""
    blank = create_blank_scenario()
    blank["graph"]["nodes"].append({"id": "n1", "kind": "action", "side": None,
                                    "action": {"type": "end_turn", "delay_ms": 100}})
    blank["graph"]["edges"].append({"id": "e1", "from": "s0", "to": "n1"})
    r = _dispatch(http_server, "validate_scenario", {"scenario": blank})
    assert r["isError"] is False
    body = json.loads(r["content"][0]["text"])
    assert body.get("ok") is False


def test_dispatch_unknown_tool_iserror(http_server):
    """unknown tool → isError=true (MCP-conventional), clean message."""
    r = _dispatch(http_server, "no_such_tool", {})
    assert r["isError"] is True


def test_dispatch_missing_required_arg_is_32602(http_server):
    """m1: missing required argument → JSON-RPC -32602 (top-level error),
    не криптый KeyError в isError-content."""
    r = _dispatch_method(http_server, "tools/call",
                         {"name": "get_scenario", "arguments": {}})  # нет 'name'
    assert "error" in r and "content" not in r
    assert r["error"]["code"] == -32602


def test_dispatch_non_dict_arguments_is_32602(http_server):
    """non-dict arguments → -32602, не AttributeError."""
    r = _dispatch_method(http_server, "tools/call",
                         {"name": "list_scenarios", "arguments": ["not", "a", "dict"]})
    assert "error" in r and "content" not in r
    assert r["error"]["code"] == -32602


def test_dispatch_missing_tool_name_is_32602(http_server):
    """tools/call без name → -32602."""
    r = _dispatch_method(http_server, "tools/call", {"arguments": {}})
    assert r["error"]["code"] == -32602


def test_dispatch_ping():
    """ping → {} result (keepalive)."""
    r = _dispatch_method(BASE, "ping", {})
    assert r == {}


def test_stdio_batch_request(http_server):
    """M2: JSON-RPC batch (array of 2 requests) → array of 2 responses,
    не краш процесса."""
    proc = _spawn_mcp(http_server)
    try:
        batch = [
            {"jsonrpc": "2.0", "id": "a", "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": "b", "method": "initialize", "params": {}},
        ]
        proc.stdin.write((json.dumps(batch) + "\n").encode())
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline().decode())
        assert isinstance(resp, list)
        ids = {x.get("id") for x in resp}
        assert ids == {"a", "b"}
        assert "result" in resp[0] and "tools" in resp[0]["result"]
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stdio_non_object_request_is_32600(http_server):
    """M2: scalar/non-object JSON line → -32600 invalid request, не краш."""
    proc = _spawn_mcp(http_server)
    try:
        proc.stdin.write(b"123\n")
        proc.stdin.flush()
        r = json.loads(proc.stdout.readline().decode())
        assert r["error"]["code"] == -32600
        assert r["id"] is None
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stdio_batch_with_notification_no_response_for_it(http_server):
    """batch с notification (no id) → ответ только на запрос с id."""
    proc = _spawn_mcp(http_server)
    try:
        batch = [
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": "x", "method": "ping", "params": {}},
        ]
        proc.stdin.write((json.dumps(batch) + "\n").encode())
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline().decode())
        assert isinstance(resp, list) and len(resp) == 1
        assert resp[0]["id"] == "x"
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


class _DummySock:
    def bind(self, *a, **k): pass
    def close(self): pass


def test_spawn_server_forwards_host_port(tmp_path):
    """M4: _spawn_server передаёт --host/--port из base_url серверу.
    Проверяем командную строку Popen через monkeypatch (не запуская сервер)."""
    import extra_orchestra.mcp_server as m
    import unittest.mock as mock
    captured = {}

    class FakeProc:
        def terminate(self): pass
        def wait(self, timeout=None): pass
        def kill(self): pass

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return FakeProc()

    with mock.patch.object(m.subprocess, "Popen", fake_popen), \
         mock.patch.object(m, "_port_responds", lambda url: False), \
         mock.patch.object(m.socket, "socket", lambda: _DummySock()):
        orig_sleep = m.time.sleep
        m.time.sleep = lambda *a, **k: None
        try:
            res = m._spawn_server("http://127.0.0.1:18095")
        finally:
            m.time.sleep = orig_sleep
    assert "--host" in captured["cmd"] and "127.0.0.1" in captured["cmd"]
    assert "--port" in captured["cmd"] and "18095" in captured["cmd"]
    assert res is None  # health не пришёл → None, не мёртвый proc


def test_spawn_server_returns_none_on_foreign_port(tmp_path):
    """M5: порт занят чужим процессом (bind OSError) → None, не падает."""
    import extra_orchestra.mcp_server as m
    import unittest.mock as mock

    class _BindFailSock:
        def bind(self, *a, **k): raise OSError("addr in use")
        def close(self): pass

    with mock.patch.object(m.socket, "socket", lambda: _BindFailSock()):
        res = m._spawn_server("http://127.0.0.1:8095")
    assert res is None


def test_store_load_path_traversal_confined(tmp_path):
    """Security: store.load('../foo') не читает файл вне scenarios/."""
    from extra_orchestra.components.scenario_store import ScenarioStore
    d = tmp_path / "scenarios"
    d.mkdir()
    outside = tmp_path / "secret.json"
    outside.write_text('{"schema":"extra_orchestra.scenario.v1","name":"secret"}', encoding="utf-8")
    store = ScenarioStore(d)
    assert store.load("../secret") is None       # alt-ветка без .json
    assert store.load("../secret.json") is None  # basename-confined
    # legit fallback внутри scenarios остаётся рабочим
    (d / "soldatik-demo.json").write_text('{"schema":"x","name":"soldatik"}', encoding="utf-8")
    got = store.load("soldatik-demo")
    assert got is not None and got["name"] == "soldatik"