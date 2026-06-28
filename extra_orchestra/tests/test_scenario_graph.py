"""Тесты v2 graph-редактора: структурная валидация, раннер, миграция v1→v2,
back-compat dispatch. Спецификация — в ``components/scenario_graph_runner.py``.

Покрывает:
  * ``validate_graph_structure`` — все ветви ошибок + валидный минимум.
  * ``run_scenario_graph`` — aoe_silence parity, end_turn flip, hold, fail-fast
    на wrong-side (action и turn), game_over mid-path, determinism, v1↔v2
    equivalence мигрированного soldatik.
  * ``migrate_v1_to_v2`` — end_with_end_turn→explicit end_turn, wait→hold,
    preserves mechanics_override/is_ready, пустой ход→hold, явный end_turn не
    дублируется.
  * dispatch — v1 бежит через dispatch идентично v1; store.save не перезаписывает
    schema v2; list() считает nodes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from extra_orchestra.components.cards_catalog import CardsCatalog
from extra_orchestra.components.scenario_engine import run_scenario as run_v1
from extra_orchestra.components.scenario_graph_runner import (
    V2,
    migrate_v1_to_v2,
    run_scenario_dispatch,
    run_scenario_graph,
    validate_graph_structure,
    validate_scenario_dispatch,
    validate_scenario_graph,
)
from extra_orchestra.components.scenario_store import ScenarioStore

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE.parent / "scenarios"
SOLDATIK = SCENARIOS / "soldatik-demo.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog():
    return CardsCatalog()


@pytest.fixture
def soldatik_v1():
    return json.loads(SOLDATIK.read_text(encoding="utf-8"))


@pytest.fixture
def soldatik_v2(soldatik_v1):
    return migrate_v1_to_v2(copy.deepcopy(soldatik_v1))


def _init_scene(**overrides) -> Dict[str, Any]:
    base = {
        "type": "init", "turn_number": 1, "starting_side": "p1", "display_ms": 1000,
        "p1": {"user_id": 1001, "nickname": "A", "rarity": "common", "mana": 6, "max_mana": 6,
               "hero": {"card_id": 1, "level": 1}, "hand": [], "board": [], "deck": []},
        "p2": {"user_id": 2002, "is_bot": True, "nickname": "B", "rarity": "common", "mana": 6,
               "max_mana": 6, "hero": {"card_id": 3, "level": 1}, "hand": [], "board": [], "deck": []},
    }
    base.update(overrides)
    return base


def _scenario(nodes, edges, start="s0", **top) -> Dict[str, Any]:
    sc = {
        "schema": V2, "name": "t", "seed": 42, "viewer_side": "p1",
        "match_id": "t", "classic_params": {"sudden_death_enabled": False, "mana_per_turn": 1,
                                            "turn_duration_seconds": 25},
        "graph": {"start": start, "nodes": nodes, "edges": edges},
        "layout": {n["id"]: {"x": 40 + 220 * i, "y": 90} for i, n in enumerate(nodes)},
    }
    sc.update(top)
    return sc


def _init_node(scene=None) -> Dict[str, Any]:
    return {"id": "s0", "kind": "scene", "scene": scene or _init_scene()}


def _hold(nid, ms=600) -> Dict[str, Any]:
    return {"id": nid, "kind": "scene", "scene": {"type": "hold", "display_ms": ms}}


def _act(nid, atype, side="p1", **a) -> Dict[str, Any]:
    action = {"type": atype, "delay_ms": 500}
    action.update(a)
    return {"id": nid, "kind": "action", "side": side, "action": action}


def _edge(eid, a, b) -> Dict[str, Any]:
    return {"id": eid, "from": a, "to": b}


def _chain(nodes, start="s0") -> Dict[str, Any]:
    """Линейный граф s0 → n1 → n2 → ... (по порядку nodes)."""
    ids = [n["id"] for n in nodes]
    edges = [_edge(f"e{i}", ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    return _scenario(nodes, edges, start=start)


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------

def test_struct_valid_minimal(catalog):
    sc = _chain([_init_node(), _hold("n1")])
    assert validate_graph_structure(sc) == {"ok": True, "error": None}
    res = validate_scenario_graph(sc, catalog)
    assert res["ok"], res["error"]
    assert res["frame_count"] == 2  # init + hold


def test_struct_single_init_only_frame(catalog):
    sc = _chain([_init_node()])
    res = run_scenario_graph(sc, catalog)
    assert res["error"] is None
    assert res["frame_count"] == 1
    assert res["frames"][0]["action_kind"] == "init"


def test_struct_no_init():
    sc = _chain([_hold("n1")])
    sc["graph"]["start"] = "n1"
    r = validate_graph_structure(sc)
    assert not r["ok"] and "init" in r["error"]


def test_struct_two_inits():
    n2 = _init_node()
    n2["id"] = "s1"
    sc = _chain([_init_node(), n2])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "one scene/init" in r["error"]


def test_struct_start_not_init():
    sc = _chain([_init_node(), _hold("n1")])
    sc["graph"]["start"] = "n1"
    r = validate_graph_structure(sc)
    assert not r["ok"] and "start must point to the init" in r["error"]


def test_struct_missing_start():
    sc = _chain([_init_node(), _hold("n1")])
    sc["graph"]["start"] = "zzz"
    r = validate_graph_structure(sc)
    assert not r["ok"] and "not in nodes" in r["error"]


def test_struct_dangling_from():
    sc = _scenario([_init_node(), _hold("n1")], [_edge("e1", "ghost", "n1")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "dangling" in r["error"]


def test_struct_dangling_to():
    sc = _scenario([_init_node(), _hold("n1")], [_edge("e1", "s0", "ghost")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "dangling" in r["error"]


def test_struct_self_loop():
    sc = _scenario([_init_node(), _hold("n1")], [_edge("e1", "n1", "n1")], start="s0")
    # нужен edge s0->n1 чтобы n1 был достижим, плюс self-loop на n1 (>1 outgoing)
    sc["graph"]["edges"] = [_edge("e1", "s0", "n1"), _edge("e2", "n1", "n1")]
    r = validate_graph_structure(sc)
    assert not r["ok"] and "self-loop" in r["error"]


def test_struct_edge_to_init():
    sc = _scenario([_init_node(), _hold("n1")], [_edge("e1", "s0", "n1"), _edge("e2", "n1", "s0")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "init" in r["error"].lower()


def test_struct_two_outgoing():
    n2 = _hold("n2")
    sc = _scenario([_init_node(), _hold("n1"), n2],
                   [_edge("e1", "s0", "n1"), _edge("e2", "s0", "n2")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "outgoing" in r["error"]


def test_struct_two_incoming():
    # orphan-узел указывает в n2 → n2 имеет 2 входящих (incoming-чек срабатывает
    # ДО reachability, поэтому изолированно ловит ветвь >1 incoming)
    sc = _scenario([_init_node(), _hold("n1"), _hold("n2"), _hold("orphan")],
                   [_edge("e1", "s0", "n1"), _edge("e2", "n1", "n2"), _edge("e3", "orphan", "n2")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "incoming" in r["error"]


def test_struct_init_has_incoming():
    sc = _scenario([_init_node(), _hold("n1")], [_edge("e1", "s0", "n1"), _edge("e2", "n1", "s0")])
    r = validate_graph_structure(sc)
    assert not r["ok"]  # edge→init поймает первым


def test_struct_edge_without_id():
    sc = _scenario([_init_node(), _hold("n1")], [{"from": "s0", "to": "n1"}])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "edge without id" in r["error"]


def test_struct_node_without_id():
    sc = _chain([_init_node(), _hold("n1")])
    sc["graph"]["nodes"].append({"kind": "scene", "scene": {"type": "hold"}})  # без id
    r = validate_graph_structure(sc)
    assert not r["ok"] and "node without id" in r["error"]


def test_struct_unreachable_orphan():
    sc = _scenario([_init_node(), _hold("n1"), _hold("orphan")],
                   [_edge("e1", "s0", "n1")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "unreachable" in r["error"] and "orphan" in r["error"]


def test_struct_cycle():
    # s0 -> n1 -> n2 -> n1 (цикл). При single-outgoing цикл неизбежно даёт
    # узлу 2 входящих → граф отвергается (цикл сам по себе запрещён структурой).
    sc = _scenario([_init_node(), _hold("n1"), _hold("n2")],
                   [_edge("e1", "s0", "n1"), _edge("e2", "n1", "n2"), _edge("e3", "n2", "n1")])
    r = validate_graph_structure(sc)
    assert not r["ok"]


def test_struct_dup_node_id():
    sc = _scenario([_init_node(), _init_node()], [])
    sc["graph"]["nodes"][1]["id"] = "s0"
    r = validate_graph_structure(sc)
    assert not r["ok"] and "duplicate node id" in r["error"]


def test_struct_dup_edge_id():
    sc = _scenario([_init_node(), _hold("n1")], [_edge("e1", "s0", "n1")])
    sc["graph"]["edges"].append(_edge("e1", "n1", "s0"))  # второе e1 (n1->s0 тоже edge→init)
    r = validate_graph_structure(sc)
    assert not r["ok"] and "duplicate edge id" in r["error"]


def test_struct_action_missing_side():
    n = _act("n1", "end_turn")
    del n["side"]
    sc = _chain([_init_node(), n])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "side" in r["error"]


def test_struct_action_bad_side():
    sc = _chain([_init_node(), _act("n1", "end_turn", side="p9")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "side" in r["error"]


def test_struct_action_bad_type():
    sc = _chain([_init_node(), _act("n1", "bogus")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "action.type" in r["error"]


def test_struct_turn_bad_side():
    sc = _chain([_init_node(), {"id": "n1", "kind": "turn", "turn": {"side": "p9"}},
                 _act("n2", "end_turn")])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "bad side" in r["error"]


def test_struct_unknown_kind():
    sc = _chain([_init_node(), {"id": "n1", "kind": "magic"}])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "unknown kind" in r["error"]


def test_struct_scene_bad_type():
    sc = _chain([_init_node(), {"id": "n1", "kind": "scene", "scene": {"type": "flash"}}])
    r = validate_graph_structure(sc)
    assert not r["ok"] and "scene.type" in r["error"]


def test_struct_zero_outgoing_terminal_ok(catalog):
    # init + end_turn без последующего — валидно (терминальный узел)
    sc = _chain([_init_node(), _act("n1", "end_turn")])
    assert validate_graph_structure(sc)["ok"]
    res = run_scenario_graph(sc, catalog)
    assert res["error"] is None


# ---------------------------------------------------------------------------
# runner semantics
# ---------------------------------------------------------------------------

def test_runner_soldatik_v2_aoe_silence_parity(soldatik_v2, catalog, soldatik_v1):
    res_v2 = run_scenario_graph(soldatik_v2, catalog)
    assert res_v2["error"] is None, res_v2["error"]
    res_v1 = run_v1(soldatik_v1, catalog)
    # parity: те же кадры и тайминги
    assert res_v2["frame_count"] == res_v1["frame_count"]
    assert res_v2["total_ms"] == res_v1["total_ms"]
    # найдём кадр после play_card (action_kind=='play_card') и проверим aoe_silence
    play_frames = [f for f in res_v2["frames"] if f["action_kind"] == "play_card"]
    assert play_frames, "нет play_card кадра"
    opp_board = play_frames[-1]["snapshot"]["opponent"]["board"]
    assert all(c["mechanics"] == [] for c in opp_board), "aoe_silence не снял механики"
    # init-кадр: ход 15, 3 вражеских юнита с механиками
    init = res_v2["frames"][0]["snapshot"]
    assert init["turn"] == 15
    assert len(init["opponent"]["board"]) == 3
    assert [c["mechanics"] for c in init["opponent"]["board"]] == [
        ["shield", "shield_refresh"], ["taunt"], ["deathrattle_aoe_damage_2"]]


def test_runner_end_turn_flips_owner(catalog):
    # init p1 -> end_turn(p1) -> hold : после end_turn owner=p2
    sc = _chain([_init_node(), _act("n1", "end_turn", side="p1"), _hold("n2")])
    res = run_scenario_graph(sc, catalog)
    assert res["error"] is None, res["error"]
    end_frame = next(f for f in res["frames"] if f["action_kind"] == "end_turn")
    hold_frame = next(f for f in res["frames"] if f["action_kind"] == "hold")
    # после end_turn owner сменился → в hold-кадре current_player_id = p2 uid
    owner_after = hold_frame["snapshot"]["current_player_id"]
    p2_uid = res["side_uids"]["p2"]
    assert owner_after == p2_uid


def test_runner_hold_frame_kind(catalog):
    sc = _chain([_init_node(), _hold("n1", 1234)])
    res = run_scenario_graph(sc, catalog)
    hold = next(f for f in res["frames"] if f["action_kind"] == "hold")
    assert hold["display_ms"] == 1234


def test_runner_hold_preserves_zero_display_ms(catalog):
    # v1 empty-turn duration_ms=0 → 0ms; v2 hold не должен коэрцать 0→600
    sc = _chain([_init_node(), _hold("n1", 0)])
    res = run_scenario_graph(sc, catalog)
    hold = next(f for f in res["frames"] if f["action_kind"] == "hold")
    assert hold["display_ms"] == 0


def test_runner_wrong_side_action_failfast(catalog):
    # init p1 -> play_card(p2): p2 не владеет ходом → fail-fast "Forgot end_turn?"
    sc = _chain([_init_node(_init_scene(starting_side="p1")),
                 _act("n1", "mana_draw", side="p2")])
    res = run_scenario_graph(sc, catalog)
    assert res["error"] is not None
    assert "current turn" in res["error"] or "Forgot end_turn" in res["error"]
    # ровно 2 кадра: init + error-кадр, дальше обход стоп
    assert res["frame_count"] == 2
    assert res["frames"][-1].get("error") == res["error"]


def test_runner_wrong_side_turn_marker(catalog):
    sc = _chain([_init_node(_init_scene(starting_side="p1")),
                 {"id": "n1", "kind": "turn", "turn": {"side": "p2", "intro_ms": 0}},
                 _act("n2", "end_turn", side="p1")])
    res = run_scenario_graph(sc, catalog)
    assert res["error"] is not None
    assert "turn" in res["error"].lower() or "current owner" in res["error"]


def test_runner_game_over_midpath(catalog):
    # p2 hero hp=1, p1 board unit attack≥1 is_ready → attack hero = lethal
    scene = _init_scene(starting_side="p1")
    scene["p2"]["hero"] = {"card_id": 3, "level": 1, "hp_override": 1}
    scene["p1"]["board"] = [{"card_id": 5, "level": 1, "attack_override": 5, "is_ready": True}]
    sc = _chain([_init_node(scene),
                 _act("n1", "attack", side="p1", attacker_index=0, target_is_hero=True),
                 _hold("n2")])
    res = run_scenario_graph(sc, catalog)
    assert res["error"] is None, res["error"]
    # должен быть game_over-кадр и обход стоп (= нет hold-кадра после)
    kinds = [f["action_kind"] for f in res["frames"]]
    assert "game_over" in kinds
    assert "hold" not in kinds, "обход должен стопнуться на game_over"


def test_runner_determinism_two_runs(soldatik_v2, catalog):
    a = run_scenario_graph(soldatik_v2, catalog)
    b = run_scenario_graph(soldatik_v2, catalog)
    assert a["frame_count"] == b["frame_count"]
    assert a["total_ms"] == b["total_ms"]
    assert json.dumps(a["frames"], ensure_ascii=False, sort_keys=True) == \
           json.dumps(b["frames"], ensure_ascii=False, sort_keys=True)


def test_runner_v1_v2_migrated_equivalence(soldatik_v1, soldatik_v2, catalog):
    r1 = run_v1(soldatik_v1, catalog)
    r2 = run_scenario_graph(soldatik_v2, catalog)
    assert r1["error"] is None and r2["error"] is None
    assert r1["frame_count"] == r2["frame_count"]
    assert r1["total_ms"] == r2["total_ms"]
    # покадровое равенство display_ms (action_kind у v1 wait vs v2 hold различается
    # меткой — это намеренно, эквивалент в кадрах/таймингах/снапшотах)
    assert [f["display_ms"] for f in r1["frames"]] == [f["display_ms"] for f in r2["frames"]]
    # снапшот после play_card: aoe_silence снял механики в обоих
    v1_play = next(f for f in r1["frames"] if f["action_kind"] == "play_card")
    v2_play = next(f for f in r2["frames"] if f["action_kind"] == "play_card")
    assert all(c["mechanics"] == [] for c in v1_play["snapshot"]["opponent"]["board"])
    assert all(c["mechanics"] == [] for c in v2_play["snapshot"]["opponent"]["board"])


# ---------------------------------------------------------------------------
# migration
# ---------------------------------------------------------------------------

def test_migrate_end_with_end_turn_appends_explicit():
    v1 = {"schema": "extra_orchestra.scenario.v1", "init_scene": _init_scene(),
          "turns": [{"id": "t1", "side": "p1", "duration_ms": 1000,
                     "end_with_end_turn": True, "nodes": []}]}
    v2 = migrate_v1_to_v2(v1)
    assert v2["schema"] == V2
    types = [(n["kind"], (n.get("action") or {}).get("type")) for n in v2["graph"]["nodes"]]
    assert ("action", "end_turn") in types, "end_with_end_turn без явного → должен добавить end_turn"


def test_migrate_explicit_end_turn_not_duplicated():
    v1 = {"schema": "extra_orchestra.scenario.v1", "init_scene": _init_scene(),
          "turns": [{"id": "t1", "side": "p1", "duration_ms": 1000,
                     "end_with_end_turn": True,
                     "nodes": [{"id": "a", "type": "end_turn", "delay_ms": 300}]}]}
    v2 = migrate_v1_to_v2(v1)
    end_turns = [n for n in v2["graph"]["nodes"]
                 if n["kind"] == "action" and n["action"]["type"] == "end_turn"]
    assert len(end_turns) == 1, "явный end_turn не должен дублироваться"


def test_migrate_wait_becomes_hold():
    v1 = {"schema": "extra_orchestra.scenario.v1", "init_scene": _init_scene(),
          "turns": [{"id": "t1", "side": "p1", "duration_ms": 1000, "end_with_end_turn": False,
                     "nodes": [{"id": "w", "type": "wait", "delay_ms": 777}]}]}
    v2 = migrate_v1_to_v2(v1)
    holds = [n for n in v2["graph"]["nodes"] if n["kind"] == "scene" and n["scene"]["type"] == "hold"]
    assert len(holds) == 1
    assert holds[0]["scene"]["display_ms"] == 777


def test_migrate_empty_turn_becomes_hold():
    v1 = {"schema": "extra_orchestra.scenario.v1", "init_scene": _init_scene(),
          "turns": [{"id": "t1", "side": "p2", "duration_ms": 2500, "end_with_end_turn": False,
                     "nodes": []}]}
    v2 = migrate_v1_to_v2(v1)
    holds = [n for n in v2["graph"]["nodes"] if n["kind"] == "scene" and n["scene"]["type"] == "hold"]
    assert len(holds) == 1 and holds[0]["scene"]["display_ms"] == 2500


def test_migrate_preserves_mechanics_override_and_ready(soldatik_v1, soldatik_v2):
    init = next(n for n in soldatik_v2["graph"]["nodes"]
                if n["kind"] == "scene" and n["scene"]["type"] == "init")
    board = init["scene"]["p2"]["board"]
    assert len(board) == 3
    # mechanics_override и is_ready сохранены из v1
    assert board[0]["mechanics_override"] == ["shield", "shield_refresh"]
    assert board[1]["mechanics_override"] == ["taunt"]
    assert board[2]["mechanics_override"] == ["deathrattle_aoe_damage_2"]
    assert all(c.get("is_ready") is True for c in board)


def test_migrate_layout_and_edges_chain(soldatik_v2):
    g = soldatik_v2["graph"]
    # каждый узел кроме start имеет ровно одно входящее; цепочка от start покрывает все
    ins = {n["id"]: 0 for n in g["nodes"]}
    for e in g["edges"]:
        ins[e["to"]] += 1
    assert ins[g["start"]] == 0
    assert all(v == 1 for k, v in ins.items() if k != g["start"])
    # layout есть для каждого узла
    assert all(n["id"] in soldatik_v2["layout"] for n in g["nodes"])


def test_migrate_idempotent_on_v2(soldatik_v2):
    again = migrate_v1_to_v2(soldatik_v2)
    assert again == soldatik_v2


def test_migrate_init_display_ms_init_intro_ms_fallback():
    scene = _init_scene()
    del scene["display_ms"]  # нет явного → должен сработать fallback на init_intro_ms
    v1 = {"schema": "extra_orchestra.scenario.v1", "init_intro_ms": 999,
          "init_scene": scene, "turns": []}
    v2 = migrate_v1_to_v2(v1)
    init = next(n for n in v2["graph"]["nodes"] if n["scene"]["type"] == "init")
    assert init["scene"]["display_ms"] == 999  # fallback на init_intro_ms


def test_migrate_init_display_ms_explicit_wins():
    scene = _init_scene(display_ms=2500)
    v1 = {"schema": "extra_orchestra.scenario.v1", "init_intro_ms": 999,
          "init_scene": scene, "turns": []}
    v2 = migrate_v1_to_v2(v1)
    init = next(n for n in v2["graph"]["nodes"] if n["scene"]["type"] == "init")
    assert init["scene"]["display_ms"] == 2500


def test_migrate_empty_turn_with_end_turn_chains_both():
    # пустой ход + end_with_end_turn → hold И end_turn, оба в цепочке
    v1 = {"schema": "extra_orchestra.scenario.v1", "init_scene": _init_scene(),
          "turns": [{"id": "t1", "side": "p1", "duration_ms": 1500,
                     "end_with_end_turn": True, "nodes": []}]}
    v2 = migrate_v1_to_v2(v1)
    nodes = v2["graph"]["nodes"]
    assert len(nodes) == 3  # init + hold + end_turn
    assert nodes[1]["kind"] == "scene" and nodes[1]["scene"]["display_ms"] == 1500
    assert nodes[2]["kind"] == "action" and nodes[2]["action"]["type"] == "end_turn"
    # цепочка: s0 → n1 → n2
    assert [(e["from"], e["to"]) for e in v2["graph"]["edges"]] == [("s0", "n1"), ("n1", "n2")]


# ---------------------------------------------------------------------------
# dispatch + store back-compat
# ---------------------------------------------------------------------------

def test_dispatch_v1_runs_identically(soldatik_v1, catalog):
    direct = run_v1(soldatik_v1, catalog)
    via_dispatch = run_scenario_dispatch(soldatik_v1, catalog)
    assert direct["frame_count"] == via_dispatch["frame_count"]
    assert direct["total_ms"] == via_dispatch["total_ms"]


def test_dispatch_v2_routes_to_graph(soldatik_v2, catalog):
    via_graph = run_scenario_graph(soldatik_v2, catalog)
    via_dispatch = run_scenario_dispatch(soldatik_v2, catalog)
    assert via_dispatch["frame_count"] == via_graph["frame_count"]
    assert via_dispatch["total_ms"] == via_graph["total_ms"]


def test_validate_dispatch_v2(soldatik_v2, catalog):
    r = validate_scenario_dispatch(soldatik_v2, catalog)
    assert r["ok"], r["error"]


def test_store_save_does_not_restamp_v2(tmp_path, soldatik_v2):
    store = ScenarioStore(tmp_path)
    soldatik_v2["name"] = "v2keep"
    path = store.save(soldatik_v2)
    loaded = store.load(path.stem)
    assert loaded["schema"] == V2, "store.save не должен перезаписывать v2-схему на v1"
    assert loaded["graph"]["start"] == soldatik_v2["graph"]["start"]


def test_store_list_counts_v2_nodes(tmp_path, soldatik_v2):
    store = ScenarioStore(tmp_path)
    soldatik_v2["name"] = "v2list"
    store.save(soldatik_v2)
    entries = store.list()
    assert len(entries) == 1
    e = entries[0]
    assert e["schema"] == V2
    assert e["nodes"] == len(soldatik_v2["graph"]["nodes"])


def test_store_save_schemaless_v2_body_keeps_v2(tmp_path, soldatik_v2):
    # v2-тело БЕЗ поля schema (есть graph) — store.save не должен штамповать v1
    store = ScenarioStore(tmp_path)
    body = {k: v for k, v in soldatik_v2.items() if k != "schema"}
    body["name"] = "noschema"
    assert "schema" not in body
    store.save(body)
    loaded = store.load("noschema")
    assert loaded["schema"] == V2
    assert loaded["graph"]["start"] == soldatik_v2["graph"]["start"]


def test_store_save_v1_body_keeps_v1(tmp_path, soldatik_v1):
    store = ScenarioStore(tmp_path)
    soldatik_v1["name"] = "v1keep"
    store.save(soldatik_v1)
    loaded = store.load("v1keep")
    assert loaded["schema"] == "extra_orchestra.scenario.v1"
    assert "turns" in loaded