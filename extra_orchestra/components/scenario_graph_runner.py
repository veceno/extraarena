"""v2 graph scenario runner — real node-and-edge graph DSL.

Сценарий v2 (``extra_orchestra.scenario.v2``) хранит граф выполнения явно:
``graph = {start, nodes[], edges[]}``. Пользователь рисует рёбра в визуальном
редакторе (``static/editor_graph.js``); раннер идёт по пути от ``start``.

Каждый action-узел несёт ОБЯЗАТЕЛЬНОЕ поле ``side ∈ {p1,p2}`` — это
восстанавливает v1 safety-guard (``scenario_engine.py:318-324``): раннер
валидирует ``side_uids[side] == env.state.current_turn_owner_id`` ПЕРЕД
``apply_action``. Забытый ``end_turn`` → явный ScenarioError, а не silent
wrong-side action (см. design-панель, lens engine-correctness, blocker-finding).

Node kinds:
  * ``scene`` type=init  — строит GameState (один, =graph.start, без входящих).
  * ``scene`` type=hold  — удерживает текущий снимок на ``display_ms`` (замена v1 wait/post).
  * ``action`` type ∈ {play_card, attack, mana_draw, end_turn} — шаг ядра.
  * ``turn``             — опц. sanity-маркер (side); ``intro_ms`` default 0 → нет кадра.

Граф = один путь: max-1 исходящее И max-1 входящее ребро на узел, ровно один
init, все узлы достижимы из start, без циклов. ``layout``/``editor`` —
top-level, раннер игнорирует (чистая executable-семантика).

Frame dict byte-identical с v1 → recorder/bridge не трогаются.
``turn_id`` = id ближайшего предшествующего turn-узла, или ``"init"`` для
init-кадра, или ``"__no_turn__"``.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import core.effects as _effects
from core.actions import EndTurnAction

from .arena_engine import OrchestraBattleEngine
from .cards_catalog import CardsCatalog
from .scenario_engine import (
    ScenarioError,
    build_initial_state,
    make_frame,
    _node_to_action,
    _result,
)

V2 = "extra_orchestra.scenario.v2"
WAIT_FRAME_DEFAULT_MS = 600


# ---------------------------------------------------------------------------
# Structural validation (без catalog)
# ---------------------------------------------------------------------------

def validate_graph_structure(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Проверка структуры графа без catalog. ``{ok, error}``."""
    graph = scenario.get("graph") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    start = graph.get("start")

    if not start:
        return {"ok": False, "error": "graph.start is required"}
    by_id: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        nid = n.get("id")
        if not nid:
            return {"ok": False, "error": "node without id"}
        if nid in by_id:
            return {"ok": False, "error": f"duplicate node id: {nid}"}
        by_id[nid] = n
    if start not in by_id:
        return {"ok": False, "error": f"graph.start '{start}' not in nodes"}

    # ровно один init
    inits = [nid for nid, n in by_id.items() if n.get("kind") == "scene"
             and (n.get("scene") or {}).get("type") == "init"]
    if len(inits) != 1:
        return {"ok": False, "error": f"expected exactly one scene/init node, got {len(inits)}"}
    if inits[0] != start:
        return {"ok": False, "error": f"graph.start must point to the init node (got '{start}', init='{inits[0]}')"}

    edge_ids = set()
    outgoing: Dict[str, List[str]] = {nid: [] for nid in by_id}
    incoming: Dict[str, List[str]] = {nid: [] for nid in by_id}
    for e in edges:
        eid = e.get("id")
        if not eid:
            return {"ok": False, "error": "edge without id"}
        if eid in edge_ids:
            return {"ok": False, "error": f"duplicate edge id: {eid}"}
        edge_ids.add(eid)
        a, b = e.get("from"), e.get("to")
        if a not in by_id:
            return {"ok": False, "error": f"edge {eid}: from '{a}' not in nodes (dangling)"}
        if b not in by_id:
            return {"ok": False, "error": f"edge {eid}: to '{b}' not in nodes (dangling)"}
        if a == b:
            return {"ok": False, "error": f"edge {eid}: self-loop on '{a}'"}
        if b == inits[0]:
            return {"ok": False, "error": f"edge {eid}: targets init/start node '{b}' (forbidden)"}
        outgoing[a].append(b)
        incoming[b].append(a)

    for nid, outs in outgoing.items():
        if len(outs) > 1:
            return {"ok": False, "error": f"node '{nid}' has {len(outs)} outgoing edges (max 1)"}
    for nid, ins in incoming.items():
        if len(ins) > 1:
            return {"ok": False, "error": f"node '{nid}' has {len(ins)} incoming edges (max 1)"}
    # init не имеет входящих
    if incoming[inits[0]]:
        return {"ok": False, "error": f"init node '{inits[0]}' must have no incoming edges"}

    # reachability from start + cycle detection (single-outgoing → один путь)
    visited: set = set()
    on_stack: set = set()
    cur = start
    while cur is not None:
        if cur in on_stack:
            return {"ok": False, "error": f"cycle detected at '{cur}'"}
        if cur in visited:
            # повторный заход невозможен при single-incoming, но защитимся
            return {"ok": False, "error": f"cycle detected at '{cur}'"}
        visited.add(cur)
        on_stack.add(cur)
        outs = outgoing[cur]
        cur = outs[0] if outs else None
    # on_stack не нужен для single-path, но оставим чистым
    reachable = visited
    orphans = [nid for nid in by_id if nid not in reachable]
    if orphans:
        return {"ok": False, "error": f"unreachable nodes from start: {orphans}"}

    # per-node semantic checks
    for nid, n in by_id.items():
        kind = n.get("kind")
        if kind == "action":
            side = n.get("side")
            if side not in ("p1", "p2"):
                return {"ok": False, "error": f"action node '{nid}' missing/invalid side '{side}' (required p1|p2)"}
            atype = (n.get("action") or {}).get("type")
            if atype not in ("play_card", "attack", "mana_draw", "end_turn"):
                return {"ok": False, "error": f"action node '{nid}' bad action.type '{atype}'"}
        elif kind == "turn":
            t = n.get("turn") or {}
            if t.get("side") is not None and t["side"] not in ("p1", "p2"):
                return {"ok": False, "error": f"turn node '{nid}' bad side '{t.get('side')}'"}
        elif kind == "scene":
            stype = (n.get("scene") or {}).get("type")
            if stype not in ("init", "hold"):
                return {"ok": False, "error": f"scene node '{nid}' bad scene.type '{stype}'"}
        else:
            return {"ok": False, "error": f"node '{nid}' unknown kind '{kind}'"}
    return {"ok": True, "error": None}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _compat_scenario_for_init(scenario: Dict[str, Any], init_node: Dict[str, Any]) -> Dict[str, Any]:
    """build_initial_state (из scenario_engine) читает init_scene/classic_params/...
    — соберём v1-совместимый dict из v2 top-level + init-узла."""
    scene = init_node.get("scene") or {}
    init_scene = {
        "turn_number": scene.get("turn_number", 1),
        "starting_side": scene.get("starting_side", "p1"),
        "display_ms": scene.get("display_ms"),
        "p1": scene.get("p1") or {},
        "p2": scene.get("p2") or {},
    }
    compat = {k: v for k, v in scenario.items() if k not in ("graph", "layout", "editor")}
    compat["init_scene"] = init_scene
    return compat


def run_scenario_graph(
    scenario: Dict[str, Any],
    catalog: Optional[CardsCatalog] = None,
) -> Dict[str, Any]:
    """Прогнать v2 graph-сценарий → тот же shape что ``run_scenario`` v1."""
    catalog = catalog or CardsCatalog()
    frames: List[Dict[str, Any]] = []

    struct = validate_graph_structure(scenario)
    if not struct["ok"]:
        return _result(frames, 0, {}, str(scenario.get("match_id", "orchestra")), struct["error"])

    graph = scenario.get("graph") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    by_id = {n["id"]: n for n in nodes}
    outgoing: Dict[str, Optional[str]] = {}
    for n in nodes:
        outgoing[n["id"]] = None
    for e in edges:
        outgoing[e["from"]] = e["to"]  # validator guarantees single-outgoing

    init_node = by_id[graph["start"]]
    compat = _compat_scenario_for_init(scenario, init_node)
    match_id = str(scenario.get("match_id", "orchestra"))

    seed = int(scenario.get("seed", 0) or 0)
    orig_random = _effects.random
    _effects.random = random.Random(seed)
    try:
        env, engine, viewer_uid, side_uids = build_initial_state(compat, catalog)
    finally:
        _effects.random = orig_random

    def _snap() -> Dict[str, Any]:
        return engine.get_full_state(viewer_uid)

    frames: List[Dict[str, Any]] = []
    err: Optional[str] = None
    last_turn_id: Optional[str] = None

    # monkeypatch на весь обход (RISK A — как v1)
    _effects.random = random.Random(seed)
    try:
        cur = graph["start"]
        while cur is not None:
            node = by_id[cur]
            nid = node["id"]
            kind = node.get("kind")

            if kind == "scene":
                scene = node.get("scene") or {}
                if scene.get("type") == "init":
                    disp = int(scene.get("display_ms") or 1200)
                    frames.append(make_frame(
                        snapshot=_snap(), sound_events=[],
                        display_ms=disp, action_kind="init",
                        turn_id="init", node_id=nid,
                    ))
                    last_turn_id = None
                else:  # hold
                    _d = scene.get("display_ms")
                    disp = int(_d) if _d is not None else WAIT_FRAME_DEFAULT_MS
                    frames.append(make_frame(
                        snapshot=_snap(), sound_events=[],
                        display_ms=disp, action_kind="hold",
                        turn_id=last_turn_id or "__no_turn__", node_id=nid,
                    ))

            elif kind == "turn":
                t = node.get("turn") or {}
                tside = t.get("side")
                if tside is not None:
                    current_owner = env.state.current_turn_owner_id
                    if side_uids[tside] != current_owner:
                        err = (f"turn {nid}: side '{tside}' (uid {side_uids[tside]}) is not "
                               f"current owner (uid {current_owner}). Author must end_turn "
                               f"explicitly to pass the turn.")
                        frames.append(make_frame(
                            snapshot=_snap(), sound_events=[],
                            display_ms=max(int(t.get("intro_ms", 0) or 0), 800),
                            action_kind="turn", turn_id=nid, node_id=nid, error=err,
                        ))
                        break
                intro = int(t.get("intro_ms", 0) or 0)
                if intro > 0:
                    frames.append(make_frame(
                        snapshot=_snap(), sound_events=[],
                        display_ms=intro, action_kind="turn_intro",
                        turn_id=nid, node_id=nid,
                    ))
                last_turn_id = nid

            elif kind == "action":
                action_spec = node.get("action") or {}
                atype = action_spec.get("type")
                side = node.get("side")
                side_uid = side_uids[side]
                current_owner = env.state.current_turn_owner_id
                if side_uid != current_owner:
                    err = (f"node {nid}: side '{side}' (uid {side_uid}) is not current turn "
                           f"owner (uid {current_owner}). Forgot end_turn?")
                    frames.append(make_frame(
                        snapshot=_snap(), sound_events=[],
                        display_ms=max(int(action_spec.get("delay_ms", 0) or 0), 800),
                        action_kind=str(atype), turn_id=last_turn_id or "__no_turn__",
                        node_id=nid, error=err,
                    ))
                    break

                action_obj = _node_to_action(engine, side_uid, action_spec)
                result = engine.apply_action(side_uid, action_obj)
                delay = int(action_spec.get("delay_ms", 0) or 0)

                if not result.get("ok"):
                    err = f"node {nid}: action '{atype}' failed: {result.get('error')}"
                    frames.append(make_frame(
                        snapshot=result.get("snapshot") or _snap(),
                        sound_events=[],
                        display_ms=max(delay, 800),
                        action_kind=str(result.get("action_kind") or atype),
                        turn_id=last_turn_id or "__no_turn__", node_id=nid, error=err,
                    ))
                    break  # fail-fast

                frames.append(make_frame(
                    snapshot=result["snapshot"],
                    sound_events=result.get("sound_events", []),
                    display_ms=delay,
                    action_kind=str(result.get("action_kind") or atype),
                    turn_id=last_turn_id or "__no_turn__", node_id=nid,
                ))

                if result.get("game_over"):
                    frames.append(make_frame(
                        snapshot=result["snapshot"], sound_events=[],
                        display_ms=0, action_kind="game_over",
                        turn_id=last_turn_id or "__no_turn__", node_id=nid,
                    ))
                    break  # стоп traversal

            cur = outgoing.get(cur)
    except ScenarioError as exc:
        err = str(exc)
    finally:
        _effects.random = orig_random

    return _result(frames, viewer_uid, side_uids, match_id, err)


def validate_scenario_graph(scenario: Dict[str, Any], catalog: Optional[CardsCatalog] = None) -> Dict[str, Any]:
    catalog = catalog or CardsCatalog()
    struct = validate_graph_structure(scenario)
    if not struct["ok"]:
        return {"ok": False, "error": struct["error"], "frame_count": 0, "total_ms": 0}
    try:
        res = run_scenario_graph(scenario, catalog)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "frame_count": 0, "total_ms": 0}
    return {
        "ok": not res.get("error"),
        "error": res.get("error"),
        "frame_count": res.get("frame_count", 0),
        "total_ms": res.get("total_ms", 0),
    }


# ---------------------------------------------------------------------------
# Dispatch (v1 ↔ v2 по schema)
# ---------------------------------------------------------------------------

def run_scenario_dispatch(scenario: Dict[str, Any], catalog: Optional[CardsCatalog] = None) -> Dict[str, Any]:
    from .scenario_engine import run_scenario as _run_v1
    if scenario.get("schema") == V2:
        return run_scenario_graph(scenario, catalog)
    return _run_v1(scenario, catalog)


def validate_scenario_dispatch(scenario: Dict[str, Any], catalog: Optional[CardsCatalog] = None) -> Dict[str, Any]:
    from .scenario_engine import validate_scenario as _val_v1
    if scenario.get("schema") == V2:
        return validate_scenario_graph(scenario, catalog)
    return _val_v1(scenario, catalog)


# ---------------------------------------------------------------------------
# v1 → v2 migration
# ---------------------------------------------------------------------------

def migrate_v1_to_v2(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Преобразовать v1 (turns[]) → v2 graph. Детерминированная auto-layout."""
    if scenario.get("schema") == V2:
        return dict(scenario)
    init = scenario.get("init_scene") or {}
    out = {k: v for k, v in scenario.items() if k not in ("init_scene", "turns")}
    out["schema"] = V2

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    layout: Dict[str, Dict[str, int]] = {}

    NODE_GAP = 220
    x = 40
    y = 90

    def place(nid: str) -> None:
        nonlocal x
        layout[nid] = {"x": x, "y": y}
        x += NODE_GAP

    # init node
    init_id = "s0"
    nodes.append({
        "id": init_id, "kind": "scene",
        "scene": {
            "type": "init",
            "turn_number": init.get("turn_number", 1),
            "starting_side": init.get("starting_side", "p1"),
            "display_ms": init.get("display_ms") or scenario.get("init_intro_ms") or 1200,
            "p1": init.get("p1") or {},
            "p2": init.get("p2") or {},
        },
    })
    place(init_id)

    prev = init_id
    edge_seq = 0
    turn_seq = 0
    node_seq = 0

    def chain(nid: str) -> None:
        nonlocal prev, edge_seq
        edge_seq += 1
        edges.append({"id": f"e{edge_seq}", "from": prev, "to": nid})
        prev = nid

    for turn in scenario.get("turns", []) or []:
        turn_seq += 1
        tside = turn.get("side", "p1")
        nodes_list = turn.get("nodes", []) or []
        explicit_end_turn = any((nn.get("type") == "end_turn") for nn in nodes_list)
        if not nodes_list:
            # пустой ход → hold на duration_ms (turn-маркер не эмитим, чтоб не добавить кадр)
            node_seq += 1
            nid = f"n{node_seq}"
            nodes.append({"id": nid, "kind": "scene",
                          "scene": {"type": "hold", "display_ms": int(turn.get("duration_ms", 600))}})
            place(nid)
            chain(nid)
        else:
            for nn in nodes_list:
                node_seq += 1
                nid = f"n{node_seq}"
                ntype = nn.get("type")
                if ntype == "wait":
                    nodes.append({"id": nid, "kind": "scene",
                                  "scene": {"type": "hold", "display_ms": int(nn.get("delay_ms", 600))}})
                else:
                    action = {k: v for k, v in nn.items() if k not in ("id",)}
                    nodes.append({"id": nid, "kind": "action", "side": tside, "action": action})
                place(nid)
                chain(nid)

        # end_with_end_turn=true БЕЗ явного end_turn-узла → добавляем явно
        # (в т.ч. для пустого хода — иначе потерялся бы переход хода)
        if turn.get("end_with_end_turn") and not explicit_end_turn:
            node_seq += 1
            nid = f"n{node_seq}"
            delay = int(turn.get("end_turn_ms", 500))
            nodes.append({"id": nid, "kind": "action", "side": tside,
                          "action": {"type": "end_turn", "delay_ms": delay}})
            place(nid)
            chain(nid)

    out["graph"] = {"start": init_id, "nodes": nodes, "edges": edges}
    out["layout"] = layout
    return out


__all__ = [
    "V2",
    "validate_graph_structure",
    "run_scenario_graph",
    "validate_scenario_graph",
    "run_scenario_dispatch",
    "validate_scenario_dispatch",
    "migrate_v1_to_v2",
]