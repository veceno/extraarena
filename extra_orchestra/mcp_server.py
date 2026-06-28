"""MCP-сервер ExtraOrchestra (stdio) — граф-сценарии для агентов.

Тонкий клиент к HTTP-серверу оркестра (порт 8095, ``python -m extra_orchestra.server``).
Авторинг — в терминах v2 graph DSL (``graph={start,nodes[],edges[]}``); раннер и
браузерная арена живут в HTTP-сервере, MCP только координирует.

Инструменты (4 группы, как просил пользователь):

  Загрузка:
    list_scenarios()                 — список сохранённых сценариев
    get_scenario(name, as_v2=true)   — загрузить (v1 авто-мигрируется в v2)
    migrate_v1_to_v2(scenario)       — явная миграция v1→v2

  Создание (графами):
    create_blank_scenario(meta?)     — пустой v2-граф с init-узлом (шаблон)
    build_graph(spec)                — собрать v2 из декларативных nodes/edges
                                        (init из spec.init_scene ИЛИ base_scenario_name),
                                        validates, возвращает сценарий + валидацию
    save_scenario(scenario)          — сохранить в store
    delete_scenario(name)            — удалить

  Предпросмотр:
    preview_frames(scenario)         — прогон → покадровые «итоговые сцены»
                                        (структурированные снапшоты p1/p2 board+hero+hp
                                        на каждом шаге графа) + run_id + arena-URL.
                                        НЕ требует vision — данные для рассуждения;
                                        vision/omni-хост может рендерить сам.
    validate_scenario(scenario)      — structure + dry-run → ok/error/frames/ms

  Экспорт:
    export_mp4(scenario, wait, timeout, inline) — записать mp4 (видео+звук арены).
                                          wait=true (по умолч.) блокирует до готовности;
                                          inline=true — отдать БАЙТЫ файла инлайн как MCP
                                          resource-content (иначе только download_url).
    export_gif(scenario, wait, timeout, inline) — то же в GIF (image-content, omni-клиенты
                                          показывают инлайн; GIF без звука).
    get_record_status(job_id)           — статус задания записи (метаданные, без байтов)
    get_record_file(job_id)             — ДОСТАТЬ ФАЙЛ: готовые байты mp4/gif инлайн как
                                          MCP content (image для gif / resource-blob для mp4)
                                          + метаданные. Это способ получить сам файл, а не URL.
    list_cards(filter?)                 — каталог карт (статы + mechanics + image)
    list_cosmetics()                    — аватары/фоны

Запуск:
    python -m extra_orchestra.mcp_server
    python -m extra_orchestra.mcp_server --base-url http://127.0.0.1:8095 --auto-start
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

V2 = "extra_orchestra.scenario.v2"
DEFAULT_BASE = "http://127.0.0.1:8095"


# ============================================================================
# OrchestraClient — HTTP-клиент к серверу оркестра
# ============================================================================

class OrchestraClient:
    def __init__(self, base_url: str, session: aiohttp.ClientSession):
        self.base = base_url.rstrip("/")
        self.session = session

    async def _req(self, method: str, path: str, **kw) -> Any:
        url = self.base + path
        timeout = aiohttp.ClientTimeout(total=kw.pop("timeout", 60))
        async with self.session.request(method, url, timeout=timeout, **kw) as r:
            text = await r.text()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": f"non-json {r.status}: {text[:200]}", "status": r.status}

    async def health(self) -> bool:
        try:
            d = await self._req("GET", "/health", timeout=5)
            return d.get("status") == "ok"
        except Exception:  # noqa: BLE001
            return False

    # --- scenarios ---
    async def list_scenarios(self) -> Dict[str, Any]:
        return await self._req("GET", "/api/orchestra/scenarios")

    async def get_scenario(self, name: str) -> Dict[str, Any]:
        return await self._req("GET", "/api/orchestra/scenarios/" + name)

    async def save_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return await self._req("POST", "/api/orchestra/scenarios", json=scenario)

    async def delete_scenario(self, name: str) -> Dict[str, Any]:
        return await self._req("DELETE", "/api/orchestra/scenarios/" + name)

    async def migrate(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return await self._req("POST", "/api/orchestra/migrate-v1", json=scenario)

    async def validate(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return await self._req("POST", "/api/orchestra/validate", json=scenario)

    async def compute_frames(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return await self._req("POST", "/api/orchestra/compute-frames", json=scenario, timeout=120)

    async def get_frames(self, run_id: str) -> Dict[str, Any]:
        return await self._req("GET", f"/api/orchestra/frames/{run_id}")

    async def record(self, scenario: Dict[str, Any], fmt: str = "mp4") -> Dict[str, Any]:
        return await self._req("POST", f"/api/orchestra/record?format={fmt}", json=scenario, timeout=120)

    async def record_status(self, job_id: str) -> Dict[str, Any]:
        return await self._req("GET", f"/api/orchestra/record/{job_id}")

    async def record_download(self, job_id: str) -> tuple:
        """Скачать готовый файл записи (mp4/gif) → ``(bytes, content_type)``.

        Возвращает ``(None, error_str)`` при ошибке. Байты нужны, чтобы отдать
        файл агенту **инлайн** через MCP: ``download_url`` — это localhost-ссылка,
        MCP-клиент в общем случае не умеет делать HTTP-запросы и не может её
        fetch'ить — без этого метода агент получил бы только URL, а не сам файл.
        """
        url = self.base + f"/api/orchestra/record/{job_id}/download"
        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as r:
            if r.status != 200:
                text = await r.text()
                return None, f"download failed (status={r.status}): {text[:200]}"
            data = await r.read()
            ctype = r.headers.get("Content-Type", "application/octet-stream")
            return data, ctype

    async def cards(self) -> Dict[str, Any]:
        return await self._req("GET", "/api/orchestra/cards")

    async def cosmetics(self) -> Dict[str, Any]:
        return await self._req("GET", "/api/orchestra/cosmetics")


# ============================================================================
# Локальные помощники (pure, без HTTP) — build_graph / blank / summary
# ============================================================================

def _default_init_scene(turn_number: int = 1, starting_side: str = "p1",
                        p1_name: str = "Демо", p2_name: str = "Оппонент",
                        p1_hero_id: int = 1, p2_hero_id: int = 3) -> Dict[str, Any]:
    return {
        "type": "init", "turn_number": turn_number, "starting_side": starting_side,
        "display_ms": 2000,
        "p1": {"user_id": 1001, "nickname": p1_name, "title": "", "rarity": "common",
               "mana": 6, "max_mana": 6, "hero": {"card_id": p1_hero_id, "level": 1},
               "avatar_url": "/DesignAssets/PlayerCosmetics/Avatars/1.png",
               "background_url": "/DesignAssets/PlayerCosmetics/Background/7.png",
               "hand": [], "board": [], "deck": []},
        "p2": {"user_id": 2002, "is_bot": True, "nickname": p2_name, "title": "",
               "rarity": "epic", "mana": 6, "max_mana": 6,
               "hero": {"card_id": p2_hero_id, "level": 1},
               "avatar_url": "/DesignAssets/PlayerCosmetics/Avatars/2.png",
               "background_url": "/DesignAssets/PlayerCosmetics/Background/3.png",
               "hand": [], "board": [], "deck": []},
    }


def create_blank_scenario(meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    m = meta or {}
    init_scene = _default_init_scene(
        turn_number=int(m.get("turn_number", 1)),
        starting_side=m.get("starting_side", "p1"),
        p1_name=m.get("p1_name", "Демо"), p2_name=m.get("p2_name", "Оппонент"),
        p1_hero_id=int(m.get("p1_hero_id", 1)), p2_hero_id=int(m.get("p2_hero_id", 3)),
    )
    return {
        "schema": V2, "name": m.get("name", "Новый сценарий"),
        "seed": int(m.get("seed", 42)), "viewer_side": m.get("viewer_side", "p1"),
        "match_id": m.get("match_id", "new-scenario"),
        "classic_params": {"sudden_death_enabled": False, "mana_per_turn": 1,
                           "turn_duration_seconds": 25},
        "graph": {"start": "s0", "nodes": [{"id": "s0", "kind": "scene", "scene": init_scene}],
                  "edges": []},
        "layout": {"s0": {"x": 60, "y": 200}}, "editor": {"zoom": 1},
    }


def build_graph(spec: Dict[str, Any], client: Optional[OrchestraClient] = None) -> Dict[str, Any]:
    """Собрать v2-сценарий из декларативного spec (pure; валидация — отдельно).

    spec:
      name?, seed?, viewer_side?, match_id?, classic_params?
      init_scene?            — полная init-сцена (p1/p2/hero/hand/board/deck/...);
                               если нет — default init. Унаследовать init+meta из
                               сохранённого сценария можно через base_scenario_name,
                               но это делает _tool-обработчик (load+migrate+inject),
                               НЕ эта pure-функция.
      nodes: [ {id?, kind, ...} ]   — kind ∈ {scene, turn, action}
      edges: [ {from, to} ]?        — если нет → авто-цепочка s0→n1→…

    Узлы принимают И плоскую форму (поля на узле), И каноническую v2-форму
    (поля вложены в `turn`/`action`) — как editor.js и сохранённые сценарии.
    """
    nodes_spec = spec.get("nodes", []) or []
    edges_spec = spec.get("edges", []) or []

    if spec.get("base_scenario_name") and not spec.get("init_scene"):
        # чистая функция ничего не грузит; _tool-обработчик подставит init_scene.
        # Если сюда дошли с base_scenario_name и без init_scene — это баг вызывающего.
        raise ValueError("build_graph: base_scenario_name требует pre-injected init_scene "
                         "(вызывающий должен загрузить base и передать spec.init_scene)")

    if spec.get("init_scene"):
        init_scene = dict(spec["init_scene"])
        init_scene["type"] = "init"
    else:
        init_scene = _default_init_scene()
    classic = spec.get("classic_params") or {"sudden_death_enabled": False, "mana_per_turn": 1,
                                             "turn_duration_seconds": 25}

    init_node = {"id": "s0", "kind": "scene", "scene": init_scene}
    built_nodes: List[Dict[str, Any]] = [init_node]
    used_ids = {"s0"}
    for i, ns in enumerate(nodes_spec):
        nid = ns.get("id")
        if not nid:
            j = i + 1
            while f"n{j}" in used_ids:
                j += 1
            nid = f"n{j}"
        if nid in used_ids:
            raise ValueError(f"duplicate node id: {nid}")
        used_ids.add(nid)
        kind = ns.get("kind")
        if kind == "scene" or ns.get("type") == "hold" or ns.get("scene_type") == "hold":
            sc = ns.get("scene") or {}
            built_nodes.append({"id": nid, "kind": "scene",
                                "scene": {"type": "hold",
                                          "display_ms": int(ns.get("display_ms",
                                                                   sc.get("display_ms", 600)))}})
        elif kind == "turn":
            t = ns.get("turn") or {}
            built_nodes.append({"id": nid, "kind": "turn",
                                "turn": {"side": ns.get("side", t.get("side")),
                                         "intro_ms": int(ns.get("intro_ms", t.get("intro_ms", 0)))}})
        elif kind == "action":
            # плоская форма (поля на узле) И каноническая (вложены в `action`)
            nested = ns.get("action") if isinstance(ns.get("action"), dict) else {}
            atype = (ns.get("type") or ns.get("action_type")
                     or nested.get("type"))
            action = {"type": atype,
                      "delay_ms": int(ns.get("delay_ms", nested.get("delay_ms", 500)))}
            for k in ("hand_index", "target_id", "target_index", "target_is_hero", "position",
                      "attacker_id", "attacker_index"):
                if k in ns:
                    action[k] = ns[k]
                elif k in nested:
                    action[k] = nested[k]
            # side обязательна (description требует); None → validate_graph_structure
            # поймает «action node '<nid>' missing side» на этапе authoring, не в runtime
            built_nodes.append({"id": nid, "kind": "action",
                                "side": ns.get("side"), "action": action})
        else:
            raise ValueError(f"node {nid}: unknown kind '{kind}' (use scene|turn|action)")

    # edges
    built_edges: List[Dict[str, Any]] = []
    if edges_spec:
        for i, e in enumerate(edges_spec):
            src = e.get("from")
            dst = e.get("to")
            if src is None or dst is None:
                raise ValueError(f"edge {i + 1}: 'from' and 'to' are required "
                                 f"(got from={src!r}, to={dst!r})")
            built_edges.append({"id": f"e{i + 1}", "from": src, "to": dst})
    else:
        # авто-цепочка s0 → built_nodes[1] → built_nodes[2] → … (по уже-built id)
        prev = "s0"
        for i, edge in enumerate(built_nodes[1:], start=1):
            built_edges.append({"id": f"e{i}", "from": prev, "to": edge["id"]})
            prev = edge["id"]

    layout = {"s0": {"x": 60, "y": 200}}
    x = 290
    for n in built_nodes[1:]:
        layout[n["id"]] = {"x": x, "y": 200}
        x += 230

    return {
        "schema": V2, "name": spec.get("name") or "Новый сценарий",
        "seed": int(spec.get("seed", 42)), "viewer_side": spec.get("viewer_side", "p1"),
        "match_id": spec.get("match_id", "new-scenario"),
        "classic_params": classic,
        "graph": {"start": "s0", "nodes": built_nodes, "edges": built_edges},
        "layout": layout, "editor": {"zoom": 1},
    }


def _summarize_card(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instance_id": c.get("instance_id"), "card_id": c.get("card_id"),
        "name": c.get("name"), "level": c.get("level"),
        "cost": c.get("mana_cost", c.get("mana")), "attack": c.get("attack"),
        "hp": c.get("hp"), "max_hp": c.get("max_hp"),
        "mechanics": c.get("mechanics", []), "is_ready": c.get("is_ready"),
        "is_frozen": c.get("is_frozen"),
    }


def _summarize_side(side_obj: Dict[str, Any]) -> Dict[str, Any]:
    hand = side_obj.get("hand", []) or []
    hand_hidden = bool(hand) and all(c.get("hidden") for c in hand)
    return {
        "user_id": side_obj.get("user_id"), "name": side_obj.get("name"),
        "mana": side_obj.get("mana"), "max_mana": side_obj.get("max_mana"),
        "hand_count": side_obj.get("hand_count", len(hand)),
        "hand_hidden": hand_hidden,
        # card_id есть только для своей руки; чужая скрыта (entries={hidden:True})
        "hand_card_ids": [c.get("card_id") for c in hand if c.get("card_id") is not None],
        "board": [_summarize_card(c) for c in side_obj.get("board", [])],
        "hero": {"card_id": side_obj.get("hero", {}).get("card_id"),
                 "name": side_obj.get("hero", {}).get("name"),
                 "hp": side_obj.get("hero", {}).get("hp"),
                 "max_hp": side_obj.get("hero", {}).get("max_hp")},
    }


def summarize_frames(run: Dict[str, Any]) -> Dict[str, Any]:
    """Покадровые «итоговые сцены» — structured, без vision."""
    side_uids = run.get("side_uids") or {}
    p1_uid = side_uids.get("p1")
    p2_uid = side_uids.get("p2")
    viewer_uid = run.get("viewer_uid")
    viewer_side = next((s for s, u in side_uids.items() if u == viewer_uid), None)
    frames = run.get("frames") or []
    out = []
    for idx, f in enumerate(frames):
        s = f.get("snapshot") or {}
        player = s.get("player") or {}
        opponent = s.get("opponent") or {}
        # map viewer-relative player/opponent → p1/p2 по user_id
        if player.get("user_id") == p1_uid:
            p1, p2 = player, opponent
        else:
            p1, p2 = opponent, player
        out.append({
            "idx": idx, "node_id": f.get("node_id"), "action_kind": f.get("action_kind"),
            "turn_id": f.get("turn_id"), "display_ms": f.get("display_ms"),
            "turn": s.get("turn"), "current_player_id": s.get("current_player_id"),
            "p1": _summarize_side(p1), "p2": _summarize_side(p2),
            "error": f.get("error"),
        })
    return {
        "run_id": run.get("run_id"), "frame_count": run.get("frame_count"),
        "total_ms": run.get("total_ms"), "viewer_uid": viewer_uid,
        "viewer_side": viewer_side,
        "side_uids": side_uids, "error": run.get("error"),
        "frames": out,
    }


# ============================================================================
# inline-файл: MCP content-item с байтами записи (чтобы агент ПОЛУЧИЛ файл)
# ============================================================================

def _mime_for(file_name: str, content_type: str) -> str:
    """Определить MIME файла записи по имени/Content-Type."""
    name = (file_name or "").lower()
    ct = (content_type or "").lower()
    if name.endswith(".gif") or "gif" in ct:
        return "image/gif"
    if name.endswith(".mp4") or "mp4" in ct:
        return "video/mp4"
    return (content_type or "").split(";")[0].strip() or "application/octet-stream"


def _file_content_item(data: bytes, mime: str, download_url: str) -> Dict[str, Any]:
    """MCP content-item с байтами файла.

    GIF → ``ImageContent`` (``{type:image, data:base64, mimeType}``): omni/vision-
    клиенты (Claude Code) показывают его инлайн, и байты доступны агенту.
    mp4 → ``ResourceContent`` c ``blob`` (``{type:resource, resource:{uri, mimeType, blob}}``):
    видео не отображается, но агент получает base64-байты и может их декодировать/сохранить.
    """
    blob = base64.b64encode(data).decode("ascii")
    if mime.startswith("image/"):
        return {"type": "image", "data": blob, "mimeType": mime}
    return {"type": "resource",
            "resource": {"uri": download_url, "mimeType": mime, "blob": blob}}


async def _attach_inline_file(meta: Dict[str, Any], client: "OrchestraClient",
                              job_id: str, download_url: str) -> None:
    """Скачать файл записи и дописать в ``meta`` инлайн-поля + ``_content`` (in-place).

    При ошибке скачивания ставит ``meta['inline_error']`` (без ``_content``) —
    вызывающий решает, как обработать (export вернёт метаданные без инлайна,
    get_record_file вернёт error).
    """
    data, ctype = await client.record_download(job_id)
    if data is None:
        meta["inline_error"] = ctype
        return
    fname = meta.get("file_name") or ""
    mime = _mime_for(fname, ctype)
    meta["inline"] = True
    meta["size_bytes"] = len(data)
    meta["mime_type"] = mime
    meta["_content"] = [_file_content_item(data, mime, download_url)]


# ============================================================================
# MCPServer
# ============================================================================

class MCPServer:
    def __init__(self, client: OrchestraClient, base_url: str):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.tools = self._build_tools()

    # ------------------------------------------------------------------
    def _build_tools(self) -> List[Dict[str, Any]]:
        return [
            # --- Загрузка ---
            {"name": "list_scenarios",
             "description": "Список сохранённых сценариев (name, schema, file, nodes/turns).",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "get_scenario",
             "description": "Загрузить сценарий по имени файла. as_v2=true (по умолчанию) авто-мигрирует v1→v2.",
             "inputSchema": {"type": "object",
                             "properties": {"name": {"type": "string"},
                                            "as_v2": {"type": "boolean", "default": True}},
                             "required": ["name"]}},
            {"name": "migrate_v1_to_v2",
             "description": "Явная миграция v1 (init_scene+turns) → v2 graph. Возвращает v2-сценарий.",
             "inputSchema": {"type": "object",
                             "properties": {"scenario": {"type": "object"}},
                             "required": ["scenario"]}},
            # --- Создание ---
            {"name": "create_blank_scenario",
             "description": "Пустой v2-граф с init-узлом (шаблон для построения нового сценария). meta: name/seed/viewer_side/match_id/turn_number/starting_side/p1_name/p2_name/p1_hero_id/p2_hero_id.",
             "inputSchema": {"type": "object",
                             "properties": {"meta": {"type": "object"}},
                             "required": []}},
            {"name": "build_graph",
             "description": (
                 "Собрать v2-сценарий из декларативного graph-spec. "
                 "spec: {name?, seed?, viewer_side?, match_id?, classic_params?, "
                 "init_scene? ИЛИ base_scenario_name?, nodes:[...], edges?:[{from,to}]}. "
                 "nodes: scene/hold → {id, kind:'scene', display_ms}; "
                 "turn → {id, kind:'turn', side?, intro_ms?}; "
                 "action → {id, kind:'action', side:'p1'|'p2', type:'play_card'|'attack'|'mana_draw'|'end_turn', delay_ms?, hand_index?, target_id?, target_index?, target_is_hero?, position?, attacker_id?, attacker_index?}. "
                 "edges необязательны (без них авто-цепочка s0→n1→…). "
                 "Возвращает {scenario, validation}. init можно унаследовать из сохранённого сценария через base_scenario_name."
             ),
             "inputSchema": {"type": "object",
                             "properties": {"spec": {"type": "object"}}, "required": ["spec"]}},
            {"name": "save_scenario",
             "description": "Сохранить v2-сценарий в store (scenarios/*.json). Возвращает {ok, file}.",
             "inputSchema": {"type": "object",
                             "properties": {"scenario": {"type": "object"}}, "required": ["scenario"]}},
            {"name": "delete_scenario",
             "description": "Удалить сценарий по имени файла.",
             "inputSchema": {"type": "object",
                             "properties": {"name": {"type": "string"}}, "required": ["name"]}},
            {"name": "validate_scenario",
             "description": "Структурная валидация графа + dry-run → {ok, error, frame_count, total_ms}.",
             "inputSchema": {"type": "object",
                             "properties": {"scenario": {"type": "object"}}, "required": ["scenario"]}},
            # --- Предпросмотр ---
            {"name": "preview_frames",
             "description": (
                 "Прогнать граф → покадровые «итоговые сцены»: на каждом шаге графа "
                 "структурированный снапшот (p1/p2 board с card_id/name/attack/hp/mechanics/is_ready, "
                 "hero hp, mana, hand_count+hand_hidden, turn, current_player_id, action_kind, display_ms). "
                 "Не требует vision — данные для рассуждения; vision/omni-хост может рендерить сам. "
                 "Возвращает {run_id, frame_count, total_ms, viewer_side, frames:[...], preview_arena_url}."
             ),
             "inputSchema": {"type": "object",
                             "properties": {"scenario": {"type": "object"}}, "required": ["scenario"]}},
            {"name": "get_frames",
             "description": (
                 "Получить кадры ранее созданного прогона по run_id (из preview_frames/export_mp4). "
                 "summary=true (по умолч.) — сокращённые «итоговые сцены» (как preview_frames, без URL); "
                 "summary=false — raw run-документ {run_id, frame_count, total_ms, frames:[...], side_uids, ...}."
             ),
             "inputSchema": {"type": "object",
                             "properties": {"run_id": {"type": "string"},
                                            "summary": {"type": "boolean", "default": True}},
                             "required": ["run_id"]}},
            # --- Экспорт ---
            {"name": "export_mp4",
             "description": (
                 "Записать mp4 (видео + SFX + фоновая музыка арены) через Playwright+ffmpeg. "
                 "Мобильный портретный mp4 (828×1792 @ device_scale_factor 2), высокое качество "
                 "(crf 10, preset slow). wait=true (по умолчанию) — блокирует до готовности "
                 "(timeout, по умолч. 180с); wait=false — сразу вернуть job_id для поллинга через "
                 "get_record_status. inline=true — дополнительно отдать БАЙТЫ mp4 инлайн как MCP "
                 "resource-content (иначе только download_url; чтобы гарантированно получить файл "
                 "в output вызова — поставь inline=true либо позови get_record_file(job_id)). "
                 "Возвращает {job_id, status, format, file_name?, download_url, size_bytes?, "
                 "mime_type?, inline?}."
             ),
             "inputSchema": {"type": "object",
                             "properties": {"scenario": {"type": "object"},
                                            "wait": {"type": "boolean", "default": True},
                                            "timeout": {"type": "integer", "default": 180},
                                            "inline": {"type": "boolean", "default": False}},
                             "required": ["scenario"]}},
            {"name": "export_gif",
             "description": (
                 "Записать GIF (двухпроходный palette: palettegen+paletteuse) через "
                 "Playwright+ffmpeg. Мобильный портрет (по умолч. ширина 540, fps 15 — см. "
                 "config.json recording.gif_width/gif_fps). GIF НЕ содержит звука (формат не "
                 "поддерживает) — чисто визуальный экспорт. wait=true (по умолчанию) — блокирует "
                 "до готовности (timeout, по умолч. 180с); wait=false — сразу вернуть job_id для "
                 "поллинга через get_record_status. inline=true — отдать БАЙТЫ GIF инлайн как MCP "
                 "image-content (omni/vision-клиенты показывают инлайн). Возвращает {job_id, "
                 "status, format:'gif', file_name?, download_url, size_bytes?, mime_type?, inline?}."
             ),
             "inputSchema": {"type": "object",
                             "properties": {"scenario": {"type": "object"},
                                            "wait": {"type": "boolean", "default": True},
                                            "timeout": {"type": "integer", "default": 180},
                                            "inline": {"type": "boolean", "default": False}},
                             "required": ["scenario"]}},
            {"name": "get_record_status",
             "description": "Статус задания записи (mp4/gif) → {status, format, file_name?, download_url?, error?}. Метаданные WITHOUT файла; чтобы получить сами байты — get_record_file(job_id).",
             "inputSchema": {"type": "object",
                             "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
            {"name": "get_record_file",
             "description": (
                 "Достать готовый файл записи (mp4/gif) инлайн — отдать БАЙТЫ как MCP content: "
                 "gif → image-content (omni/vision-клиенты рендерят инлайн), mp4 → resource-blob "
                 "(байты в base64, агент декодирует/сохраняет). Это способ ПОЛУЧИТЬ файл в output "
                 "вызова, а не только download_url (localhost-ссылку, которую MCP-клиент не может "
                 "fetch'ить). Требует status=done (иначе {error,...}). Возвращает {job_id, format, "
                 "file_name, size_bytes, mime_type, download_url, inline:true} + content-item."
             ),
             "inputSchema": {"type": "object",
                             "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
            {"name": "list_cards",
             "description": "Каталог карт (id, name, card_type, rarity, mana_cost, base_attack/hp, mechanics, image). filter — по имени/id.",
             "inputSchema": {"type": "object",
                             "properties": {"filter": {"type": "string"}}, "required": []}},
            {"name": "list_cosmetics",
             "description": "Аватары и фоны профиля (url + name).",
             "inputSchema": {"type": "object", "properties": {}}},
        ]

    # ------------------------------------------------------------------
    async def _tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self.client
        if name == "list_scenarios":
            return await c.list_scenarios()
        if name == "get_scenario":
            sc = await c.get_scenario(args["name"])
            if sc.get("error"):
                return sc
            if args.get("as_v2", True) and sc.get("schema") != V2:
                migrated = await c.migrate(sc)
                if migrated.get("error"):
                    return {"error": f"v1→v2 migration failed: {migrated['error']}", "v1": sc}
                return migrated
            return sc
        if name == "migrate_v1_to_v2":
            return await c.migrate(args["scenario"])
        if name == "create_blank_scenario":
            return create_blank_scenario(args.get("meta"))
        if name == "build_graph":
            spec = dict(args["spec"])
            base_name = spec.get("base_scenario_name")
            if base_name:
                # всегда наследовать meta из base (даже если есть init_scene);
                # init_scene из base — только если свой не задан.
                base = await c.get_scenario(base_name)
                if base.get("error"):
                    return {"error": f"base_scenario_name '{base_name}' not loaded: {base['error']}"}
                if base.get("schema") != V2:
                    base = await c.migrate(base)
                    if base.get("error"):
                        return {"error": f"base migration failed: {base['error']}"}
                if not spec.get("init_scene"):
                    init_node = next((n for n in base["graph"]["nodes"]
                                      if n["kind"] == "scene" and n["scene"]["type"] == "init"),
                                     None)
                    if init_node:
                        spec["init_scene"] = init_node["scene"]
                # `or`-fallback: base может не иметь поля (v1-migrated) → не травим None
                spec.setdefault("classic_params", base.get("classic_params")
                                or {"sudden_death_enabled": False, "mana_per_turn": 1,
                                    "turn_duration_seconds": 25})
                spec.setdefault("name", base.get("name") or "Новый сценарий")
                spec.setdefault("seed", base.get("seed", 42))
                spec.setdefault("match_id", base.get("match_id", "new-scenario"))
                spec.setdefault("viewer_side", base.get("viewer_side", "p1"))
            scenario = build_graph(spec)
            validation = await c.validate(scenario)
            return {"scenario": scenario, "validation": validation}
        if name == "save_scenario":
            return await c.save_scenario(args["scenario"])
        if name == "delete_scenario":
            return await c.delete_scenario(args["name"])
        if name == "validate_scenario":
            return await c.validate(args["scenario"])
        if name == "preview_frames":
            comp = await c.compute_frames(args["scenario"])
            if comp.get("error"):
                return comp
            run = await c.get_frames(comp["run_id"])
            if run.get("error"):
                return run
            summary = summarize_frames(run)
            auth = comp.get("auth", "")
            summary["preview_arena_url"] = (
                f"{self.base_url}/preview?id={comp['run_id']}&autoplay=1&_auth={auth}")
            return summary
        if name == "get_frames":
            run = await c.get_frames(args["run_id"])
            if run.get("error"):
                return run
            if args.get("summary", True):
                return summarize_frames(run)
            return run
        if name in ("export_mp4", "export_gif"):
            fmt = "gif" if name == "export_gif" else "mp4"
            rec = await c.record(args["scenario"], fmt=fmt)
            if rec.get("error"):
                return rec
            job_id = rec["job_id"]
            download_url = f"{self.base_url}/api/orchestra/record/{job_id}/download"
            if not args.get("wait", True):
                return {"job_id": job_id, "status": rec.get("status", "pending"),
                        "format": fmt, "download_url": download_url}
            timeout = int(args.get("timeout", 180))
            want_inline = bool(args.get("inline", False))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while loop.time() < deadline:
                st = await c.record_status(job_id)
                if st.get("status") in ("done", "failed"):
                    st["job_id"] = job_id
                    st["format"] = st.get("format", fmt)
                    st["download_url"] = download_url
                    if st.get("status") == "done" and want_inline:
                        await _attach_inline_file(st, c, job_id, download_url)
                    return st
                if st.get("error") or st.get("status") is None:
                    # lost/unknown job (server 404 → {error:...}, без status) — не висеть
                    st["job_id"] = job_id
                    st["format"] = fmt
                    st["download_url"] = download_url
                    return st
                await asyncio.sleep(2.0)
            return {"job_id": job_id, "status": "timeout", "format": fmt,
                    "download_url": download_url, "hint": "poll with get_record_status"}
        if name == "get_record_status":
            st = await c.record_status(args["job_id"])
            st["download_url"] = f"{self.base_url}/api/orchestra/record/{args['job_id']}/download"
            return st
        if name == "get_record_file":
            job_id = args["job_id"]
            download_url = f"{self.base_url}/api/orchestra/record/{job_id}/download"
            st = await c.record_status(job_id)
            if st.get("error") or st.get("status") != "done":
                return {"error": f"file not ready: status={st.get('status')!r} "
                                 f"err={st.get('error')}",
                        "job_id": job_id, "download_url": download_url}
            meta = {"job_id": job_id, "format": st.get("format", "mp4"),
                    "file_name": st.get("file_name") or st.get("mp4_name"),
                    "status": "done", "download_url": download_url}
            await _attach_inline_file(meta, c, job_id, download_url)
            if "_content" not in meta:
                # download провалился — вернуть ошибку, но с метаданными
                meta["error"] = meta.pop("inline_error", "download failed")
                return meta
            return meta
        if name == "list_cards":
            d = await c.cards()
            cards = d.get("cards", [])
            flt = (args.get("filter") or "").lower()
            if flt:
                cards = [x for x in cards if flt in (x.get("name") or "").lower()
                         or str(x.get("id")) == flt]
            return {"count": len(cards), "cards": cards}
        if name == "list_cosmetics":
            return await c.cosmetics()
        raise ValueError(f"unknown tool: {name}")

    # ------------------------------------------------------------------
    async def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "initialize":
            return {"protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "extra-orchestra", "version": "1.0"},
                    "capabilities": {"tools": {}}}
        if method == "tools/list":
            return {"tools": self.tools}
        if method == "tools/call":
            # -32602 invalid params: name обязан быть строкой, arguments — объектом
            tname = params.get("name")
            if not isinstance(tname, str) or not tname:
                return {"error": {"code": -32602,
                                  "message": "tools/call requires params.name (non-empty string)"}}
            targs = params.get("arguments", {}) or {}
            if not isinstance(targs, dict):
                return {"error": {"code": -32602,
                                  "message": "tools/call params.arguments must be an object"}}
            try:
                result = await self._tool(tname, targs)
            except KeyError as exc:
                # отсутствует обязательный аргумент → -32602, не криптый KeyError
                return {"error": {"code": -32602,
                                  "message": f"missing required argument: {exc}"}}
            except Exception as exc:  # noqa: BLE001
                logger.exception("[mcp] tool %s failed", tname)
                return {"content": [{"type": "text",
                                     "text": json.dumps({"error": str(exc)}, ensure_ascii=False)}],
                        "isError": True}
            # _tool может отдать бинарный контент инлайн (файл записи mp4/gif) через
            # result["_content"] — список MCP content-items (image/resource). Их
            # НЕ сериализуем в text (иначе base2 задвоится и раздуется) — добавляем
            # к content-массиву отдельными элементами, рядом с text-метаданными.
            extra_content: List[Dict[str, Any]] = []
            if isinstance(result, dict) and isinstance(result.get("_content"), list):
                extra_content = result.pop("_content")
            # _tool возвращает {error:...} при execution-failure (server error,
            # connection, lost job). Это НЕ success — поднимаем isError. Но
            # валидационный результат ({ok:false, error:...}) — это корректный
            # ответ инструмента, его isError=false (агент читает ok=false).
            is_error = (isinstance(result, dict) and bool(result.get("error"))
                        and "ok" not in result)
            content = [{"type": "text",
                        "text": json.dumps(result, ensure_ascii=False, default=str)}]
            content.extend(extra_content)
            return {"content": content, "isError": is_error}
        if method == "ping":
            return {}
        return {"error": {"code": -32601, "message": f"unknown method: {method}"}}


# ============================================================================
# auto-start HTTP-сервера
# ============================================================================

def _port_responds(base_url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _spawn_server(base_url: str) -> Optional[subprocess.Popen]:
    # host/port из base_url — передаём серверу, иначе не-дефолтный порт = health никогда не ответит
    from urllib.parse import urlparse
    u = urlparse(base_url)
    host = u.hostname or "127.0.0.1"
    port = u.port or 8095
    # не спавнить, если порт уже занят (чужой процесс)
    s = socket.socket()
    try:
        s.bind((host, port))
        s.close()
    except OSError:
        logger.error(
            "port %s на %s занят не-orchestra процессом (health не ответил, bind не удался); "
            "auto-start отменён — каждый tool call будет ходить в чужой/пустой порт.", port, host)
        return None
    logf = open(os.environ.get("ORCH_MCP_SERVER_LOG", "/tmp/orchestra_mcp_server.log"), "ab")
    try:
        proc = subprocess.Popen([sys.executable, "-m", "extra_orchestra.server",
                                 "--host", host, "--port", str(port)],
                                cwd=str(_REPO_ROOT), stdout=logf, stderr=logf)
    finally:
        logf.close()  # родительский fd не нужен — child'у достался dup
    # дождаться health (sync-контекст до loop'а → time.sleep, не asyncio.run)
    for _ in range(40):
        if _port_responds(base_url):
            return proc
        time.sleep(0.25)
    # health не пришёл — НЕ возвращать мёртвый proc; убить и сказать main, что бэкенда нет
    logger.error("auto-started orchestra server did not become healthy within 10s — terminating")
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try: proc.kill()
        except Exception: pass  # noqa: BLE001
    return None


# ============================================================================
# stdio loop
# ============================================================================

async def _handle_message(server: "MCPServer", msg: Any) -> Optional[Dict[str, Any]]:
    """Обработать одно JSON-RPC сообщение (dict). Возвращает response-dict ИЛИ
    None (notification / no-response). Не пишет в stdout — это забота вызывающего.
    """
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "invalid request: not an object"}}
    method = msg.get("method", "")
    params = msg.get("params", {})
    if not isinstance(params, dict):
        params = {}
    is_notification = "id" not in msg
    try:
        result = await server.dispatch(method, params)
    except Exception as exc:  # noqa: BLE001
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg.get("id"),
                "error": {"code": -32603, "message": f"internal error: {exc}"}}
    if is_notification:
        return None
    # dispatch может вернуть protocol-level error → top-level error, без result
    if isinstance(result, dict) and "error" in result and "content" not in result:
        return {"jsonrpc": "2.0", "id": msg.get("id"), "error": result["error"]}
    return {"jsonrpc": "2.0", "id": msg.get("id"), "result": result}


def _write(resp: Any) -> None:
    sys.stdout.write(json.dumps(resp, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


async def _amain(base_url: str) -> None:
    """Stdio loop. Session/client/server создаются ВНУТРИ loop'а (aiohttp требует).

    JSON-RPC 2.0: notifications (нет ``id``) не получают ответа; неизвестный метод
    → top-level ``error`` (НЕ вложенный в ``result``); parse-error → ``id: null``;
    batch (массив запросов) → массив ответов; non-object request → -32600.
    """
    async with aiohttp.ClientSession() as session:
        client = OrchestraClient(base_url, session)
        server = MCPServer(client, base_url)
        logger.info("MCP server starting (stdio). tools=%d, base=%s", len(server.tools), base_url)
        loop = asyncio.get_running_loop()
        # asyncio-cancellable stdin reader (не блокирует executor-поток на shutdown)
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while True:
            line = await reader.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                # id не определяем → null (JSON-RPC 2.0 §5.1)
                _write({"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": f"parse error: {exc}"}})
                continue
            # batch (массив запросов) → массив ответов (notifications без ответа)
            if isinstance(msg, list):
                responses = []
                for item in msg:
                    resp = await _handle_message(server, item)
                    if resp is not None:
                        responses.append(resp)
                if responses:
                    _write(responses)
                continue
            resp = await _handle_message(server, msg)
            if resp is not None:
                _write(resp)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCP-сервер ExtraOrchestra (stdio)")
    p.add_argument("--base-url", default=os.environ.get("ORCH_BASE_URL", DEFAULT_BASE))
    p.add_argument("--auto-start", action=argparse.BooleanOptionalAction,
                   default=os.environ.get("ORCH_AUTO_START", "1") not in ("0", "false", "no"),
                   help="запустить HTTP-сервер оркестра, если не отвечает (default: on; "
                        "--no-auto-start чтобы выключить)")
    p.add_argument("--log-level", default=os.environ.get("ORCH_LOG_LEVEL", "WARNING"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)
    base_url = args.base_url
    spawned = None
    if not _port_responds(base_url):
        if args.auto_start:
            logger.warning("orchestra server not responding at %s — auto-starting", base_url)
            spawned = _spawn_server(base_url)
            if spawned is None:
                logger.error("auto-start failed at %s — exiting "
                             "(запустите HTTP-сервер вручную: start_orchestra.sh)", base_url)
                sys.exit(2)
        else:
            logger.error("orchestra server not responding at %s and --auto-start off — exiting", base_url)
            sys.exit(2)
    try:
        asyncio.run(_amain(base_url))
    finally:
        if spawned is not None:
            spawned.terminate()
            try:
                spawned.wait(timeout=5)
            except subprocess.TimeoutExpired:
                spawned.kill()


if __name__ == "__main__":
    main()