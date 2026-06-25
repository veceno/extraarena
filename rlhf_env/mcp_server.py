"""MCP-сервер RLHF-среды ExtraArena (stdio) — игра агентом headless + метаданные.

Инструменты разделены на две группы:

  Управление серией:
    start_series(spec)                — создать серию, вернуть первый match_id
    next_battle(group_id)             — следующий бой серии (или series_complete)
    list_battle_groups()              — список групп
    get_battle_group_status(gid)      — статус группы
    get_battle_group_manifest(gid)    — manifest.json
    get_dataset(group_id)             — путь к каноничному NDJSON (dataset.jsonl)
    list_battle_manifests(gid)        — список battle_log + .jsonl путей
    download_battle_logs(gid, fmt)    — zip/json папка группы

  Игра (агент играет за человека, headless, без браузера/WS):
    get_state(match_id)               — полный actor-perspective state (как /api/battle/state)
    get_legal_actions(match_id)      — список легальных действий
    submit_action(match_id, action)  — выполнить действие человека; авто-advance бота
    advance_bot(match_id)            — прокрутить один ход бота (если сейчас ход бота)
    surrender(match_id)              — сдаться → финализация + NDJSON flush

Контракт совпадает с браузерной ареной (тот же RlhfBattleEngine + MatchRunner),
поэтому данные из MCP-игры и из браузера идентичны по форме.

Запуск:
    ./rlhf_env/start_rlhf_env.sh mcp
    python -m rlhf_env.mcp_server
    python -m rlhf_env.mcp_server --models-dir /path/to/v5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlhf_env import __version__  # noqa: E402
from rlhf_env.components.arena_match_manager import ArenaMatchManager  # noqa: E402
from rlhf_env.components.match_runner import MatchRunner  # noqa: E402
from rlhf_env.components.policy_factory import BOT_MAX_DIFFICULTY  # noqa: E402
from rlhf_env.components.policy_registry import PolicyRegistry  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# HeadlessHub — реестр матчей + ленивые MatchRunner'ы (без WS/broadcaster)
# ============================================================================

class HeadlessHub:
    """Связывает ArenaMatchManager с MatchRunner'ами для headless-игры."""

    def __init__(self, *, sessions_dir: str, models_dir: str, cards_path: str):
        self.manager = ArenaMatchManager(
            sessions_dir=sessions_dir, models_dir=models_dir, cards_path=cards_path,
        )
        self._runners: Dict[str, MatchRunner] = {}

    def _match(self, match_id: str):
        return self.manager.get_match(match_id)

    async def get_runner(self, match_id: str) -> Optional[MatchRunner]:
        r = self._runners.get(match_id)
        if r is not None:
            return r
        match = self.manager.get_match(match_id)
        if match is None:
            return None
        r = MatchRunner(match)
        # broadcaster=None → WS-бродкасты не нужны (headless).
        self._runners[match_id] = r
        return r


# ============================================================================
# MCP-сервер (JSON-RPC 2.0 over stdio)
# ============================================================================

class MCPServer:
    def __init__(self, hub: HeadlessHub, registry: PolicyRegistry):
        self.hub = hub
        self.registry = registry
        self.tools = self._build_tools()

    # ------------------------------------------------------------------
    # tool schemas
    # ------------------------------------------------------------------
    def _build_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "start_series",
                "description": (
                    "Создать серию боёв (человек vs модель) и вернуть первый match_id. "
                    "spec: {p2_model, battles_planned, seed?, starting_player?, "
                    "deck_strategy_p1?, deck_strategy_p2?, custom_deck_p1?, custom_deck_p2?, "
                    "p1_name?, p2_name?, ...}. Модель всегда играет на максимум (argmax); "
                    "сложность не выбирается. Агент играет за человека (p1) через submit_action."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "object",
                            "properties": {
                                "p2_model": {"type": "string", "description": "имя модели-оппонента или 'random'"},
                                "battles_planned": {"type": "integer", "default": 1, "minimum": 1, "maximum": 1000},
                                "seed": {"type": "integer", "default": 0},
                                "starting_player": {"type": "string", "default": "random", "enum": ["random", "p1", "p2"]},
                                "deck_strategy_p1": {"type": "string", "default": "random_arenaenv", "enum": ["random_arenaenv", "custom"]},
                                "deck_strategy_p2": {"type": "string", "default": "random_arenaenv", "enum": ["random_arenaenv", "custom"]},
                                "custom_deck_p1": {"type": "array", "items": {"type": "integer"}},
                                "custom_deck_p2": {"type": "array", "items": {"type": "integer"}},
                                "p1_name": {"type": "string"}, "p2_name": {"type": "string"},
                            },
                            "required": ["p2_model"],
                        },
                    },
                    "required": ["spec"],
                },
            },
            {
                "name": "next_battle",
                "description": "Перейти к следующему бою серии. Возвращает match_id или series_complete.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_state",
                "description": "Полный actor-perspective state боя (тот же формат, что /api/battle/state в браузере).",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "get_legal_actions",
                "description": "Список легальных действий для текущего (человека) игрока.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "submit_action",
                "description": (
                    "Выполнить действие человека. action: {type:'play_card'|'attack'|'end_turn', ...}. "
                    "play_card: {type:'play_card', card_id_from_hand|hand_index, target_position?, target_id?, target_is_hero?}. "
                    "attack: {type:'attack', attacker_id, target_id, target_is_hero?}. "
                    "end_turn: {type:'end_turn'}. После успешного действия ход бота прокручивается автоматически."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "match_id": {"type": "string"},
                        "action": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["play_card", "attack", "end_turn"]},
                                "card_id_from_hand": {"type": "integer"},
                                "hand_index": {"type": "integer"},
                                "target_position": {"type": "integer"},
                                "target_id": {"type": ["integer", "string"]},
                                "target_is_hero": {"type": "boolean"},
                                "attacker_id": {"type": ["integer", "string"]},
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["match_id", "action"],
                },
            },
            {
                "name": "advance_bot",
                "description": "Прокрутить один ход бота (если сейчас ход бота). Иначе no-op.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "surrender",
                "description": "Сдаться (человек) → финализация боя + flush NDJSON.",
                "inputSchema": {"type": "object", "properties": {"match_id": {"type": "string"}}, "required": ["match_id"]},
            },
            {
                "name": "list_battle_groups",
                "description": "Список всех групп боёв.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_battle_group_status",
                "description": "Статус группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_battle_group_manifest",
                "description": "Полное содержимое manifest.json группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "get_dataset",
                "description": "Путь к каноничному NDJSON (dataset.jsonl) и per-battle .jsonl файлам группы.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "list_battle_manifests",
                "description": "Список battle_log.json + analytics .jsonl путей по группе.",
                "inputSchema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
            },
            {
                "name": "download_battle_logs",
                "description": "Собрать логи группы в zip или вернуть список json-файлов.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string"},
                        "format": {"type": "string", "enum": ["json", "zip"], "default": "json"},
                    },
                    "required": ["group_id"],
                },
            },
            {
                "name": "list_models",
                "description": "Список доступных моделей-оппонентов.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    # ------------------------------------------------------------------
    # tool handlers (async)
    # ------------------------------------------------------------------
    async def _tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        hub = self.hub

        if name == "start_series":
            spec = args.get("spec", {})
            if not isinstance(spec, dict):
                raise ValueError("spec must be an object")
            spec.setdefault("battles_planned", 1)
            match = hub.manager.create_series(spec)
            # если бот ходит первым — прокручиваем (headless: агент не видит ход бота автоматически иначе)
            runner = await hub.get_runner(match.engine.match_id)
            if match.engine.is_current_player_bot() and runner is not None:
                await runner.run_bot_turn()
            return {
                "group_id": match.group_id,
                "match_id": match.engine.match_id,
                "battle_id": match.battle_id,
                "battles_planned": match.battles_planned,
                "player_ids": [match.engine.human_user_id, match.engine.bot_user_id],
                "opponent": {"model": spec.get("p2_model"), "difficulty": BOT_MAX_DIFFICULTY},
            }

        if name == "next_battle":
            gid = args["group_id"]
            match = hub.manager.next_match(gid)
            if match is None:
                return {"status": "series_complete", "group_id": gid}
            runner = await hub.get_runner(match.engine.match_id)
            if match.engine.is_current_player_bot() and runner is not None:
                await runner.run_bot_turn()
            return {"match_id": match.engine.match_id, "battle_id": match.battle_id, "group_id": gid}

        if name == "get_state":
            match = hub._match(args["match_id"])
            if match is None:
                return {"error": "match_not_found"}
            return match.engine.get_full_state(viewer_id=match.engine.human_user_id)

        if name == "get_legal_actions":
            match = hub._match(args["match_id"])
            if match is None:
                return {"error": "match_not_found"}
            uid = match.engine.human_user_id
            legal = match.engine.get_legal_actions(uid) if not match.engine.is_current_player_bot() else []
            return {"legal_actions": legal, "is_my_turn": match.engine.get_current_player_id() == uid}

        if name == "submit_action":
            match_id = args["match_id"]
            action = args.get("action", {})
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            action = dict(action)
            action.setdefault("client_action_id", f"mcp_{match_id}_{int(asyncio.get_event_loop().time()*1000)&0xffff}")
            resp = await runner.execute_human_action(action)
            # авто-advance бота уже выполнен в execute_human_action (create_task run_bot_turn),
            # но в headless-режоте create_task мог быть запланирован — дождёмся его.
            await self._drain_bot(runner)
            return resp

        if name == "advance_bot":
            match_id = args["match_id"]
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            if not runner.match.engine.is_current_player_bot():
                return {"status": "not_bot_turn"}
            await runner.run_bot_turn()
            await self._drain_bot(runner)
            return {"status": "ok", "is_ended": runner.match.engine.is_ended}

        if name == "surrender":
            match_id = args["match_id"]
            runner = await hub.get_runner(match_id)
            if runner is None:
                return {"error": "match_not_found"}
            resp = await runner.surrender()
            return resp

        if name == "list_battle_groups":
            return {"groups": hub.manager.list_groups()}

        if name == "get_battle_group_status":
            gid = args["group_id"]
            m = hub.manager.list_groups()
            for g in m:
                if g["group_id"] == gid:
                    return g
            return {"error": "group not found"}

        if name == "get_battle_group_manifest":
            gid = args["group_id"]
            path = hub.manager.sessions_dir / gid / "manifest.json"
            if not path.exists():
                return {"error": "group not found"}
            return json.loads(path.read_text(encoding="utf-8"))

        if name == "get_dataset":
            gid = args["group_id"]
            gdir = hub.manager.sessions_dir / gid
            if not gdir.exists():
                return {"error": "group not found"}
            dataset = gdir / "dataset.jsonl"
            battles_dir = gdir / "battles"
            per_battle = sorted(str(p) for p in battles_dir.glob("*.jsonl")) if battles_dir.exists() else []
            return {
                "group_id": gid,
                "dataset_jsonl": str(dataset),
                "dataset_exists": dataset.exists(),
                "dataset_rows": sum(1 for _ in dataset.open()) if dataset.exists() else 0,
                "per_battle_jsonl": per_battle,
            }

        if name == "list_battle_manifests":
            gid = args["group_id"]
            gdir = hub.manager.sessions_dir / gid / "battles"
            if not gdir.exists():
                return {"error": "group not found"}
            return {"battles": sorted(str(p) for p in gdir.glob("*.json"))}

        if name == "download_battle_logs":
            gid = args["group_id"]
            fmt = args.get("format", "json")
            gdir = hub.manager.sessions_dir / gid
            if not gdir.exists():
                return {"error": "group not found"}
            if fmt == "zip":
                zip_path = hub.manager.sessions_dir / f"{gid}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in gdir.rglob("*"):
                        if f.is_file():
                            zf.write(f, arcname=f.relative_to(gdir))
                return {"path": str(zip_path), "size": zip_path.stat().st_size, "format": "zip"}
            return {"path": str(gdir), "format": "json",
                    "files": sorted(str(p.relative_to(hub.manager.sessions_dir)) for p in gdir.rglob("*") if p.is_file())}

        if name == "list_models":
            return {"models": self.registry.list_specs()}

        raise ValueError(f"unknown tool: {name}")

    async def _drain_bot(self, runner: MatchRunner) -> None:
        """Дождаться завершения запланированной бот-рутины (если была)."""
        task = runner._bot_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("bot task timed out while draining")
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------
    async def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "rlhf-env", "version": __version__},
                "capabilities": {"tools": {}},
            }
        if method == "tools/list":
            return {"tools": self.tools}
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            try:
                result = await self._tool(name, args)
                return {"content": [{"type": "json", "data": result}], "isError": False}
            except Exception as exc:  # noqa: BLE001
                logger.exception("[mcp] tool %s failed", name)
                return {"content": [{"type": "json", "data": {"error": str(exc)}}], "isError": True}
        return {"error": {"code": -32601, "message": f"unknown method: {method}"}}


# ============================================================================
# Async stdio loop (единый event-loop — MatchRunner lock/tasks живы между вызовами)
# ============================================================================

async def _amain(server: MCPServer) -> None:
    """Stdio JSON-RPC loop. Один запрос — один ответ (строго 1:1, без unsolicited-
    banner'ов, чтобы не ломать MCP-клиенты и синхронные stdio-харнессы)."""
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            req_id = msg.get("id", 0)
            method = msg.get("method", "")
            params = msg.get("params", {}) or {}
            result = await server.dispatch(method, params)
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        except Exception as exc:  # noqa: BLE001
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": 0,
                                         "error": {"code": -32700, "message": str(exc)}}) + "\n")
        sys.stdout.flush()


# ============================================================================
# Entrypoint
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCP-сервер RLHF-среды (stdio)")
    p.add_argument("--models-dir", default=os.environ.get("RLHF_MODELS_DIR", "ai/models"))
    p.add_argument("--sessions-dir", default=os.environ.get("RLHF_SESSIONS_DIR", "rlhf_env/sessions"))
    p.add_argument("--cards-path", default=os.environ.get("RLHF_CARDS_PATH", "ai/cards.json"))
    p.add_argument("--log-level", default=os.environ.get("RLHF_LOG_LEVEL", "WARNING"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    registry = PolicyRegistry.scan(args.models_dir)
    hub = HeadlessHub(sessions_dir=args.sessions_dir, models_dir=args.models_dir, cards_path=args.cards_path)
    server = MCPServer(hub, registry)
    logger.info("MCP server starting (stdio). tools=%d, models=%d", len(server.tools), len(registry.specs))
    try:
        asyncio.run(_amain(server))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()