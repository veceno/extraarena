"""MCP-сервер для RLHF-среды ExtraArena.

stdio MCP-сервер с 6 базовыми инструментами:
  1. start_battle_group(spec)         → старт группы боёв
  2. stop_battle_group(group_id)      → остановить группу
  3. list_battle_groups()             → список групп
  4. get_battle_group_status(group_id) → статус группы
  5. get_battle_group_manifest(group_id) → содержимое manifest.json
  6. download_battle_logs(group_id, format) → путь к архиву/JSON-папке

Запуск:
    ./start_rlhf_env.sh mcp
    python -m rlhf_env.mcp_server
    python -m rlhf_env.mcp_server --models-dir /path/to/v5

ВАЖНО: MCP-сервер использует ту же SessionManager, что и web. Если web
уже запущен — онлайн-статусы берутся из его памяти; для cold-start
MCP подгружает манифесты с диска.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Добавляем корень репо в sys.path
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlhf_env import __version__  # noqa: E402
from rlhf_env.components.policy_registry import PolicyRegistry  # noqa: E402
from rlhf_env.components.session_manager import SessionManager  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# MCP-совместимый fallback (если пакет `mcp` не установлен)
# ============================================================================

class _MCPServer:
    """Минимальный MCP stdio сервер.

    Если доступен пакет `mcp` — используется его stdio-server;
    иначе — собственный fallback с тем же JSON-RPC 2.0 контрактом
    (initialize / tools/list / tools/call).
    """

    def __init__(self, manager: SessionManager, registry: PolicyRegistry):
        self.manager = manager
        self.registry = registry
        self.tools = self._build_tools()

    def _build_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "start_battle_group",
                "description": (
                    "Запустить группу боёв. spec: {p1_model, p2_model, deck_strategy, "
                    "battles_planned, seed, starting_player, max_turns, custom_deck_p1?, custom_deck_p2?}"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "object",
                            "properties": {
                                "p1_model": {"type": "string"},
                                "p2_model": {"type": "string"},
                                "deck_strategy": {"type": "string", "default": "random_arenaenv"},
                                "battles_planned": {"type": "integer", "default": 1, "minimum": 1, "maximum": 1000},
                                "seed": {"type": "integer", "default": 0},
                                "starting_player": {"type": "string", "default": "random"},
                                "max_turns": {"type": "integer", "default": 60},
                                "custom_deck_p1": {"type": "any"},
                                "custom_deck_p2": {"type": "any"},
                                "difficulty": {"type": "string", "default": "default"},
                            },
                            "required": ["p1_model", "p2_model"],
                        },
                    },
                    "required": ["spec"],
                },
            },
            {
                "name": "stop_battle_group",
                "description": "Остановить активную группу боёв.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"group_id": {"type": "string"}},
                    "required": ["group_id"],
                },
            },
            {
                "name": "list_battle_groups",
                "description": "Список всех групп (активные + завершённые на диске).",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_battle_group_status",
                "description": "Статус одной группы + winrate + manifest_path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"group_id": {"type": "string"}},
                    "required": ["group_id"],
                },
            },
            {
                "name": "get_battle_group_manifest",
                "description": "Полное содержимое manifest.json группы.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"group_id": {"type": "string"}},
                    "required": ["group_id"],
                },
            },
            {
                "name": "download_battle_logs",
                "description": "Скачать логи всех боёв группы (json или zip).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string"},
                        "format": {"type": "string", "enum": ["json", "zip"], "default": "json"},
                    },
                    "required": ["group_id"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------
    def _tool_start_battle_group(self, args: Dict[str, Any]) -> Dict[str, Any]:
        spec = args.get("spec", {})
        if not isinstance(spec, dict):
            raise ValueError("spec must be an object")
        if "battles_planned" not in spec:
            spec["battles_planned"] = 1
        gid = self.manager.start(spec)
        return {
            "group_id": gid,
            "status": "running",
            "manifest_path": str(self.manager.sessions_dir / gid / "manifest.json"),
        }

    def _tool_stop_battle_group(self, args: Dict[str, Any]) -> Dict[str, Any]:
        gid = args["group_id"]
        ok = self.manager.stop(gid)
        return {"stopped": ok, "group_id": gid}

    def _tool_list_battle_groups(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        return {"groups": self.manager.list()}

    def _tool_get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        gid = args["group_id"]
        s = self.manager.status(gid)
        if s is None:
            m = self.manager.get_manifest(gid)
            if m is None:
                return {"error": "group not found"}
            return {"group_id": gid, "status": "loaded", "manifest": m}
        return s

    def _tool_get_manifest(self, args: Dict[str, Any]) -> Dict[str, Any]:
        gid = args["group_id"]
        m = self.manager.get_manifest(gid)
        if m is None:
            return {"error": "group not found"}
        return {"group_id": gid, "manifest": m}

    def _tool_download_logs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        gid = args["group_id"]
        fmt = args.get("format", "json")
        group_dir = self.manager.sessions_dir / gid
        if not group_dir.exists():
            return {"error": "group not found"}
        if fmt == "zip":
            zip_path = self.manager.sessions_dir / f"{gid}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in group_dir.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=f.relative_to(group_dir))
            return {"path": str(zip_path), "size": zip_path.stat().st_size, "format": "zip"}
        else:
            return {"path": str(group_dir), "format": "json", "files": [
                str(p.relative_to(self.manager.sessions_dir)) for p in group_dir.rglob("*.json")
            ]}

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------
    def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
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
            args = params.get("arguments", {})
            try:
                if name == "start_battle_group":
                    result = self._tool_start_battle_group(args)
                elif name == "stop_battle_group":
                    result = self._tool_stop_battle_group(args)
                elif name == "list_battle_groups":
                    result = self._tool_list_battle_groups(args)
                elif name == "get_battle_group_status":
                    result = self._tool_get_status(args)
                elif name == "get_battle_group_manifest":
                    result = self._tool_get_manifest(args)
                elif name == "download_battle_logs":
                    result = self._tool_download_logs(args)
                else:
                    return {"error": {"code": -32601, "message": f"unknown tool: {name}"}}
                return {"content": [{"type": "json", "data": result}], "isError": False}
            except Exception as exc:
                logger.exception("[mcp] tool %s failed", name)
                return {"content": [{"type": "json", "data": {"error": str(exc)}}], "isError": True}
        return {"error": {"code": -32601, "message": f"unknown method: {method}"}}

    # ------------------------------------------------------------------
    # Stdio loop
    # ------------------------------------------------------------------
    def run_stdio(self) -> None:
        """Минимальный JSON-RPC stdio loop. Читает по одному сообщению из stdin."""
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": 0, "result": {"ready": True, "version": __version__}}) + "\n")
        sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                req_id = msg.get("id", 0)
                method = msg.get("method", "")
                params = msg.get("params", {})
                result = self.dispatch(method, params)
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
                sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": 0, "error": {"code": -32700, "message": str(exc)}
                }) + "\n")
                sys.stdout.flush()


# ============================================================================
# Entrypoint
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCP-сервер RLHF-среды")
    p.add_argument("--models-dir", default=os.environ.get("RLHF_MODELS_DIR", "ai/models"))
    p.add_argument("--sessions-dir", default=os.environ.get("RLHF_SESSIONS_DIR", "rlhf_env/sessions"))
    p.add_argument("--cards-path", default=os.environ.get("RLHF_CARDS_PATH", "ai/cards.json"))
    p.add_argument("--log-level", default=os.environ.get("RLHF_LOG_LEVEL", "INFO"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    registry = PolicyRegistry.scan(args.models_dir)
    manager = SessionManager(
        sessions_dir=args.sessions_dir,
        models_dir=args.models_dir,
        registry=registry,
    )

    server = _MCPServer(manager, registry)
    logger.info("MCP server starting (stdio). tools=%d, models=%d", len(server.tools), len(registry.specs))
    server.run_stdio()


if __name__ == "__main__":
    main()