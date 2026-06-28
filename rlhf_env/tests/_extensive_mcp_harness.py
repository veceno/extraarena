"""Reusable in-process harness for extensive MCP testing (Workflow B).

Конструирует MCPServer поверх HeadlessHub (тот же ArenaMatchManager + MatchRunner,
что и prod-MCP stdio), патчит bot-sleep'и в 0 (in-process — pacing не нужен),
и даёт ``call(server, tool, args)`` для синхронного запуска async _tool.

Используется параллельными суб-агентами оркестратора: каждый агент получает свой
tmp sessions-dir → изоляция, кросс-процесс гонки не влияют на соседние кластеры.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any, Dict, Tuple

import rlhf_env.components.match_runner as _mr
from rlhf_env.components.policy_registry import PolicyRegistry
from rlhf_env.mcp_server import HeadlessHub, MCPServer

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = str(_REPO_ROOT / "ai" / "models")
CARDS_PATH = str(_REPO_ROOT / "ai" / "cards.json")

_patched = False


def _patch_sleep() -> None:
    """Убирает реальные bot-turn delay (4–6с/ход) — in-process pacing не нужен."""
    global _patched
    if _patched:
        return
    real = _mr.asyncio
    _mr.asyncio = types.SimpleNamespace(
        sleep=lambda d: real.sleep(0),
        create_task=real.create_task,
        Lock=real.Lock,
    )
    _patched = True


def make_server(tmp_path: Path) -> Tuple[MCPServer, HeadlessHub, Path]:
    _patch_sleep()
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    hub = HeadlessHub(
        sessions_dir=str(tmp_path / "sessions"),
        models_dir=MODELS_DIR,
        cards_path=CARDS_PATH,
    )
    reg = PolicyRegistry.scan(MODELS_DIR)
    return MCPServer(hub, reg), hub, tmp_path


def call(server: MCPServer, tool: str, args: Dict[str, Any] | None = None) -> Any:
    """Синхронно запускает server._tool(tool, args). Возвращает распарсенный JSON
    (или сырой объект) ответа. Поднимает исключение при ошибке инструмента."""
    return asyncio.run(server._tool(tool, args or {}))


def start_rl_series(server: MCPServer, **spec_kwargs) -> Dict[str, Any]:
    """Удобная обёртка: start_series с p1_actor_type='rl' (auto-play до game_over)."""
    spec = {
        "p2_model": spec_kwargs.pop("p2_model", "random"),
        "battles_planned": spec_kwargs.pop("battles_planned", 1),
        "seed": spec_kwargs.pop("seed", 42),
        "p1_actor_type": spec_kwargs.pop("p1_actor_type", "rl"),
        "p1_model": spec_kwargs.pop("p1_model", "random"),
        "starting_player": spec_kwargs.pop("starting_player", "p1"),
    }
    spec.update(spec_kwargs)
    return call(server, "start_series", {"spec": spec})