"""Общие помощники для V5-тестов rlhf_env (mana_draw / теггинг актора / v5_trace).

Драйвит матч in-process через ArenaMatchManager + MatchRunner (без HTTP/сервера):
быстро, детерминированно (seed), читает v5/trace прямо из sessions-директории.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.actions import ManaDrawAction, EndTurnAction
from rlhf_env.components.arena_match_manager import ArenaMatchManager
from rlhf_env.components.match_runner import MatchRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "ai" / "models"
DEFAULT_CARDS_PATH = REPO_ROOT / "ai" / "cards.json"


def make_manager(tmp_path: Path) -> ArenaMatchManager:
    return ArenaMatchManager(
        sessions_dir=tmp_path / "sessions",
        models_dir=str(DEFAULT_MODELS_DIR),
        cards_path=str(DEFAULT_CARDS_PATH),
    )


def create_match(
    mgr: ArenaMatchManager,
    *,
    p1_actor_type: str = "llm",
    p2_model: str = "random",
    p1_model: Optional[str] = None,
    p2_model_path: Optional[str] = None,
    p2_model_kind: Optional[str] = None,
    agent_name: Optional[str] = None,
    seed: int = 42,
    starting_player: str = "p1",
    battles_planned: int = 1,
) -> Tuple[Any, Any, MatchRunner]:
    spec: Dict[str, Any] = {
        "p2_model": p2_model,
        "battles_planned": battles_planned,
        "seed": seed,
        "p1_actor_type": p1_actor_type,
        "starting_player": starting_player,
    }
    if p1_model is not None:
        spec["p1_model"] = p1_model
    if p2_model_path is not None:
        spec["p2_model_path"] = p2_model_path
    if p2_model_kind is not None:
        spec["p2_model_kind"] = p2_model_kind
    if agent_name is not None:
        spec["agent_name"] = agent_name
    match = mgr.create_series(spec)
    engine = match.engine
    runner = MatchRunner(match)
    return match, engine, runner


def p1_can_mana_draw(engine) -> bool:
    if engine.is_ended:
        return False
    if engine.get_current_player_id() != engine.human_user_id:
        return False
    legal = engine.get_legal_actions_raw(engine.human_user_id)
    return any(isinstance(a, ManaDrawAction) for a in legal)


async def drive_until_mana_draw_then(
    runner: MatchRunner,
    engine,
    *,
    on_p1_turn=None,
    max_steps: int = 80,
) -> Tuple[bool, int]:
    """Гоняет бой, пока p1 не сможет добрать карту; выполняет mana_draw.

    on_p1_turn(runner, engine, step) -> bool: вызывается на каждом ходе p1 ДО
    добора; если вернёт True — mana_draw считается выполненным кастомным колбэком
    (для bot-guard теста, где мы НЕ хотим добирать). Возвращает (drew, steps).
    """
    drew = False
    for step in range(max_steps):
        if engine.is_ended:
            break
        if engine.get_current_player_id() == engine.human_user_id:
            if not drew and p1_can_mana_draw(engine):
                if on_p1_turn is not None and on_p1_turn(runner, engine, step):
                    drew = True
                else:
                    resp = await runner.execute_human_action(
                        {"type": "mana_draw", "client_action_id": f"c_md_{step}"}
                    )
                    r = resp.get("result", {}) if isinstance(resp, dict) else {}
                    if r.get("success"):
                        drew = True
            else:
                await runner.execute_human_action(
                    {"type": "end_turn", "client_action_id": f"c_et_{step}"}
                )
        else:
            await runner.run_bot_turn()
    return drew, max_steps


def v5_dir_for(match, tmp_path: Path) -> Path:
    gdir = Path(tmp_path) / "sessions" / match.group_id
    return gdir / "battles" / match.battle_id / "v5"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))