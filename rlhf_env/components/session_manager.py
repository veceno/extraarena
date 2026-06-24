"""SessionManager — asyncio-реестр активных групп боёв.

Каждая группа боёв = background-task, выполняющая N матчей подряд.
Поддерживает:
  - start(spec) → group_id
  - status(group_id) → {status, current_battle, winrate, manifest_path, ...}
  - stop(group_id) → прерывает группу
  - list() → список всех групп (running + completed)
  - get_manifest(group_id) → содержимое manifest.json
  - find_battle_path(group_id, battle_id) → путь к battle_log

Используется из server.py (HTTP/WS) и mcp_server.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlhf_env.components.deck_builder import (
    CardCatalog,
    build_random_arena_deck,
    load_catalog,
    parse_custom_deck,
    validate_deck,
)
from rlhf_env.components.manifest import ManifestWriter
from rlhf_env.components.policy_registry import PolicyRegistry

logger = logging.getLogger(__name__)

# Лимит одновременных групп, чтобы не съесть всю память
# Максимум групп, держимых в памяти (вне зависимости от размера диска).
# Старые выгружаются, но файлы на диске остаются и доступны через /api/groups.
MAX_GROUPS_IN_MEMORY = 64


def _build_deck(
    deck_strategy: str,
    catalog: CardCatalog,
    custom_payload: Any,
    rng: random.Random,
) -> List[int]:
    """Универсальный билдер колоды по стратегии."""
    if deck_strategy == "custom":
        if custom_payload is None:
            raise ValueError("deck_strategy=custom requires custom_payload")
        ids = parse_custom_deck(custom_payload)
    elif deck_strategy in {"random_arenaenv", "random"}:
        ids = build_random_arena_deck(catalog, rng=rng)
    else:
        raise ValueError(f"unknown deck_strategy: {deck_strategy!r}")

    ok, msg = validate_deck(ids, catalog)
    if not ok:
        raise ValueError(f"deck validation failed: {msg}")
    return ids


def _build_state(
    deck_ids: List[int],
    catalog: CardCatalog,
    *,
    user_levels: Optional[Dict[int, int]] = None,
) -> Any:
    """deck_ids → GameState (через converter + classic_setup)."""
    from core.classic_setup import create_classic_game_state
    from core.converter import deck_from_card_ids

    levels = user_levels or {cid: 1 for cid in deck_ids}
    deck = deck_from_card_ids(deck_ids, catalog.cards, user_levels=levels)
    p1_user = 1000  # синтетические user_id для RLHF
    p2_user = 2000
    return create_classic_game_state(
        p1_user_id=p1_user,
        p2_user_id=p2_user,
        p1_deck=deck,
        p2_deck=deck,  # placeholder; ниже пересоздаём по факту
        p1_is_bot=True,
        p2_is_bot=True,
    )


def _build_game_state(
    p1_deck_ids: List[int],
    p2_deck_ids: List[int],
    catalog: CardCatalog,
    *,
    user_levels: Optional[Dict[int, int]] = None,
    starting_player: str = "random",
    rng: Optional[random.Random] = None,
) -> Any:
    """Создаёт GameState для двух конкретных колод."""
    from core.classic_setup import create_classic_game_state
    from core.converter import deck_from_card_ids

    rng = rng or random.Random()
    levels = user_levels or {cid: 1 for cid in set(p1_deck_ids + p2_deck_ids)}

    p1_deck = deck_from_card_ids(p1_deck_ids, catalog.cards, user_levels=levels)
    p2_deck = deck_from_card_ids(p2_deck_ids, catalog.cards, user_levels=levels)

    if starting_player == "p1":
        sp = 1000
    elif starting_player == "p2":
        sp = 2000
    else:
        sp = rng.choice([1000, 2000])

    return create_classic_game_state(
        p1_user_id=1000,
        p2_user_id=2000,
        p1_deck=p1_deck,
        p2_deck=p2_deck,
        p1_is_bot=True,
        p2_is_bot=True,
        starting_player_id=sp,
        rng=rng,
    )


class GroupState:
    """Состояние группы = цепочка боёв human-vs-model.

    Среда работает только в интерактивном режиме: человек играет против
    выбранной модели подряд N боёв (N = spec.battles_planned). На каждый
    бой заранее генерируется battle_id, чтобы клиент мог переходить от
    одного боя к другому без race-condition.
    """

    def __init__(self, group_id: str, spec: Dict[str, Any], manifest: ManifestWriter):
        self.group_id = group_id
        self.spec = spec
        self.manifest = manifest
        self.task: Optional[asyncio.Task] = None
        self.current_battle: int = 0
        self.last_error: Optional[str] = None
        self.started_at = manifest.manifest["created_at"]
        self.finished_at: Optional[str] = None
        # human_player: 1000 (P1) или 2000 (P2). По умолчанию — человек за P1.
        self.human_player: int = int(spec.get("human_player", 1000))
        # Сколько боёв подряд сыграть
        self.battles_planned: int = int(spec.get("battles_planned", 1))
        # Заранее сгенерированные battle_id для всей серии
        self.battle_ids: List[str] = [
            f"b_{uuid.uuid4().hex[:10]}" for _ in range(self.battles_planned)
        ]
        # battle_id, который сейчас играется (None если ещё не начат)
        self.active_battle_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        results = self.manifest.manifest.get("results", {})
        if self.finished_at:
            status = "completed"
        else:
            status = "running" if self.active_battle_id else "loaded"
        return {
            "group_id": self.group_id,
            "status": status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_battle": self.current_battle,
            "battles_planned": self.battles_planned,
            "battles_finished": results.get("battles_finished", 0),
            "winrate_p1": results.get("winrate_p1", 0.0),
            "winrate_p2": results.get("winrate_p2", 0.0),
            "last_error": self.last_error,
            "manifest_path": str(self.manifest.manifest_path),
            "human_player": self.human_player,
            "battle_ids": self.battle_ids,
            "active_battle_id": self.active_battle_id,
        }


class SessionManager:
    """Asyncio-реестр групп боёв."""

    def __init__(
        self,
        *,
        sessions_dir: Path | str,
        models_dir: Path | str,
        catalog: Optional[CardCatalog] = None,
        registry: Optional[PolicyRegistry] = None,
    ):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path(models_dir)
        self.catalog = catalog or load_catalog()
        self.registry = registry or PolicyRegistry.scan(self.models_dir)
        self._groups: Dict[str, GroupState] = {}

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------
    def start(self, spec: Dict[str, Any]) -> str:
        """Создаёт каркас группы human-vs-model. Возвращает group_id.

        Группа = N боёв подряд (spec.battles_planned) против одной модели.
        Каждому бою заранее выдан battle_id (state.battle_ids).
        Бои не запускаются в фоне — клиент открывает WS на конкретный
        battle_id и играет. Сервер ведёт серию: после боя N клиент получает
        battle_id боя N+1 (или сигнал «серия кончилась»).
        """
        # Генерируем group_id
        group_id = uuid.uuid4().hex[:12]
        group_dir = self.sessions_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "battles").mkdir(exist_ok=True)

        manifest = ManifestWriter(
            group_id=group_id,
            spec=spec,
            group_dir=group_dir,
            repo_root=Path.cwd(),
        )
        state = GroupState(group_id, spec, manifest)
        self._groups[group_id] = state

        logger.info(
            "[SessionManager] created group %s: human=%s vs %s, battles=%d, ids=%s",
            group_id, state.human_player,
            spec.get("p2_model" if state.human_player == 1000 else "p1_model"),
            state.battles_planned, state.battle_ids,
        )
        return group_id

    async def astart(self, spec: Dict[str, Any]) -> str:
        """Async-обёртка: удобно из тестов / скриптов под asyncio.run()."""
        return self.start(spec)

    def status(self, group_id: str) -> Optional[Dict[str, Any]]:
        s = self._groups.get(group_id)
        if s is None:
            return None
        return s.to_dict()

    def list(self) -> List[Dict[str, Any]]:
        # Сначала активные (в памяти)
        out = [g.to_dict() for g in self._groups.values()]
        # Дополнительно — completed/loaded с диска
        for entry in sorted(self.sessions_dir.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            gid = entry.name
            if gid in self._groups:
                continue
            mpath = entry / "manifest.json"
            if not mpath.exists():
                continue
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
                results = m.get("results", {})
                out.append({
                    "group_id": gid,
                    "status": "completed" if m.get("finished_at") else "loaded",
                    "started_at": m.get("created_at"),
                    "finished_at": m.get("finished_at"),
                    "current_battle": results.get("battles_finished", 0),
                    "battles_planned": m.get("spec", {}).get("battles_planned", 0),
                    "battles_finished": results.get("battles_finished", 0),
                    "winrate_p1": results.get("winrate_p1", 0.0),
                    "winrate_p2": results.get("winrate_p2", 0.0),
                    "last_error": None,
                    "manifest_path": str(mpath),
                })
            except Exception:
                pass
        return out

    def stop(self, group_id: str) -> bool:
        s = self._groups.get(group_id)
        if s is None or s.task is None or s.task.done():
            return False
        s.task.cancel()
        return True

    def get_manifest(self, group_id: str) -> Optional[Dict[str, Any]]:
        s = self._groups.get(group_id)
        if s is not None:
            return s.manifest.manifest
        mpath = self.sessions_dir / group_id / "manifest.json"
        if not mpath.exists():
            return None
        return json.loads(mpath.read_text(encoding="utf-8"))

    def find_battle_path(self, group_id: str, battle_id: str) -> Optional[Path]:
        # 1) стандартное место
        cand = self.sessions_dir / group_id / "battles" / f"{battle_id}.json"
        if cand.exists():
            return cand
        # 2) fallback — поиск по подстроке
        d = self.sessions_dir / group_id / "battles"
        if not d.exists():
            return None
        for f in d.glob("*.json"):
            if battle_id in f.stem:
                return f
        return None

    def battle_log(self, group_id: str, battle_id: str) -> Optional[Dict[str, Any]]:
        p = self.find_battle_path(group_id, battle_id)
        if p is None:
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    # Бои играются через _ws_loop (server.py) — каждый бой = одна WS-сессия
    # против бота. Никакого фонового runner'а не нужно.


__all__ = ["SessionManager", "GroupState", "MAX_GROUPS_IN_MEMORY"]