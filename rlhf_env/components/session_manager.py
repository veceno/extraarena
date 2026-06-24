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
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlhf_env.components.battle_runner import BattleRunner
from rlhf_env.components.deck_builder import (
    CardCatalog,
    build_random_arena_deck,
    load_catalog,
    parse_custom_deck,
    validate_deck,
)
from rlhf_env.components.manifest import ManifestWriter
from rlhf_env.components.policy_factory import build_policy
from rlhf_env.components.policy_registry import PolicyRegistry

logger = logging.getLogger(__name__)

# Лимит одновременных групп, чтобы не съесть всю память
MAX_CONCURRENT_GROUPS = 8


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
    """Состояние одной активной (или завершённой) группы."""

    def __init__(self, group_id: str, spec: Dict[str, Any], manifest: ManifestWriter):
        self.group_id = group_id
        self.spec = spec
        self.manifest = manifest
        self.task: Optional[asyncio.Task] = None
        self.current_battle: int = 0
        self.last_error: Optional[str] = None
        self.started_at = manifest.manifest["created_at"]
        self.finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        results = self.manifest.manifest.get("results", {})
        return {
            "group_id": self.group_id,
            "status": "completed" if self.finished_at else (
                "running" if self.task and not self.task.done() else "error"
            ),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_battle": self.current_battle,
            "battles_planned": self.spec.get("battles_planned", 0),
            "battles_finished": results.get("battles_finished", 0),
            "winrate_p1": results.get("winrate_p1", 0.0),
            "winrate_p2": results.get("winrate_p2", 0.0),
            "last_error": self.last_error,
            "manifest_path": str(self.manifest.manifest_path),
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
        """Стартует группу боёв в фоне. Возвращает group_id.

        Требует активный event loop (запускать из aiohttp app / asyncio.run()).
        """
        if len([g for g in self._groups.values() if g.task and not g.task.done()]) >= MAX_CONCURRENT_GROUPS:
            raise RuntimeError(f"too many concurrent groups (max {MAX_CONCURRENT_GROUPS})")

        # Генерируем group_id
        import uuid
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

        # Запускаем фоновый task в активном loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError(
                "SessionManager.start() requires a running asyncio event loop. "
                "Use 'await sm.astart(spec)' from inside an async context, "
                "or call from an aiohttp handler (event loop is running)."
            ) from e
        state.task = loop.create_task(self._run_group(state))
        logger.info("[SessionManager] started group %s spec=%s", group_id, spec)
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
    async def _run_group(self, state: GroupState) -> None:
        """Главный цикл группы: запускает battles_planned матчей подряд."""
        spec = state.spec
        try:
            p1_pol = build_policy({"name": spec.get("p1_model", "random")}, registry=self.registry)
            p2_pol = build_policy({"name": spec.get("p2_model", "random")}, registry=self.registry)
        except Exception as exc:
            state.last_error = f"policy load failed: {exc}"
            logger.exception("[SessionManager] policy load failed")
            return

        deck_strategy = spec.get("deck_strategy", "random_arenaenv")
        custom_p1 = spec.get("custom_deck_p1")
        custom_p2 = spec.get("custom_deck_p2")
        seed_base = int(spec.get("seed", 0))
        starting_player = spec.get("starting_player", "random")
        battles_planned = int(spec.get("battles_planned", 1))
        max_turns = int(spec.get("max_turns", 60))

        try:
            for i in range(battles_planned):
                state.current_battle = i + 1
                rng = random.Random(seed_base + i * 1009)
                try:
                    if deck_strategy == "custom" and (custom_p1 or custom_p2):
                        p1_ids = _build_deck("custom", self.catalog, custom_p1, rng)
                        p2_ids = _build_deck("custom", self.catalog, custom_p2, rng)
                    else:
                        p1_ids = _build_deck(deck_strategy, self.catalog, None, rng)
                        p2_ids = _build_deck(deck_strategy, self.catalog, None, rng)
                except Exception as exc:
                    state.last_error = f"deck build failed: {exc}"
                    logger.exception("[SessionManager] deck build failed")
                    continue

                game_state = _build_game_state(
                    p1_ids, p2_ids, self.catalog,
                    starting_player=starting_player, rng=rng,
                )
                from core.engine import ArenaEnvironment
                engine = ArenaEnvironment(game_state)

                battle_id = __import__("uuid").uuid4().hex[:12]
                battle_log_path = state.manifest.group_dir / "battles" / f"{battle_id}.json"
                runner = BattleRunner(
                    group_id=state.group_id,
                    battle_id=battle_id,
                    policy_a=p1_pol,
                    policy_b=p2_pol,
                    engine=engine,
                    battle_log_path=battle_log_path,
                    max_turns=max_turns,
                )
                battle_log = await runner.arun()
                winner, loser, status = (
                    battle_log["result"]["winner_user_id"],
                    battle_log["result"]["loser_user_id"],
                    battle_log["result"]["status"],
                )
                state.manifest.append_battle_result(
                    battle_id=battle_id,
                    battle_log_path=str(battle_log_path),
                    winner_user_id=winner,
                    loser_user_id=loser,
                    status=status,
                    turns=battle_log["final_state_summary"]["turn_number"],
                    duration_seconds=battle_log["duration_seconds"],
                )
                # сохраняем использованные колоды в манифест
                if "decks" not in state.manifest.manifest:
                    state.manifest.manifest["decks"] = {}
                state.manifest.manifest["decks"][battle_id] = {
                    "p1": p1_ids, "p2": p2_ids,
                }
        except asyncio.CancelledError:
            state.last_error = "cancelled"
            logger.info("[SessionManager] group %s cancelled", state.group_id)
        except Exception as exc:
            state.last_error = f"group failed: {exc}"
            logger.exception("[SessionManager] group %s failed", state.group_id)
        finally:
            state.manifest.finalize()
            state.finished_at = state.manifest.manifest["finished_at"]


__all__ = ["SessionManager", "GroupState", "MAX_CONCURRENT_GROUPS"]