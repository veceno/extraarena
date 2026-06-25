"""ArenaMatchManager — живой реестр матчей RLHF-арены 1:1.

Владеет жизненным циклом серии из N боёв подряд (человек vs модель):
  - create_series(spec) → первый ArenaMatch (group_id, match_id, battle_id)
  - next_match(group_id) → следующий бой серии (или None, если серия кончилась)
  - get_match(match_id) → ArenaMatch для HTTP/Socket.IO/MCP

Колоды строятся по стратегиям deck_strategy_p1/p2 (random_arenaenv | custom)
с отдельными custom_deck_p1/p2 — разводка, которая в старом server.py была
мёртвой (всегда random). Теперь custom-колоды реально доходят до движка.

Манифест/сводка пишутся через ManifestWriter в тот же sessions_dir, что и
SessionManager, поэтому существующий /api/groups browse-API продолжает работать.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.classic_setup import create_classic_game_state
from core.converter import deck_from_card_ids
from core.engine import ArenaEnvironment
from infrastructure.match_modes import resolve_mode_config

from rlhf_env.components.arena_engine import RlhfBattleEngine
from rlhf_env.components.deck_builder import CardCatalog, build_random_arena_deck, load_catalog, parse_custom_deck, validate_deck
from rlhf_env.components.manifest import ManifestWriter
from rlhf_env.components.policy_factory import BOT_MAX_DIFFICULTY, build_policy
from rlhf_env.components.policy_registry import PolicyRegistry

logger = logging.getLogger(__name__)


def _build_deck_ids(
    deck_strategy: str,
    catalog: CardCatalog,
    custom_payload: Any,
    rng: random.Random,
) -> List[int]:
    """Универсальный билдер колоды по стратегии (random_arenaenv | custom)."""
    if deck_strategy in {"custom", "specific"}:
        if custom_payload is None:
            raise ValueError("deck_strategy=custom requires custom_deck payload")
        ids = parse_custom_deck(custom_payload)
    elif deck_strategy in {"random_arenaenv", "random"}:
        ids = build_random_arena_deck(catalog, rng=rng)
    else:
        raise ValueError(f"unknown deck_strategy: {deck_strategy!r}")
    ok, msg = validate_deck(ids, catalog)
    if not ok:
        raise ValueError(f"deck validation failed: {msg}")
    return ids


# Человеко-читаемое имя модели для никнейма противника в арене.
# Бренд-префикс «extra-lr» схлопывается в «ExtraLR» (одно слово, как в
# маркетинге моделей), остальные токены маппятся по словарю.
_MODEL_TOKEN_MAP = {
    "extra": "Extra", "lr": "LR", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5",
    "max": "Max", "lite": "Lite", "medium": "Medium", "micro": "Micro", "opti": "Opti",
    "greedy": "Greedy", "face": "Face", "end": "End", "turn": "Turn", "random": "Random",
    "legal": "Legal", "biggest": "Biggest", "only": "Only", "versus": "Versus",
}


def _display_model_name(model: Any) -> str:
    """extra-lr-v4-max → «ExtraLR V4 Max»; end_turn → «End Turn»."""
    raw = str(model or "").strip()
    if not raw:
        return "Bot"
    parts = [p for p in re.split(r"[-_]", raw) if p]
    out: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i].lower() == "extra" and i + 1 < len(parts) and parts[i + 1].lower() == "lr":
            out.append("ExtraLR")
            i += 2
            continue
        low = parts[i].lower()
        out.append(_MODEL_TOKEN_MAP.get(low, parts[i][:1].upper() + parts[i][1:]))
        i += 1
    return " ".join(out) if out else "Bot"


def _starting_player_id(spec: Dict[str, Any], rng: random.Random) -> int:
    sp = str(spec.get("starting_player", "random")).lower()
    if sp == "p1":
        return 1000
    if sp == "p2":
        return 2000
    return rng.choice([1000, 2000])


# «Средние уровни» боя: одно центральное значение НА ДВОИХ (игрок+бот) из
# {3, 5, 8}, каждая отдельная карта ±2 от него (clamp 1..max карты). Вся колода
# получается «приблизительно одного уровня», но карты разноуровневые — это
# реалистичнее для обучающих данных, чем поголовно 1-й уровень. Центральное
# значение выбирается rng (репродуцируемо по seed боя) и одно на обе колоды.
BATTLE_CENTRAL_LEVELS: tuple[int, ...] = (3, 5, 8)
BATTLE_LEVEL_SPREAD: int = 2


def _card_max_level(card_data: Dict[str, Any]) -> int:
    """Макс. уровень карты: 2 для simplified_levelup (герои), иначе 10."""
    return 2 if card_data.get("simplified_levelup") else 10


def _battle_card_levels(
    card_ids: List[int],
    central: int,
    rng: random.Random,
    catalog: "CardCatalog",
) -> Dict[int, int]:
    """Уровень каждой карты = clamp(central ± spread, 1, max карты).

    Дубли card_id запрещены форматом колоды (см. build_random_arena_deck), поэтому
    словарь {card_id: level} однозначно задаёт уровень каждой карты в колоде.
    """
    levels: Dict[int, int] = {}
    for cid in card_ids:
        max_lvl = _card_max_level(catalog.cards.get(cid, {}))
        delta = rng.randint(-BATTLE_LEVEL_SPREAD, BATTLE_LEVEL_SPREAD)
        levels[cid] = max(1, min(max_lvl, central + delta))
    return levels


@dataclass
class ArenaMatch:
    """Один живой бой в серии."""

    engine: RlhfBattleEngine
    group_id: str
    battle_id: str
    battle_index: int
    battles_planned: int
    spec: Dict[str, Any]
    bot_policy: Any
    rng: random.Random
    p1_deck_ids: List[int]
    p2_deck_ids: List[int]
    p1_levels: Dict[int, int]
    p2_levels: Dict[int, int]
    manifest: ManifestWriter
    # Источник колоды P1: {"type":"random"} (регенерится каждый бой → превью)
    # или {"type":"imported","preset_number":N} (фиксирована на серию).
    p1_deck_source: Dict[str, Any] = field(default_factory=lambda: {"type": "random"})
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    finished: bool = False
    action_seq: int = 0
    battle_start_monotonic: float = 0.0
    # AnalyticsRecorder подключается в фазе 4 (match_runner).
    recorder: Any = None
    # V5TraceRecorder — максимально полная omniscient-запись боя для V5-обучения
    # (отдельная директория battles/<bid>/v5/). None — no-op.
    v5_recorder: Any = None

    def next_action_seq(self) -> int:
        self.action_seq += 1
        return self.action_seq


@dataclass
class _GroupLive:
    group_id: str
    spec: Dict[str, Any]
    manifest: ManifestWriter
    battles_planned: int
    current_index: int = -1
    current_match_id: Optional[str] = None


class ArenaMatchManager:
    """Реестр живых матчей + серия боёв."""

    def __init__(
        self,
        *,
        sessions_dir: Path | str,
        models_dir: Path | str,
        catalog: Optional[CardCatalog] = None,
        registry: Optional[PolicyRegistry] = None,
        cards_path: Optional[Path | str] = None,
    ):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path(models_dir)
        self.catalog = catalog or load_catalog()
        self.registry = registry or PolicyRegistry.scan(self.models_dir)
        self.cards_path = Path(cards_path) if cards_path else Path("ai/cards.json")
        self.catalog_hash = self._compute_catalog_hash()
        self._groups: Dict[str, _GroupLive] = {}
        self._matches: Dict[str, ArenaMatch] = {}

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------

    def create_series(self, spec: Dict[str, Any]) -> ArenaMatch:
        """Создаёт новую группу и первый бой серии. Возвращает ArenaMatch."""
        group_id = uuid.uuid4().hex[:12]
        group_dir = self.sessions_dir / group_id
        # Seed: если не задан явно (0/None) — рандомизируем. Иначе первый бой
        # каждой серии детерминирован (seed=0 → одна и та же колода два раза
        # подряд). Реальный seed пишем в spec → попадает в манифест (provenance
        # /воспроизводимость) и используется для всех боёв серии (base+idx*1009).
        if not int(spec.get("seed", 0) or 0):
            spec["seed"] = random.randrange(1, 2**31)
        manifest = ManifestWriter(
            group_id=group_id,
            spec=spec,
            group_dir=group_dir,
            repo_root=Path.cwd(),
        )
        live = _GroupLive(
            group_id=group_id,
            spec=spec,
            manifest=manifest,
            battles_planned=int(spec.get("battles_planned", 1)),
        )
        self._groups[group_id] = live
        match = self._build_match(group_id, live, battle_index=0)
        live.current_index = 0
        live.current_match_id = match.engine.match_id
        logger.info(
            "[ArenaMatchManager] create_series group=%s match=%s p2_model=%s battles=%d",
            group_id, match.engine.match_id, spec.get("p2_model"), live.battles_planned,
        )
        return match

    def next_match(self, group_id: str) -> Optional[ArenaMatch]:
        """Возвращает следующий бой серии или None, если серия закончилась."""
        live = self._groups.get(group_id)
        if live is None:
            return None
        next_index = live.current_index + 1
        if next_index >= live.battles_planned:
            live.manifest.finalize()
            live.current_match_id = None
            logger.info("[ArenaMatchManager] series complete group=%s", group_id)
            return None
        match = self._build_match(group_id, live, battle_index=next_index)
        live.current_index = next_index
        live.current_match_id = match.engine.match_id
        return match

    def finish_series(self, group_id: str) -> Dict[str, Any]:
        """Досрочно закрывает серию (человек нажал «Завершить»/ушёл в меню):
        финализирует манифест с уже записанными боями. Идемпотентно —
        повторный вызов для сыгранной до конца серии ничего не двигает."""
        live = self._groups.get(group_id)
        if live is None:
            raise KeyError("group_not_found")
        live.manifest.finalize()
        live.current_match_id = None
        logger.info(
            "[ArenaMatchManager] finish_series group=%s battles=%d/%d",
            group_id,
            live.manifest.manifest["results"]["battles_finished"],
            live.battles_planned,
        )
        return live.manifest.manifest

    def get_match(self, match_id: str) -> Optional[ArenaMatch]:
        return self._matches.get(match_id)

    def get_group(self, group_id: str) -> Optional[_GroupLive]:
        return self._groups.get(group_id)

    def current_match_of(self, group_id: str) -> Optional[ArenaMatch]:
        live = self._groups.get(group_id)
        if live is None or live.current_match_id is None:
            return None
        return self._matches.get(live.current_match_id)

    def list_groups(self) -> List[Dict[str, Any]]:
        out = []
        for live in self._groups.values():
            m = live.manifest.manifest
            out.append({
                "group_id": live.group_id,
                "status": "completed" if m.get("finished_at") else "running",
                "created_at": m.get("created_at"),
                "finished_at": m.get("finished_at"),
                "battles_planned": live.battles_planned,
                "battles_finished": m.get("results", {}).get("battles_finished", 0),
                "current_battle": live.current_index,
                "current_match_id": live.current_match_id,
                "manifest_path": str(live.manifest.manifest_path),
            })
        return out

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _build_match(
        self,
        group_id: str,
        live: _GroupLive,
        battle_index: int,
    ) -> ArenaMatch:
        spec = live.spec
        base_seed = int(spec.get("seed", 0) or 0)
        seed = base_seed + battle_index * 1009
        rng = random.Random(seed)

        p1_strategy = spec.get("deck_strategy_p1", spec.get("deck_strategy", "random_arenaenv"))
        p2_strategy = spec.get("deck_strategy_p2", spec.get("deck_strategy", "random_arenaenv"))
        p1_ids = _build_deck_ids(p1_strategy, self.catalog, spec.get("custom_deck_p1"), rng)
        p2_ids = _build_deck_ids(p2_strategy, self.catalog, spec.get("custom_deck_p2"), rng)

        # Уровни карт. По умолчанию — «средние уровни» боя: одно центральное
        # значение из {3,5,8} на обе колоды (игрок+бот), каждая карта ±2 от него.
        # Явные spec.card_levels_p1/p2 (напр. из тестов) уважаются — тогда берём их.
        explicit_p1 = spec.get("card_levels_p1")
        explicit_p2 = spec.get("card_levels_p2")
        if explicit_p1 or explicit_p2:
            p1_levels = explicit_p1 or {cid: 1 for cid in set(p1_ids)}
            p2_levels = explicit_p2 or {cid: 1 for cid in set(p2_ids)}
            battle_central_level = None
        else:
            battle_central_level = rng.choice(BATTLE_CENTRAL_LEVELS)
            p1_levels = _battle_card_levels(p1_ids, battle_central_level, rng, self.catalog)
            p2_levels = _battle_card_levels(p2_ids, battle_central_level, rng, self.catalog)
            logger.info(
                "[ArenaMatchManager] battle levels group=%s battle=%d central=%s "
                "p1=%s p2=%s",
                group_id, battle_index, battle_central_level, p1_levels, p2_levels,
            )
        all_levels = {cid: 1 for cid in set(p1_ids + p2_ids)}
        all_levels.update(p1_levels)
        all_levels.update(p2_levels)

        p1_deck = deck_from_card_ids(p1_ids, self.catalog.cards, user_levels=all_levels)
        p2_deck = deck_from_card_ids(p2_ids, self.catalog.cards, user_levels=all_levels)

        sp = _starting_player_id(spec, rng)
        game_mode = spec.get("game_mode", "classic")
        mode_config = resolve_mode_config(game_mode)

        gs = create_classic_game_state(
            1000, 2000, p1_deck, p2_deck,
            p1_is_bot=False, p2_is_bot=True,
            starting_player_id=sp, rng=rng,
        )
        arena = ArenaEnvironment(gs, classic_params=mode_config.classic, rng=rng)

        p2_model = spec.get("p2_model", "random")
        # Система сложностей удалена: модель всегда играет на максимум (argmax).
        # spec.difficulty больше не читается — фиксируем BOT_MAX_DIFFICULTY.
        difficulty = BOT_MAX_DIFFICULTY
        bot_policy = build_policy(
            {"name": p2_model, "difficulty": difficulty, "seed": seed},
            registry=self.registry,
        )

        # Профили для prebattle-рендера (arena.js читает opponent.name/title/...).
        p1_profile = {
            "name": str(spec.get("p1_name", "Вы")),
            "title": str(spec.get("p1_title", "")),
            "rarity": str(spec.get("p1_rarity", "")),
            "clan": str(spec.get("p1_clan", "")),
            "trophies": int(spec.get("p1_trophies", 0) or 0),
            "avatar_url": spec.get("p1_avatar_url"),
            "background_url": spec.get("p1_background_url"),
            "extra_pass": spec.get("p1_extra_pass"),
        }
        p2_profile = {
            "name": str(spec.get("p2_name") or _display_model_name(p2_model)),
            "title": str(spec.get("p2_title") or "BerserkInference (In-Game)"),
            "rarity": str(spec.get("p2_rarity", "common")),
            "clan": str(spec.get("p2_clan", "")),
            "trophies": int(spec.get("p2_trophies", 0) or 0),
            "avatar_url": spec.get("p2_avatar_url"),
            "background_url": spec.get("p2_background_url"),
            "extra_pass": spec.get("p2_extra_pass"),
        }

        battle_id = f"b_{uuid.uuid4().hex[:10]}"
        match_id = f"m_{uuid.uuid4().hex[:12]}"
        engine = RlhfBattleEngine(
            match_id=match_id,
            arena=arena,
            mode_config=mode_config,
            human_user_id=1000,
            bot_user_id=2000,
            p1_profile=p1_profile,
            p2_profile=p2_profile,
            bot_difficulty=difficulty,
            bot_brain_profile=p2_model,
            game_mode=game_mode,
        )
        engine.set_initial_decks(p1_ids, p2_ids)
        engine.p1_deck_source = dict(spec.get("p1_deck_source") or {"type": "random"})

        match = ArenaMatch(
            engine=engine,
            group_id=group_id,
            battle_id=battle_id,
            battle_index=battle_index,
            battles_planned=live.battles_planned,
            spec=spec,
            bot_policy=bot_policy,
            rng=rng,
            p1_deck_ids=p1_ids,
            p2_deck_ids=p2_ids,
            p1_levels=dict(p1_levels),
            p2_levels=dict(p2_levels),
            p1_deck_source=dict(spec.get("p1_deck_source") or {"type": "random"}),
            manifest=live.manifest,
            battle_start_monotonic=0.0,
        )
        # AnalyticsRecorder: каноничный NDJSON + фиксы аудита (F01/F02/F08/F12).
        from rlhf_env.components.analytics import AnalyticsRecorder
        recorder = AnalyticsRecorder(
            engine=engine,
            group_id=group_id,
            battle_id=battle_id,
            series_index=battle_index,
            p1_deck_cards=p1_deck,
            p2_deck_cards=p2_deck,
            p1_deck_ids=p1_ids,
            p2_deck_ids=p2_ids,
            p1_levels=dict(p1_levels),
            p2_levels=dict(p2_levels),
            catalog_hash=self.catalog_hash,
            p2_model=p2_model,
            difficulty=difficulty,
            bot_brain_profile=p2_model,
            game_mode=game_mode,
            side="p1",
            human_user_id=1000,
            bot_user_id=2000,
        )
        recorder.set_group_dir(live.manifest.group_dir)
        match.recorder = recorder

        # V5TraceRecorder: максимально полная omniscient-запись боя для V5-обучения
        # (turns/actions/meta + catalog.json на группу). См. rlhf-deck-import +
        # memory rlhf-training-data-v5-audit.
        try:
            from rlhf_env.components.v5_trace import V5TraceRecorder
            bot_policy_info = {
                "name": str(getattr(bot_policy, "name", p2_model)),
                "kind": str(getattr(bot_policy, "kind", "unknown")),
                "brain_profile": str(p2_model),
                "is_bot": True,
                "weights_path": getattr(bot_policy, "model_path", None),
                "weights_hash": getattr(bot_policy, "weights_hash", None),
                "weights_version": getattr(bot_policy, "weights_version", None),
            }
            v5_rec = V5TraceRecorder(
                engine=engine,
                group_id=group_id,
                battle_id=battle_id,
                battle_index=battle_index,
                battles_planned=live.battles_planned,
                p1_deck_ids=p1_ids,
                p2_deck_ids=p2_ids,
                p1_levels=p1_levels,
                p2_levels=p2_levels,
                catalog=self.catalog,
                catalog_hash=self.catalog_hash,
                group_dir=live.manifest.group_dir,
                base_seed=base_seed,
                battle_seed=seed,
                p2_model=p2_model,
                difficulty=difficulty,
                game_mode=game_mode,
                mode_config=mode_config,
                bot_policy_info=bot_policy_info,
            )
            match.v5_recorder = v5_rec
            if "v5_storage" not in live.manifest.manifest:
                live.manifest.manifest["v5_storage"] = v5_rec.manifest_storage_block()
                live.manifest._flush()
        except Exception:  # noqa: BLE001
            logger.warning("V5TraceRecorder init failed: %s", exc_info=True)

        self._matches[match_id] = match
        return match

    def _compute_catalog_hash(self) -> str:
        try:
            data = Path(self.cards_path).read_bytes()
            return hashlib.sha256(data).hexdigest()[:16]
        except Exception:
            return "unknown"


__all__ = ["ArenaMatchManager", "ArenaMatch"]