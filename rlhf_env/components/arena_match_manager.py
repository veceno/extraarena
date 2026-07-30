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
import json
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
from rlhf_env.components.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


def _model_name(p2_model: Any) -> Any:
    """Нормализует spec.p2_model к имени модели для UI/группировки.

    p2_model может быть строкой ('random' / 'extra-lr-v4-max') либо nested
    {name,path,kind}. Возвращает имя (str) или исходное значение, если это не
    dict/не содержит name.
    """
    if isinstance(p2_model, dict):
        nm = p2_model.get("name")
        return nm if nm else p2_model.get("path") or "custom"
    return p2_model


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
    # G4: True если V5TraceRecorder init упал → бой без v5/trace (для manifest/фильтров).
    v5_trace_init_failed: bool = False
    # p1 как RL-модель (model-vs-model): политика auto-play на стороне p1.
    # None для human/llm (ход через execute_human_action).
    p1_policy: Any = None
    # Кодовое имя «играющего» суб-агента оркестратора (semi-synthetic series).
    agent_name: Optional[str] = None

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
    agent_name: Optional[str] = None


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
        # Реестр кодовых имён суб-агентов оркестратора (semi-synthetic series).
        self.agent_registry = AgentRegistry(
            self.sessions_dir / "agents_index.json",
            sessions_dir=self.sessions_dir,
            cards_path=self.cards_path,
        )
        self._restore_groups_from_disk()

    def _restore_groups_from_disk(self) -> None:
        """Hydrate persisted group manifests after a process restart.

        Match objects are intentionally process-local and cannot be restored.
        A non-finalized series resumes from ``battles_finished``: the next call
        to :meth:`next_match` rebuilds precisely that battle index from the
        persisted spec/seed.  Completed groups are restored read-only so the
        browse UI and API retain history across restarts.
        """
        restored = 0
        for manifest_path in sorted(
            self.sessions_dir.glob("*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
        ):
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("manifest root must be an object")
                group_id = str(raw.get("group_id") or manifest_path.parent.name)
                if group_id != manifest_path.parent.name:
                    raise ValueError("manifest group_id does not match directory")
                spec = raw.get("spec") or {}
                if not isinstance(spec, dict):
                    raise ValueError("manifest spec must be an object")
                results = raw.get("results") or {}
                if not isinstance(results, dict):
                    raise ValueError("manifest results must be an object")

                planned = int(
                    results.get("battles_planned")
                    or spec.get("battles_planned")
                    or len(raw.get("battle_ids") or [])
                    or 1
                )
                if planned <= 0:
                    raise ValueError("battles_planned must be positive")
                battle_results = raw.get("battles_results") or []
                finished = max(
                    int(results.get("battles_finished", 0) or 0),
                    len(battle_results) if isinstance(battle_results, list) else 0,
                )
                finished = min(finished, planned)
                agent_name = raw.get("agent_name") or spec.get("agent_name")
                agent_name = str(agent_name).strip() if agent_name else None

                manifest = ManifestWriter(
                    group_id=group_id,
                    spec=dict(spec),
                    group_dir=manifest_path.parent,
                    repo_root=Path.cwd(),
                )
                self._groups[group_id] = _GroupLive(
                    group_id=group_id,
                    spec=dict(spec),
                    manifest=manifest,
                    battles_planned=planned,
                    # No live match survives a restart.  For N persisted results,
                    # next_match must construct zero-based battle index N.
                    current_index=finished - 1,
                    current_match_id=None,
                    agent_name=agent_name,
                )
                restored += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ArenaMatchManager] skipping invalid persisted manifest %s: %s",
                    manifest_path,
                    exc,
                )
        if restored:
            logger.info(
                "[ArenaMatchManager] restored %d persisted groups from %s",
                restored,
                self.sessions_dir,
            )

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------

    def create_series(self, spec: Dict[str, Any]) -> ArenaMatch:
        """Создаёт новую группу и первый бой серии. Возвращает ArenaMatch."""
        group_id = uuid.uuid4().hex[:12]
        group_dir = self.sessions_dir / group_id
        # Кодовое имя суб-агента: явное (spec.agent_name) или auto-assign из пула
        # (фиксированный список + названия карт + random-fallback). Пиним к группе.
        agent_name = str(spec.get("agent_name") or "").strip()
        if agent_name:
            if not self.agent_registry.claim(agent_name):
                # имя занято — auto-assign вместо отказа (оркестратор не должен падать).
                logger.warning(
                    "[ArenaMatchManager] agent_name=%s busy, auto-assigning instead", agent_name
                )
                agent_name = self.agent_registry.claim_auto()
        else:
            agent_name = self.agent_registry.claim_auto()
        spec["agent_name"] = agent_name
        # F5: pin_group переносим внутрь try — если он упадёт (fs/flock ошибка),
        # уже claim-ed имя (group_id=None) надо освободить по agent_name, а не
        # по group_id (release_group(group_id) его не найдёт).
        try:
            self.agent_registry.pin_group(agent_name, group_id)
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
                agent_name=agent_name,
            )
            self._groups[group_id] = live
            match = self._build_match(group_id, live, battle_index=0)
        except Exception:
            # BUG3 rollback: _build_match / pin_group упал (плохая модель/path/kind
            # или fs-ошибка) — не оставляем занятое кодовое имя и группу-полутруп.
            # Иначе start_series возвращал ошибку без group_id, имя утекало навсегда.
            self._groups.pop(group_id, None)
            # purge=True: боя не было → истории нет, имя возвращаем в пул целиком
            # (а не помечаем finished — иначе claim_auto пропустит «занятое» имя).
            # Сначала по group_id (нормальный путь), затем — на случай, что pin_group
            # упал ДО записи group_id — освобождаем явно по имени (F5).
            try:
                self.agent_registry.release_group(group_id, purge=True)
            except Exception:  # noqa: BLE001
                logger.warning("[ArenaMatchManager] release_group failed group=%s: %s", group_id, exc_info=True)
            try:
                self.agent_registry.release(agent_name, purge=True)
            except Exception:  # noqa: BLE001
                logger.warning("[ArenaMatchManager] release(%s) failed: %s", agent_name, exc_info=True)
            try:
                import shutil as _sh
                _sh.rmtree(group_dir, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass
            raise
        live.current_index = 0
        live.current_match_id = match.engine.match_id
        logger.info(
            "[ArenaMatchManager] create_series group=%s agent=%s match=%s p2_model=%s battles=%d",
            group_id, agent_name, match.engine.match_id, spec.get("p2_model"), live.battles_planned,
        )
        return match

    def next_match(self, group_id: str) -> Optional[ArenaMatch]:
        """Возвращает следующий бой серии или None, если серия закончилась."""
        live = self._groups.get(group_id)
        if live is None:
            return None
        # Explicitly finished (including early-finished) series are history,
        # never resumable merely because battles_finished < battles_planned.
        if live.manifest.manifest.get("finished_at"):
            return None
        # Never advance over an unfinished human/LLM battle.  Doing so replaces
        # ``current_match_id`` and leaves the unplayed match unfinalized, which
        # silently creates empty or partial training groups.
        if live.current_match_id is not None:
            current = self._matches.get(live.current_match_id)
            if current is not None and not bool(current.engine.is_ended):
                raise RuntimeError("current_battle_in_progress")
        next_index = live.current_index + 1
        if next_index >= live.battles_planned:
            live.manifest.finalize()
            live.current_match_id = None
            # F7(audit): НЕ удаляем ended-match из _matches здесь — клиент ещё может
            # звать get_match_status на только что завершённом матче (финальный
            # winner/is_ended). Eager pop ломал late status-чтения → ложный "draw".
            # Bounded growth закрывается в finish_series (явное закрытие серии).
            # F1: естественное завершение серии (через next_battle) — освобождаем
            # кодовое имя агента (помечаем finished, история доступна через status).
            # finish_series — досрочное закрытие, но доигранную до конца серию
            # оркестратор не обязан звать finish_series; без этого имя утекало.
            if live.agent_name:
                try:
                    self.agent_registry.release_group(group_id)
                except Exception:  # noqa: BLE001 — release не должен маскировать успех серии
                    logger.warning("[ArenaMatchManager] release_group on natural completion failed group=%s: %s", group_id, exc_info=True)
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
        ended_id = live.current_match_id
        live.current_match_id = None
        # F7(audit): убираем ended-match из _matches (bounded growth).
        if ended_id is not None:
            self._matches.pop(ended_id, None)
        # Освобождаем кодовое имя суб-агента (серия закрыта).
        # F8(audit): try/except как в next_match/reap_completed — persist/lock
        # failure не должен утекать codename, когда манифест уже финализирован.
        if live.agent_name:
            try:
                self.agent_registry.release_group(group_id)
            except Exception:  # noqa: BLE001 — release не должен маскировать успех серии
                logger.warning(
                    "[ArenaMatchManager] finish_series release_group failed group=%s: %s",
                    group_id, exc_info=True)
        logger.info(
            "[ArenaMatchManager] finish_series group=%s battles=%d/%d",
            group_id,
            live.manifest.manifest["results"]["battles_finished"],
            live.battles_planned,
        )
        return live.manifest.manifest

    def reap_completed(self, group_id: str) -> bool:
        """Self-healing release: если текущий бой серии завершён И серия доиграна
        (battles_finished >= battles_planned), освобождаем кодовое имя агента и
        финализируем манифест.

        Покрывает клиента, который после последнего боя не зовёт next_battle/
        finish_series (вкл. 1-боевые серии — оркестратор играет бой и уходит) —
        иначе имя утекало навсегда в agents_index.json (busy=True для уже
        завершённой серии, codename-pool истощался, на краше клиента — тем более).
        Вызывается из MCP read-paths (get_match_status/get_agent_status/
        get_battle_group_status/list_active_series). Идемпотентно.

        Cross-process recovery (упавший процесс оставил busy-запись, а новый
        процесс не имеет _groups-записи) — см. AgentRegistry._self_heal_locked.
        Возвращает True если(reaped."""
        live = self._groups.get(group_id)
        if live is None or live.current_match_id is None:
            return False
        match = self._matches.get(live.current_match_id)
        if match is None or not getattr(match.engine, "is_ended", False):
            return False
        res = (live.manifest.manifest.get("results") or {})
        if int(res.get("battles_finished", 0) or 0) < live.battles_planned:
            # серия не доиграна — клиент должен звать next_battle для след. боя.
            return False
        live.manifest.finalize()
        live.current_match_id = None
        # F7(audit): НЕ удаляем ended-match из _matches в reap — reap вызывается
        # из MCP read-paths (get_match_status и т.д.), и eager pop ломал поздний
        # get_match_status на только что завершённом матче (возвращал
        # match_not_found → is_ended=None/winner=None → ложный "draw"). Match
        # остаётся читаемым; bounded growth закрывается в finish_series.
        if live.agent_name:
            try:
                self.agent_registry.release_group(group_id)
            except Exception:  # noqa: BLE001 — release не должен маскировать успех
                logger.warning(
                    "[ArenaMatchManager] reap release_group failed group=%s: %s",
                    group_id, exc_info=True)
        logger.info(
            "[ArenaMatchManager] reaped completed series group=%s agent=%s",
            group_id, live.agent_name)
        return True

    def reap_all_completed(self) -> int:
        """reap_completed по всем группам. Возвращает число освобождённых."""
        return sum(1 for gid in list(self._groups) if self.reap_completed(gid))

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
        lives = sorted(
            self._groups.values(),
            key=lambda item: str(item.manifest.manifest.get("created_at") or ""),
            reverse=True,
        )
        for live in lives:
            m = live.manifest.manifest
            res = m.get("results", {}) or {}
            spec = m.get("spec", {}) or {}
            # battle_tag — из первого записанного боя (real tag: rl-vs-bot /
            # rl-vs-rl / llm-vs-bot / human-vs-bot). До первого боя — эвристика
            # по p1_actor_type (p2 пока неизвестен точно).
            results = m.get("battles_results", []) or []
            real_tag = next((r.get("battle_tag") for r in results if r.get("battle_tag")), None)
            p1_act = spec.get("p1_actor_type")
            battle_tag = real_tag or (f"{p1_act}-vs-bot" if p1_act else None)
            out.append({
                "group_id": live.group_id,
                "agent_name": live.agent_name,
                "status": "completed" if m.get("finished_at") else "running",
                "created_at": m.get("created_at"),
                "finished_at": m.get("finished_at"),
                "battles_planned": live.battles_planned,
                "battles_finished": res.get("battles_finished", 0),
                "wins": res.get("p1_wins", 0),
                "losses": res.get("p2_wins", 0),
                "draws": res.get("draws", 0),
                "p1_actor_type": p1_act,
                # F9(audit): нормализуем p2_model к имени — spec.p2_model может быть
                # nested {name,path,kind}; иначе by-model группировка в list_active_series
                # получала str(dict) как ключ.
                "p2_model": _model_name(spec.get("p2_model")),
                "battle_tag": battle_tag,
                "current_battle": max(live.current_index, 0),
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

        # --- Акторы + политики ------------------------------------------------
        # p1: human (браузер) | llm (MCP) | rl (наша RL-модель auto-play, model-vs-
        # model). p2 всегда bot (наша RL/кастом-модель). Система сложностей удалена:
        # модель всегда играет на максимум (argmax); difficulty фиксируется "max".
        p1_actor = str(spec.get("p1_actor_type", "human")).lower()
        if p1_actor not in ("human", "llm", "rl"):
            p1_actor = "human"
        p2_actor = "bot"
        difficulty = BOT_MAX_DIFFICULTY

        # p2 (оппонент): имя | nested {name,path,kind} | плоские p2_model_path/kind.
        # path+kind пробрасываются в build_policy (custom model by path+adapter).
        p2_spec = self._resolve_policy_spec(spec, "p2", seed, default="random")
        p2_model = p2_spec["name"]
        bot_policy = build_policy(p2_spec, registry=self.registry)

        # p1 как RL-модель: строим политику для auto-play (MatchRunner.run_auto).
        p1_policy = None
        if p1_actor == "rl":
            p1_spec = self._resolve_policy_spec(spec, "p1", seed, default="random")
            p1_policy = build_policy(p1_spec, registry=self.registry)

        # battle_tag: <p1>-vs-rl если p2 — RL/onnx-модель (v4/legacy_onnx/v5/...),
        # иначе <p1>-vs-bot (baseline: random/greedy_face/end_turn). Эвристика по
        # p2_policy.kind, НЕ зависит от p1 — иначе llm/human vs onnx-RL ошибочно
        # тегировался «-vs-bot» (бой против V3/V4-max помечался llm-vs-bot, хотя
        # p2 — обученная onnx-модель, не baseline). bugfix: условие на p2_kind
        # без gate p1_actor=="rl".
        p2_kind = str(getattr(bot_policy, "kind", "") or "")
        if p2_kind in ("random", "greedy_face", "end_turn", ""):
            battle_tag = f"{p1_actor}-vs-bot"
        else:
            battle_tag = f"{p1_actor}-vs-rl"

        gs = create_classic_game_state(
            1000, 2000, p1_deck, p2_deck,
            p1_is_bot=(p1_actor == "rl"), p2_is_bot=True,
            starting_player_id=sp, rng=rng,
        )
        arena = ArenaEnvironment(gs, classic_params=mode_config.classic, rng=rng)

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
            p1_actor_type=p1_actor,
            p2_actor_type=p2_actor,
        )
        # battle_tag переопределяем явно: engine по умолчанию считает f"{p1}-vs-{p2}",
        # но rl-vs-rl (обе модели RL) определяется здесь по p2_policy.kind.
        engine.battle_tag = battle_tag
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
            p1_policy=p1_policy,
            agent_name=live.agent_name,
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
            # provenance p1-политики (только для model-vs-model, p1_actor_type='rl').
            p1_policy_info = None
            if p1_policy is not None:
                p1_policy_info = {
                    "name": str(getattr(p1_policy, "name", "p1-rl")),
                    "kind": str(getattr(p1_policy, "kind", "rl")),
                    "is_bot": True,
                    "weights_path": getattr(p1_policy, "model_path", None),
                    "weights_hash": getattr(p1_policy, "weights_hash", None),
                    "weights_version": getattr(p1_policy, "weights_version", None),
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
                p1_actor_type=p1_actor,
                p2_actor_type=p2_actor,
                battle_tag=battle_tag,
                p1_policy_info=p1_policy_info,
                agent_name=live.agent_name,
            )
            match.v5_recorder = v5_rec
            match.v5_trace_init_failed = False
            if "v5_storage" not in live.manifest.manifest:
                live.manifest.manifest["v5_storage"] = v5_rec.manifest_storage_block()
                live.manifest._flush()
        except Exception:  # noqa: BLE001
            # G4: не глушить тихо — бой останется без v5/trace; сигнализируем
            # через manifest.v5_trace_ok + отдельный атрибут для collection-скриптов.
            logger.error("V5TraceRecorder init FAILED — battle will lack v5/ trace: %s",
                         exc_info=True)
            match.v5_recorder = None
            match.v5_trace_init_failed = True

        self._matches[match_id] = match
        return match

    # ------------------------------------------------------------------
    # Custom model by path+adapter (semi-synthetic orchestrator)
    # ------------------------------------------------------------------
    def _safe_model_path(self, path: str) -> str:
        """Path-traversal защита: кастомный путь модели должен лежать под
        models_dir или repo root. Относительный путь разрешается от repo root
        (cwd), если он естественно попадает в models_dir (напр.
        `ai/models/foo.onnx` — чтобы не удваивался в `models_dir/ai/models/...`);
        иначе — от models_dir (для bare-имён `foo.onnx`).
        Возвращает абсолютный путь; ValueError при выходе за допустимые корни.

        Разрешение НЕ зависит от существования файла на диске: кастомный путь
        может ещё не существовать (раньше проверка ``cand_cwd.exists()``
        пропускала удвоение `ai/models/X` → `models_dir/ai/models/X`, когда файл
        отсутствовал — regression ``test_safe_model_path_repo_relative_not_doubled``).
        """
        p = Path(path).expanduser()
        models_root = self.models_dir.resolve()
        cwd_root = Path.cwd().resolve()
        allowed_roots = [models_root, cwd_root]
        if p.is_absolute():
            chosen = p.resolve()
        else:
            cand_cwd = (cwd_root / p).resolve()
            cand_models = (models_root / p).resolve()
            # repo-relative путь, естественно попадающий в models_dir (напр.
            # "ai/models/X") → разрешаем от cwd, НЕ удваивая под models_dir.
            if cand_cwd == models_root or models_root in cand_cwd.parents:
                chosen = cand_cwd
            # bare-имя (или путь вне models_dir) → под models_dir.
            elif cand_models == models_root or models_root in cand_models.parents:
                chosen = cand_models
            # прочий repo-relative путь под cwd → от cwd.
            elif cand_cwd == cwd_root or cwd_root in cand_cwd.parents:
                chosen = cand_cwd
            else:
                chosen = cand_cwd  # провалит allowed-root проверку → ValueError
        if not any(chosen == root or root in chosen.parents for root in allowed_roots):
            raise ValueError(
                f"custom model path {path!r} resolves outside allowed roots "
                f"(models_dir / repo root)"
            )
        return str(chosen)

    def _resolve_policy_spec(
        self, spec: Dict[str, Any], side: str, seed: int, *, default: str = "random"
    ) -> Dict[str, Any]:
        """Собирает spec для build_policy из полей серии.

        side ∈ {"p1","p2"}. Поддерживаются:
          - {side}_model как строка-имя (baseline / имя из registry);
          - {side}_model как объект {name, path, kind} (nested форма);
          - плоские {side}_model_path / {side}_model_kind (custom by path+adapter).
        path+kind пробрасываются в build_policy (custom model by path+adapter).
        """
        model = spec.get(f"{side}_model")
        path = spec.get(f"{side}_model_path")
        kind = spec.get(f"{side}_model_kind")
        if isinstance(model, dict):
            name = model.get("name") or f"{side}_custom"
            path = model.get("path") or path
            kind = model.get("kind") or kind
        else:
            name = model
        if not name:
            name = default
        out: Dict[str, Any] = {"name": str(name), "difficulty": BOT_MAX_DIFFICULTY, "seed": seed}
        if path:
            out["path"] = self._safe_model_path(str(path))
        if kind:
            out["kind"] = str(kind)
        return out

    def _compute_catalog_hash(self) -> str:
        try:
            data = Path(self.cards_path).read_bytes()
            return hashlib.sha256(data).hexdigest()[:16]
        except Exception:
            return "unknown"


__all__ = ["ArenaMatchManager", "ArenaMatch"]
