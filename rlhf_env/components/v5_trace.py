"""V5TraceRecorder — максимально полная, универсальная запись данных боя для V5-обучения.

Пишет в отдельную директорию внутри директории боя:
    sessions/<group_id>/battles/<battle_id>/v5/
        meta.json     — провенанс боя + колоды + сиды + версия движка
        turns.jsonl   — одно состояние на ГЛОБАЛЬНЫЙ ход (оба игрока ПОЛНОСТЬЮ,
                        без сокрытия: hand/deck/board/hero/graveyard/mana),
                        снимается в начале хода (после добора, до первого действия)
        actions.jsonl — одно действие на строку: сырое action_json + V5-history
                        событие (actor_id, action_id, action_type, source_card,
                        target_card, 5 дельт) + ПОЛНОЕ omniscient pre_state/post_state
                        обоих игроков (V5 obs считается пошагово, не на ход — поэтому
                        состояние нужно перед каждым действием, а не только на ход)

Это «источник», а не пред-закодированный V5-тензор: будущий offline-мост
(см. memory rlhf-training-data-v5-audit) реконструирует GameState из pre_state и
кодирует encode_observation_v5 для любой комбинации InfoModeV5 флагов
(own/enemy hand/deck known) БЕЗ запуска прод-движка и БЕЗ совпадения версий.
Дельты и action_id сохранены напрямую (где безопасно вычислимы) чтобы мост не
повторял хрупкую логику classic_actions_v1 tcode.

Дельты сохранены напрямую (формулы из reward_v5.py, см. ниже) — мосту не нужно
их перевычислять. legal_action_index + action_native (engine-native to_dict
выбранного действия, единообразно для human+bot) + полный legal_actions сохранены,
чтобы мост построил action-mask и V5 tcode (0..600, dst[6]=action_id/600) из
action_native + pre_state БЕЗ запуска движка. Сам tcode НЕ сохраняется напрямую:
его кодек живёт в core/ai (classic_actions_v1) — не тащим зависимость и не
дублируем хрупкую layout-логику; мост считает tcode однократно по action_native.
  board_power            = sum(max(0, atk) * max(0, hp) over board)
  enemy_hero_hp_delta    = pre.enemy_hero_hp - post.enemy_hero_hp
  own_hero_hp_delta      = pre.my_hero_hp    - post.my_hero_hp
  my_board_count_delta  = post.my_board_count    - pre.my_board_count
  enemy_board_count_delta= post.enemy_board_count- pre.enemy_board_count
  board_power_delta      = (post.my_bp - post.enemy_bp) - (pre.my_bp - pre.enemy_bp)

visibility="omniscient_offline_only" на каждой строке/файле: полный соперник
виден; этот поток НИКОГДА не отдаётся живому webapp-клиенту (для него остаётся
actor-perspective analytics <bid>.jsonl с сокрытой рукой оппонента).
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.actions import AttackAction, EndTurnAction, ManaDrawAction, PlayCardAction

logger = logging.getLogger(__name__)

V5_STORAGE_SCHEMA = "rlhf_v5_storage_v1"
CATALOG_SCHEMA = "rlhf_catalog_v1"
CARD_PARAMS_SCHEMA = "train_v3_card_params_v1"
DECK_PARAMS_SCHEMA = "train_v3_deck_params_v1"
CARD_SHAPE_VERSION = "classic_card_shape_v1"
ACTIONS_VERSION = "classic_actions_v1"
OBS_VERSION = "classic_obs_v1"
VISIBILITY = "omniscient_offline_only"
METRONOME_MIN_LABEL_MS = 100
METRONOME_MAX_LABEL_MS = 25_000

_GIT_SHA_CACHE: Optional[str] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_short_sha() -> str:
    """Best-effort core-engine commit (для version-match при replay)."""
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=str(Path(__file__).resolve().parents[2]),
            text=True, timeout=3,
        ).strip()
        _GIT_SHA_CACHE = sha
        return sha
    except Exception:  # noqa: BLE001
        _GIT_SHA_CACHE = "unknown"
        return "unknown"


def _board_power(board) -> float:
    return float(sum(max(0, int(getattr(c, "attack", 0))) * max(0, int(getattr(c, "hp", 0))) for c in board))


class V5TraceRecorder:
    """Один боевой V5-трейс. Создаётся в ArenaMatchManager._build_match."""

    def __init__(
        self,
        *,
        engine,
        group_id: str,
        battle_id: str,
        battle_index: int,
        battles_planned: int,
        p1_deck_ids: List[int],
        p2_deck_ids: List[int],
        p1_levels: Dict[int, int],
        p2_levels: Dict[int, int],
        catalog,                      # rlhf CardCatalog
        catalog_hash: str,
        group_dir: Path,
        base_seed: int,
        battle_seed: int,
        p2_model: str,
        difficulty: str,
        game_mode: str,
        mode_config,
        bot_policy_info: Dict[str, Any],
        p1_user_id: int = 1000,
        p2_user_id: int = 2000,
        # Теггинг акторов для V5 (decision_source / battle_tag).
        p1_actor_type: str = "human",
        p2_actor_type: str = "bot",
        battle_tag: str = "human-vs-bot",
        # p1 как RL-модель (model-vs-model): provenance p1-политики + кодовое имя агента.
        p1_policy_info: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.group_id = group_id
        self.battle_id = battle_id
        self.battle_index = battle_index
        self.battles_planned = battles_planned
        self.p1_deck_ids = list(p1_deck_ids)
        self.p2_deck_ids = list(p2_deck_ids)
        self.p1_levels = dict(p1_levels)
        self.p2_levels = dict(p2_levels)
        self.catalog = catalog
        self.catalog_hash = catalog_hash
        self.group_dir = Path(group_dir)
        self.base_seed = int(base_seed)
        self.battle_seed = int(battle_seed)
        self.p2_model = p2_model
        self.difficulty = difficulty
        self.game_mode = game_mode
        self.mode_config = mode_config
        self.bot_policy_info = bot_policy_info
        self.p1_user_id = p1_user_id
        self.p2_user_id = p2_user_id
        self.p1_actor_type = str(p1_actor_type)
        self.p2_actor_type = str(p2_actor_type)
        self.battle_tag = str(battle_tag)
        self.p1_policy_info = p1_policy_info
        self.agent_name = agent_name

        self.v5_dir = self.group_dir / "battles" / battle_id / "v5"
        self.v5_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.v5_dir / "meta.json"
        self.turns_path = self.v5_dir / "turns.jsonl"
        self.actions_path = self.v5_dir / "actions.jsonl"

        self._seq = 0
        self._last_turn_snapshot: Optional[int] = None
        self._buffer: List[Dict[str, Any]] = []        # action rows awaiting after_action
        self._pre_snapshots: Dict[int, Dict[str, Any]] = {}  # handle -> pre reward snapshot
        self._finalized = False
        self._created_at = _utc_now_iso()
        self._start_monotonic = time.monotonic()
        # catalog.json — флаг успешной записи (G5: retry на неудаче, skip на успехе).
        self._catalog_written = False
        self._nemesis_collector = None

        # catalog.json — once per group (idempotent by existence).
        self.catalog_path = self.group_dir / "catalog.json"
        self._write_catalog()
        try:
            from core.nemesis_dataset import NemesisBattleCollector

            state = self.engine._arena.state

            def model_provenance(
                info: Optional[Dict[str, Any]],
            ) -> Dict[str, Any]:
                source = info if isinstance(info, dict) else {}
                return {
                    "model_id": source.get("name"),
                    "model_family": (
                        "extra-lr" if source else None
                    ),
                    "model_version": source.get("weights_version"),
                    "checkpoint_id": source.get("weights_version"),
                    "weights_hash": source.get("weights_hash"),
                    "adapter_kind": source.get("kind"),
                }

            starting_player = (
                "p1"
                if int(state.current_turn_owner_id) == int(p1_user_id)
                else "p2"
            )
            nemesis_meta = {
                "battle_id": self.battle_id,
                "match_id": str(self.engine.match_id),
                "started_at": self._created_at,
                "game_mode": self.game_mode,
                "ruleset": getattr(self.engine, "ruleset", "classic"),
                "catalog_hash": self.catalog_hash,
                "card_params_schema": CARD_PARAMS_SCHEMA,
                "deck_params_schema": DECK_PARAMS_SCHEMA,
                "starting_player": starting_player,
                "p1_user_id": int(p1_user_id),
                "p2_user_id": int(p2_user_id),
                "p1_actor_type": self.p1_actor_type,
                "p2_actor_type": self.p2_actor_type,
                "p1_deck": [
                    {
                        "slot": slot,
                        "card_id": int(card_id),
                        "level": int(self.p1_levels.get(card_id, 1)),
                    }
                    for slot, card_id in enumerate(self.p1_deck_ids)
                ],
                "p2_deck": [
                    {
                        "slot": slot,
                        "card_id": int(card_id),
                        "level": int(self.p2_levels.get(card_id, 1)),
                    }
                    for slot, card_id in enumerate(self.p2_deck_ids)
                ],
                "model_provenance": {
                    "p1": model_provenance(self.p1_policy_info),
                    "p2": model_provenance(self.bot_policy_info),
                },
                "aux_model_provenance": {"p1": {}, "p2": {}},
                "dataset_generation": 1,
            }
            self._nemesis_collector = (
                NemesisBattleCollector.from_v5_meta(
                    nemesis_meta,
                    feature_cutoff_at=self._created_at,
                    source="rlhf_headless",
                    campaign_id=self.group_id,
                    seed=self.battle_seed,
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "headless Nemesis collector init failed: %s",
                self.battle_id,
                exc_info=True,
            )

        self._write_meta(terminal=False)

    # ------------------------------------------------------------------
    # catalog
    # ------------------------------------------------------------------
    def _write_catalog(self) -> None:
        # G5: idempotent — skip если уже успешно записан (этим или прошлым боем
        # группы). При неудаче НЕ помечаем written → следующий бой retry'нет.
        if self._catalog_written or self.catalog_path.exists():
            self._catalog_written = True
            return
        try:
            cards = {str(cid): self.catalog.card(int(cid)) for cid in self.catalog.card_ids}
        except Exception:  # noqa: BLE001
            logger.warning("v5 catalog dump failed: %s", exc_info=True)
            return
        payload = {
            "schema": CATALOG_SCHEMA,
            "catalog_hash": self.catalog_hash,
            "generated_at": _utc_now_iso(),
            "cards": cards,
        }
        try:
            tmp = self.catalog_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.catalog_path)
            self._catalog_written = True
        except Exception:  # noqa: BLE001
            logger.warning("v5 catalog write failed: %s", exc_info=True)

    # ------------------------------------------------------------------
    # meta
    # ------------------------------------------------------------------
    def _write_meta(self, *, terminal: bool) -> None:
        st = self.engine._arena.state if self.engine._arena else None
        meta: Dict[str, Any] = {
            "schema_version": V5_STORAGE_SCHEMA,
            "visibility": VISIBILITY,
            "group_id": self.group_id,
            "battle_id": self.battle_id,
            "battle_index": int(self.battle_index),
            "battles_planned": int(self.battles_planned),
            "created_at": self._created_at,
            "finished_at": _utc_now_iso() if terminal else None,
            "status": (st.status.name.lower() if st is not None else "ongoing"),
            "winner_user_id": None,
            "duration_seconds": None,
            "turns": None,
            "rng_seed": {"base_seed": self.base_seed, "battle_seed": self.battle_seed},
            "engine_version": {"core_engine_commit": _git_short_sha()},
            "ruleset": getattr(self.engine, "ruleset", "classic"),
            "game_mode": self.game_mode,
            "mode_config": _serialize_mode_config_safe(self.mode_config),
            "bot_policy": self.bot_policy_info,
            "p1_policy": self.p1_policy_info,
            "agent_name": self.agent_name,
            "p2_model": self.p2_model,
            "difficulty": self.difficulty,
            "catalog_hash": self.catalog_hash,
            "catalog_path": str(self.catalog_path.relative_to(self.group_dir)) if self.catalog_path.exists() else None,
            "catalog_path_base": "group_dir",  # catalog_path разрешается от group_dir (sessions/<gid>/)
            "card_params_schema": CARD_PARAMS_SCHEMA,
            "deck_params_schema": DECK_PARAMS_SCHEMA,
            "card_shape_version": CARD_SHAPE_VERSION,
            "actions_version": ACTIONS_VERSION,
            "obs_version": OBS_VERSION,
            # p1_deck/p2_deck — selection-list provenance (pre-shuffle): card_id+level
            # в порядке выбора, instance_id=null. АВТОРИТАТИВНОЕ перетасованное
            # начальное состояние колоды (с instance_id) — в turns.jsonl turn_number=1
            # (snapshot p1.deck/p2.deck). Не использовать meta-колоды для реконструкции.
            "p1_deck": [{"card_id": int(cid), "level": int(self.p1_levels.get(cid, 1)), "instance_id": None} for cid in self.p1_deck_ids],
            "p2_deck": [{"card_id": int(cid), "level": int(self.p2_levels.get(cid, 1)), "instance_id": None} for cid in self.p2_deck_ids],
            "p1_user_id": int(self.p1_user_id),
            "p2_user_id": int(self.p2_user_id),
            "p1_is_bot": self.p1_actor_type in {"bot", "rl", "llm"},
            "p2_is_bot": self.p2_actor_type in {"bot", "rl", "llm"},
            # Теггинг актора (G15): human | llm | bot | rl. p1_is_bot/p2_is_bot
            # оставлены для back-compat (booleans), но actor_type — каноничное
            # поле для фильтрации Phase A (human) vs semi-synth (llm) vs model-vs-model (rl).
            "p1_actor_type": self.p1_actor_type,
            "p2_actor_type": self.p2_actor_type,
            "battle_tag": self.battle_tag,
            # G4: маркер что v5-trace реально создан (для collection-скриптов).
            "v5_trace_present": True,
        }
        if self._nemesis_collector is not None:
            meta["nemesis_record"] = self._nemesis_collector.snapshot()
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.meta_path)

    # ------------------------------------------------------------------
    # snapshots
    # ------------------------------------------------------------------
    def _snapshot_card(self, card) -> Optional[Dict[str, Any]]:
        if card is None:
            return None
        try:
            return self.engine._snapshot_card(card)
        except Exception:  # noqa: BLE001
            return None

    def _snapshot_player(self, p) -> Dict[str, Any]:
        return {
            "user_id": int(p.user_id),
            "is_bot": bool(p.is_bot),
            "hero": self._snapshot_card(p.hero),
            "mana": int(getattr(p, "mana", 0)),
            "max_mana": int(getattr(p, "max_mana", 0)),
            "trophies": int(getattr(p, "trophies", 0)),
            # replacement_status делает сдачу видимой в snapshot (SURRENDERED),
            # т.к. mark_surrender не мутирует state.status — терминал p2_win
            # дерайвит _finalize, но сам факт сдачи виден здесь.
            "replacement_status": str(getattr(p, "replacement_status", "ACTIVE") or "ACTIVE").rsplit(".", 1)[-1].lower(),
            "hand": [self._snapshot_card(c) for c in p.hand],
            # deck — ПОЛНЫЙ snapshot (scaled atk/hp/mechanics/card_type), симметрично
            # hand/board/graveyard. encode_card_shape для PRIVATE_INFO own/enemy_deck
            # читает attack/hp/max_hp/level/mechanics/card_type — всё тут; бриджу не
            # нужно повторять core/card_scaling (каталог даёт только base-статы).
            "deck": [self._snapshot_card(c) for c in p.deck],
            "board": [self._snapshot_card(c) for c in p.board],
            "graveyard": [self._snapshot_card(c) for c in p.graveyard],
            # V5 global-канал: сколько доборов за ману сделано в текущем своём ходу.
            "mana_draw_count_this_turn": int(getattr(p, "mana_draw_count_this_turn", 0)),
        }

    def _snapshot_state(self) -> Dict[str, Any]:
        st = self.engine._arena.state
        # G2: захватываем action_history / native history / structured V5
        # history / pending_card_feedback
        # чтобы offline-bridge мог реконструировать ПОЛНЫЙ GameState (вкл.
        # 20×162 rich history-channel для encode_observation_v5) из pre_state.
        # action_history — Deque[tuple[str,str]] (UI-лог: rebirth/crime/
        # mana_draw entries из core/engine.py); history — native List[Dict];
        # v5_history_events — model-facing Deque[Dict](maxlen=20).
        try:
            action_history = [list(t) for t in st.action_history]
        except Exception:  # noqa: BLE001
            action_history = []
        try:
            history = list(getattr(st, "history", []) or [])
        except Exception:  # noqa: BLE001
            history = []
        try:
            v5_history_events = [
                self._snapshot_v5_history_event(event)
                for event in getattr(st, "v5_history_events", ())
            ]
        except Exception:  # noqa: BLE001
            v5_history_events = []
        try:
            pending_feedback = list(getattr(st, "pending_card_feedback_events", []) or [])
        except Exception:  # noqa: BLE001
            pending_feedback = []
        return {
            "turn_number": int(st.turn_number),
            "current_turn_owner_id": int(st.current_turn_owner_id),
            "status": st.status.name.lower(),
            "p1": self._snapshot_player(st.p1),
            "p2": self._snapshot_player(st.p2),
            "action_history": action_history,
            "history": history,
            "v5_history_events": v5_history_events,
            "pending_card_feedback_events": pending_feedback,
            "visibility": VISIBILITY,
        }

    def _snapshot_v5_history_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize both JSON-safe live events and older CardInstance events."""

        out = dict(event)
        for key in ("source_card", "target_card"):
            card = out.get(key)
            if card is not None and not isinstance(card, dict):
                out[key] = self._snapshot_card(card)
        return out

    def _reward_snapshot(self, state, player_id: int) -> Dict[str, Any]:
        me = state.p1 if state.p1.user_id == player_id else state.p2
        enemy = state.p2 if state.p1.user_id == player_id else state.p1
        return {
            "my_hero_hp": int(me.hero.hp),
            "enemy_hero_hp": int(enemy.hero.hp),
            "my_board_count": len(me.board),
            "enemy_board_count": len(enemy.board),
            "my_board_power": _board_power(me.board),
            "enemy_board_power": _board_power(enemy.board),
        }

    def _deltas(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        pre_bd = pre["my_board_power"] - pre["enemy_board_power"]
        post_bd = post["my_board_power"] - post["enemy_board_power"]
        return {
            "enemy_hero_hp_delta": int(pre["enemy_hero_hp"] - post["enemy_hero_hp"]),
            "own_hero_hp_delta": int(pre["my_hero_hp"] - post["my_hero_hp"]),
            "my_board_count_delta": int(post["my_board_count"] - pre["my_board_count"]),
            "enemy_board_count_delta": int(post["enemy_board_count"] - pre["enemy_board_count"]),
            "board_power_delta": float(post_bd - pre_bd),
        }

    # ------------------------------------------------------------------
    # action_id + source/target (V5 _build_event semantics, PRE-execute)
    # ------------------------------------------------------------------
    def _resolve_action_id(self, user_id: int, action_json: Dict[str, Any], provided: Optional[int],
                           legal: Optional[List[Any]] = None) -> Optional[int]:
        if provided is not None:
            return int(provided)
        if legal is None:
            try:
                legal = self.engine.get_legal_actions_raw(user_id)
            except Exception:  # noqa: BLE001
                return None
        atype = action_json.get("type")
        if atype == "end_turn":
            for i, a in enumerate(legal):
                if isinstance(a, EndTurnAction):
                    return i
            return None
        if atype == "mana_draw":
            # Добор за ману — единственный ManaDrawAction в legal-наборе.
            for i, a in enumerate(legal):
                if isinstance(a, ManaDrawAction):
                    return i
            return None
        st = self.engine._arena.state
        me = st.p1 if st.p1.user_id == user_id else st.p2
        enemy = st.p2 if st.p1.user_id == user_id else st.p1
        if atype == "play_card":
            card_ref = action_json.get("card_ref", action_json.get("hand_index", 0))
            try:
                hi = self.engine._resolve_hand_index(me.hand, card_ref)
            except Exception:  # noqa: BLE001
                return None
            # position НЕ проверяем: движок генерирует ровно один PlayCardAction на
            # (hand_index, target_id) — position=len(board) для warrior, None для
            # potion. position никогда не дискриминатор. Старая проверка position
            # молча зануляла legal_action_index для warrior-игр: webapp шлёт
            # board_position = слот клика (0..4), а не len(board), поэтому
            # int(a.position)!=int(position) всегда true при непустом столе.
            target_is_hero = bool(action_json.get("target_is_hero", False))
            target_id = str(enemy.hero.instance_id) if target_is_hero else (
                str(action_json.get("target_id")) if action_json.get("target_id") else None)
            for i, a in enumerate(legal):
                if not isinstance(a, PlayCardAction):
                    continue
                if a.hand_index != hi:
                    continue
                a_tid = str(a.target_id) if a.target_id else None
                if a_tid != target_id and not (a_tid is None and target_id is None):
                    continue
                return i
            return None
        if atype == "attack":
            attacker_id = str(action_json.get("attacker_id"))
            target_is_hero = bool(action_json.get("target_is_hero", False))
            target_id = str(action_json.get("target_id")) if action_json.get("target_id") else None
            for i, a in enumerate(legal):
                if not isinstance(a, AttackAction):
                    continue
                if str(a.attacker_id) != attacker_id:
                    continue
                if bool(a.target_is_hero) != target_is_hero:
                    continue
                if target_is_hero:
                    return i
                a_tid = str(a.target_id) if a.target_id else None
                if a_tid == target_id:
                    return i
            return None
        return None

    def _resolve_source_target(self, user_id: int, action_json: Dict[str, Any]) -> tuple:
        """Возвращает (source_card_full|None, target_card_full|None) PRE-execute,
        симметрично env_v5._build_event."""
        st = self.engine._arena.state
        me = st.p1 if st.p1.user_id == user_id else st.p2
        enemy = st.p2 if st.p1.user_id == user_id else st.p1
        atype = action_json.get("type")
        source = target = None
        if atype == "play_card":
            try:
                hi = self.engine._resolve_hand_index(me.hand, action_json.get("card_ref", action_json.get("hand_index", 0)))
            except Exception:  # noqa: BLE001
                hi = -1
            if 0 <= hi < len(me.hand):
                source = me.hand[hi]
            tid = action_json.get("target_id")
            pool = [me.hero, enemy.hero, *me.board, *enemy.board]
            if action_json.get("target_is_hero"):
                target = enemy.hero
            elif tid is not None:
                target = _find_by_instance_id(pool, tid)
        elif atype == "attack":
            aid = action_json.get("attacker_id")
            source = _find_by_instance_id(me.board, aid)
            if action_json.get("target_is_hero"):
                target = enemy.hero
            else:
                target = _find_by_instance_id(enemy.board, action_json.get("target_id"))
        return self._snapshot_card(source), self._snapshot_card(target)

    # ------------------------------------------------------------------
    # hooks (вызывается из match_runner рядом с analytics-хуками)
    # ------------------------------------------------------------------
    @staticmethod
    def _decision_timing_fields(
        decision_source: str,
        *,
        human_decision_time_ms: Optional[int],
        decision_time_censored: Optional[bool],
        decision_censor_reason: Optional[str],
        control_source: Optional[str],
    ) -> Dict[str, Any]:
        """Normalize the action-timing contract without fabricating human labels.

        Older callers do not observe a per-action decision clock.  Their human
        rows are retained, but explicitly censored instead of silently becoming
        zero/unknown Metronome targets.  Automated actors can never carry a
        human timing label, including replacement bots on a human seat.
        """

        normalized_control = (
            str(control_source).strip() if control_source is not None else None
        )
        normalized_control = normalized_control or None
        if str(decision_source) != "human":
            return {
                "human_decision_time_ms": None,
                "decision_time_censored": False,
                "decision_censor_reason": None,
                "control_source": normalized_control,
            }

        if human_decision_time_ms is None:
            return {
                "human_decision_time_ms": None,
                "decision_time_censored": True,
                "decision_censor_reason": (
                    str(decision_censor_reason).strip()
                    if decision_censor_reason
                    else "not_observed"
                ),
                "control_source": normalized_control,
            }

        latency_ms = int(human_decision_time_ms)
        inferred_censored = not (
            METRONOME_MIN_LABEL_MS <= latency_ms <= METRONOME_MAX_LABEL_MS
        )
        censored = (
            inferred_censored
            if decision_time_censored is None
            else bool(decision_time_censored)
        )
        reason = (
            str(decision_censor_reason).strip()
            if decision_censor_reason is not None
            else None
        )
        if censored and not reason:
            reason = (
                "outside_training_window"
                if inferred_censored
                else "explicitly_censored"
            )
        if not censored:
            reason = None
        return {
            "human_decision_time_ms": latency_ms,
            "decision_time_censored": censored,
            "decision_censor_reason": reason,
            "control_source": normalized_control,
        }

    def before_action(
        self,
        user_id: int,
        action_json: Dict[str, Any],
        decision_source: str,
        action_id_provided: Optional[int] = None,
        *,
        human_decision_time_ms: Optional[int] = None,
        decision_time_censored: Optional[bool] = None,
        decision_censor_reason: Optional[str] = None,
        control_source: Optional[str] = None,
    ) -> int:
        try:
            st = self.engine._arena.state
            # turn snapshot — на первом действии каждого ГЛОБАЛЬНОГО хода
            if st.turn_number != self._last_turn_snapshot:
                self._append_turn_row()
                self._last_turn_snapshot = int(st.turn_number)

            pre_state = self._snapshot_state()
            pre_reward = self._reward_snapshot(st, user_id)
            try:
                legal = self.engine.get_legal_actions_raw(user_id)
            except Exception:  # noqa: BLE001
                legal = []
            # legal_action_index — индекс в get_legal_actions_raw (0..N-1), НЕ V5
            # tcode (0..600). tcode считает offline-мост из action_native + pre_state.
            # action_native — выбранное действие в engine-native to_dict (единообразно
            # для human+bot); legal_actions — полный набор to_dict (для action-mask).
            legal_index = self._resolve_action_id(user_id, action_json, action_id_provided, legal)
            source_card, target_card = self._resolve_source_target(user_id, action_json)
            atype = action_json.get("type") or "unknown"
            action_native = None
            if legal_index is not None and 0 <= legal_index < len(legal):
                try:
                    action_native = legal[legal_index].to_dict()
                except Exception:  # noqa: BLE001
                    action_native = None
            try:
                legal_actions = [a.to_dict() for a in legal]
            except Exception:  # noqa: BLE001
                legal_actions = []
            self._seq += 1
            actor_player = 1 if user_id == st.p1.user_id else 2
            timing_fields = self._decision_timing_fields(
                decision_source,
                human_decision_time_ms=human_decision_time_ms,
                decision_time_censored=decision_time_censored,
                decision_censor_reason=decision_censor_reason,
                control_source=control_source,
            )
            row = {
                "seq": self._seq,
                "battle_id": self.battle_id,
                "turn_number": int(st.turn_number),
                "actor_user_id": int(user_id),
                "actor_player": actor_player,
                "decision_source": decision_source,
                **timing_fields,
                "legal_action_index": legal_index,
                "action_type": atype,
                "action_json": action_json,
                "action_native": action_native,
                "source_card": source_card,
                "target_card": target_card,
                "legal_actions": legal_actions,
                "legal_action_count": len(legal),
                "pre_state": pre_state,
                "post_state": None,
                "deltas": None,
                "accepted": None,
                "error": None,
                "timestamp_ms": int(time.monotonic() * 1000),
                "visibility": VISIBILITY,
            }
            handle = len(self._buffer)
            self._buffer.append(row)
            self._pre_snapshots[handle] = pre_reward
            return handle
        except Exception:  # noqa: BLE001
            logger.warning("v5 before_action failed: %s", exc_info=True)
            return -1

    def after_action(self, handle: int, accepted: bool, error: Optional[str] = None) -> None:
        if handle is None or handle < 0 or handle >= len(self._buffer):
            return
        try:
            row = self._buffer[handle]
            pre_reward = self._pre_snapshots.pop(handle, None)
            st = self.engine._arena.state
            actor_user_id = int(row["actor_user_id"])
            post_reward = self._reward_snapshot(st, actor_user_id)
            row["post_state"] = self._snapshot_state()
            if pre_reward is not None:
                row["deltas"] = self._deltas(pre_reward, post_reward)
            row["accepted"] = bool(accepted)
            row["error"] = str(error) if error else None
            self._flush_action(row)
        except Exception:  # noqa: BLE001
            logger.warning("v5 after_action failed: %s", exc_info=True)

    def _append_turn_row(self) -> None:
        try:
            row = self._snapshot_state()
            row["visibility"] = VISIBILITY
            with self.turns_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            logger.warning("v5 turn row failed: %s", exc_info=True)

    def _flush_action(self, row: Dict[str, Any]) -> None:
        with self.actions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_terminal(
        self,
        user_id: int,
        action_type: str,
        reason: str,
        decision_source: str = "human",
        *,
        human_decision_time_ms: Optional[int] = None,
        decision_time_censored: Optional[bool] = None,
        decision_censor_reason: Optional[str] = None,
        control_source: Optional[str] = None,
    ) -> int:
        """Синтетический action-row для терминала БЕЗ обычного действия (surrender /
        non-action draw). pre_state снапается тут; post_state + deltas — в
        after_action(handle) ПОСЛЕ мутации (mark_surrender меняет status на P2_WIN).
        Без этого терминальное состояние не попадает в actions.jsonl — мосту пришлось
        бы реконструировать его из meta.json. Возвращает handle для after_action."""
        if self._finalized:
            return -1
        try:
            st = self.engine._arena.state
            if st.turn_number != self._last_turn_snapshot:
                self._append_turn_row()
                self._last_turn_snapshot = int(st.turn_number)
            pre_state = self._snapshot_state()
            pre_reward = self._reward_snapshot(st, user_id)
            self._seq += 1
            actor_player = 1 if user_id == st.p1.user_id else 2
            timing_fields = self._decision_timing_fields(
                decision_source,
                human_decision_time_ms=human_decision_time_ms,
                decision_time_censored=decision_time_censored,
                decision_censor_reason=decision_censor_reason,
                control_source=control_source,
            )
            row = {
                "seq": self._seq,
                "battle_id": self.battle_id,
                "turn_number": int(st.turn_number),
                "actor_user_id": int(user_id),
                "actor_player": actor_player,
                "decision_source": decision_source,
                **timing_fields,
                "legal_action_index": None,
                "action_type": action_type,
                "action_json": {"type": action_type, "reason": reason},
                "action_native": None,
                "source_card": None,
                "target_card": None,
                "legal_actions": [],
                "legal_action_count": 0,
                "pre_state": pre_state,
                "post_state": None,
                "deltas": None,
                "accepted": None,
                "error": None,
                "timestamp_ms": int(time.monotonic() * 1000),
                "visibility": VISIBILITY,
            }
            handle = len(self._buffer)
            self._buffer.append(row)
            self._pre_snapshots[handle] = pre_reward
            return handle
        except Exception:  # noqa: BLE001
            logger.warning("v5 record_terminal failed: %s", exc_info=True)
            return -1

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------
    def finalize(self, winner_user_id: Optional[int], status: str) -> Dict[str, Any]:
        if self._finalized:
            return {}
        self._finalized = True
        try:
            self.engine._arena.state  # noqa: B018 — touch
            st = self.engine._arena.state
            duration = round(time.monotonic() - self._start_monotonic, 3)
            meta = json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {}
            meta.update({
                "finished_at": _utc_now_iso(),
                "status": status,
                "winner_user_id": winner_user_id,
                "duration_seconds": duration,
                "turns": int(st.turn_number),
            })
            if self._nemesis_collector is not None:
                nemesis_status = (
                    status
                    if status in {"p1_win", "p2_win", "draw", "stalemate"}
                    else "draw"
                )
                meta["nemesis_record"] = (
                    self._nemesis_collector.finalize(
                        status=nemesis_status,
                        duration_seconds=max(0, int(round(duration))),
                        turns_count=int(st.turn_number),
                    )
                )
            # атомарная запись (tmp+replace), симметрично _write_meta/_write_catalog.
            tmp = self.meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.meta_path)
        except Exception:  # noqa: BLE001
            logger.warning("v5 finalize failed: %s", exc_info=True)
        return {
            "v5_dir": str(self.v5_dir.relative_to(self.group_dir)),
            "v5_meta_path": str(self.meta_path.relative_to(self.group_dir)),
            "decks_cache": {
                "p1": {"card_ids": list(self.p1_deck_ids), "levels": dict(self.p1_levels)},
                "p2": {"card_ids": list(self.p2_deck_ids), "levels": dict(self.p2_levels)},
            },
        }

    def manifest_storage_block(self) -> Dict[str, Any]:
        return {
            "schema_version": V5_STORAGE_SCHEMA,
            "catalog_path": str(self.catalog_path.relative_to(self.group_dir)) if self.catalog_path.exists() else None,
            "catalog_path_base": "group_dir",
            "catalog_hash": self.catalog_hash,
        }


def _find_by_instance_id(cards, instance_id):
    if instance_id is None:
        return None
    target = str(instance_id)
    for c in cards:
        if str(getattr(c, "instance_id", None)) == target:
            return c
    return None


def _serialize_mode_config_safe(mode_config) -> Dict[str, Any]:
    try:
        from infrastructure.match_modes import serialize_mode_config
        return serialize_mode_config(mode_config)
    except Exception:  # noqa: BLE001
        return {}


__all__ = ["V5TraceRecorder", "V5_STORAGE_SCHEMA", "CATALOG_SCHEMA", "VISIBILITY"]
