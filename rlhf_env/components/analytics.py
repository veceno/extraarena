"""AnalyticsRecorder — каноничный NDJSON-захват训练очных данных + фиксы аудита.

Реплицирует схему реального экспорта `infrastructure/database.py::export_train_v2_battle_dataset`
(строки `train_v2_admin_battle_action_jsonl_v2` / `dataset_schema=train_v3_battle_action_context_v1`),
но с исправлениями известных багов аудита (.workflow/battle-data-logs-audit-2026-06-09):

  F01 (omniscient state)  → state_json actor-perspective: рука/дека оппонента скрыты
                            ([{hidden:true}...]); добавлено поле `visibility`.
  F02 (failed == accepted)→ строка пишется с `accepted` + `decision_source` (+ `error`);
                            pre-state ловится ДО выполнения, accepted — после.
  F08 (won fragility)      → `won` проставляется ТОЛЬКО в finalize (winner==acting),
                            до этого null; draw → null.
  F12 (provenance)        → `battle_metadata` несёт полные provenance: deck_ids+levels
                            p1/p2, catalog_hash, opponent_profile{model,difficulty,brain_profile},
                            side, p2_model, group_id, series_index, recorder_version.
                            NOTE: «система сложностей» удалена — `difficulty` теперь
                            фиксированная константа "max" (модель всегда играет на
                            максимум, argmax); поле сохранено как provenance-атрибут.

Контракт хуков (вызывается из match_runner):
    handle = recorder.before_action(user_id, action_json, decision_source)  # pre-state
    recorder.after_action(handle, accepted, error=None)                     # post-execute
    recorder.finalize(winner_user_id)                                        # flush NDJSON

Файлы (files-only, без БД):
    sessions/<gid>/battles/<bid>.jsonl   — построчно один бой
    sessions/<gid>/dataset.jsonl         — накопительный по серии (append)
Каждая строка NDJSON — самодостаточный row (format/format_version/dataset_schema внутри).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.state import GameStatus  # noqa: F401  (типы)

logger = logging.getLogger(__name__)

RECORDER_VERSION = "rlhf-analytics-1.1"
CARD_PARAMS_SCHEMA = "train_v3_card_params_v1"
ACTION_CONTEXT_SCHEMA = "train_v3_action_context_v1"
DECK_PARAMS_SCHEMA = "train_v3_deck_params_v1"
FORMAT = "train_v2_admin_battle_action_jsonl_v2"
DATASET_SCHEMA = "train_v3_battle_action_context_v1"
COMPATIBLE_WITH = ["train_v2_admin_battle_action_jsonl_v1"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class AnalyticsRecorder:
    """Per-battle буфер строк NDJSON + flush в finalize."""

    def __init__(
        self,
        engine: Any,
        *,
        group_id: str,
        battle_id: str,
        series_index: int,
        p1_deck_cards: List[Any],
        p2_deck_cards: List[Any],
        p1_deck_ids: List[int],
        p2_deck_ids: List[int],
        p1_levels: Dict[int, int],
        p2_levels: Dict[int, int],
        catalog_hash: str,
        p2_model: str,
        difficulty: str,
        bot_brain_profile: Optional[str],
        game_mode: str = "classic",
        side: str = "p1",
        human_user_id: int = 1000,
        bot_user_id: int = 2000,
        sessions_dir: Optional[Path | str] = None,
    ) -> None:
        self.engine = engine
        self.group_id = group_id
        self.battle_id = battle_id
        self.series_index = series_index
        self.p1_deck_ids = list(p1_deck_ids)
        self.p2_deck_ids = list(p2_deck_ids)
        self.p1_levels = dict(p1_levels)
        self.p2_levels = dict(p2_levels)
        self.catalog_hash = catalog_hash
        self.p2_model = p2_model
        self.difficulty = difficulty
        self.bot_brain_profile = bot_brain_profile
        self.game_mode = game_mode
        self.side = side
        self.human_user_id = human_user_id
        self.bot_user_id = bot_user_id

        # sessions_dir нужен для записи файлов; если не задан — используется
        # group_dir манифеста (передаётся через set_group_dir).
        self.sessions_dir: Optional[Path] = Path(sessions_dir) if sessions_dir else None
        self._group_dir: Optional[Path] = None

        self._buffer: List[Dict[str, Any]] = []
        self._action_seq = 0
        self._start_monotonic = time.monotonic()
        self._winner_user_id: Optional[int] = None
        self._flushed = False

        # deck_param_snapshots — по начальным декам (до добора).
        self.deck_param_snapshots: Dict[str, Any] = {
            "p1": self.engine._deck_param_snapshot(p1_deck_cards),
            "p2": self.engine._deck_param_snapshot(p2_deck_cards),
        }

    # ------------------------------------------------------------------
    # конфигурация путей
    # ------------------------------------------------------------------
    def set_group_dir(self, group_dir: Path | str) -> None:
        self._group_dir = Path(group_dir)

    @property
    def group_dir(self) -> Path:
        if self._group_dir:
            return self._group_dir
        if self.sessions_dir:
            return self.sessions_dir / self.group_id
        # fallback: cwd
        return Path(self.group_id)

    @property
    def battle_jsonl_path(self) -> Path:
        return self.group_dir / "battles" / f"{self.battle_id}.jsonl"

    @property
    def dataset_jsonl_path(self) -> Path:
        return self.group_dir / "dataset.jsonl"

    # ------------------------------------------------------------------
    # actor-perspective snapshot (F01)
    # ------------------------------------------------------------------
    def _snapshot_card(self, card: Any) -> Dict[str, Any]:
        return self.engine._snapshot_card(card)

    def _player_full(self, p: Any) -> Dict[str, Any]:
        return {
            "hp": p.hero.hp,
            "max_hp": p.hero.max_hp,
            "mana": p.mana,
            "max_mana": p.max_mana,
            "hand": [self._snapshot_card(c) for c in p.hand],
            "board": [self._snapshot_card(c) for c in p.board],
            "deck": [self._snapshot_card(c) for c in p.deck],
            "graveyard": [self._snapshot_card(c) for c in p.graveyard],
            "hero": self._snapshot_card(p.hero),
        }

    def _player_hidden(self, p: Any) -> Dict[str, Any]:
        """Opponent view: hand+deck contents скрыты, board/hero/graveyard видны."""
        return {
            "hp": p.hero.hp,
            "max_hp": p.hero.max_hp,
            "mana": p.mana,
            "max_mana": p.max_mana,
            "hand": [{"hidden": True} for _ in p.hand],
            "board": [self._snapshot_card(c) for c in p.board],
            "deck": [{"hidden": True} for _ in p.deck],
            "graveyard": [self._snapshot_card(c) for c in p.graveyard],
            "hero": self._snapshot_card(p.hero),
        }

    def actor_perspective_snapshot(self, viewer_id: int) -> Dict[str, Any]:
        """F01: state_json actor-perspective — скрытая инфа оппонента не утекает."""
        st = self.engine._arena.state
        p1_is_actor = st.p1.user_id == viewer_id
        p2_is_actor = st.p2.user_id == viewer_id
        if p1_is_actor:
            p1_view, p2_view = self._player_full(st.p1), self._player_hidden(st.p2)
        elif p2_is_actor:
            p1_view, p2_view = self._player_hidden(st.p1), self._player_full(st.p2)
        else:
            # viewer не участник — обе стороны скрыты (защита)
            p1_view, p2_view = self._player_hidden(st.p1), self._player_hidden(st.p2)
        return {
            "turn": st.turn_number,
            "current_player": st.current_turn_owner_id,
            "p1": p1_view,
            "p2": p2_view,
        }

    # ------------------------------------------------------------------
    # action context (порт battle_engine._build_analytics_action_context)
    # ------------------------------------------------------------------
    def build_action_context(
        self, user_id: int, acting_player: int, action_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        st = self.engine._arena.state
        player = st.p1 if acting_player == 1 else st.p2
        acting_hand = [
            self.engine._card_params_slot_payload(card, slot=idx, zone="hand", hand_index=idx)
            for idx, card in enumerate(player.hand)
        ]

        playable_indexes: set[int] = set()
        legal_action_count = 0
        try:
            legal_actions = self.engine._arena.get_legal_actions(user_id)
            legal_action_count = len(legal_actions)
            from core.actions import PlayCardAction
            for action in legal_actions:
                if isinstance(action, PlayCardAction):
                    playable_indexes.add(int(action.hand_index))
        except Exception:  # noqa: BLE001
            legal_actions = []

        available_hand = [
            item for item in acting_hand
            if int(item.get("hand_index", -1)) in playable_indexes
        ]

        selected_card = None
        if str(action_json.get("type") or "") == "play_card":
            card_ref = action_json.get("card_ref", action_json.get("hand_index", 0))
            hand_index = self.engine._resolve_hand_index(player.hand, card_ref)
            if 0 <= hand_index < len(player.hand):
                selected_card = self.engine._card_params_slot_payload(
                    player.hand[hand_index], slot=hand_index, zone="hand", hand_index=hand_index,
                )

        return {
            "schema": ACTION_CONTEXT_SCHEMA,
            "card_params_schema": CARD_PARAMS_SCHEMA,
            "deck_params_schema": DECK_PARAMS_SCHEMA,
            "acting_player": acting_player,
            "game_mode": self.engine.game_mode,
            "ruleset": self.engine.ruleset,
            "turn_number": st.turn_number,
            "turn_start_mana": player.mana,
            "legal_action_count": legal_action_count,
            "acting_hand": acting_hand,
            "available_hand_cards": available_hand,
            "selected_card": selected_card,
            "deck_snapshot_ref": "battle_metadata.deck_param_snapshots",
        }

    # ------------------------------------------------------------------
    # provenance (F12)
    # ------------------------------------------------------------------
    def _battle_metadata(self) -> Dict[str, Any]:
        return {
            "recorder_version": RECORDER_VERSION,
            "group_id": self.group_id,
            "series_index": self.series_index,
            "side": self.side,
            "p2_model": self.p2_model,
            "difficulty": self.difficulty,
            "bot_brain_profile": self.bot_brain_profile,
            "opponent_profile": {
                "model": self.p2_model,
                "difficulty": self.difficulty,
                "brain_profile": self.bot_brain_profile,
                "is_bot": True,
            },
            "catalog_hash": self.catalog_hash,
            "decks": {
                "p1": {"card_ids": list(self.p1_deck_ids), "levels": dict(self.p1_levels)},
                "p2": {"card_ids": list(self.p2_deck_ids), "levels": dict(self.p2_levels)},
            },
            "deck_param_snapshots": self.deck_param_snapshots,
            "human_user_id": self.human_user_id,
            "bot_user_id": self.bot_user_id,
            "created_at": _utc_now_iso(),
        }

    # ------------------------------------------------------------------
    # хуки (вызывает match_runner)
    # ------------------------------------------------------------------
    def before_action(
        self,
        user_id: int,
        action_json: Dict[str, Any],
        decision_source: str,
        *,
        human_decision_time_ms: Optional[int] = None,
    ) -> int:
        st = self.engine._arena.state
        acting_player = 1 if user_id == st.p1.user_id else 2
        try:
            state_json = self.actor_perspective_snapshot(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("actor_perspective_snapshot failed: %s", exc, exc_info=True)
            state_json = {}
        try:
            context = self.build_action_context(user_id, acting_player, action_json)
        except Exception as exc:  # noqa: BLE001
            logger.warning("build_action_context failed: %s", exc, exc_info=True)
            context = {}

        self._action_seq += 1
        row: Dict[str, Any] = {
            "format": FORMAT,
            "format_version": 2,
            "dataset_schema": DATASET_SCHEMA,
            "compatible_with": COMPATIBLE_WITH,
            "id": uuid.uuid4().hex,
            "battle_id": self.battle_id,
            "group_id": self.group_id,
            "turn_number": int(st.turn_number),
            "acting_player": acting_player,
            "acting_user_id": int(user_id),
            "is_bot": bool(self.engine.is_bot(user_id)),
            "game_mode": self.game_mode,
            "winner_user_id": None,        # F08: проставляется в finalize
            "won": None,                   # F08
            "accepted": None,              # F02: проставляется в after_action
            "decision_source": decision_source,  # F02/F03
            # Server-observed interval from the moment an actionable state was
            # exposed to the browser until the next request arrived.  Deliberately
            # null for llm/rl/bot actors so synthetic pacing never contaminates the
            # future humanisation dataset.
            "human_decision_time_ms": (
                int(human_decision_time_ms)
                if decision_source == "human" and human_decision_time_ms is not None
                else None
            ),
            "action_seq": self._action_seq,
            "timestamp_ms": int(time.monotonic() * 1000),
            "state_json": state_json,
            "action_json": action_json,
            "context_json": context,
            "deck_param_snapshots": self.deck_param_snapshots,
            "card_params_schema": CARD_PARAMS_SCHEMA,
            "quality_score": None,
            "battle_metadata": self._battle_metadata(),
            "created_at": _utc_now_iso(),
            "visibility": "actor_perspective",  # F01
        }
        handle = len(self._buffer)
        self._buffer.append(row)
        return handle

    def after_action(self, handle: Any, accepted: bool, error: Optional[str] = None) -> None:
        try:
            row = self._buffer[int(handle)]
        except (IndexError, TypeError, ValueError):
            return
        row["accepted"] = bool(accepted)
        if error:
            row["error"] = str(error)

    # ------------------------------------------------------------------
    # finalize + flush
    # ------------------------------------------------------------------
    def finalize(self, winner_user_id: Optional[int]) -> Dict[str, Any]:
        if self._flushed:
            return {"rows": 0, "path": str(self.battle_jsonl_path)}
        self._flushed = True
        self._winner_user_id = winner_user_id

        for row in self._buffer:
            row["winner_user_id"] = winner_user_id
            if winner_user_id is None:
                row["won"] = None  # draw / unknown
            else:
                row["won"] = bool(winner_user_id == int(row["acting_user_id"]))
            row["finalized_at"] = _utc_now_iso()

        written = self._flush()
        logger.info(
            "[AnalyticsRecorder] finalize battle=%s rows=%d winner=%s → %s",
            self.battle_id, written, winner_user_id, self.battle_jsonl_path,
        )
        return {"rows": written, "battle_jsonl": str(self.battle_jsonl_path),
                "dataset_jsonl": str(self.dataset_jsonl_path)}

    def _flush(self) -> int:
        if not self._buffer:
            return 0
        # per-battle .jsonl (overwrite)
        self.battle_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.battle_jsonl_path.open("w", encoding="utf-8") as f:
            for row in self._buffer:
                f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
        # append to series dataset.jsonl
        self.dataset_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_jsonl_path.open("a", encoding="utf-8") as f:
            for row in self._buffer:
                f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
        return len(self._buffer)

    # ------------------------------------------------------------------
    # introspection (для MCP/тестов)
    # ------------------------------------------------------------------
    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._buffer)

    def row_count(self) -> int:
        return len(self._buffer)


def _json_safe(obj: Any) -> Any:
    """Рекурсивно делает объект JSON-сериализуемым (как infrastructure.database._json_safe)."""
    import enum
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return str(obj)


__all__ = ["AnalyticsRecorder", "RECORDER_VERSION"]
